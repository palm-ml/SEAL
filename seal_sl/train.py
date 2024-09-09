import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher
import difftopk
import argparse
import random
import numpy as np

from itertools import cycle
from dataset import CIFAR10, CIFAR100, TinyImageNet
from resnet_cifar import resnet20, resnet44
from resnet_tiny import resnet18
from meta_loss_network import MetaLossNetwork

parser = argparse.ArgumentParser()
# basic
parser.add_argument("--dataset", default="cifar10", type=str,
                    choices=["cifar10", "cifar100", "tinyimagenet"], help="dataset")
parser.add_argument('--epochs', default=200, type=int,
                    help='number of total training epochs for predictive model')

# dataloader
parser.add_argument("--data-dir", default="../data", type=str,
                    help="dataset directory")
parser.add_argument('--bs', default=128, type=int, help='mini-batch size')
parser.add_argument('--workers', default=4, type=int,
                    help='number of data loading workers')

# optimizer
parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--wd', default=1e-4, type=float, help='weight decay')

# parameterized objective function
parser.add_argument('--beta', default=10, type=int,
                    help='the beta value for the Softplus formulation')
parser.add_argument('--gamma', default=1e-2, type=float,
                    help='controls the angle of the negative slope in LeakyReLU')

# warmup
parser.add_argument('--warmup', action='store_true', 
                    help='warmup objective function and generator')
parser.add_argument('--init-step', default=5000, type=int,
                    help='init step to warmup objective function')
parser.add_argument('--inner-iter', default=2, type=int,
                    help='inner iter to warmup objective function')
parser.add_argument('--warmup-epochs', default=100, type=int,
                    help='epochs to warmup generator')
parser.add_argument("--save-folder", type=str, default="checkpoints",
                    help="folder to save warmup checkpoint")

# training loss
parser.add_argument('--temp', default=5, type=int, help='temperature scaling in KL divergence loss')
parser.add_argument('--alpha', default=0.5, type=float,
                    help='balance loss from original label and soft-pseudo-label')

# others
parser.add_argument('--print-freq', default=50, type=int,
                    help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed')
parser.add_argument('--gpu', default='0', type=str, help='device')



def warmup_h(train_loader, val_loader, model_g, model_h, device):
    step_time = AverageMeter()

    train_iter = cycle(train_loader)
    val_iter = cycle(val_loader)

    task_loss = nn.CrossEntropyLoss()
    optimizer_g = torch.optim.SGD(model_g.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4, nesterov=True)
    optimizer_h = torch.optim.Adam(model_h.parameters(), lr=1e-3)

    end = time.time()
    for i in range(args.init_step):
        model_g.reset()
        optimizer_h.zero_grad()
        with higher.innerloop_ctx(model_g, optimizer_g, copy_initial_weights=False) as (fmodel_g, diffopt_g):
            for _ in range(args.inner_iter):
                # input, target = next(iter(train_loader))
                input, target = next(train_iter)
                input, target= input.to(device), target.to(device)

                output_g = fmodel_g(input)
                loss_g = model_h(output_g, target)
                diffopt_g.step(loss_g)

            # input, target = next(iter(val_loader))
            input, target = next(val_iter)
            input, target= input.to(device), target.to(device)
            output_g_new = fmodel_g(input)
            loss_h = task_loss(output_g_new, target)
            loss_h.backward()        
        optimizer_h.step()

        # measure elapsed time
        step_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Step: [{0}/{1}]\t'
                  'Time {step_time.val:.3f} ({step_time.avg:.3f})'.format(
                      i, args.init_step, step_time=step_time))  

    model_g.reset()


def warmup_g(train_loader, model_g, optimizer_g, model_h, optimizer_h, epoch, device):
    batch_time = AverageMeter()
    losses_g = AverageMeter()
    losses_h = AverageMeter()
    top1_g = AverageMeter()
    
    task_loss = nn.CrossEntropyLoss()

    # switch to train mode
    model_g.train()
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input, target= input.to(device), target.to(device)
        
        optimizer_h.zero_grad()
        with higher.innerloop_ctx(model_g, optimizer_g, copy_initial_weights=False) as (fmodel_g, diffopt_g):
            # update soft-pseudo-label generator
            output_g = fmodel_g(input)
            loss_g = model_h(output_g, target)
            diffopt_g.step(loss_g)

            # update parameterized objective function
            output_g_new = fmodel_g(input)
            loss_h = task_loss(output_g_new, target)
            loss_h.backward()
                        
        optimizer_h.step()
        model_g.load_state_dict(fmodel_g.state_dict())
        
        # measure accuracy and record loss
        prec1_g = accuracy(output_g_new.float().data, target)[0]
        losses_g.update(loss_g.float().item(), input.size(0))
        losses_h.update(loss_h.float().item(), input.size(0))
        top1_g.update(prec1_g.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'loss_g {losses_g.val:.4f} ({losses_g.avg:.4f})\t'
                  'task_loss {losses_h.val:.4f} ({losses_h.avg:.4f})\t'
                  'Prec@1_g {top1_g.val:.3f} ({top1_g.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      losses_g=losses_g, losses_h=losses_h, top1_g=top1_g))
            

def train(train_loader, model_g, optimizer_g, model_f, optimizer_f, model_h, optimizer_h, task_loss, epoch, device):
    batch_time = AverageMeter()
    losses_g = AverageMeter()
    losses_f = AverageMeter()
    losses_h = AverageMeter()
    top1_g = AverageMeter()
    top1_f = AverageMeter()

    # switch to train mode
    model_g.train()
    model_f.train()
    
    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input, target= input.to(device), target.to(device)
        
        optimizer_h.zero_grad()
        with higher.innerloop_ctx(model_g, optimizer_g, copy_initial_weights=False) as (fmodel_g, diffopt_g):
            with higher.innerloop_ctx(model_f, optimizer_f, copy_initial_weights=False) as (fmodel_f, diffopt_f):
                # update soft-pseudo-label generator
                output_g = fmodel_g(input)
                loss_g = model_h(output_g, target)
                diffopt_g.step(loss_g)

                # update predictive model
                output_f = fmodel_f(input)
                output_g_new = fmodel_g(input)
                loss_f = (kldivloss(output_f, output_g_new) + kldivloss(output_g_new, output_f)) * args.alpha + F.cross_entropy(output_f, target)
                diffopt_f.step(loss_f)

                # update parameterized objective function
                output_f_new = fmodel_f(input)
                loss_h = task_loss(output_f_new, target)
                loss_h.backward()
                        
        optimizer_h.step()
        model_g.load_state_dict(fmodel_g.state_dict())
        model_f.load_state_dict(fmodel_f.state_dict())
        
        # measure accuracy and record loss
        prec1_g = accuracy(output_g_new.float().data, target)[0]
        prec1_f = accuracy(output_f_new.float().data, target)[0]
        losses_g.update(loss_g.float().item(), input.size(0))
        losses_f.update(loss_f.float().item(), input.size(0))
        losses_h.update(loss_h.float().item(), input.size(0))
        top1_g.update(prec1_g.item(), input.size(0))
        top1_f.update(prec1_f.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'loss_g {losses_g.val:.4f} ({losses_g.avg:.4f})\t'
                  'loss_f {losses_f.val:.4f} ({losses_f.avg:.4f})\t'
                  'loss_h {losses_h.val:.4f} ({losses_h.avg:.4f})\t'
                  'Prec@1_g {top1_g.val:.3f} ({top1_g.avg:.3f})\t'
                  'Prec@1_f {top1_f.val:.3f} ({top1_f.avg:.3f})'.format(
                      epoch, i, len(train_loader), batch_time=batch_time,
                      losses_g=losses_g, losses_f=losses_f, losses_h=losses_h,
                      top1_g=top1_g, top1_f=top1_f))


def validate(val_loader, model, device):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    criterion = nn.CrossEntropyLoss()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            input, target = input.to(device), target.to(device)

            # compute output
            output = model(input)
            loss = criterion(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                          i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))

    print(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def kldivloss(p_logit, q_logit):
    kldiv = torch.sum(
        F.softmax(q_logit / args.temp, dim=1)
        * (
            F.log_softmax(q_logit / args.temp, dim=1)
            - F.log_softmax(p_logit / args.temp, dim=1)
        ),
        dim=1,
    )
    return torch.mean(kldiv) * args.temp * args.temp

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def main():
    global args
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device("cuda:" + args.gpu)

    for key, value in vars(args).items():
        print(f"{key}: {value}")

    if args.dataset == "cifar10":
        num_classes = 10
        model = resnet20
        train_set, val_set, test_set, train_set_warmup, val_set_warmup = CIFAR10(data_dir=args.data_dir)
    elif args.dataset == "cifar100":
        num_classes = 100
        model = resnet44
        train_set, val_set, test_set, train_set_warmup, val_set_warmup = CIFAR100(data_dir=args.data_dir)
    elif args.dataset == "tinyimagenet":
        num_classes = 200
        model = resnet18
        train_set, val_set, test_set, train_set_warmup, val_set_warmup = TinyImageNet(data_dir=args.data_dir, size=32)

    # predictive model
    model_f = model(num_classes=num_classes).to(device)
    optimizer_f = torch.optim.SGD(model_f.parameters(), args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler_f = torch.optim.lr_scheduler.MultiStepLR(optimizer_f, milestones=[100, 150])

    # soft-pseudo-label generator
    model_g = model(num_classes=num_classes).to(device)
    optimizer_g = torch.optim.SGD(model_g.parameters(), args.lr, momentum=0.9, weight_decay=5e-4, nesterov=True)
    if args.dataset == 'cifar10' or args.dataset == 'cifar100':
        scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=args.epochs + args.warmup_epochs, eta_min=1e-3)
    elif args.dataset == 'tinyimagenet':
        scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=[100, 150])

    # parameterized objective function
    model_h = MetaLossNetwork(output_dim = num_classes, gamma = args.gamma, beta = args.beta).to(device)
    optimizer_h = torch.optim.Adam(model_h.parameters(), lr=1e-5)

    # task loss
    if args.dataset == 'cifar10':
        p_k = [.8, .2, 0., 0., 0.]
    elif args.dataset == 'cifar100' or args.dataset == 'tinyimagenet':
        p_k = [.5, 0., .3, 0., .2]
    task_loss = difftopk.TopKCrossEntropyLoss(
        diffsort_method='odd_even',       # the sorting / ranking method as discussed above
        inverse_temperature=2,            # the inverse temperature / steepness
        p_k=p_k,                          # the distribution P_K
        n=num_classes,                    # number of classes
        m=8,                              # the number m of scores to be sorted (can be smaller than n to make it efficient)
        distribution='cauchy',            # the distribution used for differentiable sorting networks
        art_lambda=None,                  # the lambda for the ART used if `distribution='logistic_phi'`
        device=device,                    # the device to compute the loss on
        top1_mode='sm'                    # makes training more stable and is the default value
    )       

    # load dataset
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)    
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.bs, shuffle=False, num_workers=args.workers, pin_memory=True)   
    train_loader_warmup = torch.utils.data.DataLoader(train_set_warmup, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)  
    val_loader_warmup = torch.utils.data.DataLoader(val_set_warmup, batch_size=args.bs, shuffle=True, num_workers=args.workers, pin_memory=True)
    
    epoch_best = -1
    prec1_f_val_best = 0
    prec1_f_test_best = 0

    if args.warmup:
        print('Warmup parameterized objective function...')
        warmup_h(train_loader_warmup, val_loader_warmup, model_g, model_h, device)

        print('Warmup soft-pseudo-label generator...')
        for epoch in range(0, args.warmup_epochs):
            print('current lr_g {:.5e}'.format(optimizer_g.param_groups[0]['lr']))
            print('current lr_h {:.5e}'.format(optimizer_h.param_groups[0]['lr']))
            warmup_g(train_loader, model_g, optimizer_g, model_h, optimizer_h, epoch, device)
            scheduler_g.step()

            print('soft-pseudo-label generator test acc')
            prec1_g = validate(test_loader, model_g)

        checkpoint = {
            'model_h': model_h.state_dict(),
            'model_g': model_g.state_dict(),
            'optimizer_h': optimizer_h.state_dict(),
            'optimizer_g': optimizer_g.state_dict(),
            'scheduler_g': scheduler_g.state_dict(),
        }
        save_path = os.path.join(args.save_folder, "{}.pt".format(args.dataset))
        torch.save(checkpoint, save_path)  
    else:
        print('Load the checkpoint...')
        load_path = os.path.join(args.save_folder, "{}.pt".format(args.dataset))
        checkpoint = torch.load(load_path, map_location='cpu')
        model_h.load_state_dict(checkpoint['model_h'])
        model_g.load_state_dict(checkpoint['model_g'])
        optimizer_h.load_state_dict(checkpoint['optimizer_h'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])
        scheduler_g.load_state_dict(checkpoint['scheduler_g'])    

    print('Start training...')
    if args.dataset == 'cifar10':
        optimizer_h = torch.optim.Adam(model_h.parameters(), lr=1e-6)
    for epoch in range(args.epochs):
        # train for one epoch
        print('current lr_g {:.5e}'.format(optimizer_g.param_groups[0]['lr']))
        print('current lr_f {:.5e}'.format(optimizer_f.param_groups[0]['lr']))
        print('current lr_h {:.5e}'.format(optimizer_h.param_groups[0]['lr']))
        train(train_loader, model_g, optimizer_g, model_f, optimizer_f, model_h, optimizer_h, task_loss, epoch, device)
        scheduler_g.step()
        scheduler_f.step()
        
        # evaluate
        print('soft-pseudo-label generator test acc')
        prec1_g_test = validate(test_loader, model_g)
        print('predictive model val acc')
        prec1_f_val = validate(val_loader, model_f)
        print('predictive model test acc')
        prec1_f_test = validate(test_loader, model_f)
        if prec1_f_val > prec1_f_val_best:
            epoch_best = epoch
            prec1_f_val_best = prec1_f_val
            prec1_f_test_best = prec1_f_test
        
        print('predictive model best epoch:', epoch_best)
        print('predictive model best val acc:', prec1_f_val_best)
        print('predictive model best test acc:', prec1_f_test_best)


if __name__ == '__main__':
    main()
