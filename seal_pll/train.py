import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import time
import copy
import numpy as np
import random
import higher

from utils.cifar10 import load_cifar10
from utils.cifar100 import load_cifar100
from resnet import resnet
from meta_loss_network import MetaLossNetwork

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class TimeUse:
    def __init__(self, name="", measure = "s") -> None:
        self.t = -1
        self.name = name
        self.measure = measure
        assert self.measure in ['s', 'm', 'h']

    def __enter__(self):
        self.t = time.time()
        print("{} start.".format(self.name))

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.measure == "s":
            self.use_time = (time.time() - self.t)
        if self.measure == "m":
            self.use_time = (time.time() - self.t) / 60
        if self.measure == "h":
            self.use_time = (time.time() - self.t) / 60 / 60       
        print("{} ends, using {:.2f} {}".format(self.name, self.use_time, self.measure))

def linear_rampup(current, warm_up, rampup_length=16):
    current = np.clip((current-warm_up) / rampup_length, 0.0, 1.0)
    return float(current)

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def validate(loader, model, device):
    with torch.no_grad():
        total, num_samples = 0, 0
        for images, labels in loader:
            labels, images = labels.to(device), images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += (predicted == labels).sum().item()
            num_samples += labels.size(0)

    return 100*(total/num_samples)

def confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, part_y, index):
    y_pred_aug0_probas = y_pred_aug0_probas.detach()
    y_pred_aug1_probas = y_pred_aug1_probas.detach()
    y_pred_aug2_probas = y_pred_aug2_probas.detach()

    revisedY0 = part_y.clone()

    revisedY0 = revisedY0 * torch.pow(y_pred_aug0_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug1_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug2_probas, 1 / (2 + 1))
    revisedY0 = revisedY0 / revisedY0.sum(dim=1).repeat(part_y.shape[1], 1).transpose(0, 1)

    confidence[index, :] = revisedY0.cpu().numpy()

def train(epoch, train_loader, model, optimizer, consistency_criterion, confidence, device):
    model.train()
    for i, (x_aug0, x_aug1, x_aug2, part_y, y, index) in enumerate(train_loader):
        # partial label
        part_y = part_y.float().to(device)
        # original samples with pre-processing
        x_aug0, x_aug1, x_aug2 = map(lambda x: x.to(device), (x_aug0, x_aug1, x_aug2))
        y_pred_aug0, y_pred_aug1, y_pred_aug2 = map(lambda x: model(x), (x_aug0, x_aug1, x_aug2))
        y_pred_aug0_probas_log, y_pred_aug1_probas_log, y_pred_aug2_probas_log = \
            map(lambda x: torch.log_softmax(x, dim=-1), (y_pred_aug0, y_pred_aug1, y_pred_aug2))
        y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas = \
            map(lambda x: torch.softmax(x, dim=-1), (y_pred_aug0, y_pred_aug1, y_pred_aug2))
        # consist_loss
        consist_loss0, consist_loss1, consist_loss2 = \
            map(lambda x: consistency_criterion(x, torch.tensor(confidence[index]).float().to(device)), 
                (y_pred_aug0_probas_log, y_pred_aug1_probas_log, y_pred_aug2_probas_log))
        # supervised loss
        super_loss0, super_loss1, super_loss2 = \
            map(lambda x: - torch.mean(torch.sum(torch.log(1.0000001 - F.softmax(x, dim=1)) * (1 - part_y), dim=1)), 
                (y_pred_aug0, y_pred_aug1, y_pred_aug2))
        # dynamic lam
        lam = min((epoch / 100) * args.lam, args.lam)

        # Unified loss
        final_loss = lam * (consist_loss0 + consist_loss1 + consist_loss2) + (super_loss0 + super_loss1 + super_loss2)

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()
        # update confidence
        confidence_update(confidence, y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas, part_y, index)

def confidence_update_with_meta(confidence, 
        y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas,
        y_pred_aug0_probas_after_meta, y_pred_aug1_probas_after_meta, y_pred_aug2_probas_after_meta, 
        part_y, index, update_mom=0.3):

    y_pred_aug0_probas = y_pred_aug0_probas.detach()
    y_pred_aug1_probas = y_pred_aug1_probas.detach()
    y_pred_aug2_probas = y_pred_aug2_probas.detach()

    revisedY0 = part_y.clone()

    revisedY0 = revisedY0 * torch.pow(y_pred_aug0_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug1_probas, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug2_probas, 1 / (2 + 1))
    revisedY0 = revisedY0 / revisedY0.sum(dim=1).repeat(part_y.shape[1], 1).transpose(0, 1)

    y_pred_aug0_probas_after_meta = y_pred_aug0_probas_after_meta.detach()
    y_pred_aug1_probas_after_meta = y_pred_aug1_probas_after_meta.detach()
    y_pred_aug2_probas_after_meta = y_pred_aug2_probas_after_meta.detach()

    revisedY1 = part_y.clone()

    revisedY1 = revisedY1 * torch.pow(y_pred_aug0_probas_after_meta, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug1_probas_after_meta, 1 / (2 + 1)) \
                * torch.pow(y_pred_aug2_probas_after_meta, 1 / (2 + 1))
    revisedY1 = revisedY1 / revisedY1.sum(dim=1).repeat(part_y.shape[1], 1).transpose(0, 1)

    confidence[index, :] = (1 - update_mom) * revisedY0.cpu().numpy() + update_mom * revisedY1.cpu().numpy()


def partial_kl(y1, y2, part_y=None, is_softmax=True, is_log_softmax=True):
    consistency_criterion = nn.KLDivLoss(reduction='batchmean')
    if is_log_softmax:
        y1_probas_log = torch.log_softmax(y1, dim=-1)
    if is_softmax:
        y2_norm = torch.softmax(y2, dim=1)
    if part_y is not None:
        y2_norm = y2_norm * part_y
        y2_norm = y2_norm / y2.sum(dim=1, keepdim=True)
    else:
        y2_norm = y2
    return consistency_criterion(y1_probas_log, y2_norm)

def partial_wce(y1, y2, part_y=None, is_softmax=True, is_log_softmax=True, is_target=False):
    if is_target:
        y2 = y2.detach().clone()
    if is_log_softmax:
        y1_probas_log = torch.log_softmax(y1, dim=-1)
    if is_softmax:
        y2_norm = torch.softmax(y2, dim=1)
    if part_y is not None:
        y2_norm = y2_norm * part_y
        y2_norm = y2_norm / y2.sum(dim=1, keepdim=True)
    else:
        y2_norm = y2
    return - torch.sum(y2_norm * y1_probas_log, dim=1).mean()

def reverse_ce(y, target):
    return - torch.mean(torch.sum(torch.log(1.0000001 - F.softmax(y, dim=1)) * (1 - target), dim=1))

def train_meta(
        args, epoch, device,
        train_loader, 
        model_f, optimizer_f, 
        model_g, optimizer_g, 
        model_h, optimizer_h, 
        consistency_criterion, 
        confidence):
    model_f.train()
    model_g.train()
    model_h.train()

    for i, (x_aug0, x_aug1, x_aug2, part_y, y, index) in enumerate(train_loader):
        # partial label
        part_y = part_y.float().to(device)
        # original samples with pre-processing
        x_aug0, x_aug1, x_aug2 = map(lambda x: x.to(device), (x_aug0, x_aug1, x_aug2))
        y_pred_aug0, y_pred_aug1, y_pred_aug2 = map(lambda x: model_f(x), (x_aug0, x_aug1, x_aug2))
        y_pred_aug0_probas_log, y_pred_aug1_probas_log, y_pred_aug2_probas_log = \
            map(lambda x: torch.log_softmax(x, dim=-1), (y_pred_aug0, y_pred_aug1, y_pred_aug2))
        y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas = \
            map(lambda x: torch.softmax(x, dim=-1), (y_pred_aug0, y_pred_aug1, y_pred_aug2))
        # consist_loss
        consist_loss0, consist_loss1, consist_loss2 = \
            map(lambda x: consistency_criterion(x, torch.tensor(confidence[index]).float().to(device)), 
                (y_pred_aug0_probas_log, y_pred_aug1_probas_log, y_pred_aug2_probas_log))
        # supervised loss
        super_loss0, super_loss1, super_loss2 = \
            map(lambda x: -torch.mean(torch.sum(torch.log(1.0000001 - F.softmax(x, dim=1)) * (1 - part_y), dim=1)), 
                (y_pred_aug0, y_pred_aug1, y_pred_aug2))
        # dynamic lam
        lam = min((epoch / 100) * args.lam, args.lam)

        # Unified loss
        final_loss = lam * (consist_loss0 + consist_loss1 + consist_loss2) + (super_loss0 + super_loss1 + super_loss2)

        optimizer_f.zero_grad()
        final_loss.backward()
        optimizer_f.step()
        
        # meta learning
        with higher.innerloop_ctx(model_g, optimizer_g, copy_initial_weights=False) as (fmodel_g, diffopt_g):
            with higher.innerloop_ctx(model_f, optimizer_f, copy_initial_weights=False) as (fmodel_f, diffopt_f):
                # update soft-pseudo-label generator
                inputs_x = (x_aug0, x_aug1, x_aug2)
                output_g = list(map(lambda input: fmodel_g(input), inputs_x))
                
                ramp_up_value = linear_rampup(epoch + i / len(train_loader), warm_up=0, rampup_length=args.ramp_up_epoch)
                loss_g = ramp_up_value * (1 / len(output_g)) * sum(map(lambda x: model_h(x, torch.tensor(confidence[index]).float().to(device), part_y), output_g)) + \
                    (1 - ramp_up_value + args.keep_value) * (lam * sum(map(lambda y: partial_kl(y, torch.tensor(confidence[index]).float().to(device)), output_g)) + \
                        args.super_weight * sum(map(lambda y: reverse_ce(y, part_y), output_g)))
                
                diffopt_g.step(loss_g)

                # update predictive model
                output_f = list(map(lambda input: fmodel_f(input), inputs_x))
                output_g_new = list(map(lambda input: fmodel_g(input), inputs_x))
                loss_f = (1 / len(output_f)) * (
                    lam * sum(map(lambda y1, y2: partial_wce(y1, y2, part_y), output_f, output_g_new)) +
                    sum(map(lambda y: reverse_ce(y, part_y), output_f))
                )
                diffopt_f.step(loss_f)

                # update parameterized objective function
                output_f_new = list(map(lambda input: fmodel_f(input), inputs_x))
                loss_h = (1 / len(output_f)) * (
                    lam * sum(map(lambda y: partial_kl(y, torch.tensor(confidence[index]).float().to(device)), output_f_new)) +
                    args.super_weight * sum(map(lambda y: reverse_ce(y, part_y), output_f_new))
                )
                optimizer_h.zero_grad()
                loss_h.backward()
                optimizer_h.step()

        model_g.load_state_dict(fmodel_g.state_dict())

        with torch.no_grad():
            y_pred_aug0_after_metas = \
                map(lambda x: model_g(x), inputs_x)
            y_pred_aug0_probas_after_metas = \
                map(lambda x: torch.softmax(x, dim=-1), y_pred_aug0_after_metas)
            # update confidence
            confidence_update_with_meta(confidence, 
                y_pred_aug0_probas, y_pred_aug1_probas, y_pred_aug2_probas,
                *y_pred_aug0_probas_after_metas, 
                part_y, index, update_mom=args.update_mom)


def main(args):
    for key, value in vars(args).items():
        print(f"{key}: {value}")

    set_seed(args.seed)
    device = torch.device("cuda:" + args.gpu)

    # load data
    if args.dataset == "cifar10":
        train_loader, valid_loader, test_loader, dim, K = load_cifar10(batch_size=args.batch_size, partial_rate=args.partial_rate, root=args.data_dir)
    elif args.dataset == 'cifar100':
        train_loader, valid_loader, test_loader, dim, K = load_cifar100(batch_size=args.batch_size, partial_rate=args.partial_rate, root=args.data_dir)
    else:
        assert "Unknown dataset"

    # load model
    if args.model == 'resnet':
        model_f = resnet(depth=32, num_classes=K)
    else:
        assert "Unknown model"

    # predictive model
    model_f = model_f.to(device)
    consistency_criterion = nn.KLDivLoss(reduction='batchmean').to(device)
    optimizer_f = torch.optim.SGD(model_f.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler_f = torch.optim.lr_scheduler.MultiStepLR(optimizer_f, milestones=[100, 150], last_epoch=-1)
    
    # soft-pseudo-label generator
    model_g = copy.deepcopy(model_f)
    optimizer_g = torch.optim.SGD(model_g.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.wd)
    scheduler_g = torch.optim.lr_scheduler.MultiStepLR(optimizer_g, milestones=[100, 150], last_epoch=-1)

    # parameterized objective function
    model_h = MetaLossNetwork(input_dim=3, output_dim = K, gamma = args.gamma, beta = args.beta, one_hot_encode=False).to(device)
    optimizer_h = torch.optim.Adam(model_h.parameters(), lr=args.lr_h)
    
    # init confidence
    confidence = copy.deepcopy(train_loader.dataset.given_label_matrix.numpy())
    confidence = confidence / confidence.sum(axis=1)[:, None]

    epoch_best = -1
    prec1_f_test_best = 0
    prec1_f_val_best = 0

    for epoch in range(1, args.epochs+1):
        if epoch < args.meta_start_epoch:
            print("Original training at epoch {}. ".format(epoch))
            with TimeUse("Original training", 'm'):
                train(epoch, train_loader, model_f, optimizer_f, consistency_criterion, confidence, device)
        else:
            print("Meta training at epoch {}.".format(epoch))
            model_g.load_state_dict(model_f.state_dict())
            with TimeUse("Meta training", 'm'):
                train_meta(
                    args, epoch, device,
                    train_loader,
                    model_f, optimizer_f, 
                    model_g, optimizer_g,
                    model_h, optimizer_h,
                    consistency_criterion, confidence
                )
        
        scheduler_g.step()
        scheduler_f.step()

        # evaluate on validation set
        prec1_g_test = validate(test_loader, model_g, device)
        prec1_f_val  = validate(valid_loader, model_f, device)
        prec1_f_test = validate(test_loader, model_f, device)
        print("Epoch {}, model g test acc: {:.2f}, model f val acc: {:.2f}, model f test acc: {:.2f}".format(epoch, prec1_g_test, prec1_f_val, prec1_f_test))
        if prec1_f_val > prec1_f_val_best:
            epoch_best = epoch
            prec1_f_val_best = prec1_f_val
            prec1_f_test_best = prec1_f_test           
        print('predictive model best epoch:', epoch_best)
        print('predictive model best val acc:', prec1_f_val_best)
        print('predictive model best test acc:', prec1_f_test_best)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Meta + Revisiting Consistency Regularization for Deep Partial Label Learning')
    parser.add_argument('--epochs', default=200, type=int, help='number of total epochs to run')   
    parser.add_argument('--model', type=str, choices=['resnet'], default='resnet')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100'], default='cifar10')
    
    parser.add_argument('--data-dir', default='../data/', type=str)
    parser.add_argument('--batch-size', default=256, type=int)
    parser.add_argument('--partial-rate', default=0.3, type=float,help='0.x for random')

    parser.add_argument('--lr', default=0.05, type=float)
    parser.add_argument('--wd', default=0.001, type=float)

    parser.add_argument('--meta-start-epoch', default=10, type=int)
    parser.add_argument('--ramp-up-epoch', default=200, type=int)

    parser.add_argument('--lam', default=1, type=float)
    parser.add_argument('--keep-value', default=0.1, type=float)
    parser.add_argument('--update-mom', default=0.3, type=float, help='use to update confidence')
    parser.add_argument('--super-weight', default=1, type=float)

    # parameterized objective function
    parser.add_argument('--lr-h', default=1e-5, type=float)
    parser.add_argument('--beta', default=10, type=int,
                        help='the beta value for the Softplus formulation')
    parser.add_argument('--gamma', default=1e-2, type=float,
                        help='controls the angle of the negative slope in LeakyReLU')

    # others 
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--gpu', default='0', type=str, help='device')

    args = parser.parse_args()
    main(args)