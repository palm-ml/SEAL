import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from augment.autoaugment_extra import CIFAR10Policy
from augment.cutout import Cutout
from .utils_algo import generate_instance_independent_candidate_labels

GENERATE_SEED=42
def load_cifar100(batch_size, partial_rate, root):
    test_transform = transforms.Compose(
            [transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))])
    
    temp_train = dsets.CIFAR100(root=root, train=True, download=True, transform=transforms.ToTensor())
    temp_valid = dsets.CIFAR100(root=root, train=True, transform=test_transform)
    data_size = len(temp_train)
    train_dataset, _ = torch.utils.data.random_split(temp_train,
                                                     [int(data_size * 0.9), data_size - int(data_size * 0.9)],
                                                     torch.Generator().manual_seed(GENERATE_SEED))
    _, valid_dataset = torch.utils.data.random_split(temp_valid,
                                                     [int(data_size * 0.9), data_size - int(data_size * 0.9)],
                                                     torch.Generator().manual_seed(GENERATE_SEED))
    valid_loader = torch.utils.data.DataLoader(dataset=valid_dataset, batch_size=len(valid_dataset), shuffle=False, num_workers=8)
    test_dataset = dsets.CIFAR100(root=root, train=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False, num_workers=8)

    full_train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=8)
    for data, targets in full_train_loader:
        traindata, trainlabels = data, targets.long()
    partialY, avgC = generate_instance_independent_candidate_labels(trainlabels, partial_rate=partial_rate)
    print('Average candidate num: ', avgC)

    partial_matrix_dataset = CIFAR100_Augmentention(traindata, partialY.float(), trainlabels.float())
    partial_matrix_train_loader = torch.utils.data.DataLoader(dataset=partial_matrix_dataset, 
                                                                batch_size=batch_size, 
                                                                shuffle=True, 
                                                                num_workers=8,
                                                                drop_last=True)
    dim = 32 * 32 * 3
    K = 100
    return partial_matrix_train_loader, valid_loader, test_loader, dim, K


class CIFAR100_Augmentention(Dataset):
    def __init__(self, images, given_label_matrix, true_labels):
        self.images = images
        self.given_label_matrix = given_label_matrix
        # user-defined label (partial labels)
        self.true_labels = true_labels

        # PLCR
        self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
                ])
        self.weak_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4, padding_mode='reflect'),
                    # transforms.ToTensor(),
                    Cutout(n_holes=1, length=16),
                    transforms.ToPILImage(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])
        self.strong_transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, 4, padding_mode='reflect'),
                    # transforms.ToTensor(),
                    Cutout(n_holes=1, length=16),
                    transforms.ToPILImage(),
                    CIFAR10Policy(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
                ])


    def __len__(self):
        return len(self.true_labels)
        
    def __getitem__(self, index):
        each_image_o = self.transform(self.images[index])
        each_image_w = self.weak_transform(self.images[index])
        each_image_s = self.strong_transform(self.images[index])
        each_label = self.given_label_matrix[index]
        each_true_label = self.true_labels[index]
        
        return each_image_o, each_image_w, each_image_s, each_label, each_true_label, index

