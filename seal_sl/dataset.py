import torch
import torchvision
from torchvision.datasets import ImageFolder

def CIFAR10(data_dir='./data', validation_set=True):

    # Loading the CIFAR-10 training and testing sets.
    training = torchvision.datasets.CIFAR10(
        train=True, root=data_dir, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    )

    testing = torchvision.datasets.CIFAR10(
        train=False, root=data_dir, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
    )    

    validation = None

    if validation_set is True:  # If we want to create a validation set.
        validation = torchvision.datasets.CIFAR10(
            train=True, root=data_dir, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
            ])
        )

        data_size = len(training)

        training, _ = torch.utils.data.random_split(
            training,
            [int(data_size * 0.9), data_size - int(data_size * 0.9)],
            torch.Generator().manual_seed(42),
        )
        training_temp, validation = torch.utils.data.random_split(
            validation,
            [int(data_size * 0.9), data_size - int(data_size * 0.9)],
            torch.Generator().manual_seed(42),
        )
        training_warmup, validation_warmup = torch.utils.data.random_split(
            training_temp,
            [int(data_size * 0.9 * 0.9), int(data_size * 0.9) - int(data_size * 0.9 * 0.9)],
            torch.Generator().manual_seed(42),
        )

    return training, validation, testing, training_warmup, validation_warmup


def CIFAR100(data_dir='./data', validation_set=True):

    # Loading the CIFAR-100 training and testing sets.
    training = torchvision.datasets.CIFAR100(
        train=True, root=data_dir, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.RandomCrop(32, padding=4),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    )

    testing = torchvision.datasets.CIFAR100(
        train=False, root=data_dir, download=True,
        transform=torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
    )

    validation = None

    if validation_set is True:  # If we want to create a validation set.
        validation = torchvision.datasets.CIFAR100(
            train=True, root=data_dir, download=True,
            transform=torchvision.transforms.Compose([
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
            ])
        )

        data_size = len(training)

        training, _ = torch.utils.data.random_split(
            training,
            [int(data_size * 0.9), data_size - int(data_size * 0.9)],
            torch.Generator().manual_seed(42),
        )
        training_temp, validation = torch.utils.data.random_split(
            validation,
            [int(data_size * 0.9), data_size - int(data_size * 0.9)],
            torch.Generator().manual_seed(42),
        )
        training_warmup, validation_warmup = torch.utils.data.random_split(
            training_temp,
            [int(data_size * 0.9 * 0.9), int(data_size * 0.9) - int(data_size * 0.9 * 0.9)],
            torch.Generator().manual_seed(42),
        )

    return training, validation, testing, training_warmup, validation_warmup


def TinyImageNet(data_dir='./data', size=32, validation_set=True):
    if size==64:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomRotation(20),
            torchvision.transforms.RandomHorizontalFlip(0.5),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
        ])
    elif size==32:
        train_transform = torchvision.transforms.Compose([
            torchvision.transforms.RandomResizedCrop(32),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        test_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(32),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

    train_root = data_dir + '/tiny-imagenet-200/train/'
    val_root = data_dir + '/tiny-imagenet-200/val/'

    training = ImageFolder(root=train_root, transform=train_transform)
    testing = ImageFolder(root=val_root, transform=test_transform)
    validation = None

    if validation_set is True:  # If we want to create a validation set.
        validation = ImageFolder(root=train_root, transform=test_transform)
        data_size = len(training)

        training, _ = torch.utils.data.random_split(
            training,
            [int(data_size * 0.9), data_size - int(data_size * 0.9)],
            torch.Generator().manual_seed(42),
        )
        training_temp, validation = torch.utils.data.random_split(
            validation,
            [int(data_size * 0.9), data_size - int(data_size * 0.9)],
            torch.Generator().manual_seed(42),
        )
        training_warmup, validation_warmup = torch.utils.data.random_split(
            training_temp,
            [int(data_size * 0.9 * 0.9), int(data_size * 0.9) - int(data_size * 0.9 * 0.9)],
            torch.Generator().manual_seed(42),
        )

    return training, validation, testing, training_warmup, validation_warmup