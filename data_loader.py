from torchvision import datasets, transforms
from torch.utils.data import DataLoader

training_data = datasets.MNIST(
    root='data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_data = datasets.MNIST(
    root='data',
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

def get_mnist_loaders(batch_size: int):
    training_loader = DataLoader(
        dataset=training_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    return (
        training_loader,
        test_loader
    )
