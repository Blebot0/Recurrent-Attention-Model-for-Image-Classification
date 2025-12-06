from torchvision import datasets, transforms


def get_mnist_datasets(data_dir="./data", image_size=28):
    """
    Get standard MNIST datasets.

    Args:
        data_dir: Directory to store data
        image_size: Size to resize images to

    Returns:
        train_dataset, test_dataset
    """
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )

    train_dataset = datasets.MNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset


