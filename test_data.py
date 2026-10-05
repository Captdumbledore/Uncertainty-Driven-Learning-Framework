from data import get_dataloaders


def main():
    train_loader, validation_loader, test_loader, train_dataset = get_dataloaders(
        batch_size=64
    )

    images, labels = next(iter(train_loader))

    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)

    print("Number of training batches:", len(train_loader))
    print("Number of validation batches:", len(validation_loader))
    print("Number of test batches:", len(test_loader))


if __name__ == "__main__":
    main()