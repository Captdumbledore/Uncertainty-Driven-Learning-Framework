import torch


def evaluate_model(
    model,
    test_loader,
    device="cpu"
):
    """
    Evaluate a trained model on the test dataset.

    Returns test loss and accuracy.
    """

    model = model.to(device)
    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in test_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

    test_loss = total_loss / len(test_loader)
    test_accuracy = correct / total

    print(
        f"Test Loss: {test_loss:.4f}"
    )

    print(
        f"Test Accuracy: {test_accuracy:.4f}"
    )

    return {
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
    }