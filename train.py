import torch
import torch.nn as nn
import torch.optim as optim


def train_model(
    model,
    train_loader,
    validation_loader,
    num_epochs=10,
    learning_rate=0.001,
):
    
    # Automatically use GPU if available
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Using device:", device)

    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    for epoch in range(num_epochs):

        # -------------------------
        # TRAINING
        # -------------------------
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            # Clear previous gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

            # Calculate training accuracy
            predictions = torch.argmax(
                outputs,
                dim=1
            )

            total += labels.size(0)

            correct += (
                predictions == labels
            ).sum().item()

        train_loss = running_loss / len(train_loader)

        train_accuracy = 100 * correct / total

        # -------------------------
        # VALIDATION
        # -------------------------
        model.eval()

        validation_correct = 0
        validation_total = 0

        with torch.no_grad():

            for images, labels in validation_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                validation_total += labels.size(0)

                validation_correct += (
                    predictions == labels
                ).sum().item()

        validation_accuracy = (
            100
            * validation_correct
            / validation_total
        )

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Accuracy: {train_accuracy:.2f}% | "
            f"Validation Accuracy: {validation_accuracy:.2f}%"
        )

    return model