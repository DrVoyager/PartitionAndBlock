import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model import MobileNetV1


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if (batch_idx + 1) % 100 == 0:
            print(f'  Batch [{batch_idx + 1}/{len(train_loader)}], '
                  f'Loss: {loss.item():.4f}, '
                  f'Acc: {100. * correct / total:.2f}%')

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    return 100. * correct / total


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print('Loading CIFAR-10 dataset...')
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    print('\nInitializing MobileNet V1...')
    model = MobileNetV1(num_classes=10).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    num_epochs = 76
    best_acc = 0.0

    if os.path.exists('../checkpoints/baseline_mobilenetv1.pth'):
        checkpoint = torch.load('../checkpoints/baseline_mobilenetv1.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_acc = checkpoint.get('val_acc', 0.0)
        start_epoch = checkpoint.get('epoch', 0)
        print(f'Loaded checkpoint from epoch {start_epoch} with accuracy {best_acc:.2f}%')
    else:
        start_epoch = 0

    print('\nTraining baseline MobileNet V1...')
    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch [{epoch + 1}/{num_epochs}]')
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = validate(model, test_loader, device)

        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Acc: {val_acc:.2f}%')

        if val_acc > best_acc:
            best_acc = val_acc

    os.makedirs('../checkpoints', exist_ok=True)
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
    }, '../checkpoints/baseline_mobilenetv1.pth')

    print(f'\nTraining completed!')
    print(f'Best validation accuracy: {best_acc:.2f}%')
    print(f'Final validation accuracy: {val_acc:.2f}%')


if __name__ == '__main__':
    main()
