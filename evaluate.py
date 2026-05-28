import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.model import PartitionAndBlockingModel


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * 10
    class_total = [0] * 10

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

            c = (predicted == target).squeeze()
            for i in range(len(target)):
                label = target[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    overall_acc = 100. * correct / total
    print(f'\nOverall Accuracy: {overall_acc:.2f}% ({correct}/{total})')

    print('\nPer-class Accuracy:')
    for i in range(10):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            print(f'  {classes[i]:10s}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})')

    return overall_acc


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    print('Loading CIFAR-10 test dataset...')
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

    print('Loading trained model...')
    model = PartitionAndBlockingModel(num_classes=10).to(device)

    checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Model loaded from epoch {checkpoint["epoch"]} with best accuracy: {checkpoint["best_acc"]:.2f}%')

    print('\nEvaluating model on test set...')
    test_acc = evaluate_model(model, test_loader, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'\nModel parameters:')
    print(f'  Total parameters: {total_params:,}')
    print(f'  Trainable parameters: {trainable_params:,}')


if __name__ == '__main__':
    main()
