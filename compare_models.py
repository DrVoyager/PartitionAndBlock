import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

if torch.cuda.is_available():
    device = torch.device('cuda')
elif torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)

print('='*60)
print('EVALUATING BOTH MODELS')
print('='*60)

print('\n1. Evaluating Baseline MobileNet V1...')
print('-'*60)
from train_baseline import MobileNetV1
baseline_model = MobileNetV1(num_classes=10).to(device)
baseline_checkpoint = torch.load('checkpoints/baseline_mobilenetv1.pth', map_location=device)
baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
baseline_model.eval()
baseline_correct = 0
baseline_total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = baseline_model(data)
        _, predicted = output.max(1)
        baseline_total += target.size(0)
        baseline_correct += predicted.eq(target).sum().item()
baseline_acc = 100. * baseline_correct / baseline_total
print(f'Baseline MobileNet V1 - Epoch {baseline_checkpoint["epoch"]}')
print(f'Accuracy: {baseline_acc:.2f}% ({baseline_correct}/{baseline_total})')

print('\n2. Evaluating P&B Model...')
print('-'*60)
from mobilenet_v1 import PartitionAndBlockingModel
pb_model = PartitionAndBlockingModel(num_classes=10).to(device)
pb_checkpoint = torch.load('checkpoints/best_model.pth', map_location=device)
pb_model.load_state_dict(pb_checkpoint['model_state_dict'])
pb_model.eval()
pb_correct = 0
pb_total = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = pb_model(data)
        _, predicted = output.max(1)
        pb_total += target.size(0)
        pb_correct += predicted.eq(target).sum().item()
pb_acc = 100. * pb_correct / pb_total
print(f'P&B Model - Epoch {pb_checkpoint["epoch"]}')
print(f'Accuracy: {pb_acc:.2f}% ({pb_correct}/{pb_total})')

print('\n' + '='*60)
print('COMPARISON SUMMARY')
print('='*60)
print(f'Baseline MobileNet V1: {baseline_acc:.2f}%')
print(f'P&B Model:             {pb_acc:.2f}%')
print(f'Difference:            {baseline_acc - pb_acc:.2f}%')
print('='*60)
