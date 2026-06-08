import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os


class MobileNetV1(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV1, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            self._make_depthwise(32, 64, stride=1),
            self._make_depthwise(64, 128, stride=2),
            self._make_depthwise(128, 128, stride=1),
            self._make_depthwise(128, 256, stride=2),
            self._make_depthwise(256, 256, stride=1),
            self._make_depthwise(256, 512, stride=2),
            self._make_depthwise(512, 512, stride=1),
            self._make_depthwise(512, 512, stride=1),
            self._make_depthwise(512, 512, stride=1),
            self._make_depthwise(512, 512, stride=1),
            self._make_depthwise(512, 1024, stride=2),
            self._make_depthwise(1024, 1024, stride=1),
            
            nn.AdaptiveAvgPool2d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, num_classes)
        )
    
    def _make_depthwise(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, 
                     padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, 
                     padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


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


def validate(model, val_loader, criterion, device):
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
    
    epoch_acc = 100. * correct / total
    return epoch_acc


def main():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Using device: {device}')
    print('='*60)
    
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
    
    print('\nInitializing standard MobileNet V1...')
    model = MobileNetV1(num_classes=10).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Model initialized. Total parameters: {total_params:,}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
    
    num_epochs = 76
    best_acc = 0.0
    
    if os.path.exists('checkpoints/baseline_mobilenetv1.pth'):
        checkpoint = torch.load('checkpoints/baseline_mobilenetv1.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        best_acc = checkpoint['val_acc']
        start_epoch = checkpoint['epoch']
        print(f'Loaded existing checkpoint from epoch {start_epoch} with accuracy {best_acc:.2f}%')
    else:
        start_epoch = 0
    
    print('\n' + '='*60)
    print('TRAINING BASELINE MOBILENET V1')
    print('='*60)
    print(f'Starting epoch {start_epoch + 1} of {num_epochs}...')
    print('-'*60)
    
    for epoch in range(start_epoch, num_epochs):
        print(f'\nEpoch [{epoch + 1}/{num_epochs}]')
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = validate(model, test_loader, criterion, device)
        
        print(f'\nEpoch [{epoch + 1}/{num_epochs}] Summary:')
        print(f'  Train Acc: {train_acc:.2f}%')
        print(f'  Val Acc: {val_acc:.2f}%')
        
        if val_acc > best_acc:
            best_acc = val_acc
    
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'val_acc': val_acc,
    }, 'checkpoints/baseline_mobilenetv1.pth')
    
    print(f'\nTraining completed!')
    print(f'Baseline MobileNet V1 accuracy: {val_acc:.2f}%')


if __name__ == '__main__':
    main()