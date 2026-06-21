import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import os
from mobilenet_v1 import (
    DEPTHWISE_BLOCKS,
    SPLIT_PRESETS,
    PartitionAndBlockingModel,
    split_children_from_depthwise_blocks,
)


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
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    return epoch_loss, epoch_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-name", type=str, default="best_model.pth")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint-dir/checkpoint-name if it exists.")
    parser.add_argument("--split-preset", choices=tuple(SPLIT_PRESETS), default="stem",
                        help="Named MobileNetV1 split point. stem means before the first depthwise block; dwN means after the Nth depthwise block.")
    parser.add_argument("--split-after-depthwise", type=int, default=None,
                        help=f"Split after the Nth MobileNetV1 depthwise block, from 1 to {DEPTHWISE_BLOCKS}.")
    parser.add_argument("--split-children", type=int, default=None,
                        help="Low-level override: number of spatial MobileNetV1 feature children in the P&B client.")
    parser.add_argument("--mask-side", type=int, default=None,
                        help="Override the centered protected spatial mask side length. Omit to use the default split-dependent size.")
    parser.add_argument("--require-cuda", action="store_true", help="Stop before running if CUDA is not available.")
    args = parser.parse_args()

    if args.split_after_depthwise is not None:
        args.split_children = split_children_from_depthwise_blocks(args.split_after_depthwise)
    elif args.split_children is None:
        args.split_children = SPLIT_PRESETS[args.split_preset]

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    if args.require_cuda and device.type != 'cuda':
        raise SystemExit('ERROR: CUDA is required for this run, but torch.cuda.is_available() is false.')
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
    train_dataset = datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform_test)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    print('\nInitializing Partition and Blocking model...')
    print(f'Split preset: {args.split_preset}')
    if args.split_after_depthwise is not None:
        print(f'Split after depthwise block: {args.split_after_depthwise}')
    print(f'Client split children: {args.split_children}')
    print(f'Mask side: {args.mask_side if args.mask_side is not None else "default"}')
    model = PartitionAndBlockingModel(
        num_classes=10,
        split_children=args.split_children,
        mask_side=args.mask_side,
    ).to(device)
    
    with torch.no_grad():
        model.eval()
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        dummy_output = model(dummy_input)
        model.train()
    print(f'Model initialized. Output shape: {dummy_output.shape}')
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    
    num_epochs = args.epochs
    best_acc = 0.0
    checkpoint_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    
    if args.resume and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        best_acc = checkpoint['best_acc']
        start_epoch = checkpoint['epoch']
        print(f'Loaded existing checkpoint from epoch {start_epoch} with accuracy {best_acc:.2f}%')
    else:
        start_epoch = 0
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    if not 'start_epoch' in dir():
        start_epoch = 0
    
    print('\nStarting training...')
    for epoch in range(start_epoch, num_epochs):
        start_time = time.time()
        
        print(f'\nEpoch [{epoch + 1}/{num_epochs}]')
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, test_loader, criterion, device)
        
        scheduler.step(val_acc)
        
        epoch_time = time.time() - start_time
        
        print(f'\nEpoch [{epoch + 1}/{num_epochs}] Summary:')
        print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        print(f'  Time: {epoch_time:.2f}s')
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'split_children': args.split_children,
                'split_preset': args.split_preset,
                'split_after_depthwise': args.split_after_depthwise,
                'mask_side': args.mask_side,
            }, checkpoint_path)
            print(f'  Best model saved with accuracy: {best_acc:.2f}%')
    
    print(f'\nTraining completed!')
    print(f'Best validation accuracy: {best_acc:.2f}%')


if __name__ == '__main__':
    main()
