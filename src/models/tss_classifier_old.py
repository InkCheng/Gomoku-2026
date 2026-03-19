import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os
import argparse
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

class ResBlock(nn.Module):
    def __init__(self, num_filters=256):
        super().__init__()
        self.conv1 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_filters)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class TSSClassifier(nn.Module):
    def __init__(self, in_channels=7, num_channels=128, num_res_blocks=7, board_size=15):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_channels),
            nn.ReLU(inplace=True)
        )
        self.res_blocks = nn.Sequential(*[ResBlock(num_channels) for _ in range(num_res_blocks)])
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, 1)
        )
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.training_data = []

    def forward(self, x):
        x = self.conv_block(x)
        x = self.res_blocks(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def collect_training_data(self, state, is_tss):
        self.training_data.append((state, float(is_tss)))
        if len(self.training_data) > 10000:  # 限制训练数据的数量
            self.training_data.pop(0)

class TSSDataset(Dataset):
    def __init__(self, cache_file):
        print("Loading cached data...")
        with open(cache_file, 'rb') as f:
            self.data = pickle.load(f)
        print(f"Dataset size: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def train(model, train_loader, optimizer, criterion, device, scaler):
    model.train()
    total_loss = 0
    print("开始训练循环...")
    with tqdm(train_loader, desc="训练") as pbar:
        for i, (data, target) in enumerate(pbar):
            if i % 100 == 0:  # 每100个批次打印一次
                print(f"批次 {i+1}/{len(train_loader)}")
            data, target = data.to(device), target.to(device).float()
            optimizer.zero_grad()
            try:
                with autocast():
                    output = model(data)
                    loss = criterion(output, target.unsqueeze(1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            except Exception as e:
                print(f"训练循环中出错: {e}")
                raise e
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
    return total_loss / len(train_loader)

def evaluate(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        with tqdm(test_loader, desc="Evaluating") as pbar:
            for data, target in pbar:
                data, target = data.to(device), target.to(device).float()
                with autocast():
                    output = model(data)
                    test_loss += criterion(output, target.unsqueeze(1)).item()
                pred = torch.sigmoid(output) > 0.5
                correct += pred.eq(target.view_as(pred)).sum().item()
                pbar.set_postfix({'loss': test_loss / (pbar.n + 1)})
    
    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)')
    return test_loss, accuracy

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
    
    dataset = TSSDataset(args.cache_file)
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=True)
    
    model = TSSClassifier(num_channels=args.num_channels, num_res_blocks=args.num_res_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    
    writer = SummaryWriter(log_dir=args.log_dir)
    
    best_loss = float('inf')
    no_improve_count = 0

    for epoch in range(args.epochs):
        print(f'轮次 {epoch+1}/{args.epochs}')
        train_loss = train(model, train_loader, optimizer, criterion, device, scaler)
        if epoch % 1 == 0:  # 每5个轮次评估一次
            test_loss, accuracy = evaluate(model, test_loader, criterion, device)
            
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/test', test_loss, epoch)
            writer.add_scalar('Accuracy/test', accuracy, epoch)
            
            scheduler.step(test_loss)
            
            if test_loss < best_loss:
                best_loss = test_loss
                torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_tss_classifier.pth'))
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            if no_improve_count >= args.patience:
                print("触发早停。")
                break
            
            if optimizer.param_groups[0]['lr'] < 1e-6:
                print("学习率过小，停止训练。")
                break

    torch.save(model.state_dict(), os.path.join(args.save_dir, 'final_tss_classifier_2.pth'))
    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TSS Classifier Training')
    parser.add_argument('--cache_file', type=str, default='tss_data_cache.pkl', help='Path to the cache file')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size for training')
    parser.add_argument('--num_channels', type=int, default=128, help='Number of channels in the model')
    parser.add_argument('--num_res_blocks', type=int, default=7, help='Number of residual blocks')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs to train')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save model checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs', help='Directory to save tensorboard logs')
    
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main(args)