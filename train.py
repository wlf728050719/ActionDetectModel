import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
from net.st_gcn import Model
from data_loader import get_dataloader


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, save_path):
    """绘制训练和验证的loss和accuracy曲线"""
    plt.figure(figsize=(12, 5))

    # Loss曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Training curves saved to {save_path}")

def train_epoch(model, loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    with tqdm(loader, desc=f'Epoch {epoch + 1} [Train]', unit='batch') as t:
        for data, labels in t:
            data = data.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(data)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计信息
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 更新进度条
            t.set_postfix(loss=loss.item(), acc=correct / total)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        with tqdm(loader, desc='[Validation]', unit='batch') as t:
            for data, labels in t:
                data = data.to(device)
                labels = labels.to(device)

                outputs = model(data)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                t.set_postfix(loss=loss.item(), acc=correct / total)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(device)
            labels = labels.to(device)

            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print(f'Test Accuracy: {accuracy:.2%}')
    return accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOPose+STGCN Training')
    parser.add_argument('--config_path',type=str,default='config.json',help='config file path')
    parser.add_argument('--batch_size',type=int,default=32,help='batch size')
    parser.add_argument('--epochs',type=int,default=100,help='number of epochs')
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
    parser.add_argument('--layout',type=str,default='yolopose',help='layout of the dataset:openpose,ntu-rgb+d,ntu_edge,yolopose')
    parser.add_argument('--strategy',type=str,default='spatial',help='strategy of training:uniform,distance,spatial')
    parser.add_argument('--model_save_dir',type=str,default='./checkpoints',help='model save path')
    args = parser.parse_args()
    # 训练参数
    CONFIG_PATH = args.config_path
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    LAYOUT = args.layout
    STRATEGY = args.strategy
    MODEL_SAVE_DIR = args.model_save_dir+'/'+LAYOUT+'/'+STRATEGY
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1.获取数据加载器
    num_class, train_loader, val_loader = get_dataloader(CONFIG_PATH, BATCH_SIZE)
    # 2. 初始化模型
    in_channels = 2  # (x,y)坐标
    graph_args = {'layout': LAYOUT, 'strategy': STRATEGY}
    edge_importance_weighting = True
    model = Model(in_channels, num_class, graph_args, edge_importance_weighting).to(device)

    # 3. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    # 4. 训练循环
    best_val_loss = float('inf')
    best_val_acc = 0.0

    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    for epoch in range(EPOCHS):
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, epoch)
        # 验证
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step(val_loss)

        # 绘制并打印epoch结果
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:  # 每2个epoch或最后保存一次
            plot_training_curves(train_loss_history, val_loss_history,
                                 train_acc_history, val_acc_history,
                                 os.path.join(MODEL_SAVE_DIR, 'training_curves.png'))
        print(f'Epoch {epoch + 1}/{EPOCHS}: '
              f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2%} | '
              f'Val Loss: {val_loss:.4f}, Acc: {val_acc:.2%}')

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'acc': val_acc,
            }, os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))
            print(f"Saved best model with val_loss: {val_loss:.4f}, val_acc: {val_acc:.2%}")

        # 保存最新模型
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            'acc': val_acc,
        }, os.path.join(MODEL_SAVE_DIR, 'last_model.pth'))

    print(f'Training finished. Best val acc: {best_val_acc:.2%}')
