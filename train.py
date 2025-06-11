import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import argparse
import matplotlib.pyplot as plt
from net.st_gcn import Model
from data_loader import get_dataloader

class TripletLoss(nn.Module):
    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        distance_positive = (anchor - positive).pow(2).sum(1)
        distance_negative = (anchor - negative).pow(2).sum(1)
        losses = torch.relu(distance_positive - distance_negative + self.margin)
        return losses.mean()

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.device = device
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, x, labels):
        centers_batch = self.centers[labels]
        distance = (x - centers_batch).pow(2).sum(dim=1)
        loss = distance.mean()
        return loss


def plot_training_metrics(train_losses, val_losses, train_accs, val_accs, save_dir):
    """绘制训练和验证指标曲线"""
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(12, 15))

    # Training Losses
    plt.subplot(3, 1, 1)
    plt.plot(train_losses['total_loss'], label='Train Total Loss')
    plt.plot(train_losses['weighting_triplet_loss'], label='Weighting Train Triplet Loss')
    plt.plot(train_losses['weighting_center_loss'], label='Weighting Train Center Loss')
    plt.title('Training Losses')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Validation Loss
    plt.subplot(3, 1, 2)
    plt.plot(val_losses['center_loss'], label='Validation Center Loss', color='red')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Accuracy Curves
    plt.subplot(3, 1, 3)
    plt.plot(train_accs['triplet'], label='Train Triplet Acc')
    plt.plot(val_accs['classification'], label='Validation Class Acc')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'))
    plt.close()

def calculate_triplet_accuracy(anchor, positive, negative, margin):
    with torch.no_grad():
        d_ap = torch.norm(anchor - positive, dim=1)
        d_an = torch.norm(anchor - negative, dim=1)
        correct = torch.sum(d_ap + margin < d_an).item()
        total = anchor.size(0)
        return correct / total

def calculate_classification_accuracy(features, labels, centers):
    with torch.no_grad():
        distances = torch.cdist(features, centers)
        preds = torch.argmin(distances, dim=1)
        correct = torch.sum(preds == labels).item()
        total = labels.size(0)
        return correct / total

def train_epoch(model, loader, optimizer, triplet_criterion, center_criterion,
                epoch, device, margin, lambda_center_loss, lambda_triplet_loss):
    model.train()
    total_loss = 0.0
    total_weighting_triplet_loss = 0.0
    total_weighting_center_loss = 0.0
    total_triplet_acc = 0.0
    total_batches = 0

    with tqdm(loader, desc=f'Epoch {epoch + 1} [Train]', unit='batch') as t:
        for (anchors, positives, negatives), batch_labels in t:
            anchors = anchors.to(device)
            positives = positives.to(device)
            negatives = negatives.to(device)
            batch_labels = batch_labels.to(device)

            anchor_emb = model(anchors)
            positive_emb = model(positives)
            negative_emb = model(negatives)

            triplet_loss = triplet_criterion(anchor_emb, positive_emb, negative_emb)
            center_loss = center_criterion(anchor_emb, batch_labels)
            total_batch_loss = lambda_triplet_loss * triplet_loss + lambda_center_loss * center_loss

            triplet_acc = calculate_triplet_accuracy(anchor_emb, positive_emb, negative_emb, margin)

            optimizer.zero_grad()
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item()
            total_weighting_triplet_loss += triplet_loss.item() * lambda_triplet_loss
            total_weighting_center_loss += center_loss.item() *lambda_center_loss
            total_triplet_acc += triplet_acc
            total_batches += 1

            t.set_postfix(
                loss=total_batch_loss.item(),
                triplet_loss=triplet_loss.item(),
                c_loss=center_loss.item(),
                acc=triplet_acc
            )

    metrics = {
        'total_loss': total_loss / total_batches,
        'weighting_triplet_loss': total_weighting_triplet_loss / total_batches,
        'weighting_center_loss': total_weighting_center_loss / total_batches,
        'triplet_acc': total_triplet_acc / total_batches
    }
    return metrics

def validate(model, loader, center_criterion, device, centers):
    model.eval()
    total_center_loss = 0.0
    total_class_acc = 0.0
    total_batches = 0

    with torch.no_grad():
        with tqdm(loader, desc='[Validation]', unit='batch') as t:
            for data, labels in t:
                data = data.to(device)
                labels = labels.to(device)
                embeddings = model(data)
                center_loss = center_criterion(embeddings, labels)
                class_acc = calculate_classification_accuracy(embeddings, labels, centers)
                total_center_loss += center_loss.item()
                total_class_acc += class_acc
                total_batches += 1
                t.set_postfix(c_loss=center_loss.item(), c_acc=class_acc)

    metrics = {
        'center_loss': total_center_loss / total_batches,
        'classification_acc': total_class_acc / total_batches
    }
    return metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STGCN Training with Triplet and Center Loss')
    #配置
    parser.add_argument('--config_path', type=str, default='config.json', help='config file path')

    #核心
    parser.add_argument('--max_frames', type=int, default=50,help='max number of frames')
    parser.add_argument('--embed_dim', type=int, default=1024, help='embedding dimension')
    parser.add_argument('--data_load_mode', type=str, default='slide', help='data load mode,single or slide')
    parser.add_argument('--slide_stride', type=int, default=25, help='slide stride')
    parser.add_argument('--start_frame', type=str, default='first_nonzero', help='start frame,first or first_nonzero')

    #损失相关
    parser.add_argument('--triplet_loss_margin', type=float, default=0.8, help='margin for triplet loss')
    parser.add_argument('--lambda_center_loss', type=float, default=0.001, help='weight for center loss')
    parser.add_argument('--lambda_triplet_loss', type=float, default=1.0, help='weight for triplet loss')

    #骨骼结构定义
    parser.add_argument('--layout', type=str, default='yolopose',
                      help='layout of the dataset: openpose, ntu-rgb+d, ntu_edge, yolopose')
    parser.add_argument('--strategy', type=str, default='spatial',
                      help='strategy of training: uniform, distance, spatial')

    #训练相关
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--epochs', type=int, default=400, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--model_save_dir', type=str, default='./checkpoints', help='model save path')

    args = parser.parse_args()

    CONFIG_PATH = args.config_path

    MAX_FRAMES = args.max_frames
    EMBED_DIM = args.embed_dim
    DATA_LOAD_MODE = args.data_load_mode
    SLIDE_STRIDE = args.slide_stride
    START_FRAME = args.start_frame

    TRIPLET_LOSS_MARGIN = args.triplet_loss_margin
    LAMBDA_CENTER_LOSS = args.lambda_center_loss
    LAMBDA_TRIPLET_LOSS = args.lambda_triplet_loss

    LAYOUT = args.layout
    STRATEGY = args.strategy

    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    MODEL_SAVE_DIR = args.model_save_dir+'/'+LAYOUT+'/'+ STRATEGY
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Triplet Loss Margin: {TRIPLET_LOSS_MARGIN}, Center Loss Weight: {LAMBDA_CENTER_LOSS}, Triplet Weight: {LAMBDA_TRIPLET_LOSS}")

    num_classes, train_loader, val_loader = get_dataloader(config_path=CONFIG_PATH,max_frames=MAX_FRAMES,batch_size=BATCH_SIZE,
                                                           mode=DATA_LOAD_MODE,slide_stride=SLIDE_STRIDE,start_frame=START_FRAME)

    in_channels = 2  # (x,y)坐标
    graph_args = {'layout': LAYOUT, 'strategy': STRATEGY}
    edge_importance_weighting = True

    model = Model(in_channels, EMBED_DIM, graph_args, edge_importance_weighting).to(device)
    triplet_criterion = TripletLoss(margin=TRIPLET_LOSS_MARGIN)
    center_criterion = CenterLoss(num_classes, EMBED_DIM, device)

    params = list(model.parameters()) + list(center_criterion.parameters())
    optimizer = optim.Adam(params, lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)

    best_val_acc = 0.0

    train_losses = {
        'total_loss': [],
        'weighting_triplet_loss': [],
        'weighting_center_loss': []
    }
    train_accs = {
        'triplet': []
    }
    val_losses = {
        'center_loss': []
    }
    val_accs = {
        'classification': []
    }

    for epoch in range(EPOCHS):
        train_results = train_epoch(
            model, train_loader, optimizer, triplet_criterion, center_criterion,
            epoch, device, TRIPLET_LOSS_MARGIN, LAMBDA_CENTER_LOSS, LAMBDA_TRIPLET_LOSS
        )
        val_results = validate(
            model, val_loader, center_criterion,
            device, center_criterion.centers
        )
        scheduler.step(val_results['center_loss'])

        train_losses['total_loss'].append(train_results['total_loss'])
        train_losses['weighting_triplet_loss'].append(train_results['weighting_triplet_loss'])
        train_losses['weighting_center_loss'].append(train_results['weighting_center_loss'])
        train_accs['triplet'].append(train_results['triplet_acc'])
        val_losses['center_loss'].append(val_results['center_loss'])
        val_accs['classification'].append(val_results['classification_acc'])

        print(f'\nEpoch {epoch + 1}/{EPOCHS}:')
        print(f'Train - Total Loss: {train_results["total_loss"]:.4f}, '
              f'Weighting Triplet Loss: {train_results["weighting_triplet_loss"]:.4f}, '
              f'Weighting Center Loss: {train_results["weighting_center_loss"]:.4f}, '
              f'Triplet Acc: {train_results["triplet_acc"]:.2%}')
        print(f'Val - Center Loss: {val_results["center_loss"]:.4f}, '
              f'Class Acc: {val_results["classification_acc"]:.2%}')

        if (epoch + 1) % 2 == 0 or epoch == EPOCHS - 1:
            plot_training_metrics(train_losses, val_losses, train_accs, val_accs, MODEL_SAVE_DIR)
            torch.save({
                'epoch': epoch,
                'max_frames': MAX_FRAMES,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'center_state_dict': center_criterion.state_dict(),
                'train_metrics': train_results,
                'val_metrics': val_results,
                'triplet_loss_margin': TRIPLET_LOSS_MARGIN,
                'lambda_center_loss': LAMBDA_CENTER_LOSS,
                'lambda_triplet_loss': LAMBDA_TRIPLET_LOSS,
                'layout': LAYOUT,
                'strategy': STRATEGY,
                'embedding_dim': EMBED_DIM
            }, os.path.join(MODEL_SAVE_DIR, 'last_model.pth'))

        if val_results['classification_acc'] > best_val_acc:
            best_val_acc = val_results['classification_acc']
            torch.save({
                'epoch': epoch,
                'max_frames': MAX_FRAMES,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'center_state_dict': center_criterion.state_dict(),
                'train_metrics': train_results,
                'val_metrics': val_results,
                'triplet_loss_margin': TRIPLET_LOSS_MARGIN,
                'lambda_center_loss': LAMBDA_CENTER_LOSS,
                'lambda_triplet_loss': LAMBDA_TRIPLET_LOSS,
                'layout': LAYOUT,
                'strategy': STRATEGY,
                'embedding_dim': EMBED_DIM
            }, os.path.join(MODEL_SAVE_DIR, 'best_model.pth'))
            print(f"Saved best model with val_class_acc: {best_val_acc:.2%}")

    print(f'\nTraining finished. Best val class acc: {best_val_acc:.2%}')
