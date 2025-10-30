import os
import sys
import argparse
import time
from datetime import datetime
from sklearn.metrics import roc_auc_score

import numpy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from scipy.constants import precision
from sklearn.metrics import precision_score, recall_score, f1_score

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from conf import settings
from utils import get_network, get_training_dataloader, get_test_dataloader, WarmUpLR, \
    most_recent_folder, most_recent_weights, last_epoch, best_acc_weights, get_training_dataloader_TCMEyes, get_test_dataloader_TCMEyes

def train(epoch):

    start = time.time()
    net.train()
    for batch_index, (images, labels) in enumerate(TCMEyes_training_loader):
        labels = labels.to(device)
        images = images.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(TCMEyes_training_loader) + batch_index + 1

        last_layer = list(net.children())[-1]
        for name, para in last_layer.named_parameters():
            if 'weight' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_weights', para.grad.norm(), n_iter)
            if 'bias' in name:
                writer.add_scalar('LastLayerGradients/grad_norm2_bias', para.grad.norm(), n_iter)

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * args.b + len(images),
            total_samples=len(TCMEyes_training_loader.dataset)
        ))

        #update training loss for each iteration
        writer.add_scalar('Train/loss', loss.item(), n_iter)

        if epoch <= args.warm:
            warmup_scheduler.step()

    for name, param in net.named_parameters():
        layer, attr = os.path.splitext(name)
        attr = attr[1:]
        writer.add_histogram("{}/{}".format(layer, attr), param, epoch)

    finish = time.time()

    print('epoch {} training time consumed: {:.2f}s'.format(epoch, finish - start))

@torch.no_grad()
def eval_training(epoch=0, tb=True):
    start = time.time()
    net.eval()
    test_loss = 0.0 # cost function error
    correct = 0.0
    all_preds = []
    all_labels = []
    all_probs = []  # 新增：收集预测概率
    for (images, labels) in TCMEyes_test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = net(images)
        loss = loss_function(outputs, labels)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()
        all_preds.append(preds.cpu().numpy().reshape(-1))
        all_labels.append(labels.cpu().numpy().reshape(-1))
        # 收集预测概率（用于AUC计算）
        probs = torch.softmax(outputs, dim=1)  # 多分类概率
        all_probs.append(probs.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)  # 形状: [N, C]
    # 检查数据类型
    print(f"Predictions dtype: {all_preds.dtype}, Labels dtype: {all_labels.dtype}")
    print(f"Unique labels: {np.unique(all_labels)}")

    accuracy = correct / len(all_preds)
    # precision = precision_score(all_labels, all_preds, average='weighted')
    # recall = recall_score(all_labels, all_preds, average='weighted')
    # f1 = f1_score(all_labels, all_preds, average='weighted')
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    # auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    auc = roc_auc_score(all_labels, all_probs[:, 1])  # 使用正类的概率
    
    finish = time.time()
    if args.gpu is not None:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')
    print('Evaluating Network.....')
    print(f'Test set: Epoch: {epoch}')
    print(f'Average loss: {test_loss / len(TCMEyes_test_loader.dataset):.4f}')
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC: {auc:.4f}')
    print(f'Time consumed: {finish - start:.2f}s')
    print()
    #add informations to tensorboard
    if tb:
        writer.add_scalar('Test/Average loss', test_loss / len(TCMEyes_test_loader.dataset), epoch)
        writer.add_scalar('Test/Accuracy', correct.float() / len(TCMEyes_test_loader.dataset), epoch)
        writer.add_scalar('Test/Precision', precision, epoch)
        writer.add_scalar('Test/Recall', recall, epoch)
        writer.add_scalar('Test/F1 Score', f1, epoch)
        writer.add_scalar('Test/AUC', auc, epoch)
    return correct.float() / len(TCMEyes_test_loader.dataset)

if __name__ == '__main__':

    os.environ.pop('PYTORCH_CUDA_ALLOC_CONF', None)  # 移除可能存在的冲突配置
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    # parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-gpu', type=int, default=None, help='use which gpu (default: None for CPU)')
    parser.add_argument('-b', type=int, default=128, help='batch size for dataloader')
    parser.add_argument('-warm', type=int, default=1, help='warm up training phase')
    parser.add_argument('-lr', type=float, default=0.1, help='initial learning rate')
    parser.add_argument('-resume', action='store_true', default=False, help='resume training')
    args = parser.parse_args()

    if args.gpu is not None:
        if not torch.cuda.is_available():
            raise ValueError("GPU requested but CUDA not available")
        device = torch.device(f'cuda:{args.gpu}')
        torch.cuda.set_device(device)  # 显式设置当前设备
    else:
        device = torch.device('cpu')
    net = get_network(args).to(device)

    # TCMEyes_training_loader = get_training_dataloader_TCMEyes(settings.CIFAR100_TRAIN_MEAN,
    #     settings.CIFAR100_TRAIN_STD,
    #     num_workers=4,
    #     batch_size=args.b,
    #     shuffle=True)

    # TCMEyes_test_loader = get_test_dataloader_TCMEyes(settings.CIFAR100_TRAIN_MEAN,
    #                                                           settings.CIFAR100_TRAIN_STD,
    #                                                           num_workers=4,
    #                                                           batch_size=args.b,
    #                                                           shuffle=True)

    TCMEyes_training_loader = get_training_dataloader_TCMEyes(settings.LEFTEYE_TRAIN_MEAN,
        settings.LEFTEYE_TRAIN_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True)

    TCMEyes_test_loader = get_test_dataloader_TCMEyes(settings.LEFTEYE_TEST_MEAN,
        settings.LEFTEYE_TEST_STD,
        num_workers=4,
        batch_size=args.b,
        shuffle=True)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=settings.MILESTONES, gamma=0.2) #learning rate decay
    iter_per_epoch = len(TCMEyes_training_loader)
    warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * args.warm)

    if args.resume:
        recent_folder = most_recent_folder(os.path.join(settings.CHECKPOINT_PATH, args.net), fmt=settings.DATE_FORMAT)
        if not recent_folder:
            raise Exception('no recent folder were found')

        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder)

    else:
        checkpoint_path = os.path.join(settings.CHECKPOINT_PATH, args.net, settings.TIME_NOW)

    #use tensorboard
    if not os.path.exists(settings.LOG_DIR):
        os.mkdir(settings.LOG_DIR)

    #since tensorboard can't overwrite old values
    #so the only way is to create a new tensorboard log
    writer = SummaryWriter(log_dir=os.path.join(
            settings.LOG_DIR, args.net, settings.TIME_NOW))
    input_tensor = torch.Tensor(1, 3, 224, 224)
    # if args.gpu:
    #     input_tensor = input_tensor.cuda()
    input_tensor = input_tensor.to(device)
    writer.add_graph(net, input_tensor)

    #create checkpoint folder to save model
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')

    best_acc = 0.0
    if args.resume:
        best_weights = best_acc_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if best_weights:
            weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, best_weights)
            print('found best acc weights file:{}'.format(weights_path))
            print('load best training file to test acc...')
            net.load_state_dict(torch.load(weights_path))
            best_acc = eval_training(tb=False)
            print('best acc is {:0.2f}'.format(best_acc))

        recent_weights_file = most_recent_weights(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))
        if not recent_weights_file:
            raise Exception('no recent weights file were found')
        weights_path = os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder, recent_weights_file)
        print('loading weights file {} to resume training.....'.format(weights_path))
        net.load_state_dict(torch.load(weights_path))

        resume_epoch = last_epoch(os.path.join(settings.CHECKPOINT_PATH, args.net, recent_folder))

    for epoch in range(1, settings.EPOCH + 1):
        if epoch > args.warm:
            train_scheduler.step(epoch)

        if args.resume:
            if epoch <= resume_epoch:
                continue

        train(epoch)
        acc = eval_training(epoch)

        #start to save best performance model after learning rate decay to 0.01
        if epoch > settings.MILESTONES[1] and best_acc < acc:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='best')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)
            best_acc = acc
            continue

        if not epoch % settings.SAVE_EPOCH:
            weights_path = checkpoint_path.format(net=args.net, epoch=epoch, type='regular')
            print('saving weights file to {}'.format(weights_path))
            torch.save(net.state_dict(), weights_path)

    writer.close()
