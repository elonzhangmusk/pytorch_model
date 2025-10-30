#test.py
#!/usr/bin/env python3

""" test neuron network performace
print top1 and top5 err on test dataset
of a model

author baiyu
"""

import argparse

from matplotlib import pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score,roc_auc_score
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import numpy as np

from conf import settings
from utils import get_network, get_test_dataloader,get_test_dataloader_TCMEyes

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-net', type=str, required=True, help='net type')
    parser.add_argument('-weights', type=str, required=True, help='the weights file you want to test')
    parser.add_argument('-gpu', action='store_true', default=False, help='use gpu or not')
    parser.add_argument('-b', type=int, default=16, help='batch size for dataloader')
    args = parser.parse_args()

    net = get_network(args)

    TCMEyes_test_loader = get_test_dataloader_TCMEyes(settings.LEFTEYE_TEST_MEAN,
                                                              settings.LEFTEYE_TEST_STD,
                                                              num_workers=4,
                                                              batch_size=args.b,
                                                              shuffle=True)

    net.load_state_dict(torch.load(args.weights))
    print(net)
    net.eval()

    correct_1 = 0.0
    correct_5 = 0.0
    total = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for n_iter, (image, label) in enumerate(TCMEyes_test_loader):
            print("iteration: {}\ttotal {} iterations".format(n_iter + 1, len(TCMEyes_test_loader)))

            if args.gpu:
                image = image.cuda()
                label = label.cuda()
                print('GPU INFO.....')
                print(torch.cuda.memory_summary(), end='')

            output = net(image)
            _, pred = output.max(1)

            label = label.view(-1)  # 确保label是1D张量 [batch_size]
            correct = pred.eq(label).float()
            correct_1 += correct.sum().item()  # 使用.item()获取Python标量
            all_preds.append(pred.cpu().numpy())
            all_labels.append(label.cpu().numpy())
            all_probs.append(torch.softmax(output, dim=1).cpu().numpy())

    if args.gpu:
        print('GPU INFO.....')
        print(torch.cuda.memory_summary(), end='')

    # Aggregate results
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    all_probs = np.concatenate(all_probs)
    accuracy = correct_1 / len(TCMEyes_test_loader.dataset)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    auc = roc_auc_score(all_labels, all_probs[:, 1])  # 使用正类的概率

    print()
    print("Parameter numbers: {}".format(sum(p.numel() for p in net.parameters())))
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'AUC: {auc:.4f}')
