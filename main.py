from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from models import GAT, SpGAT
from datasets.drug import DrugDataset
from utils.data import MolecularDataLoader
from utils.imbalanced import *
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.05, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--nfeat', type=int, default=31, help='Number of features.')
parser.add_argument('--nclass', type=int, default=2, help='Number of classes.')
parser.add_argument('--batch_size', type=int, default=256, help='Batch Size.')
parser.add_argument('--num_workers', type=int, default=16, help='Number of Workers.')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

print('cuda: {}'.format(args.cuda))

if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Load data
# adj, features, labels, idx_train, idx_val, idx_test = load_data()
dataset = DrugDataset('./data_covid/AID1706_binarized_sars.csv')
dataset = dataset[:20000]
train_size = int(0.9 * len(dataset))
val_size = len(dataset) - train_size
train_db, val_db = torch.utils.data.random_split(dataset, [train_size, val_size])

pos_ratio_list = [0.125, 0.25, 0.50]

valloader = MolecularDataLoader(dataset=val_db,
                                    batch_size=args.batch_size,
                                    shuffle=False,
                                    num_workers=args.num_workers)

def get_trainloader(pos_ratio):
    trainloader = MolecularDataLoader(dataset=train_db,
                                      batch_size=args.batch_size,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      sampler=ImbalancedDatasetSampler(train_db,
                                                                       callback_get_weight=WeightBinary(pos_ratio)))
    return trainloader


# Model and optimizer
model = GAT(nfeat=args.nfeat,
            nhid=args.hidden,
            nout=128,
            nlmphid=64,
            nclass=args.nclass,
            dropout=args.dropout,
            nheads=args.nb_heads,
            alpha=args.alpha)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

criterion = torch.nn.CrossEntropyLoss()


def run(epoch, dataloader, phase='train'):
    loss_train = []
    labels = []
    outputs = []
    preds = []
    pos_correct = 0.0
    neg_correct = 0.0
    pos = 0.0
    neg = 0.0
    correct = 0.0
    total = 0.0
    assert phase in ['train', 'val']
    for idx, data in enumerate(tqdm(dataloader)):
        label = data['label']
        features = data['atom_type']
        amask = data['atom_mask']
        adj = data['bond_mask']

        if args.cuda:
            model.cuda()
            features = features.cuda()
            amask = amask.cuda()
            adj = adj.cuda()
            label = label.cuda()

        features, amask, adj, label = Variable(features), Variable(amask), Variable(adj), Variable(label)

        label = label.view(-1)

        t = time.time()

        if phase == 'train':
            model.train()
        elif phase == 'val':
            model.eval()
        optimizer.zero_grad()

        if phase == 'train':
            output = model(features, amask, adj)
            loss_train_batch = criterion(output, label)
            loss_train.append(loss_train_batch.data.item())
        elif phase == 'val':
            with torch.no_grad():
                output = model(features, amask, adj)

        _, predicted = torch.max(output.data, 1)
        
        labels.extend(label.cpu().detach().numpy())
        outputs.extend(output[:, 1].cpu().detach().numpy().tolist())
        
        ones = Variable(torch.ones(label.size()).type(torch.LongTensor).cuda())
        zeros = Variable(torch.zeros(label.size()).type(torch.LongTensor).cuda())
        
        pos_correct += ((predicted == ones) & (label == ones)).sum()
        neg_correct += ((predicted == zeros) & (label == zeros)).sum()
        
        pos += (label == ones).sum()
        neg += (label == zeros).sum()
        
        correct += predicted.eq(label).cpu().sum()
        total += args.batch_size

        if phase == 'train':
            loss_train_batch.backward()
            optimizer.step()

    roc_auc = roc_auc_score(labels, outputs)
    precision, recall, _thresholds = precision_recall_curve(labels, outputs)
    prc = auc(recall, precision)
    pos_acc = 1.0 * pos_correct.data.item() / pos.data.item()
    neg_acc = 1.0 * neg_correct.data.item() / neg.data.item()
    acc = 1.0 * correct.data.item() / total

    if phase == 'train':
        print('Train Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.4f}'.format(np.mean(loss_train)),
              'acc: {:.4f}'.format(acc),
              'pos_acc: {:.4f}'.format(pos_acc),
              'neg_acc: {:.4f}'.format(neg_acc),
              'roc_auc: {:.4f}'.format(roc_auc),
              'prc: {:.4f}'.format(prc),
              'time: {:.4f}s'.format(time.time() - t))
        return np.mean(loss_train), pos_acc, neg_acc, acc, roc_auc, prc
    elif phase == 'val':
        print('Val Epoch: {:04d}'.format(epoch + 1),
              'pos_acc: {:.4f}'.format(pos_acc),
              'neg_acc: {:.4f}'.format(neg_acc),
              'acc: {:.4f}'.format(acc),
              'roc_auc: {:.4f}'.format(roc_auc),
              'prc: {:.4f}'.format(prc),
              'time: {:.4f}s'.format(time.time() - t))
        return pos_acc, neg_acc, acc, roc_auc, prc


if __name__ == '__main__':
    print('Trainset Size: {}'.format(train_size))
    print('Valset Size: {}'.format(val_size))
    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0

    for pos_ratio in pos_ratio_list:
        train_logf = open('./log/train_log_{}.txt'.format(pos_ratio), "w")
        val_logf = open('./log/val_log_{}.txt'.format(pos_ratio), "w")
        if not os.path.exists('./scalar/scalar_{}'.format(pos_ratio)):
            os.mkdir('./scalar/scalar_{}'.format(pos_ratio))
            
        writer = SummaryWriter(log_dir='./scalar/scalar_{}'.format(pos_ratio))
        
        trainloader = get_trainloader(pos_ratio=pos_ratio)
        
        print('Positive Ratio: {}'.format(pos_ratio))
        
        for epoch in range(args.epochs):
            t_loss, t_pos_acc, t_neg_acc, t_acc, t_auc, t_prc = run(epoch, dataloader=trainloader, phase='train')
            v_pos_acc, v_neg_acc, v_acc, v_auc, v_prc = run(epoch, dataloader=valloader, phase='val')
            loss_values.append(t_loss)
            train_logf.write(
                'Epoch[{:04d}] loss:{:.4f} pos_acc:{:.4f} neg_acc:{:.4f} acc:{:.4f} auc:{:.4f} prc:{:.4f}\n'.format(
                    epoch + 1, t_loss, t_pos_acc, t_neg_acc, t_acc, t_auc, t_prc))
            val_logf.write(
                'Epoch[{:04d}] pos_acc:{:.4f} neg_acc:{:.4f} acc:{:.4f} auc:{:.4f} prc:{:.4f}\n'.format(
                    epoch + 1, v_pos_acc, v_neg_acc, v_acc, v_auc, v_prc))
            train_logf.flush()
            val_logf.flush()
            writer.add_scalar('train loss', t_loss, epoch)
            writer.add_scalar('train roc_auc', t_auc, epoch)
            writer.add_scalar('train prc', t_prc, epoch)
            writer.add_scalar('train pos_acc', t_pos_acc, epoch)
            writer.add_scalar('train neg_acc', t_neg_acc, epoch)
            writer.add_scalar('val roc_auc', v_auc, epoch)
            writer.add_scalar('val prc', v_prc, epoch)
            writer.add_scalar('val pos_acc', v_pos_acc, epoch)
            writer.add_scalar('val neg_acc', v_neg_acc, epoch)

            torch.save(model.state_dict(), 'save/{}.pkl'.format(epoch))
            if loss_values[-1] < best:
                best = loss_values[-1]
                best_epoch = epoch
                bad_counter = 0
            else:
                bad_counter += 1

            if bad_counter == args.patience:
                break

            files = glob.glob('save/*.pkl')
            for file in files:
                epoch_nb = int(file.split('/')[1][:-4])
                if epoch_nb < best_epoch:
                    os.remove(file)

        files = glob.glob('save/*.pkl')
        for file in files:
            epoch_nb = int(file.split('/')[1][:-4])
            if epoch_nb > best_epoch:
                os.remove(file)

        print("Finished!")
        print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

        train_logf.close()
        val_logf.close()
        writer.close()
