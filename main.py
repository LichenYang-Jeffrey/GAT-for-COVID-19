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
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from models import GAT, SpGAT
from datasets.drug import DrugDataset
from utils.data import MolecularDataLoader
from utils.imbalanced import ImbalancedDatasetSampler
from tensorboardX import SummaryWriter

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.005, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=100, help='Patience')
parser.add_argument('--nfeat', type=int, default=31, help='Number of features.')
parser.add_argument('--nclass', type=int, default=2, help='Number of classes.')

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
dataset = DrugDataset('./data_covid/ecoli.csv')
train_db, val_db = torch.utils.data.random_split(dataset, [1945, 390])
trainloader = MolecularDataLoader(dataset=train_db,
                                  shuffle=False,
                                  sampler=ImbalancedDatasetSampler(train_db))
valloader = MolecularDataLoader(dataset=val_db,
                                shuffle=False,
                                sampler=ImbalancedDatasetSampler(val_db))
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


def run(epoch, phase='train'):
    loss_train = []
    labels = []
    outputs = []
    correct = 0.0
    total = 0.0
    assert phase in ['train','val']
    if phase == 'train':
        dataloader = trainloader
    elif phase == 'val':
        dataloader = valloader
    else:
        raise('Unkown phase.')
    for idx, data in enumerate(dataloader):
        label = data['label']
        features = data['atom_type'].squeeze(0)
        amask = data['atom_mask']
        adj = data['bond_mask'].squeeze(0)

        if args.cuda:
            model.cuda()
            features = features.cuda()
            amask = amask.cuda()
            adj = adj.cuda()
            label = label.cuda()

        features, amask, adj, label = Variable(features), Variable(amask), Variable(adj), Variable(label)

        t = time.time()

        if phase == 'train':
            model.train()
        elif phase == 'val':
            model.eval()
        optimizer.zero_grad()

        if phase == 'train':
            output = model(features, amask, adj)
            loss_train_batch = criterion(output, label.view(1))
            loss_train.append(loss_train_batch.data.item())
        elif phase == 'val':
            with torch.no_grad():
                output = model(features, amask, adj)

        _, predicted = torch.max(output.data, 1)

        labels.append(label.view(1).cpu().detach())
        outputs.append(output[:, 1].cpu().detach())

        correct += predicted.eq(label.view(1)).cpu().sum()
        total += 1.0

        if phase=='train':
            loss_train_batch.backward()
            optimizer.step()

    roc_auc = roc_auc_score(labels, outputs)
    precision, recall, _thresholds = precision_recall_curve(labels, outputs)
    prc = auc(recall, precision)
    acc = 1.0 * correct.data.item() / total

    if phase == 'train':
        print('Train Epoch: {:04d}'.format(epoch + 1),
              'loss: {:.4f}'.format(np.mean(loss_train)),
              'acc: {:.4f}'.format(acc),
              'roc_auc: {:.4f}'.format(roc_auc),
              'prc: {:.4f}'.format(prc),
              'time: {:.4f}s'.format(time.time() - t))
        return np.mean(loss_train), acc, roc_auc, prc
    elif phase == 'val':
        print('Val Epoch: {:04d}'.format(epoch + 1),
              'acc: {:.4f}'.format(acc),
              'roc_auc: {:.4f}'.format(roc_auc),
              'prc: {:.4f}'.format(prc),
              'time: {:.4f}s'.format(time.time() - t))
        return acc, roc_auc, prc


if __name__ == '__main__':
    train_logf = open('./train_log.txt', "w")
    val_logf = open('./val_log.txt', "w")
    writer = SummaryWriter(log_dir='scalar')
    # Train model
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = args.epochs + 1
    best_epoch = 0
    for epoch in range(args.epochs):
        t_loss, t_acc, t_roc, t_prc = run(epoch, phase='train')
        v_acc, v_roc, v_prc = run(epoch, phase='val')
        loss_values.append(t_loss)
        train_logf.write(
            'Epoch: {:04d}\tloss: {:.4f}\tacc: {:.4f}\troc: {:.4f}\tprc: {:.4f}'.format(
                epoch + 1, t_loss, t_acc, t_roc, t_prc))
        val_logf.write(
            'Epoch: {:04d}acc: {:.4f}\troc: {:.4f}\tprc: {:.4f}'.format(
                epoch + 1, v_acc, v_roc, v_prc))
        writer.add_scalar('train loss', t_loss, epoch)
        writer.add_scalar('train roc_auc', t_roc, epoch)
        writer.add_scalar('train prc', t_prc, epoch)
        writer.add_scalar('val roc_auc', v_roc, epoch)
        writer.add_scalar('val prc', v_prc, epoch)

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

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    train_logf.close()
    val_logf.close()
    writer.close()