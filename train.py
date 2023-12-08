from operator import ne
import os
import time
import argparse
from dgl.data import gindt
from networkx.readwrite.edgelist import parse_edgelist
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import networkx as nx
from dgl import DGLGraph
from dgl import function as fn
import dgl.ops as ops

from utils import evaluate, load_data, sample_graph
from model import PolarGCN, AngleLoss

from torch.utils.tensorboard import SummaryWriter

# torch.cuda.set_device(0)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='chameleon')
parser.add_argument('--mode', type=str, default='polar')
parser.add_argument('--struct', type=str, default='AW_distb')
parser.add_argument('--train_ratio', type=int, default=0, help='train_ratio')
parser.add_argument('--no-cuda', action="store_true", default=False)

parser.add_argument('--lr', type=float, default=0.01, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--fea_drop', type=float, default=0.3, help='Dropout rate (1 - keep probability).')
parser.add_argument('--weight_drop', type=float, default=0.0, help='Dropout rate (1 - keep probability).')

parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--patience', type=int, default=50, help='Patience')
parser.add_argument('--hidden_conv', type=int, default=64, help='Number of hidden units.')
parser.add_argument('--img_neighbor_num', type=float, default=0, help='Sample ratio.')
parser.add_argument('--num_layer', type=int, default=2, help='Number of layers')
parser.add_argument('--num_head', type=int, default=1, help='Number of similarity heads')
parser.add_argument('--m', type=float, default=0.15, help='m in Angle Loss')
parser.add_argument('--lamb', type=float, default=1.0, help='hyperparameter for position')
parser.add_argument('--eta', type=float, default=1.0, help='hyperparameter for rep')

parser.add_argument('--log_dir', type=str, default="", help='tensorboard log directory')
parser.add_argument('--compare', action="store_true", default=False)

args = parser.parse_args()

trial_name = "{}_{}_lr{}_wd{}_dropout{}w{}_dim{}_img{}_layer{}_head{}_m{}_lamb{}_eta{}".format(args.dataset, args.struct, args.lr, args.weight_decay, 
                            args.fea_drop, args.weight_drop, args.hidden_conv, args.img_neighbor_num, args.num_layer, args.num_head, args.m, args.lamb, args.eta)
if args.log_dir != '':
    writer = SummaryWriter(log_dir=args.log_dir+trial_name+'/')
    f = open(args.log_dir+trial_name+"/output.txt",'w')

print(args)

if args.no_cuda or not torch.cuda.is_available():
    device = 'cpu'
else:
    device = 'cuda'

g, nclass, features, positions, structure, labels, labels_tensor, train, val, test = load_data(dataset=args.dataset, 
                                                                struct_method=args.struct, train_ratio=args.train_ratio)
features = features.to(device)
positions = positions.to(device)
structure = structure.to(device)
labels_tensor = labels_tensor.to(device)

net = PolarGCN(in_dim=features.size()[1], hidden_dim=args.hidden_conv, str_dim=structure.shape[1], rep_dim=args.hidden_conv, out_dim=nclass, 
                fea_drop=args.fea_drop, weight_drop = args.weight_drop, num_layer=args.num_layer, lamb = args.lamb, eta = args.eta,  
                num_head=args.num_head, mode=args.mode, struct=args.struct).to(device)

if args.mode == 'polar':
    Loss = AngleLoss(m=args.m)
else:
    Loss = nn.CrossEntropyLoss()

# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = StepLR(optimizer, step_size=50, gamma=0.9)

# main loop
dur = []
los = []
loc = []
counter = 0
min_loss = 100.0
max_mic = 0.0

g_input, w_true = sample_graph(g, int(args.img_neighbor_num * g.num_nodes()), device, labels, args.compare)
g_input.ndata['h'] = features
g_input.ndata['pos'] = structure


for epoch in range(args.epochs):
    t0 = time.time()

    # train
    net.train()
    loss = 0
    for _ in range(1):
        output, logits, emb = net(g_input, features)
        loss_train = Loss(output[train], labels_tensor[train])
        loss += loss_train
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()

    # evaluation
    net.eval()
    output, logits, emb = net(g_input, features)
    loss_val = Loss(output[val], labels_tensor[val]).item()
    logits = logits.cpu().detach()
    emb = emb.cpu().detach()
    train_f1 = evaluate(logits[train], labels[train])
    val_f1 = evaluate(logits[val], labels[val])
    test_f1 = evaluate(logits[test], labels[test])

    if args.mode == 'polar':
        e = g_input.edata['w'].cpu().detach_()
        los.append([epoch, loss_val, val_f1, test_f1, emb, e])
    else:
        los.append([epoch, loss_val, val_f1, test_f1, emb])

    if loss_val < min_loss or max_mic < val_f1[1]:
        min_loss = loss_val
        max_mic = val_f1[1]
        counter = 0
    else:
        counter += 1

    # print 
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'macro & micro f1_train: {:.4f}, {:.4f}'.format(train_f1[0], train_f1[1]),
          'loss_val: {:.4f}'.format(loss_val),
          'macro & micro f1_val: {:.4f}, {:.4f}'.format(val_f1[0], val_f1[1]),
          'macro & micro f1_test: {:.4f}, {:.4f}'.format(test_f1[0], test_f1[1]),
          'time: {:.4f}s'.format(np.mean(dur)))
    
    if args.log_dir != '':
        writer.add_scalar('valid/loss', loss_val, epoch)
        writer.add_scalar('train/loss', loss_train, epoch)
        writer.add_scalar('valid/micro_f1',val_f1[1], epoch)
        writer.add_scalar('train/micro_f1', train_f1[1], epoch)
        writer.add_scalar('test/micro_f1', test_f1[1], epoch)

        f.write('Epoch: {:04d}'.format(epoch+1)+' '+
          'loss_train: {:.4f}'.format(loss_train.item())+' '+
          'macro & micro f1_train: {:.4f}, {:.4f}'.format(train_f1[0], train_f1[1])+' '+
          'loss_val: {:.4f}'.format(loss_val)+' '+
          'macro & micro f1_val: {:.4f}, {:.4f}'.format(val_f1[0], val_f1[1])+' '+
          'macro & micro f1_test: {:.4f}, {:.4f}'.format(test_f1[0], test_f1[1])+' '+
          'time: {:.4f}s'.format(np.mean(dur))+'\n')

    if counter >= args.patience:
        print('early stop')
        break

    


# res at max val mic
los.sort(key=lambda x: -x[2][1])
f1 = los[0][3]
print(los[0][0] + 1, f1)

if args.log_dir != '':
    f.write(str(los[0][0] + 1) + str(f1))

if args.log_dir != '':
    f.close()
