from model import HeteroRGCN
from loss import bpr_loss
from data import Data
from utility.batch_test import test
import torch
from tqdm import tqdm 
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")

    parser.add_argument('--weights_path', nargs='?', default='')
    parser.add_argument('--data_dir')
    parser.add_argument('--pretrain', action='store_true')

    parser.add_argument('--in_size', type=int, default=128)
    parser.add_argument('--hidden_size', type=int, default=64)
    parser.add_argument('--out_size', type=int, default=32)

    parser.add_argument('--self_loop', action='store_true')
    parser.add_argument('--drop_out', type=float, default=0.0)
    parser.add_argument('--bias', action='store_true')

    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=1024)

    parser.add_argument('--regs', nargs='?', default='[1e-5,1e-5,1e-2]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wdc', type=float, default=0.0)

    parser.add_argument('--Ks', nargs='?', default='[20, 40, 60, 80, 100]',
                        help='Output sizes of every layer')

    return parser.parse_args()

Ks = eval(args.Ks)
data = Data(args.data_dir, args.batch_size)
model = HeteroRGCN(data.G, args.in_size, args.hidden_size, args.out_size, args.bias, args.self_loop, args.dropout)
opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdc)

for epoch in tqdm(range(args.epoch)):
    model.train()
    logits = model(data.G)

    loss, mf_loss, emb_loss = 0., 0., 0.
    n_batch = data.n_train // args.batch_size + 1

    for idx in range(n_batch):
        users, pos_items, neg_items = data.sample()
        batch_mf_loss, batch_emb_loss = bpr_loss(logits['user'][users], logits['item'][pos_items], logits['item'][neg_items])
        loss = loss + batch_mf_loss + batch_emb_loss
        emb_loss += batch_emb_loss.item()
        reg_loss += batch_reg_loss.item()

    opt.zero_grad()
    loss.backward()
    opt.step() 

    print("Epoch {}: loss {}, emb_loss {}, reg_loss {}".format(epoch, loss.item(), emb_loss, reg_loss))

    if epoch%5==0:
        model.eval()
        logits = model(data.G)
        ret = test(logits, data, Ks)

        final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                        ('\t'.join(['%.5f' % r for r in ret['recall']]),
                        '\t'.join(['%.5f' % r for r in ret['precision']]),
                        '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                        '\t'.join(['%.5f' % r for r in ret['ndcg']]))
        print(final_perf)
