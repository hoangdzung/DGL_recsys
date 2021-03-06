from model import HeteroRGCN
from loss import bpr_loss
from data import Data
from utility.batch_test import test
from utility.watcher import EarlyStopping
import torch
from tqdm import tqdm 
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run NGCF.")

    parser.add_argument('--weights_path', nargs='?', default='')
    parser.add_argument('--data_dir')
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--patience', default = 5, type = int)

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

    parser.add_argument('--Ks', nargs='?', default='[20]',
                        help='Output sizes of every layer')

    return parser.parse_args()

args = parse_args()
Ks = eval(args.Ks)
data = Data(args.data_dir, args.batch_size)
model = HeteroRGCN(data.G, args.in_size, args.hidden_size, args.out_size, args.bias, args.self_loop, args.drop_out)
opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wdc)
early_stopping = EarlyStopping(patience=args.patience, verbose=True)

for epoch in range(args.epoch):
    model.train()
    logits = model(data.G)

    loss, mf_loss, emb_loss = 0., 0., 0.
    n_batch = data.n_train // args.batch_size + 1

    for idx in tqdm(range(n_batch),desc='epoch '+str(epoch)):
        users, pos_items, neg_items = data.sample()
        batch_mf_loss, batch_emb_loss = bpr_loss(logits['user'][users], logits['item'][pos_items], logits['item'][neg_items])
        loss = loss + batch_mf_loss + batch_emb_loss
        mf_loss += batch_mf_loss.item()
        emb_loss += batch_emb_loss.item()

    opt.zero_grad()
    loss.backward()
    opt.step() 

    print("Epoch {}: loss {}, emb_loss {}, mf_loss {}".format(epoch, loss.item(), emb_loss, mf_loss))

    early_stopping(loss.item())

    if early_stopping.early_stop:
        print("Early stopping")
        break

    if early_stopping.is_best:
        torch.save(model.state_dict(), args.weights_path)

    if epoch%5==0:
        model.eval()
        logits = model(data.G)
        ret = test(data, logits, Ks)

        final_perf = "recall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                        ('\t'.join(['%.5f' % r for r in ret['recall']]),
                        '\t'.join(['%.5f' % r for r in ret['precision']]),
                        '\t'.join(['%.5f' % r for r in ret['hit_ratio']]),
                        '\t'.join(['%.5f' % r for r in ret['ndcg']]))
        print(final_perf)
