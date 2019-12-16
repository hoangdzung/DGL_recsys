from collections import defaultdict
import dgl
import numpy as np
import random
import os 

class Data():
    def __init__(self, datadir, batch_size=128):
        self.train_items, self.train_edge_dict = self.get_edge_list(os.path.join(datadir, 'train.txt'))
        self.test_items, self.test_edge_dict  = self.get_edge_list(os.path.join(datadir, 'test.txt'))
        
        self.n_train = len(self.train_edge_dict[('user','ui','item')])
        self.n_test = len(self.test_edge_dict[('user','ui','item')])
        self.G = dgl.heterograph(self.train_edge_dict)

        self.n_items = self.G.number_of_nodes('item')
        self.n_users = self.G.number_of_nodes('user')
        self.users = self.G.nodes('user').detach().cpu().numpy().tolist()

        self.batch_size = batch_size
    
    @staticmethod
    def get_edge_list(datapath): 
        assert os.path.isfile(datapath)
        items =  defaultdict(list)
        edge_dict = defaultdict(list)
        for line in open(datapath):
            line = list(map(int, line.strip().split()))
            user_id = line[0]
            item_id_list = line[1:]
            items[user_id] = item_id_list
            for item_id in item_id_list:
                edge_dict[('user','ui', 'item')].append((user_id, item_id))
                edge_dict[('item','iu', 'user')].append((item_id, user_id))
        return items, edge_dict


    def sample(self):
        if self.batch_size <= self.n_users:
            users = random.sample(self.users, self.batch_size)
        else:
            users = [random.choice(self.users) for _ in range(self.batch_size)]

        def sample_pos_items_for_u(u, num):
            pos_items = self.train_items[u]
            n_pos_items = len(pos_items)
            pos_batch = []
            while True:
                if len(pos_batch) == num: break
                pos_id = np.random.randint(low=0, high=n_pos_items, size=1)[0]
                pos_i_id = pos_items[pos_id]

                if pos_i_id not in pos_batch:
                    pos_batch.append(pos_i_id)
            return pos_batch

        def sample_neg_items_for_u(u, num):
            neg_items = []
            while True:
                if len(neg_items) == num: break
                neg_id = np.random.randint(low=0, high=self.n_items,size=1)[0]
                if neg_id not in self.train_items[u] and neg_id not in neg_items:
                    neg_items.append(neg_id)
            return neg_items


        pos_items, neg_items = [], []
        for u in users:
            pos_items += sample_pos_items_for_u(u, 1)
            neg_items += sample_neg_items_for_u(u, 1)

        return users, pos_items, neg_items
