import torch 
import torch.nn.functional as F

def bpr_loss(users, pos_items, neg_items, decay=1):
    pos_scores = torch.mul(users,pos_items).sum(1) 
    neg_scores = torch.mul(users,neg_items).sum(1) 

    regularizer = torch.sum(users**2)/2 + torch.sum(pos_items**2)/2 + torch.sum(neg_items**2)/2
    regularizer = regularizer/users.shape[0]

    # In the first version, we implement the bpr loss via the following codes:
    # We report the performance in our paper using this implementation.
    maxi = F.logsigmoid(pos_scores - neg_scores)
    mf_loss = -torch.mean(maxi)

    ## In the second version, we implement the bpr loss via the following codes to avoid 'NAN' loss during training:
    ## However, it will change the training performance and training performance.
    ## Please retrain the model and do a grid search for the best experimental setting.
    # mf_loss = tf.reduce_sum(tf.nn.softplus(-(pos_scores - neg_scores)))

    emb_loss = decay * regularizer

    return mf_loss, emb_loss
