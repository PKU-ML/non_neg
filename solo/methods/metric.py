import torch
import torch.nn as nn
import torch.nn.functional as F


def cluster_acc(features, labels, eps=0.1, take_abs=False):
        # val_features_n0 = F.normalize(val_features, dim=0)        
        features = F.normalize(features, dim=1)
        if take_abs:
            features = features.abs()

        if False:
            sorted, indices = torch.sort(features.sum(dim=0), descending=True)
            indices = indices[sorted>0]
            features = features[:, indices]
            print(indices,features)

        acc_per_dim = []
        ent_per_dim = []
        for i in range(features.shape[1]):
            mask = features[:,i] > eps
            labels_selected = labels[mask]
            if mask.sum() == 0:
                 continue
            # topk, indices = features[:,i].topk(200)
            # labels_selected = labels[indices]
            
            dist = labels_selected.bincount()
            dist = dist / dist.sum()
            acc = dist.max().item()
            ent = - (dist * (dist+eps).log()).sum().item()
            acc_per_dim.append(acc)
            ent_per_dim.append(ent)
        acc_per_dim =  torch.tensor(acc_per_dim) * 100
        ent_per_dim =  torch.tensor(ent_per_dim)
        print('[cluster acc] mean {:.4f} std {:.4f}'.format(acc_per_dim.mean(), acc_per_dim.std()))
        print('[cluster ent] mean {:.4f} std {:.4f}'.format(ent_per_dim.mean(), ent_per_dim.std()))

def sparsity(features, eps=1e-5):
        features = F.normalize(features, dim=1)
        # import pdb; pdb.set_trace()        
        sparsity_per_dim =  (features.abs().sum(dim=0)<eps).float()
        # avg_sparsity_dim = (sparsity_per_dim==0).float().mean()
        # sparsity_per_sample = (features.abs().sum(dim=1)>eps).float()
        print('[sparisity dim] sum {:.4f} ratio {:.4f}'.format(sparsity_per_dim.sum(), sparsity_per_dim.mean()))
        # print('[sparisity sample] mean {:.4f} std {:.4f}'.format(sparsity_per_sample.mean(), sparsity_per_sample.std()))

def disentangle(features, eps=1e-5):
        n, d = features.shape
        #features = F.normalize(features, dim=0)
        corr = features.T @ features
        corr = corr/n
        err = (corr - torch.eye(d, device=features.device)).abs().mean()
        # import pdb; pdb.set_trace()
        print('disentangle err {:.3f}'.format(err))        
        
