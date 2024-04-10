# Copyright 2022 solo-learn development team.

# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
# Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies
# or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
# PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
# FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
# OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import inspect
import os

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
# from pytorch_lightning.loggers import WandbLogger

from solo.args.pretrain import parse_cfg
from solo.data.classification_dataloader import prepare_data as prepare_data_classification
from solo.data.pretrain_dataloader import (
    FullTransformPipeline,
    NCropAugmentation,
    build_transform_pipeline,
    prepare_dataloader,
    prepare_datasets,
)
from solo.methods import METHODS
from solo.utils.auto_resumer import AutoResumer
from solo.utils.checkpointer import Checkpointer
from solo.utils.misc import make_contiguous

try:
    from solo.data.dali_dataloader import PretrainDALIDataModule, build_transform_pipeline_dali
except ImportError:
    _dali_avaliable = False
else:
    _dali_avaliable = True

try:
    from solo.utils.auto_umap import AutoUMAP
except ImportError:
    _umap_available = False
else:
    _umap_available = True


def inference(model, loader, device=torch.device('cuda')):
    feature_vector = []
    labels_vector = []
    for step, (x, y) in enumerate(loader):
        x = x.cuda()

        # get encoding
        with torch.no_grad():
            h = model(x)
            # import pdb; pdb.set_trace()
            # if type(h) is tuple:
                # h = h[-1]
            # if type(h) is dict:
            # h = h['feats']
            h = h['z']
            # h = model.projector(h)

        feature_vector.append(h.data.to(device))
        labels_vector.append(y.to(device))

    feature_vector = torch.cat(feature_vector)
    labels_vector = torch.cat(labels_vector)
    return feature_vector, labels_vector


def plot_tsne(data, labels, n_classes, save_dir='figs', file_name='simclr', y_name='Class'):

    from sklearn.manifold import TSNE
    from matplotlib import ft2font
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    """ Input:
            - model weights to fit into t-SNE
            - labels (no one hot encode)
            - num_classes
    """
    n_components = 2
    if n_classes == 10:
        platte = sns.color_palette(n_colors=n_classes)
    else:
        platte = sns.color_palette("Set2", n_colors=n_classes)

    tsne = TSNE(n_components=n_components, init='pca', perplexity=40, random_state=0)
    tsne_res = tsne.fit_transform(data)

    v = pd.DataFrame(data,columns=[str(i) for i in range(data.shape[1])])
    v[y_name] = labels
    v['label'] = v[y_name].apply(lambda i: str(i))
    v["t1"] = tsne_res[:,0]
    v["t2"] = tsne_res[:,1]


    sns.scatterplot(
        x="t1", y="t2",
        hue=y_name,
        palette=platte,
        legend=True,
        data=v,
    )
    plt.xticks([])
    plt.yticks([])
    plt.xlabel('')
    plt.ylabel('')
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, file_name+'_t-SNE.png'))


@hydra.main(version_base="1.2")
def main(cfg: DictConfig):
    # hydra doesn't allow us to add new keys for "safety"
    # set_struct(..., False) disables this behavior and allows us to add more parameters
    # without making the user specify every single thing about the model

    # validation dataloader for when it is available
    if cfg.data.format == "dali":
        val_data_format = "image_folder"
    else:
        val_data_format = cfg.data.format

    train_loader, val_loader = prepare_data_classification(
        cfg.data.dataset,
        train_data_path=cfg.data.train_path,
        val_data_path=cfg.data.val_path,
        data_format=val_data_format,
        batch_size=cfg.optimizer.batch_size,
        num_workers=cfg.data.num_workers,
        )
    train_dataset, val_dataset = train_loader.dataset, val_loader.dataset

    # TODO: add the code that load / computes features from the model
    
    import torchvision.utils as tv
    import torch.nn.functional as F

    if 'relu' in cfg.resume_from_checkpoint:
        val_features = F.relu(val_features)
    # elif 'gelu' in cfg.resume_from_checkpoint:
    #     val_features = F.gelu(val_features)

    def sparsity(features, eps=1e-2):
        # features = F.normalize(features, dim=1)
        # import pdb; pdb.set_trace()        
        # sparsity_per_dim =  (features.abs().sum(dim=1)<eps).float() * 100
        # return sparsity_per_dim
        sp = ((features.abs()<eps).float().sum(dim=1)) / len(features[0]) * 100
        print('sparsity:', sp)
        return sp

    def cluster_acc(features, labels, eps=1e-5, take_abs=False, topk=False):
        # val_features_n0 = F.normalize(val_features, dim=0)        
        features  = features[:,F.relu(features).sum(0)>0]
        
        features = F.normalize(features, dim=1)
        if take_abs:
            features = features.abs()

        if topk:
            sorted, indices = torch.sort(features.sum(dim=0), descending=True)
            indices = indices[sorted>1]
            features = features[:, indices]

        acc_per_dim = []
        ent_per_dim = []
        for i in range(features.shape[1]):
            mask = features.abs()[:,i] > eps
            labels_selected = labels[mask]
            # if mask.sum() == 0:
            #     continue
            # topk, indices = features[:,i].topk(200)
            # labels_selected = labels[indices]
            try:
                dist = labels_selected.bincount()
                dist = dist / dist.sum()
                acc = dist.max().item()
                ent = - (dist * (dist+eps).log()).sum().item()
                acc_per_dim.append(acc)
                ent_per_dim.append(ent)
            except:
                pass
        import pdb; pdb.set_trace()
        acc_per_dim =  torch.tensor(acc_per_dim) * 100
        ent_per_dim =  torch.tensor(ent_per_dim)
        print('[cluster acc] mean {:.4f} std {:.4f}'.format(acc_per_dim.mean(), acc_per_dim.std()))
        print('[cluster ent] mean {:.4f} std {:.4f}'.format(ent_per_dim.mean(), ent_per_dim.std()))
        return acc_per_dim.numpy()    
    def orthogonality(features, eps=1e-5):
        # import pdb; pdb.set_trace() 
        features  = features[:,features.sum(0)>10]
        n, d = features.shape
        features = F.normalize(features, dim=0)
        corr = features.T @ features
        # sns.heatmap(corr)
        err = (corr - torch.eye(d, device=features.device)).abs()
        # import pdb; pdb.set_trace()
        err = err.median()        
        print('disentanglement mean {:.3f} median {:.3f}'.format(err.mean(), err.median()))
        return corr
    
    sparsity(val_features)
    cluster_acc(val_features, val_labels)
    orthogonality(val_features)
    import pdb; pdb.set_trace()

    # guess=torch.tensor([torch.bincount(temp[:,i]).argmax() for i in range(idx.shape[1])])

    # # val, idx = torch.topk(pf, k=100, dim=0, largest=False)
    # torch.logical_and(pf>1e-3, pf<1e-2)
    # mask = val[-1] > 1e-3
    # idx =  idx.T[mask]
    # idx = idx.T
    # data = torch.from_numpy(val_loader.dataset.data)[idx.cpu()] / 255.0
    # samples = data.permute(0, 1, 4, 2, 3).flatten(0, 1)
    # print(samples.shape)
    # tv.save_image(tv.make_grid(samples, nrow=100), f'{cfg.save_img}.png')


if __name__ == "__main__":
    main()
