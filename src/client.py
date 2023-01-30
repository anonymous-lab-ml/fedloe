import logging
import os
import copy
import time
import random
import math
from torch import optim
from urllib.parse import _NetlocResultMixinBytes
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torch.distributions as dist
from tqdm.auto import tqdm
from wilds.common.data_loaders import get_train_loader
from wilds.common.utils import split_into_groups
import multiprocessing as mp
from src.utils import *
from wilds.common.metrics.loss import ElementwiseLoss, Loss, MultiTaskLoss
from src.splitter import RandomSplitter
logger = logging.getLogger(__name__)


class ERM(object):
    """Class for client object having its own (private) data and resources to train a model.

    Participating client has its own dataset which are usually non-IID compared to other clients.
    Each client only communicates with the center server with its trained parameters or globally aggregated parameters.

    Attributes:
        id: Integer indicating client's id.
        data: torch.utils.data.Dataset instance containing local data.
        device: Training machine indicator (e.g. "cpu", "cuda").
        __model: torch.nn instance as a local model.
    """
    def __init__(self, seed, exp_id, client_id, device, dataset, ds_bundle, client_config):
        """Client object is initiated by the center server."""
        self.seed = seed
        self.exp_id = exp_id
        self.client_id = client_id
        self.device = device
        self._featurizer = None
        self.featurizer = None
        self._classifier = None
        self.classifier = None
        self.model = None
        self.dataset = dataset # Wrapper of WildSubset.
        self.postdataset, self.predataset = RandomSplitter(ratio=0.1, seed=self.seed).split(self.dataset, transform=self.dataset.transform)
        self.ds_bundle = ds_bundle
        self.client_config = client_config
        self.dataset.transform = self.ds_bundle.train_transform
        self.batch_size = self.client_config["batch_size"]
        self.optimizer_name = self.client_config['optimizer']
        self.optim_config = self.client_config['optimizer_config']
        try:
            self.scheduler_name = self.client_config['scheduler']
            self.scheduler_config = self.client_config['scheduler_config']
        except KeyError:
            self.scheduler_name = 'torch.optim.lr_scheduler.ConstantLR'
            self.scheduler_config = {'factor': 1, 'total_iters': 1}
        self.dataloader = get_train_loader(self.loader_type, self.predataset, batch_size=self.batch_size, uniform_over_groups=None)
        self.postdataloader = get_train_loader(self.loader_type, self.postdataset, batch_size=self.batch_size, uniform_over_groups=None)
        print(len(self.postdataset))
        print("size")
        self.saved_optimizer = True
        self.opt_dict_path = "/local/scratch/a/bai116/opt_dict/client_{}.pt".format(self.client_id)
        self.sch_dict_path = "/local/scratch/a/bai116/sch_dict/client_{}.pt".format(self.client_id)
        if os.path.exists(self.opt_dict_path): os.remove(self.opt_dict_path)

    @property
    def loader_type(self):
        return 'standard'

    def update_model(self, model_dict):
        self.model.load_state_dict(model_dict)
    
    def init_train(self, featurizer, classifier):
        assert featurizer or classifier
        self.featurizer = nn.DataParallel(self._featurizer)
        self.classifier = nn.DataParallel(self._classifier)
        self._model = nn.Sequential(self._featurizer, self._classifier) 
        self.model = nn.DataParallel(self._model)
        self.model.train()
        self.model.to(self.device)
        if featurizer and classifier:
            self.optimizer = eval(self.optimizer_name)(self.model.parameters(), **self.optim_config)
        elif classifier:
            self.optimizer = eval(self.optimizer_name)(self.classifier.parameters(), **self.optim_config)
        elif featurizer:
            self.optimizer = eval(self.optimizer_name)(self.featurizer.parameters(), **self.optim_config)
        self.scheduler = eval(self.scheduler_name)(self.optimizer, **self.scheduler_config)
        
        if self.saved_optimizer:
            try:
                self.optimizer.load_state_dict(torch.load(self.opt_dict_path))
                self.scheduler.load_state_dict(torch.load(self.sch_dict_path))
            except FileNotFoundError:
                pass
                

    def end_train(self):
        self.optimizer.zero_grad(set_to_none=True)
        self.model.to("cpu")
        torch.save(self.optimizer.state_dict(), self.opt_dict_path)
        torch.save(self.scheduler.state_dict(), self.sch_dict_path)
        del self.scheduler, self.optimizer

    def fit(self, num_epochs, featurizer=True, classifier=True):
        print('1')
        print(featurizer)
        """Update local model using local dataset."""
        self.init_train(featurizer, classifier)
        
        for e in range(num_epochs):
            # for batch in tqdm(self.dataloader):
            for batch in tqdm(self.dataloader):
                results = self.process_batch(batch)
                self.step(results)
            # print(self.scheduler.get_last_lr())
            self.scheduler.step()

        self.end_train()
        self.model.to('cpu')

    def evaluate(self):
        """Evaluate local model using local dataset (same as training set for convenience)."""
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            metric = {}
            y_pred = None
            y_true = None
            for batch in self.dataloader:
                results = self.process_batch(batch)
                if y_pred is None:
                    y_pred = results['y_pred']
                    y_true = results['y_true']
                else:
                    y_pred = torch.cat((y_pred, results['y_pred']))
                    y_true = torch.cat((y_true, results['y_true']))
            metric_new = self.dataset.eval(torch.argmax(y_pred, dim=-1).to("cpu"), y_true.to("cpu"), results["metadata"].to("cpu"))
            for key, value in metric_new[0].items():
                if key not in metric.keys():
                    metric[key] = value
                else:
                    metric[key] += value
        
                if self.device == "cuda": torch.cuda.empty_cache()
        self.model.to('cpu')
        return metric
    
    def calc_loss(self):
        self.model.eval()
        self.model.to(self.device)
        with torch.no_grad():
            metric = {}
            y_pred = None
            y_true = None
            for batch in self.dataloader:
                results = self.process_batch(batch)
                if y_pred is None:
                    y_pred = results['y_pred']
                    y_true = results['y_true']
                else:
                    y_pred = torch.cat((y_pred, results['y_pred']))
                    y_true = torch.cat((y_true, results['y_true']))
            loss = self.ds_bundle.loss.compute(y_pred, y_true, return_dict=False).mean()
        self.model.to('cpu')
        return loss
    
    def process_batch(self, batch):
        x, y_true, metadata = batch
        x = x.to(self.device)
        y_true = y_true.to(self.device)
        g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
        metadata = metadata.to(self.device)
        outputs = self.model(x)
        results = {
            'g': g,
            'y_true': y_true,
            'y_pred': outputs,
            'metadata': metadata,
        }
        return results

    def step(self, results):
        # print(results['y_true'])
        # objective = eval(self.criterion)()(results['y_pred'], results['y_true'])
        objective = self.ds_bundle.loss.compute(results['y_pred'], results['y_true'], return_dict=False).mean()
        if objective.grad_fn is None:
            pass
        try:
            objective.backward()
        except RuntimeError:
            print(objective)
            print(objective.grad_fn)
        self.optimizer.step()
        self.optimizer.zero_grad()

    def set_cat_classifier(self, cat_classifier):
        self._post_model = nn.Sequential(self._featurizer, cat_classifier) 
        self.post_model = nn.DataParallel(self._post_model)

    def prepare_post_data(self):
        self.post_model.to(self.device)
        y = []
        z = []
        metadata = []

        with torch.no_grad():
            for batch in tqdm(self.postdataloader):
                z.append(self.post_model(batch[0].to(self.device)).to('cpu'))
                y.append(batch[1])
                metadata.append(batch[2])
            self.post_model.to('cpu')
        return torch.cat(z), torch.cat(y), torch.cat(metadata)

    @property
    def name(self):
        return self.__class__.__name__
    
    def __len__(self):
        """Return a total size of the client's local data."""
        return len(self.dataset)

