import copy
import logging
from multiprocessing import pool, cpu_count
import collections
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import BatchSampler, RandomSampler
from tqdm.auto import tqdm
from collections import OrderedDict
import torch.distributions as dist

from .models import *
from .utils import *
from .client import *
from .dataset_bundle import *
from .datasets import TensorMultiDataset

logger = logging.getLogger(__name__)


class ServerIRMv1(object):
    def __init__(self, seed, exp_id, device, clients, ds_bundle, server_config):
        self.seed = seed
        self.id = exp_id
        self.ds_bundle = ds_bundle
        self.device = device
        self.clients = clients
        self.num_clients = len(self.clients)
        self.server_config = server_config
        self.loading = server_config['loading']
        if self.loading:
            self._round = server_config['start_round']
        else:
            self._round = 0
        self.num_rounds_first_stage = server_config['num_rounds_first_stage']
        self.num_rounds_second_stage = server_config['num_rounds_second_stage']
        self.num_epochs_first_stage = server_config['num_epochs_first_stage']
        self.num_epochs_train_classifier = server_config['num_epochs_train_classifier']
        self.num_epochs_train_weight = server_config['num_epochs_train_weight']
        self.batch_size = server_config["batch_size"]
        self.num_clients = 0
        self.test_dataloader = {}
        self.path = server_config['data_path']
        self.featurizer = None
        self.classifier = None
        self.post_init()

    # setup
    def init_model(self):
        """
        The model setup depends on the datasets. 
        """
        self._featurizer = self.ds_bundle.featurizer
        self._classifier = self.ds_bundle.classifier
        self._cat_classifier = CatClassifier(self.ds_bundle.feature_dimension, self.ds_bundle.n_classes, len(self.clients))
        self._invariant_classifier = InvariantClassifier(len(self.clients))
        # self._model = nn.Sequential(self._featurizer, self._cat_classifier, self._invariant_classifier)
        self._model = nn.Sequential(self._featurizer, self._classifier)
        self.featurizer = nn.DataParallel(self._featurizer)
        self.invariant_classifier = nn.DataParallel(self._invariant_classifier)
        self.cat_classifier = nn.DataParallel(self._cat_classifier)
        self.model = nn.DataParallel(self._model)

        if self.loading:
            model_file = "/local/scratch/a/bai116/models/officehome_FourierMixup_3_56.pth"
            # model_file = "/local/scratch/a/bai116/models/pacs_ERM_125_27.pth"
            self.load_model(model_state_dict=torch.load(model_file))
            # featurizer_path = f"{self.ds_bundle.name}_{self.__class__.__name__}_{self.clients[0].name}_featurizer_{self.id}_{self._round}.pth"
            # featurizer_param = torch.load(self.path + featurizer_path)
            # self.load_model(featurizer_state_dict=featurizer_param)
            # cat_classifier_path = f"{self.ds_bundle.name}_{self.__class__.__name__}_{self.clients[0].name}_cat_classifier_{self.id}_{self._round}.pth"
            # cat_classifier_param = torch.load(self.path + cat_classifier_path)
            # self.load_model(cat_classifier_state_dict=cat_classifier_param)
            # print(self.path + cat_classifier_path)
    def register_testloader(self, dataloaders):
        self.test_dataloader.update(dataloaders)
    
    # "Remote Control" and Communicate
    def update_clients(self, num_epochs, featurizer=True, classifier=True):
        """
        Description: This method will call the client.fit methods. 
        Usually doesn't need to override in the derived class.
        """
        message = f"[Round: {str(self._round).zfill(3)}] Start updating selected {len(self.clients)} clients...!"
        logging.debug(message)
        total_sample_size = 0
        for client in tqdm(self.clients, leave=False):
            client.fit(num_epochs=num_epochs,featurizer=featurizer, classifier=classifier)
            client_size = len(client)
            total_sample_size += client_size
        message = f"[Round: {str(self._round).zfill(3)}] ...{len(self.clients)} clients are updated (with total sample size: {str(total_sample_size)})!"
        logging.debug(message)
        return total_sample_size

    def evaluate_clients(self):
        message = f"[Round: {str(self._round).zfill(3)}] Evaluate clients' models...!"
        logging.debug(message)
        del message
        
        for client in tqdm(self.clients):
            client.client_evaluate()

    def aggregate_classifier(self):
        coeff = self._invariant_classifier.state_dict()['network.weight']
        weight, bias = self._cat_classifier.aggregate(coeff)
        self._classifier.load_state_dict({'weight': weight, "bias": bias})

    def transmit_model(self, only_parameter=True, featurizer=True, classifier=True):
        assert featurizer or classifier
        
            # _classifier.weight = 
            # _classifier.bias = 
        """
            Description: Send the updated global model to selected/all clients.
            This method could be overriden by the derived class if one algorithm requires to send things other than model parameters.
        """
        if not only_parameter:
            for i, client in enumerate(self.clients):
                if featurizer:
                    client._featurizer = copy.deepcopy(self._featurizer)
                if classifier:
                    local_classifier = torch.nn.Linear(self.ds_bundle.feature_dimension, self.ds_bundle.n_classes)
                    local_classifier.load_state_dict(self._cat_classifier[i])
                    client._classifier = copy.deepcopy(local_classifier)
        else:
            for client in tqdm(self.clients, leave=False):
                if featurizer:
                    client._featurizer.load_state_dict(self._featurizer.state_dict())
                if classifier:
                    client._classifier.load_state_dict(self._classifier.state_dict())
        message = f"[Round: {str(self._round).zfill(3)}] ...successfully transmitted models to all {str(self.num_clients)} clients!"
        logging.debug(message)
        del message

    def load_model(self, model_state_dict=None, featurizer_state_dict=None, classifier_state_dict=None, cat_classifier_state_dict=None, invariant_classifier_state_dict=None):
        if model_state_dict:
            self.model.load_state_dict(model_state_dict)
        else:
            if featurizer_state_dict:
                self._featurizer.load_state_dict(featurizer_state_dict)
            if classifier_state_dict:
                self._classifier.load_state_dict(classifier_state_dict)
            if cat_classifier_state_dict:   
                self._cat_classifier.load_state_dict(cat_classifier_state_dict)
            if invariant_classifier_state_dict:
                self._invariant_classifier.load_state_dict(invariant_classifier_state_dict)

    def collect_model(self):
        featurizer_list = []
        classifier_list = []
        for client in self.clients:
            classifier_list.append(client._classifier.state_dict())
            featurizer_list.append(client._featurizer.state_dict())
        return featurizer_list, classifier_list
    
    def collect_post_data(self):
        data, labels, metadata = [], [], []
        for client in self.clients:
            client.set_cat_classifier(self._cat_classifier)
            z, y, meta = client.prepare_post_data()
            data.append(z)
            labels.append(y)
            metadata.append(meta)
        data = torch.cat(data)
        labels = torch.cat(labels)
        metadata = torch.cat(metadata)
        self.post_dataset = TensorMultiDataset([data, labels, metadata])
        self.post_loader = get_train_loader("group", self.post_dataset, batch_size=self.batch_size, uniform_over_groups=None, grouper=self.ds_bundle.grouper, distinct_groups=True, n_groups_per_batch=self.n_groups_per_batch)

    # Server methods
    def fit(self):
        """
        Description: Execute the whole process of the federated learning.
        Stage 1: FedAvg.
        Train a featurizer using FedAvg.
        Stage 2: Local classifier training.
        Train a local classifier using ERM.
        Stage 3: Center classifier training.
        Train a center classifier using ERM+MinDiff
        """
        # Stage 1: Featurizer, Classifier training using FedAvg.
        message = f"Round \t "
        for testset_name in self.test_dataloader.keys():
            message += f"{testset_name} \t "
        logging.info(message)
        self.transmit_model(only_parameter=False)
        # Stage 1: Federated Averaging
        print(self._round)
        print(self.num_rounds_first_stage)
        print(self.num_rounds_first_stage + self.num_rounds_second_stage)
        while self._round < self.num_rounds_first_stage:
            
            # print(self.cat_classifier.state_dict())
            print("num of rounds at Stage 1: {}; Total: {};)".format(self._round, self.num_rounds_first_stage))
            self.transmit_model()
            # updated selected clients with local dataset
            total_sample_size = self.update_clients(num_epochs=self.num_epochs_first_stage, featurizer=True, classifier=True)
            # average each updated model parameters of the selected clients and update the global model
            mixing_coefficients = [len(client) / total_sample_size for client in self.clients]
            self.aggregate(mixing_coefficients, featurizer=True, classifier=True)
            self._round += 1
            self.save_model(self._round)
            self.evaluate_global_model()
        # Stage 2: train an personalize classifier as well as the invariant classifier . 
        # 
        while self._round < self.num_rounds_first_stage + self.num_rounds_second_stage: 
            print("num of rounds at Stage 2: {}; Total: {};)".format(self._round-self.num_rounds_first_stage, self.num_rounds_second_stage))
            self.evaluate_global_model()
            self.transmit_model(featurizer=True, classifier=True)
            total_sample_size = self.update_clients(num_epochs=self.num_epochs_train_classifier, featurizer=False, classifier=True)
            mixing_coefficients = [len(client) / total_sample_size for client in self.clients]
            self.aggregate(mixing_coefficients, featurizer=False, classifier=True)
            self.collect_post_data()
            self.post_fit()
            self._round += 1
            self.save_model(self._round)
        self.evaluate_global_model()
            
        
    def post_fit(self):
        self.optimizer = eval(self.optimizer_name)(self.invariant_classifier.parameters(), **self.optim_config)
        for r in range(self.num_epochs_train_weight):
            # print("num of rounds (Stage 2, Total): {}, {}".format(r, self._round))
            # self._round += 1
            self.post_train()
            self.aggregate_classifier()
        

    def evaluate_global_model(self):
        """Evaluate the global model using the global holdout dataset (self.data)."""
        def evaluate_single(dataloader):
            with torch.no_grad():
                y_pred = None
                y_true = None
                for batch in tqdm(dataloader):
                    data, labels, meta_batch = batch[0], batch[1], batch[2]
                    if isinstance(meta_batch, list):
                        meta_batch = meta_batch[0]
                    data, labels = data.to(self.device), labels.to(self.device)
                    prediction = self.model(data)
                    if self.ds_bundle.is_classification:
                        prediction = torch.argmax(prediction, dim=-1)
                    if y_pred is None:
                        y_pred = prediction
                        y_true = labels
                        metadata = meta_batch
                    else:
                        y_pred = torch.cat((y_pred, prediction))
                        y_true = torch.cat((y_true, labels))
                        metadata = torch.cat((metadata, meta_batch))
                metric = self.ds_bundle.dataset.eval(y_pred.to("cpu"), y_true.to("cpu"), metadata.to("cpu"))
                if self.device == "cuda": torch.cuda.empty_cache()
            return metric
        self.model.eval()
        self.model.to(self.device)
        message = f"{str(self._round).zfill(3)} \t "
        for name, dataloader in self.test_dataloader.items():
            metric = evaluate_single(dataloader)
            print(name)
            print(metric[1])
            for value in metric[0].values():
                message += f"{value:05.4} "
            message += f"\t"
        logging.info(message)
        self.model.to("cpu")

    def aggregate(self, coefficients, featurizer=True, classifier=True):
        assert featurizer or classifier
        """average the featurizer and update the classifier"""
        message = f"[Round: {str(self._round).zfill(3)}] Aggregate updated weights of {len(self.clients)} clients...!"
        logging.debug(message)
        del message
        local_featurizer_list, local_classifier_list = self.collect_model()
        if featurizer:
            averaged_featurizer = OrderedDict()
            if len(local_featurizer_list): 
                for it, local_weights in enumerate(local_featurizer_list):
                    for key in local_weights.keys():
                        if it == 0:
                            averaged_featurizer[key] = coefficients[it] * local_weights[key]
                        else:
                            averaged_featurizer[key] += coefficients[it] * local_weights[key]
            self._featurizer.load_state_dict(averaged_featurizer)
        if classifier:
            param = {'network.weight': torch.tensor(coefficients).reshape(1,-1)}
            self._invariant_classifier.load_state_dict(param)
            self._cat_classifier.concat(local_classifier_list)
            self.aggregate_classifier()

    # post train method related.
    def post_init(self):
        self.penalty_weight = self.server_config['penalty_weight']
        self.penalty_anneal_iters = self.server_config['penalty_anneal_iters']
        self.scale = torch.tensor(1.).to(self.device).requires_grad_()
        self.optimizer_name = self.server_config['optimizer']
        self.optim_config = self.server_config['optimizer_config']
        self.n_groups_per_batch = self.server_config['n_groups_per_batch']
        self.update_count = 0


    def post_train(self):
        epoch_loss, epoch_penalty = 0, 0
        self.invariant_classifier.to(self.device)
        # We use IRMv1 here.
        for batch in self.post_loader:
            x, y_true, metadata = batch
            x = x.to(self.device)
            y_true = y_true.to(self.device)
            g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
            metadata = metadata.to(self.device)
            y_pred = self.invariant_classifier(x)
            unique_groups, group_indices, _ = split_into_groups(g)
            n_groups_per_batch = unique_groups.numel()
            avg_loss = 0.
            penalty = 0.
            # torch.save(results['y_pred'], "pred.pt")
            # torch.save(results['y_true'], "true.pt")
            for i_group in group_indices: # Each element of group_indices is a list of indices
                group_losses, _ = self.ds_bundle.loss.compute_flattened(y_pred[i_group] * self.scale, y_true[i_group], return_dict=False)
                if group_losses.numel()>0:
                    avg_loss += group_losses.mean()
                penalty += self.irm_penalty(group_losses)
            avg_loss /= n_groups_per_batch
            penalty /= n_groups_per_batch
            if self.update_count >= self.penalty_anneal_iters:
                penalty_weight = self.penalty_weight
            else:
                penalty_weight = self.update_count / self.penalty_anneal_iters
            # print(self.update_count, penalty_weight)
            objective = avg_loss + penalty * penalty_weight
            # wprint(avg_loss, penalty)
            if self.update_count == self.penalty_anneal_iters:
                # Reset Adam, because it doesn't like the sharp jump in gradient
                # magnitudes that happens at this step.
                self.optimizer = eval(self.optimizer_name)(self.invariant_classifier.parameters(), **self.optim_config)
            if objective.grad_fn is None:
                pass
            objective.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            print(self.invariant_classifier.state_dict())
            self.update_count += 1
            epoch_loss += avg_loss
            epoch_penalty += penalty
        print(self.update_count, epoch_loss, epoch_penalty)
        self.invariant_classifier.to('cpu')
    
    def irm_penalty(self, losses):
        grad_1 = autograd.grad(losses[0::2].mean(), [self.scale], create_graph=True)[0]
        grad_2 = autograd.grad(losses[1::2].mean(), [self.scale], create_graph=True)[0]
        result = torch.sum(grad_1 * grad_2)
        return result

    def save_model(self, num_epoch):
        for model in ["featurizer", "classifier", "cat_classifier", "invariant_classifier"]:
            method = "self._" + model + ".state_dict()"
            model_name = f"{self.ds_bundle.name}_{self.__class__.__name__}_{self.clients[0].name}_{model}_{self.id}_{num_epoch}.pth"
            try:
                torch.save(eval(method), self.path + model_name)
            except:
                continue

class ServerFish(ServerIRMv1):
    def post_init(self):
        self.meta_lr = self.server_config['meta_lr']
        self.optimizer_name = self.server_config['optimizer']
        self.optim_config = self.server_config['optimizer_config']
        self.n_groups_per_batch = self.server_config['n_groups_per_batch']
        self.update_count = 0


    def post_train(self):
        self.invariant_classifier.to(self.device)
        # We use IRMv1 here.
        for batch in self.post_loader:
            param_dict = ParamDict(copy.deepcopy(self.invariant_classifier.state_dict()))
            x, y_true, metadata = batch
            x = x.to(self.device)
            y_true = y_true.to(self.device)
            g = self.ds_bundle.grouper.metadata_to_group(metadata).to(self.device)
            metadata = metadata.to(self.device)
            unique_groups, group_indices, _ = split_into_groups(g)
            
            # torch.save(results['y_pred'], "pred.pt")
            # torch.save(results['y_true'], "true.pt")
            for i_group in group_indices: # Each element of group_indices is a list of indices
                # print(i_group)
                group_loss = self.ds_bundle.loss.compute(self.invariant_classifier(x[i_group]), y_true[i_group], return_dict=False)
                # print(group_loss)
                # print(group_loss.grad_fn)
                if group_loss.grad_fn is None:
                    # print('jump')
                    pass
                else:
                    group_loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            param_dict = param_dict + self.meta_lr * (ParamDict(self.invariant_classifier.state_dict()) - param_dict)
            self.invariant_classifier.load_state_dict(copy.deepcopy(param_dict))
            self.update_count += 1
        self.invariant_classifier.to('cpu')

class ServerVREx(ServerIRMv1):
    def irm_penalty(self, losses):
        mean = losses.mean()
        penalty = ((losses - mean) ** 2).mean()
        return penalty