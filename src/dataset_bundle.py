import torch
import torch.nn as nn
import torchvision.transforms as transforms
from wilds.common.grouper import CombinatorialGrouper
from wilds.common.metrics.loss import ElementwiseLoss, Loss, MultiTaskLoss
from wilds.common.metrics.all_metrics import MSE

from .splitter import *
from .models import ResNet, Classifier, CNN
from transformers import DistilBertTokenizerFast


class ObjBundle(object):
    def __init__(self, dataset, feature_dimension) -> None:
        self.dataset = dataset
        self.feature_dimension = feature_dimension
        self.input_shape = self._input_shape
        self.groupby_fields = self._domain_fields
        self.grouper = CombinatorialGrouper(dataset=dataset, groupby_fields=self.groupby_fields)
        self.loss = self._loss()
        self.train_transform = self._train_transform
        self.test_transform = self._test_transform
        self.featurizer = ResNet(self.input_shape, self.feature_dimension)
        self.classifier = Classifier(self.featurizer.n_outputs, self.n_classes)

    @property
    def is_classification(self):
        return True

    @property
    def _train_transform(self):
        raise NotImplementedError

    @property
    def _test_transform(self):
        raise NotImplementedError

    def _loss(self):
        raise NotImplementedError

    @property
    def _input_shape(self):
        return None
    
    @property
    def _domain_fields(self):
        return None

    @property
    def n_classes(self):
        return self.dataset.n_classes

#### Doesn't require re-implemented by derived classes ####
    @property
    def in_channel(self):
        return self._input_shape[0]

    @property
    def name(self):
        return self.__class__.__name__.lower()


class IWildCam(ObjBundle):
    @property
    def _train_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1],self._input_shape[2])),
            transforms.RandomResizedCrop(self._input_shape[1], scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @property
    def _test_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1], self._input_shape[2])), 
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _loss(self):
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
        

    @property
    def _input_shape(self):
        return (3, 448, 448)
    
    @property
    def _domain_fields(self):
        return ['location',]


class PACS(ObjBundle):
    def _loss(self):
        return ElementwiseLoss(loss_fn=nn.CrossEntropyLoss(reduction='none', ignore_index=-100))
    
    @property
    def _input_shape(self):
        return (3, 224, 224)
    
    @property
    def _train_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1],self._input_shape[2])),
            transforms.RandomResizedCrop(self._input_shape[1], scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomGrayscale(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            
        ])
        
    @property
    def _test_transform(self):
        return transforms.Compose([
            transforms.Resize((self._input_shape[1], self._input_shape[2])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def _domain_fields(self):
        return ['domain',]


class Poverty(ObjBundle):
    @property
    def is_classification(self):
        return False

    @property
    def _train_transform(self):
        return transforms.Compose([])

    @property
    def _test_transform(self):
        return transforms.Compose([])

    def _loss(self):
        return MSE(name='loss')

    @property
    def _input_shape(self):
        return (8, 224, 224)
    
    @property
    def _domain_fields(self):
        return ['country',]

    def _oracle_training_set(self):
        return False

    @property
    def n_classes(self):
        return 1

class OfficeHome(PACS):
    def __init__(self, dataset, feature_dimension) -> None:
        super().__init__(dataset, feature_dimension)

