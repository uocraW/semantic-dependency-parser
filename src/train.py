# -*- coding: utf8 -*-
#
from config import TRAIN_PATH, DEV_PATH
from semantic_dependency_parser import SemanticDependencyParser
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

SemanticDependencyParser().fit(
    train_path=TRAIN_PATH,
    dev_path=DEV_PATH,
    pretrained_model_name='/data/hfmodel/bert-base-chinese',
    lr=1e-4,
    batch_size=32,
    epoch=100
)
