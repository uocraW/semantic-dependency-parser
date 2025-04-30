# -*- coding: utf8 -*-
#


import torch
from config import DEV_PATH, PRETRAIN_MODEL_PATH
from semantic_dependency_parser import SemanticDependencyParser
import os

import pathlib
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

eva_model_path = str(pathlib.Path('.').parent.joinpath('savepoint').joinpath('savepoint_412_bilstm').joinpath('dev_metric_8.8862e-01.pt'))


m = SemanticDependencyParser()
m.load(
    pretrained_model_name=PRETRAIN_MODEL_PATH,
    model_path=eva_model_path,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


dev = m.build_dataloader(
    DEV_PATH,
    transformer=m.tokenizer,
    batch_size=2,
    shuffle=False
)

# print(m.mst_evaluate_dataloader(dev))
print(m.evaluate_dataloader(dev))
