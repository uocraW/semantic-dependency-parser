# -*- coding: utf8 -*-
#

import torch
from config import MODEL_PATH, DEV_PATH
from semantic_dependency_parser import SemanticDependencyParser
import os
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1, 2, 3, 4, 5, 6, 7'


m = SemanticDependencyParser()
m.load(
    pretrained_model_name='/data/chd/bert-base-chinese',
    model_path=str(MODEL_PATH.joinpath('dev_metric_3.4178e-01.pt')),
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


dev = m.build_dataloader(
    DEV_PATH,
    transformer=m.tokenizer,
    batch_size=2,
    shuffle=False
)

m.mst_evaluate(dev)
# m.evaluate_dataloader(dev)
