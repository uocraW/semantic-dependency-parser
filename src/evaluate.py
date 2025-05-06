# -*- coding: utf8 -*-
#
import torch
from config import DEV_PATH, PRETRAIN_MODEL_PATH, EDU, DIALOGUE
from config import EDU_DEV_PATH, DIALOGUE_DEV_PATH
from semantic_dependency_parser import SemanticDependencyParser
import os

import pathlib
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


#################################################
# 句法依存
# eva_model_path = str(pathlib.Path('.').parent.joinpath('savepoint').joinpath('savepoint_412_bilstm').joinpath('dev_metric_8.8862e-01.pt'))

# print("\nload sentence parser...\n")
# m = SemanticDependencyParser()
# m.load(
#     pretrained_model_name=PRETRAIN_MODEL_PATH,
#     model_path=eva_model_path,
#     device='cuda' if torch.cuda.is_available() else 'cpu'
# )

# print("\nbuild sentence dev dataloader...\n")
# dev = m.build_dataloader(
#     DEV_PATH,
#     transformer=m.tokenizer,
#     batch_size=2,
#     shuffle=False
# )

# print('\nbegin sentence evaluate...\n')
# # # print(m.mst_evaluate_dataloader(dev))
# print(m.evaluate_dataloader(dev))

#################################################
# edu
# edu_model_path = str(pathlib.Path('.').parent.joinpath('savepoint').joinpath('edu').joinpath('savepoint_506').joinpath('dev_metric_9.1510e-01.pt'))

# print("\nload parser...\n")
# parser = SemanticDependencyParser(
#     enable_tag=True,
#     config=EDU
# )
# parser.load(
#     pretrained_model_name=PRETRAIN_MODEL_PATH,
#     model_path=edu_model_path,
#     device='cuda' if torch.cuda.is_available() else 'cpu'
# )

# print("\nbuild dataloader...\n")
# dev = parser.build_dataloader(
#     EDU_DEV_PATH,
#     transformer=parser.tokenizer,
#     batch_size=2,
#     shuffle=False,
#     config=EDU
# )

# print('\nbegin evaluate...\n')
# print(parser.evaluate_dataloader(dev))

#################################################
# dialogue
# dialogue_model_path = str(pathlib.Path('.').parent.joinpath('savepoint').joinpath('dialogue').joinpath('savepoint_505').joinpath('dev_metric_6.8199e-01.pt'))

# print("\nload parser...\n")
# parser = SemanticDependencyParser(
#     enable_tag=False,
#     config=DIALOGUE
# )
# parser.load(
#     pretrained_model_name=PRETRAIN_MODEL_PATH,
#     model_path=dialogue_model_path,
#     device='cuda' if torch.cuda.is_available() else 'cpu'
# )

# print("\nbuild dataloader...\n")
# dev = parser.build_dataloader(
#     DIALOGUE_DEV_PATH,
#     transformer=parser.tokenizer,
#     batch_size=2,
#     shuffle=False,
#     config=DIALOGUE
# )

# print('\nbegin evaluate...\n')
# print(parser.evaluate_dataloader(dev))
