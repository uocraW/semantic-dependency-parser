# -*- coding: utf8 -*-
#
from config import TRAIN_PATH, DEV_PATH, PRETRAIN_MODEL_PATH, DIALOGUE, SENTENCE, DIALOGUE_TRAIN_PATH, DIALOGUE_TEST_PATH
from semantic_dependency_parser import SemanticDependencyParser

# --- 句法parser ---
# SemanticDependencyParser(
#     enable_tag=True,        # 是否使用tag
#     config=DIALOGUE
# ).fit(
#     train_path=TRAIN_PATH,
#     dev_path=DEV_PATH,
#     pretrained_model_name=PRETRAIN_MODEL_PATH,
#     lr_transformer=2e-5,
#     lr_model=1e-3,
#     batch_size=32,
#     epoch=100
# )

# --- edu切分parser ---
SemanticDependencyParser(
    enable_tag=True,        # 是否使用tag
    config=DIALOGUE
).fit(
    train_path=DIALOGUE_TRAIN_PATH,
    dev_path=DIALOGUE_TEST_PATH,
    pretrained_model_name=PRETRAIN_MODEL_PATH,
    lr_transformer=2e-5,
    lr_model=1e-3,
    batch_size=32,
    epoch=100,
)

# --- 对话parser ---
SemanticDependencyParser(
    enable_tag=False,        # 是否使用tag
    config=DIALOGUE
).fit(
    train_path=DIALOGUE_TRAIN_PATH,
    dev_path=DIALOGUE_TEST_PATH,
    pretrained_model_name=PRETRAIN_MODEL_PATH,
    lr_transformer=2e-5,
    lr_model=1e-3,
    batch_size=32,
    epoch=100,
)
