# -*- coding: utf8 -*-
#

import pathlib

SENTENCE = 0
EDU = 1
DIALOGUE = 2
PRETRAIN_MODEL_PATH = pathlib.Path('/home/u210110513/hfmodels/bert-base-chinese')

#########################################################################
# 对话分析配置
DIALOGUE_DATA_PATH = pathlib.Path('.').parent.joinpath('data').joinpath('dialogue')

DIALOGUE_TRAIN_PATH = DIALOGUE_DATA_PATH.joinpath('test.json')
DIALOGUE_TEST_PATH = DIALOGUE_DATA_PATH.joinpath('train.json')
DIALOGUE_DEV_PATH = DIALOGUE_DATA_PATH.joinpath('train.json')

MODEL_PATH = pathlib.Path('.').joinpath('savepoint').joinpath('dialogue').joinpath('savepoint_506')

#########################################################################


#########################################################################
# edu切分配置
EDU_DATA_PATH = pathlib.Path('.').parent.joinpath('data').joinpath('dialogue')

EDU_TRAIN_PATH = EDU_DATA_PATH.joinpath('train.json')
EDU_TEST_PATH = EDU_DATA_PATH.joinpath('test.json')
EDU_DEV_PATH = EDU_DATA_PATH.joinpath('train.json')

# MODEL_PATH = pathlib.Path('.').joinpath('savepoint').joinpath('edu').joinpath('savepoint_506')

#########################################################################


#########################################################################
# 句法分析配置（ctb）
DATA_PATH = pathlib.Path('.').parent.joinpath('data').joinpath('origin').joinpath('ctb70')

TRAIN_PATH = DATA_PATH.joinpath('train.ctb70.conll')
DEV_PATH = DATA_PATH.joinpath('dev.ctb70.conll')
TEST_PATH = DATA_PATH.joinpath('test.ctb70.conll')

# MODEL_PATH = pathlib.Path('.').joinpath('savepoint').joinpath('savepoint_53')

#########################################################################


#########################################################################
# 句法分析配置（ptb）
# DATA_PATH = pathlib.Path('.').parent.joinpath('data').joinpath('origin').joinpath('ptb')

# TRAIN_PATH = DATA_PATH.joinpath('ptb.english.conll.train.txt.opentest.tag')
# DEV_PATH = DATA_PATH.joinpath('ptb.english.conll.dev.txt.tag')
# TEST_PATH = DATA_PATH.joinpath('ptb.english.conll.test.txt.tag')

# PRETRAIN_MODEL_PATH = pathlib.Path('/home/u210110513/hfmodels/bert-base-uncased')
# MODEL_PATH = pathlib.Path('.').joinpath('savepoint').joinpath('savepoint_414_bert')

#########################################################################


# if not MODEL_PATH.exists():
#     MODEL_PATH.mkdir(exist_ok=True)
