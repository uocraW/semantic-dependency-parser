# -*- coding: utf8 -*-
#

import pathlib

SENTENCE = 0
DIALOGUE = 1

#########################################################################

DATA_PATH = pathlib.Path('.').parent.joinpath('data').joinpath('origin').joinpath('ctb70')
DIALOGUE_DATA_PATH = pathlib.Path('.').parent.joinpath('data').joinpath('dialogue')

TRAIN_PATH = DATA_PATH.joinpath('train.ctb70.conll')
DEV_PATH = DATA_PATH.joinpath('dev.ctb70.conll')
TEST_PATH = DATA_PATH.joinpath('test.ctb70.conll')
DIALOGUE_TRAIN_PATH = DIALOGUE_DATA_PATH.joinpath('train.json')
DIALOGUE_TEST_PATH = DIALOGUE_DATA_PATH.joinpath('test.json')

PRETRAIN_MODEL_PATH = pathlib.Path('/home/u210110513/hfmodels/bert-base-chinese')

MODEL_PATH = pathlib.Path('.').joinpath('savepoint').joinpath('dialogue').joinpath('savepoint_428')

#########################################################################

# DATA_PATH = pathlib.Path('.').parent.joinpath('data').joinpath('ptb')

# TRAIN_PATH = DATA_PATH.joinpath('ptb.english.conll.train.txt.opentest.tag')
# DEV_PATH = DATA_PATH.joinpath('ptb.english.conll.dev.txt.tag')
# TEST_PATH = DATA_PATH.joinpath('ptb.english.conll.test.txt.tag')

# PRETRAIN_MODEL_PATH = pathlib.Path('/home/u210110513/hfmodels/bert-base-uncased')
# MODEL_PATH = pathlib.Path('.').joinpath('savepoint').joinpath('savepoint_414_bert')

#########################################################################

if not MODEL_PATH.exists():
    MODEL_PATH.mkdir(exist_ok=True)
