# -*- coding: utf8 -*-
#

import pathlib

DATA_PATH = pathlib.Path('.').parent.joinpath('data').joinpath('ctb70')

TRAIN_PATH = DATA_PATH.joinpath('train.ctb70.conll')
DEV_PATH = DATA_PATH.joinpath('dev.ctb70.conll')
TEST_PATH = DATA_PATH.joinpath('test.ctb70.conll')

MODEL_PATH = pathlib.Path('.').parent.joinpath('data').joinpath('savepoint')
if not MODEL_PATH.exists():
    MODEL_PATH.mkdir(exist_ok=True)
