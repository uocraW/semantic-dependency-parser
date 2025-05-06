# -*- coding: utf8 -*-
#
import math
from typing import Optional, Union

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoTokenizer, AdamW, get_linear_schedule_with_warmup, set_seed

from config import MODEL_PATH, SENTENCE, EDU, DIALOGUE
from metric import ChartMetric, EvaluateMetric
from model import SemanticDependencyModel
from transform import get_labels, SDPTransform, get_tags, EDUCutTransform, get_edu_labels, get_edu_tags, get_dialogue_labels, DialogueTransform
from utils import logger

class SemanticDependencyParser(object):
    def __init__(self, enable_tag=True, config=SENTENCE):
        self.model: Optional[SemanticDependencyModel, None] = None
        self.tokenizer = None
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        if config == SENTENCE:
            self.labels = get_labels()
            self.tags = get_tags()
        elif config == EDU:
            self.labels = get_edu_labels()
            self.tags = get_edu_tags()
        else:
            self.labels = get_dialogue_labels()
        self.enable_tag = enable_tag
        self.config = config

    def build_model(self, transformer):
        if self.enable_tag:
            self.model = SemanticDependencyModel(
                transformer=transformer,
                n_labels=len(self.labels),
                n_tags=len(self.tags),
                enable_tag=self.enable_tag
            )
        else:
                self.model = SemanticDependencyModel(
                transformer=transformer,
                n_labels=len(self.labels),
                enable_tag=self.enable_tag
            )
        self.model.to(self.device)
        logger.info(self.model)
        return self.model

    def build_tokenizer(self, pretrained_model_name: str):
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)

    def build_optimizer(
            self,
            warmup_steps: Union[float, int],
            num_training_steps: int,
            lr_transformer=2e-5, lr_model=1e-3, weight_decay=0.01,
    ):
        """
        https://github.com/huggingface/transformers/blob/7b75aa9fa55bee577e2c7403301ed31103125a35/src/transformers/trainer.py#L232
        :param warmup_steps:
        :param num_training_steps:
        :param lr:
        :param weight_decay:
        :return:
        """
        if warmup_steps <= 1:
            warmup_steps = int(num_training_steps * warmup_steps)
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ["bias", "LayerNorm.weight"]
        
        transformer_parameters = [
            {
                "params": [p for n, p in self.model.encoder.transformer.named_parameters() if not any(nd in n for nd in no_decay)],
                "lr": lr_transformer,
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.encoder.transformer.named_parameters() if any(nd in n for nd in no_decay)],
                "lr": lr_transformer,
                "weight_decay": 0.0,
            },
        ]
        model_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters()
                           if "encoder.transformer" not in n and not any(nd in n for nd in no_decay)],
                "lr": lr_model,
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                           if "encoder.transformer" not in n and any(nd in n for nd in no_decay)],
                "lr": lr_model,
                "weight_decay": 0.0,
            },
        ]
        optimizer_grouped_parameters = transformer_parameters + model_parameters
        optimizer = AdamW(optimizer_grouped_parameters)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=num_training_steps
        )
        return optimizer, scheduler

    def build_dataloader(self, path, transformer, batch_size, shuffle, config=SENTENCE):
        if config == SENTENCE:
            return SDPTransform(
                path=path,
                transformer=transformer,
                device=self.device
            ).to_dataloader(batch_size=batch_size, shuffle=shuffle)
        elif config == EDU:
            return EDUCutTransform(
                path=path,
                transformer=transformer,
                device=self.device
            ).to_dataloader(batch_size=batch_size, shuffle=shuffle)
        else:
            return DialogueTransform(
                path=path,
                transformer=transformer,
                device=self.device
            ).to_dataloader(batch_size=batch_size, shuffle=shuffle)

    def fit(self, train_path, dev_path, epoch=100, lr_transformer=2e-5, lr_model=1e-3, pretrained_model_name=None, batch_size=32,
            warmup_steps=0.1, patience=10):
        set_seed(seed=10403)

        self.build_tokenizer(pretrained_model_name=pretrained_model_name)

        train_dataloader = self.build_dataloader(
            path=train_path,
            transformer=self.tokenizer,
            batch_size=batch_size,
            shuffle=True,
            config=self.config,
        )
        dev_dataloader = self.build_dataloader(
            path=dev_path,
            transformer=self.tokenizer,
            batch_size=batch_size,
            shuffle=False,
            config=self.config,
        )

        self.build_model(transformer=pretrained_model_name)

        optimizer, scheduler = self.build_optimizer(
            warmup_steps=warmup_steps, num_training_steps=len(train_dataloader) * epoch,
            lr_transformer=lr_transformer, lr_model=lr_model
        )
        return self.fit_loop(train_dataloader, dev_dataloader, epoch=epoch, optimizer=optimizer,
                             scheduler=scheduler, patience=patience)

    def fit_loop(self, train, dev, epoch, optimizer, scheduler, patience):
        # loss
        min_train_loss, min_dev_loss = math.inf, math.inf
        # metric
        max_dev_metric = 0.0

        best_epoch = 0
        no_improve_count = 0

        for _epoch in range(1, epoch + 1):
            train_loss = self.fit_dataloader(
                train=train,
                optimizer=optimizer,
                scheduler=scheduler
            )
            if train_loss < min_train_loss:
                logger.info(f'Epoch:{_epoch} save min train loss:{train_loss} model')
                min_train_loss = train_loss
                self.save_weights(
                    save_path=str(MODEL_PATH.joinpath(f'train_loss_{train_loss:.4e}.pt'))
                )

            dev_loss, dev_metric = self.evaluate_dataloader(dev)

            if dev_loss < min_dev_loss:
                logger.info(f'Epoch:{_epoch} save min dev loss:{dev_loss} model')
                min_dev_loss = dev_loss
                self.save_weights(
                    save_path=str(MODEL_PATH.joinpath(f'dev_loss_{dev_loss:.4e}.pt'))
                )

            if dev_metric.las > max_dev_metric:
                logger.info(f'Epoch:{_epoch} save max dev metric:{dev_metric.las:.4%} model')
                max_dev_metric = dev_metric.las
                best_epoch = _epoch
                no_improve_count = 0
                self.save_weights(
                    save_path=str(MODEL_PATH.joinpath(f'dev_metric_{dev_metric.las:.4e}.pt'))
                )
            else:
                no_improve_count += 1

            # logger.info(
            #     f'Epoch:{_epoch} lr: {scheduler.get_last_lr()[0]:.4e} train loss: {train_loss} ' + \
            #     f'dev loss: {dev_loss} ' + \
            #     f'dev metric: {dev_metric}'
            # )

            lrs = ', '.join([f'{lr:.4e}' for lr in scheduler.get_last_lr()])
            logger.info(
                f'Epoch:{_epoch} lrs: [{lrs}] train loss: {train_loss} dev loss: {dev_loss} dev metric: {dev_metric}'
            )

            if no_improve_count >= patience:
                logger.info(f"Early stopping triggered at epoch {_epoch}, best dev LAS: {max_dev_metric:.4%} at epoch {best_epoch}")
                break

    def fit_dataloader(self, train, optimizer, scheduler):
        self.model.train()
        total_loss = 0.

        for data in tqdm(train, desc='fit_dataloader'):
            if len(data) == 3:
                subwords, tags, labels = data
            else:
                subwords, labels = data
                tags = None

            word_mask = subwords.ne(self.tokenizer.pad_token_id)
            mask = word_mask if len(subwords.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_label = self.model(subwords, tags)
            loss = self.model.loss(s_edge, s_label, labels, mask)
            total_loss += loss.item()
            loss.backward()

            self._step(optimizer=optimizer, scheduler=scheduler)
        total_loss /= len(train)
        return total_loss

    @torch.no_grad()
    def evaluate_dataloader(self, dev):
        self.model.eval()

        total_loss, metric = 0, EvaluateMetric()

        for data in tqdm(dev, desc='evaluate_dataloader'):
            if len(data) == 3:
                subwords, tags, labels = data
            else:
                subwords, labels = data
                tags = None

            word_mask = subwords.ne(self.tokenizer.pad_token_id)
            mask = word_mask if len(subwords.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_label = self.model(subwords, tags)
            loss = self.model.loss(s_edge, s_label, labels, mask)
            total_loss += loss.item()
            # if not label_preds.eq(-1).all():
            #     print('debug')

            label_preds = self.model.decode(s_edge, s_label)    # (B, L+1, L+1)

            # ---------- 让三者同尺寸 ----------
            L = s_edge.size(1)                                  # 真实节点数 (不含 ROOT)
            label_preds = label_preds[:, :L, :L]
            labels      = labels[:,  :L, :L]
            mask        = mask[:,    :L, :L]

            metric(label_preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
        total_loss /= len(dev)

        return total_loss, metric

    def load(self, pretrained_model_name, model_path: str, device='cpu'):
        self.device = torch.device(device)
        if not self.loaded:
            self.build_tokenizer(pretrained_model_name=pretrained_model_name)
            self.build_model(transformer=pretrained_model_name)
            self.load_weights(save_path=model_path)
            self.loaded = True

    def _step(self, optimizer, scheduler):
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

    def save_weights(self, save_path):
        if not isinstance(self.model, nn.DataParallel):
            torch.save(self.model.state_dict(), save_path)
        else:
            torch.save(self.model.module.state_dict(), save_path)

    def load_weights(self, save_path):
        if not isinstance(self.model, nn.DataParallel):
            self.model.load_state_dict(torch.load(save_path), strict=False)
        else:
            self.model.module.load_state_dict(torch.load(save_path), strict=False)


    @torch.no_grad()
    def mst_evaluate_dataloader(self, dev):
        self.model.eval()

        total_loss, metric = 0, EvaluateMetric()

        for data in tqdm(dev, desc='mst_evaluate_dataloader'):
            subwords, tags, labels = data
            word_mask = subwords.ne(self.tokenizer.pad_token_id)
            mask = word_mask if len(subwords.shape) < 3 else word_mask.any(-1)
            mask = mask.unsqueeze(1) & mask.unsqueeze(2)
            mask[:, 0] = 0
            s_edge, s_label = self.model(subwords, tags)
            loss = self.model.loss(s_edge, s_label, labels, mask)
            total_loss += loss.item()

            label_preds = self.model.mst_decode(s_edge, s_label)
            metric(label_preds.masked_fill(~mask, -1), labels.masked_fill(~mask, -1))
        total_loss /= len(dev)

        return total_loss, metric

    loaded = False
