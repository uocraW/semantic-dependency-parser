# -*- coding: utf8 -*-
#
import json
from typing import List

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import dataset, dataloader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from typing import Dict

import hanlp

# from src.config import TRAIN_PATH, DATA_PATH, DEV_PATH
from config import TRAIN_PATH, DATA_PATH, DEV_PATH, PRETRAIN_MODEL_PATH, DIALOGUE_DATA_PATH, DIALOGUE_TEST_PATH, DIALOGUE_TRAIN_PATH

hanlp.pretrained.pos.ALL
pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)

class CoNLLSentence(object):
    def __init__(self, lines: List[str]):
        self.values = [] # each line store a word with all info
        self.annotations = dict()

        for i, line in enumerate(lines):
            value = line.split('\t')
            if value[0].startswith('#') or not value[0].isdigit():
                # negetive index for annotations or invalid lines
                self.annotations[-i - 1] = line
                continue
            self.annotations[len(self.values)] = line
            self.values.append(value)
        self.values = list(zip(*self.values))

    def __repr__(self):
        # cover the raw lines
        merged = {**self.annotations,
                  **{i: '\t'.join(map(str, line))
                     for i, line in enumerate(zip(*self.values))}}
        return '\n'.join(merged.values()) + '\n'

    @property
    def words(self):
        return self.values[1]

    @property
    def tags(self):
        return self.values[3]

    def get_labels(self):
        all_edges, all_labels = self.values[6], self.values[7]
        labels = [[None] * (len(all_labels) + 1) for _ in range(len(all_labels) + 1)]
        for i in range(len(all_edges)):
            edge, label = all_edges[i], all_labels[i]
            if edge != "_" and label != "???":
                # label: [length+1, length+1] start from index 1
                labels[i+1][int(edge)] = label
        return labels

    def __len__(self):
        return len(self.words)


def get_labels() -> dict:
    if DATA_PATH.joinpath('label_map.json').exists():
        with open(DATA_PATH.joinpath('label_map.json'), 'r') as f:
            return json.loads(f.read())

    # process and make a label_map
    label_map = {'[PAD]': 0}

    def _i(path):
        with open(path, 'r') as f:
            lines = [line.strip() for line in f]
        i, start, sentences = 0, 0, []
        for line in tqdm(lines, desc='get labels'):
            if not line:
                sentences.append(CoNLLSentence(lines[start:i]))
                start = i + 1
            i += 1

        for s in sentences:
            for line in s.get_labels():
                for label in line:
                    if label is not None:
                        label_map.setdefault(label, len(label_map))

    _i(TRAIN_PATH)
    _i(DEV_PATH)
    with open(DATA_PATH.joinpath('label_map.json'), 'w') as f:
        f.write(json.dumps(label_map, ensure_ascii=False, indent=2))
    return label_map


def get_tags() -> dict:
    if DATA_PATH.joinpath('tag_map.json').exists():
        with open(DATA_PATH.joinpath('tag_map.json'), 'r') as f:
            return json.loads(f.read())

    tags_map = {'[PAD]': 0}

    def _i(path):
        with open(path, 'r') as f:
            lines = [line.strip() for line in f]
        i, start, sentences = 0, 0, []
        for line in tqdm(lines, desc='get tags'):
            if not line:
                sentences.append(CoNLLSentence(lines[start:i]))
                start = i + 1
            i += 1

        for s in sentences:
            for tag in s.tags:
                tags_map.setdefault(tag, len(tags_map))
    _i(TRAIN_PATH)
    _i(DEV_PATH)
    with open(DATA_PATH.joinpath('tag_map.json'), 'w') as f:
        f.write(json.dumps(tags_map, ensure_ascii=False, indent=2))
    return tags_map


def get_edu_labels() -> dict:
    lm_path = DIALOGUE_DATA_PATH.joinpath('label_map.json')
    if lm_path.exists():
        with open(lm_path, 'r') as f:
            return json.loads(f.read())

    label_map = {'[PAD]': 0}

    def _i(path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        for dialog in data:
            for src, rel, tgt in dialog['relationship']:
                s_src, _ = map(int, src.split('-'))
                s_tgt, _ = map(int, tgt.split('-'))
                # 跳过跨句关系
                if s_src != s_tgt:
                    continue
                if rel not in label_map:
                    label_map[rel] = len(label_map)

    _i(DIALOGUE_TRAIN_PATH)
    _i(DIALOGUE_TEST_PATH)
    with open(lm_path, 'w') as f:
        print("create label_map.json")
        f.write(json.dumps(label_map, ensure_ascii=False, indent=2))
    return label_map


def get_edu_tags() -> dict:
    if DIALOGUE_DATA_PATH.joinpath('tag_map.json').exists():
        with open(DIALOGUE_DATA_PATH.joinpath('tag_map.json'), 'r') as f:
            return json.loads(f.read())
    else:
        print("can't find tag_map")
        tags = {'[PAD]': 0, 'UNK': 1}           # 只有占位词性
        return tags


def get_dialogue_labels() -> dict:
    lm_path = DIALOGUE_DATA_PATH.joinpath("label_map_dialogue.json")
    if lm_path.exists():
        with open(lm_path, 'r') as f:
            return json.loads(f.read())
    
    label_map = {"[PAD]": 0}

    def _i(path):
        with open(path, 'r', encoding='utf8') as f:
            data = json.load(f)
        for dialog in data:
            for src, rel, tgt in dialog['relationship']:
                s_src, _ = map(int, src.split('-'))
                s_tgt, _ = map(int, tgt.split('-'))
                # 跳过句内关系
                if s_src == s_tgt:
                    continue
                if rel not in label_map:
                    label_map[rel] = len(label_map)

    _i(DIALOGUE_TRAIN_PATH)
    _i(DIALOGUE_TEST_PATH)
    with open(lm_path, 'w') as f:
        print("create label_map_dialogue.json")
        f.write(json.dumps(label_map, ensure_ascii=False, indent=2))
    return label_map


def encoder_texts(texts: List[List[str]], tokenizer):
    # 统计句子中最大的词长度
    fix_len = max([max([len(word) for word in text]) for text in texts])

    matrix = []
    for text in texts:
        vector = []

        # text = [tokenizer.cls_token, *text, tokenizer.sep_token]
        text = [tokenizer.cls_token, *text]
        input_ids = tokenizer.batch_encode_plus(
            text,
            add_special_tokens=False,
        )['input_ids']

        for _input_ids in input_ids:
            # 修复例如: texts = [['\ue5f1\ue5f1\ue5f1\ue5f1']] 这种情况
            _input_ids = _input_ids or [tokenizer.unk_token_id]
            vector.append(_input_ids + (fix_len - len(_input_ids)) * [tokenizer.pad_token_id])
        matrix.append(torch.tensor(vector, dtype=torch.long))
    return pad_sequence(matrix, batch_first=True)


class SDPTransform(dataset.Dataset):
    def __init__(self, path: str, transformer: str, device: torch.device = 'cpu'):
        super(SDPTransform, self).__init__()
        self.device = device
        # self.tokenizer = AutoTokenizer.from_pretrained(str(transformer)) if not isinstance(transformer, AutoTokenizer) else transformer

        self.tokenizer = AutoTokenizer.from_pretrained(transformer) if isinstance(transformer, str) else transformer
        self.labels = get_labels()
        self.tags = get_tags()
        self.sentences = []

        with open(path, 'r') as f:
            lines = [line.strip() for line in f]
        i, start = 0, 0
        for line in tqdm(lines, desc='transform'):
            if not line:
                self.sentences.append(CoNLLSentence(lines[start:i]))
                start = i + 1
            i += 1

        # 统计下
        l = {}
        for sentence in self.sentences:
            ll = len(sentence) // 10
            l.setdefault(ll, 0)
            l[ll] += 1
        print("句子长度情况: ", l)

        # 过滤 排序 只保留长度小于100的句子
        self.sentences = sorted([i for i in self.sentences if len(i) < 100], key=lambda x: len(x))

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return self.sentences[item]

    def collate_fn(self, batch: List[CoNLLSentence]):
        subwords = encoder_texts(texts=[i.words for i in batch], tokenizer=self.tokenizer)
        tags = []
        labels = []
        for _batch in batch:
            tag = [0, ]
            for _tag in _batch.tags:
                tag.append(self.tags[_tag])
            tags.append(tag)

            label = []
            for line in _batch.get_labels():
                label.append([])
                for _label in line:
                    label[-1].append(self.labels[_label] if _label is not None else -1)
            labels.append(label)

        tags_max_len = max([len(i) for i in tags])
        labels_max_len = max([len(i) for i in labels])
        tags_matrix = torch.zeros(len(batch), tags_max_len, dtype=torch.long)
        for index, tag in enumerate(tags):
            tags_matrix[index, :len(tag)] = torch.tensor(tag)
        labels_matrix = torch.zeros(len(batch), labels_max_len, labels_max_len, dtype=torch.long)
        for index, label in enumerate(labels):
            label_len = len(label)
            labels_matrix[index, :label_len, :label_len] = torch.tensor(label)
        return subwords.to(self.device), tags_matrix.to(self.device), labels_matrix.to(self.device)

    def to_dataloader(self, batch_size, shuffle):
        print("数据长度: ", len(self))
        return dataloader.DataLoader(self, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)


class EDUCutTransform(dataset.Dataset):
    def __init__(self, path: str, transformer: str, device: torch.device = 'cpu'):
        super().__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(transformer) \
            if isinstance(transformer, str) else transformer

        # 解析数据json，拆成句子级对象
        self.sentences = []   # 每项：dict{words,tags,edges}
        with open(path, 'r', encoding='utf8') as f:
            dataset_json = json.load(f)

        for dialog in tqdm(dataset_json, desc='parse json'):
            # 1. 先把 turn → token 数组 保存
            turns = []
            for utt in dialog['dialog']:
                words = utt['utterance'].split()
                tags = pos(words)
                turns.append({'words': words, 'tags': tags,
                              'edges': [[] for _ in range(len(words))]})

            # 2. 填充 intra‑sentence 依存边
            for src, rel, tgt in dialog['relationship']:
                sent_t, idx_t = map(int, src.split('-'))
                sent_s, idx_s = map(int, tgt.split('-'))
                if sent_s != sent_t:          # 忽略跨句
                    continue
                dep_idx = idx_s - 1
                head_idx = idx_t - 1
                if head_idx < 0:    # 根节点
                    continue
                if dep_idx >= len(turns[sent_s]['edges']) or head_idx >= len(turns[sent_s]['edges']):
                    print(f'Skip edge {src}->{tgt} in dialog {dialog["id"]}')
                    continue
                turns[sent_s]['edges'][dep_idx].append((head_idx+1, rel))

            # 3. 保存到 self.sentences
            self.sentences.extend(turns)
        
        self.tags = get_edu_tags()
        self.labels = get_edu_labels()
        # 是否更新tag_map
        updated = False
        for sent in self.sentences:
            for tag in sent['tags']:
                if tag not in self.tags:
                    self.tags[tag] = len(self.tags)
                    updated = True
        if updated:
            print("update tag_map...")
            with open(DATA_PATH.joinpath('tag_map.json'), 'w') as f:
                f.write(json.dumps(self.tags, ensure_ascii=False, indent=2))

        # 过滤超长句（与原逻辑一致）
        self.sentences = [s for s in self.sentences if len(s['words']) < 100]
        self.sentences.sort(key=lambda x: len(x['words']))

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    def collate_fn(self, batch):
        """
        将 batch(List[dict]) -> (subwords, tag_ids, label_ids) 三个 GPU 张量
        ---------------------------------------------------------------
        dict 结构:
        'words':  List[str]    已按空格切好的 token
        'tags':   List[str]    HanLP POS，长度 == len(words)
        'edges':  List[List[(head_idx+1, rel_str)]]  0-based dep, 1-based head
        """
        B = len(batch)

        subwords = encoder_texts([s['words'] for s in batch], self.tokenizer)
        L_tok = subwords.size(1)            # 已含 [CLS]

        tag_tensor = torch.zeros(B, L_tok, dtype=torch.long)   # 0 -> [PAD]/[CLS]

        for i, sent in enumerate(batch):
            # 真实 token 的 tag id 映射
            tag_ids = [self.tags[t] for t in sent['tags']]     # len == len(words)
            # 写入（位置 1 开始，因为 0 给 [CLS]）
            tag_tensor[i, 1:1+len(tag_ids)] = torch.tensor(tag_ids, dtype=torch.long)

        label_tensor = torch.full(
            (B, L_tok, L_tok),
            fill_value=-1, dtype=torch.long
        )
        for b_idx, sent in enumerate(batch):
            for dep_idx, arcs in enumerate(sent['edges']):    # dep_idx: 0-based
                row = dep_idx + 1                             # +1 对齐 [CLS]/ROOT
                for head_idx, rel in arcs:                    # head_idx 已 +1
                    # 越界保护——若标注与分词长度不一致则跳过
                    if head_idx < L_tok:
                        label_tensor[b_idx, row, head_idx] = self.labels[rel]

        return (
            subwords.to(self.device),      # (B, L_tok, L_sub)
            tag_tensor.to(self.device),    # (B, L_tok)
            label_tensor.to(self.device)   # (B, L_tok, L_tok)
        )


    def to_dataloader(self, batch_size, shuffle):
        print("数据长度: ", len(self))
        return dataloader.DataLoader(self,
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        collate_fn=self.collate_fn)


class DialogueTransform(dataset.Dataset):
    def __init__(self,
                 path: str,
                 transformer: str,
                 device: torch.device = "cpu"):
        super().__init__()
        self.device = device

        self.tok = AutoTokenizer.from_pretrained(PRETRAIN_MODEL_PATH)
        self.enc = AutoModel.from_pretrained(PRETRAIN_MODEL_PATH).to(device).eval()

        self.labels = get_dialogue_labels()
        self.dialogs: List[Dict] = []
        with open(path, "r", encoding="utf8") as f:
            raw = json.load(f)

        for dia in tqdm(raw, desc="parse json"):
            sents = [utt["utterance"] for utt in dia["dialog"]]

            edges = [[] for _ in range(len(sents))]
            for src, rel, tgt in dia["relationship"]:
                s_src, _ = map(int, src.split('-'))
                s_tgt, _ = map(int, tgt.split('-'))
                if s_src == s_tgt:
                    continue
                if s_src >= len(sents) or s_tgt >= len(sents):
                    continue
                dep, head = s_src, s_tgt
                edges[dep].append((head + 1, rel))

            self.dialogs.append({"sents": sents, "edges": edges})

        self.dialogs.sort(key=lambda d: len(d["sents"]))


    # ---------- Dataset 基本方法 ----------
    def __len__(self):
        return len(self.dialogs)

    def __getitem__(self, idx):
        return self.dialogs[idx]

    # ---------- collate_fn ----------
    def collate_fn(self, batch: List[Dict]):
        flat = sum([d["sents"] for d in batch], [])
        with torch.no_grad():
            tok = self.tok(
                flat, return_tensors="pt",
                padding=True, truncation=True
            ).to(self.device)
            cls_vec = self.enc(**tok).last_hidden_state[:, 0]

        hidden = cls_vec.size(-1)
        vec_chunks, idx = [], 0
        for d in batch:
            n = len(d["sents"])
            vec_chunks.append(cls_vec[idx:idx + n])
            idx += n

        L_max = max(v.size(0) for v in vec_chunks)
        B = len(batch)
        subwords = torch.zeros(B, L_max, hidden)
        for i, v in enumerate(vec_chunks):
            subwords[i, :v.size(0)] = v

        # ---------- 2. dummy tags ----------
        tags = torch.zeros(B, L_max, dtype=torch.long)

        # ---------- 3. 依存矩阵 ----------
        labels = torch.full(
            (B, L_max, L_max),
            fill_value=-1, dtype=torch.long
        )
        for b, d in enumerate(batch):
            for dep, arcs in enumerate(d["edges"]):
                row = dep + 1
                for head, rel in arcs:
                    if head < L_max:
                        labels[b, row, head] = self.labels[rel]

        return (
            subwords.to(self.device),
            tags.to(self.device),
            labels.to(self.device)
        )

    def to_dataloader(self, batch_size, shuffle=True):
        return dataloader.DataLoader(self, batch_size=batch_size,
                                     shuffle=shuffle, collate_fn=self.collate_fn)



if __name__ == '__main__':
    # for subwords, tags, labels in SDPTransform(
    #         path=TRAIN_PATH,
    #         transformer=PRETRAIN_MODEL_PATH
    # ).to_dataloader(batch_size=32, shuffle=False):
    #     assert (subwords.size(1) == tags.size(1) == labels.size(1) == labels.size(2))

    for sentences, labels, _ in DialogueTransform(
        path=DIALOGUE_TRAIN_PATH,
        transformer=PRETRAIN_MODEL_PATH
    ).to_dataloader(batch_size=32, shuffle=False):
        continue
