import torch, json
from semantic_dependency_parser import SemanticDependencyParser
from transform import encoder_texts
import pathlib
import hanlp
# hanlp.pretrained.mtl.ALL
hanlp.pretrained.pos.ALL

from config import PRETRAIN_MODEL_PATH, DATA_PATH, DIALOGUE_DATA_PATH, DIALOGUE_TEST_PATH, DIALOGUE_TRAIN_PATH, EDU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# edu切分模块（mst_decode）
# edu_eva_model_path = str(pathlib.Path('.').parent.joinpath('savepoint').joinpath('edu').joinpath('savepoint_430').joinpath('dev_metric_9.1639e-01.pt'))
edu_eva_model_path = str(pathlib.Path('.').parent.joinpath('savepoint').joinpath('edu').joinpath('savepoint_506').joinpath('dev_metric_9.1510e-01.pt'))
label_map = json.load(open(DIALOGUE_DATA_PATH/'label_map.json'))
tag_map = json.load(open(DIALOGUE_DATA_PATH/'tag_map.json'))

parser = SemanticDependencyParser(
    enable_tag=True,
    config=EDU
)
parser.load(
    pretrained_model_name=PRETRAIN_MODEL_PATH,
    model_path=edu_eva_model_path,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

sent = "在今年的防汛中这项工程显示出了巨大的威力"
sent = "您 好 ， 请问 有 什么 可以 帮助 您 的 么"
sent = "您 是 需要 到 站点 取 件 吗"
sent = "有 什么 问题 我 可以 帮 您 处理 或 解决 呢 ?"

# pos = hanlp.load(hanlp.pretrained.mtl.CLOSE_TOK_POS_NER_SRL_DEP_SDP_CON_ELECTRA_BASE_ZH)
# pos_res = pos([sent], task='pos').to_dict()
# words = pos_res['tok/fine'][0]
# tags = pos_res['pos/ctb'][0]

pos = hanlp.load(hanlp.pretrained.pos.CTB9_POS_ELECTRA_SMALL)
words = sent.split()
tags = pos(words)
print(words, tags)

model = parser.model
tokenizer = parser.tokenizer
label_map = parser.labels
id2label = {v:k for k,v in label_map.items()}
tag_map = parser.tags
assert len(words) == len(tags)
tag_ids = [tag_map.get(t) for t in tags]

subwords = encoder_texts([words], tokenizer).to(device)
tag_ids  = torch.tensor([[0] + tag_ids], dtype=torch.long, device=device)

model.eval()
with torch.no_grad():
    s_edge, s_label = model(subwords, tag_ids)
    pred = model.decode(s_edge, s_label)[0]

print(pred)
print("="*48)
print(f"{'ID':>2} {'WORD':<8} REL        HEAD")
print("-"*48)

for dep in range(1, pred.size(0)):
    rel_row = pred[dep]
    heads = (rel_row >= 0).nonzero(as_tuple=True)[0]

    if heads.numel() == 0:
        print(f"{dep:>2} {words[dep-1]:<6} --none--   ROOT")
        continue
    head = heads[0].item()

    rel  = id2label[rel_row[head].item()]
    head_str = "ROOT" if head == 0 else f"{head:>2} {words[head-1]}"
    print(f"{dep:>2} {words[dep-1]:<6} {rel:<10} {head_str}")
