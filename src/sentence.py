import torch, json
from semantic_dependency_parser import SemanticDependencyParser
from transform import encoder_texts
import pathlib
import hanlp
# hanlp.pretrained.mtl.ALL
hanlp.pretrained.pos.ALL

from config import PRETRAIN_MODEL_PATH, DATA_PATH, DIALOGUE_DATA_PATH, DIALOGUE_TEST_PATH, DIALOGUE_TRAIN_PATH, EDU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sent_eva_model_path = str(pathlib.Path('.').parent.joinpath('savepoint').joinpath('savepoint_412_bilstm').joinpath('dev_metric_8.8862e-01.pt'))

# 句法分析切分模块(decode)
label_map = json.load(open(DATA_PATH/'label_map.json'))
tag_map = json.load(open(DATA_PATH/'tag_map.json'))

parser = SemanticDependencyParser()
parser.load(
    pretrained_model_name=PRETRAIN_MODEL_PATH,
    model_path=sent_eva_model_path,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)


model = parser.model
tokenizer = parser.tokenizer
label_map = parser.labels
id2label = {v:k for k,v in label_map.items()}
tag_map = parser.tags

sent = "中国 鼓励 民营 企业家 投资 国家 基础 建设"
sent = "上海 浦东 开发 与 法制 建设 同步"
words = sent.split()
tags = "NR VV JJ NN VV NN NN NN".split()
tags = "NR NR NN CC NN NN VV".split()
assert len(words) == len(tags)
tag_ids = [tag_map.get(t) for t in tags]
tag_ids  = torch.tensor([[0] + tag_ids], dtype=torch.long, device=device)

subwords = encoder_texts([words], tokenizer).to(device)


model.eval()
with torch.no_grad():
    s_edge, s_label = model(subwords, tag_ids)
    # pred = model.decode(s_edge, s_label)[0]
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


