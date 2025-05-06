import torch, pathlib
from semantic_dependency_parser import SemanticDependencyParser
from config import PRETRAIN_MODEL_PATH, DIALOGUE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

eva_model_path = str(pathlib.Path('.').parent.joinpath('savepoint').joinpath('dialogue').joinpath('savepoint_52').joinpath('dev_metric_6.8199e-01.pt'))

parser = SemanticDependencyParser(
    enable_tag=False,
    config=DIALOGUE)
parser.load(
    pretrained_model_name=PRETRAIN_MODEL_PATH,
    model_path=eva_model_path,
    device=device.type
)

model = parser.model
tokenizer = parser.tokenizer
id2rel  = {v: k for k, v in parser.labels.items()}

dialog = [
    "我 买 的 东西 ， 可以 开 发票 吗 ?",
    "您 好 ， 工号 [ 数字 ] ， 花花 [ 姓名 ] 妹子 ， 很 高兴 为 您 服务 ~",
    "我 想 开 发票 ， 可以 开 办公用品 吗 ?",
    "亲爱 的 客户 ， APP 端 麻烦 您 点击 对话框 右下角 的 “ + ” ， 点击 “ 订单 ” 后 ， 选择 一下 您 需要 咨询 的 订单 哦 ， PC 端 在 我 的 订单 复制 下 哦 ， 小妹 这边 帮 您 查询 一下 哦 ~",
    "您 是 否 下 单 ?",
    "没有 呢",
    "商品编号 : 3578313 是 这个 商品编号 吗 ?",
    "内容 只 能 开具 明细 或者 大类 哦",
    "换 气扇",
    "发票 内容 将 显示 本单 商品 所属 类别 ( 电器 电子产品 及 配件 ) 及 价格 信息",
    "这个 是 大类 的 哦",
    "开具 不 了 办公用品 哦",
    "很 高兴 遇到 您 这么 善解人意 的 客户 ， 请问 还 有 其他 还 可以 帮到 您 的 吗 ?"
]

with torch.no_grad():
    tokens = tokenizer(dialog, return_tensors='pt',
                       padding=True, truncation=True).to(device)
    sent_vec = model.encoder.transformer(**tokens).last_hidden_state[:, 0]

sent_vec = torch.cat([sent_vec.new_zeros(1, sent_vec.size(-1)), sent_vec], dim=0)
sent_vec = sent_vec.unsqueeze(0)

tags_dummy = torch.zeros(1, sent_vec.size(1), dtype=torch.long, device=device)

model.eval()
with torch.no_grad():
    s_edge, s_label = model(sent_vec, tags_dummy)
    pred = model.mst_decode(s_edge, s_label)[0]

print("="*60)
print(f"{'ID':>2} {'SENTENCE':<30} REL         HEAD")
print("-"*60)

L = len(dialog)
for dep in range(1, L+1):
    heads = (pred[dep] >= 0).nonzero(as_tuple=True)[0]
    if heads.numel() == 0:
        print(f"{dep:>2} {dialog[dep-1]:<30} --none--    ROOT")
        continue
    if heads.numel() > 1:
        print(f"[Warn] node {dep} 多条弧，仅取首条")
    head = heads[0].item()
    rel  = id2rel[pred[dep, head].item()]
    head_str = "ROOT" if head == 0 else f"{head:>2} {dialog[head-1][:20]}"
    print(f"{dep:>2} {dialog[dep-1][:28]:<30} {rel:<10} {head_str}")
