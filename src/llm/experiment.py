# ds: sk-a0e0566b9da948969d3b99fa03424bec
# sil: sk-mtiiummxpaqehihutcmsgwnvxdksrgakhamlfroaivjkugtl

from openai import OpenAI
from tqdm import tqdm
import re, random

DEV_PATH = "/home/u210110513/semantic-dependency-parser/data/origin/ctb70/dev.ctb70.conll"  # 1000条
# DEV_PATH = "/home/u210110513/semantic-dependency-parser/data/origin/ctb70/test.conll"
PRED_PATH  = "/home/u210110513/semantic-dependency-parser/data/llm/ctb70_dev_pred.conll"
METRIC_PATH = "/home/u210110513/semantic-dependency-parser/data/llm/ctb70_dev_metric.txt"
API_KEY = "sk-a0e0566b9da948969d3b99fa03424bec"
BASE_URL = "https://api.deepseek.com"

def read_conll_file(filepath):
    """
    返回:
    一个列表，每个元素对应一个句子；
    每个句子是一个列表，内部每个元素为 (id, form, head, deprel)
    """
    sentences = []
    current_sentence = []

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_sentence:
                    sentences.append(current_sentence)
                    current_sentence = []
                continue
            
            cols = line.split('\t')
            # 检查列数是否足够
            if len(cols) < 8:
                continue
            
            token_id = int(cols[0])
            form = cols[1]
            head = int(cols[6])   # HEAD
            deprel = cols[7]      # DEPREL

            current_sentence.append((token_id, form, head, deprel))
    
    if current_sentence:
        sentences.append(current_sentence)

    return sentences

# 转换为prompt
def build_prompt_from_sentence(sentence_tokens):
    """
    输入：单个句子的列表
    返回：prompt
    """
    # 拼接句子
    sentence_str = " ".join(sentence_tokens)

    prompt = f"""
你是一个专业的中文依存句法分析工具。请根据提供的中文句子，给出每个词的依存关系。

要求如下：
1. 如下是对依存标签集的解释，输出的标签只能严格从中选择：
"NMOD": 名词修饰语（修饰名词的成分，如定语）,
"SBJ": 主语（句子的主语成分）,
"ROOT": 根节点（整个句子的核心谓词）,
"VMOD": 动词修饰语（修饰动词的副词等）,
"LC": 处所成分（表地点的成分，如“在学校”）,
"COOR": 并列结构（连接两个并列结构的成分）,
"DEC": 的字结构（“的”所引导的修饰结构）,
"OBJ": 宾语（动词或介词的宾语）,
"M": 数量修饰语（数词或量词修饰成分）,
"DEG": 程度修饰语（表示程度的成分，如“很”）,
"POBJ": 介词宾语（介词后面的宾语）,
"DEV": 地字结构（“地”所引导的状语结构）,
"AMOD": 形容词修饰语（形容词修饰名词）,
"PMOD": 介词修饰语（修饰介词短语的成分）,
"CS": 连词（连接句子成分的词，如“因为”）,
"PRN": 插入语（补充说明性的语段）,
"VRD": 结果补语（如“打碎”中的“碎”）,
"VC": 系动词补语（如“是”后的成分）

2.输出格式要求如下：
每一行为一个词，包含四列：词编号（从1开始）、词语、该词的中心词编号（HEAD）、依存关系标签（DEPREL）。  
请严格只输出如下一行，不要输出任何其他文字、说明或注释：
ID<TAB>FORM<TAB>HEAD_ID<TAB>DEPREL

3. 若词为根节点，则该词的HEAD_ID为0

4. 下面是一个输入与输出的例子：
示例输入句子：
上海 浦东 开发 与 法制 建设 同步 。

示例输出：
1	上海	2	NMOD
2	浦东	6	NMOD
3	开发	6	NMOD
4	与      6   NMOD
5	法制	6	NMOD
6	建设	7	SBJ
7	同步	0	ROOT
8   。      7   VMOD

请解析下面的句子：
句子: {sentence_str}
"""
    
    return prompt

def call_llm(prompt):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是一个中文自然语言处理专家，擅长对输入的中文句子进行语义依存分析。"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
    except Exception as e:
        print(f"Warning: LLM call encountered an error, skipping this case. Error detail: {e}")
        return ""
    return response.choices[0].message.content

# 解析输出，去重
def parse_llm_output(llm_response):
    """
    将大模型输出的文本解析为 (id, form, head, deprel) 的列表。
    约定大模型输出格式形如：
        1   上海    2   NMOD
        2   浦东    6   NMOD
        ...
        7   同步    0   ROOT
    不同行之间可能有空格或 tab，用正则拆分列。
    """
    predicted_sentence = []
    lines = llm_response.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        cols = re.split(r'\s+', line)
        if len(cols) < 4:
            continue
        try:
            token_id = int(cols[0])
            form = cols[1]
            head = int(cols[2])
            deprel = cols[3]
            predicted_sentence.append((token_id, form, head, deprel))
        except ValueError:
            continue
    return predicted_sentence

# 计算指标
def calc_uas_las(gold_sentences, pred_sentences):
    total_tokens = 0
    correct_uas = 0
    correct_las = 0

    for gold_sent, pred_sent in zip(gold_sentences, pred_sentences):
        for (g_id, g_form, g_head, g_dep), (p_id, p_form, p_head, p_dep) in zip(gold_sent, pred_sent):
            total_tokens += 1

            # UAS: head 预测正确
            if g_head == p_head:
                correct_uas += 1
                # LAS: head 和依存关系都正确
                if g_dep == p_dep:
                    correct_las += 1

    uas = correct_uas / total_tokens if total_tokens else 0
    las = correct_las / total_tokens if total_tokens else 0
    return uas, las

# 结果写入文件
def write_conll_file(filepath, sentences):
    """
    将预测的结果写入到一个 CoNLL-like 文件中。
    sentences: list of sentences, 
      each sentence is [(token_id, form, head, deprel), ...]
    """
    with open(filepath, 'w', encoding='utf-8') as f:
        for sent in sentences:
            for token_id, form, head, deprel in sent:
                f.write(f"{token_id}\t{form}\t{head}\t{deprel}\n")
            # 句子之间空一行
            f.write("\n")

# 主流程
def main():
    bad_case_cnt = 0
    gold_sentences = read_conll_file(DEV_PATH)

    # 逐句从 gold_sentences 中取出 tokens，构造 Prompt，调用大模型
    # 并将大模型输出解析成 pred_sentences
    pred_sentences = []

    # 抽1000条来测试
    gold_sentences = random.sample(gold_sentences, 1000)

    write_conll_file("/home/u210110513/semantic-dependency-parser/data/llm/sample.conll", gold_sentences)
    print(f"Info: 随机抽样 {1000} 条句子，已保存")
    for i, sent in enumerate(tqdm(gold_sentences, desc="Processing Sentences"), start=1):
        forms = [t[1] for t in sent]  # 词文本
        prompt = build_prompt_from_sentence(forms)

        llm_resp = call_llm(prompt)
        pred_sent = parse_llm_output(llm_resp)

        # 如果大模型输出的 token 数量、顺序与 gold 不对应，需要做对齐或检查
        # 若解析结果长度一致，直接用；否则进一步处理
        if len(pred_sent) != len(sent):
            bad_case_cnt += 1
            print("Warning: 解析结果长度与标准不一致.")
            min_len = min(len(pred_sent), len(sent))
            pred_sent = pred_sent[:min_len]

        pred_sentences.append(pred_sent)

        if i % 100 == 0:
            print(f"Info: 已处理 {i} 句，保存中...")
            write_conll_file(PRED_PATH, pred_sentences)
            uas, las = calc_uas_las(gold_sentences, pred_sentences)
            print(f"UAS = {uas*100:.2f}%")
            print(f"LAS = {las*100:.2f}%")
        print(f"bad_case_cnt = {bad_case_cnt}")


    write_conll_file(PRED_PATH, pred_sentences)
    print("Info: 最终结果已保存。")    
    uas, las = calc_uas_las(gold_sentences, pred_sentences)

    print(f"UAS = {uas*100:.2f}%")
    print(f"LAS = {las*100:.2f}%")
    print(f"bad_case_cnt = {bad_case_cnt}")
    with open(METRIC_PATH, 'w', encoding='utf-8') as f:
        f.write(f"UAS = {uas*100:.2f}%\n")
        f.write(f"LAS = {las*100:.2f}%\n")
        f.write(f"bad_case_cnt = {bad_case_cnt}\n")

if __name__ == "__main__":
    main()
