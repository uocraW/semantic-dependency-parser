# -*- coding: utf8 -*-
#

class Metric(object):

    def __lt__(self, other):
        return self.score < other

    def __le__(self, other):
        return self.score <= other

    def __ge__(self, other):
        return self.score >= other

    def __gt__(self, other):
        return self.score > other

    @property
    def score(self):
        return 0.

class ChartMetric(Metric):

    def __init__(self, eps=1e-12):
        super(ChartMetric, self).__init__()

        self.tp = 0.0
        self.utp = 0.0
        self.pred = 0.0
        self.gold = 0.0
        self.eps = eps

    def __call__(self, preds, golds):
        pred_mask = preds.ge(0)
        gold_mask = golds.ge(0)
        span_mask = pred_mask & gold_mask
        self.pred += pred_mask.sum().item()
        self.gold += gold_mask.sum().item()
        self.tp += (preds.eq(golds) & span_mask).sum().item() # 真正预测对的
        self.utp += span_mask.sum().item()
        return self

    def __repr__(self):
        return f"UP: {self.up:6.2%} UR: {self.ur:6.2%} UF: {self.uf:6.2%} P: {self.p:6.2%} R: {self.r:6.2%} F: {self.f:6.2%}"

    @property
    def score(self):
        return self.f

    @property
    def up(self):
        return self.utp / (self.pred + self.eps)

    @property
    def ur(self):
        return self.utp / (self.gold + self.eps)

    @property
    def uf(self):
        return 2 * self.utp / (self.pred + self.gold + self.eps)

    @property
    def p(self):
        return self.tp / (self.pred + self.eps)

    @property
    def r(self):
        return self.tp / (self.gold + self.eps)

    @property
    def f(self):
        return 2 * self.tp / (self.pred + self.gold + self.eps)

class EvaluateMetric(Metric):
    def __init__(self, eps=1e-12):
        super(EvaluateMetric, self).__init__()

        # 初始化计数器
        self.tp_uas = 0.0  # 正确的依存关系（UAS）
        self.tp_las = 0.0  # 正确的依存关系和标签（LAS）
        self.pred = 0.0    # 预测的依存关系数
        self.gold = 0.0    # 真实的依存关系数
        self.eps = eps

    def __call__(self, preds, golds):
        pred_mask = preds.ge(0)  # 预测出的依存关系
        gold_mask = golds.ge(0)  # 真实依存关系
        span_mask = pred_mask & gold_mask  # 计算 UAS 和 LAS 只考虑真实存在的依存关系
        
        self.pred += pred_mask.sum().item()
        self.gold += gold_mask.sum().item()

        self.tp_uas += span_mask.sum().item()
        self.tp_las += ((preds.eq(golds)) & span_mask).sum().item()
        return self

    def __repr__(self):
        return f"UAS: {self.uas:6.2%} LAS: {self.las:6.2%}"
    
    @property
    def uas(self):
        return self.tp_uas / (self.gold + self.eps)

    @property
    def las(self):
        return self.tp_las / (self.gold + self.eps)
