class EvaluationError(Exception):
    """检索评测基础异常。"""


class EvaluationDataError(EvaluationError):
    """评测数据缺失或者格式不正确。"""
