from models.model import FinancialModel
from config.settings import MODEL_CONFIGS, DECISION_THRESHOLDS


class TradingSignalModel(FinancialModel):
    def __init__(self, config=None):
        super().__init__(config or MODEL_CONFIGS["trading"], DECISION_THRESHOLDS["trading"])
