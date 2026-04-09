from models.model import FinancialModel
from config.settings import MODEL_CONFIGS, DECISION_THRESHOLDS


class CreditScoringModel(FinancialModel):
    def __init__(self, config=None):
        super().__init__(config or MODEL_CONFIGS["credit"], DECISION_THRESHOLDS["credit"])
