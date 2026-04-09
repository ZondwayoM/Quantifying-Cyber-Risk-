from models.model import FinancialModel
from config.settings import MODEL_CONFIGS, DECISION_THRESHOLDS


class AMLDetectionModel(FinancialModel):
    def __init__(self, config=None):
        super().__init__(config or MODEL_CONFIGS["aml"], DECISION_THRESHOLDS["aml"])
