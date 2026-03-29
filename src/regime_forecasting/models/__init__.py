# Models module — 4 active architectures + 1 archived (large_gru)
from regime_forecasting.models.transformer import TimeSeriesTransformer
from regime_forecasting.models.large_gru_model import LargeGRUForecaster  # archived from study, kept for reference
from regime_forecasting.models.dlinear_model import DLinearForecaster
from regime_forecasting.models.itransformer_model import iTransformerForecaster
from regime_forecasting.models.patchtst_model import PatchTSTForecaster