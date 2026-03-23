# Models module — 4 active architectures + 1 archived (large_gru)
from regime_forecasting.models.dlinear_model import DLinearForecaster as DLinearForecaster
from regime_forecasting.models.itransformer_model import iTransformerForecaster as iTransformerForecaster
from regime_forecasting.models.large_gru_model import LargeGRUForecaster as LargeGRUForecaster
from regime_forecasting.models.patchtst_model import PatchTSTForecaster as PatchTSTForecaster
from regime_forecasting.models.transformer import TimeSeriesTransformer as TimeSeriesTransformer
