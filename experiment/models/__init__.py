from .dummy_model import DummyModel
from .gcn_lstm import GCNLSTM
from .stgcn import STGCN
from .stsgcn import STSGCN
from .stid import STID
from .stgat import STGAT
from .graphsage_lstm import GraphSAGELSTM


MODEL_REGISTRY = {
    "dummy": DummyModel,
    "gcn_lstm": GCNLSTM,
    "stgcn": STGCN,
    "stsgcn": STSGCN,
    "stid": STID,
    "stgat": STGAT,
    "graphsage_lstm": GraphSAGELSTM,
}
def build_model(model_name, **kwargs):

    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available models: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_name](**kwargs)