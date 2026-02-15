from neuralflex.models.model import NeuralFlexMoEModel
from neuralflex.config.schemas import ModelConfig


def test_model_constructs() -> None:
    model = NeuralFlexMoEModel(ModelConfig())
    assert model.embed_tokens.num_embeddings == 64000
