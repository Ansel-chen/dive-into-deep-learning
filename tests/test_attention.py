import torch

from attention.bert import (
    BertEmbeddings,
    TinyBertForPretraining,
    TinyBertModel,
    bert_pretraining_loss,
)
from attention.kernel_regression import nadaraya_watson
from attention.transformer import (
    MultiHeadAttention,
    CausalTransformerLM,
    TransformerEncoder,
    causal_mask,
    scaled_dot_product_attention,
)


def test_kernel_regression_returns_normalized_weights():
    queries = torch.tensor([0.0, 1.0])
    keys = torch.tensor([-1.0, 0.0, 1.0])
    values = torch.tensor([1.0, 2.0, 4.0])
    predictions, weights = nadaraya_watson(
        queries,
        keys,
        values,
        bandwidth=0.5,
        return_weights=True,
    )
    assert predictions.shape == (2,)
    assert weights.shape == (2, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2))
    assert torch.isfinite(predictions).all()


def test_scaled_dot_product_attention_respects_boolean_mask():
    query = torch.randn(2, 3, 4)
    key = torch.randn(2, 5, 4)
    value = torch.randn(2, 5, 6)
    mask = torch.ones(2, 3, 5, dtype=torch.bool)
    mask[:, :, -1] = False
    output, weights = scaled_dot_product_attention(query, key, value, mask=mask)
    assert output.shape == (2, 3, 6)
    assert weights.shape == (2, 3, 5)
    assert torch.allclose(weights[:, :, -1], torch.zeros(2, 3), atol=1e-6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3))


def test_multi_head_attention_and_encoder_shapes():
    inputs = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, 5, dtype=torch.bool)
    mask[:, :, -1] = False
    attention = MultiHeadAttention(embed_size=8, num_heads=2)
    output, weights = attention(inputs, inputs, inputs, mask=mask, return_weights=True)
    assert output.shape == inputs.shape
    assert weights.shape == (2, 2, 5, 5)
    assert torch.allclose(weights[:, :, :, -1], torch.zeros(2, 2, 5), atol=1e-6)

    encoder = TransformerEncoder(
        embed_size=8,
        num_heads=2,
        ffn_hidden_size=16,
        num_layers=2,
    )
    encoded = encoder(inputs, mask=mask)
    assert encoded.shape == inputs.shape
    assert torch.isfinite(encoded).all()


def test_causal_transformer_returns_next_token_logits():
    tokens = torch.randint(0, 20, (2, 6))
    mask = causal_mask(tokens.shape[1])
    assert mask.dtype == torch.bool
    assert torch.equal(mask, torch.tril(torch.ones(6, 6, dtype=torch.bool)))

    model = CausalTransformerLM(
        vocab_size=20,
        embed_size=8,
        num_heads=2,
        ffn_hidden_size=16,
        num_layers=1,
        max_length=8,
    )
    logits = model(tokens)
    assert logits.shape == (2, 6, 20)
    assert torch.isfinite(logits).all()


def test_tiny_bert_embedding_and_encoder_contract():
    input_ids = torch.randint(0, 20, (2, 6))
    token_types = torch.zeros_like(input_ids)
    embeddings = BertEmbeddings(
        vocab_size=20,
        hidden_size=8,
        max_position_embeddings=8,
    )
    assert embeddings(input_ids, token_types).shape == (2, 6, 8)

    model = TinyBertModel(
        vocab_size=20,
        hidden_size=8,
        num_heads=2,
        num_layers=1,
        max_position_embeddings=8,
    )
    output = model(input_ids, token_types)
    assert output.shape == (2, 6, 8)


def test_tiny_bert_pretraining_heads_support_masked_loss():
    input_ids = torch.randint(0, 20, (2, 6))
    token_types = torch.zeros_like(input_ids)
    pred_positions = torch.tensor([[1, 4], [0, 3]])
    mlm_labels = torch.randint(0, 20, (2, 2))
    nsp_labels = torch.tensor([0, 1])
    model = TinyBertForPretraining(
        vocab_size=20,
        hidden_size=8,
        num_heads=2,
        num_layers=1,
        max_position_embeddings=8,
    )
    _, mlm_logits, nsp_logits = model(
        input_ids,
        token_types,
        pred_positions=pred_positions,
    )
    assert mlm_logits.shape == (2, 2, 20)
    assert nsp_logits.shape == (2, 2)

    loss = bert_pretraining_loss(
        mlm_logits,
        mlm_labels,
        nsp_logits,
        nsp_labels,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())
