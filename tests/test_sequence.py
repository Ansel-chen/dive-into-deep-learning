import torch

from sequence.gru import GRUEncoder
from sequence.language_model import (
    RNNLanguageModel,
    language_model_loss,
)
from sequence.preprocessing import (
    Vocabulary,
    batchify,
    normalize_text,
)
from sequence.rnn import BidirectionalRNNEncoder, RNNEncoder
from sequence.seq2seq import Decoder, Encoder, Seq2Seq


def test_text_normalization_vocabulary_and_batchify():
    assert normalize_text("Hello,  World!") == "hello world"
    vocab = Vocabulary(["<pad>", "hello", "world", "hello"])
    ids = vocab.to_indices(["hello", "unknown"])
    assert ids[0] == vocab["hello"]
    assert ids[1] == vocab.unk_index
    assert vocab.to_tokens(ids) == ["hello", "<unk>"]

    inputs, targets = batchify(torch.arange(24), batch_size=3, num_steps=4)
    assert inputs.shape == (3, 4)
    assert targets.shape == (3, 4)
    assert torch.equal(targets[:, :-1], inputs[:, 1:])


def test_rnn_and_gru_state_contracts():
    tokens = torch.randint(0, 12, (4, 5))
    rnn = RNNEncoder(vocab_size=12, embed_size=6, hidden_size=7)
    outputs, state = rnn(tokens)
    assert outputs.shape == (4, 5, 7)
    assert state.shape == (1, 4, 7)

    gru = GRUEncoder(
        vocab_size=12,
        embed_size=6,
        hidden_size=7,
        bidirectional=True,
    )
    outputs, state = gru(tokens)
    assert outputs.shape == (4, 5, 14)
    assert state.shape == (2, 4, 7)


def test_bidirectional_rnn_exposes_two_directions():
    tokens = torch.randint(0, 12, (3, 5))
    model = BidirectionalRNNEncoder(
        vocab_size=12,
        embed_size=6,
        hidden_size=7,
    )
    outputs, state = model(tokens)
    assert outputs.shape == (3, 5, 14)
    assert state.shape == (2, 3, 7)


def test_language_model_runs_one_gradient_step():
    tokens = torch.randint(0, 10, (3, 6))
    targets = torch.randint(0, 10, (3, 6))
    model = RNNLanguageModel(
        vocab_size=10,
        embed_size=5,
        hidden_size=8,
    )
    logits, state = model(tokens)
    assert logits.shape == (3, 6, 10)
    assert state.shape == (1, 3, 8)

    loss = language_model_loss(model, tokens, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_seq2seq_encoder_decoder_batch_forward():
    source = torch.randint(0, 15, (2, 5))
    target = torch.randint(0, 17, (2, 6))
    model = Seq2Seq(
        Encoder(vocab_size=15, embed_size=6, hidden_size=8),
        Decoder(vocab_size=17, embed_size=6, hidden_size=8),
    )
    logits = model(source, target)
    assert logits.shape == (2, 6, 17)
    assert torch.isfinite(logits).all()
