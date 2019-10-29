import torch


class GeneralConfig:
    printout = True
    sanity = False
    load = False
    pretraining = False
    pretraining_encoder = False
    seed = 1997
    test = True
    cuda = torch.cuda.is_available()
    valid_niter = 1000
    log_every = 100
    mode = "prune"
    model = "transformer"


class TrainConfig:
    lr = 2
    weight_decay = 0.00001
    batch_size = 32
    # teacher_forcing = 1
    clip_grad = 5.0
    lr_decay = 0.2
    max_epoch = 200
    max_epoch_pretraining = 1
    max_epoch_pretraining_encoder = 1
    patience = 3
    max_num_trial = 3
    accumulate = 8


class LSTMConfig:
    num_layers_encoder = 3
    num_layers_decoder = 3
    bidirectional_encoder = True
    residual = False
    hidden_size_encoder = 256
    hidden_size_decoder = 512
    factorization = False
    rank_encoder = 128 if factorization else None
    rank_decoder = 128 if factorization else None
    embed_size = 256
    has_output_layer = True
    dropout_layers = 0.4
    dropout_lstm_states = 0.
    context_size = 256


class DecodeConfig:
    beam_size = 10
    max_decoding_time_step = 150
    greedy_search = False
    replace = False
    batch_size = 24


class VocabConfig:
    freq_cutoff = 2
    vocab_size = 50000
    max_len_corpus = 1000
    joint = False

    subwords_source = False
    subwords_target = False
    load_subwords = False
    subwords_model_type = "bpe"
    subwords_vocab_size = 8000
    subwords_joint_vocab_size = 37000


class CompressionConfig:
    load_from = "regular"
    decode_after = True

    # mixed and quantized mode
    large_dtype = "float32"
    small_dtype = "float16"

    # only in prune mode
    prune_model = True
    retrain_pruned = False
    load_from_pruned_retrained = False
    pruning = 0.27
    pruning_where = "all"
    # possible values : all or (several in tuple form) ['encoder.lookup', 'decoder.lstm.cells.2', 'decoder.att_layer', 'encoder.lstm', 'decoder.output_layer', 'decoder.lstm.cells.0', 'encoder.out_forward', 'decoder.lstm.cells.1', 'decoder.lstm', 'decoder.lookup']
    pruning_scheme = "blind"
    max_epoch_retraining = 2

    # only in quantized mode
    quantize_model = True

    # only in mixed mode
    loss_rescale = 1

    # postfactorized mode
    factorize_model = False
    prune_before = True
    retrain_factorized = False
    prune_after = True


class TransformerConfig:
    embedding_factorization = False
    inner_factorization = False
    ffward_factorization = False
    embedding_rank = 256 if embedding_factorization else None
    inner_rank = 256 if inner_factorization else None
    ffward_rank = 256 if ffward_factorization else None
    num_layers = 3
    layer_dimension = 512
    inner_layer_dimension = 1024
    num_attention_heads = 8
    dropout = 0.1


class CBOWConfig:
    embed_size = 256
    inner_size = 32
    context_size = 5
    hidden_size = 256
    n_epochs = 2
    batch_size = 128
    lr = 0.1
    load = False
    patience = 4
    freq_cutoff = 5
    vocab_size = 80000
