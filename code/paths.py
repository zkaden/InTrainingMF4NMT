import configuration
import os
cwd = os.getcwd()
if cwd.endswith("/code"):
    MAIN_PREFIX = "../"
else:
    MAIN_PREFIX = ""
gconfig = configuration.GeneralConfig()
vconfig = configuration.VocabConfig()
cconfig = configuration.CompressionConfig()
mconfig = configuration.LSTMConfig()
results_folder = "results/"


model = results_folder + "model"
if mconfig.factorization:
    factorized_suffix = ".factorized.{}.{}{}".format(
        mconfig.embed_size, mconfig.rank_encoder, mconfig.rank_decoder)
    model = model + factorized_suffix
model_compressed_prefix = model + ".compressed"
pruning_where_suffix = cconfig.pruning_where if isinstance(
    cconfig.pruning_where, str) else "_".join(list(cconfig.pruning_where))
model_prune_suffix = ".prune" + \
    str(int(100*cconfig.pruning))+cconfig.pruning_scheme+"_" + pruning_where_suffix
model_postfactorize_suffix = "postfact0.5"

model_pruned_suffix = model_prune_suffix + ".pruned"
model_pruned_retrained_suffix = model_prune_suffix + ".retrained"
model_pruned = model_compressed_prefix + model_pruned_suffix
model_pruned_retrained = model_compressed_prefix + model_pruned_retrained_suffix

model_postfactorized_suffix = model_postfactorize_suffix + ".postfactorized"
model_postfactorized_retrained_suffix = model_postfactorize_suffix + ".retrained"
model_postfactorized = model_compressed_prefix + model_postfactorized_suffix
model_postfactorized_retrained = model_compressed_prefix + model_postfactorized_retrained_suffix

model_quantized_suffix = ".quantized."+cconfig.small_dtype
model_quantized = model_compressed_prefix + model_quantized_suffix

model_mixed = model_quantized + ".mixed_precision"

model_embedding = model+".embedding"

vocab_folder = MAIN_PREFIX+"data/vocab/"
vocab = vocab_folder+"vocab" + (".subsrc" if vconfig.subwords_source else ".words") + \
    (".subtgt" if vconfig.subwords_target else ".words") + ".bin"
vocab_bnc = vocab_folder + "vocab.bnc.bin"

decode_output_suffix = ".test.txt" if gconfig.test else ".valid.txt"
decode_output = results_folder+"decode.en"+decode_output_suffix

data_aligned_folder = MAIN_PREFIX+"data/bilingual/"
data_subwords_folder = MAIN_PREFIX+"data/subwords/"

memory_log = results_folder+"analysis_memory.log"


def get_data_path(set, mode):
    prefix = data_aligned_folder
    if mode == "tgt":
        suffix = ".de"
    else:
        assert mode == "src"
        suffix = ".en"
    return prefix+set + ".de-en"+suffix


train_source = get_data_path("train", "src")
train_target = get_data_path("train", "tgt")
dev_source = get_data_path("valid", "src")
dev_target = get_data_path("valid", "tgt")
test_source = get_data_path("test", "src")
test_target = get_data_path("test", "tgt")
newstest_target = "newstest2014-deen-ref.de.sgm"
newstest_source = "newstest2014-deen-src.en.sgm"

embed_folder = "data/embeddings/"
simverb_prefix = embed_folder+"SimVerb/"
simverb_dev = simverb_prefix+"SimVerb-500-dev.txt"
simverb_test = simverb_prefix+"SimVerb-3000-test.txt"

pretrainedw2v = embed_folder+"pretrainedw2v/model.txt"

bnc_folder = embed_folder+"BNC/download/Texts/"
bnc_extracted = embed_folder+"BNC/extracted_dataset.npy"
