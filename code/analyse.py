import configuration
from nmt.nmtmodel import NMTModel
import paths
import time
import datetime

import torch
import torch.cuda as cuda
from utils import read_corpus, zip_data


def check_memory(model):
    MAX_BATCH_SIZE = 1024
    model.to_gpu()
    logfile = paths.memory_log
    train_data_src = read_corpus(paths.train_source, source='src')
    train_data_tgt = read_corpus(paths.train_target, source='tgt')
    train_data = zip_data(train_data_src, train_data_tgt)

    now = datetime.datetime.now()
    with open(logfile, 'a') as f:
        f.write("\n==================\n")
        f.write(str(now))
        f.write("model class "+model.__class__.__name__)
    for b in range(1, MAX_BATCH_SIZE+1):
        examples = train_data[:b]
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        src_sents = [e[0] for e in examples]
        tgt_sents = [e[1] for e in examples]
        mem_before_no_retain = int(cuda.memory_allocated()//1e6)
        begin = time.time()
        try:
            loss = model(src_sents, tgt_sents, update_params=False, return_attached_loss=True)
            mem_forward = int(cuda.memory_allocated()//1e6)
            forward = round(time.time() - begin, 2)
            loss.backward(retain_graph=True)
            mem_backward_retain = int(cuda.memory_allocated()//1e6)
            backward_retain = round(time.time()-(begin+forward), 2)
            model.zero_grad()
            loss.backward(retain_graph=False)
            mem_backward_no_retain = int(cuda.memory_allocated()//1e6)
            backward_no_retain = round(time.time()-(begin+forward+backward_retain), 2)
            model.zero_grad()
            with open(logfile, 'a') as f:
                print("\nWith batch size", b, ":")
                print("Initial memory", mem_before_no_retain, "forward ("+str(forward)+"s)", mem_forward, "backward retain (" +
                      str(backward_no_retain)+"s)", mem_backward_no_retain, "backward free ("+str(backward_no_retain)+"s)", mem_backward_retain)
                print("\nWith batch size", b, ":", file=f)
                print("Initial memory", mem_before_no_retain, "forward ("+str(forward)+"s)", mem_forward, "backward retain (" +
                      str(backward_no_retain)+"s)", mem_backward_no_retain, "backward free ("+str(backward_no_retain)+"s)", mem_backward_retain, file=f)
            if b == MAX_BATCH_SIZE:
                with open(logfile, 'a') as f:
                    print("MAX_BATCH_SIZE", MAX_BATCH_SIZE, "reached !")
                    print("MAX_BATCH_SIZE", MAX_BATCH_SIZE, "reached !", file=f)
        except:
            with open(logfile, 'a') as f:
                print("\nError caught at batch", b)
                print("\nError caught at batch", b, file=f)
            break
