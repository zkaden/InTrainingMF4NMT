import configuration
import paths
from vocab import Vocab, VocabEntry
cconfig = configuration.CBOWConfig()
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

"""
class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embed_size, context_size, hidden_size, inner_size=None):
        super(CBOWModel, self).__init__()
        self.pivot_size = 2*context_size*embed_size
        if inner_size is None or inner_size >= embed_size:
            self.embeddings = nn.Embedding(vocab_size, embed_size)
        else:
            self.embeddings = nn.Sequential(nn.Embedding(
                vocab_size, inner_size), nn.Linear(inner_size, embed_size))
        self.dr = nn.Dropout(0.3)
        self.linear1 = nn.Linear(self.pivot_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size)

    def forward(self, inputs):
        # print(inputs.size())
        embedded = self.embeddings(inputs).view((inputs.size(0), -1))
        # print(embedded.size())
        hid = F.relu(self.linear1(embedded))
        out = self.linear2(hid)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def embed(self, inputs):
        # return self.embeddings(inputs)
        return torch.stack([self.linear2.weight[i] for i in inputs])

"""


class CBOWModel(nn.Module):
    def __init__(self, vocab_size, embed_size, context_size, hidden_size, inner_size=None):
        super(CBOWModel, self).__init__()
        if inner_size is None or inner_size >= embed_size:
            self.embeddings = nn.Embedding(vocab_size, embed_size)
        else:
            self.embeddings = nn.Sequential(nn.Embedding(
                vocab_size, inner_size), nn.Linear(inner_size, embed_size))
        self.dr = nn.Dropout(0.2)
        self.linear1 = nn.Linear(embed_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, inputs):
        # print(inputs.size())
        embedded = self.embeddings(inputs).sum(dim=1)
        # print(embedded.size())
        hid = F.relu(self.linear1(embedded))
        out = self.linear2(hid)
        log_probs = F.log_softmax(out, dim=1)
        return log_probs

    def embed(self, inputs, end_matrix=False):
        # return self.embeddings(inputs)
        if end_matrix:
            return torch.stack([self.linear2.weight[i] for i in inputs])
        else:
            return self.embeddings(inputs)


def train_cbow(model, train_data, dev_data):
    print_loss_every = 1000
    loss_fn = nn.NLLLoss()
    # print(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=cconfig.lr, weight_decay=1e-4)
    #optimizer = torch.optim.Adam(model.parameters(), lr=cconfig.lr, weight_decay=1e-4)

    model = model.cuda()
    patience = 0
    a = np.arange(len(train_data))
    np.random.shuffle(a)
    train_data = [train_data[i] for i in a]
    best_ppl = float("inf")
    total_loss = .0
    model.train()
    for i in range(0, len(train_data), cconfig.batch_size):
        batch = train_data[i:i+cconfig.batch_size]
        contexts, targets = zip(*batch)

        ctx_var = torch.LongTensor(contexts).cuda()
        optimizer.zero_grad()
        log_probs = model(ctx_var)

        loss = loss_fn(log_probs, torch.LongTensor(targets).cuda())

        loss.backward()
        optimizer.step()

        total_loss += loss.data.item()
        if (i//cconfig.batch_size+1) % print_loss_every == 0:
            print("Iter", i//cconfig.batch_size+1, "train ppl :",
                  round(np.exp(total_loss / print_loss_every), 1))
            total_loss = .0
            model.eval()

            for i in range(0, len(dev_data), cconfig.batch_size):

                batch = dev_data[i:i+cconfig.batch_size]
                contexts, targets = zip(*batch)

                ctx_var = torch.LongTensor(contexts).cuda()
                log_probs = model(ctx_var)

                loss = loss_fn(log_probs, torch.LongTensor(targets).cuda())
                total_loss += loss.data.item()
            ppl = np.exp(total_loss/len(dev_data)*cconfig.batch_size)
            print("Dev ppl :", round(ppl, 1))
            if ppl < best_ppl:
                best_ppl = ppl
                best_model = copy.deepcopy(model.state_dict())
                patience = 0
            else:
                if patience > cconfig.patience:
                    model.load_state_dict(best_model)
                    print("Early stopping !")
                    # break
                else:
                    patience += 1
            model.train()

            total_loss = .0

    model = model.cpu()
    return model


def preprocess_dataset(data):
    ctx_data = []
    cs = cconfig.context_size
    for s in data:
        for i in range(cs, len(s)-cs):
            #ctx = [vocab[w] for w in s[i-cs:i]]+[vocab[w] for w in s[i+1:i+cs+1]]
            #tgt = vocab[s[i]]
            ctx = np.concatenate([s[i-cs:i], s[i+1:i+cs+1]])
            tgt = s[i]
            if tgt > 3:
                ctx_data.append((ctx, tgt))
    print(len(ctx_data))
    # for a, b in ctx_data[:25]:
    #    print(a)
    #    print(b)
    return ctx_data


def get_model_outputs(model, data, end_matrix=False):
    res = []
    miss = 0
    cos = torch.nn.CosineSimilarity(dim=0)
    for u, v in data:
        if u != 3 and v != 3:
            x = torch.LongTensor([u, v])
            y = model.embed(x, end_matrix=end_matrix)
            res.append(cos(y[0], y[1]).data.item())
        else:
            miss += 1
            res.append(None)
    print("Missing pairs :", miss, "/", len(res))

    return res
