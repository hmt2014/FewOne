import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import pickle
from sklearn import metrics
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel, DistilBertConfig
import logging
import copy
import math
import torch.nn.init as init


class Embeddings(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.dim = config.dim
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_ids, params):
        """
            Parameters:
                input_ids: torch.tensor(bs, max_seq_length) The token ids to embed.

            Returns: torch.tensor(bs, max_seq_length, dim) The embedded tokens (plus position embeddings, no token_type
            embeddings)
            """
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)  # (max_seq_length)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)  # (bs, max_seq_length)

        word_embeddings = F.embedding(input_ids, params[
            'meta_learner.bert.embeddings.word_embeddings.weight'])  # (bs, max_seq_length, dim)
        position_embeddings = F.embedding(position_ids, params[
            'meta_learner.bert.embeddings.position_embeddings.weight'])  # (bs, max_seq_length, dim)

        embeddings = word_embeddings + position_embeddings  # (bs, max_seq_length, dim)
        embeddings = F.layer_norm(embeddings, (self.dim,),
                                  weight=params['meta_learner.bert.embeddings.LayerNorm.weight'],
                                  bias=params['meta_learner.bert.embeddings.LayerNorm.bias'],
                                  eps=1e-12)  # (bs, max_seq_length, dim)
        embeddings = self.dropout(embeddings)  # (bs, max_seq_length, dim)
        return embeddings


class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_layers = config.n_layers
        self.config = config

    def attention(self, query, key, value, mask, params, i):
        dim_per_head = self.config.dim // self.config.n_heads
        bs, q_length, dim = query.size()
        k_length = key.size(1)

        def shape(x):
            """ separate heads """
            return x.view(bs, -1, self.config.n_heads, dim_per_head).transpose(1, 2)

        def unshape(x):
            """ group heads """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.config.n_heads * dim_per_head)

        mask_reshp = (bs, 1, 1, k_length)

        q = shape(F.linear(query, params['meta_learner.bert.transformer.layer.' + i + ".attention.q_lin.weight"],
                           params['meta_learner.bert.transformer.layer.' + i + ".attention.q_lin.bias"]))
        k = shape(F.linear(key, params['meta_learner.bert.transformer.layer.' + i + ".attention.k_lin.weight"],
                           params['meta_learner.bert.transformer.layer.' + i + ".attention.k_lin.bias"]))
        v = shape(F.linear(value, params['meta_learner.bert.transformer.layer.' + i + ".attention.v_lin.weight"],
                           params['meta_learner.bert.transformer.layer.' + i + ".attention.v_lin.bias"]))

        q = q / math.sqrt(dim_per_head)
        scores = torch.matmul(q, k.transpose(2, 3))
        mask = (mask == 0).view(mask_reshp).expand_as(scores)  # (bs, n_heads, q_length, k_length)
        scores.masked_fill_(mask, -float("inf"))  # (bs, n_heads, q_length, k_length)

        weights = nn.Softmax(dim=-1)(scores)  # (bs, n_heads, q_length, k_length)
        dropout = nn.Dropout(p=self.config.attention_dropout)
        weights = dropout(weights)  # (bs, n_heads, q_length, k_length)

        context = torch.matmul(weights, v)  # (bs, n_heads, q_length, dim_per_head)
        context = unshape(context)  # (bs, q_length, dim)
        context = F.linear(context, params['meta_learner.bert.transformer.layer.' + i + ".attention.out_lin.weight"],
                           params['meta_learner.bert.transformer.layer.' + i + ".attention.out_lin.bias"])

        return context

    def ffn(self, input, params, i):
        x = F.linear(input, params['meta_learner.bert.transformer.layer.' + i + ".ffn.lin1.weight"],
                     params['meta_learner.bert.transformer.layer.' + i + ".ffn.lin1.bias"])
        activation = nn.GELU() if self.config.activation == "gelu" else nn.ReLU()
        x = activation(x)
        x = F.linear(x, params['meta_learner.bert.transformer.layer.' + i + ".ffn.lin2.weight"],
                     params['meta_learner.bert.transformer.layer.' + i + ".ffn.lin2.bias"])
        dropout = nn.Dropout(p=self.config.dropout)
        x = dropout(x)
        return x

    def transformer_block(self, x, attn_mask, param, i):
        sa_output = self.attention(x, x, x, attn_mask, param, i)
        sa_output = F.layer_norm(sa_output + x, (self.config.dim,),
                                 param['meta_learner.bert.transformer.layer.' + i + ".sa_layer_norm.weight"],
                                 param['meta_learner.bert.transformer.layer.' + i + ".sa_layer_norm.bias"],
                                 eps=1e-12)
        ffn_output = self.ffn(sa_output, param, i)
        ffn_output = F.layer_norm(ffn_output + sa_output, (self.config.dim,),
                                  param['meta_learner.bert.transformer.layer.' + i + ".output_layer_norm.weight"],
                                  param['meta_learner.bert.transformer.layer.' + i + ".output_layer_norm.bias"],
                                  eps=1e-12)

        return ffn_output

    def forward(
            self, x, attn_mask, params
    ):
        hidden_state = x
        for i in range(self.n_layers):
            hidden_state = self.transformer_block(hidden_state, attn_mask, params, str(i))
        return hidden_state


class Pooler(nn.Module):
    def __init__(self):
        super().__init__()

        self.activation = nn.Tanh()

    def forward(self, input, params):
        first_token_tensor = input[:, 0]
        pooled_output = F.linear(first_token_tensor, params['meta_learner.pooler.dense.weight'],
                                 params['meta_learner.pooler.dense.bias'])
        pooled_output = self.activation(pooled_output)

        return pooled_output


class BERTSentencePair(nn.Module):
    def __init__(self, config):
        super(BERTSentencePair, self).__init__()

        self.max_length = config.bert_max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased', do_lower_case=True)
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased', n_layers=config.n_layers,
                                                    n_heads=config.n_heads)

        print("initlizing ...............  models")
        for name, _ in self.bert.named_parameters():
            print(name)

        self.tanh = nn.Tanh()
        self.celoss = nn.CrossEntropyLoss(reduction="mean")
        self.pred_fc1 = nn.Linear(768, 2)
        self.config = config

        self.device = torch.device("cuda:" + str(config.device) if torch.cuda.is_available() else "cpu")

        for param in self.pred_fc1.parameters():
            if len(param.shape) >= 2:
                init.normal_(param, mean=0.0, std=config.initializer_range)
            elif len(param.shape) == 1:
                init.zeros_(param)

    def getPooling(self, h, mask):
        # ha, maska = self.getH(out_A, maska, lstm)
        vmax = h * mask.unsqueeze(2) + (1 - mask.unsqueeze(2)) * (-1e30)
        vmax = vmax.max(1)[0]
        return vmax  # , maska, ha

    def process_data(self, X):
        total = []
        total_y = []
        lens = []

        cls = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
        sep = self.tokenizer.convert_tokens_to_ids(['[SEP]'])

        o, s = X
        s_x, s_y = s
        o = o[0]

        for i, each_s in enumerate(s_x):
            cur_id = cls + o[0] + sep + each_s
            # print(len(o[0]), len(each_s), len(cur_id))
            bert_id = np.zeros([self.max_length], dtype=np.int64)

            if len(cur_id) > self.max_length:
                cur_id = cur_id[0: self.max_length]

            bert_id[0: len(cur_id)] = cur_id

            total.append(bert_id)
            lens.append(len(cur_id))
            total_y.append(s_y[i])

        total = torch.LongTensor(total).to(self.device)
        total_y = torch.LongTensor(total_y).to(self.device)
        lens = torch.LongTensor(lens).to(self.device)

        # get mask
        n = total.size()[0]
        idxes = torch.arange(self.max_length).expand(n, -1).to(self.device)
        mask = (idxes < lens.unsqueeze(1)).float()

        return total, mask, total_y

    def process_data_test(self, X):
        total = []
        total_y = []
        lens = []

        cls = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
        sep = self.tokenizer.convert_tokens_to_ids(['[SEP]'])

        for each in X:
            o, _, t = each

            t_x, t_y = t
            o = o[0]

            for i, each_t in enumerate(t_x):
                cur_id = cls + o[0] + sep + each_t
                bert_id = np.zeros([self.max_length], dtype=np.int64)

                if len(cur_id) > self.max_length:
                    cur_id = cur_id[0: self.max_length]

                bert_id[0: len(cur_id)] = cur_id

                total.append(bert_id)
                lens.append(len(cur_id))
                total_y.append(t_y[i])

        total = torch.LongTensor(total).to(self.device)
        total_y = torch.LongTensor(total_y).to(self.device)
        lens = torch.LongTensor(lens).to(self.device)

        # get mask
        n = total.size()[0]
        idxes = torch.arange(self.max_length).expand(n, -1).to(self.device)
        mask = (idxes < lens.unsqueeze(1)).float()

        return total, mask, total_y

    def forward(self, X, params=None):
        if params == None:
            return self.first_update(X)
        else:
            return self.second_update(X, params)

    def first_update(self, X):
        x, mask, y = self.process_data(X)

        out = self.bert(x, attention_mask=mask)
        out = out['last_hidden_state'][:, 0, :]

        out = self.pred_fc1(out)
        loss = self.celoss(out, y)

        pred = torch.max(out, dim=-1)[1].float().cpu()
        y = y.cpu()
        f1_val = metrics.f1_score(y, pred, average="macro")
        acc_val = torch.mean((pred.view(-1) == y.view(-1)).type(torch.FloatTensor))
        return loss, out, acc_val, f1_val

    def second_update(self, X, params):
        x, mask, y = self.process_data(X)

        disconfig = DistilBertConfig()
        disconfig.n_layers = self.config.n_layers
        disconfig.n_heads = self.config.n_heads
        emb = Embeddings(disconfig)
        embeddings = emb(x, params)

        trans = Transformer(disconfig)
        out = trans(embeddings, mask, params)

        out = out[:, 0, :]

        out = F.linear(out, params['meta_learner.pred_fc1.weight'], params['meta_learner.pred_fc1.bias'])
        loss = self.celoss(out, y)

        pred = torch.max(out, dim=-1)[1].float().cpu()
        y = y.cpu()
        f1_val = metrics.f1_score(y, pred, average="macro")
        acc_val = torch.mean((pred.view(-1) == y.view(-1)).type(torch.FloatTensor))
        return loss, out, acc_val, f1_val

    def predict(self, X):
        x, mask, y = self.process_data(X)

        out = self.bert(x, attention_mask=mask)
        out = out['last_hidden_state']
        out_feature = out[:, 0, :]

        out = self.pred_fc1(out_feature)
        loss = self.celoss(out, y)

        y = y.cpu()
        return out_feature, y


class BERT_Meta(nn.Module):
    def __init__(self, config):
        super(BERT_Meta, self).__init__()
        """
        The class defines meta-learner for MAML algorithm.
        """
        self.config = config
        self.meta_learner = BERTSentencePair(config)

    def forward(self, X, adapted_params=None):
        if adapted_params == None:
            out = self.meta_learner(X)
        else:
            out = self.meta_learner(X, adapted_params)
        return out

    def predict(self, X):
        features, labels = self.meta_learner.predict(X)
        return features, labels

    def cloned_state_dict(self):
        cloned_state_dict = {
            key: val.clone()
            for key, val in self.state_dict().items()
        }
        return cloned_state_dict


import torch
import torch.nn as nn
from torch import optim
import os
import pickle
import numpy as np
import random
from torch.backends import cudnn
import argparse
from tqdm import tqdm
from collections import OrderedDict
import copy
from sklearn import metrics
import torch.nn.functional as F
from torch.autograd import Variable


class Config():
    def __init__(self):
        pass

    model = "bert_meta"
    optimizer = "adam"
    lr = 2e-5
    task_lr = 2e-3
    batch_size = 1
    nepochs = 100
    dataset = "huffpost"
    data_path = "../datasets/oneshot_huffpost/"
    nepoch_no_imprv = 3
    device = 0
    train_epochs = 400
    dev_epochs = 300
    test_epochs = 300
    classes_per_set = 5
    samples_per_class = 10
    one_shot = 1
    dim_word = 50
    lstm_hidden = 50
    seed = 25
    num_train_updates = 1
    num_eval_updates = 3
    initializer_range = 0.1

    bert_max_length = 100
    target_type = "single" # used for ACD
    test_feature_epochs = 12

    n_layers = 1  # for ablation study in Section 5.4.2
    n_heads = 3


config = Config()
print(config)
print("current seed-----", config.seed)

seed = config.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

device = torch.device("cuda:" + str(config.device) if torch.cuda.is_available() else "cpu")


def data_loader(data_pack, epochs, config):
    data_cache = []

    for sample in range(epochs):  # every epoch has classes_per_set tasks
        n_tasks = []

        classes = np.random.choice(len(data_pack), config.classes_per_set * 2, False)
        classes_positive = classes[0: config.classes_per_set]
        classes_negative = classes[config.classes_per_set:]
        for j, cur_class in enumerate(classes_positive):  # each class
            x, y = [], []

            if config.target_type == "single":
                example_inds = np.random.choice(len(data_pack[cur_class][2]),
                                                config.one_shot, False)
                data_temp = np.array(data_pack[cur_class][2])[example_inds]
            else:
                example_inds = np.random.choice(len(data_pack[cur_class][3]),
                                                config.one_shot, False)
                data_temp = np.array(data_pack[cur_class][3])[example_inds]
            data_temp_x = data_temp[:, 0]

            positive_ids = np.random.choice(len(data_pack[cur_class][3]),
                                            config.samples_per_class, False)
            target_positive = np.array(data_pack[cur_class][3])[positive_ids]
            positive_x = target_positive[:, 0]

            negative_class_num = classes_negative[j]
            positive_class = data_pack[cur_class][1]
            negative_x = []
            negative_ids = np.random.choice(len(data_pack[negative_class_num][3]),
                                            len(data_pack[negative_class_num][3]) // 2, False)
            n = 0
            while len(negative_x) < config.samples_per_class:
                n_id = negative_ids[n]
                n += 1
                cur = data_pack[negative_class_num][3][n_id]
                if positive_class not in cur[1]:
                    negative_x.append(cur[0])

            x.extend(positive_x)
            y.extend(np.ones(len(positive_x)))

            x.extend(negative_x)
            y.extend(np.zeros(len(negative_x)))

            t_tmp = list(zip(x, y))
            random.shuffle(t_tmp)
            x[:], y[:] = zip(*t_tmp)

            n_tasks.append([
                [data_temp_x],
                [x[0: config.samples_per_class], y[0: config.samples_per_class]],
                [x[config.samples_per_class:], y[config.samples_per_class:]]
            ])

        data_cache.append(n_tasks)
    return data_cache


def data_loader_huffpost(data_pack, epochs, config):
    data_cache = []

    for sample in range(epochs):  # every epoch has classes_per_set tasks
        n_tasks = []

        classes_positive = np.random.choice(len(data_pack), config.classes_per_set, False)
        for j, cur_class in enumerate(classes_positive):  # each class
            x, y = [], []

            example_inds = np.random.choice(len(data_pack[cur_class][1]),
                                            config.one_shot + config.samples_per_class, False)

            one_shot_id = np.array([example_inds[0]])
            data_temp = np.array(data_pack[cur_class][1])[one_shot_id]
            data_temp_x = data_temp[:, 0]

            positive_ids = example_inds[1:]
            target_positive = np.array(data_pack[cur_class][1])[positive_ids]
            positive_x = target_positive[:, 0]

            # print(cur_class)
            my_range = list(range(0, cur_class)) + list(range(cur_class + 1, len(data_pack)))
            # print(my_range)
            negative_class_num = np.random.choice(my_range, 1, False)
            negative_class_num = negative_class_num[0]
            # print(negative_class_num)
            negative_ids = np.random.choice(len(data_pack[negative_class_num][1]),
                                            config.samples_per_class, False)

            neg_temp = np.array(data_pack[negative_class_num][1])[negative_ids]
            negative_x = neg_temp[:, 0]

            x.extend(positive_x)
            y.extend(np.ones(len(positive_x)))

            x.extend(negative_x)
            y.extend(np.zeros(len(negative_x)))

            t_tmp = list(zip(x, y))
            random.shuffle(t_tmp)
            x[:], y[:] = zip(*t_tmp)

            n_tasks.append([
                [data_temp_x],
                [x[0: config.samples_per_class], y[0: config.samples_per_class]],
                [x[config.samples_per_class:], y[config.samples_per_class:]]
            ])

        data_cache.append(n_tasks)
    return data_cache


def train_single_task(model, data, config):
    num_train_updates = config.num_train_updates

    model.train()

    one_shot, support, _ = data
    x = [one_shot, support]

    loss, _, _, _ = model(x)

    # clear previous gradients, compute gradients of all variables wrt loss
    def zero_grad(params):
        for p in params:
            if p.grad is not None:
                p.grad.zero_()

    zero_grad(model.parameters())
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True, allow_unused=True)

    # performs updates using calculated gradients
    # we manually compute adapted parameters since optimizer.step() operates in-place
    adapted_state_dict = model.cloned_state_dict()
    adapted_params = OrderedDict()
    for (key, val), grad in zip(model.named_parameters(), grads):
        if grad != None:
            adapted_params[key] = val - config.task_lr * grad
            adapted_state_dict[key] = adapted_params[key]

    return adapted_state_dict


def evaluate(logits, y):
    logits = logits.detach().cpu()
    y = y.detach().cpu()

    pred = torch.max(logits, dim=-1)[1].float()
    # print(logits)
    f1_val = metrics.f1_score(y, pred, average="macro")
    acc_val = torch.mean((pred.view(-1) == y.view(-1)).type(torch.FloatTensor))

    return acc_val, f1_val


def train_epoch(model, data_loader, meta_optimizer, config):
    model.train()

    total_acc, total_loss, total_f1 = [], [], []

    for step, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        adapted_state_dicts = []

        for i_batch in range(len(batch)):
            data = batch[i_batch]
            a_dict = train_single_task(model, data, config)
            adapted_state_dicts.append(a_dict)

        # Update the parameters of meta-learner
        # Compute losses with adapted parameters along with corresponding tasks
        # Updated the parameters of meta-learner using sum of the losses
        meta_loss = 0
        meta_logits, meta_label = [], []
        for i_batch in range(len(batch)):
            a_dict = adapted_state_dicts[i_batch]

            data = batch[i_batch]
            one_shot, _, target = data
            target_x, target_y = target

            y_meta = target_y
            x_meta = [one_shot, target]

            loss_t, logits_t, _, _ = model(x_meta, a_dict)

            meta_loss += loss_t
            meta_logits.append(logits_t)
            meta_label.extend(y_meta)

        meta_loss /= float(len(batch))
        # print(meta_loss)
        meta_acc, meta_f1 = evaluate(torch.cat(meta_logits, dim=0), torch.LongTensor(meta_label))

        meta_optimizer.zero_grad()
        meta_loss.backward()
        meta_optimizer.step()

        total_loss.append(meta_loss.item())
        total_acc.append(meta_acc.item())
        total_f1.append(meta_f1.item())

    total_loss = np.array(total_loss)
    total_acc = np.array(total_acc)
    total_f1 = np.array(total_f1)

    return np.mean(total_loss), np.mean(total_acc), np.mean(total_f1)


def eval_epoch(model, data_loader, config):
    model.eval()
    total_loss, total_auc = [], []
    total_results_acc, total_results_f1 = [], []

    for step, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        meta_loss = 0
        meta_logits, meta_label = [], []
        for i_batch in range(len(batch)):
            data = batch[i_batch]
            one_shot, support, target = data

            net_clone = copy.deepcopy(model)
            optim = torch.optim.Adam(net_clone.parameters(), lr=config.lr)
            for _ in range(config.num_eval_updates):
                loss_first, _, _, _ = net_clone([one_shot, support])

                optim.zero_grad()
                loss_first.backward()
                optim.step()

            loss_second, logits_t, _, _ = net_clone([one_shot, target])

            meta_loss += loss_second.item()
            meta_logits.append(logits_t)
            meta_label.extend(target[1])

        meta_loss /= float(len(batch))
        acc, f1 = evaluate(torch.cat(meta_logits, dim=0), torch.LongTensor(meta_label))

        total_loss.append(meta_loss)

        total_results_acc.append(acc.item())
        total_results_f1.append(f1.item())

    total_loss = np.array(total_loss)
    total_results_acc = np.mean(np.array(total_results_acc))
    total_results_f1 = np.mean(np.array(total_results_f1))
    return np.mean(total_loss), total_results_acc, total_results_f1


def eval_epoch_tune(model, data_loader, config):
    total_all_features, total_all_labels = [], []

    for step, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        meta_loss = 0
        meta_logits, meta_label = [], []
        for i_batch in range(len(batch)):
            data = batch[i_batch]
            one_shot, support, target = data

            net_clone = copy.deepcopy(model)
            optim = torch.optim.Adam(net_clone.parameters(), lr=config.lr)
            for _ in range(config.num_eval_updates):
                loss_first, _, _, _ = net_clone([one_shot, support])

                optim.zero_grad()
                loss_first.backward()
                optim.step()

            net_clone.eval()
            features, labels = net_clone.predict([one_shot, target])

            total_all_features.extend(features.cpu().detach().numpy())
            total_all_labels.extend(labels.cpu().detach().numpy())

    return total_all_features, total_all_labels


def eval_epoch_feature(model, data_loader, config):
    total_all_features, total_all_labels = [], []
    model.eval()
    for step, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        for i_batch in range(len(batch)):
            data = batch[i_batch]
            one_shot, _, target = data
            features, labels = model.predict([one_shot, target])

            total_all_features.extend(features.cpu().detach().numpy())
            total_all_labels.extend(labels.cpu().detach().numpy())

    return total_all_features, total_all_labels


def train(model, optimizer, train_data, valid_data, test_data, config):
    max_score = 0
    max_test_score = 0
    no_imprv = 0
    max_results = []

    if config.dataset == "acd":
        dev_data_loader = data_loader(valid_data, config.dev_epochs, config)
        test_data_loader = data_loader(test_data, config.test_epochs, config)
    else:
        dev_data_loader = data_loader_huffpost(valid_data, config.dev_epochs, config)
        test_data_loader = data_loader_huffpost(test_data, config.test_epochs, config)
    for epoch_i in range(config.nepochs):
        if config.dataset == "acd":
            train_data_loader = data_loader(train_data, config.train_epochs, config)
        else:
            train_data_loader = data_loader_huffpost(train_data, config.train_epochs, config)

        print("----------Epoch {:} out of {:}---------".format(epoch_i + 1, config.nepochs))
        print("training evaluation..........")
        train_loss, train_acc, train_f1 = train_epoch(model, train_data_loader, optimizer, config)
        train_msg = "train_loss: {:04.5f} train_acc: {:04.5f} train_f1: {:04.5f} ".format(train_loss, train_acc,
                                                                                          train_f1)
        print(train_msg)

        print("deving evaluation...........")
        dev_loss, dev_acc, dev_f1 = eval_epoch(model, dev_data_loader, config)
        dev_msg = "dev_loss: {:04.6f} dev_acc: {:04.6f} dev_f1: {:04.6f} ".format(dev_loss, dev_acc, dev_f1)
        print(dev_msg)

        print("testing evaluation...........")
        test_loss, test_acc, test_f1 = eval_epoch(model, test_data_loader, config)
        test_msg = "test_loss: {:04.6f} test_acc: {:04.6f} test_f1: {:04.6f} ".format(test_loss, test_acc, test_f1)
        print(test_msg)

        model_name = "results_" + str(config.seed) + "/" + "model.pkl"
        if dev_f1 >= max_score:
            no_imprv = 0
            torch.save(model.state_dict(), model_name)
            saved_model = copy.deepcopy(model)
            print("new best score!! The checkpoint file has been updated")
            max_score = dev_f1
            max_test_score = test_acc
            max_results = test_f1
        else:
            no_imprv += 1
            if no_imprv >= config.nepoch_no_imprv:
                break
        print("max score: {:04.4f}----max test acc: {:04.4f}".format(max_score, max_test_score))
        print("acc " + str(max_test_score) + " f1 " + str(max_results))

    # model_name = "results_"+str(config.seed)+"/" + "model.pkl"
    # torch.save(model.state_dict(), model_name)
    # saved_model = copy.deepcopy(model)

    print("-------------- getting features -----------")
    if config.dataset == "acd":
        loader = data_loader(test_data, config.test_feature_epochs, config)
    else:
        loader = data_loader_huffpost(test_data, config.test_feature_epochs, config)
    features, labels = eval_epoch_feature(saved_model, test_data_loader, config)
    pickle.dump([features, labels], open('DistilBERT_features_before_maml.p', 'wb'))
    features_tune, labels_tune = eval_epoch_tune(saved_model, loader, config)
    pickle.dump([features_tune, labels_tune], open('DisBERT_features_maml.p', "wb"))
    print(len(features), len(features_tune))
    print("done!")


def main():
    dir_output = "results_" + str(config.seed) + "/"
    path_log = dir_output

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    path = config.data_path + "data_info_distilbert.p"

    train_data, valid_data, test_data = pickle.load(open(path, "rb"))

    if config.model == "bert_meta":
        model = BERT_Meta(config)

    if torch.cuda.is_available():
        print("cuda available !!!!!!!!!!!")
        model = model.cuda(device)

    para_list = filter(lambda p: p.requires_grad, model.parameters())
    if config.optimizer == "adam":
        optimizer = optim.Adam(para_list, lr=config.lr)
    train(model, optimizer, train_data, valid_data, test_data, config)


if __name__ == '__main__':
    main()