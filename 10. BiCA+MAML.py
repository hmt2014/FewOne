import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import pickle
from sklearn import metrics
import numpy as np
import torch.nn.init as init


class NewProjModule(nn.Module):
    def __init__(self, config):
        super(NewProjModule, self).__init__()

        self.linear1 = nn.Linear(config.dim_word, config.dim_word)
        self.linear2 = nn.Linear(config.dim_word, config.dim_word)

    def forward(self, input, params=None):
        if params == None:
            i = nn.Sigmoid()(self.linear1(input))
            u = nn.Tanh()(self.linear2(input))
            out = i.mul(u)  # CMulTable().updateOutput([i, u])
        else:
            i = nn.Sigmoid()(F.linear(input, params['meta_learner.sentence_encoder.encoder.linear1.weight'],
                                      params['meta_learner.sentence_encoder.encoder.linear1.bias']))
            u = nn.Tanh()(F.linear(input, params['meta_learner.sentence_encoder.encoder.linear2.weight'],
                                   params['meta_learner.sentence_encoder.encoder.linear2.bias']))
            out = i.mul(u)
        return out


class SentenceEncoder(nn.Module):
    def __init__(self, config):
        super(SentenceEncoder, self).__init__()

        emb_path = config.data_path + "/embedding_info.p"

        # loading embeddings and other information
        if os.path.exists(emb_path):
            word_embeddings = pickle.load(open(emb_path, "rb"), encoding="iso-8859-1")[0]
        else:
            print("Error: Can not find embedding matrix")

        vocab_size = len(word_embeddings)
        self.word_embeds = nn.Embedding(vocab_size, config.dim_word, padding_idx=0)
        pretrained_weight_word = np.array(word_embeddings)
        self.word_embeds.weight.data.copy_(torch.from_numpy(pretrained_weight_word))
        self.word_embeds.weight.requires_grad = True
        self.config = config

        self.encoder = NewProjModule(config)
        self.dropout = nn.Dropout(0.1)

    def forward(self, X, params=None):

        if params == None:
            out = self.word_embeds(X).detach()
            out = self.dropout(out)
            out = self.encoder(out)
        else:
            out = F.embedding(X, params['meta_learner.sentence_encoder.word_embeds.weight']).detach()
            out = self.dropout(out)
            out = self.encoder(out, params)

        return out


params = {'aggin': 50, 'aggout': 50}


class CompareAgg(nn.Module):
    def __init__(self):
        super(CompareAgg, self).__init__()

        self.cnnin = params['aggin']
        self.cnnout = params['aggout']

        self.cnn12 = nn.Conv1d(self.cnnin, self.cnnout, 1)
        self.cnn22 = nn.Conv1d(self.cnnin, self.cnnout, 2)
        self.cnn32 = nn.Conv1d(self.cnnin, self.cnnout, 3)
        self.cnn42 = nn.Conv1d(self.cnnin, self.cnnout, 4)
        self.cnn52 = nn.Conv1d(self.cnnin, self.cnnout, 5)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.drop01 = nn.Dropout(0.1)

        self.pred_fc1 = nn.Linear(10 * self.cnnout, 2)

    def getCNNPooling2(self, out_A, params=None):
        if out_A.size(1) < 5:
            out_A = torch.cat([out_A, torch.zeros(out_A.size(0), 5 - out_A.size(1), out_A.size(2)).cuda()], 1)
        out_A = out_A.permute(0, 2, 1)
        if params == None:
            out_A1 = self.relu(self.cnn12(out_A)).max(2)[0]
            out_A2 = self.relu(self.cnn22(out_A)).max(2)[0]
            out_A3 = self.relu(self.cnn32(out_A)).max(2)[0]
            out_A4 = self.relu(self.cnn42(out_A)).max(2)[0]
            out_A5 = self.relu(self.cnn52(out_A)).max(2)[0]
        else:
            out_A1 = self.relu(
                F.conv1d(out_A, params['meta_learner.pred.cnn12.weight'], params['meta_learner.pred.cnn12.bias'])).max(
                2)[0]
            out_A2 = self.relu(
                F.conv1d(out_A, params['meta_learner.pred.cnn22.weight'], params['meta_learner.pred.cnn22.bias'])).max(
                2)[0]
            out_A3 = self.relu(
                F.conv1d(out_A, params['meta_learner.pred.cnn32.weight'], params['meta_learner.pred.cnn32.bias'])).max(
                2)[0]
            out_A4 = self.relu(
                F.conv1d(out_A, params['meta_learner.pred.cnn42.weight'], params['meta_learner.pred.cnn42.bias'])).max(
                2)[0]
            out_A5 = self.relu(
                F.conv1d(out_A, params['meta_learner.pred.cnn52.weight'], params['meta_learner.pred.cnn52.bias'])).max(
                2)[0]

        v1 = torch.cat([out_A1, out_A2, out_A3, out_A4, out_A5], -1)
        v2 = torch.stack([out_A1, out_A2, out_A3, out_A4, out_A5], 1).max(1)[0]
        v3 = torch.stack([out_A1, out_A2, out_A3, out_A4, out_A5], 1).mean(1)[0]

        return v1, v2, v3

    def forward(self, out_Q, out_A, params=None):
        if params == None:
            out_A = self.drop01(out_A)
            out_Q = self.drop01(out_Q)

            vq1, vq2, vq3 = self.getCNNPooling2(out_Q)
            va1, va2, va3 = self.getCNNPooling2(out_A)

            x_ = torch.cat([vq1, va1], -1)
            # ----- Prediction Layer -----
            out_feature = x_
            x_ = self.drop01(x_)
            x = self.pred_fc1(x_)

        else:
            out_A = self.drop01(out_A)
            out_Q = self.drop01(out_Q)

            vq1, vq2, vq3 = self.getCNNPooling2(out_Q, params)
            va1, va2, va3 = self.getCNNPooling2(out_A, params)

            x_ = torch.cat([vq1, va1], -1)
            # ----- Prediction Layer -----
            out_feature = x_
            x_ = self.drop01(x_)
            x = F.linear(x_, params['meta_learner.pred.pred_fc1.weight'], params['meta_learner.pred.pred_fc1.bias'])
        return x, out_feature


class Net(nn.Module):
    def __init__(self, config, sentence_encoder):
        super(Net, self).__init__()
        """
        The class defines meta-learner for MAML algorithm.
        """
        self.sentence_encoder = sentence_encoder

        self.embed_size = config.dim_word
        self.hidden_size = config.lstm_hidden

        self.pred = CompareAgg()
        self.sigmoid = nn.Sigmoid()

        self.config = config

        self.celoss = nn.CrossEntropyLoss(reduction="mean")
        self.device = torch.device("cuda:" + str(config.device) if torch.cuda.is_available() else "cpu")

    def getQKV(self, Q, K, V, mask1, mask2, w, dca=False, k=0):
        '''
        Q:batch_size,len1,dim ======================h
        K:batch_Size,len2,dim ======================le
        V = K
        return shape batch_size,len1,dim
        '''
        co = Q @ K.transpose(-1, -2)
        co = co * mask2.unsqueeze(1) + (1 - mask2.unsqueeze(1)) * (-1e30)
        co = F.softmax(co, -1)

        return co @ V * mask1.unsqueeze(2), co

    def process_data(self, X):
        one_shot, others = X
        one_x, one_mask = one_shot
        others_x, others_mask, y = others

        one_x = torch.LongTensor(one_x[0]).to(self.device)
        one_mask = torch.FloatTensor(one_mask[0]).to(self.device)

        others_x = torch.LongTensor(others_x).to(self.device)
        others_mask = torch.FloatTensor(others_mask).to(self.device)

        y = torch.LongTensor(y).to(self.device)

        return one_x, one_mask, others_x, others_mask, y

    def first_update(self, X):
        one_x, one_mask, others_x, others_mask, y = self.process_data(X)

        batch_size = others_x.size()[0]

        one_x = one_x.unsqueeze(0).repeat(batch_size, 1)
        one_mask = one_mask.unsqueeze(0).repeat(batch_size, 1)

        maska1 = others_mask
        maskq = one_mask

        q = self.sentence_encoder(one_x)
        a1 = self.sentence_encoder(others_x)

        q_aware_a1, ck1 = self.getQKV(a1, q, q, maska1, maskq, "w", dca='topk', k=5)
        a_aware_q1, ck2 = self.getQKV(q, a1, a1, maskq, maska1, "w", dca='topk', k=5)

        va1 = a1 * q_aware_a1
        vq1 = q * a_aware_q1

        out2, _ = self.pred(vq1, va1)
        loss = self.celoss(out2, y)

        return loss, out2

    def second_update(self, X, params):
        one_x, one_mask, others_x, others_mask, y = self.process_data(X)

        batch_size = others_x.size()[0]

        one_x = one_x.unsqueeze(0).repeat(batch_size, 1)
        one_mask = one_mask.unsqueeze(0).repeat(batch_size, 1)

        maska1 = others_mask
        maskq = one_mask

        q = self.sentence_encoder(one_x, params)
        a1 = self.sentence_encoder(others_x, params)

        q_aware_a1, ck1 = self.getQKV(a1, q, q, maska1, maskq, "w", dca='topk', k=5)
        a_aware_q1, ck2 = self.getQKV(q, a1, a1, maskq, maska1, "w", dca='topk', k=5)

        va1 = a1 * q_aware_a1
        vq1 = q * a_aware_q1

        out2, feature = self.pred(vq1, va1, params)

        loss = self.celoss(out2, y)

        return loss, out2

    def predict(self, X):
        one_x, one_mask, others_x, others_mask, y = self.process_data(X)

        batch_size = others_x.size()[0]

        one_x = one_x.unsqueeze(0).repeat(batch_size, 1)
        one_mask = one_mask.unsqueeze(0).repeat(batch_size, 1)

        maska1 = others_mask
        maskq = one_mask

        q = self.sentence_encoder(one_x)
        a1 = self.sentence_encoder(others_x)
        q_aware_a1, ck1 = self.getQKV(a1, q, q, maska1, maskq, "w", dca='topk', k=5)
        a_aware_q1, ck2 = self.getQKV(q, a1, a1, maskq, maska1, "w", dca='topk', k=5)
        va1 = a1 * q_aware_a1
        vq1 = q * a_aware_q1

        out2, features = self.pred(vq1, va1)
        loss = self.celoss(out2, y)

        return features, y

    def forward(self, X, params=None):

        if params == None:
            return self.first_update(X)
        else:
            return self.second_update(X, params)


class CompareAggregate_double_Meta(nn.Module):
    def __init__(self, config, sentence_encoder):
        super(CompareAggregate_double_Meta, self).__init__()
        """
        The class defines meta-learner for MAML algorithm.
        """
        self.config = config
        self.meta_learner = Net(config, sentence_encoder)

        self.init_weight(config)

    def init_weight(self, config):
        print("initialing weights")
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print("Linear")
                for param in m.parameters():
                    if len(param.shape) >= 2:
                        init.normal_(param, mean=0.0, std=config.initializer_range)
                    elif len(param.shape) == 1:
                        init.zeros_(param)
            elif isinstance(m, nn.Conv1d):
                print("Conv1d")
                for param in m.parameters():
                    init.normal_(param, mean=0.0, std=config.initializer_range)
            elif isinstance(m, nn.Embedding):
                print("Embedding, Do not initialize!")

    def forward(self, X, adapted_params=None):
        if adapted_params == None:
            out = self.meta_learner(X)
        else:
            out = self.meta_learner(X, adapted_params)
        return out

    def predict(self, X):
        features, y = self.meta_learner.predict(X)
        return features, y

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

    model = "compare_aggregate_double_meta"
    optimizer = "adam"
    lr = 1e-3
    task_lr = 1e-1
    batch_size = 1
    nepochs = 120
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
    seed = 20
    num_train_updates = 1
    num_eval_updates = 3
    initializer_range = 0.1

    target_type = "single" # used for ACD
    test_feature_epochs = 12


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
            x, x_mask, y = [], [], []
            example_inds = np.random.choice(len(data_pack[cur_class][2]),
                                            config.one_shot, False)

            data_temp = np.array(data_pack[cur_class][2])[example_inds]
            data_temp_x = data_temp[:, 0]
            data_temp_mask = data_temp[:, 1]

            positive_ids = np.random.choice(len(data_pack[cur_class][3]),
                                            config.samples_per_class, False)
            target_positive = np.array(data_pack[cur_class][3])[positive_ids]
            positive_x = target_positive[:, 0]
            positive_x_mask = target_positive[:, 1]

            negative_class_num = classes_negative[j]
            positive_class = data_pack[cur_class][1]
            negative_x, negative_x_mask = [], []
            negative_ids = np.random.choice(len(data_pack[negative_class_num][3]),
                                            len(data_pack[negative_class_num][3]) // 2, False)
            n = 0
            while len(negative_x) < config.samples_per_class:
                n_id = negative_ids[n]
                n += 1
                cur = data_pack[negative_class_num][3][n_id]
                if positive_class not in cur[2]:
                    negative_x.append(cur[0])
                    negative_x_mask.append(cur[1])

            x.extend(positive_x)
            x_mask.extend(positive_x_mask)
            y.extend(np.ones(len(positive_x)))

            x.extend(negative_x)
            x_mask.extend(negative_x_mask)
            y.extend(np.zeros(len(negative_x)))

            t_tmp = list(zip(x, x_mask, y))
            random.shuffle(t_tmp)
            x[:], x_mask[:], y[:] = zip(*t_tmp)

            n_tasks.append([
                [data_temp_x, data_temp_mask],
                [x[0: config.samples_per_class], x_mask[0: config.samples_per_class], y[0: config.samples_per_class]],
                [x[config.samples_per_class:], x_mask[config.samples_per_class:], y[config.samples_per_class:]]
            ])

        data_cache.append(n_tasks)
    return data_cache


def data_loader_multi(data_pack, epochs, config):
    data_cache = []

    for sample in range(epochs):  # every epoch has classes_per_set tasks
        n_tasks = []

        classes = np.random.choice(len(data_pack), config.classes_per_set * 2, False)
        classes_positive = classes[0: config.classes_per_set]
        classes_negative = classes[config.classes_per_set:]
        for j, cur_class in enumerate(classes_positive):  # each class
            x, x_mask, y = [], [], []
            example_inds = np.random.choice(len(data_pack[cur_class][3]),
                                            config.one_shot + config.samples_per_class, False)

            one_shot_id = np.array([example_inds[0]])
            data_temp = np.array(data_pack[cur_class][3])[one_shot_id]
            data_temp_x = data_temp[:, 0]
            data_temp_mask = data_temp[:, 1]

            positive_ids = example_inds[1:]
            target_positive = np.array(data_pack[cur_class][3])[positive_ids]
            positive_x = target_positive[:, 0]
            positive_x_mask = target_positive[:, 1]

            negative_class_num = classes_negative[j]
            positive_class = data_pack[cur_class][1]
            negative_x, negative_x_mask = [], []
            negative_ids = np.random.choice(len(data_pack[negative_class_num][3]),
                                            len(data_pack[negative_class_num][3]) // 2, False)
            n = 0
            while len(negative_x) < config.samples_per_class:
                n_id = negative_ids[n]
                n += 1
                cur = data_pack[negative_class_num][3][n_id]
                if positive_class not in cur[2]:
                    negative_x.append(cur[0])
                    negative_x_mask.append(cur[1])

            x.extend(positive_x)
            x_mask.extend(positive_x_mask)
            y.extend(np.ones(len(positive_x)))

            x.extend(negative_x)
            x_mask.extend(negative_x_mask)
            y.extend(np.zeros(len(negative_x)))

            t_tmp = list(zip(x, x_mask, y))
            random.shuffle(t_tmp)
            x[:], x_mask[:], y[:] = zip(*t_tmp)

            n_tasks.append([
                [data_temp_x, data_temp_mask],
                [x[0: config.samples_per_class], x_mask[0: config.samples_per_class], y[0: config.samples_per_class]],
                [x[config.samples_per_class:], x_mask[config.samples_per_class:], y[config.samples_per_class:]]
            ])

        data_cache.append(n_tasks)
    return data_cache


def data_loader_huffpost(data_pack, epochs, config):
    data_cache = []

    for sample in range(epochs):  # every epoch has classes_per_set tasks
        n_tasks = []

        classes_positive = np.random.choice(len(data_pack), config.classes_per_set, False)
        # classes_positive = classes[0: config.classes_per_set]
        # classes_negative = classes[config.classes_per_set:]
        for j, cur_class in enumerate(classes_positive):  # each class
            x, x_mask, y = [], [], []
            example_inds = np.random.choice(len(data_pack[cur_class][1]),
                                            config.one_shot + config.samples_per_class, False)

            one_shot_id = np.array([example_inds[0]])
            data_temp = np.array(data_pack[cur_class][1])[one_shot_id]
            data_temp_x = data_temp[:, 0]
            data_temp_mask = data_temp[:, 1]

            positive_ids = example_inds[1:]
            target_positive = np.array(data_pack[cur_class][1])[positive_ids]
            positive_x = target_positive[:, 0]
            positive_x_mask = target_positive[:, 1]

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
            negative_x_mask = neg_temp[:, 1]

            x.extend(positive_x)
            x_mask.extend(positive_x_mask)
            y.extend(np.ones(len(positive_x)))

            x.extend(negative_x)
            x_mask.extend(negative_x_mask)
            y.extend(np.zeros(len(negative_x)))

            t_tmp = list(zip(x, x_mask, y))
            random.shuffle(t_tmp)
            x[:], x_mask[:], y[:] = zip(*t_tmp)

            n_tasks.append([
                [data_temp_x, data_temp_mask],
                [x[0: config.samples_per_class], x_mask[0: config.samples_per_class], y[0: config.samples_per_class]],
                [x[config.samples_per_class:], x_mask[config.samples_per_class:], y[config.samples_per_class:]]
            ])

        data_cache.append(n_tasks)
    return data_cache


def train_single_task(model, data, config):
    num_train_updates = config.num_train_updates

    model.train()

    one_shot, support, _ = data
    support_x, support_mask, support_y = support
    y = support_y
    x = [one_shot, [support_x, support_mask, y]]

    loss, _ = model(x)

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
    y = y.cpu()

    pred = torch.max(logits.cpu(), dim=-1)[1].float()
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
            target_x, target_mask, target_y = target

            y_meta = target_y
            x_meta = [one_shot, [target_x, target_mask, y_meta]]

            loss_t, logits_t = model(x_meta, a_dict)

            meta_loss += loss_t
            meta_logits.append(logits_t)
            meta_label.extend(y_meta)

        meta_loss /= float(len(batch))
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
            optim = torch.optim.SGD(net_clone.parameters(), lr=config.task_lr)
            for _ in range(config.num_eval_updates):
                loss_first, _ = net_clone([one_shot, support])

                optim.zero_grad()
                loss_first.backward()
                optim.step()

            loss_second, logits_t = net_clone([one_shot, target])

            meta_loss += loss_second.item()
            meta_logits.append(logits_t)
            meta_label.extend(target[2])

        meta_loss /= float(len(batch))
        acc, f1 = evaluate(torch.cat(meta_logits, dim=0), torch.LongTensor(meta_label))

        total_loss.append(meta_loss)

        total_results_acc.append(acc.item())
        total_results_f1.append(f1.item())

    total_loss = np.array(total_loss)
    total_results_acc = np.mean(np.array(total_results_acc))
    total_results_f1 = np.mean(np.array(total_results_f1))
    return np.mean(total_loss), total_results_acc, total_results_f1


def eval_epoch_feature(model, data_loader, config):
    model.eval()
    total_all_features = []
    total_all_labels = []

    for step, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        for i_batch in range(len(batch)):
            one_shot, _, target = batch[i_batch]

            features, labels = model.predict([one_shot, target])
            total_all_features.extend(features.cpu().detach().numpy())
            total_all_labels.extend(labels.cpu().detach().numpy())

    return total_all_features, total_all_labels


def eval_epoch_finetune(model, data_loader, config):
    total_all_features, total_all_labels = [], []

    for step, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        for i_batch in range(len(batch)):
            data = batch[i_batch]
            one_shot, support, target = data

            net_clone = copy.deepcopy(model)
            optim = torch.optim.SGD(net_clone.parameters(), lr=config.task_lr)
            for _ in range(config.num_eval_updates):
                loss_first, _ = net_clone([one_shot, support])
                optim.zero_grad()
                loss_first.backward()
                optim.step()

            net_clone.eval()
            features, labels = net_clone.predict([one_shot, target])

            total_all_features.extend(features.cpu().detach().numpy())
            total_all_labels.extend(labels.cpu().detach().numpy())

    return total_all_features, total_all_labels


def train(model, optimizer, train_data, valid_data, test_data, config):
    max_score = 0
    max_test_score = 0
    no_imprv = 0
    max_results = []

    if config.dataset == "acd":
        if config.target_type == "single":
            dev_data_loader = data_loader(valid_data, config.dev_epochs, config)
            test_data_loader = data_loader(test_data, config.test_epochs, config)
        else:
            dev_data_loader = data_loader_multi(valid_data, config.dev_epochs, config)
            test_data_loader = data_loader_multi(test_data, config.test_epochs, config)
    else:
        dev_data_loader = data_loader_huffpost(valid_data, config.dev_epochs, config)
        test_data_loader = data_loader_huffpost(test_data, config.test_epochs, config)

    for epoch_i in range(config.nepochs):
        if config.dataset == "acd":
            if config.target_type == "single":
                train_data_loader = data_loader(train_data, config.train_epochs, config)
            else:
                train_data_loader = data_loader_multi(train_data, config.train_epochs, config)
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

    print("-------------- getting features -----------")
    if config.dataset == "acd":
        if config.target_type == "single":
            loader = data_loader(test_data, config.test_feature_epochs, config)
        else:
            loader = data_loader_multi(test_data, config.test_feature_epochs, config)
    else:
        loader = data_loader_huffpost(test_data, config.test_feature_epochs, config)
    features, labels = eval_epoch_feature(saved_model, test_data_loader, config)
    pickle.dump([features, labels], open('BiCA_feature_before_maml.p', "wb"))
    features_tune, labels_tune = eval_epoch_finetune(saved_model, loader, config)
    pickle.dump([features_tune, labels_tune], open('BiCA_features_maml.p', "wb"))
    print(len(features), len(features_tune))
    print("done!")


def main():
    dir_output = "results_" + str(config.seed) + "/"
    path_log = dir_output

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    path = config.data_path + "data_info.p"

    train_data, valid_data, test_data = pickle.load(open(path, "rb"))

    sentence_encoder = SentenceEncoder(config)
    if config.model == "compare_aggregate_double_meta":
        model = CompareAggregate_double_Meta(config, sentence_encoder)

    if torch.cuda.is_available():
        print("cuda available !!!!!!!!!!!")
        model = model.cuda(device)

    para_list = filter(lambda p: p.requires_grad, model.parameters())
    if config.optimizer == "adam":
        optimizer = optim.Adam(para_list, lr=config.lr)
    train(model, optimizer, train_data, valid_data, test_data, config)


if __name__ == '__main__':
    main()