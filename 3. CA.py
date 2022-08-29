import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import pickle
from sklearn import metrics
import numpy
import torch.nn.init as init

params = {'aggin': 50, 'aggout': 50}


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

    def forward(self, X):
        out = self.word_embeds(X)
        out = self.dropout(out)
        out = self.encoder(out)

        return out


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
        self.pred_fc1 = nn.Linear(5 * self.cnnout, 2)

    def getCNNPooling2(self, out_A):
        if out_A.size(1) < 5:
            out_A = torch.cat([out_A, torch.zeros(out_A.size(0), 5 - out_A.size(1), out_A.size(2)).cuda()], 1)
        out_A = out_A.permute(0, 2, 1)

        out_A1 = self.relu(self.cnn12(out_A)).max(2)[0]
        out_A2 = self.relu(self.cnn22(out_A)).max(2)[0]
        out_A3 = self.relu(self.cnn32(out_A)).max(2)[0]
        out_A4 = self.relu(self.cnn42(out_A)).max(2)[0]
        out_A5 = self.relu(self.cnn52(out_A)).max(2)[0]

        v1 = torch.cat([out_A1, out_A2, out_A3, out_A4, out_A5], -1)
        v2 = torch.stack([out_A1, out_A2, out_A3, out_A4, out_A5], 1).max(1)[0]
        v3 = torch.stack([out_A1, out_A2, out_A3, out_A4, out_A5], 1).mean(1)[0]

        return v1, v2, v3

    def forward(self, out_A):
        out_A = self.drop01(out_A)
        va1, va2, va3 = self.getCNNPooling2(out_A)

        x_ = va1
        # ----- Prediction Layer -----
        out_feature = x_
        x_ = self.drop01(x_)
        x = self.pred_fc1(x_)
        return x, out_feature


class CompareAggregate(nn.Module):
    def __init__(self, config, sentence_encoder):
        super(CompareAggregate, self).__init__()
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

    def process_data(self, X, is_tuning):
        one_shot, one_mask = [], []
        total, total_mask = [], []
        y = []

        for each in X:
            o, s, t = each

            total.extend(s[0])
            total_mask.extend(s[1])
            y.extend(s[2])

            if not is_tuning:
                total.extend(t[0])
                total_mask.extend(t[1])
                y.extend(t[2])
                length = len(s[0]) + len(t[0])
            else:
                length = len(s[0])
            one_shot.extend(numpy.repeat(o[0], length, axis=0))
            one_mask.extend(numpy.repeat(o[1], length, axis=0))

        one_x = torch.LongTensor(one_shot).to(self.device)
        one_mask = torch.FloatTensor(one_mask).to(self.device)

        total = torch.LongTensor(total).to(self.device)
        total_mask = torch.FloatTensor(total_mask).to(self.device)

        y = torch.LongTensor(y).to(self.device)

        return one_x, one_mask, total, total_mask, y

    def process_data_test(self, X):
        one_shot, one_mask = [], []
        total, total_mask = [], []
        y = []

        for each in X:
            o, s, t = each
            length = len(t[0])
            one_shot.extend(numpy.repeat(o[0], length, axis=0))
            one_mask.extend(numpy.repeat(o[1], length, axis=0))

            total.extend(t[0])
            total_mask.extend(t[1])
            y.extend(t[2])

        one_x = torch.LongTensor(one_shot).to(self.device)
        one_mask = torch.FloatTensor(one_mask).to(self.device)

        total = torch.LongTensor(total).to(self.device)
        total_mask = torch.FloatTensor(total_mask).to(self.device)

        y = torch.LongTensor(y).to(self.device)

        return one_x, one_mask, total, total_mask, y

    def forward(self, X, is_tuning=False):
        one_x, one_mask, others_x, others_mask, y = self.process_data(X, is_tuning)

        maska1 = others_mask
        maskq = one_mask

        q = self.sentence_encoder(one_x)
        a1 = self.sentence_encoder(others_x)

        q_aware_a1, ck1 = self.getQKV(a1, q, q, maska1, maskq, "w", dca='topk', k=5)

        va1 = a1 * q_aware_a1

        out2, _ = self.pred(va1)

        loss = self.celoss(out2, y)

        pred = torch.max(out2, dim=-1)[1].float().detach().cpu()
        # print(logits)
        y = y.detach().cpu()
        f1_val = metrics.f1_score(y, pred, average="macro")
        acc_val = torch.mean((pred.view(-1) == y.view(-1)).type(torch.FloatTensor))
        return loss, out2, acc_val, f1_val

    def predict(self, X):
        one_x, one_mask, others_x, others_mask, y = self.process_data_test(X)

        maska1 = others_mask
        maskq = one_mask

        q = self.sentence_encoder(one_x)
        a1 = self.sentence_encoder(others_x)

        q_aware_a1, ck1 = self.getQKV(a1, q, q, maska1, maskq, "w", dca='topk', k=5)

        va1 = a1 * q_aware_a1

        out2, out_feature = self.pred(va1)

        loss = self.celoss(out2, y)

        pred = torch.max(out2, dim=-1)[1].float().detach().cpu()
        y = y.detach().cpu()
        f1_val = metrics.f1_score(y, pred, average="macro")
        acc_val = torch.mean((pred.view(-1) == y.view(-1)).type(torch.FloatTensor))
        return loss, out_feature, acc_val, f1_val, y


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
from torch.autograd import Variable
import copy


class Config():
    def __init__(self):
        pass

    model = "compare_aggregate"
    optimizer = "adam"
    lr = 1e-3
    task_lr = 1e-1
    batch_size = 1
    nepochs = 120
    nepoch_no_imprv = 3
    device = 0
    train_epochs = 400
    dev_epochs = 300
    test_epochs = 300
    classes_per_set = 5
    samples_per_class = 10
    dim_word = 50
    lstm_hidden = 50
    seed = 25
    initializer_range = 0.1
    margin = 0.2
    one_shot = 1

    target_type = "single" #used for ACD
    dataset = "huffpost"
    data_path = "../datasets/oneshot_huffpost/"
    test_feature_epochs = 12
    num_eval_updates = 3


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


def train_epoch(model, data_loader, optimizer, config):
    model.train()

    total_loss = []
    total_results_acc = []
    total_results_f1 = []

    for step, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        # sentiment loss
        loss, logits, acc, f1 = model(batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())

        total_results_acc.append(acc.item())
        total_results_f1.append(f1.item())

    total_loss = np.array(total_loss)

    total_results_acc = np.mean(np.array(total_results_acc))
    total_results_f1 = np.mean(np.array(total_results_f1))

    return np.mean(total_loss), total_results_acc, total_results_f1


def eval_epoch(model, data_loader, config):
    model.eval()

    total_loss = []
    total_results_acc = []
    total_results_f1 = []

    for step, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        # sentiment loss
        loss, _, acc, f1, _ = model.predict(batch)

        total_loss.append(loss.item())

        total_results_acc.append(acc.item())
        total_results_f1.append(f1.item())

    total_loss = np.array(total_loss)

    total_results_acc = np.mean(np.array(total_results_acc))
    total_results_f1 = np.mean(np.array(total_results_f1))

    return np.mean(total_loss), total_results_acc, total_results_f1


def eval_epoch_finetune(model, data_loader, config):
    total_results_acc = []
    total_results_f1 = []
    total_all_features = []
    total_all_labels = []

    for step, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        net_clone = copy.deepcopy(model)
        optim = torch.optim.SGD(net_clone.parameters(), lr=config.task_lr)
        for _ in range(config.num_eval_updates):
            loss_first, _, _, _ = net_clone(batch, is_tuning=True)

            optim.zero_grad()
            loss_first.backward()
            optim.step()

        net_clone.eval()
        loss_second, features, acc, f1, labels = net_clone.predict(batch)
        total_results_acc.append(acc)
        total_results_f1.append(f1)

        total_all_features.extend(features.cpu().detach().numpy())
        total_all_labels.extend(labels.cpu().detach().numpy())

    total_results_acc = np.mean(np.array(total_results_acc))
    total_results_f1 = np.mean(np.array(total_results_f1))
    return total_results_acc, total_results_f1, total_all_features, total_all_labels


def eval_epoch_feature(model, data_loader, config):
    model.eval()
    total_all_features = []
    total_all_labels = []

    for step, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        loss_second, features, acc, f1, labels = model.predict(batch)
        total_all_features.extend(features.cpu().detach().numpy())
        total_all_labels.extend(labels.cpu().detach().numpy())

    return total_all_features, total_all_labels


def train(model, optimizer, train_data, valid_data, test_data, config):
    max_score = 0
    max_test_score = 0
    max_test_result_acc = []
    max_test_result_f1 = []
    no_imprv = 0

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
        train_msg = "train_loss: {:04.5f} train_acc: {:04.5f} train_f1: {:04.5f}".format(train_loss, train_acc,
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
            max_test_score = test_f1
            max_test_result_acc = test_acc
            max_test_result_f1 = test_f1
        else:
            no_imprv += 1
            if no_imprv >= config.nepoch_no_imprv:
                break
        print("max score: {:04.4f}----max test score: {:04.4f}".format(max_score, max_test_score))
        print("accuracy ", max_test_result_acc, " f1 ", max_test_result_f1)

    print("-------------------------------------------")
    print("---- fining tuning second time ------------")
    # fine-tune the second time, using the support set of novel classes
    acc, f1, _, _ = eval_epoch_finetune(saved_model, test_data_loader, config)
    print("acc ", acc, " f1 ", f1)

    print("-------------- getting features -----------")
    if config.dataset == "acd":
        if config.target_type == "single":
            loader = data_loader(test_data, config.test_feature_epochs, config)
        else:
            loader = data_loader_multi(test_data, config.test_feature_epochs, config)
    else:
        loader = data_loader_huffpost(test_data, config.test_feature_epochs, config)

    features, labels = eval_epoch_feature(saved_model, loader, config)
    pickle.dump([features, labels], open('CA_features.p', "wb"))
    _, _, features_tune, labels_tune = eval_epoch_finetune(saved_model, loader, config)
    pickle.dump([features_tune, labels_tune], open("CA_features_finetuned.p", "wb"))
    print(len(features), len(features_tune))
    print("done!")


def check(data):
    for each in data:
        _, name, single, multi = each
        for e in single:
            mask = e[1]
            if np.sum(np.array(mask)) == 0:
                print(name)
                print("zero length")

        for e in multi:
            mask = e[1]
            if np.sum(np.array(mask)) == 0:
                print(name)
                print("zero length")


def main():
    dir_output = "results_" + str(config.seed) + "/"
    path_log = dir_output

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    path = config.data_path + "data_info.p"

    train_data, valid_data, test_data = pickle.load(open(path, "rb"))

    sentence_encoder = SentenceEncoder(config)
    if config.model == "compare_aggregate":
        model = CompareAggregate(config, sentence_encoder)

    if torch.cuda.is_available():
        print("cuda available !!!!!!!!!!!")
        model = model.cuda(device)

    para_list = filter(lambda p: p.requires_grad, model.parameters())
    if config.optimizer == "adam":
        optimizer = optim.Adam(para_list, lr=config.lr)
    train(model, optimizer, train_data, valid_data, test_data, config)


if __name__ == '__main__':
    main()