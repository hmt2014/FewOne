import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import pickle
from sklearn import metrics
import numpy

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


class OW_Proto(nn.Module):
    def __init__(self, config, sentence_encoder):
        super(OW_Proto, self).__init__()
        """
        The class defines meta-learner for MAML algorithm.
        """
        self.sentence_encoder = sentence_encoder

        self.embed_size = config.dim_word
        self.hidden_size = config.lstm_hidden

        self.config = config

        self.celoss = nn.CrossEntropyLoss(reduction="mean")

        self.device = torch.device("cuda:" + str(config.device) if torch.cuda.is_available() else "cpu")

    def process_data(self, X):
        one_shot, one_mask = [], []
        total, total_mask = [], []
        y = []

        for each in X:
            o, s, t = each
            length = len(s[0]) + len(t[0])
            one_shot.extend(numpy.repeat(o[0], length, axis=0))
            one_mask.extend(numpy.repeat(o[1], length, axis=0))

            total.extend(s[0])
            total.extend(t[0])

            total_mask.extend(s[1])
            total_mask.extend(t[1])

            y.extend(s[2])
            y.extend(t[2])

        one_x = torch.LongTensor(one_shot).to(self.device)
        one_mask = torch.FloatTensor(one_mask).to(self.device)

        total = torch.LongTensor(total).to(self.device)
        total_mask = torch.FloatTensor(total_mask).to(self.device)

        y = torch.LongTensor(y).to(self.device)

        return one_x, one_mask, total, total_mask, y

    def __dist__(self, x, y, dim):
        return (torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

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

    def getPooling(self, h, mask):
        # ha, maska = self.getH(out_A, maska, lstm)
        vmax = h * mask.unsqueeze(2) + (1 - mask.unsqueeze(2)) * (-1e30)
        vmax = vmax.max(1)[0]
        return vmax  # , maska, ha

    def forward(self, X):
        one_x, one_mask, others_x, others_mask, y = self.process_data(X)

        maska1 = others_mask
        maskq = one_mask

        q = self.sentence_encoder(one_x)
        a1 = self.sentence_encoder(others_x)

        vq = self.getPooling(q, maskq)
        va = self.getPooling(a1, maska1)

        zero_protos = torch.zeros_like(vq)

        protos = torch.cat([zero_protos.unsqueeze(1), vq.unsqueeze(1)], dim=1)

        logits = -self.__batch_dist__(protos, va.unsqueeze(1)).squeeze(1)

        loss = self.celoss(logits, y)

        pred = torch.max(logits, dim=-1)[1].float()
        # print(logits)
        y = y.cpu().detach()
        pred = pred.cpu().detach()
        f1_val = metrics.f1_score(y, pred, average="macro")
        acc_val = torch.mean((pred.view(-1) == y.view(-1)).type(torch.FloatTensor))
        return loss, logits, acc_val, f1_val

    def predict(self, X):
        one_x, one_mask, others_x, others_mask, y = self.process_data_test(X)

        maska1 = others_mask
        maskq = one_mask

        q = self.sentence_encoder(one_x)
        a1 = self.sentence_encoder(others_x)

        vq = self.getPooling(q, maskq)
        va = self.getPooling(a1, maska1)

        zero_protos = torch.zeros_like(vq)

        protos = torch.cat([zero_protos.unsqueeze(1), vq.unsqueeze(1)], dim=1)

        logits = -self.__batch_dist__(protos, va.unsqueeze(1)).squeeze(1)

        loss = self.celoss(logits, y)

        pred = torch.max(logits, dim=-1)[1].float()
        # print(logits)
        y = y.cpu().detach()
        pred = pred.cpu().detach()
        f1_val = metrics.f1_score(y, pred, average="macro")
        acc_val = torch.mean((pred.view(-1) == y.view(-1)).type(torch.FloatTensor))
        return loss, logits, acc_val, f1_val


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


class Config():
    def __init__(self):
        pass

    model = "ow_proto"
    optimizer = "adam"
    lr = 1e-3
    batch_size = 1
    nepochs = 120
    dataset = "huffpost"
    target_type = "single"  # used for ACd
    data_path = "../datasets/oneshot_huffpost/"
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


def train_epoch(model, data_loader, crit, optimizer, config):
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


def eval_epoch(model, data_loader, crit, config):
    model.eval()

    total_loss = []
    total_results_acc = []
    total_results_f1 = []

    for step, batch in enumerate(tqdm(data_loader, total=len(data_loader))):
        # sentiment loss
        loss, logits, acc, f1 = model.predict(batch)

        total_loss.append(loss.item())

        total_results_acc.append(acc.item())
        total_results_f1.append(f1.item())

    total_loss = np.array(total_loss)

    total_results_acc = np.mean(np.array(total_results_acc))
    total_results_f1 = np.mean(np.array(total_results_f1))

    return np.mean(total_loss), total_results_acc, total_results_f1


def train(model, crit, optimizer, train_data, valid_data, test_data, config):
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
        train_loss, train_acc, train_f1 = train_epoch(model, train_data_loader, crit, optimizer, config)
        train_msg = "train_loss: {:04.5f} train_acc: {:04.5f} train_f1: {:04.5f}".format(train_loss, train_acc,
                                                                                         train_f1)
        print(train_msg)

        print("deving evaluation...........")
        dev_loss, dev_acc, dev_f1 = eval_epoch(model, dev_data_loader, crit, config)
        dev_msg = "dev_loss: {:04.6f} dev_acc: {:04.6f} dev_f1: {:04.6f} ".format(dev_loss, dev_acc, dev_f1)
        print(dev_msg)

        print("testing evaluation...........")
        test_loss, test_acc, test_f1 = eval_epoch(model, test_data_loader, crit, config)
        test_msg = "test_loss: {:04.6f} test_acc: {:04.6f} test_f1: {:04.6f} ".format(test_loss, test_acc, test_f1)
        print(test_msg)

        model_name = "results_" + str(config.seed) + "/" + "model.pkl"
        if dev_f1 >= max_score:
            no_imprv = 0
            torch.save(model.state_dict(), model_name)
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
    if config.model == "ow_proto":
        model = OW_Proto(config, sentence_encoder)

    crit = nn.CrossEntropyLoss(size_average=True)

    if torch.cuda.is_available():
        print("cuda available !!!!!!!!!!!")
        model = model.cuda(device)
        crit = crit.cuda(device)

    para_list = filter(lambda p: p.requires_grad, model.parameters())
    if config.optimizer == "sgd":
        optimizer = optim.SGD(para_list, lr=config.lr, momentum=0.9)
    elif config.optimizer == "adam":
        optimizer = optim.Adam(para_list, lr=config.lr)
    elif config.optimizer == "adadelta":
        optimizer = optim.Adadelta(para_list, lr=config.lr)
    train(model, crit, optimizer, train_data, valid_data, test_data, config)


if __name__ == '__main__':
    main()