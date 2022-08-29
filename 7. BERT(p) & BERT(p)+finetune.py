import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import os
import pickle
from sklearn import metrics
import numpy as np
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertOnlyMLMHead
import torch.nn.init as init
from transformers import BertTokenizer, AutoConfig

import warnings

warnings.filterwarnings("ignore")


class BERTSentencePair(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.max_length = self.config.bert_max_length
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        #self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert = BertModel(self.config)

        self.celoss = nn.CrossEntropyLoss(reduction="mean")
        #self.pred_fc1 = nn.Linear(768, 2)
        self.cls = BertOnlyMLMHead(self.config)
        self.init_weights()
        self.device_my = torch.device("cuda:" + str(self.config.device) if torch.cuda.is_available() else "cpu")

    def process_data(self, X, is_tuning):
        total = []
        total_y = []
        total_y_prompt = []
        lens = []
        lens_mask = []

        #cls = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
        #sep = self.tokenizer.convert_tokens_to_ids(['[SEP]'])
        #append = self.tokenizer.convert_tokens_to_ids(["It", "was", "[MASK]"])
        wenhao = self.tokenizer.convert_tokens_to_ids(["?"])
        token_mask = self.tokenizer.convert_tokens_to_ids(["[MASK]"])
        comma = self.tokenizer.convert_tokens_to_ids([","])
        yes = self.tokenizer.convert_tokens_to_ids(["yes"])[0]
        no = self.tokenizer.convert_tokens_to_ids(['no'])[0]

        for each in X:
            o, s, t = each
            # length = len(s[0]) + len(t[0])
            # one_shot.extend(numpy.repeat(o[0], length, axis=0))

            s_x, s_y = s
            t_x, t_y = t
            o = o[0]

            for i, each_s in enumerate(s_x):
                cur_id = o[0] + wenhao + token_mask
                mask_length = len(cur_id)
                cur_id = cur_id + comma + each_s
                # print(len(o[0]), len(each_s), len(cur_id))
                bert_id = np.zeros([self.max_length], dtype=np.int64)

                if len(cur_id) > self.max_length:
                    print(len(cur_id), self.max_length)
                    cur_id = cur_id[0: self.max_length]
                    exit()

                bert_id[0: len(cur_id)] = cur_id

                total.append(bert_id)
                lens.append(len(cur_id))
                lens_mask.append(mask_length)
                total_y.append(s_y[i])
                if s_y[i] == 1:
                    total_y_prompt.append(yes)
                elif s_y[i] == 0:
                    total_y_prompt.append(no)
                else:
                    print("error")

            if not is_tuning:
                for i, each_t in enumerate(t_x):
                    cur_id = o[0] + wenhao + token_mask
                    mask_length = len(cur_id)
                    cur_id = cur_id + comma + each_s
                    bert_id = np.zeros([self.max_length], dtype=np.int64)

                    if len(cur_id) > self.max_length:
                        cur_id = cur_id[0: self.max_length]
                        print(len(cur_id), self.max_length)
                        exit()

                    bert_id[0: len(cur_id)] = cur_id

                    total.append(bert_id)
                    lens.append(len(cur_id))
                    lens_mask.append(mask_length)
                    total_y.append(t_y[i])

                    if t_y[i] == 1:
                        total_y_prompt.append(yes)
                    elif t_y[i] == 0:
                        total_y_prompt.append(no)
                    else:
                        print("error")

        total = torch.LongTensor(total).to(self.device_my)
        total_y = torch.LongTensor(total_y).to(self.device_my)
        total_y_prompt = torch.LongTensor(total_y_prompt).to(self.device_my)
        lens = torch.LongTensor(lens).to(self.device_my)
        lens_mask = torch.LongTensor(lens_mask).to(self.device_my)

        # get mask
        n = total.size()[0]
        idxes = torch.arange(self.max_length).expand(n, -1).to(self.device_my)
        mask = (idxes < lens.unsqueeze(1)).float()
        #mask_last = (idxes == (lens.unsqueeze(1)-1)).float()
        mask_last = lens_mask-1

        return total, mask, mask_last, total_y, total_y_prompt

    def process_data_test(self, X):
        total = []
        total_y = []
        total_y_prompt = []
        lens = []
        lens_mask = []

        #cls = self.tokenizer.convert_tokens_to_ids(['[CLS]'])
        #sep = self.tokenizer.convert_tokens_to_ids(['[SEP]'])
        #append = self.tokenizer.convert_tokens_to_ids(["It", "was", "[MASK]"])
        wenhao = self.tokenizer.convert_tokens_to_ids(["?"])
        token_mask = self.tokenizer.convert_tokens_to_ids(["[MASK]"])
        comma = self.tokenizer.convert_tokens_to_ids([","])
        yes = self.tokenizer.convert_tokens_to_ids(["yes"])[0]
        no = self.tokenizer.convert_tokens_to_ids(['no'])[0]

        for each in X:
            o, _, t = each

            t_x, t_y = t
            o = o[0]

            for i, each_t in enumerate(t_x):
                cur_id = o[0] + wenhao + token_mask
                mask_length = len(cur_id)
                cur_id = cur_id + comma + each_t
                # print(len(o[0]), len(each_s), len(cur_id))
                bert_id = np.zeros([self.max_length], dtype=np.int64)

                if len(cur_id) > self.max_length:
                    print(len(cur_id), self.max_length)
                    cur_id = cur_id[0: self.max_length]
                    exit()

                bert_id[0: len(cur_id)] = cur_id

                total.append(bert_id)
                lens.append(len(cur_id))
                lens_mask.append(mask_length)
                total_y.append(t_y[i])

                if t_y[i] == 1:
                    total_y_prompt.append(yes)
                elif t_y[i] == 0:
                    total_y_prompt.append(no)
                else:
                    print("error")

        total = torch.LongTensor(total).to(self.device_my)
        total_y = torch.LongTensor(total_y).to(self.device_my)
        total_y_prompt = torch.LongTensor(total_y_prompt).to(self.device_my)
        lens = torch.LongTensor(lens).to(self.device_my)
        lens_mask = torch.LongTensor(lens_mask).to(self.device_my)

        # get mask
        n = total.size()[0]
        idxes = torch.arange(self.max_length).expand(n, -1).to(self.device_my)
        mask = (idxes < lens.unsqueeze(1)).float()
        mask_last = lens_mask - 1

        return total, mask, mask_last, total_y, total_y_prompt

    def forward(self, X, is_tuning=False):
        x, mask, mask_last, _, y_prompt = self.process_data(X, is_tuning)
        out = self.bert(x, attention_mask=mask)
        #out = out['pooler_output']
        out = out['last_hidden_state']
        out = out[torch.arange(out.size(0)), mask_last]
        out_feature = out

        out = self.cls(out)

        loss = self.celoss(out, y_prompt)

        pred = torch.max(out, dim=-1)[1].float().cpu()
        y_prompt = y_prompt.cpu()
        f1_val = metrics.f1_score(y_prompt, pred, average="macro")
        acc_val = torch.mean((pred.view(-1) == y_prompt.view(-1)).type(torch.FloatTensor))
        return loss, out_feature, acc_val, f1_val

    def predict(self, X):
        x, mask, mask_last, _, y_prompt = self.process_data_test(X)
        out = self.bert(x, attention_mask=mask)
        out = out['last_hidden_state']
        out = out[torch.arange(out.size(0)), mask_last]
        out_feature = out

        out = self.cls(out)

        loss = self.celoss(out, y_prompt)

        pred = torch.max(out, dim=-1)[1].float().cpu()
        y_prompt = y_prompt.cpu()
        f1_val = metrics.f1_score(y_prompt, pred, average="macro")
        acc_val = torch.mean((pred.view(-1) == y_prompt.view(-1)).type(torch.FloatTensor))
        return loss, out_feature, acc_val, f1_val, y_prompt


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

    model = "bert"
    optimizer = "adam"
    lr = 2e-5
    batch_size = 1
    nepochs = 5
    nepoch_no_imprv = 3
    device = 0
    train_epochs = 400
    dev_epochs = 300
    test_epochs = 300
    classes_per_set = 5
    samples_per_class = 10
    dim_word = 768
    hidden_size=768
    seed = 20
    initializer_range = 0.1
    margin = 0.2
    one_shot = 1

    bert_max_length = 200

    target_type = "single" # used for ACD
    dataset = "huffpost"
    data_path = "../datasets/oneshot_huffpost/"
    num_eval_updates = 3
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
        with torch.no_grad():
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
        optim = torch.optim.Adam(net_clone.parameters(), lr=config.lr)
        for _ in range(config.num_eval_updates):
            loss_first, _, _, _ = net_clone(batch, is_tuning=True)

            optim.zero_grad()
            loss_first.backward()
            optim.step()

        # net_clone.eval()
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
    saved_model = copy.deepcopy(model)
    torch.save(model.state_dict(), model_name)

    print("-------------------------------------------")
    print("---- fining tuning second time ------------")
    # loader = data_loader(test_data, config.test_epochs, config)
    # fine-tune the second time, using the support set of novel classes
    acc, f1, _, _ = eval_epoch_finetune(saved_model, test_data_loader, config)
    print("acc ", acc, " f1 ", f1)

    print("-------------- getting features -----------")
    if config.dataset == "acd":
        loader = data_loader(test_data, config.test_feature_epochs, config)
    else:
        loader = data_loader_huffpost(test_data, config.test_feature_epochs, config)
    features, labels = eval_epoch_feature(saved_model, loader, config)
    pickle.dump([features, labels], open('BERT_features.p', "wb"))
    _, _, features_tune, labels_tune = eval_epoch_finetune(saved_model, loader, config)
    pickle.dump([features_tune, labels_tune], open("BERT_features_finetuned.p", "wb"))
    print(len(features), len(features_tune))
    print("done!")


def main():
    dir_output = "results_" + str(config.seed) + "/"
    path_log = dir_output

    if not os.path.exists(dir_output):
        os.makedirs(dir_output)

    path = config.data_path + "data_info_bert.p"

    train_data, valid_data, test_data = pickle.load(open(path, "rb"))

    if config.model == "bert":

        config_bert = AutoConfig.from_pretrained("bert-base-uncased")
        config_bert.bert_max_length = config.bert_max_length
        config_bert.device = config.device

        model = BERTSentencePair(config_bert)
        model = model.from_pretrained('bert-base-uncased', config=config_bert)


    if torch.cuda.is_available():
        print("cuda available !!!!!!!!!!!")
        model = model.cuda(device)

    para_list = filter(lambda p: p.requires_grad, model.parameters())
    if config.optimizer == "adam":
        optimizer = optim.Adam(para_list, lr=config.lr)

    train(model, optimizer, train_data, valid_data, test_data, config)


if __name__ == '__main__':
    main()