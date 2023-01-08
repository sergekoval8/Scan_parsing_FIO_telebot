#model
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import torch
from tqdm import tqdm, tqdm_notebook
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# general
import cv2
import os
import string
import numpy as np
from collections import Counter
from itertools import groupby

IMG_H, IMG_W = (32, 340)
MEAN_IMG = 0.894
STD_IMG = 0.3037
CHARS = list(' !"()*,-.0123456789:№;<>?[]ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё')
N_CLASSES = len(CHARS) + 1
BLANK = 0

DEVICE = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
if not torch.cuda.is_available():
    print("А где же ГПУ?")


class ImgTextData(Dataset):
    def __init__(self, data_dir, text_files, img_files, ind_labels, text_size=25, mode='val'):
        super().__init__()
        self.img_files = img_files
        self.text_files = text_files
        self.ind_labels = ind_labels
        self.len_ = len(self.text_files)
        self.mode = mode
        self.data_dir = data_dir
        self.text_size = text_size
        self.texts = self.get_texts()
        # self.images = self.get_images()

    def __len__(self):
        return self.len_

    def tokenize_text(self, txt):
        target_length = torch.IntTensor([len(txt)])
        y = [self.ind_labels.index(ch) + 1 for ch in txt] + [BLANK] * (self.text_size - target_length)
        return torch.LongTensor(y), target_length

    def get_texts(self):
        tokenized_texts = []
        for txt_file in self.text_files:
            with open(self.data_dir + txt_file) as f:
                txt = f.read().strip()
            tokenized_texts.append((self.tokenize_text(txt)))
        return tokenized_texts

    def get_images(self):
        return [cv2.imread(self.data_dir + img_file, 0) for img_file in self.img_files]

    def __getitem__(self, index):
        img = cv2.imread(self.data_dir + self.img_files[index], 0)
        # img = self.images[index]
        bg = img[:, -3:].mean()
        transform_tr = transforms.Compose([
                                           transforms.RandomAffine(4, scale=(0.8, 1.03), fill=bg),
                                           transforms.RandomApply([
                                                                    transforms.GaussianBlur(kernel_size=5, sigma=0.8),
                                                                    transforms.GaussianBlur(kernel_size=3, sigma=0.5),
                                                                    transforms.GaussianBlur(kernel_size=3, sigma=1.)
                                                                    ])
                                          ])
        if self.mode == 'train':
            img = transforms.ToPILImage()(img)
            img = transform_tr(img)

        x = transforms.ToTensor()(img)
        # x = transforms.Normalize(mean=[MEAN_IMG], std=[STD_IMG])(x)

        # with open(self.data_dir + self.text_files[index]) as f:
        #     txt = f.read().strip()

        # y, target_length = self.tokenize_text(txt)
        y, target_length = self.texts[index]
        return x, y, target_length


class ImgTextUpperData(ImgTextData):
    def __init__(self, data_dir, text_files, img_files, ind_labels, text_size=25, mode='val'):
        super().__init__(data_dir, text_files, img_files, ind_labels, text_size, mode)

    def get_texts(self):
        tokenized_texts = []
        for txt_file in self.text_files:
            with open(self.data_dir + txt_file) as f:
                txt = f.read().strip().upper()
            tokenized_texts.append((self.tokenize_text(txt)))
        return tokenized_texts

#################################################### Models ############################################################
#                                  Base Model


class CRNN_PRO(nn.Module):
    """ base class, is used to provide "predict" and "recognize" methods in child`s classes"""
    def __init__(self):
        super().__init__()

    @torch.no_grad()
    def predict_tokens(self, y_pred):
        _, max_index = torch.max(y_pred, dim=-1)                                # max_index.shape == torch.Size([B, T]
        max_index = max_index.detach().cpu().numpy()
        dif = np.ones(max_index.shape, max_index.dtype)
        dif[:, 1:] = max_index[:, 1:] - max_index[:, :-1]

        for i, raw_prediction in enumerate(max_index):                          # len(raw_prediction) == T
            idx = dif[i].nonzero()
            pred = raw_prediction[idx] - self.BLANK
            yield pred[pred.nonzero()]

    @torch.no_grad()
    def predict_text(self, x):
        y_pred = self.forward(x)
        predictions = self.predict_tokens(y_pred)
        texts = [''.join([self.CHARS[c - 1] for c in pred]) for pred in predictions]

        log_conf_char = y_pred.detach().cpu().numpy().max(axis=-1)
        log_conf_word = log_conf_char.min(axis=-1)
        return texts, np.exp(log_conf_word)

    @torch.no_grad()
    def recognize(self, img_file_name):
        # ! batch_size = 1
        im0 = cv2.imread(img_file_name, 0)
        # 0. binarize -?
        # im0 = cv2.threshold(im0, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        # 1.1. rotate if need
        h, w = im0.shape
        if h > w:
            im0 = np.rot90(im0)

        # 1.2. adjust text size to chunk (IMG_H, IMG_W)
        text = ""
        conf = 1
        bg_color = np.uint8(np.median(im0[:, -3:]))

        while im0 is not None and (h < w):
            # 1.3. resize and fill with whitespaces
            scale_factor = self.IMG_H / h
            w_opt = int(round(self.IMG_W / scale_factor, 0))
            if w <= 1.1 * w_opt:
                img = im0.copy()
                im0 = None                                      # last iteration with long image or alone iteration with short
                if w < 0.95 * w_opt:
                    img = np.hstack((img, np.ones((h, w_opt - w), np.uint8) * bg_color))
            else:
                img = im0[:, :w_opt + 1]
                im0 = im0[:, w_opt - 1:]          # some intersection between chunks (2 pixel)
                h, w = im0.shape

            # 1.4. word, conf_word = self.process_chunk(img)
            img = cv2.resize(img, (self.IMG_W, self.IMG_H), interpolation=cv2.INTER_AREA)
            # 1.5. image to tensor
            x = transforms.ToTensor()(img)                              # x = [1, IMG_H, IMG_W]
            x = transforms.Normalize(mean=[MEAN_IMG], std=[STD_IMG])(x)
            # 2. recognize and decode, compute confidence
            text_chunk, conf_chunk = self.predict_text(x.unsqueeze(0))  # x.unsqueeze(0) = [1, 1, IMG_H, IMG_W]
            text += text_chunk[0]
            conf = min(conf, conf_chunk[0])

        return text, conf

####################################################### Model1 #######################################################


class CRNN(CRNN_PRO):
    def __init__(self, chars, blank=0, img_h=32, img_w=340, gru_hidden_size=128, gru_num_layers=2, dropout=0.1):
        super().__init__()
        self.IMG_H = img_h
        self.IMG_W = img_w
        self.CHARS = chars
        self.BLANK = blank
        self.num_classes = len(self.CHARS) + 1
        self.gru_input_size = (IMG_H // 4 - 3) * 64
        self.cnn_output_width = IMG_W // 4 - 3
        self.conv = nn.Sequential(
                                  nn.Conv2d(1, 32, kernel_size=(3, 3)),
                                  nn.InstanceNorm2d(32),
                                  nn.LeakyReLU(),

                                  nn.Conv2d(32, 32, kernel_size=(3, 3)),
                                  nn.MaxPool2d((2, 2)),
                                  nn.InstanceNorm2d(32),
                                  nn.LeakyReLU(),

                                  nn.Conv2d(32, 64, kernel_size=(3, 3)),
                                  nn.InstanceNorm2d(64),
                                  nn.LeakyReLU(),

                                  nn.Conv2d(64, 64, kernel_size=(3, 3)),
                                  nn.MaxPool2d((2, 2)),
                                  nn.InstanceNorm2d(64),
                                  nn.LeakyReLU()
                                 )

        self.gru = nn.GRU(self.gru_input_size, gru_hidden_size, gru_num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

        self.classifier = nn.Sequential(
                                nn.Linear(gru_hidden_size * 2, gru_hidden_size),
                                nn.BatchNorm1d(self.cnn_output_width),
                                nn.ReLU(),
                                nn.Linear(gru_hidden_size, self.num_classes)
                               )

    def forward(self, x):
        batch_size = x.shape[0]
        out = self.conv(x)
        out = out.view(batch_size, -1, out.shape[-1])  # [batch_size, cnn_output_height * channels_out, T]
        out = out.transpose(2, 1)                      # [batch_size, T, cnn_output_height * channels_out]
        out, _ = self.gru(out)                         # [batch_size, T, N_CLASSES]
        return F.log_softmax(self.classifier(out), dim=-1)

################################################# MODEL2 ##############################################################
# class CNNLayerNorm(nn.Module):
#     def __init__(self, n_feats):
#         super().__init__()
#         self.layer_norm = nn.LayerNorm(n_feats)

#     def forward(self, x):                      # x.shape = (batch, channel, feature, time)
#         x = x.transpose(2, 3).contiguous()     # (batch, channel, time, feature)
#         x = self.layer_norm(x)
#         return x.transpose(2, 3).contiguous()  # (batch, channel, feature, time)


class CNNMaxPool2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, pool_type='max'):
        super().__init__()
        assert pool_type in ('avg', 'max')
        if pool_type == max:
            self.layer = nn.Sequential(
                                       nn.Conv2d(in_channels, out_channels, kernel, padding=kernel//2),
                                       nn.MaxPool2d((2, 2)),
                                       nn.InstanceNorm2d(out_channels),
                                       nn.LeakyReLU()
                                       )
        else:
            self.layer = nn.Sequential(
                                       nn.Conv2d(in_channels, out_channels, kernel, padding=kernel//2),
                                       nn.AvgPool2d((2, 2)),
                                       nn.InstanceNorm2d(out_channels),
                                       nn.LeakyReLU()
                                       )

    def forward(self, x):
        return self.layer(x)


class ResidualCNN(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, dropout):
        super().__init__()
        self.layer = nn.Sequential(
                                   nn.Conv2d(in_channels, out_channels, kernel, padding=kernel//2),      #stride=1
                                   nn.InstanceNorm2d(out_channels),
                                   nn.LeakyReLU(),
                                   nn.Dropout(dropout)
                                   )

    def forward(self, x):
        out = self.layer(x)
        return x + out        # (batch, channel, feature, time)


class CNNComboPool(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0.1):
        super().__init__()

        self.rescnn1 = ResidualCNN(in_channels, in_channels, kernel=3, dropout=dropout)
        self.rescnn2 = ResidualCNN(in_channels, in_channels, kernel=5, dropout=dropout)
        self.cnn1 = CNNMaxPool2d(in_channels * 2, out_channels // 2, kernel=3, pool_type='max')
        self.cnn2 = CNNMaxPool2d(in_channels * 2, out_channels // 2, kernel=3, pool_type='avg')

    def forward(self, x):
        x1 = self.rescnn1(x)
        x2 = self.rescnn1(x)
        x = torch.cat([x1, x2], dim=1)

        x1 = self.cnn1(x)
        x2 = self.cnn2(x)
        x = torch.cat([x1, x2], dim=1)
        return x


class CRNN2(CRNN_PRO):
    def __init__(self, chars, blank=0, img_h=32, img_w=340, gru_hidden_size=128, gru_num_layers=2, dropout=0.1):
        super().__init__()
        self.IMG_H = img_h
        self.IMG_W = img_w
        self.CHARS = chars
        self.BLANK = blank
        self.num_classes = len(self.CHARS) + 1

        self.cnn_output_width = IMG_W // 8           # 340 // 8 = 42
        gru_input_size = 128 * IMG_H // 8            # 128 * 32 // 8 = 512

        self.cnn0 = CNNMaxPool2d(in_channels=1, out_channels=32, kernel=3, pool_type='max')
        self.cnn1 = CNNComboPool(32, 64, dropout=dropout)
        self.cnn2 = CNNComboPool(64, 128, dropout=dropout)

        self.gru = nn.GRU(gru_input_size, gru_hidden_size, gru_num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

        self.classifier = nn.Sequential(nn.Linear(gru_hidden_size * 2, gru_hidden_size),
                                        nn.InstanceNorm2d(self.cnn_output_width),
                                        nn.ReLU(),
                                        nn.Linear(gru_hidden_size, self.num_classes)
                                       )

    def forward(self, x):
        x = self.cnn0(x)                           # [B, 1, IMG_H, IMG_W] -> [B, 32, IMG_H // 2, IMG_W //2]
        x = self.cnn1(x)                           # [B, 32, IMG_H // 2, IMG_W //2] -> [B, 64, IMG_H // 4, IMG_W //4]
        x = self.cnn2(x)                           # [B, 64, IMG_H // 4, IMG_W //4] -> [B, 128, IMG_H // 8, IMG_W //8]

        x = x.view(x.shape[0], -1, x.shape[-1])    # [batch_size, cnn_output_height * channels_out, T]
        x = x.transpose(2, 1)                      # [batch_size, T, cnn_output_height * channels_out]
        x, _ = self.gru(x)
        x = self.classifier(x)
        return F.log_softmax(x, dim=-1)

########################################## MODEL 3 #####################################################################


class CRNN3(CRNN_PRO):
    def __init__(self, chars, blank=0, img_h=32, img_w=340, gru_hidden_size=128, gru_num_layers=2, dropout=0.1):
        super().__init__()
        self.IMG_H = img_h
        self.IMG_W = img_w
        self.CHARS = chars
        self.BLANK = blank
        self.num_classes = len(self.CHARS) + 1

        self.gru_input_size = (IMG_H // 4 - 3) * 64
        self.cnn_output_width = IMG_W // 4 - 3
        self.conv0 = nn.Sequential(
                                   nn.Conv2d(1, 32, kernel_size=(3, 3)),
                                   nn.BatchNorm2d(32),
                                   nn.LeakyReLU(),
                                   nn.Conv2d(32, 32, kernel_size=(3, 3)),
                                  )

        self.pool01 = nn.MaxPool2d((2, 2))
        self.pool02 = nn.AvgPool2d((2, 2))

        self.conv1 = nn.Sequential(
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(64, 64, kernel_size=(3, 3)),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU(),

                                    nn.Conv2d(64, 64, kernel_size=(3, 3)),
                                    nn.MaxPool2d((2, 2)),
                                    nn.BatchNorm2d(64),
                                    nn.LeakyReLU()
                                  )

        self.gru = nn.GRU(self.gru_input_size, gru_hidden_size, gru_num_layers,
                          batch_first=True, bidirectional=True, dropout=dropout)

        self.classifier = nn.Sequential(
                                nn.Linear(gru_hidden_size * 2, gru_hidden_size),
                                nn.BatchNorm1d(self.cnn_output_width),
                                nn.ReLU(),
                                nn.Linear(gru_hidden_size, self.num_classes)
                               )

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.conv0(x)
        x01 = self.pool01(x)
        x02 = self.pool02(x)
        x = torch.cat([x01, x02], dim=1)
        x = self.conv1(x)
        x = x.view(batch_size, -1, x.shape[-1])    # [batch_size, cnn_output_height * channels_out, T]
        x = x.transpose(2, 1)                      # [batch_size, T, cnn_output_height * channels_out]
        x, _ = self.gru(x)                         # [batch_size, T, N_CLASSES]
        return F.log_softmax(self.classifier(x), dim=-1)

##################################################################################################################
# train, eval


@torch.no_grad()
def acc_prediction(model, y_target, y_pred):
    y_target = y_target.numpy() - BLANK
    predictions = model.predict_tokens(y_pred)
    correct = 0

    for i, pred in enumerate(predictions):
        target = y_target[i]
        target = target[target.nonzero()]                  # in target BLANK only at tail
        correct += np.array_equal(pred, target)

    return correct / y_target.shape[0]


def train_epoh(model, criterion, train_loader, scheduler, optimizer):
    model.train()
    loss_tr = 0
    acc_tr = 0

    for x_train, y_train,  target_lengths in tqdm(train_loader):
        x_train = x_train.to(DEVICE)                      # .view(x_train.shape[0], 1, x_train.shape[1], x_train.shape[2])
        input_lengths = torch.IntTensor(y_train.shape[0]).fill_(model.cnn_output_width)

        optimizer.zero_grad()
        y_pred = model(x_train)                            # y_pred.shape == torch.Size([B, T, N_CASSES])
        loss = criterion(y_pred.permute(1, 0, 2), y_train, input_lengths, target_lengths)   # y_pred.shape : permute for loss_function
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        if scheduler:
            scheduler.step()

        loss_tr += loss.item()
        acc_tr += acc_prediction(model, y_train, y_pred)    # not more then 5% time of train epoch

    return loss_tr / len(train_loader), acc_tr / len(train_loader)


@torch.no_grad()
def eval_epoh(model, criterion, val_loader):
    acc_val = 0
    loss_val = 0

    for x_val, y_val, target_lengths in tqdm(val_loader):
        x_val = x_val.to(DEVICE)
        y_pred = model(x_val)

        input_lengths = torch.IntTensor(y_val.shape[0]).fill_(model.cnn_output_width)
        loss = criterion(y_pred.permute(1, 0, 2), y_val, input_lengths, target_lengths)

        loss_val += loss.item()
        acc_val += acc_prediction(model, y_val, y_pred)

    return loss_val / len(val_loader), acc_val / len(val_loader)


def train(model, epochs, results, scheduler, optimizer, train_loader, val_loader, n_patience, path):
    best_acc = results["val_acc"][-1] if results["val_acc"] else 0.
    cnt_bad = 0
    criterion = nn.CTCLoss(blank=BLANK, reduction='mean', zero_infinity=True)

    for k in range(epochs):
        lr = optimizer.state_dict()['param_groups'][0]['lr']
        print(f"before {k + 1} epoch: lr= {lr}")

        train_loss, train_acc = train_epoh(model, criterion, train_loader, scheduler, optimizer)
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)

        val_loss, val_acc = eval_epoh(model, criterion, val_loader)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)
        results["lr"].append(lr)

        print(f"epoh= {k + 1:2}, train_acc={train_acc:.3f}, val_acc={val_acc:.3f} || train_loss= {train_loss:.4f}, val_loss= {val_loss:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            model = model.cpu()
            torch.save(model.state_dict(), path)
            model = model.to(DEVICE)
            cnt_bad = 0
        elif cnt_bad == n_patience:
            print(f"last {n_patience} epochs has no any advantages. Aborted.")
            break
        else:
            cnt_bad += 1


def test(model, test_loader):
    model.eval()
    acc = 0

    for x_val, y_val, target_lengths in tqdm(test_loader):
        y_pred = model(x_val.to(DEVICE))
        acc += acc_prediction(model, y_val, y_pred)
    print("test_accuracy:", round(acc / len(test_loader), 4))


if __name__ == '__main__':
    from torchsummary import summary

    model_version = 3  # (1, 2, 3)
    model_kwargs = dict(IMG_H=IMG_H, IMG_W=IMG_W, gru_hidden_size=128, gru_num_layers=2, num_classes=94)
    models = {1: CRNN(**model_kwargs),
              2: CRNN2(**model_kwargs, dropout=0.35),
              3: CRNN3(**model_kwargs, dropout=0.5)
              }

    model = models[model_version].to(DEVICE)
    z = torch.Tensor(np.zeros((32, 340))).unsqueeze(0).unsqueeze(0).to(DEVICE)
    outp = model(z)
    summary(model, z.squeeze(0).shape, batch_size=64)
    print("outp.shape:", outp.shape, "model.gru_input_size:", model.gru_input_size)

    #######################################################################
