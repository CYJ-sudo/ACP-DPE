import pickle
import random
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

warnings.filterwarnings('ignore')

def Model_Evaluate(confus_matrix):
    TN, FP, FN, TP = confus_matrix.ravel()

    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    ACC = (TP + TN) / (TP + TN + FN + FP)
    MCC = ((TP * TN) - (FP * FN)) / (np.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN)))
    Pre = TP / (TP + FP)

    return SN, SP, ACC, MCC, Pre

def cal_score(pred, label):
    try:
        AUC = roc_auc_score(list(label), pred)
    except:
        AUC = 0

    pred = np.around(pred)
    label = np.array(label)

    confus_matrix = confusion_matrix(label, pred, labels=None, sample_weight=None)
    SN, SP, ACC, MCC, Pre = Model_Evaluate(confus_matrix)
    print("Model score --- SN:{0:.4f}       SP:{1:.4f}       ACC:{2:.4f}       MCC:{3:.4f}      Pre:{4:.4f}   AUC:{5:.4f}".format(SN, SP, ACC, MCC, Pre, AUC))

    return ACC

class MyDataset(Dataset):
    def __init__(self, features1, features2, labels):
        self.features1, self.features2 = features1, features2
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        feature1 = self.features1[index]
        feature2 = self.features2[index]
        label = self.labels[index]
        return feature1, feature2, label

def load_data_bicoding(file_path):
    df = pd.read_csv(file_path)
    sequences = df['Sequence'].tolist()
    labels = df['Label'].tolist()
    data_result = []
    for seq in sequences:
        seq = str(seq.strip('\n'))
        data_result.append(list(seq))

    data_result = np.array(data_result)

    return data_result, labels

def numerical_transform(file_path, length=50):
    df = pd.read_csv(file_path)
    sequences = df['Sequence'].tolist()
    labels = df['Label'].tolist()
    amino_acids = "XACDEFGHIKLMNPQRSTVWY"
    aa2int = dict((c, i) for i, c in enumerate(amino_acids))
    encoding = []
    for sequence in sequences:
        if len(sequence) >= length:
            encoding.append([aa2int[aa] for aa in sequence[:length]])
        if len(sequence) < length:
            seq = [aa2int[aa] for aa in sequence]
            for k in range(length - len(sequence)):
                seq.append(0)
            encoding.append(seq)

    return encoding, labels

def transform_token2index(sequences):
    token2index = pickle.load(open('residue2idx.pkl', 'rb'))

    for i, seq in enumerate(sequences):
        sequences[i] = list(seq)

    token_list = list()
    max_len = 0
    for seq in sequences:
        seq_id = [token2index[residue] for residue in seq]
        token_list.append(seq_id)
        if len(seq) > max_len:
            max_len = len(seq)

    return token_list, max_len

def make_data_with_unified_length(token_list, max_len):
    token2index = pickle.load(open('residue2idx.pkl', 'rb'))
    data = []
    for i in range(len(token_list)):
        token_list[i] = [token2index['[CLS]']] + token_list[i] + [token2index['[SEP]']]
        if len(token_list[i]) < max_len:
            n_pad = max_len - len(token_list[i])
            token_list[i].extend([0] * n_pad)
        else:
            token_list[i] = token_list[i][:max_len]
        data.append(token_list[i])

    return data

def load_test_bicoding(path_data):

    sequences, labels = load_data_bicoding(path_data)
    token_list, max_len = transform_token2index(sequences)
    data_train = make_data_with_unified_length(token_list, max_len)

    # np.random.seed(42)
    # np.random.shuffle(data_train)

    X_test = np.array(data_train)
    y_test = np.array(labels)

    return X_test, y_test

class Embedding(nn.Module):
    def __init__(self, vocab_size, d_model, max_len):
        super(Embedding, self).__init__()
        self.tok_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)

        embedding = self.pos_embed(pos)
        embedding = embedding + self.tok_embed(x)
        embedding = self.norm(embedding)
        return embedding

class CNN_gru(nn.Module):
    def __init__(self):
        super(CNN_gru, self).__init__()
        max_len = 50
        d_model = 64
        vocab_size = 28

        self.embed1 = nn.Embedding(21, 64)

        self.gru = nn.GRU(input_size=64, hidden_size=16, bidirectional=True, batch_first=True)

        self.embed2 = Embedding(vocab_size, d_model, max_len)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=4, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(256))

        self.conv2 = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=256, kernel_size=4, dilation=3),
            nn.ReLU(),
            nn.BatchNorm1d(256))

        self.adaptpool = nn.AdaptiveAvgPool1d(4)

        self.output1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Flatten(),
            nn.Linear(1600, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
            nn.Linear(128, 16),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

        self.output2 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(1024, 16),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x1, x2):
        embed_out1 = self.embed1(x1)
        gru_out, _ = self.gru(embed_out1)  # [64, 50, 32]
        output1 = self.output1(gru_out)

        embed_out2 = self.embed2(x2)
        embed_out2 = embed_out2.permute(0, 2, 1)
        cnn_out1 = self.conv1(embed_out2)
        cnn_out2 = self.conv2(embed_out2)

        cnn_out = torch.cat([cnn_out1, cnn_out2], dim=2)
        cnn_out = self.adaptpool(cnn_out)

        output2 = self.output2(cnn_out)

        return output1, output2

if __name__ == '__main__':

    test_path = 'dataset/ACP20mainTest.csv'

    x_test1, y_test1 = numerical_transform(test_path)
    x_test1 = np.array(x_test1)

    x_test2, y_test2 = load_test_bicoding(test_path)

    test_dataset = MyDataset(x_test1, x_test2, y_test1)

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=True)

    model = CNN_gru()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=9e-03)
    criterion = nn.BCELoss()

    model.load_state_dict(torch.load('model/main_model.pt')) 

    model.eval()

    predictions = []
    true_labels = []
    total_loss = 0

    with torch.no_grad():
        for inputs1, inputs2, labels in test_loader:
            inputs1 = inputs1.long()
            inputs2 = inputs2.long()
            labels = labels.float()

            gru_out, cnn_out = model(inputs1, inputs2)
            outputs = 0.5 * gru_out.squeeze() + 0.5 * cnn_out.squeeze()

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            predictions.extend(outputs.cpu().detach().numpy())
            true_labels.extend(labels.cpu().detach().numpy())

    accuracy = cal_score(predictions, true_labels)
    # test_loss = total_loss / len(test_loader)

    print('Accuracy on the independent test set: {:.2f}%'.format(accuracy * 100))

