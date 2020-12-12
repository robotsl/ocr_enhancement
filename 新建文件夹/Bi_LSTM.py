from utils import *
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(0)

torch.backends.cudnn.enabled = False
Seq_len, X_train = read_txt('./data/dataset.txt')
Embedding_matrix = pretrained_embedding_layer(w2v_m, w2i)
costs = []
vocab_size = len(w2i)
n_hidden = 128
embedding_dim = 300
learning_rate = 1e-3

class LM_Dataset(Dataset):
    def __init__(self, x, seq_len):
        super().__init__()
        self.x = x
        self.seq_len = seq_len

    def __getitem__(self, index):
        x, y = sentences_to_indices(self.x[index], w2i, self.seq_len[index])
        #y = convert_to_one_hot(y, len(w2i))
        return x, y

    def __len__(self):
        return self.x.shape[0]

Train_DS = LM_Dataset(X_train, Seq_len)
Train_DL = DataLoader(Train_DS, batch_size=1, shuffle=True)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, n_hidden, embedding_dim):
        super(BiLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding.from_pretrained(Embedding_matrix, freeze=True)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=n_hidden, bidirectional=True)
        self.linear = nn.Linear(n_hidden * 2, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def forward(self, X):

        input = self.word_embeddings(X) # input : [batch_size, len_seq, embedding_dim]
        input = input.to(torch.float32)
        input = input.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]

        #初始化h和c
        hidden_state = torch.zeros(1*2, len(X), self.n_hidden).to(device)   # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]
        cell_state = torch.zeros(1*2, len(X), self.n_hidden).to(device)     # [num_layers(=1) * num_directions(=2), batch_size, n_hidden]

        # output : [len_seq, batch_size, num_directions(=2)*n_hidden]
        outputs, (_, _) = self.lstm(input, (hidden_state, cell_state))
        outputs = outputs.view(-1, n_hidden * 2)
        outputs = self.linear(outputs)
        outputs = self.bn(outputs)
        outputs = self.relu(outputs)

        outputs = outputs.view(1, outputs.shape[1], outputs.shape[0])

        return outputs

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)


model = BiLSTM(vocab_size, n_hidden, embedding_dim).to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

def evaluate(model):
    model.eval()
    total_loss = 0.
    total_count = 0.
    with torch.no_grad():
        for line_num, (x, y) in enumerate(Train_DL):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                output = model(x)
            loss = loss_fn(output, y)
            total_count += np.multiply(*x.size())
            total_loss += loss.item() * np.multiply(*x.size())

    loss = total_loss / total_count
    return loss




def train(model, loss_fn, optimizer, epochs):
    perp_list = []

    print("Train Start")

    for e in range(1, epochs+1):
        for line_num, (x, y) in enumerate(Train_DL):
            model.train()
            loss = 0
            optimizer.zero_grad()

            x, y = x.to(device), y.to(device)
            z_pred = model(x)
            loss += loss_fn(z_pred, y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)  # gradient clipping
            optimizer.step()

        perp_list.append(evaluate(model))

        if e % 10 == 0:
            print(f'{"-" * 20} Epoch {e} {"-" * 20}')
            print("困惑度为:", perp_list[e-1])

    costs.append(perp_list)
    plt.plot(perp_list)
    plt.ylabel('cost')
    plt.xlabel('iterations')
    plt.title("Learning rate =" + str(learning_rate)+"; n_hidden ="+str(n_hidden))
    plt.show()

train(model, loss_fn, optimizer, epochs=150)
torch.save(model, './model/LM_128_withrelubn.pt')