from utils import *
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Seq_len, X_train = read_txt('./data/data.txt')
Embedding_matrix = pretrained_embedding_layer(w2v_m, w2i)
vocab_size = len(w2i)
n_hidden = 16
embedding_dim = 300

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

Train_DS = LM_Dataset(X_train, seq_len)
Train_DL = DataLoader(Train_DS, batch_size=1, shuffle=True)


class BiLSTM(nn.Module):
    def __init__(self, vocab_size, n_hidden, embedding_dim):
        super(BiLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding.from_pretrained(Embedding_matrix, freeze=True)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=n_hidden, bidirectional=True)
        self.softmax = nn.Linear(n_hidden * 2, vocab_size)

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
        outputs = self.softmax(outputs)
        outputs = outputs.view(1, outputs.shape[1], outputs.shape[0])

        return outputs

    def _init_weights(self, layer) -> None:
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)


model = BiLSTM(vocab_size, n_hidden, embedding_dim).to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2)

def train(model, loss_fn, optimizer, epochs):
    print("Train Start")
    for e in range(1, epochs + 1):
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

        if e % 10 == 0:
            print(f'{"-" * 20} Epoch {e} {"-" * 20}')
            print("loss is :", loss)

train(model, loss_fn, optimizer, epochs=10000)