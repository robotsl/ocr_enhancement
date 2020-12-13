from utils import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
Embedding_matrix = pretrained_embedding_layer(w2v_m, w2i)
n_hidden = 128
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, n_hidden, embedding_dim):
        super(BiLSTM, self).__init__()
        self.n_hidden = n_hidden
        self.embedding_dim = embedding_dim
        self.word_embeddings = nn.Embedding.from_pretrained(Embedding_matrix, freeze=True)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=n_hidden, bidirectional=True)
        self.softmax = nn.Linear(n_hidden * 2, vocab_size)
        self.bn = nn.BatchNorm1d(vocab_size)
        self.relu = nn.ReLU()
        self.apply(self._init_weights)

    def forward(self, X):

        input = self.word_embeddings(X) # input : [batch_size, len_seq, embedding_dim]
        input = input.to(torch.float32)
        input = input.permute(1, 0, 2) # input : [len_seq, batch_size, embedding_dim]

        #初始化 h 和 c
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