import torch
import torch.nn as nn
from transformers import BertModel


class BaseModel(nn.Module):
    """Base model class with a unified interface"""

    def __init__(self, **kwargs):
        super().__init__()
        self.embed_size = kwargs.get('embed_size')
        self.label_num = kwargs.get('label_num')
        self.label_class = kwargs.get('label_class')

    def forward(self, x):
        raise NotImplementedError


class RandomAttLinear(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab = kwargs['vocab']
        self.word2idx = kwargs['word2idx']

        self.embedding = nn.Embedding(len(self.vocab), self.embed_size, padding_idx=self.word2idx['<pad>'])
        self.linear_att = nn.Linear(self.embed_size, 128)
        self.tanh_att = nn.Tanh()
        self.u_w_att = nn.Linear(128, 1)
        self.softmax_att = nn.Softmax(dim=-1)
        self.linear = nn.Linear(self.embed_size, self.label_num * self.label_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x [batch_size, seq_len]
        embedding = self.embedding(x)
        # embedding [batch_size, seq_len, embed_size]
        attention_u = self.tanh_att(self.linear_att(embedding))
        # (batchsize, seq_len, embed_size) => (batchsize, seq_len, u_size)
        attention_a = self.softmax_att(self.u_w_att(attention_u))
        # (batchsize, seq_len, u_size) * (u_size, 1) => (batchsize, seq_len, 1)
        outputs = torch.matmul(attention_a.permute(0, 2, 1), embedding).squeeze()
        # (batchsize, 1, seq_len) * (batch_size, seq_len, embed_size) => (batchsize, embed_size)
        outputs = self.linear(outputs)
        # (batchsize, embed_size) => (batchsize, self.label_num * self.label_class)
        outputs = outputs.reshape((-1, self.label_num, self.label_class))
        # (batchsize, self.label_num * self.label_class) => (batchsize, self.label_num, self.label_class)
        logits = self.softmax(outputs)
        return logits


class RandomAverageLinear(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab = kwargs['vocab']
        self.word2idx = kwargs['word2idx']

        self.embedding = nn.Embedding(len(self.vocab), self.embed_size, padding_idx=self.word2idx['<pad>'])
        self.linear = nn.Linear(self.embed_size, self.label_num * self.label_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x [batch_size, seq_len]
        embedding = self.embedding(x)
        # embedding [batch_size, seq_len, embed_size]
        word_averaging = embedding.mean(dim=1)
        # word_averaging [batch_size, embed_size]
        output = self.linear(word_averaging)
        output = output.reshape((-1, self.label_num, self.label_class))
        # output = self.relu(output)
        # output = self.linear2(output)
        logits = self.softmax(output)
        return logits


class RandomCNNLinear(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab = kwargs['vocab']
        self.word2idx = kwargs['word2idx']
        self.num_of_filters = kwargs['num_of_filters']
        self.window_sizes = kwargs['window_sizes']
        self.maxlen = kwargs['maxlen']

        self.embedding = nn.Embedding(len(self.vocab), self.embed_size, padding_idx=self.word2idx['<pad>'])
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(in_channels=self.embed_size, out_channels=self.num_of_filters, kernel_size=h),
                           nn.ReLU(),
                           nn.MaxPool1d(self.maxlen - h + 1)
                           ) for h in self.window_sizes]
        )
        self.linear = nn.Linear(self.num_of_filters * len(self.window_sizes), self.label_num * self.label_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x [batch_size, seq_len]
        embedding = self.embedding(x)
        # embedding [batch_size, seq_len, embed_size]
        outputs = embedding.permute(0, 2, 1)
        outputs = [conv(outputs) for conv in self.convs]
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.view(-1, outputs.size(1))
        outputs = self.linear(outputs)
        outputs = outputs.reshape((-1, self.label_num, self.label_class))
        return self.softmax(outputs)


class RandomLSTMAttLinear(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab = kwargs['vocab']
        self.word2idx = kwargs['word2idx']
        self.hidden_size_lstm = kwargs['hidden_size_lstm']

        self.embedding = nn.Embedding(
            len(self.vocab), self.embed_size,
            padding_idx=self.word2idx['<pad>']
        )
        self.lstm = nn.LSTM(
            self.embed_size, self.hidden_size_lstm,
            bidirectional=True, batch_first=True
        )
        self.linear_att = nn.Linear(self.hidden_size_lstm * 2, 128)
        self.tanh_att = nn.Tanh()
        self.u_w_att = nn.Linear(128, 1)
        self.softmax_att = nn.Softmax(dim=-1)
        self.linear = nn.Linear(self.hidden_size_lstm * 2, self.label_num * self.label_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x [batch_size, seq_len]
        embedding = self.embedding(x)
        # embedding [batch_size, seq_len, embed_size]
        h0 = torch.randn(2, x.size(0), self.hidden_size_lstm).cuda()
        c0 = torch.randn(2, x.size(0), self.hidden_size_lstm).cuda()
        output, (hn, cn) = self.lstm(embedding, (h0, c0))
        attention_u = self.tanh_att(self.linear_att(output))
        # (batchsize, seq_len, embed_size) => (batchsize, seq_len, u_size)
        attention_a = self.softmax_att(self.u_w_att(attention_u))
        # (batchsize, seq_len, u_size) * (u_size, 1) => (batchsize, seq_len, 1)
        outputs = torch.matmul(attention_a.permute(0, 2, 1), output).squeeze()
        # (batchsize, 1, seq_len) * (batch_size, seq_len, embed_size) => (batchsize, embed_size)
        outputs = self.linear(outputs)
        # (batchsize, embed_size) => (batchsize, self.label_num * self.label_class)
        outputs = outputs.reshape((-1, self.label_num, self.label_class))
        # (batchsize, self.label_num * self.label_class) => (batchsize, self.label_num, self.label_class)
        logits = self.softmax(outputs)
        return logits


class RandomLSTMCNNLinear(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab = kwargs['vocab']
        self.word2idx = kwargs['word2idx']
        self.hidden_size_lstm = kwargs['hidden_size_lstm']
        self.num_of_filters = kwargs['num_of_filters']
        self.window_sizes = kwargs['window_sizes']
        self.maxlen = kwargs['maxlen']

        self.embedding = nn.Embedding(len(self.vocab), self.embed_size, padding_idx=self.word2idx['<pad>'])
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size_lstm, bidirectional=True, batch_first=True)
        self.convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels=self.embed_size + 2 * self.hidden_size_lstm, out_channels=self.num_of_filters,
                          kernel_size=h),
                nn.ReLU(),
                nn.MaxPool1d(self.maxlen - h + 1)
                ) for h in self.window_sizes]
        )

        self.linear = nn.Linear(self.num_of_filters * len(self.window_sizes), self.label_num * self.label_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        # x [batch_size, seq_len]
        embedding = self.embedding(x)
        # embedding [batch_size, seq_len, embed_size]
        h0 = torch.randn(2, x.size(0), self.hidden_size_lstm).cuda()
        c0 = torch.randn(2, x.size(0), self.hidden_size_lstm).cuda()
        output, (hn, cn) = self.lstm(embedding, (h0, c0))
        outputsplit1, outputsplit2 = output.chunk(2, dim=2)
        # print(x.shape)
        outputcat = torch.cat((outputsplit1, embedding, outputsplit2), dim=2)
        outputcat = outputcat.permute(0, 2, 1)
        x = [conv(outputcat) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = x.view(-1, x.size(1))
        x = self.linear(x)
        outputs = x.reshape((-1, self.label_num, self.label_class))
        # (batchsize, self.label_num * self.label_class) => (batchsize, self.label_num, self.label_class)
        logits = self.softmax(outputs)
        return logits


class RandomLSTMLinear(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab = kwargs['vocab']
        self.word2idx = kwargs['word2idx']
        self.hidden_size_lstm = kwargs['hidden_size_lstm']

        self.embedding = nn.Embedding(
            len(self.vocab), self.embed_size,
            padding_idx=self.word2idx['<pad>']
        )
        self.lstm = nn.LSTM(
            self.embed_size, self.hidden_size_lstm,
            bidirectional=True, batch_first=True
        )
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size_lstm * 2, 128),
            nn.Dropout(0.8),
            nn.ReLU(),
            nn.Linear(128, self.label_num * self.label_class)
        )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        embedding = self.embedding(x)
        h0 = torch.randn(2, x.size(0), self.hidden_size_lstm).to(x.device)
        c0 = torch.randn(2, x.size(0), self.hidden_size_lstm).to(x.device)
        output, (hn, _) = self.lstm(embedding, (h0, c0))
        hn_cat = torch.cat([hn[0], hn[1]], dim=-1)
        x = self.fc(hn_cat)
        return self.softmax(x.reshape(-1, self.label_num, self.label_class))


# class FastTextAttLinear(BaseModel):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.vocab = kwargs['vocab']
#         self.word2idx = kwargs['word2idx']
#         self.word_matrix = kwargs['word_matrix']  # Your word matrix
#
#         self.embedding = nn.Embedding(len(self.vocab), self.embed_size, padding_idx=self.word2idx['<pad>'])
#         self.embedding.weight.data.copy_(torch.from_numpy(self.word_matrix))
#         self.linear_att = nn.Linear(self.embed_size, 128)
#         self.tanh_att = nn.Tanh()
#         self.u_w_att = nn.Linear(128, 1)
#         self.softmax_att = nn.Softmax(dim=-1)
#         self.linear = nn.Linear(self.embed_size, self.label_num * self.label_class)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         # x [batch_size, seq_len]
#         embedding = self.embedding(x)
#         # embedding [batch_size, seq_len, embed_size]
#         attention_u = self.tanh_att(self.linear_att(embedding))
#         # (batchsize, seq_len, embed_size) => (batchsize, seq_len, u_size)
#         attention_a = self.softmax_att(self.u_w_att(attention_u))
#         # (batchsize, seq_len, u_size) * (u_size, 1) => (batchsize, seq_len, 1)
#         outputs = torch.matmul(attention_a.permute(0, 2, 1), embedding).squeeze()
#         # (batchsize, 1, seq_len) * (batch_size, seq_len, embed_size) => (batchsize, embed_size)
#         outputs = self.linear(outputs)
#         # (batchsize, embed_size) => (batchsize, self.label_num * self.label_class)
#         outputs = outputs.reshape((-1, self.label_num, self.label_class))
#         # (batchsize, self.label_num * self.label_class) => (batchsize, self.label_num, self.label_class)
#         logits = self.softmax(outputs)
#         return logits
#
#
# class FastTextAverageLinear(BaseModel):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.vocab = kwargs['vocab']
#         self.word2idx = kwargs['word2idx']
#         self.word_matrix = kwargs['word_matrix']  # Your word matrix
#
#         self.embedding = nn.Embedding(len(self.vocab), self.embed_size, padding_idx=self.word2idx['<pad>'])
#         self.embedding.weight.data.copy_(torch.from_numpy(self.word_matrix))
#         self.linear = nn.Linear(self.embed_size, self.label_num * self.label_class)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         # x [batch_size, seq_len]
#         embedding = self.embedding(x)
#         # embedding [batch_size, seq_len, embed_size]
#         word_averaging = embedding.mean(dim=1)
#         # word_averaging [batch_size, embed_size]
#         outputs = self.linear(word_averaging)
#         outputs = outputs.reshape((-1, self.label_num, self.label_class))
#         # output = self.relu(output)
#         # output = self.linear2(output)
#         logits = self.softmax(outputs)
#         return logits
#
#
# class FastTextCNNLinear(BaseModel):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.vocab = kwargs['vocab']
#         self.word2idx = kwargs['word2idx']
#         self.word_matrix = kwargs['word_matrix']  # Your word matrix
#         self.num_of_filters = kwargs['num_of_filters']
#         self.window_sizes = kwargs['window_sizes']
#
#         self.embedding = nn.Embedding(len(self.vocab), self.embed_size, padding_idx=self.word2idx['<pad>'])
#         self.embedding.weight.data.copy_(torch.from_numpy(self.word_matrix))
#
#         self.convs = nn.ModuleList(
#             [nn.Sequential(nn.Conv1d(in_channels=self.embed_size, out_channels=self.num_of_filters, kernel_size=h),
#                            nn.ReLU(),
#                            nn.MaxPool1d(self.maxlen - h + 1)
#                            ) for h in self.window_sizes]
#         )
#         self.linear = nn.Linear(self.num_of_filters * len(self.window_sizes), self.label_num * self.label_class)
#         self.softmax = nn.Softmax(dim=-1)
#
#         self.linear = nn.Linear(self.embed_size, self.label_num * self.label_class)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         # x [batch_size, seq_len]
#         embedding = self.embedding(x)
#         # embedding [batch_size, seq_len, embed_size]
#         word_averaging = embedding.mean(dim=1)
#         # word_averaging [batch_size, embed_size]
#         outputs = self.linear(word_averaging)
#         outputs = outputs.reshape((-1, self.label_num, self.label_class))
#         # output = self.relu(output)
#         # output = self.linear2(output)
#         logits = self.softmax(outputs)
#         return logits
#
#
# class FastTextLSTMAttLinear(BaseModel):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.vocab = kwargs['vocab']
#         self.word2idx = kwargs['word2idx']
#         self.word_matrix = kwargs['word_matrix']  # Your word matrix
#         self.hidden_size_lstm = kwargs['hidden_size_lstm']
#         self.num_of_filters = kwargs['num_of_filters']
#         self.window_sizes = kwargs['window_sizes']
#
#         self.embedding = nn.Embedding(len(self.vocab), self.embed_size, padding_idx=self.word2idx['<pad>'])
#         self.embedding.weight.data.copy_(torch.from_numpy(self.word_matrix))
#         self.lstm = nn.LSTM(self.embed_size, self.hidden_size_lstm, bidirectional=True, batch_first=True)
#         self.convs = nn.ModuleList(
#             [nn.Sequential(
#                 nn.Conv1d(in_channels=self.embed_size + 2 * self.hidden_size_lstm, out_channels=self.num_of_filters,
#                           kernel_size=h),
#                 nn.ReLU(),
#                 nn.MaxPool1d(self.maxlen - h + 1)
#             ) for h in self.window_sizes]
#         )
#
#         self.linear = nn.Linear(self.num_of_filters * len(self.window_sizes), self.label_num * self.label_class)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         # x [batch_size, seq_len]
#         embedding = self.embedding(x)
#         # embedding [batch_size, seq_len, embed_size]
#         h0 = torch.randn(2, x.size(0), self.hidden_size_lstm).cuda()
#         c0 = torch.randn(2, x.size(0), self.hidden_size_lstm).cuda()
#         output, (hn, cn) = self.lstm(embedding, (h0, c0))
#         outputsplit1, outputsplit2 = output.chunk(2, dim=2)
#         # print(x.shape)
#         outputcat = torch.cat((outputsplit1, embedding, outputsplit2), dim=2)
#         outputcat = outputcat.permute(0, 2, 1)
#         x = [conv(outputcat) for conv in self.convs]
#         x = torch.cat(x, dim=1)
#         x = x.view(-1, x.size(1))
#         x = self.linear(x)
#         outputs = x.reshape((-1, self.label_num, self.label_class))
#         # (batchsize, self.label_num * self.label_class) => (batchsize, self.label_num, self.label_class)
#         logits = self.softmax(outputs)
#         return logits
#
#
# class FastTextLSTMCNNLinear(BaseModel):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.vocab = kwargs['vocab']
#         self.word2idx = kwargs['word2idx']
#         self.word_matrix = kwargs['word_matrix']  # Your word matrix
#         self.hidden_size_lstm = kwargs['hidden_size_lstm']
#
#         self.embedding = nn.Embedding(len(self.vocab), self.embed_size, padding_idx=self.word2idx['<pad>'])
#         self.embedding.weight.data.copy_(torch.from_numpy(self.word_matrix))
#         self.lstm = nn.LSTM(self.embed_size, self.hidden_size_lstm, bidirectional=True, batch_first=True)
#         self.linear_att = nn.Linear(self.hidden_size_lstm * 2, 128)
#         self.tanh_att = nn.Tanh()
#         self.u_w_att = nn.Linear(128, 1)
#         self.softmax_att = nn.Softmax(dim=-1)
#
#         self.linear = nn.Linear(self.hidden_size_lstm * 2, self.label_num * self.label_class)
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         # x [batch_size, seq_len]
#         embedding = self.embedding(x)
#         # embedding [batch_size, seq_len, embed_size]
#         h0 = torch.randn(2, x.size(0), self.hidden_size_lstm).cuda()
#         c0 = torch.randn(2, x.size(0), self.hidden_size_lstm).cuda()
#         output, (hn, cn) = self.lstm(embedding, (h0, c0))
#         attention_u = self.tanh_att(self.linear_att(output))
#         # (batchsize, seq_len, embed_size) => (batchsize, seq_len, u_size)
#         attention_a = self.softmax_att(self.u_w_att(attention_u))
#         # (batchsize, seq_len, u_size) * (u_size, 1) => (batchsize, seq_len, 1)
#         outputs = torch.matmul(attention_a.permute(0, 2, 1), output).squeeze()
#         # (batchsize, 1, seq_len) * (batch_size, seq_len, embed_size) => (batchsize, embed_size)
#         outputs = self.linear(outputs)
#         # (batchsize, embed_size) => (batchsize, self.label_num * self.label_class)
#         outputs = outputs.reshape((-1, self.label_num, self.label_class))
#         # (batchsize, self.label_num * self.label_class) => (batchsize, self.label_num, self.label_class)
#         logits = self.softmax(outputs)
#         return logits
#
#
# class FastTextLSTMLinear(BaseModel):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.vocab = kwargs['vocab']
#         self.word2idx = kwargs['word2idx']
#         self.word_matrix = kwargs['word_matrix']  # Your word matrix
#         self.hidden_size_lstm = kwargs['hidden_size_lstm']
#         self.num_of_filters = kwargs['num_of_filters']
#         self.window_sizes = kwargs['window_sizes']
#
#         self.embedding = nn.Embedding(len(self.vocab), self.embed_size, padding_idx=self.word2idx['<pad>'])
#         self.embedding.weight.data.copy_(torch.from_numpy(self.word_matrix))
#         self.lstm = nn.LSTM(self.embed_size, self.hidden_size_lstm, bidirectional=True, batch_first=True)
#         self.f1 = nn.Sequential(nn.Linear(self.hidden_size_lstm * 2, 128),
#                                 nn.Dropout(0.8),
#                                 nn.ReLU(),
#                                 nn.Linear(128, self.label_num * self.label_class)
#                                 )
#         self.softmax = nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         # x [batch_size, seq_len]
#         embedding = self.embedding(x)
#         # embedding [batch_size, seq_len, embed_size]
#         h0 = torch.randn(2, x.size(0), self.hidden_size_lstm).cuda()
#         c0 = torch.randn(2, x.size(0), self.hidden_size_lstm).cuda()
#         output, (hn, cn) = self.lstm(embedding, (h0, c0))
#         # batch_size*seq_len*(self.hidden_size_lstm*2)
#         hn_cat = torch.cat([hn[0], hn[1]], dim=-1)
#         x = self.f1(hn_cat)
#         outputs = x.reshape((-1, self.label_num, self.label_class))
#         return self.softmax(outputs)

class BertAttLinear(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs['model']
        self.maxlen = kwargs['maxlen']

        self.bert = BertModel.from_pretrained(self.model_name, output_hidden_states=True, return_dict=True)
        self.dropout = nn.Dropout(0.5)
        self.linear_att = nn.Linear(self.embed_size, 128)
        self.tanh_att = nn.Tanh()
        self.u_w_att = nn.Linear(128, 1)
        self.softmax_att = nn.Softmax(dim=-1)
        self.linear = nn.Linear(self.embed_size, self.label_num * self.label_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # Returns an output dictionary
        # Use the last layer cls vector for classification
        # outputs.pooler_output: [batch_size, embed_size]
        outputs = self.dropout(outputs.last_hidden_state)
        attention_u = self.tanh_att(self.linear_att(outputs))
        # (batchsize, seq_len, embed_size) => (batchsize, seq_len, u_size)
        attention_a = self.softmax_att(self.u_w_att(attention_u))
        # (batchsize, seq_len, u_size) * (u_size, 1) => (batchsize, seq_len, 1)
        outputs = torch.matmul(attention_a.permute(0, 2, 1), outputs).squeeze()
        # (batchsize, 1, seq_len) * (batch_size, seq_len, embed_size) => (batchsize, embed_size)
        outputs = self.linear(outputs)
        # (batchsize, embed_size) => (batchsize, self.label_num * self.label_class)
        outputs = outputs.reshape((-1, self.label_num, self.label_class))
        # (batchsize, self.label_num * self.label_class) => (batchsize, self.label_num, self.label_class)
        logits = self.softmax(outputs)
        return logits


class BertAverageLinear(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs['model']
        self.maxlen = kwargs['maxlen']

        self.bert = BertModel.from_pretrained(self.model_name, output_hidden_states=True, return_dict=True)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(self.embed_size, self.label_num * self.label_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # Returns an output dictionary
        # Use the last layer cls vector for classification
        # outputs.pooler_output: [batch_size, embed_size]
        outputs = self.dropout(outputs.last_hidden_state)
        word_averaging = outputs.mean(dim=1)
        # outputs: [batch_size, label_num * label_class]
        outputs = self.linear(word_averaging)
        outputs = outputs.reshape((-1, self.label_num, self.label_class))
        # output = self.relu(output)
        # output = self.linear2(output)
        logits = self.softmax(outputs)
        return logits


class BertCLSLinear(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs['model']
        self.maxlen = kwargs['maxlen']

        self.bert = BertModel.from_pretrained(self.model_name, output_hidden_states=True, return_dict=True)
        self.dropout = nn.Dropout(0.5)
        self.linear = nn.Linear(self.embed_size, self.label_num * self.label_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # Returns an output dictionary
        # Use the last layer cls vector for classification
        # outputs.pooler_output: [batch_size, embed_size]
        outputs = self.linear(self.dropout(outputs.pooler_output))
        # outputs: [batch_size, label_num * label_class]
        outputs = outputs.reshape((-1, self.label_num, self.label_class))
        # outputs: [batch_size, label_num, label_class]
        logits = self.softmax(outputs)
        # logits = self.linear(self.dropout(outputs.pooler_output))
        return logits


class BertCNNLinear(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs['model']
        self.maxlen = kwargs['maxlen']
        self.num_of_filters = kwargs['num_of_filters']
        self.window_sizes = kwargs['window_sizes']

        self.bert = BertModel.from_pretrained(self.model_name, output_hidden_states=True, return_dict=True)
        self.dropout = nn.Dropout(0.5)
        self.convs = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(in_channels=self.embed_size, out_channels=self.num_of_filters, kernel_size=h),
                           nn.ReLU(),
                           nn.MaxPool1d(self.maxlen - h + 1)
                           ) for h in self.window_sizes]
        )
        self.linear = nn.Linear(self.num_of_filters * len(self.window_sizes), self.label_num * self.label_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # Returns an output dictionary
        # Use the last layer cls vector for classification
        # outputs.pooler_output: [batch_size, embed_size]
        outputs = self.dropout(outputs.last_hidden_state)
        outputs = outputs.permute(0, 2, 1)
        outputs = [conv(outputs) for conv in self.convs]
        outputs = torch.cat(outputs, dim=1)
        outputs = outputs.view(-1, outputs.size(1))
        outputs = self.linear(outputs)
        outputs = outputs.reshape((-1, self.label_num, self.label_class))
        return self.softmax(outputs)


class BertLSTMAttLinear(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs['model']
        self.hidden_size_lstm = kwargs['hidden_size_lstm']
        self.maxlen = kwargs['maxlen']
        self.bert = BertModel.from_pretrained(self.model_name, output_hidden_states=True, return_dict=True)
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size_lstm, bidirectional=True, batch_first=True)
        self.linear_att = nn.Linear(self.hidden_size_lstm * 2, 128)
        self.tanh_att = nn.Tanh()
        self.u_w_att = nn.Linear(128, 1)
        self.softmax_att = nn.Softmax(dim=-1)
        self.linear = nn.Linear(self.hidden_size_lstm * 2, self.label_num * self.label_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # Returns an output dictionary
        # Use the last layer cls vector for classification
        # outputs.pooler_output: [batch_size, embed_size]
        outputs = self.dropout(outputs.last_hidden_state)
        h0 = torch.randn(2, outputs.size(0), self.hidden_size_lstm).cuda()
        c0 = torch.randn(2, outputs.size(0), self.hidden_size_lstm).cuda()
        output, (hn, cn) = self.lstm(outputs, (h0, c0))
        attention_u = self.tanh_att(self.linear_att(output))
        # (batchsize, seq_len, embed_size) => (batchsize, seq_len, u_size)
        attention_a = self.softmax_att(self.u_w_att(attention_u))
        # (batchsize, seq_len, u_size) * (u_size, 1) => (batchsize, seq_len, 1)
        outputs = torch.matmul(attention_a.permute(0, 2, 1), output).squeeze()
        # (batchsize, 1, seq_len) * (batch_size, seq_len, embed_size) => (batchsize, embed_size)
        outputs = self.linear(outputs)
        # (batchsize, embed_size) => (batchsize, self.label_num * self.label_class)
        outputs = outputs.reshape((-1, self.label_num, self.label_class))
        # (batchsize, self.label_num * self.label_class) => (batchsize, self.label_num, self.label_class)
        logits = self.softmax(outputs)
        return logits


class BertLSTMCNNLinear(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs['model']
        self.hidden_size_lstm = kwargs['hidden_size_lstm']
        self.num_of_filters = kwargs['num_of_filters']
        self.window_sizes = kwargs['window_sizes']
        self.maxlen = kwargs['maxlen']

        self.bert = BertModel.from_pretrained(
            self.model_name,
            output_hidden_states=True,
            return_dict=True
        )
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size_lstm, bidirectional=True, batch_first=True)
        self.convs = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels=self.embed_size + 2 * self.hidden_size_lstm, out_channels=self.num_of_filters,
                          kernel_size=h),
                nn.ReLU(),
                nn.MaxPool1d(self.maxlen - h + 1)
            ) for h in self.window_sizes]
        )

        self.linear = nn.Linear(self.num_of_filters * len(self.window_sizes), self.label_num * self.label_class)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # Returns an output dictionary
        # Use the last layer cls vector for classification
        # outputs.pooler_output: [batch_size, embed_size]
        outputs = self.dropout(outputs.last_hidden_state)
        h0 = torch.randn(2, outputs.size(0), self.hidden_size_lstm).cuda()
        c0 = torch.randn(2, outputs.size(0), self.hidden_size_lstm).cuda()
        output, (hn, cn) = self.lstm(outputs, (h0, c0))
        outputsplit1, outputsplit2 = output.chunk(2, dim=2)
        # print(x.shape)
        outputcat = torch.cat((outputsplit1, outputs, outputsplit2), dim=2)
        outputcat = outputcat.permute(0, 2, 1)
        x = [conv(outputcat) for conv in self.convs]
        x = torch.cat(x, dim=1)
        x = x.view(-1, x.size(1))
        x = self.linear(x)
        outputs = x.reshape((-1, self.label_num, self.label_class))
        # (batchsize, self.label_num * self.label_class) => (batchsize, self.label_num, self.label_class)
        logits = self.softmax(outputs)
        return logits


class BertLSTMLinear(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model_name = kwargs['model']
        self.hidden_size_lstm = kwargs['hidden_size_lstm']
        self.maxlen = kwargs['maxlen']
        self.bert = BertModel.from_pretrained(
            self.model_name,
            output_hidden_states=True,
            return_dict=True
        )
        self.dropout = nn.Dropout(0.5)
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size_lstm, bidirectional=True, batch_first=True)
        self.f1 = nn.Sequential(nn.Linear(self.hidden_size_lstm * 2, 128),
                                nn.Dropout(0.8),
                                nn.ReLU(),
                                nn.Linear(128, self.label_num * self.label_class)
                                )
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        input_ids, attention_mask, token_type_ids = x
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)  # Returns an output dictionary
        # Use the last layer cls vector for classification
        # outputs.pooler_output: [batch_size, embed_size]
        outputs = self.dropout(outputs.last_hidden_state)
        h0 = torch.randn(2, outputs.size(0), self.hidden_size_lstm).cuda()
        c0 = torch.randn(2, outputs.size(0), self.hidden_size_lstm).cuda()
        output, (hn, cn) = self.lstm(outputs, (h0, c0))
        # batch_size*seq_len*(self.hidden_size_lstm*2)
        hn_cat = torch.cat([hn[0], hn[1]], dim=-1)
        x = self.f1(hn_cat)
        outputs = x.reshape((-1, self.label_num, self.label_class))
        return self.softmax(outputs)