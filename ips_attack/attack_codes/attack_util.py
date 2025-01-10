from warnings import filterwarnings
filterwarnings("ignore")
from sklearn.metrics import fbeta_score
from model_utils import *
import torch

Test_Data_File = {
    'Splice': './dataset/gene_test_funccall.pickle',
    'IPS': './dataset/mal_test_funccall.pickle'
}

Test_Label_File = {
    'Splice': './dataset/gene_test_label.pickle',
    'IPS': './dataset/mal_test_label.pickle'
}

Whole_Data_File = {
    'Splice': './dataset/spliceX.pickle',
}

Whole_Label_File = {
    'Splice': './dataset/spliceY.pickle',
}

num_category = {'Splice': 5, 'IPS': 1104}
num_feature = {'Splice': 60, 'IPS': 20}

Splice_Model = {
    'Normal': './classifier/Adam_RNN.4832',
    'adversarial': './classifier/Adam_RNN.17490'
}

IPS_Model = {
    'Normal': './classifier/Mal_RNN.942',
    'adversarial': './classifier/Mal_adv.705',
}

Model = {
    'Splice': Splice_Model,
    'IPS': IPS_Model,
}

def get_data_and_learner(WORKING_FOLDER, kwargs, num_seqs=500):
    data = pickle.load(open(Test_Data_File[str(WORKING_FOLDER)], 'rb'))
    label = pickle.load(open(Test_Label_File[str(WORKING_FOLDER)], 'rb'))
    dl = []
    for d, l in zip(data, label): 
        dl.append( [torch.LongTensor([d]).cuda(), torch.LongTensor([l]).cuda()] )
    if len(dl) > num_seqs:
        random.shuffle(dl)
    target_seqs = dl[:num_seqs]
    class_num = num_category[str(WORKING_FOLDER)]

    itos={i:x for i,x in enumerate(range(class_num))}
    vocab=Vocab(itos)
    return target_seqs, vocab

def load_pretrained_model(kwargs):
    print("Loading model ",str(kwargs["Dataset"]), str(kwargs["Model_Type"]))

    best_parameters_file = Model[str(kwargs["Dataset"])][str(kwargs["Model_Type"])]
    if str(kwargs["Dataset"]) == 'Splice':
        model = geneRNN()
    elif str(kwargs["Dataset"]) == 'IPS':
        model = IPSRNN()
    if torch.cuda.is_available():
        model = model.cuda()
    model.load_state_dict(torch.load(best_parameters_file, map_location='cpu'))
    print("Loaded Successfully")

    return model

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


class geneRNN(nn.Module):
    def __init__(self):
        super(geneRNN, self).__init__()
        dropout_rate = 0.5
        input_size = 30
        self.hidden_size = 30
        n_labels = 3
        self.num_layers = 4
        self.lstm = nn.LSTM(input_size, self.hidden_size, num_layers=self.num_layers, batch_first=False)
        self.fc = nn.Linear(self.hidden_size, n_labels)
        self.attention = SelfAttention(self.hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.n_diagnosis_codes = 5
        self.embed = torch.nn.Embedding(self.n_diagnosis_codes, 30)
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))

    # overload forward() method
    def forward(self, x):
        #model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        weight = self.embed(x)
        x = weight.transpose(0,1).relu() / self.n_diagnosis_codes
        # x = torch.unsqueeze(x, dim=3)
        # x = (x * weight).relu().mean(dim=2)

        h0 = torch.randn((self.num_layers, x.size()[1], self.hidden_size)).cuda()
        c0 = torch.randn((self.num_layers, x.size()[1], self.hidden_size)).cuda()
        output, h_n = self.lstm(x)

        x, attn_weights = self.attention(output.transpose(0, 1))

        x = self.dropout(x)

        logit = self.fc(x)

        #logit = self.softmax(logit)

        return logit


class SelfAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(SelfAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(True),
            nn.Linear(64, 1)
        )

    def forward(self, encoder_outputs):
        # (B, L, H) -> (B , L, 1)
        energy = self.projection(encoder_outputs)
        weights = F.softmax(energy.squeeze(-1), dim=1)
        # (B, L, H) * (B, L, 1) -> (B, H)
        outputs = (encoder_outputs * weights.unsqueeze(-1)).sum(dim=1)
        return outputs, weights


class IPSRNN(nn.Module):
    def __init__(self):
        super(IPSRNN, self).__init__()
        dropout_rate = 0.5
        input_size = 70
        hidden_size = 70
        n_labels = 3
        self.n_diagnosis_codes = 1104
        self.num_layers = 1
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=False)
        self.fc = nn.Linear(hidden_size, n_labels)
        self.attention = SelfAttention(hidden_size)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.embed = torch.nn.Embedding(self.n_diagnosis_codes, input_size)
        # self.embed.weight = nn.Parameter(torch.FloatTensor(emb_weights))
        self.model_input = torch.LongTensor(range(self.n_diagnosis_codes))

    def forward(self, x):
        # model_input = self.model_input.reshape(1, 1, self.n_diagnosis_codes).cuda()
        # weight = self.embed(model_input)
        weight = self.embed(x)
        x = weight.transpose(0,1).relu() / self.n_diagnosis_codes
        # x = torch.unsqueeze(x, dim=3)
        # x = (x * weight).relu().mean(dim=2)

        h0 = torch.randn((self.num_layers, x.size()[1], x.size()[2])).cuda()
        c0 = torch.randn((self.num_layers, x.size()[1], x.size()[2])).cuda()
        output, h_n = self.lstm(x)

        x, attn_weights = self.attention(output.transpose(0, 1))

        x = self.dropout(x)

        logit = self.fc(x)

        #logit = self.softmax(logit)
        return logit
