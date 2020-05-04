import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torchvision.models.resnet import resnet50


class Encoder(nn.Module):
    def __init__(self, embeded_dim):
        super(Encoder, self).__init__()
        self.feature = resnet50(pretrained=True)
        modules = list(self.feature.children())[:-1]
        self.feature = nn.Sequential(*modules)
        for p in self.feature.parameters():
            p.requires_grad = False

        # 将提取的特征进行embedded, resnet50 提取的特征维度为（b, 2048, 1, 1）
        self.embedding = nn.Linear(in_features=2048, out_features=embeded_dim, bias=False)

    def forward(self, x):
        x = self.feature(x).squeeze(2).squeeze(2)
        x = F.relu(self.embedding(x))
        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, en_embeded_dim, de_embeded_dim, hidden_dim, drop_out):
        super(Decoder, self).__init__()
        self.vocab_size = vocab_size
        self.en_embeded_dim = en_embeded_dim
        self.de_embeded_dim = de_embeded_dim
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, de_embeded_dim)
        self.rnn = nn.GRU(de_embeded_dim + en_embeded_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim + de_embeded_dim + en_embeded_dim, vocab_size)
        self.dropout = nn.Dropout(drop_out)

    def forward(self, x, hidden, context):
        x = x.unsqueeze(0)
        embedded = self.dropout(self.embedding(x))
        emb_con = torch.cat((embedded, context), dim=2)
        output, hidden = self.rnn(emb_con, hidden)
        output = torch.cat((embedded.squeeze(0), hidden.squeeze(0), context.squeeze(0)),
                           dim=1)
        prediction = self.fc(output)

        return prediction, hidden


class ImageCaption(nn.Module):
    def __init__(self, vocab_size, enc_embed_dim, dec_embed_dim, hidden, drop, device):
        """

        :param vocab_size: 词汇表大小
        :param enc_embed_dim: 编码器将提取的图片特征映射后作为context vector, 由于需要用来初始化解码器的hidden state, 故等于hidden
        :param dec_embed_dim: 词 embedding 输出维度
        :param hidden: 解码器 hidden 神经元个数
        :param drop:
        :param device:
        """
        super(ImageCaption, self).__init__()
        self.encoder = Encoder(embeded_dim=enc_embed_dim)
        self.decoder = Decoder(vocab_size=vocab_size, en_embeded_dim=enc_embed_dim, de_embeded_dim=dec_embed_dim,
                               hidden_dim=hidden, drop_out=drop)
        self.vocab_size = vocab_size
        self.device = device

    def forward(self, img, targ, teacher_forcing_ratio=0.5):
        batch_size = img.shape[0]
        targ_len = targ.shape[0]
        vocab_size = self.vocab_size
        outputs = torch.zeros(targ_len, batch_size, vocab_size).to(self.device)
        context = self.encoder(img).unsqueeze(0)
        hidden = context
        x = targ[0, :]
        for t in range(1, targ_len):
            # 插入token, 前一个时间步的 hidden, cell states,
            # 输出预测, hidden, cell states
            output, hidden = self.decoder(x, hidden, context)
            # 保存预测输出
            outputs[t] = output
            # 决定是否使用 teacher forcing
            teacher_force = random.random() < teacher_forcing_ratio
            # 获取当前预测的最高概率的token
            top1 = output.argmax(1)
            # 如果使用 teacher forcing, 使用实际的下一个词作为输入, 否则使用预测的作为下一个输入
            x = targ[t] if teacher_force else top1

        return outputs

    def infer(self, img, max_seq_len, sos_token, eos_token):
        # 贪心搜索
        context = self.encoder(img).unsqueeze(0)
        hidden = context
        token = torch.tensor([sos_token])
        output_tokens = []
        for t in range(1, max_seq_len):
            output, hidden = self.decoder(token, hidden, context)
            top1 = output.argmax()
            output_tokens.append(top1.item())
            if top1 == eos_token:
                break
            token = torch.tensor([top1.item()])

        return output_tokens
