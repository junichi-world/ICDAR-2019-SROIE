import torch.nn as nn


class BidirectionalLSTM(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super().__init__()
        self.rnn = nn.LSTM(n_in, n_hidden, bidirectional=True)
        self.embedding = nn.Linear(n_hidden * 2, n_out)

    def forward(self, x):
        recurrent, _ = self.rnn(x)
        t, b, h = recurrent.size()
        t_rec = recurrent.view(t * b, h)

        output = self.embedding(t_rec)
        output = output.view(t, b, -1)
        return output


class CRNN(nn.Module):
    def __init__(self, img_h, nc, nclass, nh, leaky_relu=False):
        super().__init__()
        assert img_h % 16 == 0, "img_h has to be a multiple of 16"

        ks = [3, 3, 3, 3, 3, 3, 2]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ss = [1, 1, 1, 1, 1, 1, 1]
        nm = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        def conv_relu(i, batch_normalization=False):
            n_in = nc if i == 0 else nm[i - 1]
            n_out = nm[i]
            cnn.add_module(
                f"conv{i}", nn.Conv2d(n_in, n_out, ks[i], ss[i], ps[i])
            )
            if batch_normalization:
                cnn.add_module(f"batchnorm{i}", nn.BatchNorm2d(n_out))
            if leaky_relu:
                cnn.add_module(f"relu{i}", nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module(f"relu{i}", nn.ReLU(True))

        conv_relu(0)
        cnn.add_module("pooling0", nn.MaxPool2d(2, 2))
        conv_relu(1)
        cnn.add_module("pooling1", nn.MaxPool2d(2, 2))
        conv_relu(2, True)
        conv_relu(3)
        cnn.add_module("pooling2", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        conv_relu(4, True)
        conv_relu(5)
        cnn.add_module("pooling3", nn.MaxPool2d((2, 2), (2, 1), (0, 1)))
        conv_relu(6, True)

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, nh, nh),
            BidirectionalLSTM(nh, nh, nclass),
        )

    def forward(self, x):
        conv = self.cnn(x)
        _, _, h, _ = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        return output
