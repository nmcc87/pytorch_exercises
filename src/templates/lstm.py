f = torch.sigmoid(...)
i = torch.sigmoid(...)
o = torch.sigmoid(...)
g = torch.tanh(...)

c = f * c + i * g
h = o * torch.tanh(c)