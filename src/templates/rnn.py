h = torch.zeros(hidden_size)

for t in range(seq_len):
    h = torch.tanh(x[t] @ Wx + h @ Wh + b)