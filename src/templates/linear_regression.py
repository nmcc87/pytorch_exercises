import torch

X = torch.randn(100, 1)
y = 3 * X + 2 + 0.1 * torch.randn(100, 1)

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

lr = 0.1

for i in range(100):
    y_pred = w * X + b
    loss = ((y_pred - y)**2).mean()

    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()