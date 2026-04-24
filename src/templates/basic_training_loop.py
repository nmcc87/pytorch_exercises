# Basic training loop

for epoch in range(num_epochs):
    for x, y in dataloader:

        optimizer.zero_grad()

        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        loss.backward()
        optimizer.step()

    print(f"epoch {epoch}, loss {loss.item()}")