import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. Create synthetic dataset: y = 3x + 1 + noise
torch.manual_seed(42)
X = torch.linspace(0, 1, 100).unsqueeze(1)
y = 3 * X + 1 + 0.1 * torch.randn(X.size())

# 2. Define a simple linear model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # input_dim=1, output_dim=1

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()

# 3. Define loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# 4. Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    y_pred = model(X)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch [{epoch}/{epochs}] Loss: {loss.item():.4f}")

# 5. Plot the results
model.eval()
predicted = model(X).detach()
plt.figure(figsize=(8, 5))
plt.plot(X.numpy(), y.numpy(), 'o', label='True Data')
plt.plot(X.numpy(), predicted.numpy(), label='Model Prediction')
plt.legend()
plt.title("Linear Regression with PyTorch")
plt.xlabel("X")
plt.ylabel("y")
plt.grid(True)
plt.savefig("./generated_images/prediction_plot.png")
plt.show()
