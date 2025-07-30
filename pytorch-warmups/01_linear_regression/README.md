# PyTorch Warm-up: Linear Regression

# Goal: Learn the PyTorch basics: tensor, Module, loss, optimizer, training loop
# Target Function: y = 3x + 1 (with noise)
# Output: Predicted vs ground truth plot
# Runs on: CPU 

This is a simple linear regression model implemented in PyTorch to predict synthetic data following the equation `y = 3x + 1 + noise`.

## Features
- Implements `torch.nn.Module` for model architecture
- Uses `torch.optim.SGD` and `MSELoss`
- Trains for 100 epochs and plots predictions

## Output
Generates a plot comparing predicted values vs true data points.

## Run
```bash

python linear_regression.py

```

## File Structure : 

pytorch-warmups/
├── 01_linear_regression/
│   ├── linear_regression.py
│   └── README.md