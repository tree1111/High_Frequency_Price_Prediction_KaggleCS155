# -*- coding: utf-8 -*-
"""model.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1enzH4lOBffeM9h85MmTZKXYzycqBhJBi
"""

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

training_data =  pd.read_csv('train.csv', sep=',', header = 0, skiprows = 0, nrows=10000)
values = list(training_data.columns.values)
print(len(values))
training_Y = training_data[values[-1:]]
training_Y = np.array(training_Y, dtype='int')
training_Y = np.squeeze(training_Y)
training_X = training_data[values[0:-1]]
training_X = np.array(training_X, dtype='float32')[:, 1:]
training_X = np.nan_to_num(training_X)
training_X = torch.from_numpy(training_X)
training_Y = torch.from_numpy(training_Y)
print(training_X.shape)
print(training_Y.shape)

N, D_in, D_out = 10000, 26, 2

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, 100),
    torch.nn.ReLU(),
    torch.nn.Linear(100, D_out),
)

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for t in range(500):
    # Forward pass: compute predicted y by passing x to the model. Module objects
    # override the __call__ operator so you can call them like functions. When
    # doing so you pass a Tensor of input data to the Module and it produces
    # a Tensor of output data.
    y_pred = model(training_X)
    #print(y_pred.shape)

    # Compute and print loss. We pass Tensors containing the predicted and true
    # values of y, and the loss function returns a Tensor containing the
    # loss.
    loss = loss_fn(y_pred, training_Y)
    if t % 100 == 99:
        print(t, loss.item())

    # Zero the gradients before running the backward pass.
    model.zero_grad()

    # Backward pass: compute gradient of the loss with respect to all the learnable
    # parameters of the model. Internally, the parameters of each Module are stored
    # in Tensors with requires_grad=True, so this call will compute gradients for
    # all learnable parameters in the model.
    loss.backward()

    # Update the weights using gradient descent. Each parameter is a Tensor, so
    # we can access its gradients like we did before.
    optimizer.step()

test_data =  pd.read_csv('train.csv', sep=',', header = 0, skiprows = 10000, nrows=1000)
values = list(test_data.columns.values)
print(len(values))
test_Y = test_data[values[-1:]]
test_Y = np.array(test_Y, dtype='int')
test_Y = np.squeeze(test_Y)
test_X = test_data[values[0:-1]]
test_X = np.array(test_X, dtype='float32')[:, 1:]
test_X = np.nan_to_num(test_X)
test_X = torch.from_numpy(test_X)
test_Y = torch.from_numpy(test_Y)
print(test_X.shape)
print(test_Y.shape)

correct = 0
total = 0
with torch.no_grad():
    for i in range(test_X.shape[0]):
        test_input, test_labels = test_X[i], test_Y[i]
        outputs = model(test_input)
        _, predicted = torch.max(outputs.data, 0)
        total += 1
        correct += (predicted == test_labels).sum().item()
print(correct)
print(total)

print('Accuracy of the network on the test set: %d %%' % (
    100 * correct / total))
