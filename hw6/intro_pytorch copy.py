from pkgutil import get_data
import torch
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """

    custom_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])


    train_set=datasets.FashionMNIST('./data',train=True, download=True,transform=custom_transform)
    test_set=datasets.FashionMNIST('./data', train=False, transform=custom_transform)

    if not training:
        loader = data.DataLoader(test_set, batch_size = 64)

    else:
        loader = data.DataLoader(train_set, batch_size = 64)

    return loader


def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )

    return model



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """

    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(T):
        running_loss = 0.0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

            running_loss += loss.item() * labels.size(0)
            x, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()

        print(f'Train Epoch: {epoch}    Accuracy: {correct}/{len(train_loader.sampler)}({100 * correct /len(train_loader.sampler):.2f}%)   Loss: {running_loss / len(train_loader.sampler):.3f}')


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """

    model.eval()
    correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            x, predicted  = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            total_loss += loss.item() * labels.size(0)

    if show_loss:
        print(f'Average loss: {total_loss / len(test_loader.sampler):.4f}')

    print(f'Accuracy: {100 * correct / len(test_loader.sampler):.2f}%')


def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt'
        , 'Sneaker', 'Bag', 'Ankle Boot']

    with torch.no_grad():
        logits = model(test_images[index])
        probability = F.softmax(logits, dim = 1)
        output = []
        for i, val in enumerate(probability[0], 0):
            output.append((val, class_names[i]))

        output = sorted(output, reverse=True)

        for element in output[:3]:
            print(f'{element[1]}: {100 * element[0]:.2f}%')



if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''



