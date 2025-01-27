import torch
from torchvision import transforms, datasets, models
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import seaborn as sn
import pandas as pd
from copy import copy

from torchvision.models.alexnet import AlexNet_Weights


BATCH_SIZE = int (32)
SPLIT_RATIO_TRAINING = float (0.85)
EPOCHS_LIMIT = int (200)






def task_3():
    training_losses=[]
    training_accuracies=[]
    validation_losses=[]
    validation_accuracies=[]
    best_validation_loss = np.inf
    no_improvement_counter=0
    

    # Define transformations and datasets
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])             # Normalize image to ImageNet mean and std
    ])

    data_augmentation_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),                                                      # Randomly flip image horizontally
        transforms.RandomRotation(degrees=15),                                                  # Randomly rotate image
        transforms.RandomChoice([                                                               # Randomly choose whether apply anisotropic rescaling or random cropping (2 to 1 ratio).  
            transforms.Resize((224, 224)),
            transforms.Resize((224, 224)),
            transforms.Compose([
                transforms.RandomCrop(180),
                transforms.Resize(224)
            ])
        ]),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])             # Normalize image to ImageNet mean and std
    ])



    full_training_data = datasets.ImageFolder(root="Dataset"+"/train")
    test_dataset = datasets.ImageFolder(root="Dataset"+"/test")

    train_size = int(SPLIT_RATIO_TRAINING * len(full_training_data))
    val_size = len(full_training_data) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_training_data, [train_size, val_size])

    train_dataset.dataset = copy(full_training_data)

    train_dataset.dataset.transform = data_augmentation_transform
    val_dataset.dataset.transform = transform
    test_dataset.transform=transform

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    validation_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
                       
    model =CNN_task_3()
    no_improvement_counter_limit= int (25)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.002,momentum=0.9)  # stochastic gradient descent
    loss = torch.nn.CrossEntropyLoss()

    # best_model = model.state_dict()

    for epoch in range(EPOCHS_LIMIT):
        #TRAINING
        print('EPOCH {}:'.format(epoch+ 1))
    
        #train one epoch
        model.train(True)
        for x, y in iter(train_loader):
            optimizer.zero_grad()# notice that by default, the gradients are accumulated, hence we need to set them to zero
            y_pred = model(x)
            l = loss(y_pred, y) #compute the loss
            l.backward() #backward pass
            optimizer.step() #update the weights



        training_loss, training_accuracy = calculate_loss_accuracy(model, train_loader, loss)
        training_losses.append(training_loss)
        training_accuracies.append(training_accuracy)
        print(f"Training loss: {training_loss}")
        print(f"Training accuracy: {training_accuracy*100}%")

        #VALIDATION
        model.eval()
        with torch.no_grad():   # Disable gradient computation and reduce memory consumption
            validation_loss, validation_accuracy = calculate_loss_accuracy(model, validation_loader, loss)
            validation_losses.append(validation_loss)
            validation_accuracies.append(validation_accuracy)
            print(f"Validation loss: {validation_loss}")
            print(f"Validation accuracy: {validation_accuracy*100}%")
            if validation_loss < best_validation_loss:
                print(f"!!! NEW BEST MODEL FOUND - validation loss: {validation_loss} & validation accuracy: {validation_accuracy*100}%")
                best_model = model.state_dict()
                best_validation_loss = validation_loss
                no_improvement_counter=0
            else:
                no_improvement_counter +=1
                if no_improvement_counter==no_improvement_counter_limit: 
                    break #exit here

    
    #LOAD THE BEST MODEL
    model.load_state_dict(best_model)
    torch.save(model.alexnet.classifier[6].state_dict(), "model_task_3.pt")
    
    #TEST MODEL
    total = 0
    correct = 0
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for x_test, y_test in test_loader:
            y_pred_test = model(x_test)
            _, predicted = torch.max(y_pred_test.data, 1)
            total += y_test.size(0)
            correct += (predicted == y_test).sum().item()
            all_predictions.extend(predicted.numpy())
            all_labels.extend(y_test.numpy())

    test_accuracy = 100 * correct / total
    print(f"Accuracy of the network on the test images: {test_accuracy}%")

    # Save plot
    save_loss_accuracy_plot(
        training_losses, 
        validation_losses, 
        training_accuracies, 
        validation_accuracies,  
        "loss_and_accuracy_task_3.png"
    )

    save_confusion_matrix(
        confusion_matrix(all_labels, all_predictions),
        test_accuracy,
        "confusion_matrix_task_3.png",
        test_dataset
    )





def save_loss_accuracy_plot(train_losses : list[float], val_losses : list[float], train_accuracies : list[float], 
                            val_accuracies : list[float], save_path : str) -> None:
    # Plot training and validation loss
    plt.figure(figsize=(16, 9))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(train_losses, label='Training Loss')
    ax1.plot(val_losses, label='Validation Loss')
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(train_accuracies, label='Training Accuracy')
    ax2.plot(val_accuracies, label='Validation Accuracy')
    ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.set_ylim([0, 1])
    ax2.legend()

    #plt.figtext(0.5, 0.99, values_str, ha="center", va="top", fontsize=12)

    plt.subplots_adjust(top=0.95,left=0.05,right=0.95)
    plt.savefig(save_path)
    plt.close()



def save_confusion_matrix(confusion_matrix : list[list[int]], test_accuracy : float, save_path : str, test_dataset) -> None:
    plt.figure(figsize=(16, 16))

    classes=test_dataset.classes
    cf_matrix = confusion_matrix
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])

    sn.heatmap(df_cm, annot=True, cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.figtext(0.5, 0.01, f'Final Test Accuracy: {test_accuracy:.3f}%', ha="center", fontsize=14, bbox={"facecolor":"orange", "alpha":0.5, "pad":5})
    plt.subplots_adjust(top=0.95,left=0.1,right=1.0)
    plt.savefig(save_path)
    plt.close()



def calculate_loss_accuracy(model : torch.nn.Module, loader : DataLoader, loss_function : torch.nn.modules.loss._Loss)-> tuple[float, float]:
    model.eval()
    running_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in iter(loader):
            y_pred = model(x)
            loss = loss_function(y_pred, y)
            running_loss += loss.item()
            _, predicted = torch.max(y_pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    running_loss /= len(loader)
    accuracy = correct / total
    return (running_loss, accuracy)





class CNN_task_3(nn.Module):

    def __init__(self, mean_initialization : float = 0.0, std_initialization : float = 0.01):
        super().__init__()


        self.mean_initialization = mean_initialization
        self.std_initialization = std_initialization

        self.alexnet = models.alexnet(weights=AlexNet_Weights.DEFAULT)
        self.alexnet.classifier[6] = nn.Linear(4096, 15)
        # Freeze all layers except the last one:
        for param in self.alexnet.parameters():
            param.requires_grad = False
        self.alexnet.classifier[6].apply(self._init_weights)
        self.alexnet.classifier[6].requires_grad = True
        self.alexnet.classifier[6].weight.requires_grad = True
        self.alexnet.classifier[6].bias.requires_grad = True

    def _init_weights(self, m):
        if type(m) == nn.Conv2d or type(m) == nn.Linear:
            nn.init.normal_(m.weight, mean=self.mean_initialization, std=self.std_initialization)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)

    def forward(self, x):
        return self.alexnet(x)
