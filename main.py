import torch
import numpy as np
from scipy import stats
import random

from tasks import task1,task2_individual_network,task2_ensemble_of_networks,task3

torch.manual_seed(314)

# Tasks to run:
task_1 : bool = False

task_2_IndividualNetwork : bool = True
task_2_reg : bool = False

task_2_EnsembleOfNetworks: bool = False
number_of_networks=int (10)

task_3 : bool = False

if task_1:
    task1.task_1()
    print("fine task 1")

if task_2_IndividualNetwork:
    task2_individual_network.task_2(task_2_reg)
    if task_2_reg:
        print("fine task 2 - aug with reg")
    else:
        print("fine task 2 - aug")
if task_2_EnsembleOfNetworks:
    task2_ensemble_of_networks.task_2(number_of_networks)
    print("fine task 2 - ensemble of network")


print("fine task 2")

if task_3:
    task3.task_3()
    print("fine task 3")