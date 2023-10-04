import os
os.chdir('..')

from Datasets import *
from Data import Data
import matplotlib.pyplot as plt
from GaussianProcess import ExactGPModel, ExactStreamGPModel
from kernelSearch import CKS
from gpytorch.kernels import *
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.constraints import GreaterThan, Positive
from gpytorch.priors import HorseshoePrior
import gpytorch as gpt
import torch
import globalParams
from Kernels import LinearCombinationKernel, FuzzyKernel
from Constraints import *
from helpFunctions import print_formatted_hyperparameters as pfh, get_string_representation_of_kernel as gsr
import math

globalParams.options["training"]["optimization method"] = "default"
globalParams.options["training"]["max_iter"] = 400
globalParams.options["training"]["restarts"] = 5
globalParams.options["kernel search"]["print"] = True
globalParams.options["plotting"]["border_ratio"] = 1.0

data_x = torch.linspace(-5,5,50)
# PERIODIC
data_y = torch.sin(data_x * (2))
# LINEAR
#data_y = 0.2 * data_x + 0.05 * torch.rand(data_x.size())
# SE
#data_y = 0.2 * data_x + torch.sin(data_x*3) + torch.cos(data_x*5) + 0.1*torch.rand(data_x.size())

data = Data(data_x, data_y)
#data = dataset_simul_varnoise_2cp_4[::2]
data.normalize_z()

k, l = CKS(data, GaussianLikelihood(noise_constraint=Positive()), [RBFKernel(), PeriodicKernel(), LinearKernel()], 3)
#l = GaussianLikelihood()#GaussianLikelihood(noise_constraint=Positive(), noise_prior=HorseshoePrior(0.1))
#k = PeriodicKernel()
m = ExactGPModel(data, l, k)
m.train()
#m.process_data_cpd("AKS", [ScaleKernel(RBFKernel()), ScaleKernel(PeriodicKernel()), ScaleKernel(LinearKernel())], 2, 5, 10.0, 2.0)
print(gsr(k))
m.plot_model()