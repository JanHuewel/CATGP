import os
os.chdir("..")

import itertools

import Datasets
from GaussianProcess import *
from Kernels import ChangepointKernel
from PatternMining import apriori, subset
from DataGenerator import DataGenerator
from gpytorch.kernels import *
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from helpFunctions import get_string_representation_of_kernel as gsr2, generate_kernel_from_polish_string as generate_kernel
import random
from kernelSearch import sop_CKS, CKS

import torch
import matplotlib.pyplot as plt
import time
from globalParams import options

options["plotting"]["legend"] = False

# globals

def gsr_better(kernel_expression):
    """
    Expects AdditiveKernel as the kernel expression and sorts everything for comparison.
    """
    subkernel_texts = [gsr(k) for k in kernel_expression.kernels]
    subkernel_texts.sort()
    return " + ".join(subkernel_texts)
def gsr(kernel_expression):
    """
    adjusted version of the help function get_string_representation. Here, brackets are omitted as we assume a depth < 2.
    It expects either a base kernel or a product of base kernels.

    Products of kernels are sorted alphabetically.
    """
    if kernel_expression._get_name() == "ProductKernel":
        s = ""
        subkernels = []
        for k in kernel_expression.kernels:
            subkernels.append(gsr(k))
        subkernels.sort()
        for k in subkernels:
            s += k + " * "
        return "" + s[:-3] + ""
    elif kernel_expression._get_name() == "ScaleKernel":
        return f"(c * {gsr(kernel_expression.base_kernel)})"
    elif kernel_expression._get_name() == "RBFKernel":
        return "SE"
    elif kernel_expression._get_name() == "LinearKernel":
        return "LIN"
    elif kernel_expression._get_name() == "PeriodicKernel":
        return "PER"
    else:
        return kernel_expression._get_name()
def gpm_to_itemset(gp):
    """
    assumes that the given gp is in a sum-of-products form
    """
    if not hasattr(gp, "kernels"):
        return gsr(gp)
    ret = []
    for summand in gp.kernels:
        ret.append(gsr(summand))
    return ret

def kernel_to_sop(k):
    """
    transforms a kernel expression k into sum-of-products form.
    """
    products = _kernel_to_product_set(k)
    return AdditiveKernel(*[ProductKernel(*x) for x in products])

def _kernel_to_product_set(k):
    """
    helper function for kernel_to_sop.
    """
    if not hasattr(k, "kernels"):
        return [[k]]
    elif k._get_name() == "AdditiveKernel":
        ret = []
        for x in k.kernels:
            ret.extend(_kernel_to_product_set(x))
        return ret
    elif k._get_name() == "ProductKernel":
        ret = []
        children = []
        for subkernel in k.kernels:
            children.append(_kernel_to_product_set(subkernel))
        cart_prod = itertools.product(*children)
        for curr_prod in cart_prod:
            joint_element = []
            for x in curr_prod:
                joint_element.extend(x)
            ret.append(joint_element)
        return ret


def test():
    #kernel = (LinearKernel() + (PeriodicKernel() * RBFKernel() + LinearKernel()) * PeriodicKernel()) * MaternKernel() + LinearKernel()
    kernel = (LinearKernel() + PeriodicKernel()) * (PeriodicKernel() + RBFKernel())
    print(gsr2(kernel))
    print(gsr2(kernel_to_sop(kernel)))

def main():
    data = Datasets.dataset_d8_temperature
    title = "Temperature"
    label_x = "Normalized time"
    label_y = "Normalized temperature measurements"
    base_kernels = [RBFKernel(), PeriodicKernel(), LinearKernel()]
    iterations = 5
    window_size = 50
    step_size = 30
    minSupp = 0.1


    data.normalize_z()
    models = []
    likelihoods = []

    # use kernel search on windows to determine models
    for i in range(0,len(data)-window_size, step_size):
        print(i)
        window = data[i:i+window_size]
        window.normalize_z()
        k, l = CKS(window, GaussianLikelihood(), base_kernels, iterations)
        models.append(kernel_to_sop(k))
        likelihoods.append(l)

    # convert models into item sets and perform fim
    pattern_database = [gpm_to_itemset(k) for k in models]
    frequent_patterns = apriori(pattern_database, minSupp)
    print(f"frequent_patterns: {frequent_patterns}")
    frequent_patterns.sort(key=(lambda x : x[1]), reverse=True)
    print(f"frequent_patterns: {frequent_patterns}")

    # prepare plots by finding exemplary subsequences for each frequent pattern
    example_indeces = []
    for pattern in frequent_patterns:
        for i in range(len(pattern_database)):
            if subset(pattern[0], pattern_database[i]):
            #if all([x in pattern_database[i] for x in pattern[0]]):
                example_indeces.append(i)
                break

    models_for_plotting = []
    for i in range(min(4, len(frequent_patterns))):
        print("Adding model")
        index = example_indeces[i]
        #model = models[index]
        string_model = [x.split(' * ') for x in frequent_patterns[i][0]]
        model = AdditiveKernel(*[ProductKernel(*[generate_kernel(s) for s in m]) for m in string_model])
        data_for_plotting = data[index*step_size:index*step_size+window_size]
        data_for_plotting.normalize_z()
        models_for_plotting.append(ExactGPModel(data_for_plotting, likelihoods[index], model))

    for model in models_for_plotting:
        model.optimize_hyperparameters()

    # plot the data and the found patterns
    f, axs = plt.subplot_mosaic([["data", "data", "data", "data"], ["comp1", "comp2", "comp3", "comp4"]], constrained_layout=True)
    axs["data"].plot(data.X, data.Y, label=[label_x, label_y])
    f.suptitle(title)

    if len(models_for_plotting) > 0:
        models_for_plotting[0].plot_model(figure=f, ax=axs["comp1"], return_figure=True)
        axs["comp1"].set_title("+".join(frequent_patterns[0][0]))
    if len(models_for_plotting) > 1:
        models_for_plotting[1].plot_model(figure=f, ax=axs["comp2"], return_figure=True)
        axs["comp2"].set_title("+".join(frequent_patterns[1][0]))
    if len(models_for_plotting) > 2:
        models_for_plotting[2].plot_model(figure=f, ax=axs["comp3"], return_figure=True)
        axs["comp3"].set_title("+".join(frequent_patterns[2][0]))
    if len(models_for_plotting) > 3:
        models_for_plotting[3].plot_model(figure=f, ax=axs["comp4"], return_figure=True)
        axs["comp4"].set_title("+".join(frequent_patterns[3][0]))

    plt.show()



if __name__=="__main__":
    main()