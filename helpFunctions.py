import defaultOptions
import torch
import numpy as np


def get_kernels_in_kernel_expression(kernel_expression):
    """
    returns list of all base kernels in a kernel expression
    """
    if kernel_expression == None:
        return []
    if hasattr(kernel_expression, "kernels"):
        ret = list()
        for kernel in kernel_expression.kernels:
            ret.extend(get_kernels_in_kernel_expression(kernel))
        return ret
    elif kernel_expression._get_name() == "ScaleKernel":
        return get_kernels_in_kernel_expression(kernel_expression.base_kernel)
    elif kernel_expression._get_name() == "GridKernel":
        return get_kernels_in_kernel_expression(kernel_expression.base_kernel)
    else:
        return [kernel_expression]


def get_string_representation_of_kernel(kernel_expression):
    if kernel_expression._get_name() == "AdditiveKernel":
        s = ""
        for k in kernel_expression.kernels:
            s += get_string_representation_of_kernel(k) + " + "
        return "(" + s[:-3] + ")"
    elif kernel_expression._get_name() == "ProductKernel":
        s = ""
        for k in kernel_expression.kernels:
            s += get_string_representation_of_kernel(k) + " * "
        return "(" + s[:-3] + ")"
    elif kernel_expression._get_name() == "ScaleKernel":
        return f"(c * {get_string_representation_of_kernel(kernel_expression.base_kernel)})"
    elif kernel_expression._get_name() == "RBFKernel":
        return "SE"
    elif kernel_expression._get_name() == "LinearKernel":
        return "LIN"
    elif kernel_expression._get_name() == "PeriodicKernel":
        return "PER"
    else:
        return kernel_expression._get_name()

def print_formatted_hyperparameters(kernel_expression):
    for kernel in get_kernels_in_kernel_expression(kernel_expression):
        print(kernel._get_name())
        for name, param, constraint in kernel.named_parameters_and_constraints():
            print(f"\t{name[name.rfind('.raw')+5:]:13}: {constraint.transform(param.data).item()}")

def unraw_parameter_names(generator):
    for param_name, param in generator:
        if ".raw_" not in param_name:
            yield param_name
        else:
            yield param_name.replace(".raw_", ".")

def depth_of_kernel(kernel):
    if not hasattr(kernel, "kernels"):
        return 1
    else:
        return 1 + max([depth_of_kernel(k) for k in kernel.kernels])

def amount_of_base_kernels(kernel):
    if hasattr(kernel, "base_kernel"):
        return amount_of_base_kernels(kernel.base_kernel)
    elif not hasattr(kernel, "kernels"):
        return 1
    else:
        return sum([amount_of_base_kernels(k) for k in kernel.kernels])

def kernel_contains(kernel, segment):
    """
    Returns whether or not a given kernel expression contains a subexpression as a segment, regardless of HPs
    Args:
        kernel: a kernel expression
        segment: a kernel expression

    Returns: bool
    """
    return get_string_representation_of_kernel(segment)[1:-1] in get_string_representation_of_kernel(kernel)

def show_kernel_as_tree(kernel):
    from treelib import Tree
    t = Tree()
    def get_tree(kernel, tree, parent=None, index=[]):
        tree.create_node(kernel._get_name(), str(len(index)), parent=parent)
        current_index = str(len(index))
        index.append(0)
        if hasattr(kernel, "base_kernel"):
            get_tree(kernel.base_kernel, tree, current_index)
            index.append(0)
        elif hasattr(kernel, "kernels"):
            for i in range(len(kernel.kernels)):
                get_tree(kernel.kernels[i], tree, current_index)
                index.append(0)
    get_tree(kernel, t)
    t.show()

def clean_kernel_expression(kernel):
    # option 1: scale(scale(...))
    if kernel._get_name() == "ScaleKernel":
        if kernel.base_kernel._get_name() == "ScaleKernel":
            kernel.base_kernel = kernel.base_kernel.base_kernel
    elif hasattr(kernel, "kernels"):
        for pos, sub_expression in enumerate(kernel.kernels):
            clean_kernel_expression(sub_expression)


def calculate_laplace(model, loss_of_model, variances_list=None):
    num_of_observations = len(*model.train_inputs)
    # Save a list of model parameters and compute the Hessian of the MLL
    params_list = [p for p in model.parameters()]
    mll         = (num_of_observations * (-loss_of_model))
    env_grads   = torch.autograd.grad(mll, params_list, retain_graph=True, create_graph=True)
    hess_params = []
    for i in range(len(env_grads)):
            hess_params.append(torch.autograd.grad(env_grads[i], params_list, retain_graph=True))

    # theta_mu is a vector of parameter priors
    theta_mu = torch.tensor([1 for p in range(len(params_list))]).reshape(-1,1)

    # sigma is a matrix of variance priors
    sigma = []
    if variances_list is None:
        variances_list = [4 for i in range(len(list(model.parameters())))]
    for i in range(len(params_list)):
        line = (np.zeros(len(params_list))).tolist()
        line[i] = variances_list[i]
        sigma.append(line)
    sigma = torch.tensor(sigma)


    params = torch.tensor(params_list).clone().reshape(-1,1)
    hessian = torch.tensor(hess_params).clone()


    # Here comes what's wrapped in the exp-function:
    thetas_added = params+theta_mu
    thetas_added_transposed = (params+theta_mu).reshape(1,-1)
    middle_term = (sigma.inverse()-hessian).inverse()
    matmuls    = torch.matmul( torch.matmul( torch.matmul( torch.matmul(thetas_added_transposed, sigma.inverse()), middle_term ), hessian ), thetas_added )


    # We can calculate by taking the log of the fraction:
    #fraction = 1 / (sigma.inverse()-hessian).det().sqrt() / sigma.det().sqrt()
    #laplace = mll + torch.log(fraction) + (-1/2) * matmuls

    # Conveniently we can also just express the fraction as a sum:
    laplace = mll - (1/2)*torch.log(sigma.det()) - (1/2)*torch.log( (sigma.inverse()-hessian).det() )  + (-1/2) * matmuls

    return laplace

def generate_kernel_from_polish_string(s: str):
    from gpytorch.kernels import ScaleKernel, AdditiveKernel, ProductKernel, RBFKernel, PeriodicKernel, LinearKernel
    def get_corresponding_kernel(list_of_elements):
        try:
            name = list_of_elements.pop(0)
        except:
            return None
        match name:
            case "*":
                return ProductKernel(get_corresponding_kernel(list_of_elements), get_corresponding_kernel(list_of_elements))
            case "+":
                return AdditiveKernel(get_corresponding_kernel(list_of_elements), get_corresponding_kernel(list_of_elements))
            case "c":
                return ScaleKernel(get_corresponding_kernel(list_of_elements))
            case "SE":
                return RBFKernel()
            case "RBF":
                return RBFKernel()
            case "PER":
                return PeriodicKernel()
            case "LIN":
                return LinearKernel()
    s2 = s.split(' ')
    return get_corresponding_kernel(s2)