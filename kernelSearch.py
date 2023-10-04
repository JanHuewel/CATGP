import gpytorch as gpt
import torch
from GaussianProcess import ExactGPModel
from helpFunctions import get_string_representation_of_kernel as gsr, clean_kernel_expression, print_formatted_hyperparameters as pfh
from helpFunctions import amount_of_base_kernels
from gpytorch.kernels import ScaleKernel
from Kernels import *
import threading
import copy

from globalParams import options

# ----------------------------------------------------------------------------------------------------
# ----------------------------------------HELP FUNCTIONS----------------------------------------------
# ----------------------------------------------------------------------------------------------------
def replace_internal_kernels(base, kernels):
    ret = []
    # if the current base is a (scaled) base kernel, replace it with any other base kernel
    if not hasattr(base, "kernels"):
        return [k for k in kernels if k is not None and not gsr(k)==gsr(base)]

    # if the current base is an operator, apply this function to each child
    elif hasattr(base, "kernels"):
        for position in range(len(base.kernels)):
            for k in replace_internal_kernels(base.kernels[position], kernels):
                new_expression = copy.deepcopy(base)
                new_expression.kernels[position] = k
                ret.append(new_expression)
            # additionally, if kernels should also be removed, check if there is an endpoint and remove it
            if None in kernels and not hasattr(base.kernels[position], "kernels"):
                if len(base.kernels) == 2:
                    ret.append(base.kernels[not position])
                elif len(base.kernels) > 2:
                    new_expression = copy.deepcopy(base)
                    old_kernels = list(new_expression.kernels)
                    old_kernels.pop(position)
                    new_expression.kernels = torch.nn.ModuleList(old_kernels)
                    ret.append(new_expression)
    for expression in ret:
        clean_kernel_expression(expression)
    return ret

def extend_internal_kernels(base, kernels, operations):
    ret = []
    for op in operations:
        ret.extend([op(base, k) for k in kernels])
    if hasattr(base, "kernels"):
        for position in range(len(base.kernels)):
            for k in extend_internal_kernels(base.kernels[position], kernels, operations):
                new_expression = copy.deepcopy(base)
                new_expression.kernels[position] = k
                ret.append(new_expression)
#    elif hasattr(base, "base_kernel"):
#        for k in extend_internal_kernels(base.base_kernel, kernels, operations):
#            new_expression = copy.deepcopy(base)
#            new_expression.base_kernel = k
#            ret.append(new_expression)
    return ret


# ----------------------------------------------------------------------------------------------------
# ----------------------------------------CANDIDATE FUNCTIONS-----------------------------------------
# ----------------------------------------------------------------------------------------------------
def create_candidates_CKS(base, kernels, operations):
    ret = []
    ret.extend(list(set(extend_internal_kernels(base, kernels, operations))))
    ret.extend(list(set(replace_internal_kernels(base, kernels))))
    return ret

def create_candidates_sop_CKS(base, kernels):
    ret = []
    for b in kernels:
        for i in range(len(base.kernels)):
            base_copy = copy.deepcopy(base)
            base_copy.kernels[i].kernels.append(copy.deepcopy(b))
            ret.append(copy.deepcopy(base_copy))
        base_copy = copy.deepcopy(base)
        base_copy.kernels.append(ProductKernel(copy.deepcopy(b)))
        ret.append(copy.deepcopy(base_copy))
    return ret

def create_candidates_SKC(base, kernels):
    ret = []
    ret.extend([AdditiveKernel(k, base) for k in kernels])
    ret.extend([ProductKernel(k, base) for k in kernels])
    return ret

def create_candidates_AKS(base, kernels, operations, max_complexity=5):
    ret = []
    if max_complexity and amount_of_base_kernels(base) < max_complexity:
        ret.extend(extend_internal_kernels(base, kernels, operations))
    ret.extend((replace_internal_kernels(base, [None] + kernels)))
    ret.append(base)
    ret = list(set(ret))
    return ret
# ----------------------------------------------------------------------------------------------------
# ----------------------------------------EVALUATE FUNCTIONS------------------------------------------
# ----------------------------------------------------------------------------------------------------
def evaluate_performance_via_likelihood(model):
    return - model.get_current_loss()

def SKC_lower_bound(model, inducing_points):
    # implements the VAR (variational inducing points) method
    # index defines equidistant indeces for the data, ranging from first to last element
    index = torch.round(torch.linspace(0, len(model.train_targets)-1, inducing_points)).type(torch.long)
    X_full = model.train_inputs[0].flatten()
    X = X_full[index]
    Y_full = model.train_targets
    K_NN = model.covar_module(X_full, X_full)
    K_Nm = model.covar_module(X_full, X)
    K_mm = model.covar_module(X,X)
    jitter = 1e-1 * torch.eye(len(X_full))
    K2 = K_Nm.evaluate() @ torch.linalg.pinv(K_mm.evaluate()) @ K_Nm.t().evaluate()
    dist = torch.distributions.MultivariateNormal(torch.zeros(len(X_full)), K2 + model.likelihood.noise * torch.eye(len(X_full)) + jitter)
    return dist.log_prob(Y_full) + torch.trace((K_NN - K2).evaluate()) / (2 * model.likelihood.noise)

def SKC_upper_bound(model, inducing_points):
    # index defines equidistant indeces for the data, ranging from first to last element
    index = torch.round(torch.linspace(0, len(model.train_targets) - 1, inducing_points)).type(torch.long)
    X_full = model.train_inputs[0].flatten()
    X = X_full[index]
    Y_full = model.train_targets
    K_NN = model.covar_module(X_full, X_full)
    K_Nm = model.covar_module(X_full, X)
    K_mm = model.covar_module(X, X)
    K2 = K_Nm @ torch.linalg.pinv(K_mm.evaluate()) @ K_Nm.t()
    NLD = -0.5 * torch.logdet(K2 + model.likelihood.noise * torch.eye(len(X_full)))
    def NIP(alpha):
        return 0.5 * alpha.t() @ (K_NN + model.likelihood.noise * torch.eye(len(X_full))) @ alpha - alpha.t() @ Y_full
    # theoretical optimum of NIP is the actual likelihood, which we use here instead of a CG optimizer result
    alpha_opt = torch.inverse((K_NN + model.likelihood.noise * torch.eye(len(X_full))).evaluate()) @ Y_full
    return NLD + NIP(alpha_opt)


def AIC(model):
    # Assumes that the model is optimized
    return 2 * len(list(model.parameters())) + 2 * model.get_current_loss() * model.train_inputs[0].numel()

def BIC(model):
    return len(list(model.parameters())) * torch.log(torch.tensor(len(model.train_targets))) + 2 * model.get_current_loss() * len(model.train_targets)

# ----------------------------------------------------------------------------------------------------
# ------------------------------------------- CKS ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------
def CKS(training_data, likelihood, base_kernels, iterations, **kwargs):
    operations = [gpt.kernels.AdditiveKernel, gpt.kernels.ProductKernel]
    candidates = base_kernels.copy()
    best_performance = dict()
    models = dict()
    performance = dict()
    threads = list()
    for i in range(iterations):
        for k in candidates:
            if gsr(k) in models:
                continue
            models[gsr(k)] = ExactGPModel(training_data, copy.deepcopy(likelihood), copy.deepcopy(k))
            if options["kernel search"]["multithreading"]:
                threads.append(threading.Thread(target=models[gsr(k)].optimize_hyperparameters))
                threads[-1].start()
            else:
                try:
                    models[gsr(k)].optimize_hyperparameters()
                except:
                    print(f"Failure in model selection. Current model: {gsr(k)}\n\n {pfh(k)}")
                    exit()
        for t in threads:
            t.join()
        for k in candidates:
            performance[gsr(k)] = evaluate_performance_via_likelihood(models[gsr(k)])
            if options["kernel search"]["print"]:
                print(f"KERNEL SEARCH: iteration {i} checking {gsr(k)}, loss {-performance[gsr(k)]}")
        if len(best_performance) > 0:
            if best_performance["performance"] >= max(performance.values()):
                if options["kernel search"]["print"]:
                    print("KERNEL SEARCH: no gain through additional kernel length, stopping search")
                break
        best_model = models[max(performance, key=performance.__getitem__)]
        best_performance = {"model": best_model, "performance": max(performance.values())}
        candidates = create_candidates_CKS(best_model.covar_module, base_kernels, operations)
    if options["kernel search"]["print"]:
        print(f"KERNEL SEARCH: kernel search concluded, optimal expression: {gsr(best_model.covar_module)}")
    return best_model.covar_module, best_model.likelihood

def sop_CKS(training_data, likelihood, base_kernels, iterations, **kwargs):
    candidates = [AdditiveKernel(ProductKernel(copy.deepcopy(b))) for b in base_kernels]
    best_performance = dict()
    models = dict()
    performance = dict()
    threads = list()
    for i in range(iterations):
        for k in candidates:
            models[gsr(k)] = ExactGPModel(training_data, copy.deepcopy(likelihood), copy.deepcopy(k))
            if options["kernel search"]["multithreading"]:
                def mt_opt():
                    try:
                        models[gsr(k)].optimize_hyperparameters()
                    except:
                        print(gsr(k))
                threads.append(threading.Thread(target=models[gsr(k)].optimize_hyperparameters))
                #threads.append(threading.Thread(target=mt_opt))
                threads[-1].start()
            else:
                models[gsr(k)].optimize_hyperparameters()
        for t in threads:
            t.join()
        for k in candidates:
            performance[gsr(k)] = evaluate_performance_via_likelihood(models[gsr(k)])
            if options["kernel search"]["print"]:
                print(f"KERNEL SEARCH: iteration {i} checking {gsr(k)}, loss {-performance[gsr(k)]}")
        if len(best_performance) > 0:
            if best_performance["performance"] >= max(performance.values()):
                if options["kernel search"]["print"]:
                    print("KERNEL SEARCH: no gain through additional kernel length, stopping search")
                break
        best_model = models[max(performance, key=performance.__getitem__)]
        best_performance = {"model": best_model, "performance": max(performance.values())}
        candidates = create_candidates_sop_CKS(best_model.covar_module, base_kernels)
    if options["kernel search"]["print"]:
        print(f"KERNEL SEARCH: kernel search concluded, optimal expression: {gsr(best_model.covar_module)}")
    return best_model.covar_module, best_model.likelihood

# ----------------------------------------------------------------------------------------------------
# ------------------------------------------- ABCD ---------------------------------------------------
# ----------------------------------------------------------------------------------------------------
"""
The Automatic Bayesian Covariance Discovery algorithm originally includes a CKS variant for model selection and 
translation of the resulting kernel into natural language descriptions. This implementation only includes the model
selection part.
"""
def ABCD(training_data, likelihood, base_kernels, iterations, **kwargs):
    operations = [gpt.kernels.AdditiveKernel, gpt.kernels.ProductKernel, ChangepointKernel]
    candidates = base_kernels.copy()
    best_performance = dict()
    models = dict()
    performance = dict()
    threads = list()
    for i in range(iterations):
        for k in candidates:
            models[gsr(k)] = ExactGPModel(training_data, copy.deepcopy(likelihood), copy.deepcopy(k))
            if options["kernel search"]["multithreading"]:
                threads.append(threading.Thread(target=models[gsr(k)].optimize_hyperparameters))
                threads[-1].start()
            else:
                models[gsr(k)].optimize_hyperparameters()
        for t in threads:
            t.join()
        for k in candidates:
            performance[gsr(k)] = evaluate_performance_via_likelihood(models[gsr(k)])
            if options["kernel search"]["print"]:
                print(f"KERNEL SEARCH: iteration {i} checking {gsr(k)}, loss {-performance[gsr(k)]}")
        if len(best_performance) > 0:
            if best_performance["performance"] >= max(performance.values()):
                if options["kernel search"]["print"]:
                    print("KERNEL SEARCH: no gain through additional kernel length, stopping search")
                break
        best_model = models[max(performance, key=performance.__getitem__)]
        best_performance = {"model": best_model, "performance": max(performance.values())}
        candidates = create_candidates_CKS(best_model.covar_module, base_kernels, operations)
    if options["kernel search"]["print"]:
        print(f"KERNEL SEARCH: kernel search concluded, optimal expression: {gsr(best_model.covar_module)}")
    return best_model.covar_module, best_model.likelihood


# ----------------------------------------------------------------------------------------------------
# ------------------------------------------- SKC ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------
def SKC(training_data, likelihood, base_kernels, iterations, inducing_points, kernel_buffer_size, **kwargs):
    # still need to fix jitter problem in lower bound
    def optimization_evaluation(model):
        return SKC_lower_bound(model, inducing_points)
    K = [] # kernel buffer
    C = [] # candidates
    init_candidates = base_kernels.copy()
    lowers = dict()
    uppers = dict()
    likelihoods = dict()
    for kernel in init_candidates:
        model = ExactGPModel(training_data, copy.deepcopy(likelihood), kernel)
        model.optimize_hyperparameters(optimization_evaluation)
        lowers[kernel] = SKC_lower_bound(model, inducing_points)
        uppers[kernel] = SKC_upper_bound(model, inducing_points)
        likelihoods[kernel] = model.likelihood
    k = max(uppers, key=uppers.get) # k is the base_kernel with the highest upper bound
    for i in range(iterations):
        K = [k]
        C.sort(key=uppers.get)
        for kernel in C[max(-5, -len(C)):]:
            if uppers[kernel] > lowers[k]:
                K.append(kernel)
        C = []
        for kernel in K:
            C.extend(create_candidates_SKC(kernel, base_kernels))
        for kernel in C:
            model = ExactGPModel(training_data, copy.deepcopy(likelihood), kernel)
            model.optimize_hyperparameters(optimization_evaluation)
            lowers[kernel] = SKC_lower_bound(model, inducing_points)
            uppers[kernel] = SKC_upper_bound(model, inducing_points)
            likelihoods[kernel] = model.likelihood
        k = max(uppers, key=uppers.get)
        print(f"K: {K}\n\nC: {C}\n\nk: {k}")
    return k, likelihoods[k]

# ----------------------------------------------------------------------------------------------------
# ------------------------------------------- AKS ----------------------------------------------------
# ----------------------------------------------------------------------------------------------------
def AKS(base_expression, training_data, likelihood, base_kernels, iterations, max_complexity = 99):
    operations = [gpt.kernels.AdditiveKernel, gpt.kernels.ProductKernel]
    candidates = create_candidates_AKS(base_expression, base_kernels, operations, max_complexity)
    best_performance = dict()
    models = dict()
    performance = dict()
    threads = list()
    for i in range(iterations):
        for k in candidates:
            models[gsr(k)] = ExactGPModel(training_data, copy.deepcopy(likelihood), copy.deepcopy(k))
            if options["kernel search"]["multithreading"]:
                threads.append(threading.Thread(target=models[gsr(k)].optimize_hyperparameters))
                threads[-1].start()
            else:
                models[gsr(k)].optimize_hyperparameters()
        for t in threads:
            t.join()
        for k in candidates:
            performance[gsr(k)] = evaluate_performance_via_likelihood(models[gsr(k)])
            if options["kernel search"]["print"]:
                print(f"KERNEL SEARCH: iteration {i} checking {gsr(k)}, loss {performance[gsr(k)]}")
        if len(best_performance) > 0:
            if best_performance["performance"] >= max(performance.values()):
                if options["kernel search"]["print"]:
                    print("KERNEL SEARCH: no gain through additional kernel length, stopping search")
                break
        best_model = models[max(performance, key=performance.__getitem__)]
        best_performance = {"model": best_model, "performance": max(performance.values())}
        candidates = create_candidates_AKS(best_model.covar_module, base_kernels, operations, max_complexity)
    if options["kernel search"]["print"]:
        print(f"KERNEL SEARCH: kernel search concluded, optimal expression: {gsr(best_model.covar_module)}")
    return best_model.covar_module, best_model.likelihood

