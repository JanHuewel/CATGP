# -*- coding: utf-8 -*-
"""
Created on Fri May 13 20:48:20 2022

@author: Christian Kiesow, Jan David Hüwel
"""
import gpytorch
import torch
from torch.nn import ModuleList
from gpytorch.lazy import ZeroLazyTensor, lazify
from copy import deepcopy
from gpytorch.constraints import *
from gpytorch.kernels import *
import math
from Constraints import UnitSum
from helpFunctions import get_string_representation_of_kernel as gsr
import matplotlib.pyplot as plt



class WhiteNoiseKernel(gpytorch.kernels.Kernel):
        
    has_lengthscale=True
    
    def __init__(self, noise=1):
        super().__init__()
        self.noise = noise
    
    def forward(self, x1, x2, diag=False, **params):
    
     a=x1.repeat(1,x2.shape[0])
     b=x2.repeat(1,x1.shape[0]).t()
     c=(torch.eq(a,b)).double()
        
     if not diag:
         return c*self.noise
     else:
         return c.diag()*self.noise


class TransitionKernel(gpytorch.kernels.Kernel):
    """
    A kernel that models the change from one kernel to another.

    Parameters: offset s - x-coordinate of change point
                lenght l - lenght of the transition zone

    """
    has_lengthscale = False

    def __init__(self, *kernels, length_prior=None, length_constraint=None, offset_prior=None,
                 offset_constraint=None):
        super(TransitionKernel, self).__init__()
        self.kernels = ModuleList(kernels)

        # register the raw parameter
        self.register_parameter(
            name='raw_offset', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))

        # set the parameter constraint, when nothing is specified
        if offset_constraint is None:
            offset_constraint = Interval(-3, 3, initial_value=0)

        # register the constraint
        self.register_constraint("raw_offset", offset_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if offset_prior is not None:
            self.register_prior(
                "offset_prior",
                offset_prior,
                lambda m: m.offset,
                lambda m, v: m._set_offset(v), )

        # register the raw parameter
        self.register_parameter(
            name='raw_length', parameter=torch.nn.Parameter(torch.zeros(*self.batch_shape, 1, 1)))

        # set the parameter constraint to be positive, when nothing is specified
        if length_constraint is None:
            length_constraint = Positive()

        # register the constraint
        self.register_constraint("raw_length", length_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if length_prior is not None:
            self.register_prior("length_prior", length_prior,
                                lambda m: m.length,
                                lambda m, v: m._set_length(v), )

    @property
    def offset(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_offset_constraint.transform(self.raw_offset)

    @offset.setter
    def offset(self, value):
        return self._set_offset(value)

    def _set_offset(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_offset)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_offset=self.raw_offset_constraint.inverse_transform(value))

    @property
    def length(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_length_constraint.transform(self.raw_length)

    @length.setter
    def length(self, value):
        return self._set_length(value)

    def _set_length(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_length)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_length=self.raw_length_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):

        res = ZeroLazyTensor() if not diag else 0

        first_term = self.kernels[0](x1, x2, diag=diag, **params)

        def sigma(x):
            return 0.5 * (1 + torch.tanh((self.offset - x).div(self.length)))

        if not diag:
            res = res + torch.mul(sigma(x1), sigma(x2).t()) * first_term
        else:
            res = res + (torch.mul(sigma(x1), sigma(x2).t()) * first_term).diag()

        second_term = self.kernels[1](x1, x2, diag=diag, **params)

        if not diag:
            res = res + torch.mul(1 - sigma(x1), (1 - sigma(x2)).t()) * second_term
        else:
            res = res + (torch.mul(1 - sigma(x1), (1 - sigma(x2)).t()) * second_term).diag()

        return res

    def num_outputs_per_input(self, x1, x2):
        return self.kernels[0].num_outputs_per_input(x1, x2)

    def __getitem__(self, index):
        new_kernel = deepcopy(self)
        for i, kernel in enumerate(self.kernels):
            new_kernel.kernels[i] = self.kernels[i].__getitem__(index)

        return new_kernel

class ChangepointKernel(gpytorch.kernels.Kernel):
    """
    A kernel that models the change from one kernel to another.

    Parameters: changepoints x - x-coordinate of change point

    """
    has_lengthscale = False

    def __init__(self, *kernels, changepoints_prior=None,
                 changepoints_constraint=None):
        super(ChangepointKernel, self).__init__()
        self.kernels = ModuleList(kernels)

        # register the raw parameter
        self.register_parameter(
            name='raw_changepoints', parameter=torch.nn.Parameter(torch.linspace(-1,1,len(kernels)-1)))

        # set the parameter constraint, when nothing is specified
        if changepoints_constraint is None:
            changepoints_constraint = Interval(-10, 10)

        # register the constraint
        self.register_constraint("raw_changepoints", changepoints_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if changepoints_prior is not None:
            self.register_prior(
                "changepoints_prior",
                changepoints_prior,
                lambda m: m.changepoints,
                lambda m, v: m._set_changepoints(v), )


    @property
    def changepoints(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_changepoints_constraint.transform(self.raw_changepoints)

    @changepoints.setter
    def changepoints(self, value):
        return self._set_changepoints(value)

    def _set_changepoints(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_changepoints)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(raw_changepoints=self.raw_changepoints_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):

        res = ZeroLazyTensor() if not diag else 0

        def sigma(x, cp):
            return 0.5 * (1 + torch.tanh((cp - x).div(0.05)))

        sorted_changepoints = list(sorted(self.changepoints))
        cps = [-1000] + sorted_changepoints + [1000]

        for i, k in enumerate(self.kernels):
            contribution = k(x1, x2, diag=diag, **params)
            if not diag:
                res = res + torch.mul(1-sigma(x1, cps[i]), 1-sigma(x2, cps[i]).t()) \
                      * torch.mul(sigma(x1, cps[i+1]), sigma(x2, cps[i+1]).t()) \
                      * contribution
            else:
                res = res + (torch.mul(1-sigma(x1, cps[i]), 1-sigma(x2, cps[i]).t()) \
                      * torch.mul(sigma(x1, cps[i + 1]), sigma(x2, cps[i+1]).t()) \
                      * contribution).diag()

        return res

    def num_outputs_per_input(self, x1, x2):
        return self.kernels[0].num_outputs_per_input(x1, x2)

    def __getitem__(self, index):
        new_kernel = deepcopy(self)
        for i, kernel in enumerate(self.kernels):
            new_kernel.kernels[i] = self.kernels[i].__getitem__(index)

        return new_kernel


class FuzzyKernel(gpytorch.kernels.Kernel):
    """
    The FuzzyKernel represents a distribution over possible kernel expressions without taking a definite form.
    This first implementation contains the weights of base kernels and operators as hyperparameters.
    Given an operator, two FuzzyKernels are chosen as children. A maximum depth needs to be defined.
    """
    has_lengthscale = False

    @property
    def is_stationary(self) -> bool:
        return False

    def __init__(self, kernels=[RBFKernel()], operators=[AdditiveKernel, ProductKernel], depth = 3, weight_constraint = None, **kwargs):
        super(FuzzyKernel, self).__init__(**kwargs)

        weights = torch.ones(len(kernels) + len(operators)) if depth > 0 else torch.ones(len(kernels))
        self.register_parameter(name="raw_weights", parameter=torch.nn.Parameter(weights))
        if weight_constraint is None:
            weight_constraint = Positive()
        self.register_constraint("raw_weights", weight_constraint)

        if depth > 0:
            self.kernels = ModuleList(kernels + [FuzzyKernel(kernels, operators, depth-1), FuzzyKernel(kernels, operators, depth-1)])
        else:
            self.kernels = ModuleList(kernels)
        self.depth = depth

    @property
    def weights(self):
        return self.raw_weights_constraint.transform(self.raw_weights)

    @weights.setter
    def weights(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_weights)
        self.initialize(raw_weights=self.raw_weights_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        res = ZeroLazyTensor() if not diag else 0
        for i, kern in enumerate(self.kernels):
            if self.depth > 0 and i == len(self.kernels)-2:
                next_term = self.weights[i] * AdditiveKernel(self.kernels[-2], self.kernels[-1])(x1, x2, diag=diag, **params)
            elif self.depth > 0 and i == len(self.kernels)-1:
                next_term = self.weights[i] * ProductKernel(self.kernels[-2], self.kernels[-1])(x1, x2, diag=diag, **params)
            else:
                next_term = self.weights[i] * kern(x1, x2, diag=diag, **params)
            if not diag:
                res = res + lazify(next_term)
            else:
                res = res + next_term

        return res

    def plot_weights(self):
        fig = plt.figure()
        cols = 2**self.depth
        for depth in range(self.depth+1):
            for col in range(2**depth):
                sub = fig.add_subplot(self.depth+1, 2**depth, depth*(2**depth)+col+1)
                sub.spines["top"].set_visible(False)
                sub.spines["right"].set_visible(False)
                current_kernel = self
                for i in range(depth):
                    try:
                        current_kernel = current_kernel.kernels[-2+int(bin(col)[2+i])]
                    except:
                        current_kernel = current_kernel.kernels[-2]
                if depth == self.depth:
                    sub.bar([k._get_name()[:-6] for k in current_kernel.kernels], current_kernel.weights.detach())
                else:
                    sub.bar([k._get_name()[:-6] for k in current_kernel.kernels[:-2]] + ["+", "x"], current_kernel.weights.detach())



class LinearCombinationKernel(gpytorch.kernels.Kernel):
    """
    direct linear combination of multiple standard kernels
    """
    has_lengthscale = False

    @property
    def is_stationary(self) -> bool:
        """
        Kernel is stationary if all included kernels are stationary.
        """
        return all(kernel.is_stationary for kernel in self.kernels)

    def __init__(self, *kernels, outputscale_prior = None, outputscale_constraint = None, **kwargs):
        super(LinearCombinationKernel, self).__init__(**kwargs)
        self.kernels = ModuleList(kernels)
        if outputscale_constraint is None:
            outputscale_constraint = Positive()

        outputscale = torch.zeros(len(self.kernels))
        self.register_parameter(name="raw_outputscale", parameter=torch.nn.Parameter(outputscale))
        if outputscale_prior is not None:
            self.register_prior(
                "outputscale_prior", outputscale_prior, self._outputscale_param, self._outputscale_closure
            )

        self.register_constraint("raw_outputscale", outputscale_constraint)

    def _outputscale_param(self, m):
        return m.outputscale

    def _outputscale_closure(self, m, v):
        return m._set_outputscale(v)

    @property
    def outputscale(self):
        return self.raw_outputscale_constraint.transform(self.raw_outputscale)

    @outputscale.setter
    def outputscale(self, value):
        self._set_outputscale(value)

    def _set_outputscale(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_outputscale)
        self.initialize(raw_outputscale=self.raw_outputscale_constraint.inverse_transform(value))

    def forward(self, x1, x2, diag=False, **params):
        res = ZeroLazyTensor() if not diag else 0
        for i, kern in enumerate(self.kernels):
            next_term = self.outputscale[i] * kern(x1, x2, diag=diag, **params)
            if not diag:
                res = res + lazify(next_term)
            else:
                res = res + next_term

        return res