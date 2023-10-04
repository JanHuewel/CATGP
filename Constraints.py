from gpytorch.constraints.constraints import Interval
import torch
from torch.nn import Module
from torch import sigmoid
from gpytorch.utils.transforms import inv_sigmoid

def normalize_tensor(x):
    a = torch.sum(x)
    return x/a, a

def inv_normalize_tensor(x,a):
    return x*a

class UnitSum(Module):
    """
    Parameter constraint that guarantees that a tensor has the sum 1
    """
    def __init__(self, initial_value=None):
        super().__init__()

        self._transform = normalize_tensor
        self._inv_transform = inv_normalize_tensor

        self.scaling = 1

        if initial_value is not None:
            if not isinstance(initial_value, torch.Tensor):
                initial_value = torch.tensor(initial_value)
            self._initial_value = self.inverse_transform(initial_value)
        else:
            self._initial_value = None

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        result = super()._load_from_state_dict(
            state_dict=state_dict,
            prefix=prefix,
            local_metadata=local_metadata,
            strict=False,
            missing_keys=missing_keys,
            unexpected_keys=unexpected_keys,
            error_msgs=error_msgs,
        )
        # The lower_bound and upper_bound buffers are new, and so may not be present in older state dicts
        # Because of this, we won't have strict-mode on when loading this module
        return result

    @property
    def enforced(self):
        return self._transform is not None

    def check(self, tensor):
        return bool(torch.sum(tensor) == 1)

    def check_raw(self, tensor):
        return bool(
            torch.sum(self.transform(tensor) == 1)
        )

    def transform(self, tensor):
        if not self.enforced:
            return tensor

        transformed_tensor, self.scaling = self._transform(sigmoid(tensor))

        return transformed_tensor

    def inverse_transform(self, transformed_tensor):
        """
        Applies the inverse transformation.
        """
        if not self.enforced:
            return transformed_tensor

        tensor = inv_sigmoid(self._inv_transform(transformed_tensor, self.scaling))

        return tensor

    @property
    def initial_value(self):
        """
        The initial parameter value (if specified, None otherwise)
        """
        return self._initial_value

    def __repr__(self):
            return super().__repr__()

    def __iter__(self):
        yield 0
        yield 1
        yield self.scaling