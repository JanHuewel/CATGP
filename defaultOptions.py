"""
This script provides all custom default options used in different steps of GP operations
"""

default_options = {
    "training": {"print_training_output" : False,
                 "print_optimizing_output": False,
                 "max_iter": 50,
                 "learning_rate": 0.1,
                 "restarts": 10,
                 "optimization method": "default"},
    "kernel search": {"print": False,
                      "probability graph": False,
                      "multithreading": True},
    "plotting": {"border_ratio": 0.0,
                 "sample_points": 1000,
                 "legend": True}}


hyperparameter_limits = {"RBFKernel": {"lengthscale": [1e-3,1]},
                         "LinearKernel": {"variance": [1e-4,1]},
                         "PeriodicKernel": {"lengthscale": [1e-4,1],
                                            "period_length": [1e-4,3]},
                         "ScaleKernel": {"outputscale": [1e-3,100]},
                         "Noise": [1e-4,2e-2],
                         "Mean": [0.0,1.0]}