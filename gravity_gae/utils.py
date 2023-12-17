import numpy as np
from math import log10, floor
from pandas import DataFrame

# Model utilities
def reset_parameters(module):
    for layer in module.children():
        if hasattr(layer, 'reset_parameters'):
            print(f"resetting {layer}")
            layer.reset_parameters()
        elif len(list(layer.children())) > 0:
            reset_parameters(layer)

def print_model_parameters_names(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)

def summarize_link_prediction_evaluation(performances):
    mean_std_dict = {}
    for metric in ['AUC', 'F1', 'hitsk', 'AP']:
        vals = [run[metric] for run in performances]
        filtered_vals = list(filter(None, vals))
        mean_std_dict[metric] = {
            "mean": np.nanmean(filtered_vals), 
            "std": np.nanstd(filtered_vals)
        }
    return mean_std_dict

def round_to_first_significative_digit(x):
    digit = -int(floor(log10(abs(x))))
    return digit, round(x, digit)

def pretty_print_link_performance_evaluation(mean_std_dict, model_name):
    performances_strings = {}
    for metric, mean_std in mean_std_dict.items():
        if np.isnan(mean_std["mean"]):
            performances_strings[metric] = str(None)
        elif mean_std["std"] == 0:
            digit, mean_rounded = round_to_first_significative_digit(mean_std["mean"]) 
            performances_strings[metric] = f"{mean_rounded} +- {mean_std['std']}"
        else:
            digit, std_rounded = round_to_first_significative_digit(mean_std["std"])
            mean_rounded = round(mean_std["mean"], digit)
            performances_strings[metric] = f"{mean_rounded} +- {std_rounded}"
    
    df = DataFrame(performances_strings.values(), columns=[model_name], index=performances_strings.keys())
    return df.to_markdown(index=True)