import torch
import argparse


def get_model_config(model_type:str, precision:int, heterogeneous:bool=True, pareto:bool=False):
    """
    Returns the model configuration for the given model type and precision
    """
    if model_type == "gpt2XL":
        model_config = {"hidden_size": torch.tensor([1600]).float(), # also known as d_model
                    "sequence_length": torch.tensor([1024]).float(),
                    "num_layers": torch.tensor([48]).float(), 
                    "vocab_size":torch.tensor([50257]).float(),
                    "num_attention_heads": torch.tensor([16]).float(),
                    "type": "gpt2XL",
                    "precision":torch.tensor(precision).float()}
        gbs = 64
        exp_name = "gpt2"
    elif model_type == "bert":
        # for more information: https://catalog.ngc.nvidia.com/orgs/nvidia/models/megatron_bert_345m
        model_config = {"hidden_size": torch.tensor([1024]).float(), # also known as d_model
                    "sequence_length": torch.tensor([512]).float(),
                    "num_layers": torch.tensor([24]).float(),
                    "vocab_size":torch.tensor([30522]).float(),
                    "num_attention_heads": torch.tensor([16]).float(),
                    "type": "bert",
                    "precision":torch.tensor(precision).float()}
        gbs = 256
        exp_name = "bert"
    elif model_type == "T5":
        model_config = {"hidden_size": torch.tensor([1024]).float(), # also known as d_model
                    "sequence_length": torch.tensor([512]).float(),
                    "num_layers": torch.tensor([48]).float(),
                    "vocab_size":torch.tensor([30522]).float(), # we are using Bert's vocab file, original paper size is 32k
                    "num_attention_heads": torch.tensor([32]).float(),
                    "type": "T5",
                    "precision":torch.tensor(precision).float(),
                    "dkv": torch.tensor([128]).float()}
        gbs = 128
        exp_name = "T5"
    else:
        assert False, "Model type not supported"
    
    if heterogeneous==True:
        exp_name = exp_name + "_heterogeneous"
    if pareto==True:
        exp_name = exp_name + "_pareto_device_placement(8,8)"

    return model_config, gbs, exp_name