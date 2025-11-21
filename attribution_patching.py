import torch
from torch import Tensor
import einops
from patching_base import PatchingBase
from torchtyping import TensorType
from typing import Union, Dict, Tuple, List
from tqdm import tqdm
from fancy_einsum import einsum
import gc
import numpy as np
import os
class AttributionPatching(PatchingBase):

    def __init__(self, model, tokenizer, xr, xc, xr_answers, xc_answers, model_family: str, metric,
                 threshold: float):
        super().__init__(model, tokenizer, xr, xc, xr_answers, xc_answers, model_family, threshold)
        self.metric = metric

    def run(self, patching_from_clean_to_corrupted: bool = True, selected_components=None, module_type="attn", language_prefix: str="fr-de") -> Union[TensorType["layers", "components"], Dict[Tuple[int, int], TensorType["layers", "components"]]]:
        device = torch.device('cuda:0')
        n_layers = self.model.config.num_hidden_layers
        n_samples = len(self.xr_token_ids)
        num_heads = self.model.config.num_attention_heads
        head_dim = self.model.config.hidden_size // num_heads

        clean_input_toks = self.xr_token_ids
        corrupted_input_toks = self.xc_token_ids
        clean_token_id = self.xr_answers_token_ids
        corrupted_token_id = self.xc_answers_token_ids
        clean_token_pos = self.xr_lengths
        corrupted_token_pos = self.xc_lengths

        # Choose different inputs based on the patching direction
        if not patching_from_clean_to_corrupted:
            clean_input_toks, corrupted_input_toks = corrupted_input_toks, clean_input_toks
            clean_token_id, corrupted_token_id = corrupted_token_id, clean_token_id
            clean_token_pos, corrupted_token_pos = corrupted_token_pos, clean_token_pos

        # Get logits and activations for clean and corrupted inputs
        clean_tensor, clean_logits = self.get_activation(clean_input_toks, module_type, clean_token_pos, device)
        ld_clean = self.get_logit_diff(clean_token_id,
                                            corrupted_token_id, 
                                            clean_logits)
        clean_tensor = clean_tensor.to(device)
        corr_tensor, corrupted_logits = self.get_activation(corrupted_input_toks, module_type, corrupted_token_pos, device)
        ld_corrupted = self.get_logit_diff(clean_token_id,
                                                corrupted_token_id,
                                                corrupted_logits)
        corr_tensor = corr_tensor.to(device)
        CLEAN_BASELINE = ld_clean.item()
        CORRUPTED_BASELINE = ld_corrupted.item()

        if module_type == "attn":
            clean_tensor = clean_tensor.view(n_layers, n_samples, num_heads, head_dim)
            corr_tensor = corr_tensor.view(n_layers, n_samples, num_heads, head_dim)
        
        # Define the backward function to compute the logit difference before and after patching

        backward_func = lambda logits: self.metric.compute(
            self.get_logit_diff(clean_token_id, corrupted_token_id,
                                    logits), CLEAN_BASELINE, CORRUPTED_BASELINE)
        output_grad_tensor = self.get_gradient(corrupted_input_toks, 
                                        backward_func, None, module_type, get_input_grad=False, token_pos=corrupted_token_pos, device=device)
        output_grad_tensor = output_grad_tensor.to(device)
        if module_type == "attn":
            grad_tensor = output_grad_tensor.view(n_layers, n_samples, num_heads, head_dim)
        elif module_type == "mlp_down":
            grad_tensor = output_grad_tensor

        # If selected_components is not None, only return the attribution for the specified components
        if selected_components is not None:
            grad_tensor = grad_tensor.to(device)
            
            for (end_layer, end_neuron) in selected_components:
                end_neuron_grad = grad_tensor[end_layer, :, end_neuron]  
                up_proj = self.model.model.layers[end_layer].mlp.up_proj.weight.to(device)
                ln = self.model.model.layers[end_layer].post_attention_layernorm.weight.to(device)
                d_neuron = clean_tensor.shape[2]
                with torch.no_grad():
                    attribution_matrix = torch.zeros((n_layers, d_neuron), device=device)
                    
                    '''
                        To compute the edge attribution through the end_neuron, we need to map the end_neuron gradient back to the model's residual stream space.
                        This is because we are effectively cloning the path from the start neuron to the end neuron through the residual stream, ignoring all other paths.
                        The overall process is as follows:
                        1. Map the end_neuron gradient to the model's residual stream space: we only consider the path through up_proj, 
                            first dividing the neuron gradient by the silu output, then multiplying by the up_proj weight matrix and passing through ln.
                        2. The upstream neuron's activation is multiplied by the down_proj weight matrix to project it onto the residual stream.
                        3. Calculate the attribution.
                    '''
                    
                    # Project the end_neuron gradient to the model's residual stream space
                    end_neuron_vector = up_proj[end_neuron]
                    corr_silu_input, _ = self.get_activation(corrupted_input_toks, "mlp_silu") 
                    end_neuron_silu_output = self.model.model.layers[end_layer].mlp.act_fn(corr_silu_input[end_layer, :, end_neuron]).to(device)
                    # Gradient of the end_neuron only through up_proj
                    scaled_end_neuron_grad = end_neuron_grad * end_neuron_silu_output 
                    mediated_grad = einsum(
                        "batch, d_model, d_model -> batch d_model", 
                        scaled_end_neuron_grad,    # [n_samples]
                        ln,                        # [d_model]
                        end_neuron_vector          # [d_model]
                    )

                    for layer_idx in range(n_layers):
                        if layer_idx >= end_layer:  # Only consider layers before the end_layer
                            continue
                            
                        # Load the down projection weight matrix for the current layer
                        down_proj_weight = self.model.model.layers[layer_idx].mlp.down_proj.weight.T.to(device)  # [d_neuron, hidden_size]
                        
                        # Calculate the attribution matrix
                        attribution_matrix[layer_idx] = einsum(
                            "batch neuron, neuron d_model, batch d_model -> neuron",
                            (clean_tensor[layer_idx] - corr_tensor[layer_idx]),
                            down_proj_weight,
                            mediated_grad
                        )
                                            
                        del down_proj_weight
                        torch.cuda.empty_cache()
                        gc.collect()
                    output_dir = f"../results/neuron-attributions/{language_prefix}"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path_pt = f"{output_dir}/{end_layer}-{end_neuron}-attribution-matrix.npy"
                    np.save(output_path_pt, attribution_matrix.detach().cpu().numpy())
                    
                    del up_proj, ln, end_neuron_vector, corr_silu_input, attribution_matrix
                    del end_neuron_silu_output, scaled_end_neuron_grad, mediated_grad
                    del end_neuron_grad
                    torch.cuda.empty_cache()
        else:
            if module_type == "attn":
                sample_score = einops.reduce(
                grad_tensor * (clean_tensor - corr_tensor),
                    "layer sample head d_head -> layer head",
                    "sum",
                )
            elif module_type == "mlp_down":
                sample_score = einops.reduce(
                    grad_tensor * (clean_tensor - corr_tensor),
                    "layer sample d_neuron -> layer d_neuron",
                    "sum",
                )
            return sample_score
