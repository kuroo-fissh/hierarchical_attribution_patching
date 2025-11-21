from torch import Tensor
import torch
from typing import List, Tuple, Union
from torchtyping import TensorType

class PatchingBase:
    def __init__(self, model, tokenizer, xr: List[str], xc: List[str], xr_answers: List[str], xc_answers: List[str], model_family: str, threshold: float):
        self.model = model
        self.tokenizer = tokenizer
        self.xr = xr
        self.xc = xc
        self.xr_answers = xr_answers
        self.xc_answers = xc_answers
        self.model_family = model_family
        self.threshold = threshold

        self.module_name_format = {
            "qwen2": {
                "mlp": "model.layers.{i}.mlp",
                "attn": "model.layers.{i}.self_attn.o_proj",
                "res_post": "model.layers.{i}",
                "mlp_down": "model.layers.{i}.mlp.down_proj",
                "mlp_silu": "model.layers.{i}.mlp.act_fn",
            }
        }

        self.check_input_types()
        self.prepare_tokens()

    def prepare_tokens(self):
        xr_tokens = self.tokenizer(self.xr, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        xc_tokens = self.tokenizer(self.xc, padding=True, truncation=True, return_tensors="pt").to(self.model.device)
        
        self.xr_token_ids = xr_tokens['input_ids']
        self.xc_token_ids = xc_tokens['input_ids']

        self.xr_lengths = xr_tokens['attention_mask'].sum(dim=1)
        self.xc_lengths = xc_tokens['attention_mask'].sum(dim=1)
        self.xr_lengths = [length - 1 for length in self.xr_lengths]
        self.xc_lengths = [length - 1 for length in self.xc_lengths]

        self.xr_answers_token_ids = torch.stack([self.tokenizer.encode(x, return_tensors="pt")[0][0].to(self.model.device) for x in self.xr_answers])
        self.xc_answers_token_ids = torch.stack([self.tokenizer.encode(x, return_tensors="pt")[0][0].to(self.model.device) for x in self.xc_answers])
    def check_input_types(self):
        if not isinstance(self.xr, list) or not all(isinstance(x, str) for x in self.xr):
            raise TypeError("xr must be a list of strings")
        
        if not isinstance(self.xc, list) or not all(isinstance(x, str) for x in self.xc):
            raise TypeError("xc must be a list of strings")
            
        if len(self.xr) == 0:
            raise ValueError("xr and xc cannot be empty")
            
        if not isinstance(self.threshold, (int, float)):
            raise TypeError(f"threshold must be int, instead of {type(self.threshold)}")
        
        if not 0 <= self.threshold:
            raise ValueError(f"threshold must be [0, ], instead of : {self.threshold}")
        
        if not isinstance(self.model_family, str):
            raise TypeError(f"model_family must be str, instead of {type(self.model_family)}")
        if not self.model_family in ["qwen2", "gpt2"]:
            raise ValueError("Invalid model_family specified.")

    def get_logit_diff(self, xr_token_ids: List[int], xc_token_ids: List[int], logits: TensorType["batch", "pos", "vocab_size"], patch_all_samples_at_once=True) -> Union[float, TensorType["batch"]]:
        if len(logits.shape) == 3:
            # Get final logits only
            logits = logits[:, -1, :]
        correct_logits = logits.gather(1, xr_token_ids.unsqueeze(1))
        incorrect_logits = logits.gather(1, xc_token_ids.unsqueeze(1))
        if patch_all_samples_at_once:
            return (correct_logits - incorrect_logits).mean()
        else:
            return correct_logits - incorrect_logits
        
    def get_model_submodule(self, module_type: str, layer_idx: int):
        if self.model_family in self.module_name_format.keys():
            module_name = self.module_name_format[self.model_family][module_type].format(i=layer_idx)
            module = self.model.get_submodule(module_name)
        else:
            gpt2_module_map = {
                "mlp": self.model.transformer.h[layer_idx].mlp,
                "attn": self.model.transformer.h[layer_idx].attn.c_proj,
                "res_post": self.model.transformer.h[layer_idx],
                "unembed": self.model.lm_head,
                "c_attn": self.model.transformer.h[layer_idx].attn.c_attn,
                "ln_f": self.model.transformer.ln_f
            }
            
            if module_type not in gpt2_module_map:
                raise ValueError(f"未知的模块类型 '{module_type}' 用于 'gpt2' 模型")
            module = gpt2_module_map[module_type]
        return module

    def get_activation(self, input_tokens: Tensor, module_type: str, token_pos: List[int] = None, device = None) -> Tuple[TensorType["layer", "batch", "d_model"], TensorType["batch", "pos", "vocab_size"]]:
        features = []
        def hook_fn(module, input):
            # input: Tuple[Tensor["batch", "pos", "d_model"], ]
            feat = input[0].detach()
            if token_pos is not None:
                # Get hidden states at token_pos for each sample
                batch_size = feat.shape[0]
                gathered_feats = torch.zeros(batch_size, feat.shape[-1], device=device)
                for i in range(batch_size):
                    gathered_feats[i] = feat[i, token_pos[i]]
                features.append(gathered_feats)
            else:
                # Get hidden states at the last token for each sample
                features.append(feat[:, -1])
        hooks = []
        n_layers = self.model.config.num_hidden_layers
        for i in range(n_layers):
            module = self.get_model_submodule(module_type, i)
            hooks.append(module.register_forward_pre_hook(hook_fn))
        with torch.no_grad():
            logits = self.model(input_tokens).logits
        for h in hooks:
            h.remove()
        # move features to the same device as the model
        features = [feat.to(device) for feat in features]
        features = torch.stack(features, dim=0)
        return features, logits

    def get_gradient(self, input_tokens: Tensor, backward_func, backward_value, module_type: str, get_input_grad: bool, token_pos: List[int] = None, device=None) -> TensorType["layer", "batch", "d_model"]:
        captured_features = []
        def hook_input_fn(module, grad_input, grad_output):
            # grad_output: Tuple[Tensor["batch", "pos", "d_model"], ]
            # grad_input and grad_output correspond to the output gradient and input gradient of the module, respectively. 
            # The positions of input and output are consistent with the forward order, but the order of backward calculation is opposite.
            grad = grad_output[0]
            if token_pos is not None:
                # Get gradient at token_pos for each sample
                batch_size = grad.shape[0]
                gathered_grad = torch.zeros(batch_size, grad.shape[-1], device=device)
                for i in range(batch_size):
                    gathered_grad[i] = grad[i, token_pos[i]].detach().to(device)
                captured_features.append(gathered_grad)
            else:
                # Get gradient at the last token for each sample
                captured_features.append(grad[:, -1]).detach().to(device)
            return None
        
        def hook_output_fn(module, grad_input, grad_output):
            # grad_input: Tuple[Tensor["batch", "pos", "d_model"], ]
            grad = grad_input[0]
            if token_pos is not None:
                batch_size = grad.shape[0]
                gathered_grad = torch.zeros(batch_size, grad.shape[-1], device=device)
                for i in range(batch_size):
                    gathered_grad[i] = grad[i, token_pos[i]].detach().to(device)
                captured_features.append(gathered_grad)
            else:
                captured_features.append(grad[:, -1]).detach().to(device)
            return None

        hooks = []
        n_layers = self.model.config.num_hidden_layers
        hook_fn = hook_input_fn if get_input_grad else hook_output_fn
        for i in range(n_layers):
            module = self.get_model_submodule(module_type, i)
            hooks.append(module.register_full_backward_hook(hook_fn))
        output = self.model(input_tokens)
        # print("Backward value: ", backward_func(output.logits))
        if backward_func is None and backward_value is not None:
            backward_value.backward()
        else:
            backward_func(output.logits).backward()
        # Since backward starts from the last layer, we need to reverse captured_features along the layer dimension
        captured_features = captured_features[::-1]
        captured_features = [feat.to(device) for feat in captured_features]
        captured_features = torch.stack(captured_features, dim=0)
        self.model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        for h in hooks:
            h.remove()
        return captured_features

    def run(self):
        raise NotImplementedError("run is not implemented. Please implement it in the subclass.")