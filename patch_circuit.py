import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time
import json
from attribution_patching import AttributionPatching
from effect_metrics import IOIMetric, AttPMetric
from analyze_tools import get_top_k_neurons_list
from circuit_discovery import discover_circuits
import logging
import argparse
logging.basicConfig(level=logging.FATAL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run patching experiments with specified configurations.")
    parser.add_argument("--model_path",
                        type=str,
                        required=True,
                        help="Path to the pretrained model.")
    parser.add_argument("--metric_name",
                        type=str,
                        choices=["ioi-metric", "attp-metric"],
                        required=True,
                        help="Metric to use for evaluation.")
    parser.add_argument(
        "--patching_from_clean_to_corrupted",
        type=lambda x: x.lower() == 'true',
        help=
        "Direction of patching: clean to corrupted (True) or corrupted to clean (False)."
    )
    parser.add_argument("--model_family",
                        type=str,
                        required=True,
                        help="Model family name (e.g., qwen2).")
    parser.add_argument("--rerun",
                        type=lambda x: x.lower() == 'true',
                        help="Whether to rerun the patching process.")
    parser.add_argument("--xr_language",
                        type=str,
                        required=True,
                        help="Language code for clean prompts.")
    parser.add_argument("--xc_language",
                        type=str,
                        required=True,
                        help="Language code for corrupted prompts.")
    parser.add_argument("--module_type",
                        type=str,
                        choices=["attn", "mlp", "mlp_down", "res_post"],
                        required=True,
                        help="Type of module to patch")
    args = parser.parse_args()

    model_path = args.model_path
    metric_name = args.metric_name
    patching_from_clean_to_corrupted = args.patching_from_clean_to_corrupted
    model_family = args.model_family
    rerun = args.rerun
    xr_language = args.xr_language
    xc_language = args.xc_language
    module_type = args.module_type

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="cuda:0",
    )
    model_name = model_path.split('/')[-1]

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    metric_instance_map = {
        "ioi-metric": IOIMetric(),
        "attp-metric": AttPMetric()
    }
    metric_instance = metric_instance_map[metric_name]

    clean_prompt = []
    corrupted_prompt = []
    clean_answers = []
    corrupted_answers = []
    language_prefix = xr_language + '-' + xc_language

    # Prepare the data
    data_list = json.load(
        open(
            f"../data/generated-counterfact-patching/counterfact-{xr_language}-{xc_language}-7b-instruct.json",
            'r'))
    clean_prompt = [data['xr'] for data in data_list]
    corrupted_prompt = [data['xc'] for data in data_list]
    clean_answers = [data['xr_answer'] for data in data_list]
    corrupted_answers = [data['xc_answer'] for data in data_list]

    print(f"xr[0]: {clean_prompt[0]}")
    print(f"xr answer[0]: {clean_answers[0]}")
    print(f"xc[0]: {corrupted_prompt[0]}")
    print(f"xc answer[0]: {corrupted_answers[0]}")
    print("Len of clean_prompt:", len(clean_prompt))
    print("Len of corrupted_prompt:", len(corrupted_prompt))

    # Initiate AttributionPatching instance for circuit discovery
    attribution_patching = AttributionPatching(model=model,
                                               tokenizer=tokenizer,
                                               xr=clean_prompt,
                                               xc=corrupted_prompt,
                                               xr_answers=clean_answers,
                                               xc_answers=corrupted_answers,
                                               model_family=model_family,
                                               metric=metric_instance,
                                               threshold=0.5)

    save_dir = f"../results/attp-{xr_language}-{xc_language}-{patching_from_clean_to_corrupted}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Run attribution patching for the first time, save the results
    attr_scores_save_path = f"{save_dir}/attr-{model_name}-metric-{metric_name}-patching-direction-{patching_from_clean_to_corrupted}-neuron-level-1.pt"
    if not os.path.exists(attr_scores_save_path) or rerun:
        start_time = time.time()
        attr_scores = attribution_patching.run(
            selected_components=None,
            patching_from_clean_to_corrupted=patching_from_clean_to_corrupted,
            module_type=module_type,
            language_prefix=language_prefix)
        print(
            f"time taken for attribution_patching_before.run: {time.time() - start_time:.4f} seconds"
        )
        torch.save(attr_scores, attr_scores_save_path)
    else:
        attr_scores = torch.load(attr_scores_save_path)

    selected_components = get_top_k_neurons_list(attr_scores, 30)['top_30_neurons']
    discover_circuits(
        attribution_patching=attribution_patching,
        level1_neurons=selected_components,
        max_level=5 ,    
        output_dir=f"{save_dir}/circuit_discovery/new",
        language_prefix=language_prefix,
        threshold=0.0001
    )