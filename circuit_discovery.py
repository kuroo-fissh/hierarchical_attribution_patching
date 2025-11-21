import os
import json
import numpy as np
import torch
from tqdm import tqdm

def discover_circuits(attribution_patching, level1_neurons, max_level=5, output_dir="../results/circuit_discovery", language_prefix="", threshold=0.001):
    """
    Iteratively discover circuits in the model by analyzing neuron attributions.

    Args:
        attribution_patching: An instance of AttributionPatching.
        level1_neurons: A list of neurons at the first level, formatted as [(layer, neuron), ...].
        max_level: The maximum number of levels to explore.
        output_dir: The directory to save the results.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_neurons = set()
    current_level_neurons = level1_neurons.copy()
    
    for level in range(1, max_level + 1):
        print(f"Processing Level {level}, Neuron Number: {len(current_level_neurons)}")
        
        level_attributions = {}
        next_level_neurons = set()
        
        for end_layer, end_neuron in tqdm(current_level_neurons):
            if (end_layer, end_neuron) in processed_neurons:
                continue
            
            processed_neurons.add((end_layer, end_neuron))
            
            # Calculate the attribution matrix for the current neuron
            attribution_matrix_path = f"../results/neuron-attributions/{language_prefix}/{end_layer}-{end_neuron}-attribution-matrix.npy"
            if os.path.exists(attribution_matrix_path):
                attribution_matrix = np.load(attribution_matrix_path)
            else:
                attribution_patching.run(
                    patching_from_clean_to_corrupted=True,
                    selected_components=[(end_layer, end_neuron)],
                    module_type="mlp_down",
                    language_prefix=language_prefix
                )
                
                # Load the attribution matrix
                attribution_matrix = np.load(attribution_matrix_path)
                
            important_neurons = []
            candidate_neurons = []
            
            for layer_idx in range(attribution_matrix.shape[0]):
                if layer_idx >= end_layer:  # Only consider layers before the end_layer
                    continue
                
                # Find neurons in this layer with attribution value greater than 3*sigma
                for neuron_idx in range(attribution_matrix.shape[1]):
                    attr_value = attribution_matrix[layer_idx, neuron_idx]
                    if abs(attr_value) > threshold:
                        important_neurons.append((layer_idx, neuron_idx, float(attr_value)))
                        # If the neuron is not in the first layer and has not been processed yet, add to candidate neurons
                        if layer_idx > 1 and (layer_idx, neuron_idx) not in processed_neurons:
                            candidate_neurons.append((layer_idx, neuron_idx, float(attr_value)))

            sorted_neurons = sorted(important_neurons, key=lambda x: abs(x[2]), reverse=True)
            top_neurons = sorted_neurons[:min(10, len(sorted_neurons))]
            # Add top neurons to the next level's processing list
            for layer_idx, neuron_idx, _ in top_neurons:
                next_level_neurons.add((layer_idx, neuron_idx))
            
            level_attributions[f"{end_layer},{end_neuron}"] = top_neurons
            
        with open(f"{output_dir}/level_{level}_attributions.json", 'w') as f:
            json.dump(level_attributions, f)
        
        # If there are no more neurons to process or if we have reached the shallowest layer, the circuit discovery is complete
        if not next_level_neurons or all(layer <= 1 for layer, _ in next_level_neurons):
            print(f"Finished circuit discovery, reaching the maximum level {level} or the shallowest layer.")
            break
            
        # Update the current level neurons for the next iteration
        current_level_neurons = list(next_level_neurons)

