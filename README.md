# hierarchical_attribution_patching

We provide complete implementation for hierarchical attribution patching, organized as follows:

   - `patching_base.py`: Base class containing core methods (data validation, activation/gradient extraction) shared by all patching variants (e.g. AttributionPatching, ActivationPatching)
   - `attribution_patching.py`: Computes neuron-to-output and neuron-to-neuron attribution scores
   - `circuit_discovery.py`: Builds circuits by iteratively adding neurons exceeding attribution thresholds
   - `metric_base.py`: Base class for different metrics;
   - `effect_metrics.py`: Implements our `AttPMetric` (extends `MetricBase` class)
   - `patch_circuit.py`: Main function to run hierarchical patching, where we pass in model information, data and hyper-parameters for our circuit discover experiments. 