# Code for *Content Effects in Human and LLM World Models*

## Setup

Run ./setup_env.sh and acivate the new environment. enter huggingface/openai/google API keys as prompted, or leave blank. Note that you will need GPU access to run most experiments (require running models locally).

## Experiments
All experiment results can be generated through exp_battery.py. All final analyses can be found in analyses/.

### 1. General behavioral evaluation
Run the full model suite on the behavioral evaluation stimuli. 

*python3 exp_battery.py run_model_general*

### 2. Fine grained content effects evaluation
Run the full model suite on the content effects evaluation stimuli

*python3 exp_battery.py run_model_fine_grained*

### 3. QK attention
Extract QK matrix processing features for full model suite

### 4. Residual processing
Extract residual stream processing features for full model suite

### 5. Train linear probes
Train linear probes for full model suite

### 6. Steer linear probes
Steer model responses with linear probes, and extract logit metrics, logit lens (unembed projections), 

### 7. Steer prompting
Steer model responses with prompting


## Analyses
All analyses are in the analysis/ folder. Separate experiment analyses are separated by notebook and numbered according to manuscript sections.

###

