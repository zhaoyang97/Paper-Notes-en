---
title: >-
  [Paper Note] Steering Generative Models with Experimental Data for Protein Fitness Optimization
description: >-
  [NeurIPS 2025][Medical Imaging][protein fitness optimization] This work systematically evaluates strategies for steering protein generative models (discrete diffusion models and language models) toward fitness optimizati…
tags:
  - "NeurIPS 2025"
  - "Medical Imaging"
  - "protein fitness optimization"
  - "generative model steering"
  - "discrete diffusion models"
  - "protein language models"
  - "Bayesian optimization"
date: 2026-05-08
content_hash: 710d737ce7e8e6d6
---

# Steering Generative Models with Experimental Data for Protein Fitness Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2505.15093](https://arxiv.org/abs/2505.15093)
**Code**: [Available](https://github.com/jsunn-y/SGPO)
**Area**: Medical Imaging / Computational Biology
**Keywords**: protein fitness optimization, generative model steering, discrete diffusion models, protein language models, Bayesian optimization

## TL;DR

This work systematically evaluates strategies for steering protein generative models (discrete diffusion models and language models) toward fitness optimization, finding that plug-and-play guidance methods using small labeled datasets (~200 samples)—particularly DAPS—outperform RL-based fine-tuning, and proposes a Thompson sampling strategy incorporating predictive uncertainty for adaptive optimization.

## Background & Motivation

Protein fitness optimization is a central challenge in bioengineering: given a known protein, the goal is to identify sequence variants that maximize desired properties (activity, stability, fluorescence, etc.). The design space is enormous ($20^M$ possible sequences for a protein of length $M$), while wet-lab experimental throughput is extremely limited (only $10^2$–$10^3$ fitness labels).

**Steered Generation for Protein Optimization (SGPO)** is a promising methodological framework that combines a generative prior (capturing evolutionary knowledge of natural protein distributions) with fitness-labeled data (to steer generation toward high-fitness regions). However, existing work suffers from three key limitations: (1) most methods require large labeled datasets or computational surrogate metrics, leaving low-label regimes underexplored; (2) no systematic comparison exists across different models and steering strategies; (3) adaptive optimization principles such as uncertainty-driven exploration have not been adequately integrated.

## Method

### Overall Architecture

The SGPO workflow proceeds in three steps: (1) train a generative prior $p(\mathbf{x})$ on a multiple sequence alignment (MSA) of natural protein sequences; (2) train a value function $p(\mathbf{y}|\mathbf{x}) \propto \exp(f(\mathbf{x})/\beta)$ using a small number of fitness labels; (3) sample high-fitness sequences from the posterior $p(\mathbf{x}|\mathbf{y}) \propto p(\mathbf{x}) \exp(f(\mathbf{x})/\beta)$ via a steering strategy.

### Key Designs

1. **Training and evaluation of multiple generative prior models**: The study trains and compares continuous diffusion (27.9M parameters), D3PM with uniform-noise discrete diffusion (37.9M, fine-tuned from EvoDiff), MDLM with masked discrete diffusion (28.6M), and ARLM autoregressive language model (151M, fine-tuned from ProGen2). D3PM most faithfully captures the natural sequence distribution while maintaining high diversity; continuous diffusion performs poorly; UDLM is prone to mode collapse.

2. **Plug-and-play steering strategies**: Three strategies are evaluated—classifier guidance (CG, which biases the discrete diffusion rate matrix using a time-dependent value function), decoupled annealed posterior sampling (DAPS, **first adapted to protein optimization**, requiring only a clean-data value function), and noise-optimized sampling (NOS, which optimizes in the continuous embedding space of discrete tokens). DPO fine-tuning of the language model serves as the baseline. DAPS achieves the best overall performance, followed by CG.

3. **Ensemble-based Thompson sampling**: Within the adaptive optimization loop, an ensemble of 10 neural network regressors forms a frequentist ensemble of value functions. At each generation step, one value function is randomly sampled from the ensemble to guide sequence generation—analogous to Thompson sampling in Bayesian optimization—leveraging predictive uncertainty to balance exploration and exploitation.

### Loss & Training

- **Prior training**: Each model is trained on MSA data using standard diffusion or language modeling objectives.
- **Value function training**: A neural network regressor is trained on approximately 200 fitness labels using MSE loss.
- **Adaptive loop**: Each round samples 100 sequences, evaluates fitness using a computational oracle, and updates the labeled dataset and value function.
- Plug-and-play methods require tuning only a single hyperparameter (guidance strength) and do not modify the prior model weights.

## Key Experimental Results

### Main Results

Evaluation is conducted on TrpB enzyme (15 residues), CreiLOV fluorescent protein (119 residues), and GB1 binding protein (56 residues).

| Method | Model | Steerability | Fitness | Diversity | Compute Cost |
|--------|-------|-------------|---------|-----------|--------------|
| DAPS | MDLM | ★★★★★ | Highest | Moderate | Low (minutes) |
| CG | D3PM | ★★★★ | High | Moderate | Low |
| NOS | D3PM | ★★ | Moderate | Higher | Low |
| DPO | ARLM | ★★ | Lower | Lower | High (hours) |
| APEXGo (BO) | VAE | ★★ | Lower | — | Moderate |

### Ablation Study

| Design Choice | Preferred Option | Notes |
|---------------|-----------------|-------|
| Guided vs. unguided | Guidance provides significant gains | 200 labels suffice for effective steering |
| Plug-and-play vs. DPO fine-tuning | Plug-and-play superior | DPO shows poor steerability under low data |
| Ensemble vs. single value function | Ensemble superior | Thompson sampling better explores sequence space |
| Continuous vs. discrete diffusion | Discrete superior | Continuous diffusion prior captures natural distribution poorly |
| DAPS vs. CG | DAPS marginally superior | Especially on continuous models |

### Key Findings

- As few as 200 fitness labels suffice for effective guidance, substantially reducing experimental cost.
- Plug-and-play guidance outperforms DPO fine-tuning: only a single hyperparameter requires tuning, and training completes within minutes.
- DAPS is the best steering strategy, first adapted to protein optimization and demonstrated to be advantageous.
- Thompson sampling in the multi-round adaptive optimization loop achieves higher maximum fitness.
- SGPO outperforms latent-space Bayesian optimization (APEXGo), which struggles with trust region calibration under low-data, few-round settings.

## Highlights & Insights

- **Systematic and comprehensive**: The first work to compare 7 generative models × 4 steering strategies within a unified framework, providing clear best-practice guidelines.
- **Practically oriented**: Strategy selection recommendations can directly inform real protein engineering experimental design.
- Plug-and-play methods do not modify prior weights, incur minimal training cost, and the prior can be reused across different tasks.
- Thompson sampling introduces the exploration–exploitation trade-off from Bayesian optimization into generative model steering, representing a novel conceptual integration.

## Limitations & Future Work

- Evaluation relies on computational oracles rather than real wet-lab experiments; transferability of conclusions remains to be validated.
- Insertion and deletion mutations are not considered; the approach is restricted to fixed-length sequence design.
- For fitness objectives that diverge substantially from natural protein function, the value of the evolutionary prior may diminish.
- Gaussian processes perform poorly as the Thompson sampling backend; uncertainty quantification requires further improvement.
- RL-based methods on discrete diffusion models have not been tested.

## Related Work & Insights

- **Comparison with APEXGo**: Latent-space Bayesian optimization underperforms SGPO in low-data, few-round settings.
- **Relation to Blalock et al.**: RL fine-tuning may hold advantages in large-data regimes ($>10^3$ labels).
- **Broader implications**: The SGPO framework is transferable to discrete sequence design domains such as small molecules and controllable natural language generation; multi-objective optimization is an important future direction.

## Rating

- Novelty: ⭐⭐⭐⭐ (Systematic comparison and DAPS adaptation are innovative, though individual components have prior foundations)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (3 proteins, 7 models, 4 strategies, and adaptive optimization experiments are comprehensive)
- Writing Quality: ⭐⭐⭐⭐⭐ (Excellent structure, highly informative figures, strong practical guidance)
- Value: ⭐⭐⭐⭐⭐ (Provides clear best practices for ML-assisted design in protein engineering)

This work systematically evaluates strategies for steering protein generative models (discrete diffusion models and language models) using small experimental fitness datasets, finding that plug-and-play guidance based on DAPS outperforms reinforcement learning fine-tuning, and proposes an adaptive sequence selection strategy analogous to Thompson sampling.

## Background & Motivation

Core challenges in protein fitness optimization:
1. **Vast design space**: A protein of length $M$ admits $20^M$ possible sequences, of which only a negligible fraction are functional.
2. **Low experimental throughput**: Wet-lab assays provide only $10^2$–$10^3$ fitness labels.
3. **Inefficiency of classical directed evolution**: Each round accumulates only a single mutation, constraining search to local neighborhoods.

Limitations of existing approaches:
- **Zero-shot methods** (prior only): Unable to handle fitness objectives diverging from natural function.
- **Purely supervised methods** (labels only): Lack evolutionary priors and generalize poorly.
- **MLDE** (enumerate and score): Design space limited to $N < 9$ residues.

This work proposes the **SGPO (Steered Generation for Protein Optimization)** framework, combining a generative prior $p(x)$ trained on natural protein sequences with steering from a small number of fitness labels to efficiently optimize protein fitness over large design spaces. The central questions are: which generative model and steering strategy combination is most effective, and how can uncertainty be leveraged to improve exploration?

## Method

### Overall Architecture

SGPO operates in two stages:
1. **Generative prior pretraining**: A generative model is trained on a multiple sequence alignment (MSA) of homologous proteins to capture the natural protein distribution $p(x)$.
2. **Guided generation**: A value function $f(x)$ is trained on a small number of fitness labels and used via plug-and-play guidance to sample high-fitness variants from the posterior $p(x|y) \propto p(x) \cdot \exp(f(x)/\beta)$.

### Key Designs

1. **Systematic evaluation of multiple generative prior models**

   Five classes of generative models are trained and compared:
   - **Continuous diffusion**: Diffusion in continuous space, 27.9M parameters.
   - **Continuous-ESM diffusion**: Diffusion in ESM embedding space, 25.5M parameters.
   - **D3PM**: Discrete diffusion with uniform noise, 37.9M parameters, fine-tuned from EvoDiff.
   - **MDLM**: Discrete diffusion with absorbing (mask) noise, 28.6M parameters.
   - **ARLM**: Autoregressive language model, fine-tuned from ProGen2-small, 151M parameters.

   Key finding: D3PM most faithfully captures the natural distribution with high generation diversity; continuous diffusion performs poorly; UDLM is prone to mode collapse.

2. **Three plug-and-play steering strategies**

   - **Classifier guidance (CG)**: Trains a time-dependent value function $p(y|x_t, t)$ and biases the rate matrix during the reverse discrete diffusion process. Requires a classifier trained at each noise level $t$.
   - **Decoupled annealed posterior sampling (DAPS)**: A variable-splitting approach that decomposes posterior sampling into alternating denoising and data-consistency steps. Requires only a clean-data value function $p(y|x_0)$; simpler and generally most effective.
   - **Noise-optimized sampling (NOS)**: Trains a value function in the continuous embedding space of discrete tokens and optimizes embeddings for higher fitness.

   Baseline: **DPO fine-tuning of ARLM**—direct preference optimization of language model weights.

3. **Adaptive optimization via Thompson sampling-style strategy**

   The approach simulates iterative protein engineering: each round samples a batch of sequences, evaluates fitness, and updates the guidance model.

   Key innovation:
   - A **frequentist ensemble** of 10 neural network regressors serves as the value function.
   - At each generation step, one value function is randomly drawn from the ensemble to guide sampling (analogous to Thompson sampling).
   - Ensemble predictive uncertainty promotes exploration of the design space, balancing exploration and exploitation.
   - MDLM combined with CG or DAPS serves as the primary configuration.

### Loss & Training

- **Prior pretraining**: Trained on MSA-aligned homologous sequences using standard diffusion or language model losses.
- **Value function training**: A regressor is trained on a small set (~200) of sequence–fitness pairs, modeled as $p(y|x) \propto \exp(f(x)/\beta)$.
- **Guidance strength control**: The temperature parameter $\beta$ or method-specific hyperparameters regulate guidance strength, balancing high fitness against sequence diversity.
- **Adaptive optimization**: 100 sequences are sampled per round; only unique and novel samples are retained.

## Key Experimental Results

### Main Results

Evaluation on three protein fitness datasets (TrpB enzymatic activity, CreiLOV fluorescence, GB1 binding):

| Method | Model | TrpB Mean Fitness | CreiLOV Mean Fitness | Advantage |
|--------|-------|-------------------|----------------------|-----------|
| DAPS | MDLM | Highest | Highest | Strongest steering, minimal hyperparameters |
| CG | MDLM | Second highest | Second highest | Strong steering |
| CG | D3PM | Competitive | Competitive | High prior quality |
| DPO | ARLM | Lower | Lower | Weak steerability under low data |
| NOS | D3PM | Lower | Lower | Limited steering range |
| Unguided | MDLM | Baseline | Baseline | Relies solely on evolutionary prior |

**Adaptive optimization** (multi-round iteration, MDLM + CG/DAPS):

| Method | Strategy | TrpB Max Fitness | CreiLOV Max Fitness | Notes |
|--------|----------|-----------------|---------------------|-------|
| DAPS + ensemble | Thompson sampling | Highest | Highest | Uncertainty-driven exploration |
| CG + ensemble | Thompson sampling | Second highest | Second highest | Same |
| Single value function | Greedy guidance | Lower | Lower | Insufficient exploration |
| APEXGo | Latent-space BO | Lower | Lower | Trust region difficult to calibrate under low data |
| DPO | ARLM fine-tuning | Lowest | Lowest | High compute cost, weak steerability |

### Ablation Study

| Configuration | Key Finding | Notes |
|--------------|-------------|-------|
| Uniform noise vs. absorbing noise | Comparable performance | D3PM ≈ MDLM |
| Continuous vs. discrete diffusion | Discrete superior | Continuous model prior quality inferior |
| ESM embedding space diffusion | No improvement | No additional benefit on this task |
| Ensemble vs. single value function | Ensemble superior | Thompson sampling improves maximum fitness |
| Varying label counts | 200 labels sufficient | Plug-and-play methods show clear advantage under low data |

### Key Findings

- **Plug-and-play guidance >> RL fine-tuning**: In the low-data regime (~200 labels), DAPS/CG guidance of discrete diffusion models substantially outperforms DPO fine-tuning of language models.
- **DAPS is overall best**: Requires only a clean-data value function (no time-dependent classifier), and achieves the strongest guidance.
- **Ensemble + Thompson sampling improves exploration**: Achieves higher maximum fitness in adaptive optimization.
- **Computationally efficient**: Prior pretraining takes <1 hour (single H100); steering experiments complete in minutes; DPO fine-tuning requires several hours.
- **Generated sequences are largely novel**, without over-steering toward known sequences.

## Highlights & Insights

- **Systematic benchmark**: The first comprehensive comparison of different generative model and steering strategy combinations on real protein fitness data, providing actionable guidelines.
- **DAPS adapted to discrete diffusion**: The first application of decoupled annealed posterior sampling to protein optimization with discrete diffusion models.
- **Practical advantages of plug-and-play**: Only a single hyperparameter (guidance strength) requires tuning; prior weights are not modified; computational cost is minimal.
- **Practical hyperparameter selection strategy**: Sweep guidance strength and select the maximum value that generates $n$ unique novel sequences (where $n$ equals the throughput of the next screening round).
- **Bayesian optimization principles integrated into generative frameworks**: Ensemble + Thompson sampling represents a natural combination of classical Bayesian optimization and modern generative models.

## Limitations & Future Work

- Only fitness objectives close to natural function are tested; generalization to non-natural activities (e.g., engineered novel enzymatic activities) remains unvalidated.
- Fitness is approximated via computational oracles rather than real wet-lab experimental validation.
- RL-based methods (DPO/RTB) on discrete diffusion models have not been tested.
- Thompson sampling employs a frequentist ensemble rather than a true Bayesian posterior (GP-based approaches perform poorly).
- Insertion and deletion mutations are not considered; only fixed-length sequence design is addressed.
- ARLM generation requires manual mapping back to the design space; more elegant alternatives such as inpainting have not been explored.

## Related Work & Insights

- **EvoDiff**: Discrete diffusion protein model from which the D3PM baseline is fine-tuned.
- **DAPS (Zhang et al., 2025)**: Decoupled annealed posterior sampling, first adapted here to discrete diffusion for proteins.
- **APEXGo**: Latent-space Bayesian optimization baseline.
- **ProGen2**: ARLM baseline.
- Broader implications: The low-data advantages of plug-and-play guidance combined with Bayesian exploration strategies are transferable to other discrete sequence design problems, including small molecules and controllable natural language generation.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic comparison of SGPO methods + first application of DAPS to protein optimization.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three proteins, multiple models and steering strategies, and adaptive optimization experiments are highly comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure and well-designed method taxonomy figures.
- Value: ⭐⭐⭐⭐ Provides practical best-practice guidelines for protein engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Flow Density Control: Generative Optimization Beyond Entropy-Regularized Fine-Tuning](flow_density_control_generative_optimization_beyond_entropy-regularized_fine-tun.md)
- [\[NeurIPS 2025\] Generative Modeling of Full-Atom Protein Conformations using Latent Diffusion on Graph Embeddings](generative_modeling_of_full-atom_protein_conformations_using_latent_diffusion_on.md)
- [\[NeurIPS 2025\] Multiscale Guidance of Protein Structure Prediction with Heterogeneous Cryo-EM Data](multiscale_guidance_of_protein_structure_prediction_with_heterogeneous_cryo-em_d.md)
- [\[NeurIPS 2025\] Energy Matching: Unifying Flow Matching and Energy-Based Models for Generative Modeling](energy_matching_unifying_flow_matching_and_energy-based_models_for_generative_mo.md)
- [\[NeurIPS 2025\] GraphFLA: Augmenting Biological Fitness Prediction Benchmarks with Landscape Features](augmenting_biological_fitness_prediction_benchmarks_with_landscapes_features_fro.md)

</div>

<!-- RELATED:END -->
