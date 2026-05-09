---
title: >-
  [Paper Note] Neurosymbolic Diffusion Models
description: >-
  [NeurIPS 2025][Autonomous Driving][neurosymbolic] This paper proposes Neurosymbolic Diffusion Models (NeSyDM), which integrates discrete masked diffusion models with symbolic programs to overcome the conditional independence assumption in traditional neurosymbolic predictors. NeSyDM models inter-concept dependencies and uncertainty while maintaining scalability, achieving state-of-the-art accuracy and calibration on visual reasoning and autonomous driving benchmarks.
tags:
  - NeurIPS 2025
  - Autonomous Driving
  - neurosymbolic
  - diffusion models
  - discrete diffusion
  - visual reasoning
date: 2026-05-08
content_hash: a6ed585a1a12573b
---

# Neurosymbolic Diffusion Models

**Conference**: NeurIPS 2025
**arXiv**: [2505.13138](https://arxiv.org/abs/2505.13138)
**Code**: None
**Area**: Autonomous Driving
**Keywords**: neurosymbolic, diffusion models, discrete diffusion, visual reasoning, autonomous driving

## TL;DR

This paper proposes Neurosymbolic Diffusion Models (NeSyDM), which integrates discrete masked diffusion models with symbolic programs to overcome the conditional independence assumption in traditional neurosymbolic predictors. NeSyDM models inter-concept dependencies and uncertainty while maintaining scalability, achieving state-of-the-art accuracy and calibration on visual reasoning and autonomous driving benchmarks.

## Background & Motivation

1. **Core problem in neurosymbolic prediction**: Existing neurosymbolic (NeSy) predictors extract symbolic concepts via neural networks and infer output labels through symbolic programs. Since training relies only on input-output pairs and concepts are unannotated latent variables, there is a risk of "Reasoning Shortcuts" (RS) where learned concepts diverge from their intended semantics.

2. **Fundamental flaw of the conditional independence assumption**: The vast majority of NeSy predictors assume concepts are conditionally independent given the input, i.e., $p_\theta(\mathbf{c}|\mathbf{x}) = \prod_i p_\theta(c_i|\mathbf{x})$. Although this assumption enables efficient Weighted Model Counting (WMC), it has been theoretically shown to be incapable of simultaneously expressing correct uncertainty and maximizing likelihood.

3. **Reasoning shortcuts cause OOD generalization failure**: When data and programs admit multiple valid concept assignments, independent models can only deterministically commit to one mapping, failing to distribute probability mass over all consistent concept configurations, leading to overconfidence and poor out-of-distribution generalization.

4. **Insufficient scalability of existing alternatives**: Mixture models and probabilistic circuits require knowledge compilation (worst-case exponential time) and do not scale to high-dimensional problems; autoregressive models do not permit efficient marginalization due to the non-commutativity of marginalization and conditional factorization.

5. **Compatibility with masked diffusion models**: Masked Diffusion Models (MDMs) employ a local conditional independence assumption at each denoising step while globally modeling dependencies. This is highly compatible with the independence structure of NeSy predictors, enabling direct reuse of efficient NeSy inference mechanisms.

6. **Requirements of real-world scenarios such as autonomous driving**: In autonomous driving tasks such as BDD-OIA, models must extract high-level concepts (e.g., pedestrian presence, traffic light states) from dashcam images and reason about permitted driving actions via logical rules. Concept calibration and uncertainty quantification are critical for safety.

## Method

### Core Framework: NeSyDM

NeSyDM integrates MDMs into NeSy predictors through three extensions:

1. **Input conditioning**: The denoising model $p_\theta(\tilde{\mathbf{c}}^0 | \mathbf{c}^t, \mathbf{x})$ is conditioned on input $\mathbf{x}$.
2. **Joint modeling of concepts and outputs**: The diffusion process is defined jointly over concepts $\mathbf{c}$ and outputs $\mathbf{y}$, with concepts treated as latent variables.
3. **Symbolic program feedback**: A program $\varphi$ maps concepts to outputs, providing differentiable gradient signals.

### Forward Process

A continuous-time masked diffusion process is adopted, progressively masking data $\mathbf{c}^0$ to $\mathbf{c}^t$:

$$q(\mathbf{c}^t | \mathbf{c}^s) = \prod_{i=1}^C \frac{\alpha_t}{\alpha_s} \mathbb{1}[c_i^t = c_i^s] + (1 - \frac{\alpha_t}{\alpha_s})\mathbb{1}[c_i^t = \text{m}]$$

Each dimension is masked with probability $1 - \alpha_t/\alpha_s$, and once masked, remains masked.

### Loss Function (NELBO)

The NeSyDM loss comprises three components:

- **Concept unmasking loss $\mathcal{L}_\mathbf{c}$**: Analogous to standard MDM; samples $\mathbf{c}^0$ from the variational distribution, partially masks them, and requires the model to reconstruct the originals.
- **Output unmasking loss $\mathcal{L}_\mathbf{y}$**: WMC is computed independently for each output dimension, leveraging the conditional independence of the concept denoising model for efficient computation.
- **Variational entropy $\mathcal{L}_{H[q]}$**: Maximizes the entropy of the variational distribution, encouraging coverage of all concept assignments consistent with the input-output pair.

### Variational Posterior and Gradient Estimation

Since exact sampling from the constrained posterior is NP-hard, a relaxed constraint $r_\beta$ is used for approximate sampling: $K$ samples are drawn from $p_\theta$ and the sample with the fewest constraint violations is selected. Gradient optimization employs the REINFORCE Leave-One-Out (RLOO) estimator, decomposing the WMC problem into $Y$ independent subproblems.

### Inference

A majority voting strategy is used: $L$ concept samples $\mathbf{c}_l^0$ are drawn, outputs are computed via program $\varphi$, and the most frequent output is taken as the prediction.

## Key Experimental Results

### Experiment 1: Scalability (Visual Path Planning)

| Method | 12×12 Accuracy | 30×30 Accuracy |
|--------|---------------|---------------|
| I-MLE (continuous cost) | 97.20±0.5 | 93.70±0.6 |
| EXAL | 94.19±1.74 | 80.85±3.83 |
| A-NeSI | 94.57±2.27 | 17.13±16.32 |
| A-NeSI+RL | 98.96±1.33 | 67.57±36.76 |
| **NeSyDM (Ours)** | **99.41±0.06** | **97.40±1.23** |

On the 30×30 grid ($5^{900}$ combinatorial space), NeSyDM achieves 97.40% accuracy, substantially outperforming all baselines. A-NeSI collapses on high-dimensional problems due to the independence assumption (only 17.13%), while NeSyDM exhibits markedly lower variance (0.06 vs. 1.33), demonstrating superior reliability.

### Experiment 2: Reasoning Shortcut Awareness (RSBench)

| Method | MNIST Half Acc_c(ID)↑ | ECE_c(ID)↓ | MNIST E-O ECE_c(ID)↓ |
|--------|-----------------------|------------|----------------------|
| PNP (independent) | 42.76±0.14 | 69.40±0.35 | 81.04±1.15 |
| SL (independent) | 42.88±0.09 | 70.61±0.18 | 82.18±1.57 |
| BEARS (ensemble) | 43.26±0.75 | 36.81±0.17 | 28.82±2.19 |
| NeSyDM (conditional entropy) | **71.16±1.77** | **4.18±2.56** | **2.70±1.21** |

NeSyDM achieves comprehensive improvements in both concept accuracy and calibration error (ECE). The conditional entropy variant achieves an ECE of only 4.18%, far below the 69–70% of independent models and 36.81% of BEARS, demonstrating that NeSyDM effectively mitigates reasoning shortcuts. On the BDD-OIA autonomous driving task, NeSyDM improves both output prediction performance and concept calibration simultaneously.

## Highlights & Insights

- **Solid theoretical contributions**: Proves that the continuous-time NELBO of MDMs extends to non-factorized distributions, providing theoretical foundations for MDM architectures beyond the NeSy setting.
- **Exceptional scalability**: Achieves 97.4% accuracy on 30×30 path planning (900-dimensional discrete space), surpassing the best baseline by 27 percentage points.
- **Outstanding uncertainty quantification**: Reduces ECE calibration error from the 70% range to 4%, enabling reliable deployment in settings such as active learning.
- **Elegant architecture**: Reusing existing NeSy predictor networks requires only additional conditioning on the current masked state, incurring low implementation overhead.

## Limitations & Future Work

- **Variational entropy estimation is a biased approximation**: Unconditional or 1-step approximations are used rather than exact variational entropy, offering limited theoretical guarantees.
- **Slower inference**: Multi-step diffusion sampling combined with majority voting requires multiple network forward passes, making inference orders of magnitude slower than independent models.
- **Sensitivity to loss weight hyperparameters**: The weights $\gamma_\mathbf{c}, \gamma_\mathbf{y}, \gamma_H$ for the three loss components critically affect performance; ablation studies show that improper settings significantly degrade results.
- **RLOO gradient estimation still exhibits variance in high-dimensional settings**: When the probability space is extremely large, the probability of sampling consistent concepts is low, potentially yielding insufficient gradient signal.

## Related Work & Insights

### vs. BEARS (Marconato et al., 2024)

BEARS achieves RS awareness through ensembles of independent distributions, requiring knowledge compilation to build logical circuits and additional parameters for each mixture component. NeSyDM naturally models dependencies through the diffusion process without explicit circuit compilation, and scales far better than BEARS on high-dimensional problems (BEARS is infeasible on 30×30 path planning). However, BEARS offers faster single-step inference.

### vs. A-NeSI (van Krieken et al., 2023)

A-NeSI is an efficient approximate NeSy method that retains the independence assumption; it performs comparably to NeSyDM on low-dimensional problems (MNIST addition: 92.56 vs. 92.49), but collapses on high-dimensional path planning (17.13% vs. 97.40%). The advantages of NeSyDM are most pronounced on complex tasks that require modeling inter-concept dependencies.

### vs. I-MLE (Niepert et al., 2021)

I-MLE uses continuous cost prediction rather than a discrete NeSy framework, achieving 93.70% on 30×30 path planning. NeSyDM surpasses I-MLE (97.40%) while using discrete concepts, and additionally provides interpretable concept extraction and uncertainty quantification — capabilities absent in I-MLE.

## Rating

| Dimension | Score | Comments |
|-----------|-------|----------|
| Novelty | ⭐⭐⭐⭐ | First integration of discrete diffusion models into NeSy predictors, with innovations in both theory and methodology |
| Technical Depth | ⭐⭐⭐⭐⭐ | Rigorous continuous-time NELBO derivation, non-factorized extension theorem, and scalable gradient estimation |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Covers synthetic and real-world tasks, multiple baselines, and 10-seed statistical testing, but lacks inference efficiency comparisons |
| Practical Value | ⭐⭐⭐ | Practically meaningful for safety-critical applications such as autonomous driving, but inference overhead and hyperparameter sensitivity limit deployment |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Distilling Diffusion Models to Efficient 3D LiDAR Scene Completion](../../ICCV2025/autonomous_driving/distilling_diffusion_models_to_efficient_3d_lidar_scene_completion.md)
- [\[NeurIPS 2025\] SQS: Enhancing Sparse Perception Models via Query-based Splatting in Autonomous Driving](sqs_enhancing_sparse_perception_models_via_query-based_splatting_in_autonomous_d.md)
- [\[NeurIPS 2025\] Towards Foundational LiDAR World Models with Efficient Latent Flow Matching](towards_foundational_lidar_world_models_with_efficient_latent_flow_matching.md)
- [\[NeurIPS 2025\] CymbaDiff: Structured Spatial Diffusion for Sketch-based 3D Semantic Urban Scene Generation](cymbadiff_structured_spatial_diffusion_for_sketch-based_3d_semantic_urban_scene_.md)
- [\[NeurIPS 2025\] RAW2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving](raw2drive_reinforcement_learning_with_aligned_world_models_for_end-to-end_autono.md)

</div>

<!-- RELATED:END -->
