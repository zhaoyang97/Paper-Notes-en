---
title: >-
  [Paper Note] Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks
description: >-
  [ICML 2025][LLM Pretraining][Neural Scaling Laws] The first Bayesian extrapolation method for Neural Scaling Laws. By designing customized prior distributions (covering Down, Down-Down, and Down-Up-Down functional families) and leveraging Prior-data Fitted Networks (PFNs) to meta-learn extrapolation capability, this approach outperforms existing methods in both point estimation accuracy and uncertainty quantification quality.
tags:
  - "ICML 2025"
  - "LLM Pretraining"
  - "Neural Scaling Laws"
  - "Bayesian Inference"
  - "PFN"
  - "Uncertainty Quantification"
  - "Meta-learning"
  - "Extrapolation Prediction"
date: 2026-05-08
content_hash: b5ca56b2e5d79c5a
---

# Bayesian Neural Scaling Law Extrapolation with Prior-Data Fitted Networks

**Conference**: ICML 2025  
**arXiv**: [2505.23032](https://arxiv.org/abs/2505.23032)  
**Code**: [github.com/DongWooLee-Eli/nslpfn](https://github.com/DongWooLee-Eli/nslpfn)  
**Area**: LLM Pre-training  
**Keywords**: Neural Scaling Laws, Bayesian Inference, PFN, Uncertainty Quantification, Meta-learning, Extrapolation Prediction

## TL;DR

The first Bayesian extrapolation method for Neural Scaling Laws. By designing customized prior distributions (covering Down, Down-Down, and Down-Up-Down functional families) and leveraging Prior-data Fitted Networks (PFNs) to meta-learn extrapolation capability, this approach outperforms existing methods in both point estimation accuracy and uncertainty quantification quality.

## Background & Motivation

Scaling is a primary driver behind recent advances in deep learning. Numerous empirical studies demonstrate that scaling laws generally follow power laws, which has led to various functional variants being proposed to predict large-scale behavior. However, existing methods suffer from fundamental limitations:

**Point estimation only**: They fail to quantify uncertainty, which is crucial for decision-making (e.g., "is it worth investing an extra $x \times$ compute?").

**Fixed functional forms**: A specific functional family (such as M1-M4 or BNSL) must be pre-selected; mis-specification leads to extrapolation failure.

**Inability to handle chaotic behaviors**: Such as double descent (decrease-increase-decrease), which simple functional families cannot represent.

**Difficulties in MCMC methods**: Setting priors for non-monotonic behavior is challenging, and the optimization landscape is non-convex and multi-modal.

Key Motivation: The true value of scaling law extrapolation lies in guiding high-stakes decisions (e.g., allocating compute budget, collecting data); relying solely on point estimates is too risky. This work addresses this issue by combining the flexible priors of PFNs with amortized inference.

## Method

### Overall Architecture

NSL-PFN (Neural Scaling Law PFN) is built on the following core ideas:

1. Design a prior distribution capable of sampling infinite synthetic scaling curves.
2. Meta-train a PFN (Transformer) on these synthetic datasets to learn to predict future behavior and uncertainty given partial observations.
3. Perform extrapolation during inference in a single forward pass.

Mathematical foundation of PFN: The training objective is equivalent to minimizing the KL divergence between the predictive posterior distribution $q_\theta$ and the true prior predictive distribution (PPD):

$$\min_\theta \mathbb{E}_{p(\mathcal{C}, X^*)}[\text{KL}[p(Y^*|X^*, \mathcal{C}) \| q_\theta(Y^*|X^*, \mathcal{C})]]$$

### Key Designs

**Prior design for three functional families**:

1. **Down**: A simple monotonic decreasing trend without breaks. It randomly selects $\mathcal{M}_3$ or $\mathcal{M}_4$:

    - $\mathcal{M}_3: y = a(x^{-1} + d)^b$ (power law with offset)
    - $\mathcal{M}_4: y = g^{-1}(x)$, containing $\alpha$ to control the inflection point (initially flat, followed by a sharp decline)

2. **Down-Down**: A complex decreasing trend with breaks. It randomly selects 1-3 breaks (creating 2-4 segments), sampling each segment independently from $\mathcal{M}_3$ or $\mathcal{M}_4$, and aligning them via translation along the y-axis.

3. **Down-Up-Down**: Captures chaotic behaviors like double descent. It uses exactly 2 breaks (3 segments), where the first and third segments are modeled with $\mathcal{M}_3$/$\mathcal{M}_4$ (decreasing), and the second segment uses the **BetaCDF** (an ascending S-shaped curve).

**Auxiliary functions**:
- **Norm**: Renormalizes each segment using the sampled max/min values.
- **Noise**: Adds observation noise (likelihood function).

**Meticulous design of cutoff distributions**:
- Down: Cutoff can occur at any position.
- Down-Down: Cutoff occurs only within the final segment (assuming future breaks are unpredictable).
- Down-Up-Down: Cutoff occurs in either the second or third segment (if in the ascending phase, the model must learn to predict that it will eventually decrease).

Design Philosophy: This distinguishes two scenarios — (1) when the final segment is decreasing, assuming the trend continues; (2) when the final segment is increasing, utilizing empirical knowledge that it will eventually decrease.

### Loss & Training

**Training Objective** (augmented version):

$$\max_\theta \mathbb{E}[\log q_\theta(Y^*|X^*, \mathcal{C}) + \log q_\theta(Y|X, \mathcal{C})]$$

The second term is the context regression loss (auto-regressive objective), which improves fitting quality near the cutoff point.

**Interpolation loss** (optional): To facilitate Bayesian active learning, targets are randomly sampled to be included in the context, training the model to possess both interpolation and extrapolation capability.

**Architecture**: A 12-layer, 4-head Transformer with a hidden dimension of 512, discretizing the output distribution into 1000 bins. It is trained on 1.6M synthetic curves for 100K iterations (approx. 2.6 hours on a single A100 GPU).

## Key Experimental Results

### Main Results

**Datasets**: IC (72 vision curves), NLP (20 NMT/LM/BB curves), Nano (24 nanoGPT curves), ColPret (192 LLM curves), DD (16 double descent curves).

**Image Domain (IC) Average RMSLE↓ / LL↑**:

| Method | RMSLE | LL |
|------|-------|-----|
| M4 (Best Point Estimate) | 0.0415 | - |
| BNSL | 0.0406 | - |
| MCMC (M4) | 0.0422 | 3.024 |
| MCMC (BNSL) | 0.0645 | 2.921 |
| LC-PFN | 0.0428 | 2.429 |
| **NSL-PFN** | **0.0280** | **3.330** |

**Language Domain (NLP+Nano) Average**:

| Method | RMSLE | LL |
|------|-------|-----|
| BNSL | 0.0223 | - |
| MCMC (M4) | 0.0235 | 2.216 |
| LC-PFN | 0.0398 | 1.519 |
| **NSL-PFN** | **0.0194** | **2.773** |

**Double Descent (DD)**:

| Method | RMSLE | LL |
|------|-------|-----|
| BNSL | 0.0468 | - |
| MCMC (BNSL) | 0.0494 | 1.250 |
| LC-PFN | 0.0706 | 1.321 |
| **NSL-PFN** | **0.0335** | **2.565** |

### Ablation Study

**Prior Design Ablation** (Table 5):

| Config | IC RMSLE | DD RMSLE |
|------|----------|----------|
| M3 only | 0.064 | 0.131 |
| M4 only | 0.050 | 0.058 |
| M3+M4 | 0.038 | 0.071 |
| +Breaks | 0.032 | 0.051 |
| +Up (Full) | **0.028** | **0.033** |

Every design choice contributes, with the Up segment yielding the largest improvement for DD ($0.051 \rightarrow 0.033$).

**Inference Efficiency** (Table 7, average seconds per curve):

| M4 | BNSL | MCMC(M4) | MCMC(BNSL) | LC-PFN | NSL-PFN |
|----|------|----------|------------|--------|---------|
| 15.65 | 98.79 | 154.64 | 280.55 | **0.02** | **0.02**|

NSL-PFN is approximately 4-5 orders of magnitude faster!

### Key Findings

1. NSL-PFN achieves optimal or near-optimal RMSLE and LL across all datasets.
2. The advantage is most pronounced in DD: while MCMC methods collapse during the ascending phase, NSL-PFN successfully predicts the eventual decrease.
3. Increasing the number of MCMC samples (300 to 3000) yields almost no improvement, and in some cases, even degrades performance.
4. In Bayesian active learning experiments, NSL-PFN significantly outperforms baselines, continuously improving as more observations are added.
5. Calibration (measured by MSCE) is also significantly better than all baselines, particularly on ColPret and DD.

## Highlights & Insights

1. **Prior design is the core innovation**: Instead of naively applying PFN, the authors carefully designed three families (Down, Down-Down, and Down-Up-Down) based on a deep understanding of scaling behavior, with distinct cutoff strategies for each.
2. **Automatic inference of functional forms and break counts**: Unlike BNSL, which requires cross-validation, PFN naturally performs model selection during inference.
3. **Extremely fast inference**: Driven by a single forward pass, it is $10000\times$ faster than MCMC, making it highly practical for large-scale applications.
4. **High-quality uncertainty estimation**: It excels in data-constrained scenarios, such as Bayesian active learning.
5. **Elegant handling of double descent**: By incorporating the Down-Up-Down pattern into the prior, the model "knows" that an upward turn will eventually be followed by a decline.

## Limitations & Future Work

1. **Prior design relies heavily on domain knowledge**: Handcrafting the three functional families requires a deep understanding of scaling behaviors.
2. **Potential over-modeling of simple scaling laws**: It is slightly outperformed by MCMC (M4) on simple curves like NMT/LM.
3. **Regularization issues**: Prior hyperparameters are tuned manually via visual matching; Bayesian optimization only yields modest improvements.
4. **Output discretization**: Discretizing the output into 1000 bins might lead to accuracy loss at extreme values.
5. **No testing on multi-variable scaling laws**: E.g., joint scaling laws that simultaneously scale dataset size and model parameter size.
6. **Overly restrictive cutoff assumptions**: The assumption in Down-Down that the cutoff only occurs in the final segment may not always hold true.

## Related Work & Insights

- **Relationship with LC-PFN (Adriaensen et al., 2023)**: LC-PFN is a PFN designed for learning curves (training epochs), whose prior is not tailored for scaling laws. NSL-PFN's prior is specially targeted to cover power-law variants, breaks, and double descent.
- **Complementarity with BNSL (Caballero et al., 2022)**: BNSL introduced the concept and functional forms of breaks but only provides point estimates. NSL-PFN builds upon BNSL's functional families with Bayesian inference.
- **Connection to Chinchilla (Hoffmann et al., 2022)**: NSL-PFN can help facilitate more reliable decisions regarding compute-optimal training.
- **Insights**: The PFN paradigm of "prior $\rightarrow$ finite synthetic data $\rightarrow$ meta-learn" is highly applicable to other structured extrapolation problems requiring uncertainty quantification.

## Rating

- Novelty: ⭐⭐⭐⭐ (4/5) — First Bayesian neural scaling law method, with an exquisitely designed prior.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (5/5) — Evaluated on 6 datasets with various baselines; comprehensive studies across ablation, efficiency, calibration, and active learning.
- Writing Quality: ⭐⭐⭐⭐ (4/5) — Well-structured with rich illustrations.
- Value: ⭐⭐⭐⭐⭐ (5/5) — High practical value, addressing a fundamental pain point in scaling law prediction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Position: The Future of Bayesian Prediction Is Prior-Fitted](position_the_future_of_bayesian_prediction_is_prior-fitted.md)
- [\[ICML 2025\] Does Data Scaling Lead to Visual Compositional Generalization?](does_data_scaling_lead_to_visual_compositional_generalization.md)
- [\[NeurIPS 2025\] Generalization Bounds for Rank-sparse Neural Networks](../../NeurIPS2025/llm_pretraining/generalization_bounds_for_rank-sparse_neural_networks.md)
- [\[NeurIPS 2025\] Neural Collapse under Gradient Flow on Shallow ReLU Networks for Orthogonally Separable Data](../../NeurIPS2025/llm_pretraining/neural_collapse_under_gradient_flow_on_shallow_relu_networks_for_orthogonally_se.md)
- [\[ICML 2025\] Scaling Inference-Efficient Language Models](scaling_inference-efficient_language_models.md)

</div>

<!-- RELATED:END -->
