---
title: >-
  [Paper Note] Establishing a Scale for Kullback–Leibler Divergence in Language Models Across Various Settings
description: >-
  [ACL 2026][Model Compression][KL divergence] This paper embeds language models of diverse architectures into a unified space via log-likelihood vectors…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "KL divergence"
  - "language models"
  - "pretraining trajectory"
  - "log-likelihood vectors"
  - "anomalous diffusion"
date: 2026-05-08
content_hash: 72bb46b3c4b6c19e
---

# Establishing a Scale for Kullback–Leibler Divergence in Language Models Across Various Settings

**Conference**: ACL 2026
**arXiv**: [2505.15353](https://arxiv.org/abs/2505.15353)
**Code**: [GitHub](https://github.com/shimo-lab/modelmap)
**Area**: Model Analysis / Training Dynamics
**Keywords**: KL divergence, language models, pretraining trajectory, log-likelihood vectors, anomalous diffusion

## TL;DR

This paper embeds language models of diverse architectures into a unified space via log-likelihood vectors, systematically measures the characteristic KL divergence scales across multiple settings—pretraining, model scale, random seeds, quantization, fine-tuning, and inter-layer analysis—and reveals that pretraining trajectories exhibit subdiffusive behavior in log-likelihood space: despite continuous drift in weight space, the output distributions stabilize early in training.

## Background & Motivation

**Background**: Understanding the learning dynamics and intermediate representations of language models requires quantifying behavioral changes and enabling cross-model comparison. Traditional analyses rely on weight parameters, but the permutation symmetry and architecture dependence of weights hinder direct comparison across models trained with different methods or designs.

**Limitations of Prior Work**: (1) Weight-space comparisons are confounded by permutation symmetry—hidden units arranged in different orders can implement identical functions; (2) models with different architectures cannot be compared within a common coordinate system; (3) there is no unified reference scale for interpreting the magnitude of KL divergence across different settings.

**Key Challenge**: An architecture-agnostic and interpretable unified metric is needed to compare behavioral differences between language models, yet existing approaches either depend on architecture (weight space) or lack cross-setting reference scales.

**Goal**: Establish a consistent scale for KL divergence across multiple settings to provide a practical reference for model comparison.

**Key Insight**: Building on the log-likelihood vector framework proposed by Oyama et al. (2025), this work extends it to training checkpoints, quantized models, and intermediate layers, enabling analysis within a unified coordinate system.

**Core Idea**: Log-likelihood vectors define a common space in which the squared Euclidean distance approximates KL divergence, reducing model comparison to a geometric problem. By systematic measurement, each setting yields a characteristic KL divergence scale.

## Method

### Overall Architecture

Each language model $p$ is represented as a log-likelihood vector $\ell = (\log p(x_1), \ldots, \log p(x_N))^\top$ evaluated on a predefined text corpus. Double-centering the log-likelihood matrix yields a matrix $Q$ such that $2\text{KL}(p_i, p_j) \approx \|q_i - q_j\|^2 / N$. KL divergences across various settings are then systematically measured within this "model map."

### Key Designs

1. **Establishing KL Divergence Scales Across Settings**:

    - Function: Provide interpretable quantitative references for model differences across diverse scenarios.
    - Mechanism: 10,000 texts from the Pile corpus serve as the text set; KL divergence is reported in bits/byte (normalized by average text length). Six categories of settings are systematically measured: (a) consecutive late-pretraining checkpoints: ~0.01–0.05 bits/byte; (b) early training: ~0.05–0.1 bits/byte; (c) different random seeds: ~0.1 bits/byte; (d) different model scales: ~0.15–1.7 bits/byte; (e) 8-bit/4-bit quantization: ~0.44/0.49 bits/byte; (f) fine-tuning: ~0.40 bits/byte.
    - Design Motivation: A value of 0.1 bits/byte may represent a substantial change between consecutive checkpoints yet be negligible in cross-model-type comparisons—cross-setting reference scales are necessary for correct interpretation.

2. **Diffusion Analysis of Pretraining Trajectories**:

    - Function: Reveal the stability characteristics of model behavior during training.
    - Mechanism: For the Pythia model family (410M–6.9B, 7 random seeds), diffusion exponents are analyzed in both weight space and log-likelihood space. Weight space exhibits Brownian motion ($c_w \approx 1$), whereas log-likelihood space exhibits strong subdiffusion ($c_q \approx 0.2$), indicating that output distributions stabilize early in training despite continued weight drift.
    - Design Motivation: Weight drift does not imply behavioral change—this finding has important implications for understanding when model training truly "converges."

3. **Hölder Regularity and Geometric Folding**:

    - Function: Explain why large changes in weight space correspond to small changes in log-likelihood space.
    - Mechanism: The mapping $f: W \mapsto q(W)$ from weights to log-likelihoods has an effective Hölder exponent $\alpha = c_q/c_w \approx 0.2$, far below Lipschitz continuity ($\alpha = 1$). This implies a strong "folding" effect—owing to the permutation symmetry of hidden units, many distinct weight configurations map to identical or nearly identical output distributions. The effective fractal dimensions are $D_w \approx 2$ and $D_q \approx 10$.
    - Design Motivation: Provides a theoretical explanation for why weights continue to drift in late training while model behavior remains essentially unchanged.

### Loss & Training

This is an analytical study and introduces no new training strategy. It uses publicly available pretraining checkpoints from the Pythia family (410M–6.9B) and a subset of the 1,018 language models analyzed by Oyama et al. (2025). For inter-layer analysis, the logit lens is applied to treat each layer's sub-network as an independent model.

## Key Experimental Results

### Main Results

| Setting | Median KL Divergence (bits/byte) |
|---------|----------------------------------|
| Late-pretraining consecutive checkpoints | 0.011 |
| Early-pretraining consecutive checkpoints | 0.067 |
| Different random seeds (410M) | 0.12 |
| Different model scales | 0.48 |
| 8-bit quantization | 0.44 |
| 4-bit quantization | 0.49 |
| Fine-tuning | 0.40 |
| Random pairs within same model type | 0.95 |
| Random pairs across model types | 2.2 |
| Adjacent layers | 3.0 |

### Ablation Study

| Model Scale | $c_w$ (weight diffusion) | $c_q$ (likelihood diffusion) | $\alpha$ (Hölder) |
|-------------|--------------------------|------------------------------|-------------------|
| 410M | 1.1 | 0.15 | 0.14 |
| 1B | 0.83 | 0.20 | 0.24 |
| 1.4B | 0.91 | 0.21 | 0.23 |
| 2.8B | 0.90 | 0.26 | 0.29 |
| 6.9B | 0.92 | 0.33 | 0.36 |

### Key Findings

- KL divergence spans more than two orders of magnitude across settings (0.01 to 3.0 bits/byte), with each setting exhibiting a characteristic scale.
- KL divergence induced by quantization is highly consistent in both direction and magnitude within the same model family (cosine similarity 0.91–0.98), indicating that quantization constitutes a structured perturbation rather than random noise.
- Fine-tuning-induced changes (0.40) are smaller than random within-type model pairs (0.95) and far smaller than cross-type pairs (2.2).
- The Hölder exponent $\alpha$ increases with model scale, suggesting that the weight-to-behavior mapping becomes smoother for larger models.

## Highlights & Insights

- The finding that "weights drift but behavior stabilizes" has important practical implications—model convergence should be assessed via output distributions rather than weight changes.
- The characterization of quantization as a "structured perturbation" explains why quantized models generally retain strong performance: the perturbation direction and magnitude are consistent within the same model family.
- Hölder regularity establishes a quantitative link between weight space and behavior space, offering a new perspective on understanding overparameterization in deep learning.
- The versatility of the log-likelihood vector framework is noteworthy—it uniformly handles checkpoints, quantization, fine-tuning, and inter-layer analysis.

## Limitations & Future Work

- Only 10,000 texts from the Pile corpus are used; the effect of text sets from other domains is not examined.
- Pretraining trajectory analysis is limited to the Pythia family with checkpoint intervals of 1k steps; finer-grained behavior remains unknown.
- Inter-layer analysis using the logit lens is noisy in shallow layers; the tuned lens may improve results but has limited current availability.
- The Hölder exponent is estimated only along the training trajectory and does not characterize the global properties of the mapping.

## Related Work & Insights

- **vs. Weight-space analysis**: Weight space cannot directly compare models of different architectures or training methods due to permutation symmetry; the log-likelihood space overcomes this limitation.
- **vs. Kunin et al. (2024)**: The latter identifies anomalous diffusion in weight space ($c_w \approx 1$); this work finds stronger subdiffusion in log-likelihood space ($c_q \approx 0.2$) and quantitatively links the two via Hölder regularity.
- **vs. Oyama et al. (2025)**: This work extends their log-likelihood vector framework from fully trained models to checkpoints, quantized models, and intermediate layers.

## Rating

- Novelty: ⭐⭐⭐⭐ Substantially extends the log-likelihood vector framework; the subdiffusion finding is novel, though the core framework builds on prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers six categories of settings and multiple model scales with careful analysis, though coverage of model families is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically rigorous, with excellent visualizations and clearly interpretable conclusions.
- Value: ⭐⭐⭐⭐ Provides a practical quantitative reference framework for model comparison; the subdiffusion finding has far-reaching implications for understanding training dynamics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](../../CVPR2026/model_compression/quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[NeurIPS 2025\] Order-Level Attention Similarity Across Language Models: A Latent Commonality](../../NeurIPS2025/model_compression/order-level_attention_similarity_across_language_models_a_latent_commonality.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](compositional_steering_of_large_language_models_with_steering_tokens.md)
- [\[ACL 2026\] SeLaR: Selective Latent Reasoning in Large Language Models](selar_selective_latent_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](training-free_test-time_contrastive_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
