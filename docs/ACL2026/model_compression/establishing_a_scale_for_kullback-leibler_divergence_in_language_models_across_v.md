---
title: >-
  [Paper Note] Establishing a Scale for Kullback–Leibler Divergence in Language Models Across Various Settings
description: >-
  [ACL 2026][Model Compression][KL Divergence] This paper utilizes log-likelihood vectors to embed language models of different architectures into a unified space…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "KL Divergence"
  - "Language Models"
  - "Pre-training Trajectory"
  - "Log-likelihood Vectors"
  - "Anomalous Diffusion"
date: 2026-05-08
content_hash: 66e8f2bf0fcf483e
---

# Establishing a Scale for Kullback–Leibler Divergence in Language Models Across Various Settings

**Conference**: ACL 2026  
**arXiv**: [2505.15353](https://arxiv.org/abs/2505.15353)  
**Code**: [GitHub](https://github.com/shimo-lab/modelmap)  
**Area**: Model Analysis / Training Dynamics  
**Keywords**: KL Divergence, Language Models, Pre-training Trajectory, Log-likelihood Vectors, Anomalous Diffusion

## TL;DR

This paper utilizes log-likelihood vectors to embed language models of different architectures into a unified space, systematically measuring the characteristic scales of KL divergence across settings such as pre-training, model scale, random seeds, quantization, fine-tuning, and layers. It discovers that pre-training trajectories exhibit sub-diffusive behavior in the log-likelihood space—despite continuous drift in the weight space, model output distributions tend to stabilize early in training.

## Background & Motivation

**Background**: Quantifying behavioral changes and performing cross-model comparisons are essential for understanding the learning dynamics and intermediate representations of language models. Traditional analyses rely on weight parameters, but the permutation symmetry and architecture dependence of weights hinder direct comparisons between models with different learning methods or designs.

**Limitations of Prior Work**: (1) Comparisons in weight space are constrained by permutation symmetry—different permutations of hidden units can correspond to the same function; (2) Models with different architectures cannot be compared within the same coordinate system; (3) There is a lack of a unified metric scale to interpret the "magnitude" of KL divergence across different settings.

**Key Challenge**: An architecture-agnostic, interpretable, and unified metric is needed to compare behavioral differences in language models; however, existing methods either depend on specific architectures (weight space) or lack a reference scale across various settings.

**Goal**: To establish a consistent scale for KL divergence across multiple settings, providing a practical reference for model comparison.

**Key Insight**: Building upon the log-likelihood vector framework proposed by Oyama et al. (2025), this work extends it to training checkpoints, quantized models, and intermediate layers for analysis in a unified coordinate system.

**Core Idea**: Log-likelihood vectors define a common space where the squared Euclidean distance approximates the KL divergence, transforming model comparison into a geometric problem. Through systematic measurement, a characteristic KL divergence scale is assigned to each setting.

## Method

### Overall Architecture

A language model $p$ is represented as its log-likelihood vector $\ell = (\log p(x_1), \ldots, \log p(x_N))^\top$ over a predefined text set. Double-centering the log-likelihood matrix yields a $Q$ matrix such that $2\text{KL}(p_i, p_j) \approx \|q_i - q_j\|^2 / N$. Various settings are systematically measured in this "model map."

### Key Designs

1. **Establishment of KL Divergence Scales Across Settings**:

    - Function: Provide an interpretable quantitative reference for model differences in various scenarios.
    - Mechanism: 10,000 text sequences from the Pile corpus are used as the text set, with KL divergence measured in bits/byte (normalized by average text length). Six categories are measured: (a) ~0.01-0.05 bits/byte between consecutive checkpoints in late pre-training; (b) ~0.05-0.1 bits/byte in early training; (c) ~0.1 bits/byte between different random seeds; (d) ~0.15-1.7 bits/byte for different model scales; (e) ~0.44/0.49 bits/byte for 8-bit/4-bit quantization; (f) ~0.40 bits/byte for fine-tuning.
    - Design Motivation: A value of 0.1 bits/byte might represent a significant change between consecutive checkpoints but a negligible difference when comparing across model types—a reference scale across settings is necessary for correct interpretation.

2. **Diffusion Analysis of Pre-training Trajectories**:

    - Function: Reveal the stability characteristics of model behavior during training.
    - Mechanism: For the Pythia series (410M-6.9B, 7 random seeds), diffusion exponents in both the weight space and log-likelihood space are analyzed. The weight space exhibits Brownian motion ($c_w \approx 1$), but the log-likelihood space shows strong sub-diffusion ($c_q \approx 0.2$), indicating that model output distributions stabilize early despite continuous weight drift.
    - Design Motivation: Weight drift does not equate to behavioral change—this finding is crucial for understanding when model training truly "converges."

3. **Hölder Regularity and Geometric Folding**:

    - Function: Explain why large changes in weight space correspond to small changes in log-likelihood space.
    - Mechanism: The mapping $f: W \mapsto q(W)$ from weights to log-likelihoods has an effective Hölder exponent $\alpha = c_q/c_w \approx 0.2$, which is significantly smaller than Lipschitz continuity ($\alpha = 1$). This implies a strong "folding" effect in the mapping—due to permutation symmetry of hidden units, many different weight configurations map to identical or similar output distributions. The effective fractal dimensions are $D_w \approx 2$ and $D_q \approx 10$.
    - Design Motivation: To provide a theoretical explanation for why model behavior remains essentially unchanged even as weights continue to drift in late training.

### Loss & Training

Ours is an analytical work and does not involve new training strategies. Publicly available pre-training checkpoints from the Pythia series (410M-6.9B) are used, alongside a subset of 1,018 language models analyzed by Oyama et al. (2025). Layer-wise analysis treats each sub-network as an independent model using the logit lens.

## Key Experimental Results

### Main Results

| Setting | Median KL Divergence (bits/byte) |
|------|--------------------------|
| Consecutive late pre-training checkpoints | 0.011 |
| Consecutive early pre-training checkpoints | 0.067 |
| Different random seeds (410M) | 0.12 |
| Different model scales | 0.48 |
| 8-bit quantization | 0.44 |
| 4-bit quantization | 0.49 |
| Fine-tuning | 0.40 |
| Random pairs of same architecture | 0.95 |
| Random pairs across architectures | 2.2 |
| Adjacent layers | 3.0 |

### Ablation Study

| Model Scale | $c_w$ (Weight Diffusion) | $c_q$ (Likelihood Diffusion) | $\alpha$ (Hölder) |
|----------|-----------------|-----------------|------------------|
| 410M | 1.1 | 0.15 | 0.14 |
| 1B | 0.83 | 0.20 | 0.24 |
| 1.4B | 0.91 | 0.21 | 0.23 |
| 2.8B | 0.90 | 0.26 | 0.29 |
| 6.9B | 0.92 | 0.33 | 0.36 |

### Key Findings

- KL divergence spans over two orders of magnitude (0.01 to 3.0 bits/byte) across different settings, with each setting having a characteristic scale.
- The direction and magnitude of KL divergence caused by quantization are highly consistent within the same model family (cosine similarity 0.91-0.98), suggesting quantization is a structured perturbation rather than random noise.
- Changes induced by fine-tuning (0.40) are smaller than random pairings of the same model type (0.95) and much smaller than cross-type pairings (2.2).
- $\alpha$ increases with model scale, suggesting that the weight-to-behavior mapping is smoother for larger models.

## Highlights & Insights

- The discovery that "weights drift while behavior remains stable" has important practical implications—judging model convergence should be based on output distribution rather than weight changes.
- Identifying quantization as a "structured perturbation" explains why quantized models generally maintain good performance—the direction and magnitude of perturbation are consistent within the same model family.
- A quantitative link between weight space and behavior space is established through Hölder regularity, providing a new perspective on understanding "over-parameterization" in deep learning.
- The versatility of the log-likelihood vector framework is impressive—it allows for a unified treatment of checkpoints, quantization, fine-tuning, and layer-wise analysis.

## Limitations & Future Work

- Only 10,000 text sequences from the Pile corpus were used; the influence of cross-domain text sets has not been tested.
- Pre-training trajectory analysis is limited to the Pythia series with 1k-step checkpoint intervals; finer-grained behavior remains unknown.
- Layer-wise analysis using the logit lens is noisy in shallower layers; using a tuned lens might improve results but currently has limited availability.
- The Hölder exponent is estimated only along training trajectories and does not characterize the global properties of the mapping.

## Related Work & Insights

- **vs. Weight Space Analysis**: Weight space cannot directly compare models of different architectures or methods due to permutation symmetry; the log-likelihood space overcomes this limitation.
- **vs. Kunin et al. (2024)**: While Kunin et al. found anomalous diffusion in weight space ($c_w \approx 1$), this work discovers stronger sub-diffusion in log-likelihood space ($c_q \approx 0.2$), with the two being quantitatively linked via Hölder regularity.
- **vs. Oyama et al. (2025)**: This work extends the log-likelihood vector framework from fully trained models to checkpoints, quantization, and intermediate layers.

## Rating

- Novelty: ⭐⭐⭐⭐ Significantly extends the log-likelihood vector framework; the sub-diffusion discovery is novel, though the core framework builds on prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers six categories of settings and multiple model scales with detailed analysis, though model family coverage is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically rigorous, excellent visualizations, and clear, interpretable conclusions.
- Value: ⭐⭐⭐⭐ Provides a practical quantitative reference framework for model comparison; findings on sub-diffusion have profound implications for understanding training dynamics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)
- [\[CVPR 2026\] QuantVLA: Scale-Calibrated Post-Training Quantization for Vision-Language-Action Models](../../CVPR2026/model_compression/quantvla_scale-calibrated_post-training_quantization_for_vision-language-action_.md)
- [\[NeurIPS 2025\] Order-Level Attention Similarity Across Language Models: A Latent Commonality](../../NeurIPS2025/model_compression/order-level_attention_similarity_across_language_models_a_latent_commonality.md)
- [\[ACL 2026\] IntroLM: Introspective Language Models via Prefilling-Time Self-Evaluation](introlm_introspective_language_models_via_prefilling-time_self-evaluation.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](training-free_test-time_contrastive_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
