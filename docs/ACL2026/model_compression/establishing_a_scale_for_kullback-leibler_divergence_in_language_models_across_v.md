---
title: >-
  [Paper Note] Establishing a Scale for Kullback–Leibler Divergence in Language Models Across Various Settings
description: >-
  [ACL 2026][Model Compression][Paper Note] This paper utilizes log-likelihood vectors to embed language models of various architectures into a unified space, systematically measuring characteristic scales of KL divergence across settings including pre-training, model scale, random seeds, quantization, fine-tuning, and layers. It discovers that pre-training traj
tags:
  - ACL 2026
  - Model Compression
date: 2026-05-08
content_hash: 91a4e4bd0244b8bf
---
# Establishing a Scale for Kullback–Leibler Divergence in Language Models Across Various Settings

**Conference**: ACL 2026 Findings  
**arXiv**: [2505.15353](https://arxiv.org/abs/2505.15353)  
**Code**: [GitHub](https://github.com/shimo-lab/modelmap)  
**Area**: Model Analysis / Training Dynamics  
**Keywords**: KL Divergence, Language Models, Pre-training Trajectory, Log-likelihood Vectors, Anomalous Diffusion

## TL;DR

This paper utilizes log-likelihood vectors to embed language models of various architectures into a unified space, systematically measuring characteristic scales of KL divergence across settings including pre-training, model scale, random seeds, quantization, fine-tuning, and layers. It discovers that pre-training trajectories exhibit sub-diffusive behavior in log-likelihood space—model output distributions stabilize early despite continuous drifting in the weight space.

## Background & Motivation

**Background**: Understanding the learning dynamics and intermediate representations of language models requires quantifying behavioral changes and enabling cross-model comparisons. Traditional analyses rely on weight parameters; however, permutation symmetry and architectural dependence of weights hinder direct comparisons between models with different learning methods or designs.

**Limitations of Prior Work**: (1) Weight space comparisons are restricted by permutation symmetry—different configurations of hidden units can correspond to the same function; (2) Models with different architectures cannot be compared within the same coordinate system; (3) There is a lack of a unified metric scale to interpret the magnitude of KL divergence across different settings.

**Key Challenge**: An architecture-agnostic, interpretable, and unified metric is needed to compare behavioral differences in language models; however, existing methods either depend on architecture (weight space) or lack reference scales across settings.

**Goal**: Establish a consistent scale for KL divergence across multiple settings to provide a practical reference for model comparison.

**Key Insight**: Building on the log-likelihood vector framework proposed by Oyama et al. (2025), this work extends it to training checkpoints, quantized models, and intermediate layers for analysis within a unified coordinate system.

**Core Idea**: Log-likelihood vectors define a common space where the squared Euclidean distance approximates KL divergence, transforming model comparison into a geometric problem. Through systematic measurement, each setting is mapped to a characteristic KL divergence scale.

## Method

### Overall Architecture

A language model $p$ is represented as its log-likelihood vector $\ell = (\log p(x_1), \ldots, \log p(x_N))^\top$ over a predefined text set. Double-centering the log-likelihood matrix yields the $Q$ matrix, such that $2\text{KL}(p_i, p_j) \approx \|q_i - q_j\|^2 / N$. Various settings are systematically measured in this "model map."

### Key Designs

**1. Establishing KL Divergence Scales across Settings: Providing a Reference for "How Large is 0.1 bits/byte?"**

KL divergence is an abstract number; in isolation, its impact on behavioral differences is unclear. A value of 0.1 bits/byte might be significant between two adjacent checkpoints but negligible in cross-model comparisons. Using 10,000 text samples from the Pile corpus, this study normalizes KL divergence to bits/byte based on average text length and measures characteristic scales for six categories: adjacent checkpoints in late pre-training (~0.01–0.05), early training (~0.05–0.1), different random seeds (~0.1), different model scales (~0.15–1.7), 8-bit/4-bit quantization (~0.44/0.49), and fine-tuning (~0.40 bits/byte). This yardstick, spanning two orders of magnitude, allows any measured KL divergence to be contextualized.

**2. Pre-training Trajectory Diffusion Analysis: Weights Drift While Behavior Stabilizes**

It is often assumed that if weights are changing, the model is still learning; this is not necessarily true. This study measures diffusion indices for the Pythia series (410M–6.9B, 7 random seeds) in both weight and log-likelihood spaces. While the weight space follows standard Brownian motion (diffusion index $c_{w} \approx 1$), the log-likelihood space exhibits strong sub-diffusion ($c_{q} \approx 0.2$). This contrast indicates that output distributions stabilize early in training; thereafter, despite weight drift, behavior remains nearly fixed. This distinguishes "weight convergence" from "behavioral convergence."

**3. Hölder Regularity and Geometric Folding: Theoretical Explanation for Small Behavioral Changes From Large Weight Changes**

Sub-diffusion requires a mechanistic explanation. The effective Hölder index $\alpha = c_q / c_w \approx 0.2$ for the mapping $f: W \mapsto q(W)$ is calculated, which is significantly lower than the $\alpha = 1$ required for Lipschitz continuity. This suggests a strong "folding" effect: due to hidden unit permutation symmetry, numerous distinct weight configurations are compressed into identical or similar output distributions. This is corroborated by the effective fractal dimension—$D_w \approx 2$ on the weight side, expanding to $D_q \approx 10$ on the log-likelihood side. This geometric folding causes model behavior to remain static even as weights continue to wander in late training.

### Loss & Training

This study is an analytical work and does not involve new training strategies. It utilizes public pre-training checkpoints from the Pythia series (410M-6.9B) and a subset of 1,018 language models analyzed by Oyama et al. (2025). Layer-wise analysis treats each sub-network layer as an independent model using the logit lens.

## Key Experimental Results

### Main Results

| Setting | Median KL Divergence (bits/byte) |
| :--- | :--- |
| Late pre-training consecutive checkpoints | 0.011 |
| Early pre-training consecutive checkpoints | 0.067 |
| Different random seeds (410M) | 0.12 |
| Different model scales | 0.48 |
| 8-bit quantization | 0.44 |
| 4-bit quantization | 0.49 |
| Fine-tuning | 0.40 |
| Random pairs of the same type | 0.95 |
| Random pairs of different types | 2.2 |
| Adjacent layers | 3.0 |

### Ablation Study

| Model Scale | $c_w$ (Weight Diffusion) | $c_q$ (Likelihood Diffusion) | $\alpha$ (Hölder) |
| :--- | :--- | :--- | :--- |
| 410M | 1.1 | 0.15 | 0.14 |
| 1B | 0.83 | 0.20 | 0.24 |
| 1.4B | 0.91 | 0.21 | 0.23 |
| 2.8B | 0.90 | 0.26 | 0.29 |
| 6.9B | 0.92 | 0.33 | 0.36 |

### Key Findings

- KL divergence spans over two orders of magnitude (0.01 to 3.0 bits/byte) across settings, with each setting possessing a characteristic scale.
- KL divergence induced by quantization is highly consistent in direction and magnitude within the same model family (cosine similarity 0.91-0.98), suggesting quantization is a structured perturbation rather than random noise.
- Changes induced by fine-tuning (0.40) are smaller than random pairings of the same model type (0.95) and much smaller than different types (2.2).
- $\alpha$ increases with model scale, suggesting that weight-to-behavior mapping is smoother in larger models.

## Highlights & Insights

- The discovery of "weight drift but behavioral stability" implies that model convergence should be judged by output distributions rather than weight changes.
- The finding that quantization serves as a "structured perturbation" explains why quantized models generally maintain performance—the direction and magnitude of the perturbation are consistent within a model family.
- Quantitative links between weight space and behavioral space established via Hölder regularity provide a new perspective on "over-parameterization" in deep learning.
- The versatility of the log-likelihood vector framework is significant—it uniformly handles checkpoints, quantization, fine-tuning, and layer-wise analysis.

## Limitations & Future Work

- Only 10,000 texts from the Pile corpus were used; the impact of cross-domain text sets was not examined.
- Pre-training trajectory analysis is limited to the Pythia series with 1k-step checkpoint intervals; finer-grained behavior remains unknown.
- Layer-wise analysis using the logit lens is noisy in shallower layers; using a tuned lens might improve results but current availability is limited.
- The Hölder index was only estimated along training trajectories and does not represent a global property of the mapping.

## Related Work & Insights

- **vs. Weight Space Analysis**: Weight space cannot directly compare models with different architectures/methods due to permutation symmetry; the log-likelihood space overcomes this limitation.
- **vs. Kunin et al. (2024)**: While they found anomalous diffusion in weight space ($c_w \approx 1$), this work finds stronger sub-diffusion in log-likelihood space ($c_q \approx 0.2$), linked quantitatively via Hölder regularity.
- **vs. Oyama et al. (2025)**: This work extends their log-likelihood vector framework from fully trained models to checkpoints, quantization, and intermediate layers.

## Rating

- Novelty: ⭐⭐⭐⭐ Substantial expansion of the log-likelihood vector framework and the novelty of the sub-diffusion discovery; however, the core framework is based on prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers six settings and various model scales with detailed analysis, though model family coverage is limited.
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematically rigorous with excellent visualization and clear, interpretable conclusions.
- Value: ⭐⭐⭐⭐ Provides a practical quantitative reference for model comparison; sub-diffusion findings are significant for understanding training dynamics.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LightReasoner: Can Small Language Models Teach Large Language Models Reasoning?](lightreasoner_can_small_language_models_teach_large_language_models_reasoning.md)
- [\[NeurIPS 2025\] Order-Level Attention Similarity Across Language Models: A Latent Commonality](../../NeurIPS2025/model_compression/order-level_attention_similarity_across_language_models_a_latent_commonality.md)
- [\[ICML 2026\] Entropy-Aware On-Policy Distillation of Language Models](../../ICML2026/model_compression/entropy-aware_on-policy_distillation_of_language_models.md)
- [\[ACL 2026\] IntroLM: Introspective Language Models via Prefilling-Time Self-Evaluation](introlm_introspective_language_models_via_prefilling-time_self-evaluation.md)
- [\[ACL 2026\] Training-Free Test-Time Contrastive Learning for Large Language Models](training-free_test-time_contrastive_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
