---
title: >-
  [Paper Note] RollingQ: Reviving the Cooperation Dynamics in Multimodal Transformer
description: >-
  [ICML2025][Multimodal VLM][Multimodal fusion] This work reveals that the self-attention mechanism in multimodal Transformers loses its dynamic adaptability (favoring a single modality) due to a "self-reinforcing loop." It proposes the RollingQ algorithm to break this loop by rotating query vectors, thereby reviving cross-modal cooperation dynamics.
tags:
  - "ICML2025"
  - "Multimodal VLM"
  - "Multimodal fusion"
  - "Attention mechanism"
  - "Modality bias"
  - "Dynamic fusion"
  - "Transformer"
date: 2026-05-08
content_hash: 9bbfb999dffc2e94
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# RollingQ: Reviving the Cooperation Dynamics in Multimodal Transformer

**Conference**: ICML2025  
**arXiv**: [2506.11465](https://arxiv.org/abs/2506.11465)  
**Code**: [GeWu-Lab/RollingQ_ICML2025](https://github.com/GeWu-Lab/RollingQ_ICML2025)  
**Area**: Multimodal VLM  
**Keywords**: Multimodal fusion, Attention mechanism, Modality bias, Dynamic fusion, Transformer

## TL;DR

This work reveals that the self-attention mechanism in multimodal Transformers loses its dynamic adaptability (favoring a single modality) due to a "self-reinforcing loop." It proposes the RollingQ algorithm to break this loop by rotating query vectors, thereby reviving cross-modal cooperation dynamics.

## Background & Motivation

The core challenge of multimodal learning lies in how to effectively fuse information from different modalities. Existing fusion paradigms can be categorized into:

- **Static Fusion**: Employs fixed weights for each modality during inference, assuming uniform contributions.
- **Dynamic Fusion**: Adaptively adjusts the weights of each modality based on input data characteristics (e.g., Transformer attention mechanisms).

The authors observe a counter-intuitive phenomenon on the Kinetic-Sound dataset: **dynamic fusion based on self-attention (67.0%) performs even worse than simple static fusion concatenation (68.0%)**.

A deep analysis of the attention scores reveals that the model allocates disproportionately high attention to the audio modality; **even when the audio input is replaced with Gaussian noise, the model still overwhelmingly attends to the audio**. This indicates that the attention mechanism completely loses its dynamic adaptability.

**Root Cause Analysis**: There is a significant distribution gap in attention keys across different modalities. The keys of the biased modality consistently exhibit higher cosine similarity with the query of the class token, leading to a "self-reinforcing loop":

1. Forward propagation: The biased modality receives higher attention scores.
2. Backward propagation: Higher attention scores provide larger gradients to the encoder of the biased modality.
3. Feature quality of the biased modality is further enhanced $\rightarrow$ returning to step 1.

This loop continuously exacerbates the key distribution gap between modalities, eventually causing the attention mechanism to lose its capability for dynamic adjustment.

## Method

### Overall Architecture

Given bimodal inputs $x_i = (x_i^a, x_i^v)$, features $z_i^m = \Phi^m(x_i^m; \theta^m)$ are obtained through their respective encoders. In an attention layer, the query of the class token $q = z_{cls} W^Q$ performs a dot product with the keys of each modality to compute the attention score:

$$A_i = \frac{q K_i^T}{\sqrt{d}}, \quad h_i = \text{softmax}(A_i) V_i$$

where $K_i = [K_i^a, K_i^v]$. The attention score for modality $m$ can be decomposed as:

$$\sum_{j=1}^{L^m} \frac{q k_{(i,j)}^m}{\sqrt{d}} = \frac{L^m}{\sqrt{d}} \|q\|_2 \|\hat{k}_i^m\|_2 \cos\theta_i^m$$

where $\hat{k}_i^m$ represents the average key of modality $m$, and $\cos\theta_i^m$ is the cosine similarity between the query and the average key.

### Theoretical Analysis of the Self-Reinforcing Loop

- **Early Training Phase**: $\mathbb{E}[Q] = 0$, where $Q$ and $\hat{K}^m$ are independent, hence the expected attention scores for both modalities are identical.
- **During Training**: Due to the greedy nature of multimodal learning, the model tends to favor the modality with higher feature quality, leading to a gap in key distributions.
- **Gradient Analysis**: The biased modality obtains a larger $s(\cdot) \frac{\partial V_i^m}{\partial z_i^m}$ term, further reinforcing the optimization of its encoder.

### RollingQ Algorithm

**Step 1: Detecting the Self-Reinforcing Loop** — Define the Attention Imbalance Rate (AIR):

$$AIR = \mathbb{E}[\cos\theta^a - \cos\theta^v] \in [-2, 2]$$

When $|AIR| \geq \beta$ (threshold hyperparameter), the distribution gap is considered significant.

**Step 2: Constructing a Balanced Anchor** — Compute the target query position that distributes more attention to the weaker modality:

$$q_b = \left(\alpha \frac{\mathbb{E}[\hat{K}^a]}{\|\mathbb{E}[\hat{K}^a]\|_2} + (1-\alpha) \frac{\mathbb{E}[\hat{K}^v]}{\|\mathbb{E}[\hat{K}^v]\|_2}\right) \|\mathbb{E}[Q]\|_2$$

where the weight $\alpha$ is determined by the AIR:

$$\alpha = \frac{1}{2}[1 + \tanh(-\rho \cdot AIR)]$$

When $AIR > 0$ (modality $a$ is biased), $\alpha < 0.5$, pulling the anchor closer to the weaker modality $v$.

**Step 3: Query Rotation** — Compute the rotation matrix via SVD:

$$R_b = SVD([\mathbb{E}[Q], q_b]), \quad q_r = q \cdot R_b$$

The rotated query learns in a new balanced regime, encouraging a reduction in the key distribution gap between modalities.

### Implementation Details

- Hyperparameters: $\rho > 0$ controls the rotation intensity, and $\beta$ is the AIR threshold.
- Rotation limits: Capped at 1 rotation for CREMA-D/MOSEI, and 3 rotations for Kinetic-Sound.
- Multi-layer Extension: Uses progressive training, applying RollingQ to each layer sequentially.
- Extra parameters: Only increases parameters by approximately **1%** and GFLOPs by about **0.1%**.

## Key Experimental Results

### Main Results (Table 1)

| Method | Type | CREMA-D Acc | Kinetic-Sound Acc | CMU-MOSEI Acc |
|------|------|-------------|-------------------|---------------|
| Audio (Unimodal) | - | 47.6 | 53.9 | - |
| Visual (Unimodal) | - | 36.3 | 57.0 | 47.1 |
| Concat (Static) | Static | 49.3 | 68.0 | 62.8 |
| OGM | Static | 51.2 | 68.2 | 62.7 |
| PMR | Static | 50.1 | 68.2 | 63.0 |
| Vanilla MT | Dynamic | 48.8 | 67.0 | 62.7 |
| MBT | Dynamic | 51.5 | 72.2 | 63.0 |
| MMML | Dynamic | 52.0 | 69.8 | 62.8 |
| **Vanilla MT + RollingQ** | Dynamic | **51.9** | **69.3** | **63.2** |
| **MMML + RollingQ** | Dynamic | **52.7** | **70.7** | **63.2** |

RollingQ brings consistent improvements to Vanilla MT: CREMA-D +3.1%, Kinetic-Sound +2.3%, and MOSEI +0.5%.

### Noise Robustness Test (Table 3, Kinetic-Sound)

| Noise Level | Vanilla MT | Vanilla MT + RollingQ |
|----------|-----------|----------------------|
| 0.00 | 67.0 | 69.3 |
| 0.25 | 62.7 (-4.3) | 67.2 (-1.9) |
| 0.50 | 52.9 (-14.1) | 58.2 (-11.1) |
| 1.00 | 34.7 (-32.3) | 40.6 (-28.7) |

RollingQ consistently exhibits superior performance and experiences less degradation across all noise levels.

### Computational Efficiency (Table 4, CREMA-D)

| Method | Acc | Params | GFLOPs |
|------|-----|--------|--------|
| MBT | 51.5 | 114.21M | 2746.90 |
| MMML | 52.0 | 77.88M | 1828.29 |
| JMT | 50.7 | 62.23M | 1494.87 |
| **Vanilla MT + RollingQ** | **51.9** | **60.46M** | **1489.20** |

RollingQ achieves highly competitive performance comparable to complex architectures with minimal parameters and computational overhead.

### Pearson Correlation Analysis (Table 2)

| Method | CREMA-D coef | KS coef |
|------|-------------|---------|
| Vanilla MT | 0.52 | 0.44 |
| Vanilla MT + RollingQ | **0.76** | **0.78** |

RollingQ significantly increases the correlation between attention scores and input quality (p < 0.01), proving that dynamic adaptability is restored.

## Highlights & Insights

1. **Deep Problem Exploration**: This work uncovers the "dynamically named but virtually static" attention mechanism in multimodal Transformers, clearly demonstrating the self-reinforcing loop through theoretical analysis and noise experiments.
2. **Minimalist yet Highly Effective**: Requiring only a single rotation matrix and a ~1% parameter increase, RollingQ consistently improves performance across multiple datasets.
3. **Comprehensive Evaluation**: The efficacy of RollingQ in restoring dynamic collaboration is validated through key distribution visualization, gradient analysis, Pearson correlation, noise robustness, and OOD detection.
4. **Strong Versatility**: It integrates effectively across multiple architectures (such as Vanilla MT, MulT, and MMML) and is also applicable to ResNet backbones.

## Limitations & Future Work

1. **Theoretical analysis is limited to single-layer attention**: Although a progressive training approach for multiple layers is provided, indepth modeling of complex dynamic interactions in multi-layer Transformers remains insufficient.
2. **No direct enhancement of unimodal encoders**: RollingQ only adjusts the attention query allocation strategy without directly optimizing unimodal features like OGM/PMR. Combining the two methodologies could yield further gains.
3. **Small dataset scales**: The three primary datasets (CREMA-D, Kinetic-Sound, CMU-MOSEI) are relatively small-scale; the concept lacks validation in large-scale multimodal pretraining scenarios (e.g., VLMs).
4. **Manual setting of the rotation frequency**: The maximum number of rotations varies across datasets (1 or 3); adaptive selection strategies require further exploration.
5. **Only validated in bimodal scenarios**: Scalability to three or more modalities remains to be extensively discussed.

## Related Work & Insights

- **Imbalanced Multimodal Learning**: Methods such as OGM (Gradient Modulation) and PMR (Prototype Modality Rebalancing) primarily target static fusion. This study is the first to systematically analyze the imbalance issue in the context of dynamic fusion.
- **Multimodal Transformers**: Work such as MBT (Attention Bottlenecks) and MulT (Cross-modal Attention) design architectures to improve fusion processing, but do not address the inherent failure of the attention mechanism itself.
- **Greedy Multimodal Learning**: Wu et al. (2022) identified greediness in multimodal learning, which this study extends to analyze the dynamic collapse of attention mechanisms.

**Insights**: This approach can be extended to detect attention bias toward specific modalities (e.g., vision vs. language) in large-scale VLMs (such as LLaVA or Qwen-VL), and can also inspire expert-balancing strategies in Mixture-of-Experts (MoE) architectures.

## Rating
- Novelty: ⭐⭐⭐⭐ — The identified problem is novel and critical, particularly the observation of "attention dynamics collapse."
- Experimental Thoroughness: ⭐⭐⭐⭐ — Evaluated across multiple datasets, architectures, and dimensions, including noise, OOD, and ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Clear theoretical analysis, intuitive diagrams, and highly logical arguments.
- Value: ⭐⭐⭐⭐ — Offers valuable insights into understanding multimodal Transformers, though limited by dataset scales.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] In-Context Compositional Learning via Sparse Coding Transformer](../../NeurIPS2025/multimodal_vlm/in-context_compositional_learning_via_sparse_coding_transformer.md)
- [\[ICML 2026\] Dimension-Free Multimodal Sampling via Preconditioned Annealed Langevin Dynamics](../../ICML2026/multimodal_vlm/dimension-free_multimodal_sampling_via_preconditioned_annealed_langevin_dynamics.md)
- [\[ACL 2026\] Towards Visually Grounded Multimodal Summarization via Cross-Modal Transformer and Gated Attention](../../ACL2026/multimodal_vlm/towards_visually_grounded_multimodal_summarization_via_cross-modal_transformer_a.md)
- [\[ICCV 2025\] Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy](../../ICCV2025/multimodal_vlm/dita_scaling_diffusion_transformer_for_generalist_visionlang.md)
- [\[CVPR 2026\] Dynamics-Aware Preference Optimization for Vision-Language Models](../../CVPR2026/multimodal_vlm/dynamics-aware_preference_optimization_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
