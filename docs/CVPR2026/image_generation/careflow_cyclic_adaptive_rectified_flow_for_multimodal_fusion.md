---
title: >-
  [Paper Note] CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion
description: >-
  [CVPR 2026][Image Generation][rectified flow] Proposes CaReFlow, the first work to utilize rectified flow for multimodal distribution mapping to bridge the modality gap: it enables source modality data points to observe the global distribution of the target modality through one-to-many mapping, applies different alignment intensities to modality pairs with varying correlation via adaptive relaxed alignment, and ensures no information loss after mapping through cyclic rectified flow. It achieves SOTA on multiple multimodal affective computing benchmarks even with simple concatenation fusion.
tags:
  - CVPR 2026
  - Image Generation
  - rectified flow
  - modality gap
  - multimodal fusion
  - affective computing
  - distribution mapping
date: 2026-05-08
content_hash: b0437e76abc4317c
---

# CaReFlow: Cyclic Adaptive Rectified Flow for Multimodal Fusion

**Conference**: CVPR 2026  
**arXiv**: [2602.19140](https://arxiv.org/abs/2602.19140)  
**Code**: TBD  
**Area**: Image Generation  
**Keywords**: rectified flow, modality gap, multimodal fusion, affective computing, distribution mapping

## TL;DR
Proposes CaReFlow, the first work to utilize rectified flow for multimodal distribution mapping to bridge the modality gap: it enables source modality data points to observe the global distribution of the target modality through one-to-many mapping, applies different alignment intensities to modality pairs with varying correlation via adaptive relaxed alignment, and ensures no information loss after mapping through cyclic rectified flow. It achieves SOTA on multiple multimodal affective computing benchmarks even with simple concatenation fusion.

## Background & Motivation
**Background**: Multimodal Affective Computing (MAC) requires fusing visual, acoustic, and language modalities to analyze human emotional states. The core challenge lies in the vast differences in feature distributions across modalities (the modality gap), which often results in simple multimodal concatenation performing worse than unimodal language models.

**Limitations of Prior Work**: Existing methods for bridging the modality gap (e.g., contrastive learning, GANs, diffusion models) mostly employ one-to-one alignment—pushing each source modality data point towards a single fixed target point. This presents two issues: (a) too few paired data points per sample lead to insufficient alignment; (b) source modality data points cannot see the overall distribution of the target modality, leading to less robust alignment.

**Key Challenge**: One-to-one mapping limits the source modality's perception of the target modality's global distribution. If one-to-many mapping (like vanilla rectified flow) is used directly, it creates ambiguous flow directions (each source point matching multiple target points) and requires multiple recursive training rounds to learn straight trajectories.

**Goal**: (a) How to let the source modality perceive the global distribution of the target modality? (b) How to avoid ambiguity in one-to-many mapping? (c) How to prevent source modality information loss during the mapping process?

**Key Insight**: Rectified flow naturally maps one distribution to another, and the learned trajectories are straight (simulatable with few Euler steps). The authors observe that the training process of rectified flow is inherently one-to-many—randomly sampling distribution pairs for training—which happens to expose global distribution information.

**Core Idea**: Use rectified flow for modality distribution mapping, solve the ambiguity problem through adaptive relaxed alignment, and preserve source modality information via cyclic flows.

## Method

### Overall Architecture
Input: Tri-modal feature sequences $\mathbf{U}_m \in \mathbb{R}^{T_m \times d_m}$ ($m \in \{a, v, l\}$, representing acoustic, visual, and language) are passed through unimodal networks to extract representations $\mathbf{X}_m \in \mathbb{R}^d$. Since language is the dominant modality in MAC, CaReFlow maps the distributions of visual and acoustic modalities to the language modality distribution: $\mathbf{X}_{m,l} = \text{CaReFlow}_{m,l}(\mathbf{X}_m)$, followed by simple concatenation + MLP for fusion and prediction. During inference, distribution conversion is completed in 2 Euler steps.

### Key Designs

1. **One-to-Many Mapping**:

    - **Function**: Enables each source modality data point to observe multiple target modality data points during training, rather than just the one within the same sample.
    - **Mechanism**: During rectified flow training, data pairs are randomly sampled from two distributions to train the velocity field $\mathbf{V}_{m_1,m_2}$. In implementation, intra-sample modality pairs are constructed first within each mini-batch, followed by randomly sampled inter-sample modality pairs (at a ratio of $\beta$ times). This allows each source modality data point to perceive the global distribution of the target modality.
    - **Design Motivation**: Alleviates the issue of insufficient paired data per sample, making distribution mapping more robust. Experiments show a performance drop of about 3 percentage points (Acc2) without this module, making it the most impactful component.

2. **Adaptive Relaxed Alignment**:

    - **Function**: Imposes different degrees of alignment strictness for modality pairs with different correlations.
    - **Mechanism**: Modifies the rectified flow objective function by introducing a margin $\eta_{m_1,m_2}$:
    $\mathcal{L}^f_{m_1,m_2} = \mathbb{E}\left[\max\left(\|\mathbf{V}_{m_1,m_2}(\mathbf{X}^t_{m_1,m_2}, t) - (\mathbf{X}_{m_2} - \mathbf{X}_{m_1})\|_2 - \eta_{m_1,m_2}, 0\right)\right]$
      where $\eta_{m_1,m_2} = 0$ (same sample) or $\eta_{m_1,m_2} = \epsilon + \|y_i - y_j\|_2$ (different samples, determined adaptively by label distance).
    - **Design Motivation**: Modalities of the same sample should be strictly aligned ($\eta=0$ degrades to standard rectified flow), while relaxation is applied to different samples to allow for variance. Less relaxation is applied as labels become more similar. This simultaneously solves the ambiguity in one-to-many mapping and eliminates the need for recursive training of rectified flow.

3. **Cyclic Rectified Flow**:

    - **Function**: Constructs a backward rectified flow to ensure features $\mathbf{X}_{m_1,m_2}$ from the forward mapping can be mapped back to the original features $\mathbf{X}_{m_1}$.
    - **Mechanism**: Trains a backward velocity field $\hat{\mathbf{V}}_{m_1,m_2}$ to restore the forward output:
    $\mathcal{L}^b_{m_1,m_2} = \mathbb{E}\left[\|\hat{\mathbf{V}}_{m_1,m_2}(\hat{\mathbf{X}}^t_{m_1,m_2}, t) - (\mathbf{X}_{m_1} - \mathbf{X}_{m_1,m_2})\|_2\right]$
      Key detail: The backward loss applies `detach` to $\mathbf{X}_{m_1}$ but not to $\mathbf{X}_{m_1,m_2}$, allowing the backward loss to backpropagate and influence the forward rectified flow.
    - **Design Motivation**: Prevents the loss of discriminative information from the source modality during distribution mapping, ensuring that the fused multimodal representation retains sufficient modality-specific information.

4. **Implementation of Velocity Field Network $\mathbf{V}_{m_1,m_2}$**:

    - Uses sine/cosine positional encoding to inject time information $\mathbf{TE}^t$, then concatenates input features and time encoding before feeding them into an MLP.
    - Applies `detach` to input features so that forward loss $\mathcal{L}^f$ only trains the velocity field without updating the unimodal networks, achieving decoupling between the distribution alignment task and the main task.

### Loss & Training
$$\mathcal{L}_{total} = \mathcal{L} + \sum_{m \in \{a,v\}} (\alpha_f \times \mathcal{L}^f_{m,l} + \alpha_b \times \mathcal{L}^b_{m,l})$$
where $\mathcal{L}$ is the main task prediction loss, and $\alpha_f$ and $\alpha_b$ are weights for forward and backward losses, respectively. $\alpha_f$ requires a larger value (to ensure sufficient distribution mapping), while $\alpha_b$ requires a moderate value (too large will hinder distribution conversion).

## Key Experimental Results

### Main Results: MSA (CMU-MOSI & CMU-MOSEI)

| Method | Conference | MOSI Acc7 | MOSI Acc2 | MOSI MAE↓ | MOSEI Acc2 | MOSEI MAE↓ |
|------|------|-----------|-----------|-----------|------------|------------|
| ITHP | ICLR 2024 | 47.7 | 88.5 | 0.663 | 87.1 | 0.550 |
| DLF | AAAI 2025 | 49.4 | 88.7 | 0.669 | 87.5 | 0.515 |
| AtCAF | IF 2025 | 46.5 | 88.6 | 0.650 | 87.0 | 0.508 |
| **Ours** | - | **50.6** | **89.8** | **0.616** | **87.9** | **0.504** |

On the CH-SIMS-v2 dataset, Acc5 reaches 57.9 (Prev. SOTA KuDA was 53.1, Gain +4.8), and Acc2 reaches 82.9 (Prev. SOTA AV-MC was 80.6, Gain +2.3).

### Ablation Study

| Configuration | CH-SIMS-v2 Acc5 | CH-SIMS-v2 Acc2 | MOSI Acc7 | MOSI Acc2 |
|------|-----------------|-----------------|-----------|-----------|
| W/O Distribution Alignment | 55.1 | 78.4 | 45.9 | 86.7 |
| W/O Cyclic Information Flow | 56.3 | 81.7 | 47.2 | 87.9 |
| W/O Adaptive Relaxed Alignment | 56.8 | 82.4 | 47.9 | 88.5 |
| W/O One-to-Many Mapping | 57.4 | 79.8 | 47.2 | 87.0 |
| **Full CaReFlow** | **57.9** | **82.9** | **50.6** | **89.8** |

### Comparison with Other Distribution Mapping Methods

| Method | Type | MOSI Acc7 | MOSI Acc2 | Parameters | CH-SIMS-v2 Acc5 |
|------|------|-----------|-----------|--------|-----------------|
| ARGF | GAN | 50.5 | 87.4 | 184.43M | 51.8 |
| MulT | Transformer | 40.7 | 88.1 | 185.51M | 54.6 |
| CLGSI | Contrastive | 45.8 | 89.0 | 186.31M | 52.5 |
| Diffusion Bridge | Diffusion | 47.3 | 86.9 | 185.46M | 52.5 |
| **Ours** | Rectified Flow | **50.6** | **89.8** | **185.38M** | **57.9** |

### Key Findings
- **One-to-Many contributes the most**: Removing it drops MOSI Acc2 by 2.8% and CH-SIMS-v2 Acc2 by 3.1%, making it the most critical module.
- **Robust Hyperparameters**: Performance is stable for cross-modal ratio $\beta$ in the 3-7 range; changes in Euler steps from 2-5 are minimal (indicating trajectories are already very straight).
- CaReFlow has a moderate parameter count (185.38M), fewer than CLGSI (186.31M) and MulT (185.51M); performance gains come from alignment effectiveness rather than increased parameters.
- t-SNE visualization intuitively shows CaReFlow bridges the modality gap more effectively than ARGF/CLGSI/Diffusion Bridge/DLF.
- Replacing with advanced fusion methods (e.g., tensor fusion) can further improve results, such as CH-SIMS-v2 Acc2 increasing from 82.9 to 83.6.

## Highlights & Insights
- **A New Perspective on Rectified Flow for Modality Alignment**: Redefines the modality gap problem as a distribution mapping task, leveraging the geometric intuition of rectified flow (straight trajectories mapping two distributions) to bridge the gap. This is simpler and faster than GANs and diffusion models.
- **Clever Adaptive Margin Design**: Uses label distance $\|y_i - y_j\|$ to adaptively control the degree of relaxation, retaining the global field-of-view advantage of one-to-many mapping while guiding the model to distinguish between "should-be-aligned" and "relaxable" modality pairs. This solves both the ambiguity and recursive training issues in one go.
- **Task Decoupling via Detach Operations**: Forward loss applies `detach` to inputs (not updating unimodal networks), while backward loss applies `detach` to $\mathbf{X}_{m_1}$ but not $\mathbf{X}_{m_1,m_2}$—this precise gradient control allows for full individual optimization of distribution alignment and the main task.
- **Fusion-Method Agnostic**: As a preprocessing module, CaReFlow is independent of the fusion mechanism and can be plugged into any multimodal system.

## Limitations & Future Work
- Currently only validated on multimodal affective computing tasks, and has not been tested on broader multimodal tasks like vision-language understanding.
- Uses language as a fixed target modality (since it is dominant in MAC); it remains unclear how to select target modalities for tasks where language is not dominant.
- The velocity field is implemented with a simple MLP, which may lack expressive power in more complex distribution mapping scenarios.
- Hyperparameters ($\beta$, $\epsilon$, $\alpha_f$, $\alpha_b$) require tuning; although the authors demonstrate robustness, optimal values vary by dataset.

## Related Work & Insights
- **vs Diffusion Bridge**: Both use generative models for distribution mapping, but diffusion models have slow inference and only perform one-to-one mapping; CaReFlow uses rectified flow which completes in 2 steps and supports one-to-many.
- **vs CLGSI (Contrastive Learning)**: CLGSI also implements one-to-many (same-category positive pairs) but does not differentiate the importance between same-sample pairs and same-category pairs; CaReFlow achieves finer control through adaptive margins.
- **vs ARGF (GAN)**: GAN training is unstable and difficult for guaranteeing information preservation; CaReFlow explicitly constrains information preservation through cyclic flow.

## Rating
- Novelty: ⭐⭐⭐⭐ Utilizing rectified flow for modality alignment for the first time, with three intertwined innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ 5 datasets across 3 tasks + sufficient ablation and visualization.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation derivation and well-formalized methodology.
- Value: ⭐⭐⭐⭐ The modality gap is a core issue in multimodal fusion; the rectified flow solution is concise and effective.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Efficient Rectified Flow for Image Fusion](../../NeurIPS2025/image_generation/efficient_rectified_flow_for_image_fusion.md)
- [\[CVPR 2026\] WISER: Wider Search, Deeper Thinking, and Adaptive Fusion for Training-Free Zero-Shot Composed Image Retrieval](wiser_wider_search_deeper_thinking_and_adaptive_fusion_for_training-free_zero-sh.md)
- [\[ICLR 2026\] Free Lunch for Stabilizing Rectified Flow Inversion](../../ICLR2026/image_generation/free_lunch_for_stabilizing_rectified_flow_inversion.md)
- [\[NeurIPS 2025\] Balanced Conic Rectified Flow](../../NeurIPS2025/image_generation/balanced_conic_rectified_flow.md)
- [\[NeurIPS 2025\] Rectified-CFG++ for Flow Based Models](../../NeurIPS2025/image_generation/rectified-cfg_for_flow_based_models.md)

<!-- RELATED:END -->
