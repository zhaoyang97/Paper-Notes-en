---
title: >-
  [Paper Note] Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning
description: >-
  [NeurIPS 2025][Model Compression][Class-incremental learning] This paper proposes learning beneficial "mixture of noise" to suppress parameter drift in pre-trained models during class-incremental learning. By dynamically mixing task-specific noise with learned weights across tasks, the method achieves state-of-the-art performance, particularly in the challenging 50-step incremental setting.
tags:
  - NeurIPS 2025
  - Model Compression
  - Class-incremental learning
  - pre-trained models
  - parameter drift
  - positive noise
  - catastrophic forgetting suppression
date: 2026-05-08
content_hash: 6f03ee8a5854d712
---

# Mixture of Noise for Pre-Trained Model-Based Class-Incremental Learning

**Conference**: NeurIPS 2025
**arXiv**: [2509.16738](https://arxiv.org/abs/2509.16738)
**Code**: [https://github.com/ASCIIJK/MiN-NeurIPS2025](https://github.com/ASCIIJK/MiN-NeurIPS2025)
**Area**: Model Compression / Incremental Learning
**Keywords**: Class-incremental learning, pre-trained models, parameter drift, positive noise, catastrophic forgetting suppression

## TL;DR

This paper proposes learning beneficial "mixture of noise" to suppress parameter drift in pre-trained models during class-incremental learning. By dynamically mixing task-specific noise with learned weights across tasks, the method achieves state-of-the-art performance, particularly in the challenging 50-step incremental setting.

## Background & Motivation

**State of the Field**: Pre-trained models (PTMs) exhibit strong performance when fine-tuned on downstream tasks; however, continual fine-tuning induces **parameter drift**, which corrupts discriminative features of previous tasks and allows new-task features to interfere with existing decision boundaries.

**Limitations of Prior Work**: Conventional class-incremental learning methods focus on improving feature utilization efficiency (e.g., prompt learning, prototype networks), while overlooking inter-task feature interference. Parameter drift has been uniformly treated as a purely negative phenomenon.

**Core Idea**: Noise is not inherently harmful — positive-incentive noise (Pi-Noise) can improve classification by masking inter-class confusion and highlighting discriminative features. Parameter drift constitutes "destructive noise," yet "beneficial noise" can be learned to counteract it.

**Paper Goals**: Rather than pursuing efficient feature utilization, this work actively learns beneficial noise to suppress inter-task confusion patterns.

**Starting Point**: From an information-theoretic perspective, noise is modeled as a latent variable, and efficient inference is achieved via the reparameterization trick and dynamic mixing.

**Core Idea**: Through noise expansion (learning task-specific noise modules) combined with noise mixture (dynamic weight-based fusion), parameter drift is transformed from "catastrophic forgetting" into a "controllable positive signal."

## Method

### Overall Architecture

The MiN (Mixture of Noise) framework is realized through two core strategies:
- **Noise Expansion**: Learning dedicated noise generation modules for each new task.
- **Noise Mixture**: Dynamically learning weights to mix noise from different tasks, enabling single-pass inference.

### Key Designs

1. **Noise Expansion Strategy (Section 4.1)**

    - **Function**: Inserts π-noise layers into intermediate layers of the pre-trained backbone.
    - **Mechanism**: Given an intermediate feature $r_l$, the noise generation process is defined as $\varepsilon_t = \varepsilon \cdot \phi_t^{\sigma}(r_l W_{down}) + \phi_t^{\mu}(r_l W_{down})$, where $\varepsilon$ is sampled from a standard normal distribution, $\phi_t^{\sigma}$ and $\phi_t^{\mu}$ are two-layer MLPs generating variance and mean vectors respectively, and $W_{down}$ projects high-dimensional features into a low-dimensional space $d_2 \ll d_1$.
    - **Design Motivation**: The number of trainable parameters is minimal (only two $d_2 \times d_2$ matrices), keeping the model lightweight.

2. **Noise Mixture Strategy (Section 4.2)**

    - **Function**: For task $t$, an accumulated set of noise modules $\{\varepsilon_1, \ldots, \varepsilon_t\}$ is maintained, and dynamic weights are learned for optimal mixing.
    - **Mechanism**: Weights are initialized based on task prototype similarity $s_{t,i} = \frac{\mu_t \cdot \mu_i}{\|\mu_t\| \|\mu_i\|}$, and the normalized weighted mixture is computed as $\varphi(\{\varepsilon_1, \ldots, \varepsilon_t\}) = \sum_{i=1}^{t}\varepsilon_i \omega_i$.
    - **Design Motivation**: Applying each noise module independently would result in inference complexity growing linearly with the number of tasks; mixing enables a single forward pass.

3. **Three-Step Training Pipeline (Algorithm 1)**

    - **Function**: Iteratively updates through three steps: analytic learning → noise learning → classifier update.
    - **Mechanism**: Step 1 updates the classifier $W_t$ via analytic learning (gradient-free); Step 2 trains task-specific noise using an auxiliary classifier; Step 3 updates the classifier again.
    - **Design Motivation**: The auxiliary classifier $W_{aux}$ is initialized to zero to fit residuals, preventing the main classifier from being disturbed during noise training.

### Loss & Training

$$\mathcal{L}_{cls} = \ell(z_L W_{aux}, y - z_L W_t)$$

The auxiliary classifier learns residuals, and the noise ultimately acts as perturbations in the feature space.

## Key Experimental Results

### Main Results

| Method | CIFAR-100 10-step | CIFAR-100 20-step | CIFAR-100 50-step | CUB-200 10-step | CUB-200 20-step |
|--------|-------------------|-------------------|-------------------|-----------------|-----------------|
| L2P | 85.92 | 81.90 | 74.29 | 84.29 | 81.75 |
| DualPrompt | 89.65 | 85.57 | 73.66 | 84.39 | 83.79 |
| CODA-Prompt | 91.05 | 87.51 | 69.54 | 84.15 | 83.89 |
| SLCA | 92.67 | 93.32 | 90.76 | 86.83 | 83.38 |
| FeCAM | 93.23 | 91.86 | 90.92 | 92.73 | 92.89 |
| **MiN (Ours)** | **94.12** | **93.89** | **92.34** | **93.45** | **93.67** |

### Ablation Study

| Configuration | CIFAR-100 20-step | Notes |
|---------------|-------------------|-------|
| Noise Expansion only | 90.56 | Basic noise learning |
| + Initial weights | 91.23 | Similarity-based initialization |
| + Weight learning | 93.12 | Dynamic weight optimization |
| + Three-step pipeline | **93.89** | Full pipeline |

### Key Findings

- Significant improvement over the previous SOTA in the 50-step incremental setting (the most challenging scenario): 90.92 → 92.34, with the smallest standard deviation.
- Even in the 50-step setting, the total number of trainable parameters remains minimal (fewest per task among compared methods).
- Grad-CAM visualizations demonstrate that MiN substantially improves attention to key discriminative regions while suppressing responses to irrelevant areas.

## Highlights & Insights

- **Novel Perspective**: Parameter drift is reinterpreted as a controllable noise signal, extracting a positive mechanism from what was previously considered a purely negative phenomenon. This represents a fundamental conceptual shift — from "avoiding drift" to "exploiting drift."
- **Lightweight Design**: The noise generation modules introduce very few parameters, far fewer than prompt-based methods, and mixing noise at inference incurs no additional overhead.
- **Robustness in Extreme Settings**: The method maintains stable performance under 50-step incremental learning (2 new classes per step), substantially outperforming competing methods in this regime.

## Limitations & Future Work

- Experiments are conducted on CIFAR-100 (10K samples) and CUB-200 (12K samples); large-scale real-world validation is absent.
- Evaluation is limited to ViT-B/16-IN21K; the effectiveness on other backbones (e.g., Swin, ConvNeXt) remains unexplored.
- The temperature coefficient and initialization weight computation leave room for further tuning.
- Applicability to non-visual domains (NLP, multimodal incremental learning) has yet to be verified.

## Related Work & Insights

- **vs. L2P/DualPrompt**: These methods adapt to new tasks via prompt learning, but prompts do not share knowledge across tasks; MiN's noise mixture inherently enables cross-task sharing.
- **vs. FeCAM**: FeCAM relies on prototypes and feature space alignment, whereas MiN injects positive noise into the feature space — the two approaches are complementary.
- **Transferable Directions**: Noise augmentation in knowledge distillation and contrastive learning; noise feature modeling in cross-domain transfer learning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The parameter drift–beneficial noise framework represents a genuinely novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Six datasets with comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Logic is clear and methodological exposition is precise.
- Value: ⭐⭐⭐⭐⭐ Provides actionable guidance for effective adaptation of pre-trained models with high practical deployment value.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Integrating Task-Specific and Universal Adapters for Pre-Trained Model-based Class-Incremental Learning](../../ICCV2025/model_compression/integrating_task-specific_and_universal_adapters_for_pre-trained_model-based_cla.md)
- [\[AAAI 2026\] Compensating Distribution Drifts in Class-incremental Learning of Pre-trained Vision Transformers](../../AAAI2026/model_compression/compensating_distribution_drifts_in_class-incremental_learning_of_pre-trained_vi.md)
- [\[ICCV 2025\] Achieving More with Less: Additive Prompt Tuning for Rehearsal-Free Class-Incremental Learning](../../ICCV2025/model_compression/achieving_more_with_less_additive_prompt_tuning_for_rehearsal-free_class-increme.md)
- [\[NeurIPS 2025\] Toward Efficient Inference Attacks: Shadow Model Sharing via Mixture-of-Experts](toward_efficient_inference_attacks_shadow_model_sharing_via_mixture-of-experts.md)
- [\[NeurIPS 2025\] Online Mixture of Experts: No-Regret Learning for Optimal Collective Decision-Making](online_mixture_of_experts_no-regret_learning_for_optimal_collective_decision-mak.md)

<!-- RELATED:END -->
