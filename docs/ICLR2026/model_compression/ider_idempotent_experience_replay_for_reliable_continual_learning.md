---
title: >-
  [Paper Note] IDER: IDempotent Experience Replay for Reliable Continual Learning
description: >-
  [ICLR 2026][Model Compression][Continual Learning] This paper introduces idempotence into continual learning, enforcing output self-consistency during new task acquisition via two components—a Standard Idempotent Module…
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Continual Learning"
  - "Idempotency"
  - "Experience Replay"
  - "Calibration Error"
  - "Catastrophic Forgetting"
date: 2026-05-08
content_hash: 964833709987968a
---

# IDER: IDempotent Experience Replay for Reliable Continual Learning

**Conference**: ICLR 2026
**arXiv**: [2603.00624](https://arxiv.org/abs/2603.00624)  
**Code**: [GitHub](https://github.com/YutingLi0606/Idempotent-Continual-Learning)  
**Area**: Model Compression
**Keywords**: Continual Learning, Idempotency, Experience Replay, Calibration Error, Catastrophic Forgetting

## TL;DR
This paper introduces idempotence into continual learning, enforcing output self-consistency during new task acquisition via two components—a Standard Idempotent Module and an Idempotent Distillation Module—simultaneously improving prediction reliability (reduced calibration error) and significantly mitigating catastrophic forgetting.

## Background & Motivation
Continual learning faces the core challenge of catastrophic forgetting—models rapidly lose knowledge of previous tasks when learning new ones. Replay-based methods (e.g., ER, DER++) mitigate this by storing old samples in a buffer, yet tend to be overconfident and poorly calibrated (high ECE), with particular bias toward recent tasks. In safety-critical domains such as healthcare and autonomous driving, models must not only be accurate but also "know what they do not know."

Existing uncertainty-aware CL methods (e.g., NPCL) employ neural processes but suffer from: (1) non-negligible parameter growth; and (2) incompatibility with logits-based replay methods due to the stochasticity of Monte Carlo sampling. There is thus a need for a **lightweight and compatible** principle for building reliable CL systems.

Core Idea: Idempotence—a function that yields the same result when applied repeatedly ($f(f(x)) = f(x)$). If a model remains idempotent on old data, its outputs lie on a learned stable manifold, indicating self-consistent and reliable predictions.

## Method

### Overall Architecture
IDER augments the experience replay framework with two modules: (1) a Standard Idempotent Module that enforces idempotence on current-task data; and (2) an Idempotent Distillation Module that imposes cross-task idempotent constraints using an old model checkpoint. Only two forward passes are required, introducing virtually no additional parameters.

### Key Designs

1. **Modified Network Architecture**:

    - Function: Enables the model to accept two inputs—an image and an auxiliary label signal.
    - Mechanism: The ResNet backbone is split into $f_t^1$ (front half) and $f_t^2$ (back half). A second input (one-hot label $y$ or a uniform "null" signal $\mathbf{0}$) is transformed via a linear layer followed by LeakyReLU into a feature of the same dimensionality as the output of $f_t^1$, added to the intermediate feature map, and then passed into $f_t^2$. The model output (softmax over logits) can be recycled as the second input.
    - Design Motivation: At inference time, when no ground-truth label is available, $\mathbf{0}$ is used as the second input to obtain a prediction, which is then fed back to verify idempotence.

2. **Standard Idempotent Module**:

    - Function: Trains the model to be idempotent on current-task data.
    - Mechanism: Minimizes the cross-entropy of two forward passes against the ground truth:
    $\mathcal{L}_{ice} = \sum_{(x,y)\in\mathcal{T}_t} [\mathcal{L}_{ce}(f_t(x,y^*),y) + \mathcal{L}_{ce}(f_t(x,f_t(x,y^*)),y)]$
      where $y^*$ is chosen as the ground-truth label with probability $1-P$ and as the null signal $\mathbf{0}$ with probability $P$.
    - Design Motivation: After training, $f_t(x,\mathbf{0}) \approx y$ and $f_t(x,f_t(x,\mathbf{0})) \approx y$, achieving self-consistent predictions.

3. **Idempotent Distillation Module**:

    - Function: Leverages old model checkpoints to constrain the current model and prevent prediction distribution drift.
    - Mechanism: Rather than naively minimizing $\|f_t(x,\mathbf{0}) - f_t(x,f_t(x,\mathbf{0}))\|^2$ (which may reinforce erroneous predictions due to the bias of $f_t$ toward current data), the frozen old checkpoint $f_{t-1}$ is used for the second forward pass:
    $\mathcal{L}_{ide} = \sum_{(x,y)\in\mathcal{T}_t,M} \|f_t(x,\mathbf{0}) - f_{t-1}(x,f_t(x,\mathbf{0}))\|_2^2$
    - Design Motivation: Freezing $f_{t-1}$ preserves prior knowledge and a stable distribution; only $f_t$ is updated, guiding $y_0$ in the correct direction and avoiding error reinforcement. This simultaneously serves as knowledge distillation.

### Loss & Training
The total loss is a weighted sum of three terms: $\mathcal{L}_{IDER} = \mathcal{L}_{ice} + \alpha\mathcal{L}_{ide} + \beta\mathcal{L}_{rep\text{-}ice}$, where the replay loss $\mathcal{L}_{rep\text{-}ice}$ also applies idempotent training on buffer data. IDER can be integrated as a plug-and-play component into methods such as ER, BFP, and CLS-ER.

## Key Experimental Results

### Main Results (Final Average Accuracy)

| Method | CIFAR-10 Buf200 | CIFAR-10 Buf500 | CIFAR-100 Buf500 | CIFAR-100 Buf2000 |
|--------|----------------|----------------|-----------------|------------------|
| ER | 44.46 | 58.84 | 23.41 | 40.47 |
| DER++ | 62.19 | 70.10 | 37.69 | 51.82 |
| XDER | 64.10 | 67.42 | 48.14 | 57.57 |
| BFP | 68.64 | 73.51 | 46.70 | 57.39 |
| **ER+ID (Ours)** | **71.02** | **74.74** | 44.82 | 56.59 |
| **BFP+ID (Ours)** | **71.99** | **76.65** | **48.53** | **57.74** |

### Ablation Study (GCIL-CIFAR-100 Uniform)

| Method | Buf200 | Gain | Buf500 | Gain |
|--------|--------|------|--------|------|
| ER | 16.34 | — | 28.76 | — |
| ER+ID | 26.66 | +10.32 | 40.54 | +11.78 |
| CLS-ER | 22.37 | — | 36.80 | — |
| CLS-ER+ID | 31.17 | +8.80 | 37.57 | +0.77 |

### Key Findings
- ER+ID achieves a gain of up to 26% on CIFAR-10 Buf200 (44.46→71.02), the largest improvement among all methods evaluated.
- Idempotent distillation effectively alleviates recency bias, yielding more uniform prediction probabilities across tasks.
- ECE (calibration error) is consistently reduced across all baselines, indicating more "honest" model predictions.
- Improvements are even more pronounced in the more challenging GCIL setting (e.g., CLS-ER+ID gains 8.4% under the Longtail configuration).

## Highlights & Insights
- The paper elegantly maps the algebraic notion of idempotence directly to a "prediction self-consistency" constraint in CL, offering a mathematically clear and principled intuition.
- The method is extremely lightweight—requiring only one additional forward pass and the previous checkpoint, with virtually no parameter overhead.
- Its plug-and-play nature makes it a general-purpose tool for enhancing the reliability of existing CL methods.

## Limitations & Future Work
- Storing the previous checkpoint $f_{t-1}$ is required, which, while modest in storage cost, increases system complexity.
- All experiments are conducted on ResNet-18; the effectiveness on Transformer backbones and larger models remains unexplored.
- The theoretical foundation of the idempotency assumption warrants further strengthening—a formal analysis of why idempotence necessarily leads to better generalization is absent.

## Related Work & Insights
- **vs. NPCL**: NPCL incurs parameter growth through neural processes and is incompatible with logits-based methods, whereas IDER is lightweight and general-purpose.
- **vs. DER++**: DER++ simply stores and matches logits, while IDER provides stronger regularization through idempotent constraints.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Applying idempotence to CL represents a genuinely novel perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers CIL, GCIL, ECE, and plug-and-play validation comprehensively.
- Writing Quality: ⭐⭐⭐⭐ — Logic is clear; the derivation from motivation to method flows naturally.
- Value: ⭐⭐⭐⭐ — Provides a new mathematical principle and practical tool for the CL community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Elastic Weight Consolidation Done Right for Continual Learning](../../CVPR2026/model_compression/elastic_weight_consolidation_done_right_for_continual_learning.md)
- [\[ICLR 2026\] Rethinking Continual Learning with Progressive Neural Collapse](rethinking_continual_learning_with_progressive_neural_collapse.md)
- [\[AAAI 2026\] Beyond Sharpness: A Flatness Decomposition Framework for Efficient Continual Learning](../../AAAI2026/model_compression/beyond_sharpness_a_flatness_decomposition_framework_for_efficient_continual_lear.md)
- [\[ICLR 2026\] Revisiting Weight Regularization for Low-Rank Continual Learning](revisiting_weight_regularization_for_low-rank_continual_learning.md)
- [\[ICLR 2026\] FlyPrompt: Brain-Inspired Random-Expanded Routing with Temporal-Ensemble Experts for General Continual Learning](flyprompt_brain-inspired_random-expanded_routing.md)

</div>

<!-- RELATED:END -->
