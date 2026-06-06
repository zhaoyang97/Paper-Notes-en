---
title: >-
  [Paper Note] Adversarial Robust Memory-Based Continual Learner
description: >-
  [ICCV 2025][LLM Safety][Continual Learning] This paper identifies two compounding challenges when combining continual learning with adversarial training—accelerated forgetting and gradient confusion—and proposes two plug…
tags:
  - "ICCV 2025"
  - "LLM Safety"
  - "Continual Learning"
  - "Adversarial Robustness"
  - "Experience Replay"
  - "Logit Calibration"
  - "Gradient Confusion"
date: 2026-05-08
content_hash: d5427edf344cf78b
---

# Adversarial Robust Memory-Based Continual Learner

**Conference**: ICCV 2025
**arXiv**: [2311.17608](https://arxiv.org/abs/2311.17608)  
**Code**: N/A  
**Area**: Continual Learning / Adversarial Robustness
**Keywords**: Continual Learning, Adversarial Robustness, Experience Replay, Logit Calibration, Gradient Confusion

## TL;DR

This paper identifies two compounding challenges when combining continual learning with adversarial training—accelerated forgetting and gradient confusion—and proposes two plug-and-play modules, Anti-Forgettable Logit Calibration (AFLC) and Robustness-Aware Experience Replay (RAER), achieving up to 8.13% improvement in adversarial robustness on Split-CIFAR10/100 and Split-Tiny-ImageNet.

## Background & Motivation

Continual learning enables models to learn from non-i.i.d. data streams, yet existing methods remain highly vulnerable to adversarial examples. Directly combining continual learning with adversarial training introduces two core challenges:

**Challenge 1: Adversarial Training Accelerates Forgetting**
- Adversarial examples amplify negative gradients on past classes from current task data, exacerbating catastrophic forgetting.
- Adversarial samples $\tilde{x}_t$ produce higher logits on past classes than clean samples $x_t$: $h_\theta(\tilde{x_t})_p > h_\theta(x_t)_p$.
- This induces larger negative gradients in the direction of past classes, accelerating knowledge forgetting.

**Challenge 2: Continual Learning Aggravates Gradient Confusion**
- Limited replay data causes the model to overfit memorized samples.
- Gradient directions deviate significantly from those under joint training (verified via cosine similarity).
- This constitutes a form of "fragmented gradients" unique to the continual learning setting.

Prior work by Chen et al. attempted to mitigate this using large amounts of unlabeled data; this paper aims to address the problem **without any additional data**.

## Method

### Overall Architecture

Two modules are introduced on top of memory-based continual learning frameworks (e.g., ER) augmented with adversarial training:
1. **AFLC (Anti-Forgettable Logit Calibration)**: Applies task-order-aware calibration to logits before the softmax layer.
2. **RAER (Robustness-Aware Experience Replay)**: Selects replay data based on an adversarial robustness difficulty factor.

Both modules are plug-and-play and compatible with continual learning algorithms (ER, DER, DER++, X-DER) as well as adversarial training methods (Vanilla AT, TRADES).

### Key Designs

**AFLC (Anti-Forgettable Logit Calibration)**:
- Partitions classifier logits $h_\theta(x)$ into three groups: past (p), current (c), and future (u) classes.
- Applies a calibration value $v_i$ to each class $i$: $h_\theta^{lc}(\tilde{x})_i = h_\theta(\tilde{x})_i - v_i$.
- Key constraint: $v_p > v_c$, i.e., larger calibration for past classes.
- Effect: reduces negative gradients from current adversarial samples toward past classes, and increases negative gradients from past adversarial samples toward current classes.
- Formulation: $v_i = -\log(\frac{n_i}{\sum n_j}) - \alpha_i$, where $n_i$ is the number of samples for class $i$ in memory plus current data.
- **Future-class Prior Adjustment (FP)**: Sets future-class calibration values to the mean of existing classes, preventing implicit amplification of future-class logits.
- No logit calibration is applied at inference time.

**RAER (Robustness-Aware Experience Replay)**:
- Defines a robustness difficulty factor $k$: the number of PGD-10 steps required to successfully attack a sample.
- Larger $k$ indicates a more vulnerable sample, closer to the decision boundary.
- Sets a threshold $\rho$; only samples with $k < \rho$ are stored in memory.
- Filters out samples that overfit the current decision boundary, prioritizing robust and representative data.
- This is the **opposite** of conventional continual learning data selection strategies (which favor boundary-proximal samples).

### Loss & Training

Training objective using ER+AT as an example:

$$\mathcal{L}_t = \text{CE}(f_\theta^{lc}(\tilde{x_t}), y_t) + \text{CE}(f_\theta^{lc}(\tilde{x}_\mathcal{M}), y_\mathcal{M})$$

Adversarial examples are generated via PGD, logits are calibrated by AFLC, and replay data are selected by RAER.

Training procedure:
1. Sample a batch from the current task and a batch from memory.
2. Generate adversarial examples via PGD and record robustness factor $K$.
3. Calibrate logits with AFLC, compute the loss, and update parameters.
4. Use RAER to filter current-task samples based on $K$ before storing in memory.

Hyperparameter settings: $\alpha = 3.5$, $\rho = 5$ (threshold on successful attack steps within PGD-10).

## Key Experimental Results

### Main Results

Split-CIFAR10, Buffer=200, ER as the base framework:

| Method | w/ AT | Class-IL Clean FAA↑ | Class-IL Adv FAA↑ | Task-IL Clean FAA↑ | Task-IL Adv FAA↑ |
|--------|:-----:|---------------------|-------------------|--------------------|--------------------|
| ER | ✗ | 48.80 | 0.27 | 92.89 | 0.01 |
| ER+AT | ✓ | 28.18 | 17.86 | 84.49 | 44.30 |
| ER+AT+Ours | ✓ | **Improved** | **+8.13%** | **Improved** | **Improved** |
| Joint (upper bound) | ✓ | 79.33 | 50.93 | 94.80 | 74.63 |

Key observation: directly incorporating AT drops clean FAA from 48.80 to 28.18 (−20.62), while adversarial FAA increases from 0.27 to 17.86.

### Ablation Study

Contribution of each module (ER+AT, Buffer=200, Split-CIFAR10):

| Configuration | CRD↓ | FRI↓ | RRD↓ |
|---------------|------|------|------|
| ER+AT (baseline) | 14.51 | 12.91 | 31.70 |
| +AFLC | Reduced | Reduced | Reduced |
| +RAER | Reduced | Reduced | Reduced |
| +AFLC+RAER | **Lowest** | **Lowest** | **Lowest** |

- **CRD** (Clean Relative Decrease): drop in clean accuracy after adding AT.
- **FRI** (Forgetting Relative Increase): increase in forgetting on clean data.
- **RRD** (Robust Relative Decrease vs. Joint): robustness gap relative to the joint training model.

Comparison across four continual learning methods (ER, DER, DER++, X-DER) combined with AT:
- ER+AT performs best under the Class-IL setting (especially at buffer=200).
- X-DER excessively suppresses the model's ability to learn new tasks.
- Larger buffer (5120) substantially improves all metrics.

### Key Findings

1. **Adversarial training is incompatible with direct integration into continual learning**: CRD is positive across all methods, indicating significant clean accuracy degradation.
2. **Gradient confusion is negatively correlated with buffer size**: increasing buffer from 200 to 5120 notably improves cosine similarity of gradients relative to joint training.
3. **The counter-intuitive design of RAER is effective**: contrary to conventional practice (retaining hard samples), storing robust samples is more beneficial for continual adversarial learning.
4. AFLC can replace X-DER's logit masking with superior performance—logit masking is too extreme ($v_p = +\infty$), suppressing the learning of new knowledge.

## Highlights & Insights

1. **Problem diagnosis is more valuable than the solution**: the paper is the first to systematically analyze the dual interference between continual learning and adversarial training, establishing a foundation for future research.
2. **A new perspective on gradient confusion**: unlike conventional gradient confusion (arising from defense methods), the confusion here is a structural gradient deviation caused by limited replay data.
3. **Plug-and-play design**: AFLC and RAER can be freely combined with different continual learning algorithms and adversarial training methods.
4. **No additional data required**: unlike prior work, the approach relies entirely on existing training data and memory.

## Limitations & Future Work

1. Experiments are primarily conducted on small-scale datasets (CIFAR-10/100, Tiny-ImageNet); validation on large-scale settings is absent.
2. Only $\ell_\infty$-norm PGD attacks are considered; robustness to other attack types is not evaluated.
3. The calibration value $v_i$ in AFLC depends on a manually set hyperparameter $\alpha$; adaptive mechanisms remain to be explored.
4. The threshold $\rho$ in RAER is fixed; different tasks may require different thresholds.
5. Application to pretrained models (e.g., ViT, CLIP) is not discussed.

## Related Work & Insights

- **ER (Experience Replay)**: the simplest memory replay method and the primary vehicle for this paper's contributions.
- **TRADES**: an adversarial training method balancing clean accuracy and robustness; this paper extends its use to the continual learning setting.
- **X-DER**: employs logit masking to mitigate forgetting; AFLC is a more flexible generalization thereof.
- Insight: the intersection of adversarial robustness and continual learning remains largely open, particularly in more complex real-world scenarios.

## Rating

| Dimension | Score (1–5) |
|-----------|-------------|
| Novelty | 3.5 |
| Technical Depth | 4 |
| Experimental Thoroughness | 4 |
| Writing Quality | 3.5 |
| Value | 3.5 |
| Overall | 3.5 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Finding Structure in Continual Learning](../../NeurIPS2025/llm_safety/finding_structure_in_continual_learning.md)
- [\[AAAI 2026\] Attention Retention for Continual Learning with Vision Transformers](../../AAAI2026/llm_safety/attention_retention_for_continual_learning_with_vision_transformers.md)
- [\[ICML 2026\] Understanding Generalization and Forgetting in In-Context Continual Learning](../../ICML2026/llm_safety/understanding_generalization_and_forgetting_in_in-context_continual_learning.md)
- [\[NeurIPS 2025\] ReliabilityRAG: Effective and Provably Robust Defense for RAG-based Web-Search](../../NeurIPS2025/llm_safety/reliabilityrag_effective_and_provably_robust_defense_for_rag-based_web-search.md)
- [\[ICCV 2025\] Enhancing Adversarial Transferability by Balancing Exploration and Exploitation with Gradient-Guided Sampling](enhancing_adversarial_transferability_by_balancing_exploration_and_exploitation_.md)

</div>

<!-- RELATED:END -->
