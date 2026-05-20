---
title: >-
  [Paper Note] Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression
description: >-
  [ICML 2026][Multimodal VLM][MLLM Regression] This work reframes the "regression to the mean" issue of MLLM continuous value regression under long-tail distributions as a distribution-aware RL problem. Within the GRPO fra…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "MLLM Regression"
  - "Long-tail Distribution"
  - "GRPO"
  - "Concordance Correlation Coefficient"
  - "Batch-level Reward"
date: 2026-05-08
content_hash: 1d1864c97e70c7b5
---

# Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression

**Conference**: ICML 2026  
**arXiv**: [2605.01402](https://arxiv.org/abs/2605.01402)  
**Code**: To be released after acceptance (not available yet)  
**Area**: Multimodal VLM / Reinforcement Learning / Deep Imbalanced Regression  
**Keywords**: MLLM Regression, Long-tail Distribution, GRPO, Concordance Correlation Coefficient, Batch-level Reward

## TL;DR
This work reframes the "regression to the mean" issue of MLLM continuous value regression under long-tail distributions as a distribution-aware RL problem. Within the GRPO framework, the Concordance Correlation Coefficient (CCC) is used as a batch-level reward—simultaneously considering correlation, variance, and mean—thus explicitly penalizing prediction distribution collapse. On four long-tail regression tasks and Qwen2.5-VL-3B/7B, it consistently outperforms SFT, SoftLabel, and various point-wise RL methods, with especially significant MAE reductions in medium/few-shot regions.

## Background & Motivation

**Background**: MLLMs are increasingly used for "continuous value regression" (e.g., age, rating, bone age), but the mainstream approach is token-level SFT (splitting numbers into tokens for cross-entropy) or GRPO with point-wise regression rewards (MAE, Reward Reward, etc.).

**Limitations of Prior Work**: (i) Token-level SFT treats regression as discrete classification, so predicting "5 years as 6" and "5 years as 50" may incur the same token loss, being insensitive to numerical distance; (ii) Under long-tail supervision, most samples cluster in the head, and SFT causes model predictions to collapse toward the mean (as shown in Fig 1); (iii) Existing regression methods either modify the architecture (Rex-Omni adds coordinate tokens, GEODE adds regression heads), breaking the unified MLLM generation framework, or rely on CoT reasoning (slow inference), or SoftLabel smooths hard one-hot labels—but all still provide "local per-token signals"; (iv) RL methods (Visual-RFT, VLM-R1, Perception-R1) still use per-sample MAE as reward, evaluating each sample independently and failing to address long-tail structure.

**Key Challenge**: Long-tail regression requires "cross-sample relative relationships" to maintain global distribution structure, but SFT and per-sample RL rewards only consider pointwise errors, failing to convey "you cannot predict all samples as the median".

**Goal**: (i) Solve MLLM long-tail regression collapse purely via post-training, without architecture changes or CoT; (ii) Enable supervision signals to sense the consistency between "predicted distribution vs. ground-truth distribution"; (iii) Explicitly penalize mean collapse and variance shrinkage.

**Key Insight**: The authors observe that RL's advantage is the ability to compute arbitrary rewards on decoded values. Rather than rewarding "pointwise closeness to ground truth", it is better to reward "the distribution of a batch of predictions matching the distribution of a batch of ground truths"—upgrading the "numerical" problem to a "distributional" one.

**Core Idea**: For each minibatch, concatenate each sampled prediction with the mean predictions of other samples to form a vector, and compute CCC with the corresponding ground-truth vector. Use CCC as the reward—CCC simultaneously rewards correlation, penalizes variance collapse, and penalizes mean shift, achieving all three objectives.

## Method

### Overall Architecture
The GRPO framework remains unchanged: for each input $x_i$, sample $K$ generation trajectories to obtain numerical predictions $\{q_k(x_i)\}_{k=1}^K$, and take the mean $\mu(x_i) = \frac{1}{K}\sum_k q_k(x_i)$ as a stable anchor. When evaluating the $k$-th trajectory, concatenate it with the mean predictions of other samples in the minibatch to form $\mathbf{q}_{i,k} = [q_k(x_i), \{\mu(x_j)\}_{j\neq i}]$, and the corresponding ground-truth vector $\mathbf{y}_i = [y_i, \{y_j\}_{j\neq i}]$; the reward is $r_k(x_i) = \text{CCC}(\mathbf{q}_{i,k}, \mathbf{y}_i)$. An additional lightweight format-check reward ensures decodability. Finally, standard GRPO normalizes within the group to obtain relative advantage for policy updates.

### Key Designs

1. **Batch-level Relative Comparison Vector Construction**:

    - **Function**: Each sampled prediction is evaluated in the context of the "minibatch group", introducing cross-sample relationships.
    - **Mechanism**: For minibatch $\{x_1,\ldots,x_B\}$, sample $K$ trajectories per $x_i$; when evaluating $q_k(x_i)$, concatenate it with the mean predictions of the other $B-1$ samples to form a length-$B$ vector $\mathbf{q}_{i,k}$; the ground-truth vector $\mathbf{y}_i$ is constructed in the same index order. These two vectors are scored by CCC. Using $\mu(x_j)$ as the anchor for others, rather than a single sample, reduces reward noise from cross-sample randomness.
    - **Design Motivation**: Traditional GRPO reward is $r_k = \text{MAE}(q_k, y_i)$, always pointwise; as long as each sample independently "matches its ground truth", the model is rewarded, which encourages collapse to high-density regions. Embedding "comparison with others" into the reward vector injects "group distribution shape" signals into the gradient.

2. **CCC Reward: Simultaneously Controls Correlation, Variance, and Mean**:

    - **Function**: Provides a scalar reward encoding three aspects—correlation, scale consistency, and mean alignment.
    - **Mechanism**: $\text{CCC}(\mathbf{q}, \mathbf{y}) = \frac{2\,\text{Cov}(\mathbf{q}, \mathbf{y})}{\text{Var}(\mathbf{q}) + \text{Var}(\mathbf{y}) + (\mu_{\mathbf{q}} - \mu_{\mathbf{y}})^2}$. The numerator is covariance (rewards order consistency); in the denominator, small $\text{Var}(\mathbf{q})$ (collapse) is penalized, and large $(\mu_{\mathbf{q}} - \mu_{\mathbf{y}})^2$ (systematic bias) is also penalized. Pure Pearson only rewards correlation, ignoring scale and mean; pure ranking only considers order, not values. CCC handles all three.
    - **Design Motivation**: The two typical pathologies of long-tail regression collapse—variance shrinkage (predicting a single value) and mean shift (systematic bias toward the head)—are directly penalized by the two denominator terms of CCC. In sparse regions, CCC is less prone to "illusory good results" from compressed but correlated predictions than pure Pearson.

3. **Supporting Data Benchmark: DIR-for-MLLM Unified Evaluation Protocol**:

    - **Function**: Establishes a fair benchmark for evaluating MLLM long-tail regression, avoiding incomparable results due to different splits.
    - **Mechanism**: Unifies four long-tail regression tasks—AgeDB-DIR, IMDB-WIKI-DIR, IMDB-Movie-DIR (movie poster rating, newly constructed), BoneAge-DIR—into dialogue-format MLLM inputs. The training set maintains the natural long-tail distribution; the test set is split by shot region (many >100, medium 20-100, few <20) for balanced evaluation; totaling over 129k samples. Evaluation uses MAE + GM (geometric mean, more sensitive to uniformity).
    - **Design Motivation**: Traditional DIR methods are developed on CNN + regression heads, not token-decoder settings. Porting DIR to MLLM requires first establishing data/evaluation protocols; otherwise, subsequent comparisons lack foundation.

### Loss & Training
The GRPO optimizer is used as is, with only the reward changed; reward = CCC + lightweight format reward; within-group z-score normalization yields relative advantage; backbone uses Qwen2.5-VL-3B and 7B; shot-aware evaluation on the test set ensures fair head/tail comparison.

## Key Experimental Results

### Main Results

| Dataset | Method (Qwen2.5-VL-3B) | All MAE | Many MAE | Medium MAE | Few MAE |
|---------|------------------------|---------|----------|------------|---------|
| AgeDB-DIR | SFT | 6.37 | 5.78 | 7.67 | 8.36 |
| AgeDB-DIR | Regression Reward (Tan 2025) | 5.85 | 5.48 | 6.52 | 7.58 |
| AgeDB-DIR | DISCO MAE Reward | 5.95 | 5.64 | 6.73 | 6.75 |
| AgeDB-DIR | **CCC-GRPO (Ours)** | **5.52** | 5.42 | **5.62** | **6.40** |
| IMDB-Movie-DIR | SFT | 7.44 | 4.87 | 11.21 | 21.51 |
| IMDB-Movie-DIR | Regression Reward | 7.42 | 5.06 | 10.51 | 21.14 |
| IMDB-Movie-DIR | **CCC-GRPO** | **6.89** | 5.60 | **8.12** | **16.35** |

Results on 7B are also best: AgeDB All MAE 5.33 vs SFT 5.82; Movie All MAE 5.95 vs SFT 6.42.

### Ablation Study

| Setting | Key Phenomenon | Description |
|---------|----------------|-------------|
| SFT (point-wise CE) | Prediction collapses to head (Fig 1) | Long-tail collapse baseline |
| GRPO + MAE reward | Still per-sample, no cross-sample structure | Limited improvement in medium/few regions |
| GRPO + DISCO MAE (Zhou 2025) | Frequency-weighted reward | Slightly better in medium/few but still pointwise |
| **GRPO + CCC reward (Ours)** | Explicitly penalizes collapse + drift | Significant drop in medium/few |
| BoneAge-DIR (multi-modal distribution) | CCC-GRPO still superior | 23.55% overall MAE improvement over SFT (Table 12) |

### Key Findings
- **Greatest gains in medium/few-shot**: On Movie, few-shot MAE drops from SFT's 21.51 to 16.35 (−24%); AgeDB few-shot from 8.36 to 6.40 (−23%), confirming CCC mainly addresses "long-tail compression".
- **No sacrifice in head**: Many-shot MAE is comparable to or slightly higher than SFT/Reg Reward (e.g., Movie 4.87→5.60), but the large gains in medium/few make this a meaningful trade-off.
- **GM improvement is notable**: AgeDB many GM 5.78→5.42, indicating error distribution is more uniform rather than dominated by a few large errors.
- **BoneAge multi-modal scenario**: Training labels are multi-modal (not just long-tail), yet CCC-GRPO still achieves a 23.55% overall improvement, showing the reward does not rely on a "unimodal long-tail" assumption but generally rewards "distributional shape consistency".
- **VisualQuality counterexample**: CoT-based VisualQuality yields MAE as high as 24.43 on Movie and similar tasks, indicating unsuitability for perceptual regression—highlighting that post-training reward selection is more effective than reasoning.

## Highlights & Insights
- **"Reward shape determines prediction shape"**: The main insight is that GRPO's reward can shape the overall prediction distribution—switching from "pointwise similarity" to "distributional similarity" enables the model to naturally preserve variance and scale, without any architectural or loss-level hacks.
- **CCC is an underrated metric**: Common in medical/psychometric fields, but rarely used as RL reward; its geometric property of combining "correlation + variance + mean" directly addresses the two main issues of long-tail collapse, and is worth trying in RM/preference learning.
- **Intra-group "other mean anchor"**: Using $\mu(x_j)$ as a representative for other samples to reduce noise, akin to Polyak averaging, is a simple yet effective technique to lower reward variance.
- **Establishing DIR-for-MLLM benchmark**: Transferring the mature DIR protocol from classification to generative MLLMs will facilitate substantial follow-up work.

## Limitations & Future Work
- CCC reward requires sufficient diversity within the minibatch to be meaningful; performance may degrade with small batches (<8) or in scenarios with very low intra-class variance. Batch size sensitivity was not studied.
- Only tested on four 2D vision regression tasks; whether extension to multivariate regression (e.g., bbox 4D, 3D pose) requires multivariate CCC is unexplored.
- CCC is a non-differentiable signal in the formal objective, relying on GRPO's policy gradient estimation; variance remains high when $K$ is small, and the optimal $K$ was not analyzed.
- No comparison with the latest GRPO variants such as DAPO, RLOO, so complementarity between reward and optimizer improvements is unclear.
- Code is not officially released ("after acceptance"), so reproduction must wait.

## Related Work & Insights
- **vs Classic DIR (Yang 2021, RankSim, VIR)**: Classic methods use regression heads + label/feature smoothing, not applicable to token-decoder MLLMs; this work essentially brings "distribution smoothing" to the reward layer.
- **vs SoftLabel (Wang 2025b)**: SoftLabel smooths supervision at the token loss level, still a local token signal; CCC operates at the sequence-level reward, cross-sample, at a higher granularity.
- **vs DISCO MAE Reward (Zhou 2025)**: DISCO scales reward by domain/difficulty but remains per-sample; CCC directly treats "inter-sample relationships" as a first-order reward term.
- **vs Reasoning-based VisualQuality (Wu 2025)**: CoT reasoning is ineffective for perceptual regression; this work shows that "choosing the right reward" is more important than "adding reasoning".
- **vs Rex-Omni / GEODE**: They modify vocab or add heads for retraining, which is heavy; this work leaves the architecture unchanged, enabling post-training application.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing CCC into RL reward and explicitly targeting "distribution collapse" in long-tail regression is a relatively fresh combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 datasets × 2 backbones × multiple baselines + shot-aware protocol + ranking error curves; density is adequate, though key ablations like batch size and $K$ are missing.
- Writing Quality: ⭐⭐⭐⭐ Fig 2's three-column comparison clearly illustrates the differences among SFT / GRPO / CCC-GRPO; Fig 5 directly shows long-tail region gains via MAE.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play solution for "fine-grained numerical prediction with MLLMs" without architecture changes, highly practical for industrial applications (age, bone age, rating prediction).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Targeted Exploration via Unified Entropy Control for Reinforcement Learning](../../ACL2026/multimodal_vlm/targeted_exploration_via_unified_entropy_control_for_reinforcement_learning.md)
- [\[ICML 2026\] Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives](multimodal_continual_learning_with_mllms_from_multi-scenario_perspectives.md)
- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](../../ACL2026/multimodal_vlm/multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)
- [\[CVPR 2026\] Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](../../CVPR2026/multimodal_vlm/reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)
- [\[CVPR 2026\] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](../../CVPR2026/multimodal_vlm/moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)

</div>

<!-- RELATED:END -->
