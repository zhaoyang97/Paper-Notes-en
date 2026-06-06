---
title: >-
  [Paper Note] Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression
description: >-
  [ICML 2026][Multimodal VLM][MLLM Regression] This paper treats the "regression to the mean" problem of MLLM continuous value regression under long-tail distributions as a distribution-aware RL problem. Within the GRPO fr…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "MLLM Regression"
  - "Long-tail distribution"
  - "GRPO"
  - "Concordance Correlation Coefficient (CCC)"
  - "Batch-level reward"
date: 2026-05-08
content_hash: 44f22e597698b275
---

# Injecting Distributional Awareness into MLLMs via Reinforcement Learning for Deep Imbalanced Regression

**Conference**: ICML 2026  
**arXiv**: [2605.01402](https://arxiv.org/abs/2605.01402)  
**Code**: The paper states it will be released upon acceptance (currently unavailable)  
**Area**: Multimodal VLM / Reinforcement Learning / Deep Imbalanced Regression  
**Keywords**: MLLM Regression, Long-tail distribution, GRPO, Concordance Correlation Coefficient (CCC), Batch-level reward

## TL;DR
This paper treats the "regression to the mean" problem of MLLM continuous value regression under long-tail distributions as a distribution-aware RL problem. Within the GRPO framework, it uses the Concordance Correlation Coefficient (CCC) as a batch-level reward—evaluating correlation, variance, and mean simultaneously—to explicitly penalize predictive distribution collapse. On 4 long-tail regression tasks using Qwen2.5-VL-3B/7B, this approach consistently outperforms SFT, SoftLabel, and various point-wise RL methods, with MAE dropping significantly in medium/few-shot regions.

## Background & Motivation

**Background**: MLLMs are increasingly utilized for "continuous value regression" (age, ratings, bone age, etc.). However, current mainstream strategies rely on token-level SFT (decomposing numbers into tokens for cross-entropy) or GRPO with point-wise regression rewards (MAE, Reward Reward, etc.).

**Limitations of Prior Work**: (i) Token-level SFT treats regression as discrete classification, where predicting "6 years old" for a "5-year-old" might yield the same token loss as predicting "50 years old," losing all sensitivity to numerical distance. (ii) Under long-tail supervision, where most samples belong to the head, SFT causes model predictions to collapse toward the mean (clearly visible in Fig 1). (iii) Existing regression-specific methods either modify the architecture (e.g., Rex-Omni's coordinate tokens, GEODE's regression head), which breaks the unified MLLM generation framework, or rely on slow CoT reasoning, or use SoftLabel to smooth hard one-hots—yet all remain "per-token local signals." (iv) RL methods (Visual-RFT, VLM-R1, Perception-R1) still use per-sample MAE rewards, evaluating each sample independently and failing to address the long-tail structure.

**Key Challenge**: Long-tail regression requires "cross-sample relative relationships" to maintain the global distribution structure, whereas both SFT and per-sample RL rewards only consider single-point errors, failing to convey the signal that "not all samples should be predicted as the median."

**Goal**: (i) Solve MLLM long-tail regression collapse via pure post-training without architecture changes or CoT; (ii) enable the supervision signal to perceive the consistency between the "predicted distribution vs. ground-truth distribution"; (iii) explicitly penalize mean collapse and variance collapse.

**Key Insight**: The advantage of RL lies in the ability to calculate arbitrary rewards on decoded numerical values. Rather than rewarding "single-point proximity to ground truth," it is more effective to reward "the distribution of a batch of predictions for being close to the distribution of a batch of ground truths"—upgrading the numerical problem to a distribution problem.

**Core Idea**: Concatenate each sampled prediction in a minibatch with the mean predictions of other samples into a vector. Calculate the CCC between this vector and the corresponding ground-truth vector. Use CCC as the reward, as it simultaneously rewards correlation, penalizes variance collapse, and penalizes mean shift.

## Method

### Overall Architecture
The GRPO framework remains unchanged: for each input $x_i$, $K$ generation trajectories are sampled to obtain numerical predictions $\{q_k(x_i)\}_{k=1}^K$, with the mean $\mu(x_i) = \frac{1}{K}\sum_k q_k(x_i)$ serving as a stable anchor. When evaluating the $k$-th trajectory, it is concatenated with the mean predictions of other samples in the minibatch to form $\mathbf{q}_{i,k} = [q_k(x_i), \{\mu(x_j)\}_{j\neq i}]$, corresponding to the ground-truth vector $\mathbf{y}_i = [y_i, \{y_j\}_{j\neq i}]$. The reward is defined as $r_k(x_i) = \text{CCC}(\mathbf{q}_{i,k}, \mathbf{y}_i)$, with an additional lightweight format validation reward to ensure decodability. Relative advantages for policy updates are derived using standard intra-group normalization in GRPO.

### Key Designs

1. **Batch-level Relative Comparison Vector Construction**:
    - **Function**: Allows each sampled prediction to be evaluated within the context of a "minibatch group," introducing cross-sample relationships.
    - **Mechanism**: For a minibatch $\{x_1,\ldots,x_B\}$, $K$ trajectories are sampled for each $x_i$. When evaluating $q_k(x_i)$, it is concatenated with the "mean predictions of the other B-1 samples" to form a vector $\mathbf{q}_{i,k}$ of length B. The ground-truth side is concatenated in the same index order to form $\mathbf{y}_i$. These two vectors are used to compute CCC. Utilizing $\mu(x_j)$ as an anchor for others instead of a single sample reduces reward noise from cross-sample stochasticity.
    - **Design Motivation**: Traditional GRPO rewards use $r_k = \text{MAE}(q_k, y_i)$, which only looks at a single point. As long as each sample independently moves "closer to its ground truth," the model is rewarded, which inadvertently encourages collapse toward high-density regions. Embedding the "comparison with others" into the reward vector passes the "group distribution shape" signal into the gradient.

2. **CCC Reward: Simultaneously Managing Correlation, Variance, and Mean**:
    - **Function**: Provides a scalar reward that encodes three things: correlation, scale consistency, and mean alignment.
    - **Mechanism**: 
    $$\text{CCC}(\mathbf{q}, \mathbf{y}) = \frac{2\,\text{Cov}(\mathbf{q}, \mathbf{y})}{\text{Var}(\mathbf{q}) + \text{Var}(\mathbf{y}) + (\mu_{\mathbf{q}} - \mu_{\mathbf{y}})^2}$$
    The numerator is the covariance (rewarding order consistency). In the denominator, a small $\text{Var}(\mathbf{q})$ (collapse) or a large $(\mu_{\mathbf{q}} - \mu_{\mathbf{y}})^2$ (systematic bias) is penalized. Unlike Pearson correlation (which ignores scale/mean) or ranking metrics (which ignore values), CCC addresses all three.
    - **Design Motivation**: The two typical symptoms of long-tail regression collapse—variance shrinkage (predicting a uniform value) and mean drift (systematic bias toward the head center)—are exactly addressed by the two terms in the CCC denominator. In sparse regions, CCC is less likely than pure Pearson to produce "pseudo-good" results that appear correlated but are actually compressed.

3. **Supporting Benchmark: DIR-for-MLLM Unified Evaluation Protocol**:
    - **Function**: Establishes a benchmark for fair evaluation of MLLM long-tail regression, avoiding inconsistencies between different methods' splits.
    - **Mechanism**: Four long-tail regression tasks—AgeDB-DIR, IMDB-WIKI-DIR, IMDB-Movie-DIR (movie poster ratings, newly constructed by the authors), and BoneAge-DIR—are unified into a dialogue-format MLLM input. The training set maintains a natural long-tail distribution, while the test set is split by shot regions (many >100, medium 20-100, few <20) for balanced evaluation, totaling 129k+ samples. Evaluation uses MAE and Geometic Mean (GM), which is more sensitive to uniformity.
    - **Design Motivation**: Traditional DIR methods developed on CNNs + regression heads lack the token-decoder setting. Porting DIR to MLLMs required establishing these protocols first.

### Loss & Training
The approach strictly follows the GRPO optimizer, modifying only the reward function (Reward = CCC + Format Reward). Intra-group z-score normalization yields the relative advantage. Backbones used include Qwen2.5-VL-3B and 7B. Shot-aware evaluation on the test set ensures fair comparisons between head and tail regions.

## Key Experimental Results

### Main Results

| Dataset | Method (Qwen2.5-VL-3B) | All MAE | Many MAE | Medium MAE | Few MAE |
|--------|----------------------|---------|----------|------------|---------|
| AgeDB-DIR | SFT | 6.37 | 5.78 | 7.67 | 8.36 |
| AgeDB-DIR | Regression Reward (Tan 2025) | 5.85 | 5.48 | 6.52 | 7.58 |
| AgeDB-DIR | DISCO MAE Reward | 5.95 | 5.64 | 6.73 | 6.75 |
| AgeDB-DIR | **CCC-GRPO (Ours)** | **5.52** | 5.42 | **5.62** | **6.40** |
| IMDB-Movie-DIR | SFT | 7.44 | 4.87 | 11.21 | 21.51 |
| IMDB-Movie-DIR | Regression Reward | 7.42 | 5.06 | 10.51 | 21.14 |
| IMDB-Movie-DIR | **CCC-GRPO** | **6.89** | 5.60 | **8.12** | **16.35** |

Performance on 7B is also superior: AgeDB All MAE 5.33 vs SFT 5.82; Movie All MAE 5.95 vs SFT 6.42.

### Ablation Study

| Setting | Key Findings | Description |
|------|---------|------|
| SFT (point-wise CE) | Prediction collapses toward the head (Fig 1) | Baseline for long-tail collapse |
| GRPO + MAE reward | Still per-sample, no cross-sample structure | Limited improvement in medium/few regions |
| GRPO + DISCO MAE (Zhou 2025) | Frequency-weighted reward | Slight improvement in medium/few but still point-to-point |
| **GRPO + CCC reward (Ours)** | Explicitly penalizes collapse + drift | Substantial decrease in medium/few MAE |
| BoneAge-DIR (Multi-modal) | CCC-GRPO remains superior | 23.55% overall MAE improvement over SFT (Table 12) |

### Key Findings
- **Highest gains in medium/few-shot regions**: In Movie, few-shot MAE dropped from 21.51 (SFT) to 16.35 (−24%); in AgeDB, it dropped from 8.36 to 6.40 (−23%), confirming that CCC primarily fixes "long-tail compression."
- **No sacrifice of the head**: Many-shot MAE remains comparable to or slightly higher than SFT/Reg Reward (e.g., 4.87 → 5.60 in Movie), but this is a meaningful trade-off for the massive medium/few gains.
- **Significant GM improvement**: In AgeDB, many GM 5.78 → 5.42, indicating a more uniform error distribution rather than one dominated by extreme outliers.
- **Multimodal BoneAge scenario**: Even with multi-modal training labels (not simple long-tail), CCC-GRPO improves by 23.55%, showing the reward generalizes to "distribution shape consistency" rather than relying on a single-peak assumption.
- **VisualQuality Counter-example**: The CoT-based method VisualQuality yielded an MAE as high as 24.43 on Movie, showing that reasoning-based approaches are ill-suited for perceptual regression. This highlights that choosing the right reward in post-training is more effective than adding "reasoning."

## Highlights & Insights
- **"Reward shape determines prediction shape"**: The main insight is that GRPO's reward can shape the overall distribution of predictions. By switching from "point-wise similarity" to "distributional similarity," the model naturally learns to preserve variance and scale without architecture hacks.
- **CCC is an undervalued metric**: Often used in medicine/psychometrics, it is rarely seen as an RL reward. Its trinity of "correlation + variance + mean" properties happens to hit both symptoms of long-tail collapse, making it a viable candidate for RM/preference learning elsewhere.
- **Intra-group "Mean Anchor" strategy**: Using $\mu(x_j)$ as a representative for other samples acts like a Polyak average, serving as a simple yet effective technique to reduce reward variance.
- **Establishing the DIR-for-MLLM benchmark**: Porting mature DIR protocols from the classification era into generative MLLMs provides a necessary foundation for subsequent work.

## Limitations & Future Work
- CCC reward requires sufficient diversity within the minibatch; results might degrade for small batches (<8) or scenarios with minimal intra-class variance (batch size sensitivity was not explored).
- Only 2D visual regression tasks were tested; whether it extends to multivariate regression (e.g., 4D bbox, 3D pose) using multivariate CCC remains unexplored.
- CCC is a non-differentiable signal estimated via policy gradients in GRPO; variance may still be high when $K$ is small.
- Comparisons against the latest GRPO variants like DAPO or RLOO are missing, so it is unclear if reward improvements are complementary to optimizer improvements.
- No public code yet ("after acceptance").

## Related Work & Insights
- **vs. Classic DIR (Yang 2021, RankSim, VIR)**: Classic methods use regression heads + label/feature smoothing, which are inapplicable to token-decoder MLLMs. This paper effectively moves the "distribution smoothing" idea to the reward layer.
- **vs. SoftLabel (Wang 2025b)**: SoftLabel smooths supervision at the token loss layer (local signal). CCC operates at the sequence-level reward (cross-sample signal), which is higher-level.
- **vs. DISCO MAE Reward (Zhou 2025)**: DISCO scales rewards by domain/difficulty but remains per-sample. CCC treats "inter-sample relationships" as a first-order term of the reward.
- **vs. Reasoning-based VisualQuality (Wu 2025)**: CoT reasoning is largely ineffective for perceptual regression. This work shows "right reward" > "adding reasoning" for this task.
- **vs. Rex-Omni / GEODE**: These require modifying vocabularies or adding heads; this approach is architecture-agnostic and only requires post-training.

## Rating
- Novelty: ⭐⭐⭐⭐ Introducing CCC to RL rewards to specifically target "distribution collapse" in long-tail regression is a refreshing combination.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 datasets × 2 backbones × multiple baselines + shot-aware protocols + ranking error curves; however, it lacks critical ablations on batch size and K.
- Writing Quality: ⭐⭐⭐⭐ The three-column comparison in Fig 2 clearly illustrates the differences between SFT, GRPO, and CCC-GRPO. Fig 5 effectively uses MAE gain to show tail-end benefits.
- Value: ⭐⭐⭐⭐ Provides a plug-and-play solution for fine-grained numerical prediction in MLLMs, highly practical for industrial applications like age or rating prediction.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] iVGR: Internalizing Visually Grounded Reasoning for MLLMs with Reinforcement Learning](ivgr_internalizing_visually_grounded_reasoning_for_mllms_with_reinforcement_lear.md)
- [\[ICML 2026\] Deep Pre-Alignment for VLMs](deep_pre-alignment_for_vlms.md)
- [\[ICML 2026\] Multimodal Continual Learning with MLLMs from Multi-scenario Perspectives](multimodal_continual_learning_with_mllms_from_multi-scenario_perspectives.md)
- [\[CVPR 2026\] Reason-SVG: Enhancing Structured Reasoning for Vector Graphics Generation with Reinforcement Learning](../../CVPR2026/multimodal_vlm/reason-svg_enhancing_structured_reasoning_for_vector_graphics_generation_with_re.md)
- [\[CVPR 2026\] MoE-GRPO: Optimizing Mixture-of-Experts via Reinforcement Learning in Vision-Language Models](../../CVPR2026/multimodal_vlm/moe-grpo_optimizing_mixture-of-experts_via_reinforcement_learning_in_vision-lang.md)

</div>

<!-- RELATED:END -->
