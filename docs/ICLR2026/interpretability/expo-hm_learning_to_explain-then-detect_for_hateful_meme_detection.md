---
title: >-
  [Paper Note] ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection
description: >-
  [ICLR 2026][hate speech detection] ExPO-HM is proposed, inspired by the training pipeline of human content moderators. By combining policy manual SFT warm-up, GRPO curriculum learning, and a Conditional Decision Entropy (CDE) reward, it is the first Explain-then-Detect system to comprehensively surpass direct detection baselines across binary classification, fine-grained classification, and reasoning quality in hateful meme detection, achieving up to 15–17% F1 improvement.
tags:
  - ICLR 2026
  - hate speech detection
  - multimodal
  - GRPO
  - curriculum learning
  - conditional decision entropy
  - interpretability
date: 2026-05-08
content_hash: 48ff1a3303b9ffde
---

# ExPO-HM: Learning to Explain-then-Detect for Hateful Meme Detection

**Conference**: ICLR 2026
**arXiv**: [2510.08630](https://arxiv.org/abs/2510.08630)
**Code**: [GitHub](https://github.com/JingbiaoMei/ExPO-HM)
**Area**: Interpretability
**Keywords**: hate speech detection, multimodal, GRPO, curriculum learning, conditional decision entropy, interpretability

## TL;DR

ExPO-HM is proposed, inspired by the training pipeline of human content moderators. By combining policy manual SFT warm-up, GRPO curriculum learning, and a Conditional Decision Entropy (CDE) reward, it is the first Explain-then-Detect system to comprehensively surpass direct detection baselines across binary classification, fine-grained classification, and reasoning quality in hateful meme detection, achieving up to 15–17% F1 improvement.

## Background & Motivation

Hateful meme detection is a highly challenging online content moderation task. Existing approaches fall into two main paradigms:

**Direct Detection**: Outputs only a binary label (hateful/benign). Representative works such as RA-HMD rely on CLIP-based methods, achieving strong performance but providing no explanations, which does not satisfy real-world moderation requirements.

**Explain-then-Detect**: Generates natural language explanations before classification. However, existing systems of this type (e.g., LOREHM, U-CoT+), which use CoT prompting or agent frameworks, **perform worse than simple SFT baselines**. Even post-training methods such as GRPO fail to close this gap.

The authors identify two key issues:

**Model explanations omit critical cues**: Policy-relevant information such as attack targets and attack types is not considered as plausible explanatory hypotheses by the model.

**Binary reward signals are insufficient to guide reasoning**: Just as human annotators cannot learn effectively from yes/no labels alone, models also require more fine-grained feedback.

The central analogy is the **human moderator training pipeline**—first studying a detailed policy manual, then practicing from fine-grained category judgments to binary decisions—which motivates the design of ExPO-HM.

## Method

### Overall Architecture

ExPO-HM consists of three stages that simulate the human moderator training process:

1. **SFT-PM Warm-up**: Policy manual-enhanced supervised fine-tuning, teaching the model to understand moderation policies.
2. **GRPO-CL**: GRPO with curriculum learning, progressing from fine-grained to binary classification.
3. **CDE Reward**: Conditional Decision Entropy as a proxy reward for reasoning quality.

### Key Designs

**SFT Policy Manual Warm-up (SFT-PM)**: Fine-grained dataset labels are converted into structured policy manuals used as input prompts, with each policy item accompanied by descriptions from annotation guidelines. The LMM is trained with policy manual-augmented inputs, and the target responses are fine-grained labels. Human-written gold-standard explanations are not used (off-policy usage leads to worse performance).

**GRPO Curriculum Learning (GRPO-CL)**: A simple 50/50/50 strategy is adopted:
- First 50% of training steps: fine-grained data only (encouraging reasoning exploration)
- Remaining 50%: 50/50 mixture of fine-grained and binary classification data

Key effect: Standard GRPO produces an average response length of only 28 tokens for binary classification, whereas GRPO-CL nearly doubles this to 52 tokens, indicating more detailed reasoning.

**Conditional Decision Entropy (CDE)**:

Definition: Given explanation $\mathbf{e}$ and input $\mathbf{x}$, CDE is the entropy of the decision conditioned on the explanation:

$$H(d \mid \mathbf{e}, \mathbf{x}) = -\mathbb{E}_{d \sim \pi_\theta(\cdot|\mathbf{e},\mathbf{x})}[\log \pi_\theta(d \mid \mathbf{e}, \mathbf{x})]$$

Rationale: Good reasoning should lead to clear and correct decisions (low entropy), while poor reasoning produces confusion (high entropy).

**CDE Reward Design**:

$$r_{\text{CDE}}(h, \delta) = \delta \cdot f_{\text{correct}}(h) + (1-\delta) \cdot f_{\text{wrong}}(h)$$

where $\delta = \mathbf{1}[d = d^*]$ indicates prediction correctness. Correct and confident (low CDE) → reward; incorrect but confident → penalty (coefficient $\rho$); incorrect but uncertain → tolerated.

### Loss & Training

Total reward: $r = r_{\text{format}} + r_{\text{acc}} + w \cdot r_{\text{CDE}}$

where $r_{\text{format}} \in \{0,1\}$ checks output format and $r_{\text{acc}} \in [0,1\}$ measures prediction correctness. Standard GRPO clipped surrogate loss and KL regularization are applied. Default hyperparameters: $a=0.1$, $b=0.5$, $w=0.2$, $\rho=0.25$.

## Key Experimental Results

### Main Results

Evaluation is conducted on three datasets—HatefulMemes, MAMI, and PrideMM—using Qwen2.5-VL-3B and 7B as backbone models.

**Qwen2.5-VL-7B results on HatefulMemes:**

| Method | Binary F1 | Attack F1 | Target F1 | LLM Judge | CDE ↓ |
|--------|-----------|-----------|-----------|-----------|-------|
| Zero-shot | 65.9 | 44.7 | 64.5 | 5.0 | 0.33 |
| SFT | 74.5 | 58.4 | 69.4 | 5.0 | 0.33 |
| DPO | 73.6 | 63.2 | 66.6 | 4.9 | 0.32 |
| GRPO | 74.5 | 61.2 | 64.5 | 5.2 | 0.26 |
| RA-HMD (SOTA direct detection) | 80.2 | — | — | 5.5 | — |
| **ExPO-HM** | **81.1** | **75.6** | **77.2** | **6.2** | **0.03** |

ExPO-HM is the first Explain-then-Detect system to comprehensively surpass the direct detection SOTA (RA-HMD), while substantially leading in reasoning quality.

**Cross-dataset consistency** (7B model):

| Dataset | GRPO Binary F1 | ExPO-HM Binary F1 | Gain |
|---------|---------------|-------------------|------|
| HatefulMemes | 74.5 | 81.1 | +6.6 |
| MAMI | 76.8 | 82.3 | +5.5 |
| PrideMM | 73.2 | 78.7 | +5.5 |

### Ablation Study

| # | SFT-PM | GRPO-CL | CDE | Binary F1 | Attack F1 | Target F1 | LLM ↑ | CDE ↓ |
|---|--------|---------|-----|-----------|-----------|-----------|-------|-------|
| 1 | - | - | - | 74.5 | 61.2 | 64.5 | 5.2 | 0.263 |
| 2 | ✓ | - | - | 75.8 | 70.8 | 70.2 | 5.6 | 0.092 |
| 3 | ✓ | ✓ | - | 78.4 | 74.3 | 76.1 | 5.8 | 0.056 |
| 4 | ✓ | ✓ | ✓ | **81.1** | **75.6** | **77.2** | **6.2** | **0.026** |

All three components contribute: SFT-PM substantially improves fine-grained metrics, GRPO-CL further advances all metrics, and CDE markedly enhances reasoning quality (LLM Judge 5.8→6.2).

### Key Findings

1. **Explain-then-Detect surpasses Direct Detection for the first time**: All prior systems of this type fall short of the SFT baseline.
2. **CDE strongly correlates with LLM-Judge**: Pearson $r=-0.78$, Spearman $\rho=-0.81$ ($p<0.001$).
3. **SFT warm-up strategy is critical**: Binary-only SFT performs worse than the no-warm-up baseline after the RL stage.
4. **CDE does not cause policy entropy collapse**: Overall policy entropy is comparable to baselines without CDE.
5. **Human evaluation validates the approach**: ExPO-HM achieves 100% logical consistency vs. 96% for GRPO, with helpfulness scores of 2.2 vs. 1.6.

## Highlights & Insights

1. **The analogy to human annotator training is highly precise**: The progressive pipeline—policy manual → fine-grained practice → binary judgment—translates naturally into the algorithm design.
2. **CDE is an excellent proxy for reasoning quality**: Conceptually simple (conditional entropy), strongly correlated with human evaluation, and usable as a differentiable reward signal.
3. **Key finding: good SFT does not necessarily lead to good RL**: Binary SFT achieves the best SFT-stage performance but yields the worst results after RL.
4. **Consistency across three datasets**: The method generalizes well across different hateful content types.
5. **Exceptionally comprehensive evaluation**: Ablations, warm-up strategy comparisons, CDE analysis, calibration analysis, and human evaluation are all included.

## Limitations & Future Work

1. **Limited dataset scale**: Annotated hateful meme data, especially with explanations, is extremely scarce.
2. **Single-turn interaction**: Only single-turn reasoning is evaluated; multi-turn moderation dialogue scenarios are not considered.
3. **Cultural dependency**: Moderation policies are highly culture-specific; cross-cultural applicability is not validated.
4. **Backbone model constraints**: Validation is limited to Qwen2.5-VL 3B/7B.
5. The framework is extensible to other content moderation tasks (e.g., misinformation detection, cyberbullying identification).

## Related Work & Insights

- **RA-HMD** (Mei et al., 2025): The previous SOTA direct detection method, which ExPO-HM is the first to surpass.
- **LOREHM** (Huang et al., 2024): A reasoning agent framework based on LLaVA-Next-34B.
- **GRPO** (Shao et al., 2024): ExPO-HM builds upon this by incorporating the CDE reward and curriculum learning.
- Insight: The "reasoning quality proxy" concept underlying CDE is generalizable to other tasks requiring interpretable reasoning.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The CDE concept is original, and the curriculum learning strategy is elegantly designed.
- **Technical Depth**: ⭐⭐⭐⭐ — The mapping from the human training pipeline to concrete algorithmic design is complete and coherent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three datasets, multiple baselines, ablations, and human evaluation make for an exceptionally thorough evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clear and experiments are well organized.
- **Value**: ⭐⭐⭐⭐ — Directly applicable to content moderation in practice.
- **Overall Recommendation**: ⭐⭐⭐⭐ (4/5)

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Language Models Can Explain Visual Features via Steering](../../CVPR2026/interpretability/language_models_can_explain_visual_features_via_steering.md)
- [\[ICLR 2026\] Behavior Learning (BL): Learning Hierarchical Optimization Structures from Data](behavior_learning_bl_learning_hierarchical_optimization_structures_from_data.md)
- [\[ICLR 2026\] Decoupling Dynamical Richness from Representation Learning: Towards Practical Measurement](decoupling_dynamical_richness_from_representation_learning_towards_practical_mea.md)
- [\[ICLR 2026\] Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning](decomposing_representation_space_into_interpretable_subspaces_with_unsupervised_.md)
- [\[ICLR 2026\] Tokenizing Single-Channel EEG with Time-Frequency Motif Learning](tokenizing_single-channel_eeg_with_time-frequency_motif_learning.md)

<!-- RELATED:END -->
