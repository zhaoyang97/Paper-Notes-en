---
title: >-
  [Paper Note] DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition
description: >-
  [ICLR2026][Reinforcement Learning][fine-grained recognition] This paper proposes DiVE-k, a framework that constructs multiple-choice questions (MCQs) from the top-k outputs of a large vision-language model (LVLM) and trains the model via GRPO reinforcement learning to perform differential visual reasoning, achieving substantial improvements in base-to-novel generalization for fine-grained image recognition.
tags:
  - ICLR2026
  - Reinforcement Learning
  - fine-grained recognition
  - reinforcement-learning
  - GRPO
  - visual reasoning
  - multiple-choice question
  - LVLM
date: 2026-05-08
content_hash: 168f1cf2f391a056
---

# DiVE-k: Differential Visual Reasoning for Fine-grained Image Recognition

**Conference**: ICLR2026
**arXiv**: [2511.18305](https://arxiv.org/abs/2511.18305)
**Code**: [raja-kumar/DiVE-k](https://github.com/raja-kumar/DiVE-k)
**Area**: Reinforcement Learning
**Keywords**: fine-grained recognition, reinforcement-learning, GRPO, visual reasoning, multiple-choice question, LVLM

## TL;DR

This paper proposes DiVE-k, a framework that constructs multiple-choice questions (MCQs) from the top-k outputs of a large vision-language model (LVLM) and trains the model via GRPO reinforcement learning to perform differential visual reasoning, achieving substantial improvements in base-to-novel generalization for fine-grained image recognition.

## Background & Motivation

Despite possessing rich textual knowledge, LVLMs struggle with fine-grained image recognition and fail to distinguish visually similar categories. The authors identify two key observations:

1. **A large gap exists between Pass@1 and Pass@K**: Models frequently cover the correct answer within K samples, yet exhibit low Pass@1 accuracy. This indicates over-reliance on coarse salient features and a lack of fine-grained differential reasoning.
2. **LVLMs inherently contain detailed knowledge of fine-grained categories** (e.g., part attributes, appearance descriptions), but existing methods fail to elicit this knowledge for discriminating similar classes.

Existing RL-based methods such as ViRFT use exact string matching as a reward signal, which suffers from three issues: (a) brittle string matching due to inconsistencies between scientific and common names; (b) incentivizing rote memorization of training class names; (c) inability to encourage attribute-level differential reasoning. These deficiencies lead to poor generalization from base to novel classes.

## Core Problem

How can LVLMs be trained to perform effective differential reasoning in fine-grained image recognition — i.e., identifying the correct answer among visually similar candidates by comparing key discriminative attributes — such that this reasoning ability generalizes to unseen novel categories?

## Method

The DiVE-k framework consists of two stages:

### Stage 1: Offline Top-k Option Mining

For each training image $I$ and query $q$, the base model $\pi_\theta$ is sampled $K$ times ($K=20$), and all predicted class names are collected. The top-$k$ most frequent classes are selected as the option set $\mathcal{O}_{top-k}$ ($k = \min(5, |\mathcal{C}|)$). If the ground-truth answer is absent from the top-k, it replaces the least frequent option. Options are then randomly shuffled and formatted as a standard MCQ (A/B/C/...).

Key design: **easy samples are filtered out** — if all $K$ samples yield only the correct class, the sample is excluded, focusing training on hard-negative cases.

### Stage 2: GRPO-based Reinforcement Learning

The constructed MCQ dataset $\mathcal{D} = \{(I, q, \mathcal{O}_{enum}, \hat{a})\}$ is used for training. For each sample, $N$ rollouts are generated and group advantages are computed:

$$A_i = \frac{r_i - \text{mean}\{r_1, \dots, r_N\}}{\text{std}\{r_1, \dots, r_N\} + \delta}$$

The reward consists of two components:
- **MCQ reward** $r_{mcq}$: 1.0 for selecting the correct option, 0.0 otherwise (easily verifiable).
- **Format reward** $r_{format}$: encourages correct use of `<think>` and `<answer>` tags.

The total reward is $r = \lambda_f r_{format} + \lambda_m r_{mcq}$, optimized via the standard GRPO loss.

### Inference

Inference also proceeds in two steps: (1) the trained model performs $K$ sampling passes to generate top-k candidates; (2) these candidates are composed into an MCQ prompt for the model to select from. Ground-truth labels are not artificially inserted during inference.

## Key Experimental Results

### Experimental Setup
- Base model: Qwen2.5-VL-7B-Instruct
- Datasets: OxfordFlowers-102, CUB-200, OxfordPets-37, StanfordCars-196, FGVC Aircraft-100
- Training: 3× A6000 GPUs, batch size 6, 4 GRPO rollouts
- Metrics: Base/Novel accuracy and Harmonic Mean (HM)

### Base-to-Novel Generalization (Table 1)

| Method | Base | Novel | HM |
|--------|------|-------|----|
| QWEN2.5-VL-7B | 68.9 | 70.5 | 69.7 |
| ViRFT | 73.0 | 74.2 | 73.6 |
| **DiVE-k** | **80.8** | **78.8** | **79.8** |

DiVE-k outperforms ViRFT by **+6.2%** in HM and QWEN2.5-VL-7B by **+10.1%**. The most significant gain is on CUB (HM +14.9). Performance approaches Gemini2.5-flash-lite (80.0 HM) and is slightly below GPT-5-mini (83.4 HM).

### Mixed-Dataset Generalization (Table 2)
Training a single model on base classes across all datasets, DiVE-k achieves HM 78.7, surpassing ViRFT by +4.0 and QWEN2.5 by +9.0. When ViRFT adopts the two-step inference pipeline, its novel accuracy actually drops below the original QWEN2.5 (76.2 vs. 77.0), whereas DiVE-k reaches 78.8.

### Few-shot (4-shot, Table 3)
DiVE-k achieves an average accuracy of 74.75%, outperforming ViRFT by +7.73% and QWEN2.5 by +10.85%.

## Highlights & Insights

1. **Elegant training signal design**: Using the model's own top-k predictions as MCQ options provides meaningful hard-negatives derived from the model's own confusion distribution, while enabling easily verifiable rewards that avoid the brittleness of string matching.
2. **Emergent differential reasoning**: MCQ-based training leads the model to compare fine-grained attribute differences among candidate classes (e.g., floral inflorescence morphology, bird beak shape) rather than relying solely on coarse features.
3. **Resolves ViRFT's memorization issue**: ViRFT shows virtually no improvement on novel classes under two-step inference (77.0→77.1), while DiVE-k yields substantive gains on novel classes (+1.8), demonstrating that the model learns generalizable reasoning rather than class name memorization.
4. **Clear ablation design**: Detailed ablations cover option generation strategies (random vs. text-embedding vs. top-k), the roles of visual and textual components, and the effect of $K$, providing thorough justification.

## Limitations & Future Work

1. **Minor regression on OxfordPets**: DiVE-k slightly underperforms ViRFT on the Pets dataset (HM 91.6 vs. 92.9), likely because more options increase the error probability in the second step. For already simple datasets, the MCQ format introduces unnecessary distraction.
2. **Two-step inference increases computational overhead**: Inference requires $K$ sampling passes followed by a selection step, roughly increasing computation by $K+1$ times compared to single-pass inference, which may be a deployment bottleneck.
3. **Validated only on Qwen2.5-VL-7B**: Although the appendix reports experiments with Gemma-3, validation on larger-scale models (e.g., 72B) is lacking; whether the method continues to yield gains on stronger base models remains to be confirmed.
4. **Fixed option count of 5**: $m=5$ is used as a fixed hyperparameter, whereas different datasets may have different optimal values (e.g., $K=2$ is optimal on Pets); adaptive option count selection could potentially improve performance further.
5. **Inconsistency between offline option generation and online training**: As training progresses, the model distribution shifts, yet the top-k options are generated offline using the initial model, potentially introducing distribution mismatch.

## Related Work & Insights

| Method | Mechanism | Reward Signal | Novel Generalization |
|--------|-----------|---------------|----------------------|
| CLIP | Visual-text embedding matching | None (zero-shot) | Limited |
| Prompt Learning | Learning text prompt vectors | Cross-entropy | Moderate |
| FuDD | LLM-generated offline discriminative features with VLM | No RL | Fixed strategy |
| ViRFT | Open-ended generation + exact string matching reward | Brittle | Weak (memorization) |
| SFT (Full) | Supervised fine-tuning | Cross-entropy | Very poor (Novel −33.5%) |
| **DiVE-k** | Top-k MCQ + GRPO | Easily verifiable | **Strong** |

The key distinction of DiVE-k lies in reformulating the open-ended classification problem as a closed-set selection problem over model-generated candidates, while leveraging the MCQ format to simplify the reward signal from "string matching" to "option index matching," fundamentally addressing reward brittleness and memorization.

**Broader implications:**
1. **Top-k self-refinement paradigm**: Using a model's own output distribution as a training signal is generalizable to other classification tasks (medical imaging, remote sensing, etc.) and potentially to multi-choice VQA training.
2. **MCQ as a universal reasoning interface**: Reformulating open-ended generation as MCQ is broadly applicable — the MCQ format inherently provides verifiable rewards, reducing RL training difficulty.
3. **Connection to Best-of-N sampling**: DiVE-k's inference pipeline resembles Best-of-N but goes further — rather than simple majority voting, it presents the candidate set to the model for explicit attribute-level comparative reasoning.
4. **Automated hard-negative mining**: Hard negatives are automatically obtained from the model's own sampling without external annotations or feature retrieval, making the approach concise and elegant.

## Rating
- Novelty: ⭐⭐⭐⭐ — The combination of top-k self-generated MCQ and RL is novel and addresses key deficiencies of ViRFT.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 5 datasets × 3 settings with detailed ablations, though validation across model scales is insufficient.
- Writing Quality: ⭐⭐⭐⭐ — Motivation is clear, figures are intuitive, and comparative experiments are well-designed.
- Value: ⭐⭐⭐⭐ — Proposes a practical RL training paradigm for fine-grained recognition with well-validated generalization ability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning](rewardmap_tackling_sparse_rewards_in_fine-grained_visual_reasoning_via_multi-sta.md)
- [\[CVPR 2026\] Specificity-aware Reinforcement Learning for Fine-grained Open-world Classification](../../CVPR2026/reinforcement_learning/specificity-aware_reinforcement_learning_for_fine-grained_open-world_classificat.md)
- [\[ICLR 2026\] PreferThinker: Reasoning-based Personalized Image Preference Assessment](preferthinker_reasoning-based_personalized_image_preference_assessment.md)
- [\[ICLR 2026\] Reasoning as Representation: Rethinking Visual Reinforcement Learning in Image Quality Assessment](reasoning_as_representation_rethinking_visual_reinforcement_learning_in_image_qu.md)
- [\[NeurIPS 2025\] NoisyRollout: Reinforcing Visual Reasoning with Data Augmentation](../../NeurIPS2025/reinforcement_learning/noisyrollout_reinforcing_visual_reasoning_with_data_augmenta.md)

</div>

<!-- RELATED:END -->
