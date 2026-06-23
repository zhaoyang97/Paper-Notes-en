---
title: >-
  [Paper Note] Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead
description: >-
  [ICLR 2026][Alignment & RLHF][RLVR] This paper demonstrates, through experiments involving over 100 models and 1 million GPU hours, that the common assumption in reasoning LLM post-training—"higher SFT scores lead to better RL performance"—is a widespread fallacy. It proposes **Validation Set Generalization Loss** and **Pass@large k** as reliable indicat
tags:
  - ICLR 2026
  - Alignment & RLHF
  - RLVR
  - GRPO
  - Pass@k
date: 2026-05-08
content_hash: adc678c01bb51a26
---
# Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=uLM3BfKo19](https://openreview.net/forum?id=uLM3BfKo19)  
**Code**: https://github.com/feiyang-k/SFT-RL-ICLR  
**Area**: Alignment RLHF / LLM Reasoning / Post-training Analysis  
**Keywords**: SFT-RL Post-training, RLVR, GRPO, Generalization Loss, Pass@k

## TL;DR
This paper demonstrates, through experiments involving over 100 models and 1 million GPU hours, that the common assumption in reasoning LLM post-training—"higher SFT scores lead to better RL performance"—is a widespread fallacy. It proposes **Validation Set Generalization Loss** and **Pass@large k** as reliable indicators to predict final RL performance, improving prediction accuracy ($R^2$, Spearman rank correlation) by up to 0.5 (approximately 2x) compared to using SFT scores directly.

## Background & Motivation
**Background**: Since DeepSeek-R1, post-training for reasoning LLMs has generally followed a two-stage pipeline: SFT for cold-starting, followed by RLVR (Reinforcement Learning with Verifiable Rewards, typically GRPO). In industry, these stages are often managed by different teams; SFT teams maximize "post-SFT accuracy on benchmarks" and pass the highest-scoring model to the RL team, assuming it will perform best after RL.

**Limitations of Prior Work**: The assumption that "highest SFT score $\rightarrow$ strongest post-RL performance" frequently fails. The authors provide numerous counterexamples: repeatedly training on the same data for more epochs consistently raises SFT scores, but over-training compresses the headroom for RL improvement, sometimes resulting in worse post-RL outcomes. Training only on "shortest/simplest" samples yields faster SFT gains, but these models learn almost nothing during RL. In extreme cases, applying RL to an SFT-enhanced model yields lower final performance than applying RL directly to the base model.

**Key Challenge**: A systemic misalignment exists between the optimization objective of the SFT phase (post-SFT accuracy) and the true objective of post-training (post-RL accuracy). SFT scores favor **simpler, more homogeneous, and repetitive** data—precisely what many SFT data selection methods prefer—meaning "optimizing SFT scores" can drive models toward configurations unfavorable for subsequent RL exploration. Furthermore, RL is extremely costly (e.g., 3 epochs taking up to 5 days) and lacks early stopping indicators, leading to inefficient iterations between post-training teams, characterized as the "quagmire" in the title.

**Goal**: This work focuses on "predictability"—judging the post-RL potential of a post-SFT model without actually running the expensive RL process. This is decomposed into two research questions: RQ1: Do models with better pre-RL performance always perform better after RLVR? If not, what are the failure modes? RQ2: Which SFT paradigms or data recipes are better for subsequent RL, and can the suitability of an SFT model be judged before running RL?

**Core Idea**: Instead of the misleading "post-SFT accuracy," the authors propose two proxy metrics that reflect "potential" rather than "current achievement": Validation Set Generalization Loss (to capture overfitting) and Pass@large k (to capture inherent problem-solving limits). These are used to predict and rank RL outcomes.

## Method

### Overall Architecture
This is an analysis and methodology paper that proposes a "diagnostic followed by metric replacement" predictive framework rather than a new training algorithm. The methodology consists of three steps: First, it systematically reveals the "SFT Metric Trap" across **Dataset-level** (same distribution, varying epochs/sample size/learning rate) and **Instance-level** (fixed config, varying data selection) scenarios to prove the divergence between post-SFT and post-RL performance. Second, it introduces two new predictive metrics—Validation Set Generalization Loss and Pass@large k—to replace post-SFT accuracy. Third, it provides a workflow: use generalization loss to filter poor candidates, use Pass@large k to rank the remainder, and optionally run RL on a few SFT models to calibrate a linear predictor for absolute value estimation. The value lies in identifying the best SFT models for RL without (or with minimal) RL execution.

### Key Designs

**1. SFT Metric Trap: Exposing the Fallacy via Dataset and Instance-level Counterexamples**

This diagnostic foundation targets the pain point of optimizing SFT and RL in isolation. The authors design two control scenarios: **Dataset-level**, where SFT samples come from the same distribution but training configurations (unique samples, epochs, learning rate) vary, mimicking industrial trade-offs between "more epochs on limited data" vs. "single epoch on full data." **Instance-level**, where the model and process are fixed but SFT datasets vary (e.g., shortest, longest, random subsets), mimicking "SFT data selection." At the dataset-level, a linear fit between post-SFT and post-RL performance yields $R^2 = 0.43$, showing that SFT performance explains only 43% of the post-RL variance. They observe that over-training (up to 8 epochs) raises SFT scores while post-RL performance saturates or even degrades (Qwen3) after 2 epochs. At the instance-level, training on only the shortest samples yields the fastest SFT gains, but because these are too close to the model's original output, post-RL performance is significantly lower. This sections demonstrates that post-SFT performance is systematically biased by homogeneous/simple data.

**2. Generalization Loss: Capturing Over-training via Validation Loss "Flare-up"**

To address the "over-training harms RL" failure mode in dataset-level scenarios, the authors use generalization loss on a held-out validation set during SFT as a potential indicator. The intuition is that as epochs increase, post-SFT accuracy may continue to rise while validation generalization loss eventually "flares up," signaling strong overfitting. This loss increase correlates strongly with reduced gains in subsequent RL. Practitially, models with "low accuracy and high generalization loss" can be immediately discarded. A noted limitation is that this is only applicable to dataset-level (same distribution) selection; when comparing different datasets, the validation loss is confounded by distribution shifts.

**3. Pass@large k: Predicting RL Gains via Inherent Problem-Solving Capacity**

For instance-level (cross-dataset) selection, the authors use Pass@large k. This is based on the insight that RLVR (like GRPO) aims to maximize expected reward, effectively "compressing" Pass@k capability into Pass@1 performance. Since GRPO only progresses if at least one sample in a group is correct, RL dynamics are tightly coupled with the original model's Pass@k. Therefore, the Pass@k of a post-SFT model at a large $k$ better reflects its "ceiling" and how much can be extracted by RL. To estimate this efficiently, an unbiased estimator is used: for $n$ generated responses with $c$ correct ones:

$$ \text{Pass@}k = \mathbb{E}\left[1 - \frac{\binom{n-c}{k}}{\binom{n}{k}}\right] $$

This holds for all $k \le n$. The authors use the post-SFT model with the highest Pass@large k to predict the highest post-RL Pass@1 **without running RL**. Because this measures the intrinsic ability to generate correct solutions, it is robust to training data distribution shifts and serves as a replacement when generalization loss fails.

### Loss & Training
The paper introduces no new loss functions. SFT uses standard supervised fine-tuning. RL uses GRPO (Group Relative Policy Optimization) for RLVR, with binary verifiable rewards for math answers. For metrics: Generalization loss is the loss on the SFT validation set. Pass@large k uses the unbiased estimator above (experimentally $k=64$ with 256 samples per problem). For absolute value prediction, RL is run on a small subset of SFT models to collect calibration points for a linear predictor.

## Key Experimental Results

### Main Results
Scale: Training over 100 models (up to 12B), including Llama3-8B-Instruct, Mistral-Nemo-12B-Instruct, and Qwen3-4B-base. SFT data includes Llama-Nemotron-SFT / AceReasoner1.1-SFT, and RL data includes MATH(train) / DeepScaleR. Evaluation uses Pass@1 (average of 64) across 7 math benchmarks (MATH-500, AIME, GSM8k, etc.), totaling >1 million A100 GPU hours.

Dataset-level Prediction (Table 1/2, Metric Comparison):

| Model | Metric | post-SFT Pass@1 (Base) | Generalization Loss | Pass@large k (k=64) | Average of Both |
|------|------|------|------|------|------|
| Llama3-8B | Spearman | 0.75 | 0.94 | 0.95 | 0.97 (+0.22) |
| Mistral-NeMo-12B | Spearman | 0.78 | 0.90 | 0.92 (+0.14) | 0.90 |
| Llama3-8B | $R^2$ | 0.57±0.29 | 0.88±0.09 | 0.87±0.10 | 0.94±0.04 (+0.37) |
| Mistral-NeMo-12B | $R^2$ | 0.29±0.38 | 0.79±0.26 (+0.50) | 0.57±0.32 | 0.72±0.24 |

Instance-level Prediction (Table 3/4, Generalization Loss omitted due to cross-distribution inapplicability):

| Model | Metric | post-SFT Pass@1 (Base) | Pass@large k (k=64) |
|------|------|------|------|
| Llama3-8B | Spearman | 0.69 | 0.94 (+0.25) |
| Mistral-NeMo-12B | Spearman | 0.70 | 0.98 (+0.28) |
| Llama3-8B | $R^2$ | 0.58±0.20 | 0.92±0.05 (+0.34) |
| Mistral-NeMo-12B | $R^2$ | 0.73±0.16 | 0.98±0.01 (+0.25) |

### Ablation Study
Rather than traditional module ablation, the paper uses metric comparison to ablate the effectiveness of three predictive signals across scenarios:

| Configuration | Dataset-level | Instance-level | Description |
|------|---------|--------|------|
| post-SFT Pass@1 (Base) | $R^2$ as low as 0.29 | $R^2$≈0.58–0.73 | Direct SFT score prediction is high-variance and unreliable. |
| Generalization Loss | $R^2$ Gain up to +0.50 | N/A | Best for same-distribution accuracy; fails across datasets. |
| Pass@large k | Spearman ≥ 0.92 | $R^2$ up to 0.98 | Most stable ranking; robust to distribution shift. |

### Key Findings
- **Pass@large k is the most stable ranking tool**: Spearman $\ge 0.90$ (up to 0.98) in both scenarios, with a 59% $R^2$ improvement in instance-level tasks due to its focus on intrinsic capacity.
- **Generalization Loss has high accuracy but narrow boundaries**: Gains in $R^2$ reach +0.50 for Mistral (0.29 $\rightarrow$ 0.79) under same-distribution conditions, but it is explicitly unsuitable for instance-level selection.
- **"Half data x 2 epochs" often beats "Full data x 1 epoch"**: Given the same compute budget, the former is usually superior both post-SFT and post-RL. Furthermore, training only on short samples yields misleadingly high SFT scores but poor RL results.
- **Heterogeneous responses across models**: Mistral shows correlated gains between SFT and RL, while Qwen3 displays almost no correlation (post-RL performance is independent of SFT improvement), indicating that SFT scores cannot be extrapolated across different models.

## Highlights & Insights
- **Studying "Wrong Metrics" as a Systemic Failure**: The authors do not just fix a training trick; they identify that the industry's default proxy (post-SFT accuracy) is biased. This "critique of measurement" provides more leverage than hyperparameter tuning.
- **Theoretical Coupling of Pass@large k and GRPO**: The argument that "GRPO compresses Pass@k into Pass@1" explains why large-k performance predicts the RL ceiling. The use of an unbiased estimator ensures high efficiency.
- **Direct Engineering Value**: The SOP—"filter with generalization loss, rank with Pass@large k, calibrate with minimal RL"—is a plug-and-play methodology that significantly reduces costs in the post-training pipeline.
- **Honest Boundary Marking**: By explicitly stating that generalization loss cannot be used across distributions, the authors increase the credibility and practical utility of their findings.

## Limitations & Future Work
- The research is limited to **mathematical reasoning**; generalization to code, science, or agentic tasks remains unverified.
- Only mainstream **online RL + GRPO** was studied; relationships under offline RL, DPO, or other algorithms might differ.
- **Cost of Pass@large k evaluation**: Direct evaluation requires at least $k$ samples, which is expensive for long sequences. Extrapolating from small $k$ to large $k$ is a potential optimization.
- Contextual limitations: $R^2$/Spearman values are calculated within specific model+data combinations and are not directly comparable across task difficulties or budgets.

## Related Work & Insights
- **vs. "SFT Cold-start Necessity" (DeepSeek-R1)**: R1 argues SFT is necessary for RL; this paper does not negate SFT but shows that "maximizing SFT scores" is not necessarily the best starting point for RL.
- **vs. "Over-SFT constrains RL" (Llama-4) / "Pure RL is better" (Chen et al. 2025a)**: This paper unifies these observation via a predictability framework, explaining scattered phenomena (over-training or simple data harming RL) as manifestations of "metric bias."
- **vs. Difficulty/Influence based SFT Selection (Muennighoff et al. 2025)**: While those methods optimize for post-SFT performance, this work shows such objectives favor simple/homogeneous data and argues for switching to generalization loss or Pass@large k as selection signals.

## Rating
- Novelty: ⭐⭐⭐⭐ (Not a new algorithm, but the "metric critique" perspective is rare and rigorous)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (100+ models, 3 backbones, 1M+ GPU hours)
- Writing Quality: ⭐⭐⭐⭐ (Problem-driven, clear counterexamples)
- Value: ⭐⭐⭐⭐⭐ (Directly addresses the industrial "RL selection" pain point)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SFT Overtraining Predicts Rank Inversion via Entropy Collapse Under RLVR](../../ICML2026/llm_alignment/sft_overtraining_predicts_rank_inversion_via_entropy_collapse_under_rlvr.md)
- [\[ICLR 2026\] Spectrum Tuning: Post-Training for Distributional Coverage and In-Context Steerability](spectrum_tuning_post-training_for_distributional_coverage_and_in-context_steerab.md)
- [\[ICLR 2026\] Fluent Alignment with Disfluent Judges: Post-training for Lower-Resource Languages](fluent_alignment_with_disfluent_judges_post-training_for_lower-resource_language.md)
- [\[ICLR 2026\] Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)
- [\[CVPR 2025\] Continual SFT Matches Multimodal RLHF with Negative Supervision](../../CVPR2025/llm_alignment/continual_sft_matches_multimodal_rlhf_with_negative_supervision.md)

</div>

<!-- RELATED:END -->
