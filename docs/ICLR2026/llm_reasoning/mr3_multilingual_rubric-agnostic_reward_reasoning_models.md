---
title: >-
  [Paper Note] mR3: Multilingual Rubric-Agnostic Reward Reasoning Models
description: >-
  [ICLR 2026][LLM Reasoning][multilingual reward models] This paper introduces mR3, a family of multilingual rubric-agnostic reward reasoning models covering 72 languages. Through systematic data construction (GPT-OSS-120B distillation with difficulty filtering) and curriculum learning, the 14B model surpasses the 120B teacher model and all comparable baselines on multilingual evaluation benchmarks, while supporting point-wise, pair-wise, and binary evaluation paradigms.
tags:
  - ICLR 2026
  - LLM Reasoning
  - multilingual reward models
  - reasoning-based evaluation
  - curriculum learning
  - rubric-based assessment
  - knowledge distillation
date: 2026-05-08
content_hash: 932786bc15055935
---

# mR3: Multilingual Rubric-Agnostic Reward Reasoning Models

**Conference**: ICLR 2026
**arXiv**: [2510.01146](https://arxiv.org/abs/2510.01146)
**Code**: [github.com/rubricreward/mr3](https://github.com/rubricreward/mr3)
**Area**: LLM Reasoning / Alignment & RLHF
**Keywords**: multilingual reward models, reasoning-based evaluation, curriculum learning, rubric-based assessment, knowledge distillation

## TL;DR
This paper introduces mR3, a family of multilingual rubric-agnostic reward reasoning models covering 72 languages. Through systematic data construction (GPT-OSS-120B distillation with difficulty filtering) and curriculum learning, the 14B model surpasses the 120B teacher model and all comparable baselines on multilingual evaluation benchmarks, while supporting point-wise, pair-wise, and binary evaluation paradigms.

## Background & Motivation

**Background**: The LLM-as-judge evaluation paradigm has been widely adopted in English settings, but support for non-English languages remains extremely limited. Existing reward models (e.g., ArmoRM, RM-R1) focus almost exclusively on English, while multilingual evaluation models (e.g., m-Prometheus) cover only 6 languages and lack systematic study of training strategies.

**Limitations of Prior Work**:
   - Existing reward models exhibit significant accuracy degradation in non-English settings
   - LLMs lack coherent reasoning capability in low-resource languages (LRLs)
   - Multilingual evaluation lacks a standardized framework; existing work supports only pair-wise comparison, without point-wise or binary evaluation
   - There is no systematic investigation into how to construct high-quality training data for multilingual reward models, including what languages to use for instructions, rubrics, and reasoning chains

**Key Challenge**: Multilingual evaluation requires both strong reasoning ability and cross-lingual knowledge transfer, yet existing models' reasoning capabilities degrade substantially in non-English languages. How to simultaneously improve both under limited multilingual data conditions remains an open challenge.

**Goal**:
   - Design a multilingual reward reasoning model covering 72 languages
   - Systematically study optimal combinations of instruction language, reasoning language, and target language
   - Explore data selection and curriculum learning strategies
   - Support point-wise, pair-wise, and binary evaluation paradigms

**Key Insight**: Rather than training conventional scalar reward models, this work trains generative reward models that produce reasoning traces alongside scores, improving evaluation interpretability and cross-lingual robustness through explicit reasoning.

**Core Idea**: Construct a 72-language alignment dataset (100K samples) via GPT-OSS-120B distillation combined with difficulty filtering and curriculum learning, training generative reward reasoning models that outperform the teacher model despite having far fewer parameters.

## Method

### Overall Architecture
Input: task instruction $t$ + input instance $i$ + candidate response $a$ + evaluation rubric $r$
Output: reasoning trace + brief explanation $e$ + score $s$
Formally, $f(x) = y$, where $x = (t, i, a, r)$ and $y = (\text{trace}, e, s)$

Three evaluation modes: point-wise (scoring a single response), pair-wise (comparing two responses), and binary (correct/incorrect judgment).

### Key Designs

1. **Multilingual Data Construction Pipeline**

    - **Function**: Filter and construct a 100K high-quality multilingual training set from over 3 million samples
    - **Mechanism**:
        - The initial data pool is drawn from 6 public datasets (Human Arena Preference, HelpSteer3, MMMLU, HumanEval-XL, MATH-500 Multilingual, PolyGuardMix), covering 125 languages
        - For data lacking rubrics, English rubrics are automatically generated using GPT-4.1
        - GPT-OSS-120B is used to distill outputs under three language strategies: eng-eng (English instruction + English reasoning), tgt-eng (target-language instruction + English reasoning), and tgt-tgt (target-language instruction + target-language reasoning)
        - **Quality filtering**: Only samples correctly answered under all three strategies are retained
        - **Difficulty filtering**: Samples that gpt-oss-20b answers correctly in 5 consecutive attempts are discarded as "easy"
        - The dataset is downsampled to 100K, prioritizing harder samples

2. **Curriculum Learning Strategy**

    - **Function**: Optimize the ordering of training data
    - **Mechanism**: Random shuffling, English-first, difficulty ordering, and mixed schemes are compared; ordering from **easy to hard** yields the best results (difficulty is measured by prediction consistency and token length)
    - **Design Motivation**: Easy samples first establish foundational capabilities; hard samples fine-tune later, avoiding disruption by noisy samples in early training

3. **Multilingual Reasoning Strategy Study**

    - **Function**: Systematically compare the effectiveness of eng-eng, tgt-eng, and tgt-tgt reasoning paths
    - **Key Findings**:
        - eng-eng achieves the highest overall performance (most mature English reasoning capability)
        - tgt-eng follows closely; larger models are more robust to non-English prompts
        - tgt-tgt is weakest before fine-tuning but shows the **largest gains** after fine-tuning, even surpassing the base model's eng-eng performance
    - **Design Motivation**: Target-language reasoning is critical for interpretability and accessibility in low-resource language settings

4. **Training Objective: SFT over RL**

    - **Function**: Standard cross-entropy training to maximize the log-likelihood of target tokens
    - **Core Formula**: $\mathcal{L}_{\text{SFT}}(\theta) = -\frac{1}{N}\sum_{i=1}^{N}\sum_{t=1}^{T_i}\log \pi_\theta(y_t^{(i)} | y_{<t}^{(i)}, x^{(i)})$
    - **Design Motivation**: Experiments show that RL-based methods (e.g., RLVR) are less effective than SFT in this setting

### Loss & Training
- SFT cross-entropy loss, based on the Qwen3 model family (4B/8B/14B)
- Curriculum learning: training data ordered from easy to hard by difficulty
- Multilingual alignment: each sample is aligned across all three language strategies

## Key Experimental Results

### Main Results (Pairwise Evaluation Benchmarks, eng-eng Setting)

| Model | m-RewardBench (23 lang) | RewardBench (1 lang) | MM-Eval (18 lang) | IndoPref (1 lang) |
|------|----------------------|--------------------|-----------------|-----------------| 
| GPT-OSS-120B | 89.05 | 90.30 | 85.01 | 72.15 |
| Nemotron-Multi-49B | 89.03 | 89.62 | 76.27 | 68.40 |
| R3-Qwen3-14B-LoRA | 88.07 | **91.00** | 84.04 | 72.65 |
| **mR3-Qwen3-14B** | **89.18** | 90.79 | **86.05** | **74.14** |
| **mR3-Qwen3-8B** | 88.44 | 90.50 | 84.84 | 72.86 |
| **mR3-Qwen3-4B** | 87.61 | 89.74 | 82.62 | 72.22 |

mR3-Qwen3-14B surpasses the 120B teacher model with only 14B parameters (+0.13 on m-RB, +1.04 on MM-Eval, +1.99 on IndoPref), while being 3.5× faster than the 49B Nemotron model.

### Ablation Study

| Configuration | Key Finding |
|------|---------|
| Curriculum: easy→hard vs. random | Easy→hard achieves the best results on HelpSteer3 validation set |
| Data scale: 50K vs. 100K vs. 200K | 100K is the sweet spot; 200K yields no significant improvement |
| Language strategy: eng-eng vs. tgt-tgt | eng-eng achieves higher absolute scores, but tgt-tgt shows the largest gains after fine-tuning |
| Difficulty filtering: with vs. without | Removing easy samples significantly improves model performance |
| Training method: SFT vs. RLVR | SFT consistently outperforms RL-based methods on this task |

### Key Findings
- **Small model, large impact**: The 14B model systematically outperforms the 120B teacher and the 49B competitor, demonstrating that high-quality data and correct training strategies matter more than scale
- **Step-change improvement of tgt-tgt**: The base model's target-language reasoning is the weakest, yet it shows the largest gains after fine-tuning, even surpassing the base model's eng-eng performance — indicating that multilingual training effectively "activates" cross-lingual reasoning capabilities
- **Downstream DPO validation**: Using mR3-Qwen3-14B as the reward model for DPO on Qwen3-30B-A3B improves the English win rate on m-ArenaHard-v2.0 from 49.1% to 57.3%
- **Human evaluation**: 20 native speakers evaluated across 12 languages; mR3's reasoning traces substantially outperform the Qwen3 baseline on factuality (2.78 vs. 2.06) and logical coherence (2.67 vs. 2.05)

## Highlights & Insights
- The **unified 72-language training framework** represents a major advance in multilingual reward modeling, far exceeding the prior best of 6 languages in m-Prometheus. The three-strategy alignment data design (eng-eng/tgt-eng/tgt-tgt) is particularly elegant, enabling controlled research while covering realistic usage scenarios
- **Easy-to-hard curriculum learning is effective for reward model training**: This finding is directly transferable to the training of other generative evaluation models
- **Data quality > data scale**: A 14B model trained on 100K curated samples outperforms models trained on 3M+ samples, underscoring the importance of multi-stage filtering (three-strategy consistency + difficulty filtering)
- **Interpretability value of target-language reasoning**: Although English reasoning achieves higher accuracy, target-language reasoning is critical for accessibility and user trust in low-resource language settings, and fine-tuning effectively closes the gap

## Limitations & Future Work
- The distillation outputs from GPT-OSS-120B carry inherent language bias (strongest in English), which propagates to mR3
- Coverage of low-resource languages among the 72 languages may be uneven, as the source datasets are biased toward high- and medium-resource languages
- Training relies solely on SFT; the potential of RL-based post-training (e.g., GRPO) is not fully explored
- Human evaluation covers only 12 languages (already more than comparable work), falling short of all 72 training languages
- **Future directions**: Specialized data augmentation for low-resource languages (e.g., high-resource → low-resource translation with back-translation), and exploring whether online RL fine-tuning can yield further improvements

## Related Work & Insights
- **vs. R3 (Anugraha et al., 2025)**: R3 is the English-only predecessor of mR3, trained exclusively on English data. mR3 inherits its rubric-agnostic framework and extends it to 72 languages, substantially outperforming R3 on multilingual benchmarks (m-RewardBench: 89.18 vs. 88.07), while R3 retains a slight edge on the English-only RewardBench (91.00 vs. 90.79)
- **vs. m-Prometheus (Pombal et al., 2025)**: Covers only 6 languages with 480K training samples; m-RewardBench score of 79.51 vs. mR3's 89.18, a substantial margin
- **vs. Nemotron-Multilingual-49B (Wang et al., 2025)**: With 49B parameters, it supports pair-wise evaluation in only 13 languages; mR3-14B comprehensively surpasses it with 1/3.5 the parameters and 7.2× the language coverage

## Rating
- Novelty: ⭐⭐⭐⭐ The unified 72-language framework and three-strategy alignment data construction are novel, though the model architecture and training method (SFT) are relatively standard
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 7 benchmarks, multiple ablations, curriculum learning comparisons, DPO downstream validation, and a 20-annotator 12-language human evaluation — exceptionally comprehensive
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rich tables and figures, though the paper is lengthy (extensive appendix) and core contributions require extraction from a large volume of experiments
- Value: ⭐⭐⭐⭐⭐ Addresses a critical gap in multilingual reward modeling with direct practical impact on non-English LLM alignment

## Key Experimental Results

| Model | mR3-RewardBench | Size |
|------|----------------|------|
| GPT-OSS-120B | ~88% | 120B |
| **mR3-Qwen-14B** | **88.46%** | **14B (9× smaller)** |

Human evaluators (20 annotators, 12 languages) prefer the reasoning quality of mR3.

## Rating
- Novelty: ⭐⭐⭐⭐ First large-scale multilingual reward reasoning model
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 72 languages + human evaluation
- Value: ⭐⭐⭐⭐⭐ Foundational infrastructure for multilingual LLM alignment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners](../../ACL2026/llm_reasoning/large_reasoning_models_are_not_yet_multilingual_latent_reasoners.md)
- [\[ICLR 2026\] DRPO: Efficient Reasoning via Decoupled Reward Policy Optimization](drpo_efficient_reasoning_via_decoupled_reward_policy_optimization.md)
- [\[ICLR 2026\] Is It Thinking or Cheating? Detecting Implicit Reward Hacking by Measuring Reasoning Effort](is_it_thinking_or_cheating_detecting_implicit_reward_hacking_by_measuring_reason.md)
- [\[ICLR 2026\] Why is Your Language Model a Poor Implicit Reward Model?](why_is_your_language_model_a_poor_implicit_reward_model.md)
- [\[ICLR 2026\] Fixing the Broken Compass: Diagnosing and Improving Inference-Time Reward Modeling](fixing_the_broken_compass_diagnosing_and_improving_inference-time_reward_modelin.md)

</div>

<!-- RELATED:END -->
