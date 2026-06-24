---
title: >-
  [Paper Note] EvoLM: In Search of Lost Language Model Training Dynamics
description: >-
  [NeurIPS 2025 Oral][Reinforcement Learning][training dynamics] Systematically trains 100+ 1B/4B parameter LMs from scratch to transparently investigate training dynamics across pre-training, continued pre-training (CPT), SFT, and RL stages, revealing diminishing returns from overtraining, strategies to alleviate catastrophic forgetting, and complex trade-offs in SFT/RL configurations.
tags:
  - "NeurIPS 2025 Oral"
  - "Reinforcement Learning"
  - "training dynamics"
  - "scaling law"
  - "continued pre-training"
  - "reinforcement-learning"
  - "SFT"
date: 2026-05-08
content_hash: 1b7ce154bda0b148
---

# EvoLM: In Search of Lost Language Model Training Dynamics

**Conference**: NeurIPS 2025 Oral  
**arXiv**: [2506.16029](https://arxiv.org/abs/2506.16029)  
**Code**: Yes (models, data, training, and evaluation pipelines are fully open-sourced)  
**Area**: Reinforcement Learning  
**Keywords**: training dynamics, scaling law, continued pre-training, reinforcement-learning, SFT

## TL;DR

Systematically trains 100+ 1B/4B parameter LMs from scratch to transparently investigate training dynamics across pre-training, continued pre-training (CPT), SFT, and RL stages, revealing diminishing returns from overtraining, strategies to alleviate catastrophic forgetting, and complex trade-offs in SFT/RL configurations.

## Background & Motivation

Modern language model training is split into multiple stages (pre-training, continued pre-training, SFT, RL), yet it is highly challenging for downstream practitioners to evaluate the impact of design choices in each stage. Existing literature suffers from several key limitations:

**Opaque analysis**: Many post-training studies use off-the-shelf base models, without strictly controlling key variables such as model size and data volume.

**Reliability of intermediate checkpoints**: Evaluating intermediate checkpoints underestimates the model's true capability because the learning rate has not fully decayed.

**Unclear interactions between stages**: How does the amount of pre-training affect RL performance? How should data allocation be configured between SFT and RL? These questions lack systematic research.

This work eliminates these confounding factors by training over 100 models from scratch, with each model completing a full learning rate schedule.

## Method

### Overall Architecture

Four-stage pipeline:
1. **Pre-training**: Trained on FineWeb-Edu with token budgets ranging from Chinchilla-optimal ($20 \times$ model parameters) up to 320B tokens.
2. **Continued Pre-training (CPT)**: Continued training on FineMath for 2B–42B tokens, employing a data replay strategy to mitigate forgetting.
3. **SFT**: Fine-tuned on augmented GSM8K/MATH datasets, utilizing model consistency to filter low-quality samples.
4. **RL**: Conducted using PPO + binary verifiable rewards, with non-overlapping data relative to SFT.

Models are initialized with the LLaMA-2 architecture in 1B and 4B parameter scales. All configurations use a full learning rate schedule, with only the final checkpoints evaluated.

### Key Designs

The dual-dimensional design of the **evaluation protocol** is a key highlight of this work:
- **Upstream tasks**: 0-shot accuracy on HellaSwag, Winogrande, PIQA, etc. $\rightarrow$ measures language modeling capability.
- **Downstream tasks**: GSM8K-Platinum, MATH (In-Domain, ID) + CRUXEval, BGQA, TabMWP, StrategyQA (Out-of-Domain, OOD) $\rightarrow$ measures reasoning capability.
- **Four sampling strategies**: Pass@1 (greedy), Maj@16 (majority vote), RM@16 (best-of-16 by ORM scoring), and Pass@16 (any-correct).

**Data replay strategy**: Injecting a small amount of pre-training data (FineWeb) during CPT; the optimal ratio is approximately 5% (8B replay + 42B domain-specific data).

### Loss & Training

- Pre-training and CPT utilize the standard next-token prediction loss.
- SFT utilizes the standard cross-entropy loss.
- RL utilizes the PPO algorithm + binary verifiable rewards (1 for correct answers, 0 for incorrect).
- All models complete a full learning rate decay.

## Key Experimental Results

### Main Results: Influence of Pre-training Scale

| Model | ID Maj@16 (SFT) | ID Maj@16 (SFT+RL) | OOD Maj@16 (SFT) | OOD Maj@16 (SFT+RL) |
|------|----------------|---------------------|-------------------|----------------------|
| 1B-20B | ~8% | — | — | — |
| 1B-80B | ~15% | 21.4% | 24.6% | 31.0% |
| 1B-160B | 14.2% | 22.5% | 25.6% | 31.6% |
| 1B-320B | 16.1% | 25.0% | 24.8% | 29.9% |
| 4B-160B | 26.4% | 34.8% | 26.0% | 33.2% |

### Interaction between Model Scale and Pre-training Budget

| Comparison | ID Greedy (SFT/SFT+RL) | ID Pass@16 (SFT/SFT+RL) |
|------|------------------------|--------------------------|
| 1B-320B (same compute) | 14.1/20.1 | 36.0/49.0 |
| 4B-80B (same compute) | 11.3/15.7 | 34.2/43.0 |
| 1B-160B (same tokens) | 12.8/17.5 | 34.5/45.1 |
| 4B-160B (same tokens) | 22.0/27.8 | 47.6/58.4 |

### Ablation Study

**CPT Data Replay Ratio** (1B-160B base, 50B tokens total CPT):

| Configuration | GSM8K-Platinum Pass@1 |
|------|-----------------------|
| No CPT | 6.04% |
| FineMath 50B (no replay) | 19.27% |
| FineWeb 1.6B + FineMath 48.4B | 16.21% |
| **FineWeb 8B + FineMath 42B** | **21.01%** |
| FineWeb 16B + FineMath 34B | 15.22% |

**SFT/RL Data Allocation** (fixed at 100K samples, 4 epochs):
- Allocating more to SFT $\rightarrow$ Maximizes in-domain performance (ID Greedy saturates at 70K SFT).
- Allocating more to RL $\rightarrow$ Enhances out-of-domain generalization (OOD performance peaks at 10K SFT / 90K RL).

### Key Findings

**Takeaway 1**: Overtraining in pre-training does not always improve downstream performance and may even lead to degradation (saturating at $80 \times - 160 \times$ model parameters).

**Takeaway 3**: CPT causes catastrophic forgetting, which can be effectively mitigated by a 5% data replay.

**Takeaway 4-5**: Sufficient domain-specific CPT is a prerequisite for successful post-training; without CPT, RL may even degrade performance.

**Takeaway 7-8**: Excessive SFT improves in-domain but hurts out-of-domain performance, while limiting subsequent gains from RL.

**Takeaway 10**: RL primarily increases the sampling probability of already correct outputs, rather than genuinely improving reasoning capability (Correct Ratio rises but Pass@16 decreases).

**Takeaway 12**: ORM score is a reliable unsupervised surrogate metric, with the scores of an 8B reward model showing a 0.62–0.84 Pearson correlation with the 1B model's accuracy.

## Highlights & Insights

- **End-to-End Transparency**: Over 100 models were trained completely from scratch with a full decay cycle, eliminating the confounding factor of intermediate checkpoints (empirically supported by Table 3, which shows that intermediate checkpoints significantly underestimate true performance).
- **Insight into the Essence of RL**: RL does not "teach the model new capabilities" but rather "amplifies the probability of already correct behaviors"—evidenced by a decrease in Pass@16 paired with an increase in Correct Ratio.
- **Unlocking Conditions for Model Scale**: Smaller models can surprisingly outperform larger ones when pre-training is insufficient; the advantages of model scale only manifest after reaching the saturated pre-training phase.
- **Practical Value of ORM Score as a Surrogate Metric**: Particularly valuable in tasks where annotation is highly challenging.

## Limitations & Future Work

- Model scale is limited to 4B; whether these trends generalize to larger models remains unverified.
- Focuses exclusively on post-training for reasoning tasks; objectives like safety alignment, instruction following, and coding are unexplored.
- RL is restricted to PPO + verifiable rewards; alternative methods such as GRPO and DPO are not examined.
- It remains uncertain whether findings in the mathematics domain transfer to other specialized fields (e.g., law, medicine).

## Related Work & Insights

- **Chinchilla scaling law**: This work builds upon it to investigate downstream performance in the "overtraining" regime.
- **Springer et al. (overtrained)**: Discovered that overtraining in pre-training degrades SFT performance; this study extends the scope to RL and validates the findings on generative reasoning tasks.
- **Yue et al.**: Parallel findings suggesting that RL primarily increases confidence rather than reasoning capacity; this work provides a fine-grained trade-off across both epochs and data volume.
- **Zhao et al. (Echo)**: Concluded that RL amplifies pre-trained patterns rather than creating new ones; this work provides complementary evidence from the perspective of training dynamics.

## Rating

- **Novelty**: 7/10 — Strong systemization, but individual findings are mostly quantitative validations of already known trends.
- **Experimental Thoroughness**: 10/10 — The systematic controlled experiments over 100+ models set a benchmark.
- **Value**: 9/10 — The 12 takeaways offer direct, actionable guidance for LM training practices.
- **Writing Quality**: 9/10 — Well-structured with rich illustrations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Training Language Models to Reason Efficiently](training_language_models_to_reason_efficiently.md)
- [\[ICLR 2026\] SSVPO: Toward Effective Step-level Credit Assignment for Language Model RL Training](../../ICLR2026/reinforcement_learning/ssvpo_effective_step-level_credit_assignment_for_rl_training_of_language_models.md)
- [\[NeurIPS 2025\] RePIC: Reinforced Post-Training for Personalizing Multi-Modal Language Models](repic_reinforced_post-training_for_personalizing_multi-modal_language_models.md)
- [\[ICML 2026\] The Surprising Difficulty of Search in Model-Based Reinforcement Learning](../../ICML2026/reinforcement_learning/the_surprising_difficulty_of_search_in_model-based_reinforcement_learning.md)
- [\[ICLR 2026\] MOBODY: Model-Based Off-Dynamics Offline Reinforcement Learning](../../ICLR2026/reinforcement_learning/mobody_model-based_off-dynamics_offline_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
