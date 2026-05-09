---
title: >-
  [Paper Note] Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation
description: >-
  [ACL 2026][Model Compression][Knowledge Distillation] This paper proposes PerSyn (Personalized data Synthesis), which adopts a "Route then Generate" paradigm where a router assigns the optimal teacher model to each prompt by jointly considering student learnability and teacher response quality. Compared to the conventional "Generate then Select" paradigm, PerSyn is more efficient and effective, consistently outperforming all baselines across instruction tuning and mathematical reasoning tasks.
tags:
  - ACL 2026
  - Model Compression
  - Knowledge Distillation
  - Synthetic Data
  - Multi-Teacher
  - Routing Mechanism
  - Personalized Distillation
date: 2026-05-08
content_hash: 6f6f067cc8ec7291
---

# Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation

**Conference**: ACL 2026
**arXiv**: [2510.10925](https://arxiv.org/abs/2510.10925)
**Code**: None (but PerSyn-Math dataset is open-sourced)
**Area**: Model Compression / Knowledge Distillation
**Keywords**: Knowledge Distillation, Synthetic Data, Multi-Teacher, Routing Mechanism, Personalized Distillation

## TL;DR

This paper proposes PerSyn (Personalized data Synthesis), which adopts a "Route then Generate" paradigm where a router assigns the optimal teacher model to each prompt by jointly considering student learnability and teacher response quality. Compared to the conventional "Generate then Select" paradigm, PerSyn is more efficient and effective, consistently outperforming all baselines across instruction tuning and mathematical reasoning tasks.

## Background & Motivation

**Background**: Generating synthetic data from powerful teacher models to train smaller student models is the dominant paradigm in knowledge distillation. It is commonly assumed that stronger teachers produce higher-quality data, leading to better student learning.

**Limitations of Prior Work**: Recent studies have shown that "a stronger model is not necessarily a better teacher"—outputs from stronger models may be overly complex and diverge from the student's distribution, making effective learning difficult. The Mix approach blends data from strong and weak teachers, while CAR selects a single best teacher per prompt; however, both follow the "Generate then Select" paradigm, requiring all teachers to generate responses for every prompt, with cost scaling linearly with the number of teachers.

**Key Challenge**: (1) *Efficiency*—"Generate then Select" requires all candidate teachers to generate responses for every prompt; with 20 teachers and 100K prompts, this amounts to 2 million generations. (2) *Granularity*—existing methods select a single teacher or a fixed mixing ratio, ignoring the fact that different prompts may require different teachers.

**Goal**: Design a prompt-level optimal teacher assignment mechanism to construct personalized synthetic datasets at lower cost.

**Key Insight**: The authors observe that the optimal teacher varies across prompts—some simple prompts are better served by weaker teachers (whose outputs better match the student's level), while difficult prompts require stronger teachers.

**Core Idea**: Train a lightweight router (based on Qwen2.5-1.5B) that predicts the optimal teacher for each prompt based on student learnability and teacher quality, shifting the paradigm from "Generate then Select" to the more efficient "Route then Generate."

## Method

### Overall Architecture

Given a prompt set $\mathcal{X}$ and a teacher model pool $\mathcal{M}$, the PerSyn router $\pi(x)$ outputs a score vector $\mathbf{o} \in \mathbb{R}^{|\mathcal{M}|}$ for each prompt $x$, and the teacher with the highest score is selected. Each teacher generates responses only for the subset of prompts assigned to it; all outputs are merged into the final synthetic dataset $\mathcal{D}$ for SFT training of the student model.

### Key Designs

1. **Dual-Dimensional Teacher Evaluation Criterion**:

    - *Function*: Jointly measures how suitable each teacher's response is for a specific student.
    - *Mechanism*: The total reward is $r(y_i^{\mathcal{M}_n}, \theta) = (1-\alpha) \cdot r_q(y_i^{\mathcal{M}_n}) + \alpha \cdot r_l(y_i^{\mathcal{M}_n}, \theta)$, where the learnability reward $r_l$ is measured by the average log-likelihood under the student model (higher values indicate better alignment with student capacity), and the quality reward $r_q$ is computed by an external reward model (Skywork-Reward for instruction tuning; binary correctness for mathematical reasoning). Both rewards are normalized and combined with weight $\alpha=0.4$.
    - *Design Motivation*: Relying solely on learnability biases toward simple or low-quality responses; relying solely on quality biases toward overly complex outputs. Ablation studies confirm that removing the quality reward has a greater negative impact than removing learnability.

2. **Bradley-Terry Router Training**:

    - *Function*: Train the router on a small calibration set and generalize to the full prompt collection.
    - *Mechanism*: Parallel responses from all teachers are generated for only 2.5K prompts, from which pairwise preference labels are derived. The Bradley-Terry model is used to parameterize pairwise preference probabilities: $\mathbb{P}(B \succ A | z, x) = \sigma(z^\top \pi(x))$, where $z$ is a two-hot encoding. The router is built on Qwen2.5-1.5B with the language modeling head replaced by a scoring head (output dimension equals the number of teachers), and trained with binary cross-entropy loss.
    - *Design Motivation*: Parallel responses for 2.5K prompts are sufficient to train a high-quality router (stable Hit@3), achieving 20–40× greater efficiency than the Oracle router (which requires parallel responses for all prompts).

3. **"Route then Generate" Paradigm**:

    - *Function*: Each teacher generates responses only for its assigned prompts, substantially reducing generation cost.
    - *Mechanism*: The router partitions the prompt set into subsets $\mathcal{X}_{\mathcal{M}_i}$ assigned to teacher $\mathcal{M}_i$, and each teacher generates responses only for its own subset. Experiments show that >95% of prompts are routed to smaller teacher models, further reducing computational cost.
    - *Design Motivation*: The cost of the conventional paradigm grows linearly with the number of teachers, whereas router inference requires only a single forward pass and is negligible in cost.

### Loss & Training

Router training uses binary cross-entropy loss. Student model training follows standard SFT, computing loss only on response tokens. Full-parameter fine-tuning is applied to student models smaller than 14B; LoRA is used for larger models.

## Key Experimental Results

### Main Results

Average performance (%) of five student models across six benchmarks:

| Student Model | Strong | Mix | CAR | **PerSyn** |
|---|---|---|---|---|
| Qwen2.5-0.5B | 28.51 | 30.75 | 32.77 | **34.13** |
| Qwen2.5-1.5B | 46.82 | 47.79 | 49.21 | **50.63** |
| Gemma-2-2B | 28.45 | 29.76 | 31.41 | **32.85** |
| Qwen2.5-3B | 55.38 | 55.39 | 57.17 | **58.09** |
| Llama-3.2-3B | 31.37 | 31.75 | 32.99 | **34.81** |

Specific gains on Llama-3.2-3B (PerSyn vs. CAR): IFEval +5.8%, TruthfulQA +4.1%, MATH +7.5%.

### Ablation Study

Ablation of learnability and quality rewards (averaged over all student models):

| Setting | Effect |
|---|---|
| PerSyn (full) | Best performance |
| PerSyn w/o Learnability | ~1–2% drop |
| PerSyn w/o Quality | ~2–4% drop (larger) |

Router efficiency comparison:

| Router | Qwen2.5-0.5B | Qwen2.5-3B | Llama-3.2-3B |
|---|---|---|---|
| PerSyn Router | 27.18 | 40.53 | 30.35 |
| Oracle Router | 27.63 | 41.02 | 30.18 |

### Key Findings

- More than 95% of prompts are routed to smaller teacher models, with extremely few assigned to models such as Llama-3.1-405B.
- Qwen2.5-72B-Instruct consistently receives high assignment proportions across all student models, making it the most universally effective teacher.
- Long-CoT models (e.g., DeepSeek-R1) receive only a small share of assignments, but are indispensable—replacing them with Short-CoT teachers causes a 1.3% performance drop.
- The Strong baseline trained entirely on Long-CoT data performs worse, as models tend to generate repetitive reasoning chains.

## Highlights & Insights

- The "Route then Generate" paradigm shift is elegant and principled, fundamentally addressing the efficiency bottleneck of multi-teacher distillation.
- The Bradley-Terry router requires only 2.5K calibration samples to generalize effectively, making it highly practical.
- Quality is more important than learnability ($\alpha=0.4$), yet both are indispensable—this corrects the two extremes of "always use the strongest teacher" and "always use the best-matching teacher."

## Limitations & Future Work

- Validation is limited to instruction tuning and mathematical reasoning; code generation, multimodal tasks, and other settings remain unexplored.
- Student model scale is restricted to 14B and below; whether larger models also benefit from personalized distillation remains to be verified.
- The router requires separate training for each (setting, student model) combination, and cumulative cost may become non-trivial when configurations change frequently.

## Related Work & Insights

- Li et al. (2025) first identified the "learnability gap" issue and proposed the Mix strategy; PerSyn refines this from the dataset level to the prompt level.
- CAR (Xu et al., 2025) selects a single teacher per prompt; PerSyn demonstrates that different prompts require different teachers.
- Insight: Distillation is not merely a "data quality" problem but a "data–student matching" problem; the routing mechanism has the potential to generalize to other data selection scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The paradigm shift is clearly motivated and the router design is practical, though the core idea (using different teachers for different samples) is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five student models × three model families × two settings × six benchmarks, with highly comprehensive ablations and analyses.
- Writing Quality: ⭐⭐⭐⭐ Figures and tables are well-designed; Table 1 presents a compelling comparison, and the overall narrative is coherent.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](../../ICLR2026/model_compression/pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[CVPR 2026\] Distilling Balanced Knowledge from a Biased Teacher](../../CVPR2026/model_compression/distilling_balanced_knowledge_from_a_biased_teacher.md)
- [\[ICCV 2025\] A Good Teacher Adapts Their Knowledge for Distillation](../../ICCV2025/model_compression/a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [\[ICLR 2026\] STAR: Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models](../../ICLR2026/model_compression/star_similarity-guided_teacher-assisted_refinement_for_super-tiny_function_calli.md)
- [\[NeurIPS 2025\] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](../../NeurIPS2025/model_compression/single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)

</div>

<!-- RELATED:END -->
