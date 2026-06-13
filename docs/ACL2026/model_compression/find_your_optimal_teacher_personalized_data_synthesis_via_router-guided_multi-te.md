---
title: >-
  [Paper Note] Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation
description: >-
  [ACL 2026][Model Compression][Knowledge Distillation] Ours proposes PerSyn (Personalized data Synthesis), which adopts a "Route then Generate" paradigm where a router assigns the optimal teacher model to each prompt. By…
tags:
  - "ACL 2026"
  - "Model Compression"
  - "Knowledge Distillation"
  - "Synthetic Data"
  - "Multi-teacher"
  - "Routing Mechanism"
  - "Personalized Distillation"
date: 2026-05-08
content_hash: 49af76dd0faafa81
---

# Find Your Optimal Teacher: Personalized Data Synthesis via Router-Guided Multi-Teacher Distillation

**Conference**: ACL 2026  
**arXiv**: [2510.10925](https://arxiv.org/abs/2510.10925)  
**Code**: None (but the PerSyn-Math dataset is open-sourced)  
**Area**: Model Compression / Knowledge Distillation  
**Keywords**: Knowledge Distillation, Synthetic Data, Multi-teacher, Routing Mechanism, Personalized Distillation

## TL;DR

Ours proposes PerSyn (Personalized data Synthesis), which adopts a "Route then Generate" paradigm where a router assigns the optimal teacher model to each prompt. By considering both student learnability and teacher response quality, this approach is more efficient and effective than the traditional "Generate then Select" paradigm, consistently outperforming all baselines in instruction tuning and mathematical reasoning scenarios.

## Background & Motivation

**Background**: Generating synthetic data with powerful teacher models to train smaller student models is a mainstream approach in knowledge distillation. It is generally assumed that stronger teachers produce higher-quality data, leading to better student performance.

**Limitations of Prior Work**: Recent studies have found that "a stronger model is not necessarily a better teacher"—outputs from strong models may be overly complex or deviate from the student's distribution, making it difficult for students to learn effectively. Methods like Mix blend data from strong and weak teachers, while CAR selects a single best teacher. However, both follow the "Generate then Select" paradigm, requiring all teachers to generate responses for all prompts, which results in costs that scale linearly with the number of teachers.

**Key Challenge**: (1) Efficiency issue—"Generate then Select" requires every candidate teacher to generate a response for every prompt (e.g., 20 teachers $\times$ 100K prompts = 2 million generations). (2) Granularity issue—existing methods select a single teacher or use a fixed mixing ratio, ignoring the fact that different prompts require different teachers.

**Goal**: Design a prompt-level optimal teacher assignment mechanism to construct personalized synthetic datasets at a lower cost.

**Key Insight**: The authors observe that the optimal teacher varies for different prompts—some simple prompts are better suited for weak teachers (whose outputs match the student's level), while difficult prompts require strong teachers.

**Core Idea**: A lightweight router (based on Qwen2.5-1.5B) is trained to predict the optimal teacher for each prompt based on student learnability and teacher quality, shifting the paradigm from "Generate then Select" to a more efficient "Route then Generate" approach.

## Method

### Overall Architecture

Given a prompt set $\mathcal{X}$ and a teacher model pool $\mathcal{M}$, the PerSyn router $\pi(x)$ outputs a score vector $\mathbf{o} \in \mathbb{R}^{|\mathcal{M}|}$ for each prompt $x$, selecting the teacher with the highest score. That teacher generates responses only for its assigned subset of prompts. The outputs from all teachers are merged into the final synthetic dataset $\mathcal{D}$ for SFT training of the student model.

### Key Designs

1.  **Dual-dimension Teacher Evaluation Criteria**:

    - **Function**: Comprehensively measures the suitability of each teacher's response for a specific student.
    - **Mechanism**: The total reward $r(y_i^{\mathcal{M}_n}, \theta) = (1-\alpha) \cdot r_q(y_i^{\mathcal{M}_n}) + \alpha \cdot r_l(y_i^{\mathcal{M}_n}, \theta)$, where the learnability reward $r_l$ is measured by the average log-likelihood of the student model (higher values indicate a better match with student capability), and the quality reward $r_q$ is measured by an external reward model (Skywork-Reward for instruction tuning, binary correctness for math reasoning). Both are normalized and weighted with $\alpha=0.4$.
    - **Design Motivation**: Focusing solely on learnability favors simple/low-quality responses; focusing solely on quality favors overly complex outputs. Ablation studies confirm that removing quality has a greater impact than removing learnability.

2.  **Bradley-Terry Router Training**:

    - **Function**: Trains the router using a small amount of calibration data to generalize to the full set of prompts.
    - **Mechanism**: Parallel responses from all teachers are generated for only 2.5K prompts to calculate pairwise preference labels. A Bradley-Terry model is used to model pairwise preference probabilities: $\mathbb{P}(B \succ A | z, x) = \sigma(z^\top \pi(x))$, where $z$ is a two-hot encoding. The router is based on Qwen2.5-1.5B, replacing the language modeling head with a coefficient head (output dimension = number of teachers), and is trained using binary cross-entropy loss.
    - **Design Motivation**: Parallel responses for 2.5K prompts are sufficient to train a high-quality router (stable Hit@3), which is 20-40 times more efficient than an Oracle router (which requires full parallel responses).

3.  **"Route then Generate" Paradigm**:

    - **Function**: Each teacher only generates responses for its assigned prompts, significantly reducing generation costs.
    - **Mechanism**: The router partitions the prompt set into subsets $\mathcal{X}_{\mathcal{M}_i}$ assigned to teacher $\mathcal{M}_i$. Each teacher is responsible for generating only its own subset. Experiments show that $>95\%$ of prompts are routed to smaller teacher models, further reducing computational costs.
    - **Design Motivation**: Traditional paradigms scale linearly with the number of teachers. Router prediction requires only one forward pass, making its cost negligible.

### Loss & Training

Router training: binary cross-entropy loss; student model training: standard SFT, with loss calculated only on response tokens. Full-parameter fine-tuning is used for student models smaller than 14B, while LoRA is used for larger models.

## Key Experimental Results

### Main Results

Average performance (%) of five student models across six benchmarks:

| Student Model | Strong | Mix | CAR | **PerSyn** |
| :--- | :--- | :--- | :--- | :--- |
| Qwen2.5-0.5B | 28.51 | 30.75 | 32.77 | **34.13** |
| Qwen2.5-1.5B | 46.82 | 47.79 | 49.21 | **50.63** |
| Gemma-2-2B | 28.45 | 29.76 | 31.41 | **32.85** |
| Qwen2.5-3B | 55.38 | 55.39 | 57.17 | **58.09** |
| Llama-3.2-3B | 31.37 | 31.75 | 32.99 | **34.81** |

Specific gains on Llama-3.2-3B (PerSyn vs CAR): IFEval +5.8%, TruthfulQA +4.1%, MATH +7.5%.

### Ablation Study

Ablation of learnability and quality rewards (average across all student models):

| Setting | Impact |
| :--- | :--- |
| PerSyn (Full) | Highest performance |
| PerSyn w/o Learnability | Decrease of approx. 1-2% |
| PerSyn w/o Quality | Decrease of approx. 2-4% (Larger) |

Router efficiency comparison:

| Router | Qwen2.5-0.5B | Qwen2.5-3B | Llama-3.2-3B |
| :--- | :--- | :--- | :--- |
| PerSyn Router | 27.18 | 40.53 | 30.35 |
| Oracle Router | 27.63 | 41.02 | 30.18 |

### Key Findings

- $>95\%$ of prompts are routed to smaller teacher models, with ultra-large models like Llama-3.1-405B receiving minimal assignments.
- Qwen2.5-72B-Instruct consistently achieves a high assignment ratio across all student models, proving to be the most versatile teacher.
- Long-CoT models (e.g., DeepSeek-R1) only account for a small portion of assignments but are indispensable—replacing them with Short-CoT teachers leads to a 1.3% performance drop.
- The Strong baseline, which uses only Long-CoT data for training, performs worse because the model tends to produce repetitive reasoning.

## Highlights & Insights

- The "Route then Generate" paradigm shift is simple and elegant, fundamentally addressing the efficiency bottleneck of multi-teacher distillation.
- The Bradley-Terry router requires only 2.5K calibration samples to generalize, making it highly practical.
- Quality is more important than learnability ($\alpha=0.4$), but both are essential—correcting the extreme views of "using only the strongest teacher" versus "using only the best-matching teacher."

## Limitations & Future Work

- Validated only in instruction tuning and math reasoning; code generation and multi-modality remain unexplored.
- Student model scale is limited to under 14B; whether larger models benefit from personalized distillation remains to be verified.
- The router needs to be trained separately for each (setting, student model) combination, which could lead to cumulative costs if settings change frequently.

## Related Work & Insights

- Li et al. (2025) first revealed the "learnability gap" and proposed the Mix strategy; PerSyn refines this from the dataset level to the prompt level.
- CAR (Xu et al., 2025) selects a single teacher; PerSyn demonstrates that different prompts require different teachers.
- Insight: Distillation is not just a "data quality" issue but a "data-student matching" issue. The routing mechanism is expected to extend to other data selection scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The paradigm shift is clear and the router design is practical, though the core idea (different teachers for different samples) is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Five student models $\times$ three families $\times$ two scenarios $\times$ six benchmarks, with extremely detailed ablation and analysis.
- Writing Quality: ⭐⭐⭐⭐ Excellent chart design; the comparison in Table 1 is intuitive and powerful, with a smooth overall narrative.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Pedagogically-Inspired Data Synthesis for Language Model Knowledge Distillation](../../ICLR2026/model_compression/pedagogically-inspired_data_synthesis_for_language_model_knowledge_distillation.md)
- [\[CVPR 2026\] Distilling Balanced Knowledge from a Biased Teacher](../../CVPR2026/model_compression/distilling_balanced_knowledge_from_a_biased_teacher.md)
- [\[ICLR 2026\] STAR: Similarity-guided Teacher-Assisted Refinement for Super-Tiny Function Calling Models](../../ICLR2026/model_compression/star_similarity-guided_teacher-assisted_refinement_for_super-tiny_function_calli.md)
- [\[ICCV 2025\] A Good Teacher Adapts Their Knowledge for Distillation](../../ICCV2025/model_compression/a_good_teacher_adapts_their_knowledge_for_distillation.md)
- [\[NeurIPS 2025\] Single-Teacher View Augmentation: Boosting Knowledge Distillation via Angular Diversity](../../NeurIPS2025/model_compression/single-teacher_view_augmentation_boosting_knowledge_distillation_via_angular_div.md)

</div>

<!-- RELATED:END -->
