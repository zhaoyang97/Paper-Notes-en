---
title: >-
  [Paper Note] RetrySQL: Text-to-SQL Training with Retry Data for Self-Correcting Query Generation
description: >-
  [AAAI 2026][Image Generation][Text-to-SQL] This paper proposes the RetrySQL training paradigm, which injects retry data (erroneous steps + [BACK] token + correct steps) into reasoning chains during continual pre-training…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "Text-to-SQL"
  - "self-correction"
  - "retry data"
  - "chain of reasoning"
  - "small language models"
date: 2026-05-08
content_hash: 6dce33751f6ccb37
---

# RetrySQL: Text-to-SQL Training with Retry Data for Self-Correcting Query Generation

**Conference**: AAAI 2026
**arXiv**: [2507.02529](https://arxiv.org/abs/2507.02529)
**Code**: [https://github.com/allegro/RetrySQL](https://github.com/allegro/RetrySQL)
**Area**: Image Generation
**Keywords**: Text-to-SQL, self-correction, retry data, chain of reasoning, small language models

## TL;DR

This paper proposes the RetrySQL training paradigm, which injects retry data (erroneous steps + [BACK] token + correct steps) into reasoning chains during continual pre-training of small encoder models. This approach enables a 1.5B open-source model to acquire self-correction capabilities, achieving improvements of up to 4 and 3.93 percentage points in overall execution accuracy on the BIRD and SPIDER benchmarks, respectively, with gains of up to 9 percentage points on challenging samples.

## Background & Motivation

### State of the Field

Text-to-SQL is a core challenge in NLP, aiming to translate natural language questions into SQL queries. Current approaches fall into two main categories:
- **End-to-end pipelines**: combining retrieval (schema linking), generation, and correction stages
- **Large model solutions**: GPT-4o, Gemini, and similar models still lag significantly behind human performance on BIRD and SPIDER benchmarks
- **Fine-tuning solutions**: parameter-efficient fine-tuning (e.g., LoRA) of open-source models, primarily focused on larger models (7B+)

### Limitations of Prior Work

**Underexplored potential of small models**: 1.5B encoder models (e.g., OpenCoder, Qwen2.5-Coder) perform poorly in zero-shot settings (EX_overall ~30–37%), yet are highly efficient to train and cheap to infer, making them attractive for practical deployment.

**Self-correction not yet applied to Text-to-SQL**: While DeepSeek-R1 and others have demonstrated that RL can teach self-correction, more recent work shows that standard autoregressive training with retry data can also learn self-correction (in mathematical reasoning tasks). However, this approach **has not yet been applied to the Text-to-SQL domain**.

**Existing correction is post-hoc**: Correction steps in approaches such as DIN-SQL are applied after generation rather than during the generation process itself.

### Root Cause

Small models underperform on Text-to-SQL tasks, yet scaling up model size is constrained by limited training data (BIRD contains only ~9K examples) and computational resources. The central question is: how can the generative capability of small models be substantially improved **without increasing model parameters**?

### Starting Point

Inspired by the "Physics of Language Models" line of work, which demonstrates that retry data can teach self-correction in mathematical reasoning, this paper transfers the paradigm to Text-to-SQL. Training data containing errors and their corrections is constructed, and continual pre-training is used to enable the model to learn a "make error → identify → backtrack → correct" capability.

## Method

### Overall Architecture

The RetrySQL training paradigm consists of three stages:

1. **Reasoning step generation** (Figure 1a): GPT-4o is used to generate a chain of reasoning for each SQL query in the training set.
2. **Retry data construction** (Figure 1b): Random perturbations are applied to the reasoning steps (replacing correct steps with erroneous ones), with errors annotated using the [BACK] token followed by the correct step.
3. **Continual pre-training** (Figure 1c): Open-source coding models are continually pre-trained on training examples containing retry data.

### Key Designs

#### 1. Reasoning Step Generation

**Core Idea**: GPT-4o is used to generate synthetic reasoning chains for the SQL query of each training example, formatted similarly to solution steps.

**Implementation Details**:
- DDL (Data Definition Language) is used as the schema representation, as it provides concise table/column names, data types, and relational information.
- Perfect schema linking is adopted (including only the tables/columns referenced by the query) to focus the study on SQL generation capability.
- The correctness of reasoning steps is manually verified on 100 randomly sampled examples.

**Design Motivation**: Prior work shows that retry data is only effective when reasoning steps are present. Reasoning steps provide **intermediate targets for backtracking**, making the "error → correction" mechanism feasible.

#### 2. Retry Data Construction

**Core Idea**: Random perturbations are applied to a reasoning step sequence $(r_1, r_2, ..., r_N)$. Given step $r_i$, an erroneous step $r_{error}$ is selected with probability $p_{retry}$ to replace it, then annotated with the [BACK] token followed by the correct $r_i$.

Four perturbation strategies:
- **FS (Forward Single)**: A randomly selected **subsequent step** is used as the error (once).
- **FM (Forward Multiple)**: Multiple subsequent steps are randomly selected as consecutive errors.
- **FBS (Forward-Back Single)**: A randomly selected step from **all other steps** is used as the error.
- **FBM (Forward-Back Multiple)**: Multiple steps from all other steps are selected as errors.

Each $r_i$ is replaced by: $(r_{error}, \text{[BACK]}, r_i)$

**Design Motivation**:
- The FS strategy is most effective because errors in SQL generation typically arise from **prematurely using a step that has not yet been reached** (i.e., "skipping ahead").
- Backward-referencing errors (FBS/FBM) do not reflect typical error patterns in SQL generation.
- Multiple perturbations (FM/FBM) may introduce excessive noise.

#### 3. Training Data Format

Special tokens are introduced—[CONTEXT], [QUESTION], [REASONING], [SQL]—to delimit the database schema, external knowledge, reasoning steps, and SQL query:

```
[CONTEXT] DDL schema + external knowledge
[QUESTION] natural language question  
[REASONING] step1 [BACK] correct_step1 step2 step3 ...
[SQL] SELECT ...
```

At inference time, the model receives input up to the [REASONING] token and autoregressively generates the reasoning steps and SQL query.

#### 4. Linear Probe Validation

**Validation of a Key Prerequisite**: Before training, it is first verified whether the base model **intrinsically possesses the ability to distinguish correct from incorrect reasoning steps**.

Method: The weights of OpenCoder 1.5B are frozen, and a binary classification head is trained to predict the correctness of reasoning steps. Results:
- balanced_accuracy = 82%
- f1_score = 71%

These results substantially exceed the random baseline of 50%, confirming that the model already possesses a latent self-correction capacity, and that retry data merely "unlocks" this ability.

### Loss & Training

- OpenCoder 1.5B and Qwen2.5-Coder 1.5B are used as base models.
- 2× NVIDIA A100 80GB GPUs, DeepSpeed Zero-2.
- Effective batch size of 128, trained for 5 epochs.
- Learning rate of 5e-5 with cosine scheduling.
- AdamW optimizer (β₁=0.9, β₂=0.95).
- Each training run requires approximately 4.47 GPU hours.

## Key Experimental Results

### Main Results

Execution accuracy (EX) of OpenCoder 1.5B on the BIRD dataset:

| Data Variant | EX_simple | EX_moderate | EX_challenging | EX_overall |
|---|---|---|---|---|
| Zero-shot | 47.14 | 27.63 | 17.52 | 38.44 |
| Without reasoning steps | 43.78 | 28.88 | 24.83 | 37.48 |
| With reasoning steps (no retry) | 62.70 | 43.53 | 39.45 | 54.71 |
| **Retry FS 0.2** | **68.22** | **45.47** | 40.28 | **58.70** |
| **Retry FS 0.3** | 68.00 | 44.91 | **43.31** | 58.68 |

Qwen2.5-Coder 1.5B on the SPIDER dataset:

| Data Variant | EX_easy | EX_medium | EX_hard | EX_extra | EX_overall |
|---|---|---|---|---|---|
| Without retry | 92.34 | 74.75 | 65.52 | 48.80 | 73.25 |
| **Retry FS 0.2** | 90.40 | **80.94** | **69.66** | **55.18** | **77.18** |

### Ablation Study

Comparison of different retry strategies (OpenCoder 1.5B, BIRD):

| Strategy | $p_{retry}$ | EX_overall | EX_challenging | Notes |
|---|---|---|---|---|
| No retry | - | 54.71 | 39.45 | Baseline |
| **FS 0.2** | 0.2 | **58.70** | 40.28 | Best overall |
| **FS 0.3** | 0.3 | 58.68 | **43.31** | Best on challenging |
| FM 0.3 | 0.3 | 57.17 | 37.24 | Multiple perturbations inferior to single |
| FBS 0.4 | 0.4 | 57.63 | 35.86 | Backward reference performs poorly |
| FBM 0.3 | 0.3 | 57.43 | 37.93 | Overall inferior to FS |
| FM 0.5 | 0.5 | 48.63 | 28.28 | Excessive perturbation is harmful |

### End-to-End Pipeline Results

| Model | EX_overall (BIRD) | Parameters |
|---|---|---|
| GPT-4o-mini | 32.53 | ~8B |
| **RetrySQL (OpenCoder 1.5B)** | **51.36** | 1.5B |
| GPT-4o | 54.99 | ~200B |

### Self-Correction Behavior Analysis

Changes in model confidence before and after the [BACK] token:

| Position | Mean Max Softmax Score | Softmax Std. Dev. |
|---|---|---|
| Before [BACK] (erroneous tokens) | Lower | Higher (uncertain) |
| After [BACK] (correction tokens) | Higher | Lower (certain) |

This demonstrates that self-correction is **an actively learned behavior**, rather than a simple effect of data augmentation.

### Key Findings

- **Reasoning steps themselves are critical**: Adding reasoning steps (without retry) improves EX_overall from 37.48 to 54.71 (+17.2 p.p.).
- **FS strategy is consistently optimal**: Among the four perturbation strategies, Forward Single performs best across all settings.
- **Largest gains on challenging samples**: Retry FS 0.3 improves EX_challenging on BIRD from 39.45 to 43.31 (+3.86 p.p.); Qwen2.5-Coder improves from 30.21 to 39.10 on BIRD (+8.89 p.p.).
- **Optimal $p_{retry}$ is 0.2–0.3**: Performance degrades with values that are too low (0.1) or too high (0.5).
- **RetrySQL reduces reasoning step length**: Models trained with retry data generate an average of 9.47 steps vs. 12.48 steps without retry, as self-correction prevents "continuing down the wrong path."
- **1.5B models can challenge GPT-4o**: In the end-to-end pipeline, the RetrySQL-trained 1.5B model achieves 51.36 EX_overall, surpassing GPT-4o-mini (32.53) and approaching GPT-4o (54.99).

## Highlights & Insights

1. **Self-correction is a universally learnable property**: The effectiveness of retry data across mathematical reasoning and Text-to-SQL supports the hypothesis that self-correction is a general capability of language models, not limited to specific tasks.
2. **Linear probing enables prior validation**: Verifying model potential through probe experiments before committing training resources represents a principled "validate-then-train" methodology worth generalizing.
3. **Persuasiveness of confidence analysis**: The paper not only demonstrates performance gains but also **proves** the existence of self-corrective behavior through statistical analysis of softmax scores.
4. **Practical value of small models**: The 1.5B model trained with RetrySQL approaches GPT-4o performance in an end-to-end pipeline, with significant implications for cost-sensitive production environments.
5. **Training cost of 4.47 GPU hours**: Substantial improvements are obtained at extremely low computational cost.

## Limitations & Future Work

- Training data is limited (~9K examples in BIRD); synthetic data generation is not explored.
- The effect of scaling model size is not investigated (only 1.5B models are evaluated).
- The schema linking and correction stages of the end-to-end pipeline are relatively simple and do not employ state-of-the-art methods.
- The optimal $p_{retry}$ may be dataset-dependent.
- Reasoning steps are generated by GPT-4o, introducing a dependency on a strong LLM.
- The combination of retry data with alignment methods such as RLHF/DPO is not explored.

## Related Work & Insights

- **Relation to DeepSeek-R1**: R1 learns self-correction via RL, whereas RetrySQL learns it through data construction and standard pre-training—the latter being simpler and more efficient.
- **Relation to s1 (budget forcing)**: s1 steers reasoning by enforcing a thinking budget during SFT, while RetrySQL naturally learns backtracking through retry data.
- **Validation of Physics of LMs**: This paper validates the "self-correction as a universal law" hypothesis in a new domain (Text-to-SQL) and on new model architectures (Qwen2.5-Coder).
- **Broader implications**: The retry data paradigm may be applicable to other reasoning-intensive tasks, such as code generation, theorem proving, and planning.

## Rating

- Novelty: ⭐⭐⭐⭐ (First application of the retry data paradigm to Text-to-SQL, validating its generality)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Multiple models, benchmarks, four strategies, multiple probabilities, confidence analysis, pipeline evaluation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, in-depth analysis, strong reproducibility)
- Value: ⭐⭐⭐⭐ (The small-model + self-correction paradigm has practical deployment implications, though broader domain validation is needed)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Infinite-Story: A Training-Free Consistent Text-to-Image Generation](infinite-story_a_training-free_consistent_text-to-image_gene.md)
- [\[AAAI 2026\] Targeted Data Protection for Diffusion Model by Matching Training Trajectory](targeted_data_protection_for_diffusion_model_by_matching_training_trajectory.md)
- [\[AAAI 2026\] Difficulty Controlled Diffusion Model for Synthesizing Effective Training Data](difficulty_controlled_diffusion_model_for_synthesizing_effec.md)
- [\[AAAI 2026\] Self-NPO: Data-Free Diffusion Model Enhancement via Truncated Diffusion Fine-Tuning](self-npo_data-free_diffusion_model_enhancement_via_truncated_diffusion_fine-tuni.md)
- [\[ICLR 2026\] Neon: Negative Extrapolation From Self-Training Improves Image Generation](../../ICLR2026/image_generation/neon_negative_extrapolation_image_generation.md)

</div>

<!-- RELATED:END -->
