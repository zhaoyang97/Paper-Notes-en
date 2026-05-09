---
title: >-
  [Paper Note] FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation
description: >-
  [ACL 2026][AIGC Detection][content moderation] FlexGuard proposes an LLM moderation model that outputs continuous risk scores (0-100) instead of binary safe/unsafe judgments, achieving SOTA robustness and accuracy across varying enforcement strictness levels via rubric-guided distillation and GRPO risk alignment training.
tags:
  - ACL 2026
  - AIGC Detection
  - content moderation
  - continuous risk scoring
  - strictness adaptation
  - LLM safety
  - reinforcement learning
date: 2025-05-08
content_hash: db4e97105a50509b
---

# FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation

**Conference**: ACL 2026  
**arXiv**: [2602.23636](https://arxiv.org/abs/2602.23636)  
**Code**: [GitHub](https://github.com/)  
**Area**: AI Safety / Content Moderation  
**Keywords**: content moderation, continuous risk scoring, strictness adaptation, LLM safety, reinforcement learning

## TL;DR

FlexGuard outputs continuous risk scores (0-100) instead of binary safe/unsafe judgments for LLM content moderation, achieving SOTA robustness and accuracy across varying strictness deployment scenarios through rubric-guided distillation and GRPO risk alignment training.

## Background & Motivation

**State of the Field**: LLM content moderation models (LlamaGuard, WildGuard, etc.) have evolved through multiple generations and are widely used to detect harmful content in user inputs and model outputs. The vast majority of existing moderation models define content moderation as a fixed binary classification task.

**Limitations of Prior Work**: Enforcement strictness—the degree to which a platform is conservative about "harmful"—varies significantly across platforms and time periods. For example, platform X allows appropriately labeled adult content, while certain Reddit communities require all-ages content. Binary moderation models are implicitly bound to the safety definition of their training data and cannot adapt to changing strictness requirements, leading to inconsistent cross-strictness performance: Qwen3Guard drops 19.2% from strict to loose in prompt moderation.

**Root Cause**: The "safe/unsafe" boundary in moderation decisions is not fixed but varies with deployment context, yet existing models and benchmarks assume a single fixed safety definition.

**Paper Goals**: (1) Build a benchmark (FlexBench) that evaluates moderation models under different strictness levels; (2) design a moderation model (FlexGuard) that adapts to strictness changes.

**Starting Point**: Replace binary classification with continuous risk scoring so that strictness adaptation reduces to simple threshold selection. Use rubric-guided distillation to obtain continuous labels, then optimize score-severity alignment via GRPO reinforcement learning.

**Core Idea**: Calibrated continuous risk scores + deployment-time threshold selection = strictness-adaptive moderation.

## Method

### Overall Architecture

The system has two parts: (1) FlexBench—a strictness-annotated benchmark with 4K instances covering seven risk categories and five severity levels, supporting strict/moderate/loose evaluation modes; (2) FlexGuard—a Qwen3-8B-based moderation model trained in two stages (SFT warmup + GRPO alignment) to output risk categories and continuous scores, with deployment-time threshold adaptation for strictness.

### Key Designs

1. **FlexBench Strictness-Adaptive Benchmark**:

    - Function: Evaluate moderation model reliability under different strictness levels
    - Mechanism: Defines five severity levels (BENIGN/LOW/MODERATE/HIGH/EXTREME) mapped to three strictness regimes—strict (only BENIGN is safe), moderate (BENIGN+LOW safe), loose (BENIGN+LOW+MODERATE safe). Covers seven risk categories (violence/illegal/sexual/privacy/discrimination/misinformation/jailbreak), containing 2K prompt instances and 2K response instances. Uses human-AI collaborative annotation: LLM generates candidate labels, five human annotators verify and correct, disagreements are resolved by senior annotators
    - Design Motivation: Existing benchmarks use fixed binary labels and cannot evaluate model robustness under strictness changes

2. **Rubric-Guided Distillation Pipeline**:

    - Function: Generate pseudo-labels for continuous risk scores during training
    - Mechanism: Use expert-designed scoring rubrics to guide a strong LLM (e.g., GPT-5) to generate risk category $c(x)$, score $r'(x) \in [0, 100]$, and reasoning process for each instance. The key step is label consistency calibration—aligning LLM scores with source dataset binary labels by linearly mapping raw scores $r'(x)$ to label-consistent intervals (safe: [0,40], unsafe: [40,100]), suppressing cross-boundary outliers
    - Design Motivation: Public moderation corpora mostly have binary labels, and direct continuous annotation is prohibitively expensive; LLM distillation enables large-scale generation, and calibration ensures consistency with existing labels

3. **Two-Stage Risk Alignment Training**:

    - Function: Train the model to produce continuous scores aligned with risk severity
    - Mechanism: Stage 1 uses LoRA SFT warmup to teach the model to follow rubric reasoning and output formatted $(\hat{c}(x), \hat{r}(x))$; Stage 2 uses GRPO reinforcement learning with dense reward $R(x) = s_{\text{category}}(x) + s_{\text{score}}(x)$, where category accuracy reward $s_{\text{category}} \in \{-1, +1\}$ and score regression reward $s_{\text{score}} = 2 - \frac{4}{E_{\max}} |\hat{r}(x) - r(x)| \in [-2, 2]$, with $E_{\max}$ normalizing to make errors comparable across different target scores
    - Design Motivation: SFT provides stable initialization; GRPO directly optimizes score alignment objective; dense linear regression reward provides richer gradient signal than binary reward

### Loss & Training

Two-stage training: Stage 1 standard SFT with LoRA, Stage 2 GRPO with combined category accuracy + score regression dense reward. Trained on 8×H20 GPUs.

## Key Experimental Results

### Main Results

**FlexBench Strictness-Adaptive Moderation (Harmfulness F1 %)**

| Method | Prompt Avg | Prompt Worst | Response Avg | Response Worst |
|------|-----------|-------------|-------------|---------------|
| GPT-5 | 73.26 | 70.95 | 77.43 | 74.07 |
| Qwen3Guard-8B | 75.10 | 67.06 | 76.61 | 69.16 |
| BingoGuard-8B | 74.22 | 68.31 | 76.59 | 74.80 |
| **FlexGuard (calibrated threshold)** | **81.78** | **78.26** | **80.29** | **75.81** |

### Ablation Study

| Config | Key Metric | Note |
|------|---------|------|
| FlexGuard full | Avg 81.78 / Worst 78.26 | Best |
| SFT only (no GRPO) | Decreased | Insufficient score-severity alignment |
| No label consistency calibration | Decreased | More cross-boundary outliers |
| Rubric threshold (no calibration) | 80.29 / 76.63 | Still competitive |

### Key Findings

- FlexGuard's cross-strictness performance drop is significantly lower than competitors: best-worst gap on prompts is only 5.73%, vs. 15.95% for Qwen3Guard and 13.52% for BingoGuard
- Rubric thresholds achieve competitive performance without a validation set (Prompt Avg 80.29); calibrated thresholds improve by ~1.5%
- On public benchmarks (no strictness variation), FlexGuard also matches or exceeds SOTA (Prompt Avg 85.36, Response Avg 87.85)
- GRPO stage significantly improves score quality: MAE decreases and score distributions become more separated across severity levels

## Highlights & Insights

- Redefines content moderation from a "classification problem" to a "risk assessment problem"; the continuous scoring + threshold selection design elegantly decouples model capability from deployment requirements
- Label consistency calibration is a key technical detail—aligning LLM-distilled scores with existing binary labels solves pseudo-label quality issues
- Dense linear regression reward design (vs. common binary reward) provides richer gradient signal for GRPO

## Limitations & Future Work

- FlexBench currently supports English only; strictness adaptation behavior in multilingual scenarios is unknown
- Three strictness levels may be too coarse—real deployments may need finer-grained control
- Interpretability of continuous scores needs strengthening—users may need to understand score semantics
- Adversarial input (jailbreak attacks) score stability has not been tested

## Related Work & Insights

- **vs LlamaGuard/WildGuard**: These models output binary labels and adapt strictness poorly via logit thresholds; FlexGuard natively outputs continuous scores
- **vs BingoGuard/PKU-SafeRLHF**: These output discrete severity levels with limited granularity; FlexGuard's continuous scoring provides finer risk discrimination

## Rating

- Novelty: ⭐⭐⭐⭐ Problem definition (strictness-adaptive moderation) is novel and practical; continuous scoring approach is naturally sound
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Custom benchmark + public benchmarks, multiple baselines, complete ablation
- Writing Quality: ⭐⭐⭐⭐ Clear structure with well-motivated problem; some details could be more concise
- Value: ⭐⭐⭐⭐⭐ Directly addresses industry deployment pain points; FlexBench can become a new standard for moderation evaluation

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content](../../NeurIPS2025/aigc_detection/can_llms_write_faithfully_an_agent-based_evaluation_of_llm-generated_islamic_con.md)
- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)
- [\[ACL 2026\] BIASEDTALES-ML: A Multilingual Dataset for Analyzing Narrative Attribute Distributions in LLM-Generated Stories](biasedtales-ml_a_multilingual_dataset_for_analyzing_narrative_attribute_distribu.md)
- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)

<!-- RELATED:END -->
---
title: >-
  [Paper Note] FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation
description: >-
  [ACL 2026][AI Safety][Content Moderation] FlexGuard proposes an LLM moderation model that outputs continuous risk scores (0-100) instead of binary safe/unsafe judgments, achieving SOTA robustness and accuracy across varying strictness levels via rubric-guided distillation and GRPO risk alignment training.
tags:
  - ACL 2026
  - AI Safety
  - Content Moderation
  - Continuous Risk Scoring
  - Strictness Adaptation
  - Reinforcement Learning
date: 2025-05-08
content_hash: db4e97105a50509b
---

# FlexGuard: Continuous Risk Scoring for Strictness-Adaptive LLM Content Moderation

**Conference**: ACL 2026
**arXiv**: [2602.23636](https://arxiv.org/abs/2602.23636)
**Code**: [GitHub](https://github.com/)
**Area**: AI Safety / Content Moderation
**Keywords**: content moderation, continuous risk scoring, strictness adaptation, LLM safety, reinforcement learning

## TL;DR

FlexGuard proposes an LLM moderation model that outputs continuous risk scores (0-100) instead of binary safe/unsafe judgments, achieving SOTA robustness and accuracy across varying enforcement strictness levels via rubric-guided distillation and GRPO risk alignment training.

## Background & Motivation

**State of the Field**: LLM content moderation models (LlamaGuard, WildGuard, etc.) have evolved through multiple generations and are widely used to detect harmful content in user inputs and model outputs. The vast majority of existing moderation models define content moderation as a fixed binary classification task.

**Limitations of Prior Work**: Enforcement strictness — the degree to which a platform is conservative about what counts as "harmful" — varies significantly across platforms and over time. For example, platform X allows appropriately labeled adult content, while certain Reddit communities require all-ages content. Binary moderation models are implicitly tied to the safety definition of their training data and cannot adapt to changing strictness requirements, leading to inconsistent cross-strictness performance: Qwen3Guard drops 19.2% from strict to loose in prompt moderation.

**Root Cause**: The "safe/unsafe" boundary of moderation decisions is not fixed but varies with deployment contexts, yet existing models and benchmarks all assume a single fixed safety definition.

**Paper Goals**: (1) Build a benchmark (FlexBench) that can evaluate moderation models across different strictness levels; (2) design a moderation model (FlexGuard) that can adapt to strictness changes.

**Starting Point**: Replace binary classification with continuous risk scoring, so strictness adaptation reduces to a simple threshold selection problem. Acquire continuous labels through rubric-guided distillation, then optimize score-severity consistency via GRPO reinforcement learning.

**Core Idea**: Calibrated continuous risk scores + deployment-time threshold selection = strictness-adaptive moderation.

## Method

### Overall Architecture

The system consists of two parts: (1) FlexBench — a benchmark with strictness annotations, containing 4K instances covering seven risk categories and five severity levels, supporting strict/moderate/loose evaluation modes; (2) FlexGuard — a Qwen3-8B-based moderation model trained through two-stage training (SFT warmup + GRPO alignment) to output risk categories and continuous scores, with deployment-time strictness adaptation via threshold selection.

### Key Designs

1. **FlexBench Strictness-Adaptive Benchmark**:

    - Function: Evaluate moderation model reliability across different strictness levels
    - Mechanism: Define five severity levels (BENIGN/LOW/MODERATE/HIGH/EXTREME), mapped to three strictness regimes — strict (only BENIGN is safe), moderate (BENIGN+LOW are safe), loose (BENIGN+LOW+MODERATE are safe). Covers seven risk categories (violence/illegal/sexual/privacy/discrimination/misinformation/jailbreak), containing 2K prompt instances and 2K response instances. Uses a human-AI collaborative annotation pipeline where LLMs generate candidate labels, five human annotators verify and correct, and disagreements are resolved by senior annotators
    - Design Motivation: Existing benchmarks use fixed binary labels and cannot evaluate model robustness when strictness changes

2. **Rubric-Guided Distillation Pipeline**:

    - Function: Generate pseudo-labels for training continuous risk scores
    - Mechanism: Use expert-designed scoring rubrics to guide a strong LLM (e.g., GPT-5) to generate risk category $c(x)$, score $r'(x) \in [0, 100]$, and reasoning for each instance. The key step is label consistency calibration — aligning LLM scores with source dataset binary labels by linearly mapping raw scores $r'(x)$ to label-consistent intervals (safe: [0,40], unsafe: [40,100]), suppressing cross-boundary outliers
    - Design Motivation: Public moderation corpora mostly have only binary labels, and direct continuous score annotation is prohibitively expensive; LLM distillation can generate at scale, and the calibration step ensures consistency with existing labels

3. **Two-Stage Risk Alignment Training**:

    - Function: Train the model to produce continuous scores consistent with risk severity
    - Mechanism: Stage 1 uses LoRA SFT warmup to teach the model to follow rubric reasoning and output formatted $(\hat{c}(x), \hat{r}(x))$; Stage 2 uses GRPO reinforcement learning with a dense reward $R(x) = s_{\text{category}}(x) + s_{\text{score}}(x)$, where category accuracy reward $s_{\text{category}} \in \{-1, +1\}$ and score regression reward $s_{\text{score}} = 2 - \frac{4}{E_{\max}} |\hat{r}(x) - r(x)| \in [-2, 2]$, with $E_{\max}$ normalizing to make errors across different target scores comparable
    - Design Motivation: SFT provides stable initialization; GRPO directly optimizes the score consistency objective; the dense linear regression reward provides richer gradient signals than binary rewards

### Loss & Training

Two-stage training: Stage 1 standard SFT with LoRA; Stage 2 GRPO with combined category accuracy + score regression dense reward. Trained on 8×H20 GPUs.

## Key Experimental Results

### Main Results

**FlexBench Strictness-Adaptive Moderation (Harmfulness F1 %)**

| Method | Prompt Avg | Prompt Worst | Response Avg | Response Worst |
|---|---|---|---|---|
| GPT-5 | 73.26 | 70.95 | 77.43 | 74.07 |
| Qwen3Guard-8B | 75.10 | 67.06 | 76.61 | 69.16 |
| BingoGuard-8B | 74.22 | 68.31 | 76.59 | 74.80 |
| **FlexGuard (calibrated threshold)** | **81.78** | **78.26** | **80.29** | **75.81** |

### Ablation Study

| Config | Key Metric | Note |
|---|---|---|
| FlexGuard full | Avg 81.78 / Worst 78.26 | Best |
| SFT only (no GRPO) | Degraded | Insufficient score-severity consistency |
| No label consistency calibration | Degraded | More cross-boundary outliers |
| Rubric threshold (no calibration) | 80.29 / 76.63 | Still competitive |

### Key Findings

- FlexGuard's cross-strictness performance drop is significantly lower than competitors: Prompt best-worst gap is only 5.73%, vs. 15.95% for Qwen3Guard and 13.52% for BingoGuard
- Rubric thresholds achieve competitive performance without a validation set (Prompt Avg 80.29); calibrated thresholds further improve by ~1.5%
- On public benchmarks (no strictness variation), FlexGuard also achieves or exceeds SOTA (Prompt Avg 85.36, Response Avg 87.85)
- The GRPO stage significantly improves score quality: score MAE decreases, and score distributions across severity levels become more separated

## Highlights & Insights

- Redefining content moderation from a "classification problem" to a "risk assessment problem": the continuous scoring + threshold selection design elegantly decouples model capability from deployment requirements
- Label consistency calibration is a critical technical detail — aligning LLM-distilled scores with existing binary labels resolves pseudo-label quality issues
- The dense linear regression reward design (rather than common binary rewards) provides richer gradient signals for GRPO

## Limitations & Future Work

- FlexBench currently supports English only; strictness adaptation behavior in multilingual settings is unknown
- Three strictness levels may not be granular enough — real deployments may require more continuous control
- Interpretability of continuous scores needs strengthening — users may need to understand what scores mean
- Robustness of scores under adversarial inputs (jailbreak attacks) has not been tested

## Related Work & Insights

- **vs LlamaGuard/WildGuard**: These models output binary labels and adapt strictness via logit thresholds with poor results; FlexGuard natively outputs continuous scores
- **vs BingoGuard/PKU-SafeRLHF**: Output discrete severity levels with limited granularity; FlexGuard's continuous scoring provides finer risk discrimination

## Rating

- Novelty: ⭐⭐⭐⭐ Novel and practical problem definition (strictness-adaptive moderation); the continuous scoring approach is natural and sound
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Custom benchmark + public benchmarks, multi-baseline comparison, complete ablations
- Writing Quality: ⭐⭐⭐⭐ Clear structure, well-motivated problem; some details could be more concise
- Value: ⭐⭐⭐⭐⭐ Directly addresses industrial deployment pain points; FlexBench could become a new standard for moderation evaluation

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Can LLMs Write Faithfully? An Agent-Based Evaluation of LLM-generated Islamic Content](../../NeurIPS2025/aigc_detection/can_llms_write_faithfully_an_agent-based_evaluation_of_llm-generated_islamic_con.md)
- [\[ACL 2026\] Temporal Flattening in LLM-Generated Text: Comparing Human and LLM Writing Trajectories](temporal_flattening_in_llm-generated_text_comparing_human_and_llm_writing_trajec.md)
- [\[ACL 2026\] Who Wrote This Line? Evaluating the Detection of LLM-Generated Classical Chinese Poetry](who_wrote_this_line_evaluating_the_detection_of_llm-generated_classical_chinese_.md)
- [\[ACL 2026\] BIASEDTALES-ML: A Multilingual Dataset for Analyzing Narrative Attribute Distributions in LLM-Generated Stories](biasedtales-ml_a_multilingual_dataset_for_analyzing_narrative_attribute_distribu.md)
- [\[ACL 2026\] Beyond the Final Actor: Modeling the Dual Roles of Creator and Editor for Fine-Grained LLM-Generated Text Detection](beyond_the_final_actor_modeling_the_dual_roles_of_creator_and_editor_for_fine-gr.md)

<!-- RELATED:END -->
