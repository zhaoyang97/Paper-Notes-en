---
title: >-
  [Paper Note] Training-free LLM Merging for Multi-task Learning
description: >-
  [ACL 2025][LLM (Other)][Model Merging] This paper proposes Hi-Merging, a **hierarchical iterative training-free model merging** method. By utilizing model-wise and layer-wise pruning and scaling operations combined with contribution analysis, it identifies and resolves parameter conflicts. This merges specialized LLMs across different tasks/languages into a single unified multi-task model, outperforming mixed-data fine-tuning baselines in most scenarios.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Model Merging"
  - "Multi-Task Learning"
  - "Layer-wise Pruning"
  - "Conflict Elimination"
  - "Training-free"
date: 2026-05-08
content_hash: 3142419ae794c400
---

# Training-free LLM Merging for Multi-task Learning

**Conference**: ACL 2025  
**arXiv**: [2506.12379](https://arxiv.org/abs/2506.12379)  
**Code**: [GitHub](https://github.com/Applied-Machine-Learning-Lab/Hi-Merging)  
**Area**: LLM Model Merging  
**Keywords**: Model Merging, Multi-Task Learning, Layer-wise Pruning, Conflict Elimination, Training-free

## TL;DR

This paper proposes Hi-Merging, a **hierarchical iterative training-free model merging** method. By utilizing model-wise and layer-wise pruning and scaling operations combined with contribution analysis, it identifies and resolves parameter conflicts. This merges specialized LLMs across different tasks/languages into a single unified multi-task model, outperforming mixed-data fine-tuning baselines in most scenarios.

## Background & Motivation

With the release of open-source large language models like LLaMA and Qwen, there are now over 1 million specialized LLMs fine-tuned for various tasks and languages on Hugging Face. A natural question arises: **Can these specialized models be merged into a single unified multi-task model?**

The direct solution is to collect all fine-tuning data and retrain them, which faces three major difficulties:
1. **Data Inaccessibility**: Models are publicly available, but fine-tuning data is often proprietary.
2. **High Computational Cost**: Retraining LLMs requires massive computing resources.
3. **Seesaw Effect**: When training on mixed data, improving performance on one task often degrades performance on another.

**Model Merging** has thus become an attractive alternative, but existing methods face two core challenges:
- **Noise Interference**: Noise parameters introduced during fine-tuning due to data bias or overfitting can impair the generalization of the merged model.
- **Knowledge Misalignment**: Independently trained models follow different optimization trajectories, leading to mismatched knowledge alignment in the parameter space. Direct merging causes incompatibility.

Existing methods, such as TIES-Merging and DARE, lack explicit guidance for conflict localization, leading to high performance variance. The proposed Hi-Merging systematically addresses these issues through layer-wise analysis.

## Method

### Overall Architecture

Hi-Merging adopts a two-stage hierarchical processing architecture:
1. **Model-wise Pruning & Scaling**: De-noising and scaling the entire delta vector of each fine-tuned model.
2. **Layer-wise Pruning & Scaling**: Identifying the layers with the most severe conflicts through contribution analysis and iteratively eliminating parameter conflicts.

The core mathematical foundation is the delta vector: $\boldsymbol{\delta}_m = \boldsymbol{\theta}_m - \boldsymbol{\theta}_F$, which represents the parameter difference between the fine-tuned model and the base model.

### Key Designs

1. **Model-wise Pruning & Scaling**:
    - **Pruning threshold $p$**: Retains the top $p\%$ parameters in the delta vector with the largest absolute values and sets the rest to zero, eliminating noise parameters introduced by data bias.
    - **Scaling factor $s$**: Multiplies the retained delta vector by $s \in [0,1]$, mitigating overly aggressive parameter updates caused by overfitting.
    - Experimental validation: Setting $p=0.1, s=0.9$ (retaining only 10% of parameters and scaling by 0.9) already outperforms the original model.
    - Complementary operations: Pruning eliminates minor perturbations, while scaling regulates extreme parameter changes.

2. **Contribution Analysis**:
    - **Pruning Impact $\alpha$**: Constructs an initial merged model ℳ_G and measures the performance drop of ℳ_m on its original task when removing the delta vector of a specific layer.
    - **Addition Impact $\beta$**: Adds the delta vector of a specific layer to the base model ℳ_F and measures the performance gain of ℳ_m on its original task.
    - Total contribution $c = \alpha + \beta$. Conflict degree $\gamma_m^l = c_{m,m}^l - c_{m,G}^l$.
    - Sort layers by $\Gamma^l = \sum_m \gamma_m^l$ to identify the most conflict-prone layers.

3. **Iterative Conflict Elimination**: Processes each layer in descending order of conflict severity, categorized into three cases:
    - **Severe Conflict** ($\gamma_A > 0$ and $\gamma_B > 0$): Both capabilities are degraded by merging $\rightarrow$ keep only the delta vector with the larger contribution and zero out the other.
    - **Partial Conflict** ($\gamma_A \cdot \gamma_B < 0$): One model's overfitting degrades the other $\rightarrow$ prune and scale the conflicting model's delta vector further.
    - **Mutual Enhancement** ($\gamma_A \leq 0$ and $\gamma_B \leq 0$): Both capabilities improve after merging $\rightarrow$ no adjustment needed.

### Loss & Training

**Completely training-free**. Hi-Merging is a parameter post-processing method:
- Base Model: Qwen2-7B-Instruct
- Fine-tuning: LLaMA-Factory + LoRA (rank=8, alpha=16, dropout=0.01)
- Merging Tool: mergekit
- Model-wise $p$ and $s$ searched in the range of 0.1 to 1.0 (step size 0.1).
- Layer-wise $p$ and $s$ are set to half of the model-wise values.
- Evaluation Metrics: Accuracy for MCQA, BLEU-4 and ROUGE-1/2/L for QA.

## Key Experimental Results

### Main Results

**Bilingual MCQA Task Merging (English MedQA + Chinese CMExam):**

| Method | MedQA (Acc) | CMExam (Acc) | Avg Impr. | Avg Rank |
|------|------------|-------------|-----------|----------|
| Qwen2-7B Base | 51.41 | 74.62 | - | 17.0 |
| Single-Task Fine-tuning A (English) | 59.14 | 83.78 | +13.40% | 10.0 |
| Mixed-Data Fine-tuning | 60.08 | 88.22 | +17.67% | 3.5 |
| Task Arithmetic | 59.53 | 88.77 | +17.67% | 4.0 |
| TIES | 59.06 | 88.78 | +17.31% | 4.5 |
| DARE | 58.67 | 88.69 | +16.93% | 7.5 |
| **Hi-Merging** | **60.16** | **89.07** | **+18.41%** | **1.0** |

**Monolingual Multi-task Merging (English MCQA+QA):**

| Method | MedQA Acc | HCMagic BLEU-4 | HCMagic ROUGE-L | Avg Impr. | Avg Rank |
|------|----------|---------------|----------------|-----------|----------|
| Mixed-Data Fine-tuning | 59.22 | 35.60 | 20.46 | +25.23% | 8.3 |
| TIES | 60.47 | 35.79 | 20.37 | +26.78% | 4.2 |
| DARE | 58.44 | 36.58 | 20.39 | +26.29% | 4.4 |
| **Hi-Merging** | 60.16+ | Best-level | Best-level | **Best** | **1.0** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Model-wise processing only | Avg Rank ~4 | Outperforms no-processing, but inferior to the full hierarchical method |
| Layer-wise processing only (no model-wise) | Avg Rank ~3 | Lacks global de-oising, limiting the layer-wise optimization space |
| Full Hi-Merging | Avg Rank 1.0 | Achieves the best hierarchical coordination effect |
| $p=0.1, s=0.9$ (single model) | Outperforms original | Validates that pruning + scaling is also beneficial for single models |
| Different base models (Yi-1.5-9B, Baichuan2-7B) | Effective but baselines differ | The method is robust to the choice of base models |

### Key Findings

1. **Hi-Merging Consistently Ranks First**: Across the three settings (bilingual MCQA, monolingual multi-task, and cross-lingual cross-task), the average rank is consistently 1.0.
2. **Outperforming Mixed-data Fine-tuning**: Under most scenarios, the training-free Hi-Merging outperforms the mixed-data fine-tuning baseline which requires extra training.
3. **High Variance in Existing Methods**: TIES and DARE occasionally perform best on individual metrics but exhibit overall instability due to the lack of guidance.
4. **10% Parameters Suffice for Performance Preservation**: When pruned to retain only 10% of delta parameters, proper scaling still maintains or even enhances performance.
5. **Severe Conflict Layers Identified and Resolved**: Contribution analysis effectively identifies the most problematic layers in merging, and the three corresponding conflict elimination strategies are highly targeted.

## Highlights & Insights

- **Value of Hierarchical Thinking**: Decomposing the global model merging problem into model-wise de-noising followed by layer-wise conflict elimination makes the problem more analytical and controllable.
- **Innovation in Contribution Analysis**: Quantifying the conflict severity of each layer by simultaneously measuring "pruning impact" and "addition impact" is more direct than statistical-based methods.
- **Categorized Treatment of Three Conflict Types**: The categorization of severe conflict, partial conflict, and mutual enhancement is intuitive and provides targeted resolution strategies.
- **Complementarity of Pruning and Scaling**: Pruning removes minor noise, while scaling adjusts large parameters. Their complementarity addresses two common issues in the fine-tuning process.
- **High Practical Usability**: Implemented based on mergekit with a reasonable hyperparameter search space (10x10 grid), making it friendly to the open-source community.

## Limitations & Future Work

1. **Primarily Explores Two-Model Merging**: Although the framework can theoretically scale to multiple models, experiments are mainly pairwise, and multi-model scenarios have not been fully validated.
2. **Limited Task Types**: Validated only on MCQA and QA tasks in the medical domain, without covering other critical tasks like code generation and reasoning.
3. **Computational Overhead of Contribution Analysis**: It requires pruning/addition experiments and performance evaluation for every single layer. The overhead is non-negligible when scaling up model and task combinations.
4. **LoRA Fine-tuning Assumption**: The experiments utilize LoRA for fine-tuning. The effectiveness of merging full-parameter fine-tuned models is yet to be validated.
5. Future work can explore **adaptive selection of $p$ and $s$** to lower the hyperparameter search overhead.

## Related Work & Insights

- **Task Arithmetic (Ilharco et al., 2023)**: Proposes the fundamental framework of delta vector merging, on top of which Hi-Merging incorporates layer-wise optimization.
- **TIES-Merging (Yadav et al., 2023)** and **DARE (Yu et al., 2024)**: Mitigate parameter conflicts through various strategies, but struggle with explicit conflict localization.
- **DELLA (Deep et al., 2024)**: Incorporates parameter magnitude, but remains a global processing approach.
- **Model Breadcrumbs (Davari & Belilovsky, 2024)**: Progressively sparsifies parameters without involving layer-wise analysis.
- **Layer Swapping (Bandarkar et al., 2025)**: Utilizes a layer swapping strategy but lacks fine-grained conflict analysis.
- **Insights**: Model merging should not be a brute-force, single-step operation. Layer-wise analysis and iterative optimization can significantly reduce parameter conflicts.

## Rating

- Novelty: ⭐⭐⭐⭐ The hierarchical conflict analysis and elimination framework is highly novel in the model merging domain, with an innovative contribution analysis method.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers three merging scenarios (bilingual, multi-task, and cross-lingual cross-task), compared against 10+ baselines, including detailed ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear mathematical derivations, intuitive visualization of the three conflict types, and a well-structured layout.
- Value: ⭐⭐⭐⭐ This training-free method outperforms training-based baselines, providing substantial practical guidance for model integration in the LLM community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] A Training-free LLM-based Approach to General Chinese Character Error Correction](a_training-free_llm-based_approach_to_general_chinese_character_error_correction.md)
- [\[ICCV 2025\] FW-Merging: Scaling Model Merging with Frank-Wolfe Optimization](../../ICCV2025/llm_nlp/fw-merging_scaling_model_merging_with_frank-wolfe_optimization.md)
- [\[ACL 2025\] Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](token_prepending_training_free.md)
- [\[ACL 2025\] Cheaper and Better Diffusion Language Model via Task-Specific Training](cheaper_and_better_diffusion_language_model_via_task-specific_training.md)
- [\[ACL 2025\] Meta-Reflection: A Feedback-Free Reflection Learning Framework](meta-reflection_a_feedback-free_reflection_learning_framework.md)

</div>

<!-- RELATED:END -->
