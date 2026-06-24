---
title: >-
  [Paper Note] One for All: Update Parameterized Knowledge Across Multiple Models with Once Edit
description: >-
  [ACL 2025][LLM (Other)][Knowledge Editing] OnceEdit is proposed to update knowledge across multiple LLMs through "once edit, multi-model update" by editing a lightweight plug-in model and utilizing heterogeneous model ensemble techniques to transfer the edited knowledge. It significantly outperforms existing methods on the ZsRE and Counterfact datasets.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Knowledge Editing"
  - "Model Ensemble"
  - "Multi-Model Update"
  - "Parameterized Knowledge"
  - "LLM"
date: 2026-05-08
content_hash: 69c3f62ef0b54223
---

# One for All: Update Parameterized Knowledge Across Multiple Models with Once Edit

**Conference**: ACL 2025  
**arXiv**: [2506.00817](https://arxiv.org/abs/2506.00817)  
**Code**: Yes (to be released soon)  
**Area**: Others  
**Keywords**: Knowledge Editing, Model Ensemble, Multi-Model Update, Parameterized Knowledge, LLM

## TL;DR

OnceEdit is proposed to update knowledge across multiple LLMs through "once edit, multi-model update" by editing a lightweight plug-in model and utilizing heterogeneous model ensemble techniques to transfer the edited knowledge. It significantly outperforms existing methods on the ZsRE and Counterfact datasets.

## Background & Motivation

Although large language models (LLMs) encode a vast amount of world knowledge during the pre-training phase, this knowledge gradually becomes outdated as real-world information dynamically changes, leading to errors and hallucinations. Knowledge Editing provides a more efficient knowledge-updating alternative than retraining by directly modifying specific parameters in the model to update knowledge.

However, existing knowledge editing methods face two core limitations:

**Editing restricted to a single model**: Methods like ROME, MEMIT, and MEND are designed for a single model. In practical scenarios, an organization might deploy multiple distinct LLMs (e.g., Llama2, Mistral, GPT-J). When a specific factual knowledge needs to be updated, editing operations must be executed separately for each model. This is not only computationally expensive but also requires tuning hyperparameters for each individual model.

**Unstable cross-model performance**: Existing methods are highly sensitive to hyperparameter settings, resulting in significantly different editing performances across different models. For instance, MEMIT performs well on GPT-J-6B but shows mediocre results on Llama2-7B, which limits its scalability to new models.

The key insight of OnceEdit is: **instead of performing an edit for each model, it is better to edit a unified small model and then transfer the edited knowledge to all target models via model ensemble**.

## Method

### Overall Architecture

OnceEdit consists of two stages:

1. **Editing Stage**: Edits knowledge on a lightweight plug-in model (e.g., TinyLlama).
2. **Ensemble Stage**: Ensembles the edited plug-in model with various target LLMs to achieve knowledge transfer.

Based on DEEPEN (a heterogeneous model ensemble method), two key improvement mechanisms are introduced.

### Key Designs

#### 1. **Dynamic Weight Mechanism**

**Function**: Dynamically assigns ensemble weights of the plug-in model and target LLM for each input instance.

**Core Problem**: Traditional ensemble methods use fixed weights, but in the knowledge editing scenario, **edit-related** inputs should rely more on the plug-in model's knowledge, while **irrelevant inputs** should depend on the original LLM.

**Solution**: A special token `[WEIGHT]` is introduced into the vocabulary of the plug-in model, and the sigmoid of this token's output logit is used as the ensemble weight $\alpha$:

$$\alpha = \phi(\text{logit}_w(x))$$

During training, both the generation loss and the weight prediction loss are optimized simultaneously:

$$\mathcal{L}_{\text{edit}}(\theta) = \mathcal{L}_{\text{gen}}(\theta) + \lambda \cdot \mathcal{L}_{\text{weight}}(\theta)$$

where weight prediction uses Binary Cross-Entropy (BCE) loss, with a label of 1 for edit-related inputs and 0 for irrelevant inputs.

**Design Motivation**: Consequently, edit-related queries receive higher weights for the plug-in model, whereas irrelevant queries maintain their reliance on the original LLM, thereby preserving unedited knowledge while updating the target knowledge.

#### 2. **Ensemble Enhancement**

In decoding, the original DEEPEN uses the LLM's output distribution as the initial search point, treating the ensemble distribution as a "perturbation" of the LLM. This is feasible in traditional ensembles (since model outputs are similar), but in knowledge editing, the plug-in model and the LLM may have **completely different** output distributions—the LLM retains old knowledge, while the plug-in model has updated to new knowledge. Initializing with the LLM distribution causes decoding results to overly rely on old knowledge.

Two strategies:

**Search-space Zero Initialization**: Instead of initializing with the LLM's distribution, the search starts from a zero vector:

$$\textbf{p}_{init} = \text{zeros\_like}(\textbf{p}_l)$$

**Target Augmentation**: Converts the aggregated distribution into a one-hot vector (taking the token with the highest probability) to reinforce the sharpened expression of the fused knowledge:

$$\bar{\mathbf{P_o}} = \begin{cases} 1, & i = \arg\max_j \bar{\mathbf{P}}_j \\ 0, & \text{otherwise} \end{cases}$$

These two strategies work synergistically: zero initialization eliminates the bias towards old knowledge, and Target Augmentation ensures that the new knowledge has an explicit "vote" during decoding.

### Loss & Training

- Generation loss $\mathcal{L}_{\text{gen}}$: Standard language model cross-entropy
- Weight loss $\mathcal{L}_{\text{weight}}$: BCE loss, used to train the `[WEIGHT]` token to distinguish between edit-related and irrelevant inputs
- Both losses are balanced via a hyperparameter $\lambda$; experiments show that performance is insensitive to $\lambda$

## Key Experimental Results

### Main Results (Teacher-forced setting, 1,000 edits)

| Method | Llama2-7B Avg | Mistral-7B Avg | GPT-J-6B Avg | Overall Score | Edit Count |
|------|-------------|----------------|-------------|-----------|---------|
| FT-L | 0.23 | 0.50 | 0.23 | 0.32 | 3 |
| MEND | 0.00 | 0.00 | 0.00 | 0.00 | 3 |
| ROME | 0.05 | 0.01 | 0.01 | 0.02 | 3 |
| MEMIT | 0.69 | 0.77 | 0.89 | 0.78 | 3 |
| WISE | 0.87 | 0.77 | 0.81 | 0.82 | 3 |
| **OnceEdit** | **0.97** | **0.93** | **0.87** | **0.92** | **1** |

Results on the ZsRE dataset. OnceEdit requires only one edit operation to update three models, with the overall score leading the runner-up by over 10%.

### Ablation Study (Component Contributions)

| Method | Llama2-7B | Mistral-7B | GPT-J-6B | Total Score |
|------|-----------|------------|----------|------|
| DEEPEN (Baseline) | 0.57 | 0.47 | 0.18 | 0.41 |
| + Dynamic Weight (DW) | 0.88 | 0.78 | 0.46 | 0.71 |
| **+ DW + Ensemble Enhancement (EE)** | **0.96** | **0.91** | **0.86** | **0.91** |

ZsRE dataset. The dynamic weight mechanism primarily improves Locality (protecting unedited knowledge), while the ensemble enhancement mechanism significantly boosts Reliability and Generality.

### Extension Experiments (More/Larger Models)

| Model | ZsRE Avg | Counterfact Avg |
|------|---------|----------------|
| Llama3-8B | 0.93 | 0.69 |
| Mistral-7B-v0.3 | 0.94 | 0.69 |
| Qwen2.5-7B | 0.85 | 0.70 |
| Llama3-70B | 0.80 | 0.56 |

Using the same TinyLlama plug-in model, OnceEdit can stably edit various different LLMs, including models of 70B scale.

### Editing Efficiency Comparison

| Method | Llama2 | Mistral | GPT-J | Total (Normalized) |
|------|--------|---------|-------|------------|
| FT-L | 0.69x | 0.71x | 0.73x | 2.13x |
| DEFER | 1.49x | 1.47x | 1.40x | 4.36x |
| WISE | 1.35x | 1.33x | 1.26x | 3.94x |
| **OnceEdit** | **1x** | **1x** | **1x** | **1x** |

OnceEdit achieves the minimum total time for editing three models—other methods require 2 to 4 times longer.

### Key Findings

1. **Win-win on effectiveness and efficiency**: OnceEdit leads WISE (the runner-up) by 14% on ZsRE, while requiring only 1/3 of the editing operations.
2. **Strong cross-model stability**: Other methods (such as MEMIT) show highly fluctuating performance across different models/datasets (0.20 to 0.92), whereas OnceEdit remains consistent.
3. **Distinct contributions from both mechanisms**: The dynamic weight mechanism protects unedited knowledge (significant improvement in Locality), while the ensemble enhancement mechanism boosts editing effectiveness (significant improvement in Reliability/Generality).
4. **Plug-in model choice is influential but not critical**: TinyLlama outperforms Qwen2.5-1.5B, but both surpass all baselines.
5. **Low transfer cost**: Constructing the relative transformation matrix for a new model only incurs computational overhead equivalent to forward propagation over roughly 600–1000 tokens.
6. **Scalable to 70B models**: Although the performance decreases slightly as the model size increases, it still demonstrates feasibility.

## Highlights & Insights

- **Resolves a real-world engineering pain point**: "An organization deploys 5 different LLMs, and a certain fact needs to be updated"—OnceEdit reduces the editing frequency from 5 times to once.
- **Ingenious design of the `[WEIGHT]` token**: Employs a special token to predict "whether the current input is relevant to the edit," embedding routing decisions directly into the model itself rather than relying on external judgment.
- **Simple yet effective zero-initialization strategy**: Merely changing the initial search point from the LLM contribution to a zero vector significantly enhances the transmission of new knowledge—indicating that removing the prior is more crucial than adding information when knowledge conflicts exist.
- **Orthogonality of the method framework**: Editing the plug-in model in OnceEdit can be executed with any existing knowledge editing method (currently full fine-tuning is used), which can be replaced with more advanced methods in the future.

## Limitations & Future Work

1. Introducing a plug-in model increases inference overhead (requiring running two models simultaneously); although decoding can be parallelized, it still incurs costs.
2. Only tested under the batch editing setting, without exploring sequential editing and multi-hop editing.
3. Tuning the plug-in model currently relies on full fine-tuning, which might lead to knowledge degradation under a large volume of edits.
4. All methods exhibit relatively low Locality on the Counterfact dataset, which remains an inherent challenge for counterfactual data.
5. Performance degrades slightly on 70B models, suggesting that adapting to ultra-large-scale models warrants further investigation.

## Related Work & Insights

- **ROME/MEMIT**: Locate-then-edit methods based on causal tracing, which represent the most popular single-model editing approaches.
- **DEEPEN**: The fundamental method for heterogeneous model ensembles, on top of which OnceEdit adapts the setup for knowledge editing scenarios.
- **GRACE/WISE**: Memory-based editing methods that perform well in sequential editing but exhibit limited generalization.
- The core insight of OnceEdit is to redefine knowledge editing from "internal parameter modification of the model" to "external knowledge module + model ensemble fusion."

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — "Multi-model once edit" is a novel problem formulation, and the ensemble-based solution is refreshingly creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Includes 3+4 models, 2 datasets, multiple evaluation settings, and comprehensive efficiency analysis; however, sequential editing and multi-hop editing are lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, detailed method description, and well-designed figures/tables; however, some mathematical notations are dense.
- **Value**: ⭐⭐⭐⭐⭐ — Resolves the practical deployment demand for synchronous knowledge updates across multiple models, offering a highly generalizable and efficient approach.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Can we Retrieve Everything All at Once? ARM: An Alignment-Oriented LLM-based Retrieval Method](can_we_retrieve_everything_all_at_once_arm_an_alignment-oriented_llm-based_retri.md)
- [\[ACL 2025\] Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations](knowledge_boundary_crosslingual.md)
- [\[ACL 2025\] Efficient Ensemble for Fine-tuning Language Models on Multiple Datasets](efficient_ensemble_for_fine-tuning_language_models_on_multiple_datasets.md)
- [\[CVPR 2025\] MG-MotionLLM: A Unified Framework for Motion Comprehension and Generation across Multiple Granularities](../../CVPR2025/llm_nlp/mg-motionllm_a_unified_framework_for_motion_comprehension_and_generation_across_.md)
- [\[ACL 2025\] Knowledge Boundary of Large Language Models: A Survey](knowledge_boundary_survey.md)

</div>

<!-- RELATED:END -->
