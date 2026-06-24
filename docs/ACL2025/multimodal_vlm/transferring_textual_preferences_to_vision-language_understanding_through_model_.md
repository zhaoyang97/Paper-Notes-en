---
title: >-
  [Paper Note] Transferring Textual Preferences to Vision-Language Understanding through Model Merging
description: >-
  [ACL 2025][Multimodal VLM][Model Merging] This work proposes a training-free method to transfer the preference capability of text-only reward models (RMs) into Large Vision-Language Models (LVLMs) through model parameter merging, building a Vision-Language Reward Model (VLRM) that outperforms direct LVLM scoring and text-only RMs across multiple multimodal evaluation benchmarks.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Model Merging"
  - "Vision-Language Reward Model"
  - "Preference Transfer"
  - "Training-Free"
  - "RLHF"
date: 2026-05-08
content_hash: af6259c2f4a1b974
---

# Transferring Textual Preferences to Vision-Language Understanding through Model Merging

**Conference**: ACL 2025  
**arXiv**: [2502.13487](https://arxiv.org/abs/2502.13487)  
**Authors**: Chen-An Li, Tzu-Han Lin, Yun-Nung Chen, Hung-yi Lee (National Taiwan University)  
**Code**: [GitHub](https://github.com/lca0503/MergeToVLRM)  
**Area**: Multimodal VLM  
**Keywords**: Model Merging, Vision-Language Reward Model, Preference Transfer, Training-Free, RLHF  

## TL;DR

This work proposes a training-free method to transfer the preference capability of text-only reward models (RMs) into Large Vision-Language Models (LVLMs) through model parameter merging, building a Vision-Language Reward Model (VLRM) that outperforms direct LVLM scoring and text-only RMs across multiple multimodal evaluation benchmarks.

## Background & Motivation

### Background
Large Vision-Language Models (LVLMs) perform exceptionally well on multimodal tasks, but their ability to evaluate the quality of generated content remains limited. Training specialized Vision-Language Reward Models (VLRMs) requires collecting expensive multimodal preference data and performing large-scale training, which is extremely costly. Meanwhile, the textual domain has accumulated rich preference datasets and high-quality text reward models.

### Limitations of Prior Work
- Existing VLRM training relies heavily on large amounts of multimodal preference data, making collection and labeling costs extremely high.
- Direct scoring using LVLMs (LVLM-as-a-Judge) performs poorly on challenging tasks such as VL-RewardBench, especially in scoring and batch ranking tasks.
- Although text RMs exhibit strong preference judgment capabilities, they completely lack visual understanding and cannot handle multimodal scenarios.
- Cascaded approaches (describing the image with an LVLM first, then scoring with a text RM) suffer from information loss during transmission.

### Design Motivation
The language modules of many LVLMs (such as Llama-3.2-Vision) are inherently constructed by extending pre-trained language models with visual encoders and adapters. This architectural characteristic implies that if a text RM is also derived from the same pre-trained language model, their transformer parameters reside in the same parameter space. Consequently, their respective capabilities can be combined through parameter merging—with the LVLM providing visual understanding and the text RM providing preference judgment.

## Method

### Overall Architecture

The core idea is extremely simple: merge the parameters of the shared transformer layers of the LVLM and the text RM, while retaining the LVLM's visual encoder and adapter, and keeping the text RM's reward head to assemble a complete VLRM.

Specifically, the merged VLRM is composed of five parts:
$$\theta^{\text{MERGE}} = \{\theta^{\text{LVLM}}_{\text{venc}}, \theta^{\text{LVLM}}_{\text{adapt}}, \theta^{\text{MERGE}}_{\text{emb}}, \theta^{\text{MERGE}}_{\text{trans}}, \theta^{\text{RM}}_{\text{rm}}\}$$

- **Visual Encoder and Adapter**: Completely sourced from the LVLM, preserving visual understanding capabilities.
- **Embedding Layer and Transformer Layers**: Fused using a model merging strategy.
- **Reward Head**: Completely sourced from the text RM, mapping hidden states into scalar reward values.

The prerequisite is that both models must share the same pre-trained language model base (both based on Llama-3.1-8B in this paper).

### Key Designs

#### 1. Four Merging Strategies

**Weighted Average (Linear)**: The most direct merging method, which linearly combines the transformer parameters of both models using a weight $\lambda$. It is simple but can lead to parameter interference.

**Task Arithmetic**: First calculates the "task vectors" (i.e., parameter increments) of the LVLM and the RM relative to the pre-trained model individually, then overlays both task vectors onto the pre-trained parameters proportionally. This approach avoids the parameter cancellation issue caused by direct averaging.

**TIES**: Adds three steps on top of Task Arithmetic—pruning small-magnitude parameters, resolving sign conflicts (retaining parameters in the direction of the larger cumulative magnitude), and averaging the retained parameters. The retention ratio is controlled via a density parameter $d$.

**DARE**: Randomly drops delta parameters in the task vectors with probability $p$, and rescales the remaining parameters by a factor of $1/(1-p)$. It can be combined with Task Arithmetic or TIES.

#### 2. Embedding Layer Merging Strategy

Since LVLMs and RMs may have different vocabulary extensions (e.g., the LVLM adds visual tokens), merging the embedding layer requires special handling. The strategy from MergeKit is adopted:
1. Tokens already present in the pre-trained model use the pre-trained embedding.
2. Tokens appearing in only one of the models directly use that model's embedding.
3. Tokens appearing in both models take the average of their embeddings.

#### 3. Hyperparameter Selection

A validation set of 400 instances sampled from the RLAIF-V training set is used for hyperparameter search. For Linear and Task Arithmetic, the search space is $\lambda \in [0.0, 0.1, ..., 1.0]$. For TIES and DARE, the search space is $\lambda \in [0.5, 0.7, 1.0]$ and $d \in [0.2, 0.4, 0.6, 0.8]$.

## Key Experimental Results

### Experiment 1: Main Benchmark Results (Merged with Tulu-2.5-RM)

| Method | VL-RB General | VL-RB Hallucination | VL-RB Reasoning | VL-RB Overall | TextVQA | MMMU-Pro Std | MMMU-Pro Vision |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Llama-3.2-Vision | 33.3 | 38.4 | 56.6 | 42.9 | 46.4 | 28.8 | 19.8 |
| Tulu-2.5-RM (Text-only) | 43.2 | 31.4 | 54.1 | 38.9 | 42.6 | 29.8 | 21.4 |
| Cascade | 44.8 | 37.8 | 57.2 | 43.8 | 43.2 | 30.9 | 23.4 |
| Linear | 39.3 | 52.3 | 54.4 | 51.0 | 54.7 | 27.8 | 22.1 |
| Task Vec. | 48.6 | 59.4 | 59.7 | 57.9 | 59.0 | 31.0 | 22.7 |
| TIES | 43.7 | 58.2 | 58.5 | 56.2 | **64.2** | 29.1 | 22.6 |
| DARE + Task Vec. | **49.2** | **61.7** | **61.0** | **59.7** | 58.8 | 30.3 | 22.4 |
| DARE + TIES | 49.2 | 59.1 | 58.2 | 57.4 | 57.3 | **31.6** | 22.0 |

The merged VLRM significantly improves the Overall score on VL-RewardBench from 42.9% to 59.7% (+16.8) and on TextVQA from 46.4 to 64.2 (+17.8).

### Experiment 2: Comparison with Large-Scale and Commercial Models

| Method | VL-RB General | VL-RB Hallucination | VL-RB Reasoning |
|------|:---:|:---:|:---:|
| Llama-3.2-Vision (11B) | 33.3 | 38.4 | 56.6 |
| Llama-3.2-Vision (90B) | 42.6 | 57.3 | 61.7 |
| GPT-4o-mini | 41.7 | 34.5 | 58.2 |
| Gemini-1.5-Flash | 47.8 | 59.6 | 58.4 |
| Gemini-1.5-Pro | 50.8 | 72.5 | 64.2 |
| GPT-4o | 49.1 | 67.6 | 70.5 |
| DARE + Task Vec. (Ours) | 49.2 | 61.7 | 61.0 |

The merged 11B VLRM outperforms the 90B LVLM and is competitive with GPT-4o and Gemini-1.5-Pro on the General and Hallucination dimensions.

### Experiment 3: Ablation of the Visual Encoder (Removing Image Input)

| Method | VL-RB (with Image) | VL-RB (without Image) | TextVQA (with Image) | TextVQA (without Image) |
|------|:---:|:---:|:---:|:---:|
| Task Vec. | 57.9 | 44.9 | 59.0 | 38.7 |
| DARE + Task Vec. | 59.7 | 44.5 | 58.8 | 36.2 |
| TIES | 56.2 | 42.7 | 64.2 | 40.9 |

Removing the images leads to a steep decline in performance (a drop of approximately 13-15 points on VL-RB and 20-28 points on TextVQA), proving that the merged VLRM indeed leverages the visual encoder rather than solely relying on the textual RM.

## Key Findings

- Advanced merging strategies (Task Arithmetic, TIES, DARE) clearly outperform simple weighted averaging, indicating that addressing parameter interference is critical.
- Even when task vectors are pruned to retain only 20%-40% of the parameters, the merged model maintains robust performance, aligning with existing findings in the LLM merging domain.
- Optimal hyperparameters vary across benchmarks—VL-RewardBench is insensitive to $\lambda$, whereas MMMU-Pro achieves its best performance at $\lambda=1.0$.
- The merging method outperforms the cascaded approach (Cascade), indicating that direct fusion in the parameter space captures more information than the "describe image then score" pipeline.

## Highlights & Insights

- **Minimalist yet Effective Design Philosophy**: The entire framework requires zero training and zero multimodal preference data; capabilities are combined solely via parameter merging, with computational overhead low enough to be handled entirely on a CPU.
- **In-depth Architectural Insight**: By exploiting the fact that the LVLM language module and the text RM share the same pre-trained base, cross-modal preference transfer is converted into vector operations within the exact same parameter space.
- **High Practical Value**: The entire merging process completes in 1.5–6 hours on CPU, and validation inference takes only about 1.5 hours of GPU time, which is substantially lower than the cost of training a VLRM from scratch.
- **11B Merged Model Defeats 90B Model**: Demonstrates that composing capabilities is more efficient than simply scaling up model size.

## Limitations & Future Work

- **Constrained Model Architectures**: Only validated on the LLaMA architecture, which requires the LVLM and the RM to share the same pre-trained base, limiting its general applicability.
- **Single Scale**: Only the 11B LVLM + 8B RM combination was tested, leaving the effects of larger or smaller models unexplored.
- **Sensitivity to Hyperparameters**: The optimal hyperparameters vary by task, requiring a carefully constructed validation set for selection, which adds extra cost in practical deployment.
- **No Comparison with Training-Based Methods**: Lacks a systematic comparison with VLRMs trained directly on multimodal preference data.
- **No RLHF Integration Experiments**: The merged VLRM was not utilized in RLHF algorithms such as PPO, leaving its end-to-end effectiveness unevaluated.

## Related Work & Insights

This work lies at the intersection of **model merging** and **reward models**. Similar to DogeRM (Lin et al., 2024), which injects domain knowledge into RMs through merging, this work operates in the opposite direction—injecting RM capabilities into an LVLM across modalities. REMEDY (Zhu et al., 2025) also studies LVLM merging, but focuses on merging among homogeneous LVLMs rather than heterogeneous models (LVLM + RM).

**Insight**: This paradigm of "borrowing existing capabilities to inject them training-free via merging" can be generalized—for example, merging code-generation RMs into general LLMs, or security-alignment RMs into domain-specific models. The key prerequisite is a shared base model.

## Rating

- Novelty: ⭐⭐⭐⭐ — The perspective of cross-modal training-free preference transfer is novel, though the merging techniques themselves are existing.
- Experimental Thoroughness: ⭐⭐⭐⭐ — The multiple benchmarks, ablation experiments, and hyperparameter analyses are comprehensive, but direct comparison with training-based approaches is missing.
- Writing Quality: ⭐⭐⭐⭐ — The structure is clear, the motives are well-formulated, and qualitative case studies assist in understanding.
- Value: ⭐⭐⭐⭐ — Highly practical, providing a viable, low-cost path to building VLRMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Value-Spectrum: Quantifying Preferences of Vision-Language Models via Value Decomposition](value_spectrum_vlm_pref.md)
- [\[NeurIPS 2025\] RobustMerge: Parameter-Efficient Model Merging for MLLMs with Direction Robustness](../../NeurIPS2025/multimodal_vlm/robustmerge_parameter-efficient_model_merging_for_mllms_with_direction_robustnes.md)
- [\[CVPR 2026\] Beyond Layer-Wise Merging: Chain-of-Merging for Vision-Language Models](../../CVPR2026/multimodal_vlm/beyond_layer-wise_merging_chain-of-merging_for_vision-language_models.md)
- [\[ACL 2025\] Jailbreak Large Vision-Language Models Through Multi-Modal Linkage](jailbreak_large_vision-language_models_through_multi-modal_linkage.md)
- [\[CVPR 2025\] Cropper: Vision-Language Model for Image Cropping through In-Context Learning](../../CVPR2025/multimodal_vlm/cropper_vision-language_model_for_image_cropping_through_in-context_learning.md)

</div>

<!-- RELATED:END -->
