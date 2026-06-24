---
title: >-
  [Paper Note] HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model
description: >-
  [ACL 2025][Multimodal VLM][Continual learning] Through CKA analysis, it is discovered that the top layer of MLLM learns task-specific information while the remaining layers learn general knowledge. This paper proposes HiDe-LLaVA: MoE-style task-specific expansion for top-layer LoRA (using dual-modal anchor matching) paired with uniform merging for LoRA in the remaining layers. On the newly constructed UCIT benchmark (free of information leakage)…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Continual learning"
  - "MLLM"
  - "instruction tuning"
  - "catastrophic forgetting"
  - "hierarchical decoupling"
  - "LoRA fusion"
date: 2026-05-08
content_hash: a83795b8df383c0e
---

# HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model

**Conference**: ACL 2025  
**arXiv**: [2503.12941](https://arxiv.org/abs/2503.12941)  
**Code**: [https://github.com/Ghy0501/HiDe-LLaVA](https://github.com/Ghy0501/HiDe-LLaVA)  
**Area**: LLM / NLP / Multimodal  
**Keywords**: Continual learning, MLLM, instruction tuning, catastrophic forgetting, hierarchical decoupling, LoRA fusion

## TL;DR
Through CKA analysis, it is discovered that the top layer of MLLM learns task-specific information while the remaining layers learn general knowledge. This paper proposes HiDe-LLaVA: MoE-style task-specific expansion for top-layer LoRA (using dual-modal anchor matching) paired with uniform merging for LoRA in the remaining layers. On the newly constructed UCIT benchmark (free of information leakage), it achieves a 5.8% improvement over the best baseline.

## Background & Motivation
**Background**: MLLM is adapted to specific tasks through instruction tuning. However, in practice, users need to continuously fine-tune the model on different datasets over time, requiring the model to learn new tasks without forgetting old ones (continual instruction tuning, CIT).

**Limitations of Prior Work**: (1) The existing benchmark CoIN suffers from **information leakage**—the downstream datasets overlap with the SFT stage of LLaVA, resulting in only a 13.19% gap between multi-task and zero-shot performance; (2) existing methods (such as MoELoRA, O-LoRA) lack theoretical guidance, making it unclear which layers should perform task isolation and which layers should share knowledge.

**Key Challenge**: Expanding all layers yields good performance but leads to parameter explosion (5.2$\times$); merging all layers keeps parameters minimal but suffers from performance degradation. There is a need to identify the exact boundaries between "which layers should be isolated and which should be shared."

**Goal**: (1) Correct the information leakage issues in existing benchmarks; (2) propose an efficient continual learning approach based on hierarchical analysis.

**Key Insight**: Using CKA similarity to analyze the output differences of each model layer across different tasks, it is found that only the top layer (layer 32) exhibits task-specificity, while the remaining layers are highly similar.

**Core Idea**: Perform MoE-style task-specific expansion only at the top layer, and directly merge the LoRA parameters uniformly for the remaining 31 layers, thereby achieving optimal continual learning performance with minimal parameter overhead.

## Method

### Overall Architecture
Based on LLaVA-v1.5-7b, a LoRA module (injected into all linear layers, rank=8) is trained for each task. Post-processing is conducted after training: (1) LoRAs in layers 1-31 are uniformly merged into a set of shared parameters; (2) layer 32 retains the LoRA modules of all tasks, which are selected via dual-modal anchor matching during inference.

### Key Designs

1. **CKA-Driven Hierarchical Decoupling**:

    - **Function**: Analyzes CKA similarities across LLaVA layers on 6 instruction-tuning datasets.
    - **Key Findings**: CKA similarity between different tasks drops drastically at the top layer (layer 32, task-specific), while layers 1-31 maintain high similarity (task-general).
    - **Validation**: Merging non-top-layer LoRA only + selecting top-layer correctly $\to$ optimal performance; choosing the incorrect top-layer $\to$ significant drop.
    - **Design Motivation**: CKA analysis provides a theoretical basis for "where to cut" rather than relying on empirical guesswork.

2. **Task-Specific Expansion (Top-Layer MoE)**:

    - **Function**: Extracts dual-modal feature anchors for each task during training (using the mean of the LLaVA vision encoder for vision, and the mean of the CLIP text encoder for text).
    - **Mechanism**: During inference, the cosine similarities $r_v^c, r_{ins}^c$ between the test input and each task's anchors are calculated. After a weighted combination, they are processed via softmax (T=0.1) to yield the weight $d_c$ for each task LoRA. The top-layer output is $O_{top} = \sum_{i=1}^T d_i E_i(\mathbf{h})$.
    - **Design Motivation**: Avoids training an additional router (which is required by MoELoRA) by directly utilizing the feature space of pretrained encoders for matching.

3. **Task-General Merging (Remaining Layers)**:

    - **Function**: Uniformly merges the LoRA parameters of all tasks in layers 1-31: $\bar{E}_T = \sum_{i=1}^T \epsilon_i E_i$, with coefficient $\epsilon_i = 1.0$.
    - **Design Motivation**: Knowledge in these layers is inherently task-general (validated by CKA), so simple merging does not lose much information while significantly reducing parameters.

### Loss & Training
Standard LoRA fine-tuning (1 epoch, LR=2e-4) without additional training overhead. Hierarchical decoupling is a post-processing operation. Parameters are only 44.27M (vs 229.62M for expanding all layers).

## Key Experimental Results

### Main Results (UCIT Benchmark - Last Metric)

| Method | Image-R | ArxivQA | IconQA | CLEVR | Average |
|------|---------|---------|--------|-------|------|
| FineTune | 37.63 | 72.33 | 41.7 | 35.63 | 48.12 |
| O-LoRA | 69.36 | 82.42 | 53.66 | 42.53 | 58.36 |
| MoELoRA | 49.87 | 77.63 | 46.40 | 36.47 | 52.06 |
| **HiDe-LLaVA** | **80.50** | **89.83** | **62.90** | **47.97** | **64.19** |

### Ablation Study

| Configuration | Last | Avg | Parameter Size |
|------|------|-----|--------|
| **HiDe-LLaVA** | **64.19** | **68.94** | **44.27M** |
| Merge all (1-32) | 61.26 | 65.43 | 38.27M |
| Expand all (1-32) | 67.62 | 70.91 | 229.62M (5.2$\times$) |
| Merge (1-31) only | 60.64 | 63.28 | 44.27M |

### Key Findings
- The gap between multi-task and zero-shot on the UCIT benchmark is 43.10% (compared to only 13.19% in CoIN), confirming the information leakage issue.
- Compared to the best baseline O-LoRA, HiDe-LLaVA improves the Last metric on UCIT by 5.8% (64.19 vs 58.36).
- Expanding the top layer only vs. expanding all layers: parameters are reduced by 5.2$\times$, while performance decreases by only 3.43 percentage points—extremely cost-effective.
- Dual-modal anchor matching outperforms single-modal: some tasks have similar images but different instructions, while others have similar instructions but different images.
- Robust to task order: performance fluctuations under different orders are $\le 1\%$.
- Merging coefficient $\epsilon=1.0$ (uniform) yields the best results.

## Highlights & Insights
- **CKA analysis provides an interpretable design basis**: Deciding which layers should be shared or isolated is not arbitrary but derived from quantitative analysis showing the uniqueness of the top layer. This "first analyze, then design" methodology is highly generalizable.
- **Extreme parameter efficiency**: Expanding just a single layer (increasing parameters from 38M to 44M, which is +6M) achieves performance close to expanding all layers (230M), demonstrating exceptional cost-effectiveness.
- **The discovery of information leakage** serves as a warning to the entire continual learning community—methods that seemingly "resolve" forgetting on CoIN might simply be exploiting pre-existing pretraining information.

## Limitations & Future Work
- Hierarchical decoupling is a post-processing operation and cannot exploit hierarchical information for optimization during training; optimizing with a hierarchical strategy directly during training can be a future research direction.
- Only validated on LLaVA-v1.5-7b; the CKA patterns in larger models or different architectures (such as InternVL) remain unknown.
- Merging operations still incur information loss, and how to reduce task conflicts between layers is not thoroughly explored.
- The issue of maintaining the model's original zero-shot generalization capability is not addressed.

## Related Work & Insights
- **vs. O-LoRA**: O-LoRA avoids knowledge overwrite via orthogonal constraints but does not differentiate between layers, leading to lower resource efficiency.
- **vs. MoELoRA**: MoELoRA applies MoE across all layers and requires training a router (+20% overhead), whereas HiDe-LLaVA does it only at the top layer and requires no trained router.
- This paradigm of "CKA-driven hierarchical analysis $\to$ differentiated processing" can be transferred to LLM continual learning, multi-task learning, and other scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ The CKA-driven hierarchical decoupling approach is novel, and the discovery of information leakage is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐ New benchmark + two old benchmarks + detailed ablations + task order testing.
- Writing Quality: ⭐⭐⭐⭐ Clear CKA analysis and comprehensive methodology description.
- Value: ⭐⭐⭐⭐ Holds practical guidance and significance for the MLLM continual learning community.
   - The top layer retains LoRA modules of all learned tasks, producing MoE-style weighted outputs during inference.
   - Instead of a traditional MoE Router, CLIP encoders are used to extract image-text anchors (calculating average image-text features for each task during training), with weights assigned based on cosine similarity during inference: $d_c = \text{softmax}(\bar{r^c}/T)$, $\bar{r^c} = \alpha r_v^c + \beta r_{ins}^c$.
   - Output: $O_{top} = \sum_i d_i E_i(h)$.

3. **Task-General Merging (Remaining Layers)**:
    - Directly merge LoRA modules of all tasks using the merging coefficient: $\bar{E}_T = \sum \epsilon_i E_i$.
    - Share general knowledge after merging, eliminating the need for task selection.

4. **Leakage-Free Benchmark Construction**:
    - Evaluate the model's zero-shot performance on each candidate dataset, filtering out tasks already known to the model (those seen during the SFT phase).
    - Retain only tasks where the model shows poor zero-shot performance, ensuring that what is evaluated is "newly learned" rather than "already known."

## Key Experimental Results

| Method | New Benchmark Average | Old Benchmark Average | Info Leakage |
|------|:---:|:---:|:---:|
| Sequential FT | Low (Severe Forgetting) | Low | - |
| MoELoRA | Baseline | High (Potentially Inflated) | **Yes** |
| ModalPrompt | Better | High | **Yes** |
| **HiDe-LLaVA** | **SOTA** | High | **Fixed** |

### Key Findings
- **Small performance gap among methods on old benchmarks** (as information leakage makes the model "already know" them), but **significant differences on the new benchmark**—HiDe-LLaVA leads by a wide margin.
- CLIP image-text anchor routing is more effective than a traditional Router (no training required, leverages pretrained knowledge).
- Top-layer expansion only + merging of remaining layers = optimal balance (high efficiency and performance).

## Highlights & Insights
- **CKA Analysis-Driven Design**: Rather than arbitrary layering, data-driven similarity analysis determines what functions various layers should perform.
- **Revealing Benchmark Information Leakage**: Serves as an important warning regarding evaluation fairness for the entire continual instruction tuning field.
- **CLIP Anchor Replacing Router**: Zero training cost for inference-time task selection, and is more stable than training a Router.

## Limitations & Future Work
- Validated only on LLaVA; applicability to other MLLM architectures (such as Qwen-VL) has not been tested.
- The $\alpha$/$\beta$ of image-text anchors and temperature $T$ require tuning.
- The number of top-layer LoRAs increases linearly as the number of tasks grows.

## Rating
- Novelty: ⭐⭐⭐⭐ CKA-driven hierarchical decoupling + revealing benchmark information leakage.
- Experimental Thoroughness: ⭐⭐⭐⭐ New vs. old benchmark comparison + ablation study + CKA visualization.
- Writing Quality: ⭐⭐⭐⭐ Clear logic in CKA analysis and experimental design.
- Value: ⭐⭐⭐⭐ Substantial progress for MLLM continual learning, with the benchmark correction bringing community value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Multimodal Continual Instruction Tuning with BranchLoRA](branchlora_continual_instruction.md)
- [\[ACL 2025\] AVG-LLaVA: An Efficient Large Multimodal Model with Adaptive Visual Granularity](avg-llava_an_efficient_large_multimodal_model_with_adaptive_visual_granularity.md)
- [\[ACL 2025\] MEIT: Multimodal Electrocardiogram Instruction Tuning on Large Language Models for Report Generation](meit_multimodal_electrocardiogram_instruction_tuning_on_large_language_models_fo.md)
- [\[ICCV 2025\] SMoLoRA: Exploring and Defying Dual Catastrophic Forgetting in Continual Visual Instruction Tuning](../../ICCV2025/multimodal_vlm/smolora_exploring_and_defying_dual_catastrophic_forgetting_in_continual_visual_i.md)
- [\[ICML 2025\] Dynamic Mixture of Curriculum LoRA Experts for Continual Multimodal Instruction Tuning](../../ICML2025/multimodal_vlm/dynamic_mixture_of_curriculum_lora_experts_for_continual_multimodal_instruction_.md)

</div>

<!-- RELATED:END -->
