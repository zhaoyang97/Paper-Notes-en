---
title: >-
  [Paper Note] SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model
description: >-
  [CVPR 2025][VLM Reasoning][3D affordance] This work proposes the Sequential 3D Affordance Reasoning task and constructs a benchmark of 180K instruction-point cloud pairs. By introducing a `<SEG>` token and a multi-granular language-point integration module into a 3D MLLM, the model reasons and segments sequential affordance regions from complex human instructions.
tags:
  - "CVPR 2025"
  - "VLM Reasoning"
  - "3D affordance"
  - "sequential reasoning"
  - "multimodal large language model"
  - "point cloud segmentation"
  - "embodied AI"
date: 2026-05-08
content_hash: 15d205f41f6a62e8
---

# SeqAfford: Sequential 3D Affordance Reasoning via Multimodal Large Language Model

**Conference**: CVPR 2025  
**arXiv**: [2412.01550](https://arxiv.org/abs/2412.01550)  
**Code**: [https://github.com/seq-afford](https://github.com/seq-afford)  
**Area**: Multimodal VLM  
**Keywords**: 3D affordance, sequential reasoning, multimodal large language model, point cloud segmentation, embodied AI

## TL;DR

This work proposes the Sequential 3D Affordance Reasoning task and constructs a benchmark of 180K instruction-point cloud pairs. By introducing a `<SEG>` token and a multi-granular language-point integration module into a 3D MLLM, the model reasons and segments sequential affordance regions from complex human instructions.

## Background & Motivation

Existing 3D affordance works are limited to the "single object - single affordance" paradigm: given a simple instruction (e.g., "open the door"), the model only needs to locate a single corresponding affordance region (e.g., the door handle). However, in real-world scenarios, human instructions often involve long-horizon multi-step tasks across multiple objects—for example, "heating food from the bowl in the microwave" sequentially requires "grasping the bowl → opening the microwave door → putting in the bowl". Existing methods cannot actively reason about such implicit sequential affordances.

Furthermore, prior approaches either rely on independent language encoders (such as BERT/RoBERTa), which lack reasoning and world knowledge, or utilize pure LLMs to localize 2D objects before retrieving 3D counterparts, lacking joint vision-language alignment capability. Thus, a unified 3D multimodal large language model is needed to bridge the gap between reasoning and fine-grained segmentation.

## Method

### Overall Architecture

SeqAfford consists of three main components: (1) a 3D visual encoder (Uni3D) based on large-scale 3D representation learning, providing foundational features for dense prediction; (2) a 3D multimodal large language model (ShapeLLM) that leverages world knowledge for affordance reasoning and outputs text containing a `<SEG>` token; and (3) a multi-granular language-point integration (MGLP) module, which injects LLM reasoning results into dense point cloud features to generate affordance masks.

### Key Designs

1. **Sequential Affordance Reasoning**:
    - **Function**: Decomposes complex human instructions into multi-step affordance sequences.
    - **Mechanism**: A special `<SEG>` token is added to the vocabulary of the 3D MLLM. Given the point cloud $\mathbf{X}_{\text{point}}$ and instruction $\mathbf{X}_{\text{txt}}$, the model outputs text $\tilde{\mathbf{y}}_{\text{txt}} = \mathcal{F}(\mathbf{X}_{\text{point}}, \mathbf{X}_{\text{txt}})$ containing several `<SEG>` tokens, where each `<SEG>` represents one affordance segmentation result. Its last-layer embedding is extracted and projected via an MLP to obtain a segmentation vector $\mathbf{H}_{\text{seg}}^{(i)} = \text{Proj}(\mathbf{h}_{\text{seg}}^{(i)})$.
    - **Design Motivation**: Inspired by 2D segmentation MLLMs like LISA, the segmentation capability is embedded within the generative process of the LLM, enabling reasoning and segmentation to be completed in a unified framework.

2. **Multi-Granular Language-Point Integration**:
    - **Function**: Injects abstract semantics reasoned by the LLM into 3D point cloud dense features to achieve affordance mask prediction.
    - **Mechanism**: Done in two stages: (a) Multi-granular feature propagation: propagating intermediate features progressively to dense features $\mathbf{f}_{\text{dense}}$ through hierarchical upsampling and FPS; (b) Point-language fusion: performing cross-attention with $\mathbf{H}_{\text{seg}}^{(i)}$ as the Query and $\mathbf{f}_{\text{dense}}$ as Key/Value, and then merging with sparse features $\mathbf{f}_{\text{sparse}}$ to obtain $\mathbf{A}_f^{(i)} = \mathcal{G}(\mathbf{f}_{\text{dense}}, \mathbf{f}_{\text{sparse}}, \mathbf{H}_{\text{seg}}^{(i)})$, which is finally decoded to output the mask.
    - **Design Motivation**: Direct dense prediction is unfeasible using only the LLM's global semantic embeddings. Multi-granular point cloud features are necessary to provide spatial details, while semantic tokens provide guidance on "where to segment."

3. **Large-scale Benchmark Construction (180K Instruction-Point Cloud Pairs)**:
    - **Function**: Provides training and testing data for both single affordance and sequential affordance settings.
    - **Mechanism**: Based on point clouds from 3D AffordanceNet and mesh rendered images from PartNet, combined with HOI images from IAGNet, GPT-4o is prompted using 4 modalities (text-only, mesh rendering, mesh + HOI image, mesh + scene description) to generate diverse instructions.
    - **Design Motivation**: Addresses the limitation of existing datasets where instructions are overly simplistic (e.g., sharing the same text for the same class of objects), generating personalized instructions for each point cloud instance.

### Loss & Training

The total loss is composed of three parts: $\mathcal{L} = \lambda_c \mathcal{L}_c + \lambda_b \mathcal{L}_b + \lambda_d \mathcal{L}_d$, where $\mathcal{L}_c$ is the autoregressive cross-entropy loss (text generation), $\mathcal{L}_b$ is the binary cross-entropy loss (mask prediction), and $\mathcal{L}_d$ is the Dice loss (mask prediction). ShapeLLM-7B is efficiently fine-tuned using LoRA (rank=8) while freezing the 3D encoder. AdamW optimizer is used with a learning rate of 2e-4, a cosine scheduler, and trained for 10 epochs on 1x A100.

## Key Experimental Results

### Main Results

| Setting | Method | mIoU↑ | AUC↑ | SIM↑ | MAE↓ |
|------|------|-------|------|------|------|
| Seen | PointRefer (SOTA) | 16.3 | 84.3 | 0.568 | 0.108 |
| Seen | **SeqAfford** | **19.5** | **86.9** | **0.594** | **0.098** |
| Unseen | PointRefer | 12.4 | 76.1 | 0.502 | 0.132 |
| Unseen | **SeqAfford** | **13.8** | **82.4** | **0.518** | **0.128** |
| Sequential | PointRefer* | 14.3 | 80.7 | 0.521 | 0.124 |
| Sequential | **SeqAfford** | **14.6** | **84.2** | **0.573** | **0.118** |

*\*Note: \* indicates the baseline uses the ground-truth sequential order, as it lacks sequential reasoning capabilities on its own.*

### Ablation Study

| Configuration | mIoU↑ (Single) | AUC↑ (Single) | mIoU↑ (Seq) | AUC↑ (Seq) |
|------|----------------|---------------|-------------|------------|
| w/o MGLP | 12.1 | 83.4 | 11.7 | 80.3 |
| w/ MGLP (Ours) | **19.5** | **86.9** | **14.6** | **84.2** |

| 3D Vision Encoder | mIoU↑ | AUC↑ | SIM↑ | MAE↓ |
|-------------|-------|------|------|------|
| ULIP | 17.9 | 84.8 | 0.574 | 0.109 |
| OpenShape | 18.4 | 85.3 | 0.582 | 0.103 |
| Recon++ | 19.1 | 86.4 | 0.588 | 0.099 |
| **Uni3D** | **19.5** | **86.9** | **0.594** | **0.098** |

### Key Findings

- The MGLP module significantly improves mIoU (Single: 12.1 → 19.5, +61%), indicating that multi-granular integration is key to bridging reasoning and segmentation.
- Under the Unseen setting, AUC increases from 76.1 to 82.4 (+8.3%), demonstrating the capability of 3D MLLMs to perform open-world generalization utilizing world knowledge.
- In sequential tasks, SeqAfford significantly outperforms the baseline in AUC and SIM, even when the baseline is supplied with the ground-truth sequence order.

## Highlights & Insights

- **Novel Task Formulation**: For the first time, 3D affordance is extended from "single instruction → single affordance" to "complex instruction → sequential affordance," mimicking real robotic manipulation needs more closely.
- **Unified Framework Design**: Blends reasoning (LLM world knowledge) and segmentation (dense prediction) into a single model, preventing information loss from pipeline-based methods.
- **Ingenious Data Construction**: Leverages combinations of four modalities to prompt GPT-4o for diverse instructions, which is much more realistic than previous approaches that share a single text template per object category.

## Limitations & Future Work

- Affordance segmentation is currently performed only at the object level, and has not yet been extended to scene-level fine-grained reasoning.
- Relies on ShapeLLM (7B) as the backbone, resulting in high computational overhead.
- The mIoU for sequential tasks (14.6) still has considerable room for improvement, indicating that the accuracy of multi-step reasoning remains a challenge.
- The dataset covers only 23 categories based on 3D AffordanceNet, with limited coverage.

## Related Work & Insights

- LISA introduced the `<SEG>` token into 2D MLLMs for reasoning segmentation; this work extends it to the 3D domain.
- Although 3D MLLMs like ShapeLLM/PointLLM understand 3D objects, they lack dense prediction capabilities.
- For embodied AI, affordance reasoning is a pivotal link connecting perception and manipulation, with sequential affordance serving as the foundation for long-horizon planning.

## Rating
- Novelty: ⭐⭐⭐⭐ First to propose the sequential 3D affordance reasoning task; both the task definition and benchmark construction are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive multi-setting comparisons and ablation studies, though lacking physical validation in downstream robotic tasks.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clear explanations of task motivations.
- Value: ⭐⭐⭐⭐ Provides a new paradigm of affordance reasoning for long-horizon manipulation in embodied AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ACT as Human: Multimodal Large Language Model Data Annotation with Critical Thinking](../../NeurIPS2025/vlm_reasoning/act_as_human_multimodal_large_language_model_data_annotation.md)
- [\[NeurIPS 2025\] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](../../NeurIPS2025/vlm_reasoning/affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)
- [\[ICLR 2026\] SportR: A Benchmark for Multimodal Large Language Model Reasoning in Sports](../../ICLR2026/vlm_reasoning/sportr_a_benchmark_for_multimodal_large_language_model_reasoning_in_sports.md)
- [\[CVPR 2025\] Insight-V: Exploring Long-Chain Visual Reasoning with Multimodal Large Language Models](insight-v_exploring_long-chain_visual_reasoning_with_multimodal_large_language_m.md)
- [\[CVPR 2026\] PointThinker: Point-Incentivized Parallel Thinking for Multimodal Large Language Model](../../CVPR2026/vlm_reasoning/pointthinker_point-incentivized_parallel_thinking_for_multimodal_large_language_.md)

</div>

<!-- RELATED:END -->
