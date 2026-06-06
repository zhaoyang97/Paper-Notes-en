---
title: >-
  [Paper Note] NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints
description: >-
  [NeurIPS 2025][Multimodal VLM][Native MLLM] This paper systematically investigates the design space and scaling properties of native multimodal large language models (Native MLLMs) under data constraints. It identifies a…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Native MLLM"
  - "Scaling Law"
  - "Visual Encoder"
  - "Mixture-of-Experts"
  - "End-to-End Training"
date: 2026-05-08
content_hash: 57636cbd0f4f0138
---

# NaViL: Rethinking Scaling Properties of Native Multimodal Large Language Models under Data Constraints

**Conference**: NeurIPS 2025
**arXiv**: [2510.08565](https://arxiv.org/abs/2510.08565)  
**Code**: [GitHub](https://github.com/OpenGVLab/NaViL)  
**Area**: Multimodal VLM
**Keywords**: Native MLLM, Scaling Law, Visual Encoder, Mixture-of-Experts, End-to-End Training

## TL;DR

This paper systematically investigates the design space and scaling properties of native multimodal large language models (Native MLLMs) under data constraints. It identifies a positive log-linear optimal scaling relationship between the visual encoder and the LLM, and based on this finding proposes NaViL, which achieves competitive performance with state-of-the-art MLLMs using only approximately 600 million pre-training image-text pairs.

## Background & Motivation

**Background**: The dominant MLLM paradigm adopts compositional training — independently pre-training a visual encoder and an LLM, then aligning them through multimodal fine-tuning.

**Limitations of Prior Work**: Compositional training makes it difficult to explore joint scaling properties of the visual and language components, and vision-language alignment is constrained by the limitations of separate training.

**Key Challenge**: Native MLLMs (end-to-end training) exhibit greater potential for favorable scaling laws, yet existing studies evaluate them primarily under the assumption of unlimited resources, leaving their practical feasibility under data- and compute-constrained settings largely unexplored.

**Goal**: Can native MLLMs match or even surpass the performance ceiling of top-tier compositional MLLMs under realistic data constraints?

**Key Insight**: Systematically explore key architectural choices for native MLLMs (LLM initialization, MoE, visual encoder design) and the joint scaling laws of the vision-language components.

**Core Idea**: The optimal parameter count of the visual encoder grows log-linearly with the LLM parameter count; the two must be scaled jointly to achieve optimal performance.

## Method

### Overall Architecture

NaViL is an end-to-end trained native MLLM composed of three components: a visual encoder $\mathcal{V}_{d,w}$, an MLP connector $\mathcal{C}$, and an MoE-augmented LLM. It supports arbitrary-resolution inputs and employs Visual Multi-scale Packing to enhance inference performance.

### Key Designs

1. **LLM Initialization**: Language parameters are initialized from a pre-trained LLM (InternLM2-Base) rather than trained from scratch. Experiments show that initialized models converge more than 10× faster than their from-scratch counterparts and exhibit significantly superior zero-shot image captioning performance. This is attributed to the substantially lower textual diversity and quality of multimodal training corpora compared to pure language pre-training data.

2. **Modality-Specific MoE**: Modality-specific attention experts and FFN experts are introduced at every LLM layer. Separate projection matrices $W_Q^m, W_K^m, W_V^m, W_O^m$ process visual and textual features, while a unified global attention computation is maintained:
$$x_{i,m}^{l'} = x_{i,m}^{l-1} + \text{MHA-MMoE}(\text{RMSNorm}(x_{i,m}^{l-1}))$$
$$x_{i,m}^{l} = x_{i,m}^{l'} + \text{FFN-MMoE}(\text{RMSNorm}(x_{i,m}^{l'}))$$
MoE enables the model to reach the same validation loss with only 1/10 of the data, without increasing training or inference cost.

3. **Visual Encoder Architecture Search**: Under a fixed parameter budget, the optimal combination of depth $d$ and width $w$ is explored (parameter count $\mathcal{N} = 12 \times d \times w^2$). Experiments show that extreme depth or width configurations perform poorly, while moderate configurations show little difference at large data scales.

4. **Visual Multi-scale Packing**: At inference time, the input image is progressively downsampled at rate $\tau$ to generate a multi-scale image sequence $\{I_i\}_{i=0}^n$. Each scale is processed by the visual encoder independently and then concatenated before being fed into the LLM, with special tokens `<end_of_scale>` separating different scales.

### Scaling Property Findings

- **Scaling LLM Independently**: Validation loss decreases log-linearly as LLM parameter count increases, consistent with conventional language scaling laws.
- **Scaling Visual Encoder Independently**: Performance gains diminish — when the LLM is fixed, increasing the visual encoder beyond a threshold yields negligible benefit, indicating that the performance ceiling is constrained by LLM capacity.
- **Joint Scaling**: The logarithm of the optimal visual encoder size grows linearly with the logarithm of the LLM size, necessitating joint scaling. This stands in contrast to the compositional approach of using a fixed-size visual encoder.

### Loss & Training

- **Stage 1 — Multimodal Generative Pre-training**: 500 million image-text pairs are used for training (300 million web-crawled + 200 million synthetic captions), with only visual parameters updated; subsequently, 185 million high-quality samples are used to unfreeze the attention layer text parameters.
- **Stage 2 — Supervised Fine-tuning**: All parameters are unfrozen and fine-tuned on 68 million high-quality multimodal samples.

## Key Experimental Results

### Main Results

| Model | #Active Params | Avg | MMVet | MMMU | MMB | MME | MathVista | OCRBench | CCB |
|-------|---------------|-----|-------|------|-----|-----|-----------|----------|-----|
| InternVL-2.5-2B (Compositional) | 2.2B | 67.0 | 60.8 | 43.6 | 74.7 | 2138 | 51.3 | 804 | 81.7 |
| Mono-InternVL (Native) | 1.8B | 56.4 | 40.1 | 33.7 | 65.5 | 1875 | 45.7 | 767 | 66.3 |
| **NaViL-2B** (Native) | **2.4B** | **67.1** | **78.3** | **41.8** | **71.2** | **1822** | **50.0** | **796** | **83.9** |

NaViL-2B surpasses the compositional baseline InternVL-2.5-2B on most metrics and substantially outperforms all existing native MLLMs.

### Ablation Study

| Design Choice | Effect |
|--------------|--------|
| With vs. without LLM initialization | Initialized version converges 10×+ faster with significantly better zero-shot captioning |
| With vs. without MoE | MoE version reaches the same loss with only 1/10 of the data |
| Visual encoder $d$ = 3/6/12/24/48 | Extreme configurations underperform; moderate configurations show minimal differences |

### Key Findings

- Native MLLMs achieve performance competitive with top-tier compositional MLLMs for the first time at the 2B parameter scale.
- The optimal visual encoder size grows log-linearly with LLM size.
- MoE architecture is critical for handling heterogeneous multimodal data.

## Highlights & Insights

- **First Systematic Study**: Provides a comprehensive exploration of the design space and scaling properties of native MLLMs under data constraints.
- **High Practicality**: A highly competitive native MLLM can be trained with only approximately 600 million pre-training samples.
- **Novel Scaling Law**: The identified optimal joint scaling relationship between the visual encoder and the LLM offers important guidance for native MLLM design.
- **Modality-Specific MoE**: Introducing attention experts in addition to FFN experts addresses the inter-modal feature scale discrepancy problem.

## Limitations & Future Work

- Only the image modality is explored; extension to video, audio, and other modalities remains unaddressed.
- The validation of scaling laws is limited in scope (LLM up to 7B); whether the findings hold at larger scales is yet to be confirmed.
- The visual encoder architecture directly reuses Transformer layers from the LLM; vision-specific architectural designs are not explored.
- Visual Multi-scale Packing introduces additional computational overhead at inference time.

## Related Work & Insights

- **Chameleon**: A native MLLM trained from scratch; its substantially inferior performance compared to NaViL underscores the importance of LLM initialization.
- **Mono-InternVL**: The first native MLLM to introduce modality-specific MoE; NaViL extends this by additionally incorporating attention experts.
- **Compositional MLLMs** (InternVL, Qwen2VL): The strategy of using a fixed-size visual encoder is shown to be suboptimal.

## Rating

- Novelty: ⭐⭐⭐⭐ Systematic investigation of native MLLM scaling laws is a novel contribution, though the individual architectural components are not entirely new.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 14 benchmarks, extensive ablation studies, and comprehensive scaling analyses.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, rich figures, and systematic analysis.
- Value: ⭐⭐⭐⭐⭐ Provides important practical guidance and theoretical insights for native MLLM design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Scaling Laws for Native Multimodal Models](../../ICCV2025/multimodal_vlm/scaling_laws_for_native_multimodal_models.md)
- [\[NeurIPS 2025\] BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](bioclip_2_emergent_properties_from_scaling_hierarchical_contrastive_learning.md)
- [\[NeurIPS 2025\] ACT as Human: Multimodal Large Language Model Data Annotation with Critical Thinking](act_as_human_multimodal_large_language_model_data_annotation.md)
- [\[NeurIPS 2025\] FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models](flowcut_rethinking_redundancy_via_information_flow_for_effic.md)
- [\[NeurIPS 2025\] Rethinking Multimodal Learning from the Perspective of Mitigating Classification Ability Disproportion](rethinking_multimodal_learning_from_the_perspective_of_mitig.md)

</div>

<!-- RELATED:END -->
