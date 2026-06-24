---
title: >-
  [Paper Note] Efficient Transfer Learning for Video-language Foundation Models
description: >-
  [CVPR 2025][Video Understanding][Parameter-Efficient Fine-Tuning] The authors propose a Multimodal Spatiotemporal Adapter (MSTA), which achieves efficient transfer of video-language foundation models to downstream tasks with only 2-7% of trainable parameters, through a vision-language shared projection layer and spatiotemporal description-guided consistency constraints.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Parameter-Efficient Fine-Tuning"
  - "Video Action Recognition"
  - "Multimodal Adapter"
  - "Transfer Learning"
  - "Generalization Ability"
date: 2026-05-08
content_hash: 74e20e16f8aad3c3
---

# Efficient Transfer Learning for Video-language Foundation Models

**Conference**: CVPR 2025  
**arXiv**: [2411.11223](https://arxiv.org/abs/2411.11223)  
**Code**: [https://github.com/chenhaoxing/ETL4Video](https://github.com/chenhaoxing/ETL4Video)  
**Area**: Video Understanding  
**Keywords**: Parameter-Efficient Fine-Tuning, Video Action Recognition, Multimodal Adapter, Transfer Learning, Generalization Ability

## TL;DR

The authors propose a Multimodal Spatiotemporal Adapter (MSTA), which achieves efficient transfer of video-language foundation models to downstream tasks with only 2-7% of trainable parameters, through a vision-language shared projection layer and spatiotemporal description-guided consistency constraints.

## Background & Motivation

Pre-trained video-language foundation models (such as CLIP, ViCLIP) require adaptation to downstream video tasks. Existing methods face two Key Challenges:

1. **Conflict between parameters and generalization**: Methods such as ActionCLIP and XCLIP introduce a large number of extra parameters to model temporal information. Although they improve downstream task performance, they trigger catastrophic forgetting, severely damaging the generalization performance on unseen categories.
2. **Limitations of single-modality PEFT**: Although parameter-efficient methods such as LoRA and AdaptFormer use fewer parameters, they are designed for single-modality models. When applied independently to the vision and text branches, they ignore cross-modal interactions, failing to effectively align video and text representations.

In addition, as ViCLIP is a pre-trained model specifically designed for videos, existing CLIP-based methods cannot be directly transferred to it, leaving a lack of efficient fine-tuning schemes specifically tailored for ViCLIP.

## Method

### Overall Architecture

Taking ViCLIP (a video-version CLIP model that replaces CLIP's original attention with spatiotemporal attention) as the backbone, lightweight Multimodal Spatiotemporal Adapters (MSTA) are injected into the high-end Transformer blocks of the video and text encoders. During training, the pre-trained parameters are frozen, and only the adapters are optimized. Meanwhile, a spatiotemporal description-guided consistency constraint ($\mathcal{L}_{CC}$) is employed to mitigate overfitting.

### Key Designs

1. **MSTA Adapter Architecture**:
    - Function: Establishes parameter-efficient cross-modal alignment between the video and text branches.
    - Mechanism: Each adapter consists of three components: modality-specific down-projection layers $\mathbf{W}_v^{kd}$/$\mathbf{W}_t^{kd}$, a **cross-modality shared** intermediate bottleneck projection layer $\mathbf{W}^{ks}$, and modality-specific up-projection layers. The up-projection of the video branch is split into a spatial up-projection $\mathbf{W}_v^{ku-s}$ (a linear layer) and a temporal up-projection $\mathbf{W}_v^{ku-t}$ (a 3D convolutional layer), with their outputs added together. A scaling factor $\lambda$ controls the intensity of the adapter output: $[c_j, x_j] = \mathcal{E}^j_v([c_{j-1}, x_{j-1}]) + \lambda \mathcal{A}^j_v([c_{j-1}, x_{j-1}])$.
    - Design Motivation: The shared intermediate layer can simultaneously receive gradient updates from both the vision and text modalities during fine-tuning, thereby optimizing cross-modal alignment. The separate down- and up-projection layers preserve the specificities of each modality. The spatial and temporal joint up-projection design enhances the model's adaptation to spatial and temporal features, respectively.

2. **Selective Layer Injection Strategy**:
    - Function: Injects adapters only into higher-level Transformer blocks to protect the general features learned in lower-level layers.
    - Mechanism: MSTA adapters are added from the $k$-th block to the final block $L$, while the lower-level layers from $1$ to $k-1$ remain frozen. Different values of $k$ are selected based on task settings (blocks 1-12 for base-to-novel, blocks 8-12 for few-shot).
    - Design Motivation: Lower Transformer layers learn generic features, while higher layers learn task-specific features. In scenarios requiring substantial generalization, such as few-shot learning, fine-tuning only the higher layers preserves the pre-trained knowledge more effectively.

3. **Spatiotemporal Description-guided Consistency Constraint**:
    - Function: Prevents overfitting and enhances generalization via knowledge distillation.
    - Mechanism: Large language models (e.g., DeepSeek) are leveraged to generate spatial descriptions $\text{DES}_s$ and temporal descriptions $\text{DES}_t$ for each action category. A standard prompt template ("a video of {cls}") is input into the trainable branch, while the LLM-generated descriptions are fed into the frozen pre-trained branch. The outputs of both branches are aligned using a consistency constraint based on cosine distance: $\mathcal{L}_{CC} = 2 - \cos(w^c, D_s^c) - \cos(w^c, D_t^c)$.
    - Design Motivation: Knowledge distillation forces the trainable encoders not to deviate too far from the pre-trained model. Spatiotemporal descriptions provide richer semantic information than simple templates, guiding the model to learn more discriminative representations in the spatiotemporal semantic space.

### Loss & Training

The final objective function is a weighted sum of the cross-entropy loss and the consistency constraint:

$$\mathcal{L} = \mathcal{L}_{CE} + \alpha \mathcal{L}_{CC}$$

where $\mathcal{L}_{CE}$ is the standard video-text contrastive loss and $\alpha=1.0$ is the optimal weight. The AdamW optimizer is used with a weight decay of 0.001, and $N=2$ descriptions is found to be optimal. All modules in MSTA are initialized using Kaiming initialization.

## Key Experimental Results

### Main Results (Base-to-Novel Generalization, Harmonic Mean (HM) Averaged over 4 Datasets)

| Method | Trainable Params | K-400 HM | HMDB-51 HM | UCF-101 HM | SSv2 HM |
|------|-----------|----------|------------|------------|---------|
| ViFi-CLIP (Full Fine-Tuning) | All | 68.2 | 62.5 | 78.7 | 14.2 |
| ViCLIP (Full Fine-Tuning) | 124.3M | 71.3 | 62.7 | 81.6 | 17.0 |
| +AdaptFormer | 7.9M | 71.1 | 64.3 | 82.3 | 17.0 |
| +LoRA | 9.4M | 70.9 | 64.0 | 82.1 | 16.0 |
| **+MSTA+$\mathcal{L}_{CC}$** | **8.7M** | **72.0** | **66.3** | **82.9** | **18.9** |

### Ablation Study

| Configuration | Base | Novel | HM | Description |
|------|------|-------|----|------|
| Language Adapter Only | 66.1 | 51.5 | 57.9 | Inadequate single modality |
| Vision Adapter Only | 65.7 | 51.7 | 57.9 | Inadequate single modality |
| No Shared Layer | 68.0 | 52.9 | 59.5 | Lacks cross-modality alignment |
| Full MSTA | 68.6 | 53.5 | 60.1 | Shared layer improves HM by 0.6 |

### Key Findings

- MSTA achieves state-of-the-art (SOTA) performance across all four evaluation settings (zero-shot, few-shot, base-to-novel, and fully-supervised), while using only 2-7% of the original model parameters.
- On SSv2 (a dataset with strong temporal dependencies), the accuracy of Novel categories increases from OST's 11.5 to 16.5 (+43%), verifying the significance of spatiotemporal modeling.
- The shared projection layer yields consistent HM improvements (59.5 → 60.1), demonstrating the efficacy of cross-modal gradient sharing.
- The consistency constraint is particularly effective in few-shot learning, successfully mitigating overfitting in low-data regimes.
- The number of descriptions $N=2$ is optimal; larger values of $N$ introduce more noise due to large language model hallucinations.
- Injecting adapters only into higher layers (8→12) performs better in few-shot scenarios compared to full-layer injection.

## Highlights & Insights

- **Shared projection layer** is the core innovation of the proposed method: a simple idea (cross-modal shared intermediate layers) brings significant performance improvements almost without increasing parameters.
- The **dual spatial + temporal up-projection** design is simple yet effective, utilizing linear layers to capture spatial information and 3D convolutions to capture temporal information.
- The combination of **LLM-generated descriptions + knowledge distillation** successfully transfers knowledge from large language models to downstream tasks, serving as a solid example of the "LLM as teacher" paradigm.
- The method is highly versatile and can be adapted to both CLIP and ViCLIP simultaneously.

## Limitations & Future Work

- The consistency constraint relies heavily on the generation quality of the LLM, and hallucination issues at larger values of $N$ limit its scalability.
- The method has only been verified on ViT-B/16, leaving its performance on larger-scale models uncertain.
- The generation of spatiotemporal descriptions is offline and decoupled from training; online adaptive generation might yield better results.
- The spatial and temporal up-projections are simply added together; more sophisticated fusion strategies (e.g., gating mechanisms) might yield further improvements.

## Related Work & Insights

- PEFT methods such as LoRA and AdaptFormer are efficient but neglect multimodal alignment; this work highlights the critical importance of cross-modal interactions.
- The OST method also leverages LLM-generated descriptions but relies on full-parameter fine-tuning, which is prone to overfitting; this study utilizes descriptions for distillation constraints, which is a more elegant approach.
- The MoTE method introduces a mixture of temporal experts, which requires a large parameter budget (88M vs. 8.7M in this work) yet yields slightly worse performance.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of cross-modal shared projection and description-guided consistency constraints is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers four evaluation settings, six datasets, and comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, though some parts have heavy mathematical formulations that require careful reading.
- Value: ⭐⭐⭐⭐ Provides a practical solution for the efficient transfer of video-language models with a wide scope of applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Video-Panda: Parameter-efficient Alignment for Encoder-free Video-Language Models](video-panda_parameter-efficient_alignment_for_encoder-free_video-language_models.md)
- [\[ECCV 2024\] R²-Tuning: Efficient Image-to-Video Transfer Learning for Video Temporal Grounding](../../ECCV2024/video_understanding/r2tuning_efficient_imagetovideo_transfer_learning_for_video.md)
- [\[CVPR 2025\] Video Summarization with Large Language Models](video_summarization_with_large_language_models.md)
- [\[CVPR 2025\] On the Consistency of Video Large Language Models in Temporal Comprehension](on_the_consistency_of_video_large_language_models_in_temporal_comprehension.md)
- [\[CVPR 2025\] PAVE: Patching and Adapting Video Large Language Models](pave_patching_and_adapting_video_large_language_models.md)

</div>

<!-- RELATED:END -->
