---
title: >-
  [Paper Note] ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models
description: >-
  [ICCV 2025][Model Compression][Cross-architecture distillation] This paper proposes ViT-Linearizer, a cross-architecture distillation framework that transfers the "quadratic knowledge" learned by ViT self-attention into linear-complexity recurrent models (Mamba-based Adventurer) via two core mechanisms: **activation matching** and **masked prediction**. The approach achieves 84.3% accuracy on ImageNet while delivering up to 4.2× inference speedup on high-resolution tasks.
tags:
  - "ICCV 2025"
  - "Model Compression"
  - "Cross-architecture distillation"
  - "ViT"
  - "Mamba"
  - "linear complexity"
  - "activation matching"
  - "masked prediction"
date: 2026-05-08
content_hash: e6831ae767a25de2
---

# ViT-Linearizer: Distilling Quadratic Knowledge into Linear-Time Vision Models

**Conference**: ICCV 2025
**Code**: Not released  
**Area**: Model Compression
**Keywords**: Cross-architecture distillation, ViT, Mamba, linear complexity, activation matching, masked prediction
**Authors**: Guoyizhe Wei, Rama Chellappa (Johns Hopkins University)

## TL;DR

This paper proposes ViT-Linearizer, a cross-architecture distillation framework that transfers the "quadratic knowledge" learned by ViT self-attention into linear-complexity recurrent models (Mamba-based Adventurer) via two core mechanisms: **activation matching** and **masked prediction**. The approach achieves 84.3% accuracy on ImageNet while delivering up to 4.2× inference speedup on high-resolution tasks.

## Background & Motivation

Vision Transformers (ViTs) have demonstrated exceptional performance in visual representation learning through their global self-attention mechanism, yet their $\mathcal{O}(L^2)$ quadratic complexity becomes a severe computational bottleneck when processing high-resolution inputs. As the demand for high-resolution and high-fidelity visual inputs continues to grow rapidly, efficiently leveraging the "quadratic knowledge" encoded in ViTs becomes increasingly important.

On the other hand, RNN-style token mixers such as Mamba, RWKV, and xLSTM have exhibited competitive predictive performance and more favorable accuracy–computation trade-offs on vision tasks. These recurrent vision models scale **linearly** in both computational cost and memory with respect to sequence length, making them a promising alternative to the quadratic complexity explosion of self-attention. However, unlike ViTs, which have benefited from extensive research investment, recurrent vision models remain largely confined to smaller data scales and model sizes.

These limitations motivate the authors to develop a **cross-architecture distillation approach** that effectively transfers ViT capabilities to linear-time recurrent models. A key finding is that naïve distillation fails between ViT and Mamba; the combination of **activation matching** and **masked prediction** is essential.

## Method

### Overall Architecture

The overall pipeline of ViT-Linearizer is illustrated in Figure 2. The complete input image is fed to the frozen teacher model (ViT), while a randomly masked image is fed to the student model (Mamba-2-based Adventurer). Token-level activation matching is performed at $K$ intermediate stages, and at the final layer the student predicts the teacher's representations for unseen (masked) tokens. Only the student network is trained; the teacher remains frozen throughout.

The teacher model is CLIP ViT-Base/16, and the student model is Adventurer (equipped with a Mamba-2 token mixer). Their token mixer formulations are compared as follows:

**Self-attention** ($\mathcal{O}(L^2)$):

$$\mathbf{y} = \text{softmax}(\mathbf{q}\mathbf{k}^T/\sqrt{d})\mathbf{v}$$

**Mamba-2** ($\mathcal{O}(L)$):

$$\mathbf{y} = \left(\exp(-\text{softplus}(\delta)\alpha)\mathbf{S} + \mathbf{v}\mathbf{k}^T\right)\mathbf{q}$$

where $\mathbf{S}_t = \exp(-\text{softplus}(\delta_t)\alpha)\mathbf{S}_{t-1} + \mathbf{v}_t \mathbf{k}_t^T$ is a hidden state that accumulates token dependencies in a recurrent manner.

### Activation Matching

**Core Insight**: ViT models typically capture richer information content in intermediate-layer activation maps than in final-layer outputs. These activation maps directly reflect token-level dependencies learned under the quadratic computational cost of self-attention.

Concretely, the teacher and student blocks are partitioned into $K$ stages (default $K=4$). At each stage, an $\mathbb{R}^{L \times L}$ activation map is computed as the pairwise cosine similarity between all tokens:

$$\mathbf{A}_{\text{tea}}^k = \frac{\mathbf{f}_{\text{tea}}^k(i) \cdot \mathbf{f}_{\text{tea}}^k(j)}{\|\mathbf{f}_{\text{tea}}^k(i)\|_2 \|\mathbf{f}_{\text{tea}}^k(j)\|_2}$$

After $\ell_2$-normalizing each row, the activation matching loss is defined as:

$$\mathcal{L}_{\text{act}} = \frac{1}{KL}\sum_{k=1}^{K}\sum_{i=1}^{L}\left[1 - \langle\bar{\mathbf{A}}_{\text{tea}}^k(i,:), \bar{\mathbf{A}}_{\text{stu}}^k(i,:)\rangle\right]$$

This loss itself entails $\mathcal{O}(L^2)$ computation—termed a "quadratic constraint" by the authors—and is experimentally verified to be a necessary component for distilling quadratic knowledge.

### Masked Prediction

A standard asymmetric setup is adopted: the teacher receives the full image while the student receives a masked input, where a portion of patch tokens are randomly replaced with learnable [mask] tokens, using the 75% masking ratio from MAE. The student is trained to predict the teacher's output representations at masked positions:

$$\mathcal{L}_{\text{mask}} = \frac{1}{aL}\sum_{i \in \Omega}\text{Smooth}\ell_1(\mathbf{Y}_{\text{tea}}(i,:), \mathbf{Y}_{\text{stu}}(i,:))$$

### Integration of Activation Matching and Masking

A critical design consideration is that masking alters the student's effective representation space—intermediate features at masked positions constitute predictions of unseen information rather than representations of corresponding input tokens. **Applying activation matching to unseen tokens would cause direct information leakage**, causing masked prediction at the final layer to trivially collapse.

Accordingly, activation matching is applied exclusively to the tokens visible to the student, yielding activation maps of size $\mathbb{R}^{(1-a)L \times (1-a)L}$.

**Total loss**: $\mathcal{L} = \mathcal{L}_{\text{act}} + \lambda \mathcal{L}_{\text{mask}}$, with default $\lambda = 1$.

## Key Experimental Results

### Main Results: ImageNet-1k Classification

| Model | Token Mixer | Input Size | Memory | Throughput | Acc. (%) |
|-------|------------|------------|--------|------------|----------|
| CLIP ViT-B/16 | Self-attention | 224² | 14.4GB | 613 | 84.7 |
| Vim-Base | Mamba | 224² | 20.0GB | 180 | 81.9 |
| Adventurer-Base (supervised) | Mamba-2 | 224² | 13.0GB | 736 | 82.6 |
| **Adventurer-Base (ours)** | Mamba-2 | 224² | 13.0GB | 736 | **84.3** |
| CLIP ViT-B/16 | Self-attention | 448² | >80GB | 95 | 85.3 |
| **Adventurer-Base (ours)** | Mamba-2 | 448² | 45.2GB | 199 | **85.0** |

At 448×448 input resolution, the distilled model achieves **2.1× inference speedup** with only 0.3% accuracy loss.

### Semantic Segmentation Results

| Dataset | Backbone | Params | Throughput | mIoU (%) |
|---------|---------|--------|------------|----------|
| **ADE20k** | CLIP ViT-B/16 | 119M | 1.00× | 51.0 |
| | Adventurer-Base (ours) | 115M | **2.74×** | **51.3** |
| **Cityscapes** | CLIP ViT-B/16 | 122M | 1.00× | 81.8 |
| | Adventurer-Base (ours) | 118M | **4.21×** | **82.0** |

On Cityscapes, the proposed model achieves **4.21× speedup** while surpassing the teacher in mIoU.

### Ablation Study

| Setting | Masked Pred. | Act. Match | IN1k Acc. | ADE mIoU |
|---------|-------------|-----------|-----------|----------|
| supervised | ✗ | ✗ | 82.6 | 47.8 |
| no act. match | ✓ | ✗ | 83.6 | 49.7 |
| no mask pred. | ✗ | ✓ | 83.8 | 50.1 |
| **default** | ✓ | ✓ | **84.3** | **51.3** |

| Activation Matching Scope | IN1k Acc. | ADE mIoU |
|--------------------------|-----------|----------|
| Class token only | 83.7 | 50.0 |
| **Visible tokens only** | **84.3** | **51.3** |
| All tokens | 83.4 | 49.0 |

- Both mechanisms contribute significant individual gains, with their combination being optimal.
- Matching all tokens (including masked ones) causes information leakage and degrades performance.
- Matching only the class token (an $\mathcal{O}(L)$ constraint) is insufficient for fully transferring quadratic knowledge.

### Cross-Scale Distillation

| Teacher | Student | Acc. (%) |
|---------|---------|----------|
| CLIP ViT-B/16 (86M) | Adventurer-S (44M) | 83.1 |
| CLIP ViT-B/16 (86M) | Adventurer-B (99M) | 84.3 |
| CLIP ViT-B/16 (86M) | **Adventurer-L (346M)** | **85.0** |
| CLIP ViT-L/14 (307M) | Adventurer-L (346M) | 85.2 |

A "reverse distillation" phenomenon is observed: a larger student can still benefit from a smaller teacher. Adventurer-L reaches 85.0% (compared to 83.4% under supervised training alone), establishing a new state of the art for Mamba-based architectures.

## Highlights & Insights

1. **Validation of cross-architecture distillation**: This work systematically transfers ViT knowledge to Mamba architectures for the first time, demonstrating that recurrent models can inherit ViT's "attention knowledge."
2. **Activation matching as the cornerstone**: In contrast to simple output-layer distillation, effective knowledge transfer is achieved by matching token-wise dependency relationships at intermediate layers.
3. **Elegant integration of masking and matching**: The information leakage problem is identified, and the solution of restricting matching to visible tokens is proposed.
4. **Efficiency advantage grows with resolution**: The speedup scales from 1.2× at 224² to 4.21× on Cityscapes, with the linear-complexity advantage becoming increasingly pronounced at longer sequence lengths.
5. **Reverse distillation**: Larger students can benefit from smaller teachers, indicating that ViT-Linearizer not only reduces inference cost but also endows recurrent models with attention knowledge and masked modeling capability.

## Limitations & Future Work

1. The activation matching loss during distillation is itself $\mathcal{O}(L^2)$—although used only at training time, it increases training cost.
2. Experiments are primarily conducted at Base/Large scale; the effectiveness of ViT-Linearizer at larger scales remains to be explored.
3. Validation is currently limited to classification and semantic segmentation; performance on generation, multimodal reasoning, and other tasks requires further investigation.
4. The approach depends on a strong teacher model (CLIP ViT), and teacher quality directly determines the performance ceiling.

## Related Work & Insights

- **Theoretical connection to Transformers are SSMs (Dao & Gu, 2024)**: The high formal similarity between self-attention and Mamba-2 provides a theoretical foundation for distillation.
- **Comparison with DeiT (Touvron et al., 2021)**: DeiT pioneered CNN→ViT distillation; this work explores the reverse direction of ViT→Mamba.
- **Implications for high-resolution vision**: In future ultra-fine-grained patchification scenarios (50K+ tokens/image), the advantages of linear-complexity models will become even more critical.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The cross-architecture distillation framework is cleverly designed, with clear technical insight motivating the activation matching and masked prediction combination.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Multi-task validation across classification and segmentation, comprehensive ablations, and cross-teacher/cross-scale experiments.
- **Writing Quality**: ⭐⭐⭐⭐ — Well-organized, with sufficient motivation and concise formulations.
- **Value**: ⭐⭐⭐⭐ — Introduces a new technical route for ViT inference acceleration with meaningful implications for the development of recurrent vision models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] EA-ViT: Efficient Adaptation for Elastic Vision Transformer](ea-vit_efficient_adaptation_for_elastic_vision_transformer.md)
- [\[ICML 2025\] Distilling Tool Knowledge into Language Models via Back-Translated Traces](../../ICML2025/model_compression/distilling_tool_knowledge_into_language_models_via_back-translated_traces.md)
- [\[CVPR 2026\] Masking Teacher and Reinforcing Student for Distilling Vision-Language Models](../../CVPR2026/model_compression/masking_teacher_and_reinforcing_student_for_distilling_vision-language_models.md)
- [\[CVPR 2026\] Distilling Balanced Knowledge from a Biased Teacher](../../CVPR2026/model_compression/distilling_balanced_knowledge_from_a_biased_teacher.md)
- [\[ACL 2025\] Towards the Law of Capacity Gap in Distilling Language Models](../../ACL2025/model_compression/law_of_capacity_gap_distilling_language_models.md)

</div>

<!-- RELATED:END -->
