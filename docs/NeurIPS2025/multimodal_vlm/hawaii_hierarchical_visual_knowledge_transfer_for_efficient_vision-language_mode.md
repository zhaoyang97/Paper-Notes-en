---
title: >-
  [Paper Note] HAWAII: Hierarchical Visual Knowledge Transfer for Efficient VLM
description: >-
  [NeurIPS 2025][Multimodal VLM][Knowledge Distillation] This paper proposes the Hawaii framework, which distills knowledge from multiple visual experts into a single visual encoder via Mixture of LoRA Adapters (MoLA) and…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "Knowledge Distillation"
  - "Visual Encoder"
  - "LoRA"
  - "MoE"
  - "Multi-Teacher Distillation"
date: 2026-05-08
content_hash: 84e34ad5bc7520a3
---

# HAWAII: Hierarchical Visual Knowledge Transfer for Efficient VLM

**Conference**: NeurIPS 2025  
**arXiv**: [2506.19072](https://arxiv.org/abs/2506.19072)  
**Code**: [Available](https://github.com/yimuwangcs/wise-hawaii)  
**Area**: Multimodal VLM  
**Keywords**: Knowledge Distillation, Visual Encoder, LoRA, MoE, Multi-Teacher Distillation

## TL;DR

This paper proposes the Hawaii framework, which distills knowledge from multiple visual experts into a single visual encoder via Mixture of LoRA Adapters (MoLA) and Hierarchical Knowledge Distillation (HKD), significantly improving the visual understanding capability of VLMs without incurring any additional inference cost.

## Background & Motivation

The performance of VLMs depends heavily on the capability of the visual encoder. Recent studies have shown that integrating multiple visual experts (e.g., SAM, ConvNeXt, EVA) can substantially boost performance, but introduces serious efficiency issues:

**High inference-time cost**: Multi-expert approaches require computing visual tokens from all experts during both training and inference, resulting in significant computational and latency overhead.

**Multi-teacher knowledge conflicts**: Different teachers vary in training data, architecture, and objectives, and naive distillation leads to noisy and redundant knowledge transfer.

**Insufficiency of existing distillation methods**: MoVE-KD employs a fixed set of LoRA adapters for all teachers, making it difficult to effectively disentangle knowledge from different teachers.

Core problem: **How to efficiently absorb complementary knowledge from multiple visual experts while maintaining single-encoder inference efficiency?**

## Method

### Overall Architecture

Hawaii follows the standard VLM architecture (visual encoder → projector → LLM), with the key innovation residing in the visual encoder. It comprises two core modules:
1. **MoLA (Mixture of LoRA Adapters)**: Manages teacher-specific and general-purpose knowledge adapters.
2. **HKD (Hierarchical Knowledge Distillation)**: Performs knowledge distillation at both fine-grained and coarse-grained levels.

### Key Designs

#### 1. Mixture of LoRA Adapters (MoLA)

MoLA is applied to every feed-forward layer of the student encoder (CLIP) and consists of two groups of adapters:

**Teacher-specific LoRA adapters** $\{a_i^T\}_{i=1}^{N_t}$:
- Each adapter aligns to a single teacher, avoiding inter-teacher knowledge conflicts.
- Dynamically selected by a sparse router $f_r^T(\cdot)$ based on the hidden state input.

**General-purpose LoRA adapters** $\{a_i^G\}_{i=1}^{N_g}$:
- Learn the collective consensus across multiple teachers.
- Selected by an independent sparse router $f_r^G(\cdot)$.

The feed-forward layer output is: $F^*(h) = F(h) + a_i^T(h) + a_j^G(h)$, where $i = \text{argmax}(f_r^T(h))$ and $j = \text{argmax}(f_r^G(h))$.

Each adapter is a LoRA block (rank=32); routers are 2-layer MLPs with GELU activation. Only the top-1 adapter is activated at each step (sparse design).

#### 2. Coarse-Grained Knowledge Distillation (CGKD)

Objective: Distill the collective consensus from multiple teachers.

- Teacher visual features are unified to the student's token length via pixel unshuffle.
- Features are concatenated and passed through a 2-layer MLP to produce an aggregated representation: $I_{cg}^T = f_{cg}(\text{Concat}(I_1^T, I_2^T, ..., I_{N_t}^T))$
- An MSE loss aligns the student output with the aggregated features: $\mathcal{L}_{cg} = \text{MSE}(I^S, I_{cg}^T)$

General-purpose LoRA adapters are active at this stage to facilitate global alignment.

#### 3. Fine-Grained Knowledge Distillation (FGKD)

Objective: Precisely learn the unique knowledge from each individual teacher.

**Teacher-specific adaptation**: When the $i$-th teacher-specific LoRA is activated, the student output $I_i^S$ is aligned only to the $i$-th teacher's features $I_i^T$.

**Token importance scoring**: Not all tokens are equally informative. The most informative tokens are selected via a similarity-based score:
$$s_i = \text{mean}\left(\text{softmax}\left(\frac{\text{Concat}(\hat{I}_i^T, \hat{T})(\hat{I}_i^T)^\top}{\sqrt{D}}\right)\right)$$

This score jointly considers teacher visual tokens and the input text instruction $T$, prioritizing tokens more relevant to the task.

Fine-grained distillation loss: $\mathcal{L}_{fg} = \frac{1}{N_t} \sum_{i=1}^{N_t} s_i \cdot \text{MSE}(I_i^S, \hat{I}_i^T)$

### Loss & Training

**Overall training objective**:
$$\mathcal{L} = \mathcal{L}_{gen} + \lambda_1(\mathcal{L}_{fg} + \mathcal{L}_{cg}) + \lambda_2 \mathcal{L}_{mb}$$

- $\mathcal{L}_{gen}$: Autoregressive text generation loss
- $\lambda_1 = 0.5$: Distillation loss weight
- $\lambda_2 = 0.05$: MoE load balancing loss weight
- $\mathcal{L}_{mb}$: MoE load balancing loss

**Two-stage training** (following the LLaVA-1.5 paradigm):
1. **Pre-training**: 558K image-text pairs; only the projector, LoRA adapters, and routers are trained.
2. **Instruction tuning**: 665K instruction-following samples; full model training.

**Teacher configurations**:
- Hawaii (base): CLIP + ConvNeXt + EVA-02 (3 teachers)
- Hawaii†: additionally includes Pix2Struct (4 teachers)
- Hawaii‡: CLIP + ConvNeXt + EVA-02 + SAM

Hardware: 8 × NVIDIA A6000 (48GB)

## Key Experimental Results

### Main Results

| Method | VQA-T | VizWiz | GQA | SQA | POPE | MME | MMB | MMMU | SeedB |
|--------|-------|--------|-----|-----|------|-----|-----|------|-------|
| LLaVA-1.5 (Baseline) | 58.2 | 50.0 | 62.0 | 66.8 | 85.9 | 1510.7 | 64.3 | 34.7 | 66.1 |
| MoVE-KD | 58.3 | 52.3 | 63.2 | 69.4 | 86.9 | 1524.5 | 66.3 | - | - |
| **Hawaii** | **58.7** | **53.9** | **62.8** | **70.5** | **87.3** | **1540.2** | **66.9** | **36.6** | **67.5** |
| Δ vs Baseline | +0.5 | +3.9 | +0.8 | +3.7 | +1.4 | +29.5 | +2.6 | +1.9 | +1.4 |

Hawaii outperforms both LLaVA-1.5 and MoVE-KD across all benchmarks. The most notable gains are on VizWiz (+3.9%) and SQA (+3.7%).

### Ablation Study

| Configuration | Avg. |
|---------------|------|
| LLaVA-1.5 (Baseline) | 61.9 |
| + FGKD (w/o token scoring) | 63.2 |
| + token scoring | 63.5 |
| + CGKD (full Hawaii) | **63.7** |

| # General Adapters | MME | POPE | SeedB |
|--------------------|-----|------|-------|
| 1 | 1516.2 | 84.5 | 67.4 |
| **3** | **1540.2** | **87.3** | **67.5** |
| 5 | 1530.2 | 85.2 | 66.9 |

### Key Findings

1. **Each component contributes incrementally**: FGKD → token scoring → CGKD, validating the effectiveness of the hierarchical design.
2. **Teacher-specific LoRA outperforms shared LoRA**: Compared to MoVE-KD (shared adapters), Hawaii's dedicated adapter strategy yields superior results.
3. **Three general-purpose adapters is optimal**: Performance degrades with too many (5) or too few (1).
4. **Effectiveness scales to 13B models**: Hawaii-13B also outperforms LLaVA-1.5-13B and MoVE-KD-13B on corresponding benchmarks.
5. **Teacher combination matters**: Adding SAM (Hawaii‡) outperforms adding Pix2Struct (Hawaii†) on certain tasks.

## Highlights & Insights

1. **Zero inference overhead**: After distillation, only a single visual encoder (with LoRA) is used, incurring the same inference cost as the baseline.
2. **Elegant MoE design in MoLA**: The dual-routing mechanism of teacher-specific and general-purpose adapters avoids inter-teacher conflicts while capturing cross-teacher consensus.
3. **Multi-signal token importance scoring**: Jointly leveraging teacher visual features and text instructions is more principled than vision-only token selection.
4. **Generalizable hierarchical distillation paradigm**: FGKD enables precise individual alignment; CGKD ensures global consistency—the two are complementary.

## Limitations & Future Work

1. Teacher models still require forward passes during training (inference overhead is zero only after distillation), leading to substantial training-time computation.
2. Comparisons are limited to LLaVA-1.5 as the baseline; stronger VLMs (e.g., InternVL2, Qwen-VL2) are not evaluated.
3. Teacher selection is relatively fixed (CLIP/ConvNeXt/EVA/SAM); an automatic teacher selection mechanism is lacking.
4. The LoRA rank (32) and adapter count are set empirically without systematic search.
5. Dynamic routing weights (instead of hard top-1 selection) could be explored.

## Related Work & Insights

- **Key difference from MoVE-KD**: MoVE-KD uses a fixed set of LoRA adapters shared across all teachers, whereas Hawaii employs teacher-specific adapters to eliminate inter-teacher conflicts.
- **Key difference from Eagle/Cambrian**: These methods require multiple experts at inference time, while Hawaii relies on multiple experts only during training.
- **Broader inspiration**: The MoLA design principle is generalizable to other multi-source knowledge fusion scenarios, such as multimodal fusion and multi-task learning.

## Rating

- Novelty: ⭐⭐⭐⭐ (The combination of MoLA and hierarchical distillation is creative)
- Experimental Thoroughness: ⭐⭐⭐⭐ (10 benchmarks + detailed ablations, though the baseline is the relatively dated LLaVA-1.5)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure and intuitive figures)
- Value: ⭐⭐⭐⭐ (Zero inference overhead with consistent improvements; strong practical utility)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Harnessing Textual Semantic Priors for Knowledge Transfer and Refinement in CLIP-Driven Continual Learning](../../AAAI2026/multimodal_vlm/harnessing_textual_semantic_priors_for_knowledge_transfer_and_refinement_in_clip.md)
- [\[ICCV 2025\] SparseVILA: Decoupling Visual Sparsity for Efficient VLM Inference](../../ICCV2025/multimodal_vlm/sparsevila_decoupling_visual_sparsity_for_efficient_vlm_inference.md)
- [\[NeurIPS 2025\] SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation](spatialtracegen_high-fidelity_traces_for_efficient_vlm_spatial_reasoning_distill.md)
- [\[CVPR 2026\] KEC: Hierarchical Textual Knowledge for Enhanced Image Clustering](../../CVPR2026/multimodal_vlm/kec_hierarchical_textual_knowledge_clustering.md)
- [\[NeurIPS 2025\] BioCLIP 2: Emergent Properties from Scaling Hierarchical Contrastive Learning](bioclip_2_emergent_properties_from_scaling_hierarchical_contrastive_learning.md)

</div>

<!-- RELATED:END -->
