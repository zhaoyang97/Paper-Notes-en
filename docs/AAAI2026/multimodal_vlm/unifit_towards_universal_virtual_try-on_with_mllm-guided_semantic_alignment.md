---
title: >-
  [Paper Note] UniFit: Towards Universal Virtual Try-on with MLLM-Guided Semantic Alignment
description: >-
  [AAAI 2026][Multimodal VLM][Virtual try-on] This paper proposes UniFit, a universal virtual try-on framework driven by a multimodal large language model (MLLM). An MLLM-Guided Semantic Alignment (MGSA) module bridges the…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Virtual try-on"
  - "MLLM"
  - "semantic alignment"
  - "Diffusion Transformer"
  - "self-synthesis training"
date: 2026-05-08
content_hash: 3cd619a215ca03eb
---

# UniFit: Towards Universal Virtual Try-on with MLLM-Guided Semantic Alignment

**Conference**: AAAI 2026
**arXiv**: [2511.15831](https://arxiv.org/abs/2511.15831)  
**Code**: [github.com/zwplus/UniFit](https://github.com/zwplus/UniFit)  
**Area**: Multimodal VLM
**Keywords**: Virtual try-on, MLLM, semantic alignment, Diffusion Transformer, self-synthesis training

## TL;DR

This paper proposes UniFit, a universal virtual try-on framework driven by a multimodal large language model (MLLM). An MLLM-Guided Semantic Alignment (MGSA) module bridges the semantic gap between textual instructions and reference images. A two-stage progressive training strategy combined with a self-synthesis pipeline overcomes data scarcity in complex scenarios. UniFit is the first single framework to support all 6 VTON tasks.

## Background & Motivation

### Problem Definition

Image-based virtual try-on (VTON) aims to synthesize a realistic image of a person wearing a specified garment. Despite significant progress, building a universal VTON framework that flexibly handles diverse and complex tasks remains a major challenge.

### Root Cause

Existing text-instruction-guided VTON methods face two critical limitations:

**Semantic gap**: Abstract textual representations extracted by text encoders (e.g., CLIP or T5) struggle to precisely correspond to concrete visual details (textures, logo shapes, etc.) in images, resulting in low fidelity and weak controllability in generated outputs.

**Data scarcity**: Public datasets (e.g., VITON-HD, DressCode) only provide single-garment/try-on result pairs, lacking training data for complex scenarios such as multi-garment try-on and model-to-model try-on.

### Capability Comparison with Existing Methods

| Method | Single Try-on | Model-free Try-on | Garment Reconstruction | Multi-view | Multi-garment | Model-to-Model |
|------|---------|-----------|---------|--------|--------|----------|
| AnyFit | ✓ | - | - | - | ✓ | - |
| CatVTON | ✓ | - | - | - | - | ✓ |
| MV-VTON | ✓ | - | - | ✓ | - | - |
| Any2AnyTryon | ✓ | ✓ | ✓ | - | - | - |
| **UniFit** | **✓** | **✓** | **✓** | **✓** | **✓** | **✓** |

UniFit is the first method to support all 6 VTON tasks within a single framework.

## Method

### Overall Architecture

UniFit consists of three core components:

1. **MGSA module** (red): Encodes multimodal inputs into coherent semantic guidance.
2. **VAE encoder** (blue): Extracts low-level visual features from reference images.
3. **DiT (Diffusion Transformer)** (gray): Generates output images conditioned on semantic guidance and low-level visual features.

The generation pipeline contains two parallel streams:
- MGSA employs Qwen2-VL and learnable queries to capture the semantic relationship between textual instructions and reference images, producing a high-level semantic representation $T_q$.
- The VAE encoder processes reference images to extract fine-grained visual features $r = \{r_1, \ldots, r_n\}$.
- $T_q$, the noisy latent $z_t$, and reference tokens $r$ are concatenated as the DiT input $[T_q; z_t; r_1; \ldots; r_n]$.

### Key Designs

#### 1. **MLLM-Guided Semantic Alignment Module (MGSA)**: Bridging the Semantic Gap Between Text and Vision

**Mechanism**: MLLM (Qwen2-VL-2B-Instruct) is used to jointly process textual instructions and visual inputs, rather than handling them independently with separate text and image encoders as in prior methods.

**Learnable queries**: Learnable queries $T_q \in \mathbb{R}^{N_q \times D_q}$ ($N_q = 486$, $D_q = 1536$) are appended to the Qwen2-VL input sequence. Via causal attention, the queries distill task-relevant signals from the verbose multimodal sequence into a compact representation.

**Semantic alignment loss**: $T_q$ is aligned with the ground-truth visual representation $T_v$ of the target image (extracted via a frozen ViT):

$$\mathcal{L}_{\text{align}} = -\frac{1}{N_v} \sum_{n=1}^{N_v} \cos(T_{v,n}, \text{MLP}(T_{q,n}))$$

Token-level cosine similarity alignment ensures that the query representation semantically corresponds to the target output.

**Design Motivation**:
- Learnable queries address the redundancy and computational overhead caused by excessively long MLLM output sequences.
- The semantic alignment loss guides MGSA to fuse multimodal inputs and produce explicit guidance that is meaningful for DiT.
- Compared to abstract textual features from CLIP/T5, MLLM can jointly understand the semantic relationship between text and images.

#### 2. **Spatial Attention Focusing Loss**: Guiding DiT to Focus on Task-Relevant Regions

**Mechanism**: Cross-attention in DiT tends to disperse over irrelevant regions, causing detail degradation and visual artifacts. By explicitly regularizing cross-attention maps, the model is forced to focus on key regions.

The cross-attention map $AttnMap \in \mathbb{R}^{l_{r_i} \times l_{z_t}}$ is computed and, depending on task type:
- **Try-on tasks**: Averaged along the reference token axis to obtain an output-centric response map $M \in \mathbb{R}^{l_{z_t}}$.
- **Garment reconstruction tasks**: Averaged along the output token axis to obtain a reference-centric response map $M \in \mathbb{R}^{l_{r_i}}$.
- **Model-to-model**: Both response maps are computed and supervised simultaneously.

MSE loss aligns the maps with the ground-truth spatial mask $M_{\text{target}}$:

$$\mathcal{L}_{\text{focus}} = \frac{1}{N_R \times N_L} \sum_{j=1}^{N_L} \sum_{i=1}^{N_R} \|M_i^j - M_{\text{target},i}\|_2^2$$

**Design Motivation**: Although MGSA provides strong high-level semantic guidance, DiT's attention may still scatter, requiring explicit spatial constraints to ensure faithful transfer of fine-grained details.

#### 3. **Two-Stage Progressive Training + Self-Synthesis Pipeline**: Overcoming Data Scarcity in Complex Scenarios

**Stage I – Base Pretraining**:
- A Base Model is trained on VITON-HD and DressCode to learn three fundamental tasks (single try-on, garment reconstruction, model-free try-on).
- Approximately 59K training pairs per task, 120K steps.

**Self-Synthesis**:
- **Multi-garment try-on**: The garment reconstruction capability of the Base Model is leveraged to extract separate upper/lower garments from full-body images, creating 10K paired samples.
- **Model-to-model try-on**: The model-free try-on capability is used to synthesize new person images conditioned on existing garments, creating 30K paired samples.
- Dual filtering: DreamSim perceptual similarity + Qwen2.5-VL-7B-Instruct consistency check.

**Stage II – Joint Finetuning**:
- The model is fine-tuned for 80K steps on a mixture of real and synthesized data covering all 6 VTON tasks.

### Loss & Training

The overall training objective combines three loss terms:
1. **Flow matching loss**: Standard generative loss.
2. **Semantic alignment loss $\mathcal{L}_{\text{align}}$**: Cosine alignment between MGSA queries and target visual tokens.
3. **Spatial attention focusing loss $\mathcal{L}_{\text{focus}}$**: MSE between DiT cross-attention maps and spatial masks.

Training details:
- MGSA is based on Qwen2-VL-2B-Instruct; the first 14 layers are frozen and the last 14 layers are fine-tuned.
- DiT backbone: StableDiffusion-3.5 Medium.
- Training resolution: $1024 \times 768$ (for most tasks).
- AdamW optimizer, learning rate $4 \times 10^{-5}$, gradient clipping 1.0, batch size 16.

## Key Experimental Results

### Main Results

**Garment Reconstruction** (VITON-HD):

| Method | SSIM ↑ | LPIPS ↓ | DISTS ↓ | FID ↓ |
|------|--------|---------|---------|-------|
| TryOffDiff | 0.792 | 0.337 | 0.227 | 21.40 |
| Any2AnyTryon | 0.762 | 0.367 | 0.231 | 13.57 |
| **UniFit** | **0.775** | **0.281** | **0.202** | **12.58** |

**Single Try-on** (VITON-HD):

| Method | SSIM ↑ | LPIPS ↓ | FID ↓ | KID ↓ |
|------|--------|---------|-------|-------|
| CatVTON | 0.888 | 0.075 | 9.128 | 1.130 |
| FitDiT | 0.895 | 0.067 | 9.326 | 0.913 |
| Any2AnyTryon | 0.839 | 0.088 | 8.965 | 0.981 |
| **UniFit** | **0.883** | **0.065** | **8.799** | **0.702** |

FID is reduced to 8.799 (best), and KID is substantially reduced to 0.702.

**Model-free Try-on** (VITON-HD):

| Method | CLIP-AS ↑ | CLIP-I ↑ | MP-LPIPS ↓ |
|------|----------|---------|-----------|
| IMAGDressing-v1 | 4.96 | 0.880 | 0.107 |
| Any2AnyTryon | 4.95 | 0.843 | 0.127 |
| **UniFit** | 4.91 | **0.914** | **0.078** |

**Multi-view Try-on** (MVG):

| Method | SSIM ↑ | LPIPS ↓ | FID ↓ | KID ↓ |
|------|--------|---------|-------|-------|
| MV-TON | 0.930 | 0.062 | 37.09 | 3.23 |
| **UniFit** | **0.935** | 0.072 | **35.62** | 3.85 |

### Ablation Study

(VITON-HD single try-on, Stage I Base Model):

| Configuration | SSIM ↑ | LPIPS ↓ | FID ↓ | KID ↓ | Note |
|------|--------|---------|-------|-------|------|
| w/o MGSA (T5 replacement) | 0.851 | 0.098 | 9.133 | 1.053 | Significant overall degradation |
| w/o $\mathcal{L}_{\text{align}}$ | 0.863 | 0.074 | 8.937 | 0.951 | Alignment loss is important |
| w/o $\mathcal{L}_{\text{focus}}$ | 0.872 | 0.069 | 8.870 | 0.835 | Focusing loss contributes |
| **Full model (Stage I)** | **0.887** | **0.071** | **8.813** | **0.785** | Best |

### Key Findings

1. **MGSA is the core contribution**: Replacing MGSA with T5 causes SSIM to drop from 0.887 to 0.851 and FID to rise from 8.813 to 9.133—a substantial degradation.
2. **Semantic alignment loss is critical**: Without $\mathcal{L}_{\text{align}}$, SSIM drops from 0.887 to 0.863, demonstrating that explicit semantic alignment is essential for guidance quality.
3. **Self-synthesis pipeline is effective**: Data synthesized by the Base Model enables the model to handle complex tasks (multi-garment, model-to-model) that are absent from the original training data.
4. **Multi-task framework performs excellently**: Achieving SOTA or competitive performance across all 6 tasks validates the feasibility of a universal framework.

## Highlights & Insights

1. **Novel integration of MLLM into VTON**: MLLM is deeply integrated into the VTON framework for the first time—not merely for instruction understanding, but as a bridge to close the multimodal semantic gap.
2. **Elegant self-synthesis training strategy**: The already-trained model is used to generate training data for new tasks, forming a positive feedback loop that elegantly resolves the data scarcity bottleneck.
3. **Spatial attention focusing loss**: Explicitly supervising DiT's attention maps addresses the fine-grained detail degradation caused by attention dispersion.
4. **Mask-free design**: The framework does not rely on garment masks; instead, image inpainting is used to synthesize triplet training samples, improving practicality.

## Limitations & Future Work

1. **Uncertain performance in the wild**: Due to the distribution of existing datasets (predominantly studio settings), performance may degrade under extreme lighting or heavy occlusion.
2. **No layered try-on support**: The framework cannot handle garment layering (e.g., a jacket over an inner shirt).
3. **No text-editable try-on**: Garment attributes cannot be modified via textual descriptions (e.g., "change to red").
4. **Self-synthesized data quality**: Despite DreamSim + Qwen2.5-VL filtering, distributional bias in synthesized data may limit generalization.
5. **High computational cost**: The combination of MLLM (Qwen2-VL-2B) and DiT (SD3.5-Medium) may incur substantial inference overhead.

## Related Work & Insights

- **CatVTON / FitDiT**: DiT-based single-task VTON methods; UniFit extends these to multi-task settings.
- **Any2AnyTryon**: The closest prior work to universal VTON, but lacking multi-garment and model-to-model support, with weaker text instruction following.
- **DreamO**: Source of inspiration for the spatial attention focusing loss.
- **Qwen2-VL**: The MLLM backbone adopted by UniFit, providing multimodal understanding capabilities.
- **Insight**: MLLMs can serve not only for understanding and generation, but also as "semantic bridges" connecting different modalities to guide generative models.

## Rating

- Novelty: ⭐⭐⭐⭐ (The integration of MLLM with VTON is novel; the self-synthesis training is also highly creative.)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive evaluation across 6 tasks, multiple datasets, and complete ablations.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure, rich illustrations, well-defined problem formulation.)
- Value: ⭐⭐⭐⭐⭐ (The first universal framework supporting 6 VTON tasks, with extremely high practical utility.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Exo2Ego: Exocentric Knowledge Guided MLLM for Egocentric Video Understanding](exo2ego_exocentric_knowledge_guided_mllm_for_egocentric_vide.md)
- [\[AAAI 2026\] Learning to Tell Apart: Weakly Supervised Video Anomaly Detection via Disentangled Semantic Alignment](learning_to_tell_apart_weakly_supervised_video_anomaly_detection_via_disentangle.md)
- [\[CVPR 2026\] Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models](../../CVPR2026/multimodal_vlm/uncertainty-guided_compositional_alignment_with_part-to-whole_semantic_represent.md)
- [\[NeurIPS 2025\] Guiding Cross-Modal Representations with MLLM Priors via Preference Alignment](../../NeurIPS2025/multimodal_vlm/guiding_cross-modal_representations_with_mllm_priors_via_preference_alignment.md)
- [\[CVPR 2026\] VLM-Guided Group Preference Alignment for Diffusion-based Human Mesh Recovery](../../CVPR2026/multimodal_vlm/vlm-guided_group_preference_alignment_for_diffusion-based_human_mesh_recovery.md)

</div>

<!-- RELATED:END -->
