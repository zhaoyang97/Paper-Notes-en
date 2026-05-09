---
title: >-
  [Paper Note] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging
description: >-
  [NeurIPS 2025][Multimodal VLM][3D medical VLM] This paper proposes BTB3D, a 3D CT tokenizer based on causal convolutional codec, 3D Haar wavelet compression, and a three-stage progressive training strategy. It achieves substantial state-of-the-art improvements on two downstream tasks—radiology report generation and text-conditioned CT synthesis—demonstrating that "better tokens matter more than larger language models."
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - 3D medical VLM
  - CT tokenization
  - causal convolution
  - wavelet transform
  - report generation
  - text-to-CT synthesis
date: 2026-05-08
content_hash: a73e797ca849d6b6
---

# Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging

**Conference**: NeurIPS 2025
**arXiv**: [2510.20639](https://arxiv.org/abs/2510.20639)
**Code**: [ibrahimethemhamamci/BTB3D](https://github.com/ibrahimethemhamamci/BTB3D)
**Area**: Multimodal VLM
**Keywords**: 3D medical VLM, CT tokenization, causal convolution, wavelet transform, report generation, text-to-CT synthesis

## TL;DR

This paper proposes BTB3D, a 3D CT tokenizer based on causal convolutional codec, 3D Haar wavelet compression, and a three-stage progressive training strategy. It achieves substantial state-of-the-art improvements on two downstream tasks—radiology report generation and text-conditioned CT synthesis—demonstrating that "better tokens matter more than larger language models."

## Background & Motivation

**The bottleneck in 3D medical VLMs lies in the visual encoder, not the LLM**: Existing methods (CT-CHAT, Merlin) rely on CLIP-style contrastive pretraining for visual representations, but the large volume of 3D CT scans constrains training to small batches and shallow models, resulting in poor encoding quality.

**The assumptions of contrastive learning do not hold in radiology**: Contrastive learning assumes that unpaired samples are semantically unrelated; however, different radiologists may describe the same lesion in highly varied styles, and forcing such samples apart degrades semantic understanding.

**Slice-level tokenization discards spatial continuity**: Existing methods encode 2D slices independently and concatenate them, leading to loss of fine-grained anatomical structures (e.g., small nodules, textures) and poor cross-slice consistency.

**2D and 3D representations are incompatible**: Large-scale 2D pretrained models cannot directly process volumetric data, while 3D annotated data is extremely scarce, making transfer difficult.

**Codec bottlenecks in text-conditioned 3D generation**: GenerateCT employs cascaded 2D upsampling, causing inter-slice discontinuities; MedSyn uses a lightweight 3D architecture with insufficient fidelity.

**Core Insight**: What is needed is a codec tokenizer that can (a) scale to arbitrarily long sequences, (b) unify 2D/3D training, and (c) decode with high fidelity—rather than focusing solely on the LLM side.

## Method

### Overall Architecture

BTB3D = **3D Haar Wavelet Compression** → **Causal 3D Convolutional Encoder** → **Lookup-Free Quantization (LFQ)** → **Causal Decoder** → **Inverse Wavelet Reconstruction**. Training follows a three-stage progressive strategy, proceeding from short clips to full CT volumes.

### Key Designs

#### 1. 3D Haar Wavelet Compression

- A 3D Haar wavelet transform is applied to the input $x \in \mathbb{R}^{D \times H \times W}$, producing $\mathbb{R}^{D/2 \times H/2 \times W/2 \times 8}$ (1 low-frequency + 7 high-frequency subbands).
- Resolution is reduced by $2\times$ along each axis while retaining frequency information, shrinking the volume by $8\times$ and substantially reducing downstream computation.
- The transform is equally applicable to 2D slices ($D=1$), where high-frequency components along the z-axis approach zero, preserving architectural consistency.

#### 2. Causal Convolutional Codec

- Spatial convolutions $1 \times k \times k$ (sagittal–coronal planes) and temporal convolutions $k \times 1 \times 1$ (axial direction) are used with causal padding, applying zero-padding only in the past direction.
- The $t$-th token depends only on slices $\leq t$, preventing future leakage and enabling natural compatibility with autoregressive generation.
- **8×8×8 variant**: 2 stride-2 spatial downsampling steps + 2 stride-2 temporal downsampling steps, combined with the $2\times$ wavelet compression, yielding $8\times$ compression in all directions—suitable for fine-grained generation tasks.
- **16×16×8 variant**: An additional stride-2 spatial step provides $16\times$ spatial compression—suitable for memory-constrained semantic tasks such as report generation.

#### 3. Lookup-Free Quantization (LFQ)

- The encoder output is binarized dimension-wise via the sign function to obtain $b \in \{-1, 1\}^d$, yielding a codebook of size $K = 2^d$ (e.g., $K = 262{,}144$ for $d = 18$).
- No explicit codebook lookup is required, offering high speed and memory efficiency.
- Entropy regularization $\mathcal{L}_{\text{entropy}}$ is applied to prevent codebook collapse.

### Three-Stage Training Strategy

| Stage | Components Trained | Input Length | Primary Objective |
|---|---|---|---|
| Stage 1 | Encoder + Decoder + Quantizer | Single slice / 9-slice sub-volume | Learn local spatial and short-range temporal features |
| Stage 2 | Encoder + Decoder | 201 slices (overlapping window tiling) | Extend to long sequences; maintain temporal consistency |
| Stage 3 | Decoder only (encoder + codebook frozen) | 241 slices | Enhance long-range anatomical dependency modeling |

**Overlapping window tiling mechanism**: Each 9-slice window produces 2 tokens. For the first window, both tokens are retained; for subsequent windows, the first token (covering only the overlapping slice) is discarded and the second token (covering all 9 slices) is retained. The final sequence is assembled as $[z_1^1, z_2^1, z_2^2, \ldots, z_2^T]$.

### Loss & Training

$$\mathcal{L} = \underbrace{\|x - \hat{x}\|_1}_{\text{Reconstruction (L1)}} + \lambda_{\text{adv}} \underbrace{(-\log D(\hat{x}))}_{\text{Adversarial}} + \underbrace{\|sg[y] - e\|_2^2 + \beta\|y - sg[e]\|_2^2}_{\text{Quantization}}$$

- L1 reconstruction loss (sharper than L2).
- A 3D discriminator applies adversarial supervision in the CT domain rather than the wavelet domain.
- VGG perceptual loss is deliberately excluded, as the domain gap between natural images and grayscale medical images is shown experimentally to degrade performance.

## Key Experimental Results

### Reconstruction Quality: Progressive Gains Across Three Stages

| Stage | Compression | PSNR ↑ | SSIM ↑ | MSE ↓ |
|---|---|---|---|---|
| Stage 1 | 8³ | 9.35 | 0.206 | 0.117 |
| Stage 2 | 8³ | 23.98 | 0.697 | 0.005 |
| **Stage 3** | **8³** | **28.17** | **0.760** | **0.001** |
| Stage 1 | 16²×8 | 11.07 | 0.353 | 0.079 |
| Stage 2 | 16²×8 | 23.81 | 0.700 | 0.005 |
| **Stage 3** | **16²×8** | **26.75** | **0.749** | **0.002** |

**Key Findings**: Stage 2 yields the largest improvement (PSNR +14 dB, SSIM $\times$3); Stage 3 refines inter-slice detail.

### Main Results

#### Radiology Report Generation (CT-RATE Dataset)

| Model | F1 ↑ | Precision ↑ | Recall ↑ | CRG ↑ | BLEU-1 ↑ | BLEU-mean ↑ | METEOR ↑ |
|---|---|---|---|---|---|---|---|
| CT2Rep | 0.160 | 0.435 | 0.128 | 0.359 | 0.372 | 0.280 | 0.197 |
| Merlin | 0.160 | 0.295 | 0.112 | 0.352 | 0.231 | 0.154 | 0.148 |
| CT-CHAT | 0.184 | 0.450 | 0.158 | 0.368 | 0.373 | 0.272 | 0.215 |
| **BTB3D-16** | **0.258** | 0.260 | **0.260** | **0.370** | **0.439** | **0.305** | **0.223** |

- Clinical F1 improves by **40%** over CT-CHAT; BLEU-1 improves by 18%.
- On the OOD dataset RadChestCT, F1 reaches 0.266, a **46% improvement** over the strongest baseline.
- Merlin exhibits high precision but low recall (missed findings); CT2Rep/CT-CHAT show high recall but low precision (hallucinations); BTB3D achieves the best balance.

#### Text-Conditioned CT Synthesis

| Model | FID-mean ↓ | FVD (CT-Net) ↓ | FVD (I3D) ↓ | CLIP Text-Img ↑ |
|---|---|---|---|---|
| GenerateCT | 9.51 | 7.66 | 1512.5 | 23.63 |
| MedSyn | 12.59 | 13.93 | 725.8 | 23.57 |
| **BTB3D-8** | **2.24** | **3.96** | **325.5** | **24.27** |

- FID decreases by **76.5%** (9.51 → 2.24); FVD decreases by **48.3%**.
- Full-resolution 512×512×241 volumes are generated with clear anatomical structures and inter-slice consistency.
- The 8³ variant is superior for generation tasks; the 16²×8 variant is superior for report generation, reflecting a task-dependent compression-rate trade-off.

### Ablation Study

**Experimental Setup**:
- Dataset: CT-RATE (25,692 chest CT cases; 20,000 train / 1,304 test).
- Training resources: 64×H100 GPUs (Stages 1–3); 40×H100 (report generation); 16×H100 (CT synthesis).
- LLM: LLaMA 3.1-8B (LoRA fine-tuning), using the same configuration as CT-CHAT for fair comparison.

## Highlights & Insights

1. **Profound core insight**: "Better tokens, not larger LLMs"—the work demonstrates that high-quality 3D tokenization, rather than scaling the language backbone, drives VLM improvements, validated across two distinct tasks.
2. **Elegant causal convolution design**: Unifies 2D/3D training and inference, supports CT scans of arbitrary length, and is inherently compatible with autoregressive generation.
3. **Practical three-stage training**: Enables generalization from 9-slice training clips to 241-slice full-volume inference, effectively addressing the short-training-to-long-inference gap.
4. **Simple yet effective overlapping window tiling**: Discards overlapping tokens while retaining fully-covered tokens, maintaining temporal consistency at low memory cost.
5. **Comprehensive experiments with large margins**: Report generation F1 +40%, CT synthesis FID −76.5%; both fundamentally different downstream tasks substantially surpass prior methods.

## Limitations & Future Work

1. **Validation limited to chest CT**: All experiments are conducted on CT-RATE (chest CT); generalization to other anatomical regions (abdomen, head and neck, etc.) remains untested.
2. **Extremely high computational requirements**: Three-stage training requires 64 H100 GPUs in total, making reproduction difficult for most research groups.
3. **Dense prediction tasks not evaluated**: Only report generation and image synthesis are assessed; the utility of the tokens for pixel-level tasks such as segmentation and detection is not examined.
4. **Fixed LFQ codebook size**: $d=18$ yields 262K codewords; the effect of varying codebook scale is not explored.
5. **Inference efficiency not fully discussed**: Tiled inference requires multiple windowed encoding passes; latency implications for real-time clinical deployment are not analyzed.

## Related Work & Insights

| Method | Visual Encoding | Pretraining Strategy | Limitations |
|---|---|---|---|
| CT2Rep | CT-ViT | None | No pretrained model; weak generation capability |
| CT-CHAT | CLIP contrastive pretraining | Contrastive learning | Contrastive learning assumptions ill-suited for radiology |
| Merlin | I3D-ResNet | Masked + contrastive | High precision, low recall; pretrained weights not publicly available |
| GenerateCT | Low-res 3D AE + cascaded 2D diffusion | Self-supervised | Inter-slice discontinuities; cascade introduces artifacts |
| MedSyn | Lightweight 3D Transformer | Self-supervised | Insufficient fidelity |
| **BTB3D** | **Causal 3D CNN + Wavelet + LFQ** | **Three-stage reconstruction** | **Unified codec; direct end-to-end approach** |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of causal convolution, wavelet compression, and three-stage training is pioneering in 3D medical VLMs, with clearly motivated design choices.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Two downstream tasks, reconstruction ablations, OOD validation, comprehensive baselines, and both quantitative and qualitative analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure, rich figures and tables, and a coherent motivation–method–experiment narrative.
- **Value**: ⭐⭐⭐⭐ — Reveals the critical role of tokenization quality in 3D medical VLMs and has the potential to shift the field's paradigm; open-source code further amplifies its impact.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Learning from Videos for 3D World: Enhancing MLLMs with 3D Vision Geometry Priors](learning_from_videos_for_3d_world_enhancing_mllms_with_3d_vision_geometry_priors.md)
- [\[NeurIPS 2025\] Are Vision Language Models Ready for Clinical Diagnosis? A 3D Medical Benchmark for Tumor-centric Visual Question Answering](are_vision_language_models_ready_for_clinical_diagnosis_a_3d_medical_benchmark_f.md)
- [\[NeurIPS 2025\] Text to Robotic Assembly of Multi Component Objects using 3D Generative AI and Vision Language Models](text_to_robotic_assembly_of_multi_component_objects_using_3d_generative_ai_and_v.md)
- [\[NeurIPS 2025\] AffordBot: 3D Fine-grained Embodied Reasoning via Multimodal Large Language Models](affordbot_3d_fine-grained_embodied_reasoning_via_multimodal_large_language_model.md)
- [\[NeurIPS 2025\] Situat3DChange: Situated 3D Change Understanding Dataset for Multimodal Large Language Models](situat3dchange_situated_3d_change_understanding_dataset_for_multimodal_large_lan.md)

<!-- RELATED:END -->
