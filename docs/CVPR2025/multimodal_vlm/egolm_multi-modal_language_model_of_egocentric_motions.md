---
title: >-
  [Paper Note] EgoLM: Multi-Modal Language Model of Egocentric Motions
description: >-
  [CVPR 2025][Multimodal VLM][Egocentric motions] This work proposes a unified multimodal language model framework that integrates egocentric motion tracking (sparse sensors $\rightarrow$ full-body motion) and motion understanding (motion $\rightarrow$ language description). By combining a VQ-VAE motion tokenizer and a GPT-2 backbone, the framework jointly models four modalities (text, motion tokens, sensors, and video). Incorporating egocentric video reduces tracking errors by…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Egocentric motions"
  - "sparse sensor tracking"
  - "motion-language model"
  - "VQ-VAE"
  - "multimodal unification"
date: 2026-05-08
content_hash: 84e56ffe0d9e8e80
---

# EgoLM: Multi-Modal Language Model of Egocentric Motions

**Conference**: CVPR 2025  
**arXiv**: [2409.18127](https://arxiv.org/abs/2409.18127)  
**Code**: [https://hongfz16.github.io/projects/EgoLM](https://hongfz16.github.io/projects/EgoLM)  
**Area**: Multimodal VLM  
**Keywords**: Egocentric motions, sparse sensor tracking, motion-language model, VQ-VAE, multimodal unification

## TL;DR
This work proposes a unified multimodal language model framework that integrates egocentric motion tracking (sparse sensors $\rightarrow$ full-body motion) and motion understanding (motion $\rightarrow$ language description). By combining a VQ-VAE motion tokenizer and a GPT-2 backbone, the framework jointly models four modalities (text, motion tokens, sensors, and video). Incorporating egocentric video reduces tracking errors by 10-20mm.

## Background & Motivation

**Background**: Egocentric motion tracking recovers full-body poses from sparse sensors (3-point or 1-point 6-DoF) on head-mounted devices, which is a core task for AR/VR. Current methods such as AvatarPoser and BoDiffusion rely solely on sensor data and cannot leverage the egocentric cameras of head-mounted devices. Meanwhile, motion understanding (translating motion into natural language descriptions) is investigated as an independent research direction, separate from tracking tasks.

**Limitations of Prior Work**: (1) Sparse sensor tracking is highly under-constrained—recovering 22 joints from 3 sensors leaves the lower body almost unconstrained, leading to massive lower limb errors ($>150$mm). (2) Egocentric video contains rich environmental and interaction clues for disambiguation, but existing tracking methods cannot fuse video information. (3) Motion tracking and motion understanding are highly correlated but studied in isolation.

**Key Challenge**: The information provided by sparse sensors is insufficient for accurate full-body and lower-body motion recovery, requiring an additional modality (video) to provide constraints; meanwhile, motion and language belong to different modalities, and a unified modeling framework is lacking.

**Goal**: To design a unified framework that simultaneously handles egocentric motion tracking and motion understanding, leveraging egocentric video to disambiguate sensor data and enabling bidirectional translation between motion and language.

**Key Insight**: To quantize motion into discrete tokens (using VQ-VAE) that share the same vocabulary space as text tokens, enabling GPT-2 to utilize next-token prediction to uniformly handle both tracking (generating motion tokens) and understanding (generating text tokens) tasks.

**Core Idea**: Discretize continuous motion sequences into tokens using VQ-VAE, and uniformly model four modalities (sensors, video, motion tokens, and text) via GPT-2 to achieve joint optimization of tracking and understanding.

## Method

### Overall Architecture
Three-stage training: (1) VQ-VAE motion tokenizer training (encoding continuous motion into discrete tokens) $\rightarrow$ (2) Motion pre-training (GPT-2 performing next-token prediction on motion token sequences) $\rightarrow$ (3) Multimodal instruction tuning (introducing sensor and video encoders to train four tasks: tracking, understanding, M2T, and T2M).

### Key Designs

1. **Motion VQ-VAE Tokenizer (Product Quantization)**:

    - **Function**: Compresses the 279-dimensional/frame continuous motion representation into a discrete token sequence.
    - **Mechanism**: Fully convolutional encoder-decoder architecture with $4\times$ temporal downsampling. The key innovation is Product Quantization, which splits the latent features into $N=2$ segments, each independently quantized using an 8192-sized codebook (dimension 64). Ultimately, each frame generates $N \times (T/r) = 2 \times (T/4)$ tokens. The reconstruction loss consists of three terms: raw representation, joint position, and rotational velocity.
    - **Design Motivation**: Single-codebook quantization (PQ=1) yields an MPJPE of 51.6mm, whereas Product Quantization (PQ=2) reduces it to 34.5mm ($-33\%$). This is because the dual-codebook combination provides $8192^2 \approx 67\text{M}$ effective code entries, substantially improving representation accuracy.

2. **Egocentric Video Disambiguation**:

    - **Function**: Provides additional visual constraints for the under-constrained sparse sensor tracking task.
    - **Mechanism**: Each frame of egocentric video is processed by a CLIP image encoder to extract features, which are then mapped to the LLM feature space via linear projection. The video features are concatenated with the output of the sensor encoder to serve as conditional inputs for GPT-2.
    - **Design Motivation**: Ablation studies show that after incorporating video, the full-body error for 3-point tracking drops from 83.88mm to 73.38mm ($-12.5\%$), and from 127.45mm to 106.95mm ($-16.1\%$) for 1-point tracking. Video provides environmental clues about "what the person is doing" (e.g., walking, bending down, jumping), with the lower limbs showing the most notable improvement (3-point: 148.37 to 124.58mm).

3. **Multi-Task Instruction Tuning**:

    - **Function**: Unifies the training of both tracking and understanding tasks.
    - **Mechanism**: Instruction templates are designed to distinguish between four tasks: tracking (sensors + video $\rightarrow$ motion tokens), understanding (sensors + video $\rightarrow$ text), M2T (motion $\rightarrow$ text), and T2M (text $\rightarrow$ motion). All tasks share GPT-2 parameters, with instruction templates routing input and output formats.
    - **Design Motivation**: Joint training allows tracking supervision to assist understanding—the motion priors provided by motion tracking improve the quality of language descriptions. Ablation results show that the understanding performance of joint training (BERT score 19.40) is close to that of a cascaded approach (19.97).

### Loss & Training
**VQ-VAE stage**: Reconstruction loss (raw representation + joint positions + rotational velocity) + commitment loss + EMA codebook update. **LM stage**: Next-token prediction cross-entropy loss. The model backbone is GPT-2 Medium (345M), and GPT-2 Large (1.5B) was also tested. The dataset used is Nymeria (147.89 hours of training data).

## Key Experimental Results

### Main Results

| Method | Input | Full Body (mm) | Upper Body (mm) | Lower Body (mm) |
|------|------|---------|-----------|-----------|
| AvatarPoser | 3pts | 85.89 | 52.78 | 165.18 |
| BoDiffusion | 3pts | 79.80 | 52.79 | 152.68 |
| EgoLM | 3pts | 83.88 | 54.06 | 148.37 |
| **EgoLM** | **3pts+Vid** | **73.38** | **49.67** | **124.58** |
| AvatarPoser† | 1pt | 129.23 | 94.19 | 192.34 |
| **EgoLM** | **1pt+Vid** | **106.95** | **83.73** | **141.26** |

### Ablation Study

| Configuration | MPJPE | Description |
|------|-------|------|
| VQ-VAE PQ=1 | 51.60mm | Single codebook |
| VQ-VAE PQ=2 | **34.49mm** | Dual codebook, -33% |
| 60 frames without video | 83.88mm | Baseline |
| 120 frames without video | 79.61mm | Long window helps |
| 60 frames + video | **73.38mm** | Video > long window |
| GPT-2 Medium (345M) | BERT 18.38 | Baseline |
| GPT-2 Large (1.5B) | BERT 19.56 | LM scale improves understanding |

### Key Findings
- **Video is more effective than long temporal windows**: 60 frames + video (73.38mm) outperforms 120 frames without video (79.61mm), indicating that environmental context is more valuable than longer motion histories.
- **Lower-body improvement is the most significant**: The 3-point tracking lower-body error drops from 148.37mm to 124.58mm ($-16\%$). Since there are no sensors on the lower body, the footsteps and ground interactions visible in the video provide critical constraints.
- **Video understanding surpasses motion understanding**: V2T (BERT 16.62) outperforms M2T (15.90). Since many motion descriptions involve environmental information (e.g., "walking into a tunnel"), the video directly provides scenic semantics that the motion sequence cannot express.
- **Product Quantization is key**: Moving from PQ=1 to PQ=2 reduces MPJPE by more than 17mm, demonstrating that the precision of motion representations is critical for downstream tasks.

## Highlights & Insights
- **The "motion discretization + LLM unified modeling" framework** elegantly transforms continuous motor control issues into language modeling problems, unifying tracking and understanding. This paradigm can be generalized to robotic manipulation (sensors $\rightarrow$ actions $\rightarrow$ language descriptions).
- **Empirical value of video disambiguation**: It explicitly quantifies the assistance of egocentric video to sparse tracking (10-20mm), providing solid evidence for multi-sensor fusion in AR/VR devices.
- **By-product capabilities**: The framework naturally supports unconditional motion generation and text-to-motion generation, covering four tasks with a single model.

## Limitations & Future Work
- The reconstruction error of the VQ-VAE (34.5mm) sets an upper bound on tracking precision; better quantization methods (such as Residual Quantization RQ-VAE) may yield improvements.
- Frame-by-frame encoding of video via CLIP loses fine-grained temporal information (such as specific object names); a video encoder could be used instead.
- GPT-2 Medium has only 345M parameters; switching to a larger LLM could significantly boost understanding performance (a trend already shown by the 1.5B model).
- The language output suffers from hallucination issues, lacking factual correctness assurance mechanisms.

## Related Work & Insights
- **vs AvatarPoser / BoDiffusion**: Traditional tracking methods only utilize sensor data. EgoLM reduces errors by 10+mm by incorporating video; however, it underperforms compared to BoDiffusion when using only 3-point sensors, surpassing it only after video is integrated.
- **vs MotionGPT / TM2T**: These are specialized models for motion-to-language translation. EgoLM outperforms them on understanding tasks (BERT score 19.97 vs 14.09), thanks to multi-task joint training and video information.
- **vs EgoEgo**: EgoEgo directly predicts motion from egocentric video but yields inferior results (132.16mm). EgoLM treats video as an auxiliary signal rather than the sole input, which is more effective.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First to unify egocentric motion tracking and understanding, with an elegant VQ-VAE + LLM framework design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Complete comparisons and ablation studies in both tracking and understanding directions, with exhaustive VQ-VAE parameter searches.
- **Writing Quality**: ⭐⭐⭐⭐ Clear motivation for the framework; the argument for multi-task unification is convincing.
- **Value**: ⭐⭐⭐⭐ Directly applicable to egocentric interaction in AR/VR; the multimodal unified modeling paradigm is inspiring.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Rethinking Vision-Language Model in Face Forensics: Multi-Modal Interpretable Forged Face Detector](rethinking_vision-language_model_in_face_forensics_multi-modal_interpretable_for.md)
- [\[CVPR 2025\] MMRL: Multi-Modal Representation Learning for Vision-Language Models](mmrl_multi-modal_representation_learning_for_vision-language_models.md)
- [\[CVPR 2025\] GeoMM: On Geodesic Perspective for Multi-Modal Learning](geomm_on_geodesic_perspective_for_multi-modal_learning.md)
- [\[CVPR 2026\] Point Cloud as a Foreign Language for Multi-modal Large Language Model](../../CVPR2026/multimodal_vlm/point_cloud_as_a_foreign_language_for_multi-modal_large_language_model.md)
- [\[ICCV 2025\] Large Multi-modal Models Can Interpret Features in Large Multi-modal Models](../../ICCV2025/multimodal_vlm/large_multi-modal_models_can_interpret_features_in_large_multi-modal_models.md)

</div>

<!-- RELATED:END -->
