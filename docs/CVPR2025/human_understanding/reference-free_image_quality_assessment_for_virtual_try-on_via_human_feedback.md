---
title: >-
  [Paper Note] Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback
description: >-
  [CVPR2025][Human Understanding][virtual try-on] VTON-IQA is proposed as a reference-free virtual try-on image quality assessment framework. It achieves image-level quality prediction aligned with human perception through a large-scale human-annotated benchmark VTON-QBench (62,688 try-on images + 431,800 annotations) and an Interleaved Cross-Attention module.
tags:
  - "CVPR2025"
  - "Human Understanding"
  - "virtual try-on"
  - "image quality assessment"
  - "human feedback"
  - "cross-attention"
  - "benchmark"
date: 2026-05-08
content_hash: a6a165beff0a7808
---

# Reference-Free Image Quality Assessment for Virtual Try-On via Human Feedback

**Conference**: CVPR2025  
**arXiv**: [2603.13057](https://arxiv.org/abs/2603.13057)  
**Code**: [GitHub](https://github.com/litelightlite/VTON-IQA)  
**Area**: Human Understanding  
**Keywords**: virtual try-on, image quality assessment, human feedback, cross-attention, benchmark

## TL;DR

VTON-IQA is proposed as a reference-free virtual try-on image quality assessment framework. It achieves image-level quality prediction aligned with human perception through a large-scale human-annotated benchmark VTON-QBench (62,688 try-on images + 431,800 annotations) and an Interleaved Cross-Attention module.

## Background & Motivation

**Dilemma of virtual try-on evaluation**: In real-world scenarios, ground-truth images of the same person wearing the target clothing are usually unavailable, rendering reference-based metrics (SSIM, LPIPS) inapplicable.

**Limitations of distribution-level metrics**: Metrics such as FID and KID measure dataset-level statistical properties and cannot reflect the perceptual quality of individual generated images.

**Limitations of Prior Work**: The VTONQA dataset has a limited scale (748 pairs vs. 13,153 pairs in ours), VTON-VLLM focuses on textual criticism rather than quantitative prediction, and VTBench does not directly learn a unified quality model from large-scale human annotations.

**Specificity of quality assessment**: Virtual try-on quality assessment differs from traditional single-image IQA because it requires simultaneously verifying clothing fidelity and the preservation of non-target regions (human identity, background, etc.).

**Need for cross-image interaction**: Quality evaluation requires modeling the cross-relationships between the generated try-on image, the input clothing image, and the person image, which traditional IQA methods cannot achieve.

**Lack of reproducible benchmarks**: Existing methods lack open-source implementations and standardized evaluation benchmarks, making reproducible evaluation difficult.

## Method

### Overall Architecture

A three-branch Transformer architecture processes the clothing image $I_G$, the person image $I_P$, and the generated try-on image $I_V$. The first half of the layers perform independent feature extraction, while the second half introduces the Interleaved Cross-Attention (ICA) module to model cross-image interactions.

### VTON-QBench Dataset Construction

- **Data Augmentation**: Synthetic clothing-person pairs are generated based on FLUX.1-dev, covering casual/street/formal/minimal/vintage styles, expanding the number of pairs from 6,981 to 13,153 (approximately 1.9$\times$).
- **Pseudo-triplet Construction**: A strong model, Nano Banana Pro, is used to generate pseudo ground-truths to construct $(I_G, I_P, I_R)$ triplets, supporting comparisons with reference-based metrics.
- **Generation by 14 VTON Models**: Covering GAN-based (VITON-HD, HR-VITON, SD-VITON), U-Net diffusion (IDM-VTON, CatVTON, OOTDiffusion), DiT diffusion (FitDit, CatVTON-FLUX), and proprietary editing models (Nano Banana Pro, GPT-Image-1.5).
- **Annotation Protocol**: A three-level ordinal scale (Unnatural / Slightly unnatural / Completely natural) is adopted, where 13,838 qualified annotators provided 431,800 annotations.
- **Data Cleaning**: A two-stage filtering process—dummy-task consistency check + Krippendorff's $\alpha$ threshold filtering (questionnaires with $\alpha \le 0.4$ are discarded), improving $\alpha$ from 0.286 to 0.550.

### Interleaved Cross-Attention (ICA) Module

- In the latter $L/2$ layers of standard Transformer blocks, a cross-attention layer is inserted between self-attention and MLP.
- **Asymmetric Interaction Design**: The try-on image representation aggregates contributions from both clothing and person $\hat{X}_V = \tilde{X}_V + C_{V \leftarrow G} + C_{V \leftarrow P}$, while the clothing and person branches only retrieve information from the try-on branch.
- This design avoids unnecessary $G \leftrightarrow P$ coupling and emphasizes quality judgment centered on the generated try-on image.

### Scoring Module

- $[CLS]$ tokens from each branch are extracted as compact global representations $c_G, c_P, c_V$.
- Intermediate Relation Score: A learnable weight $\alpha$ balances the cosine similarities of clothing consistency and non-target region preservation.
- $\tilde{s} = \alpha \frac{c_G^\top c_V}{\|c_G\|\|c_V\|} + (1-\alpha) \frac{c_P^\top c_V}{\|c_P\|\|c_V\|}$
- Final Score: $\hat{s} = \tanh(a\tilde{s}+b)$, where a learnable affine transformation and a $\tanh$ activation restrict the score to $[-1,1]$.

### Loss & Training

- **Pairwise Preference Term**: The Bradley-Terry model is used to model pairwise preferences, with soft-label cross-entropy aligning predictions with the distribution of human preferences.
- **Score Regression Term**: $L_2$ loss enforces consistency between the predicted scores and human ratings.
- Joint optimization balances both relative ranking and absolute score alignment.

## Key Experimental Results

| Method | $\rho_{\text{SRCC}}\uparrow$ | $\rho_{\text{PLCC}}\uparrow$ | $R^2\uparrow$ | $A_{\text{macro}}\uparrow$ | $A_{\text{micro}}\uparrow$ |
|------|---------|---------|------|----------|----------|
| SSIM | – | 0.135 | – | 0.596 | 0.593 |
| LPIPS | – | 0.387 | – | 0.701 | 0.695 |
| DINOv3 (zero-shot) | – | 0.261 | – | 0.637 | 0.641 |
| VTON-IQA w/o ICA | 0.617 | 0.615 | 0.372 | 0.722 | 0.747 |
| **VTON-IQA** | **0.750** | **0.751** | **0.553** | **0.781** | **0.790** |
| Human | 0.760 | 0.762 | 0.536 | 0.782 | 0.791 |

**Key Findings**:
- The ICA module brings significant improvement (SRCC: 0.617 to 0.750), validating the effectiveness of modeling cross-image interactions.
- In terms of pairwise accuracy ($A_{\text{macro}}/A_{\text{micro}}$), the model is close to human-level performance (0.781 vs. 0.782).
- There is still room for improvement in correlation metrics (0.750 vs. 0.760), indicating that fine-grained perceptual alignment needs further enhancement.
- In the benchmark of 14 VTON models, Nano Banana Pro achieves the highest overall scores on Dress Code and VITON-HD.
- GAN-based methods (VITON-HD, HR-VITON, SD-VITON) score significantly lower than diffusion-based methods under VTON-IQA.
- Qualitative analysis shows that VTON-IQA is robust to changes in pose and scale, whereas SSIM/LPIPS excessively penalize global transformations.

## Highlights

1. **Unprecedentedly Large Human-Annotated Benchmark**: VTON-QBench is currently the largest dataset for subjective human evaluation of virtual try-on, with the number of annotators (13,838) and the count of annotations (431,800) far exceeding prior works.
2. **Asymmetric ICA Design**: It precisely models $V \leftrightarrow G$ and $V \leftrightarrow P$ interactions while avoiding redundant $G \leftrightarrow P$ coupling, aligning perfectly with the essence of virtual try-on quality assessment.
3. **Human-Like Pairwise Accuracy**: Achieving consistency comparable to humans on pairwise ranking tasks.
4. **Comprehensive Benchmark**: Providing a systematic evaluation covering 14 representative VTON models, offering a standardized reference for the virtual try-on community.

## Limitations

1. There is still a gap in correlation metrics compared to human performance (SRCC 0.750 vs. 0.760), indicating that fine-grained perceptual alignment needs improvement.
2. The annotation scale has only three levels, which is relatively coarse-grained and might limit the discriminability of quality scores.
3. Both training and evaluation are based on VTON-QBench, and the generalization ability to completely new VTON models or extreme scenarios remains to be verified.
4. It only evaluates upper-body/full-body try-on, and does not cover sub-scenarios like accessories or footwear.
5. The backbone is based on DINOv3 ViT-L/16, which has high inference costs (requiring three-branch forward propagation for three images).

## Related Work

- **Virtual Try-On**: Evolving from GAN-based two-stage pipelines (VITON-HD, HR-VITON) $\rightarrow$ U-Net diffusion (IDM-VTON, CatVTON) $\rightarrow$ DiT (FitDit, Any2AnyTryon) $\rightarrow$ proprietary editing models (Nano Banana Pro, GPT-Image-1.5).
- **Quality Assessment**: VTONQA (limited-scale annotation training evaluator), VTON-VLLM (textual criticism-oriented), and VTBench (hierarchical benchmark without a unified quality model).
- **General IQA**: Q-Align, CLIP-IQA (single-image IQA without modeling cross-image interactions).

## Rating

- Novelty: ⭐⭐⭐⭐ — The asymmetric ICA interaction design is highly focused, and the large-scale crowdsourced annotation dataset construction process is mature.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Comprehensive benchmark of 14 models, ablation studies, and comparisons with humans.
- Writing Quality: ⭐⭐⭐⭐ — Clear structure and well-defined problem.
- Value: ⭐⭐⭐⭐ — Provides the virtual try-on community with much-needed standardized evaluation tools and large-scale benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] RefTon: Reference Person Shot Assist Virtual Try-on](../../CVPR2026/human_understanding/refton_reference_person_shot_assist_virtual_try-on.md)
- [\[CVPR 2025\] UNOPose: Unseen Object Pose Estimation with an Unposed RGB-D Reference Image](unopose_unseen_object_pose_estimation_with_an_unposed_rgb-d_reference_image.md)
- [\[CVPR 2025\] VTON 360: High-Fidelity Virtual Try-On from Any Viewing Direction](vton_360_high-fidelity_virtual_try-on_from_any_viewing_direction.md)
- [\[CVPR 2026\] Mobile-VTON: High-Fidelity On-Device Virtual Try-On](../../CVPR2026/human_understanding/mobile_vton_ondevice_virtual_tryon.md)
- [\[CVPR 2025\] X-Dyna: Expressive Dynamic Human Image Animation](x-dyna_expressive_dynamic_human_image_animation.md)

</div>

<!-- RELATED:END -->
