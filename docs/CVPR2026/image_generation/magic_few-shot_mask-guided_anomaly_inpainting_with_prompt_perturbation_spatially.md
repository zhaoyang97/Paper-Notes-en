---
title: >-
  [Paper Note] MAGIC: Few-Shot Mask-Guided Anomaly Inpainting with Prompt Perturbation, Spatially Adaptive Guidance, and Context Awareness
description: >-
  [CVPR2026 Findings][Image Generation][Few-shot anomaly generation] This paper proposes the MAGIC framework, which fine-tunes an inpainting diffusion model and incorporates three complementary modules—Gaussian prompt perturbation, mask-guided spatial noise injection, and context-aware mask alignment—to generate high-fidelity, diverse, and spatially plausible industrial anomaly images under few-shot conditions, achieving state-of-the-art performance on downstream tasks using MVTec-AD.
tags:
  - CVPR2026 Findings
  - Image Generation
  - Few-shot anomaly generation
  - diffusion models
  - inpainting
  - industrial inspection
  - prompt perturbation
  - spatially adaptive guidance
  - mask alignment
date: 2026-05-08
content_hash: a4eea24074bc0031
---

# MAGIC: Few-Shot Mask-Guided Anomaly Inpainting with Prompt Perturbation, Spatially Adaptive Guidance, and Context Awareness

**Conference**: CVPR2026 Findings
**arXiv**: [2507.02314](https://arxiv.org/abs/2507.02314)
**Code**: [GitHub](https://github.com/Jaeihk/MAGIC-Anomaly-generation)
**Area**: Image Generation / Anomaly Detection
**Keywords**: Few-shot anomaly generation, diffusion models, inpainting, industrial inspection, prompt perturbation, spatially adaptive guidance, mask alignment
**Authors**: JaeHyuck Choi, MinJun Kim, Je Hyeong Hong (Hanyang University)

## TL;DR

This paper proposes the MAGIC framework, which fine-tunes an inpainting diffusion model and incorporates three complementary modules—Gaussian prompt perturbation, mask-guided spatial noise injection, and context-aware mask alignment—to generate high-fidelity, diverse, and spatially plausible industrial anomaly images under few-shot conditions, achieving state-of-the-art performance on downstream tasks using MVTec-AD.

## Background & Motivation

In industrial inspection scenarios, normal images are abundantly available, whereas anomalous images are extremely scarce. Although anomaly detection can be trained solely on normal samples (e.g., one-class classification and reconstruction-based methods), anomaly classification—which is critical for root cause analysis—still requires labeled anomalous samples. Consequently, synthesizing realistic anomaly images via generative models becomes an essential need.

Existing diffusion-model-based approaches suffer from two categories of problems:

- **Global anomaly generation (GAG)** methods (e.g., DualAnoDiff) jointly generate anomaly images and masks but frequently corrupt normal background textures because they do not accept normal images as guidance.
- **Mask-guided anomaly generation (MAG)** methods (e.g., AnomalyDiffusion, AnoGen) preserve the background but suffer from misalignment between the generated anomaly region and the input mask, mask displacement outside object boundaries, and limited generation quality due to frozen backbone networks.

The root cause lies in the fact that directly fine-tuning an inpainting model with few shots can guarantee background fidelity and mask alignment, but leads to severe overfitting—resulting in a lack of diversity and poor generation quality when masks are placed at semantically implausible locations.

## Method

### Overall Architecture

MAGIC builds upon the Stable Diffusion 2 inpainting model and is fine-tuned using DreamBooth. A fixed rare token (e.g., "sks") is adopted as the anomaly prompt, eliminating the need for object-specific textual descriptions. The overall pipeline is as follows:

- **Training phase**: The anomaly image $I_A$, ground-truth mask $M_{GT}$, and the masked normal background $I_A^M$ are concatenated as input, and the inpainting network is trained with a Gaussian-perturbed prompt embedding $c_p$.
- **Inference phase**: Given a normal image $I_N$ and an automatically generated mask $M$, CAMA first aligns the mask to a semantically plausible location to obtain $M_a$; denoising then proceeds with a randomly perturbed $c_p$ and MGNI-based local noise injection.

### Key Design 1: Gaussian Prompt Perturbation (GPP)

The core idea of GPP is to inject Gaussian noise into the prompt embedding space to enhance global texture diversity. After encoding the fixed prompt $\mathcal{P}$, perturbation is added as follows:

$$c_p = \tau(\mathcal{P}) + \delta, \quad \delta \sim \mathcal{N}(0, \sigma^2 I)$$

The key innovation is that GPP is applied **both** during training and inference. Applying noise only at inference time causes distribution shift, producing unrealistic textures. By incorporating perturbations drawn from the same distribution during training, the model learns a smooth mapping from the embedding space to the image space—effectively associating the anomaly concept with a ball in embedding space rather than a single point. Sampling from this ball at inference time naturally produces diverse yet realistic anomalies. $\sigma$ is set to 1.0.

### Key Design 2: Mask-Guided Noise Injection (MGNI)

During the DDIM denoising process, MGNI injects additional random noise exclusively within the mask region to enhance local texture diversity. The noise intensity is controlled by a scale factor $a$ (uniformly sampled from $[0, 0.6]$) and a time-decay function:

$$\lambda(t) = a \cdot \mathbb{1}_{t > t_{\min}}$$

Noise is injected in the early denoising steps (when $t \approx 1$) to enrich texture variety, while standard DDIM updates are restored in later steps (as $t \to 0$) to ensure fidelity. The DDIM update formula is augmented with a localized noise term $\sqrt{1-\alpha_{t-1}} \cdot \lambda(t) \cdot M \cdot \eta_t$, which acts only on masked pixels and thus leaves the background unaffected.

### Key Design 3: Context-Aware Mask Alignment (CAMA)

CAMA addresses the problem of semantically implausible mask placement. For object-type categories (e.g., screws, cables), anomalies should only appear within specific semantic sub-regions. CAMA employs the pretrained GeoAware-SC semantic correspondence model, extracting three keypoints from anomalous training samples—the mask centroid $p_c$, upper boundary point $p_u$, and lower boundary point $p_\ell$—and establishing semantic correspondences with the normal image:

1. A similarity map $S_u, S_c, S_\ell$ is generated for each keypoint.
2. The upper and lower boundary points are matched to obtain $q_u^*, q_\ell^*$, forming a candidate line $\mathcal{L}$.
3. The centroid position $q_c^*$ is optimized under the joint constraints of the candidate line, foreground mask $M_f$, and similarity map $S_c$.
4. The mask is translated to the new position and intersected with the foreground.

This approach achieves robust mask transfer using only three keypoints, balancing accuracy and computational efficiency.

## Key Experimental Results

### Generation Quality Evaluation (MVTec-AD, Table 1)

| Method | KID (×10³) ↓ | IC-LPIPS ↑ |
|--------|-------------|-----------|
| AnomalyDiffusion | 104.01 | 0.30 |
| AnoGen | 105.39 | 0.31 |
| DualAnoDiff | 96.82 | **0.36** |
| **MAGIC (Ours)** | **46.06** | 0.30 |

MAGIC achieves a substantially lower KID score (more than 52% improvement), indicating that its generated distribution is closest to real anomalies. The higher IC-LPIPS of DualAnoDiff is partially attributable to spurious diversity introduced by background corruption.

### Downstream Anomaly Classification Accuracy (ResNet-34, Table 2)

| Method | Mean Classification Accuracy (%) |
|--------|----------------------------------|
| Crop-Paste | 56.17 |
| AnomalyDiffusion | 64.90 |
| AnoGen | 56.92 |
| DualAnoDiff | 68.50 |
| **MAGIC (Ours)** | **76.39** |

MAGIC outperforms the second-best method, DualAnoDiff, by 7.89 percentage points. Particularly notable gains are observed on categories such as hazelnut (95.83%) and screw (83.95%).

### Downstream Anomaly Detection and Localization (U-Net, Table 3)

| Method | AUROC-P | AP-P | F1-P | AP-I |
|--------|---------|------|------|------|
| Crop-Paste | 94.4 | 69.1 | 70.7 | 98.9 |
| AnomalyDiffusion | 98.2 | 75.0 | 73.2 | 99.1 |
| DualAnoDiff | 97.4 | 76.8 | 72.9 | 98.6 |
| **MAGIC (Ours)** | **99.0** | **81.7** | **77.4** | **99.5** |

MAGIC achieves the best performance across all pixel-level and image-level metrics, with AP-P surpassing the second-best method by nearly 5 percentage points.

### Ablation Study (Table 4)

| GPP | MGNI | CAMA | KID↓ | Classification Acc. (%) |
|-----|------|------|------|------------------------|
| ✗ | ✗ | ✗ | 40.36 | 70.09 |
| ✓ | ✗ | ✗ | 33.87 | 74.07 |
| ✓ | ✓ | ✗ | 40.13 | 74.50 |
| ✓ | ✓ | ✓ | 38.76 | **76.39** |

GPP alone significantly reduces KID and improves classification accuracy by approximately 3%; MGNI increases diversity and slightly improves downstream performance despite a marginal KID increase; CAMA provides an additional ~2.85% gain on object-type categories.

## Key Findings

- Injecting Gaussian perturbations into the prompt embedding space is more effective at increasing global texture diversity than simply varying the random seed.
- **Applying GPP during training** is critical—using it only at inference time causes distribution shift and produces unrealistic textures.
- Spatially localized noise injection (MGNI) and prompt-level perturbation (GPP) enhance local and global diversity, respectively, and are complementary to each other.
- Efficient mask alignment can be achieved with only three keypoints for semantic correspondence, at a substantially lower computational cost than dense correspondence methods.

## Highlights & Insights

1. **Precise problem formulation**: The paper explicitly identifies three simultaneous requirements for an anomaly generator (background fidelity, mask alignment, and semantically plausible placement), whereas existing methods satisfy at most two.
2. **Transfer from personalized generation**: The approach draws on DreamBooth fine-tuning for fidelity and restores diversity through embedding-space perturbation—essentially finding a balance between overfitting and underfitting.
3. **No object-specific textual descriptions required**: Using only a rare token such as "sks" improves generalizability to industrial parts without semantic labels.
4. **Evaluation fairness**: All baseline methods are uniformly re-implemented and evaluated under a consistent protocol without manual curation, ensuring credible comparisons.
5. **Symmetry between training and inference** (GPP applied in both phases) reflects a deep understanding of distributional consistency.

## Limitations & Future Work

- CAMA relies on a rough correspondence between the input mask and the actual defect shape; semantic correspondence degrades when the deviation is too large.
- The framework depends on pretrained components (U2-Net for foreground extraction and GeoAware-SC for semantic correspondence), which may fail on repetitive structures or unseen domains.
- Validation is conducted solely on MVTec-AD; other commonly used anomaly datasets such as VisA have not been evaluated.
- Training takes approximately 1.5 hours (5,000 steps) per anomaly category, making the total training cost substantial when many categories are involved.
- CAMA increases inference time by approximately 5×, limiting real-time applicability.

## Related Work & Insights

- **AnomalyDiffusion**: Employs a frozen backbone with textual inversion for anomaly generation; MAGIC instead fine-tunes the inpainting model to achieve better fidelity.
- **DualAnoDiff**: A global approach with dual-stream attention sharing that achieves high diversity but severely corrupts backgrounds.
- **DreamBooth / Textual Inversion**: Two paradigms for personalized generation; MAGIC adopts DreamBooth's fidelity and compensates for diversity loss through perturbation.
- **DreamDistribution**: Similarly performs distribution sampling in embedding space to increase diversity, but targets general personalized generation rather than anomaly synthesis.
- **DefectFill**: A concurrent work that also fine-tunes an inpainting model, but requires object-specific prompts and does not address mask misalignment.

**Insight**: The technique of Gaussian perturbation in embedding space combined with symmetric training–inference design is broadly applicable and can be transferred to other few-shot conditional generation tasks (e.g., medical image augmentation, few-shot style transfer). The lightweight semantic correspondence strategy in CAMA is also worth adapting for generative tasks requiring spatial priors.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Each of the three modules is independently novel; the symmetric GPP design for training and inference is particularly elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Ablations are complete, comparisons are fair, and downstream task coverage is comprehensive, though evaluation on a single dataset is a mild limitation.
- **Writing Quality**: ⭐⭐⭐⭐ — Problem motivation is clearly articulated, method descriptions are complete, and figures and tables are of high quality.
- **Value**: ⭐⭐⭐⭐ — Practically valuable for data augmentation in industrial anomaly detection, with transferable technical insights.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Adaptive Auxiliary Prompt Blending for Target-Faithful Diffusion Generation](adaptive_auxiliary_prompt_blending_for_target-faithful_diffusion_generation.md)
- [\[NeurIPS 2025\] Token Perturbation Guidance for Diffusion Models](../../NeurIPS2025/image_generation/token_perturbation_guidance_for_diffusion_models.md)
- [\[CVPR 2026\] Few-shot Acoustic Synthesis with Multimodal Flow Matching](few-shot_acoustic_synthesis_with_multimodal_flow_matching.md)
- [\[CVPR 2026\] Uni-DAD: Unified Distillation and Adaptation of Diffusion Models for Few-step Few-shot Image Generation](uni-dad_unified_distillation_and_adaptation_of_diffusion_models_for_few-step_few.md)
- [\[AAAI 2026\] FreeInpaint: Tuning-free Prompt Alignment and Visual Rationality Enhancement in Image Inpainting](../../AAAI2026/image_generation/freeinpaint_tuning-free_prompt_alignment_and_visual_rationality_enhancement_in_i.md)

<!-- RELATED:END -->
