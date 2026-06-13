---
title: >-
  [Paper Note] Face Time Traveller: Travel Through Ages Without Losing Identity
description: >-
  [CVPR2026][Human Understanding][Face aging] This paper proposes FaceTT, a framework that achieves high-fidelity, identity-consistent face age transformation via three core modules—face-attribute-aware prompt refinement…
tags:
  - "CVPR2026"
  - "Human Understanding"
  - "Face aging"
  - "diffusion models"
  - "identity preservation"
  - "attention control"
  - "tuning-free inversion"
date: 2026-05-08
content_hash: 1d1555b4b0b5ab5d
---

# Face Time Traveller: Travel Through Ages Without Losing Identity

**Conference**: CVPR 2026 Findings  
**arXiv**: [2602.22819](https://arxiv.org/abs/2602.22819)  
**Code**: To be confirmed  
**Area**: Human Understanding
**Keywords**: Face aging, diffusion models, identity preservation, attention control, tuning-free inversion

## TL;DR

This paper proposes FaceTT, a framework that achieves high-fidelity, identity-consistent face age transformation via three core modules—face-attribute-aware prompt refinement, angular inversion, and adaptive attention control (AAC)—surpassing existing methods across multiple benchmarks.

## Background & Motivation

**Face aging is an ill-posed problem**: Influenced by both intrinsic factors (genetics) and extrinsic factors (environment, lifestyle), realistic age transformation must simultaneously alter age-related features (wrinkles, skin tone) while preserving age-invariant features (identity, expression), making the balance extremely challenging.

**Limitations of GAN-based methods**: Methods such as HRFAE and CUSP fall short in capturing high-resolution details and preserving identity, prone to artifacts or inaccurate reconstruction—especially suffering from severe identity drift under large age gaps.

**High cost of diffusion model inversion**: Existing diffusion models rely on iterative optimization-based inversion (e.g., Null-Text Inversion), which incurs substantial computational overhead and unstable reconstruction quality, making it difficult to achieve efficient editing while preserving facial details.

**Simple prompts are insufficient for aging**: A prompt such as "Photo of a X years old person" fails to capture the complex semantics of aging—interactions between intrinsic biological factors (skin texture changes) and extrinsic environmental factors (UV exposure, lifestyle habits) are neglected.

**Deficiencies of static attention control**: Methods such as P2P and PnP apply a uniform attention strategy across the entire image, unable to isolate and prioritize age-related regions, leading to background hallucination and loss of accessories.

**Imperfect evaluation protocols**: Conventional evaluation compares re-aged images against real target-age images, but paired ground truth is scarce, rendering identity consistency assessment unreliable.

## Method

### Overall Architecture

FaceTT is built upon a pretrained Stable Diffusion model with lightweight fine-tuning (150 steps) on the FFHQ-Aging dataset. Given a source-age face, three core modules produce the target-age face:

1. **Face-Attribute-Aware Prompt Refinement** → generates attribute-rich text prompts
2. **Angular Inversion** → maps the image to the diffusion latent space with high fidelity and without optimization
3. **Adaptive Attention Control (AAC)** → dynamically balances semantic transformation and structural preservation

### Key Designs

**Face-Attribute-Aware Prompt Refinement**: The visual-language model FastVLM is employed to extract age, gender, skin tone and texture (intrinsic factors), as well as external condition descriptions (extrinsic factors) from the input face. The refined prompt follows the format: `Photo of a <src_age> years old <gender> with <skin tone & texture>, due to <cause/condition description>`. This enables the model to associate high-level semantics such as "hair loss" and "weight gain" with their corresponding visual features.

**Angular Inversion**: The source and target branches are decoupled and optimized independently. The core idea is, at each denoising step:

- Compute the angular deviation between the inversion trajectory $z_t^*$ and the forward trajectory $z_t^{src/tgt}$
- Scale the update by exponential decay $\exp(-\xi \cdot \theta)$ according to the angular magnitude—a larger angle indicates poorer alignment, resulting in lower update weight
- Adaptively weight the correction terms for the source and target branches via cosine similarity—high similarity emphasizes editing fidelity, while low similarity emphasizes source image preservation
- The hyperparameter $\xi = 1.2$ controls the decay rate

**Adaptive Attention Control (AAC)**: Attention strategies are dynamically switched according to the denoising stage:

- **Early stage** ($t > \tau_1 = 35$): Cross-attention control injects semantic aging cues (wrinkles, skin tone, hair color)
- **Middle stage** ($\tau_2 \leq t \leq \tau_1$): The KL divergence $\eta$ measures the discrepancy between source and target cross-attention—if $\eta > \eta_{th} = 0.05$, cross-attention is prioritized to introduce significant semantic transformation; otherwise, self-attention is prioritized to preserve fine-grained structure
- **Late stage** ($t < \tau_2 = 15$): Self-attention replacement maintains facial geometry, expression, and identity consistency
- In the middle stage, an adaptive blending weight $w_t = 1 - H(M)$ based on attention map entropy enables smooth interpolation between source and target attention

### Loss & Training

- Stable Diffusion is fine-tuned for only 150 steps on FFHQ-Aging (70k images, 10 age groups)
- Adam optimizer, learning rate $5 \times 10^{-6}$, batch size 2
- No additional optimization is required at inference; processing a single image takes approximately 5 seconds on an A100 GPU—26× faster than FADING (~130 seconds)

## Key Experimental Results

### Main Results

**CelebA-HQ (young→60) quantitative comparison**:

| Method | Predicted Age | Blur ↓ | Gender ↑ | Smiling ↑ |
|--------|--------------|--------|----------|-----------|
| HRFAE | 55.05±9.18 | 3.42 | 94.80 | 74.60 |
| CUSP | 57.57±7.88 | 3.39 | 89.79 | 75.88 |
| FADING | 69.88±6.20 | 2.18 | 98.44 | 76.17 |
| **FaceTT** | **62.05±6.81** | **2.18** | **99.79** | **78.31** |

FaceTT achieves the closest predicted age to the ground-truth label (65.14) and the highest gender preservation rate of 99.79%.

**FFHQ-Aging full age-range comparison**:

| Metric | HRFAE | CUSP | FADING | **FaceTT** |
|--------|-------|------|--------|-----------|
| MAE (mean) | 21.84 | 16.40 | 13.47 | **11.40** |
| Gender Acc. | 0.45 | 0.51 | 0.57 | **0.62** |
| KID (mean) | 0.34 | 3.06 | 2.03 | **1.58** |

FaceTT reduces MAE by 15% and KID by 22% compared to FADING.

### Ablation Study

| Angular Inv. | AAC | Predicted Age | Gender ↑ | Smiling ↑ |
|:---:|:---:|--------------|----------|-----------|
| ✗ | ✗ | 69.88 | 98.44 | 76.17 |
| ✗ | ✓ | 61.70 | 99.22 | 73.78 |
| ✓ | ✗ | 61.25 | 99.02 | 68.58 |
| ✓ | ✓ | **62.05** | **99.79** | **78.31** |

Each module contributes independently, and their combination achieves the best balance across all metrics. Hyperparameter sensitivity analysis confirms that $\xi=1.2$, $\eta_{th}=0.05$, and $(\tau_1,\tau_2)=(35,15)$ constitute the optimal configuration.

### Key Findings

- **Identity preservation**: FaceTT achieves the best cyclic identity similarity ($ID_{sim}^{cyc}$) of 0.69 on FFHQ and 0.80 on the celebrity test set; reference identity similarity ($ID_{sim}^{ref}$) reaches 0.55, surpassing FADING's 0.50
- **Biometric verification**: On FNMR@FMR=0.1% over a 35-year age gap, FaceTT achieves only 0.01, far outperforming FADING (0.07) and CUSP (0.09)
- **Inference speed**: ~5 seconds per image vs. ~130 seconds for FADING, representing an approximately 26× speedup
- **Background and accessory preservation**: Qualitative comparisons demonstrate that FaceTT consistently preserves accessories such as glasses and earrings, as well as background details

## Highlights & Insights

- The angular inversion method cleverly exploits geometric angular deviation to control latent space updates, achieving high-quality inversion without iterative optimization and improving inference speed by an order of magnitude
- The adaptive attention control mechanism dynamically switches between cross-attention and self-attention based on the denoising stage and KL divergence, offering greater flexibility than static approaches
- The proposed cyclic identity similarity evaluation protocol eliminates reliance on paired ground-truth data, providing a more reliable measure for face aging assessment
- Face-attribute-aware prompt refinement effectively leverages the capabilities of VLMs to encode intrinsic and extrinsic aging factors as semantically rich text conditions

## Limitations & Future Work

- Validation is conducted only on static images; extension to temporally coherent age progression in video sequences remains unexplored
- The method relies on FastVLM for attribute prompt extraction, making the final output sensitive to the accuracy of VLM attribute descriptions
- KID scores in the extreme age range (0–2 years) remain relatively high (12.18), indicating room for improvement in infant face generation
- Fine-tuning is performed exclusively on FFHQ-Aging; generalization to non-Western faces has not been thoroughly validated

## Related Work & Insights

- **GAN-based methods**: HRFAE (latent space manipulation), CUSP (style-content disentanglement), MyTimeMachine (personalized temporal aging)—all constrained by identity distortion from GAN inversion
- **Diffusion inversion**: DDIM Inversion (deterministic but with reconstruction error), Null-Text Inversion (stable but slow), Direct Inversion (decoupled branches but limited for nonlinear editing)
- **Attention-based editing**: P2P, PnP, MasaCtrl (static strategies), FPE (stable transformation), Inversion-Free Editing (joint cross- and self-attention control)
- **Recent unpublished works**: Aging Multiverse (multi-trajectory aging at the cost of fine-grained control), TimeBooth (personalized but with poor background consistency)

## Rating

- Novelty: ⭐⭐⭐⭐ — Angular inversion and adaptive attention control represent valuable technical contributions; the cyclic evaluation protocol is also a meaningful addition
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive experimental design covering multiple datasets, metrics, ablations, user studies, and biometric verification
- Writing Quality: ⭐⭐⭐⭐ — Clear structure, detailed algorithmic pseudocode, and rich figures and tables
- Value: ⭐⭐⭐⭐ — Substantially faster inference with superior performance over prior SOTA, demonstrating strong practical utility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Seeing without Pixels: Perception from Camera Trajectories](seeing_without_pixels_perception_from_camera_trajectories.md)
- [\[AAAI 2026\] Robust Long-term Test-Time Adaptation for 3D Human Pose Estimation through Motion Discretization](../../AAAI2026/human_understanding/robust_long-term_test-time_adaptation_for_3d_human_pose_estimation_through_motio.md)
- [\[CVPR 2026\] All in One: Unifying Deepfake Detection, Tampering Localization, and Source Tracing with a Robust Landmark-Identity Watermark](all_in_one_unifying_deepfake_detection_tampering_localization_and_source_tracing.md)
- [\[CVPR 2026\] IDperturb: Enhancing Variation in Synthetic Face Generation via Angular Perturbations](idperturb_enhancing_variation_in_synthetic_face_generation_via_angular_perturbat.md)
- [\[CVPR 2026\] LaMoGen: Language to Motion Generation Through LLM-Guided Symbolic Inference](lamogen_language_to_motion_generation_through_llm-guided_symbolic_inference.md)

</div>

<!-- RELATED:END -->
