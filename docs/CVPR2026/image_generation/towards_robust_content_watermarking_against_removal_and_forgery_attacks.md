---
title: >-
  [Paper Note] Towards Robust Content Watermarking Against Removal and Forgery Attacks
description: >-
  [CVPR 2026][Image Generation][Content watermarking] This paper proposes ISTS, an instance-specific two-sided detection watermarking method that dynamically selects watermark injection timestep and location based on image…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Content watermarking"
  - "diffusion models"
  - "removal attacks"
  - "forgery attacks"
  - "instance-specific watermarking"
date: 2026-05-08
content_hash: c12a2ca6e00db614
---

# Towards Robust Content Watermarking Against Removal and Forgery Attacks

**Conference**: CVPR 2026 Findings  
**arXiv**: [2604.06662](https://arxiv.org/abs/2604.06662)  
**Code**: None  
**Area**: Image Generation / Digital Watermarking
**Keywords**: Content watermarking, diffusion models, removal attacks, forgery attacks, instance-specific watermarking

## TL;DR

This paper proposes ISTS, an instance-specific two-sided detection watermarking method that dynamically selects watermark injection timestep and location based on image semantics to resist both removal and forgery attacks. A two-sided detection mechanism is further designed to counter reverse latent representation attacks. ISTS achieves state-of-the-art robustness under both average and worst-case scenarios across three removal attacks and three forgery attacks.

## Background & Motivation

1. **Background**: Content watermarking (e.g., Tree-Ring) has been widely studied in text-to-image diffusion models, embedding identity markers into the latent space during generation to verify image provenance. These methods exhibit strong robustness against common image transformations (rotation, cropping, compression, etc.).
2. **Limitations of Prior Work**: Recent studies (Müller et al., Yang et al., Jain et al.) reveal that existing watermarks are highly vulnerable to removal and forgery attacks—post-removal detection AUC drops below 0.1 (e.g., Gaussian-Shading), while post-forgery AUC approaches 1.0 (watermarks can be trivially forged). This means watermarks can be both erased and fabricated, seriously undermining the reliability of copyright protection.
3. **Key Challenge**: Existing methods employ static, uniform watermark patterns (e.g., Tree-Ring injects fixed ring patterns at the center of Fourier space), whose consistency inadvertently leaks structural features, enabling attackers to extract or replicate watermarks using surrogate models.
4. **Goal**: To design a watermarking scheme that is robust against both removal and forgery attacks.
5. **Key Insight**: The key insight is that "static watermarks = information leakage." If the watermark pattern and injection parameters differ for each image, attackers cannot extract generalizable watermark features from a single or small number of reference images.
6. **Core Idea**: Instance-specific dynamic watermarking (semantics-guided selection of injection timestep and location) combined with two-sided detection (jointly examining both forward and inverse latent representations to block reverse optimization attack paths).

## Method

### Overall Architecture

**Generation stage**: Given a text prompt → generate an unwatermarked image → extract semantic features via CLIP encoder → map to watermark parameters $(t, l)$ (injection timestep and frequency-domain coordinates) via a pretrained semantic selector → execute the first $T-t$ DDIM steps → inject the watermark at frequency-domain coordinate $l$ at step $t$ → complete the remaining denoising to obtain the watermarked image.
**Detection stage**: Extract semantic features from the suspicious image → recover parameters $(t, l)$ → apply DDIM inversion to step $t$ → extract the watermark region at coordinate $l$ → apply two-sided detection for decision.

### Key Designs

1. **Instance-Specific Dynamic Watermarking (Dynamic Pattern + Dynamic Timestep)**

    - **Function**: Makes the watermark pattern and injection location unique per image, blocking the attacker's feature extraction pathway.
    - **Mechanism**: An unwatermarked image is first generated and encoded into a semantic vector via CLIP. A parameter selector $f = \phi \circ h \circ g$ is then constructed using K-Means clustering and a classifier, mapping semantic features to a specific injection timestep $t$ and frequency-domain coordinate $l$. Since watermark injection has negligible impact on image semantics, the semantic features of watermarked and unwatermarked images are highly consistent, enabling accurate parameter recovery from the suspicious image at detection time. Dynamic patterns disrupt the attack path of "extracting watermark patterns from reference images," while dynamic timesteps prevent gradient-based optimization attacks from tracing back to the correct injection step.
    - **Design Motivation**: Under static watermarking, an attacker can extract the watermark pattern from a single reference image and forge it (Müller et al.), or average multiple watermarked images to extract the common pattern (Yang et al.). With dynamic watermarking, watermark features across different images cancel each other out, rendering averaging-based attacks ineffective.

2. **Two-Sided Detection**

    - **Function**: Blocks attack paths that remove watermarks by optimizing the latent representation toward the opposite direction of the watermark pattern.
    - **Mechanism**: Traditional one-sided detection computes $d = \frac{1}{|M|} \sum_{i \in M} |W_i - \mathcal{F}(z_T)_i|$, which only checks the watermark-matching direction. Attackers can push the watermarked latent representation to the opposite direction to evade detection. Two-sided detection takes the minimum over both directions: $d = \min\{\frac{1}{|M|}\sum|W_i - \mathcal{F}(z_T)_i|, \frac{1}{|M|}\sum|W_i + \mathcal{F}(z_T)_i|\}$. For unwatermarked images (standard Gaussian distribution), sign flipping does not affect the distribution, leaving the detection metric unchanged; for watermarked images, the signal is captured regardless of direction.
    - **Design Motivation**: The removal attack of Müller et al. exploits precisely this vulnerability of one-sided detection by pushing the watermark to the opposite direction. Two-sided detection closes this attack surface at negligible cost (only one additional distance computation with a min operation).

3. **Semantic Parameter Selector Training**

    - **Function**: Establishes a deterministic mapping from image semantics to watermark parameters.
    - **Mechanism**: (1) Generate unwatermarked images from a prompt set → (2) Extract feature vectors via CLIP → (3) Cluster into $N$ categories via K-Means → (4) Map category labels to $(t, l)$ parameters via a predefined modular mapping $\phi$ → (5) Train classifier $h$ to perform feature-to-category mapping. The final selector is $f = \phi \circ h \circ g$ (where $g$ is the CLIP encoder).
    - **Design Motivation**: A deterministic mapping is required to ensure identical parameters are recovered during both generation and detection. K-Means clustering naturally assigns semantically similar images to the same parameter group, guaranteeing consistency.

### Loss & Training

- Selector training requires only one K-Means clustering step followed by simple classifier training.
- Watermark injection and detection require no additional training, directly leveraging the pretrained diffusion model.
- Stable-Diffusion-2-1-base is used; 100 image pairs are evaluated under adversarial attacks, and 1,000 pairs under non-adversarial scenarios.

## Key Experimental Results

### Main Results (Robustness Against Removal Attacks)

| Watermark Method | Original AUC | Imp-Removal | Avg-Removal | Mean AUC | Worst AUC |
|---------|----------|-------------|-------------|---------|---------|
| Tree-Ring | 1.000 | 0.267 | 0.527 | 0.589 | 0.267 |
| Gaussian-Shading | 1.000 | 0.000 | 0.371 | 0.457 | 0.000 |
| ROBIN | 1.000 | 0.082 | 0.742 | 0.595 | 0.082 |
| SEAL | 1.000 | 0.508 | 0.959 | 0.752 | 0.508 |
| **ISTS (Ours)** | 1.000 | **0.821** | **0.990** | **0.936** | **0.821** |

### Ablation Study

| Configuration | Imp-Removal AUC | Imp-Forgery AUC | Note |
|------|----------------|-----------------|------|
| Full ISTS | **0.821** | **0.634** | All three components combined |
| w/o dynamic pattern | ~0.71 | 0.72 | Fixed pattern is easier to forge |
| w/o dynamic timestep | Reduced | Reduced | Gradient attacks can trace back |
| w/o two-sided detection | ~0.71 | Comparable | Reverse latent attack is effective |

### Key Findings

- **Imp-Removal is the strongest removal attack**: Nearly all existing methods drop below AUC 0.7; ISTS maintains 0.821 (an improvement of 20%+).
- **ISTS achieves best performance under forgery attacks**: Mean AUC of 0.686 (lower is better), worst-case AUC of 0.949, outperforming all baselines.
- **Dynamic pattern contributes most to forgery resistance**: Removing it raises Imp-Forgery AUC from 0.62 to 0.72 (easier to forge).
- **Two-sided detection contributes most to removal resistance**: Removing it drops Imp-Removal AUC from 0.82 to ~0.71.
- **No image quality degradation**: PSNR, SSIM, and LPIPS are comparable to ROBIN (the highest-quality baseline); CLIP-Score remains consistent.
- **Robustness to conventional image transformations**: Mean AUC of 0.974 (vs. Tree-Ring's 0.975), worst-case AUC of 0.933 (vs. Tree-Ring's 0.928), on par with the best baseline.

## Highlights & Insights

- **The deep insight that "static = leakage"**: Although black-box attackers nominally have no knowledge of the watermarking algorithm, the consistent patterns of static watermarks in practice provide attackers with additional prior information. This observation reveals the general principle in security design that "implementation details can become side channels."
- **The elegant minimalism of two-sided detection**: Closing the reverse optimization attack path requires only one additional distance computation with a min operation (near-zero overhead). This "symmetrized detection metric" paradigm has broader applicability to other distance-based security detection schemes.
- **Semantic consistency guarantees detection reliability**: By exploiting the physical property that watermark injection has negligible impact on image semantics, the parameters recovered from a watermarked image are guaranteed to match those used at generation time. This is a clever design that decouples adversarial robustness from functional correctness.

## Limitations & Future Work

- Generating an unwatermarked version of each image to extract semantic features is required, effectively doubling the generation cost.
- The parameter selector relies on the assumption of CLIP semantic consistency, which may be violated after extreme image editing.
- Validation is conducted only on Stable-Diffusion-2-1-base; newer models such as SDXL and FLUX have not been tested.
- The selection of the K-Means cluster count $N$ and the parameter mapping $\phi$ lacks theoretical guidance.
- Evaluation uses 100 image pairs, which is a relatively small sample size that may limit statistical significance.
- Future work could explore adaptive dynamic strategies, such as adjusting subsequent watermark parameters in response to detected attack signals.

## Related Work & Insights

- **vs. Tree-Ring**: Tree-Ring injects fixed ring patterns at the center of the frequency domain; its static pattern causes AUC to drop to 0.267–0.527 under three removal attacks, whereas ISTS maintains 0.821–0.990 after dynamic adaptation.
- **vs. SEAL**: SEAL uses SimHash to control denoising randomness and shows moderate resistance to forgery (AUC 0.703), but is highly vulnerable to conventional transformations such as rotation, blurring, and cropping (worst-case AUC 0.523); ISTS achieves robustness against both adversarial attacks and conventional transformations.
- **vs. RingID**: RingID excels at conventional robustness (worst-case AUC 0.953) but fails almost entirely under forgery attacks (Imp-Forgery AUC = 1.0); ISTS provides guarantees on both fronts.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of instance-specific watermarking and two-sided detection effectively addresses the fundamental vulnerabilities of existing watermarks, with clear motivation and theoretical support.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers three removal attacks, three forgery attacks, and six image transformations with comprehensive average/worst-case analysis; however, the sample size is small and only a single model is evaluated.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation and method motivation are clearly articulated, algorithmic pseudocode is well-structured, and the threat model is rigorously defined.
- **Value**: ⭐⭐⭐⭐ The first work to systematically address the dual threat of removal and forgery in content watermarking, with practical significance for copyright protection in generative AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Rel-Zero: Harnessing Patch-Pair Invariance for Robust Zero-Watermarking Against AI Editing](rel-zero_harnessing_patch-pair_invariance_for_robust_zero-watermarking_against_a.md)
- [\[CVPR 2026\] SPDMark: Selective Parameter Displacement for Robust Video Watermarking](spdmark_selective_parameter_displacement_for_robust_video_watermarking.md)
- [\[CVPR 2026\] TRACE: Structure-Aware Character Encoding for Robust and Generalizable Document Watermarking](trace_structure-aware_character_encoding_for_robust_and_generalizable_document_w.md)
- [\[CVPR 2026\] Editing Away the Evidence: Diffusion-Based Image Manipulation and the Failure Modes of Robust Watermarking](editing_away_the_evidence_diffusion-based_image_manipulation_and_the_failure_mod.md)
- [\[AAAI 2026\] Creating Blank Canvas Against AI-Enabled Image Forgery](../../AAAI2026/image_generation/creating_blank_canvas_against_ai-enabled_image_forgery.md)

</div>

<!-- RELATED:END -->
