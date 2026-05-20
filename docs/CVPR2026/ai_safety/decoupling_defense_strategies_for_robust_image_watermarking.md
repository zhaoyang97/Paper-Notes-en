---
title: >-
  [Paper Note] AdvMark: Decoupling Defense Strategies for Robust Image Watermarking
description: >-
  [CVPR2026][AI Safety][Image Watermarking] AdvMark proposes a two-stage decoupled defense framework: Stage 1 Encoder Adversarial Training (EAT) pushes watermarked images into non-attackable regions to resist adversarial a…
tags:
  - "CVPR2026"
  - "AI Safety"
  - "Image Watermarking"
  - "Adversarial Robustness"
  - "Diffusion Regeneration Attack"
  - "Decoupled Training"
  - "Adversarial Training"
  - "Image Quality"
date: 2026-05-08
content_hash: 9571e0ed0fb38bbc
---

# AdvMark: Decoupling Defense Strategies for Robust Image Watermarking

**Conference**: CVPR2026
**arXiv**: [2602.20053](https://arxiv.org/abs/2602.20053)  
**Code**: N/A  
**Area**: AI Security
**Keywords**: Image Watermarking, Adversarial Robustness, Diffusion Regeneration Attack, Decoupled Training, Adversarial Training, Image Quality

## TL;DR
AdvMark proposes a two-stage decoupled defense framework: Stage 1 Encoder Adversarial Training (EAT) pushes watermarked images into non-attackable regions to resist adversarial attacks; Stage 2 performs direct image optimization to defend against distortion and regeneration attacks while preserving adversarial robustness. Evaluated across 9 watermarking methods × 10 attack types, AdvMark improves distortion/regeneration/adversarial accuracy by 29%/33%/46% respectively, while achieving the best image quality.

## Background & Motivation

**Background**: Deep learning image watermarking (DL watermarking) embeds information into images via an encoder and extracts it via a decoder, and has become a core technology for copyright protection and content tracing. Attack methods have escalated in recent years, forming a triple threat.

**Triple Threat**:
- **Adversarial Attack**: e.g., WEvade, which causes the decoder to extract incorrect information via imperceptible perturbations, with no visible change to the image.
- **Regeneration Attack**: Uses diffusion models to add noise to watermarked images and then denoise them, effectively "washing out" the watermark.
- **Distortion Attack**: Traditional image processing operations such as JPEG compression, Gaussian blur, and cropping.

**Two Major Problems with Joint Adversarial Training (JAT)**:
- **Problem 1**: Adversarial training of the decoder degrades clean accuracy — to correctly decode adversarial examples, the decoder is forced to expand its decision boundary, which in turn reduces accuracy on clean images.
- **Problem 2**: Simultaneously training against three attack types leads to slow convergence and poor performance — conflicting gradient directions across the three attacks result in a complex optimization landscape that JAT struggles to satisfy jointly.

**Core Insight**: Adversarial attacks are fundamentally different from distortion/regeneration attacks. Adversarial attacks exploit weaknesses in the model's decision boundary (model-specific), whereas distortion/regeneration attacks operate at the signal level (model-agnostic). Defense strategies should therefore be decoupled rather than jointly trained.

**Core Idea**: Two-stage decoupling — Stage 1 uses EAT to push encoded images into non-attackable regions; Stage 2 applies direct image optimization to handle distortion and regeneration attacks.

## Core Problem
How to simultaneously defend against adversarial attacks, regeneration attacks, and distortion attacks while avoiding the gradient conflicts and clean accuracy degradation inherent in joint training?

## Method

### Overall Architecture
AdvMark adopts a two-stage decoupled design. Stage 1 EAT focuses on adversarial robustness by fine-tuning the encoder (rather than expanding the decoder boundary) to move watermarked images into safe regions. Stage 2 directly optimizes the encoded image to resist distortion and regeneration attacks, using constraints to preserve the adversarial robustness established in Stage 1.

### Key Designs

1. **Stage 1: Encoder Adversarial Training (EAT)**:

    - **Function**: Constructs defender-tailored adversarial examples and primarily fine-tunes the encoder to move watermarked images away from the region reachable by adversarial attacks.
    - **Mechanism**:
        - **Adversarial example construction** (Eq. 2): $\min_{\delta} |0.5 - l(\text{clamp}(D(x_w + \delta), 0, 1), m)|$, which searches for the perturbation $\delta$ that most easily drives the decoder output toward 0.5 (maximum uncertainty); these constitute the defender-tailored adversarial examples.
        - **Encoder-primary update strategy**: Adversarial examples are fed back to the encoder, training it to embed watermarked images into safe regions far from the decision boundary. The decoder is conditionally updated only once when bit accuracy falls below $\tau_1$.
    - **Design Motivation**: Conventional adversarial training (AT) updates both encoder and decoder; expanding the decoder's decision boundary to accommodate adversarial examples sacrifices clean accuracy. EAT takes the opposite approach — rather than enlarging the boundary, it trains the encoder to relocate images to regions beyond the boundary's reach.
    - **Key Distinction**: In EAT, the encoder is the primary training target; the decoder is largely frozen.

2. **Stage 2: Direct Image Optimization**:

    - **Function**: Further optimizes the watermarked image $x_{w1}$ output by Stage 1 to obtain $x_{w2}$, enabling it to simultaneously resist distortion and regeneration attacks.
    - **Mechanism**:
        - **Optimization objective**: Directly optimizes $x_{w2}$ in pixel space (without updating network parameters) so that the decoder can still correctly extract the watermark after distortion/regeneration attacks.
        - **Constrained Image Loss**: Constrains the deviation of $x_{w2}$ from $x_{w1}$, ensuring that the optimized image does not stray from the non-attackable region established in Stage 1, thereby preserving adversarial robustness. A theoretical guarantee is provided: under the constraint $\|x_{w2} - x_{w1}\| \leq \epsilon$, the adversarial robustness of Stage 1 is maintained with high probability.
        - **Quality-aware Early-stop**: Instead of fixed $\epsilon$-ball projection (which leads to uneven image quality), image quality metrics (PSNR/SSIM) are monitored and optimization is stopped early when quality degrades below a threshold.
    - **Design Motivation**: Distortion/regeneration attacks are model-agnostic signal-level corruptions that are difficult to address through encoder training alone; direct pixel optimization is more direct and efficient, and the constraint preserves the adversarial defense gains of Stage 1.

3. **Theoretical Guarantee for Two-Stage Decoupling**:

    - **Function**: Proves that Stage 2 optimization does not compromise the adversarial robustness established in Stage 1.
    - **Mechanism**: If $x_{w1}$ is safe within adversarial radius $r$, and $\|x_{w2} - x_{w1}\| \leq \epsilon$, then $x_{w2}$ remains safe within radius $r - \epsilon$.
    - **Design Motivation**: Decoupling the two stages requires guaranteeing that the latter stage does not undermine the former; the theoretical guarantee makes the framework reliable.

### Training and Inference Pipeline
- **Stage 1**: Iteratively trains the encoder on adversarial examples ($K$-step PGD attack + encoder update), with conditional decoder freezing.
- **Stage 2**: Fixes the encoder/decoder and directly optimizes the pixel values of $x_{w2}$ via gradient descent, with quality-aware early-stop.
- **Inference**: Standard encoder embedding → Stage 2 optimization → output final watermarked image.

## Key Experimental Results

### Main Results — 9 Watermarking Methods × 10 Attack Types

| Defense Strategy | Distortion Acc (%) | Regeneration Acc (%) | Adversarial Acc (%) | PSNR ↑ | SSIM ↑ |
|---|---|---|---|---|---|
| No Defense (Baseline) | ~60–70 | ~50–60 | ~20–30 | Highest | Highest |
| JAT (Joint Training) | ~65–75 | ~55–65 | ~40–50 | Lower | Lower |
| AT + Distortion | ~70–78 | ~58–68 | ~45–55 | Low | Low |
| **AdvMark (Ours)** | **+29%** | **+33%** | **+46%** | **Highest** | **Highest** |

### Ablation Study

| Configuration | Adversarial Acc | Distortion Acc | Regeneration Acc | Image Quality |
|---|---|---|---|---|
| Stage 1 only (EAT) | High | Medium | Medium | High |
| Stage 2 only (DIO) | Low | High | High | Medium |
| JAT (Joint Training) | Medium | Medium | Medium | Low |
| Standard AT + DIO (non-EAT) | Medium | — | — | Low |
| EAT + DIO w/o constraint | Low | High | High | Medium |
| **AdvMark (EAT + constrained DIO)** | **High** | **High** | **High** | **High** |

### Key Findings
- **EAT vs. Standard AT**: Standard AT expands the decoder boundary, causing clean BA to drop from ~99% to ~92%; EAT maintains clean BA at ~98–99% while achieving stronger adversarial robustness.
- **Importance of the constraint**: Removing the image constraint in Stage 2 significantly degrades adversarial accuracy, validating the theoretical analysis.
- **Quality-aware early-stop vs. $\epsilon$-ball projection**: Early-stop achieves on average 1–2 dB higher PSNR at equivalent accuracy.
- **Generalizability**: Consistent improvements are observed across 9 watermarking methods with different architectures, demonstrating that AdvMark is a plug-and-play general-purpose framework.
- **Largest gain in adversarial accuracy (+46%)**: Indicates that EAT's "move into safe region" strategy is more effective than "expand the boundary."

## Highlights & Insights
- **"Move into safe region vs. expand boundary"**: This is the most central insight of the paper. Conventional AT makes the decoder more tolerant; EAT trains the encoder to deliver images to a safe location. By analogy: rather than making a house earthquake-resistant (modifying the decoder), build the house where earthquakes do not occur (modifying the encoder).
- **Conceptual depth of the decoupling strategy**: Adversarial attacks are model-specific (exploiting decision boundary weaknesses), while distortion/regeneration attacks are model-agnostic (signal-level corruption). The two classes of attacks are fundamentally distinct, and their defenses should be decoupled accordingly — a design driven by deep problem understanding.
- **Complete chain from theory to practice**: The paper first theoretically proves that robustness is preserved under the constraint, then implements quality-aware early-stop to operationalize this guarantee in practice.
- **General-purpose framework**: Plug-and-play compatibility with 9 existing watermarking methods demonstrates broad applicability and practical value.

## Limitations & Future Work
- Stage 2 direct image optimization requires additional inference time (tens of optimization steps per image), which may limit applicability in real-time scenarios.
- The threshold for quality-aware early-stop requires tuning for different application settings and is not entirely hyperparameter-free.
- The theoretical guarantee rests on the assumption $\|x_{w2} - x_{w1}\| \leq \epsilon$, which may not hold exactly in practice.
- Validation is limited to image watermarking; applicability to other modalities such as video and audio watermarking remains to be explored.
- Adversarial attack evaluation is primarily based on WEvade; testing against a broader range of adaptive attacks would strengthen credibility.

## Related Work & Insights
- **vs. RivaGAN/StegaStamp and similar watermarking methods**: These methods do not consider adversarial robustness in their encoder-decoder training; AdvMark can be applied as a plug-and-play post-processing step to enhance their robustness.
- **vs. Joint Adversarial Training (JAT)**: JAT simultaneously trains against three attack types, leading to gradient conflicts and clean accuracy degradation; AdvMark's two-stage decoupling optimizes each stage independently, yielding superior performance and image quality.
- **vs. DiffPure and similar diffusion-based purification methods**: DiffPure uses diffusion models to purify adversarial examples, but such models are precisely the tool used in regeneration attacks against watermarks. AdvMark must defend against scenarios where diffusion models act as attackers.
- **Broader inspiration**: The decoupling strategy for multi-type attack defense is generalizable to other security scenarios, such as multi-modal adversarial defense and federated learning robustness.

## Rating
- Novelty: ⭐⭐⭐⭐ The "move into safe region" perspective of EAT is novel, and the two-stage decoupled design demonstrates conceptual depth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Large-scale comparisons across 9 methods × 10 attacks are highly comprehensive, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is thorough; the narrative contrasting "expand boundary vs. move into safe region" is clear and compelling.
- Value: ⭐⭐⭐⭐ The plug-and-play general-purpose framework offers direct practical guidance for watermarking defense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ClusterMark: Towards Robust Watermarking for Autoregressive Image Generators with Visual Token Clustering](clustermark_towards_robust_watermarking_for_autoregressive_image_generators_with.md)
- [\[CVPR 2026\] TIACam: Text-Anchored Invariant Feature Learning with Auto-Augmentation for Camera-Robust Zero-Watermarking](tiacam_text-anchored_invariant_feature_learning_with_auto-augmentation_for_camer.md)
- [\[CVPR 2026\] RecoverMark: Robust Watermarking for Localization and Recovery of Manipulated Faces](recovermark_robust_watermarking_for_localization_and_recovery_of_manipulated_fac.md)
- [\[AAAI 2026\] Robust Watermarking on Gradient Boosting Decision Trees](../../AAAI2026/ai_safety/robust_watermarking_on_gradient_boosting_decision_trees.md)
- [\[CVPR 2026\] Domain-Skewed Federated Learning with Feature Decoupling and Calibration](domain-skewed_federated_learning_with_feature_decoupling_and_calibration.md)

</div>

<!-- RELATED:END -->
