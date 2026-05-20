---
title: >-
  [Paper Note] FAST: Foreground-aware Diffusion with Accelerated Sampling Trajectory for Segmentation-oriented Anomaly Synthesis
description: >-
  [NeurIPS 2025][Segmentation][Industrial anomaly synthesis] FAST introduces explicit mechanisms to preserve anomaly regions throughout the diffusion trajectory: AIAS compresses the multi-step reverse process of discrete d…
tags:
  - "NeurIPS 2025"
  - "Segmentation"
  - "Industrial anomaly synthesis"
  - "foreground-aware diffusion"
  - "accelerated sampling"
  - "anomaly segmentation"
  - "mask-guided"
date: 2026-05-08
content_hash: 68b85f917e154908
---

# FAST: Foreground-aware Diffusion with Accelerated Sampling Trajectory for Segmentation-oriented Anomaly Synthesis

**Conference**: NeurIPS 2025
**arXiv**: [2509.20295](https://arxiv.org/abs/2509.20295)  
**Code**: [https://github.com/Chhro123/fast-foreground-aware-anomaly-synthesis](https://github.com/Chhro123/fast-foreground-aware-anomaly-synthesis)  
**Area**: Image Segmentation / Industrial Anomaly Detection
**Keywords**: Industrial anomaly synthesis, foreground-aware diffusion, accelerated sampling, anomaly segmentation, mask-guided

## TL;DR

FAST introduces explicit mechanisms to preserve anomaly regions throughout the diffusion trajectory: AIAS compresses the multi-step reverse process of discrete diffusion into a small number of coarse-to-fine analytical updates, while FARM reconstructs and reinjects anomaly foregrounds at each step, yielding a method that is both fast and better suited for generating training data for downstream anomaly segmentation models.

## Background & Motivation

Industrial anomaly segmentation differs fundamentally from standard anomaly detection: instead of simply classifying whether a defect exists, it requires precisely delineating defect boundaries at the pixel level.

The bottleneck in such tasks is not the classifier but the pixel-level annotation process.

Real production lines produce rare and morphologically diverse anomalies, many of which are non-reproducible, making it infeasible to cover a sufficiently rich anomaly space through manual collection and labeling alone.

Consequently, the industrial vision community has increasingly adopted a paradigm of synthesizing anomalies first and then training segmentation models on the resulting data.

However, existing methods suffer from three core limitations.

The first category consists of handcrafted or weakly-learned methods, such as patch replacement, texture erosion, and external texture blending.

These methods can quickly produce images that appear contaminated, but the synthesized anomalies typically lack the structural consistency of real industrial defects, which is particularly detrimental to boundary learning in segmentation models.

The second category comprises GAN-based methods.

While generally producing more visually realistic results than handcrafted approaches, these methods offer limited control over the location, shape, and extent of anomalies, and tend to generate outputs holistically, lacking the fine-grained local controllability that segmentation tasks genuinely require.

The third category involves diffusion-based anomaly synthesis.

Diffusion models offer stronger image fidelity and semantic consistency and integrate more naturally with text prompts, but most such methods treat foreground anomaly regions and background normal regions uniformly.

This "uniform noising and denoising" strategy may suffice for general image generation, but is inadequate for segmentation-oriented anomaly synthesis.

The anomaly region is precisely the local structure that most needs to be preserved; if it is continuously diluted by background statistics along a long denoising trajectory, the resulting anomaly will exhibit blurred boundaries, weak localization, and structural instability.

A practical concern is efficiency.

Standard DDPM often requires hundreds to thousands of reverse sampling steps, which is prohibitively slow for industrial changeovers, rapid data augmentation synthesis, and online iterative experimentation.

Existing post-training or training-free acceleration methods can shorten sampling but generally do not incorporate the importance of anomaly regions into the sampling trajectory itself.

The starting point of this paper is therefore well-defined.

Rather than simply pursuing greater visual realism or faster sampling in isolation, the goal is to align synthetic outputs more closely with the objective of improving downstream segmentation performance.

The authors accordingly make two key observations.

First, anomaly regions must be explicitly maintained throughout the diffusion trajectory, rather than being left to implicit noise modeling to incidentally capture.

Second, the multi-step updates of discrete diffusion can be analytically merged within short temporal windows, provided that such merging still preserves anomaly-relevant information.

FAST is built upon these two observations.

## Method

### Overall Architecture

FAST is built on a latent diffusion model.

The inputs consist of the latent representation of a normal image, an anomaly mask, and a text prompt.

During training, the model first diffuses the latent to a given timestep, then uses FARM to reconstruct a pseudo-clean representation containing only the anomaly region from the noisy latent, and subsequently reinjects noise into the anomaly region to form an anomaly-aware latent.

This anomaly-aware latent is then fed into the original diffusion denoiser for noise prediction.

During inference, the full reverse diffusion is no longer iterated step by step over 1000 steps; instead, AIAS partitions it into several coarse-to-fine segments.

Within each segment, a closed-form formula analytically aggregates multiple DDPM reverse transitions into a single update.

After each segment update, FARM reconstructs and reinforces the anomaly region to prevent anomaly information from being diluted during accelerated sampling.

The final output is not merely a visually appealing anomaly image, but an anomaly image–mask pair better suited as training data for segmentation models.

### Key Designs

1. **AIAS: Anomaly-Informed Accelerated Sampling**

    - Function: Compresses the long-chain reverse process of standard DDPM into a small number of analytical coarse-to-fine updates.
    - Mechanism: Starting from the linear Gaussian form of the DDPM single-step posterior, the authors prove that when $\hat{x}_0$ is approximately constant within a short temporal window, multiple reverse transitions can be aggregated into a single affine Gaussian kernel $x_{t_e}=\Pi_{t_e}^{t_s}x_{t_s}+\Sigma_{t_e}^{t_s}\hat{x}_0+\varepsilon_{t_e}$.
    - Design Motivation: Per-step iteration wastes computation on repetitive fine-grained numerical progression; analytically merging multiple timesteps enables significant acceleration without retraining the model.
    - Distinction from DDIM: DDIM relies on single-step deterministic updates and focuses on fast sampling; AIAS emphasizes closed-form multi-step aggregation within the discrete DDPM setting and continues to preserve anomaly-relevant mask information in subsequent reconstruction.
    - Distinction from PLMS: PLMS approximates the trajectory using a fixed multi-step solver, whereas AIAS performs analytical aggregation based on the original variance schedule, more faithfully respecting the discrete-time modeling assumptions of this work.

2. **FARM: Foreground-Aware Reconstruction Module**

    - Function: Explicitly reconstructs the anomaly foreground at each timestep and reinjects anomaly-aware noise into the masked region.
    - Mechanism: FARM is an encoder–decoder with time embeddings. The encoder extracts features from the noisy latent; a background-adaptive soft mask suppresses background responses; the decoder integrates binary masks at multiple resolutions to reconstruct an anomaly-exclusive pseudo-clean latent $\hat{x}_0^{an}$.
    - The reconstruction is then re-noised forward to the current timestep to obtain an anomaly-aware noisy representation $\hat{x}_{t_s}^{an}$, which is written back into the original latent via the mask.
    - Design Motivation: Standard diffusion processes anomalies and background within the same noise space, making anomalies susceptible to being averaged out by background statistics; FARM provides an additional pathway that continuously reminds the model of the importance of the anomaly region.

3. **Explicit Foreground–Background Decoupling**

    - Function: Maintains global consistency in the background while preserving local saliency in the anomaly.
    - Mechanism: The clean sample is decomposed into the anomaly foreground and the normal background. The background undergoes independent forward diffusion, the anomaly foreground is reconstructed by FARM, and the two are merged via the mask.
    - Design Motivation: The primary concern for segmentation tasks is not insufficient visual complexity in the anomaly, but inconsistency in noise levels at the anomaly boundary relative to the background. Explicit decoupling ensures that the local anomaly noise intensity is synchronized with the global background at the same timestep.

4. **Final 1–2 Step Refinement Strategy**

    - Function: Compensates for high-frequency detail loss that may result from accelerated sampling.
    - Mechanism: In the very-low-noise regime, the method reverts to standard DDPM posterior sampling to restore textural fidelity.
    - Design Motivation: Early segments primarily determine structure, while the final few steps primarily govern texture; the two serve different roles, so there is no need to use fine step sizes throughout the entire trajectory.

### Loss & Training

The training objective of FAST comprises two terms.

The first is the standard diffusion noise prediction loss, which encourages the denoiser output to approximate the true noise as closely as possible.

The second is the FARM reconstruction loss, which encourages FARM to recover the anomaly-only pseudo-clean content within the masked region as accurately as possible.

Together, these terms ensure that the main diffusion model maintains stability along the overall generative trajectory, while FARM anchors the anomaly structures that segmentation genuinely relies upon.

In practice, the authors synthesize training pairs on MVTec-AD and BTAD using normal images, anomaly masks, and text prompts.

For each anomaly category, 500 image–mask samples are generated; approximately one-third are used for training, with the remainder reserved for downstream evaluation.

Masks are sourced from geometric augmentations of real anomaly masks, as well as new masks synthesized by an LDM trained on real anomaly masks and subsequently filtered manually.

This indicates that FAST is not entirely free of human prior knowledge; rather, human effort is concentrated on mask quality control, which is the single most impactful investment.

## Key Experimental Results

### Main Results

The authors train real-time segmentation networks on data generated by different anomaly synthesis methods and compare pixel-level segmentation results on extended MVTec-AD.

The central finding is that FAST achieves consistently superior average mIoU and average Acc.

| Method | Avg. mIoU ↑ | Avg. Acc ↑ | Notes |
|--------|------------|-----------|-------|
| CutPaste | 55.87 | 63.81 | Handcrafted local replacement; limited boundary realism |
| DRAEM | 66.86 | 74.75 | Strong baseline; structural consistency still insufficient |
| GLASS | 56.23 | 61.44 | Framework oriented toward detection; advantage diminishes when transferred to segmentation |
| DFMGAN | 67.71 | 74.07 | GAN synthesis is more realistic but lacks controllability |
| RealNet | 62.89 | 71.70 | More focused on realistic anomaly modeling |
| AnomalyDiffusion | 62.88 | 72.06 | Text-driven diffusion; regions processed uniformly |
| **FAST** | **76.72** | **83.97** | Foreground-aware + accelerated sampling; overall best |

A per-category analysis reveals that FAST yields the largest gains on difficult categories.

For instance, mIoU on *capsule* improves from 51.39 (DRAEM) to 63.22; *grid* improves from 47.75 to 52.45; and *transistor* improves from 84.22 to 91.80.

These categories share the common characteristics of complex structure, fine boundaries, and irregular anomaly morphology, demonstrating that FAST's advantage lies not merely in faster generation but in sustaining local anomaly integrity.

The paper also reports the same trend on BTAD, indicating that FAST's performance is not an artifact of tuning to a single dataset.

### Ablation Study

The authors primarily validate two aspects: whether FARM is necessary, and whether the step compression of AIAS is genuinely beneficial.

| Configuration | Avg. mIoU ↑ | Avg. Acc ↑ | Conclusion |
|---------------|------------|-----------|------------|
| w/o FARM | 65.33 | 71.24 | Accelerated sampling only, without explicit foreground reconstruction |
| w/ FARM | 76.42 | 83.97 | Significant improvement in anomaly saliency and boundary localization |
| Gain | +11.09 | +12.73 | FARM is the decisive component |

The paper also provides finer per-category observations.

After incorporating FARM, mIoU on *capsule* improves by approximately 14.1, on *grid* by approximately 14.7, and on *transistor* by approximately 29.5.

These figures indicate that FARM's contribution is not a uniform marginal improvement across all categories, but is most pronounced on the hardest structurally complex anomalies.

| Sampling Strategy / Steps | Finding | Implication for Segmentation |
|---------------------------|---------|------------------------------|
| DDPM 1000 steps | High visual fidelity at high cost | Suitable for pure image quality; impractical for large-scale industrial synthesis |
| DDIM 50 steps | Faster, but unstable on *capsule*, *grid*, *transistor* | More prone to boundary–background mismatch |
| PLMS 50 steps | Stronger multi-step solver but still not task-specialized | Insufficient preservation of local anomaly structure |
| AIAS 10 steps | Already approaches full-step DDPM | Demonstrates the effectiveness of coarse-to-fine aggregation |
| AIAS 50 steps | Near-optimal performance | Best trade-off between speed and quality |

### Key Findings

- The most important experimental finding is not that "faster sampling still works," but that segmentation-oriented anomaly synthesis and pure visual fidelity are not fully aligned objectives.
- Excessive sampling steps continue to refine details but do not necessarily improve anomaly localization consistency, and may in fact average out the anomaly region over a long trajectory.
- FARM repeatedly emphasizes the anomaly region at every step, yielding the greatest benefit for segmentation models most sensitive to fine-grained boundaries.
- The value of AIAS lies not in pursuing acceleration as an end in itself, but in providing a practically viable synthesis throughput for industrial scenarios.

## Highlights & Insights

- **Downstream segmentation objectives are incorporated directly into the generative design.** Many anomaly synthesis works implicitly assume that greater visual realism translates to greater utility; this paper explicitly identifies structural alignment, boundary stability, and local anomaly saliency as what segmentation tasks truly require.
- **The analytical aggregation in AIAS is theoretically clean.** Rather than training a separate fast model, AIAS derives closed-form multi-step updates within the original discrete diffusion framework, resulting in low transfer cost and straightforward engineering reuse.
- **FARM transforms the mask from a static conditioning signal into a dynamic memory mechanism.** Most methods treat the mask as an input prompt; this work makes the mask a foreground maintenance mechanism that persists throughout the entire trajectory — a design worth borrowing for any locally-editing diffusion task.
- **"Local structure first" is more appropriate than "globally more realistic" for industrial tasks.** This insight is well-grounded in the application domain and explains why AIAS can improve segmentation performance even with very few steps.
- **Foreground–background decoupling is a broadly applicable methodology.** Beyond industrial anomalies, tasks such as medical lesion synthesis, defect inpainting, and controllable editing can all benefit from this design principle.

## Limitations & Future Work

Although FAST achieves dual optimization of speed and segmentation performance, several notable limitations remain.

First, mask generation is not fully automated.

Both real-mask geometric augmentation and the pipeline of LDM-synthesized masks followed by manual filtering indicate that the system still relies on external priors.

Second, evaluation is primarily conducted on standard industrial benchmarks such as MVTec-AD and BTAD.

Real production lines may involve far more complex material properties, lighting conditions, acquisition noise, and defect definitions; evidence for cross-domain generalization remains limited.

Third, the closed-form aggregation in AIAS relies on the approximation that $\hat{x}_0$ changes slowly within a short temporal window.

The paper acknowledges that residual artifacts can appear when the number of steps is extremely small (e.g., 1–2 steps), indicating that this approximation does not hold universally.

Fourth, the paper focuses exclusively on segmentation-oriented anomaly synthesis and does not directly optimize for metrics beyond detection and localization, such as calibration, open-set recognition, or cross-category generalization.

Several natural directions for future work exist.

One direction is to enable joint learning of mask generation and anomaly synthesis, rather than relying on separately prepared external inputs.

Another direction is to extend FARM into a more general region memory module applicable to tasks such as lesion synthesis, local inpainting, and text-controllable editing.

A further direction is to combine the analytical aggregation idea of AIAS with stronger consistency or rectified flow frameworks to further reduce the number of required steps.

## Related Work & Insights

- **vs. CutPaste / DRAEM**: These methods function more as "anomaly appearance perturbers" — capable of producing defects but not necessarily preserving structural boundaries; FAST is instead a "trajectory-aware local structure generator."
- **vs. AnomalyDiffusion**: Both employ text-driven diffusion synthesis, but AnomalyDiffusion is oriented toward general diffusion generation, whereas FAST additionally incorporates foreground awareness and downstream segmentation objectives into the sampling process.
- **vs. BDG-type foreground–background decoupling works**: FAST does not apply attention gating inside the denoiser but adds an explicit reconstruction branch outside it, making it more akin to a "plug-in structural enhancer."
- **Transferable insight**: In medical image lesion synthesis, lesion regions can analogously be treated as anomaly-only foregrounds, and a FARM-like mechanism can be employed to maintain lesion boundary and textural consistency.

## Rating

- **Novelty**: ⭐⭐⭐⭐☆ — The combination of AIAS's analytical acceleration and FARM's persistent foreground reconstruction is natural and well-motivated, supported by both theoretical derivation and task-specific innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ — Main results, per-category breakdowns, FARM ablations, and sampling step analyses are relatively comprehensive, though further validation in real industrial deployment scenarios would strengthen the work.
- **Writing Quality**: ⭐⭐⭐⭐☆ — The methodological logic is clearly presented, and the paper effectively explains why faster sampling can paradoxically be more suitable for segmentation; some formula-heavy passages carry a higher reading burden.
- **Value**: ⭐⭐⭐⭐⭐ — Highly practical for industrial anomaly segmentation; the paper genuinely operationalizes the principle that "synthetic data must serve downstream tasks" at the level of method design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation](../../CVPR2026/segmentation/task-oriented_data_synthesis_and_control-rectify_sampling_for_remote_sensing_sem.md)
- [\[NeurIPS 2025\] Fast and Fluent Diffusion Language Models via Convolutional Decoding and Rejective Fine-tuning](fast_and_fluent_diffusion_language_models_via_convolutional_decoding_and_rejecti.md)
- [\[NeurIPS 2025\] HAODiff: Human-Aware One-Step Diffusion via Dual-Prompt Guidance](haodiff_human-aware_one-step_diffusion_via_dual-prompt_guidance.md)
- [\[NeurIPS 2025\] HopaDIFF: Holistic-Partial Aware Fourier Conditioned Diffusion for Referring Human Action Segmentation in Multi-Person Scenarios](hopadiff_holistic-partial_aware_fourier_conditioned_diffusion_for_referring_huma.md)
- [\[ICCV 2025\] UniGlyph: Unified Segmentation-Conditioned Diffusion for Precise Visual Text Synthesis](../../ICCV2025/segmentation/uniglyph_unified_segmentation-conditioned_diffusion_for_precise_visual_text_synt.md)

</div>

<!-- RELATED:END -->
