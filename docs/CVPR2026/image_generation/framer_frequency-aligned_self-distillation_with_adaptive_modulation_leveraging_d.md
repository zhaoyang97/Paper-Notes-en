---
title: >-
  [Paper Note] FRAMER: Frequency-Aligned Self-Distillation with Adaptive Modulation Leveraging Diffusion Priors for Real-World Image Super-Resolution
description: >-
  [CVPR 2026][Image Generation][Real-World Image Super-Resolution] FRAMER proposes a frequency-aligned self-distillation training framework that uses final-layer feature maps as teacher supervision for intermediate layers.…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Real-World Image Super-Resolution"
  - "Self-Distillation"
  - "Frequency Awareness"
  - "Diffusion Priors"
  - "Plug-and-Play"
date: 2026-05-08
content_hash: 716101ef114d053c
---

# FRAMER: Frequency-Aligned Self-Distillation with Adaptive Modulation Leveraging Diffusion Priors for Real-World Image Super-Resolution

**Conference**: CVPR 2026
**arXiv**: [2512.01390](https://arxiv.org/abs/2512.01390)
**Code**: [https://cmlab-korea.github.io/FRAMER/](https://cmlab-korea.github.io/FRAMER/)
**Area**: Diffusion Models / Image Generation
**Keywords**: Real-World Image Super-Resolution, Self-Distillation, Frequency Awareness, Diffusion Priors, Plug-and-Play

## TL;DR
FRAMER proposes a frequency-aligned self-distillation training framework that uses final-layer feature maps as teacher supervision for intermediate layers. By applying IntraCL and InterCL contrastive losses to low-frequency (LF) and high-frequency (HF) components respectively, along with Frequency-based Adaptive Weight (FAW) and Frequency-based Adaptive Modulation (FAM), FRAMER significantly improves high-frequency detail recovery in diffusion-based real-world image super-resolution without modifying the network architecture or inference pipeline.

## Background & Motivation

1. **Background**: Real-world image super-resolution (Real-ISR) aims to recover high-resolution images from low-resolution inputs degraded by complex unknown distortions. Diffusion models have surpassed GANs as the dominant approach, and leveraging rich priors from pretrained text-to-image models (e.g., SD2's U-Net, SD3's DiT) is a promising direction.
2. **Limitations of Prior Work**: Diffusion models struggle to reconstruct fine high-frequency (HF) details, often producing overly smooth results. Standard noise prediction losses apply uniform supervision across all layers and frequencies, neglecting the internal frequency hierarchy of the model.
3. **Key Challenge**: The authors trace the problem to a fundamental low-frequency (LF) bias arising from two sources: (a) the frequency distribution of natural images is LF-dominant, which is exacerbated in LR inputs, causing the noise prediction loss to favor LF components to minimize overall loss; (b) a "LF-first, HF-later" hierarchical structure exists along network depth — LF features stabilize in early layers while HF features only converge near the final layers.
4. **Goal**: How to apply targeted supervision to LF and HF components during training, correcting the LF bias, without altering the inference architecture?
5. **Key Insight**: Self-distillation — treating final-layer feature maps as the teacher and intermediate layers as students. Compared to external frequency-domain losses, this avoids domain mismatch since teacher and student operate in the same feature space.
6. **Core Idea**: Decompose the self-distillation signal by frequency band, applying intra-sample contrastive learning on LF to stabilize structure and inter-sample contrastive learning on HF to sharpen details, with adaptive mechanisms that match the internal frequency hierarchy of the model.

## Method

### Overall Architecture
FRAMER is a pure training strategy that adds auxiliary self-distillation terms to the standard noise prediction loss. At inference, the original backbone is used with no modifications. At each denoising step, the final-layer feature map serves as the teacher and all intermediate layers serve as students. Teacher and student feature maps are decomposed into LF and HF bands via FFT masking. The LF band is stabilized via IntraCL (intra-sample contrastive loss); the HF band is sharpened via InterCL (inter-sample contrastive loss). FAW and FAM adaptively modulate distillation strength per layer and per frequency band.

### Key Designs

1. **Intra Contrastive Loss (IntraCL) — Low-Frequency Stabilization**:

    - Function: Stabilizes globally shared structural representations in the LF band.
    - Mechanism: For each intermediate layer $i$, the cosine similarity between its LF representation $\mathbf{F}_{LF}^{(i)}$ and the teacher's LF representation $\mathbf{F}_{LF}^{(n)}$ forms the positive pair, while the LF representation of a randomly sampled layer $j$ forms the negative pair. The loss takes a log-softmax form: $\mathcal{L}_{IntraCL}^{(i)} = -\log \frac{\exp(s_{+,LF}^{(i)})}{\exp(s_{+,LF}^{(i)}) + \exp(s_{-,LF}^{(i)})}$. Cross-sample negatives are not used, as LF features exhibit high inter-sample similarity and in-batch negatives would become false negatives.
    - Design Motivation: LF features are highly similar across training samples (shared structural information), making in-batch negatives prone to false negatives. Intra-sample contrast is sufficient to drive student convergence toward the teacher via inter-layer discrepancies.

2. **Inter Contrastive Loss (InterCL) — High-Frequency Sharpening**:

    - Function: Sharpens instance-specific detail representations in the HF band.
    - Mechanism: Pulls the student's HF representation closer to the teacher's while pushing away two types of negatives: (i) HF representations from randomly sampled layers of the same image (enforcing inter-layer progression); (ii) HF representations from other images in the batch (encouraging instance discrimination). $\mathcal{L}_{InterCL}^{(i)} = -\log \frac{\exp(s_{+,HF}^{(i)})}{\exp(s_{+,HF}^{(i)}) + \exp(s_{-,HF}^{(i)}) + S_{neg}^{(i)}}$.
    - Design Motivation: HF features have low inter-sample similarity (instance-specific details), making in-batch negatives informative true negatives. This directly counteracts the LF bias by providing targeted optimization signals for the slowly converging HF components.

3. **Frequency-based Adaptive Weight (FAW) — Adaptive Weighting**:

    - Function: Adaptively modulates distillation weight per layer and per frequency band based on discrepancy from the teacher.
    - Mechanism: Computes the FFT magnitude mean $E_{LF}^{(i)}$, $E_{HF}^{(i)}$ for each layer's LF/HF bands and the relative discrepancy $\Delta^{(i)}$ from the final layer. The weight follows the inverse-discrepancy formula $w^{(i)} = 1/(1+\Delta^{(i)})$. Frequency bands closer to the teacher receive higher weights; early-layer LF weights exceed HF weights.
    - Design Motivation: Matches the "LF-first, HF-later" hierarchical structure, avoiding redundant gradients for already-converged LF layers while providing sufficient signals for immature HF layers.

### Loss & Training
The final training objective is $\mathcal{L}_{total} = \mathcal{L}_{noise} + \sum_i \mathcal{L}_{FRAMER}^{(i)}$, where the FRAMER term is a weighted sum of FAW- and FAM-gated IntraCL and InterCL. FAM gates distillation strength via a student–teacher alignment score (with ReLU and stop-gradient) to prevent collapse in early layers. All auxiliary heads are removed at test time, incurring zero inference overhead.

## Key Experimental Results

### Main Results

| Dataset | Metric | FRAMER_U (Ours) | PiSA-SR (Baseline) | Gain | FRAMER_D (Ours) | DiT4SR (Baseline) | Gain |
|--------|------|-----------------|----------------|------|-----------------|---------------|------|
| DrealSR | PSNR↑ | 26.96 | 26.18 | +3.0% | 24.73 | 23.64 | +4.6% |
| DrealSR | SSIM↑ | 0.786 | 0.752 | +4.5% | 0.687 | 0.640 | +7.3% |
| DrealSR | LPIPS↓ | 0.333 | 0.368 | +9.5% | 0.412 | 0.442 | +6.8% |
| DrealSR | MANIQA↑ | 0.595 | 0.490 | +21.4% | 0.514 | 0.441 | +16.6% |
| RealSR | PSNR↑ | 24.81 | 24.02 | +3.3% | 23.23 | 21.94 | +5.9% |
| RealSR | MANIQA↑ | 0.484 | 0.412 | +17.5% | 0.564 | 0.459 | +22.9% |

### Ablation Study
The paper ablates the effectiveness of the final-layer teacher and random-layer negatives (detailed data in supplementary material). Core findings:

| Configuration | Effect |
|------|---------|
| Noise prediction loss only (baseline) | Severe LF bias, insufficient HF recovery |
| + IntraCL | Improved LF stability, more consistent structure |
| + InterCL | Significant sharpening of HF details |
| + FAW | Layer-aware weight allocation, balanced overall improvement |
| + FAM | Prevents early-layer collapse, more stable training |

### Key Findings
- FRAMER yields the most significant gains on perceptual metrics (MANIQA, MUSIQ), with a 21.4% MANIQA improvement on DrealSR, confirming substantially enhanced HF recovery.
- Effective across both U-Net and DiT architectures, validating architecture-agnostic applicability.
- Advantages are more pronounced on the more challenging RealLR200 and RealLQ250 datasets.
- Training overhead is minimal (only auxiliary loss computation); inference overhead is zero.

## Highlights & Insights
- **Depth of the Frequency Hierarchy Finding**: The paper not only identifies the LF bias but also reveals the "LF-first, HF-later" phenomenon through layer-wise cosine similarity analysis, providing strong empirical motivation for frequency-decomposed self-distillation.
- **Differentiated Design for LF/HF Contrastive Learning**: LF uses intra-sample contrast (to avoid false negatives) while HF uses inter-sample contrast (to exploit true negatives) — an elegant design grounded in the analysis of inter-sample feature similarity.
- **Practical Plug-and-Play Value**: Requiring no architectural changes and no inference overhead, FRAMER can be directly applied to SR training with any diffusion backbone, offering broad practical utility.

## Limitations & Future Work
- Frequency decomposition relies on fixed FFT binary masks to partition LF/HF bands, which may not be optimal; learnable frequency splitting is worth exploring.
- In U-Net, additional 1×1 convolutions and resize operations are required to align feature dimensions, increasing integration complexity.
- The paper does not investigate generalization to other low-level vision tasks such as denoising or deblurring.
- Hyperparameters introduced by FAW/FAM (e.g., epsilon) may require per-backbone tuning.
- The LF/HF dichotomy may be overly coarse; three-band or continuous frequency decomposition could yield further improvements.
- Detailed ablation data are relegated to supplementary material, leaving the main paper without explicit per-component incremental results.

## Related Work & Insights
- **vs. SeeSR**: Methods such as SeeSR apply uniform noise prediction loss across all layers and frequencies without exploiting the internal frequency hierarchy; FRAMER directly addresses this gap through frequency-decomposed self-distillation.
- **vs. Frequency-Aware Diffusion Methods**: Existing frequency-aware approaches (e.g., FreeU) rely on fixed inference-time modulation; FRAMER adaptively adjusts supervision during training based on each layer's actual state.
- **vs. Self-Distillation Methods**: Conventional self-distillation aligns entire feature maps, implicitly inheriting the LF bias; FRAMER explicitly counteracts it through frequency separation.

## Rating
- Novelty: ⭐⭐⭐⭐ The frequency-aligned self-distillation framework is novel in design, though individual components (contrastive learning, adaptive weighting) are relatively standard.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four datasets, six metrics, cross-architecture evaluation on U-Net and DiT, and comprehensive ablation analysis.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly derived; the logical chain from observation to method is complete and figures are intuitive.
- Value: ⭐⭐⭐⭐ The plug-and-play training strategy is broadly applicable and can directly enhance existing SR methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] OARS: Process-Aware Online Alignment for Generative Real-World Image Super-Resolution](oars_process-aware_online_alignment_for_generative_real-world_image_super-resolu.md)
- [\[AAAI 2026\] Realism Control One-step Diffusion for Real-World Image Super-Resolution](../../AAAI2026/image_generation/realism_control_one-step_diffusion_for_real-world_image_super-resolution.md)
- [\[CVPR 2026\] DUO-VSR: Dual-Stream Distillation for One-Step Video Super-Resolution](duo-vsr_dual-stream_distillation_for_one-step_video_super-resolution.md)
- [\[AAAI 2026\] Continuous Degradation Modeling via Latent Flow Matching for Real-World Super-Resolution](../../AAAI2026/image_generation/continuous_degradation_modeling_via_latent_flow_matching_for_real-world_super-re.md)
- [\[AAAI 2026\] Mixture of Ranks with Degradation-Aware Routing for One-Step Real-World Image Super-Resolution](../../AAAI2026/image_generation/mixture_of_ranks_with_degradation-aware_routing_for_one-step_real-world_image_su.md)

</div>

<!-- RELATED:END -->
