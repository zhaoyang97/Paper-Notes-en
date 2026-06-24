---
title: >-
  [Paper Note] V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration
description: >-
  [CVPR2025][Image Generation][video generative prior] Redefines image restoration as a **progressive video generation process**, leveraging pre-trained video generative priors (Wan2.2-TI2V-5B) to achieve competitive multi-task image restoration using only 1,000 multi-task training samples (less than 2% of existing methods).
tags:
  - "CVPR2025"
  - "Image Generation"
  - "video generative prior"
  - "few-shot image restoration"
  - "progressive restoration"
  - "drift correction"
  - "all-in-one restoration"
date: 2026-05-08
content_hash: e451031a303e2131
---

<!-- 由 src/gen_stubs.py 自动生成 -->
# V-Bridge: Bridging Video Generative Priors to Versatile Few-shot Image Restoration

**Conference**: CVPR2025  
**arXiv**: [2603.13089](https://arxiv.org/abs/2603.13089)  
**Code**: [V-Bridge](https://github.com/Shenghe-Zheng/V-Bridge) (open-source project)  
**Area**: Image Generation  
**Keywords**: video generative prior, few-shot image restoration, progressive restoration, drift correction, all-in-one restoration

## TL;DR
Redefines image restoration as a **progressive video generation process**, leveraging pre-trained video generative priors (Wan2.2-TI2V-5B) to achieve competitive multi-task image restoration using only 1,000 multi-task training samples (less than 2% of existing methods).

## Background & Motivation
**Video models contain rich visual priors**: Large-scale video generation models pre-trained on massive datasets internalize rich structural, semantic, and dynamic visual priors, which have yet to be utilized in low-level vision tasks.

**Data bottleneck in image restoration**: Traditional restoration methods require huge amounts of supervised data (over a million samples), and training must be conducted separately for each degradation type, which is highly expensive.

**Limitations of task-specific modeling**: Mainstream methods rely on architectures carefully designed for each specific degradation type, which is disconnected from progress in generative modeling.

**Insights from Chain-of-Frames inference**: The Chain-of-Frames (CoF) inference capability of video models has been demonstrated in high-level semantic tasks, but its potential in low-level vision tasks remains entirely unexplored.

**Resolution gap**: Video models are typically pre-trained at medium resolutions (e.g., 720p), whereas practical restoration tasks require high resolution (e.g., 4K), presenting a significant gap.

**Paradigm shift requirement**: A paradigm shift is needed, moving from "static regression" to "progressive generation" to redefine image restoration.

## Method

### Overall Architecture
V-Bridge reformulates image restoration as a time-evolving process: the degraded image serves as the initial state, the high-quality reconstruction is the endpoint, and intermediate frames form a progressive quality trajectory. A video generation model is used to simulate this progressive restoration from low-quality to high-quality.

### Key Designs

**1. Pseudo-Temporal Data Construction**
- Given a low-quality/high-quality image pair $(\mathbf{I}_{LQ}, \mathbf{I}_{HQ})$, a pseudo-temporal sequence of length $T+1$ is constructed.
- Intermediate frames are generated via linear interpolation: $\mathbf{I}_t = (1-\alpha_t)\mathbf{I}_{LQ} + \alpha_t \mathbf{I}_{HQ}$, where $\alpha_t = t/T$.
- This provides temporally consistent supervision, guiding the model to learn the complete low-quality to high-quality trajectory.

**2. Progressive Curriculum Training**
- Build a resolution curriculum $\{r_t\}_{t=1}^T$, gradually increasing from low to high resolution.
- Perform resolution-dependent down-and-up sampling on video samples in each stage: $v_i^{(t)} = \text{DownUp}(v_i, r_t)$.
- Global structural consistency is established first, followed by the progressive enhancement of high-frequency detail generation.
- This bridges the gap between pre-training resolution and high-resolution restoration.

**3. Drift Correction**
- Due to the pre-training resolution bias, the base model prediction $\hat{x}$ suffers a systematic drift from the high-fidelity manifold.
- Train an additional drift correction model $g_\phi: p_\theta^{LR}(x) \rightarrow p_{HR}(x)$.
- Train with short pseudo-temporal sequences between the base model output and the ground truth to achieve a smooth transition.
- Efficiently eliminates the resolution bias using only a few intermediate frames.

### Loss & Training
- Unified training objective: $\mathcal{L}(\theta) = \mathbb{E}_{(\mathbf{I}_0, \mathbf{I}_t)}[\ell(f_\theta(\mathbf{I}_0, t), \mathbf{I}_t)]$
- A reconstruction loss $\ell(\cdot, \cdot)$ is used to supervise the video model's simulation of the restoration process.
- The same objective is used across all stages, with the difficulty controlled solely by the resolution curriculum.

## Key Experimental Results

### Main Results: FoundIR Test Set (PSNR/SSIM)

| Method | Data Volume | Blur | Haze | Rain | LowLight | B+N | Overall Trend |
|------|--------|------|------|------|----------|-----|----------|
| FoundIR-G | 1M | 24.34/0.79 | 16.65/0.63 | 33.09/0.94 | 12.35/0.72 | 22.53/0.77 | Requires huge data |
| V-Bridge (w/o DC) | 1K | 21.02/0.69 | 21.25/0.78 | 26.40/0.80 | 19.18/0.83 | 23.80/0.77 | Effective with few data |
| V-Bridge | 1K | 24.92/0.78 | 21.70/0.69 | 25.42/0.78 | **26.94**/0.89 | **27.31**/0.85 | Drift correction brings massive improvement |

Key Findings:
- Competitiveness is achieved using only **1K samples** (FoundIR uses 1M) across multiple degradation types.
- Low-light task: V-Bridge (26.94dB) significantly outperforms FoundIR-G (12.35dB) with a +14.59dB improvement.
- Hybrid degradation (B+N, L+N, etc.): V-Bridge performs exceptionally well, demonstrating the robust generalization capability of video priors.

### Ablation Study
- **Role of Drift Correction**: Adding DC yields an average improvement of 1-6 dB, especially in low-light (+7.76dB) and hybrid degradation scenarios.
- **Progressive Curriculum Training**: Compared to direct high-resolution training, the progressive strategy improves both training efficiency and final performance.
- **Data Efficiency**: Powerful performance can be achieved with only 50 samples per class, proving the highly efficient transferability of video priors.
- **Impact of Sequence Length**: Longer pseudo-temporal sequences provide smoother transitions but increase computational overhead linearly.

### Key Findings
- Video generation models implicitly learn powerful, transferable restoration priors, which can be activated with minimal data.
- Demonstrates excellent generalization capabilities in out-of-distribution (OOD) degradation scenarios.
- Drift correction is crucial for bridging the resolution gap.

## Highlights & Insights
1. **Paradigm Innovation**: Redefines static image restoration as a special case of video generation, opening a new direction for using video priors in low-level vision.
2. **Extreme Data Efficiency**: Achieving competitive performance with 1K samples compared to methods trained on millions of samples, challenging the traditional notion that "low-level vision requires massive data."
3. **Mathematical Interpretation of Drift Correction**: Modeling the performance degradation caused by resolution as a distribution drift, solved by correcting with an additional short-trajectory generation model, which is elegant and efficient.
4. **High Versatility**: A single model can handle multiple degradation types, such as denoising, deblurring, dehazing, deraining, and low-light enhancement.

## Limitations & Future Work
1. Performance on certain single degradation types (e.g., Noise, JPEG) is still inferior to specialized methods (e.g., FoundIR-G achieves 38.61 on Noise vs. V-Bridge's 32.87).
2. Inference speed is constrained by the generative process of the video model, requiring multi-step denoising sampling, which can be much slower than feed-forward restoration methods.
3. Progressive generation requires multi-frame inference ($T+1$ frames), incurring larger computational overhead compared to multi-frame feed-forward methods.
4. Constructing intermediate frames via linear interpolation is relatively simple and may not capture non-linear quality evolution.
5. The upper limit of the drift correction model itself is bounded by the capabilities of the base video model.
6. Robustness against extreme degradation (e.g., severe blur + noise + low-light triple degradation) has not been fully verified.

## Related Work & Insights
- **vs. FoundIR**: Achieves competitive performance with 1/1000 of the data volume, proving that pre-trained priors > large-scale supervision.
- **vs. PromptIR/AirNet**: These methods require degradation-aware mechanisms (contrastive learning/prompting), whereas V-Bridge handles multiple degradations implicitly using video priors.
- **vs. DiffUIR/DiffBIR**: Although both are generative model-based methods, V-Bridge utilizes the spatio-temporal consistency priors embedded in the video model, which are richer than those of image diffusion models.
- **vs. Chain-of-Frames Inference**: Integrates low-level vision tasks into the CoF framework, validating the effectiveness of CoF in pixel-level reconstruction for the first time.
- **Insights**: Video generation models might be ideal candidates for general visual foundation models, not limited to generative tasks, as low-level vision can also benefit; future work can explore unifying more low-level tasks into the video generation paradigm.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — First to model image restoration as a video generation process, pioneering the utilization of video priors for low-level vision.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Complete evaluations across various degradations, OOD testing, and ablations, but lacks inference speed comparisons.
- Writing Quality: ⭐⭐⭐⭐ — Clear motivation, smooth methodological presentation, and standard mathematical formulations.
- Value: ⭐⭐⭐⭐⭐ — Unlocks the great potential of video generation models as general visual priors, with far-reaching impacts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Zero-Shot Image Restoration Using Few-Step Guidance of Consistency Models (and Beyond)](zero-shot_image_restoration_using_few-step_guidance_of_consistency_models_and_be.md)
- [\[CVPR 2025\] AS-Bridge: A Bidirectional Generative Framework Bridging Next-Generation Astronomical Surveys](as-bridge_a_bidirectional_generative_framework_bridging_next-generation_astronom.md)
- [\[CVPR 2025\] DualAnoDiff: Dual-Interrelated Diffusion Model for Few-Shot Anomaly Image Generation](dual-interrelated_diffusion_model_for_few-shot_anomaly_image_generation.md)
- [\[ECCV 2024\] Rejection Sampling IMLE: Designing Priors for Better Few-Shot Image Synthesis](../../ECCV2024/image_generation/rejection_sampling_imle_designing_priors_for_better_few-shot_image_synthesis.md)
- [\[ICLR 2026\] Bridging Degradation Discrimination and Generation for Universal Image Restoration](../../ICLR2026/image_generation/bridging_degradation_discrimination_and_generation_for_universal_image_restorati.md)

</div>

<!-- RELATED:END -->
