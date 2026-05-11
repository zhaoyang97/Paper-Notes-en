---
title: >-
  [Paper Note] Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training
description: >-
  [CVPR 2026][Segmentation][Curriculum Learning] This paper proposes Data Warmup, a curriculum learning strategy that requires no modifications to the model or loss function. It schedules training images from easy to hard…
tags:
  - "CVPR 2026"
  - "Segmentation"
  - "Curriculum Learning"
  - "Diffusion Models"
  - "Data Complexity"
  - "Foreground Saliency"
  - "Training Efficiency"
date: 2026-05-08
content_hash: 6ce033ac0849a4ff
---

# Data Warmup: Complexity-Aware Curricula for Efficient Diffusion Training

**Conference**: CVPR 2026
**arXiv**: [2604.07397](https://arxiv.org/abs/2604.07397)
**Code**: Available
**Area**: Segmentation / Diffusion Model Training Acceleration
**Keywords**: Curriculum Learning, Diffusion Models, Data Complexity, Foreground Saliency, Training Efficiency

## TL;DR
This paper proposes Data Warmup, a curriculum learning strategy that requires no modifications to the model or loss function. It schedules training images from easy to hard using a semantics-aware image complexity metric (foreground dominance × foreground typicality). On ImageNet 256×256, it yields improvements of up to +6.11 IS and −3.41 FID for the SiT family. Notably, the reversed curriculum (hard-to-easy) performs worse than the uniform baseline, demonstrating that ordering itself is the key mechanism.

## Background & Motivation

**Background**: Diffusion model training is expensive (hundreds of GPU-days). Much computation is wasted in early training, where randomly initialized networks are forced to process the full spectrum of images—from simple to complex—resulting in noisy, uninformative gradients and wasted compute.

**Limitations of Prior Work**: (1) Traditional curriculum learning relies on training-time signals (loss/gradients), incurring per-iteration overhead and coupling with optimizer dynamics. (2) Pixel-level statistics (frequency, compressibility) are poor proxies for complexity—what matters is semantic structure.

**Core Intuition**: No art teacher begins with Picasso's *Guernica*—students learn simple concepts before complex ones. But what counts as "simple" for a diffusion model?

**Core Idea**: (1) Foreground dominance $\Omega_{dom}$: a large foreground-to-image ratio signals simplicity; (2) Foreground typicality $\Omega_{prot}$: a canonical viewpoint signals simplicity. Both are computed offline and used in temperature-controlled softmax sampling with easy-to-hard annealing.

## Method

### Key Designs

1. **Semantic Image Complexity Metric (offline, one-time computation, ~10 min on a single H100)**:

    - **Foreground Separation**: DINOv2 spatial tokens → PCA first principal component projection → threshold $\theta=0.05$ → foreground token set $\mathbf{Z}_i^{fg}$
    - **Foreground Dominance $\Omega_{dom}$**: Background ratio $r_i^{bg}$ corrected via sigmoid: $\Omega_{dom} = \frac{1}{1+e^{-(\kappa r_i^{bg} + \alpha(v_{min}))}}$. The nonlinearity captures the intuition that "80%→60% foreground has little effect, but 40%→20% has a large effect."
    - **Foreground Typicality $\Omega_{prot}$**: Mean of foreground tokens → k-means clustering ($K=1000$) → distance to nearest centroid. Greater distance = more atypical = harder.
    - **Overall Complexity**: $\Omega_i = \Omega_{dom} \times \Omega_{prot}$ (multiplicative: both dimensions must be simple for an image to be considered simple)
    - Intra-cluster normalization eliminates distributional bias across visual concepts

2. **Temperature-Controlled Sampling Schedule**:

    - $P(i|t) = \frac{\exp(-\tilde{\Omega}_i/\tau(t))}{\sum_j \exp(-\tilde{\Omega}_j/\tau(t))}$
    - Low $\tau$ → concentrated on simple images; high $\tau$ → approaches uniform sampling
    - Effective dataset size $|\mathcal{D}_\tau|$ grows from $|\mathcal{D}_0|$ to $|\mathcal{D}_{max}|$ via a power-2 schedule
    - Switches to uniform sampling after $T_w$ iterations

### Key Properties
- **Zero per-iteration overhead**: Complexity is fully precomputed offline
- **Orthogonal to model and loss**: Can be applied on top of any diffusion training pipeline
- **Ordering direction is critical**: The reversed (hard-to-easy) curriculum performs worse than the uniform baseline

## Key Experimental Results

### Directional Validation (SiT-B/2, ImageNet 256×256)

| Strategy | IS↑ | FID↓ |
|------|:---:|:---:|
| Uniform Sampling (Baseline) | 41.40 | 36.16 |
| **Data Warmup (Easy→Hard)** | **45.70 (+4.30)** | **32.75 (−3.41)** |
| Inverse (Hard→Easy) | 36.60 (−4.80) | 41.05 (+4.89) |

The gap between easy→hard and hard→easy is ΔIS ≈ 9 and ΔFID ≈ 8, confirming that direction is the key source of non-uniform benefit.

### Combined with REPA

| Method | IS↑ | FID↓ |
|------|:---:|:---:|
| REPA | 55.36 | 27.54 |
| **REPA + Data Warmup** | **58.08 (+2.72)** | **25.84 (−1.70)** |

### Across Model Scales

| Backbone | IS Gain | FID Reduction |
|----------|:---:|:---:|
| SiT-S/2 | +3.85 | −2.10 |
| SiT-B/2 | +4.30 | −3.41 |
| SiT-L/2 | +5.52 | −2.88 |
| SiT-XL/2 | +6.11 | −2.53 |

### Key Findings
- **Directional asymmetry is the central finding**: Easy→hard improves training while hard→easy is harmful, ruling out the hypothesis that non-uniform sampling is inherently beneficial
- Complementary gains when combined with REPA (a model-level accelerator), indicating data-level and model-level acceleration are orthogonal
- Consistent effectiveness across all model scales (S→XL), confirming this is not a scale-specific trick
- Foreground dominance contributes more individually than foreground typicality—the foreground-to-image ratio is the most important dimension of simplicity

## Highlights & Insights
- **A data-centric perspective on diffusion acceleration**: Without modifying the model, loss, or architecture, simply reordering data presentation yields significant improvements—a refreshing contrast to the model-centric methods that dominate this area
- **Rigorous validation of easy→hard ordering**: The inverse experiment cleanly eliminates confounding factors, establishing that ordering itself—not any incidental effect—drives the improvement
- **Semantic vs. pixel-level complexity**: The paper demonstrates that pixel-level statistics (frequency, entropy) are poor proxies, while semantic structure (foreground dominance + typicality) correctly characterizes difficulty for diffusion models
- **Minimal cost**: ~10 minutes of single-GPU (H100) offline preprocessing with zero per-iteration overhead—an extremely low barrier to adoption

## Limitations & Future Work
- Relies on DINOv2 for foreground separation; alternative approaches may be needed for domains where DINOv2 is unsuitable (e.g., medical imaging)
- The number of k-means clusters ($K=1000$) and sigmoid parameters ($\kappa=12$, $v_{min}=0.002$) are empirically chosen hyperparameters
- Validated only on ImageNet; generalization to other datasets (e.g., LAION) and text-conditional diffusion models remains to be confirmed
- The curriculum only affects the first $T_w$ iterations; its effectiveness when continuing to train already well-trained models is an open question

## Related Work & Insights
- **vs. Traditional Curriculum Learning**: Conventional methods compute difficulty at training time (loss-based), incurring per-iteration overhead. Data Warmup is fully offline with zero per-iteration cost.
- **vs. REPA**: REPA operates at the model level (aligning pretrained features); Data Warmup operates at the data level—the two are orthogonal and complementary.
- **vs. Data Selection / Importance Sampling**: Data selection reduces dataset size; Data Warmup reorders without reduction, ensuring all data is seen during training.
- The sigmoid correction for foreground dominance is a noteworthy design choice—linear mapping fails to reflect the true difficulty distribution.
- The same data complexity mismatch problem in diffusion training likely exists in other modalities (video, 3D, audio), suggesting broad applicability.

## Technical Details

- **Effective Dataset Size**: $|\mathcal{D}_{\tau(t)}| = \sum_{i=1}^{|\mathcal{D}|}[1-(1-P(i|t))^{|\mathcal{D}|}]$; temperature $\tau$ is recovered via binary search
- **Power-2 Schedule**: Rapidly transitions through simple samples early, spending more iterations on harder samples later
- **Intra-cluster Normalization**: $\tilde{\Omega}_i = \frac{\Omega_i - \Omega_{\min}^{k(i)}}{\Omega_{\max}^{k(i)} - \Omega_{\min}^{k(i)}}$, eliminating distributional differences across visual concepts

## Rating
- Novelty: ⭐⭐⭐⭐ Simple yet overlooked idea, with rigorous directional validation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-scale models, directional validation, combination with REPA, hyperparameter ablations
- Writing Quality: ⭐⭐⭐⭐⭐ Clear intuition ("no teacher starts with Guernica")
- Value: ⭐⭐⭐⭐⭐ A general, near-zero-cost training acceleration strategy with broad practical value for the diffusion model community

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Making Training-Free Diffusion Segmentors Scale with the Generative Power](making_training-free_diffusion_segmentors_scale_with_the_generative_power.md)
- [\[CVPR 2026\] SGMA: Semantic-Guided Modality-Aware Segmentation for Remote Sensing with Incomplete Multimodal Data](sgma_semantic-guided_modality-aware_segmentation_for_remote_sensing_with_incompl.md)
- [\[CVPR 2026\] Task-Oriented Data Synthesis and Control-Rectify Sampling for Remote Sensing Semantic Segmentation](task-oriented_data_synthesis_and_control-rectify_sampling_for_remote_sensing_sem.md)
- [\[ICLR 2026\] Efficient-SAM2: Accelerating SAM2 with Object-Aware Visual Encoding and Memory Retrieval](../../ICLR2026/segmentation/efficient-sam2_accelerating_sam2_with_object-aware_visual_encoding_and_memory_re.md)
- [\[CVPR 2026\] Live Interactive Training for Video Segmentation](live_interactive_training_for_video_segmentation.md)

</div>

<!-- RELATED:END -->
