---
title: >-
  [Paper Note] Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods
description: >-
  [CVPR 2026][Image Generation][Zero-shot detection] This paper proposes STALL, a training-free zero-shot generated video detector that jointly models per-frame spatial likelihoods and inter-frame temporal likelihoods in a…
tags:
  - "CVPR 2026"
  - "Image Generation"
  - "Zero-shot detection"
  - "generated video detection"
  - "likelihood estimation"
  - "whitening transform"
  - "spatial-temporal modeling"
date: 2026-05-08
content_hash: 02f202a6cf8e231f
---

# Training-free Detection of Generated Videos via Spatial-Temporal Likelihoods

**Conference**: CVPR 2026
**arXiv**: [2603.15026](https://arxiv.org/abs/2603.15026)  
**Code**: Available  
**Area**: Image Generation
**Keywords**: Zero-shot detection, generated video detection, likelihood estimation, whitening transform, spatial-temporal modeling

## TL;DR

This paper proposes STALL, a training-free zero-shot generated video detector that jointly models per-frame spatial likelihoods and inter-frame temporal likelihoods in a whitened embedding space. It requires only real video calibration and achieves robust detection across diverse generative models.

## Background & Motivation

**1. State of the Field**: Video generation technologies (e.g., Sora, Veo3) have advanced rapidly, producing highly realistic and temporally coherent videos, which introduces serious risks including misinformation and fraud. Reliable detection of generated videos has thus become critically important.

**2. Limitations of Prior Work**:
- **Image detectors**: Process each frame independently, completely ignoring temporal dynamics, and fail to capture cross-frame artifacts such as motion inconsistencies;
- **Supervised video detectors**: Require large annotated datasets for training and generalize poorly to unseen generative models, which continue to emerge rapidly;
- **D3 (the only existing zero-shot video detector)**: Relies solely on temporal cues (second-order inter-frame differences), ignores per-frame spatial information, and lacks theoretical grounding.

**3. Root Cause**: Neither spatial nor temporal information alone is sufficient—spatial detectors are insensitive to motion artifacts, while temporal detectors are insensitive to per-frame visual anomalies. A method that jointly models both is needed.

**4. Paper Goals**: Design a zero-shot (requiring neither generated samples nor training) and theoretically grounded video detection method that leverages both spatial and temporal evidence simultaneously.

**5. Starting Point**: High-dimensional visual embeddings approximate a Gaussian distribution after whitening, as theoretically guaranteed by the Maxwell–Poincaré lemma. The closed-form log-likelihood can therefore serve as a "authenticity" measure. This idea is extended from static images to inter-frame transition vectors in video.

**6. Core Idea**: Spatial likelihoods are computed from frame embeddings, and temporal likelihoods are computed from normalized inter-frame difference vectors. The two are fused into a unified detection score via percentile normalization. Generated videos deviate from the real data distribution in either the spatial or temporal dimension, and are thereby identified.

## Method

### Overall Architecture

STALL (Spatial-Temporal Aggregated Log-Likelihoods) consists of three components:

1. **Calibration stage (offline)**: A set of real videos (calibration set, e.g., 33k videos from VATEX) is used to extract frame embeddings via a visual encoder (DINOv3). Spatial whitening parameters $(μ, W)$ and temporal whitening parameters $(μ_Δ, W_Δ)$ are computed, and the likelihood score distributions over the calibration set are recorded.
2. **Inference stage**: For a test video, spatial likelihoods are computed per frame, and temporal likelihoods are computed from normalized inter-frame differences; each is aggregated and then fused via percentile normalization.
3. **Decision**: A lower fused score indicates a higher probability of the video being generated.

### Key Designs

#### Spatial Likelihood

- **Function**: Measures the likelihood of each frame under the real image distribution.
- **Mechanism**: The frame embedding $x_t = E(f_t)$ is whitened as $y_t = W(x_t - μ)$, yielding zero mean and identity covariance. Under the Gaussian approximation of whitened coordinates, $y \sim \mathcal{N}(0, I_d)$, and the log-likelihood is $\ell(y) = -\frac{1}{2}(d\log(2\pi) + \|y\|_2^2)$.
- **Design Motivation**: Prior work (e.g., ZED) has verified that CLIP/DINO embeddings exhibit Gaussianity after whitening (validated by Anderson–Darling and D'Agostino–Pearson tests). This paper extends the verification to video frame embeddings and confirms that DINOv3 satisfies the Gaussian assumption on video frames.

#### Temporal Likelihood

- **Function**: Measures the motion consistency of inter-frame transitions.
- **Mechanism**: Inter-frame differences $\Delta_t = x_{t+1} - x_t$ are computed, but raw differences exhibit large norm variance and do not satisfy the Gaussian assumption. The **key innovation** is L2 normalization of the difference vector: $\tilde{\Delta}_t = \Delta_t / \|\Delta_t\|$, projecting directions onto the unit hypersphere. By the Maxwell–Poincaré lemma, vectors uniformly distributed on a high-dimensional hypersphere are approximately Gaussian under any low-dimensional projection. The normalized vectors are then whitened as $z_t = W_Δ(\tilde{\Delta}_t - μ_Δ)$, and the same closed-form log-likelihood is applied.
- **Design Motivation**: Video motion directions are inherently isotropic (no preferred direction), while norm magnitudes vary substantially. Normalization removes the confounding norm effect, and the resulting directional uniformity validates the Gaussian assumption. Unnatural motion patterns in generated videos correspond to lower temporal likelihood values.
- **Edge cases**: If consecutive frames are identical ($\Delta_t = 0$), the transition is discarded; if all frames are identical, the method degrades to purely spatial detection.

#### Score Aggregation and Fusion

- **Frame-level → video-level aggregation**: Spatial likelihoods are aggregated via **maximum** (max); temporal likelihoods via **minimum** (min). Correlation analysis shows that max-spatial and min-temporal scores are least correlated, providing the most complementary information.
- **Percentile normalization**: Since spatial and temporal likelihoods differ in scale, test scores are converted to percentile ranks relative to the calibration set: $\text{perc}(s) = \frac{1}{n}|\{i : s_i \le s\}|$.
- **Unified score**: $s_{\text{video}} = \frac{1}{2}(\text{perc}_{\text{sp}} + \text{perc}_{\text{temp}})$.

### Loss & Training

The proposed method requires **no training whatsoever**. The calibration stage only computes statistics (mean, covariance, whitening matrix) and is a purely statistical estimation process. No learnable parameters are involved at inference time. The only "hyperparameters" are the calibration set size (experiments show that $\geq$ 5k samples yield stable results) and the frame sampling strategy (default: 8 FPS, 16 frames).

## Key Experimental Results

### Main Results

Zero-shot comparisons against image detectors (AEROBLADE, RIGID, ZED) and video detectors (D3-L2, D3-cos) on three benchmarks, with AUC as the primary metric:

| Benchmark | AEROBLADE | RIGID | ZED | D3 (L2) | D3 (cos) | **STALL** |
|-----------|-----------|-------|-----|---------|----------|-----------|
| VideoFeedback (11 models, avg) | 0.58 | 0.63 | 0.54 | 0.54 | 0.55 | **0.83** |
| GenVideo (10 models, avg) | 0.59 | 0.65 | 0.55 | 0.72 | 0.70 | **0.80** |
| ComGenVid (Sora+Veo3, avg) | 0.69 | 0.57 | 0.55 | 0.73 | 0.73 | **0.85** |
| **Overall average** | 0.62 | 0.61 | 0.57 | 0.64 | 0.64 | **0.82** |

STALL achieves the highest average AUC across all benchmarks and is the **only method that maintains AUC > 0.5 on every individual generator** (other methods exhibit decision boundary reversal on certain generators).

Compared against supervised detectors (Figure 6b), STALL's zero-shot performance even surpasses several variants of T2VE and AIGVdet that were trained on the test generators.

### Ablation Study

**Encoder ablation** (Table 2, GenVideo benchmark):

| Encoder | DINOv3 | MobileNet-v3 | ResNet-18 | ViCLIP-L/14 | VideoMAE |
|---------|--------|-------------|-----------|-------------|---------|
| AUC | 0.81 | 0.82 | 0.79 | 0.59 | 0.61 |

- Image encoders (including lightweight MobileNet) all perform well; video encoders underperform because compressing an entire video into a single vector loses the per-frame and per-transition statistical information essential to STALL.

**Calibration set size** (Figure 7a): Varying from 1k to 34k, results stabilize above 5k with negligible standard deviation.

**Robustness tests** (Figure 7b): Under JPEG compression, Gaussian blur, crop-and-resize, and additive noise at five intensity levels, STALL maintains high separability across all perturbations.

**Temporal ablation** (Figure 8): Results are robust to variations in temporal stride, video length, and frame rate.

### Key Findings

1. **Both spatial and temporal cues are essential**: Each dimension alone has blind spots—ZED (spatial only) fails when temporal inconsistency dominates, and D3 (temporal only) fails when spatial anomalies dominate. STALL's joint modeling eliminates both failure modes.
2. **Normalization is critical for temporal likelihood**: Raw inter-frame differences do not satisfy the Gaussian assumption; L2 normalization is required, as jointly validated by the Maxwell–Poincaré lemma and empirical experiments.
3. **Lightweight and efficient**: Inference latency is only 0.49s per video (16 frames), comparable to the fastest baseline D3 and substantially faster than AEROBLADE and AIGVdet.
4. **Insensitivity to calibration set choice**: Using different sources (VATEX, Kinetics-400, real videos from VideoFeedback) as calibration sets yields similar performance.

## Highlights & Insights

- **Theory-driven design**: Rather than a purely empirical approach, STALL is grounded in Gaussian likelihoods and the Maxwell–Poincaré lemma, providing an interpretable and verifiable theoretical framework. Failures can be quantitatively diagnosed as arising from spatial or temporal scores.
- **Elegant simplicity**: The entire method has no learnable parameters; the core operations are whitening, norm computation, and percentile ranking—straightforward to implement yet substantially outperforming complex supervised approaches.
- **Zero-shot generalization**: Detects outputs from the latest models such as Sora and Veo3 without any adaptation.
- **Percentile fusion**: Avoids the scale incompatibility between spatial and temporal likelihoods, yielding greater robustness than direct weighted averaging.

## Limitations & Future Work

1. **Degradation on static videos**: When inter-frame variation is minimal, the temporal signal vanishes and the method degrades to purely spatial detection, potentially missing carefully crafted generated content that is only anomalous temporally.
2. **Calibration set dependency**: Although the paper claims robustness to calibration set selection, real video data are still required; performance under extreme domain shift (e.g., medical or satellite videos) remains unknown.
3. **Limitations of the Gaussian assumption**: For embedding spaces with special structure (e.g., severe concentration on narrow cones), the Gaussian approximation may lose accuracy.
4. **Fully generated videos only**: The method does not address partial replacement or editing scenarios (deepfakes), which require pixel-level localization capabilities.
5. **Potential adaptive attacks**: If an adversary is aware of the detection mechanism, they may attempt to match the spatial/temporal statistics of generated videos to those of real ones; adversarial robustness is not thoroughly discussed.
6. **Fixed frame sampling strategy**: Uniform sampling may miss locally anomalous segments; adaptive sampling strategies are worth exploring.

## Related Work & Insights

- **ZED** (zero-shot image detection): The spatial likelihood in this paper directly inherits the whitening + Gaussian likelihood framework from ZED; the core contribution is its extension to the temporal dimension.
- **D3** (first zero-shot video detector): Relies on empirical assumptions about second-order inter-frame differences, lacks theoretical grounding, and ignores spatial information. STALL surpasses D3 by using first-order normalized differences with theoretical guarantees.
- **Maxwell–Poincaré lemma**: Provides rigorous theoretical support for the normalization operation—projections of uniform distributions on high-dimensional hyperspheres are approximately Gaussian—which forms the theoretical foundation of STALL's temporal modeling.
- **Broader implications**: The zero-shot detection paradigm is transferable to other modalities (e.g., audio generation detection, 3D generation detection), as long as the embedding space satisfies the Gaussian assumption. The calibration set requires only real data, which sets an extremely low barrier to deployment.

## Rating

⭐⭐⭐⭐ Theoretically elegant, experimentally rigorous, and methodologically minimalist with striking effectiveness—a benchmark work for zero-shot generated video detection. One star is withheld for its limitation to fully generated scenarios and insufficient analysis of adversarial robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Object-WIPER: Training-Free Object and Associated Effect Removal in Videos](object-wiper_training-free_object_and_associated_effect_removal_in_videos.md)
- [\[CVPR 2026\] Just-in-Time: Training-Free Spatial Acceleration for Diffusion Transformers](just-in-time_training-free_spatial_acceleration_for_diffusion_transformers.md)
- [\[CVPR 2026\] Diversity over Uniformity: Rethinking Representation in Generated Image Detection](diversity_over_uniformity_rethinking_representation_in_generated_image_detection.md)
- [\[ICCV 2025\] Free4D: Tuning-free 4D Scene Generation with Spatial-Temporal Consistency](../../ICCV2025/image_generation/free4d_tuning-free_4d_scene_generation_with_spatial-temporal_consistency.md)
- [\[CVPR 2026\] SketchDeco: Training-Free Latent Composition for Precise Sketch Colourisation](sketchdeco_training-free_latent_composition_for_precise_sketch_colourisation.md)

</div>

<!-- RELATED:END -->
