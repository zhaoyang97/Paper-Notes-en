---
title: >-
  [Paper Note] DiffEye: Diffusion-Based Continuous Eye-Tracking Data Generation Conditioned on Natural Images
description: >-
  [NeurIPS 2025][Image Generation][Diffusion models] This paper proposes DiffEye, the first diffusion-based framework that directly utilizes raw eye-tracking data to generate continuous and diverse eye movement trajectorie…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "Diffusion models"
  - "eye movement trajectory generation"
  - "scanpath prediction"
  - "visual attention modeling"
  - "corresponding position embedding"
date: 2026-05-08
content_hash: 5f837d155592d3f4
---

# DiffEye: Diffusion-Based Continuous Eye-Tracking Data Generation Conditioned on Natural Images

**Conference**: NeurIPS 2025
**arXiv**: [2509.16767](https://arxiv.org/abs/2509.16767)  
**Code**: Available ([https://diff-eye.github.io/](https://diff-eye.github.io/))  
**Area**: Image Generation / Diffusion Models / Eye Tracking
**Keywords**: Diffusion models, eye movement trajectory generation, scanpath prediction, visual attention modeling, corresponding position embedding

## TL;DR

This paper proposes DiffEye, the first diffusion-based framework that directly utilizes raw eye-tracking data to generate continuous and diverse eye movement trajectories conditioned on natural images, while introducing Corresponding Position Embedding (CPE) to align the gaze space with the image semantic space.

## Background & Motivation

- **Importance of eye tracking**: Eye tracking provides critical visual attention data for fields such as VR, developmental science (e.g., autism research), and advertising effectiveness analysis, but data collection is costly and time-consuming.
- **Two major limitations of existing methods**:
  1. Most methods operate on scanpaths (discrete fixation point sequences), discarding the rich spatiotemporal information of raw trajectories (average scanpath in MIT1003: 8.4 steps vs. raw trajectory: 723.7 steps).
  2. Existing methods either simulate variability via autoregressive sampling or perform deterministic prediction, failing to truly capture the inherent distribution of human gaze behavior.
- **Core assumption**: Training generative models on complete eye movement trajectories can more effectively characterize visual attention dynamics than scanpath-based methods; diffusion models are particularly well-suited for modeling this intrinsic stochasticity.

## Method

### Overall Architecture

DiffEye adopts the DDPM framework to learn the conditional distribution $p_\theta(\mathcal{R}|\mathcal{S})$, i.e., the eye movement trajectory distribution given visual stimulus $I$:

- **Input**: Visual stimulus image $I \in \mathbb{R}^{H \times W \times 3}$ (resized to 224×224) and fixed-length trajectory $R \in \mathbb{R}^{L \times 2}$ ($L=720$).
- **Forward process**: Gaussian noise is progressively added to the trajectory over $T_{diff}=1000$ steps.
- **Reverse process**: A U-Net predicts the noise and denoises conditioned on the image to recover the original trajectory.
- **Inference**: Starting from pure noise, DDIM sampling with 50 steps generates trajectories, which can be converted to scanpaths or saliency maps.

### Key Designs

#### 1. Backbone: 1D U-Net

- Downsampling, middle, and upsampling block structure using **1D convolutions** for temporal downsampling/upsampling of trajectories.
- Each block includes self-attention layers to capture temporal dependencies in the trajectory.
- Diffusion timestep $t_{diff}$ is injected into each block via sinusoidal positional encoding + MLP + SiLU.

#### 2. Image Conditioning Mechanism (Progressive Design)

| Scheme | Approach | Effect |
|--------|----------|--------|
| Global feature | DINOv2 global vector concatenated to trajectory tokens | Poor generation quality, lacks spatial semantics |
| Patch features + single-layer cross-attention | DINOv2 patch tokens and trajectory tokens perform cross-attention | Insufficient conditioning |
| Patch features + multi-layer cross-attention | Cross-attention appended to each U-Net block | Significant improvement |
| **FeatUp high-resolution features** | Replace DINOv2 patches with FeatUp (32×32 vs. 16×16) | Final design, higher spatial precision |

#### 3. Corresponding Position Embedding (CPE) — Core Contribution

The key idea of CPE: trajectory tokens and image patch tokens **share the same 2D sinusoidal positional encoding grid**, enabling spatial alignment:

- Construct a positional embedding grid $P \in \mathbb{R}^{H \times W \times D}$.
- For trajectory step $i$ at coordinate $(x_i, y_i)$, extract the corresponding positional encoding: $R_i^{CPE} = R_{proj}[i] + P[y_i, x_i, :]$.
- For image features, interpolate the positional grid to patch resolution and add to features: $F_{CPE} = F_{proj} + P'$.
- In cross-attention, trajectory points and image patches at the same spatial location receive matching positional signals.

#### 4. Data Preprocessing

- Uses the MIT1003 dataset (15 subjects × 1,003 images × 3-second free viewing, 240 Hz sampling).
- Blinks and NaN values are removed; sequences of ≥720 steps are retained, yielding 8,934 trajectories in total.
- All sequences are uniformly truncated/downsampled to 720 steps; 90%/10% train/test split by image.

### Loss & Training

- **Loss function**: Standard DDPM noise prediction loss $\min_\theta \mathbb{E}_{t,R^{(0)},\epsilon}\left[\|\epsilon - \epsilon_\theta(R^{(t)}, t, I)\|^2\right]$
- Adam optimizer, fixed learning rate $1 \times 10^{-4}$, trained for 3,000 epochs.
- Linear noise schedule $[1 \times 10^{-4}, 2 \times 10^{-2}]$.
- **Classifier-Free Guidance (CFG)** is used to enhance conditional control.
- DDIM 50-step sampling at inference.

## Key Experimental Results

### Main Results: Scanpath Generation (MIT1003 + OSIE)

| Test Set | Method | Levenshtein ↓ (Mean/Best) | DFD ↓ ×10² | DTW ↓ ×10³ | TDE ↓ (Mean/Best) |
|----------|--------|--------------------------|------------|------------|-------------------|
| MIT1003 | IOR-ROI | 13.574/11.092 | 3.777/2.460 | 1.834/1.317 | 108.284/80.944 |
| MIT1003 | DeepGaze III (seen) | 14.415/11.856 | 3.553/2.160 | 1.757/1.141 | 96.456/65.408 |
| MIT1003 | **DiffEye** | **13.009/9.709** | **3.529/2.449** | **1.573/1.067** | **88.661/53.486** |
| OSIE | DeepGaze III | 15.507/12.532 | 3.206/2.077 | 1.765/1.166 | 84.337/57.786 |
| OSIE | **DiffEye** | **14.771/12.077** | **3.068/2.238** | **1.552/1.089** | **81.925/54.347** |

DiffEye is trained on substantially fewer trajectories (8,934) than the baselines (DeepGaze III uses 615K scanpaths), yet achieves consistent improvements across all metrics.

### Ablation Study

| Configuration | Levenshtein ↓ ×10² | DTW ↓ ×10⁴ | TDE ↓ (Mean) |
|---------------|---------------------|------------|--------------|
| DiffEye (full) | **0.130** | **0.157** | **88.661** |
| w/o FeatUp | 0.133 | 0.163 | 91.007 |
| w/o CPE | 0.141 | 0.180 | 100.792 |
| w/o U-Net cross-attention | 0.143 | 0.189 | 107.962 |
| w/o patch-level features (global only) | 0.153 | 0.209 | 116.226 |

### Key Findings

1. **Each component contributes**: patch-level features > multi-layer cross-attention > CPE > FeatUp high resolution; cumulative gains are substantial.
2. **Small data surpasses large-scale baselines**: DiffEye with only 8,934 trajectories outperforms HAT and GazeFormer trained on 60K+ scanpaths.
3. **Strong generalization**: DiffEye achieves state-of-the-art performance on the completely unseen OSIE dataset.
4. **First continuous eye movement trajectory generation**: prior methods only generate discrete scanpaths.

## Highlights & Insights

- **Value of raw data**: Directly utilizing raw eye movement trajectories (720 steps) rather than compressed scanpaths (~8 steps) yields approximately 85× more information, substantially enhancing generation quality.
- **Elegant CPE design**: By sharing a positional encoding grid, CPE achieves trajectory–image spatial alignment with zero additional parameters — a simple yet effective solution.
- **Natural advantage of diffusion models**: Diverse trajectories are generated without autoregressive sampling, genuinely modeling the stochastic distribution of human gaze behavior.
- **Downstream convertibility**: Generated continuous trajectories can be converted on demand to scanpaths or saliency maps, enabling a single model to serve multiple tasks.

## Limitations & Future Work

1. **Fixed-length output only**: The current design is fixed at 720 steps (240 Hz) and cannot adapt to different sampling rates or sequence lengths.
2. **Single dataset**: MIT1003 is the only natural image dataset providing raw eye movement trajectories, limiting scale.
3. **Free-viewing task only**: Performance on other visual tasks such as visual search or visual question answering has not been validated.
4. **Suboptimal saliency prediction**: Saliency maps derived indirectly from trajectories are less accurate than dedicated saliency models.
5. Future directions include: transfer learning, variable-length generation, and personalized modeling (e.g., autism populations vs. neurotypical individuals).

## Related Work & Insights

- **DiffGaze** (Jiao et al., 2024): Diffusion-based gaze generation on 360° images, but conditioned only on global features; DiffEye demonstrates that patch-level features + CPE substantially outperform global features.
- **HAT** (Yang et al., 2024): A unified Transformer model for FV/TP/TA tasks, but with a sampling strategy similar to DeepGaze III.
- **FeatUp** (Fu et al., 2024): A model-agnostic feature upsampling framework that provides high-resolution semantic features for DiffEye.
- **Insight**: The paradigm of diffusion models combined with spatial position alignment is generalizable to other sequence-conditioned-on-image generation tasks.

## Rating

| Dimension | Score | Comment |
|-----------|-------|---------|
| Novelty | ★★★★☆ | First diffusion-based eye movement trajectory generation on natural images; CPE design is elegant |
| Technical Depth | ★★★★☆ | Well-motivated architecture, comprehensive ablation, but limited theoretical analysis |
| Experimental Thoroughness | ★★★★☆ | Multi-metric evaluation, multi-baseline comparison, thorough ablation study |
| Value | ★★★★☆ | Applicable to VR, developmental science, and other domains; code is open-source |
| Writing Quality | ★★★★☆ | Clear figures, well-stated motivation, logical flow |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Detecting Generated Images by Fitting Natural Image Distributions](detecting_generated_images_by_fitting_natural_image_distributions.md)
- [\[NeurIPS 2025\] Mitigating Sexual Content Generation via Embedding Distortion in Text-conditioned Diffusion Models](mitigating_sexual_content_generation_via_embedding_distortion_in_text-conditione.md)
- [\[NeurIPS 2025\] Continuous Diffusion Model for Language Modeling](continuous_diffusion_model_for_language_modeling.md)
- [\[NeurIPS 2025\] SAO-Instruct: Free-form Audio Editing using Natural Language Instructions](sao-instruct_free-form_audio_editing_using_natural_language_instructions.md)
- [\[NeurIPS 2025\] WMCopier: Forging Invisible Image Watermarks on Arbitrary Images](wmcopier_forging_invisible_image_watermarks_on_arbitrary_images.md)

</div>

<!-- RELATED:END -->
