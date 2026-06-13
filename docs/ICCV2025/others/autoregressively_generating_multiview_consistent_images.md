---
title: >-
  [Paper Note] Auto-Regressively Generating Multi-View Consistent Images (MV-AR)
description: >-
  [ICCV 2025][autoregressive generation] This paper is the first to introduce autoregressive (AR) models into multi-view image generation. By generating views sequentially…
tags:
  - "ICCV 2025"
  - "autoregressive generation"
  - "multi-view consistency"
  - "condition injection"
  - "data augmentation"
  - "unified multimodal"
date: 2026-05-08
content_hash: e3d88f5cf8c201d5
---

# Auto-Regressively Generating Multi-View Consistent Images (MV-AR)

**Conference**: ICCV 2025
**arXiv**: [2506.18527](https://arxiv.org/abs/2506.18527)  
**Code**: [https://github.com/MILab-PKU/MVAR](https://github.com/MILab-PKU/MVAR)  
**Area**: Other
**Keywords**: autoregressive generation, multi-view consistency, condition injection, data augmentation, unified multimodal

## TL;DR

This paper is the first to introduce autoregressive (AR) models into multi-view image generation. By generating views sequentially, the model leverages all preceding views to enhance consistency across distant viewpoints. It further proposes a unified multimodal condition injection architecture and a Shuffle Views data augmentation strategy, enabling a single model to handle text, image, and geometry conditions simultaneously.

## Background & Motivation

Existing multi-view image generation methods are predominantly based on diffusion models (e.g., MVDream, Zero123++, SyncDreamer, Wonder3D), which typically generate all views simultaneously via mechanisms such as cross-view attention. However, this "simultaneous generation" paradigm has a fundamental limitation: when the reference view and the target view are far apart (e.g., generating a back view from a front view), the visual overlap is minimal, rendering the reference information nearly ineffective and causing severe degradation in consistency across distant views.

Human perception of 3D objects is inherently incremental—one side is observed first, then the next, gradually building a comprehensive understanding. AR models are naturally suited to this progressive generation paradigm: when generating the $n$-th view, all $n{-}1$ preceding views can be exploited as references, thereby providing sufficient context for distant views.

## Core Problem

Applying AR models to multi-view generation presents three key challenges:

1. **Insufficient condition injection (Issue 1)**: AR models lack effective mechanisms for injecting conditions such as camera pose, reference images, and geometry.
2. **Limited high-quality data (Issue 2)**: AR models require large-scale data to avoid overfitting, yet high-quality 3D object data is scarce compared to the billions of text tokens available in NLP.
3. **Accumulated errors (Issue 3)**: In AR generation, a low-quality intermediate view will serve as a reference for subsequent views, causing error propagation.

## Method

### Overall Architecture

MV-AR is built upon a pretrained text-to-image AR model (LLamaGen). A 2D VQVAE encodes $N$ views into token sequences, which are concatenated in view order for autoregressive modeling. The Transformer backbone follows Llama, employing RMSNorm, SwiGLU, and AdaLN. Text and geometry conditions are prepended as context tokens, camera poses are injected via Shift Position Encoding (SPE), and image conditions are injected token-by-token through the IWC module.

### Key Designs

1. **Split Self-Attention (SSA) — Text Condition**: Text is encoded with FLAN-T5 XL and prepended as prefilling tokens. To prevent subsequent image tokens from corrupting the text tokens (modality misalignment), SSA is designed so that after standard self-attention, the outputs at text positions are zeroed and the original text tokens are added back. This ensures text conditions are not contaminated by image tokens while still allowing image tokens to attend to text information. Experiments show SSA significantly improves CLIP Score.

2. **Shift Position Encoding (SPE) — Camera Condition**: Plücker Ray Embedding encodes the ray origin and direction (6-dimensional) for each spatial position. These embeddings are directly added to token embeddings as positional offsets, informing the model of the view and position of each token and providing precise physical angle guidance.

3. **Image Warp Controller (IWC) — Image Condition**: Rather than using high-level semantic features from CLIP or DINO, the VQVAE encoder extracts low-level features from the reference image. These are then processed through Self-Attention + Cross-Attention (cross-attended with camera pose) + FFN to predict overlapping content and texture between the current and reference views, and injected token-by-token into the model via residual connections. Low-level features better preserve color and texture details.

4. **Geometry Condition Injection**: Point clouds (8,192 surface-sampled points with normals) serve as the geometry condition. A pretrained shape encoder (Michelangelo) maps them to a fixed-length latent token sequence, inserted between text tokens and the start token as prepended context.

### Loss & Training

**Loss Function**: Standard AR negative log-likelihood loss, averaged over all vocabulary positions:

$$\mathcal{L}_{ar} = -\frac{1}{T}\sum_{t=1}^{T}\log p(q_t|q_{<t})$$

**Shuffle Views (ShufV) Data Augmentation**: The order of $N$ views is randomly permuted to construct training sequences. For $N$ views, this yields $\frac{N(N-1)}{2}$ permutation combinations, expanding the effective training data by orders of magnitude. ShufV also enhances the IWC's ability to capture overlapping regions between arbitrary view pairs (bidirectional learning of A→B and B→A transformations).

**Progressive Learning**:
- A text-to-multi-view (t2mv) model is first trained as a baseline.
- An X-to-multi-view (X2mv) model is subsequently trained on top of t2mv: text conditions are randomly dropped, and image/geometry conditions are randomly combined.
- The probability of condition dropout and combination linearly increases from 0 to 0.5 over the first 10k iterations and is held at 0.5 thereafter.
- When text is dropped, it is replaced by a generic prompt that excludes the target description (e.g., "Generate multi-view images of the following \<img\>").

## Key Experimental Results

**Text-to-Multi-View (GSO dataset, 30 objects)**

| Method | FID↓ | IS↑ | CLIP-Score↑ |
|--------|------|-----|-------------|
| MVDream† | 143.72 | 7.93 | 28.95 |
| LLamaGen | 146.11 | 5.78 | 28.36 |
| **MV-AR (Ours)** | **144.29** | **8.00** | **29.49** |

**Image-to-Multi-View Ablation (GSO dataset)**

| Image Condition Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|------------------------|-------|-------|--------|
| In-context | 11.92 | 0.538 | 0.477 |
| Cross Attention | 15.13 | 0.709 | 0.310 |
| **IWC (Ours)** | **22.99** | **0.907** | **0.084** |

**Image-to-Multi-View Comparison (GSO dataset)**

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| RealFusion | 15.26 | 0.722 | 0.283 |
| Zero123 | 18.93 | 0.779 | 0.166 |
| SyncDreamer | 19.89 | 0.801 | 0.129 |
| Wonder3D | 22.82 | 0.892 | 0.062 |
| Era3D | 22.73 | 0.911 | 0.071 |
| **MV-AR (Ours)** | **22.99** | **0.907** | 0.084 |

### Ablation Study

| Variant | FID/IS (t2mv) | PSNR/SSIM/LPIPS (i2mv) |
|---------|---------------|------------------------|
| w/o SPE | 147.29 / 7.26 | 21.30 / 0.843 / 0.118 |
| w/o ShufV | 173.51 / 4.77 | 18.27 / 0.778 / 0.194 |
| **MV-AR (Full)** | **144.29 / 8.00** | **22.99 / 0.907 / 0.084** |

- **Effect of SPE**: Removing SPE increases FID by 3 points and decreases PSNR by 1.69, confirming that camera pose encoded as positional offsets is critical for multi-view consistency.
- **Effect of ShufV**: Removing ShufV causes FID to surge to 173 and PSNR to drop sharply to 18.27, making it the most impactful design component. This demonstrates that data augmentation is highly effective in mitigating AR model overfitting under limited data.
- **IWC vs. other image condition methods**: IWC (PSNR 22.99) substantially outperforms Cross Attention (15.13) and In-context (11.92), as the AR base model lacks image-to-image capability and IWC achieves precise control through token-level low-level feature injection.

## Highlights & Insights

1. **Paradigm innovation**: MV-AR is the first to introduce AR models into multi-view image generation, replacing simultaneous generation with progressive generation and naturally resolving the consistency problem across distant views.
2. **ShufV data augmentation**: Simple yet effective—shuffling view order expands the training data by a factor of $\frac{N(N-1)}{2}$ while enhancing the model's ability to transform between arbitrary view pairs.
3. **Unified multimodal framework**: A single model supports arbitrary combinations of text, image, and geometry conditions, making MV-AR the first unified X-to-multi-view generation model.
4. **Design philosophy of IWC**: Using low-level VQVAE features rather than high-level CLIP/DINO features for image conditioning better preserves color and texture consistency.

## Limitations & Future Work

1. **Limitation of 2D VQVAE**: The authors deliberately avoid 3D VAE (since inter-view information exchange during encoding would violate the AR motivation), but the 2D VQVAE may constrain 3D geometric understanding; the authors suggest exploring causal 3D VAE in future work.
2. **Suboptimal LPIPS**: In the i2mv task, LPIPS ranks third (0.084 vs. Wonder3D's 0.062), suggesting that overly strict low-level feature constraints may sacrifice perceptual quality.
3. **Accumulated errors not fully addressed**: Although ShufV partially mitigates the issue, no direct technical solution targeting error accumulation is proposed.
4. **Limited evaluation scale**: Evaluation is conducted on only 30 GSO objects, lacking large-scale quantitative assessment.
5. **Resolution constraint**: The training resolution of 256×256 is relatively low given current trends in high-resolution generation.
6. **Inference speed**: A comparison of AR token-by-token generation speed against diffusion model multi-step denoising is absent.

## Related Work & Insights

- **vs. MVDream**: MVDream generates four views simultaneously via cross-view attention, suffering from poor consistency across distant views; MV-AR's AR paradigm naturally leverages all preceding views and achieves superior CLIP Score (29.49 vs. 28.95).
- **vs. Wonder3D/Era3D**: These diffusion-based methods achieve comparable i2mv performance (PSNR gap <0.3), but rely on image-to-image pretraining priors in diffusion models, whereas MV-AR attains competitive results starting from an AR model.
- **vs. LLamaGen**: MV-AR extends LLamaGen, and the improvements in IS (5.78→8.00) and CLIP Score (28.36→29.49) brought by SSA validate the improved text condition handling.
- **vs. VAR/PixelCNN and other AR methods**: These methods are limited to single-image generation; MV-AR is the first to extend the AR paradigm to multi-view scenarios.

## Highlights & Insights

1. **Feasibility of AR for 3D generation**: This work demonstrates that AR models can rival diffusion models in multi-view generation, suggesting the potential of AR models in broader 3D tasks (e.g., 3D reconstruction, 4D generation).
2. **Generalizability of Shuffle Views**: The strategy of augmenting data by shuffling sequence order can be extended to other sequential generation tasks (e.g., shuffling frame order in video generation).
3. **Value of unified condition frameworks**: The progressive training with random condition dropout strategy serves as a general paradigm for building unified multimodal models, applicable to other multi-condition generation tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ First application of AR models to multi-view generation with a clear paradigm contribution; individual condition injection modules are relatively standard in design.
- Experimental Thoroughness: ⭐⭐⭐ Ablation study is well-designed, but the evaluation set covers only 30 objects, lacking large-scale assessment and comparison with more recent methods.
- Writing Quality: ⭐⭐⭐⭐ Problem analysis is clear (three Issues are identified and addressed in sequence) with a well-organized structure; some formula presentations are slightly redundant.
- Value: ⭐⭐⭐⭐ Provides a new AR-based baseline for multi-view generation; the unified multimodal condition framework has practical significance and offers meaningful insights for subsequent AR-based 3D generation research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Φ-GAN: Physics-Inspired GAN for Generating SAR Images Under Limited Data](ph-gan_physics-inspired_gan_for_generating_sar_images_under_limited_data.md)
- [\[ICCV 2025\] Intra-view and Inter-view Correlation Guided Multi-view Novel Class Discovery](intra-view_and_inter-view_correlation_guided_multi-view_novel_class_discovery.md)
- [\[ICCV 2025\] Recover Biological Structure from Sparse-View Diffraction Images with Neural Volumetric Prior](recover_biological_structure_from_sparse-view_diffraction_images_with_neural_vol.md)
- [\[ICCV 2025\] Switch-a-View: View Selection Learned from Unlabeled In-the-wild Videos](switch-a-view_view_selection_learned_from_unlabeled_in-the-wild_videos.md)
- [\[NeurIPS 2025\] Incomplete Multi-view Clustering via Hierarchical Semantic Alignment and Cooperative Completion](../../NeurIPS2025/others/incomplete_multi-view_clustering_via_hierarchical_semantic_alignment_and_coopera.md)

</div>

<!-- RELATED:END -->
