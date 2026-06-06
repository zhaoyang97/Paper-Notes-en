---
title: >-
  [Paper Note] Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable
description: >-
  [NeurIPS 2025][Image Generation][AI-generated image detection] This paper proposes Dual Data Alignment (DDA), which generates synthetic training images via pixel-domain and frequency-domain dual alignment to eliminate sp…
tags:
  - "NeurIPS 2025"
  - "Image Generation"
  - "AI-generated image detection"
  - "data alignment"
  - "frequency-domain alignment"
  - "generalizability"
  - "VAE reconstruction"
  - "dataset bias"
date: 2026-05-08
content_hash: 63610dc21072fcce
---

# Dual Data Alignment Makes AI-Generated Image Detector Easier Generalizable

**Conference**: NeurIPS 2025
**arXiv**: [2505.14359](https://arxiv.org/abs/2505.14359)  
**Code**: [roy-ch/Dual-Data-Alignment](https://github.com/roy-ch/Dual-Data-Alignment)  
**Area**: Image Generation
**Keywords**: AI-generated image detection, data alignment, frequency-domain alignment, generalizability, VAE reconstruction, dataset bias

## TL;DR

This paper proposes Dual Data Alignment (DDA), which generates synthetic training images via pixel-domain and frequency-domain dual alignment to eliminate spurious correlations caused by dataset bias. By forcing the detector to learn only forgery-relevant features, DDA achieves an average accuracy of 90.7% across 11 benchmarks, substantially outperforming existing methods.

## Background & Motivation

**Growing threat of AI-generated images (AIGI)**: The rapid advancement of generative models such as diffusion models and autoregressive models has made forged images a serious security concern in contexts including misinformation, fraud, and copyright infringement, underscoring the urgent need for reliable detection methods.

**Poor generalizability of existing detectors**: Current detectors perform well on their training sets but suffer significant performance degradation in zero-shot scenarios involving cross-dataset or cross-generator evaluation, particularly when confronted with unseen generative paradigms.

**Dataset bias as the root cause of poor generalizability**: Real and synthetic images in existing datasets exhibit systematic differences in non-causal attributes such as format (JPEG vs. PNG), resolution (diverse vs. fixed multiples of 128), and semantic content. Detectors tend to exploit these spurious correlations rather than learning genuine forgery traces.

**Reconstruction-based alignment methods are incomplete**: Methods such as DRCT and AlignedForensics attempt to reduce content bias by aligning synthetic images with real images at the pixel level through diffusion or VAE reconstruction.

**Frequency-domain bias has been overlooked**: The authors find that even after pixel-level alignment, reconstructed images exhibit substantial frequency-domain discrepancies. VAE reconstruction restores high-frequency details lost due to JPEG compression in real images, resulting in reconstructed synthetic images with far greater high-frequency energy than their real counterparts, thereby introducing a new spurious cue.

**Frequency-domain bias is exploitable by detectors**: Experiments show that the frequency-domain detector SAFE can identify visually near-identical VAE-reconstructed images with 93% success rate, yet detection rates drop sharply when high-frequency information is slightly masked—demonstrating that detectors indeed overfit to high-frequency bias rather than genuine forgery features.

## Method

### Overall Architecture

DDA consists of three steps: (1) VAE reconstruction for pixel-level alignment; (2) high-frequency fusion to eliminate frequency-domain bias; and (3) pixel-level Mixup for further pixel-domain alignment. This pipeline produces synthetic images that are highly aligned with real images in both the pixel and frequency domains for use as training data. A DINOv2 + LoRA fine-tuned detector is then trained on these aligned images to achieve strong generalizable detection.

### Key Design 1: VAE Reconstruction (Pixel-Level Alignment)

- **Function**: Real images are reconstructed via the VAE encoder-decoder of Stable Diffusion as $\hat{x} = \text{Decoder}(\text{Encoder}(x))$, without modifying the latent space, yielding synthetic images with high pixel-level similarity to the originals.
- **Mechanism**: The VAE decoder constitutes the final stage of diffusion generators; the artifacts it introduces exhibit cross-generator generality. Learning decision boundaries corresponding to these artifacts enables generalization to more "distant" synthetic images (e.g., those produced by text-to-image generation).
- **Design Motivation**: Compared to diffusion reconstruction (which alters semantics by modifying the latent space) or text-to-image generation (which lacks precise supervision), pure VAE reconstruction minimizes pixel-level discrepancy between real and synthetic images, yielding the tightest real–fake image pairs.

### Key Design 2: Frequency-Domain Alignment (JPEG Compression Matching)

- **Function**: The JPEG quality factor of each real image is estimated, and during training the corresponding VAE-reconstructed image is subjected to JPEG compression at the same quality factor with 50% probability, aligning the high-frequency energy distributions of both images.
- **Mechanism**: The root cause of frequency-domain bias is that real images undergo JPEG compression (attenuating high frequencies), whereas VAE-reconstructed images do not (preserving high frequencies intact). Applying equivalent compression to the reconstructed images eliminates this discrepancy.
- **Design Motivation**: Removing the shortcut whereby detectors exploit "rich high frequencies = synthetic" forces the model to learn genuine forgery artifacts rather than compression-induced differences.

### Key Design 3: Pixel-Level Mixup

- **Function**: Real images and frequency-aligned synthetic images are blended at the pixel level as $x_{\text{mix}} = r_{\text{pixel}} \cdot x_{\text{real}} + (1 - r_{\text{pixel}}) \cdot x_{\text{syn}}$, where $r_{\text{pixel}} \sim \mathcal{U}(0, R_{\text{pixel}})$.
- **Mechanism**: Controllable pixel mixing further narrows the real–fake gap, positioning synthetic images near the boundary of the real data manifold and encouraging the model to learn a more compact and transferable decision boundary.
- **Design Motivation**: t-SNE visualizations demonstrate that DDA-generated synthetic images have cluster centroids closest to real images among all compared methods (proximity order: DDA < VAE Rec. < Diff. Rec. < T2I). A tighter decision boundary implies stronger generalizability.

### Key Design 4: Two New Evaluation Benchmarks

- **DDA-COCO**: 5K real MSCOCO images paired with 25K synthetic images produced by five variants of VAE reconstruction with frequency-domain alignment, designed to assess whether detectors rely on genuine forgery features rather than bias cues.
- **EvalGEN**: 2,765 images generated by five recent generators—FLUX, GoT, Infinity, NOVA, and OmniGen (including autoregressive models)—to evaluate generalizability to unseen generators.

## Loss & Training

- Backbone: DINOv2 + LoRA (rank=8) fine-tuning
- Input resolution: 336×336; random crop during training, center crop during validation
- Training data: **MSCOCO images and their DDA-aligned counterparts only** (118K real + 118K synthetic)
- VAE used: SD 2.1 VAE
- Frequency alignment probability: JPEG compression applied with 50% probability during training
- All evaluations use a **single model** without dataset-specific fine-tuning or threshold adjustment

## Key Experimental Results

### Table 1: Overall Comparison on 11 Benchmarks (Balanced Accuracy)

| Method | Average Acc. | Minimum Acc. | Std. Dev. |
|--------|-------------|--------------|-----------|
| DDA (Ours) | **90.7%** | **81.4%** | Smallest (±5.3) |
| AlignedForensics | 75.0% | 53.9% | ±11.1 |
| DRCT | 70.1% | 50.6% | ±14.6 |
| C2P-CLIP | 62.1% | 38.9% | ±15.6 |
| FatFormer | 59.6% | 45.6% | ±14.6 |

DDA surpasses the second-best method by **15.7%** in average accuracy and **27.5%** in minimum accuracy, with a standard deviation roughly half that of competing methods.

### Table 2: Generalization to Novel Generators on EvalGEN (Balanced Accuracy)

| Method | Flux | GoT | Infinity | NOVA | OmniGen | Average |
|--------|------|-----|----------|------|---------|---------|
| DDA | 89.9 | 99.5 | 97.8 | 99.5 | 99.5 | **97.2 ±4.2** |
| DRCT | 72.5 | 81.4 | 77.9 | 84.6 | 72.5 | 77.8 ±5.4 |
| AlignedForensics | 32.0 | 72.3 | 74.0 | 84.8 | 77.0 | 68.0 ±20.7 |
| C2P-CLIP | 8.7 | 49.6 | 35.3 | 86.4 | 14.5 | 38.9 ±31.2 |

DDA demonstrates strong cross-architecture generalizability on the latest generators, including autoregressive models.

### Other Key Results

- **In-the-wild datasets**: Chameleon 82.4% (2nd place: 71.0%), WildRF 90.3% (2nd: 80.1%), BFree-Online 95.1% (2nd: 68.5%)
- **Robustness**: Under JPEG-60, RESIZE-2.0, and BLUR-2.0 post-processing, DDA outperforms the second-best by 10.5%, 4.1%, and 5.7%, respectively
- **Data generation efficiency**: Full DDA dataset construction requires only 5.9h, far below DRCT (64.6h) and B-Free (258.79h)
- **Ablation study**: Performance remains stable for $P_{\text{pixel}}$ and $R_{\text{pixel}}$ in the range 0.2–0.8; the SD 2.1 VAE yields the best results

## Highlights & Insights

1. **Deep problem insight**: This work is the first to identify and systematically validate the frequency-domain bias overlooked in pixel-level alignment methods, providing a frequency-perspective explanation for the shortcomings of existing alignment approaches.
2. **Simple yet effective solution**: DDA requires only three steps—VAE reconstruction, JPEG compression matching, and pixel Mixup—without complex training pipelines or additional models.
3. **Highly thorough experimentation**: 11 benchmark datasets (4 in-the-wild), 9 competing methods, robustness analysis, ablation study, and visualization analyses provide coverage that is unmatched in this field.
4. **Clear efficiency advantage**: Training uses only 118K MSCOCO images with dataset construction taking just 5.9h, far below competing methods.
5. **Two high-quality evaluation benchmarks**: DDA-COCO and EvalGEN fill a gap in AIGI detection evaluation.

## Limitations & Future Work

1. **Performance gap under heavy post-processing**: The authors acknowledge that there remains room for improvement when images undergo extensive post-processing in real-world settings (e.g., social media compression).
2. **Interference from smartphone AI enhancement**: Modern smartphone photography pipelines embed AI enhancement, which may introduce synthetic-like artifacts into real photographs, increasing detection complexity.
3. **Dependence on VAE architecture assumption**: The generalizability of DDA rests on the assumption that the VAE constitutes the final stage of diffusion generators; its applicability to non-VAE architectures (e.g., pure Transformer generators) requires further investigation.
4. **Weaker performance on ForenSynths**: Detection of images produced by early GANs (e.g., ProGAN, CycleGAN) is comparatively weaker.

## Related Work & Insights

| Dimension | DDA (Ours) | DRCT (ICML'24) | AlignedForensics (ICLR'25) | B-Free |
|-----------|-----------|----------------|--------------------------|--------|
| Alignment | Pixel + frequency dual alignment | Diffusion reconstruction | Pure VAE reconstruction | Diffusion reconstruction + inpainting |
| Frequency handling | JPEG compression matching | None | None | None |
| Format alignment | ✓ | ✓ | ✗ | ✗ |
| Training data (real/syn) | 118K/118K | 118K/354K | 179K/179K | 51K/309K |
| Construction time | 5.9h | 64.6h | 8.73h | 258.79h |
| 11-benchmark average | 90.7% | 70.1% | 75.0% | N/A |

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The discovery of frequency-domain bias is an important insight; the DDA solution is simple yet precisely addresses the core issue.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 11 benchmarks, 9 competing methods, with comprehensive ablation, robustness, and visualization analyses.
- **Writing Quality**: ⭐⭐⭐⭐ — Motivation is clearly derived, figures are informative, and the argumentation chain is complete.
- **Value**: ⭐⭐⭐⭐ — Provides a systematic solution to dataset bias in AIGI detection with high practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Physics-Driven Spatiotemporal Modeling for AI-Generated Video Detection](physics-driven_spatiotemporal_modeling_for_ai-generated_video_detection.md)
- [\[NeurIPS 2025\] Epistemic Uncertainty for Generated Image Detection](epistemic_uncertainty_for_generated_image_detection.md)
- [\[NeurIPS 2025\] GeneMAN: Generalizable Single-Image 3D Human Reconstruction from Multi-Source Human Data](geneman_generalizable_single-image_3d_human_reconstruction_from_multi-source_hum.md)
- [\[NeurIPS 2025\] Is Artificial Intelligence Generated Image Detection a Solved Problem?](is_artificial_intelligence_generated_image_detection_a_solved_problem.md)
- [\[NeurIPS 2025\] UtilGen: Utility-Centric Generative Data Augmentation with Dual-Level Task Adaptation](utilgen_utility-centric_generative_data_augmentation_with_dual-level_task_adapta.md)

</div>

<!-- RELATED:END -->
