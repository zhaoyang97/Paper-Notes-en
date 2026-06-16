---
title: >-
  [Paper Note] RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression
description: >-
  [CVPR 2026][Model Compression][Paper Note] RDVQ achieves end-to-end joint rate-distortion optimization for VQ-based image compression for the first time by introducing a differentiable relaxation of codebook distributions, obtaining superior or competitive perceptual quality at ultra-low bitrates with less than 20% of the parameters.
tags:
  - CVPR 2026
  - Model Compression
date: 2026-05-08
content_hash: 3640983c8d87f69c
---
# RDVQ: Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression

**Conference**: CVPR 2026 Oral  
**arXiv**: [2604.10546](https://arxiv.org/abs/2604.10546)  
**Code**: [https://github.com/CVL-UESTC/RDVQ](https://github.com/CVL-UESTC/RDVQ)  
**Area**: Image Compression/Restoration  
**Keywords**: Vector Quantization, Rate-Distortion Optimization, Generative Image Compression, Entropy Model, Differentiable Relaxation

## TL;DR
RDVQ achieves end-to-end joint rate-distortion optimization for VQ-based image compression for the first time by introducing a differentiable relaxation of codebook distributions, obtaining superior or competitive perceptual quality at ultra-low bitrates with less than 20% of the parameters.

## Background & Motivation

**Background**: Learned image compression primarily employs scalar quantization (SQ), where differentiable approximations (e.g., noise addition/STE) allow gradients to backpropagate to the encoder for end-to-end rate-distortion optimization. Vector quantization (VQ) preserves better structural information and perceptual quality, making it particularly suitable for ultra-low bitrates.

**Limitations of Prior Work**: The discrete nearest-neighbor assignment in VQ obstructs gradient propagation from the rate loss to the encoder. The implicit prior distribution induced by the encoder cannot be directly optimized by the rate objective, leading to a fundamental decoupling between representation learning and the entropy model.

**Key Challenge**: While VQ offers advantages in reconstruction quality, it lacks the capability for end-to-end joint rate-distortion optimization like SQ, relying instead on heuristic methods such as adjusting codebook size or selective transmission to control the bitrate.

**Goal**: Restore the differentiable gradient path from the rate objective to the encoder in VQ compression to achieve true end-to-end rate-distortion optimization.

**Key Insight**: Replace the hard nearest-neighbor assignment with a distance-aware soft distribution, used exclusively in the rate estimation branch, while maintaining standard hard quantization for reconstruction.

**Core Idea**: During training, use a softmax-relaxed codebook distribution to estimate the rate, allowing rate gradients to flow to the encoder; during inference, switch back to standard hard VQ to maintain compatibility.

## Method

### Overall Architecture
The core challenge RDVQ addresses is that the "codeword selection" in VQ compression is a discrete nearest-neighbor choice, which severs the gradient flow from the rate loss to the encoder, forcing bitrate control to rely on heuristics like codebook sizing. The mechanism involves establishing a "shadow" gradient path: reconstruction proceeds with hard quantization as usual, but rate estimation employs a differentiable soft distribution, allowing rate loss to backpropagate to the encoder via this soft path. Specifically, an input image is processed by the analysis transform $g_a$ to extract multi-scale features, which are then flattened into a sequence. The VQ module simultaneously produces three outputs: hard quantized embeddings for the synthesis transform $g_s$ reconstruction, discrete indices for encoding, and a relaxed distribution used for rate calculation only during training. The entropy model is a Masked Transformer that performs autoregressive probability prediction on these indices. During inference, rate adjustment is achieved by transmitting only a prefix of the indices, allowing the entropy model to autoregressively complete the rest without retraining.

```mermaid
graph TD
    A["Input Image"] --> B["Analysis Transform g_a<br/>Extract multi-scale features and flatten into sequence"]
    B --> C["VQ Module"]
    C -->|Hard Quantized Embeddings| D["Synthesis Transform g_s Reconstructs Image"]
    C -->|Soft Distribution (Training)| E["Differentiable Soft Relaxation<br/>Temperature Softmax for Rate Estimation"]
    C -->|Discrete Indices| F["Dependency-aware Autoregressive Entropy Model<br/>Masked Transformer, Coarse-to-Fine Sorting"]
    F --> E
    E -.->|Rate Loss Gradient Backpropagation| B
    F -->|Inference| G["Test-time Rate Adjustment<br/>Transmit index prefix, autoregressive completion at decoder"]
    G --> H["Bitstream"]
```

### Key Designs

**1. Differentiable Soft Relaxation: Replacing "codeword selection" with temperature softmax to open a gradient path for rate loss to the encoder**

Standard VQ's nearest-neighbor assignment is an argmax operation where gradients stop, preventing rate loss from reaching the encoder and decoupling representation learning from the entropy model. RDVQ calculates a distance $d_{b,l,k}$ between the encoder output and each codeword, then converts it into a soft distribution using temperature-scaled softmax: $p_{\text{soft}}(b,l,k) = \text{softmax}_k(-d_{b,l,k}/\tau)$—where closer codewords receive higher probabilities. During training, the rate objective calculates cross-entropy using this continuous distribution, allowing gradients to flow back to the encoder. The reconstruction branch continues to use standard hard quantization. This "dual-path" design ensures no inconsistency during inference. Removing this relaxation causes the FID to spike from 19.96 to 86.93 in ablation studies, proving that end-to-end optimization is impossible without this path.

**2. Dependency-aware Autoregressive Entropy Model: Hierarchical dependencies for rate estimation**

Codebook indices are not independent; in multi-scale structures, coarse scales determine the expected features of fine scales. Standard raster scanning misses this structure. RDVQ concatenates multi-scale features into a unified sequence and constructs a dependency-aware ordering vector $o$ to generate a mask $M = (o > o^\top)$ for the attention mechanism. Coarse-scale tokens are placed earlier in the sequence, meaning fine-scale tokens can only attend to coarser ones, completing a "coarse-to-fine" autoregressive factorization in a single parallel forward pass. This ordering aligns with natural hierarchical dependencies, leading to tighter rate estimation.

**3. Test-time Rate Adjustment: Sliding bitrate control via index prefixing and autoregressive completion**

RDVQ enables flexible bitrate adjustment on a single model without retraining. Since the latent space is trained to be highly predictable via joint rate-distortion optimization, only a prefix of the index sequence needs to be transmitted. The decoder can then use the same autoregressive entropy model to complete the remaining indices. Shorter prefixes result in lower bitrates with smooth quality degradation. This allows sliding rate control within a range (approx. 0.02–0.32 bpp) without modifying model weights.

### Loss & Training
A three-stage training strategy is employed: (1) Pre-training the autoencoder and codebook (reconstruction loss); (2) Pre-training the entropy model (rate objective); (3) Joint fine-tuning of the entire model (rate + distortion), followed by high-resolution adaptation. The loss function includes GAN loss, LPIPS perceptual loss, and the relaxed cross-entropy rate loss.

## Key Experimental Results

### Main Results

| Dataset | Metric | RDVQ | RDEIC | Rate Savings |
|--------|------|------|-------|---------|
| DIV2K-val | DISTS | Best | Second | -75.71% |
| DIV2K-val | LPIPS | Best | Second | -37.63% |
| Kodak | DISTS | SOTA | - | - |
| CLIC2020 | CLIPIQA | SOTA | - | - |

### Ablation Study

| Configuration | bpp | DISTS | LPIPS | FID |
|------|-----|-------|-------|-----|
| RDVQ (full) | 0.0247 | 0.1005 | 0.2321 | 19.96 |
| w/o Relaxation | 0.0464 | 0.2147 | 0.5031 | 86.93 |
| K-means VQ | 0.0247 | 0.1253 | 0.2831 | 28.08 |

### Key Findings
- Performance drops drastically without differentiable relaxation; even at higher bitrates, it cannot match the full model, proving relaxation is central to end-to-end RD optimization.
- K-means bitrate control shows significantly worse quality than RDVQ at equivalent bitrates, as heuristic methods fail to eliminate redundancy in index distributions.
- As the bitrate decreases, encoder features become smoother and codebook utilization more concentrated, demonstrating that the model automatically learns compression strategies.

## Highlights & Insights
- **Elegant Separation of Relaxation**: Relaxation is used only in the rate estimation branch during training, while the reconstruction path remains hard quantized. This "dual-path" design solves the gradient problem while maintaining deployment compatibility.
- **Unified Vision Tokenization and Compression**: Existing VQ tokenizers can be converted into compression models by introducing entropy constraints; conversely, compression techniques can improve tokenizer efficiency.

## Limitations & Future Work
- Test-time rate adjustment has a limited range (0.02-0.32 bpp), with quality degrading significantly outside this window.
- While having significantly fewer parameters than baselines, 251.9M parameters is still not considered lightweight.
- Future work could explore applying this framework to entropy-aware training of visual tokenizers.

## Related Work & Insights
- **vs OSCAR/RDEIC**: Methods based on diffusion or large model priors have massive parameter counts; RDVQ is trained from scratch with less than 20% of their parameters.
- **vs DLF**: Dual-branch SQ+VQ hybrid methods essentially still fail to perform rate-distortion optimization on the VQ branch.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First implementation of end-to-end RD optimization for VQ, clear theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multiple metrics across three datasets, comprehensive ablation and analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Precise problem definition and clear derivation.
- Value: ⭐⭐⭐⭐⭐ Highly significant for both VQ compression and visual tokenization.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Differentiable Vector Quantization for Rate-Distortion Optimization of Generative Image Compression](differentiable_vector_quantization_for_rate-distortion_optimization_of_generativ.md)
- [\[CVPR 2026\] ProGIC: Progressive and Lightweight Generative Image Compression with Residual Vector Quantization](progic_progressive_and_lightweight_generative_image_compression_with_residual_ve.md)
- [\[CVPR 2026\] CADC: Content Adaptive Diffusion-Based Generative Image Compression](cadc_content_adaptive_diffusion-based_generative_image_compression.md)
- [\[AAAI 2026\] Reinforced Rate Control for Neural Video Compression via Inter-Frame Rate-Distortion Awareness](../../AAAI2026/model_compression/reinforced_rate_control_for_neural_video_compression_via_inter-frame_rate-distor.md)
- [\[CVPR 2026\] Parallax to Align Them All: An OmniParallax Attention Mechanism for Distributed Multi-View Image Compression](parallax_to_align_them_all_an_omniparallax_attention_mechanism_for_distributed_m.md)

</div>

<!-- RELATED:END -->
