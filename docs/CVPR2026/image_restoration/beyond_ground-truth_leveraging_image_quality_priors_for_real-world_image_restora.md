---
title: >-
  [Paper Note] Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration
description: >-
  [CVPR 2026][Image Restoration][Image Quality Prior] This paper proposes IQPIR, a framework that introduces image quality priors (IQP) derived from pretrained NR-IQA models as conditioning signals. Through three mechanism…
tags:
  - "CVPR 2026"
  - "Image Restoration"
  - "Image Quality Prior"
  - "Dual Codebook"
  - "NR-IQA"
  - "Quality-Conditioned"
date: 2026-05-08
content_hash: 08aa748c3c0d5804
---

# Beyond Ground-Truth: Leveraging Image Quality Priors for Real-World Image Restoration

**Conference**: CVPR 2026
**arXiv**: [2603.29773](https://arxiv.org/abs/2603.29773)
**Code**: [https://github.com/fengyang1399-pixel/IQPIR](https://github.com/fengyang1399-pixel/IQPIR)
**Area**: Image Restoration
**Keywords**: Image Restoration, Image Quality Prior, Dual Codebook, NR-IQA, Quality-Conditioned

## TL;DR
This paper proposes IQPIR, a framework that introduces image quality priors (IQP) derived from pretrained NR-IQA models as conditioning signals. Through three mechanisms—quality-conditioned Transformer, dual Codebook architecture, and quality optimization in discrete representation space—IQPIR guides the restoration process toward maximal perceptual quality, achieving state-of-the-art performance on blind face restoration and related tasks.

## Background & Motivation

**Background**: Real-world image restoration aims to recover high-quality images from inputs suffering from complex degradations. Codebook-based methods reformulate restoration as a code prediction problem in discrete representation space, effectively reducing reconstruction ambiguity.

**Limitations of Prior Work**: All existing methods implicitly assume that ground-truth (GT) images are perfect and serve as the sole supervision source. However, as shown in Figure 1, the perceptual quality of GT datasets (e.g., FFHQ) is inconsistent—most GT quality scores fall between 5 and 8, with very few reaching 9. As a result, models converge to the **average quality level** of the GT rather than the highest achievable quality.

**Key Challenge**: (1) Training exclusively on the highest-quality GT leads to insufficient data diversity, causing artifacts and degraded features; (2) Training on all GT images pulls the model toward average quality.

**Key Insight**: GT images of different quality levels serve distinct roles—HQ+ GT excels at fine structural control, while average-quality GT is better suited for recovering large blurred regions.

**Core Idea**: NR-IQA scores are injected as conditioning signals into the restoration model; setting the score to its maximum value at inference time guides the network to produce the highest-quality output. A dual Codebook learns general structures and HQ+-specific details separately.

## Method

### Overall Architecture
The framework consists of two stages: (1) **Codebook learning**—a dual Codebook architecture that learns general and HQ+-specific features separately; (2) **Code prediction**—a quality-conditioned Transformer predicts dual code sequences with a quality optimization loss.

### Key Designs

1. **Dual Codebook Architecture**:

    - **Function**: Decouple general structural features from high-quality-specific fine details.
    - **Mechanism**: The Common Codebook is trained on all GT images; the HQ+ Codebook participates in quantization only when the GT quality score satisfies $S > S_{thr}$. The fused representation is $Z_q = Z_q^1 + \alpha Z_q^2$ (or simply $Z_q^1$ when $S \leq S_{thr}$). The decoder reconstructs images from the fused representation.
    - **Design Motivation**: Fine visual details in HQ GT (e.g., hair strand tips) require a dedicated Codebook for encoding, while the Common Codebook ensures broad degradation recovery capability.

2. **Quality-Conditioned Transformer**:

    - **Function**: Predict code sequences conditioned on quality scores.
    - **Mechanism**: An NR-IQA model estimates the GT quality score $S$, which is embedded as a vector $\mathbf{s} \in \mathbb{R}^{h \times w \times c}$ and added directly to the LQ features: $\hat{Z}_l = Z_l + \mathbf{s}$. The Transformer takes $\hat{Z}_l$ as input and predicts two code sequences $\mathbf{c}_1, \mathbf{c}_2$ to query the respective Codebooks.
    - **At inference**: $S$ is set to its maximum value, guiding the network to produce restorations of the highest perceptual quality.
    - **Design Motivation**: Analogous to class-conditional generation, the model learns quality-image correspondences, enabling **controllable quality restoration**.

3. **Quality Optimization in Discrete Representation Space**:

    - **Function**: Perform quality optimization in discrete representation space to avoid over-optimization in continuous space.
    - **Mechanism**: An NR-IQA model computes the quality loss $\mathcal{L}_{quality}$ on the restored output.
    - **Design Motivation**: Directly optimizing IQA scores in continuous space is prone to over-optimization and artifacts; the discrete Codebook naturally constrains the output space, inherently mitigating this issue.

4. **Quality Prior Integration**:

    - Scores from multiple NR-IQA models are averaged as $S = \frac{1}{n}\sum s_i$, reducing bias from any single model.

### Loss & Training
Codebook stage: reconstruction loss + quantization commitment loss + perceptual loss. Code prediction stage: cross-entropy for code prediction + quality optimization loss.

## Key Experimental Results

### Main Results (Blind Face Restoration, LFW-Test)

| Method | TOPIQ-G↑ | Musiq-G↑ | Q-Align↑ | CLIP-IQA↑ |
|------|----------|----------|----------|-----------|
| CodeFormer | 0.809 | 0.832 | 4.31 | 0.697 |
| DAEFR | 0.814 | 0.827 | 4.33 | 0.696 |
| WaveFace | 0.786 | 0.799 | 4.43 | 0.788 |
| Interlcm | 0.831 | 0.834 | 4.55 | 0.721 |
| **IQPIR (Ours)** | **0.861** | **0.878** | **4.67** | **0.790** |

IQPIR also achieves comprehensive improvements on WebPhoto-Test and WIDER-Test.

### Ablation Study

| Configuration | Key Metric | Note |
|------|---------|------|
| w/o quality conditioning | Degraded | Demonstrates necessity of IQP conditioning |
| Single Codebook | Degraded | HQ+ Codebook is critical for fine details |
| Quality optimization in continuous space | Over-optimized | Highlights advantage of discrete space |
| Single IQA model | Slightly degraded | Multi-model ensemble is more robust |

### Key Findings
- **IQP is a general quality-enhancement strategy**: Applying the proposed quality conditioning to DifFace (DifFace+) also yields significant quality improvements, demonstrating plug-and-play applicability.
- The HQ+ Codebook primarily improves fine-grained details such as hair tips and skin texture.
- Setting the quality score to its maximum value at inference time leads to perceptual quality that substantially exceeds the average GT level.

## Highlights & Insights
- **Challenging the perfect GT assumption**: This is the first work to systematically reveal the impact of inconsistent GT quality on restoration models, proposing a "beyond GT" restoration paradigm.
- **Plug-and-play quality conditioning**: IQP can be inserted into any restoration architecture as an independent module without structural modifications.
- **Discrete-space quality optimization**: The work elegantly leverages the discrete nature of VQ-VAE to avoid the over-optimization pitfalls of IQA reward optimization in continuous space.

## Limitations & Future Work
- NR-IQA models themselves carry biases (some may favor particular styles); ensemble integration mitigates but does not fully eliminate this issue.
- The threshold $S_{thr}$ and weight $\alpha$ require manual tuning.
- When GT quality is extremely low, the HQ+ Codebook has limited information to learn from.
- Extending quality priors to video restoration and 3D restoration is a promising direction.

## Related Work & Insights
- **vs. CodeFormer/DAEFR**: These methods assume GT perfection and rely on direct supervision. This work breaks that constraint by introducing a quality dimension.
- **vs. GAN/diffusion-based restoration**: Generative priors are powerful but may introduce hallucinations; Codebook priors combined with quality priors offer greater controllability.
- **vs. NR-IQA research**: This work repositions IQA from an evaluation tool to a training signal, substantially broadening the application scope of IQA.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The idea of leveraging quality priors for restoration is original; the system design integrating dual Codebook, quality conditioning, and discrete optimization is coherent and complete.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Multi-dataset, multi-metric evaluation with thorough ablations; plug-and-play validation is convincing.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation figure (GT quality distribution) is intuitive and compelling.
- **Value**: ⭐⭐⭐⭐⭐ The general quality-guided strategy has broad implications for the image restoration community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond the Ground Truth: Enhanced Supervision for Image Restoration](beyond_the_ground_truth_enhanced_supervision_for_image_restoration.md)
- [\[CVPR 2026\] Toward Real-world Infrared Image Super-Resolution: A Unified Autoregressive Framework and Benchmark Dataset](real_iisr_infrared_image_super_resolution_autoregressive.md)
- [\[CVPR 2026\] Disentangled Textual Priors for Diffusion-based Image Super-Resolution](disentangled_textual_priors_for_diffusion-based_image_super-resolution.md)
- [\[CVPR 2026\] PhaSR: Generalized Image Shadow Removal with Physically Aligned Priors](phasr_generalized_image_shadow_removal_with_physically_aligned_priors.md)
- [\[CVPR 2026\] TM-BSN: Triangular-Masked Blind-Spot Network for Real-World Self-Supervised Image Denoising](tm-bsn_triangular-masked_blind-spot_network_for_real-world_self-supervised_image.md)

</div>

<!-- RELATED:END -->
