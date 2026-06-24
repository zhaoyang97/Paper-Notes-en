---
title: >-
  [Paper Note] Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient
description: >-
  [CVPR 2025][Image Generation][Visual Autoregressive (VAR)] This work proposes CoDe (Collaborative Decoding), which decomposes the multi-scale inference of VAR into a collaborative workflow: a large model drafts (low-frequency small scales) and a small model refines (high-frequency large scales). It achieves a 1.7× speedup and a 50% memory reduction, with the FID only slightly increasing from 1.95 to 1.98.
tags:
  - "CVPR 2025"
  - "Image Generation"
  - "Visual Autoregressive (VAR)"
  - "Next-scale Prediction"
  - "Collaborative Decoding"
  - "Inference Acceleration"
  - "KV Cache Optimization"
date: 2026-05-08
content_hash: c5f15fcdd03972f7
---

# Collaborative Decoding Makes Visual Auto-Regressive Modeling Efficient

**Conference**: CVPR 2025  
**arXiv**: [2411.17787](https://arxiv.org/abs/2411.17787)  
**Code**: [https://github.com/czg1225/CoDe](https://github.com/czg1225/CoDe)  
**Area**: Diffusion Models / Model Compression  
**Keywords**: Visual Autoregressive (VAR), Next-scale Prediction, Collaborative Decoding, Inference Acceleration, KV Cache Optimization

## TL;DR
This work proposes CoDe (Collaborative Decoding), which decomposes the multi-scale inference of VAR into a collaborative workflow: a large model drafts (low-frequency small scales) and a small model refines (high-frequency large scales). It achieves a 1.7× speedup and a 50% memory reduction, with the FID only slightly increasing from 1.95 to 1.98.

## Background & Motivation

**Background**: Visual Auto-Regressive (VAR) models replace traditional GPT-style next-token prediction with next-scale prediction, parallelly decoding and generating $256 \times 256$ images in just 10 steps, achieving breakthrough performance in both quality and speed. However, the progressive zooming strategy of VAR results in a total token sequence length of up to 680 (which is 2.7 times that of traditional AR), and KV cache consumption accounts for approximately 80% of the total GPU memory (57GB out of 70GB is used for the KV cache).

**Limitations of Prior Work**: The sequence length is heavily concentrated in the last few scales—with the final scale alone accounting for 38% of the tokens. Self-attention computation scales quadratically with sequence length. Although using a smaller VAR model is faster, it suffers from a significant drop in quality (the FID of VAR-d20 increases from 1.95 to 2.61). Existing AR acceleration methods (such as speculative decoding) target the next-token prediction paradigm and are inapplicable to the next-scale prediction of VAR.

**Key Challenge**: VAR has the largest computational overhead at large scales (high-resolution token maps), yet the parameter requirement is minimal—experiments show that 2B and 0.3B models perform almost identically on the final scale. Meanwhile, the generation patterns of small and large scales are completely different (low-frequency vs. high-frequency), and simultaneously learning both in a single model leads to optimization interference.

**Goal**: How to leverage both the parameter redundancy at large scales of VAR and the mutual exclusivity of cross-scale generation patterns to significantly enhance inference efficiency.

**Key Insight**: Since large scales do not require large models and small scales do not need small models, the large model can be designated to generate only small scales (drafting), while the small model is designated to handle large scales (refining). Once each model focuses on its own scale range, specialized fine-tuning can be applied to eliminate cross-scale interference.

**Core Idea**: Decompose the 10-step multi-scale inference of VAR into the first $N$ steps handled by the large model (low-frequency draft) and the subsequent $K-N$ steps handled by the small model (high-frequency refinement), coupled with specialized fine-tuning to eliminate training interference.

## Method

### Overall Architecture
Pre-trained VAR-d30 (2B) is used as the drafter, and VAR-d16 (0.3B) as the refiner. The drafter generates the token map $R_L$ for the first $N$ scales (low-frequency global structure). After releasing the KV cache, the refiner uses $R_L$ as a prefix to generate the token map $R_H$ for the remaining $K-N$ scales (high-frequency details). Both models undergo specialized fine-tuning on their respective scales. Finally, the residual quantization function and multi-scale VQVAE decoder are utilized to reconstruct the image.

### Key Designs

1. **Collaborative Decoding with Large & Small Models**

    - **Function**: Replace the large model with a small model for processing computationally intensive large scales while maintaining generation quality.
    - **Mechanism**: The drafter (2B) is responsible for the first $N$ steps (computationally sparse but high parameter demand), while the refiner (0.3B) is responsible for the remaining $K-N$ steps (computationally intensive but low parameter demand). Supporting evidence shows that the final 3 steps account for 64% of the total inference time, but the small model performs only slightly worse than the large model at these steps. The refiner is 4.6× faster than the drafter at the final step. After the drafter completes, it releases its KV cache, and the refiner only needs to maintain its own smaller KV cache, drastically reducing GPU memory.
    - **Design Motivation**: Fourier analysis confirms that the first 3 scales mainly generate low-frequency components, while the last 3 scales generate high-frequency components, which demand entirely different capabilities.

2. **Specialized Fine-Tuning**

    - **Function**: Eliminate cross-scale training interference and make each model more precise on its designated scales.
    - **Mechanism**: The drafter is fine-tuned on the first $N$ scales using CSE loss (for 5% of the original training epochs, with $lr=1e-6$). The refiner is trained via KL divergence knowledge distillation from the large model, with a dynamic weight $\lambda_{ep}$ linearly decaying from 1 to 0 to gradually shift the learning focus from all tokens to refinement tokens (for 25% of training epochs, with $lr=1e-5$). Experiments demonstrate that if fine-tuning the refinement scales accidentally harms the global modeling capability, it causes the FID to spike from 3.30 to 21.93.
    - **Design Motivation**: During pre-training, a single model learning both low-frequency and high-frequency components simultaneously suffers from mutual interference. Specialized fine-tuning allows each model to purely optimize its own scales.

3. **Flexible Speed-Quality Trade-Off**

    - **Function**: Flexibly balance speed and quality by adjusting the drafting steps $N$.
    - **Mechanism**: A smaller $N$ implies that the drafter does less and the refiner does more, which yields larger speedups but a slight degradation in quality. Selecting $N=9$ (1.2× speedup) yields an FID of 1.94, which is even better than the original model; $N=8$ (1.7× speedup) achieves an FID of 1.98; $N=6$ (2.9× speedup) achieves an FID of 2.27. Even in a training-free setting (without fine-tuning), CoDe outperforms smaller VAR models under comparable speedup ratios.
    - **Design Motivation**: Different applications have varied requirements for speed and quality, and a flexible trade-off enhances the practical utility of the method.

### Loss & Training
Drafter: Handled by CSE loss to align the generation distribution with ground-truth labels. Refiner: Uses KL divergence distillation to learn from the large model, with dynamic weights that gradually focus on refinement scales. Training is conducted on 4 × NVIDIA L20 GPUs with a batch size of 1024 (using gradient accumulation) and the AdamW optimizer.

## Key Experimental Results

### Main Results (Class-Conditional Generation on ImageNet-256)

| Method | Steps | Speedup↑ | Memory↓ | FID↓ | IS↑ |
|------|------|------|------|------|-----|
| VAR-d30 (Original) | 10 | 1.0× | 39.2GB | 1.95 | 301 |
| VAR-d24 | 10 | 1.7× | 25.1GB | 2.11 | 311 |
| **CoDe N=8** | 8+2 | **1.7×** | **21.0GB** | **1.98** | 302 |
| VAR-d20 | 10 | 2.8× | 17.8GB | 2.61 | 301 |
| **CoDe N=6** | 6+4 | **2.9×** | **19.9GB** | **2.27** | 297 |
| DiT-XL/2 | 50 | 0.2× | 11.4GB | 2.26 | 239 |
| LlamaGen-XXL | 384 | <0.1× | 42.6GB | 2.34 | 254 |

### Ablation Study

| Configuration | N=6 FID | N=8 FID | Description |
|------|---------|---------|------|
| Training-free CoDe | 2.42 | 2.10 | Direct collaboration using pre-trained models |
| **+ Specialized Fine-Tuning** | **2.27** | **1.98** | Fine-tuning eliminates interference, yielding significant improvement |

### Key Findings
- The FID of CoDe with $N=9$ (1.94) is even better than the original VAR-d30 (1.95), proving that specialized fine-tuning eliminates cross-scale interference and results in a more precise model.
- Under comparable speedup ratios, CoDe significantly outperforms directly using a smaller VAR model (e.g., $N=8$ achieves an FID of 1.98 vs. VAR-d24 with an FID of 2.11).
- The KV cache memory decreases from 28.7GB to 4.1GB (batch size = 64), which is the primary source of GPU memory savings.
- At the last scale, the refiner is 4.6× faster than the drafter, validating the observation of parameter redundancy at large scales.
- CoDe is currently the fastest method to achieve an FID of $<2$.

## Highlights & Insights
- **The observation that "large scales do not require large models"** is simple yet powerful: replacement experiments quantitatively prove that the 2B and 0.3B models show comparable performance at the largest scale, providing a solid theoretical foundation for collaborative decoding.
- **The scale mutual-exclusivity of low-frequency and high-frequency components** is double-validated via Fourier analysis and perturbation experiments, explaining why learning all scales in a single model is sub-optimal.
- **The method is extremely simple and practical**: it requires no changes to the architecture or training pipeline, only switching models during inference coupled with lightweight fine-tuning.

## Limitations & Future Work
- It requires maintaining two models (2B + 0.3B), which increases the total model footprint, although they do not occupy GPU memory at the same time during inference.
- Specialized fine-tuning still incurs training costs (5% epochs for the drafter + 25% epochs for the refiner).
- The method is only validated on ImageNet-256 class-conditional generation, while more complex tasks like text-to-image generation remain untested.
- Sampling hyperparameters were modified (top-k decreased from 900 to 600 + temperature set to 1.1), which might affect fair comparison.

## Related Work & Insights
- **vs. Directly using small VAR**: CoDe with $N=8$ (1.98 FID, 1.7× speedup) significantly outperforms VAR-d24 (2.11 FID, 1.7× speedup) because the initial steps are still safeguarded by the large model.
- **vs. DiT-XL/2**: CoDe with $N=6$ achieves an FID of 2.27 while being 15× faster, demonstrating the huge efficiency advantage of the VAR paradigm.
- **vs. Speculative Decoding (LANTERN, SJD)**: These methods target next-token prediction and are incompatible with next-scale prediction in VAR. CoDe is an acceleration scheme custom-designed for VAR.

## Rating
- Novelty: ⭐⭐⭐⭐ A simple, observation-driven method with valuable core insights (large-scale parameter redundancy + scale mutual exclusivity).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extremely detailed efficiency analysis (latency/VRAM/throughput), with comprehensive ablation and qualitative results.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logical flow from the two observations to the method design, and excellent figures/tables.
- Value: ⭐⭐⭐⭐ The fastest method to achieve an FID of $<2$, offering direct practical value for VAR deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] HMAR: Efficient Hierarchical Masked Auto-Regressive Image Generation](hmar_efficient_hierarchical_masked_auto-regressive_image_generation.md)
- [\[CVPR 2025\] MMAR: Towards Lossless Multi-Modal Auto-Regressive Probabilistic Modeling](mmar_towards_lossless_multi-modal_auto-regressive_probabilistic_modeling.md)
- [\[NeurIPS 2025\] Distilled Decoding 2: One-step Sampling of Image Auto-regressive Models with Conditional Score Distillation](../../NeurIPS2025/image_generation/distilled_decoding_2_onestep_sampling_of_image_autoregressiv.md)
- [\[CVPR 2025\] Visual-ERM: Reward Modeling for Visual Equivalence](visual-erm_reward_modeling_for_visual_equivalence.md)
- [\[CVPR 2025\] Multi-party Collaborative Attention Control for Image Customization](multi-party_collaborative_attention_control_for_image_customization.md)

</div>

<!-- RELATED:END -->
