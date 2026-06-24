---
title: >-
  [Paper Note] AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer
description: >-
  [ECCV 2024][Model Compression][PTQ] This paper proposes AdaLog, an adaptive logarithmic base quantizer that addresses the power-law distribution of post-Softmax and post-GELU activations in ViTs by replacing fixed $\log_2$/$\log_{\sqrt{2}}$ quantizers with a searchable logarithmic base. Additionally, a Fast Progressive Combinatorial Search (FPCS) strategy is designed to efficiently determine quantization hyperparameters, which significantly outperforms existing ViT PTQ method…
tags:
  - "ECCV 2024"
  - "Model Compression"
  - "PTQ"
  - "Non-uniform Quantization"
  - "Adaptive Logarithmic Base"
  - "Hyperparameter Search"
  - "ViT Quantization"
date: 2026-05-08
content_hash: 73e024cdd906bae6
---

# AdaLog: Post-Training Quantization for Vision Transformers with Adaptive Logarithm Quantizer

**Conference**: ECCV 2024  
**arXiv**: [2407.12951](https://arxiv.org/abs/2407.12951)  
**Code**: [https://github.com/GoatWu/AdaLog](https://github.com/GoatWu/AdaLog)  
**Area**: Image Segmentation  
**Keywords**: PTQ, Non-uniform Quantization, Adaptive Logarithmic Base, Hyperparameter Search, ViT Quantization  

## TL;DR
This paper proposes AdaLog, an adaptive logarithmic base quantizer that addresses the power-law distribution of post-Softmax and post-GELU activations in ViTs by replacing fixed $\log_2$/$\log_{\sqrt{2}}$ quantizers with a searchable logarithmic base. Additionally, a Fast Progressive Combinatorial Search (FPCS) strategy is designed to efficiently determine quantization hyperparameters, which significantly outperforms existing ViT PTQ methods under ultra-low bit (3/4-bit) configurations.

## Background & Motivation
Vision Transformers (ViTs) have achieved outstanding performance in vision tasks such as classification, detection, and segmentation, but their huge computational and memory overheads hinder deployment on resource-constrained edge devices. Post-Training Quantization (PTQ) is highly efficient as it requires only a small amount of calibration data for model compression. However, two specific types of activations in ViTs—post-Softmax and post-GELU—exhibit **power-law distributions**, where values heavily concentrate near zero. Uniform quantizers introduce severe quantization errors in such scenarios.

Existing non-uniform quantization schemes, such as the $\log_2$ quantizer in FQ-ViT and the $\log_{\sqrt{2}}$ quantizer in RepQ-ViT, employ a fixed logarithmic base. This design is problematic: $\log_2$ leads to massive rounding errors for large values in 4-bit settings, while $\log_{\sqrt{2}}$ truncates most values to 0 under 3-bit settings. The optimal logarithmic base varies across different layers and bit-widths, making fixed-base approaches inadequate. Furthermore, the $\log_{\sqrt{2}}$ quantizer requires floating-point multiplications during dequantization, which is hardware-unfriendly.

## Core Problem
1. **Inflexible Logarithmic Base**: Fixed $\log_2$ or $\log_{\sqrt{2}}$ bases cannot adapt across different bit-widths and layers, causing severe accuracy degradation at ultra-low bits.
2. **Sparse Hyperparameter Search Space**: Due to the wide activation range of ViTs, traditional uniform sparse grid search easily trapped in local optima.
3. **Hardware Unfriendly**: The dequantization process of the $\log_{\sqrt{2}}$ quantizer involves element-wise floating-point multiplications, preventing fully integer-based inference.

## Method

### Overall Architecture
AdaLog is applied to the post-Softmax layer (MatMul2) and post-GELU layer (FC2) in standard ViT blocks. Other convolutional and linear layers (QKV, Proj, FC1) still utilize uniform quantizers, while employing FPCS to search for optimal hyperparameters. For each layer, 32 calibration images are utilized to capture input/output activations. Then, FPCS search is executed to determine the quantization hyperparameters (searching for the logarithmic base $b$ and scale factor $s$ in AdaLog layers, and scale factors/zero points in uniform layers), resulting in the final quantized model.

### Key Designs
1. **Adaptive Logarithmic Base Quantizer (AdaLog)**: This design generalizes the quantization formulation from a fixed base to an arbitrary base $b$: $A^{(\mathbb{Z})} = \text{clamp}(\lfloor -\log_b \frac{A}{s} \rceil, 0, 2^{bit}-1)$. The crucial technique is to approximate $\log_2 b \approx q/r$ with a rational number, simplifying dequantization into look-up table (LUT) and bit-shift operations: $\hat{A} = s \cdot (2^{-\tilde{A}^{(\mathbb{Z})}} \circ 2^{-\tilde{U}})$, where both $\tilde{A}^{(\mathbb{Z})}$ and $2^{-\tilde{U}}$ can be precomputed and stored in LUTs (with a table length of only $2^{bit}$). Inference requires only two table lookups and one bit shift, avoiding any floating-point computations and facilitating hardware deployment.

2. **Bias Reparameterization**: Post-GELU activations contain negative values (mostly concentrated in (-0.17, 0]), whereas logarithmic quantization requires non-negative inputs. The solution is to rewrite the linear layer of FC2 as $Y = W \cdot (X + 0.17) + (b - 0.17 \cdot \hat{W} \cdot \mathbf{1})$, making the input to AdaLog $X' = X + 0.17$ non-negative, while incorporating the offset into the bias to preserve output equivalence.

3. **Fast Progressive Combinatorial Search (FPCS)**: Inspired by beam search in NLP, this method first evaluates all candidate combinations on a coarse-grained grid (complexity $O(xy)$, $xy=n$) and selects the top-$k$ optimal combinations. It then performs fine-grained expansion around each optimal point (with $z$ candidates for each expansion, $kz=n$), iterating for $p$ steps. The total complexity is $O(pn)$, which is comparable to coordinate descent (alternating search), but avoids local optima by progressively refining the search space, achieving accuracy close to brute-force search (complexity $O(nm)$).

### Loss & Training
- The search objective is the **layer-wise MSE loss**: $\text{MSE}(\phi^{(l)}(X_l, a, b), O_l)$, which is the mean squared error between the quantized layer output and the full-precision output.
- Calibration data only requires 32 ImageNet images (classification) or 32 COCO images (detection/segmentation).
- Weights utilize channel-wise quantization, while activations utilize layer-wise quantization + scale reparameterization techniques.
- Hyperparameter settings: $r=37$ (prime number to ensure coprime), $n=128$, search steps $p=4$.

## Key Experimental Results

| Dataset | Model | bit(W/A) | Metric | AdaLog | RepQ-ViT | Gain |
|--------|------|----------|------|--------|----------|------|
| ImageNet | ViT-S | W4/A4 | Top-1 | 72.75% | 65.05% | +7.70% |
| ImageNet | ViT-B | W4/A4 | Top-1 | 79.68% | 68.48% | +11.20% |
| ImageNet | DeiT-B | W4/A4 | Top-1 | 78.03% | 75.61% | +2.42% |
| ImageNet | Swin-B | W4/A4 | Top-1 | 82.47% | 78.32% | +4.15% |
| ImageNet | ViT-S | W3/A3 | Top-1 | 13.88% | 0.10% | +13.78% |
| ImageNet | ViT-B | W3/A3 | Top-1 | 37.91% | 0.10% | +37.81% |
| COCO | Mask R-CNN Swin-T | W4/A4 | AP^box | 39.1% | 36.1% | +3.0% |
| COCO | Cascade Mask R-CNN Swin-S | W4/A4 | AP^box | 50.6% | 49.3% | +1.3% |

### Ablation Study
- **AdaLog contributes the most**: On ViT-S W4A4, adding only AdaLog (without FPCS) improves Top-1 accuracy from 62.20% to 72.01% (+9.81%), whereas adding only FPCS yields an improvement of less than 1%.
- **FPCS becomes more important at 3-bit**: On ViT-B W3A3, FPCS improves accuracy from 9.68% to 15.50% (+5.82%), as the search space is more critical at lower bits.
- **The combination of AdaLog+FPCS performs best**: The joint implementation of both outperforms individual components across all models and bit-widths.
- **Comparison of post-Softmax quantizers**: AdaLog maintains reasonable accuracy at 2-bit (e.g., ViT-S 70.36%), whereas both $\log_2$ and $\log_{\sqrt{2}}$ collapse to 0.10%.
- **Comparison of post-GELU quantizers**: AdaLog achieves the highest accuracy across all 7 models, while other quantizers exhibit unstable performance.
- **FPCS vs Brute-force search**: Accuracy is close (DeiT-T W3A3: 31.56% vs 32.04%), but the execution time is only 1/45 (4.1min vs 183min).
- **Incorporate into existing frameworks**: Adding AdaLog to the BRECQ framework improves the accuracy of LSQ training activation parameters from collapse (0.93%) to 62.50%.
- **Efficiency**: The memory overhead of the look-up table is extremely small; for 4-bit DeiT-T, it requires only ~3KB, which is less than 0.2% of the model size.

## Highlights & Insights
- The core idea of **making the logarithmic base learnable** is simple yet powerful, addressing all limitations of fixed-base quantizers with a single hyperparameter.
- The hardware-friendly dequantization scheme leveraging table lookups and bit-shifting is elegantly designed: via the rational approximation of $\log_2 b$, arbitrary-base power operations are converted into two table lookups and one bit-shifting operation.
- Bias reparameterization enables AdaLog to accommodate negative activation values (post-GELU), extending its applicability.
- Adapting the beam search concept from NLP to quantization hyperparameter search (FPCS) achieves a fair balance between search accuracy and efficiency.
- Under extreme ultra-low bits like 3-bit, where other methods completely collapse, AdaLog still achieves reasonable accuracy, demonstrating the core utility of non-uniform quantization in low-bit scenarios.

## Limitations & Future Work
- AdaLog is only employed in post-Softmax and post-GELU layers, while remaining layers still use uniform quantizers. Can adaptive non-uniform quantization be unified across all layers?
- The experiments only cover classification, detection, and segmentation, without validating performance on generative tasks (e.g., diffusion models) or larger-scale models (e.g., VLMs).
- The absolute accuracy under 3-bit remains low (only 13.88% on ViT-S), limiting its practicality, and it is almost unusable under 2-bit.
- The selection of Search Range and $r$ values for the logarithmic base lacks theoretical guidance and relies on empirical settings.
- The potential of integrating with QAT is not explored—is the non-uniform quantization function of AdaLog differentiable?

## Related Work & Insights
- **vs RepQ-ViT (ICCV 2023)**: RepQ-ViT uses a fixed $\log_{\sqrt{2}}$ base + scale reparameterization, but its dequantization requires element-wise floating-point multiplication, which is hardware-unfriendly. AdaLog features a searchable base and purely integer inference, outperforming RepQ-ViT by 5.13% on average under W4A4.
- **vs PTQ4ViT (ECCV 2022)**: PTQ4ViT utilizes a Twin-Uniform quantizer to handle power-law distributions, which is inherently a variant of uniform quantization. AdaLog's non-uniform logarithmic quantization matches power-law distributions better. PTQ4ViT almost completely collapses under 3-bit.
- **vs FQ-ViT (IJCAI 2022)**: A pioneering work proposing the $\log_2$ quantizer, but the limitations of its fixed base are fully exposed under low-bit settings.

## Related Work & Insights
- The progressive search concept of FPCS can be transferred to any quantization scenario requiring joint multi-hyperparameter search.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of an adaptive logarithmic base is intuitive yet previously unexplored; the table lookup design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Conducts comprehensive ablation studies across three tasks (classification/detection/segmentation) with 7 models under 3/4/6-bit settings.
- Writing Quality: ⭐⭐⭐⭐ Formulations and derivation are clear, figures are intuitive, but some sections have dense notations that require careful reading.
- Value: ⭐⭐⭐⭐ Consistently advances low-bit PTQ for Vision Transformers, with the FPCS search strategy holding strong reusability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] MetaAug: Meta-Data Augmentation for Post-Training Quantization](metaaug_meta-data_augmentation_for_post-training_quantization.md)
- [\[ECCV 2024\] PQ-SAM: Post-training Quantization for Segment Anything Model](pq-sam_post-training_quantization_for_segment_anything_model.md)
- [\[CVPR 2025\] FIMA-Q: Post-Training Quantization for Vision Transformers by Fisher Information Matrix Approximation](../../CVPR2025/model_compression/fima-q_post-training_quantization_for_vision_transformers_by_fisher_information_.md)
- [\[CVPR 2025\] Q-DiT: Accurate Post-Training Quantization for Diffusion Transformers](../../CVPR2025/model_compression/q-dit_accurate_post-training_quantization_for_diffusion_transformers.md)
- [\[ICLR 2026\] Faster Vision Transformers with Adaptive Patches](../../ICLR2026/model_compression/faster_vision_transformers_with_adaptive_patches.md)

</div>

<!-- RELATED:END -->
