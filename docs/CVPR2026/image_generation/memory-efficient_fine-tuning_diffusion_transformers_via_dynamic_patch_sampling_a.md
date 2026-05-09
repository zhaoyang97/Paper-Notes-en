---
title: >-
  [Paper Note] Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping
description: >-
  [CVPR 2026][Image Generation][Diffusion Transformer] This paper proposes DiT-BlockSkip, a framework that reduces LoRA fine-tuning memory on FLUX by approximately 50% while maintaining comparable personalized generation quality. It achieves this through two components: timestep-aware dynamic patch sampling (low-resolution training with dynamically adjusted crop sizes) and a block skipping strategy that identifies critical blocks via cross-attention analysis and precomputes residual features for skipped blocks.
tags:
  - CVPR 2026
  - Image Generation
  - Diffusion Transformer
  - Efficient Fine-Tuning
  - Dynamic Patch Sampling
  - Block Skipping
  - Personalized Generation
date: 2026-05-08
content_hash: 81c84fb9d0aa20d5
---

# Memory-Efficient Fine-Tuning Diffusion Transformers via Dynamic Patch Sampling and Block Skipping

**Conference**: CVPR 2026
**arXiv**: [2603.20755](https://arxiv.org/abs/2603.20755)
**Code**: None
**Area**: Diffusion Models / Efficient Fine-Tuning
**Keywords**: Diffusion Transformer, Efficient Fine-Tuning, Dynamic Patch Sampling, Block Skipping, Personalized Generation

## TL;DR

This paper proposes DiT-BlockSkip, a framework that reduces LoRA fine-tuning memory on FLUX by approximately 50% while maintaining comparable personalized generation quality. It achieves this through two components: timestep-aware dynamic patch sampling (low-resolution training with dynamically adjusted crop sizes) and a block skipping strategy that identifies critical blocks via cross-attention analysis and precomputes residual features for skipped blocks.

## Background & Motivation

1. **Background**: Diffusion Transformer (DiT)-based text-to-image models (e.g., FLUX, SANA) have substantially improved image generation quality. Personalized fine-tuning typically employs PEFT methods such as LoRA to adapt models on small sets of reference images.

2. **Limitations of Prior Work**: (a) DiT models are extremely large (FLUX has 19 double-stream and 38 single-stream blocks), and even LoRA requires full forward and backward passes, resulting in substantial memory overhead (~30 GiB for FLUX LoRA at 512×512); (b) quantization methods degrade model precision; (c) gradient-free methods (e.g., ZOODiP) suffer from unstable optimization and require up to 30,000 steps to converge.

3. **Key Challenge**: The depth and capacity of DiT architectures produce activation memory during training that far exceeds that of U-Net-based models. Existing memory-efficient methods (e.g., HollowedNet) are designed for U-Net and cannot be directly transferred to DiT.

4. **Goal**: Achieve substantial memory reduction on DiT-based models while preserving personalization quality, with the ultimate goal of enabling on-device deployment.

5. **Key Insight**: (a) Different timesteps in the diffusion process encode different types of information—high-noise steps capture global structure while low-noise steps encode fine-grained details; (b) not all DiT blocks contribute equally to personalization—intermediate blocks are most critical.

6. **Core Idea**: Dynamic cropping combined with low-resolution training reduces forward/backward memory; selective skipping of non-critical blocks with precomputed residual features reduces parameter and optimizer state memory.

## Method

### Overall Architecture

DiT-BlockSkip consists of two orthogonal components: (1) **Dynamic Patch Sampling**, which dynamically adjusts the crop region size according to the diffusion timestep and then uniformly resizes the cropped region to a fixed low resolution before feeding it to the model; and (2) **Block Skipping**, which identifies critical blocks through cross-attention masking experiments, skips non-critical blocks (a number of leading and trailing blocks), and precomputes residual features for skipped blocks to preserve information integrity. The two components are used in combination, and end-to-end LoRA training is performed only on the critical intermediate blocks.

### Key Designs

1. **Timestep-Aware Dynamic Patch Sampling**

   - **Function**: Reduces training resolution while preserving the model's ability to learn both global structure and local detail.
   - **Mechanism**: Given diffusion timestep $t$, the crop size is determined by $f(s_{min}, s_{max}, t) = s_{min} + \frac{t}{T} \cdot (s_{max} - s_{min})$. At high timesteps (high noise), a larger region is cropped to capture global structure; at low timesteps, a smaller region is cropped to focus on fine details. The cropped region is uniformly resized to $s_{min} \times s_{min}$ (e.g., 256×256), with patch sizes discretized according to the VAE downsampling factor (16).
   - **Design Motivation**: Naively reducing resolution discards fine details, while fixing a small crop region loses global structure. Dynamic adjustment of the crop range allows the model to observe information at different scales across timesteps, approximating the representational capacity of high-resolution training.

2. **Cross-Attention Masking-Based Block Selection**

   - **Function**: Identifies the Transformer blocks in DiT that are most critical for personalization.
   - **Mechanism**: On a LoRA fine-tuned model, the cross-attention (image query to text key) of 14 consecutive blocks at different positions is sequentially masked, and the semantic distance between the generated images and those from the full model is measured. Masking intermediate blocks causes the subject to disappear (largest semantic distance), while masking leading or trailing blocks has minimal effect. The optimal skip pair $(n^*, m^*)$ is searched over 30 categories from CustomConcept101 by computing DINO embedding distances, minimizing the combined masking impact of the first $n$ and last $m$ blocks.
   - **Design Motivation**: Unlike U-Net, DiT lacks an explicit hierarchical structure, necessitating an empirical approach to determine block importance. Cross-attention masking provides an efficient probe; once precomputed, it enables rapid lookup for any desired skip ratio.

3. **Residual Feature Precomputation**

   - **Function**: Preserves information integrity when skipping non-critical blocks, avoiding train-inference discrepancy.
   - **Mechanism**: For $l$ consecutive skipped blocks, the residual is precomputed as $\Delta f_{i,i+l} = f_{i+l} - f_i$ (the difference between the output of the last skipped block and its input). During training, the residual is added to the updated input: $f'_{i+l} = f'_i + \Delta f_{i,i+l}$. Residuals are extracted from the original model and stored prior to training.
   - **Design Motivation**: Directly skipping blocks causes severe feature distribution shift. Naive skipping in the style of HollowedNet significantly degrades performance on DiT (DINO drops from 0.73 to 0.43). Residual precomputation compensates for the information loss of block skipping at negligible storage overhead.

### Loss & Training

- The standard conditional flow matching loss is used, consistent with the original FLUX/SANA training objectives.
- LoRA adapters are injected only into the non-skipped blocks.
- Parameters of skipped blocks are offloaded from GPU to CPU; precomputed residual features are loaded on demand.
- Each subject is trained using 4–6 reference images; evaluation uses 25 category-specific prompts with 4 generated images per prompt.

## Key Experimental Results

### Main Results

**Personalization quality comparison on DreamBooth dataset with FLUX:**

| Method | Skip Ratio | Train Resolution | DINO↑ | CLIP-I↑ | CLIP-T↑ |
|--------|-----------|-----------------|-------|---------|---------|
| LoRA (baseline) | – | 512×512 | 0.7324 | 0.8146 | 0.3173 |
| LISA | – | 512×512 | 0.7387 | 0.8194 | 0.3177 |
| HollowedNet | 50% | 512×512 | 0.4435 | 0.6930 | 0.3094 |
| **Ours** | **30%** | **256×256** | **0.7194** | **0.8036** | **0.3199** |
| **Ours** | **40%** | **256×256** | **0.7171** | **0.8034** | **0.3194** |
| **Ours** | **50%** | **256×256** | **0.6963** | **0.7877** | **0.3184** |

**Memory comparison (FLUX BFloat16):**

| Method | Training Memory (GiB) | TFLOPs |
|--------|-----------------------|--------|
| LoRA 512×512 | ~30 | ~28 |
| Ours 30% 256×256 | ~15 | ~7 |
| Ours 50% 256×256 | ~12 | ~5 |

### Ablation Study

| Configuration | DINO | CLIP-I | CLIP-T | Notes |
|---------------|------|--------|--------|-------|
| LoRA 512×512 | 0.7324 | 0.8146 | 0.3173 | Baseline |
| + Resize to 256 | 0.7164 | 0.8044 | 0.3176 | Naive resolution reduction |
| + Dynamic Patch | 0.7253 | 0.8099 | 0.3196 | Dynamic sampling outperforms naive resize |
| Block Skip (no residual) 50% | 0.4301 | 0.6794 | 0.3047 | Naive skipping collapses |
| Block Skip + residual 50% | 0.7150 | 0.8035 | 0.3182 | Residual restores performance |
| Skip first 50% blocks | 0.6651 | 0.7646 | 0.3193 | Inferior to proposed strategy |
| Skip last 50% blocks | 0.4808 | 0.7111 | 0.3090 | Trailing blocks are more critical |
| Ours (first + last) 50% | 0.7150 | 0.8035 | 0.3182 | Optimal skip combination |

### Key Findings

- **Dynamic patch sampling outperforms naive resize**: DINO 0.7253 vs. 0.7164, confirming the effectiveness of timestep-aware scale variation.
- **Residual feature precomputation is essential for block skipping**: Without residuals, DINO drops sharply from 0.73 to 0.43; with residuals, it recovers to 0.72.
- **Skipping trailing blocks is more damaging than skipping leading blocks**: Skipping the last 50% of blocks alone yields DINO of only 0.48, validating the importance of intermediate layers.
- **30% skip ratio offers the best efficiency–quality trade-off**: DINO 0.7194 is within 1.8% of the LoRA baseline (0.7324) with approximately half the memory.
- **HollowedNet completely fails on DiT**: DINO of only 0.44, whereas the proposed method achieves 0.70+ at the same skip ratio.

## Highlights & Insights

- **Timestep–scale alignment**: Leveraging an intrinsic property of the diffusion process (high noise ↔ coarse structure, low noise ↔ fine detail) to guide training strategy design is both principled and generalizable. This idea is transferable to efficient training of video diffusion models.
- **Residual feature precomputation**: A minimalist approach that compensates for information loss from block skipping by approximating the skipped blocks' contribution with a fixed bias offset—negligible computational overhead but significant performance recovery.
- **Cross-attention masking as a probe**: Using attention masking experiments rather than gradient-based analysis to identify critical layers is more intuitive and computationally lightweight, offering a new perspective for DiT interpretability research.

## Limitations & Future Work

- **No inference memory reduction**: The method optimizes only the training phase; inference still requires a full model forward pass. Additional pruning or distillation would be needed for on-device inference.
- **Storage overhead of precomputed residuals**: Residual features must be stored for each training iteration, which may become a bottleneck in large-scale training scenarios.
- **Architecture-specific optimal skip ratios**: The optimal skip configuration for SANA and FLUX differs, requiring a new block selection analysis when generalizing to new architectures.
- **Directions for improvement**: (a) Explore dynamic residuals instead of static precomputation, allowing residuals to adapt as LoRA weights are updated; (b) extend the block selection strategy to inference acceleration; (c) combine with gradient checkpointing to further reduce activation memory.

## Related Work & Insights

- **vs. HollowedNet**: HollowedNet is a layer-skipping method designed for U-Net; when directly applied to DiT, performance collapses (DINO 0.44). This paper resolves the block-skipping problem for DiT via cross-attention analysis and residual precomputation.
- **vs. ZOODiP**: Zeroth-order optimization avoids backpropagation but requires 30,000 steps to converge and exhibits instability; the proposed method converges within standard training steps.
- **vs. LISA / LoRA-FA**: These efficient training methods from the LLM domain yield inconsistent results on DiT (significant degradation on SANA), whereas the proposed method demonstrates broader applicability.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Dynamic patch sampling and block skipping are individually not entirely new, but their combination with cross-attention-based block selection constitutes a meaningful contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers two architectures (FLUX and SANA) with comprehensive ablations, though validation on a broader set of models is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with clear motivation and well-designed figures and tables.
- **Value**: ⭐⭐⭐⭐ Practically significant for efficient DiT fine-tuning; approximately halves memory usage with negligible quality degradation, making on-device deployment more feasible.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] MPDiT: Multi-Patch Global-to-Local Transformer Architecture for Efficient Flow Matching](mpdit_multi-patch_global-to-local_transformer_architecture_for_efficient_flow_ma.md)
- [\[NeurIPS 2025\] DEFT: Decompositional Efficient Fine-Tuning for Text-to-Image Models](../../NeurIPS2025/image_generation/deft_decompositional_efficient_finetuning_for_texttoimage_mo.md)
- [\[CVPR 2026\] EdgeDiT: Hardware-Aware Diffusion Transformers for Efficient On-Device Image Generation](edgedit_hardware-aware_diffusion_transformers_for_efficient_on-device_image_gene.md)
- [\[CVPR 2026\] CRAFT: Aligning Diffusion Models with Fine-Tuning Is Easier Than You Think](craft_aligning_diffusion_models_with_finetuning_is_easier_than_you_think.md)
- [\[AAAI 2026\] DogFit: Domain-guided Fine-tuning for Efficient Transfer Learning of Diffusion Models](../../AAAI2026/image_generation/dogfit_domain-guided_fine-tuning_for_efficient_transfer_learning_of_diffusion_mo.md)

<!-- RELATED:END -->
