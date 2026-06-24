---
title: >-
  [Paper Note] When Do We Not Need Larger Vision Models?
description: >-
  [ECCV2024][3D Vision][Multi-scale features] This paper proposes the Scaling on Scales (S2) strategy: freezing a small model (e.g., ViT-B) to run on multiple image scales and concatenating the features, which matches or even outperforms large models (ViT-H/G) on tasks like classification, segmentation, depth estimation, and MLLMs without increasing parameters. Furthermore, it demonstrates both theoretically and experimentally that the representations learned by large models ca…
tags:
  - "ECCV2024"
  - "3D Vision"
  - "Multi-scale features"
  - "Vision Model Scaling"
  - "S2-Wrapper"
  - "ViT"
  - "Multimodal Large Language Models (MLLMs)"
date: 2026-05-08
content_hash: 5ef40742927c807f
---

# When Do We Not Need Larger Vision Models?

**Conference**: ECCV2024  
**arXiv**: [2403.13043](https://arxiv.org/abs/2403.13043)  
**Code**: [https://github.com/bfshi/scaling_on_scales](https://github.com/bfshi/scaling_on_scales)  
**Area**: 3D Vision  
**Keywords**: Multi-scale features, Vision Model Scaling, S2-Wrapper, ViT, Multimodal Large Language Models (MLLMs)

## TL;DR
This paper proposes the Scaling on Scales (S2) strategy: freezing a small model (e.g., ViT-B) to run on multiple image scales and concatenating the features, which matches or even outperforms large models (ViT-H/G) on tasks like classification, segmentation, depth estimation, and MLLMs without increasing parameters. Furthermore, it demonstrates both theoretically and experimentally that the representations learned by large models can be largely approximated linearly by multi-scale small models.

## Background & Motivation

**Background**: Visual representation learning follows the "larger is better" paradigm—ViT-B $\rightarrow$ ViT-L $\rightarrow$ ViT-H $\rightarrow$ ViT-G, with parameter sizes scaling from 86M to billions. Almost all SOTA methods use giant backbones by default.

**Limitations of Prior Work**: Large models have massive parameters, slow inference, and high deployment costs, experiencing diminishing returns on certain downstream tasks (especially dense prediction). Meanwhile, existing multi-scale representation research is limited to specific architectures and cannot be directly applied to arbitrary pre-trained models.

**Key Challenge**: Computational capacity scaling is mainly achieved by increasing model parameters, but image understanding has another dimension—resolution/scale. High-resolution inputs provide more details but are constrained by the $O(n^2)$ self-attention complexity.

**Goal**: (1) Can multi-scale small models be used to replace large models? (2) What exactly is the representational advantage of large models, and can it be replicated by small models? (3) Does incorporating S2 during the pre-training of small models further close the gap?

**Key Insight**: The authors observe that by scaling up an image and splitting it into sub-images to pass through the same frozen small model, the resulting multi-scale features overlap heavily in information with large models—simple concatenation can replace the large model.

**Core Idea**: Use "Scaling on Scales" (S2) to replace "Scaling on Model Size" to achieve better visual representation with zero additional parameters.

## Method

### Overall Architecture
S2-Wrapper is a plug-and-play, parameter-free module applicable to any pre-trained vision model. An input image is interpolated to multiple scales (e.g., $224^2$, $448^2$, $672^2$), each scale is split into $224^2$ sub-images, and all sub-images are passed through the same frozen backbone to extract features. The features are then reconstructed into feature maps of corresponding scales, average-pooled to the original spatial size, and finally concatenated along the channel dimension. The output feature spatial size remains unchanged while the channel dimension is multiplied (e.g., 768 $\to$ 1536).

### Key Designs

1. **Sub-image splitting instead of full-image processing**:

    - **Function**: Splits large-scale images into sub-images matching the pre-training resolution, performing independent inference.
    - **Mechanism**: Avoids the explosion of the $O(n^2)$ self-attention complexity in ViTs, while bypassing performance degradation caused by positional embedding interpolation.
    - **Design Motivation**: Direct positional embedding interpolation on large images causes a significant drop in ViT performance (proven by existing literature, e.g., Bolya et al. 2023), whereas sub-image splitting ensures the model always operates at the resolution seen during training.

2. **Multi-scale feature pooling and concatenation**:

    - **Function**: Feature maps of each scale are average-pooled to the original spatial dimension and then concatenated along the channel dimension.
    - **Mechanism**: Keeps the number of output tokens constant (e.g., $16^2 = 256$ tokens), eliminating the need to modify downstream LLMs/decoders.
    - **Design Motivation**: Without pooling, the number of tokens would scale quadratically with the resolution (e.g., $448^2$ generates 4 times the tokens), making the LLM inference cost unacceptable for MLLMs.

3. **Interpolation instead of raw high-resolution input**:

    - **Function**: Upsamples the original $224^2$ image to $448^2$/$672^2$ rather than directly using high-resolution raw images.
    - **Mechanism**: Ensures a fair comparison with model size scaling, as the large model has also only seen the $224^2$ resolution.
    - **Design Motivation**: Eliminates confounding variables brought by extra high-resolution information; direct use of high-resolution raw images is recommended in practical applications.

### Loss & Training
S2-Wrapper itself requires no training (zero parameters, zero training). In MLLM scenarios, the visual backbone is frozen, and only the projection layer and LoRA are trained, following the standard LLaVA-1.5 training pipeline. The paper also validates introducing S2 in the pre-training stage (using multi-scale inputs when training ViT and DINOv2 on ImageNet-21k), showing that S2 pre-training can further improve the generalization capability of small models.

## Key Experimental Results

### Main Results

| Model | Task | ViT-B+S2 vs ViT-H/G | Parameter Ratio | GFLOPs |
|------|------|---------------------|---------|--------|
| ViT (IN-21k) | ImageNet Classification | B+S2 ≈ H | 0.28× | Comparable |
| DINOv2 | ADE20k Segmentation | B+S2 > L (+1.5 mIoU) | 0.28× | Comparable |
| DINOv2 | NYUv2 Depth | B+S2 > L | 0.28× | Comparable |
| OpenCLIP | 6/9 Tasks | B+S2 > Large Models | 0.07×~0.28× | Comparable |

**MLLM Results (LLaVA-1.5 + S2 @1008²)**:

| Model | V*_Att | V*_Spa | VQAv2 | TextVQA | MMBench | SEED |
|------|--------|--------|-------|---------|---------|------|
| LLaVA-1.5-7B | 43.5 | 56.6 | 78.5 | 58.2 | 64.3 | 65.7 |
| LLaVA-1.5-7B-S2 | **51.3** | **61.8** | **80.0** | **61.0** | **66.2** | **67.9** |
| GPT-4V | 51.3 | 60.5 | 77.2 | 78.0 | 75.8 | 71.6 |

On the V* benchmark, the S2 version reaches GPT-4V level, outperforming all open-source MLLMs in fine-grained visual understanding.

### Ablation Study

| Configuration | Description | Key Conclusion |
|------|------|----------|
| Base-S2 (post-pretrain) | S2 added only after pre-training | Outperforms large models on 6/9 tasks, but has weaker generalization |
| Base-S2 (pre-train) | S2 used during pre-training | ViT classification 82.4 vs Large 81.6, outperforming the large model |
| Feature reconstruction experiment | Linear reconstruction of Large features using Base-S2 | Reconstruction rate 96-100%, covering almost all information of the large model |
| Robot grasping | MVP-Base+S2 vs MVP-Large | S2 success rate +20%, far exceeding the +6% from model scaling |

### Key Findings
- **Dense prediction tasks benefit the most**: On segmentation and depth estimation, the advantage of S2 is most prominent (multi-scale features are naturally suited for fine-grained tasks).
- **The only failure case**: For DINOv2 on ImageNet classification, B+S2 is worse than L (lower training loss but worse test performance $\to$ overfitting), but this can be fixed by incorporating S2 during pre-training.
- **Feature approximability**: 99%+ of the useful representations of large models can be linearly reconstructed by multi-scale small models, suggesting that large models do not learn fundamentally different representations.
- **The sole advantage of large models**: They generalize better on rare/difficult examples (e.g., a sculpture of a TV shape), but this can be compensated for by S2 pre-training.

## Highlights & Insights
- **Extremely simple yet powerful design**: S2-Wrapper has no trainable parameters and can be applied to any vision model with a single line of code, yet its performance is comparable to or surpasses large models with costly parameters. This philosophy of "scaling resolution instead of parameters" is highly elegant.
- **Unique feature reconstruction analysis**: Quantitative analysis using mutual information and linear reconstruction to answer "what exactly do small models lack" reveals the answer is "almost nothing", providing a new perspective on model scaling for the community.
- **Constant token count**: Pooling back to the original spatial size introduces zero overhead on the LLM side, solving the core bottleneck of high-resolution MLLMs. It can be directly migrated to any LLaVA-like architecture.
- **Parallel processing potential**: Independent inference of sub-images naturally supports parallelization, which is of great significance for latency-sensitive scenarios.

## Limitations & Future Work
- **No reduction in computation**: Although having fewer parameters, the GFLOPs are matched ($448^2$ requires 4x inference), so the actual inference speedup depends on the parallel implementation.
- **No interaction between sub-images**: Each sub-image passes through the backbone independently, losing global context across sub-images (e.g., when a large object is cut off). Lightweight cross-window attention could be considered.
- **Only validated on the ViT family**: ConvNeXt experiments are placed in the appendix and are less significant than ViT; suitability for other architectures (such as Vision Mamba models) is unknown.
- **Limited scale of the pre-trained S2 experiments**: Only validated on ImageNet-21k, while performance on larger datasets like LAION-2B remains to be validated.

## Related Work & Insights
- **vs Multi-scale ViT (MViT, Swin)**: These methods design multi-scale architectures internally and require specialized training. S2 is a post-processing step for any existing model with zero training cost, and it can be applied cumulatively.
- **vs Dynamic resolution in LLaVA-NeXT / InternVL**: Subsequent works adopt a similar sub-image splitting approach with improvements (e.g., dynamic resolution selection); S2 is a pioneering work in this direction.
- **vs High-resolution pre-training (DINOv2 @518²)**: DINOv2 pre-training is limited to a maximum of $518^2$ due to computational resource constraints. S2 allows expansion to $1008^2$+ during inference without retraining.

## Rating
- Novelty: ⭐⭐⭐⭐ Multi-scale features themselves are not new, but the systematic study of this as a general scaling strategy and the argument that "large models are not necessary" is highly novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 5 main tasks of classification/segmentation/depth/MLLM/robotics, 3 pre-training systems, and includes theoretical analysis of feature reconstruction.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear logic, step-by-step progression from empirical findings to theory, and then to solutions.
- Value: ⭐⭐⭐⭐⭐ Directly influenced the design of subsequent high-resolution MLLM research like LLaVA-NeXT and InternVL; accessible for use with a single line of code.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] Sapiens: Foundation for Human Vision Models](sapiens_foundation_for_human_vision_models.md)
- [\[NeurIPS 2025\] How Many Tokens Do 3D Point Cloud Transformer Architectures Really Need?](../../NeurIPS2025/3d_vision/how_many_tokens_do_3d_point_cloud_transformer_architectures_really_need.md)
- [\[ICLR 2026\] Do 3D Large Language Models Really Understand 3D Spatial Relationships?](../../ICLR2026/3d_vision/do_3d_large_language_models_really_understand_3d_spatial_relationships.md)
- [\[ICLR 2026\] DepthLM: Metric Depth from Vision Language Models](../../ICLR2026/3d_vision/depthlm_metric_depth_from_vision_language_models.md)
- [\[ECCV 2024\] PointLLM: Empowering Large Language Models to Understand Point Clouds](pointllm_empowering_large_language_models_to_understand_point_clouds.md)

</div>

<!-- RELATED:END -->
