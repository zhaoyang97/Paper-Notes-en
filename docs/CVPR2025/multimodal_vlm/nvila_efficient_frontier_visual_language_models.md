---
title: >-
  [Paper Note] NVILA: Efficient Frontier Visual Language Models
description: >-
  [CVPR 2025][Multimodal VLM][Visual Language Models] NVILA proposes the "Scale-then-Compress" paradigm. By scaling up spatial and temporal resolutions and subsequently compressing visual tokens, it maintains or even surpasses SOTA accuracy while reducing training costs by 1.9-5.1x, prefill latency by 1.6-2.2x, and decoding latency by 1.2-2.8x.
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "Visual Language Models"
  - "Efficient Training"
  - "Visual Token Compression"
  - "Multi-resolution Processing"
  - "Model Deployment"
date: 2026-05-08
content_hash: 822f96dbda45b132
---

# NVILA: Efficient Frontier Visual Language Models

**Conference**: CVPR 2025  
**arXiv**: [2412.04468](https://arxiv.org/abs/2412.04468)  
**Code**: [https://github.com/NVlabs/NVILA](https://github.com/NVlabs/NVILA)  
**Area**: Multimodal VLMs  
**Keywords**: Visual Language Models, Efficient Training, Visual Token Compression, Multi-resolution Processing, Model Deployment

## TL;DR

NVILA proposes the "Scale-then-Compress" paradigm. By scaling up spatial and temporal resolutions and subsequently compressing visual tokens, it maintains or even surpasses SOTA accuracy while reducing training costs by 1.9-5.1x, prefill latency by 1.6-2.2x, and decoding latency by 1.2-2.8x.

## Background & Motivation

Visual Language Models (VLMs) have recently made remarkable progress in terms of accuracy, but their efficiency has long been overlooked. VLMs are computationally expensive across multiple dimensions: (1) training a 7B VLM requires 400 GPU-days; (2) full fine-tuning of a 7B VLM requires over 64GB of VRAM; (3) edge deployment (laptops, robotics) is highly resource-constrained. Existing approaches either sacrifice accuracy for efficiency, or naively scale up resolution, leading to computational explosion (doubling resolution quadruples self-attention computation). **Key Challenge**: The irreconcilable contradiction between the information gain from high resolution/long videos and the explosion of token counts. NVILA addresses this via a "Scale-then-Compress" approach, capturing maximum information first and then compressing it efficiently, thereby systematically optimizing the entire life cycle (training $\rightarrow$ fine-tuning $\rightarrow$ deployment). **Core Idea**: High-information-density compressed tokens are superior to raw, low-resolution tokens.

## Method

### Overall Architecture

NVILA is built upon VILA as an autoregressive VLM consisting of three components: a vision encoder (SigLIP) to extract visual features, a projector (2-layer MLP) to align cross-modal embeddings, and a token processor (Qwen2 LLM) that receives visual and language tokens and outputs language tokens. The overall pipeline adopts a five-stage training process: projector initialization $\rightarrow$ vision encoder pre-training $\rightarrow$ token processor pre-training $\rightarrow$ image instruction tuning $\rightarrow$ video instruction tuning.

### Key Designs

1. **Dynamic-S2 Spatial Scaling**:
    - **Function**: Adaptively processes high-resolution images with various aspect ratios.
    - **Mechanism**: Based on the S2 multi-scale tiling strategy, the image is no longer forced to resize to a square at the largest scale. Instead, the image size is adaptively adjusted to the closest resolution to the original aspect ratio that is divisible by $448^2$. Feature maps from each scale are interpolated to the size of the largest scale and concatenated along the channel dimension.
    - **Design Motivation**: The original S2 always resizes images to squares, causing severe distortion to narrow or wide images. Dynamic-S2, inspired by InternVL's dynamic resolution strategy, delivers up to a 30% accuracy improvement on text-intensive benchmarks.

2. **Spatial Token Compression (STC + VEP)**:
    - **Function**: Compresses the number of visual tokens by 2.4x while maintaining accuracy.
    - **Mechanism**: Uses a $3\times3$ Spatial-to-Channel (STC) reshape to compress the number of tokens from $16\times16=256$ to $11\times11=121$. Since aggressive compression makes projector training difficult, an additional Vision Encoder Pre-training (VEP) phase is introduced to jointly tune the vision encoder and the projector.
    - **Design Motivation**: Simple $2\times2$ STC lossless compression has been verified by VILA, but larger compression ratios (such as $3\times3$) lead to a drop of about 10% on DocQA. The VEP phase successfully recovers most of the accuracy loss. Experiments also show that learnable compression methods like TokenLearner and Perceiver Resampler do not outperform simple STC at equivalent compression ratios.

3. **Temporal Token Compression (Temporal Averaging)**:
    - **Function**: Processes long videos (up to 256 frames) while controlling token count.
    - **Mechanism**: Groups video frames and applies average pooling along the temporal dimension within each group, leveraging the inherent temporal continuity of video to eliminate redundancy. For example, compressing 32 frames by 4x yields the same token count as an 8-frame baseline, yet achieves over 5% higher accuracy.
    - **Design Motivation**: Consecutive frames typically contain similar information; temporal pooling effectively reduces redundancy while preserving crucial spatio-temporal information. Further scaling to 256 frames with 8x compression achieves SOTA performance on Video-MME among 7B models.

### Loss & Training

- **DeltaLoss Dataset Pruning**: Scores training data using the output log-probability ratio of a large model to a small model, i.e., $\log \frac{p_{\text{large}}(x)}{p_{\text{small}}(x)}$. Samples with a ratio close to 0 are too simple (both models are correct or both are incorrect), negative ratios indicate distracting samples (small model is correct but large model is incorrect), and positive ratios indicate the most valuable samples to learn from (small model is incorrect but large model is correct). Pruning 50% of the data causes almost no loss in accuracy while speeding up training by 2x.
- **FP8 Mixed-Precision Training**: Leverages native FP8 support on H100 GPUs, quantizing weights and activations to FP8, which allows the batch size to expand from 4 to 16 and boosts training throughput by 2x. When combined with gradient checkpointing, it still yields a 1.2x acceleration.
- **Efficient Fine-Tuning Strategy**: The learning rate for the ViT component should be 5–50 times smaller than that of the LLM component. Fine-tuning only the LayerNorm in ViT achieves performance comparable to LoRA while reducing training time by 25%. When paired with QLoRA, fine-tuning fits within 24GB of VRAM.
- **Quantized Deployment**: W8A8 quantization for the vision encoder is virtually lossless, and W4A16 quantization for the LLM backbone, combined with optimized FP16 accumulation GEMM kernels, brings a 1.7x core acceleration.

## Key Experimental Results

### Main Results (Image)

| Benchmark | Metric | NVILA-8B | Qwen2-VL-8B | LLaVA-OV-8B | GPT-4o |
|-----------|------|----------|-------------|-------------|--------|
| AI2D | test | **92.3** | 83.0 | 81.4 | 94.2 |
| DocVQA | test | **93.7** | 94.5 | 87.5 | 92.8 |
| ChartQA | test | **86.1** | 83.0 | 80.0 | 85.7 |
| TextVQA | val | 80.1 | **84.3** | 78.3 | 78.7 |
| MMMU | val | 49.9 | **54.1** | 48.8 | 69.1 |
| MathVista | testmini | **65.4** | 58.2 | 63.2 | 63.8 |

### Main Results (Video)

| Benchmark | NVILA-8B (256 frames) | Qwen2-VL-8B | LLaVA-OV-8B | GPT-4o |
|-----------|-----------------|-------------|-------------|--------|
| Video-MME (w/o sub) | **64.2** | 63.3 | 58.2 | 71.9 |
| Video-MME (w/ sub) | **70.0** | 69.0 | 61.5 | 77.2 |
| MLVU m-avg | **70.1** | 65.5 | 64.7 | 64.6 |
| MVBench | **68.1** | 67.0 | 56.7 | - |
| ActivityNet-QA | **60.9** | - | 56.6 | - |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Data Pruning 50% (DeltaLoss) | IM-10: 75.5 vs Baseline 75.6 | Almost no loss in accuracy, training speedup 2x |
| Data Pruning 50% (Random) | IM-10: 74.0 | Significantly worse than DeltaLoss |
| FP8 without GC | Throughput 390.1 vs BF16 199.2 | 2.0x speedup |
| Spatial Compression 3x3 STC + VEP | IM-10: 70.8 vs without VEP 67.1 | VEP recovers 3.7 points |
| Temporal Compression 32 frames 4x | Video-MME: 60.1 vs uncompressed 61.0 | Only 0.9% accuracy loss, token count reduced by 4x |

### Key Findings

- The "Scale-then-Compress" strategy enables NVILA to achieve higher accuracy than directly using low resolution under the same token budget.
- Simple STC spatial compression combined with the VEP pre-training stage outperforms more complex learnable methods like TokenLearner and Perceiver Resampler.
- DeltaLoss dataset pruning significantly outperforms random and clustering-based pruning, especially on tasks like DocQA.
- NVILA-8B outperforms all open-source models of comparable size across all video understanding benchmarks, even approaching GPT-4o mini.

## Highlights & Insights

- **Full-Lifecycle Efficiency Optimization**: From training data pruning, FP8 training, and efficient fine-tuning to quantized deployment, establishing a comprehensive methodology for efficiency optimization.
- **Simplicity Over Complexity**: STC reshape is more effective than TokenLearner or Perceiver Resampler, validating the intuition that "good information compression does not require learnable parameters."
- **The VEP Stage is a Key Innovation**: Addresses the difficulty in training the projector caused by aggressive compression, representing a successful practice in "compensatory pre-training."
- **Philosophy of DeltaLoss Data Pruning**: Samples that are too simple or too difficult do not benefit learning. Only samples that are "informative to the large model but challenging for the small model" hold the highest value.

## Limitations & Future Work

- Performance still lags behind Qwen2-VL on knowledge-reasoning benchmarks like MMMU, indicating that "compression" might lose some high-level semantic information.
- DeltaLoss requires running inference with both large and small models on all data to calculate scores, which incurs significant computational overhead itself.
- Processing 256 frames still requires the vision encoder to run inference frame-by-frame (though this is not the main bottleneck); future work can explore more efficient video encoding methods.
- The paper focuses on token compression strategies with a "simple design", which might not be flexible enough in extreme-resolution scenarios.

## Related Work & Insights

- **vs LLaVA-OneVision**: NVILA's training cost is only 1/5 of LLaVA-OneVision's while achieving higher accuracy, demonstrating that "efficiency vs accuracy" is not necessarily a trade-off but can be optimized concurrently.
- **vs Qwen2-VL**: NVILA achieves 1.6–2.8x faster inference at comparable accuracy, thanks to token compression making visual inputs more compact.
- **vs InternVL2**: NVILA adopts its dynamic resolution concept but further introduces token compression to achieve higher token efficiency.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The "Scale-then-Compress" paradigm is clear and elegant, though the individual components are not entirely brand new; the VEP stage serves as a valuable engineering innovation.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Extremely comprehensive, covering 10+ image benchmarks, 6+ video benchmarks, efficiency comparisons, ablation studies, and downstream applications (robotics and medicine).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Well-structured, with the "scale-then-compress" thread consistently woven throughout, complemented by exquisite figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ The full-stack efficiency optimization solutions hold exceptional practical value for the VLM community, alongside open-sourced code and models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] FastVLM: Efficient Vision Encoding for Vision Language Models](fastvlm_efficient_vision_encoding_for_vision_language_models.md)
- [\[CVPR 2025\] Skip Tuning: Pre-trained Vision-Language Models are Effective and Efficient Adapters Themselves](skip_tuning_pre-trained_vision-language_models_are_effective_and_efficient_adapt.md)
- [\[NeurIPS 2025\] Efficient Multi-modal Large Language Models via Progressive Consistency Distillation](../../NeurIPS2025/multimodal_vlm/efficient_multi-modal_large_language_models_via_progressive_consistency_distilla.md)
- [\[ACL 2025\] Activating Distributed Visual Region within LLMs for Efficient and Effective Vision-Language Training and Inference](../../ACL2025/multimodal_vlm/activating_distributed_visual_region_within_llms_for_efficient_and_effective_vis.md)
- [\[ICLR 2026\] ERGO: Efficient High-Resolution Visual Understanding for Vision-Language Models](../../ICLR2026/multimodal_vlm/ergo_efficient_high-resolution_visual_understanding_for_vision-language_models.md)

</div>

<!-- RELATED:END -->
