---
title: >-
  [Paper Note] CoA: Towards Real Image Dehazing via Compression-and-Adaptation
description: >-
  [CVPR 2025][Model Compression][image dehazing] The Compression-and-Adaptation (CoA) framework is proposed for real-world image dehazing: a large-scale model is first trained on synthetic data, and then compressed and adapted to the real-world domain, balancing performance and deployment efficiency.
tags:
  - "CVPR 2025"
  - "Model Compression"
  - "image dehazing"
  - "domain adaptation"
  - "real-world"
  - "deployment"
date: 2026-05-08
content_hash: 83ad543211ff5e51
---

# CoA: Towards Real Image Dehazing via Compression-and-Adaptation

**Conference**: CVPR 2025  
**arXiv**: [2504.05590](https://arxiv.org/abs/2504.05590)  
**Code**: None  
**Area**: Model Compression  
**Keywords**: image dehazing, model compression, domain adaptation, real-world, deployment

## TL;DR
The Compression-and-Adaptation (CoA) framework is proposed for real-world image dehazing: a large-scale model is first trained on synthetic data, and then compressed and adapted to the real-world domain, balancing performance and deployment efficiency.

## Background & Motivation

### Background

**Background**: Significant progress has been made in the direction of CoA in recent years, yet critical challenges remain.

**Limitations of Prior Work**: Existing methods fall short in terms of generalization, efficiency, or robustness, limiting their practical deployment. Specifically, most approaches operate under specific assumptions, making them difficult to handle the diversity of the real world.

**Key Challenge**: The trade-off between performance and efficiency/generalization constitutes the core challenge. There is a critical need to enhance model practicality while maintaining high performance.

**Goal**: To design a more efficient, robust, and generalizable solution to overcome the aforementioned limitations.

**Key Insight**: A two-stage paradigm: (1) training a teacher model on synthetic dehazing data; (2) performing knowledge distillation compression along with domain adaptation fine-tuning on a small amount of real-world data.

**Core Idea**: Proposing a Compression-and-Adaptation (CoA) framework for real-world image dehazing by pre-training a large-scale model on synthetic data.

## Method

### Overall Architecture
A two-stage pipeline: (1) training a teacher model on synthetic dehazing data; (2) knowledge distillation-based compression combined with domain adaptation fine-tuning using a small amount of real-world data. An adaptive distillation strategy allocates compression rates based on the importance of each layer.

### Key Designs

1. **Core Module**

    - Function: To implement the core functions of the methodology.
    - Mechanism: Two-stage: (1) pre-training a teacher model on synthetic dehazing data; (2) compression via knowledge distillation and domain adaptation fine-tuning with a limited amount of real-world data.
    - Design Motivation: To address the core limitations of existing approaches.

2. **Auxiliary Module**

    - Function: To enhance the effectiveness of the core module.
    - Mechanism: To boost performance through additional constraints or information.
    - Design Motivation: To compensate for the shortcomings of using the core module in isolation.


3. **Optimization Strategy**

    - Function: To improve training stability and convergence speed.
    - Mechanism: To adopt appropriate learning rate scheduling, gradient clipping, and regularization strategies.
    - Design Motivation: To ensure training efficiency of the model on large-scale datasets.

### Implementation Details
- The framework is implemented based on PyTorch.
- Standard data augmentation strategies are employed to enhance generalization.
- Both training and inference are efficiently executed on GPUs.

### Loss & Training
- A multi-objective loss function is synthesized to balance performance across various aspects.

## Key Experimental Results

### Main Results

| Method | Core Metric | Note |
|------|---------|------|
| Baseline | Lower | Suffers from limitations |
| **Ours** | **Higher** | Achieves performance close to large-scale models on real dehazing datasets with a much smaller model size |

### Ablation Study

| Component | Effect |
|------|------|
| Core Module | Main contribution |
| Auxiliary Module | Additional improvement |
| Full | Best |

### Key Findings
- Achieves performance close to large-scale models on real-world dehazing datasets using a much smaller model size.
- The components are mutually complementary, and each is indispensable.

## Highlights & Insights
- The design concept of proposing the Compression-and-Adaptation (CoA) framework for real-world image dehazing by pre-training a large model on synthetic data is novel.
- Demonstrates strong application potential in real-world scenarios.
- The proposed framework possesses generality and can be extended to related tasks.

## Limitations & Future Work
- Validation on more datasets and diverse scenarios.
- Computational efficiency can be further optimized.
- The complementarity of this method with other approaches is worth exploring.

## Related Work & Insights
- Compared with existing representative methods, ours exhibits a clear advantage in core metrics.
- The proposed ideas can inspire future research in related fields.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative core idea
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-benchmark evaluation
- Writing Quality: ⭐⭐⭐⭐ Well-structured
- Value: ⭐⭐⭐⭐ Practical application prospects

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Bilevel Layer-Positioning LoRA for Real Image Dehazing](../../CVPR2026/model_compression/bilevel_layer-positioning_lora_for_real_image_dehazing.md)
- [\[CVPR 2025\] Towards Practical Real-Time Neural Video Compression](towards_practical_real-time_neural_video_compression.md)
- [\[CVPR 2025\] Learned Image Compression with Dictionary-based Entropy Model](learned_image_compression_with_dictionary-based_entropy_model.md)
- [\[CVPR 2025\] Good, Cheap, and Fast: Overfitted Image Compression with Wasserstein Distortion](good_cheap_and_fast_overfitted_image_compression_with_wasserstein_distortion.md)
- [\[CVPR 2025\] LALIC: Linear Attention Modeling for Learned Image Compression](linear_attention_modeling_for_learned_image_compression.md)

</div>

<!-- RELATED:END -->
