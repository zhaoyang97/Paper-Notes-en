---
title: >-
  [Paper Note] Quantized Prompt for Efficient Generalization of Vision-Language Models
description: >-
  [ECCV 2024][Multimodal Efficiency][Quantization] By treating quantization error as a form of regularization noise, this work applies ultra-low-bit quantization (down to 1-bit) to the learnable prompts of VLMs. This significantly reduces storage overhead (up to $16\times$ compression) while markedly improving the model's generalization capability to unseen classes. QCoOp achieves superior performance over various state-of-the-art (SOTA) methods using only 0.26KB of storage.
tags:
  - "ECCV 2024"
  - "Multimodal Efficiency"
  - "Quantization"
  - "prompt tuning"
  - "generalization"
  - "vision-language models"
  - "parameter-efficient fine-tuning"
date: 2026-05-08
content_hash: e0ee35863392bbdb
---

# Quantized Prompt for Efficient Generalization of Vision-Language Models

**Conference**: ECCV 2024  
**arXiv**: [2407.10704](https://arxiv.org/abs/2407.10704)  
**Code**: [GitHub](https://github.com)  
**Area**: Multimodal VLM  
**Keywords**: Quantization, prompt tuning, generalization, vision-language models, parameter-efficient fine-tuning

## TL;DR
By treating quantization error as a form of regularization noise, this work applies ultra-low-bit quantization (down to 1-bit) to the learnable prompts of VLMs. This significantly reduces storage overhead (up to $16\times$ compression) while markedly improving the model's generalization capability to unseen classes. QCoOp achieves superior performance over various state-of-the-art (SOTA) methods using only 0.26KB of storage.

## Background & Motivation
Adapting large-scale pre-trained vision-language models (such as CLIP) to downstream tasks faces two core issues: **overfitting** and **catastrophic forgetting**. Existing methods (CoOp, CoCoOp, MaPLe, etc.) utilize prompt tuning for parameter-efficient fine-tuning, but as these methods grow increasingly complex, storage and inference costs become more prominent.

This work stems from a key observation: **moderate random noise can suppress overfitting and catastrophic forgetting**. The authors further point out that since quantization error is essentially a form of noise, quantization can be exploited to regularize VLMs. Compared to Gaussian noise, quantization error is more controllable, and quantization itself significantly reduces storage. This unique perspective organically unifies model compression and generalization enhancement.

The key challenge lies in the trade-off: excessive noise degrades the model's adaptation capability, whereas insufficient noise fails to provide effective regularization. Thus, the quantization scheme must be meticulously designed to maintain the quantization error at an "optimal" level.

## Method

### Overall Architecture
Based on an in-depth analysis of the prompt weight distribution characteristics, the QPrompt method adopts K-Means clustering as the foundation for quantization, combining it with normalization/denormalization and a constrained adaptive clustering strategy. Overall pipeline: during training, gradients are backpropagated through the quantization operator using the Straight-Through Estimator (STE); during storage, FP16 parameters are converted into $b$-bit indices along with a codebook.

### Key Designs

1. **Analysis of the Relationship Between Noise and Generalization**:

    - Function: Apply Gaussian noise of varying intensities to prompt weights and observe the changes in accuracy on base/new classes.
    - Key Findings: During training, the baseline's generalization capability continuously decreases while its specialization capability increases. A moderate amount of noise (e.g., 0.01) improves accuracy on unseen classes without significantly compromising accuracy on seen classes.
    - Design Motivation: Excessive noise (0.1) severely weakens adaptation capacity, while negligible noise (0.001) fails to regularize the model. Only moderate noise proves beneficial, which provides a theoretical justification for designing the quantization scheme.

2. **Analysis of Prompt Weight Distribution Characteristics**:

    - Function: Analyze changes in the distribution of prompt weights during CoOp training.
    - Key Findings: (1) The shape of the weight distribution remains largely unchanged throughout training; (2) The variance increases rapidly at the early stage of training; (3) There are almost no outliers; (4) Weight transitions between adjacent stages are smooth.
    - Design Motivation: These characteristics directly guide the design principles of the quantization scheme.

3. **K-Means Quantization and Normalization**:

    - Function: Construct a quantization mapping $Q$ using K-Means clustering to map prompt weights to $2^b$ discrete values.
    - Mechanism: First normalize the weights as $\hat{W} = \frac{W - \mu}{\sigma}$, perform K-Means clustering in the normalized space, and then denormalize to obtain the quantized weights $W_q = \sigma Q(\hat{W}) + \mu$.
    - Design Motivation: Normalization eliminates the effects of distribution translation and scaling (since the distribution shape remains constant during training, and changes primarily originate from variance). K-Means remains robust as prompt weights do not contain outliers.

4. **Constrained Adaptive Clustering (CAC)**:

    - Function: Dynamically control the update frequency of the K-Means codebook.
    - Mechanism: Instead of clustering at every iteration, a minimum update interval $t$ is set, and the difference between the current weight distribution and the cached weight distribution is measured via KL divergence. Re-clustering is triggered only when the KL divergence exceeds a threshold $T_{KL}$.
    - Design Motivation: (1) K-Means is computationally expensive; frequent execution impairs training efficiency. (2) Only moderate quantization error benefits generalization; continuously minimizing quantization error can be counterproductive. (3) Weight transitions between adjacent stages are smooth, making frequent updates redundant.
    - KL Divergence Calculation: First map the new and old weights to the same event space (the index space of K-Means clustering), then compute the KL divergence between the index probability distributions.

5. **Storage Optimization**:

    - Function: Replace FP16 parameters with $b$-bit indices and a codebook.
    - Storage Footprint: $bN + 2^b \times 16$ bits, compared to the baseline's $16N$ bits.
    - When $b=1$, this yields approximately $16\times$ compression.

### Loss & Training
- During training, gradients are propagated directly through the quantization operator using STE: $\frac{\partial Q(x)}{\partial x} = x$
- Standard CLIP classification loss: Cross-entropy loss based on image-text similarity.
- The method can be integrated into CoOp (yielding QCoOp) or MaPLe (yielding QMaPLe).

## Key Experimental Results

### Main Results: Base-to-New Generalization (Average over 11 Datasets)

| Method | Parameter Size | Base | New | H (Harmonic Mean) |
|------|---------|------|-----|--------------|
| CLIP | 0KB | 69.34 | 74.22 | 71.70 |
| CoOp | 4.1KB | 82.69 | 63.22 | 71.66 |
| CoCoOp | 70.8KB | 80.47 | 71.69 | 75.83 |
| ProGrad | 16.4KB | 82.79 | 68.55 | 75.00 |
| **QCoOp** | **0.26KB** | **80.68** | **74.44** | **77.43** |
| MaPLe | 7096KB | 82.28 | 75.14 | 78.55 |
| **QMaPLe** | **1774KB** | **83.02** | **75.57** | **79.12** |

### Ablation Study

| Configuration | Base | New | H | Description |
|------|------|-----|---|------|
| K-Means only | 78.71 | 72.55 | 75.50 | Basic quantization |
| K-Means + Norm | 78.84 | 73.09 | 75.85 | Normalization improves new acc |
| K-Means + Norm + CAC | 78.24 | 74.02 | 76.07 | CAC further improves new acc |
| QAT | 80.72 | 72.35 | 76.31 | QAT outperforms PTQ |
| PTQ | 82.21 | 68.50 | 74.73 | PTQ shows weak generalization |

### Key Findings
- QCoOp ($0.26\text{KB}$) is $63\times$ smaller than ProGrad ($16.4\text{KB}$) while achieving higher accuracy, demonstrating extreme efficiency.
- QMaPLe requires only $0.25\times$ the storage space compared to MaPLe, yet improves H by $0.57\%$.
- In cross-dataset transfer, QCoOp achieves the highest accuracy on 5 out of 10 target datasets.
- In few-shot learning, QCoOp outperforms CLIP, CoOp, and CLIP-Adapter across all shot settings.
- Evaluation based on the SLIP model: The new class accuracy of QCoOp ($74.04\%$) is significantly higher than that of CoOp ($46.60\%$), with the H score soaring from $55.51\%$ to $71.07\%$.
- Increasing the number of quantization bits does not necessarily yield better results: $b=1$ ($75.92\%$) $\ge$ $b=2$ ($75.91\%$) $\ge$ $b=4$ ($75.76\%$).

## Highlights & Insights
- **Novel Perspective**: For the first time, quantization error is regarded as a regularization tool rather than a defect to be minimized, completely subverting the optimization objective of conventional quantization.
- **Unification of Theory and Practice**: Design principles of the quantization scheme are derived from detailed analysis of prompt weight distribution rather than being designed out of thin air.
- **Extreme Efficiency**: The $0.26\text{KB}$ model size enables adaptation on highly resource-constrained devices.
- **High Versatility**: The proposed method can be seamlessly integrated into various existing methods (such as CoOp and MaPLe) to achieve consistent improvements.
- **Viability of 1-bit Quantization**: 1-bit quantization, which is traditionally deemed "aggressive", surprisingly performs the best on prompts.

## Limitations & Future Work
- The choice of quantization bits ($b=1, 2, 4$) is currently a hyperparameter that lacks an adaptive selection mechanism.
- There is a lack of theoretical guidance for setting the KL-divergence threshold $T_{KL}$ and the minimum update interval $t$.
- The method has only been validated on classification tasks; it has not been evaluated on downstream tasks such as detection and segmentation.
- Only the prompt and certain linear layer parameters are quantized, leaving the quantization of the backbone unexplored.

## Related Work & Insights
- **CoOp/CoCoOp/MaPLe**: Foundational methods for prompt tuning, upon which QPrompt is built to provide quantization-driven enhancement.
- **Regularization & Noise**: Traditional regularization methods include Dropout, data augmentation, etc. This paper extends this concept to quantization noise regularization in the weight space.
- **K-Means Quantization**: Traditionally heavily vulnerable to outliers, but the absence of outliers in prompt weights makes it viable.
- **Insight**: Model compression does not necessarily come at the expense of performance; rather, appropriate "information loss" can potentially enhance generalization.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The perspective of redefining quantization error as regularization noise is highly unique.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across four configurations, 11 datasets, and various baselines, with detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, with a coherent derivation from observation to principles to design.
- Value: ⭐⭐⭐⭐⭐ Achieves dual gains in both efficiency and effectiveness, presenting strong practicality.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ECCV 2024\] IVTP: Instruction-Guided Visual Token Pruning for Large Vision-Language Models](ivtp_instruction-guided_visual_token_pruning_for_large_vision-language_models.md)
- [\[CVPR 2026\] QVGGT: Post-Training Quantized Visual Geometry Grounded Transformer](../../CVPR2026/vlm_efficiency/qvggt_post-training_quantized_visual_geometry_grounded_transformer.md)
- [\[CVPR 2026\] VLM-PTQ: Efficient Post-Training Quantization for Large Vision-Language Models](../../CVPR2026/vlm_efficiency/vlm-ptq_efficient_post-training_quantization_for_large_vision-language_models.md)
- [\[CVPR 2026\] Attention-aware Inference Optimizations for Large Vision-Language Models with Memory-efficient Decoding](../../CVPR2026/vlm_efficiency/attention-aware_inference_optimizations_for_large_vision-language_models_with_me.md)
- [\[CVPR 2026\] AdaptVision: Efficient Vision-Language Models via Adaptive Visual Acquisition](../../CVPR2026/vlm_efficiency/adaptvision_efficient_vision-language_models_via_adaptive_visual_acquisition.md)

</div>

<!-- RELATED:END -->
