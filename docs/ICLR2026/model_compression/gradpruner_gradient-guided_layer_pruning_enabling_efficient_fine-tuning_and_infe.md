---
title: >-
  [Paper Note] GradPruner: Gradient-guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs
description: >-
  [ICLR 2026][Model Compression][Layer Pruning] GradPruner calculates the importance of each layer (IGIA-Matrix) using gradients accumulated during the **initial 1% of steps** of LoRA fine-tuning for layer pruning. It then performs "same-sign merging" of pruned layers into retained layers, achieving simultaneous training and inference speedups on downstream tasks: 40% parameter reduction with only 0.99% accuracy loss.
tags:
  - "ICLR 2026"
  - "Model Compression"
  - "Layer Pruning"
  - "Structured Pruning"
  - "LoRA Fine-tuning"
  - "Gradient Importance"
  - "Layer Merging"
date: 2026-05-08
content_hash: 99717d53ae75e553
---

# GradPruner: Gradient-guided Layer Pruning Enabling Efficient Fine-Tuning and Inference for LLMs

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=bxzJorqyYM](https://openreview.net/forum?id=bxzJorqyYM)  
**Code**: [https://github.com/secretflow/ACoLab/tree/main/PaperCode/GradPrune](https://github.com/secretflow/ACoLab/tree/main/PaperCode/GradPrune)  
**Area**: Model Compression / LLM Structured Pruning  
**Keywords**: Layer Pruning, Structured Pruning, LoRA Fine-tuning, Gradient Importance, Layer Merging  

## TL;DR
GradPruner calculates the importance of each layer (IGIA-Matrix) using gradients accumulated during the **initial 1% of steps** of LoRA fine-tuning for layer pruning. It then performs "same-sign merging" of pruned layers into retained layers, achieving simultaneous training and inference speedups on downstream tasks: 40% parameter reduction with only 0.99% accuracy loss.

## Background & Motivation
**Background**: LLMs often require fine-tuning on downstream data for vertical domains like healthcare and finance to achieve optimal performance, but full fine-tuning is slow and expensive. While structured pruning improves inference efficiency, existing methods typically follow a two-stage process—evaluating parameter importance with calibration data followed by training/distillation for recovery—which consumes additional time and VRAM.

**Limitations of Prior Work**: Most existing pruning methods rely on forward passes (e.g., intermediate activations) on calibration data to evaluate parameter importance. However, since LLMs exhibit weak initial performance in specialized domains, this approach introduces significant bias. Works focused on "efficient training + inference" have drawbacks: APT only supports LoRA fine-tuning, while SAT dynamically modifies the structure during training but restores the model to a dense form in the final step, failing to accelerate inference.

**Key Challenge**: A trade-off exists where more pruning speeds up training/inference but sacrifices accuracy, while less pruning maintains accuracy but remains slow. A balance must be found between "efficient importance evaluation" and "maximizing layer pruning without accuracy loss." The authors observed that loss drops sharply in the first 1% of fine-tuning steps, indicating the model rapidly acquires downstream knowledge and that learning capabilities vary significantly across parameters.

**Goal**: To measure parameter importance for specific downstream data and models without increasing training time or VRAM, maximize structural preservation during pruning, and support both full-fine-tuning and LoRA.

**Core Idea**: **[Early Gradients as Importance]** Construct an IGIA-Matrix using cumulative gradients from the very early steps of LoRA fine-tuning to evaluate layer importance for pruning. **[Same-Sign Layer Merging]** Merge key parameters from pruned layers (after sparsification) into retained layers to further increase the pruning rate without compromising accuracy.

## Method

### Overall Architecture
GradPruner consists of three steps: First, gather gradients from the initial $t$ steps ($t \ll T$) of LoRA fine-tuning to construct the "Initial Gradient Information Accumulation Matrix" (IGIA-Matrix) for each linear layer. Second, sum the IGIA-Matrices of all linear layers within each transformer layer to derive layer importance scores and prune low-scoring layers. Finally, sparsify the pruned layers and merge them into preceding retained layers based on sign compatibility to maintain accuracy at higher pruning rates.

```mermaid
flowchart LR
    A[Downstream Data D] --> B[LoRA fine-tuning first t steps<br/>Collect ∇W_A, ∇W_B]
    B --> C[Simulate W gradient<br/>∇W = ∇W_B · ∇W_A]
    C --> D[IGIA-Matrix F_W<br/>= Mean Squared Gradients]
    D --> E[Layer Importance = Sum of<br/>IGIA of linear layers]
    E --> F[Prune low-score layers]
    F --> G[Sparsify pruned layers via IGIA top-p%]
    G --> H[Same-sign merging into previous retained layer]
```

### Key Designs

**1. IGIA-Matrix: Simulating $W$ importance via initial LoRA gradients.** This is the foundation of the method. Instead of expensive full-training, it utilizes gradients $\nabla_{W_A}L$ and $\nabla_{W_B}L$ from the first $t$ steps of LoRA. Since LoRA parameters can be merged with original parameters, the authors multiply the gradients of the two paths to align with the dimensions of $W$, obtaining a "simulated gradient" for $W$ at step $i$: $\nabla_W L(x,y)^{sim}_i = \nabla_{W_B}L_i \cdot \nabla_{W_A}L_i$. The IGIA-Matrix is then computed as the mean of squared simulated gradients: $F_W = \frac{1}{t}\sum_{i=1}^{t}\big(\nabla_W L(x,y)_i\big)^2$. Squaring eliminates signs and amplifies parameters undergoing intense learning (critical for downstream tasks), allowing early-stage measures to approximate full-training results.

**2. Layer-level Pruning: Preserving overall structure.** After obtaining the IGIA-Matrix for each linear layer, a layer score is calculated by summing the importance of all $M$ linear layers (each with $H$ parameters) within layer $j$: $\text{Layer}_j=\sum_{k=1}^{M}\sum_{l=1}^{H}F_{W_{kl}}$. Layers with the lowest scores are pruned. Choosing layer-level pruning (over neurons/channels) aims to maintain the model's overall architecture, as ablation studies show that pruning parameters of important layers significantly degrades downstream accuracy. However, layer pruning has an upper bound—accuracy drops sharply when more than ~30% of layers are pruned (e.g., pruning more than 10 layers in Llama 3.1-8B).

**3. Same-sign Layer Merging: Breaking the pruning limit.** Instead of discarding pruned layers, their useful information is merged back into retained layers. Given a retained layer $W_1$ and pruned layers $\{W_2,...,W_n\}$, two steps are followed: ① **Sparsification**—Use the IGIA-Matrix as a criterion to keep only the top-p% parameters of pruned layers, setting others to zero to obtain $\{\hat W_2,...,\hat W_n\}$. ② **Same-sign Merging**—To prevent values of opposite signs from canceling each other out, use the sign $\gamma$ of $W_1$ as a baseline. Elements in $\hat W$ are added only if their sign matches $W_1$: when signs conflict, $W_1$ is kept unchanged; when signs match, $(W_j)_{kl}+\hat{(W_{j+n})}_{kl}$ is executed. Pruned layers are merged only with their immediate preceding retained layer. This allows pruning 1–3 additional layers beyond the 10-layer limit while maintaining performance close to the dense model.

## Key Experimental Results

### Main Results (40% Sparse Pruning, Average Score across Eight Datasets)

| Method | Llama3.1-8B (FFT) | Llama3.1-8B (LoRA) | Mistral-7B (FFT) | Mistral-7B (LoRA) |
|---|---|---|---|---|
| Dense Model (Upper Bound) | 0.784 | 0.794 | 0.781 | 0.790 |
| LLMPruner | 0.734 | 0.733 | 0.730 | 0.728 |
| LaCo | 0.736 | 0.740 | 0.738 | 0.737 |
| MINITRON | 0.734 | 0.734 | 0.734 | 0.731 |
| SAT | 0.750 | 0.745 | 0.748 | 0.743 |
| APT | — | 0.759 | — | 0.750 |
| FT(Llama3.2-3B) | 0.777 | 0.774 | — | — |
| **GradPruner** | **0.782** | **0.786** | **0.770** | **0.780** |

GradPruner outperforms all baselines across all four settings. Compared to the dense model, the average drop is only 0.99% (FFT/Llama3.1). Furthermore, the pruned 8B model exceeds the accuracy of a directly fine-tuned Llama 3.2-3B and is approximately 5 percentage points higher than LLMPruner/LaCo/MINITRON.

### Gain (Normalized to Dense Model, Lower is Better)

| Method | Training Time | Training VRAM | Inference Time | Inference VRAM |
|---|---|---|---|---|
| Dense Model | 100% | 100% | 100% | 100% |
| LLMPruner | 78.3% | 284.4% | 67.4% | 65.3% |
| LaCo | 73.8% | 64.4% | 59.7% | 61.3% |
| SAT | 75.5% | 79.4% | 98.9% | 103.6% |
| **GradPruner (FFT)** | **62.4%** | **65.8%** | **61.5%** | **60.9%** |

GradPruner saves approximately 36% in training time/VRAM and 39% in inference time/VRAM, comparable to LaCo. Meanwhile, SAT provides almost no inference savings due to dense restoration, and APT's training time reaches 158% due to distillation.

### Ablation Study (Llama3.1-8B / FFT, Effect of Merged Layers on Accuracy)

| Layers Merged | GradPruner | w/o Merging |
|---|---|---|
| 1 | 0.785 | 0.775 |
| 2 | 0.786 | 0.767 |
| 3 | 0.782 | 0.741 |

### Key Findings
- **Layer merging is critical for high pruning rates**: Accuracy drops sharply without merging as more layers are removed (dropping to 0.741 with 13 layers pruned), but remains stable near the dense model performance with merging.
- **Layer pruning has a hard upper bound**: Pruning up to ~10 layers in Llama 3.1-8B has minimal impact, but further pruning causes significant decline.
- **Sparsity rate is not "the higher the better"**: Accuracy is harmed at both extremes within the 50%–90% range; a "sweet spot" exists.
- **Early gradients are sufficiently reliable**: Gradient sensitivity analysis confirms that layers identified as important in the first 1% of steps are highly consistent with those identified after full training.

## Highlights & Insights
- **Moving pruning forward into early fine-tuning**: Traditional pruning uses forward activations from calibration data, which can be distorted by the model's initial weakness in vertical domains. GradPruner uses early gradients from the task itself, naturally fitting downstream data while eliminating evaluation overhead.
- **Simulating $W$ gradients via LoRA dual-path multiplication**: This ingenious approach avoids direct gradient computation for the frozen large matrix $W$, using the product of low-rank path gradients to balance VRAM efficiency and importance interpretability.
- **Sign-aware merging**: Merging is not simple addition. By addressing sign conflicts that cause mutual cancellation, it converts pruned layers from "trash" to "recycled residuals," breaking the 30% limit of layer pruning.
- **Simultaneous optimization of training and inference**: Unlike most pruning works that focus only on inference, this method reduces training time/VRAM, making it highly suitable for practical scenarios involving repeated fine-tuning in vertical domains.

## Limitations & Future Work
- **Coarse granularity**: The method only performs layer-level pruning. It does not address finer channel or attention head levels, and the pruning rate is limited by the discrete nature of layers.
- **Reliance on LoRA early gradient transferability**: The method relies on the empirical observation that early gradients represent long-term importance, which may not hold for tasks with unstable training or slow loss convergence.
- **Approximation error in merging**: Same-sign merging discards conflicting elements and sparsifies pruned layers; how this lossy operation's error accumulation scales with model size lacks theoretical characterization.
- **Evaluation metrics**: Similarity assessments (BertScore+ROUGE-L) for medical/financial QA still differ from real clinical/compliance needs, and the method was only validated on the 7B/8B scale.

## Related Work & Insights
- **Structured Pruning**: LLMPruner (gradient-based coupling), LaCo (layer collapsing), and MINITRON (joint depth/width pruning + distillation) represent the two-stage "prune then train/distill" route. GradPruner integrates importance evaluation into the fine-tuning process.
- **Efficient Training + Inference Pruning**: APT (dynamic LoRA pruning) and SAT (restoring to dense) are direct competitors. Ours improves upon their limitations regarding LoRA exclusivity and lack of inference acceleration.
- **Gradient/Fisher Importance**: Using full-training gradient information for importance (Matena & Raffel, Daheim et al.) is known. GradPruner's contribution is proving that early gradients suffice and operationalizing this into a prune-and-merge pipeline.
- **Insights**: Utilizing "early adaptation dynamics" as a signal for compression/selection may be more effective for vertical domains than static calibration; the same-sign merging concept could be transferred to model merging or task vectors.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — The combination of early LoRA IGIA-Matrix and same-sign merging is novel, targeting training and inference efficiency simultaneously.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Comprehensive testing across 2 LLMs, 8 datasets, and FFT/LoRA settings, covering accuracy, efficiency, and multiple ablations. However, limited to the 7B/8B range.
- **Writing Quality**: ⭐⭐⭐ — Clear motivation and diagrams, but Equation (2) has notation inconsistencies ($\nabla_{W_B}\cdot\nabla_{W_B}$ vs. the text's intended $W_B\cdot W_A$), and minor clerical errors exist.
- **Value**: ⭐⭐⭐⭐ — Highly practical for efficient compression in vertical domain fine-tuning. Achieving 40% reduction with 0.99% loss is very attractive for production deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Reassessing Layer Pruning in LLMs: New Insights and Methods](reassessing_layer_pruning_in_llms_new_insights_and_methods.md)
- [\[ICLR 2026\] TRAC: Tensor-Train Based Across-Layer Compression for Parameter-Efficient Fine-Tuning](trac_tensor-train_based_across-layer_compression_for_parameter-efficient_fine-tu.md)
- [\[ICLR 2026\] Gradient Intrinsic Dimensionality Alignment：Narrowing The Gap Between Low-Rank Adaptation and Full Fine-Tuning](gradient_intrinsic_dimensionalityalignmentnarrowing_the_gap_between_low-rank_ad.md)
- [\[CVPR 2026\] LoPrune: Efficient Data Pruning for LoRA-Based Fine-Tuning of Vision Transformer](../../CVPR2026/model_compression/loprune_efficient_data_pruning_for_lora-based_fine-tuning_of_vision_transformer.md)
- [\[ICLR 2026\] GPTailor: Large Language Model Pruning Through Layer Cutting and Stitching](gptailor_large_language_model_pruning_through_layer_cutting_and_stitching.md)

</div>

<!-- RELATED:END -->
