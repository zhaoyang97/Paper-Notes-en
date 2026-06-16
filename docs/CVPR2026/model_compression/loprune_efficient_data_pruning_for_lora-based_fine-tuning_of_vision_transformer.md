---
title: >-
  [Paper Note] LoPrune: Efficient Data Pruning for LoRA-Based Fine-Tuning of Vision Transformer
description: >-
  [CVPR 2026][Model Compression][K-FAC] Addressing the overlooked "data redundancy" bottleneck in on-device LoRA fine-tuning, LoPrune proposes projecting sample influence functions onto the LoRA trainable subspace to calculate the TSA Score. By utilizing K-FAC curvature approximation for efficient single-epoch scoring, it reduces fine-tuning overhead by up t
tags:
  - CVPR 2026
  - Model Compression
  - K-FAC
date: 2026-05-08
content_hash: 74bff599dd58e390
---
# LoPrune: Efficient Data Pruning for LoRA-Based Fine-Tuning of Vision Transformer

**Conference**: CVPR 2026  
**论文**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/He_LoPrune_Efficient_Data_Pruning_for_LoRA_Based_Fine_Tuning_of_Vision_Transformer_CVPR_2026_paper_html)  
**Code**: To be confirmed  
**Area**: Model Compression / Efficient Fine-Tuning  
**Keywords**: Data Pruning, LoRA Fine-Tuning, Influence Function, Trainable Subspace, K-FAC

## TL;DR
Addressing the overlooked "data redundancy" bottleneck in on-device LoRA fine-tuning, LoPrune proposes projecting sample influence functions onto the LoRA trainable subspace to calculate the TSA Score. By utilizing K-FAC curvature approximation for efficient single-epoch scoring, it reduces fine-tuning overhead by up to 72.9% and accelerates training by up to 3.69× on models like ViT/DeiT/Swin/DETR, while achieving accuracy gains of up to 3.50%.

## Background & Motivation

**Background**: Vision models are extensively deployed on IoT/edge devices and require continuous on-device fine-tuning to adapt to dynamic changes in lighting, weather, and user behavior. Constrained by computation, memory, and battery life, edge devices typically employ Parameter-Efficient Fine-Tuning (PEFT), with LoRA being the mainstream solution due to its approach of adding low-rank increments to weight matrices while freezing the backbone.

**Limitations of Prior Work**: While LoRA reduces overhead at the "parameter level," **redundancy at the data level persists**. A large number of samples in local datasets contribute minimally to model improvement yet still consume training resources. Although data pruning could theoretically select the most informative subset to save computation, existing pruning methods are poorly suited for direct application in LoRA fine-tuning.

**Key Challenge**: Existing pruning methods follow two paths, both conflicting with LoRA. First, methods based on learning dynamics (EL2N, Forgetting, GraNd) assume that "harder to learn means more important" and require accumulation of statistics over multiple epochs. However, LoRA fine-tuning often converges in a few epochs; if the warm-up is too short, scores become inaccurate (potentially worse than random pruning), while adding extra epochs for logging dynamics defeats the purpose of saving overhead. Second, influence-function-based methods evaluate sample importance in the **full parameter space**. Since LoRA only updates low-rank adapters and freezes the backbone, significant gradient energy is projected onto frozen directions that are never updated, leading to systematic overestimation or misjudgment of sample importance.

**Goal**: To design a low-cost data pruning method **aligned with LoRA's update directions that can be completed within the first epoch**, simultaneously reducing on-device fine-tuning overhead from both data and model perspectives.

**Key Insight**: Since LoRA only updates within a low-rank subspace, sample importance should be measured within this "trainable subspace" rather than the full parameter space—effectively rewarding only those directions that can actually be updated.

**Core Idea**: Use a Jacobian to project the influence function from the full parameter space onto the LoRA subspace to obtain the TSA Score (calculating only updatable directions and reducing dimensionality). Then, use K-FAC curvature approximation to transform the Hessian inversion into near-linear time, enabling single-epoch pruning.

## Method

### Overall Architecture
LoPrune models data pruning as selecting a subset $S(m)$ from the training set $D=\{z_i\}_{i=1}^N$ using a binary vector $m$, minimizing performance loss while removing as many samples as possible. The workflow consists of three steps: First, use a Jacobian to map the sample influence from the full parameter space $\theta$ to the LoRA trainable subspace $\phi$ to obtain the exact TSA Score. Second, use K-FAC approximation to calculate layer-wise scores from the input covariance $\Sigma_x$ and output gradient covariance $\Sigma_\delta$ of each LoRA layer, aggregating them into the final TSA Score. Third, retain high-scoring samples in descending order to form the subset $S(m)$ for fine-tuning. The $x_t$ and $\delta_t$ required for scoring are obtained from standard forward/backward passes without introducing extra computation, allowing completion within a single epoch.

### Key Designs

**1. TSA Score: Projecting Influence Functions to the LoRA Trainable Subspace**

This design directly addresses the issue where full-parameter space scoring includes gradient energy from frozen directions. LoPrune first concatenates all LoRA factors into a trainable parameter vector $\phi = (\mathrm{vec}(A_1), \mathrm{vec}(B_1), \dots)$. Let $\theta(\phi)$ be the equivalent full model weights after injecting low-rank increments, and $J=\partial\,\mathrm{vec}(\theta)/\partial\phi$ be the Jacobian of this mapping. Using the chain rule, gradients and curvature in the subspace can be obtained without explicitly constructing $J$: $g_\phi(z)=J^\top\mathrm{vec}(g_\theta(z))$ and $H_\phi=J^\top H_\theta J$. The exact TSA Score is defined as:

$$s(z) = g_\phi(z)^\top H_\phi^{-1} g_\phi(z)$$

This is equivalent to projecting the influence direction from the full space to the LoRA subspace under the $H_\theta$ metric. This provides two benefits: first, it only rewards directions that are "actually updatable," preventing overestimation of samples that the pre-trained model has already learned well due to large gradients on the frozen backbone. Second, it reduces the inverse Hessian-vector product (iHVP) problem from the full space to the low-rank subspace, avoiding gradient/curvature calculations for frozen weights and significantly reducing overhead. Ablations show that adding this projection (L) improves accuracy by approximately 1.33%–3.38% and reduces pruning time by 82.4%–93.2% compared to not using it.

**2. K-FAC Approximate Scoring: Near-Linear Time Hessian Inversion**

Even in the subspace, directly solving the Hessian is expensive. LoPrune utilizes Kronecker-factored approximate curvature (K-FAC), assuming that cross-layer parameter terms are approximately zero and focusing on intra-layer relations. By leveraging the fact that network gradients can be decomposed into input and output contributions, $H_\theta \approx \Sigma_x \otimes \Sigma_\delta$, where $\Sigma_x=\mathbb{E}[xx^\top]$ is the input activation covariance and $\Sigma_\delta=\mathbb{E}[ \delta\delta^\top]$ is the output gradient covariance. Per the Kronecker product theorem $H_\theta^{-1}=\Sigma_x^{-1}\otimes\Sigma_\delta^{-1}$, the TSA Score is calculated by averaging over all trainable blocks:

$$s(z) = \frac{1}{T}\sum_{t=1}^{T}\left(x_t^\top \Sigma_{x,t}^{-1} x_t\right)\left(\delta_t^\top \Sigma_{\delta,t}^{-1} \delta_t\right)$$

where $t$ indexes all trainable $A$ and $B$ blocks. After substituting the Jacobian mapping, the B-block mapping projects the input side onto the subspace spanned by $A$, and the A-block mapping projects the output side onto the subspace spanned by $B$. Since $x_t$ and $\delta_t$ are already available from backpropagation, the scoring introduces no additional computation. This step reduces scoring to near-linear time, which is key to LoPrune’s "single-epoch, low-overhead" capability. In ablations, L+K reduces time by 91.9%–93.3% compared to L+F (traditional Hessian inversion approximation) with almost no impact on accuracy (~0.05% difference).

## Key Experimental Results

### Main Results
Evaluation across four models and three datasets with a pruning rate of 0.5 (except No pruning). Results report Accuracy / Total Time in seconds (Scoring + Training). Top-1 for classification, AP50 for detection:

| Method | ViT-S/CIFAR100 | ViT-S/Tiny-IN | Swin-T/COCO | DETR/COCO |
|------|----------------|---------------|-------------|-----------|
| No pruning | 90.05 (561.1) | 84.91 (992.1) | 67.18 (5403.1) | 60.23 (7220.2) |
| Random | 88.85 (343.3) | 83.86 (561.4) | 66.11 (3348.2) | 59.13 (4464.3) |
| MoSo 2023 | 89.25 (893.3) | 84.17 (2394.3) | 66.40 (8532.7) | 59.74 (11160.3) |
| TFDP 2025 | 88.96 (544.2) | 83.78 (783.2) | 65.82 (5670.3) | 59.54 (7416.1) |
| **Ours (LoPrune)** | **89.59 (371.5)** | **84.78 (622.1)** | **66.70 (3620.0)** | **59.92 (4804.2)** |

Ours leads in both accuracy and total time: compared to the best baseline, classification Top-1 improves by 0.34%–6.62%, and total overhead is reduced by 1.07×–5.72×. Compared to No pruning, it typically saves 25%–37% of time with only a 0.10%–1.21% drop in accuracy. On Tiny-ImageNet, pruning 50% of data results in only a 0.22% drop in Top-1 while saving 37.4% time.

### Ablation Study
Ablation of TSAS (L) and K-FAC (K) components, where F is the traditional Hessian inversion approximation baseline:

| L | K | F | CIFAR-100 Acc | Time(s) |
|---|---|---|---------------|---------|
| ✓ | - | - | 89.64 | 5542.3 |
| - | ✓ | - | 86.88 | 2114.4 |
| ✓ | - | ✓ | 89.13 | 489.2 |
| - | - | ✓ | 86.21 | 4793.3 |
| ✓ | ✓ | - | **89.59** | **371.5** |

### Key Findings
- **TSAS (L) drives accuracy**: Accuracy drops significantly without L. L+K / L+F improves accuracy by 2.33%–2.92% over K / F and reduces scoring time by 82.5%–89.8%, proving the necessity of "scoring in the trainable subspace."
- **K-FAC (K) drives efficiency**: While it provides minor accuracy improvements, it compresses scoring into near-linear time. L+K is the optimal combination for accuracy and overhead.
- **Advantage is most pronounced at high pruning rates**: At pruning rates above 0.5, LoPrune outperforms other methods by 0.24%–13.72%. At low pruning rates (e.g., 10%), most pruning methods even outperform full training by removing noisy samples with negative contributions.
- Interesting observation: Methods requiring multi-epoch warm-ups like EL2N and GraNd struggle to converge in short-term fine-tuning. At high pruning rates, they are 5.08%–10.34% lower than LoPrune, and their extra scoring overhead often exceeds the savings from pruning.

## Highlights & Insights
- **"Where to score" is more critical than the "scoring function"**: Projecting the influence function into the LoRA subspace alone accounts for most of the accuracy gains. It highlights the overlooked mismatch in the PEFT era—relying on full-parameter space importance metrics.
- **Zero-extra-overhead scoring via training statistics**: $x_t$ and $\delta_t$ are already calculated during training. K-FAC reuses these "free" intermediate values, making "single-epoch pruning" possible.
- **First data pruning specifically designed for LoRA**: This work bridges "data redundancy" and "parameter redundancy" compression for the first time within the PEFT framework, providing a dual-layer path for low-cost on-device fine-tuning.

## Limitations & Future Work
- K-FAC relies on the assumption that cross-layer parameter terms are zero and intra-layer terms are Kronecker-decomposable. The impact of approximation errors under extremely small ranks or unusual covariance structures is not fully discussed.
- Experiments focus on ViT-based vision models and LoRA. Generalization to other PEFT methods (e.g., Adapter, Prefix-tuning) or CNN/multimodal architectures remains to be verified.
- At low pruning rates (<30%), performance gaps between methods are small; LoPrune's advantage is primarily in high-pruning scenarios. Actual savings depend on the acceptable pruning ratio.
- Hyperparameters like the starting pruning epoch affect results (Fig. suggests starting at the 2nd epoch is better) and may require task-specific tuning.

## Related Work & Insights
- **vs EL2N / GraNd / Forgetting**: These rely on multi-epoch learning dynamics. They struggle to converge in short LoRA sessions, and their scoring overhead can exceed training savings. LoPrune scores in a single epoch in near-linear time.
- **vs Optimization-based / MoSo**: These also use influence functions or empirical risk changes but evaluate in the full parameter space with high computational costs. LoPrune projects into the LoRA subspace for both alignment and dimensionality reduction.
- **vs TFDP 2025**: TFDP uses training-free heuristics based on sample complexity (shape/category). LoPrune aligns with actual LoRA update directions, offering more stable accuracy at high pruning rates.

## Rating
- Novelty: ⭐⭐⭐⭐ The perspective of measuring importance in the LoRA subspace is novel and represents the first data pruning method for LoRA.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers four models, three datasets, and both classification and detection tasks with clear ablations and pruning rate scans.
- Writing Quality: ⭐⭐⭐⭐ Solid motivation and derivation; however, the boundaries of K-FAC assumptions are briefly addressed.
- Value: ⭐⭐⭐⭐ Significant for low-cost continuous fine-tuning on edge/IoT devices; the methodology is transferable to other PEFT scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] TaskIT: Memory-Efficient Fine-Tuning of Multi-LoRA LLMs via Cross-Task Importance Transfer](taskit_memory-efficient_fine-tuning_of_multi-lora_llms_via_cross-task_importance.md)
- [\[CVPR 2026\] TAS-LoRA: Transformer Architecture Search with Mixture-of-LoRA Experts](tas-lora_transformer_architecture_search_with_mixture-of-lora_experts.md)
- [\[CVPR 2026\] Mining Attribute Subspaces for Efficient Fine-tuning of 3D Foundation Models](mining_attribute_subspaces_for_efficient_fine-tuning_of_3d_foundation_models.md)
- [\[CVPR 2026\] ThinkingViT: Matryoshka Thinking Vision Transformer for Elastic Inference](thinkingvit_matryoshka_thinking_vision_transformer_for_elastic_inference.md)
- [\[CVPR 2026\] ReFTA: Breaking the Weight Reconstruction Bottleneck in Tensorized Parameter-Efficient Fine-Tuning](refta_breaking_the_weight_reconstruction_bottleneck_in_tensorized_parameter-effi.md)

</div>

<!-- RELATED:END -->
