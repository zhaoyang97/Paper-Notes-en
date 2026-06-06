---
title: >-
  [Paper Note] Elastic ViTs from Pretrained Models without Retraining
description: >-
  [NeurIPS 2025][Model Compression][Vision Transformer Pruning] SnapViT proposes a post-training structured pruning method that combines a local Hessian diagonal approximation derived from self-supervised gradients with gl…
tags:
  - "NeurIPS 2025"
  - "Model Compression"
  - "Vision Transformer Pruning"
  - "Elastic Inference"
  - "Hessian Approximation"
  - "Evolutionary Algorithms"
  - "Self-Supervised Importance Scoring"
date: 2026-05-08
content_hash: e8d28fbbc80e3349
---

# Elastic ViTs from Pretrained Models without Retraining

**Conference**: NeurIPS 2025
**arXiv**: [2510.17700](https://arxiv.org/abs/2510.17700)  
**Code**: [elastic.ashita.nl](https://elastic.ashita.nl/)  
**Area**: Model Compression / Structured Pruning
**Keywords**: Vision Transformer Pruning, Elastic Inference, Hessian Approximation, Evolutionary Algorithms, Self-Supervised Importance Scoring

## TL;DR
SnapViT proposes a post-training structured pruning method that combines a local Hessian diagonal approximation derived from self-supervised gradients with global cross-module correlations estimated via evolutionary algorithms. Without any retraining or labels, it generates elastic ViT sub-networks spanning continuous sparsity levels in a single run, requiring less than 5 minutes on an A100 GPU.

## Background & Motivation

**Background**: Powerful vision foundation models are typically released in only a few fixed sizes (e.g., DINOv3 from 21M to 6.7B parameters), forcing users to select "the largest model that fits within their constraints," which frequently leads to suboptimal deployment.

**Limitations of Prior Work**: (a) Knowledge distillation requires a predefined target architecture and typically non-public pretraining data; (b) elastic inference methods (Matryoshka/Matformer) require nested structures to be designed during pretraining and cannot be applied to existing models; (c) existing pruning methods target specific computational budgets and tasks, generally require retraining, and can only optimize a single sparsity level per run.

**Key Challenge**: Diagonal (or block-diagonal/K-FAC) Hessian approximations capture only local, intra-layer dependencies, neglecting cross-module correlations between layers—while the full Hessian contains $N^2$ elements and is computationally intractable.

**Goal**: Extract a family of sub-networks spanning continuous sparsity levels from any pretrained ViT, without retraining, without labels, and at minimal computational cost.

**Key Insight**: Decompose the Hessian into a local term (diagonal approximation via self-supervised gradients) and a global term (cross-module correlations learned by an evolutionary algorithm), combining both into a unified pruning score.

**Core Idea**: Self-supervised gradients provide local sensitivity estimates; the xNES evolutionary algorithm learns global cross-module correlations—together they yield an elastic model covering all sparsity levels in a single run.

## Method

### Overall Architecture
Input: any pretrained ViT → Step 1: compute parameter-wise squared gradients (local Hessian diagonal approximation) using the DINO self-supervised loss on a small set of unlabeled data → Step 2: optimize global structural scaling factors via the xNES evolutionary algorithm (global Hessian approximation) → Step 3: multiply the two terms to obtain a unified pruning score → globally rank scores and prune to any target sparsity.

### Key Designs

1. **Local Hessian Approximation (Self-Supervised Gradients)**:

    - **Function**: Estimate the local sensitivity of each parameter.
    - **Mechanism**: $H^{(l)} \approx \frac{1}{N_D}\sum_{i=1}^{N_D} \|\nabla_\theta \mathcal{L}_i\|^2$, retaining only the diagonal of the Hessian. The DINO self-supervised objective $\mathcal{L}^{\text{SSL}} = \sum_k \sum_m \mathcal{L}_{\text{CE}}(z_k^g, z_m^l)$ (cross-view consistency loss between global and local crops) is used, requiring no classification head.
    - **Design Motivation**: The self-supervised loss makes the method applicable to any model (with or without a classification head) and generalizes well to downstream tasks.

2. **Global Hessian Estimation (xNES Evolutionary Algorithm)**:

    - **Function**: Capture cross-module correlations among attention heads and FFN blocks.
    - **Mechanism**: A search distribution $\mathcal{N}(\mu, \Sigma)$ is parameterized with $\Sigma = BB^T$ (where $B = e^A$). At each iteration, a structure-level scaling vector $c$ is sampled, multiplied with the local scores, and used for pruning. Fitness is evaluated without labels via cosine similarity of PCA-projected embeddings from the original and pruned models. The xNES natural gradient update drives $\Sigma^{-1}$ to approximate the cross-module Hessian: $H^{(g)} \approx \alpha \Sigma^{-1}$.
    - **Design Motivation**: Directly computing the structure-level Hessian remains intractable, but xNES implicitly models it through black-box optimization—contracting variance along sensitive directions and expanding it along flat ones.
    - **Fitness Function**: $F = \frac{1}{|\mathcal{S}|} \sum_{s \in \mathcal{S}} \text{sim}(\text{PCA}(z), \text{PCA}(z_{p_s}))$, evaluated across multiple sparsity levels $s$.

3. **Elastic Single-Run Pruning**:

    - **Function**: Produce sub-networks at all sparsity levels from a single computation.
    - **Mechanism**: The unified score is $P = \text{diag}(\frac{1}{N_D}\sum_i \|\nabla_\theta \mathcal{L}^{\text{SSL}}\|^2) \odot Mc$, where $M \in \{0,1\}^{N \times B}$ is a membership matrix that broadcasts module-level factors to parameter level. After global ranking, the sub-network at sparsity $S$ is defined as $\Theta_S = \{\theta_i | \text{rank}(P_i) < |\Theta|(1-S)\}$.
    - **Design Motivation**: A single evolutionary optimization covers all sparsity levels, whereas each baseline method requires an independent run per target sparsity.

### Loss & Training
No model weight training or fine-tuning is required. xNES runs for 50–500 iterations (more for large-scale pretrained models), evaluating on a small set of unlabeled images per iteration. Total runtime is less than 5 minutes on an A100 GPU.

## Key Experimental Results

### Main Results (DINO ViT-B/16, k-NN + Linear, 7-dataset average)

| Sparsity | Method | Avg k-NN | Avg Linear | Notes |
|----------|--------|----------|------------|-------|
| 0% | Unpruned | 69.1 | 72.0 | Original model |
| 40% | SnapViT (Ours) | ~65 | ~68 | <5% accuracy drop, 1.58× speedup |
| 40% | SNIP Magnitude | ~56 | ~57 | Significantly behind |
| 40% | LAMP | ~45 | ~42 | Severe degradation |
| 50% | SnapViT (Ours) | 63.5 | — | Effect of modeling FFN+head interactions |
| 50% | Only FFN (12 interactions) | 60.1 | — | FFN only |
| 50% | None (0 interactions) | 56.6 | — | No global correlations |

### Ablation Study

| Dimension | Configuration | Key Result |
|-----------|---------------|------------|
| Global interactions | 0 interactions | 56.6% k-NN (50% sparsity) |
| Global interactions | FFN only (12) | 60.1% (+3.5) |
| Global interactions | FFN + heads (156) | **63.5%** (+6.9) |
| Evolutionary iterations | 50 iter | 42.2% Linear (50% sparsity) |
| Evolutionary iterations | 500 iter | **44.0%** (+1.8) |
| Number of optimized sparsities | 1 | Slightly lower than 6 |
| Number of optimized sparsities | 6 | More robust |
| Loss function | SSL vs. CE | SSL only marginally below CE |

### ImageNet-1k Full Fine-Tuning (DeiT ViT-B/16, 50% sparsity, 300 epochs)

| Method | Avg k-NN | Avg Linear | ImageNet-1k |
|--------|----------|------------|-------------|
| Unpruned | 75.8 | 78.5 | 81.8 |
| NViT | 73.7 | 72.0 | **83.3** |
| SnapViT (Ours) | **75.4** | **75.9** | 82.6 |

SnapViT approaches NViT on ImageNet while substantially outperforming it in 7-dataset generalization (k-NN +1.7%, Linear +3.9%).

### Key Findings
- **Cross-module correlations are critical**: Increasing from 0 to 156 interactions improves k-NN by 6.9%, demonstrating that diagonal Hessian approximations are severely insufficient.
- Large-scale pretrained models (DINOv3/SigLIPv2) are harder to prune—training on massive data distributes representations uniformly across parameters.
- Pruning naturally removes FFN neurons first (especially in deeper blocks 8–12), while attention heads are more robust.
- The performance gap between self-supervised and supervised gradients is negligible, making the method truly label-free.
- A single weight correction step (SparseGPT-style) substantially recovers performance at extreme sparsity levels.

## Highlights & Insights
- **Evolutionary algorithm as a global Hessian proxy**: The approach circumvents the intractability of explicitly computing $N^2$ Hessian elements by implicitly learning inter-module correlations through black-box fitness evaluation—this constitutes the central contribution.
- **Elastic single-run pruning**: All baselines require one independent run per target sparsity, whereas a single xNES optimization in SnapViT covers all sparsity levels simultaneously—a practically significant advantage.
- **Self-supervision enables generality**: The DINO objective decouples the scoring from classification heads, enabling effective pruning of foundation models such as DINOv3 and SigLIPv2 for the first time.

## Limitations & Future Work
- Large-scale pretrained models (SigLIPv2/DINOv3) suffer sharp performance degradation beyond 30% sparsity; weight correction partially mitigates this but does not fully resolve it.
- The number of xNES iterations requires manual tuning (50 for small models, 500 for large-scale pretraining), with no automatic stopping criterion.
- Validation is limited to ViT architectures; applicability to CNNs and hybrid architectures remains unexplored.
- Pruning granularity is at the FFN row-column and full attention head level; finer granularity (e.g., channel-level) may yield better efficiency–accuracy trade-offs.
- The fitness function relies on cosine similarity of PCA-projected embeddings, which may discard high-dimensional structural information.

## Related Work & Insights
- **vs. LAMP**: Magnitude-based pruning that degrades severely on self-supervised models (approximately 21% below SnapViT at 50% sparsity). The root cause is the absence of global dependency modeling.
- **vs. LLM Surgeon**: Performs 5-shot iterative pruning with weight correction and requires a classification head. SnapViT matches or exceeds it in a single label-free run.
- **vs. Matformer**: Requires nested structures to be incorporated during pretraining. SnapViT is directly applicable to any existing pretrained model.
- **vs. NViT**: Requires 300 epochs of full fine-tuning to surpass SnapViT; SnapViT approaches its ImageNet performance without any training and generalizes better after fine-tuning.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The combination of xNES-based global Hessian approximation, self-supervised scoring, and single-run elastic pruning is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 8 datasets, 5 model families, and 3 protocols (k-NN / Linear / Segmentation), with comprehensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ The methodological derivation is well-structured, though notation is dense in places.
- **Value**: ⭐⭐⭐⭐⭐ Directly practical for ViT deployment; generating elastic models in under 5 minutes is immediately adoptable in industry settings.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Understanding Differential Transformer Unchains Pretrained Self-Attentions](understanding_differential_transformer_unchains_pretrained_self-attentions.md)
- [\[NeurIPS 2025\] AutoJudge: Judge Decoding Without Manual Annotation](autojudge_judge_decoding_without_manual_annotation.md)
- [\[NeurIPS 2025\] Robustifying Learning-Augmented Caching Efficiently without Compromising 1-Consistency](robustifying_learning-augmented_caching_efficiently_without_compromising_1-consi.md)
- [\[NeurIPS 2025\] Deterministic Continuous Replacement: Fast and Stable Module Replacement in Pretrained Transformers](deterministic_continuous_replacement_fast_and_stable_module_replacement_in_pretr.md)
- [\[ICCV 2025\] EA-ViT: Efficient Adaptation for Elastic Vision Transformer](../../ICCV2025/model_compression/ea-vit_efficient_adaptation_for_elastic_vision_transformer.md)

</div>

<!-- RELATED:END -->
