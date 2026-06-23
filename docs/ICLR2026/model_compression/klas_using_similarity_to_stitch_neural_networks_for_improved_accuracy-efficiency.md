---
title: >-
  [Paper Note] KLAS: Using Similarity to Stitch Neural Networks for Improved Accuracy-Efficiency Tradeoffs
description: >-
  [ICLR 2026][Model Compression][model stitching] KLAS utilizes **KL divergence** to measure the similarity of intermediate representations in pretrained models, automatically selecting optimal "anchor + block pairs" from $O(k^2n^2)$ stitching configurations. It shifts the accuracy-efficiency curve of stitched networks upward at the same fine-tuning cost as baselines
tags:
  - ICLR 2026
  - Model Compression
  - model stitching
  - KL divergence
  - accuracy-efficiency tradeoff
  - many-to-many NAS
  - linear probe
date: 2026-05-08
content_hash: 9f9b17158889f2e2
---
# KLAS: Using Similarity to Stitch Neural Networks for Improved Accuracy-Efficiency Tradeoffs

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=iTaQmRWa7Y](https://openreview.net/forum?id=iTaQmRWa7Y)  
**Code**: To be confirmed  
**Area**: Model Compression / Model Stitching / Neural Architecture Search  
**Keywords**: model stitching, KL divergence, accuracy-efficiency tradeoff, many-to-many NAS, linear probe  

## TL;DR
KLAS utilizes **KL divergence** to measure the similarity of intermediate representations in pretrained models, automatically selecting optimal "anchor + block pairs" from $O(k^2n^2)$ stitching configurations. It shifts the accuracy-efficiency curve of stitched networks upward at the same fine-tuning cost as baselines (ImageNet-1K +1.21% top-1 at the same compute, or 1.33× FLOPs savings at the same accuracy).

## Background & Motivation

**Background**: To address diverse deployment targets from cloud servers to edge devices, flexible model selection within a given compute budget is required. Traditional NAS is computationally expensive; while one-shot/zero-shot NAS reduces costs, they are limited to single design spaces and cannot reuse existing pretrained model libraries. Recent "many-to-many NAS + model stitching" (represented by SN-Net) connects blocks from different pretrained "anchor" models. It uses a lightweight linear stitching layer to feed intermediate activations from one network into another, interpolating a deployable model spectrum across the accuracy-efficiency trade-off at extremely low cost.

**Limitations of Prior Work**: The key to stitching lies in "configuration selection"—choosing which anchors to use and which two blocks to stitch. Even with only two anchors of depth $n$, the configuration space is $O(n^2)$, making exhaustive search infeasible. SN-Net relies on **naive heuristics**: "nearest stitching" for anchors (connecting only models with adjacent complexity) and "paired/unpaired" assumptions for blocks. These heuristics ignore the true compatibility between blocks, resulting in: (1) suboptimal accuracy-efficiency tradeoffs; (2) lack of generality across model families. Paper experiments demonstrate that for Swin, skipping the intermediate S-size and directly performing "distant stitching" between Ti and B actually yields 0.9% higher top-1 accuracy than SN-Net's Ti-S-B at the same FLOPs.

**Key Challenge**: To select superior stitches, one must **explicitly characterize the similarity between the two models being stitched**, yet existing similarity metrics are unreliable. Empirical findings in the paper (Tab.1) show that CKA, MSE, CE, DM, and even SN-Net heuristics achieve a maximum recovery overlap of only 61.1% for the optimal Swin Ti-B configuration, with CKA as low as 5.5%. This is because they only capture either "representational similarity" or "functional similarity," but not both.

**Goal**: Design an **automatic, generalizable, and zero-additional-fine-tuning-cost** framework for stitching configuration selection that satisfies both representational alignment (making the stitching layer easy to learn) and functional alignment (preserving downstream accuracy).

**Key Insight**: **Use KL divergence as a unified similarity metric**. Rooted in information theory, KL divergence measures both the statistical distance of intermediate activation distributions (representational similarity) and the degree of category-discriminative information retention (functional similarity). It naturally achieves two goals at once—addressing the deficiencies of MSE, CE, and CKA respectively.

## Method

### Overall Architecture
KLAS (KL divergence based Anchor Stitching) decouples stitching into two steps: "anchor selection" and "block pair selection." The entire process relies on scores calculated via KL divergence from softmax distributions estimated by a **linear probe** attached after each block. This avoids the need to actually instantiate or train stitching layers, resulting in zero fine-tuning cost for evaluating candidate configurations. Once the candidates are selected, the framework proceeds to the same stitching supernet fine-tuning stage as SN-Net.

```mermaid
flowchart LR
    A[Pretrained Anchor Pool<br/>Swin/DeiT/ResNet...] --> B[ProbeNet: Attach 1×1 linear probes to every block<br/>Jointly train all probes in one pass]
    B --> C[Anchor Selection<br/>Minimum KL divergence between last block softmax distributions → Most compatible]
    C --> D[Block Pair Selection<br/>Stitch score Γ=Ω/Σ + Bucketing + Threshold τ]
    D --> E[Candidate Stitching Config Set S]
    E --> F[Stitching Supernet Fine-tuning 50 epochs<br/>Same settings as SN-Net]
    F --> G[Accuracy-Efficiency Pareto Frontier]
```

### Key Designs

**1. Unified Representational and Functional Similarity via KL Divergence: The "True" Scorer for Stitching.** Given the first $i$ blocks $f_{\le i}$ of source network $f$ and the latter portion $g_{>j}$ of target network $g$, the stitched network is $g_{>j}\circ T\circ f_{\le i}$, where $T$ is the stitching layer mapping source activations $A^f_i$ to target activations $A^g_j$. KLAS recognizes that **stitching success depends on whether the source and target activation distributions are close**. If they are close, $T$ only needs to perform a slight affine transformation (representational alignment), while the source block output retains sufficient category-discriminative information for the target to maintain its original decision boundary (functional alignment). KL divergence characterizes both: a low $D_{KL}$ implies the mapping is easy to learn without losing accuracy. This is the fundamental advantage of KLAS over "representation-only" (CKA) or "task-loss-only" (TLM) optimization.

**2. ProbeNet: Joint Training of All Linear Probes to Reduce Probe Cost to Near Zero.** Calculating the KL divergence for any block pair requires projecting intermediate activations into the output class space. KLAS attaches a $1\times1$ convolutional probe after every block (e.g., 24 for Swin-B, 12 for DeiT-S). However, training them independently is expensive. The paper proposes the **ProbeNet** architecture: it activates only one probe per forward-backward pass, compressing the cost of training 24 probes from $24\times0.25$ GPU-days down to **0.25 GPU-days**. Probes converge rapidly (by the 4th epoch, Fig.3), making it a negligible one-time overhead. After training, the average KL divergence on the validation set is used as the block-pair similarity score:

$$\Theta(P^f_i, P^g_j)=\frac{\sum_{x\in D_v} D_{KL}\!\left(P^f_i(x)\,\|\,P^g_j(x)\right)}{|D_v|}$$

where $P^f_i(x)$ is the softmax distribution of the probe after block $i$.

**3. Anchor Selection via Minimum KL Divergence of the Last Block.** The first step in stitching is choosing which two pretrained models to use as anchors. KLAS directly compares the KL divergence between the softmax distributions of the **last blocks** of different anchors. A lower value indicates more similar decision boundaries and confidence distributions, making them suitable for stitching. This step does not even require training probes (the last block already outputs softmax). Results (Tab.3) prove it provides the correct answer across model families—within the Swin family, Ti-B has the smallest last-block KL ($2.38\times10^{-4}$), corresponding to the optimal curve; however, in the DeiT family, Ti-S/S-B are the smallest. This explains why the "nearest stitching heuristic" is not universal—KLAS adaptively identifies the optimal anchor pairs for different families using a unified criterion.

**4. Block Pair Selection via Stitch Score $\Gamma=\Omega/\Sigma$, Bucketing, and Thresholding.** Once anchors are chosen, KLAS calculates a stitch score $\Gamma(i,j)$ for every block pair $(i,j)$:

$$\Gamma(i,j)=\frac{\overbrace{\Theta(P^f_i, P^g_j)}^{\Omega: \text{Inter-anchor activation distance}}}{\underbrace{\Theta(P^g_j, P^g_{j+1})}_{\Sigma: \text{Intra-anchor block capacity}}}$$

The numerator $\Omega$ measures the magnitude of transformation required for a hypothetical stitching layer—lower is better. The denominator $\Sigma$ measures the representational change within the target anchor between blocks $j \to j+1$; a large $\Sigma$ suggests block $j+1$ has high learning capacity to "absorb" distribution mismatches from the source. A smaller $\Gamma$ is more optimal. To produce a Pareto frontier with controllable density, KLAS partitions the compute space between $f$ and $g$ into several FLOPs buckets $B$. In each bucket, it selects all configurations where $\Gamma$ is below the "inner minimum $\times (1+\tau)$" while ensuring at least the $\arg\min\Gamma$ configuration is preserved for coverage:

$$S=\bigcup_{b\in B} R^*_b,\quad R^*_b=\left\{(i,j)\,|\,\Gamma(i,j)\le\tau\right\}\cup\left\{\arg\min_{(i,j)\in b}\Gamma(i,j)\right\}$$

The threshold $\tau$ balances "frontier sparsity" vs. "noisy configurations," defaulting to 5%.

## Key Experimental Results

Setup: ImageNet-1K / CIFAR-100, anchors including DeiT(Ti/S/B), Swin(Ti/S/B), LeViT, ResNet, and cross-architecture ResNet-Swin; stitched models are fine-tuned for 50 epochs (matching SN-Net). The end-to-end overhead for the Swin family is only 16 hours on 8×A40.

### Main Results

| Task / Family | Metric | SN-Net | KLAS | Gain |
|---|---|---|---|---|
| ImageNet-1K Swin | top-1 at same compute | — | — | **+1.21%** (or **1.33× FLOPs** savings) |
| Swin AUC | ∆AUC | 0.8345 | 0.8950 | **+0.06** |
| DeiT | ∆AUC | — | — | +0.012 |
| LeViT | ∆AUC | — | — | +0.006 |
| ResNet | ∆AUC | — | — | +0.014 |
| ResNet-Swin (Cross-arch) | ∆AUC | — | — | +0.002 |
| Swin CIFAR-100 | ∆AUC | — | — | +0.014 |
| ADE20K Seg (Mask2Former, mIoU) | Same compute bucket | 32.6 | **33.5** | Up to **+0.9% mIoU** |
| LLM (LLaMA 1B→3B, TruthfulQA) | ROUGE-1/2 | ESTA 0.631/0.353 | **0.645/0.379** (fewer params) | +0.017 / +0.033 |

Comparison of similarity metrics (Swin family, AUC, Tab.2): KLAS **0.8950** > SN-Net 0.8345 > CKA 0.8124 > CE 0.8023 > DM 0.7642 > MSE 0.7564. Configuration recovery overlap (Tab.1): KL divergence **88.9%** vs. SN-Net 61.1% vs. CKA only 5.5%.

### Ablation Study

| Threshold $\tau$ | Avg Top-1(%) | AUC | | # Buckets | Avg Top-1(%) | AUC |
|---|---|---|---|---|---|---|
| 1% | 83.72 | 0.8934 | | 10 | 83.65 | 0.8902 |
| 3% | 83.74 | 0.8942 | | 15 | 83.75 | 0.8947 |
| **5%** | **83.76** | **0.8950** | | **20** | **83.76** | **0.8950** |
| 10% | 83.69 | 0.8931 | | | | |

### Key Findings
- $\tau$ too large introduces noisy configurations, while too small leads to sparse frontiers; 5% is optimal. Sensitivity to bucket granularity (10–20) is low, indicating KLAS's robustness.
- KLAS **automatically recovers** configurations considered effective by SN-Net heuristics (28/38 overlap for Swin, 40/66 for DeiT) while **discovering new configurations** (e.g., Swin-Ti→Swin-B "distant stitching" missed by the nearest stitching heuristic).
- Even when **anchors are fixed to SN-Net's Ti-S-B**, KLAS's curve still outperforms SN-Net using only its KL-based block selection (Fig.6)—indicating gains are not solely from anchor selection.

## Highlights & Insights
- **Transforms "where to stitch" from manual heuristics into a computable similarity problem**, noting that the key is satisfying both representational and functional similarity. This dual perspective provides a convincing explanation for why CKA and MSE fail in isolation.
- **Zero fine-tuning cost for candidate evaluation**: Using the softmax distribution from linear probes to calculate KL avoids actual instantiation of stitching layers, which is why KLAS remains inexpensive despite exploring a massive configuration space.
- **Strong Generalization**: Positive gains are observed across ViTs, CNNs, cross-architecture ViT-CNN, and three task categories (classification/segmentation/LLM), proving KL divergence is a universal metric for stitching similarity.
- The design of $\Gamma = \Omega/\Sigma$ is intuitive: the numerator captures "stitching ease" while the denominator captures "target block mismatch tolerance," encoding both stitching difficulty and target capacity into a single scalar.

## Limitations & Future Work
- Absolute gains are relatively small: most family ∆AUC gains are between +0.006 and +0.014; the cross-architecture ResNet-Swin is only +0.002, where some improvements are close to the noise margin.
- The stitching direction is fixed as "Source Complexity < Target Complexity," without exploring reverse stitching or multi-source fusion.
- LLM experiments are limited to LLaMA 1B→3B and a single dataset (TruthfulQA); the scale and breadth are limited.
- The paper notes that generative workloads have dual-phase latency constraints (TTFT, TBT) and KV-cache compatibility issues that current similarity metrics do not consider, marking a clear direction for future work.

## Related Work & Insights
- **Stitching Genealogy**: From being used as a representation analysis tool by Lenc & Vedaldi (2015) and Bansal (2021), to SN-Net applying it to NAS (reducing training costs by 22×), and recently ESTA/StitchLLM porting heuristics to LLMs—KLAS is the first to treat "configuration selection" as a principled similarity problem.
- **Response to Negative Findings on Similarity Metrics**: Responding to Csiszárik (2021) and Balogh & Jelasity (2025)—who found common metrics like CKA do not correlate with stitching accuracy and advocated for task loss matching—KLAS proves that by switching to KL divergence (which encodes functional information), similarity metrics can indeed predict stitching quality.
- **Insight**: The approach of "zero-cost compatibility assessment via probes + distribution divergence" can be migrated to model merging, early exiting/cascading, modular reuse, or any scenario requiring a judgment on whether two network segments can be interfaced.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Reconstructs stitching configuration selection as a KL divergence similarity problem with a clear dual-similarity (representation + functional) perspective, effectively addressing previous negative conclusions about similarity metrics.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers ViT/CNN/Cross-architecture + Classification/Segmentation/LLM + Multiple datasets with complete ablations; however, gains in some families are small and LLM validation is somewhat lean.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation progresses logically (heuristic failure → metric failure → KL as a dual solution), with well-aligned formulas and figures.
- **Value**: ⭐⭐⭐⭐ Provides a universal, low-cost, and generalizable configuration selector for "reusing pretrained model libraries for flexible deployment," making it highly engineering-friendly.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Adaptive Width Neural Networks](adaptive_width_neural_networks.md)
- [\[ICML 2026\] Partial Fusion of Neural Networks: Efficient Tradeoffs Between Ensembles and Weight Aggregation](../../ICML2026/model_compression/partial_fusion_of_neural_networks_efficient_tradeoffs_between_ensembles_and_weig.md)
- [\[ICLR 2026\] A Recovery Guarantee for Sparse Neural Networks](a_recovery_guarantee_for_sparse_neural_networks.md)
- [\[ICLR 2026\] Fine-tuning Quantized Neural Networks with Zeroth-order Optimization](fine-tuning_quantized_neural_networks_with_zeroth-order_optimization.md)
- [\[ICLR 2026\] BEP: A Binary Error Propagation Algorithm for Binary Neural Networks Training](bep_a_binary_error_propagation_algorithm_for_binary_neural_networks_training.md)

</div>

<!-- RELATED:END -->
