---
title: >-
  [Paper Note] Partial Fusion of Neural Networks: Efficient Tradeoffs Between Ensembles and Weight Aggregation
description: >-
  [ICML 2026][Model Compression][Model Fusion] The authors propose **Partial Fusion**: using partial optimal transport (partial OT) to merge only the "most similar" neurons in two networks while keeping the remaining neuro…
tags:
  - "ICML 2026"
  - "Model Compression"
  - "Model Fusion"
  - "Ensemble"
  - "Partial Optimal Transport"
  - "Generalized Pruning"
  - "Neuron Similarity"
date: 2026-05-08
content_hash: aab8e5fe8877c6c7
---

# Partial Fusion of Neural Networks: Efficient Tradeoffs Between Ensembles and Weight Aggregation

**Conference**: ICML 2026  
**arXiv**: [2605.22350](https://arxiv.org/abs/2605.22350)  
**Code**: https://github.com/Fabian-Mor/partial_fusion_nn  
**Area**: Model Compression  
**Keywords**: Model Fusion, Ensemble, Partial Optimal Transport, Generalized Pruning, Neuron Similarity  

## TL;DR
The authors propose **Partial Fusion**: using partial optimal transport (partial OT) to merge only the "most similar" neurons in two networks while keeping the remaining neurons independent. This creates a smooth, monotonic, and adjustable accuracy–parameter curve between "weight aggregation ($1\times$ parameters)" and "full ensemble ($2\times$ parameters)." It is further unified into a "generalized pruning of ensembles" perspective, allowing the same toolkit to compress individual models.

## Background & Motivation

**Background**: There are two main approaches to combining multiple independently trained neural networks. One is the **ensemble**: retaining all models and averaging their outputs during inference. This is robust and accurate but has parameter counts and inference times that scale linearly with the number of models. The other is **weight aggregation / model fusion**: aligning neurons of different networks through permutation alignment (Git Re-Basin, Ainsworth et al. 2023) or optimal transport alignment (OT Fusion, Singh & Jaggi 2020) and then averaging the weights to obtain a fused model of the same size as a single network.

**Limitations of Prior Work**: These two routes stand at opposite ends of the curve, leaving the **middle region empty**. Ensembles are expensive but accurate; fusion is cheap but often less accurate than ensembles. Especially when networks are trained on heterogeneous data slices and neuron functions differ significantly, forcing all neurons to pair up averages "functionally non-overlapping neurons," causing unnecessary accuracy loss.

**Key Challenge**: existing methods only offer a choice between the two extremes of "full pairing (fusion)" or "full retention (ensemble)." However, the most informative aspect is the **neuron-level variance**: identifying which neurons are truly similar and worth merging, and which have unique functions and should remain separate. Utilizing this variance could draw a continuous, smooth Pareto curve between accuracy and parameter count.

**Goal**: (1) Provide a fusion method that can interpolate arbitrarily between weight aggregation and ensembles; (2) Place it within a more general "generalized pruning of ensembles" framework to prove they are the same concept; (3) Apply the same framework back to single model compression.

**Key Insight**: Observing that leaving the "least similar neurons" unmerged causes the average similarity of the "merged part" to significantly increase (Appendix L). Thus, **pairing only the most similar subset** and keeping others as independent branches recovers substantial accuracy with few extra parameters.

**Core Idea**: Use **partial optimal transport (partial OT)** to match only a $(1-\alpha)$ proportion of neuron mass. The remaining $\alpha$ proportion of neurons is retained as independent channels in the fused network, resulting in a "partially fused network" of size $(1+\alpha)\times$ the single network—where $\alpha=0$ is OT Fusion and $\alpha=1$ is an ensemble.

## Method

### Overall Architecture
Input: Two $L$-layer feedforward networks $A$ and $B$ with weights $W_\ell^A, W_\ell^B$, an interpolation factor $\lambda \in [0,1]$, and an "independent retention ratio" $\alpha \in [0,1]$. Output: A **partial fusion network** $C$, with layers sized between a single network ($\alpha=0$) and an ensemble ($\alpha=1$). The process follows four steps: (i) Extract feature vectors (activations or weight columns) for neurons at each layer $\ell$ and assign probability distributions $\mu_\ell^A, \mu_\ell^B$; (ii) Solve $\alpha$-partial OT to obtain a partial coupling matrix $\tilde\pi_\ell^{A,B}$, identifying "matched" and "independent" neurons; (iii) Weight-merge the matched parts as in OT Fusion and retain independent parts as new channels; (iv) Assemble the layer weights $W_\ell^C$ into a 3-block structure: independent blocks for $A$ (top-left) and $B$ (bottom-right), a fused block (center), and stochastic kernels $K_\ell^{A\to B}, K_\ell^{B\to A}$ derived from OT to stitch the transitions.

### Key Designs

1.  **Partial OT Fusion**:
    - **Function**: Determines which neurons at each layer are merged versus left independent and provides the alignment matrix for merging.
    - **Mechanism**: Views neurons as support points in probability measures, with similarity measured by Euclidean distance of feature vectors. While standard OT Fusion requires the coupling $\pi$ to transport $\mu^A$ entirely to $\mu^B$, this work uses $\alpha$-partial OT, relaxing constraints to $\sum_j \tilde\pi[i,j]\le\mu^A[i]$ and $\sum_{i,j}\tilde\pi[i,j]=1-\alpha$, solving $\tilde\pi_\ell^{A,B}=\arg\min_{\pi\in\Pi_\alpha}\int\|x-y\|^2\,\pi(dx,dy)$. Neurons corresponding to untransported mass are "independent"; matched parts are normalized to form stochastic kernels via $K_\ell^{A\to B}=(\pi^{A,B})^T/\mu^A$ and $K_\ell^{B\to A}=\pi^{A,B}/\mu^B$. Partial OT is computationally equivalent to standard OT.
    - **Design Motivation**: Avoids forced merging of functionally dissimilar neurons and turns "merging" into a continuous knob $\alpha$, making the tradeoff between model size and accuracy smooth and controllable at layer granularity.

2.  **3-block Fusion Layer + Global Fixed-point Alignment**:
    - **Function**: Unifies "independent retention + weighted fusion" into a single weight matrix and jointly optimizes alignment across layers.
    - **Mechanism**: The weight matrix $W_\ell^C$ from layer $\ell$ to $\ell+1$ is partitioned into a $3\times 3$ block structure (corresponding to $A$-independent, fused, and $B$-independent). Diagonal blocks either copy original weights or fuse them via $W_\ell^C=(1-\lambda)W_\ell^B+\lambda K_{\ell+1}^{A\to B} W_\ell^A[F,F] K_\ell^{B\to A}$, while off-diagonal blocks facilitate transitions between independent and fused branches using $K$. Following Ainsworth et al. (2023), alignment is formulated as a global objective $(\pi_\ell^{A,B})_\ell=\arg\min\sum_\ell\int\|x-y\|^2\pi_\ell(dx,dy)$, solved via **fixed-point iteration**—updating one $\pi_\ell$ at a time while freezing others, maintaining a (partial) OT problem at each step.
    - **Design Motivation**: When features are defined by weight columns, alignment at layer $\ell$ depends on layer $\ell+1$; joint solving is necessary. Fixed-point iteration achieves higher accuracy than greedy approaches (Figure 5(a) vs 5(b)). The 3-block structure is a clean way to represent ensembles and fusion simultaneously in tensors.

3.  **Generalized Pruning via Clustering**:
    - **Function**: Reconciles "pruning an ensemble" with "pruning a single network," providing a form of pruning that allows for linear combinations of neurons rather than just deletion.
    - **Mechanism**: For a large model $E$ (e.g., an ensemble or any over-parameterized network), stochastic kernels $K_\ell^{E\to S}\in\mathbb{R}^{n_\ell^S\times n_\ell^E}$ and $K_\ell^{S\to E}$ are introduced to map to a small model $S$, defining $W_\ell^S:=K_{\ell+1}^{E\to S} W_\ell^E K_\ell^{S\to E}$. If $K$ is a 0/1 row/column stochastic matrix, this reduces to standard pruning. This work generalizes it to kernels derived from **clustering**: solving $\pi_\ell^{E,S}=\arg\min_{\pi\in\Pi(\mu^E,*_m)}\int\|x-y\|^2\,\pi(dx,dy)$, where the second marginal is supported on at most $m$ centroids (K-means for uniform $\mu^E$). Neurons within a cluster are linearly combined into one centroid to form $S$. Partial fusion is proven to be a special case of generalized pruning applied to the $A\oplus B$ ensemble with inductive biases (merging only cross-network neurons, maximum pairwise combinations).
    - **Design Motivation**: Standard pruning only deletes (losing processing steps), while OT post-processing only merges (blurring processing steps). These error sources vary with compression rates. Clustering-based generalized pruning allows both operations and balances them automatically, significantly outperforming both on MLP-on-MNIST (Figure 1(b)). In practice, Lloyd's K-means performs poorly in relevant compression ranges; **hierarchical clustering** is used for near-global optimal solutions.

### Loss & Training
The entire process is **training-free**: partial fusion and generalized pruning are post-hoc weight reorganizations. Only in comparison experiments (Section 3.2), **slight fine-tuning** is performed using 1% of MNIST or 5% of CIFAR-10 data to verify the further potential of fused models. $\alpha$ and $\lambda$ are the core adjustable parameters: $\lambda$ controls the relative weight of the two networks in fused blocks, and $\alpha$ controls the "independent retention ratio," both in $[0,1]$.

## Key Experimental Results

### Main Results

Accuracy comparison under heterogeneous data slices + fine-tuning (Table 1):

| Model / Data | $\alpha=0.0$ (OT Fusion) | $\alpha=0.4$ | $\alpha=0.5$ | $\alpha=0.8$ | $\alpha=1.0$ (Ensemble) | Single Model A / B |
|---|---|---|---|---|---|---|
| MLP/MNIST Fusion (No FT) | 84.1 | 87.4 | 87.5 | 87.9 | 88.1 | 93.8 / 87.8 |
| MLP/MNIST FT (1% data) | 95.1 | 96.1 | 96.2 | 96.5 | 96.5 | 93.8 / 87.8 |
| ResNet-18/CIFAR-10 Fusion | 66.4 | 83.4 | 87.4 | 90.6 | 91.3 | 79.8 / 76.7 |
| ResNet-18/CIFAR-10 FT (5%) | 85.3 | 90.0 | 90.3 | 91.4 | 91.8 | 79.8 / 76.7 |

Key Observations: (i) Accuracy rises **monotonically** with $\alpha$ without abrupt jumps, validating that partial fusion provides a truly continuous interpolation between OT Fusion and ensembles; (ii) On ResNet-18, $\alpha=0.4$ recovers 83.4%, significantly higher than OT Fusion's 66.4% while using only $\approx 1.4\times$ the parameters of a single network; (iii) After fine-tuning, all $\alpha$ configurations outperform single models, suggesting independent neurons provide capacity for the optimizer to bridge distribution shifts quickly.

### Ablation Study

Relative performance of partial fusion / generalized pruning on MLP-MNIST splits (Qualitative summary from Figures 5, 6):

| Configuration | Key Phenomenon | Explanation |
|------|---------|------|
| Partial OT (weight feat., fixed-point) | Highest, most monotonic accuracy | Global joint alignment is optimal; recommended config |
| Partial OT (weight feat., greedy) | Slightly below fixed-point | Greedy single-layer misses cross-layer coupling |
| Partial OT (activation feat.) | One tier lower than weight-based | Activation features are noisier |
| Generalized Pruning (Clustering) | Better than Partial OT at large $\alpha$ | Higher precision via NP-hard clustering |
| Unstructured Pruning of Ensemble | **Non-monotonic**, beats ensemble at $\lambda{=}0.3$ | Double scaling of weights in output + saliency; equivalent $\lambda$ shift |
| Single VGG11 Pruning (clustering vs OT) | Clustering wins slightly | Advantage narrows when blurring error dominates |
| Fixed 1/2/3 layers as ensemble | Significant boost for all methods | Huge variance in "fusibility" across layers |

### Key Findings
- **Core Mechanism Validated**: Keeping the few most dissimilar neurons independent significantly increases the average similarity of the merged portion (Appendix L), which is the root cause of efficiency in partial fusion and clustering pruning.
- **Task-Dependent "Winners"**: On CNNs, the inductive bias of partial OT (cross-network merging, pairwise combinations) is more effective; on MLPs, more flexible clustering pruning wins.
- **Significant Layer Heterogeneity**: Keeping wide layers as ensembles while using partial fusion for narrow layers allowed VGG11 to outperform single models with only a 38% increase in channels. This suggests future work should focus on **automated per-layer $\alpha$**.
- **Anomalies**: Unstructured ensemble pruning showed non-monotonic curves, attributed to implementations where $\lambda$ scales both output weights and neuron importance simultaneously.

## Highlights & Insights
- **Unified Perspective**: Framing model fusion, ensembles, pruning, and post-processed pruning within a framework that maps a large network $E$ to a small network $S$ using a stochastic kernel $K$. Partial fusion is a specific case with added inductive bias. This redefines pruning from "selecting 0/1 matrices" to "linear combinations as a valid atomic operation."
- **Precise Application of Partial OT**: Existing OT Fusion's requirement to transport all mass is a mathematical constraint rather than a problem requirement; releasing the $\alpha$ portion perfectly matches the "voluntary mismatch" intent without increasing solving complexity.
- **Deletion vs. Blurring Dichotomy**: The authors categorize pruning errors into "losing steps (deletion)" and "blurring steps (linear combination)," noting that clustering pruning balances these automatically by supporting both operations.
- **Layer Heterogeneity Hint**: The observation that fixing just 3 layers as ensembles provides massive gains implies that for LLMs/ViTs, the value of this work lies in adaptive $\alpha$ per layer rather than a global $\alpha$.

## Limitations & Future Work
- **Scale Constraints**: Experiments are limited to small architectures (MLP, VGG11, ResNet-18) and small datasets (MNIST, CIFAR-10). Scalability to ViTs/LLMs is unverified; clustering pruning involves NP-hard paths that may be prohibitive for billion-parameter models.
- **Limited Similarity Metrics**: Partial OT and clustering rely on Euclidean distance and permutation invariance. The authors acknowledge that richer metrics like CCA or Procrustes might be more appropriate.
- **Lack of Automated $\alpha$**: Selecting which layers to remain as ensembles was manual. A principled criterion for "layer fusibility" is needed for scaling.
- **Non-monotonicity in Unstructured Pruning**: Highlighting risks where weights are scaled in multiple places simultaneously, requiring careful semantic alignment of scaling factors in future generalizations.

## Related Work & Insights
- **vs. OT Fusion (Singh & Jaggi 2020)**: This is a strict generalization (equivalent at $\alpha=0$) and introduces the fixed-point joint optimization from Ainsworth et al. (2023) to OT-based methods for the first time.
- **vs. Git Re-Basin (Ainsworth et al. 2023)**: While Git Re-Basin is restricted to permutation matrices (same-size layers), this method uses stochastic kernels to handle heterogeneous layers, unequal widths, and partial matches.
- **vs. ZipIt! (Stoica et al. 2024)**: Shares the "concatenate then merge" philosophy; this work formalizes it specifically as "generalized pruning of an ensemble" and identifies the exact inductive bias of partial fusion.
- **vs. Luenam et al. 2025**: Uses similar clustering-based aggregation for multi-network merging; this work explicitly places it in a pruning framework and demonstrates that clustering acts as an excellent pruning kernel.

## Rating
- Novelty: ⭐⭐⭐⭐ Significant framework contribution by unifying partial OT, fusion, and generalized pruning.
- Experimental Thoroughness: ⭐⭐⭐ Multiple tasks and backbones, but limited to small-scale scenarios; lacks large model validation.
- Writing Quality: ⭐⭐⭐⭐ Good pace; 3-block weight diagrams make abstract designs intuitive.
- Value: ⭐⭐⭐⭐ Provides a unified coordinate system for "merging ↔ ensemble ↔ pruning." High theoretical value; engineering value depends on scalability to LLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SURGE: Surrogate Gradient Adaptation in Binary Neural Networks](surge_surrogate_gradient_adaptation_in_binary_neural_networks.md)
- [\[ICLR 2026\] Adaptive Width Neural Networks](../../ICLR2026/model_compression/adaptive_width_neural_networks.md)
- [\[NeurIPS 2025\] Synergy between the Strong and the Weak: Spiking Neural Networks Are Inherently Superior in Temporal Processing](../../NeurIPS2025/model_compression/synergy_between_the_strong_and_the_weak_spiking_neural_networks_are_inherently_s.md)
- [\[ICML 2026\] Quantifying the Uncertainty of Foundation Models with Singular Value Ensembles](quantifying_the_uncertainty_of_foundation_models_with_singular_value_ensembles.md)
- [\[AAAI 2026\] Explore and Establish Synergistic Effects between Weight Pruning and Coreset Selection](../../AAAI2026/model_compression/explore_and_establish_synergistic_effects_between_weight_pruning_and_coreset_sel.md)

</div>

<!-- RELATED:END -->
