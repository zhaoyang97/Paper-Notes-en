---
title: >-
  [Paper Note] LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models
description: >-
  [ICML 2026][Self-Supervised Learning][Tabular Foundation Models] Addressing two critical pathologies in tabular foundation models like TabPFN-v2—severe low-rank collapse in shallow layers and negligible contribution of s…
tags:
  - "ICML 2026"
  - "Self-Supervised Learning"
  - "Tabular Foundation Models"
  - "RBF Embedding"
  - "Low-Rank Collapse"
  - "Bidirectional Attention"
  - "Tabular ICL"
date: 2026-05-08
content_hash: 8fcee706b061dd39
---

# LimiX-2M: Mitigating Low-Rank Collapse and Attention Bottlenecks in Tabular Foundation Models

**Conference**: ICML 2026  
**arXiv**: [2606.04485](https://arxiv.org/abs/2606.04485)  
**Code**: https://github.com/limix-ldm-ai/LimiX  
**Area**: Tabular Foundation Models / Transformer Architecture / Numerical Embedding  
**Keywords**: Tabular Foundation Models, RBF Embedding, Low-Rank Collapse, Bidirectional Attention, Tabular ICL

## TL;DR
Addressing two critical pathologies in tabular foundation models like TabPFN-v2—severe low-rank collapse in shallow layers and negligible contribution of sample-attention in the final layer to prediction signals—the authors propose using Radial Basis Functions (RaBEL) to expand each scalar into a set of local responses, unlocking "value-direction" degrees of freedom. Furthermore, they reorder bidirectional attention blocks from F→S→N to S→N→F to ensure all attention paths flow into the readout. With only 2M parameters, this approach consistently outperforms the 7M TabPFN-v2 and 27M TabICL on major tabular benchmarks.

## Background & Motivation

**Background**: Tabular Foundation Models (TFMs) such as TabPFN, TabPFN-v2, TabICL, and LimiX redefine tabular learning as "in-context inference" by pre-training Transformers on synthetic tasks. They have pushed long-standing leaders like Gradient Boosting Decision Trees (XGBoost, LightGBM, CatBoost) into the second tier on several small-to-medium scale benchmarks. The standard practice involves projecting each numerical cell $x_{i,j}$ into a latent space using a $1\times p$ linear layer, then superimposing column IDs or positional embeddings.

**Limitations of Prior Work**: The authors performed SVD on the latent states of each layer in TabPFN-v2 using OpenML-CC18 and discovered extremely severe low-rank collapse. In a 192-dimensional latent space across 12 layers (36 modules), merely 5–10 singular components often retain 95% of the energy. Truncated SVD on the original 192-dimensional input suggests that compressing the rank to 50 (~25%) results in almost no AUC drop, and even at a rank of 20, competitive AUC is maintained (0.8985 vs 0.9177). This indicates that most of the latent space is wasted.

**Key Challenge**: The authors prove (Proposition 3.1) that under pure linear embedding and column IDs, given $n$ scalars $x_1,\dots,x_n\in\mathbb{R}$, the rank of the embedding matrix is **at most 2**. After single-head self-attention, the rank remains at most 2, and multi-head attention can only increase it to $H+1$. Column IDs or positional encodings can "distinguish columns" but **cannot increase the effective degrees of freedom per scalar** entering the model. Simultaneously, prevailing bidirectional blocks are ordered F→S→N (feature-attention → sample-attention → FFN), forcing the first-layer feature-attention to perform cross-column fusion on raw values without any column-level statistics. Worse, during prediction, only the target token is read, meaning the final layer's sample-attention path barely affects the readout, wasting significant computation.

**Goal**: (1) Introduce sufficient non-linearity in the embedding layer to elevate the effective rank of shallow layers. (2) Reorder attention sequences so that every attention calculation contributes to the final prediction.

**Key Insight**: Classical Radial Basis Function (RBF) local kernel expansion naturally exhibits "different responses across different value ranges," which is equivalent to a set of Gaussian kernel bases centered at quantiles. Mapping a single scalar to an $M$-dimensional vector followed by a shared linear projection expands the "degrees of freedom per scalar" from 1 to $M$. Meanwhile, placing sample-attention at the front of a block allows the model to aggregate column-level statistics before performing feature-attention, aligning with the natural computational order of "statistics then interaction."

**Core Idea**: Utilizing RaBEL (Radial Basis Embedding Layer) as a "pre-non-linearity" to break the value bottleneck + S→N→F reordering for "readout alignment"—together pushing a 2M parameter model beyond the 7M TabPFN-v2.

## Method

### Overall Architecture
Given an input $X\in\mathbb{R}^{N\times D}$ ($N$ samples, $D$ columns), $z$-score normalization is first applied per column to obtain $\tilde{x}_{i,j}$. RaBEL then expands each scalar $\tilde{x}_{i,j}$ into $M$-dimensional RBF responses, which are linearly projected into a $d$-dimensional latent space, forming the cell embedding tensor $E\in\mathbb{R}^{N\times D\times d}$. This tensor is passed through $L$ **reordered bidirectional attention blocks** (S→N→F). Each block performs sample-dimension attention (aggregating column statistics) → FFN (conditional compression) → feature-dimension attention (learning inter-column relationships under better conditions). Finally, attention pooling aggregates all feature tokens into a prediction vector, ensuring every attention layer "sees" the gradient from the readout.

### Key Designs

1.  **RaBEL: Radial Basis Embedding Layer + Exponential Gating**:
    *   **Function**: Expands a single scalar into $M$ local RBF responses before a linear projection, increasing "value-direction" degrees of freedom from 1 to $M$.
    *   **Mechanism**: For each column $j$, $M$ centers $\{c_{j,m}\}$ (initialized via empirical quantiles) and bandwidths $\{\sigma_{j,m}\}$ (initialized via IQR and learned end-to-end) are selected. Defining $\kappa_{j,m}=\exp(-(x_{i,j}-c_{j,m})^2 / (2\sigma_{j,m}^2))$, $\phi_j(x_{i,j})=[\kappa_{j,1},\dots,\kappa_{j,M}]$ is mapped to $\mathbb{R}^d$ via a shared projection $W_{\mathrm{rbf}}\in\mathbb{R}^{d\times M}$ and LayerNorm. To handle cross-magnitude/heteroscedastic real-world data, "exponential gating" is added: $\ell_{i,j}=\log_\beta(|x_{i,j}|+\tau)$ is soft-assigned to exponential buckets $\mathcal{B}$ via a temperature kernel to obtain a scale context $z^{\exp}_{i,j}$. A small MLP then outputs two positive scalar gates $(\gamma^c_{i,j},\gamma^\sigma_{i,j})$ to scale all $M$ centers and bandwidths simultaneously, achieving scale-equivariance and heteroscedastic robustness.
    *   **Design Motivation**: Under pure linear embedding, the rank is at most 2 (Prop 3.1). The locality of RBFs provides diverse activation patterns across different value ranges, equivalent to disentangling common tabular non-linear structures (piecewise trends, local cycles, quantization, heavy tails, heteroscedasticity) in the first layer. Exponential gating allows the model to automatically adjust bump widths across magnitudes, preventing RBFs from becoming "all zeros" in large ranges or overly smooth in small ones.

2.  **S→N→F: Reordered Bidirectional Attention Blocks**:
    *   **Function**: Replaces the standard TabPFN F→S→N block with S→N→F—sample-attention, then FFN, then feature-attention.
    *   **Mechanism**: The S stage performs attention along the sample dimension, allowing the model to aggregate column-level statistics (mean, frequency, missing patterns) within each column. The N (FFN) stage conditionally compresses these statistical signals. Finally, the F stage learns inter-column interactions on tokens already enriched with column-level summaries. For prediction, attention pooling is used across all feature tokens instead of just the target token, providing gradient signals to every attention path.
    *   **Design Motivation**: F→S→N has two flaws: (a) The first F layer must perform inter-column fusion using raw, unconditioned values, exacerbating low-rank collapse. (b) The S layer in the final block does not directly influence the target token's readout, effectively wasting computation. Visualization on synthetic DAG data (Fig 2) shows that in F→S→N, feature-attention is dominated by self-attention, whereas S→N→F allocates attention to the target's true causal parents (e.g., Node 0), proving that reordering restores the natural dependency structure of "statistics before interaction."

3.  **2M Parameter Baseline with Identical Training Recipes**:
    *   **Function**: Decouples architectural changes from training recipes to prove that gains stem from RaBEL + S→N→F.
    *   **Mechanism**: A 2M parameter "SNF baseline" shares the exact same SNF layer sequence, training data, and hyperparameters as LimiX-2M, using standard Linear embeddings instead of RaBEL. Additionally, RaBEL is separately applied to an MLP backbone and a 2M pure Transformer backbone for ablation.
    *   **Design Motivation**: A common critique in the tabular FM field is that gains come from training data, duration, or tuning. This triangular comparison (Linear-SNF / RaBEL-MLP / RaBEL-SNF) isolates architectural contributions from engineering efforts.

### Loss & Training
Inherits the "masked prediction pre-training on synthetically generated tabular tasks" paradigm from LimiX/TabPFN-v2. The training objective remains unchanged; only the embedding layer and intra-block order are modified. RaBEL centers, bandwidths, exponential bucket embeddings, and gating MLPs are all learned end-to-end.

## Key Experimental Results

### Main Results

Comparison of different embeddings on a 2M parameter Transformer backbone across BCCO-CLS / BCCO-REG:

| Benchmark | Metric | Transformer+MLP | Transformer+Periodic | Transformer+PLE | Transformer+RaBEL |
|-----------|------|-----------------|----------------------|-----------------|-------------------|
| BCCO-CLS  | AUC ↑ | 83.52 | 83.88 | 84.66 | **85.04** |
| BCCO-CLS  | Acc ↑ | 76.82 | 77.80 | 77.68 | **77.99** |
| BCCO-CLS  | F1 ↑  | 66.57 | 68.65 | 67.74 | **69.01** |
| BCCO-REG  | $R^2$ ↑ | 0.7731 | 0.6859 | 0.7410 | **0.7792** |
| BCCO-REG  | RMSE ↓ | 0.4043 | 0.4321 | 0.4216 | **0.3964** |

Average rankings across 11 aggregated benchmarks (OpenML-CC18, TabZilla, etc.) show LimiX-2M achieving best (red) or second-best (blue) results on most columns, surpassing GBDT giants (CatBoost, LightGBM, XGBoost), deep tabular models (FT-Transformer, ExcelFormer, SAINT), and larger models like TabPFN-v2 (7M) and TabICL (27M).

### Ablation Study

Shallow-layer rank comparison: LimiX-2M vs. 2M SNF baseline (same recipe, different embedding):

| Configuration | Numerical Rank | Rank@99% | Rank@95% |
|------|----------------|----------|----------|
| 2M Baseline (Linear + SNF) | 58.41 | 13.94 | 6.73 |
| **LimiX-2M (RaBEL + SNF)** | **78.62** (+34.6%) | **25.35** (+82.0%) | **12.31** (+83.2%) |

Horizontal comparison of RaBEL on an MLP backbone (across 13 datasets) also shows MLP-RaBEL ranking 1st on 9/13 datasets, significantly outperforming MLP-MLP, MLP-PLE, and MLP-Periodic.

### Key Findings
- **RaBEL Directly Increases Rank**: Using identical SNF architecture and training recipes, replacing Linear with RaBEL nearly doubled the Rank@95% in shallow layers (6.73→12.31), confirming that the "value bottleneck" diagnosed in Prop 3.1 is the real culprit.
- **S→N→F Modifies "Attention Direction," Not "Compute"**: On synthetic DAG data, F→S→N acts almost entirely as self-attention (weights on diagonal), while S→N→F correctly attributes high attention to causal parents. This explains why gains are achieved without adding new parameters.
- **TabPFN-v2 is Severely Over-parameterized**: Truncating its 192-dimensional latent space to 50 dimensions (25%) only dropped AUC from 0.9177 to 0.9143. Truncating to 20 dimensions (10%) still yielded 0.8985. This supports the notion that embedding, not parameter count, is the bottleneck.
- **2M Beats 7M / 27M**: Across 11 aggregated benchmarks, LimiX-2M generally outperformed TabPFN-v2 (7M) and TabICL (27M) with lower training and inference costs, proving that value lies in "identifying bottlenecks" rather than "scaling up."

## Highlights & Insights
- **Theoretical Diagnosis to Methodological Solution**: Prop 3.1 provides a rigid upper bound (rank ≤ 2) for linear embeddings, SVD measurements replicate this bound, and RBF expansion lifts the "scalar degree of freedom" from 1 to $M$. This logical chain makes it a rare paper that "diagnoses the disease before prescribing the cure."
- **Scale-Equivariance via Exponential Gating**: Decoupling magnitude (log-domain exponential buckets) from "patterns within magnitude" (RBF responses) handles the longstanding tabular challenge of unit and magnitude sensitivity with a lightweight module.
- **Param-free Reordering**: S→N→F introduces no new parameters; it simply reorders components and switches readout to attention pooling. The fact that it learns causal direction on synthetic DAGs suggests this "free architectural bonus" is highly applicable to other in-context learning scenarios.

## Limitations & Future Work
- **Validated Only up to 2M Parameters**: Experiments focused on the 2M scale without showing scaling curves for 16M/70M. It is possible that RaBEL's benefits might be diminished in much larger models.
- **Categorical Column Handling**: RaBEL targets numerical columns; categorical columns degrade to standard entity embeddings. There is room for improvement in tables with heavy categorical presence using RBF-style frequency or TF-IDF bases.
- **Column-level Bias in RBF Hyperparameters**: Using empirical quantiles and IQR for initialization might not be expressive enough for multimodal or heavy-tailed columns. Hybrid RBF + Periodic approaches could be considered.
- **S→N→F on Variable-length/Streaming Tables**: The in-context assumption here is fixed; performance in real-world online prediction scenarios involving latency and drift remains unexplored.

## Related Work & Insights
- **vs FT-Transformer / TabTransformer**: While they use standard "Linear + Column ID" embeddings, this work proves the rank bound ≤ 2 for that recipe and systematically breaks it with RBF.
- **vs Periodic / PLE** (Gorishniy et al. 2022): While Periodic uses global cycles and PLE uses piecewise linear binning, RaBEL uses local RBFs. RaBEL's superior RMSE (0.3964) on BCCO-REG compared to Periodic (0.4321) and PLE (0.4216) suggests local kernels are more suitable for most tabular data.
- **vs TabPFN-v2 / TabICL**: This work adopts the in-context framework of TabPFN-v2 and the bidirectional attention of TabICL but reverses the F→S→N order and changes the readout. This serves as a "minimal architectural patch" that allows a 2M model to compete with 7M and 27M models.
- **vs SAINT / VIME / MET**: These use cross-sample attention or self-supervision. The S-stage in SNF places "cross-sample statistics" in the correct position, representing a proper placement of SAINT’s row-attention within the in-context FM framework.

## Rating
- Novelty: ⭐⭐⭐⭐ Combination of Prop 3.1 rank proof, RaBEL expansion, and S→N→F reordering is original in its diagnostic-driven approach.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 benchmarks, triangular ablation, DAG visualization, and quantitative rank ratios cover nearly all bases.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, rigorous formulas, and effective visualization; some terminology stacking takes effort to parse.
- Value: ⭐⭐⭐⭐⭐ 2M beating 7M/27M while reducing costs is a highly practical contribution for Tabular FMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Mitigating Label Shift in Tabular In-Context Learning via Test-Time Posterior Adjustment](mitigating_label_shift_in_tabular_in-context_learning_via_test-time_posterior_ad.md)
- [\[AAAI 2026\] Robust Tabular Foundation Models](../../AAAI2026/self_supervised/robust_tabular_foundation_models.md)
- [\[ICML 2026\] From Zero to Hero: Advancing Zero-Shot Foundation Models for Tabular Outlier Detection](from_zero_to_hero_advancing_zero-shot_foundation_models_for_tabular_outlier_dete.md)
- [\[NeurIPS 2025\] Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models](../../NeurIPS2025/self_supervised/mitra_mixed_synthetic_priors_for_enhancing_tabular_foundation_models.md)
- [\[ICML 2026\] The Geometry of Projection Heads: Conditioning, Invariance and Collapse](the_geometry_of_projection_heads_conditioning_invariance_and_collapse.md)

</div>

<!-- RELATED:END -->
