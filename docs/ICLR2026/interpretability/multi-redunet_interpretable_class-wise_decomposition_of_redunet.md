---
title: >-
  [Paper Note] Multi-ReduNet: Interpretable Class-Wise Decomposition of ReduNet
description: >-
  [ICLR 2026][Interpretability][ReduNet] The global MCR² objective of ReduNet is theoretically decomposed into $K$ independent "class-wise subproblems." Combined with the Woodbury identity, the complexity of per-layer matrix inversion is reduced from $O(d^3)$ to $O(m_j^3)$. In high-dimensional undersampled scenarios ($m \ll d$), this approach achieves higher
tags:
  - ICLR 2026
  - Interpretability
  - ReduNet
  - MCR²
date: 2026-05-08
content_hash: 9f93a839118a4767
---
# Multi-ReduNet: Interpretable Class-Wise Decomposition of ReduNet

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=wLcTAJ7DF9](https://openreview.net/forum?id=wLcTAJ7DF9)  
**Code**: To be confirmed  
**Area**: Interpretability / White-box Networks  
**Keywords**: ReduNet, MCR², White-box Networks, Class-wise Decomposition, Undersampling, Woodbury Identity  

## TL;DR
The global MCR² objective of ReduNet is theoretically decomposed into $K$ independent "class-wise subproblems." Combined with the Woodbury identity, the complexity of per-layer matrix inversion is reduced from $O(d^3)$ to $O(m_j^3)$. In high-dimensional undersampled scenarios ($m \ll d$), this approach achieves higher accuracy, approximately 2× training acceleration, and nearly an order of magnitude improvement in learning rate robustness while maintaining white-box interpretability.

## Background & Motivation
**Background**: ReduNet (Chan et al., 2022) is a white-box deep network based on the Maximal Coding Rate Reduction (MCR²) principle. Each layer possesses a closed-form analytical update, the geometric meaning of feature maps is transparent, and it carries provable optimization guarantees. It is one of the few works in "interpretable deep learning" capable of deriving an entire network from first principles.

**Limitations of Prior Work**: ReduNet performs dense operations on global feature matrices, where the complexity per parameter is $O(d^3)$ relative to the feature dimension $d$. In "undersampled" scenarios prevalent in finance, biomedicine, and rare disease imaging ($d \gg m$; e.g., the ARCENE dataset has $d=10{,}000$ but only $m=200$ samples, $m/d=0.02$), this $O(d^3)$ cost becomes prohibitive. Simultaneously, the global covariance $ZZ^\top$ couples all classes together, failing to explicitly exploit class-specific structures, which makes it slow and difficult to converge under class imbalance.

**Key Challenge**: The interpretability of white-box methods stems from global closed-form updates, yet these global dense operators create an $O(d^3)$ computational wall. Accelerating the model usually requires architectural changes that risk destroying the theoretical optimality of MCR².

**Goal**: To decompose the global optimization into smaller class-wise problems without sacrificing MCR² optimality or white-box interpretability, while enhancing discriminative power and hyperparameter robustness in undersampled scenarios.

**Key Insight**: **Class orthogonality is a property of the optimal solution rather than an external constraint.** The authors prove that the global optimal solution of MCR² must satisfy inter-class orthogonality $(Z^i)^\top Z^j=0$. Consequently, the global objective can be losslessly decomposed into $K$ independent class-wise subproblems, each of which can utilize the Woodbury identity to replace $d \times d$ inversions with $m_j \times m_j$ inversions.

## Method

### Overall Architecture
The method progresses in three steps: first, **imp-ReduNet** uses the Woodbury identity to compress single-parameter complexity from $O(d^3)$ to $O(m^3)$ (solving the dimension bottleneck without changing class structure); second, two theorems prove that the global MCR² objective can be **losslessly decomposed into class-wise subproblems** (further splitting $m \times m$ into $K$ subproblems of $m_j \times m_j$); finally, **Multi-ReduNet** and its variant **Multi-ReduNet-LastNorm** are designed to optimize closed-form update operators for each class in parallel.

```mermaid
flowchart TD
    A[Input X ∈ R^{d×m}<br/>Undersampled m≪d] --> B[ReduNet Original<br/>Global MCR² Objective O d^3]
    B --> C[imp-ReduNet<br/>Woodbury Identity<br/>O d^3 → O m^3]
    C --> D[Theorem 1 Class Orthogonality<br/>is a Property of Global Optima]
    D --> E[Theorem 2 Global Objective<br/>Losslessly Decomposed into K Subproblems]
    E --> F[Multi-ReduNet<br/>Class-wise Parallel Updates O m_j^3]
    E --> G[Multi-ReduNet-LastNorm<br/>Normalization only at Last Layer]
    F --> H[White-box Features + SVM/KNN/NSC Classification]
    G --> H
```

### Key Designs

**1. imp-ReduNet: Eliminating the dimension bottleneck via the Woodbury identity by "switching sides for inversion."** ReduNet requires inverting $d \times d$ matrices when calculating update operators $E_l$ and $C_l^j$, costing $O(d^3)$. When $m \ll d$, the authors apply the Woodbury identity $(I+\alpha XX^\top)^{-1}=I-\alpha X(I+\alpha X^\top X)^{-1}X^\top$ to replace the $d \times d$ inversion with an $m \times m$ inversion. For the ARCENE dataset ($d=10{,}000$, $m_{\text{train}}=159$), the theoretical acceleration for the inversion step alone is approximately $(10000/159)^3 \approx 250{,}000 \times$. This step addresses high dimensionality but not class count; if the total sample size $m$ is large, the $m \times m$ inversion remains expensive, leading to class-wise decomposition.

**2. Class Orthogonality as an Optimality Condition (Theorem 1): Grounding decomposition in theory.** The authors prove that for the MCR² objective $\max_Z \frac{1}{2}\log\det(I+\alpha ZZ^\top)-\sum_j\frac{m_j}{2m}\log\det(I+\alpha_j Z\Pi_j Z^\top)$, any global optimal solution $Z^\star$ must satisfy inter-class orthogonality $(Z^i)^\top Z^j=0\ (\forall i \neq j)$. This is proven by contradiction: if features from two classes are not orthogonal, a determinant inequality regarding sums of PSD matrices (Corollary 1) shows that the global coding rate $\det(I+\sum_j Z^j(Z^j)^\top)$ is strictly less than $\prod_j\det(I+Z^j(Z^j)^\top)$. Thus, a solution $Z'$ with a higher objective value can be constructed via SVD re-orthogonalization. Crucially, class orthogonality is an **inherent geometric property of the optimal solution**, not a hard constraint added during training, providing the foundation for optimizing classes separately.

**3. Class-wise Lossless Decomposition (Theorem 2): Splitting the global objective into K independent subproblems.** Under the orthogonal structure from Theorem 1, if each class's features satisfy $\mathrm{rank}(Z^j) \le d_j$ and $\sum_j d_j \le d$, the global MCR² objective is exactly equivalent to the sum of $K$ independent subproblems: $\max_{Z^j}\frac{1}{2}[\log\det(I+\frac{d}{m\epsilon^2}Z^j(Z^j)^\top)-\frac{m_j}{m}\log\det(I+\frac{d}{m_j\epsilon^2}Z^j(Z^j)^\top)]$ subject to $\|Z^j\|_F^2=m_j$. The proof uses a double inequality: class-wise feasible solutions are globally feasible ($v_2 \le v_1$), while global optima are feasible for class-wise problems due to orthogonality ($v_1 \le v_2$). When $m \ll d$, the condition $\sum_j \mathrm{rank}(Z^j) \le \sum_j m_j = m \ll d$ is naturally satisfied. This further reduces the per-parameter cost from $O(m^3)$ to $\sum_j O(m_j^3)$, offering more savings as class imbalance increases, and serves as the **first practical class-wise MCR² optimization algorithm.**

**4. Multi-ReduNet and LastNorm Variant: Parallel class updates + normalization strategy trade-offs.** According to Theorem 2, each class maintains its own update operators $E_l^j = \alpha(I+\alpha Z_l^j(Z_l^j)^\top)^{-1}$ and $C_l^j = \alpha_j(I+\alpha_j Z_l^j(Z_l^j)^\top)^{-1}$. Layer-wise gradient ascent is performed: $Z_{l+1}^j \leftarrow Z_l^j + \eta(E_l^j Z_l^j - \frac{m_j}{m} C_l^j Z_l^j)$. The expansion term (coefficient $\alpha$) stretches features globally, while the compression term (coefficient $\alpha_j$) pulls each class towards a compact subspace. Both are calculated from **intra-class covariance** rather than global $ZZ^\top$, decoupling optimization. **Multi-ReduNet** applies spherical projection $\mathcal{P}_{S^{d-1}}$ (column-wise normalization) at every layer to satisfy $\|Z^j\|_F^2=m_j$. **Multi-ReduNet-LastNorm** relaxes intermediate normalization and **projects only at the final layer**, allowing more flexible intermediate representations, reducing projection overhead, and significantly improving hyperparameter robustness. During inference, test samples use soft assignments $\hat\pi_l^j$ to aggregate class updates.

## Key Experimental Results

### Main Results
Six undersampled datasets ($m_{\text{train}}/d \in [0.016, 0.5]$), $L=5$ layers, $\epsilon^2=0.1$, fixed $\eta_0=0.05$. Final layer features classified using SVM/KNN/NSC (average of 3 seeds):

| Dataset | Model | SVM | KNN | NSC |
|---------|-------|-----|-----|-----|
| Reuters | ReduNet | 0.802 | 0.670 | 0.922 |
| Reuters | Multi-ReduNet-LastNorm | **0.985** | **0.943** | **0.957** |
| DrivFace| ReduNet | 0.432 | 0.393 | 0.366 |
| DrivFace| Multi-ReduNet-LastNorm | **1.000** | **0.978** | **0.995** |
| ARCENE  | ReduNet | 0.439 | 0.415 | 0.463 |
| ARCENE  | Multi-ReduNet-LastNorm | **0.829** | **0.732** | **0.805** |
| MNIST   | ReduNet | **0.906** | **0.930** | 0.903 |
| MNIST   | Multi-ReduNet-LastNorm | 0.842 | 0.903 | 0.873 |

Averaged across four learning rates $\{0.5, 0.1, 0.05, 0.01\}$ and three classifiers, Multi-ReduNet(-LastNorm) outperforms ReduNet by **8.5–52.7 percentage points** (Reuters +30.7pp, DrivFace +52.7pp) and reduces wall-clock training time by ~2× (1.4–2.6× across datasets).

### Ablation Study
Learning rate robustness (Accuracy Range, lower is more stable) and LastNorm gains:

| Dataset | ReduNet Range(pp) | Multi-ReduNet Range(pp) | LastNorm Range(pp) | LN vs MR Δ(pp) |
|---------|:--:|:--:|:--:|:--:|
| Reuters | 67.5 | 3.3 | 3.2 | +0.0 |
| MNIST   | 86.3 | 27.1 | 20.6 | +0.0 |
| Fashion | 71.7 | 10.7 | 8.1 | +1.3 |
| ARCENE  | 41.4 | 9.7 | 2.4 | +0.0 |
| **Mean**| **62.6** | **9.0** | **6.4** | **+0.2** |

Multi-ReduNet-LastNorm is comparable to Multi-ReduNet in accuracy (+0.2pp on average) but is approximately **9.8×** more robust to learning rates than ReduNet (Range 6.4pp vs 62.6pp), also surpassing Multi-ReduNet (6.4 vs 9.0).

### Key Findings
- **Greater benefits from undersampling**: Gains are largest on heavily undersampled and noisy datasets like DrivFace and ARCENE (accuracy jumps from 0.43–0.46 to 0.73–1.00). On sub-sampled image datasets like MNIST/Fashion where ReduNet already performs well, Multi-ReduNet shows a slight decrease of a few percentage points, suggesting class-wise flexibility is primarily advantageous in difficult high-dimensional scenarios like microarrays or facial images.
- **Acceleration scales with depth**: Relative speedup remains stable at 1.4–2.6×, but the absolute wall-clock difference increases with the number of layers $L$, which is particularly significant for deep ($L>20$) high-dimensional ($d>10{,}000$) models.
- **Comparison with classical methods**: On Reuters, Multi-ReduNet-LastNorm (98.8%) outperforms PCA (97.5%). However, on ARCENE, LDA (87.8%) still beats this method (82.9%), suggesting classical methods remain competitive on well-structured datasets.
- **Visualization evidence**: t-SNE shows that Multi-ReduNet variants produce more compact and well-separated class clusters.

## Highlights & Insights
- **Turning "Decomposition" from a heuristic into a theorem**: Theorems 1 and 2 utilize class orthogonality to prove that the decomposition of the global MCR² objective is a lossless equivalence rather than an approximation trick, providing rare theoretical guarantees for accelerating white-box networks.
- **Orthogonal superposition of two complexity reductions**: Woodbury (removing $d$ bottleneck) and class-wise decomposition (removing $m$ class coupling) address different bottlenecks. Combined, they yield $O(d^3) \to \sum_j O(m_j^3)$, providing maximum benefit in undersampled and imbalanced cases.
- **LastNorm is a high-yield design**: Deferring normalization to the final layer maintains accuracy while yielding nearly an order of magnitude improvement in learning rate robustness, which is beneficial for practical deployment where hyperparameter tuning is difficult.
- **Maintenance of white-box identity**: All updates remain closed-form analytical operators in the style of ReduNet. The decomposition introduces no black-box components, ensuring interpretability throughout.

## Limitations & Future Work
- **Reliance on $m \ll d$**: The conditions for Theorem 1/2 decomposition ($\sum_j \mathrm{rank}(Z^j) \le d$) are naturally met in undersampled scenarios. In large-scale data scenarios where $m \gg d$, the advantages and validity of decomposition are no longer obvious.
- **Theoretical optimality ≠ Practical orthogonality**: The authors acknowledge that due to local optima, finite steps, numerical precision, and data separability, learned class representations are only "approximately orthogonal." Decomposition should be understood as a "rational reparameterization at the level of global optimality."
- **Not a universal replacement for classic methods**: The method underperforms compared to LDA on ARCENE and slightly lags behind the original ReduNet on MNIST/Fashion, clearly defining its niche: high-dimensional undersampling and imbalance.
- **Real-world acceleration is less than theoretical**: Theoretical $O((d/m)^3)$ inversion speedup is diluted by memory transfers and interpreter overhead, resulting in an empirical speedup of ~2×.
- **Future directions**: Extending decomposition to convolutional or translation-invariant versions of ReduNet, integrating it with modern self-supervised learning objectives, or verifying parallel scalability under a larger number of classes $K$.

## Related Work & Insights
- **White-box Network Genealogy**: Directly built upon ReduNet (Chan et al., 2022) and MCR² (Yu et al., 2020), this work stands as an efficiency-focused extension of the lineage that derives networks from rate reduction principles.
- **High-dimensional Undersampled Learning**: Compared to PCA/LDA (global statistical modeling without explicit class structure encoding), black-box few-shot/meta-learning (Prototypical/Matching Networks, MAML—effective but black-box), and information-theoretic objectives (InfoMax, Information Bottleneck—relying on variational bounds without closed-form updates), this work simultaneously achieves class-specific structure, closed-form updates, and geometric interpretability.
- **Inspiration**: Proving that a certain symmetry or orthogonality is an inherent property of the optimal solution and then using it to decompose global optimization into parallel subproblems is a transferable methodology. Similar strategies might yield lossless acceleration in other coupled objectives like contrastive learning or spectral methods.

## Rating
- **Novelty**: ⭐⭐⭐⭐ — Uses the optimality of class orthogonality to prove the lossless equivalence of class-wise MCR² decomposition, creating the first practical class-wise MCR² algorithm; however, it remains a targeted extension within the ReduNet framework rather than a brand-new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers six cross-domain undersampled datasets, three classifiers, multiple learning rates/depths, accuracy, efficiency, robustness, t-SNE, classical baselines, and failure mode analysis. Lacks cross-comparison with broader modern deep learning methods or large-scale validation.
- **Writing Quality**: ⭐⭐⭐⭐ — The three-step progression (Woodbury → Theorems → Architecture) is logically clear, though the high density of formulas creates a steep threshold for readers unfamiliar with MCR².
- **Value**: ⭐⭐⭐⭐ — Simultaneously improves accuracy, speed, and robustness for high-dimensional undersampling—a real-world pain point—while preserving interpretability. This is significant for data-scarce fields like finance and medicine.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TimeSeg: An Information-Theoretic Segment-Wise Explainer for Time-Series Predictions](timeseg_an_information-theoretic_segment-wise_explainer_for_time-series_predicti.md)
- [\[ICLR 2026\] Towards Understanding the Nature of Attention with Low-Rank Sparse Decomposition](towards_understanding_the_nature_of_attention_with_low-rank_sparse_decomposition.md)
- [\[CVPR 2026\] HUMORCHAIN: Theory-Guided Multi-Stage Reasoning for Interpretable Multimodal Humor Generation](../../CVPR2026/interpretability/humorchain_theory-guided_multi-stage_reasoning_for_interpretable_multimodal_humo.md)
- [\[ICLR 2026\] A Comprehensive Information-Decomposition Analysis of Large Vision-Language Models](a_comprehensive_information-decomposition_analysis_of_large_vision-language_mode.md)
- [\[ICML 2026\] Neural Collapse by Design: Learning Class Prototypes on the Hypersphere](../../ICML2026/interpretability/neural_collapse_by_design_learning_class_prototypes_on_the_hypersphere.md)

</div>

<!-- RELATED:END -->
