---
title: >-
  [Paper Note] From Data Statistics to Feature Geometry: How Correlations Shape Superposition
description: >-
  [ICLR 2026][Interpretability][superposition] This paper argues that the classic "superposition = interference = noise" paradigm is incomplete for real-world data. When features are correlated, interference can be **constructive**: models arrange features based on co-activation patterns, allowing interference between active features to mutually reinforce signals. This enables reconstruction with smaller weight norms and lower rank, naturally explaining geometric structures lik…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "superposition"
  - "mechanistic interpretability"
  - "feature geometry"
  - "sparse autoencoder"
  - "constructive interference"
date: 2026-05-08
content_hash: 481fd2d9203dbda1
---

# From Data Statistics to Feature Geometry: How Correlations Shape Superposition

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=7akSRQS5Xh](https://openreview.net/forum?id=7akSRQS5Xh)  
**Code**: [https://github.com/LucasPrietoAl/correlations-feature-geometry](https://github.com/LucasPrietoAl/correlations-feature-geometry)  
**Area**: Mechanistic Interpretability / Superposition  
**Keywords**: superposition, mechanistic interpretability, feature geometry, sparse autoencoder, constructive interference  

## TL;DR
This paper argues that the classic "superposition = interference = noise" paradigm is incomplete for real-world data. When features are correlated, interference can be **constructive**: models arrange features based on co-activation patterns, allowing interference between active features to mutually reinforce signals. This enables reconstruction with smaller weight norms and lower rank, naturally explaining geometric structures like semantic clustering and circular arrangements of months observed in real language models.

## Background & Motivation

**Background**: A core idea in mechanistic interpretability (MI) is that neural networks represent significantly more features than their dimensions, forming an overcomplete basis through "superposition" at the cost of allowing inter-feature interference. This framework has driven the development of dictionary learning methods like Sparse Autoencoders (SAEs), successfully used for feature recovery in frontier large language models.

**Limitations of Prior Work**: Previous studies on superposition were almost exclusively conducted under **idealized settings**—where features are sparse and mutually independent (e.g., the toy model by Elhage et al. 2022). In these settings, interference is understood as noise that must be **geometrically minimized and then filtered out** using non-linearities like ReLU. This leads to local structures such as regular polytopes, where pairwise inner products of features are near zero and negative interference is suppressed.

**Key Challenge**: This paradigm fails to explain structures observed in real language models. Instead of local polytopes, researchers have found **ordered feature rings** (e.g., circular arrangements of the twelve months) and **anisotropic superposition** (where related features cluster rather than minimize inner products) in LLM activations. Standard superposition theory predicts these should not occur.

**Goal**: To resolve this contradiction and provide a unified characterization of superposition geometry under real-world data distributions.

**Key Insight**: The authors argue that the root of the contradiction is that "real features are neither sparse nor independent." **[Key Insight] When features are correlated, interference need not be purely harmful—it can be constructive.** By arranging features according to co-activation patterns, interference among active features reinforces each other (e.g., "December" aids the reconstruction of "Christmas"), while ReLU is still employed to avoid false positives. To verify this in a controlled environment, the authors propose the **Bag-of-Words Superposition (BOWS)** framework: encoding binary bag-of-words representations of internet text into superposition using an autoencoder, which captures real feature correlations with known ground-truth features.

## Method

### Overall Architecture

BOWS situates the "superposition geometry" problem in a **controlled yet realistic** setting: binary bag-of-words vectors are sampled from a corpus (WikiText-103, vocabulary $V=10,000$). An autoencoder with tied weights $\hat{f}=\sigma(W^\top W f + b)$ is trained ($W\in\mathbb{R}^{m\times V}, m<V$), comparing ReLU AE and Linear AE decoding. The authors validate their claims across three levels: **theoretical characterization** (when interference becomes constructive), **synthetic experiments** (12-dimensional cyclic covariance), and **real data** (semantic clusters and month rings in WikiText), while distinguishing between presence-coding and value-coding to account for structures that appear even without data correlation.

```mermaid
flowchart LR
    A[Internet Text<br/>WikiText-103] --> B[Binary BOW Vectors<br/>x∈{0,1}^V]
    B --> C[ReLU Autoencoder<br/>f̂=ReLU W^T W f + b]
    C --> D{Covariance Σ Structure}
    D -->|Sparse/Weak Correlation| E[Interference=Noise<br/>Filtered by ReLU<br/>→Antipodal/Polytope]
    D -->|Low-rank Correlation| F[Constructive Interference<br/>Linear Superposition<br/>→Semantic Clusters/Rings]
    F --> G[Smaller Weight Norm<br/>‖W‖²=m < d]
```

### Key Designs

**1. Signal-Interference Decomposition: Redefining the Role of Interference**: For a tied-weight AE, the reconstruction of feature $f_i$ can be decomposed into signal and interference terms: $\hat{f}_i=\sigma\big(\|w_i\|^2 f_i + \sum_{j\neq i}\langle w_i,w_j\rangle f_j + b_i\big)$, where $I_i=\sum_{j\neq i}\langle w_i,w_j\rangle f_j$ is the interference. While the standard view treats $I_i$ as noise to be filtered, this paper proves that when the feature covariance $\Sigma=\mathbb{E}[ff^\top]$ has a sufficiently strong low-rank structure, $I_i$ **aligns** with the signal. Specifically, the optimal mapping $P=W^\top W$ for a Linear AE is an orthogonal projection onto the top $m$ principal components of $\Sigma$. Here, $P_{ij}=\langle w_i,w_j\rangle$ exactly reflects the correlation between features $i$ and $j$ within the principal subspace—each $f_j$ contributes to the reconstruction of $f_i$ proportional to their shared variance. In other words, **PCA itself is a form of superposition**, where correlated features are arranged so that interference reinforces the signal rather than needing suppression.

**2. Distinguishing Linear and Non-linear Superposition**: The authors define the boundary based on whether a feature can be recovered via a **linear decoder**. When $\mathrm{rank}(\Sigma)\le m$, the data lies entirely within the principal subspace, the residual $\varepsilon=0$, and interference $I_i=(1-P_{ii})f_i$ is proportional to the signal. This is **linear superposition** (Definition 2: there exists linear $\psi_{\text{lin}}$ such that $R_i^2\ge 1-\varepsilon$). When $\Sigma$ is only approximately low-rank, the residual $\|\varepsilon\|^2=\sum_{k>m}\lambda_k(\Sigma)$ (the Eckart–Young lower bound) introduces false positives that require ReLU and negative biases for suppression, resulting in **non-linear superposition**. The more concentrated the spectrum and smaller the residual, the more effective the constructive interference.

**3. Weight Decay + Tight Bottleneck Bias Towards Low-rank Solutions**: The two mechanisms incur different norm costs. The classic scheme of filtering interference with ReLU requires each feature direction to have near-unit norm, leading to $\|W\|_F^2\approx d$ (proportional to the number of features). Conversely, projecting onto an $m$-dimensional low-rank subspace yields an optimal rank-$m$ projector where $\|W\|_F^2=\mathrm{tr}(P)=\sum_k\lambda_k(P)=m<d$. Thus, **when $m\ll d$ and weight decay is applied, even non-linear models are pushed toward solutions leveraging low-rank structures**, as they achieve accurate reconstruction with significantly smaller weight norms. This explains why semantic clusters re-emerge in models with weight decay.

**4. Coexistence of Mechanisms**: When $\Sigma$ is only approximately low-rank, weight geometry aligns most interference with the signal, while ReLU and negative biases suppress harmful interference caused by residuals. The authors demonstrate this with the word "Beatles": in supportive contexts, related words (Lennon, McCartney) provide positive pre-activation to help reconstruct the target word. When a similar context lacks the target word, the ReLU threshold filters out the false positive.

**5. Presence-coding vs Value-coding**: Circular structures in modular addition occur even without data correlation, which standard superposition cannot explain. The authors distinguish between two types of features: **Presence-coding** features detect discrete attributes ("is it the word 'cat'"), whose structural arrangement depends on data correlation (the superposition discussed above). **Value-coding** features linearly encode continuous variables $v(x)\in\mathbb{R}$ (e.g., angles, coordinates). $k$ such features span a $k$-dimensional value space; projecting samples onto it reveals rings or maps—**but these structures are explained entirely by linear value codes and exist even without superposition.** This distinction separates "geometric superposition" from "feature manifolds."

## Key Experimental Results

### Synthetic Data: Two Mechanisms under Cyclic Covariance (Figure 2)

A Linear AE and ReLU AE were trained on 12-dimensional cyclic covariance data, varying the latent dimension $m$:

| Latent Dim $m$ | Linear AE | ReLU AE |
|---|---|---|
| Small $m$ ($m<6$) | Project to circular principal subspace | **Reproduces ring structure** (Linear Superposition) |
| Medium $m$ (6–10) | Still circular projection | **Abandons ring, forms antipodal pairs** (Non-linear Superposition, ReLU filters interference) |
| $m\to 12$ (≈Input dim) | — | Weights approach identity, structure disappears |

### Main Results: Month Features as Linear Superposition

- Training a linear decoder to reconstruct month features yielded $R_i^2(W_{\text{months}},\psi_{\text{lin}})=0.98\pm 0.00015$, with no two months being mutually orthogonal. Per Definition 2, **month features are in linear superposition**.
- Months exhibit cyclic correlation in WikiText (January co-occurs more with February/December than August). Their PCA shows a ring (Figure 4b), and the PCA of learned encoding weights $W$ reproduces the same ring (Figure 4c). "Christmas" aligns with its co-occurring months; "December" constructively contributes to its reconstruction. When all months appear simultaneously, interference cancels out to avoid false positives.

### Value-coding Ablation (Table 1)

VC+ keeps the value-coding subspace and zeros the orthogonal complement; VC− removes value-coding coordinates:

| Condition | MAP Loss↓ | MAP Acc↑ | Key-freq Loss↓ | Key-freq Acc↑ |
|---|---|---|---|---|
| Baseline | 0.0536 | 97.94 | 0.0001 | 100.00 |
| VC+ (Keep VC Only) | 0.2879 | 93.16 | 0.1649 | 93.99 |
| VC− (Remove VC) | 6.3943 | 22.43 | 11.7464 | 3.11 |

Replacing 90%+ of dimensions with the mean still preserves most accuracy, while removing value-coding features causes performance to collapse. This proves value-coding features are the true units of computation. In a city relative position task, a linear probe predicted coordinates with $R^2=0.98$, correctly reconstructing a map of the US from held-out cities.

### Key Findings

- **Sequential Disappearance of Structures** (Figure 6): As $m$ increases, the month structure orthogonalizes earlier than Roman numerals—different feature groups in real data occupy different positions on the linear-to-non-linear superposition spectrum based on frequency/correlation.
- **Frequency-based Stratification**: In an $m=800$ model with a threshold of $R=0.5$, 4,073 words are in non-linear superposition, while the 522 highest-frequency words are in linear superposition.
- **Benefits of Constructive Interference**: In validation samples containing "Beatles," 81% of contexts achieved better reconstruction than one-hot reconstruction.

## Highlights & Insights

- **Paradigm Reframing**: Flips the "interference = noise" narrative to "interference can be constructive," bridging the gap between statistical low-rank structure and MI feature geometry by identifying PCA as a form of superposition.
- **Falsifiable Definitions**: Distinguishes linear/non-linear superposition based on linear reconstructibility ($R^2\ge 1-\varepsilon$), providing a clean empirical criterion (e.g., $R^2=0.98$ for months).
- **Clever BOWS Design**: Bag-of-words provides real correlation while maintaining ground-truth features, filling the gap between toy models and real LLMs.
- **Value/Presence Coding Distinction**: Resolves long-standing confusion over structures like modular addition rings and provides a principled way to separate "feature geometry" from "feature manifolds."

## Limitations & Future Work

- The setup is limited to **Bag-of-Words Autoencoders** rather than full Transformer language models; the generalizability of BOWS to real LLM representations requires further validation.
- Theoretical analysis primarily assumes **tied weights, ReLU, and approximately low-rank covariance**, providing limited characterization of superposition geometry under complex non-linearities or attention mechanisms.
- **Automatic distinction** between value-coding and geometric superposition in complex real-world scenarios remains an open problem.
- Downstream impacts on SAE training, knowledge editing, and adversarial robustness are discussed qualitatively without direct task validation.

## Related Work & Insights

- **Origins of Superposition**: Elhage et al. (2022) used superposition to explain polysemanticity, predicting polytopes under sparse independent conditions; this paper shows that picture is incomplete for correlated data.
- **SAE Dictionary Learning**: Bricken et al. (2023) and others recovered linear features and observed anisotropic clustering; this paper provides the geometric cause.
- **Feature Rings/Manifolds**: Engels et al. (2025) found rings for periodic concepts, while Nanda et al. (2023) studied modular addition rings; this paper attributes these to linear superposition and value-coding, respectively.
- **Insights**: (1) Weight decay and bottleneck width during training directly shape superposition geometry, serving as "geometric knobs" for interpretability. (2) When interpreting SAE feature clusters, one should first ask if they result from data-driven superposition geometry or task-driven value codes to avoid misinterpreting manifolds as superposition.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reframing "interference" and linking PCA to superposition is a substantial theoretical advancement.
- Experimental Thoroughness: ⭐⭐⭐⭐ Synthetic + Real + Ablation studies provide strong evidence, though end-to-end Transformer validation is missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear definitions, intuitive diagrams, and a smooth transition from theory to empirical results.
- Value: ⭐⭐⭐⭐ Provides a unified perspective for understanding feature geometry in SAEs and robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Temporal Superposition and Feature Geometry of RNNs under Memory Demands](temporal_superposition_and_feature_geometry_of_rnns_under_memory_demands.md)
- [\[ICLR 2026\] Emergence of Superposition: Unveiling the Training Dynamics of Chain of Continuous Thought](emergence_of_superposition_unveiling_the_training_dynamics_of_chain_of_continuou.md)
- [\[ICLR 2026\] On The Geometry and Topology of Representations: the Manifolds of Modular Addition](on_the_geometry_and_topology_of_representations_the_manifolds_of_modular_additio.md)
- [\[ICLR 2026\] Mixing Mechanisms: How Language Models Retrieve Bound Entities In-Context](mixing_mechanisms_how_language_models_retrieve_bound_entities_in-context.md)
- [\[ICLR 2026\] The Geometry of Reasoning: Flowing Logics in Representation Space](the_geometry_of_reasoning_flowing_logics_in_representation_space.md)

</div>

<!-- RELATED:END -->
