---
title: >-
  [Paper Note] On the Spectral Differences Between NTK and CNTK and Their Implications for Point Cloud Recognition
description: >-
  [ICLR 2026][Learning Theory][NTK] Under the assumption of tensor data in arbitrary dimensions, this paper provides two **distribution-independent** spectral difference theorems for NTK and CNTK (NTK eigenvalues have a larger mean and a more concentrated spectrum). Based on these, it defines a "Convolutional Suitability" metric to measure "how suitable data is for convolution" and infers that point clouds depend more on convolutional structures than images. Finally…
tags:
  - "ICLR 2026"
  - "Learning Theory"
  - "Neural Tangent Kernel"
  - "NTK"
  - "CNTK"
  - "Spectral Analysis"
  - "Convolutional Suitability"
  - "Point Cloud Recognition"
date: 2026-05-08
content_hash: 7e1c45a432765081
---

# On the Spectral Differences Between NTK and CNTK and Their Implications for Point Cloud Recognition

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=U6SnDgI3gG](https://openreview.net/forum?id=U6SnDgI3gG)  
**Code**: TBC  
**Area**: Learning Theory / Neural Tangent Kernel  
**Keywords**: NTK, CNTK, Spectral Analysis, Convolutional Suitability, Point Cloud Recognition

## TL;DR
Under the assumption of tensor data in arbitrary dimensions, this paper provides two **distribution-independent** spectral difference theorems for NTK and CNTK (NTK eigenvalues have a larger mean and a more concentrated spectrum). Based on these, it defines a "Convolutional Suitability" metric to measure "how suitable data is for convolution" and infers that point clouds depend more on convolutional structures than images. Finally, a hybrid CNTK-NTK kernel (PointNTK) significantly outperforms the NTK baseline in few-shot point cloud recognition.

## Background & Motivation

**Background**: The Neural Tangent Kernel (NTK) transforms the training dynamics of infinite-width neural networks into a fixed kernel, allowing precise analysis of network inductive bias and convergence from a kernel regression perspective. Its convolutional variant, CNTK, corresponds to infinite-width CNNs; several works have studied its properties as a kernel and extended it in various directions.

**Limitations of Prior Work**: Although NTK and CNTK have been studied extensively individually, **where the spectral differences (eigenvalue distributions) between the two lie when applied to the same data** has rarely been systematically discussed. More problematically, most existing spectral analyses assume data is uniformly distributed on a high-dimensional sphere, completely ignoring the **tensor structure** of real-world data (images as 2D tensors, point clouds as 1D sequences). This makes it difficult to explain empirical phenomena such as "why convolutional networks generalize better" or "why deep CNNs are followed by several MLP layers."

**Key Challenge**: The inductive bias brought by convolutional structures is reflected in the spectrum of the kernel matrix, but there is a lack of a **distribution-independent** spectral measure to directly compare NTK and CNTK. Simultaneously, there is no computable indicator to determine "how suitable a specific dataset actually is for convolution."

**Goal**: (1) Characterize the spectral differences between NTK and CNTK under the assumption of general-dimensional tensor data; (2) Convert spectral differences into a data-level "Convolutional Suitability" metric; (3) Use this theory to explain and guide point cloud recognition—a field previously almost untouched by kernel regression.

**Key Insight**: The authors abandon the spherical uniformity assumption and instead assume $d+1$ dimensional data lies on a $d$-dimensional tensor space, following an arbitrary distribution with covariance matrix $\sigma_H$ (where the covariance is generally non-diagonal). They generalize CNTK to **arbitrary convolutional dimensions** (degenerating to NTK when $d=0$). This places NTK and CNTK in a unified framework where spectral differences are directly characterized by the covariance matrix.

**Core Idea**: Use three spectral statistics (mean $m_K$, mean square $s_K$, and dispersion $\beta_K$) as a "unified ruler" for NTK/CNTK. It is proved that $\beta_K$ equals 1 for CNTK at any layer and is consistently less than 1 for NTK. Thus, $1-\beta^{(0)}_{\mathrm{NTK}}$ naturally serves as a measure of "how suitable the data is for convolution."

## Method

### Overall Architecture
This paper is not a "network pipeline" but a **theoretical derivation chain**: Generalize CNTK to arbitrary dimensions and unify with NTK (general-dimensional CNTK recursion) $\to$ Define three spectral statistics $m_K, s_K, \beta_K$ $\to$ Prove the distribution-independent differences between NTK and CNTK for these three quantities (two spectral theorems) $\to$ Use the first-layer spectral difference $1-\beta^{(0)}_{\mathrm{NTK}}$ to define "Convolutional Suitability" and determine that point clouds rely more on convolution than images $\to$ Provide a closed-form hybrid kernel of CNTK followed by NTK, implemented as PointNTK for point cloud kernel regression.

The logic of the entire chain is that "spectral properties determine generalization, spectral properties are determined by data covariance, and the covariance structure in point clouds is closer to diagonal." Therefore, the theoretical conclusions lead directly to the practical suggestion that "point clouds should use convolutional kernel regression." Since the core involves matrix spectra and covariance operations rather than data flow branches, no architecture diagram is provided; the steps are explained via formulas.

### Key Designs

**1. General-Dimensional CNTK Recursion: Unifying NTK and CNTK in a Single Framework**

To compare NTK and CNTK, they must first be "comparable." This paper generalizes CNTK, usually written for 2D, to arbitrary convolutional dimension $d$. For tensor samples $x, x' \in \mathbb{R}^{\prod_{i=1}^{d} h_i \times n_0}$, the covariance $\Lambda^{(l)}_{p,q}$ and zero/first-order expectations $\Sigma^{(l)}_{p,q}, \dot\Sigma^{(l)}_{p,q}$ are maintained layer by layer at position $p$ and its convolutional neighborhood $N(p)$. The recursion for the tensor kernel is:

$$K^{(l)}_{p,q}(x,x') = \mathrm{tr}\!\left(K^{(l-1)}_{N(p),N(q)} \odot \dot\Sigma^{(l-1)}_{N(p),N(q)}(x,x')\right) + E^{(l-1)}_{p,q}(x,x'),$$

where $\odot$ denotes the Schur (element-wise) product, and the recursion base $\Sigma^{*(0)}_{p,q}(x,x') = x \otimes x'$ uses the Kronecker product to characterize interactions between internal elements of the tensors. With pooling, the kernel is $K_{wp}=P_{av}(K^{(L)})$; without pooling, $K_{wop}=\mathrm{tr}(K^{(L)})$. The key point is: **when $d=0$, where $p=q$ is fixed and $N(p)=p$, the recursion degenerates precisely to the standard NTK**. This relationship provides the common ground for all subsequent "NTK vs CNTK" comparisons.

**2. Three Spectral Statistics and Two Distribution-Independent Spectral Theorems: Explaining Why Convolution Generalizes Better**

Regarding the spectral differences between NTK and CNTK, this paper defines three computable spectral statistics:

$$m_K = \tfrac{1}{N}\mathrm{tr}(K),\quad s_K = \tfrac{1}{N^2}\mathrm{tr}(KK^\top),\quad \beta_K = \frac{s_K}{m_K^2}.$$

$m_K$ is the mean eigenvalue (reflecting convergence speed), $s_K$ is the mean squared eigenvalue, and $\beta_K$ characterizes the dispersion of eigenvalues—where $1/\beta_K$ is equivalent to the effective rank defined by Bartlett et al., which upper bounds the generalization error of kernel regression. Based on this, two conclusions **independent of any specific data distribution assumption** are provided:

- **Theorem 2 (Mean Difference)**: As $N\to\infty$, $m^{(L)}_{\mathrm{NTK}} \ge m^{(L)}_{\mathrm{CNTK}}$, meaning NTK has a larger average eigenvalue. The proof uses a stronger point-wise conclusion $K^{(L)}_{\mathrm{NTK}}(x,x)\ge K^{(L)}_{\mathrm{CNTK}}(x,x)$, requiring only that the dual function of the activation function is non-convex on the diagonal (not limited to ReLU).
- **Theorem 3 (Dispersion Difference)**: For any layers $L, L'$, $\beta^{(L)}_{\mathrm{NTK}} \le \beta^{(L')}_{\mathrm{CNTK}} = 1$. That is, **$\beta_K$ for CNTK is consistently 1 for any layer**, while for NTK, it is strictly less than 1 at each layer—this arises solely from the positive semi-definiteness of the covariance matrix under random initialization.

These two theorems explain empirical phenomena: a smaller $\beta_K$ indicates a more "diagonalized" kernel matrix and poorer generalization. Since NTK's $\beta$ is consistently smaller than CNTK's, **convolutional networks generalize better than MLPs**. Meanwhile, NTK's larger $m_K$ corresponds to faster convergence, explaining **why adding a few fully connected layers after deep convolutional networks accelerates training without damaging generalization**.

**3. Convolutional Suitability: A Computable Metric Based on Data Covariance**

The theorems state that "CNTK always has a more dispersed spectrum than NTK," but the degree of dispersion varies across data. This paper quantifies this as a data metric using the **first-layer spectral difference**. Proposition 4 gives the closed-form for the first layer ($L{=}0$):

$$\beta^{(0)}_{\mathrm{NTK}} = \frac{\mathrm{mean}(\sigma_H \odot \sigma_H)}{\mathrm{mean}(\mathrm{diag}(\sigma_H)\otimes\mathrm{diag}(\sigma_H))},\qquad \beta^{(0)}_{\mathrm{CNTK}}=1,$$

where $\sigma_H$ is the covariance matrix of the tensor input. Since $\beta$ for CNTK is always 1, the authors define:

$$\text{Convolutional Suitability} = \beta^{(L)}_{\mathrm{CNTK}} - \beta^{(0)}_{\mathrm{NTK}} = 1 - \beta^{(0)}_{\mathrm{NTK}}$$

as the measure of "how suitable the data is for convolutional networks": the smaller $\beta^{(0)}_{\mathrm{NTK}}$ is (closer the data covariance is to diagonal), the higher the suitability. This metric depends only on data covariance without training. Applying it to real data yields a key judgment: **Because of the disorder of points in point clouds, correlations between samples are weak, covariance is near-diagonal, and $\beta_K$ is smaller; thus, point clouds are more suitable for convolutional structures than images** (ModelNet series $\beta_K\!\sim\!10^{-3}$ in Table 1, whereas CIFAR10 reaches $0.137$).

**4. CNTK-NTK Hybrid Kernel and PointNTK: Applying Theory to Point Cloud Kernel Regression**

The theory suggests that "convolution followed by MLP combines the wide spectrum (II) of CNTK and the large offset (I) of NTK." Based on this, the paper derives the closed-form kernel for this practical architecture. Proposition 1 provides the combined kernel $K_c = P_{av}(K^{(L_1+L_2)})$ for a $d_1$-dimensional convolutional subnet $f_1$ followed by a $d_2$-dimensional subnet $f_2$ (connected via local average pooling $P^{d_1\to d_2}_{av}$). When $d_2=0$ (followed by MLP), this yields **CNTK-NTK**.

When applied to point clouds, the authors note that the shared-MLP in PointNet is essentially a **1D convolution with kernel size 1**. Consequently, they replicate the PointNet architecture to create **PointNTK**: using $L_1{=}4$ layers of 1D convolution and $L_2{=}3$ layers of MLP, with $d_1{=}1, d_2{=}0$. This is the first application of CNTK to point cloud tasks.

### Loss & Training
The kernel regression method does not require training; it provides a closed-form solution given the input. The control group PointNetm replaces training loss with least squares and max pooling with average pooling to align with the kernel regression setup. Ablations used 500 samples for training; few-shot experiments used 100 fixed samples covering all categories. Implementation used Python 3.7 + PyTorch 1.8 on a single RTX 3090, with no data augmentation (only global normalization).

## Key Experimental Results

### Main Results
Direct comparison of kernel regression: 1dCNTK for point clouds and 2dCNTK for images, both with $L=7$. The improvement of CNTK over NTK is much larger on point clouds than on images, confirming that "point clouds rely more on convolution."

| Dataset | NTK | CNTK | Gain | $\beta_K$ |
|--------|-----|------|------|-----------|
| ModelNet103 | 17.58 | 76.65 | +59.07 | 1.25e-3 |
| ModelNet106 | 16.19 | 91.96 | +75.77 | 1.12e-3 |
| ModelNet40 | 11.10 | 60.62 | +49.52 | 1.53e-3 |
| MNIST | 94.00 | 96.00 | +2.00 | 3.38e-2 |
| CIFAR10 | 59.19 | 76.79 | +17.60 | 1.37e-1 |

PointNTK vs PointNet (Selection, $f$ indicates training with only 100 samples):

| Method | ModelNet103 | ModelNet10$^f_3$ | ModelNet106 | ModelNet10$^f_6$ | ModelNet40 | ModelNet40$^f$ |
|------|------|------|------|------|------|------|
| PointNet | 91.12 | 68.92 | 92.08 | 70.22 | 88.71 | 38.32 |
| NTK | 17.58 | 11.12 | 16.19 | 11.78 | 11.10 | 3.69 |
| 1dCNTK | 76.65 | 71.03 | 91.96 | 81.16 | 60.62 | 45.22 |
| PointNTK | 86.56 | 73.68 | 91.19 | 78.52 | 80.47 | 45.71 |

Key phenomenon: PointNTK consistently outperforms 1dCNTK; it is slightly inferior to trained PointNet under **sufficient data** but **surpasses PointNet/PointNetm in all few-shot (100 samples) settings**.

### Ablation Study
Fixed at 500 samples, ablating network depth and kernel size.

| Configuration | Conclusion |
|------|------|
| Increasing only 1D convolution depth | Minimal gain or performance degradation |
| Fixed convolution, increasing FC depth | Regression performance decreases |
| Deepening convolution + FC at the end | Gain across all datasets, most effective |
| Both parts deepened simultaneously | Inferior to deepening only 1D convolution |
| Kernel size 1 vs >1 | Kernel size 1 is optimal |

### Key Findings
- **Terminal MLP is critical**: Simply deepening the 1D convolution is of little benefit, but "deepening convolution + terminal fully connected layers" yields improvements—matching the "wide spectrum + large offset" combination predicted by theorems (I)(II).
- **Kernel size must be 1**: Point clouds are unordered. When kernel size exceeds 1, points rolled into the same kernel lack meaningful spatial relationships, thus size=1 (shared-MLP) is optimal.
- **Few-shot is CNTK's home turf**: Kernel regression is deterministic and stable, outperforming trained PointNet on extremely small data (100 samples).

## Highlights & Insights
- **Distribution-Independent Spectral Theorems**: $\beta^{(L')}_{\mathrm{CNTK}}\equiv 1$ and $\beta^{(L)}_{\mathrm{NTK}}<1$ hold exclusively based on covariance positive semi-definiteness, offering universal conclusions.
- **Quantifying "Convolutional Suitability" as $1-\beta^{(0)}_{\mathrm{NTK}}$**: Predicts architecture suitability solely from data covariance and links it to effective rank/generalization bounds.
- **Theory Guiding Architecture**: Deduces "convolution followed by MLP" from "wide spectrum (II) + large offset (I)" and recognizes PointNet's shared-MLP as size-1 1D convolution.
- **First use of CNTK for point cloud kernel regression**, filling a gap in the field.

## Limitations & Future Work
- Calculating $\beta_K$ for arbitrary $L$-layer NTK still requires distribution assumptions; in practice, only the first layer ($L{=}0$) is used for assessment.
- PointNTK is still **inferior to trained PointNet** under sufficient data; its advantages are concentrated in few-shot scenarios.
- "Convolutional Suitability" assumes data follows a Gaussian distribution for covariance fitting; applicability to non-Gaussian/heavy-tailed data is not fully verified.
- Datasets are limited to ModelNet and CIFAR/MNIST; not tested on larger scales or more complex point cloud tasks.

## Related Work & Insights
- **vs Standard NTK (Jacot et al.) / CNTK (Arora et al.)**: Previous works provided recursions but did not systematically compare spectra on the same data; this paper provides the first distribution-independent spectral theorem.
- **vs Bartlett et al. (benign overfitting / effective rank)**: Borrows the $1/\beta_K$ definition but turns it from an analysis tool into a "Suitability Metric."
- **vs PointNet (Qi et al.)**: PointNet empirically proposed shared-MLP + pooling; this paper theoretically justifies size-1 1D convolution and terminal MLPs, providing the corresponding PointNTK.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐ 
- Writing Quality: ⭐⭐⭐ 
- Value: ⭐⭐⭐⭐ 

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Separable Neural Networks: Approximation Theory, NTK Regime, and Preconditioned Gradient Descent](separable_neural_networks_approximation_theory_ntk_regime_and_preconditioned_gra.md)
- [\[ICLR 2026\] Sublinear Spectral Clustering Oracle with Little Memory](sublinear_spectral_clustering_oracle_with_little_memory.md)
- [\[ICLR 2026\] Provable Separations between Memorization and Generalization in Diffusion Models](provable_separations_between_memorization_and_generalization_in_diffusion_models.md)
- [\[ICLR 2026\] From Neural Networks to Logical Theories: The Correspondence between Fibring Modal Logics and Fibring Neural Networks](from_neural_networks_to_logical_theories_the_correspondence_between_fibring_moda.md)
- [\[ICLR 2026\] A New Initialization to Control Gradients in Sinusoidal Neural Networks](a_new_initialization_to_control_gradients_in_sinusoidal_neural_networks.md)

</div>

<!-- RELATED:END -->
