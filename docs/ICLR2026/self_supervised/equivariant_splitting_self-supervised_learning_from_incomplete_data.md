---
title: >-
  [Paper Note] Equivariant Splitting: Self-supervised learning from incomplete data
description: >-
  [ICLR 2026][Self-Supervised Learning][inverse problems] By combining the invariance prior of "Equivariant Imaging (EI)" with the efficient unbiased properties of "measurement splitting," this paper proposes the Equivariant Splitting (ES) loss. This allows training a reconstructor that approximates the MMSE using only a **single highly under-sampled forward operator**, without requiring multiple forward evaluations as in EI.
tags:
  - "ICLR 2026"
  - "Self-Supervised Learning"
  - "inverse problems"
  - "equivariant networks"
  - "measurement splitting"
  - "MMSE estimation"
date: 2026-05-08
content_hash: 63bd3798ad971478
---

# Equivariant Splitting: Self-supervised learning from incomplete data

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=upMIVpe467](https://openreview.net/forum?id=upMIVpe467)  
**Code**: [vsechaud/Equivariant-Splitting](https://github.com/vsechaud/Equivariant-Splitting)  
**Area**: Self-supervised Learning / Inverse Problem Reconstruction  
**Keywords**: Self-supervised learning, inverse problems, equivariant networks, measurement splitting, MMSE estimation

## TL;DR
By combining the invariance prior of "Equivariant Imaging (EI)" with the efficient unbiased properties of "measurement splitting," this paper proposes the Equivariant Splitting (ES) loss. This allows training a reconstructor that approximates the MMSE using only a **single highly under-sampled forward operator**, without requiring multiple forward evaluations as in EI.

## Background & Motivation
**Background**: Inverse problems $y = Ax + \varepsilon$ (MRI, CT, compressed sensing, inpainting) typically involve rank-deficient forward matrices $A$. Since supervised training requires ground-truth labels that are difficult to obtain, self-supervised reconstruction has become a critical necessity.

**Limitations of Prior Work**:
- **Measurement Splitting** partitions measurements into input/target pairs to predict each other, serving as an unbiased estimate of the supervised loss. However, this only holds when the dataset contains **diverse operators** (e.g., accelerated MRI where each image has a different mask); it fails in single-operator scenarios.
- **Equivariant Imaging (EI)** can handle a single operator but requires 2–3 forward passes through the network per step (expensive). Furthermore, its equivariant loss is only valid when reconstruction is nearly perfect, proves unreliable for ill-posed problems, and does not guarantee convergence to the MMSE estimator.

**Key Challenge**: Under a single, fixed, highly incomplete forward operator, splitting lacks operator diversity, while EI is both slow and suboptimal.

**Goal**: In single-operator under-sampling settings (e.g., accelerated MRI with a fixed mask, sparse-view CT), realize the efficient unbiased nature of splitting while utilizing the invariance prior of EI to recover information in the null space.

**Core Idea**: Under the invariance assumption $p(T_g x)=p(x)$, a single set of measurements $y=Ax$ can be re-interpreted as a "virtual ground truth $x_g=T_g^{-1}x$ + virtual operator $A_g=AT_g$." This **artificially creates an implicit multi-operator structure** $\{AT_g\}_{g\in G}$, enabling the application of measurement splitting.

## Method

### Overall Architecture
The key observation of ES is: $y = Ax + \varepsilon = AT_g\,T_g^{-1}x + \varepsilon = A_g x_g + \varepsilon$. By using group transformations $\{T_g\}$, a single operator is disguised as multiple operators, and measurement splitting is applied to these virtual operators. Furthermore, if the **reconstructor architecture is equivariant**, the ES loss collapses into the standard splitting loss, achieving "zero-overhead implicit data augmentation" without explicit transformation sampling or multiple forward passes.

```mermaid
graph LR
    A[Single measurement y, Single operator A] --> B[Invariance assumption p(Tg·x)=p(x)]
    B --> C["Virtual multi-operator {A·Tg}<br/>y=A·Tg·(Tg⁻¹x)"]
    C --> D[Apply measurement splitting to virtual operators]
    D --> E["Equivariant reconstructor<br/>f(y,A·Tg)=Tg⁻¹·f(y,A)"]
    E --> F["Thm 3: ES loss ≡ Standard splitting loss<br/>No extra forward passes"]
    F --> G[Global optimum = MMSE estimator]
```

### Key Designs
**1. Equivariant Splitting Loss: Transforming Invariance into Implicit Multi-operators**
The ES loss is defined as the splitting loss marginalized over the group transformations:
$$\mathcal{L}_{\mathrm{ES}}(y,A,f) = \mathbb{E}_g\big\{\mathcal{L}_{\mathrm{SPLIT}}(y, AT_g, f)\big\} = \mathbb{E}_g\Big\{\mathbb{E}_{y_1,A_1|y,AT_g}\big[\|AT_g f(y_1,A_1)-y\|^2\big]\Big\}$$
where $A_1$ is a random split of $AT_g$. **Theorem 1** guarantees optimality: under noiseless conditions and an invariant $p(x)$, if the matrix $Q_{A_1}=\mathbb{E}_{g|A_1}[(AT_g)^\top AT_g]$ is full rank, the global minimizer of ES is the supervised MMSE estimator $f^*(y_1,A_1)=\mathbb{E}_{x|y_1,A_1}\{x\}$. This is a property not guaranteed by EI or previous splitting methods.

**2. New Definition of Equivariant Reconstructors: Necessary Condition for Null Space Recovery**
The authors introduce an equivariance definition specifically **targeted at reconstruction functions** (rather than image-to-image mappings):
$$f(y, AT_g) = T_g^{-1} f(y, A),\quad \forall y, g, A.$$
**Theorem 2** proves that a wide class of classical reconstructors satisfy this: artifact removal networks $f=\phi(A^\top y)$, unrolled networks (containing gradient steps + equivariant denoiser $\phi$), Reynolds averaging, MAP, and MMSE estimators. Meanwhile, **Corollary 1** points out a critical constraint: the forward operator itself **cannot** be equivariant to the transformation group ($\exists g, AT_g\neq T_g A$), otherwise $Q_{A_1}$ will not be full rank, and null space information cannot be learned. This guides the choice of transformations for each task: translation equivariance for compressed sensing/inpainting, and rotation/reflection equivariance for MRI/CT.

**3. Computational Synergy: Equivariant Architectures Reduce ES to Zero-Overhead Splitting**
**Theorem 3**: If $f$ is an equivariant reconstructor, then $\mathcal{L}_{\mathrm{ES}}(y,A,f)=\mathcal{L}_{\mathrm{SPLIT}}(y,A,f)$. This means **explicit sampling and applying $T_g$ transformations are unnecessary**. As long as equivariance constraints are built into the architecture (using translation-equivariant UNets or Reynolds averaging for rotation/reflection), the effect of "data augmentation over all $T_g$" is automatically achieved without any transformation overhead or extra forward passes—providing a significant efficiency advantage over EI (which requires 2–3 passes).

**4. Extension to Noisy Data: Replacing Consistency Terms with R2R**
The ES loss can be decomposed into a measurement consistency term and a prediction term. When measurements contain Gaussian noise, the consistency term is replaced with the Recorrupted2Recorrupted (R2R) loss:
$$\mathcal{L}_{\text{G-ES}} = \mathbb{E}\Big\{\big\|A_1 f(y_1+\alpha\omega, A_1)-(y_1-\tfrac{\omega}{\alpha})\big\|^2 + \|A_2 f(y_1+\alpha\omega,A_1)-y_2\|^2\Big\},\ \omega\sim\mathcal{N}(0,\sigma^2 I)$$
R2R provides an unbiased estimate of the consistency term, ensuring Theorem 1 still holds and the model converges to the MMSE. Non-Gaussian generalizations can be applied similarly. During inference, averaging over multiple random splits (e.g., 10) and synthetic noise is performed.

## Key Experimental Results
Four types of inverse problems are covered: compressed sensing (MNIST), inpainting (DIV2K, ~30% pixels), accelerated MRI (FastMRI, ×8), and sparse-view CT (LIDC-IDRI, 50 views). A unified architecture and training pipeline are used for fair comparison.

### Main Results

| Task | Metric | ES (Ours) | EI (Prev. SOTA) | Supervised Upper Bound | Gain vs EI |
|--------|------|------|------|------|---|
| Inpainting (DIV2K) | PSNR | **27.45** | 25.89 | 28.46 | **+1.56 dB** |
| Inpainting | SSIM | **0.8737** | 0.8332 | 0.8982 | **+0.040** |
| MRI ×8, 40dB | PSNR | **28.54** | 27.88 | 28.74 | **+0.66 dB** |
| MRI ×8 (Real measurements) | PSNR | **28.30** | 27.88 | 28.81 | **+0.42 dB** |
| CT 50 views, 50dB | PSNR | **32.62** | 28.61 | 33.99 | **+4.01 dB** |
| CT 50 views | SSIM | **0.8570** | 0.7400 | 0.8819 | **+0.117** |

ES outperforms self-supervised baselines like EI, SURE, and MC across all tasks and approaches the supervised upper bound. The lead over EI is most significant in highly ill-posed problems like CT (+4 dB). Compressive sensing curves show that the gap for EI widens as the compression ratio increases, while ES consistently remains close to the supervised line.

### Ablation Study (Impact of Equivariant Architectures and Splitting Loss)

| Task | Equivariant Architecture | PSNR | EQUIV |
|------|----------|------|-------|
| Inpainting | ✓ | **27.45** | **27.46** |
| Inpainting | ✗ | 27.20 | 26.52 |
| MRI ×8 | ✓ | **28.54** | **31.53** |
| MRI ×8 | ✗ | 28.18 | 27.28 |

### Key Findings
- There is a **synergy** between equivariant architectures and the splitting loss: the equivariant versions consistently perform better and exhibit higher equivariance (EQUIV), validating the theoretical predictions of Theorem 3.
- Non-equivariant architectures still exhibit "unexpectedly" high equivariance—a phenomenon termed learned equivariance—which might explain why splitting methods perform decently even without equivariant architectures.
- Baselines that cannot recover null space information, such as SURE, IDFT, and FBP, fail significantly on MRI and CT, confirming the theory.

## Highlights & Insights
- **Elegant Conceptual Reframing**: By using "$y=A_g x_g$", the invariance assumption is translated into an implicit multi-operator framework, unifying splitting (which requires multiple operators) and EI (which handles a single operator) in a single-operator setting.
- **First Equivariant Definition for Reconstruction Functions** $f(y,AT_g)=T_g^{-1}f(y,A)$, which differs from the forced $f(AT_g x,A)=T_g f(Ax,A)$ in EI (the latter being equivalent only under perfect reconstruction). Provision of proofs for unrolled/artifact-removal/MAP/MMSE models is a contribution.
- **Win-Win for Theory and Efficiency**: The method provides an MMSE optimality guarantee (lacking in EI) while leveraging architectural equivariance to eliminate the 2–3 forward pass overhead of EI.

## Limitations & Future Work
- Optimality depends on $Q_{A_1}$ / $\bar{Q}_A$ being full rank. When closed-form solutions are unavailable, unweighted averages (Eq. 12) are used as approximations; actual optimality is determined by the number of "non-negligible eigenvalues" in the spectrum, which may be compromised in highly ill-posed cases.
- Invariance assumptions are natural for natural/remote-sensing/microscopy images, but medical images only exhibit "approximate invariance," and the method relies on this approximation being valid.
- Only distortion metrics (PSNR/SSIM) are reported; perceptual quality was not evaluated as the authors noted the inherent trade-off between perception and distortion.
- The transformation group must be manually selected based on the problem (guided by Corollary 1); automatic selection remains an open problem.

## Related Work & Insights
- **vs. Measurement Splitting (Millard & Chiew 2023; Daras 2023)**: Original methods require operator diversity in the dataset; this work extends it to single-operator under-sampling (fixed mask MRI, sparse CT) for the first time.
- **vs. Equivariant Imaging EI (Chen 2021/2022)**: Shares the invariance assumption, but ES replaces loss-based equivariance with architecture-based equivariance, becoming faster, more stable, and offering MMSE guarantees.
- **vs. Equivariant Networks (Cohen & Welling; Chaman & Dokmanić)**: Re-purposes equivariant architectures, traditionally used to "improve test-time generalization," as a tool for "learning from incomplete data," expanding the utility of equivariant networks.
- **Insight**: Invariance is a powerful prior for learning from incomplete measurements. Internalizing "data augmentation" into the architecture rather than the loss yields both efficiency and optimality in self-supervised inverse problems.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Unifies splitting and EI via "virtual multi-operators" and provides the first equivariant definition for reconstruction functions; conceptually clear and fills a gap in single-operator under-sampling.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 4 types of inverse problems + real MRI + ablations with fair comparisons; however, perceptual metrics and larger-scale natural images are missing.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical progression from theory (3 theorems + corollary) to motivation; well-established background and high readability.
- Value: ⭐⭐⭐⭐ Practical for scenarios without GT (medical imaging, astronomy, microscopy); its theoretical guarantees and efficiency advantage position it to become a new baseline for self-supervised inverse problems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Understanding the Robustness of Distributed Self-Supervised Learning Frameworks Against Non-IID Data](understanding_the_robustness_of_distributed_self-supervised_learning_frameworks_.md)
- [\[ICLR 2026\] On the Alignment Between Supervised and Self-Supervised Contrastive Learning](on_the_alignment_between_supervised_and_self-supervised_contrastive_learning.md)
- [\[ICLR 2026\] Incomplete Multi-View Multi-Label Classification via Shared Codebook and Fused-Teacher Self-Distillation](incomplete_multi-view_multi-label_classification_via_shared_codebook_and_fused-t.md)
- [\[ICLR 2026\] Understanding the Learning Phases in Self-Supervised Learning via Critical Periods](understanding_the_learning_phases_in_self-supervised_learning_via_critical_perio.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](soft_equivariance_regularization_for_invariant_self-supervised_learning.md)

</div>

<!-- RELATED:END -->
