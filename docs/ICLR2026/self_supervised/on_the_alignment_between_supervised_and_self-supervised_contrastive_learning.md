---
title: >-
  [Paper Note] On the Alignment Between Supervised and Self-Supervised Contrastive Learning
description: >-
  [ICLR 2026][Self-Supervised Learning][Contrastive Learning] This paper theoretically proves that under shared randomness, self-supervised contrastive learning (CL) and a supervised surrogate—"Negative-only Supervised Contrastive Learning" (NSCL)—maintain a high level of alignment in the **representation similarity space** throughout training (with high-probability lower bounds for CKA/RSA), even while their **parameters** might diverge exponentially. This establishes NSCL as…
tags:
  - "ICLR 2026"
  - "Self-Supervised Learning"
  - "Contrastive Learning"
  - "Self-supervised"
  - "NSCL"
  - "Representation Alignment"
  - "CKA/RSA"
date: 2026-05-08
content_hash: 08764bb227e4a80a
---

# On the Alignment Between Supervised and Self-Supervised Contrastive Learning

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=JkitQScjuL](https://openreview.net/forum?id=JkitQScjuL)  
**Code**: https://dlfundamentals.github.io/cl-nscl-representation-alignment  
**Area**: Self-Supervised / Representation Learning / Theory of Contrastive Learning  
**Keywords**: Contrastive Learning, Self-supervised, NSCL, Representation Alignment, CKA/RSA

## TL;DR
This paper theoretically proves that under shared randomness, self-supervised contrastive learning (CL) and a supervised surrogate—"Negative-only Supervised Contrastive Learning" (NSCL)—maintain a high level of alignment in the **representation similarity space** throughout training (with high-probability lower bounds for CKA/RSA), even while their **parameters** might diverge exponentially. This establishes NSCL as a principled bridge connecting self-supervised and supervised learning.

## Background & Motivation
**Background**: Self-supervised contrastive learning (SimCLR, MoCo, CPC, etc.) can learn representations comparable to or even exceeding supervised pre-training. The core mechanism involves "pulling together augmented views of the same sample and pushing apart other samples." A long-standing puzzle is how CL, despite having no labels, learns features that align closely with semantic category boundaries.

**Limitations of Prior Work**: Recently, Luthra et al. (2025) provided an explanation at the **loss level**—the InfoNCE objective of CL approaches a supervised variant, NSCL (which excludes same-class samples from the denominator and normalizes over negative samples only), at a rate of $O(1/C)$ as the number of classes $C$ increases. However, this only indicates that the two **objective functions** are close; it does not guarantee that the two **optimization trajectories** will converge. Curvature differences, gradient noise, and learning rate schedules could amplify small differences in loss, causing SGD trajectories to diverge.

**Key Challenge**: Similarity in loss $\neq$ similarity in representations. Downstream behavior is determined by the geometry of representations, not loss values. Therefore, the question of whether CL merely converges to a solution similar to NSCL or remains coupled with NSCL throughout the entire training process remains unresolved.

**Goal**: Characterize the alignment between CL and NSCL throughout the training process under **shared randomness** (same initialization, same mini-batch, same augmentation): (1) Are their representations consistently similar? (2) Under what conditions does alignment occur, and what factors control the strength of this alignment? (3) Does the coupling also hold in the parameter space?

**Key Insight**: Instead of performing analysis in the parameter space (where parameter drift is uncontrollable under non-convex dynamics and sensitive to reparameterization), this study analyzes the **similarity matrix space**—a perspective that is invariant to reparameterization and directly characterizes representation geometry.

**Core Idea**: A "similarity descent" surrogate is used to track the evolution of the similarity matrices for CL and NSCL. It is proven that their Frobenius drift is bounded by a term that varies systematically with the number of classes, batch size, and temperature. This bound is then translated into high-probability lower bounds for CKA and RSA.

## Method

### Overall Architecture
This is a **pure theoretical analysis** paper (no structural innovations). Its "method" consists of a linked chain of proofs. Given a dataset $S=\{(x_i,y_i)\}_{i=1}^N$ with $C$ classes, an encoder $f_w$ maps inputs to embeddings, and similarity is measured using $\ell_2$-normalized cosine similarity. For an anchor $i$ within a batch, the per-anchor loss for CL treats all other samples (including those of the same class) as negatives in the denominator, while NSCL only includes **different-class** samples in the denominator:

$$\ell^{CL}_i = -\log\frac{\exp(\mathrm{sim}(z_i,z_i')/\tau)}{\sum_{t\neq i}\exp(\mathrm{sim}(z_i,z_t)/\tau)+\exp(\mathrm{sim}(z_i,z_t')/\tau)},\quad \ell^{NSCL}_i=-\log\frac{\exp(\mathrm{sim}(z_i,z_i')/\tau)}{\sum_{j\in I_i^-}[\cdots]}$$

where $I_i^-=\{j:y_j\neq y_i\}$ is the index set of different-class negative samples. The analysis starts from the observation that "analysis in the parameter space will explode," shifts to the similarity space, and proceeds in three steps: ① The similarity updates induced by parameter SGD are approximated as a "similarity descent" surrogate dynamics that only updates entries touched by the current batch; ② The Frobenius drift of the two similarity trajectories is proven to satisfy a recurrence, yielding an exponential coupling bound (Theorem 1); ③ This bound is translated into high-probability lower bounds for CKA and RSA, which are standard representation alignment metrics (Corollaries 1–2). Finally, instability results in the parameter space (Theorem 2) serve as a contrast, showing that "parameter divergence" and "representation alignment" are not contradictory.

To quantify representation similarity, the paper uses two metrics: Linear CKA is the normalized Frobenius inner product of two **centered** similarity matrices, $\mathrm{CKA}(Z,Z')=\frac{\langle H\Sigma(Z)H,\,H\Sigma(Z')H\rangle_F}{\|H\Sigma(Z)H\|_F\,\|H\Sigma(Z')H\|_F}$, where $H=I-\frac1N\mathbf{1}\mathbf{1}^\top$ is the centering projection; RSA is the Pearson correlation between the upper triangular off-diagonal elements of two dissimilarity matrices $\mathrm{RDM}=\mathbf{1}\mathbf{1}^\top-\Sigma$. Both metrics fall within $[0,1]$, where values closer to 1 indicate more consistent similarity structures.

### Key Designs

**1. Similarity Space Surrogate Dynamics: Bypassing Uncontrollability in Parameter Space**

The authors first argue that studying CL and NSCL trajectories directly in the parameter space is problematic—on non-convex losses without convexity or strong-convexity assumptions, small reparameterizations can distort distances, and parameter drift grows uncontrollably over time. Consequently, the object of analysis is shifted from weights $w$ to the similarity matrix $\Sigma_t\in[-1,1]^{N\times N}$ (pairwise cosine similarities of embeddings for a fixed reference set). To make the analysis analytical, a "similarity descent" surrogate is defined: each step only updates the entries touched by the current mini-batch, $\Sigma^{CL}_{t+1}=\Sigma^{CL}_t-\eta_t G^{CL}_t$, and $\Sigma^{NSCL}_{t+1}=\Sigma^{NSCL}_t-\eta_t G^{NSCL}_t$, where $G_t=\nabla_\Sigma\bar\ell_{B_t}(\Sigma_t)$ is the batch gradient map when the loss is written as a function of similarity entries, with untouched entries set to zero. Appendix D proves that under regularity conditions—such as bounded Jacobian spectral norm $\|J(w)\|_{2\to2}\le L_\Sigma$, bounded second-order Taylor remainders, and a learning rate schedule where $\sum_t\eta_t/(\tau^2 B)$ and $\sum_t\eta_t^2$ are bounded—this surrogate trajectory uniformly approximates the similarity trajectory $\hat\Sigma_t=\Sigma(w_t)$ induced by true parameter SGD. The intuition is that with small step sizes, sufficiently large batches, and moderate temperatures, the way parameter SGD drives similarity is nearly equivalent to performing gradient descent directly in the similarity space. This step is the pivot of the entire analysis, converting a problem that is sensitive to reparameterization and prone to explosion into one that is invariant to reparameterization and controllable.

**2. Similarity Coupling Bound: Controlling Drift with Discrete Grönwall**

This is the core result (Theorem 1). Under shared randomness, the authors first provide an estimate of the per-step gradient mismatch: the CL–NSCL batch gradient difference is decomposed into (i) a **reweighting error** (the normalization difference caused by NSCL excluding same-class samples, bounded by $\Delta_{\pi,\delta}(B;\tau)$ according to total variation) and (ii) a **stability term** (the dependence of the gradient map on the current similarity, controlled by the $\frac{1}{2\tau^2 B}$-Lipschitz property of the batch gradient map). Leveraging block orthogonality across anchors to combine reweighting contributions via the sum of squares, they obtain:

$$\big\|G^{CL}_t-G^{NSCL}_t\big\|_F\le\frac1\tau\cdot\frac{\Delta_{\pi,\delta}(B;\tau)}{\sqrt B}+\frac{1}{2\tau^2 B}\big\|\Sigma^{CL}_t-\Sigma^{NSCL}_t\big\|_F.$$

Substituting this into the update yields the drift recurrence $\|\Sigma^{CL}_{t+1}-\Sigma^{NSCL}_{t+1}\|_F\le(1+\frac{\eta_t}{2\tau^2 B})\|\Sigma^{CL}_t-\Sigma^{NSCL}_t\|_F+\eta_t\frac{\Delta_{\pi,\delta}(B;\tau)}{\tau\sqrt B}$—the first term propagates existing error, while the second injects new differences. Expanding this recurrence (Discrete Grönwall Inequality) gives a bound that holds with **at least $1-\delta$ probability**:

$$\big\|\Sigma^{CL}_T-\Sigma^{NSCL}_T\big\|_F\le\exp\!\Big(\frac{1}{2\tau^2 B}\sum_{t=0}^{T-1}\eta_t\Big)\Big(\frac{1}{\tau\sqrt B}\sum_{t=0}^{T-1}\eta_t\Big)\,\Delta_{\pi,\delta}(B;\tau),$$

where $\Delta_{\pi,\delta}(B;\tau)=\dfrac{2\,e^{2/\tau}(\pi_{\max}+\epsilon_{B,\delta})}{1-\pi_{\max}-\epsilon_{B,\delta}}$, $\epsilon_{B,\delta}=\sqrt{\frac{1}{2B}\log(TB/\delta)}$, and $\pi_{\max}=\max_c\pi_c$ is the maximum class prior. The value of this bound is that it **translates the abstract notion of "when alignment occurs" into monotonic relationships with common CL hyperparameters**: as the number of classes $C$ increases ($\pi_{\max}\approx1/C$ decreases for balanced classes, reducing $\Delta$), batch size $B$ increases (simultaneously lowering concentration error $\epsilon_{B,\delta}$, the pre-factor $1/\sqrt B$, and the exponent $\frac{1}{2\tau^2 B}$), temperature $\tau$ increases (lowering $1/\tau$, $1/\tau^2$, and $e^{2/\tau}$), or effective step size $\sum_t\eta_t$ decreases, the bound becomes tighter—precisely the empirical regimes where "CL behaves like NSCL." The key insight is that the "instability rate" in similarity space is only $\frac{1}{2\tau^2 B}$, which is negligible for typical $B\sim10^2$–$10^3$, fundamentally different from the growth rate in parameter space governed by the smoothness coefficient $\beta$.

**3. From Similarity Drift to CKA/RSA Guarantees: Observational Metrics**

Theorem 1 controls the Frobenius drift, but in practice, CKA and RSA are the observed metrics. This step translates the bound. Since centering is a contractive mapping $\|HXH\|_F\le\|X\|_F$, the bound on $\Sigma$ automatically controls the difference of the centered Gram matrix $K=H\Sigma H$. Defining the relative deviation as $\rho_T=\|K^{CL}_T-K^{NSCL}_T\|_F/\|K^{CL}_T\|_F$, Corollary 1 provides a **high-probability CKA lower bound** $\mathrm{CKA}_T\ge\frac{1-\rho_T}{1+\rho_T}$. Similarly for RSA, defining $r_T=\|b_T-a_T\|_2/(\sqrt M\,\sigma_{D,T})$ (where $a_T,b_T$ are upper triangular vectors of RDMs, $M=\binom N2$, and $\sigma_{D,T}$ is the empirical standard deviation of $a_T$), Corollary 2 gives $\mathrm{RSA}_T\ge\frac{1-r_T}{1+r_T}$. In realistic regimes ($C\sim10^3$, $B\sim10^2$–$10^3$), $\rho_T, r_T \ll 1$, meaning both metrics are anchored close to 1. This explains the phenomenon in Figure 1: **Even if weights diverge, the induced representations evolve in a coupled and stable manner**.

**4. Intrinsic Instability in Parameter Space: Divergence and Alignment are Compatible**

For completeness, the authors provide a parameter drift bound under $\beta$-smoothness assumptions (Theorem 2), proving that the **weight difference between CL and NSCL can grow exponentially with training time**—the growth rate is dominated by the smoothness coefficient $\beta$, which is far more aggressive than the $\frac{1}{2\tau^2 B}$ in similarity space. This contrast forms the conceptual claim of the paper: parameter coupling is inherently unstable, while representation coupling is inherently stable. Thus, "two models' weights growing further apart while their representations remain aligned" can occur simultaneously. This fundamentally explains why the relationship between self-supervised and supervised learning should be measured in similarity space rather than parameter space.

### Loss & Training
The analysis targets the CL (InfoNCE type) and NSCL (negative-only normalization excluding same-class samples) losses themselves. In experiments, CL uses a decoupled DCL loss (to avoid positive-negative coupling); supervised baselines include NSCL, SCL (Supervised Contrastive), and CE (Cross-Entropy). Key technical assumptions for the proof include shared randomness, $\beta$-smoothness, bounded Jacobian spectral norm, and a high-probability batch composition guarantee (Corollary 3: ensuring the proportion of negative samples in the denominator does not deviate significantly from the expected value to exclude "bad batches" with too many positive samples).

## Key Experimental Results

Experiments use a ResNet-50 encoder + two-layer MLP projection head (2048→2048→ReLU→128), LARS optimizer, and batch size $B=1024$. Models are trained following the SimCLR recipe across CIFAR-10/100, Mini-ImageNet, Tiny-ImageNet, and ImageNet-1K.

### Main Results
Downstream evaluation uses Nearest Class Center Classifier (NCCC) and Linear Probe (LP) accuracy (%). Note: The argument is not that NSCL is the strongest for downstream tasks, but that NSCL and CL representations are most aligned—thus, higher SCL/CE downstream performance does not weaken the claim.

| Dataset | Metric | CL | NSCL | SCL | CE |
|--------|------|----|------|-----|-----|
| CIFAR-10 | NCCC / LP | 88.37 / 90.16 | 94.47 / 94.09 | 94.93 / 94.67 | 92.97 / 93.39 |
| CIFAR-100 | NCCC / LP | 54.62 / 65.65 | 60.14 / 68.38 | 64.06 / 69.52 | 67.35 / 68.04 |
| Mini-ImageNet | NCCC / LP | 60.78 / 65.30 | 63.92 / 72.60 | 74.78 / 76.00 | 75.20 / 74.00 |
| Tiny-ImageNet | NCCC / LP | 40.59 / 44.61 | 40.76 / 45.79 | 48.63 / 48.73 | 48.28 / 52.57 |

The most compelling alignment data: after 1000 epochs on Tiny-ImageNet, the **CKA between CL and NSCL reaches 0.87, while the CKA between CL and SCL is only 0.043**—NSCL tracks CL much more closely than any other supervised objective.

### Ablation Study

| Control Variable | Observation | Theoretical Correspondence |
|----------|------|----------|
| Class count $C'$ (Training 1000 ep on $C'$-way subset) | CKA/RSA increases monotonically with $C'$ (Fig 3, consistent across datasets) | $1/C$ term in $\Delta$ decreases as $C$ increases |
| Temperature $\tau\in\{0.1,0.5,1.0\}$ (300 ep) | Highest alignment throughout for $\tau=1.0$ | $1/\tau$, $1/\tau^2$ terms in pre-factor/exponent decrease as $\tau$ grows |
| Batch size $B$ and LR scaling | Alignment **decreases** with $B$ when $\eta=O(B)$; **increases** when $\eta$ is $O(\sqrt B)/O(\sqrt[4]B)/$ constant | The effect of $B$ on the bound depends on how $\eta$ scales; empirical directions match theoretical signs |
| Training duration | CL aligns most closely with NSCL for the first ~1000 ep; alignment drops later | NSCL enters the Neural Collapse regime later than SCL/CE |
| Weight space | Both NSCL and SCL weights increasingly diverge from CL during training | Theorem 2: Parameter divergence can grow exponentially |

### Key Findings
- **NSCL is consistently the supervised objective most aligned with CL**, far surpassing SCL and CE. The mechanism is that all three induce Neural Collapse, but NSCL's structure is most similar to CL (both pull one positive, push negatives, and perform instance-level discrimination), whereas SCL imposes stronger class-level constraints forming clusters faster, with CE in between.
- The batch experiments provide the most granular verification: the direction of $B$'s impact on alignment **flips depending on learning rate scaling**. The empirical alignment changes across four scaling methods match theoretical predictions, indicating the bound captures non-trivial dynamics.
- Representation alignment and weight divergence were **observed simultaneously in the same experiments**, confirming that "parameter divergence $\neq$ representation divergence."

## Highlights & Insights
- **Solvability through Space Transformation**: By shifting the question from "parameter trajectory coupling" (which explodes) to "similarity matrix coupling" (which is invariant to reparameterization and controllable via Grönwall), a formerly intractable problem becomes solvable.
- **Bounds as Control Knobs**: Theorem 1 provides an explicit bound with monotonic factors for class count, batch size, temperature, and learning rate. The theory provides an actionable recipe: to make self-supervised learning more like supervised learning, increase $C$/$B$/$\tau$ and decrease step size.
- **Perspective Shift**: While traditional theory focuses on whether self-supervised loss minimization guarantees downstream classification accuracy, this paper shifts focus to whether CL and NSCL induce similar similarity structures, which is more relevant for tasks relying on representation geometry (e.g., interpretability, segmentation).
- **Explanation of a Counter-Intuitive Conclusion**: Geometric alignment can persist despite exponential parameter divergence. This warns future work on model similarity or merging not to use parameter distance as a proxy for representation similarity.

## Limitations & Future Work
- The theory is built on the strong assumption of shared randomness (same initialization, batch, and augmentation); whether alignment holds and how strong it is for independently trained models is not theoretically covered.
- Surrogate dynamics' uniform approximation of true SGD depends on regularity conditions (bounded Jacobian norm, controlled second-order remainder, specific LR schedules) that may not strictly hold in deep networks.
- While "alignment drops after very long training" was observed, the bound in Theorem 1 only loosens with $\sum_t\eta_t$ and does not predict the precise shape of an alignment **reset**, leaving a gap between theory and long-term empirical observation.
- All experiments were limited to computer vision (ResNet-50 + SimCLR). Applicability to other contrastive learning scenarios like language, audio, or multimodal data remains to be tested.

## Related Work & Insights
- **vs Luthra et al. (2025)**: They proved CL and NSCL converge at the **loss level** at $O(1/C)$ and characterized NSCL minima; this paper extends this by asking if **representations** remain close throughout training, moving from objective functions to training dynamics and similarity geometry.
- **vs Balestriero & LeCun (2024)**: They proved SSL objectives like VICReg are equivalent to supervised quadratic losses in linear models; this paper is not limited to linear models or specific architectures and provides a labels-independent similarity coupling bound for the entire training process.
- **vs alignment/uniformity (Wang & Isola 2020, etc.)**: That line of work uses positive pair clustering and negative pair uniformity to characterize CL geometry but doesn't explain semantic organization. This paper answers how supervised signals are implicit in CL by aligning it with a supervised objective's similarity structure.
- **vs Grigg et al. (2021)**: They empirically observed geometric alignment between supervised and self-supervised models; this paper provides a theoretical explanation and quantifies alignment as a function of class count, batch size, temperature, and learning rate.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Advancing alignment analysis from loss level to training dynamics in similarity space is highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five datasets verify four types of theoretical predictions, though limited to a single architecture in vision.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, clean theoretical narrative, and consistent correspondence between theory and empirical results.
- Value: ⭐⭐⭐⭐ Provides a principled bridge for why self-supervised learning rivals supervised learning and offers actionable tuning recipes for alignment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Dual Perspectives on Non-Contrastive Self-Supervised Learning](dual_perspectives_on_non-contrastive_self-supervised_learning.md)
- [\[NeurIPS 2025\] Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning](../../NeurIPS2025/self_supervised/self-supervised_contrastive_learning_is_approximately_supervised_contrastive_lea.md)
- [\[ICLR 2026\] Understanding the Robustness of Distributed Self-Supervised Learning Frameworks Against Non-IID Data](understanding_the_robustness_of_distributed_self-supervised_learning_frameworks_.md)
- [\[ICLR 2026\] Soft Equivariance Regularization for Invariant Self-Supervised Learning](soft_equivariance_regularization_for_invariant_self-supervised_learning.md)
- [\[ICLR 2026\] Equivariant Splitting: Self-supervised learning from incomplete data](equivariant_splitting_self-supervised_learning_from_incomplete_data.md)

</div>

<!-- RELATED:END -->
