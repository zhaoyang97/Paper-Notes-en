---
title: >-
  [Paper Note] CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations
description: >-
  [ICML 2026][Multi-task learning] The authors reattribute the root cause of "negative transfer" in multi-task learning from "gradient conflict" to the "entanglement of semantics and noise within shared representations." T…
tags:
  - "ICML 2026"
  - "Multi-task learning"
  - "Gradient conflict"
  - "Causal disentanglement"
  - "OOD generalization"
  - "Counterfactual augmentation"
date: 2026-05-08
content_hash: c1fcb3a87054cf63
---

# CORE-MTL: Rethinking Gradient Balancing via Causal Orthogonal Representations

**Conference**: ICML 2026  
**arXiv**: [2606.02221](https://arxiv.org/abs/2606.02221)  
**Code**: https://github.com/Hope-Rita/CORE-MTL  
**Area**: Optimization / Multi-Task Learning / Causal Representation Learning  
**Keywords**: Multi-task learning, Gradient conflict, Causal disentanglement, OOD generalization, Counterfactual augmentation

## TL;DR
The authors reattribute the root cause of "negative transfer" in multi-task learning from "gradient conflict" to the "entanglement of semantics and noise within shared representations." They propose CORE-MTL: a dual-stream encoder splits representations into semantic $\hat{Z}_s$ and residual $\hat{Z}_r$, implementing "causal orthogonality" via CKA independence constraints + counterfactual style substitution + inverse rendering reconstruction. Theoretically, it provides a tighter OOD upper bound than gradient balancing. Experimentally, it outperforms ten baselines including PCGrad, GradNorm, STCH, and FairGrad across ID settings (NYUv2/Cityscapes) and OOD settings (GTA5→Cityscapes, Cityscapes-C).

## Background & Motivation

**Background**: The mainstream approach to Multi-Task Learning (MTL) is divided into two factions: the optimization faction—adjusting task weights or projecting gradient directions during updates (GradNorm, PCGrad, MGDA, STCH, FairGrad); and the architecture faction—allocating different slices of the backbone to each task (MTAN). A common characteristic is treating the shared representation as a black box, intervening only at the gradient or routing level.

**Limitations of Prior Work**: Negative transfer remains widespread, and performance drops sharply under Out-of-Distribution (OOD) scenarios (distribution shifts, stylistic perturbations, synthetic-to-real). The authors point out a deeper issue: shared representations entangle "task-relevant invariant semantics" with "nuisance factors" like style, lighting, or background. Downstream heads take "shortcuts" by relying on these nuisances in the training distribution.

**Key Challenge**: Gradient conflict is often a symptom rather than the cause of negative transfer. When nuisance factors are already encoded into the shared representation, downstream predictions are forced to carry these spurious correlations regardless of how gradients are projected or reweighted—meaning the optimization faction cannot alter the "representation geometry."

**Goal**: To prove from a causal perspective that optimization-based methods possess an OOD error lower bound, and to design a "representation-centric" framework that structurally separates semantic and residual streams, ensuring task heads only access the semantic stream.

**Key Insight**: Assuming the input is generated from invariant semantic factors $Z_s$ and residual nuisance factors $Z_r$ via a mechanism $X=g(Z_s,Z_r)$, distribution shifts only change the covariance $\Sigma_r$ of $Z_r$, while the distribution of $Z_s$ remains constant. Under a linear-Gaussian SCM, if the encoder's learned representation mixes $Z_s$ and $Z_r$ via a rotation angle $\psi$, the OOD error has a lower bound of $c\sin^2(\psi)\|\Sigma_r^T-\Sigma_r^S\|_F$. This $\sin^2\psi$ term cannot be eliminated by gradient manipulation alone.

**Core Idea**: Instead of patching the gradient space, structurally split the representation into a semantic stream $\hat{Z}_s$ and a residual stream $\hat{Z}_r$, forcing heads to read only the semantic stream. Once decoupled, gradient orthogonality emerges automatically as a "geometric byproduct" without requiring post-hoc interventions like PCGrad.

## Method

### Overall Architecture
Input $x$ enters a shared encoder $\Phi_\theta$, and the output is explicitly split into two streams: $(\hat{Z}_s,\hat{Z}_r)=\Phi_\theta(x)$. The $K$ task heads $f_{\phi_t}$ read only the semantic stream $\hat{Z}_s$; the residual stream $\hat{Z}_r$ does not enter the heads. During training, three regularizations act on the representations: a CKA independence loss forces $\hat{Z}_s\perp\hat{Z}_r$; Counterfactual Augmentation (CFA) samples $\tilde{Z}_r$ from the empirical residual distribution and concatenates it with the original $\hat{Z}_s$ to form a "reskinned" input $\tilde{x}=\mathcal{D}(\hat{Z}_s,\tilde{Z}_r)$, which is passed through the encoder and heads to calculate a task loss, forcing heads to be invariant to style perturbations; a reconstruction loss feeds the pair $(\hat{Z}_s,\hat{Z}_r)$ to a decoder $\mathcal{D}$ to recover $x$, providing anchors for the "division of labor" between the two streams. During inference, only the encoder and task heads are used, discarding the decoder and counterfactual branches, resulting in zero additional overhead.

### Key Designs

1.  **Dual-Stream Encoding + Semantic-Only Heads**:
    *   **Function**: Architecturally ensures that task predictions cannot directly utilize residual signals.
    *   **Mechanism**: The encoder output is split along the channel dimension; $\hat{Z}_s$ goes to the heads, while $\hat{Z}_r$ only goes to the decoder and CFA. This topologically cuts the shortcut of "task loss sensitivity to residuals." Defining a leakage coefficient $\lambda_{\text{leak}}=\sup\|g_s(z_s,z_r)-g_s(z_s,z_r')\|/\|z_r-z_r'\|$ to measure semantic stream partial derivatives w.r.t residuals yields a tight OOD bound $\mathcal{E}_T(h)-\mathcal{E}_S(h)\leq C_{\text{cap}}+\alpha\lambda_{\text{leak}}W_1(P_S(Z_r),P_T(Z_r))$. As $\lambda_{\text{leak}}\to 0$, the OOD gap decouples from residual shift magnitude.
    *   **Design Motivation**: To replace the irreducible $\sin^2(\psi)$ lower bound of GradNorm/PCGrad methods, lifting robustness from "optimization dynamics" to "representation geometry."

2.  **CKA Independence Constraint**:
    *   **Function**: Pushes the two streams toward zero correlation at a statistical level, reinforcing the architectural split.
    *   **Mechanism**: Linear CKA on mini-batches is used as a regularization term $\mathcal{L}_{\text{CKA}}=\text{CKA}(\mathbf{Z}_s,\mathbf{Z}_r)$ to minimize linear dependence between feature matrices. Under linear-Gaussian assumptions, the authors prove that minimizing CKA is equivalent to reducing the cross-term of the encoder's Jacobian, acting as a differentiable proxy for $\lambda_{\text{leak}}$. A direct byproduct (Proposition 2.5) is $\mathbb{E}[\cos^2(g_{\text{task}},g_{\text{res}})]\leq c\cdot\text{CKA}(Z_s,Z_r)+\delta$, meaning task gradients and auxiliary residual gradients are nearly orthogonal at the final shared layer, resolving gradient conflict at the source.
    *   **Design Motivation**: Architectural splitting only ensures "information separability"; CKA forces "information separation." It transforms "gradient orthogonality"—usually forced via PCGrad projections—into a continuously differentiable representation regularizer.

3.  **Counterfactual Style Substitution + Reconstruction Grounding**:
    *   **Function**: Assigns explicit semantic roles to both streams and trains heads to be robust against style perturbations.
    *   **Mechanism**: CFA samples $\tilde{Z}_r$ from the empirical residual distribution of the current batch, concatenates it with the original $\hat{Z}_s$, and passes it through the decoder to synthesize a "same semantics, different style" image $\tilde{x}=\mathcal{D}(\hat{Z}_s,\tilde{Z}_r)$. This is fed back to the encoder, requiring heads to produce labels consistent with the original: $\mathcal{L}_{\text{CFA}}=\sum_t w_t\mathcal{L}_t(f_{\phi_t}([\Phi_\theta(\tilde{x})]_s),y_t)$. BN statistics are frozen during the counterfactual pass to prevent style leakage. Reconstruction grounding has two instantiations: **Hard Grounding** implements the decoder as physics-based inverse rendering $\hat{x}\approx\mathcal{A}(\hat{Z}_r)\odot\mathcal{S}(\mathcal{N}(\hat{Z}_s),\mathbf{L}(\hat{Z}_r))$, making $\hat{Z}_s$ responsible for geometry (normals) and $\hat{Z}_r$ for photometry (albedo + lighting); **Soft Grounding** uses a general convolutional decoder + $L_1$ reconstruction when physics priors are absent, relying on "heads-only-reading-$\hat{Z}_s$" and CKA to push discriminative info and reconstruction residuals into the correct streams.
    *   **Design Motivation**: Pure statistical independence can be satisfied by degenerate solutions (e.g., random channel splits); roles must be "anchored" to avoid character swapping. Physical priors are the strongest anchors, while soft grounding uses architectural bottlenecks as weak anchors.

### Loss & Training
The total objective is $\mathcal{L}_{\text{total}}=\sum_t w_t\mathcal{L}_t+\lambda_{\text{CKA}}\mathcal{L}_{\text{CKA}}+\lambda_{\text{CFA}}\mathcal{L}_{\text{CFA}}+\lambda_{\text{rec}}\mathcal{L}_{\text{rec}}$. Task weights can be fixed (equal weighting in experiments) or combined with GradNorm. $\mathcal{L}_{\text{rec}}$ switches between hard and soft grounding. The backbone is consistently ResNet-50, with frozen BN statistics in counterfactual paths for strict robustness evaluation.

## Key Experimental Results

### Main Results
In-Distribution results for NYUv2 (3 tasks) + Cityscapes (2 tasks) (selected metrics):

| Method | NYUv2 mIoU↑ | NYUv2 Depth Abs↓ | NYUv2 Normal Mean↓ | Cityscapes mIoU↑ | Cityscapes Depth Rel↓ |
|------|------------|------------------|--------------------|------------------|----------------------|
| Single Task | 0.5192 | 0.5260 | 24.27 | 0.6869 | 47.96 |
| Equal Weighting | 0.5316 | 0.3911 | 24.29 | 0.6962 | 44.03 |
| PCGrad | 0.5222 | 0.3916 | 24.39 | 0.6998 | 44.56 |
| GradNorm | 0.5264 | 0.3896 | 24.39 | 0.7015 | 44.55 |
| STCH | 0.5377 | 0.3917 | 23.20 | 0.6952 | 42.84 |
| MTAN | 0.5401 | 0.3822 | 24.02 | 0.7023 | 45.57 |
| FairGrad | 0.5291 | 0.3944 | 23.07 | 0.6986 | 43.74 |
| RepMTL | 0.5492 | 0.3727 | 24.53 | 0.7079 | 44.32 |
| **CORE-MTL** | **0.5693** | **0.3544** | **22.49** | **0.7229** | **19.61** |

OOD Results for GTA5→Cityscapes (Sim-to-Real) and Cityscapes-C: CORE-MTL achieved a target domain mIoU of 0.5435 (PCGrad 0.5047), Pixel Acc of 0.8401 (PCGrad 0.7943), and Depth Rel of 235.04 (PCGrad 301.61). On Cityscapes-C, metrics (mIoU 0.6104 / Pixel Acc 0.8670) significantly outperformed all baselines, validating the prediction in Theorem 2.4 that reducing leakage minimizes the OOD gap.

### Ablation Study
Ablation of components on NYUv2:

| Configuration | Seg mIoU↑ | Depth Abs↓ | Normal Mean↓ | Description |
|------|----------|-----------|--------------|------|
| Vanilla MTL | 0.5249 | 0.4418 | 25.62 | No dual-stream, standard sharing |
| + DS | 0.5352 | 0.3813 | 23.30 | Dual-stream w/o recon/reg; significant error drop |
| + DS + Grounding | 0.5424 | 0.3827 | 23.22 | + Reconstruction anchoring, seg improvement |
| + DS + Grounding + CKA | — | — | — | Independence constraint completes role division |
| Full (+ CFA) | 0.5693 | 0.3544 | 22.49 | Counterfactual provides most significant robustness gain |

Scalability on CelebA (Tasks K=10→40): CORE-MTL training time was near-constant (~300 s/epoch), whereas PCGrad increased linearly from 690 s to 2806 s. Average attribute accuracy was consistently highest.

### Key Findings
- **Gradient Orthogonality as a Structural Byproduct**: Visualization shows task and reconstruction gradient cosine similarity near 0 post-training, with a structured block pattern in the task-task gradient matrix, eliminating the need for PCGrad's projections.
- **Stability is an Order of Magnitude Higher**: Feature substitution experiments in cross-domain settings showed $\Delta Z_r/\Delta Z_s=3.28$, proving the semantic stream "withstood" style perturbations while the residual stream absorbed them.
- **PCGrad/FairGrad Regress in OOD**: Many gradient surgery methods showed larger $\Delta$ (source-target gap) than Equal Weighting, empirically confirming the lower bound in Theorem 2.3—gradient manipulation cannot solve OOD issues on entangled representations.
- **Colored-Cityscapes Shortcut Test**: CORE-MTL remained top-performing even when category colors were shuffled, proving it suppresses "residuals as shortcuts" rather than depending on perfect semantic-residual independence.

## Highlights & Insights
- **Reattributing "Gradient Conflict" to "Representation Geometry"**: The use of a clean linear-Gaussian SCM to derive an irreducible $\sin^2(\psi)$ lower bound theoretically "dooms" optimization-based MTL in OOD, while decoupled representations provide a tighter reachable upper bound.
- **CKA as a Differentiable Proxy for Leakage**: Bridging geometric ($\lambda_{\text{leak}}$) and statistical (CKA) measures makes the "representation independence" slogan a backpropagatable loss term, transferable to disentanglement, domain generalization, etc.
- **Counterfactual Style Substitution without External Data**: CFA samples from the in-batch empirical residual distribution to synthesize $\tilde{x}$, avoiding the need for external style libraries found in mixup or stylization methods.
- **Hard / Soft Grounding Duality**: High-prior scenarios (geometry) use inverse rendering, while low-prior scenarios (attributes) use general decoders, providing a standardized template for representation-centric methods across domains.

## Limitations & Future Work
- **Linear-Gaussian SCM Assumption**: The assumption of $Z_s\perp Z_r$ and fixed rotation encoders is strong; real-world semantics and context (e.g., pedestrians on crosswalks) are highly coupled.
- **Dependency on Good Decoders**: Hard grounding requires a physical forward model; finding equivalent "physical" priors in domains like medical imaging or time-series remains an open problem.
- **Non-negligible Training Overhead**: Dual-stream + CFA + reconstruction takes ~300 s/epoch (vs ~90 s for EW). While it doesn't scale linearly with $K$, it is ~3× the baseline; training requires more VRAM for decoders despite zero inference cost.
- **Narrow Task Range**: Experiments focused on dense vision and attribute classification; performance on NLP MTL (GLUE) or Multi-objective RL is unverified.

## Related Work & Insights
- **vs PCGrad / GradNorm / FairGrad**: These operate in the gradient space to project or reweight; CORE-MTL argues they face an irreducible OOD lower bound and instead obtains orthogonality as a free byproduct of decoupling.
- **vs MTAN (Architecture)**: MTAN uses attention paths to slice the backbone per task; CORE-MTL slices by "semantics vs. residuals," aligning more with causal disentanglement and allowing better cross-task sharing.
- **vs RepMTL (Representation-centric)**: RepMTL focuses on statistical alignment without explicit causal structure; CORE-MTL introduces SCM interpretation, counterfactuals, and physical grounding, making it more theoretically complete.
- **vs IRM / DANN (OOD)**: IRM pursues invariant representations across environments; CORE-MTL grafts "invariance" onto the internal head-stream relationship within MTL and works without multi-environment labels.

## Rating
- Novelty: ⭐⭐⭐⭐ Elevates negative transfer from optimization to representation level; theoretical framework + implementation is cohesive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers ID, OOD, scalability, and ablation against ten baselines with thorough visualization.
- Writing Quality: ⭐⭐⭐⭐ Clear logical skeleton (Theorems 2.3 & 2.4); physical grounding details are slightly brief in the main text.
- Value: ⭐⭐⭐⭐ Provides a new perspective for the MTL community; the lower bound conclusion serves as a warning for future "gradient-only" methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Addressing Divergent Representations from Causal Interventions on Neural Networks](../../ICLR2026/others/addressing_divergent_representations_causal.md)
- [\[ICML 2026\] Continual Learning of Domain-Invariant Representations](continual_learning_of_domain-invariant_representations.md)
- [\[ICML 2026\] Rethinking FID Through the Geometry of the Reference Dataset](rethinking_fid_through_the_geometry_of_the_reference_dataset.md)
- [\[ICML 2026\] Rethinking Evaluation Paradigms in IBP-based Certified Training](rethinking_evaluation_paradigms_in_ibp-based_certified_training.md)
- [\[CVPR 2026\] Rethinking SNN Online Training and Deployment: Gradient-Coherent Learning via Hybrid-Driven LIF Model](../../CVPR2026/others/rethinking_snn_online_training_and_deployment_grad.md)

</div>

<!-- RELATED:END -->
