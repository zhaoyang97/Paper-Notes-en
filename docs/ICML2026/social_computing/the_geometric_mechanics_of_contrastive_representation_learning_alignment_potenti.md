---
title: >-
  [Paper Note] The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence
description: >-
  [ICML 2026][Social Computing][InfoNCE] This paper elevates the InfoNCE loss to a deterministic "population energy" over representation distributions using a measure-theoretic framework…
tags:
  - "ICML 2026"
  - "Social Computing"
  - "InfoNCE"
  - "CLIP"
  - "Modality Gap"
  - "population energy"
  - "Gibbs equilibrium"
date: 2026-05-08
content_hash: a27f36a99d3d4d31
---

# The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence

**Conference**: ICML 2026  
**arXiv**: [2601.19597](https://arxiv.org/abs/2601.19597)  
**Code**: None  
**Area**: Representation Learning Theory / Contrastive Learning / Multimodal  
**Keywords**: InfoNCE, CLIP, Modality Gap, population energy, Gibbs equilibrium

## TL;DR
This paper elevates the InfoNCE loss to a deterministic "population energy" over representation distributions using a measure-theoretic framework, proving that the unimodal case is convex and converges to a unique Gibbs equilibrium, while the symmetric multimodal case exhibits persistent negative symmetric KL coupling, which geometrically and inevitably induces a modality gap.

## Background & Motivation

**Background**: InfoNCE is the unified objective for current self-supervised and multimodal contrastive learning, forming the basis of systems from SimCLR/MoCo to CLIP/SigLIP. The most classic theoretical analysis is the alignment-uniformity decomposition by Wang & Isola (2020), and the density-ratio perspective that describes the optimal critic as pointwise mutual information.

**Limitations of Prior Work**: (i) The alignment-uniformity explanation only addresses the asymptotic trade-off, but does not clarify what "population distribution" InfoNCE itself prefers; (ii) In multimodal InfoNCE, as in CLIP, strong pairwise alignment is achieved, yet the marginal distributions of the two modalities remain separated (modality gap), for which existing theory lacks a mechanistic explanation; (iii) Existing identifiability results focus only on "what can be learned" under generative assumptions, without characterizing the geometric preference of the training objective itself.

**Key Challenge**: Viewing InfoNCE solely as "pairwise discrimination" loses crucial information—the softmax denominator is actually a kernel average over the current representation distribution, so the optimization direction fundamentally depends on the distribution, not just the pairs. In the multimodal case, the "force exerted by the distribution on itself" couples with the "force exerted on the other modality," so even strong pairwise alignment cannot control the marginals.

**Goal**: (i) Formulate stochastic InfoNCE strictly as a deterministic functional over representation distributions; (ii) Explain the geometry of the unimodal case (convexity, Gibbs equilibrium, low-temperature concentration); (iii) Derive the "cross-coupling" structure of symmetric multimodal InfoNCE, distinct from the unimodal case, and provide a first-principles explanation for the modality gap.

**Key Insight**: Treat the representation space $\mathcal{Z}$ as a compact manifold with volume measure $\mu$, with the encoder push-forwarding the data distribution onto $\mathcal{Z}$. The softmax denominator converges in the large-batch limit to the "population partition field" $\Gamma_{\theta,\tau}(\mathbf{z})$, a distribution-dependent energy field.

**Core Idea**: In the large batch limit, InfoNCE is equivalent to a population energy functional over the representation distribution; for the unimodal case, this functional is strictly convex and has a unique Gibbs solution (i.e., entropy acts as a "dispersion selector" within the alignment basin), while in the multimodal case, the functional contains a negative symmetric KL coupling term, causing the two modalities to act as "walls" for each other when sharpening their respective potentials, thus stably maintaining the modality gap.

## Method

### Overall Architecture
Analysis workflow: (i) Define representation laws $q_\theta=(f_\theta)_\# p_x$ and positive-pair laws $\pi_{\theta\theta}$ on compact $\mathcal{Z}$; (ii) Introduce the partition field $\Gamma_{\theta,\tau}(\mathbf{z})=\int_\mathcal{Z}\kappa_\tau(\mathbf{z},\mathbf{w})\mathrm{d}q_\theta(\mathbf{w})$ and kernel-smoothed density $\tilde\rho_{\theta,\tau}=\Gamma_{\theta,\tau}/V_\kappa(\tau)$; (iii) Prove that stochastic InfoNCE, as $N\to\infty$, is value- and gradient-consistent with a parametric energy $\mathcal{J}_\tau(\theta)$; (iv) Elevate $\mathcal{J}_\tau$ to an "intrinsic free energy" $\mathcal{F}_{\tau,U}$, analyzing its convexity, minimizer, and low-temperature concentration; (v) Repeat the same process for symmetric multimodal InfoNCE, obtaining $\mathcal{F}_{\tau,\mathbf{U}_{1,2}}^{\text{Sym}}$ with negative symmetric KL coupling, and analyze its geometric differences from the unimodal case.

### Key Designs

1. **Large-batch Consistency from Stochastic Loss to Deterministic Energy**:

    - **Function**: Establishes a strict equivalence between InfoNCE and population energy—both at the value and gradient levels—providing mathematical justification for all geometric analyses, rather than relying on intuition.
    - **Mechanism**: For the unimodal case, defines $\mathcal{J}_\tau(\theta)=\frac{1}{\tau}\int_\mathcal{Z}U_\theta(\mathbf{z})\mathrm{d}q_\theta(\mathbf{z})-H_\times(q_\theta,\tilde\rho_{\theta,\tau})$, where the alignment potential field $U_\theta(\mathbf{z})=-\int_\mathcal{Z}s(\mathbf{z},\mathbf{w})\mathrm{d}\nu_{\theta,\mathbf{z}}(\mathbf{w})$ comes from the disintegration of positive pairs. Theorem 3.1 proves, under consistent regularization of encoder and critic, kernel volume constant, and finite batch control, that $|\mathcal{L}_{\text{NCE}}(\theta)-\mathcal{J}_\tau(\theta)-\log(NV_\kappa(\tau))|\to0$ and $\|\nabla_\theta\mathcal{L}_{\text{NCE}}-\nabla_\theta\mathcal{J}_\tau\|\to0$.
    - **Design Motivation**: Previous alignment-uniformity decompositions were manual approximations; this work insists on "value + gradient consistency," ensuring that stochastic gradient descent with large batches is strictly equivalent to population energy descent, so all subsequent convexity and equilibrium analyses directly correspond to the actual training process.

2. **Intrinsic Free Energy + Gibbs Equilibrium**:

    - **Function**: Elevates the parametric energy to the distribution space, stripping away implicit parameterization to obtain a strictly convex free energy with a unique minimizer, and proves under the sharp kernel assumption that it matches the parametric energy at low temperature.
    - **Mechanism**: Defines $\mathcal{F}_{\tau,U}(\rho)=\frac{1}{\tau}\int_\mathcal{Z}U(\mathbf{z})\rho(\mathbf{z})\mathrm{d}\mu(\mathbf{z})-H(\rho)$, proves strict convexity on $\mathcal{P}_\mu(\mathcal{Z})$, and that the unique minimizer is the Gibbs form $\rho^*(\mathbf{z})=\exp(-U(\mathbf{z})/\tau)/Z_\tau$. Under the sharp diagonal peak assumption, shows $|\mathcal{J}_\tau(\theta)-\mathcal{F}_{\tau,U_\theta}(\rho_\theta)|\leq 2\varepsilon_{\text{kde}}^{(\theta)}(\tau)/\underline\rho_\theta$, and finally, via a low-temperature concentration proposition, proves that as $\tau\to0^+$, the Gibbs equilibrium concentrates on the near-minimum region of $U$.
    - **Design Motivation**: Reinterprets "uniformity" as entropy-driven dispersion within the alignment basin, rather than a global force opposing alignment. This is a conceptual upgrade over the Wang & Isola perspective: alignment determines "which basin" convergence occurs in, while uniformity determines the degree of dispersion "within the basin."

3. **Multimodal Negative Symmetric KL Coupling and the Inevitability of the Modality Gap**:

    - **Function**: Elevates symmetric InfoNCE to a free energy with cross-coupling, proving that the coupling term is a negative symmetric KL, meaning each modality treats the other's density field as a "barrier," leading to persistent marginal separation.
    - **Mechanism**: Defines $\mathcal{J}_\tau^{\text{Sym}}(\theta,\phi)=\frac{1}{2}(\mathcal{J}_\tau^{x\to y}+\mathcal{J}_\tau^{y\to x})$, where each directional energy evaluates its own cross-entropy on the other modality's smoothed density. Lifting to the distribution space yields $\mathcal{F}_{\tau,\mathbf{U}_{1,2}}^{\text{Sym}}(\rho_1,\rho_2)=\frac{1}{2}(\mathcal{F}_{\tau,U_{1\to2}}(\rho_1)+\mathcal{F}_{\tau,U_{2\to1}}(\rho_2))-D_{\text{KL}}^{\text{Sym}}(\rho_1,\rho_2)$, with the crucial negative sign. Thus, each modality, when optimizing itself, seeks both to sharpen alignment to its own potential and to increase the KL divergence from the other modality's marginal (i.e., "mutually acting as barriers"), so at equilibrium, a knife-edge compatibility condition is required for marginal matching; otherwise, the modality gap is inevitable.
    - **Design Motivation**: The modality gap is not an optimization failure, but a geometric inevitability of InfoNCE under heterogeneous conditionals—once this conclusion is established, all community efforts to eliminate the gap via better hard negatives or larger batches face a fundamental limit.

### Loss & Training
This is a theoretical paper; no new models are trained. The experimental section uses synthetic bimodal Gaussian mixtures for controlled experiments to visualize the modality gap and consistency conclusions, and measures marginal distances on pretrained OpenCLIP (CNN + ViT backbone) to test the prediction that "disrupting cross-modal compatibility systematically increases the gap."

## Key Experimental Results

### Main Results

| Experiment | Setting | Key Observation |
|------------|---------|-----------------|
| Unimodal Low-Temperature Concentration | Synthetic data + various $\tau$ | As $\tau\to0^+$, the Gibbs measure concentrates mass in low-potential regions, matching theory |
| Multimodal Marginal Separation | Synthetic heterogeneous modalities | Even with perfect pairwise alignment, the two modality marginals remain separated; the gap increases monotonically with compatibility mismatch |
| OpenCLIP marginal gap | CNN / ViT backbone | Strong retrieval performance coexists with a significant modality gap; weakening cross-modal compatibility systematically enlarges the gap |

### Ablation Study

| Configuration | Phenomenon | Note |
|---------------|------------|------|
| Sharp kernel + low temperature | $\mathcal{J}_\tau\approx\mathcal{F}_{\tau,U_\theta}$ | Verifies that KDE bias is controllable in the sharp regime |
| Unidirectional vs. Symmetric InfoNCE | Symmetric adds negative KL coupling | No inevitable modality gap in the unidirectional case; only the symmetric version exhibits this |
| Compatibility perturbation | Gap increases with perturbation | Verifies the prediction that "compatibility determines the gap" |

### Key Findings
- "Uniformity" should be understood as entropy-driven dispersion within the alignment basin, not as a global force opposing alignment—this directly corrects a long-standing explanation.
- The essence of the multimodal modality gap is negative symmetric KL coupling: as both modalities minimize their own potentials, they are forced to push their marginals apart; stronger pairwise alignment actually entrenches the gap.
- Exact marginal matching requires a knife-edge compatibility condition (the two modality conditional laws must be identical), which is rarely met in real data, so the gap is a generic phenomenon, not an accidental failure.

## Highlights & Insights
- Upgrading contrastive learning analysis from "pointwise discrimination" to "population geometry" is a true perspective shift; all conclusions (Gibbs equilibrium, negative KL coupling) are now measure-theoretically provable, not just intuitive.
- The negative symmetric KL term definitively explains the modality gap—it reframes "why the gap persists" as "this is how InfoNCE loss is structured." For practitioners, rather than designing more sophisticated hard negatives, it is more effective to directly constrain the marginals.
- The two-layer analysis (intrinsic vs. parametric) plus KDE error control is a clean paradigm: first prove geometry in distribution space, then use sharp kernels to transfer conclusions back to parameter space; this can be generalized to any contrastive objective with a softmax denominator.

## Limitations & Future Work
- All proofs rely on assumptions such as compact manifolds, isotropic kernels, and sharp diagonal peaks, and are strictly valid only at very low temperatures and extremely sharp kernels; the temperatures used in practical CLIP training are not that low.
- Experiments are limited to synthetic data and existing OpenCLIP models; no new models were trained from scratch using this theory to demonstrate that the modality gap can be systematically reduced.
- The multimodal analysis is limited to symmetric bimodal cases; whether the coupling structure for three or more modalities is still dominated by "pairwise KL negatives" is not addressed.
- The geometric framework is not directly linked to downstream metrics such as zero-shot performance or retrieval rank, so there remains a gap between theory and practical outcomes.

## Related Work & Insights
- **vs Wang & Isola 2020 (alignment-uniformity)**: This work shows that uniformity is not a global force but an entropy selector within the basin, refining rather than overturning the original perspective.
- **vs Liang et al. 2022 (modality gap)**: They first empirically demonstrated the modality gap and attributed it to the cone effect and initialization; this work provides a first-principles geometric explanation.
- **vs Identifiability works (Zimmermann 2021)**: Those works answer "what can be learned," while this work answers "what geometry the objective prefers," offering complementary perspectives.
- **vs Brumley / Park 2024 (LRH)**: Both CRH and this work advance the study of representation geometry; CRH focuses on steering, while this work focuses on the population energy of the training objective, sharing a common conceptual lineage.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The conclusions "InfoNCE = population energy" and "modality gap = negative symmetric KL" are both theoretical firsts and represent a structural upgrade for the community's perspective.
- Experimental Thoroughness: ⭐⭐⭐ Experiments serve the theory and are concise, without systematic validation on large-scale training.
- Writing Quality: ⭐⭐⭐⭐ Measure-theoretic notation is rigorous, and the authors thoughtfully present the unimodal-multimodal duality, though the entry barrier is high.
- Value: ⭐⭐⭐⭐⭐ Provides directional significance for future research in contrastive learning and multimodal alignment, clearly stating that "pairwise tuning alone cannot eliminate the gap."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Cross-modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition](../../AAAI2026/social_computing/cross-modal_prompting_for_balanced_incomplete_multi-modal_emotion_recognition.md)
- [\[ICML 2026\] Alignment Tampering: How Reinforcement Learning from Human Feedback Is Exploited to Optimize Misaligned Biases](alignment_tampering_how_reinforcement_learning_from_human_feedback_is_exploited_.md)
- [\[AAAI 2026\] Multi-modal Dynamic Proxy Learning for Personalized Multiple Clustering](../../AAAI2026/social_computing/multi-modal_dynamic_proxy_learning_for_personalized_multiple_clustering.md)
- [\[ICCV 2025\] Gradient Extrapolation for Debiased Representation Learning](../../ICCV2025/social_computing/gradient_extrapolation_for_debiased_representation_learning.md)
- [\[ICML 2026\] MIND: Multi-Rationale Integrated Discriminative Reasoning Framework for Multi-Modal Fake News](mind_multi-rationale_integrated_discriminative_reasoning_framework_for_multi-mod.md)

</div>

<!-- RELATED:END -->
