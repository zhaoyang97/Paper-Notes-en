---
title: >-
  [Paper Note] The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence
description: >-
  [ICML 2026][Social Computing][InfoNCE] This paper employs a measure-theoretic framework to elevate the InfoNCE loss to a deterministic "population energy" over representation distributions. It proves that the unimodal ca…
tags:
  - "ICML 2026"
  - "Social Computing"
  - "InfoNCE"
  - "CLIP"
  - "Modality Gap"
  - "population energy"
  - "Gibbs equilibrium"
date: 2026-05-08
content_hash: 3344fbdceff51d35
---

# The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-modal Divergence

**Conference**: ICML 2026  
**arXiv**: [2601.19597](https://arxiv.org/abs/2601.19597)  
**Code**: None  
**Area**: Representation Learning Theory / Contrastive Learning / Multimodal  
**Keywords**: InfoNCE, CLIP, Modality Gap, population energy, Gibbs equilibrium

## TL;DR
This paper employs a measure-theoretic framework to elevate the InfoNCE loss to a deterministic "population energy" over representation distributions. It proves that the unimodal case is convex and converges to a unique Gibbs equilibrium, while the symmetric multimodal case exhibits persistent negative symmetric KL coupling, which geometrically necessitates a modality gap.

## Background & Motivation

**Background**: InfoNCE serves as the unified objective for current self-supervised and multimodal contrastive learning, underpining systems from SimCLR/MoCo to CLIP/SigLIP. The most classic theoretical analyses include the alignment-uniformity decomposition by Wang & Isola (2020) and the density-ratio perspective describing the optimal critic as pointwise mutual information.

**Limitations of Prior Work**: (i) The alignment-uniformity explanation only addresses asymptotic trade-offs without specifying which "population distribution" InfoNCE favors; (ii) Multimodal InfoNCE in systems like CLIP achieves strong pairwise alignment, yet the marginal distributions of the two modalities remain separated (modality gap), a phenomenon for which existing theories lack a mechanistic explanation; (iii) Existing identifiability results focus on "what can be learned" under generative assumptions rather than characterizing the geometric preferences of the training objective itself.

**Key Challenge**: Treating InfoNCE solely as a "pair-wise discrimination" task loses critical information—the softmax denominator is essentially a kernel average of the current representation distribution, meaning the optimization direction depends fundamentally on the distribution rather than just individual pairs. In the multimodal setting, this "force exerted by the distribution on itself" couples with the "force exerted on the other modality," such that strong pairwise alignment cannot necessarily control the marginals.

**Goal**: (i) Formulate stochastic InfoNCE strictly as a deterministic functional of representation distributions; (ii) Explain the geometry of the unimodal case (convexity, Gibbs equilibrium, low-temperature concentration); (iii) Derive the "cross-coupling" structure of symmetric multimodal InfoNCE that differs from the unimodal case to provide a first-principles explanation for the modality gap.

**Key Insight**: The representation space $\mathcal{Z}$ is viewed as a compact manifold with a volume measure $\mu$. The encoder pushes forward the data distribution onto $\mathcal{Z}$. In the large-batch limit, the softmax denominator converges to a "population partition field" $\Gamma_{\theta,\tau}(\mathbf{z})$, which is a distribution-dependent energy field.

**Core Idea**: In the large-batch limit, InfoNCE is equivalent to a population energy functional of the representation distribution. In the unimodal case, this functional is strictly convex with a unique Gibbs solution (where entropy acts as a "dispersion selector" within the alignment basin). In the multimodal case, the functional contains a negative symmetric KL coupling term, causing the two modalities to act as "walls" to each other while sharpening their respective potentials, thus stably maintaining the modality gap.

## Method

### Overall Architecture
The analysis pipeline consists of: (i) Defining representation laws $q_\theta=(f_\theta)_\# p_x$ and positive-pair laws $\pi_{\theta\theta}$ on a compact $\mathcal{Z}$; (ii) Introducing the partition field $\Gamma_{\theta,\tau}(\mathbf{z})=\int_\mathcal{Z}\kappa_\tau(\mathbf{z},\mathbf{w})\mathrm{d}q_\theta(\mathbf{w})$ and the kernel-smoothed density $\tilde\rho_{\theta,\tau}=\Gamma_{\theta,\tau}/V_\kappa(\tau)$; (iii) Proving that stochastic InfoNCE is value- and gradient-consistent with a parametric energy $\mathcal{J}_\tau(\theta)$ as $N\to\infty$; (iv) Elevating $\mathcal{J}_\tau$ to an "intrinsic free energy" $\mathcal{F}_{\tau,U}$ and analyzing its convexity, minimizers, and low-temperature concentration; (v) Repeating the process for symmetric multimodal InfoNCE to obtain $\mathcal{F}_{\tau,\mathbf{U}_{1,2}}^{\text{Sym}}$ containing the negative symmetric KL coupling and analyzing its geometric differences from the unimodal case.

### Key Designs

1.  **Large-batch consistency from stochastic loss to deterministic energy**:
    -   **Function**: Establishes a strict equivalence between InfoNCE and population energy—at both the value and gradient levels—providing a mathematical foundation for geometric analysis rather than relying on intuition.
    -   **Mechanism**: For the unimodal case, define $\mathcal{J}_\tau(\theta)=\frac{1}{\tau}\int_\mathcal{Z}U_\theta(\mathbf{z})\mathrm{d}q_\theta(\mathbf{z})-H_\times(q_\theta,\tilde\rho_{\theta,\tau})$, where the alignment potential field $U_\theta(\mathbf{z})=-\int_\mathcal{Z}s(\mathbf{z},\mathbf{w})\mathrm{d}\nu_{\theta,\mathbf{z}}(\mathbf{w})$ arises from the disintegration of positive pairs. Theorem 3.1 proves that under consistent regularization of encoder and critic, constant kernel volume, and controlled finite batch size, $|\mathcal{L}_{\text{NCE}}(\theta)-\mathcal{J}_\tau(\theta)-\log(NV_\kappa(\tau))|\to0$ and $\|\nabla_\theta\mathcal{L}_{\text{NCE}}-\nabla_\theta\mathcal{J}_\tau\|\to0$.
    -   **Design Motivation**: Previous decompositions of InfoNCE into alignment-uniformity were manual approximations. This work insists on "consistency in both value and gradient," ensuring that stochastic gradient descent in the large-batch limit is strictly equivalent to the descent of population energy.

2.  **Intrinsic free energy + Gibbs equilibrium**:
    -   **Function**: Elevates the parametric energy to the distribution space, stripping away the implicit relationship of parameterization to obtain a strictly convex free energy with a unique minimizer and proving its consistency with parametric energy at low temperatures.
    -   **Mechanism**: Define $\mathcal{F}_{\tau,U}(\rho)=\frac{1}{\tau}\int_\mathcal{Z}U(\mathbf{z})\rho(\mathbf{z})\mathrm{d}\mu(\mathbf{z})-H(\rho)$, and prove that it is strictly convex on $\mathcal{P}_\mu(\mathcal{Z})$ with a unique minimizer in Gibbs form $\rho^*(\mathbf{z})=\exp(-U(\mathbf{z})/\tau)/Z_\tau$. Using a sharp diagonal peak assumption, it is shown that $|\mathcal{J}_\tau(\theta)-\mathcal{F}_{\tau,U_\theta}(\rho_\theta)|\leq 2\varepsilon_{\text{kde}}^{(\theta)}(\tau)/\underline\rho_\theta$. A low-temperature concentration proposition proves that as $\tau\to0^+$, the Gibbs equilibrium concentrates in the near-minimal regions of $U$.
    -   **Design Motivation**: Reinterprets "uniformity" as entropy-driven dispersion within alignment basins, rather than a global force opposing alignment. This step represents a conceptual upgrade to the Wang & Isola perspective: alignment determines "which basin" to converge into, while uniformity determines the dispersion "within the basin."

3.  **Negative symmetric KL coupling in multimodal and the necessity of modality gap**:
    -   **Function**: Elevates symmetric InfoNCE to a free energy with cross-coupling, proving that the coupling term is a negative symmetric KL, which implies that the density fields of the two modalities act as "potential barriers" to each other, resulting in persistent marginal separation.
    -   **Mechanism**: Define $\mathcal{J}_\tau^{\text{Sym}}(\theta,\phi)=\frac{1}{2}(\mathcal{J}_\tau^{x\to y}+\mathcal{J}_\tau^{y\to x})$, where each directional energy evaluates its cross-entropy against the smoothed density of the other modality. Elevating this to distribution space yields $\mathcal{F}_{\tau,\mathbf{U}_{1,2}}^{\text{Sym}}(\rho_1,\rho_2)=\frac{1}{2}(\mathcal{F}_{\tau,U_{1\to2}}(\rho_1)+\mathcal{F}_{\tau,U_{2\to1}}(\rho_2))-D_{\text{KL}}^{\text{Sym}}(\rho_1,\rho_2)$, where the negative sign is critical. Thus, while optimizing, each modality aims to sharpen alignment with its own potential while pulling away from the other modality's marginal KL (acting as the other's barrier). At steady state, a "knife-edge" compatibility condition between marginals must be met; otherwise, a modality gap is inevitable.
    -   **Design Motivation**: The modality gap is not an optimization failure but a geometric necessity of InfoNCE under heterogeneous conditionals. Once established, this suggests an inherent limit to efforts to eliminate the gap using better hard negatives or larger batches.

### Loss & Training
As this is a theoretical paper, no new models were trained. The experimental section uses controlled experiments with synthetic two-modality Gaussian mixtures to visualize the modality gap and consistency results. It also evaluates pre-trained OpenCLIP (CNN and ViT backbones) to measure distances between marginals and test the prediction that "disrupting cross-modal compatibility systematically increases the gap."

## Key Experimental Results

### Main Results

| Experiment | Setting | Key Observation |
| :--- | :--- | :--- |
| Unimodal Low-temp Concentration | Synthetic data + various $\tau$ | As $\tau\to0^+$, Gibbs measure mass tends to 1 in low-potential regions, matching theory. |
| Multimodal Marginal Separation | Synthetic heterogeneous modalities | Even with perfect pairwise alignment, marginals remain separated; gap increases with compatibility mismatch. |
| OpenCLIP Marginal Gap | CNN / ViT backbones | Strong retrieval performance coexists with significant modality gaps; weakening cross-modal compatibility increases gaps. |

### Ablation Study

| Configuration | Phenomenon | Explanation |
| :--- | :--- | :--- |
| Sharp kernel + Low-temp | $\mathcal{J}_\tau\approx\mathcal{F}_{\tau,U_\theta}$ | Validates that KDE bias is controllable in the sharp regime. |
| Unidirectional vs Sym. InfoNCE | Symmetric adds negative KL coupling | The unidirectional case has no necessity for a modality gap, unlike the symmetric version. |
| Compatibility Perturbation | Gap increases with perturbation | Validates the prediction that "compatibility determines the gap." |

### Key Findings
- "Uniformity" should be understood as entropy-driven dispersion within alignment basins rather than a force globally antagonistic to alignment—this directly refines a long-standing interpretation.
- The essence of the multimodal modality gap is negative symmetric KL coupling: both modalities are forced to push away each other's marginals while minimizing their own potentials; stronger pairwise alignment can paradoxically solidify the gap.
- Exact marginal matching requires a "knife-edge" compatibility condition (identical conditional laws), which is rarely satisfied by real-world data, making the gap a generic phenomenon rather than an accidental failure.

## Highlights & Insights
- Shifting contrastive learning analysis from "pointwise discrimination" to "population geometry" is a substantial perspective upgrade; all conclusions (Gibbs equilibrium, negative KL coupling) are measure-theoretically provable rather than intuitive.
- The use of a negative sign (negative symmetric KL) provides a definitive explanation for the modality gap, reframing "why we cannot eliminate the gap" as "it is an inherent property of the InfoNCE loss." For practitioners, this suggests that adding constraints to the marginals may be more effective than designing sophisticated hard negatives.
- The two-layer analysis (intrinsic vs parametric) plus KDE error control provides a clean paradigm: prove geometry in the distribution space first, then use sharp kernels to translate conclusions back to the parameter space.

## Limitations & Future Work
- All proofs rely on assumptions such as compact manifolds, isotropic kernels, and sharp diagonal peaks, which strictly hold only when the temperature is minimal and kernels are extremely sharp; real-world CLIP temperatures are not necessarily minimal.
- Experiments are validated only on synthetic data and off-the-shelf OpenCLIP; no from-scratch training of a model with a modified loss to systematically reduce the gap was demonstrated.
- Multimodal analysis is limited to the symmetric two-modality case; the paper does not address whether the coupling structure of three or more modalities is still dominated by pairwise negative KL terms.
- The geometry is not directly linked to downstream metrics such as zero-shot performance or retrieval rank.

## Related Work & Insights
- **vs Wang & Isola 2020 (alignment-uniformity)**: This work proves uniformity is entropy-driven dispersion within basins rather than a global force, refining rather than overturning the original view.
- **vs Liang et al. 2022 (modality gap)**: They first empirically observed the gap and attributed it to the cone effect and initialization; this paper provides a first-principles geometric explanation.
- **vs Identifiability works (Zimmermann 2021)**: Those works address "what can be learned," while this paper addresses "what geometry the objective prefers," offering complementary perspectives.
- **vs Brumley / Park 2024 (LRH)**: CRH and this work both advance along the representation geometry line; while CRH focuses on steering, this paper focuses on the population energy of the training objective.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The conclusions regarding "InfoNCE = population energy" and "modality gap = negative symmetric KL" are novel in theoretical literature and represent a structural upgrade to the community's perspective.
- **Experimental Thoroughness**: ⭐⭐⭐ Experiments serve the theory well but do not systematically validate whether the theory can guide large-scale model design.
- **Writing Quality**: ⭐⭐⭐⭐ Measure-theoretic notation is rigorous, and the unimodal-multimodal dual presentation is well-structured, though the technical barrier is high.
- **Value**: ⭐⭐⭐⭐⭐ Highly significant for future research in contrastive learning and multimodal alignment, explicitly informing the community that the gap cannot be eliminated solely through pairwise tuning.

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
