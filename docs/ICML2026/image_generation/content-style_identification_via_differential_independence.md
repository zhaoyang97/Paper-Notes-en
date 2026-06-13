---
title: >-
  [Paper Note] Content-Style Identification via Differential Independence
description: >-
  [ICML 2026][Image Generation][content-style disentanglement] This paper proposes CSDI (content-style differential independence) as a novel identifiability condition. It proves that unpaired multi-domain content-style blo…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "content-style disentanglement"
  - "differential independence"
  - "Jacobian orthogonality"
  - "GAN"
  - "identifiability"
date: 2026-05-08
content_hash: 1287878ce691fcb0
---

# Content-Style Identification via Differential Independence

**Conference**: ICML 2026  
**arXiv**: [2605.17827](https://arxiv.org/abs/2605.17827)  
**Code**: https://github.com/subashtimilsina/CSDI (Available)  
**Area**: Image Generation / Content-Style Disentanglement / Identifiability  
**Keywords**: content-style disentanglement, differential independence, Jacobian orthogonality, GAN, identifiability

## TL;DR
This paper proposes CSDI (content-style differential independence) as a novel identifiability condition. It proves that unpaired multi-domain content-style blocks are identifiable under settings where content and style are statistically **correlated** and the Jacobians are **dense**, provided the Jacobian column spaces of the generator with respect to content and style are mutually orthogonal on the data manifold. By employing Hutchinson noise probing, this condition is implemented as a scalable regularization term $\mathcal{L}_{\rm orth}$ for StyleGAN2-ADA. On AFHQ and CelebA-HQ, this method reduces FID from 5.2 / 4.6 to 4.4 / 4.3 and improves LPIPS from 0.40 / 0.26 to 0.45 / 0.34 in counterfactual generation and cross-domain translation tasks.

## Background & Motivation
**Background**: Learning the latent variable decomposition $\bm{x}^{(n)} = \bm{g}(\bm{c}, \bm{s}^{(n)})$—representing "shared content + domain-specific style"—from unpaired multi-domain data is the common framework for image translation, counterfactual generation, and domain adaptation. In the absence of paired samples, additional structural assumptions must be introduced to ensure that the learned $\widehat{\bm{c}}, \widehat{\bm{s}}^{(n)}$ are consistent with the ground truth $\bm{c}, \bm{s}^{(n)}$, rather than merely achieving distribution matching at the cost of semantic consistency.

**Limitations of Prior Work**: Existing identifiability conditions are primarily categorized into two restrictive types: (i) **Statistical Independence** (Xie 2023 / Kong 2022 / Shrestha & Fu 2025), which requires content and style to be block-independent at the probability level. However, in reality, style often depends on content (e.g., lighting depends on object geometry, or cellular states influence gene variation). (ii) **Jacobian Sparsity** (Yan 2023), which requires each style dimension to affect only a small, non-overlapping set of data features, failing in dense influence scenarios like single-cell analysis.

**Key Challenge**: The "information separation" required for identifiability is typically imposed at the **distributional level** or the **support level**, both of which are overly strong. Statistical correlation does not imply information entanglement, and dense influence does not imply non-separability.

**Goal**: To identify a set of conditions that neither require $\bm{c} \perp\!\!\!\perp \bm{s}^{(n)}$ nor require the Jacobian $\bm{J}\bm{g}$ to be sparse, and that can scale to high-resolution image generation.

**Key Insight**: From a differential geometry perspective, "content and style being unrelated" can be localized. As long as perturbations of the generator with respect to $\bm{c}$ and $\bm{s}^{(n)}$ result in **orthogonal directions** on the data manifold $\mathcal{X}^{(n)}$, disentanglement is possible even if they are statistically correlated or have dense Jacobians. This intuition aligns with IMA (Gresele 2021), StyleGAN2 path-length regularization, and Hessian penalty, but previous works lacked rigorous proofs for content-style block-level identifiability.

**Core Idea**: Use "tangent space orthogonality" (differential independence) instead of "distributional independence / sparse support" as the structural condition for content-style identifiability. This is implemented via Hutchinson vector-Jacobian products (VJP) as a differentiable regularizer $\mathcal{L}_{\rm orth}$ with $\mathcal{O}(K)$ rather than $\mathcal{O}(d)$ backward complexity.

## Method

### Overall Architecture
CSDI-GAN adopts a dual-branch generative structure common in multi-domain GANs: two learnable latent space mappings $\bm{e}_C, \bm{e}_S^{(n)}$ encode Gaussian noise $\bm{r}_C, \bm{r}_S^{(n)}$ into $\widehat{\bm{c}}, \widehat{\bm{s}}^{(n)}$, which are then fed into a shared generator $\widehat{\bm{g}}$ to obtain $\widehat{\bm{x}}^{(n)} = \widehat{\bm{g}}(\widehat{\bm{c}}, \widehat{\bm{s}}^{(n)})$. Finally, domain-specific discriminators $\widehat{\bm{d}}^{(n)}$ perform distribution matching with real $\bm{x}^{(n)}$. The key differences from B.I. GAN / I-StyleGAN are: (a) Explicit construction of statistically correlated $\widehat{\bm{c}}, \widehat{\bm{s}}^{(n)}$ (Reichenbach common cause) by letting $\bm{r}_C$ and $\bm{r}_S^{(n)}$ **share a sub-vector** $\bm{r}_{C_1}$; (b) Addition of a Jacobian subspace orthogonal regularizer $\mathcal{L}_{\rm orth}$ to the GAN loss to enforce the CSDI condition. The total training objective is $\mathcal{L}_{\rm GAN} + \lambda_{\rm inv} \mathcal{L}_{\rm inv} + \lambda_{\rm orth} \mathcal{L}_{\rm orth}$, where $\mathcal{L}_{\rm inv}$ is a cyclic reconstruction loss using inverse mappings $\bm{t}_C, \bm{t}_S^{(n)}$ to implicitly enforce an invertible generator (following Zimmermann 2021, Shrestha & Fu 2025).

### Key Designs

1.  **CSDI Assumption + Dual Identifiability Theorems**:
    *   **Function**: Relaxes "content/style disentanglement" from distributional independence to tangent space orthogonality and proves that $\bm{c}, \bm{s}^{(n)}$ are both identifiable up to invertible transformations in the unpaired multi-domain setting.
    *   **Mechanism**: At each point $\bm{x}^{(n)}$, the tangent space is decomposed as $T_{\bm{x}^{(n)}}\mathcal{X}^{(n)} = \mathcal{R}(\bm{J}_{\bm{c}}\bm{g}) \oplus \mathcal{R}(\bm{J}_{\bm{s}^{(n)}}\bm{g})$, assuming these two subspaces are orthogonal (Assumption 3.1). Combined with the domain variability assumption (Assumption 3.3) and distribution matching constraints (3b), Theorem 3.4 establishes $\widehat{\bm{c}} = \bm{\gamma}(\bm{c})$. Further, with $\mathrm{rank}(\bm{J}_{\bm{s}^{(n)}}\bm{g}) = d_S$, it follows that $\widehat{\bm{s}}^{(n)} = \bm{\delta}(\bm{s}^{(n)})$. Theorem 3.6 provides an upper bound for style contamination by content in inexact cases: $\|\bm{J}_{\bm{c}} \widehat{\bm{s}}^{(n)}\|_2 \le \sin\xi \cdot \|\bm{J}_{\bm{c}}\bm{g}\|_2 / \sigma_{\min}(\bm{J}_{\widehat{\bm{s}}}\widehat{\bm{g}})$.
    *   **Design Motivation**: Prior conditions were restrictive because they defined "unrelatedness" at a global probabilistic level. Switching to a local geometric level preserves the physical meaning of "disentanglement" while allowing content and style to share common factors in their distributions, which accurately reflects real-world scenarios where style depends on content.

2.  **Correlated Dual-Noise Sampling Structure**:
    *   **Function**: Explicitly represents the statistical correlation between $\bm{c}$ and $\bm{s}^{(n)}$ in the GAN architecture, remaining compatible with the CSDI assumption.
    *   **Mechanism**: The content noise is split into two parts $\bm{r}_C = (\bm{r}_{C_1}, \bm{r}_{C_2})$, and the style noise is constructed as $\bm{r}_S^{(n)} = (\bm{r}_{C_1}, \bm{r}_{S_1}^{(n)})$—where $\bm{r}_{C_1}$ enters both content and style channels. Thus, $\widehat{\bm{c}} = \bm{e}_C(\bm{r}_C)$ and $\widehat{\bm{s}}^{(n)} = \bm{e}_S^{(n)}(\bm{r}_S^{(n)})$ depend on the common variable $\bm{r}_{C_1}$, while the orthogonal regularizer ensures that this "informational dependence" does not collapse into "tangent space entanglement."
    *   **Design Motivation**: Pure independent sampling (B.I. GAN) precludes correlation structures at the source, making it unable to model Reichenbach-style common causes. This work uses "shared sub-noise" to introduce correlation lightly, while full responsibility for semantic disentanglement is placed on $\mathcal{L}_{\rm orth}$.

3.  **Orthogonal Regularization $\mathcal{L}_{\rm orth}$ via Hutchinson Noise Probing**:
    *   **Function**: Estimates the orthogonality of Jacobian subspaces with $\mathcal{O}(K)$ ($K \ll d$) backward complexity in high-dimensional image generation, avoiding the explicit construction of $d \times d_C$ / $d \times d_S$ Jacobian matrices.
    *   **Mechanism**: Defines $\mathcal{L}_{\rm orth} = \sum_n \mathbb{E}\big[ \|\bm{J}_{\widehat{\bm{s}}^{(n)}}^{\top} \bm{J}_{\widehat{\bm{c}}}\|_F^2 / (\|\bm{J}_{\widehat{\bm{c}}}\|_F^2 \|\bm{J}_{\widehat{\bm{s}}^{(n)}}\|_F^2 + \epsilon) \big]$. Both numerator and denominator are estimated using vector-Jacobian products (VJP) with random vectors $\bm{v}, \mathbb{E}[\bm{v}\bm{v}^{\top}] = \bm{I}_d$: $\bm{J}_{\widehat{\bm{c}}}^{\top}\bm{v} = \nabla_{\widehat{\bm{c}}} \langle \widehat{\bm{g}}, \bm{v} \rangle$, $\bm{J}_{\widehat{\bm{s}}^{(n)}}^{\top}\bm{v} = \nabla_{\widehat{\bm{s}}^{(n)}} \langle \widehat{\bm{g}}, \bm{v} \rangle$. Normalization by the Frobenius norm in the denominator is critical to avoid the trivial "false orthogonality" solution where $\bm{J} \to \bm{0}$.
    *   **Design Motivation**: Directly forming the Jacobian requires $\mathcal{O}(Bd(d_C + d_S))$ memory and $\mathcal{O}(Bd)$ backwards passes, which is infeasible for high-res images. Unlike the path-length regularization of Karras 2020b or the finite-difference methods of Peebles 2020 / Wei 2021, this method solves both scalability and the "false orthogonality" problem via VJP + normalized Frobenius inner product.

### Loss & Training
The complete training objective is $\mathcal{L} = \mathcal{L}_{\rm GAN} + \lambda_{\rm inv} \mathcal{L}_{\rm inv} + \lambda_{\rm orth} \mathcal{L}_{\rm orth}$. $\mathcal{L}_{\rm GAN}$ uses domain-specific discriminators for standard minimax optimization; $\mathcal{L}_{\rm inv} = \mathbb{E}\|\bm{t}_C(\bm{e}_C(\bm{r}_C)) - \bm{r}_C\|_2^2 + \sum_n \mathbb{E}\|\bm{t}_S^{(n)}(\bm{e}_S^{(n)}(\bm{r}_S^{(n)})) - \bm{r}_S^{(n)}\|_2^2$ implicitly forces an invertible generator; $\mathcal{L}_{\rm orth}$ is estimated via Hutchinson probing with $K$ random vectors per step. The core architecture uses DCGAN for MNIST and StyleGAN2-ADA for AFHQ / CelebA-HQ training from scratch.

## Key Experimental Results

### Main Results
Counterfactual generation and cross-domain translation were evaluated on AFHQ (3 domains: dog/cat/wild) and CelebA-HQ (2 domains: male/female).

| Task | Dataset | Metric | StyleGAN2-ADA | I-StyleGAN | B.I. GAN | CSDI-GAN |
|------|--------|------|---------------|------------|----------|----------|
| Generation | AFHQ | FID ↓ | 6.5 | 5.6 | 5.2 | **4.4** |
| Generation | AFHQ | LPIPS ↑ | – | 0.3436 | 0.3995 | **0.4452** |
| Generation | CelebA-HQ | FID ↓ | 5.0 | 4.8 | 4.6 | **4.3** |
| Generation | CelebA-HQ | LPIPS ↑ | – | 0.2799 | 0.2628 | **0.3392** |
| Translation | AFHQ | FID ↓ | 15.0 (StarGANv2) | 17.6 | 10.5 | **7.1** |
| Translation | AFHQ | LPIPS ↑ | 0.3578 | 0.3701 | 0.4107 | **0.4392** |
| Translation | CelebA-HQ | FID ↓ | 14.3 (StarGANv2) | 19.7 | 24.6 | **12.9** |
| Translation | CelebA-HQ | LPIPS ↑ | 0.3148 | 0.2003 | 0.2828 | 0.3105 |

CSDI-GAN outperforms all content-style baselines in both FID and LPIPS for generation tasks on both datasets. In translation, the AFHQ FID is reduced by 32% compared to B.I. GAN, and the CelebA-HQ FID drops from 24.6 to 12.9.

### Ablation Study
| Configuration | AFHQ FID ↓ | AFHQ LPIPS ↑ | CelebA-HQ FID ↓ | CelebA-HQ LPIPS ↑ |
|------|-----------|-------------|----------------|------------------|
| CSDI-GAN (Full) | **4.4** | **0.4452** | **4.3** | **0.3392** |
| CSDI-GAN w/o $\mathcal{L}_{\rm orth}$ | 5.3 | 0.4079 | 6.0 | 0.2467 |
| B.I. GAN (Independent Latent) | 5.2 | 0.3995 | 4.6 | 0.2628 |

Removing $\mathcal{L}_{\rm orth}$ leads to a 27% drop in LPIPS on CelebA-HQ (0.34 → 0.25) and returns FID to levels worse than B.I. GAN. This suggests that modeling correlation via "shared sub-noise" alone is insufficient; the orthogonal regularizer is the essential component for realizing the CSDI assumption.

### Key Findings
- $\mathcal{L}_{\rm orth}$ is the actual driver of identifiability: without it, style (dog breed) in cat2dog translations on AFHQ drifts randomly into tigers or leopards within the same row. This qualitative failure mode, identical to B.I. GAN, proves that both "explicit distributional correlation" and "tangent space orthogonality" are indispensable.
- Inexact orthogonality does not contaminate content (Theorem 3.6 (a)), but it allows content information to leak into style. This leakage is bounded by $\sin\xi \cdot \|\bm{J}_{\bm{c}}\bm{g}\|_2 / \sigma_{\min}(\bm{J}_{\widehat{\bm{s}}}\widehat{\bm{g}})$, providing a quantitative rationale for allowing minor violations in practical scenarios.
- I-StyleGAN tends to preserve cat ear shapes during cat2dog translation (spurious correlation), revealing that disentanglement based on statistical independence fails when content and style are highly co-variant. CSDI-GAN, by not relying on the independence assumption, correctly assigns ear shape to the style component.

## Highlights & Insights
- **From Probabilistic Independence to Geometric Orthogonality**: Shifting "disentanglement" from the distribution level to the tangent space level is not only mathematically weaker (independence implies tangent space orthogonality, but not vice versa) but also naturally connects empirical branches such as IMA, Hessian penalty, and StyleGAN2 path-length. Regularizers that were previously "empirically effective but theoretically opaque" now gain an explanation through identifiability.
- **Engineering Ingenuity of Normalized Frobenius Fractions**: Orthogonal regularization can be bypassed by the network making the Jacobian approach zero if not normalized. Placing $\|\bm{J}\|_F^2$ in the denominator forces the cost of trivial solutions to decrease at the same rate, effectively compelling the network to achieve true disentanglement. This trick is transferable to all Jacobian-based regularizers.
- **Division of Responsibility between Shared Noise and Orthogonal Regularization**: Statistical correlation is introduced into the latent space via $\bm{r}_{C_1}$, while semantic disentanglement is enforced in the tangent space via $\mathcal{L}_{\rm orth}$. This dual-track design—modeling correlation and regularizing disentanglement—can be applied to any causal representation learning scenario where correlation must be permitted while maintaining identifiability.

## Limitations & Future Work
- The authors acknowledge that the implementation is restricted to GAN architectures, relying on explicit content-style branches. Moving the CSDI constraint to modern diffusion or flow-matching models, which have different architectures and training mechanisms, is non-trivial and remains future work.
- In CelebA-HQ, while CSDI-GAN achieves a better FID (12.9) than StarGANv2 (14.3) and B.I. GAN (24.6), its LPIPS (0.3105) is slightly lower than StarGANv2 (0.3148). The benefits of orthogonal constraints may diminish in binary domains with low-dimensional style.
- Hutchinson estimation variance is sensitive to the number of probes $K$ and batch size. The paper does not fully explore the relationship between $K$ and training stability. Potential training instabilities introduced by orthogonal regularization in long, high-resolution runs require further verification.
- Assumption 3.3 (Domain Variability) requires that $\bm{s}^{(n)}$ exhibits actual distributional differences across domains $n$. In weak multi-domain scenarios where styles are nearly identical, the identification power may decrease.

## Related Work & Insights
- **vs B.I. GAN (Shrestha & Fu 2025)**: B.I. GAN assumes $p(\bm{c}, \bm{s}^{(1)}, \ldots, \bm{s}^{(N)}) = p(\bm{c}) \prod_n p(\bm{s}^{(n)})$ (block independence) and requires at least 2 domains for identification. This work removes the independence requirement by enforcing tangent space orthogonality, covering scenarios where style depends on content.
- **vs I-StyleGAN (Xie et al. 2023) / Kong et al. 2022**: Their identifiability relies on component-wise statistical independence and at least $2d_s + 1$ domains. CSDI-GAN has weaker conditions, fewer domain requirements, and significantly higher LPIPS in experiments.
- **vs Yan et al. 2023**: Uses Jacobian sparsity to disentangle correlated content-style; this work uses Jacobian orthogonality, avoiding "non-overlapping feature support" assumptions that are unrealistic for dense data like natural images.
- **vs IMA (Gresele 2021, Buchholz 2022)**: IMA enforces elementwise Jacobian orthogonality, and its identifiability hasn't been fully proven yet. This work focuses on content-style **block** orthogonality, providing a complete proof tailored for multi-domain settings.
- **vs StyleGAN2 path-length / Hessian penalty (Peebles 2020) / Wei 2021**: These methods empirically encourage Jacobian orthogonality but either control single variables or use finite-difference estimation with high tuning costs. This work uses cross-block VJP and Frobenius normalization, offering a cleaner theoretical and practical approach.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to provide a complete proof and high-res GAN implementation for the third major line of content-style identifiability (Orthogonality).
- Experimental Thoroughness: ⭐⭐⭐⭐ Solid results on MNIST, AFHQ, and CelebA-HQ across two tasks with thorough ablations, though lacking comparison with diffusion models and $K$-sensitivity curves.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical exposition; Remarks effectively clarify differences with existing Jacobian regularizers. The bound explanation in the inexact section (Sec 3.3) is somewhat brief.
- Value: ⭐⭐⭐⭐⭐ Provides an identifiability-based explanation for empirically effective Jacobian regularizers; open-sourced code makes it accessible for both theory and engineering communities.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](../../ICCV2025/image_generation/scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [\[ICCV 2025\] AIComposer: Any Style and Content Image Composition via Feature Integration](../../ICCV2025/image_generation/aicomposer_any_style_and_content_image_composition_via_feature_integration.md)
- [\[ICCV 2025\] CSD-VAR: Content-Style Decomposition in Visual Autoregressive Models](../../ICCV2025/image_generation/csd-var_content-style_decomposition_in_visual_autoregressive_models.md)
- [\[ICML 2026\] RAIGen: Rare Attribute Identification in Text-to-Image Generative Models](raigen_rare_attribute_identification_in_text-to-image_generative_models.md)
- [\[ICCV 2025\] StyleMotif: Multi-Modal Motion Stylization using Style-Content Cross Fusion](../../ICCV2025/image_generation/stylemotif_multi-modal_motion_stylization_using_style-content_cross_fusion.md)

</div>

<!-- RELATED:END -->
