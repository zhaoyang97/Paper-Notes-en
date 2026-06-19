---
title: >-
  [Paper Note] Content-Style Identification via Differential Independence
description: >-
  [ICML 2026][Image Generation][content-style disentanglement] This paper proposes CSDI (content-style differential independence), a novel identifiability condition. It proves that unpaired multi-domain content-style block identifiability can be achieved under statistically **correlated** content-style and **dense** Jacobian settings, provided the column spaces of the generator's
tags:
  - ICML 2026
  - Image Generation
  - content-style disentanglement
  - differential independence
  - Jacobian orthogonality
  - GAN
  - identifiability
date: 2026-05-08
content_hash: 1a72fabf485491fc
---
# Content-Style Identification via Differential Independence

**Conference**: ICML 2026  
**arXiv**: [2605.17827](https://arxiv.org/abs/2605.17827)  
**Code**: https://github.com/subashtimilsina/CSDI (Available)  
**Area**: Image Generation / Content-Style Disentanglement / Identifiability  
**Keywords**: content-style disentanglement, differential independence, Jacobian orthogonality, GAN, identifiability

## TL;DR
This paper proposes CSDI (content-style differential independence), a novel identifiability condition. It proves that unpaired multi-domain content-style block identifiability can be achieved under statistically **correlated** content-style and **dense** Jacobian settings, provided the column spaces of the generator's Jacobian with respect to content and style are mutually orthogonal on the data manifold. This condition is implemented as a scalable regularization term $\mathcal{L}_{\rm orth}$ via Hutchinson noise probing on StyleGAN2-ADA. In counterfactual generation and cross-domain translation on AFHQ/CelebA-HQ, the method reduces FID from 5.2 / 4.6 to 4.4 / 4.3 and improves LPIPS from 0.40 / 0.26 to 0.45 / 0.34.

## Background & Motivation
**Background**: Learning the latent variable decomposition $\bm{x}^{(n)} = \bm{g}(\bm{c}, \bm{s}^{(n)})$ as "shared content + domain-specific style" from unpaired multi-domain data forms the backbone of image translation, counterfactual generation, and domain adaptation. In the absence of paired samples, additional structural assumptions are required to ensure that the learned $\widehat{\bm{c}}, \widehat{\bm{s}}^{(n)}$ align with the true $\bm{c}, \bm{s}^{(n)}$, rather than merely matching distributions while losing semantic consistency.

**Limitations of Prior Work**: Existing identifiability conditions fall into two restrictive categories. (i) **Statistical Independence** (Xie 2023 / Kong 2022 / Shrestha & Fu 2025): Requires content and style to be block-independent at the probability level. However, real-world style often depends on content (e.g., lighting depends on geometry), violating the independence assumption. (ii) **Jacobian Sparsity** (Yan 2023): Requires each style dimension to affect only a small, non-overlapping subset of data features, which is unrealistic in dense influence scenarios like single-cell data or natural images.

**Key Challenge**: Traditional "information separation" conditions for identifiability are typically imposed at the **distributional** or **support** levels, both of which are too strong. Distributional correlation does not imply information entanglement, and dense influence does not imply non-decouplability.

**Goal**: To find a set of identifiability conditions that require neither $\bm{c} \perp\!\!\!\perp \bm{s}^{(n)}$ nor Jacobian sparsity, while remaining scalable to high-resolution image generation.

**Key Insight**: From a differential geometry perspective, the notion that "content and style are unrelated" can be localized. As long as perturbations of the generator with respect to $\bm{c}$ and $\bm{s}^{(n)}$ yield **orthogonal directions** on the data manifold $\mathcal{X}^{(n)}$, disentanglement can be achieved even if they are statistically correlated and the Jacobian is dense. This intuition aligns with IMA (Gresele 2021), StyleGAN2 path-length regularization, and Hessian penalty, but previous works lacked rigorous block-level identifiability proofs for content-style.

**Core Idea**: Use "tangent space orthogonality" (differential independence) instead of "distributional independence / sparse support" as the structural condition for content-style identifiability. This is implemented as a differentiable regularization using Hutchinson VJP, requiring $\mathcal{O}(K)$ rather than $\mathcal{O}(d)$ backward passes.

## Method

### Overall Architecture
CSDI-GAN addresses block identification of content and style in unpaired multi-domain settings. It reformulates "independence" as tangent space orthogonality and implements this geometric condition as a differentiable regularizer for StyleGAN2-ADA. The architecture follows a two-branch generation structure: two learnable latent mappings $\bm{e}_C, \bm{e}_S^{(n)}$ encode Gaussian noise $\bm{r}_C, \bm{r}_S^{(n)}$ into $\widehat{\bm{c}}, \widehat{\bm{s}}^{(n)}$, which are fed into a shared generator $\widehat{\bm{g}}$ to produce $\widehat{\bm{x}}^{(n)} = \widehat{\bm{g}}(\widehat{\bm{c}}, \widehat{\bm{s}}^{(n)})$. A domain-specific discriminator $\widehat{\bm{d}}^{(n)}$ performs distribution matching. Key differences from B.I. GAN / I-StyleGAN include: 1) allowing content and style noise to share a sub-vector to explicitly inject statistical correlation; 2) adding a Jacobian subspace orthogonality regularizer $\mathcal{L}_{\rm orth}$ alongside the GAN loss.

### Key Designs

**1. CSDI Assumption and Dual Identifiability Theorems: Relaxing Disentanglement from Independence to Orthogonality**

Prior conditions were restrictive because they defined "independence" globally. This work moves to the local geometric level: at each data point $\bm{x}^{(n)}$, the tangent space is decomposed as $T_{\bm{x}^{(n)}}\mathcal{X}^{(n)} = \mathcal{R}(\bm{J}_{\bm{c}}\bm{g}) \oplus \mathcal{R}(\bm{J}_{\bm{s}^{(n)}}\bm{g})$, requiring only that these two Jacobian column spaces are mutually orthogonal (Assumption 3.1). This allows content/style to share factors in their distributions. Combined with domain variability (Assumption 3.3) and distribution matching (3b), Theorem 3.4 proves content identifiability $\widehat{\bm{c}} = \bm{\gamma}(\bm{c})$. Theorem 3.5 further proves style identifiability $\widehat{\bm{s}}^{(n)} = \bm{\delta}(\bm{s}^{(n)})$ given $\mathrm{rank}(\bm{J}_{\bm{s}^{(n)}}\bm{g}) = d_S$. Theorem 3.6 provides an upper bound for style contamination by content in the inexact case: $\|\bm{J}_{\bm{c}} \widehat{\bm{s}}^{(n)}\|_2 \le \sin\xi \cdot \|\bm{J}_{\bm{c}}\bm{g}\|_2 / \sigma_{\min}(\bm{J}_{\widehat{\bm{s}}}\widehat{\bm{g}})$, where $\xi$ is the angular deviation.

**2. Correlated Dual-Noise Sampling: Explicitly Injecting Content-Style Correlation**

Pure independent sampling (B.I. GAN) assumes $p(\bm{c}, \bm{s}^{(1)}, \ldots) = p(\bm{c}) \prod_n p(\bm{s}^{(n)})$, which prevents modeling common factors. This work uses shared sub-noise: content noise is split into $(\bm{r}_{C_1}, \bm{r}_{C_2})$, and style noise is constructed as $\bm{r}_S^{(n)} = (\bm{r}_{C_1}, \bm{r}_{S_1}^{(n)})$. The shared vector $\bm{r}_{C_1}$ creates statistical dependency between $\widehat{\bm{c}} = \bm{e}_C(\bm{r}_C)$ and $\widehat{\bm{s}}^{(n)} = \bm{e}_S^{(n)}(\bm{r}_S^{(n)})$. Here, common noise models correlation, while $\mathcal{L}_{\rm orth}$ enforces semantic disentanglement.

**3. Orthogonal Regularizer $\mathcal{L}_{\rm orth}$ via Hutchinson Probing: Scaling Tangent Orthogonality**

Explicitly constructing $d \times d_C$ or $d \times d_S$ Jacobians for high-resolution images is infeasible as it requires $\mathcal{O}(Bd)$ backward passes. This work defines a normalized subspace orthogonality loss $\mathcal{L}_{\rm orth} = \sum_n \mathbb{E}\big[ \|\bm{J}_{\widehat{\bm{s}}^{(n)}}^{\top} \bm{J}_{\widehat{\bm{c}}}\|_F^2 / (\|\bm{J}_{\widehat{\bm{c}}}\|_F^2 \|\bm{J}_{\widehat{\bm{s}}^{(n)}}\|_F^2 + \epsilon) \big]$ and uses Hutchinson probing to estimate the numerator and denominator using vector-Jacobian products (VJP) with random vectors $\bm{v}$: $\bm{J}_{\widehat{\bm{c}}}^{\top}\bm{v} = \nabla_{\widehat{\bm{c}}} \langle \widehat{\bm{g}}, \bm{v} \rangle$. By sampling $K \ll d$ probes, the cost is reduced to $\mathcal{O}(K)$ backward passes. The Frobenius norm normalization in the denominator is crucial to prevent the network from finding a trivial solution by shrinking the Jacobian to zero.

### Loss & Training
The objective is $\mathcal{L} = \mathcal{L}_{\rm GAN} + \lambda_{\rm inv} \mathcal{L}_{\rm inv} + \lambda_{\rm orth} \mathcal{L}_{\rm orth}$. $\mathcal{L}_{\rm GAN}$ is a standard minimax loss; $\mathcal{L}_{\rm inv} = \mathbb{E}\|\bm{t}_C(\bm{e}_C(\bm{r}_C)) - \bm{r}_C\|_2^2 + \sum_n \mathbb{E}\|\bm{t}_S^{(n)}(\bm{e}_S^{(n)}(\bm{r}_S^{(n)})) - \bm{r}_S^{(n)}\|_2^2$ provides cyclic reconstruction via inverse mappings $\bm{t}_C, \bm{t}_S^{(n)}$ to encourage an invertible generator; $\mathcal{L}_{\rm orth}$ is the Hutchinson-estimated regularizer. For backbone networks, DCGAN is used for MNIST, and StyleGAN2-ADA is trained from scratch for AFHQ and CelebA-HQ.

## Key Experimental Results

### Main Results
Counterfactual generation and cross-domain translation on AFHQ (dog/cat/wild) and CelebA-HQ (male/female).

| Task | Dataset | Metric | StyleGAN2-ADA | I-StyleGAN | B.I. GAN | CSDI-GAN |
|------|--------|------|---------------|------------|----------|----------|
| Generation | AFHQ | FID ↓ | 6.5 | 5.6 | 5.2 | **4.4** |
| Generation | AFHQ | LPIPS ↑ | – | 0.3436 | 0.3995 | **0.4452** |
| Generation | CelebA-HQ | FID ↓ | 5.0 | 4.8 | 4.6 | **4.3** |
| Generation | CelebA-HQ | LPIPS ↑ | – | 0.2799 | 0.2628 | **0.3392** |
| Translation | AFHQ | FID ↓ | 15.0 (StarGANv2) | 17.6 | 10.5 | **7.1** |
| Translation | AFHQ | LPIPS ↑ | 0.3578 | 0.3701 | 0.4107 | **0.4392** |
| Translation | CelebA-HQ | FID ↓ | **14.3** (StarGANv2) | 19.7 | 24.6 | 12.9 |
| Translation | CelebA-HQ | LPIPS ↑ | 0.3148 | 0.2003 | 0.2828 | 0.3105 |

CSDI-GAN outperforms all content-style baselines in FID and LPIPS. In AFHQ translation, FID is reduced by 32% compared to B.I. GAN.

### Ablation Study

| Configuration | AFHQ FID ↓ | AFHQ LPIPS ↑ | CelebA-HQ FID ↓ | CelebA-HQ LPIPS ↑ |
|------|-----------|-------------|----------------|------------------|
| CSDI-GAN (Full) | **4.4** | **0.4452** | **4.3** | **0.3392** |
| CSDI-GAN w/o $\mathcal{L}_{\rm orth}$ | 5.3 | 0.4079 | 6.0 | 0.2467 |
| B.I. GAN (Independent Latent) | 5.2 | 0.3995 | 4.6 | 0.2628 |

Removing $\mathcal{L}_{\rm orth}$ causes LPIPS on CelebA-HQ to drop by 27%, demonstrating that the orthogonal regularizer is essential for grounding the CSDI assumption.

### Key Findings
- $\mathcal{L}_{\rm orth}$ is the primary driver of identifiability. Without it, cat2dog translation results in "style drift" (e.g., dogs morphing into tigers), as seen in the failure modes of B.I. GAN.
- Inexact orthogonality does not contaminate content but allows style to be polluted by content, governed by the bound in Theorem 3.6.
- I-StyleGAN retains spurious correlations (e.g., cat ear shapes in dog translations) because its disentanglement relies on statistical independence, which fails when content and style co-vary. CSDI-GAN correctly assigns ear shape to the style branch.

## Highlights & Insights
- **From Distributional Independence to Geometric Orthogonality**: Shifting disentanglement to the tangent space level is a weaker requirement mathematically but aligns naturally with empirical evidence from IMA and Hessian penalties. This provides a theoretical identifiability explanation for previously "heuristically effective" regularizers.
- **Hutchinson Normalized Fraction**: Normalizing the loss with the Frobenius norm of the Jacobian prevents trivial solutions. This trick is applicable to other Jacobian-based losses like path-length or Hessian penalties.
- **Division of Responsibility**: Statistical correlation is handled by shared noise $\bm{r}_{C_1}$, while semantic disentanglement is enforced by $\mathcal{L}_{\rm orth}$ in the tangent space. This dual-track design is applicable to any causal representation learning requiring "correlated yet identifiable" features.

## Limitations & Future Work
- Implementation is currently tied to GAN architectures with explicit content-style branches. Generalizing CSDI to modern diffusion or flow-matching models is non-trivial.
- In CelebA-HQ translation, while CSDI-GAN achieves better FID than B.I. GAN, its LPIPS gain is marginal compared to StarGANv2 when the style dimension is low.
- The variance of Hutchinson estimation is sensitive to the number of probes ($K$) and batch size, which may affect stability during high-resolution training.
- Identifiability decreases in "weak multi-domain" scenarios where the style distribution differs minimally across domains (Assumption 3.3).

## Related Work & Insights
- **vs B.I. GAN (Shrestha & Fu 2025)**: B.I. GAN requires block independence, which is a stronger assumption. CSDI covers scenarios where style depends on content.
- **vs I-StyleGAN (Xie et al. 2023)**: Requires component-wise independence and more domains. CSDI requires fewer domains and achieves higher LPIPS.
- **vs Yan et al. 2023**: Uses Jacobian sparsity. CSDI uses orthogonality, which is more realistic for dense data like natural images.
- **vs StyleGAN2 path-length / Hessian penalty**: These methods empirically encourage orthogonality but lack identifiability proofs and use more expensive estimation methods. This work provides both the theory and a scalable cross-block VJP implementation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to provide full proofs and implementation of content-style identifiability via orthogonality in high-res GANs.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive benchmarks and ablations, though lacking sensitivity curves for $K$ and comparison with diffusion models.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical narrative and distinction from existing methods; however, the inexact bound explanation in Section 3.3 is somewhat brief.
- Value: ⭐⭐⭐⭐⭐ Provides a rigorous foundation for a class of effective Jacobian regularizers; open-source code makes it highly accessible.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SplitFlux: Learning to Decouple Content and Style from a Single Image](../../CVPR2026/image_generation/splitflux_learning_to_decouple_content_and_style_from_a_single_image.md)
- [\[ECCV 2024\] Implicit Style-Content Separation using B-LoRA](../../ECCV2024/image_generation/implicit_style-content_separation_using_b-lora.md)
- [\[ICML 2026\] RAIGen: Rare Attribute Identification in Text-to-Image Generative Models](raigen_rare_attribute_identification_in_text-to-image_generative_models.md)
- [\[ICCV 2025\] SCFlow: Implicitly Learning Style and Content Disentanglement with Flow Models](../../ICCV2025/image_generation/scflow_implicitly_learning_style_and_content_disentanglement_with_flow_models.md)
- [\[CVPR 2026\] CRAFT-LoRA: Content-Style Personalization via Rank-Constrained Adaptation and Training-Free Fusion](../../CVPR2026/image_generation/craft-lora_content-style_personalization_via_rank-constrained_adaptation_and_tra.md)

</div>

<!-- RELATED:END -->
