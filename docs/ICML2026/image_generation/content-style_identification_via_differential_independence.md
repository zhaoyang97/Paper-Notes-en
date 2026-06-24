---
title: >-
  [Paper Note] Content-Style Identification via Differential Independence
description: >-
  [ICML 2026][Image Generation][content-style disentanglement] This paper proposes CSDI (content-style differential independence), a novel identifiability condition. It proves that unpaired multi-domain content-style block identifiability is achievable under settings where content and style are statistically **correlated** and Jacobians are **dense**, provided that the column spaces of the generator's Jacobians with respect to content and style are mutually orthogonal on the da…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "content-style disentanglement"
  - "differential independence"
  - "Jacobian orthogonality"
  - "GAN"
  - "identifiability"
date: 2026-05-08
content_hash: f24615540ecf5366
---

# Content-Style Identification via Differential Independence

**Conference**: ICML 2026  
**arXiv**: [2605.17827](https://arxiv.org/abs/2605.17827)  
**Code**: https://github.com/subashtimilsina/CSDI (Available)  
**Area**: Image Generation / Content-Style Disentanglement / Identifiability  
**Keywords**: content-style disentanglement, differential independence, Jacobian orthogonality, GAN, identifiability

## TL;DR
This paper proposes CSDI (content-style differential independence), a novel identifiability condition. It proves that unpaired multi-domain content-style block identifiability is achievable under settings where content and style are statistically **correlated** and Jacobians are **dense**, provided that the column spaces of the generator's Jacobians with respect to content and style are mutually orthogonal on the data manifold. Using Hutchinson noise probing, this condition is implemented as a scalable regularization term $\mathcal{L}_{\rm orth}$ for StyleGAN2-ADA. In counterfactual generation and cross-domain translation on AFHQ / CelebA-HQ, FID is reduced from 5.2 / 4.6 to 4.4 / 4.3, and LPIPS is improved from 0.40 / 0.26 to 0.45 / 0.34.

## Background & Motivation
**Background**: Learning the latent variable decomposition $\bm{x}^{(n)} = \bm{g}(\bm{c}, \bm{s}^{(n)})$—representing "shared content + domain-specific style"—from unpaired multi-domain data is the fundamental framework for image translation, counterfactual generation, and domain adaptation. In the absence of paired samples, additional structural assumptions are necessary to ensure that the learned $\widehat{\bm{c}}, \widehat{\bm{s}}^{(n)}$ are consistent with the true $\bm{c}, \bm{s}^{(n)}$, rather than merely achieving "distribution matching with scrambled semantics."

**Limitations of Prior Work**: Existing identifiability conditions fall into two main categories, both of which are quite restrictive. (i) **Statistical Independence** (Xie 2023 / Kong 2022 / Shrestha & Fu 2025): Requires content and style to be block-independent at the probability level. However, in reality, lighting depends on object geometry, and cellular states influence genetic mutations; such "intrinsic dependence of style on content" contradicts the independence assumption. (ii) **Jacobian Sparsity** (Yan 2023): Requires each style dimension to affect only a small, non-overlapping set of data features, which is invalid in dense influence scenarios like single-cell data.

**Key Challenge**: The "information separation" required for identifiability is conventionally defined at the **distribution level** or the **support level**, both of which are too strong. Statistical correlation does not imply information entanglement, and dense influence does not imply non-disentangleability.

**Goal**: To identify a set of conditions for identifiability that require neither $\bm{c} \perp\!\!\!\perp \bm{s}^{(n)}$ nor sparsity of $\bm{J}\bm{g}$, and can scale to high-resolution image generation.

**Key Insight**: From a differential geometry perspective, the concept that "content and style are unrelated" can be localized. As long as the perturbations of the generator with respect to $\bm{c}$ and $\bm{s}^{(n)}$ result in **orthogonal directions** on the data manifold $\mathcal{X}^{(n)}$, disentanglement is possible even if they are statistically correlated or have dense Jacobians. This intuition aligns with IMA (Gresele 2021), StyleGAN2 path-length regularization, and Hessian penalty, but previous work lacked a rigorous proof of content-style block-level identifiability.

**Core Idea**: Replace "distribution independence / sparse support" with "tangent space orthogonality" (differential independence) as the structural condition for content-style identifiability. This is implemented as a differentiable regularization term using Hutchinson VJP, requiring $\mathcal{O}(K)$ rather than $\mathcal{O}(d)$ backward passes.

## Method

### Overall Architecture
CSDI-GAN addresses content-style block identification in unpaired multi-domain settings: aligning learned $\widehat{\bm{c}}, \widehat{\bm{s}}^{(n)}$ with true latent variables without paired samples, even when content and style are correlated and Jacobians are dense. It redefines the independence of content and style from probabilistic independence to tangent space orthogonality and formulates this geometric condition as a differentiable regularization term for StyleGAN2-ADA. The architecture follows a dual-branch generation structure for multi-domain GANs: two learnable latent mappings $\bm{e}_C, \bm{e}_S^{(n)}$ encode Gaussian noise $\bm{r}_C, \bm{r}_S^{(n)}$ into $\widehat{\bm{c}}, \widehat{\bm{s}}^{(n)}$, which are fed into a shared generator $\widehat{\bm{g}}$ to produce $\widehat{\bm{x}}^{(n)} = \widehat{\bm{g}}(\widehat{\bm{c}}, \widehat{\bm{s}}^{(n)})$. Distribution matching is performed by domain-specific discriminators $\widehat{\bm{d}}^{(n)}$ against real $\bm{x}^{(n)}$. The primary differences from B.I. GAN / I-StyleGAN are: first, sharing a sub-vector between content and style noise to explicitly inject statistical correlation; and second, adding a Jacobian subspace orthogonality regularization $\mathcal{L}_{\rm orth}$ to enforce the CSDI condition during training.

### Key Designs

**1. CSDI Assumptions and Dual Identifiability Theorems: Relaxing "Disentanglement" to Tangent Space Orthogonality**

Prior identifiability conditions were restrictive because they defined the independence of content and style at the global probability level (statistical independence) or support level (Jacobian sparsity). In real-world scenarios, such as lighting depending on geometry or cellular states affecting gene mutations, style naturally depends on content. Ours shifts to the local geometric level: at each data point $\bm{x}^{(n)}$, the tangent space is decomposed as $T_{\bm{x}^{(n)}}\mathcal{X}^{(n)} = \mathcal{R}(\bm{J}_{\bm{c}}\bm{g}) \oplus \mathcal{R}(\bm{J}_{\bm{s}^{(n)}}\bm{g})$, requiring only that these two Jacobian column spaces be mutually orthogonal (Assumption 3.1). This condition preserves the physical meaning of "disentanglement" while allowing content and style to share common factors in their distributions. Combined with the domain variability assumption (Assumption 3.3) and distribution matching constraints (3b), Theorem 3.4 establishes content identifiability $\widehat{\bm{c}} = \bm{\gamma}(\bm{c})$. By adding the condition $\mathrm{rank}(\bm{J}_{\bm{s}^{(n)}}\bm{g}) = d_S$, Theorem 3.5 further achieves style identifiability $\widehat{\bm{s}}^{(n)} = \bm{\delta}(\bm{s}^{(n)})$. Recognizing that exact orthogonality is rarely met in practice, Theorem 3.6 provides an upper bound for style contamination by content in the inexact case: $\|\bm{J}_{\bm{c}} \widehat{\bm{s}}^{(n)}\|_2 \le \sin\xi \cdot \|\bm{J}_{\bm{c}}\bm{g}\|_2 / \sigma_{\min}(\bm{J}_{\widehat{\bm{s}}}\widehat{\bm{g}})$, where $\xi$ represents the actual angular deviation between the two subspaces.

**2. Correlated Dual-Noise Sampling: Explicitly Injecting Content–Style Correlation**

Pure independent sampling (as in B.I. GAN) assumes $p(\bm{c}, \bm{s}^{(1)}, \ldots) = p(\bm{c}) \prod_n p(\bm{s}^{(n)})$, which precludes correlation structures and fails to model Reichenbach's common cause. Ours introduces correlation via shared sub-noise: content noise is split as $\bm{r}_C = (\bm{r}_{C_1}, \bm{r}_{C_2})$, and style noise is constructed as $\bm{r}_S^{(n)} = (\bm{r}_{C_1}, \bm{r}_{S_1}^{(n)})$. The shared sub-vector $\bm{r}_{C_1}$ enters both channels, creating statistical dependence between $\widehat{\bm{c}} = \bm{e}_C(\bm{r}_C)$ and $\widehat{\bm{s}}^{(n)} = \bm{e}_S^{(n)}(\bm{r}_S^{(n)})$. This corresponds to the "statistically correlated but tangent-space orthogonal" setting of the CSDI hypothesis. Notably, there is a clear division of labor: modeling correlation is handled by shared noise, while semantic disentanglement is enforced by $\mathcal{L}_{\rm orth}$.

**3. Orthogonal Regularization $\mathcal{L}_{\rm orth}$ via Hutchinson Noise Probing: Scalable Tangent Space Orthogonality**

Directly implementing the CSDI assumption requires measuring the orthogonality of two Jacobian column spaces. However, explicitly constructing $d \times d_C$ and $d \times d_S$ Jacobians for high-resolution images would require $\mathcal{O}(Bd(d_C + d_S))$ memory and $\mathcal{O}(Bd)$ backward passes, which is infeasible. This paper defines a normalized subspace orthogonality loss $\mathcal{L}_{\rm orth} = \sum_n \mathbb{E}\big[ \|\bm{J}_{\widehat{\bm{s}}^{(n)}}^{\top} \bm{J}_{\widehat{\bm{c}}}\|_F^2 / (\|\bm{J}_{\widehat{\bm{c}}}\|_F^2 \|\bm{J}_{\widehat{\bm{s}}^{(n)}}\|_F^2 + \epsilon) \big]$ and uses Hutchinson noise probing to replace both terms with vector-Jacobian product (VJP) estimates using random vectors $\bm{v}$ ($\mathbb{E}[\bm{v}\bm{v}^{\top}] = \bm{I}_d$): $\bm{J}_{\widehat{\bm{c}}}^{\top}\bm{v} = \nabla_{\widehat{\bm{c}}} \langle \widehat{\bm{g}}, \bm{v} \rangle$ and $\bm{J}_{\widehat{\bm{s}}^{(n)}}^{\top}\bm{v} = \nabla_{\widehat{\bm{s}}^{(n)}} \langle \widehat{\bm{g}}, \bm{v} \rangle$. Sampling only $K \ll d$ probe vectors per step reduces the cost from $\mathcal{O}(d)$ to $\mathcal{O}(K)$ backward passes. The Frobenius norm normalization in the denominator is another critical design: without it, the network could "cheat" the orthogonality constraint by pushing the entire Jacobian toward $\bm{0}$ (a "false orthogonality" trivial solution). Normalizing by $\|\bm{J}\|_F^2$ ensures the cost of this shortcut scales down proportionally, forcing the network to achieve true disentanglement.

### Loss & Training
The complete training objective is $\mathcal{L} = \mathcal{L}_{\rm GAN} + \lambda_{\rm inv} \mathcal{L}_{\rm inv} + \lambda_{\rm orth} \mathcal{L}_{\rm orth}$. $\mathcal{L}_{\rm GAN}$ uses domain-specific discriminators for standard minimax training. $\mathcal{L}_{\rm inv} = \mathbb{E}\|\bm{t}_C(\bm{e}_C(\bm{r}_C)) - \bm{r}_C\|_2^2 + \sum_n \mathbb{E}\|\bm{t}_S^{(n)}(\bm{e}_S^{(n)}(\bm{r}_S^{(n)})) - \bm{r}_S^{(n)}\|_2^2$ employs inverse mappings $\bm{t}_C, \bm{t}_S^{(n)}$ for cyclic reconstruction, implicitly enforcing an invertible generator. $\mathcal{L}_{\rm orth}$ is the Hutchinson estimator described above. For the backbone, DCGAN is used for MNIST control experiments, and StyleGAN2-ADA is trained from scratch for AFHQ / CelebA-HQ.

## Key Experimental Results

### Main Results
Counterfactual generation and cross-domain translation were evaluated on AFHQ (3 domains: dog/cat/wild) and CelebA-HQ (2 domains: male/female).

| Task | Dataset | Metric | StyleGAN2-ADA | I-StyleGAN | B.I. GAN | CSDI-GAN (Ours) |
|------|--------|------|---------------|------------|----------|----------|
| Generation | AFHQ | FID ↓ | 6.5 | 5.6 | 5.2 | **4.4** |
| Generation | AFHQ | LPIPS ↑ | – | 0.3436 | 0.3995 | **0.4452** |
| Generation | CelebA-HQ | FID ↓ | 5.0 | 4.8 | 4.6 | **4.3** |
| Generation | CelebA-HQ | LPIPS ↑ | – | 0.2799 | 0.2628 | **0.3392** |
| Translation | AFHQ | FID ↓ | 15.0 (StarGANv2) | 17.6 | 10.5 | **7.1** |
| Translation | AFHQ | LPIPS ↑ | 0.3578 | 0.3701 | 0.4107 | **0.4392** |
| Translation | CelebA-HQ | FID ↓ | **14.3** (StarGANv2) | 19.7 | 24.6 | 12.9 |
| Translation | CelebA-HQ | LPIPS ↑ | 0.3148 | 0.2003 | 0.2828 | 0.3105 |

CSDI-GAN outperforms all content-style baselines in both FID and LPIPS for generation. In translation, FID on AFHQ is reduced by 32% compared to B.I. GAN, and CelebA-HQ FID drops from 24.6 to 12.9.

### Ablation Study

| Configuration | AFHQ FID ↓ | AFHQ LPIPS ↑ | CelebA-HQ FID ↓ | CelebA-HQ LPIPS ↑ |
|------|-----------|-------------|----------------|------------------|
| CSDI-GAN (Full) | **4.4** | **0.4452** | **4.3** | **0.3392** |
| CSDI-GAN w/o $\mathcal{L}_{\rm orth}$ | 5.3 | 0.4079 | 6.0 | 0.2467 |
| B.I. GAN (Independent Latent Baseline) | 5.2 | 0.3995 | 4.6 | 0.2628 |

Removing $\mathcal{L}_{\rm orth}$ results in a 27% drop in LPIPS on CelebA-HQ (0.34 → 0.25) and an FID worse than B.I. GAN. This suggests that modeling correlation via "shared noise" alone is insufficient; the orthogonal regularization is the essential component for realizing the CSDI assumption.

### Key Findings
- $\mathcal{L}_{\rm orth}$ is the primary driver of identifiability: without it, styles (e.g., dog breed) in cat2dog translation randomly drift within the same row, a failure mode shared with B.I. GAN. This indicates that both explicit distribution correlation and tangent space orthogonality are necessary.
- Inexact orthogonality does not contaminate content (Theorem 3.6 (a)), but it does allow style to inherit content information. The contamination is bounded by $\sin\xi \cdot \|\bm{J}_{\bm{c}}\bm{g}\|_2 / \sigma_{\min}(\bm{J}_{\widehat{\bm{s}}}\widehat{\bm{g}})$, providing a quantitative basis for "allowable violations" in practice.
- I-StyleGAN retains cat ear shapes during cat2dog translation (spurious correlation), revealing that disentanglement based on statistical independence fails when content and style highly covary. CSDI-GAN, by not relying on independence, correctly assigns ear shape to the style component.

## Highlights & Insights
- **From Probabilistic Independence to Geometric Orthogonality**: Moving "disentanglement" from the distribution level to the tangent space level is not only mathematically weaker (independence implies tangent space orthogonality, but not vice versa), it also naturally connects IMA, Hessian penalty, and StyleGAN2 path-length regularization. Previously "empirically effective but theoretically unclear" regularizations are given an identifiability explanation within this framework.
- **Engineering Ingenuity of the Hutchinson Normalized Fraction**: Orthogonal regularization is easily bypassed if not normalized. Using $\|\bm{J}\|_F^2$ in the denominator instead of a constant ensures that the trivial solution's cost decreases proportionally, forcing the network toward true disentanglement. This trick is transferable to all Jacobian-based regularizations.
- **Division of Labor between Shared Noise and Orthogonal Regularization**: Shared noise $\bm{r}_{C_1}$ introduces statistical correlation into the latent space, while $\mathcal{L}_{\rm orth}$ enforces semantic disentanglement in the tangent space. This dual-track design—modeling correlation while regularizing for disentanglement—is applicable to any causal representation learning scenario requiring identifiability despite correlation.

## Limitations & Future Work
- The authors acknowledge that the implementation is restricted to GAN architectures with explicit content-style branches. Porting CSDI constraints to modern diffusion or flow-matching models, which have different architectures and training mechanisms, is non-trivial and remains future work.
- Translation FID on CelebA-HQ (12.9) is better than StarGANv2 (14.3), but LPIPS (0.3105) is slightly lower than StarGANv2 (0.3148)—indicating that the benefits of orthogonality may be marginal in binary domains with low style dimensionality.
- The variance of the Hutchinson estimator is sensitive to both $K$ (number of probes) and batch size. The relationship between $K$ and training stability was not fully explored.
- Assumption 3.3 (Domain Variability) requires $\bm{s}^{(n)}$ to have true distributional differences across domains; identifiability may degrade in "weak multi-domain" scenarios where styles are nearly identical (e.g., same identity under different lighting).

## Related Work & Insights
- **vs B.I. GAN (Shrestha & Fu 2025)**: B.I. GAN assumes block independence $p(\bm{c}, \bm{s}^{(1)}, \ldots, \bm{s}^{(N)}) = p(\bm{c}) \prod_n p(\bm{s}^{(n)})$, requiring at least 2 domains. Ours foregoes this independence in favor of tangent space orthogonality, covering broader scenarios where style depends on content.
- **vs I-StyleGAN (Xie et al. 2023) / Kong et al. 2022**: Their identifiability relies on component-wise statistical independence and at least $2d_s + 1$ domains. Ours has weaker conditions and requires fewer domains, achieving significantly higher LPIPS in experiments.
- **vs Yan et al. 2023**: Uses Jacobian sparsity to disentangle correlated content and style; ours uses Jacobian orthogonality, avoiding the unrealistic "non-overlapping feature support" assumption in dense data like natural images.
- **vs IMA (Gresele 2021, Buchholz 2022)**: IMA applies elementwise Jacobian orthogonality, and its identifiability hasn't been fully proven. Ours focuses on content-style **block** orthogonality with a complete proof specifically for multi-domain settings.
- **vs StyleGAN2 path-length / Hessian penalty (Peebles 2020) / Wei 2021**: These methods empirically encourage Jacobian orthogonality but typically control a single variable or use finite-difference estimates that are costly to tune. Ours uses cross-block VJP and Frobenius normalization for a cleaner theoretical and practical implementation.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to provide a complete proof for the third path of content-style identifiability (orthogonality vs. independence/sparsity) and implement it in high-res GANs.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Includes MNIST control experiments, AFHQ / CelebA-HQ benchmarks, and both generation/translation tasks. Key ablations are present, though diffusion comparisons and $K$-sensitivity curves are missing.
- **Writing Quality**: ⭐⭐⭐⭐ Clear theoretical exposition; Remarks effectively distinguishours from existing Jacobian methods. The explanation of the bound in the inexact section (Section 3.3) is slightly brief.
- **Value**: ⭐⭐⭐⭐⭐ Provides an identifiability explanation for a class of "empirically effective but theoretically messy" Jacobian regularizations. Open-sourced code makes it accessible to both engineers and theorists.

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
