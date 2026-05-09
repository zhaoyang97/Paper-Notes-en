---
title: >-
  [Paper Note] HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars
description: >-
  [CVPR 2026][3D Vision][Gaussian splatting] This paper proposes HyperGaussians, which extends 3DGS to high-dimensional multivariate Gaussians. Expression-dependent attribute variations are modeled via conditional distributions, and an inverse covariance trick enables efficient conditioning. Integrated as a plug-and-play module into FlashAvatar and GaussianHeadAvatar, the method significantly improves high-frequency detail quality.
tags:
  - CVPR 2026
  - 3D Vision
  - Gaussian splatting
  - face avatars
  - high-dimensional Gaussians
  - facial animation
  - conditional distribution
date: 2026-05-08
content_hash: 6c313b6af36d5111
---

# HyperGaussians: High-Dimensional Gaussian Splatting for High-Fidelity Animatable Face Avatars

**Conference**: CVPR 2026
**arXiv**: [2507.02803](https://arxiv.org/abs/2507.02803)
**Code**: [https://gserifi.github.io/HyperGaussians](https://gserifi.github.io/HyperGaussians)
**Area**: 3D Vision
**Keywords**: Gaussian splatting, face avatars, high-dimensional Gaussians, facial animation, conditional distribution

## TL;DR
This paper proposes HyperGaussians, which extends 3DGS to high-dimensional multivariate Gaussians. Expression-dependent attribute variations are modeled via conditional distributions, and an inverse covariance trick enables efficient conditioning. Integrated as a plug-and-play module into FlashAvatar and GaussianHeadAvatar, the method significantly improves high-frequency detail quality.

## Background & Motivation
1. **Background**: 3D Gaussian splatting (3DGS) has become the standard approach for face avatar modeling. Current SOTA methods bind Gaussians to 3DMM meshes and use MLPs to predict expression-dependent offsets for handling dynamics.
2. **Limitations of Prior Work**: Existing methods still struggle with nonlinear deformations (eye closure, mouth opening), complex lighting effects (specular reflections on glasses), and fine-grained structures (tooth gaps, glasses frames)—high-frequency details that are central to the uncanny valley effect.
3. **Key Challenge**: The expressiveness of 3D Gaussian primitives is inherently limited. Each Gaussian has only a few attributes (position, rotation, scale, color), and even with MLP-predicted expression-dependent offsets, linear combinations of attributes remain limited in expressiveness.
4. **Goal**: How can the expressiveness of 3D Gaussian primitives be enhanced to capture high-frequency dynamic effects without redesigning the entire pipeline?
5. **Key Insight**: Inspired by HyperNeRF—modeling deformation fields in a high-dimensional space and slicing them can handle topological changes. This idea is applied to Gaussian primitives: each Gaussian is defined as a multivariate Gaussian distribution in a high-dimensional space, and conditioning (slicing) yields expression-dependent 3D Gaussians.
6. **Core Idea**: Standard 3D Gaussians are extended into $(m+n)$-dimensional HyperGaussians. Conditioning on an $n$-dimensional latent embedding yields dynamically adaptive 3D attributes, and an inverse covariance trick maintains efficiency.

## Method

### Overall Architecture
HyperGaussians is a plug-and-play representation enhancement module. In the original pipeline (e.g., FlashAvatar), the MLP outputs expression-dependent offsets $\Delta\mu, \Delta r, \Delta s$. With HyperGaussians, the MLP instead outputs a latent vector $z_\psi$, from which MAP-estimated offsets are computed via the conditional distribution of the high-dimensional Gaussian. Only the Gaussian representation is replaced; all other components (loss functions, hyperparameters) remain unchanged.

### Key Designs
1. **HyperGaussian High-Dimensional Extension**:
    - Function: Extends standard 3D Gaussians to $(m+n)$-dimensional multivariate Gaussians with an additional $n$-dimensional latent space, enhancing representational capacity.
    - Mechanism: A joint distribution $\gamma = (\gamma_a, \gamma_b)^\top \sim \mathcal{N}(\mu, \Sigma)$ is defined, where $\gamma_a \in \mathbb{R}^m$ corresponds to Gaussian attributes (position, rotation, scale) and $\gamma_b \in \mathbb{R}^n$ corresponds to latent embeddings. Conditioning $p(\gamma_a|\gamma_b)$ on a given latent embedding yields an ordinary 3D Gaussian: $\mu_{a|b} = \mu_a + \Sigma_{ab}\Sigma_{bb}^{-1}(\gamma_b - \mu_b)$. From a Bayesian perspective, this is equivalent to MAP estimation, where the prior $p(\gamma_a)$ acts as implicit regularization.
    - Design Motivation: Compared to NDGS, which only models the conditional distribution of position (rotation and scale are independent of the latent code), HyperGaussians also models $p(\Delta r|z)$ and $p(\Delta s|z)$, allowing Gaussians to rotate and scale according to the latent code, yielding strictly greater expressiveness.

2. **Inverse Covariance Trick**:
    - Function: Reduces the computational complexity of high-dimensional conditioning from $O(n^3 + mn^2)$ to $O(m^3 + m^2n)$—from cubic in $n$ to linear in $n$.
    - Mechanism: Reparameterizing via the precision matrix $\Lambda = \Sigma^{-1}$, the conditional mean becomes $\mu_{a|b} = \mu_a - \Lambda_{aa}^{-1}\Lambda_{ab}(\gamma_b - \mu_b)$ and the conditional covariance becomes $\Sigma_{a|b} = \Lambda_{aa}^{-1}$. Only the small matrix $\Lambda_{aa} \in \mathbb{R}^{m \times m}$ (where $m=3$ or $4$) needs to be stored and inverted, reducing the parameter count from $O((m+n)^2)$ to $O(m^2 + mn)$.
    - Design Motivation: The naïve implementation is prohibitively slow and memory-intensive at large latent dimensions ($n > 8$). At $n=8$, this yields a 150% speedup; at $n=128$, a 15000% speedup with 48–90% memory reduction, making large latent dimensions practical.

3. **Plug-and-Play Integration Strategy**:
    - Function: Improves performance by directly replacing 3DGS primitives without modifying architecture or hyperparameters.
    - Mechanism: Using FlashAvatar as an example, the deformation MLP output is changed from direct offsets to a latent code $z_\psi$, from which offsets are computed via the HyperGaussian conditional distribution. All other components (losses, training strategy, learning rates) remain unchanged. A latent dimension of $n=8$ achieves the best performance. GaussianHeadAvatar is integrated in the same manner.
    - Design Motivation: As an improvement orthogonal to architectural design, HyperGaussians enhances quality without touching the model architecture, meaning it can be freely combined with any future architectural improvements.

### Loss & Training
All loss functions and hyperparameters are inherited from the base methods (FlashAvatar / GaussianHeadAvatar). The learning rate for HyperGaussian parameters is set to $10^{-4}$. FlashAvatar trains with ~15k Gaussians for 30k iterations (15–20 minutes on RTX 4090); GaussianHeadAvatar trains for 600k iterations (~2 days). HyperGaussians add only ~1% training overhead (30 minutes). Key design decisions: three independent HyperGaussian distributions are maintained per Gaussian primitive for position, rotation, and scale respectively; Cholesky parameterization ensures positive definiteness of the covariance matrix. The latent dimension $n=8$ is determined via ablation as the optimal trade-off. Parameterizing the precision matrix requires only the two blocks $\Lambda_{aa}$ and $\Lambda_{ab}$, significantly reducing memory usage. Code is publicly released to facilitate integration into other pipelines.

## Key Experimental Results

### Main Results (29 subjects, 6 datasets)

| Method | PSNR↑ | SSIM↑ | LPIPS↓ |
|--------|-------|-------|--------|
| SplattingAvatar | 28.58 | 0.9396 | 0.0902 |
| MonoGaussianAvatar | 29.94 | 0.9456 | 0.0655 |
| FlashAvatar | 29.43 | 0.9466 | 0.0511 |
| **Ours (FA)** | **29.99** | **0.9510** | **0.0498** |
| GaussianHeadAvatar | 24.10 | 0.8819 | 0.2027 |
| **Ours (GHA)** | **24.38** | **0.8819** | **0.1977** |

### Ablation Study

| Configuration | LPIPS↓ | FPS |
|---------------|--------|-----|
| HyperGaussians ($n=8$) | 0.0498 | 300 |
| Deeper MLP (same param count) | 0.0572 (+15%) | 158 (−47%) |
| Wider MLP (same param count) | 0.0512 (+3%) | 178 (−41%) |

### Key Findings
- HyperGaussians significantly outperforms increased MLP capacity under the same parameter budget, achieving better LPIPS without sacrificing speed (300 FPS vs. 158 FPS).
- Improvements are most prominent in high-frequency details: specular reflections, glasses frames, tooth gaps, and skin wrinkles—precisely the areas where standard 3DGS is weakest—confirming that high-dimensional conditioning genuinely enhances local expressiveness.
- Even $n=1$ (the minimal extension) already outperforms the baseline; $n=8$ is the optimal trade-off; $n=128$ remains feasible with the inverse covariance trick but yields diminishing returns.
- The inverse covariance trick is critical for practicality: without it, $n=128$ is nearly infeasible (15000% speedup, >90% memory reduction with the trick).
- The method is also effective in cross-reenactment scenarios, indicating that the learned capability is general high-frequency modeling rather than overfitting to specific expressions.
- In multi-view settings (GaussianHeadAvatar), HyperGaussians likewise improve wrinkle and reflection quality with only 1% additional training overhead.
- **Inverse covariance efficiency**: at $n=8$, naïve implementation uses 42MB per conditioning step vs. 22MB with the trick (−48%); at $n=128$, memory is reduced by >90%.
- **Cross-dataset consistency**: consistent improvements across 6 different datasets including INSTA, NerFace, IMAvatar, and NeRSemble.
- Training convergence is also accelerated—HyperGaussians reach the baseline's final quality in fewer iterations.

## Highlights & Insights
- **Elegant Bayesian Unification**: NDGS, 4DGS, 6DGS, 7DGS, and other multi-dimensional Gaussian extensions are unified as special cases of conditional Gaussian distributions, with their expressiveness limitations (inability to condition rotation/scale) clearly identified. This theoretical unification is itself a significant contribution.
- **Improvement Orthogonal to Architecture**: Quality is enhanced without modifying any model design, meaning HyperGaussians can be freely combined with any future architectural improvements—this "free upgrade" property is highly attractive.
- **Generality of the Inverse Covariance Trick**: The trick is not limited to face avatars; any scenario requiring high-dimensional Gaussian conditioning can benefit. The reduction from $O(n^3)$ to $O(n)$ is highly significant for real-time applications.

## Limitations & Future Work
- Validation is currently limited to face avatars; effectiveness on full-body dynamic scenes (clothing, hair motion, and other large-scale nonlinear deformations) remains unknown.
- The conditional covariance matrix $\Sigma_{a|b} = \Lambda_{aa}^{-1}$, which encodes uncertainty in 3D Gaussian parameters, is not utilized in the current implementation—it could be leveraged for active learning or rendering quality assessment.
- Storing and inverting the precision matrix blocks $\Lambda_{aa}$ and $\Lambda_{ab}$ is required, and while the parameter count is reduced from $O((m+n)^2)$ to $O(m^2+mn)$, the implementation is more complex than standard 3DGS.
- For highly extreme out-of-distribution expressions, the extrapolation capability of the conditional distribution may be limited, as the linear conditioning assumption of multivariate Gaussians presupposes linear relationships among parameters.
- Quantitative improvements, while consistent, are modest in magnitude (PSNR +0.56, LPIPS −0.0013); visual improvements are concentrated in high-frequency detail regions.
- The choice of latent dimension $n=8$ is determined by ablation, but different scenarios may require different optimal values. An adaptive dimension selection mechanism remains to be developed.

## Related Work & Insights
- **vs. NDGS/6DGS/7DGS**: These methods are special cases of HyperGaussians—they cannot condition rotation/scale and are computationally infeasible at large latent dimensions. HyperGaussians addresses both limitations and eliminates the Gaussian vanishing problem under large deformations.
- **vs. FlashAvatar**: Replacing only the Gaussian representation reduces LPIPS from 0.0511 to 0.0498 and improves PSNR from 29.43 to 29.99, demonstrating the value of improving the underlying representation. FlashAvatar's MLP-predicted offsets constitute "external modulation," whereas HyperGaussians internalize modulation within the representation itself.
- **vs. HyperNeRF**: The concept of modeling deformations in a high-dimensional space is inherited from HyperNeRF, but HyperGaussians adapts it to the Gaussian splatting framework and achieves real-time rendering (300 FPS) via the inverse covariance trick. HyperNeRF's "slicing" and HyperGaussians' "conditioning" are formally analogous but differ entirely in implementation and theoretical framework.
- **vs. MonoGaussianAvatar**: MonoGaussianAvatar requires 100k+ Gaussians and 12 hours of training, whereas FlashAvatar + HyperGaussians uses only ~15k Gaussians and 20 minutes, with superior performance—demonstrating that better representations matter more than more parameters.
- **vs. SplattingAvatar**: SplattingAvatar does not account for the effect of expression changes on appearance (e.g., specular reflection shifts), leading to blurry rendering. HyperGaussians naturally models this dependency through conditional distributions.
- **Future Outlook**: The HyperGaussians framework can be applied to dynamic scenes beyond faces (full body, animals, deformable objects); the inverse covariance trick makes higher latent dimensions feasible.
- **vs. 4DGS**: 4DGS adds a temporal dimension but requires custom CUDA kernels; HyperGaussians adds arbitrary dimensions and can use standard 3DGS rasterizers via the inverse covariance trick.
- **Gaussian Mixture Model Perspective**: NDGS is essentially a multi-dimensional extension of Gaussian mixture models, but density functions that depend on the joint distribution become unstable under large deformations; HyperGaussians avoids this by using conditional distributions. Since the conditional mean computation is equivalent to MAP estimation, the prior $p(\mathcal{A})$ acts as implicit regularization, helping preserve fine details at test time.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Triple innovation of high-dimensional Gaussians + Bayesian perspective + inverse covariance trick, with outstanding theoretical depth and unification of multiple existing methods
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 29 subjects across 6 datasets, including ablations, speed comparisons, and both monocular and multi-view settings
- Writing Quality: ⭐⭐⭐⭐⭐ Mathematical derivations are clear and elegant; the Bayesian unification analysis is particularly well executed
- Value: ⭐⭐⭐⭐⭐ Plug-and-play representation upgrade with strong generality; the inverse covariance trick has broad application prospects

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] ProgressiveAvatars: Progressive Animatable 3D Gaussian Avatars](progressiveavatars_progressive_animatable_3d_gaussian_avatars.md)
- [\[CVPR 2026\] Motion-Aware Animatable Gaussian Avatars Deblurring](motion-aware_animatable_gaussian_avatars_deblurring.md)
- [\[CVPR 2026\] 3D Gaussian Splatting with Self-Constrained Priors for High Fidelity Surface Reconstruction](3d_gaussian_splatting_with_self-constrained_priors_for_high_fidelity_surface_rec.md)
- [\[CVPR 2026\] Using Gaussian Splats to Create High-Fidelity Facial Geometry and Texture](using_gaussian_splats_to_create_high-fidelity_facial_geometry_and_texture.md)
- [\[CVPR 2026\] Neural Gabor Splatting: Enhanced Gaussian Splatting with Neural Gabor for High-frequency Surface Reconstruction](neural_gabor_splatting.md)

<!-- RELATED:END -->
