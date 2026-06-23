---
title: >-
  [Paper Note] Carré du champ Flow Matching: 用几何感知噪声改善生成模型的质量-泛化权衡
description: >-
  [ICLR 2026][Image Generation][Flow Matching] This paper proposes **CDC-FM (Carré du champ Flow Matching)**, which replaces the isotropic homogeneous Gaussian noise in standard Flow Matching with **anisotropic, spatially varying noise** determined by the local geometry of the data manifold. This significantly suppresses memorisation and enhances generalisation wit
tags:
  - ICLR 2026
  - Image Generation
  - Flow Matching
  - Carré du champ
date: 2026-05-08
content_hash: 7feef8619ee30dfb
---
# Carré du champ Flow Matching: Improving the Quality-Generalisation Trade-off in Generative Models via Geometry-Aware Noise

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=psmrKQ5lJe](https://openreview.net/forum?id=psmrKQ5lJe)  
**Code**: To be confirmed  
**Area**: Generative Models / Flow Matching  
**Keywords**: Flow Matching, Memorisation, Generalisation, Data Manifold, Anisotropic Noise, Carré du champ  

## TL;DR
This paper proposes **CDC-FM (Carré du champ Flow Matching)**, which replaces the isotropic homogeneous Gaussian noise in standard Flow Matching with **anisotropic, spatially varying noise** determined by the local geometry of the data manifold. This significantly suppresses memorisation and enhances generalisation without sacrificing sample quality, making it particularly suitable for scientific scenarios with sparse data or strong geometric structures.

## Background & Motivation
- **Background**: Continuous Normalising Flows (CNF) and the unified framework Flow Matching (FM) achieve extremely high generation quality on tasks like images, molecules, and weather forecasting by constructing a deterministic probability path from a Gaussian source distribution to the target distribution.
- **Limitations of Prior Work**: High quality often comes at the cost of **memorisation** — the model reproduces training points directly rather than generalising to the underlying data geometry. Geometrically, memorisation manifests as a sudden collapse of the intrinsic dimension of the data manifold, where the learned distribution degenerates into an empirical measure supported on isolated training points. This compromises diversity and novelty while posing data privacy risks.
- **Key Challenge**: FM induces a **fixed-bandwidth, isotropic** Gaussian kernel approximation around each training point as $t\to1$. In practice, $\sigma_{\min}=0$ is often used to maximise precision, causing the probability path to converge directly to the empirical density. The result is a "quality-generalisation frontier": early-stopping models generalise well but have poor quality, while longer training improves quality but increases memorisation. This frontier exists consistently across different datasets and architectures (MLP/UNet/Transformer).
- **Goal**: Find a regularisation approach that can **surpass** this frontier — maintaining quality while stabilising the intrinsic dimension of the manifold and preserving a non-degenerate tangent space.
- **Key Insight**: **Replace isotropic noise with geometric noise aligned with the data manifold**. Swap the covariance of the conditional probability path from $\sigma_t^2 I$ to an anisotropic covariance field $\hat\Gamma(x)$ that captures local geometry. This smooths the noise along the tangent space and tightens it perpendicular to the manifold, thereby suppressing tangential flows that lead to memorisation.

## Method

### Overall Architecture
CDC-FM replaces the affine conditional flow path of FM with a **geometry-aware displacement interpolation**: the starting point is an isotropic Gaussian $\mathcal N(0,I)$, and the endpoint is a Gaussian centred at the training point $x_1$ with an anisotropic covariance $\hat\Gamma(x_1)$. This $\hat\Gamma$ is estimated from data using diffusion maps, approximating the projection matrix of the local tangent space. The training loss follows the original FM regression objective, changing only the probability path, thus allowing direct integration into existing FM pipelines.

```mermaid
flowchart LR
    A[Training Point x1] --> B[Diffusion Geometry Estimation<br/>Local Covariance Γ̂x1]
    B --> C[Anisotropic Conditional Path<br/>ψ_t = t·x1 + Σ_t^1/2·X0]
    C --> D[FM Regression Loss<br/>Learn Velocity Field û_θ]
    D --> E[Sampling: Flow Perpendicular to Manifold<br/>Suppress Memorisation]
```

### Key Designs

**1. Anisotropic Conditional Probability Path: Aligning Noise with Manifold Geometry** — The standard FM conditional path is $\psi_t(X|X_1)=tX_1+\sigma_t X$, inducing the probability path $p_t(x|x_1)=\mathcal N(x|tx_1,\sigma_t^2 I)$, which is a geometry-agnostic isotropic Gaussian. CDC-FM modifies this to:
$$\psi_t^\Gamma(X|X_1)=tX_1+\Sigma_t^\Gamma(X_1)^{1/2}X,\quad \Sigma_t^\Gamma(x)=\big((1-t)I+t\hat\Gamma(x)^{1/2}\big)^2,$$
corresponding to the path $p_t(x|x_1)=\mathcal N(x|tx_1,\Sigma_t^\Gamma(x_1))$, which is the **optimal transport (displacement) interpolation** between $\mathcal N(0,I)$ and "an anisotropic Gaussian centred at $x_1$". Marginalising this over the target distribution yields $\nu\simeq\frac1N\sum_i\mathcal N(x|x^{(i)};\hat\Gamma(x^{(i)}))$ — FM's mixture of Gaussians at isolated points is replaced by an anisotropic mixture spread along the manifold. The authors prove (Appendix B, Thm 1) that simply performing data augmentation $x^{(i)}\mapsto\mathcal N(x^{(i)},\hat\Gamma(x^{(i)}))$ and using the original FM path is **strictly suboptimal**, necessitating the modification of the path itself.

**2. Manifold-Perpendicular Velocity Field Bias: Suppressing Tangential Flows at Source** — Substituting the new path into the FM loss, the target velocity becomes $\frac{d}{dt}\psi_t=x_1+(\hat\Gamma(x_1)^{1/2}-I)X_0$. Its stochastic component is approximately $(\hat\Gamma(x_1)^{1/2}-I)X_0\sim\mathcal N(0,I-\hat\Gamma(x_1))$. When $\hat\Gamma(x_1)$ approximates the projection matrix of the local tangent space, the dominant component of the velocity is **approximately perpendicular to the manifold**, thereby minimising tangential flows strongly correlated with memorisation. In other words, geometric noise is not a post-hoc filter but encodes "do not collapse onto training points" into the learning objective. The authors further prove (Appendix D, Prop 2) that this path introduces a geometry-aware anisotropic diffusion term in the continuity equation, where the amount of smoothing is exactly characterised by the Dirichlet energy (the carré du champ field $\hat\Gamma$), providing the theoretical basis for the name "Carré du champ".

**3. Robust Estimation of $\hat\Gamma$ via Diffusion Geometry Scalable to Large Data** — $\hat\Gamma$ is obtained via local kernel density estimation of the diffusion maps Laplacian: first constructing Markov transition probabilities $P_{ij}$ using a variable-bandwidth Gaussian kernel $w_\epsilon(x^{(i)},x^{(j)})=\exp(-\|x^{(i)}-x^{(j)}\|^2/(\epsilon_i\epsilon_j))$ (where $\epsilon$ is the $k_{bw}$-nearest neighbour distance), then calculating the local covariance $\hat\Gamma(x^{(i)})=\mathbb E_{X\sim P_i}[(X-m^*)(X-m^*)^T]$. The authors prove (Appendix E, Thm 2) that this is the **optimal** Gaussian covariance for a given Markov kernel. In practice, three things are done: scaling down $\hat\Gamma$ to make only a small first-order correction to the FM path, using a rank $d_{cdc}$ approximation (via grid search), and scaling regularisation strength with a global hyperparameter $\gamma$. The overall complexity is $O(N\log N)$ with $O(N)$ memory, and inference speed (NFE) is comparable to or lower than FM.

## Key Experimental Results

### Main Results: Single-cell Gene Expression Trajectories (Earth Mover Distance, average of 5 runs)

| Method | Cite ↓ | Multi ↓ |
|------|--------|---------|
| I-FM | 48.276 ± 3.281 | 57.262 ± 3.855 |
| **Ours (I-CDC-FM)** | **46.657 ± 3.412** | **54.419 ± 0.629** |
| OT-FM | 45.393 ± 0.416 | 54.814 ± 5.858 |
| **Ours (OT-CDC-FM)** | **44.410 ± 0.993** | **52.043 ± 1.948** |

Regardless of whether Optimal Transport pairing (I-/OT-) is used, CDC-FM consistently outperforms FM in reconstruction error across both single-cell benchmarks.

### Latent Space Image Generation: CelebA-HQ (Subset of 1000, Stable Diffusion VAE Latent Space)

| Epoch | FID↓ FM | FID↓ CDC-FM | NLL↓ FM | NLL↓ CDC-FM |
|-------|---------|-------------|---------|-------------|
| 1000 | 15.60 | **12.72** | **6.80** | 7.18 |
| 3000 | 13.56 | **10.55** | 6.80 | **6.68** |
| 4000 | 13.82 | **11.70** | 6.69 | **6.53** |
| 5000 | 13.10 | **10.85** | 6.68 | **6.48** |

At 3k epochs, once both models stabilise, CDC-FM simultaneously improves quality (FID) and generalisation (NLL), indicating that geometric regularisation is also effective for latent space generation.

### Key Findings
- **Surpassing the FM Frontier**: On Drosophila motion capture data (Transformer architecture), FM with varying $\sigma_{\min}$ is always stuck on the quality-generalisation frontier; CDC-FM ($\gamma>0$, optimal at approximately $\gamma=0.3$) **simultaneously** improves quality, generalisation, and memorisation, truly surpassing the frontier.
- **Sparse Regions are Most Affected**: Memorisation occurs mainly in sparse regions of the data manifold (slow, complex movements) and is strongly correlated with distance to the nearest neighbour; CDC-FM shows lower memorisation and is less sensitive to sparsity.
- **Early Stopping is No Longer Critical**: In FM, generalisation monotonically degrades and memorisation monotonically increases with training, necessitating early stopping; in CDC-FM, test performance plateaus and memorisation remains low, allowing training until target quality is reached.
- **Clear Benefits for Geometric Data**: On Mt. Rainier LiDAR point clouds (40–200 points, MLP), FM reconstructions are patchy and fragmented; CDC-FM is smoother and more coherent, with better quality and generalisation.
- **Robust to Dimensionality**: On $T^d$ torus synthetic data, FM memorises almost the entire dataset in high dimensions; CDC-FM memorisation decreases with dimension and remains low, with better generalisation (though higher dimensions require more data to maintain quality).
- **Convergence with Large Data**: On CIFAR-10 small training sets (<10k), FM undergoes a memorisation "phase transition," while only a few percent are memorised with CDC-FM; however, with sufficient training points, the two converge as implicit regularisation from architecture and loss begins to dominate — **the gains of geometric noise are greatest in low-data, heterogeneous, and strongly geometric scenarios.**

## Highlights & Insights
- **Translating "Memorisation" into Geometric Language for Targeted Solution**: The authors seize the observation that "Memorisation ⟺ Intrinsic Dimension Collapse, Tangent Space Degeneracy" and inject manifold-aligned anisotropic noise directly into the probability path. The mechanism is clear and the motivation is provable.
- **Modifying the Path Rather than the Data**: Proving that naive data augmentation is suboptimal highlights the fundamental difference between "aligning geometry at the optimal transport interpolation level" and "post-hoc noise addition."
- **Closed Loop of Theory-Algorithm-Engineering**: From the diffusion term of the continuity equation and Dirichlet energy to optimal covariance estimation via diffusion maps and a $O(N\log N)$ scalable implementation, the chain is complete and ready for plug-and-play in existing FM pipelines.
- **Cross-domain and Cross-architecture Validation**: Synthetic manifolds, point clouds (LiDAR), single-cell genomics, animal motion capture, and images, using MLP/CNN/Transformer architectures — the coverage is exceptionally broad.

## Limitations & Future Work
- **The Curse of Dimensionality is Not Eradicated**: Maintaining quality in high dimensions still requires enough data because kNN graph kernel estimation degrades in high dimensions; the quality of $\hat\Gamma$ is significantly affected by $k$ (number of neighbours).
- **Limited Scenarios for Gains**: On large-scale, uniformly sampled data, CDC-FM converges with FM. The marginal gains of geometric noise vanish as implicit regularisation strengthens — the method's positioning is for "low-data/heterogeneous/strong geometric structure" rather than a universal performance boost.
- **Hyperparameter Sensitivity**: $\gamma, d_{cdc}, k, k_{bw}$ require grid search; a $d_{cdc}$ rank that is too large may cause slight quality drops due to noise leakage in off-manifold directions.
- **Future Work**: Combining with learned manifold methods, latent space diffusion, or adaptive regularisation strength selection could further extend the benefits to higher-dimensional large-scale data scenarios.

## Related Work & Insights
- **Generative Modelling under the Manifold Hypothesis**: Unlike methods constrained to predefined or learned manifolds (Riemannian FM, Kapusniak et al. 2024), this work treats geometry as **regularisation** rather than a **constraint** — adding anisotropic diffusion only in the probability path, making it lighter and more flexible.
- **Quality-Generalisation Trade-off in Diffusion Models**: Echoes research by Yoon et al. (2023), Ross et al. (2025), and Achilli et al. (2024) on memorisation and intrinsic dimensions, transferring these insights from diffusion to FM, which was previously less studied (Bertrand et al. 2025 proved optimal FM vector fields memorise).
- **Geometric Regularisation**: Compared to earlier uses of tangent information (Simard 1991, Rifai 2011) to regularise losses in supervised learning, CDC-FM brings geometric regularisation to generative modelling and applies it to the probability path rather than the loss.
- **Insights**: Integrating "Data Geometry $\leftrightarrow$ Generalisation/Memorisation" into a unified mathematical framework suggests that local geometric noise can serve as a universal, plug-and-play regularisation tool for generative models, especially in low-sample AI for Science scenarios.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Introducing carré du champ / diffusion geometry into FM probability paths and designing anisotropic noise based on the geometric root cause of memorisation is a novel and systematic approach.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Five types of data × three architectures, including synthetic controlled experiments and real scientific/image data. Solid ablations ($\gamma, d_{cdc}, k$, dimension, data volume); however, gains on large-scale images are limited, and comparison with more SOTA regularisation baselines is lacking.
- **Writing Quality**: ⭐⭐⭐⭐ — Progressively builds motivation; good coordination between text and figures; clear correspondence between theoretical propositions and algorithms. High formula density may pose a barrier for readers without a geometry background.
- **Value**: ⭐⭐⭐⭐ — Plug-and-play and scalable, with high practical value for low-sample/heterogeneous data scenarios in AI for Science. Provides a reusable geometric framework for memorisation research in generative models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Delay Flow Matching](delay_flow_matching.md)
- [\[ICLR 2026\] Flow Matching with Semidiscrete Couplings](flow_matching_with_semidiscrete_couplings.md)
- [\[ICLR 2026\] Edit-Based Flow Matching for Temporal Point Processes](edit-based_flow_matching_for_temporal_point_processes.md)
- [\[ICML 2026\] A Kinetic Energy Perspective of Flow Matching](../../ICML2026/image_generation/a_kinetic_energy_perspective_of_flow_matching.md)
- [\[ICLR 2026\] DenseGRPO: From Sparse to Dense Reward for Flow Matching Model Alignment](densegrpo_from_sparse_to_dense_reward_for_flow_matching_model_alignment.md)

</div>

<!-- RELATED:END -->
