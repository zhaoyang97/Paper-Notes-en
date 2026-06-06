---
title: >-
  [Paper Note] Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves
description: >-
  [ICML 2026][Graph Learning][Neural Layer Diffusion] PolyNSD replaces the "one-step spatial diffusion" in Sheaf Neural Networks with a learnable $K$-order polynomial spectral filter on the normalized sheaf Laplacian…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Neural Layer Diffusion"
  - "Chebyshev Polynomial Filtering"
  - "Heterogeneous Graphs"
  - "Oversmoothing"
  - "Diagonal Restriction Maps"
date: 2026-05-08
content_hash: 52f125dda818d7cc
---

# Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves

**Conference**: ICML 2026  
**arXiv**: [2512.00242](https://arxiv.org/abs/2512.00242)  
**Code**: None  
**Area**: Graph Neural Networks / Sheaf Neural Networks / Spectral Graph Filtering / Heterogeneous Graphs  
**Keywords**: Neural Layer Diffusion, Chebyshev Polynomial Filtering, Heterogeneous Graphs, Oversmoothing, Diagonal Restriction Maps

## TL;DR
PolyNSD replaces the "one-step spatial diffusion" in Sheaf Neural Networks with a learnable $K$-order polynomial spectral filter on the normalized sheaf Laplacian, computed stably via Chebyshev three-term recurrence. A single layer thus achieves $K$-hop receptive field and controllable low/band/high-pass response. An unexpected finding is that using only diagonal restriction maps outperforms all existing NSDs requiring dense high-dimensional stalks, with significant reductions in parameters, memory, and runtime.

## Background & Motivation
**Background**: GNNs have achieved great success on homogeneous graphs, but face challenges on heterogeneous graphs (where adjacent nodes have different types) and with deep stacking (oversmoothing). One remedy is cellular sheaf theory: assign each node a local feature space (stalk), each edge a linear restriction map, and construct a sheaf Laplacian for "transport-aware" diffusion, which is more suitable for heterogeneity than isotropic message passing. Bodnar et al. (2022) introduced Neural Sheaf Diffusion (NSD), learning the sheaf Laplacian and achieving SOTA.

**Limitations of Prior Work**: NSD essentially performs one-step spatial propagation $X^{(t+1)}=X^{(t)}-\sigma(\Delta_{\mathcal{F}(t)}(I_{nd}\otimes W_1^{(t)})X^{(t)}W_2^{(t)})$, with four structural issues: (i) Each layer only propagates one step, so long-range dependencies require deep stacking, exacerbating oversmoothing; (ii) Relies on dense per-edge restriction maps (diagonal/bundle/general), with parameters and memory dominated by stalk dimension $d$; (iii) Training requires normalization or decomposition such as SVD, leading to numerical instability; (iv) Model performance is highly dependent on large stalk dimension, making it impossible to decouple "accuracy" from "computational cost".

**Key Challenge**: The "transport-aware + partial order" advantage of the sheaf Laplacian is spatially local, but multi-hop interactions require repeated stacking, which triggers oversmoothing and expensive sheaf relearning. In other words, **long-range propagation is tied to layer depth**.

**Goal**: (1) Enable single-layer $K$-hop mixing while retaining sheaf transport expressiveness; (2) Make the diffusion frequency response learnable instead of "implicitly low-pass"; (3) Decouple performance from stalk dimension, making diagonal restriction sufficient; (4) Provide a stable polynomial implementation on par with standard GNNs.

**Key Insight**: Inspired by classical spectral polynomial filters like ChebNet/GPRGNN—since Chebyshev recurrence on the graph Laplacian is proven stable and efficient, and the sheaf Laplacian is a symmetric positive semi-definite matrix, the same spectral functional analysis ($p(L)=Up(\Lambda)U^\top$) theoretically applies.

**Core Idea**: Replace single-step $(aI+bL)$ diffusion with $p(L)=\sum_{k=0}^K c_k L^k$, and use spectral rescaling $\widetilde{L}=2L/\lambda_\text{max}-I$ to ensure Chebyshev basis stability and boundedness, upgrading sheaf diffusion to a controllable spectral filtering layer in one shot.

## Method

### Overall Architecture
Input: Graph $G=(V,E)$ + raw node features $x_v^\text{raw}\in\mathbb{R}^F$. The pipeline consists of 10 steps (corresponding to Fig.1 in the paper): (1-3) Each node is lifted to a $d$-dimensional stalk, yielding $x_v\in\mathbb{R}^{d\times F}$, stacked into a 0-cochain $x\in\mathbb{R}^{(Nd)\times F}$; (4) The sheaf learner $\Psi$ uses an MLP to predict edge-wise restriction maps (diagonal/bundle/general) from neighboring node feature pairs, constructing the sheaf Laplacian $L\in\mathbb{R}^{Nd\times Nd}$ (unnormalized $L_\mathcal{F}$ or normalized $\Delta_\mathcal{F}$); (5-6) After estimating $\lambda_\text{max}$, perform affine rescaling $\widetilde{L}=2L/\lambda_\text{max}-I$ to compress the spectrum to $[-1,1]$; (7) Use Chebyshev three-term recurrence $T_0=I,T_1=\widetilde{L},T_{k+1}=2\widetilde{L}T_k-T_{k-1}$ to compute the polynomial response $p_\theta(\widetilde{L})=\sum_{k=0}^K \alpha_k T_k(\widetilde{L})$, where $\alpha=\text{softmax}(\eta)$ are convex mixing coefficients; (8) Add high-pass correction $h_\text{hp}=x-\lambda_\text{max}^{-1}Lx$, $z=p_\theta(\widetilde{L})x+\alpha_\text{hp}h_\text{hp}$; (9-10) Gated residual $x^+=(I+\tanh\varepsilon)x-\phi(z)$ outputs the result. The entire structure is compatible with existing SheafNNs and is a model-agnostic operator.

### Key Designs

1. **Chebyshev Polynomial Spectral Filtering on Sheaf Laplacian**:

    - **Function**: Replaces single-step spatial diffusion with a single-layer $K$-hop spectral filter, with learnable frequency response.
    - **Mechanism**: $L$ is symmetric PSD, $L=U\Lambda U^\top$. The polynomial $p(L)=\sum_{k=0}^K c_k L^k$ is spectrally equivalent to $Up(\Lambda)U^\top$, i.e., $p$ acts as a multiplier $p(\lambda_i)$ on the $i$-th sheaf Fourier mode. Monotonically decreasing $p$ yields low-pass; banded $p$ yields band-pass; increasing $p$ yields high-pass, all learned via $\alpha_k$. Directly learning monomials $\{c_k\}$ is numerically unstable (Vandermonde ill-conditioning), so Chebyshev basis $T_k(\xi)=\cos(k\arccos\xi)$ is used on $\widetilde{L}=2L/\lambda_\text{max}-I\in[-1,1]$; $|T_k(\xi)|\le 1$ ensures the convex mixture $\alpha=\text{softmax}(\eta)$ yields a non-expansive operator $\|p_\theta(\widetilde{L})\|_2\le 1$. For normalized $\Delta_\mathcal{F}$, set $\lambda_\text{max}=2$; for unnormalized $L_\mathcal{F}$, estimate via Gershgorin bound or short power iteration.
    - **Design Motivation**: Bodnar's $X^{(t+1)}=X^{(t)}-\sigma(\Delta_{\mathcal{F}}(\dots))$ is equivalent to $K=1$ case, i.e., $aI+bL$, propagating only one hop per layer. This design raises $K$ to arbitrary order. **Proposition 1** proves $(p(L))_{vu}=0$ when $\text{dist}_G(v,u)>K$, i.e., $K$-hop locality holds strictly in space—no need to stack $K$ layers or relearn the sheaf repeatedly.

2. **Spectral Rescaling + Convex Mixture for Stability (Dirichlet Energy Monotonicity)**:

    - **Function**: Ensures the polynomial filter is numerically stable and does not amplify disagreement (prevents training divergence).
    - **Mechanism**: Chebyshev polynomials grow exponentially for $|\xi|>1$, so the spectrum **must** be compressed to $[-1,1]$. **Proposition 2**: If $0\le p(\lambda)\le 1$ holds on $\sigma(L)$, then Dirichlet energy $\langle p(L)x,Lp(L)x\rangle=\sum\lambda_i p(\lambda_i)^2\hat x_i^2\le\sum\lambda_i\hat x_i^2=\langle x,Lx\rangle$, i.e., $p(L)$ can only dampen disagreement, not amplify it. Polynomial filters commute: $p(L)q(L)=(pq)(L)$, so stacking layers is equivalent to learning higher-order polynomials.
    - **Design Motivation**: Makes "frequency selectivity" a first-class design: users can directly control whether the layer is low-pass (smoothing), band-pass (extracting mid-frequencies), or high-pass (preserving disagreement) via $\alpha$; ensures stable training even for large $K$.

3. **High-pass Skip + Gated Residual + Diagonal Restriction Maps**:

    - **Function**: (a) Mitigates the inherent low-pass bias of diffusion; (b) Maintains gradients in deep layers; (c) Decouples stalk dimension from "must be large" to "diagonal suffices".
    - **Mechanism**: High-pass component $h_\text{hp}=x-\lambda_\text{max}^{-1}Lx$ is linearly combined with the polynomial output as $z=p_\theta(\widetilde{L})x+\alpha_\text{hp}h_\text{hp}$; since $\widetilde{L}$ and $L$ share eigenbasis, the entire linear mapping is diagonal in the eigenbasis, with multiplier $m(\lambda)=p_\theta(2\lambda/\lambda_\text{max}-1)+\alpha_\text{hp}(1-\lambda/\lambda_\text{max})$, so a single scalar $\alpha_\text{hp}$ analytically shapes the frequency response. Gated residual $x^+=(I+\tanh\varepsilon)x-\phi(z)$ uses a 1-Lipschitz nonlinearity $\phi$, ensuring the overall Lipschitz constant has an explicit upper bound. The restriction map uses the cheapest diagonal form, with only $K+1$ scalar parameters plus the sheaf learner, and per-layer cost $\mathcal{O}(K\cdot\text{nnz}(L)\cdot C)$.
    - **Design Motivation**: Traditional NSD relies on large stalk dimensions to compensate for expressiveness loss; here, expressiveness is shifted to "learnable frequency response", enabling diagonal restriction (the cheapest sheaf form) to match bundle/general. The paper constructs **PolySpectralGNN**—a version without sheaf transport, with stalk=1 and identity transport—as ablation, showing that the extra benefit of sheaf is not from polynomial filtering alone.

### Loss & Training
The task is node classification (standard supervised CE); the focus is on the propagation operator rather than the loss. Key training points: (a) $\lambda_\text{max}$ is estimated via power iteration, avoiding per-step eigendecomposition; (b) Convex mixture $\alpha=\text{softmax}(\eta)$ ensures $\|p\|_\infty\le 1$; (c) After spectral rescaling, the whole layer operator is Lipschitz $\le 1$, allowing deeper stacking without gradient explosion; (d) Diagonal restriction is the default, bundle/general are for comparison only.

## Key Experimental Results

### Main Results
Nine node classification datasets (homogeneous: Cora/Citeseer/Pubmed; heterogeneous: Texas/Wisconsin/Film/Squirrel/Chameleon/Cornell). Top three highlighted by color. The most heterogeneous (homophily 0.11) Texas and most homogeneous (0.81) Cora are excerpted for comparison:

| Model | Texas (hetero 0.11) | Wisconsin | Squirrel | Chameleon | Cora (homo 0.81) | Citeseer | Pubmed |
|------|---------------------|-----------|----------|-----------|------------------|----------|--------|
| **DiagPolySD (Ours)** | **90.00±4.68** | 88.63±3.59 | **56.61±2.06** | **71.45±2.03** | 88.79±1.13 | 77.74±1.26 | 89.70±0.32 |
| BundlePolySD | 89.74±5.32 | 89.41±4.04 | 55.76±2.02 | 71.18±1.46 | 88.33±1.34 | 77.57±1.55 | **89.75±0.34** |
| Diag-NSD (Bodnar) | 85.67±6.95 | 88.63±2.75 | 54.78±1.81 | 68.68±1.73 | 87.14±1.06 | 77.14±1.85 | 89.42±0.43 |
| Gen-NSD (Bodnar) | 82.97±5.13 | 89.21±3.84 | 53.17±1.31 | 67.93±1.58 | 87.30±1.15 | 76.32±1.65 | 89.33±0.35 |
| GGCN | 84.86±4.55 | 86.86±3.29 | 55.17±1.58 | 71.14±1.84 | 87.95±1.05 | 77.14±1.45 | 89.15±0.37 |
| H2GCN | 84.86±7.23 | 87.65±4.98 | 36.48±1.86 | 60.11±2.15 | 87.87±1.20 | 77.11±1.57 | 89.49±0.38 |
| GPRGNN | 78.38±4.36 | 82.94±4.21 | 31.61±1.24 | 46.58±1.71 | 87.95±1.18 | 77.13±1.67 | 87.54±0.38 |
| GCN | 55.14±5.16 | 51.76±3.06 | 53.43±2.01 | 64.82±2.24 | 86.98±1.27 | 76.50±1.36 | 88.42±0.50 |

DiagPolySD improves over Diag-NSD by 4.3 points on the most heterogeneous Texas, and by 35 points over GCN; achieves SOTA on Squirrel/Chameleon (56.6 / 71.5); and matches or slightly outperforms on homogeneous data.

### Ablation Study

| Configuration | Key Change | Observation |
|---------------|-----------|-------------|
| Full DiagPolySD ($K>1$) | Complete model | Top-3 on all 9 datasets |
| $K=1$ (degenerates to NSD) | Polynomial order reduced to 1 | Equivalent to $aI+bL$, reverts to Bodnar's single-layer diffusion |
| No spectral rescaling | Directly learn $\{c_k\}$ monomials | Vandermonde ill-conditioning, training diverges |
| Remove high-pass skip | Only polynomial response | Stronger low-pass bias, performance drops on heterogeneous data |
| Remove gated residual | Use standard residual | Unstable gradients in deep layers |
| PolySpectralGNN | stalk=1, identity transport | Significant drop on heterogeneous data (Texas 64.6 vs 90.0), proving sheaf transport is irreplaceable for heterogeneity |
| Bundle/General restriction | Use dense restriction | Similar to diagonal, performance no longer depends on large stalk |

### Key Findings
- **Diagonal restriction suffices**: Traditional NSD requires bundle/general to squeeze out performance, with parameters and memory dominated by stalk dimension; PolyNSD matches or exceeds with diagonal restriction—shifting "expressiveness" from restriction dimension to polynomial frequency response, a paradigm-level decoupling.
- **$K>1$ consistently brings gains**: Raising polynomial order from 1 to higher improves all datasets, validating that "single-layer $K$-hop" is both faster and more accurate than stacking $K$ NSD layers.
- **Sheaf is irreplaceable for heterogeneity**: PolySpectralGNN (removing sheaf, keeping only Chebyshev filtering) lags by 20+ points on highly heterogeneous data like Texas/Wisconsin/Cornell, indicating transport-awareness is key for heterogeneity, with polynomial filtering amplifying this effect.
- **Spectral rescaling is essential for stability**: Removing $\widetilde{L}=2L/\lambda_\text{max}-I$ leads to training divergence; convex mixture via $\text{softmax}$ is crucial for enforcing $\|p\|_2\le 1$ at the code level.

## Highlights & Insights
- **Transplanting ChebNet's "old trick" to the sheaf Laplacian** is seemingly obvious but nontrivial in practice: the sheaf Laplacian is $Nd\times Nd$, requiring careful handling of $\lambda_\text{max}$ estimation, three-term recurrence, and spectrum compression to $[-1,1]$; the paper implements all these details robustly.
- **Paradigm overturn: performance ≠ large stalk**: The community previously assumed NSD needed large stalk + general restriction for top performance; PolyNSD breaks this consensus with diagonal restriction + polynomial filtering, democratizing the sheaf framework much like ChebNet did for spectral graph convolution.
- **Interpretable frequency response**: The learned $p_\theta(\lambda)$ can be directly visualized—multi-band or even high-pass responses on heterogeneous data, low-pass on homogeneous—providing spectral interpretability for sheaf models. The paper also analyzes long-range oversquashing attenuation.
- **Transferable design template**: (1) The "sparse operator + spectral rescaling + Chebyshev recurrence" trio applies to any symmetric PSD operator (GNN/implicit neural networks/PDE solvers); (2) The "high-pass skip + gated residual" combination can mitigate low-pass bias in any diffusion model.

## Limitations & Future Work
- **Reliance on accurate $\lambda_\text{max}$ estimation**: For normalized $\Delta_\mathcal{F}$, $\lambda_\text{max}=2$ is safe, but for unnormalized $L_\mathcal{F}$, power iteration is used, whose iteration count depends on the graph spectrum and may be unstable for extreme cases.
- **Chebyshev is the default but not the only basis**: The paper claims basis-agnostic implementation but only systematically validates Chebyshev; whether other orthogonal bases (Jacobi, Bernstein) perform better on different data remains to be explored.
- **$K$ is a hyperparameter, no adaptive mechanism**: Ideally, $K$ should be chosen automatically based on graph diameter or task; currently, it is manually tuned.
- **Sheaf learner is still an MLP, with potential instability**: More complex structures like multi-edge consistency are not specially handled.
- **Engineering for large-scale graphs**: $L$ is $Nd\times Nd$, and $K$-order sparse multiplication remains challenging for very large graphs; the paper does not validate on OGB-scale graphs.

## Related Work & Insights
- **vs Bodnar et al. (NSD 2022)**: NSD is the $K=1$ special case; PolyNSD generalizes to arbitrary order with stability proofs, breaking the "performance depends on large stalk" assumption.
- **vs ChebNet (Defferrard 2016)**: Essentially the same spectral polynomial filtering technique, but transplanted to the sheaf Laplacian, gaining both "transport-awareness" and "frequency controllability".
- **vs GPRGNN / FAGCN / H2GCN**: These are SOTA polynomial or frequency response baselines for heterogeneous graphs, but use the graph Laplacian rather than the sheaf Laplacian; this work outperforms them by 20+ points on Texas/Wisconsin/Squirrel, quantifying the gain from sheaf transport.
- **vs PolySpectralGNN (own ablation)**: Removing the sheaf and keeping only spectral polynomial filtering leads to significant performance drop, elegantly proving via ablation that "sheaf is not just parameter overhead".
- **Insights**: Applying the "any spatial diffusion model → spectral polynomial version" idea to Hodge Laplacian, normalized attention matrix, or graph wavelet operator may yield immediate benefits.

## Rating
- Novelty: ⭐⭐⭐⭐ Not inventing a new operator from scratch, but rigorously transplanting ChebNet's mature spectral filtering framework to the sheaf Laplacian, and discovering the counter-consensus result that "diagonal restriction suffices"; both the transplantation and decoupling are insightful and substantial.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 datasets (homogeneous + 6 heterogeneous), 4 restriction types, full comparison with 14+ baselines; ablations dissect sheaf vs. polynomial, $K$, spectral rescaling, high-pass, and also cover oversmoothing/oversquashing/continuous ODE extensions.
- Writing Quality: ⭐⭐⭐⭐ Formula-dense but well-structured, with Proposition 1/2 providing theoretical guarantees for spatial locality and energy monotonicity; Fig. 1 clearly illustrates the 10-step pipeline.
- Value: ⭐⭐⭐⭐ Brings sheaf neural networks from "expensive niche" to "affordable and practical", and opens the research path of "designing spectral filters on the sheaf Laplacian", with promising community impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](../../AAAI2026/graph_learning/sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [\[ICML 2026\] Information-Geometric Adaptive Sampling for Graph Diffusion](information-geometric_adaptive_sampling_for_graph_diffusion.md)
- [\[ICLR 2026\] Cooperative Sheaf Neural Networks](../../ICLR2026/graph_learning/cooperative_sheaf_neural_networks.md)
- [\[ICML 2026\] Quantile-Free Uncertainty Quantification in Graph Neural Networks](quantile-free_uncertainty_quantification_in_graph_neural_networks.md)
- [\[ICML 2026\] Full-Spectrum Graph Neural Network: Expressive and Scalable](full-spectrum_graph_neural_network_expressive_and_scalable.md)

</div>

<!-- RELATED:END -->
