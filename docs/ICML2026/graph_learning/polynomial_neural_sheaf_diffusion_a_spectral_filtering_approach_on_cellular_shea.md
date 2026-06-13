---
title: >-
  [Paper Note] Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves
description: >-
  [ICML 2026][Graph Learning][Neural Sheaf Diffusion] PolyNSD replaces the "one-step spatial diffusion" of Sheaf Neural Networks with a learnable $K$-th order polynomial spectral filter on the normalized sheaf Laplacian. B…
tags:
  - "ICML 2026"
  - "Graph Learning"
  - "Neural Sheaf Diffusion"
  - "Chebyshev Polynomial Filtering"
  - "Heterophilous Graphs"
  - "Oversmoothing"
  - "Diagonal Transport Maps"
date: 2026-05-08
content_hash: 4ca5146b306854ee
---

# Polynomial Neural Sheaf Diffusion: A Spectral Filtering Approach on Cellular Sheaves

**Conference**: ICML 2026  
**arXiv**: [2512.00242](https://arxiv.org/abs/2512.00242)  
**Code**: None  
**Area**: Graph Neural Networks / Sheaf Neural Networks / Spectral Graph Filtering / Heterophilous Graphs  
**Keywords**: Neural Sheaf Diffusion, Chebyshev Polynomial Filtering, Heterophilous Graphs, Oversmoothing, Diagonal Transport Maps

## TL;DR
PolyNSD replaces the "one-step spatial diffusion" of Sheaf Neural Networks with a learnable $K$-th order polynomial spectral filter on the normalized sheaf Laplacian. By utilizing Chebyshev three-term recurrence for stable computation, a single layer achieves a $K$-hop receptive field and controllable low/band/high-pass responses. Paradoxically, the study finds that using only diagonal restriction maps outperforms existing NSD models that require dense, high-dimensional stalks, leading to significant reductions in parameters, memory, and runtime.

## Background & Motivation
**Background**: GNNs are successful on homophilous graphs but struggle with heterophilous graphs (where adjacent nodes belong to different categories) and deep stacking (oversmoothing). Cellular sheaf theory offers a remedy: by assigning a local feature space (stalk) to each node and a linear restriction map to each edge, the constructed sheaf Laplacian enables "transport-aware" diffusion, which is more suitable for heterogeneity than isotropic message passing. Bodnar et al. (2022) achieved SOTA with Neural Sheaf Diffusion (NSD) by learning the sheaf Laplacian.

**Limitations of Prior Work**: NSD is essentially one-step spatial propagation $X^{(t+1)}=X^{(t)}-\sigma(\Delta_{\mathcal{F}(t)}(I_{nd}\otimes W_1^{(t)})X^{(t)}W_2^{(t)})$, which faces four structural issues: (i) each layer only performs a one-hop expansion, requiring deep stacks for long-range dependencies, which exacerbates oversmoothing; (ii) it relies on dense per-edge restriction maps (diagonal/bundle/general), causing parameters and memory to be bottlenecked by the stalk dimension $d$; (iii) it requires numerical normalization or decomposition (e.g., SVD) during training, which is unstable; (iv) performance is highly dependent on large stalk dimensions, failing to decouple "accuracy" from "computational cost."

**Key Challenge**: The advantages of the sheaf Laplacian, such as "transport-awareness" and "partial order," are spatially local. Utilizing these for multi-hop interactions necessitates repeated layer stacking, which triggers oversmoothing and expensive re-learning of the sheaf. In other words, **long-range propagation is tied to layer depth**.

**Goal**: (1) Retain the expressivity of sheaf transport while allowing a single layer to perform $K$-hop mixing; (2) Convert the frequency response of the diffusion from "implicitly low-pass" to learnable; (3) Decouple performance from stalk dimension so that diagonal restrictions suffice; (4) Provide a stable polynomial implementation of the same order as standard GNNs.

**Key Insight**: The authors draw inspiration from classical spectral polynomial filters like ChebNet and GPRGNN. Since Chebyshev recurrence is proven stable and efficient on graph Laplacians, and the sheaf Laplacian is a symmetric positive semi-definite (PSD) matrix, the same spectral functional analysis ($p(L)=Up(\Lambda)U^\top$) is theoretically applicable.

**Core Idea**: Replace one-step $(aI+bL)$ diffusion with $p(L)=\sum_{k=0}^K c_k L^k$, and use spectral rescaling $\widetilde{L}=2L/\lambda_\text{max}-I$ to keep the Chebyshev bases stable and bounded, effectively upgrading sheaf diffusion into a controllable spectral filtering layer.

## Method

### Overall Architecture
Input: Graph $G=(V,E)$ + raw node features $x_v^\text{raw}\in\mathbb{R}^F$. The pipeline consists of 10 steps (corresponding to Fig.1 in the paper): (1-3) Lift each node to a $d$-dimensional stalk to obtain $x_v\in\mathbb{R}^{d\times F}$ and stack them into a 0-cochain $x\in\mathbb{R}^{(Nd)\times F}$; (4) The sheaf learner $\Psi$ uses an MLP to predict edge-wise restriction maps (diagonal/bundle/general) from adjacent node feature pairs, constructing the sheaf Laplacian $L\in\mathbb{R}^{(Nd)\times (Nd)}$ (unnormalized $L_\mathcal{F}$ or normalized $\Delta_\mathcal{F}$); (5-6) Estimate $\lambda_\text{max}$ and perform affine scaling $\widetilde{L}=2L/\lambda_\text{max}-I$ to compress the spectrum into $[-1,1]$; (7) Use Chebyshev three-term recurrence $T_0=I, T_1=\widetilde{L}, T_{k+1}=2\widetilde{L}T_k-T_{k-1}$ to compute the polynomial response $p_\theta(\widetilde{L})=\sum_{k=0}^K \alpha_k T_k(\widetilde{L})$, where $\alpha=\text{softmax}(\eta)$ are convex mixing coefficients; (8) Add a high-pass correction $h_\text{hp}=x-\lambda_\text{max}^{-1}Lx$, setting $z=p_\theta(\widetilde{L})x+\alpha_\text{hp}h_\text{hp}$; (9-10) Output via a gated residual $x^+=(I+\tanh\varepsilon)x-\phi(z)$. This entire structure is compatible with existing SheafNNs and serves as a model-agnostic operator.

### Key Designs

1.  **Chebyshev Polynomial Spectral Filtering on Sheaf Laplacian**:
    -   **Function**: Replaces one-step spatial diffusion with a single-layer $K$-hop spectral filter with learnable frequency response.
    -   **Mechanism**: $L$ is a symmetric PSD matrix, $L=U\Lambda U^\top$. The polynomial $p(L)=\sum_{k=0}^K c_k L^k$ is equivalent to $Up(\Lambda)U^\top$ via spectral functionals, meaning the multiplier for the $i$-th sheaf Fourier mode is $p(\lambda_i)$. The response is low-pass if $p$ is monotonically decreasing, band-pass if it is shaped like a band, and high-pass if it is increasing—all learned via $\alpha_k$. Directly learning monomials $\{c_k\}$ is numerically unstable (Vandermonde ill-conditioning), so the first-kind Chebyshev $T_k(\xi)=\cos(k\arccos\xi)$ is used as a basis on $\widetilde{L}=2L/\lambda_\text{max}-I \in [-1,1]$. $|T_k(\xi)|\le 1$ ensures that convex mixing $\alpha=\text{softmax}(\eta)$ automatically yields a non-expansive operator with $\|p_\theta(\widetilde{L})\|_2\le 1$. For normalized $\Delta_\mathcal{F}$, $\lambda_\text{max}=2$; for unnormalized $L_\mathcal{F}$, $\lambda_\text{max}$ is estimated using Gershgorin bounds or power iteration.
    -   **Design Motivation**: Bodnar’s $X^{(t+1)}=X^{(t)}-\sigma(\Delta_{\mathcal{F}}(\dots))$ is equivalent to $aI+bL$ with $K=1$, meaning a single layer only performs a one-hop expansion. This design increases $K$ to any order. **Proposition 1** proves that $(p(L))_{vu}=0$ when $\text{dist}_G(v,u)>K$, establishing that $K$-hop locality holds strictly in space—without needing to stack $K$ layers or repeatedly learn the sheaf.

2.  **Spectral Rescaling + Convex Mixing for Stability (Monotonically Non-increasing Dirichlet Energy)**:
    -   **Function**: Ensures the polynomial filter is numerically stable and does not amplify disagreement (avoiding training divergence).
    -   **Mechanism**: Chebyshev polynomials grow exponentially when $|\xi|>1$, making spectral compression to $[-1,1]$ **mandatory**. **Proposition 2** states: if $0\le p(\lambda)\le 1$ holds on $\sigma(L)$, then the Dirichlet energy $\langle p(L)x,Lp(L)x\rangle=\sum\lambda_i p(\lambda_i)^2\hat x_i^2\le\sum\lambda_i\hat x_i^2=\langle x,Lx\rangle$, meaning $p(L)$ can only damp disagreement and cannot amplify it. Additionally, polynomial filters commute ($p(L)q(L)=(pq)(L)$), making multi-layer stacking equivalent to learning higher-order polynomials.
    -   **Design Motivation**: Makes "frequency selectivity" a first-class design: users can directly control via $\alpha$ whether a layer is low-pass (smoothing), band-pass (extracting mid-frequencies), or high-pass (retaining disagreement), while ensuring training stability even for large $K$.

3.  **High-pass Skip + Gated Residual + Diagonal Restriction Maps**:
    -   **Function**: (a) Alleviate the inherent low-pass bias of diffusion; (b) maintain gradients in deep networks; (c) decouple the stalk dimension from being "necessarily large" to "diagonal is sufficient."
    -   **Mechanism**: The high-pass component $h_\text{hp}=x-\lambda_\text{max}^{-1}Lx$ is linearly combined with the polynomial output as $z=p_\theta(\widetilde{L})x+\alpha_\text{hp}h_\text{hp}$. Since $\widetilde{L}$ and $L$ share the same eigenbasis, this linear mapping is diagonal on the basis with multipliers $m(\lambda)=p_\theta(2\lambda/\lambda_\text{max}-1)+\alpha_\text{hp}(1-\lambda/\lambda_\text{max})$, allowing a single scalar $\alpha_\text{hp}$ to analytically adjust the frequency response. A gated residual $x^+=(I+\tanh\varepsilon)x-\phi(z)$ with a 1-Lipschitz non-linearity $\phi$ ensures an explicit upper bound on the global Lipschitz constant. Using diagonal restriction maps (the cheapest form) results in parameters of only $K+1$ scalars plus the sheaf learner, with a single-layer cost of $\mathcal{O}(K\cdot\text{nnz}(L)\cdot C)$.
    -   **Design Motivation**: Traditional NSD must rely on large stalk dimensions to compensate for expressivity loss. This approach shifts expressivity to "learnable frequency response," allowing diagonal restrictions to rival bundle/general ones. The paper constructs **PolySpectralGNN**—a version without sheaf transport, where stalk=1 and transport=identity—as an ablation to prove that the sheaf's contribution is independent of the polynomial filtering itself.

### Loss & Training
The task is node classification (standard supervised CE). The focus is on the propagation operator rather than the loss. Key training points: (a) $\lambda_\text{max}$ is estimated with power iteration to avoid full eigendecomposition at each step; (b) convex mixing $\alpha=\text{softmax}(\eta)$ ensures $\|p\|_\infty\le 1$; (c) spectral rescaling ensures a layer Lipschitz constant $\le 1$, allowing depth without gradient explosion; (d) diagonal restriction is the default, with bundle/general used only for comparison.

## Key Experimental Results

### Main Results
Evaluation on 9 node classification datasets (homophilous: Cora/Citeseer/Pubmed; heterophilous: Texas/Wisconsin/Film/Squirrel/Chameleon/Cornell). The following table extracts results for the most heterophilous (Texas, homophily 0.11) and most homophilous (Cora, 0.81) datasets:

| Model | Texas (Hetero 0.11) | Wisconsin | Squirrel | Chameleon | Cora (Homo 0.81) | Citeseer | Pubmed |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **DiagPolySD (Ours)** | **90.00±4.68** | 88.63±3.59 | **56.61±2.06** | **71.45±2.03** | 88.79±1.13 | 77.74±1.26 | 89.70±0.32 |
| BundlePolySD | 89.74±5.32 | 89.41±4.04 | 55.76±2.02 | 71.18±1.46 | 88.33±1.34 | 77.57±1.55 | **89.75±0.34** |
| Diag-NSD (Bodnar) | 85.67±6.95 | 88.63±2.75 | 54.78±1.81 | 68.68±1.73 | 87.14±1.06 | 77.14±1.85 | 89.42±0.43 |
| Gen-NSD (Bodnar) | 82.97±5.13 | 89.21±3.84 | 53.17±1.31 | 67.93±1.58 | 87.30±1.15 | 76.32±1.65 | 89.33±0.35 |
| GGCN | 84.86±4.55 | 86.86±3.29 | 55.17±1.58 | 71.14±1.84 | 87.95±1.05 | 77.14±1.45 | 89.15±0.37 |
| H2GCN | 84.86±7.23 | 87.65±4.98 | 36.48±1.86 | 60.11±2.15 | 87.87±1.20 | 77.11±1.57 | 89.49±0.38 |
| GPRGNN | 78.38±4.36 | 82.94±4.21 | 31.61±1.24 | 46.58±1.71 | 87.95±1.18 | 77.13±1.67 | 87.54±0.38 |
| GCN | 55.14±5.16 | 51.76±3.06 | 53.43±2.01 | 64.82±2.24 | 86.98±1.27 | 76.50±1.36 | 88.42±0.50 |

DiagPolySD improves over Diag-NSD by 4.3 points and over GCN by 35 points on the highly heterophilous Texas dataset. It pushes SOTA on Squirrel/Chameleon to 56.6/71.5 and remains competitive or slightly superior on homophilous data.

### Ablation Study

| Configuration | Change | Observation |
| :--- | :--- | :--- |
| Full DiagPolySD ($K>1$) | Complete model | Top-3 on all 9 datasets |
| $K=1$ (degrades to NSD) | Polynomial order becomes 1 | Equivalent to $aI+bL$, reverts to Bodnar's single-layer diffusion |
| No spectral rescaling | Directly learn monomial $\{c_k\}$ | Vandermonde ill-conditioning, training diverges |
| No high-pass skip | Polynomial response only | Stronger low-pass bias, performance drops on hetero data |
| No gated residual | Standard residual | Deep gradients become unstable |
| PolySpectralGNN | stalk=1, identity transport | Significant degradation on hetero data (Texas 64.6 vs 90.0), proving sheaf transport is indispensable for heterogeneity |
| Bundle/General restriction | Dense restriction | Performance similar to diagonal; accuracy no longer depends on stalk size |

### Key Findings
-   **Diagonal restriction is sufficient**: Traditional NSD requires bundle/general restrictions for performance, which ties complexity to stalk dimensions. PolyNSD matches or exceeds this performance using only diagonal restrictions by shifting expressivity to the polynomial frequency response—a paradigm-shifting decoupling.
-   **$K>1$ brings consistent gains**: Increasing the polynomial order beyond 1 improves results across all datasets, validating that "single-layer $K$-hop" is both faster and more accurate than "stacking $K$ NSD layers."
-   **Sheaf is indispensable for heterogeneity**: PolySpectralGNN (Chebyshev filtering without sheaf) lags by 20+ points on extremely heterophilous data like Texas, indicating that transport-awareness is the key for heterophilous graphs; polynomial filtering merely amplifies this advantage.
-   **Spectral rescaling is necessary for stability**: Removing $\widetilde{L}=2L/\lambda_\text{max}-I$ leads to immediate training divergence. The $\text{softmax}$ convex mixing is also crucial for implementing $\|p\|_2\le 1$ at the code level.

## Highlights & Insights
-   **Adapting ChebNet techniques to the sheaf Laplacian** is conceptually straightforward but non-trivial in engineering: the sheaf Laplacian is $Nd\times Nd$, and estimating $\lambda_\text{max}$, performing three-term recurrence, and ensuring $[-1,1]$ spectral mapping require careful implementation. This paper successfully navigates these details.
-   **Paradigm Overturn: Performance $\neq$ Large Stalk**: In the past, the community assumed NSD required large stalks and general restrictions to win. PolyNSD breaks this consensus using diagonal restrictions + polynomial filtering, making the sheaf framework more accessible, much like ChebNet popularized spectral graph convolutions.
-   **Spectral Interpretability**: The learned $p_\theta(\lambda)$ can be visualized. On heterophilous data, the model learns multi-band or high-pass responses, while it leans low-pass on homophilous data. This provides spectral interpretability for sheaf models, supplemented by analysis of long-range influence decay in oversquashing.
-   **Transferable Design Template**: (1) The triad of "sparse operator + spectral rescaling + Chebyshev recurrence" is applicable to any symmetric PSD operator (GNNs/Implicit NNs/PDE solvers); (2) the "high-pass skip + gated residual" combo can mitigate low-pass bias in any diffusion-based model.

## Limitations & Future Work
-   **Dependence on accurate $\lambda_\text{max}$ estimation**: While $\lambda_\text{max}=2$ is safe for normalized $\Delta_\mathcal{F}$, using power iteration for unnormalized $L_\mathcal{F}$ depends on spectral properties and may be unstable for extreme graphs.
-   **Chebyshev is the default but not only choice**: The authors acknowledge their approach is basis-agnostic but only systematically validate Chebyshev. Other orthogonal bases (Jacobi, Bernstein) might perform better on specific data.
-   **$K$ is a hyperparameter**: Ideally, $K$ should be automatically selected based on graph diameter or task, but it currently requires manual tuning.
-   **Sheaf learner is still an MLP**: There is potential instability, and more complex structures like multi-edge consistency are not addressed.
-   **Engineering for large-scale graphs**: With $L$ being $Nd\times Nd$, $K$ sparse multiplications remain challenging for ultra-large graphs; validation on OGB-level graphs is missing.

## Related Work & Insights
-   **vs Bodnar et al. (NSD 2022)**: NSD is a special case where $K=1$. PolyNSD extends this to arbitrary orders with stability proofs and breaks the performance dependence on large stalks.
-   **vs ChebNet (Defferrard 2016)**: Similar spectral polynomial filtering but applied to the sheaf Laplacian to achieve dual expressivity: transport-aware and frequency-controllable.
-   **vs GPRGNN / FAGCN / H2GCN**: These are SOTA polynomial or frequency-response baselines for heterophilous graphs using the standard graph Laplacian. PolyNSD outperforms them by 20+ points on Texas/Wisconsin/Squirrel, quantifying the gain from sheaf transport.
-   **vs PolySpectralGNN (Ablation)**: Removing the sheaf while keeping the polynomial filtering leads to significant drops, elegantly proving that "sheaf is not just a waste of parameters."
-   **Insight**: Transitioning any spatial diffusion model to a spectral polynomial version (applied to Hodge Laplacians, normalized attention matrices, or graph wavelet operators) might yield immediate performance benefits.

## Rating
-   Novelty: ⭐⭐⭐⭐ Rather than inventing a new operator, strictly porting the ChebNet framework to the sheaf Laplacian and discovering the counter-intuitive "diagonal restriction is enough" is a significant contribution.
-   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 9 datasets (homo + 6 hetero), 4 types of restrictions, comparison with 14+ baselines. Ablations decouple sheaf vs. polynomial, $K$, rescaling, and high-pass components. Includes oversmoothing/oversquashing/ODE analysis.
-   Writing Quality: ⭐⭐⭐⭐ Dense formulas but logical hierarchy. Propositions 1 & 2 provide theoretical guarantees for spatial locality and energy monotonicity. Fig. 1 clearly illustrates the 10-step pipeline.
-   Value: ⭐⭐⭐⭐ Moves sheaf neural networks from "expensive and niche" to "affordable and practical," opening a research path for designing spectral filters on sheaf Laplacians.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Deep Neural Sheaf Diffusion](deep_neural_sheaf_diffusion.md)
- [\[AAAI 2026\] Sheaf Graph Neural Networks via PAC-Bayes Spectral Optimization](../../AAAI2026/graph_learning/sheaf_graph_neural_networks_via_pac-bayes_spectral_optimization.md)
- [\[ICML 2026\] L2G-Net: Local to Global Spectral Graph Neural Networks via Cauchy Factorizations](l2g-net_local_to_global_spectral_graph_neural_networks_via_cauchy_factorizations.md)
- [\[ICML 2026\] Rethinking Feature Alignment in Generalist Graph Anomaly Detection: A Relational Fingerprint-based Approach](rethinking_feature_alignment_in_generalist_graph_anomaly_detection_a_relational_.md)
- [\[ICML 2026\] Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion](generative_representation_learning_on_hyper-relational_knowledge_graphs_via_mask.md)

</div>

<!-- RELATED:END -->
