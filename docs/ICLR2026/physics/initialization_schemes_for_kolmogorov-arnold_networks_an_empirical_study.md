---
title: >-
  [Paper Note] Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study
description: >-
  [ICLR 2026][Physics & Scientific Computing][KAN] This study presents the first systematic investigation into initialization schemes for spline-based KANs. It proposes LeCun/Glorot-inspired variance preservation schemes and a tunable power-law initialization family. Large-scale experiments involving over 126K model instances demonstrate that power-law initialization consistently outperforms baselines in function fitting and PDE solving. Furthermore…
tags:
  - "ICLR 2026"
  - "Physics & Scientific Computing"
  - "KAN"
  - "Initialization Schemes"
  - "Variance Preservation"
  - "Power-Law Initialization"
  - "Neural Tangent Kernel"
date: 2026-05-08
content_hash: d2cdb56e08686132
---

# Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study

**Conference**: ICLR 2026  
**arXiv**: [2509.03417](https://arxiv.org/abs/2509.03417)  
**Code**: [GitHub](https://github.com/srigas/KAN_Initialization_Schemes)  
**Area**: Deep Learning Theory/KAN  
**Keywords**: KAN, Initialization Schemes, Variance Preservation, Power-Law Initialization, Neural Tangent Kernel

## TL;DR

This study presents the first systematic investigation into initialization schemes for spline-based KANs. It proposes LeCun/Glorot-inspired variance preservation schemes and a tunable power-law initialization family. Large-scale experiments involving over 126K model instances demonstrate that power-law initialization consistently outperforms baselines in function fitting and PDE solving. Furthermore, the Glorot scheme shows significant gains in large-parameter models, with NTK spectral analysis revealing the underlying optimization dynamics.

## Background & Motivation

**Background**: Kolmogorov-Arnold Networks (KANs) replace fixed activation functions in MLPs with trainable B-spline basis functions, showing unique advantages in scientific computing tasks such as function fitting, PDE solving, and operator learning. Each layer of a KAN outputs $y_j = \sum_{i=1}^{n_{\text{in}}} (r_{ji} R(x_i) + c_{ji} \sum_{m=1}^{G+k} b_{jim} B_m(x_i))$, involving three types of trainable parameters: residual weights $r_{ji}$ , scale weights $c_{ji}$, and spline basis coefficients $b_{jim}$. Currently, the KAN community generally follows the original paper's initialization—scale weights set to 1, residual weights using Glorot initialization, and basis coefficients sampled from $\mathcal{N}(0, 0.1)$—which has never been systematically challenged.

**Limitations of Prior Work**: The MLP field has accumulated extensive initialization theories (LeCun 1998, Glorot 2010, He 2015), centered on "variance preservation"—ensuring signals neither amplify nor decay across layers. However, these theories cannot be directly applied to KANs: first, KANs have three parameter types per layer instead of one, complicating variance decomposition; second, spline basis functions $B_m(x)$ depend on the grid partition, and their statistical moments lack general closed-form solutions; third, the residual branch uses SiLU rather than linear functions, further complicating the analysis.

**Key Challenge**: Initialization is vital for training deep networks—poor initialization leads to signal explosion/vanishing, early hidden layer saturation, and slow convergence. The $\sigma=0.1$ scheme long used by the KAN community is an arbitrary choice lacking theoretical grounding, with issues worsening as parameter counts increase.

**Goal**: To establish a systematic initialization theory for spline KANs and answer three questions: (1) Can MLP variance preservation principles adapt to KANs? (2) Which initialization strategy consistently outperforms the baseline across tasks and architectures? (3) How does initialization affect KAN optimization dynamics?

**Key Insight**: The authors derive KAN variance formulas starting from two classical MLP initialization perspectives—forward variance preservation (LeCun) and joint forward+backward preservation (Glorot)—while introducing a tunable power-law initialization family as an empirical benchmark. Conclusions are established through three-level validation: large-scale grid search, NTK spectral analysis, and the Feynman physics formula dataset.

**Core Idea**: Deriving variance preservation formulas across three KAN parameter types and proposing a power-law initialization family, proving the overall superiority of power-law initialization through systematic experiments on 126K+ models.

## Method

### Overall Architecture

The paper addresses "how to initialize KANs" along two paths: a **theoretical path**—starting from MLP variance preservation principles to derive a LeCun scheme (stabilizing forward variance) and a Glorot scheme (stabilizing both forward and backward variance) for the three parameter types ($r_{ji}, c_{ji}, b_{jim}$); and an **empirical path**—bypassing difficult moment calculations of basis functions by providing a two-parameter power-law initialization family for which the optimal exponents are determined via large-scale grid search. The input is the KAN architecture parameters (layers, width, grid size $G$), and the output is the standard deviation ($\sigma_r, \sigma_b$) for each weight type. These schemes are subsequently compared in a massive horizontal experiment, with Neural Tangent Kernel (NTK) spectral analysis used to explain why certain schemes yield healthier optimization dynamics.

### Key Designs

**1. LeCun-inspired Forward Variance Preservation: Matching Output and Input Variance**

The most basic requirement is preventing signals from amplifying or decaying layer-by-layer during forward propagation. This scheme aims to constrain $\text{Var}(y_j) = \text{Var}(x_i)$. The authors fix scale weights $c_{ji}=1$, set residual weights and spline coefficients to zero-mean Gaussians $r_{ji} \sim \mathcal{N}(0, \sigma_r^2)$ and $b_{jim} \sim \mathcal{N}(0, \sigma_b^2)$, and expand the single-layer KAN variance using independence assumptions to solve for:

$$\sigma_r = \sqrt{\frac{\text{Var}(x_i)}{n_{\text{in}}(G+k+1)\mu_R^{(0)}}}$$

where $\mu_R^{(0)} = \mathbb{E}[R(x_i)^2]$ is the second moment of the SiLU residual branch. The formula for $\sigma_b$ follows the same structure, replacing $\mu_R^{(0)}$ with the second moment of the spline basis $\mu_B^{(0)} = \mathbb{E}[B_m(x_i)^2]$. This essentially ports LeCun's classical MLP analysis. The difficulty lies in the "residual + spline" dual-branch structure of each KAN layer and the fact that $\mu_B^{(0)}$ lacks an analytic form. Two variants are derived: LeCun-numerical estimates this moment via numerical sampling during initialization; LeCun-normalized standardizes the basis functions so the second moment is always 1, i.e., $\tilde{B}_m(x_i) = (B_m(x_i) - \mathbb{E}[B_m]) / \sqrt{\mu_B^{(0)} - \mathbb{E}^2[B_m]}$.

**2. Glorot-inspired Joint Forward + Backward Variance Preservation: Stabilizing Activations and Gradients**

Stabilizing forward variance alone is insufficient—gradients can also degrade during backpropagation. This scheme overlays a "variance preservation for backpropagation" constraint onto the LeCun model. Solving for both directions yields:

$$\sigma_r = \sqrt{\frac{2}{(G+k+1)\,(n_{\text{in}}\mu_R^{(0)} + n_{\text{out}}\mu_R^{(1)})}}$$

Here, $\mu_R^{(1)} = \mathbb{E}[R'(x_i)^2]$ is the second moment of the SiLU derivative; $\sigma_b$ takes a similar form involving the moment of the basis function derivative $\mu_B^{(1)}$, estimated via automatic differentiation. Conceptually, this migrates the heuristic that Glorot is more robust than LeCun in MLPs to KANs. A key difference from the LeCun scheme is the inclusion of both $n_{\text{in}}$ and $n_{\text{out}}$, meaning the standard deviation adapts to the dimensions of each layer.

**3. Power-Law Family: Bypassing Moment Estimation with Two Exponents**

While theoretical formulas are elegant, they hinge on estimating basis function moments—especially in PDE scenarios where normalization can propagate standard deviations into higher-order derivatives, causing issues. The power-law family abandons derivation in favor of a two-parameter empirical formula:

$$\sigma_r = \big(n_{\text{in}}(G+k+1)\big)^{-\alpha}, \qquad \sigma_b = \big(n_{\text{in}}(G+k+1)\big)^{-\beta}$$

where $\alpha, \beta$ are chosen from $\{0.0, 0.25, 0.5, \ldots, 2.0\}$. Optimal combinations are found via grid search across 81 combinations. This method requires no moment calculation or sampling and outperforms theoretical schemes requiring precise moment estimation across over 120,000 model instances. Once optimal exponent ranges are identified for a task type (e.g., $\alpha \approx 0.25, \beta \approx 1.0\text{-}1.75$ for function fitting), they can be reused for similar new problems.

## Key Experimental Results

**Setup**: Three benchmarks covered: (1) Function fitting on 5 2D target functions over 2000 epochs; (2) PDE solving for Allen-Cahn, Burgers, and Helmholtz equations over 5000 epochs; (3) A subset of the Feynman physics formula dataset. Architecture search scope: 1-4 hidden layers, widths from $2^1$ to $2^6$, grid size $G \in \{5, 10, 20, 40\}$. Results are medians over 5 random seeds. All experiments utilized JAX/jaxKAN on an RTX 4090.

### Function Fitting Grid Search (126,240 model instances)

| Initialization Scheme | $f_1$ better than Baseline (Loss/L2/Both) | $f_3$ better than Baseline (Loss/L2/Both) | $f_5$ better than Baseline (Loss/L2/Both) |
|:---|:---|:---|:---|
| LeCun-numerical | 18.75% / 6.25% / 1.04% | 12.50% / 5.21% / 0.00% | 26.04% / 2.08% / 0.00% |
| LeCun-normalized | 19.79% / 11.46% / 2.08% | 19.79% / 11.46% / 5.21% | 31.25% / 6.25% / 1.04% |
| Glorot | 78.13% / 78.13% / **78.13%** | 78.13% / 78.13% / **78.13%** | 72.92% / 72.92% / 64.59% |
| **Power-Law** | **100%** / **100%** / **100%** | **100%** / **100%** / **100%** | 98.96% / 96.88% / **95.83%** |

### PDE Benchmark Grid Search (56,882 model instances)

| Initialization Scheme | Allen-Cahn (Loss/L2/Both) | Burgers (Loss/L2/Both) | Helmholtz (Loss/L2/Both) |
|:---|:---|:---|:---|
| LeCun-numerical | 11.11% / 16.67% / 8.33% | 11.11% / 22.22% / 6.94% | 8.33% / 15.28% / 2.78% |
| LeCun-normalized | 2.78% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| Glorot | 55.56% / 51.39% / 41.67% | 50.00% / 54.17% / 36.11% | **76.39%** / **72.22%** / **62.50%** |
| **Power-Law** | **98.61%** / **94.44%** / **94.44%** | **100%** / 73.61% / 73.61% | 98.61% / 87.50% / 87.50% |

### Representative Results on Feynman Dataset (Large Arch G=20, 3×32)

| Formula | Baseline L2 | Glorot L2 | Power-Law L2 | Power-Law Gain |
|:---|:---|:---|:---|:---|
| I.12.11 | 3.77×10⁻¹ | 1.47×10⁻³ | **1.66×10⁻⁴** | 2271× |
| I.16.6 | 6.31×10⁻¹ | 1.63×10⁻² | **1.48×10⁻²** | 43× |
| I.26.2 | 1.10×10⁰ | 8.98×10⁻³ | **1.25×10⁻³** | 880× |
| I.30.3 | 7.72×10⁻¹ | 2.92×10⁻³ | **4.17×10⁻⁴** | 1851× |
| II.6.15a | 7.60×10⁰ | 5.47×10⁻² | **4.40×10⁻³** | 1727× |
| II.35.18 | 1.19×10⁰ | 1.18×10⁻² | **7.77×10⁻⁴** | 1531× |
| III.10.19 | 2.74×10⁻¹ | 9.89×10⁻⁴ | **8.70×10⁻⁵** | 3149× |

### Key Findings

- **Universal Dominance of Power-Law**: In function fitting, the optimal configuration $(\alpha, \beta) = (0.25, 1.0)$ improved both Loss and L2 error in 87.5%–97.9% of cases. Optimal ranges cluster around $\alpha \in \{0.25, 0.5\}$, $\beta \geq 1.0$.
- **Glorot’s Rise for Large Models**: As parameter counts grow (deeper, wider, or finer grids), Glorot's win rate jumps from near-baseline levels to 60-78%. Large KANs require joint forward+backward variance preservation.
- **Critical Failure of LeCun-normalized on PDEs**: Win rates near 0%. Normalized basis functions propagate standard deviations into all derivatives, altering the stiffness of the PDE residual and breaking the balance of physics-informed losses.
- **NTK Spectrum Mechanism**: Baseline and LeCun NTK eigenvalues are highly skewed (dominated by a few large values) and collapse further during training, leading to low effective rank and optimization difficulty. Power-Law produces spectra that decay power-law-like and remain stable; Glorot follows as a strong second.
- **Baseline "Inverse Degradation" on Large Models**: For many Feynman formulas, switching from small to large architectures caused Baseline L2 error to worsen (e.g., I.12.11 worsening from 3.67×10⁻³ to 3.77×10⁻¹), whereas Glorot and Power-Law errors dropped by 2–3 orders of magnitude.

## Highlights & Insights

- **First Complete Adaptation**: Systematically migrates classical MLP initialization theories (forward and joint variance preservation) to the three-parameter KAN framework. This provides a methodological template for further KAN variants.
- **Elegance of the Power-Law Family**: A simple formula with two parameters $(\alpha, \beta)$ outperforms theoretical schemes requiring precise moment estimation across over 120,000 instances without requiring sampling or architecture changes. "Simple yet effective" empirical methods prove their worth over complex theory.
- **NTK as a Diagnosis Tool**: Provides a mechanistic explanation rather than just saying one scheme is "better." A good initialization should produce a dispersed, stable eigenvalue spectrum—a framework applicable to assessing any new KAN initialization.
- **Lessons from PDEs**: The failure of LeCun-normalized on PDEs shows that physics-informed losses involving high-order derivatives are far more sensitive to initialization than standard regression.

## Limitations & Future Work

- **Restricted to Spline KANs**: Variants like Chebyshev, Fourier, or Wavelet KANs have different basis function structures; optimal strategies may differ entirely.
- **Lack of Theory for Power-Law Exponents**: The reason why $\alpha \approx 0.25, \beta \approx 1.0\text{-}1.75$ works best remains empirical and lacks deep theoretical backing.
- **Narrow Task Domains**: Only function fitting and PDEs were verified; classification, RL, and generative models were not addressed. Cross-domain transferability of exponents remains unconfirmed.
- **Scale Ceiling**: The largest architecture tested was 3 layers × 32 width × G=20, which is smaller than many real-world KANs. Initialization behavior in KANs with millions of parameters deserves study.
- **Interaction with Adaptive Optimizers**: All experiments used a fixed learning rate. The interaction between initialization and modern techniques like Adam or learning rate warmups is not fully explored.

## Related Work & Insights

- **vs Original KAN (Liu et al., 2025)**: Original KAN uses a fixed $\sigma=0.1$. This study proves it is insufficient for large models, leading to performance degradation on the Feynman dataset.
- **vs Chebyshev KAN Initialization (Rigas et al., 2026)**: Chebyshev KAN has used Glorot but removed the residual branch; this differs fundamentally from the spline + residual setting analyzed here.
- **vs He Initialization**: He initialization is designed for ReLU (considering asymmetry at zero). Since SiLU in KANs is smooth and symmetric, LeCun/Glorot are more appropriate starting points.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic exploration of KAN initialization, filling a clear gap.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Massive search across 126K+ model instances, with NTK and Feynman validation.
- Writing Quality: ⭐⭐⭐⭐ Solid connection between theoretical derivation and experimental analysis.
- Value: ⭐⭐⭐⭐ Provides ready-to-use, effective initialization strategies for the KAN community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] KANO: Kolmogorov–Arnold Neural Operator](kano_kolmogorov-arnold_neural_operator.md)
- [\[AAAI 2026\] Catastrophic Forgetting in Kolmogorov-Arnold Networks](../../AAAI2026/physics/catastrophic_forgetting_in_kolmogorov-arnold_networks.md)
- [\[AAAI 2026\] Learning Fair Representations with Kolmogorov-Arnold Networks](../../AAAI2026/physics/learning_fair_representations_with_kolmogorov-arnold_networks.md)
- [\[AAAI 2026\] FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer](../../AAAI2026/physics/flashkat_understanding_and_addressing_performance_bottlenecks_in_the_kolmogorov-.md)
- [\[CVPR 2025\] KAC: Kolmogorov-Arnold Classifier for Continual Learning](../../CVPR2025/physics/kac_kolmogorov-arnold_classifier_for_continual_learning.md)

</div>

<!-- RELATED:END -->
