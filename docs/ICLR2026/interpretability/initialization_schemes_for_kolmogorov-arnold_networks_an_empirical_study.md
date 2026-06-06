---
title: >-
  [Paper Note] Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study
description: >-
  [ICLR 2026][Interpretability][KAN] This work presents the first systematic study of initialization strategies for spline-based KANs. It proposes variance-preserving schemes inspired by LeCun/Glorot and a tunable power-la…
tags:
  - "ICLR 2026"
  - "Interpretability"
  - "KAN"
  - "initialization schemes"
  - "variance preservation"
  - "power-law initialization"
  - "neural tangent kernel"
date: 2026-05-08
content_hash: 1d160a392e9bffe2
---

# Initialization Schemes for Kolmogorov-Arnold Networks: An Empirical Study

**Conference**: ICLR 2026  
**arXiv**: [2509.03417](https://arxiv.org/abs/2509.03417)  
**Code**: [GitHub](https://github.com/srigas/KAN_Initialization_Schemes)  
**Area**: Deep Learning Theory / KAN  
**Keywords**: KAN, initialization schemes, variance preservation, power-law initialization, neural tangent kernel

## TL;DR

This work presents the first systematic study of initialization strategies for spline-based KANs. It proposes variance-preserving schemes inspired by LeCun/Glorot and a tunable power-law initialization family. Large-scale experiments spanning 126K+ model instances demonstrate that power-law initialization consistently outperforms baselines on function fitting and PDE solving, while the Glorot scheme yields significant gains for larger models. NTK eigenspectrum analysis further reveals the underlying optimization dynamics.

## Background & Motivation

**Background**: Kolmogorov-Arnold Networks (KANs) replace the fixed activation functions in MLPs with trainable B-spline basis functions, exhibiting unique advantages on scientific computing tasks such as function fitting, PDE solving, and operator learning. The output of each KAN layer is $y_j = \sum_{i=1}^{n_{\text{in}}} (r_{ji} R(x_i) + c_{ji} \sum_{m=1}^{G+k} b_{jim} B_m(x_i))$, which contains three types of trainable parameters: residual weights $r_{ji}$, scaling weights $c_{ji}$, and spline basis coefficients $b_{jim}$. The KAN community has universally adopted the initialization strategy from the original paper—scaling weights set to 1, residual weights initialized with Glorot, and basis coefficients sampled from $\mathcal{N}(0, 0.1)$—without ever systematically challenging it.

**Limitations of Prior Work**: The MLP literature has accumulated rich initialization theory (LeCun 1998, Glorot 2010, He 2015), centered on the principle of variance preservation—ensuring that signals neither amplify nor vanish as they propagate through layers. However, this theory cannot be directly transferred to KANs: first, each KAN layer has three parameter types rather than one, making variance decomposition more complex; second, the spline basis functions $B_m(x)$ depend on grid partitions, and their statistical moments have no universal closed-form expression; third, the residual branch employs SiLU rather than a linear function, further complicating the analysis.

**Key Challenge**: Initialization is critical for training deep networks—poor initialization leads to signal explosion or vanishing, early saturation of hidden layers, and slow convergence. Yet the $\sigma=0.1$ scheme long used by the KAN community is an arbitrary choice with no theoretical justification, and the problem worsens as model size increases.

**Goal**: To establish a systematic initialization theory for spline KANs by addressing three questions: (1) Can the variance-preservation principle from MLPs be adapted to KANs? (2) What initialization strategy can consistently outperform baselines across tasks and architectures? (3) How does initialization affect the optimization dynamics of KANs?

**Key Insight**: The authors derive KAN variance formulas from the two classical MLP initialization perspectives—forward variance preservation (LeCun) and joint forward–backward preservation (Glorot)—while introducing a tunable power-law initialization family as an empirical counterpart. Conclusions are established through three levels of validation: large-scale grid search, NTK eigenspectrum analysis, and the Feynman physics formula dataset.

**Core Idea**: Derive variance-preserving formulas for the three parameter types in KAN and propose a power-law initialization family; systematic experiments across 126K+ models demonstrate that power-law initialization achieves overall superiority.

## Method

### Overall Architecture

The methodological framework operates on three levels: (1) theoretical derivation—deriving LeCun- and Glorot-adapted initialization formulas for KAN from variance-preservation principles; (2) empirical exploration—designing a power-law initialization family and identifying optimal configurations via large-scale grid search; (3) dynamics analysis—using the NTK eigenspectrum to explain performance differences across initialization schemes. The input is the KAN architecture specification (depth, width, grid size), and the output is the initialization standard deviation $\sigma_r$ and $\sigma_b$ for each parameter type.

### Key Designs

1. **LeCun-Inspired Forward Variance-Preserving Initialization**:

    - **Function**: Ensures that the output variance of each layer equals its input variance, preventing signals from amplifying or vanishing during forward propagation.
    - **Mechanism**: With $c_{ji}=1$ fixed, and $r_{ji} \sim \mathcal{N}(0, \sigma_r^2)$, $b_{jim} \sim \mathcal{N}(0, \sigma_b^2)$, the constraint $\text{Var}(y_j) = \text{Var}(x_i)$ is derived under an independence assumption, yielding $\sigma_r = \sqrt{\text{Var}(x_i) / (n_{\text{in}}(G+k+1)\mu_R^{(0)})}$, where $\mu_R^{(0)} = \mathbb{E}[R(x_i)^2]$ is the second moment of SiLU. An analogous formula holds for $\sigma_b$, involving the second moment of the spline basis $\mu_B^{(0)} = \mathbb{E}[B_m(x_i)^2]$.
    - **Design Motivation**: Directly parallels LeCun's analysis for MLPs, adapted to KAN's dual residual-plus-spline branch structure. Since $\mu_B^{(0)}$ depends on the grid and cannot be computed analytically, two variants are introduced: LeCun-numerical (numerically estimating moments by sampling at initialization) and LeCun-normalized (normalizing basis functions so that their moment is identically 1, i.e., $\tilde{B}_m(x_i) = (B_m(x_i) - \mathbb{E}[B_m]) / \sqrt{\mu_B^{(0)} - \mathbb{E}^2[B_m]}$).

2. **Glorot-Inspired Joint Forward–Backward Variance-Preserving Initialization**:

    - **Function**: Simultaneously stabilizes the variance of forward activations and backward gradients, preventing signal degradation in either direction.
    - **Mechanism**: Extending LeCun with an additional backward variance-preservation constraint yields $\sigma_r = \sqrt{(G+k+1)^{-1} \cdot 2 / (n_{\text{in}}\mu_R^{(0)} + n_{\text{out}}\mu_R^{(1)})}$, where $\mu_R^{(1)} = \mathbb{E}[R'(x_i)^2]$ is the second moment of the SiLU derivative. An analogous formula for $\sigma_b$ involves the derivative moments of the basis functions $\mu_B^{(1)}$, estimated numerically via automatic differentiation.
    - **Design Motivation**: Glorot initialization is more robust than LeCun in MLPs; the same "bidirectional balance" idea is transferred to KANs here. The additional dependence on $n_{\text{out}}$ and derivative moments allows the standard deviation to adapt automatically to the input and output dimensionality of each layer.

3. **Power-Law Initialization Family**:

    - **Function**: Provides a concise empirical formula family that covers a broad initialization space by tuning two exponent parameters.
    - **Mechanism**: $\sigma_r = (n_{\text{in}}(G+k+1))^{-\alpha}$, $\sigma_b = (n_{\text{in}}(G+k+1))^{-\beta}$, with $\alpha, \beta \in \{0.0, 0.25, 0.5, \ldots, 2.0\}$. No theoretical derivation is involved; the optimal configuration is found purely through grid search over 81 combinations of $(\alpha, \beta)$.
    - **Design Motivation**: Although theoretically derived formulas are elegant, they require estimating basis function moments—a non-trivial issue in PDE settings due to normalization propagation. The power-law family bypasses this difficulty entirely. Once a favorable exponent range is identified for a task class (e.g., $\alpha \approx 0.25$, $\beta \approx 1.0\text{–}1.75$ for function fitting), it can be directly reused for new problems of the same type.

### Experimental Design

Experiments span three benchmarks: (1) function fitting—5 two-dimensional target functions, trained for 2,000 epochs; (2) PDE solving—three forward PDEs (Allen–Cahn, Burgers, Helmholtz), trained for 5,000 epochs; (3) a subset of the Feynman physics formula dataset. Architecture search space: 1–4 hidden layers, widths from $2^1$ to $2^6$, grid sizes $G \in \{5, 10, 20, 40\}$. Each configuration is run with 5 random seeds and evaluated by median (3 seeds for grid search to reduce computational cost). All experiments are implemented in JAX/jaxKAN on a single RTX 4090.

## Key Experimental Results

### Function Fitting Grid Search (126,240 model instances)

| Initialization | $f_1$ beats baseline (Loss/L2/Both) | $f_3$ beats baseline (Loss/L2/Both) | $f_5$ beats baseline (Loss/L2/Both) |
|:---|:---|:---|:---|
| LeCun-numerical | 18.75% / 6.25% / 1.04% | 12.50% / 5.21% / 0.00% | 26.04% / 2.08% / 0.00% |
| LeCun-normalized | 19.79% / 11.46% / 2.08% | 19.79% / 11.46% / 5.21% | 31.25% / 6.25% / 1.04% |
| Glorot | 78.13% / 78.13% / **78.13%** | 78.13% / 78.13% / **78.13%** | 72.92% / 72.92% / 64.59% |
| **Power-Law** | **100%** / **100%** / **100%** | **100%** / **100%** / **100%** | 98.96% / 96.88% / **95.83%** |

### PDE Benchmark Grid Search (56,882 model instances)

| Initialization | Allen-Cahn (Loss/L2/Both) | Burgers (Loss/L2/Both) | Helmholtz (Loss/L2/Both) |
|:---|:---|:---|:---|
| LeCun-numerical | 11.11% / 16.67% / 8.33% | 11.11% / 22.22% / 6.94% | 8.33% / 15.28% / 2.78% |
| LeCun-normalized | 2.78% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% | 0.00% / 0.00% / 0.00% |
| Glorot | 55.56% / 51.39% / 41.67% | 50.00% / 54.17% / 36.11% | **76.39%** / **72.22%** / **62.50%** |
| **Power-Law** | **98.61%** / **94.44%** / **94.44%** | **100%** / 73.61% / 73.61% | 98.61% / 87.50% / 87.50% |

### Representative Feynman Dataset Results (Large Architecture: G=20, 3×32)

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

- **Power-law initialization dominates comprehensively**: In function fitting, the optimal configuration $(\alpha, \beta) = (0.25, 1.0)$ simultaneously improves both Loss and L2 error across all 5 target functions in 87.5%–97.9% of cases. The optimal range concentrates at $\alpha \in \{0.25, 0.5\}$, $\beta \geq 1.0$.
- **Glorot emerges at larger scales**: As model capacity increases (deeper, wider, or finer grids), Glorot's win rate rises from near-baseline levels to 60–78%, indicating that larger KANs benefit more from bidirectional variance preservation.
- **LeCun-normalized completely fails on PDEs**: Win rates approach 0%. Normalizing the basis functions propagates the modified standard deviation into all derivatives, altering the stiffness of PDE residuals and disrupting the balance of physics-informed losses.
- **NTK eigenspectrum reveals the mechanism**: The NTK eigenvalue distributions of the baseline and LeCun schemes are highly skewed (dominated by a few large eigenvalues) and collapse further during training, resulting in low effective rank and optimization difficulty. Power-law initialization yields an approximately power-law-decaying spectrum that remains stable throughout training; Glorot ranks second but is still far superior to the baseline.
- **Baseline degrades with scale**: On many Feynman formulas, switching from small to large architectures worsens baseline L2 error (e.g., I.12.11 degrades from 3.67×10⁻³ to 3.77×10⁻¹), whereas Glorot and Power-Law generally achieve 2–3 orders of magnitude lower error on larger models.

## Highlights & Insights

- **First complete adaptation**: MLP classical initialization theory (forward variance preservation and joint forward–backward preservation) is systematically transferred to KAN's three-parameter framework. The derivation is clear and rigorous, establishing a methodological template for initialization research in KAN variants.
- **Elegance of the power-law family**: The two-hyperparameter $(\alpha, \beta)$ power-law formula requires no computation of basis function moments, no numerical sampling, and no architectural modifications, yet comprehensively outperforms theoretically motivated schemes requiring precise moment estimation across over 120,000 model instances. Simple but sufficiently good empirical methods sometimes surpass elaborately derived theoretical ones.
- **NTK as an initialization diagnostic tool**: Rather than merely claiming that one scheme is better, the paper provides mechanistic explanations through NTK eigenvalue spectra—good initialization should produce dispersed and stable eigenspectra. This analytical framework can be directly applied to evaluating any new KAN initialization scheme.
- **A special lesson for PDE settings**: LeCun-normalized performs reasonably on function fitting but completely fails on PDEs, revealing that physics-informed losses involving high-order derivatives are far more sensitive to initialization than standard regression tasks.

## Limitations & Future Work

- **Restricted to spline KANs**: Variants such as Chebyshev KAN, Fourier KAN, and Wavelet KAN employ different basis function structures, and their optimal initialization strategies may differ substantially.
- **Lack of theoretical explanation for optimal power-law exponents**: Why do $\alpha \approx 0.25$ and $\beta \approx 1.0\text{–}1.75$ work well? The paper acknowledges these as purely empirical findings without deeper theoretical grounding.
- **Narrow task scope**: Only function fitting and PDE solving are evaluated; classification, reinforcement learning, and generative modeling scenarios are not covered. Whether the optimal exponent range transfers across domains remains unconfirmed.
- **Scale ceiling**: The largest architecture evaluated is 3 layers × 32 width × G=20, far smaller than large-scale KANs that may appear in real applications. Initialization behavior at the tens-of-thousands-of-parameter scale warrants further investigation.
- **No interaction with adaptive optimization**: All experiments use a fixed learning rate (scheduler experiments appear in the appendix but are not a primary focus). The interaction between initialization and modern training practices such as Adam or learning rate warmup has not been sufficiently studied.

## Related Work & Insights

- **vs. original KAN (Liu et al., 2025)**: The original KAN uses a fixed initialization of $\sigma=0.1$, which this work demonstrates to be severely inadequate for large models—performance on the Feynman dataset actually degrades when using baseline initialization with larger architectures.
- **vs. Chebyshev KAN initialization (Rigas et al., 2026)**: Glorot initialization has been successfully applied to Chebyshev KAN, but without the residual branch; this is fundamentally different from the spline-plus-residual setting in this paper and cannot be directly transferred.
- **vs. He initialization**: He initialization is designed for ReLU (accounting for the asymmetry at the zero point). SiLU, used in KAN's residual branch, is smooth and symmetric, making LeCun/Glorot more appropriate starting points than He.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic exploration of KAN initialization, filling a clear gap in the literature.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Massive grid search across 126K+ model instances, with three progressive validation levels: NTK analysis and Feynman dataset verification.
- Writing Quality: ⭐⭐⭐⭐ Theoretical derivations and experimental analyses are tightly connected; figures and tables are clearly designed.
- Value: ⭐⭐⭐⭐ Provides the KAN community with plug-and-play practical initialization strategies.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer](../../AAAI2026/interpretability/flashkat_understanding_and_addressing_performance_bottlenecks_in_the_kolmogorov-.md)
- [\[ICLR 2026\] Modal Logical Neural Networks for Financial AI](modal_logical_neural_networks_for_financial_ai.md)
- [\[ICML 2026\] Understanding LoRA as Knowledge Memory: An Empirical Analysis](../../ICML2026/interpretability/understanding_lora_as_knowledge_memory_an_empirical_analysis.md)
- [\[ICLR 2026\] SALVE: Sparse Autoencoder-Latent Vector Editing for Mechanistic Control of Neural Networks](salve_sparse_autoencoder-latent_vector_editing_for_mechanistic_control_of_neural.md)
- [\[ACL 2026\] Rhetorical Questions in LLM Representations: A Linear Probing Study](../../ACL2026/interpretability/rhetorical_questions_in_llm_representations_a_linear_probing_study.md)

</div>

<!-- RELATED:END -->
