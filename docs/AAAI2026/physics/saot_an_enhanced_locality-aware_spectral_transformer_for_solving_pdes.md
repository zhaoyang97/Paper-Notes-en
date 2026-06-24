---
title: >-
  [Paper Note] SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs
description: >-
  [AAAI 2026][Physics & Scientific Computing][Neural Operator] This paper proposes SAOT (Spectral Attention Operator Transformer), which captures high-frequency local details via linear-complexity Wavelet Attention (WA) and complements it with the global receptive field of Fourier Attention (FA) through a gated fusion mechanism. SAOT achieves state-of-the-art performance on 6 operator learning benchmarks, reducing the relative error on Navier-Stokes by 22.3% compared to Transol…
tags:
  - "AAAI 2026"
  - "Physics & Scientific Computing"
  - "Neural Operator"
  - "Wavelet Transform"
  - "Fourier Attention"
  - "Spectral Transformer"
  - "PDE Solving"
date: 2026-05-08
content_hash: 77d3554209520302
---

# SAOT: An Enhanced Locality-Aware Spectral Transformer for Solving PDEs

**Conference**: AAAI 2026
**arXiv**: [2511.18777](https://arxiv.org/abs/2511.18777)  
**Code**: [https://github.com/chenhong-zhou/SAOT](https://github.com/chenhong-zhou/SAOT)  
**Authors**: Chenhong Zhou, Jie Chen, Zaifeng Yang
**Area**: Scientific Computing / Operator Learning
**Keywords**: Neural Operator, Wavelet Transform, Fourier Attention, Spectral Transformer, PDE Solving

## TL;DR

This paper proposes SAOT (Spectral Attention Operator Transformer), which captures high-frequency local details via linear-complexity Wavelet Attention (WA) and complements it with the global receptive field of Fourier Attention (FA) through a gated fusion mechanism. SAOT achieves state-of-the-art performance on 6 operator learning benchmarks, reducing the relative error on Navier-Stokes by 22.3% compared to Transolver.

## Background & Motivation

### State of the Field

PDE solving is a fundamental task in scientific computing. Traditional numerical methods (finite element, finite difference) offer high accuracy but at substantial computational cost. Deep learning-driven neural operator approaches have emerged rapidly in recent years. FNO (Fourier Neural Operator) established the frequency-domain learning paradigm: it parameterizes the integral operator kernel in Fourier space, which is equivalent to global convolution in the spatial domain, enabling efficient capture of long-range dependencies. Subsequent variants such as U-FNO, Geo-FNO, and F-FNO, as well as Transformer-based architectures like Transolver, have further advanced the field.

### Core Problem: Spectral Bias of Fourier Methods

Despite the strength of Fourier methods in capturing global features, they suffer from a fundamental limitation—**spectral bias**—manifested in two ways:

**Over-smoothing**: Global convolution operations tend to produce overly smooth solutions, suppressing high-frequency components. The paper demonstrates this through energy spectrum analysis: FA aligns well with the ground truth at low wavenumbers but exhibits a sharp energy decay at high wavenumbers, deviating from the true distribution.

**Loss of local detail**: The Fourier transform is inherently a global operation whose basis functions have support over the entire domain, lacking spatial locality. However, PDE solutions often contain features such as boundary layers, shocks, and vortical structures that require both spatial locality and high-frequency information to be accurately captured.

**Limited effectiveness of existing remedies**: MWT (Multi-Wavelet Operator) and WNO (Wavelet Neural Operator) attempt to replace the Fourier transform with wavelet transforms, but have not achieved the expected accuracy gains in practice—particularly as Transformer-based architectures become increasingly dominant.

### Starting Point

The wavelet transform inherently possesses **time-frequency localization**: wavelet basis functions are localized in both space and frequency, preserving frequency information while retaining spatial position. This directly compensates for the locality deficiency of the Fourier transform. The paper's core idea is: **Wavelet Attention captures local high-frequency features; Fourier Attention captures global low-frequency features; the two are adaptively fused via a gating mechanism**. This is a classical complementary strategy in signal processing, but is here effectively realized in the operator learning domain for the first time.

## Method

### Overall Architecture

SAOT follows a standard Encoder–Processor–Decoder Transformer architecture:

- **Encoder**: A linear layer lifts the input function from $\mathbb{R}^{d_a}$ to a high-dimensional feature space $\mathbb{R}^D$.
- **Processor**: $L$ stacked pre-norm Transformer blocks, where the standard self-attention is replaced by the proposed **Spectral Attention (SA)**.
- **Decoder**: A linear layer projects to the output dimension $\mathbb{R}^{d_u}$.

Each Transformer block follows: $\hat{X}^l = \text{SA}(\text{LN}(X^{l-1})) + X^{l-1}$, $X^l = \text{MLP}(\text{LN}(\hat{X}^l)) + \hat{X}^l$.

### Wavelet Attention (WA) — Core Innovation

WA is designed to learn locality-sensitive features in the wavelet domain with linear complexity $O(ND^2)$. The pipeline proceeds as follows:

**Step 1 — Channel reduction**: The input feature $X \in \mathbb{R}^{H \times W \times D}$ is first projected to $\bar{X} \in \mathbb{R}^{H \times W \times D/4}$ via a convolutional layer. This dimensionality reduction prepares for channel concatenation after FWT decomposition, ensuring the final dimensionality does not expand.

**Step 2 — Fast Wavelet Transform (FWT) decomposition**: Using the Haar wavelet, $\bar{X}$ is decomposed into 4 subbands. The high-pass filter $f_H = (1/\sqrt{2}, -1/\sqrt{2})$ and low-pass filter $f_L = (1/\sqrt{2}, 1/\sqrt{2})$ are applied sequentially along rows and columns, yielding:

- $X_{LL}$: low-frequency component capturing coarse-grained global features;
- $X_{LH}, X_{HL}, X_{HH}$: three high-frequency components preserving fine-grained details in different orientations.

Each subband has size $\mathbb{R}^{H/2 \times W/2 \times D/4}$; concatenating along the channel dimension gives $\tilde{X} \in \mathbb{R}^{H/2 \times W/2 \times D}$. The halved spatial resolution reduces subsequent attention computation to one-quarter of the original cost.

**Step 3 — Local convolution enhancement**: Optionally, a $3 \times 3$ convolution is applied to $\tilde{X}$ to reinforce spatial local correlations across wavelet subbands, producing locally contextualized features $X^w$.

**Step 4 — Linearized attention**: $X^w$ is reshaped into $Q^w, K^w, V^w \in \mathbb{R}^{n \times D}$ (where $n = H/2 \times W/2$), and standard attention is linearized using the kernel feature map $\phi(x) = \text{elu}(x) + 1$. The key formula is:

$$s'_i = \frac{\phi(q_i)^T \left(\sum_j \phi(k_j) \otimes v_j\right)}{\phi(q_i)^T \sum_l \phi(k_l)}$$

Since $\sum_j \phi(k_j) \otimes v_j$ and $\sum_l \phi(k_l)$ can be precomputed and reused across all queries, the complexity is reduced from $O(N^2)$ to $O(N)$.

**Step 5 — Inverse wavelet transform and output synthesis**: The linear attention output is reshaped back to $\mathbb{R}^{H/2 \times W/2 \times D}$, reconstructed via IFWT to $X^r \in \mathbb{R}^{H \times W \times D/4}$, concatenated with the original input $X$, and passed through a linear layer to produce the final output $X^{WA}$.

**Design Motivation Summary**: The core insight of WA is to exploit the wavelet transform to decompose the input into multiple frequency subbands and perform attention computation across these subbands—enabling the model to simultaneously learn low-frequency global patterns and high-frequency local details in a compact representation space, while linearized attention ensures computational efficiency.

### Fourier Attention (FA)

FA follows the AFNO design, approximating the integral operator kernel in Fourier space:

$$\mathcal{K}(X)(g) = \mathcal{F}^{-1}(R_\psi \cdot \mathcal{F}(X))(g)$$

where $R_\psi$ is implemented via block-diagonal MLP layers (rather than the learnable complex-valued weight tensors in FNO), substantially reducing parameter count and memory. Unlike AFNO, **no sparsification or truncation** is applied—all frequency modes are retained to preserve full expressiveness. The complexity is $O(ND\log N)$, with a residual connection on the output: $X^{FA} = X + X'$. FA and WA operate in parallel.

### Gated Fusion Block

Different spatial regions have different demands for global versus local information (e.g., boundary regions rely more on local high frequencies; flat regions rely more on global low frequencies). An adaptive gated fusion is therefore designed:

$$G = \sigma(\text{Linear}(\text{Concat}(X^{FA}, X^{WA})))$$
$$X^{SA} = G \odot X^{FA} + (1 - G) \odot X^{WA}$$

where $\sigma$ is the sigmoid function and $G \in \mathbb{R}^{N \times D}$ is an element-wise gating weight. This design allows the network to adaptively determine, for each position and channel, how much to rely on global versus local information based on the data itself.

### Computational Complexity

The complexity of SA is $O(\max(ND^2, ND\log N))$, and the full SAOT with $L$ layers has complexity $O(L \cdot \max(ND^2, ND\log N))$—significantly more efficient than standard self-attention at $O(N^2D)$ for large-scale meshes.

### Training Details

Relative $L^2$ error serves as both the loss and evaluation metric. The Adam optimizer is used with an initial learning rate of $10^{-3}$, training for 500 epochs on a single V100S-32GB GPU.

## Key Experimental Results

### Table 1: Main Results — Relative $L^2$ Error on 6 Benchmarks (↓ lower is better)

| Model | Darcy | NS | Airfoil | Pipe | Plasticity | Elasticity |
|------|-------|------|---------|------|-----------|------------|
| FNO | 0.0108 | 0.1556 | - | - | - | - |
| Geo-FNO | 0.0108 | 0.1556 | 0.0138 | 0.0067 | 0.0074 | 0.0229 |
| MWT | 0.0067 | 0.1553 | 0.0076 | 0.0072 | 0.0027 | 0.0334 |
| WNO | 0.0242 | 0.1613 | 0.0188 | 0.0070 | - | 0.0465 |
| GNOT | 0.0105 | 0.1380 | 0.0076 | - | - | 0.0086 |
| IPOT | 0.0085 | 0.0885 | 0.0088 | - | 0.0033 | 0.0156 |
| Transolver | 0.0058 | 0.0985 | 0.0053 | 0.0043 | 0.0012 | 0.0067 |
| **SAOT** | **0.0049** | **0.0688** | **0.0048** | 0.0063 | **0.0008** | 0.0080 |
| vs Transolver | ↓15.5% | **↓22.3%** | ↓9.4% | ↑46.5% | **↓33.3%** | ↑19.4% |

SAOT achieves best performance on Darcy, NS, Airfoil, and Plasticity, with the most significant improvements on NS and Plasticity. It underperforms Transolver on Pipe and Elasticity.

### Table 2: Ablation Study — Attention Mechanism Comparison

| Attention | Darcy Params (M) | Darcy $L^2$ | Elasticity Params (M) | Elasticity $L^2$ |
|--------|----------------|------------|---------------------|-----------------|
| FA only | 0.651 | 0.0058 | 0.576 | 0.0232 |
| WA only | 2.361 | 0.0057 | 1.514 | 0.0129 |
| **SA (FA+WA)** | 2.694 | **0.0049** | 2.040 | **0.0080** |

- WA alone substantially outperforms FA alone (Elasticity: 0.0129 vs. 0.0232, ↓44.4%), validating the importance of high-frequency local information.
- SA further improves upon either branch alone—gated fusion achieves genuine complementary gains.
- WA has approximately 3.6× more parameters than FA, but delivers a qualitative performance leap.

## Highlights & Insights

1. **Energy spectrum analysis as a diagnostic tool**: By comparing the energy spectra of FA and WA predictions against the ground truth, the paper visually exposes the root cause of spectral bias in Fourier methods. WA aligns more closely with the true energy spectrum at high wavenumbers. This analytical approach can be generalized to diagnose frequency-capture deficiencies in any frequency-domain method.

2. **Effective realization of wavelet–Fourier complementarity**: The time-frequency complementarity between wavelets and Fourier transforms is a classical idea in signal processing, but this paper is the first to integrate both into a Transformer architecture for operator learning with consistent improvements. The gated fusion is critical—rather than simple weighted averaging, it allows the network to adaptively assign weights per spatial location and channel.

3. **Strong zero-shot super-resolution generalization**: A model trained at resolution $85^2$ maintains the lowest error across the range $43^2$–$421^2$ at test time, demonstrating strong discretization invariance. While all methods exhibit a U-shaped error curve (error increases as the test resolution deviates from the training resolution), SAOT's U-shape is the shallowest.

4. **WA substantially outperforms existing wavelet operators**: WA's Darcy error (0.0057) is far below MWT (0.0067) and WNO (0.0242), indicating that embedding wavelet attention within a Transformer framework is more effective than directly constructing operator layers from wavelets.

## Limitations & Future Work

1. **Underperformance on Pipe and Elasticity vs. Transolver**: The error on Pipe is 46.5% higher (0.0063 vs. 0.0043) and on Elasticity 19.4% higher. Transolver's physics-aware slice attention may be better suited to specific geometric structures.

2. **Exclusive use of Haar wavelets**: The Haar wavelet is the simplest wavelet basis and is inferior to higher-order wavelets such as Daubechies or Coiflet in terms of smoothness and approximation capability. The paper acknowledges in its conclusion that exploring alternative wavelet bases is a direction for future work.

3. **Significant parameter growth**: SA (2.694M parameters) is 4.1× larger than FA-only (0.651M), making the efficiency–performance trade-off a concern for parameter-constrained deployment scenarios.

4. **Validation limited to 2D PDEs**: All benchmarks are 2D problems; effectiveness on higher-dimensional PDEs (3D fluids, spatiotemporal problems) remains unverified.

## Related Work & Insights

- **vs. Transolver**: Transolver uses physics-inspired slice attention to aggregate mesh points into physical slices for attention computation, focusing on physical structure; SAOT designs attention from the perspective of frequency-domain complementarity. The two approaches are complementary, each excelling on different PDE types.
- **vs. MWT / WNO**: Both belong to the wavelet methods family, but MWT employs multi-wavelet bases as operator layers and WNO uses wavelet integral layers—neither is effectively integrated with Transformer architectures. SAOT's WA embeds wavelets into an attention mechanism, achieving a qualitative leap.
- **vs. AFNO / FNet / GFNet**: These spectral Transformers perform token mixing exclusively in the frequency domain and lack locality. SAOT achieves an Elasticity error of only 0.0080, far below GFNet (0.0230) and AFNO (0.0228).
- **vs. FNO and variants**: The FNO family performs global Fourier convolution; SAOT additionally introduces a wavelet branch to capture local details, comprehensively outperforming FNO-based methods on all regular-grid benchmarks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of wavelet attention and Fourier attention for operator learning is novel and effective; the gated fusion design is well-motivated.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 6 benchmarks, 11+ baselines, detailed ablations, energy spectrum analysis, super-resolution generalization tests, and additional comparisons with spectral Transformers.
- **Writing Quality**: ⭐⭐⭐⭐ — The motivation chain is clear (spectral bias → wavelet locality → complementary fusion); energy spectrum visualizations are intuitive and compelling.
- **Value**: ⭐⭐⭐⭐ — Provides a new direction for frequency-domain attention design in operator learning; code is publicly available.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Knowledge-Guided Masked Autoencoder with Linear Spectral Mixing and Spectral-Angle-Aware Reconstruction](knowledge-guided_masked_autoencoder_with_linear_spectral_mixing_and_spectral-ang.md)
- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/physics/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[AAAI 2026\] FlashKAT: Understanding and Addressing Performance Bottlenecks in the Kolmogorov-Arnold Transformer](flashkat_understanding_and_addressing_performance_bottlenecks_in_the_kolmogorov-.md)
- [\[ICML 2026\] TINNs: Time-Induced Neural Networks for Solving Time-Dependent PDEs](../../ICML2026/physics/tinns_time-induced_neural_networks_for_solving_time-dependent_pdes.md)
- [\[ICLR 2026\] Riesz Neural Operator for Solving Partial Differential Equations](../../ICLR2026/physics/riesz_neural_operator_for_solving_partial_differential_equations.md)

</div>

<!-- RELATED:END -->
