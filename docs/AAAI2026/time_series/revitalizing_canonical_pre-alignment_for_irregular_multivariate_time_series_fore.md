---
title: >-
  [Paper Note] Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting
description: >-
  [AAAI 2026][Time Series][Irregular Multivariate Time Series] This paper is the first to argue that Canonical Pre-Alignment (CPA) should not be abandoned for Irregular Multivariate Time Series (IMTS) forecasting. It proposes KAFNet, which addresses the efficiency bottleneck of CPA via three modules—Pre-Convolution smoothing, Temporal Kernel Aggregation (TKA), and Frequency-domain Linear Attention (FLA)—achieving state-of-the-art accuracy on 4 IMTS benchmarks while reducing parameters by 7.2× and accelerating training/inference by 8.4×.
tags:
  - AAAI 2026
  - Time Series
  - Irregular Multivariate Time Series
  - Canonical Pre-Alignment
  - Temporal Kernel Aggregation
  - Frequency-domain Linear Attention
  - Efficient Forecasting
date: 2026-05-08
content_hash: e2b397fb97fe838f
---

# Revitalizing Canonical Pre-Alignment for Irregular Multivariate Time Series Forecasting

**Conference**: AAAI 2026
**arXiv**: [2508.01971](https://arxiv.org/abs/2508.01971)
**Code**: [github.com/zhouziyu02/KAFNet](https://github.com/zhouziyu02/KAFNet)
**Area**: Time Series
**Keywords**: Irregular Multivariate Time Series, Canonical Pre-Alignment, Temporal Kernel Aggregation, Frequency-domain Linear Attention, Efficient Forecasting

## TL;DR

This paper is the first to argue that Canonical Pre-Alignment (CPA) should not be abandoned for Irregular Multivariate Time Series (IMTS) forecasting. It proposes KAFNet, which addresses the efficiency bottleneck of CPA via three modules—Pre-Convolution smoothing, Temporal Kernel Aggregation (TKA), and Frequency-domain Linear Attention (FLA)—achieving state-of-the-art accuracy on 4 IMTS benchmarks while reducing parameters by 7.2× and accelerating training/inference by 8.4×.

## Background & Motivation

Irregular Multivariate Time Series (IMTS) are prevalent in traffic, meteorology, healthcare, and other domains, characterized by:

**Intra-series irregularity**: Non-uniform observation intervals within a single variable.

**Inter-series asynchrony**: Misaligned sampling timestamps across different variables.

**The two sides of Canonical Pre-Alignment (CPA)**:

CPA is the classical preprocessing approach for IMTS—aligning all variables to a unified time grid with zero-padding at missing positions. It resolves inter-series asynchrony, unifies sequence lengths, and facilitates batch training. However, its critical drawback is **sequence length explosion due to zero-padding**: when the number of variables is large, the merged global timestamp set is far larger than the observation count of any individual variable, creating computational and memory bottlenecks.

**Recent trend—bypassing CPA**: Graph neural network methods such as tPatchGNN, GraFITi, and TimeCHEAT directly process raw irregular sequences via patching or graph structures, avoiding CPA's length explosion. However, these methods rely on local message passing and struggle to capture **global cross-variable correlations**—when two variables never co-occur at the same timestamp, messages cannot be directly exchanged.

**Core position of this paper**: CPA should not be abandoned but "revitalized." By resolving its efficiency issues, CPA-based models can surpass state-of-the-art graph models that bypass CPA—a bold and counter-mainstream argument.

## Method

### Overall Architecture

KAFNet consists of four modules:
1. **Pre-Convolution**: Smooths the sparse CPA-aligned sequence and injects temporal embeddings.
2. **Temporal Kernel Aggregation (TKA)**: Compresses long sequences into fixed-length representations using learnable Gaussian kernels.
3. **Frequency Linear Attention (FLA) Blocks**: Models cross-variable dependencies via linear attention in the frequency domain.
4. **Output Layer**: An MLP that generates query-specific forecasts.

### Key Designs

1. **Pre-Convolution Smoothing**: Zero-padding in CPA results in highly uneven information distribution—observed positions contain values while the majority are zero. Feeding such sparse sequences into complex models hampers learning. Two lightweight convolutional layers are first applied to smooth the sequence:

    $\tilde{x}^n = \text{Conv}_{1\times 1}(\sigma(\text{Conv}_{1\times 3}(x^n))) \in \mathbb{R}^L$

   Since the CPA-generated time grid indices no longer reflect true temporal intervals, a continuous temporal embedding is introduced:

    $\text{TE}(t) = [w_s t + b_s \oplus \sin(\mathbf{w}_p t + \mathbf{b}_p) \oplus \cos(\mathbf{w}_c t + \mathbf{b}_c)]$

   The convolutional features and temporal embeddings are summed to yield a time-aware representation $\hat{x}^n = \tilde{x}^n + \mathbf{w}_t^\top \text{TE}(\mathbf{t}^n)$.

   **Design Motivation**: The 3×1 convolution leverages local neighborhood information to "diffuse" observed values into zero-padded positions, mitigating sparsity. Temporal embeddings restore true temporal distance information that CPA destroys.

2. **Temporal Kernel Aggregation (TKA)**: This is the core module addressing CPA's sequence length explosion. Each variable's long sequence $\hat{x}^n \in \mathbb{R}^L$ is compressed into a fixed-length representation of size $K$.

   Concretely, $K$ equally-spaced Gaussian kernels $\{c_k, \sigma_k\}$ with learnable bandwidths $\sigma_k$ are placed on the min-max normalized time axis $[0,1]$. The affinity weight from timestamp $\hat{t}_l^n$ to the $k$-th kernel is:

    $w_{l,k}^n = \exp\left[-\frac{1}{2}(\hat{t}_l^n - c_k)^2 / \sigma_k^2\right] \cdot m_l^n$

   where $m_l^n$ is the CPA mask (ensuring only actual observations contribute). After normalization, a weighted aggregation is performed:

    $h_k^n = \sum_{l=1}^{L} a_{l,k}^n \hat{x}_l^n, \quad a_{l,k}^n = \frac{w_{l,k}^n}{\sum_j w_{j,k}^n}$

   Gating $\tilde{\mathbf{h}}^n = \text{Sigmoid}(\mathbf{g}) \odot \mathbf{h}^n$ followed by linear projection yields $\mathbf{z}^n \in \mathbb{R}^d$.

   **Design Motivation**:
    - Gaussian kernels form a **soft temporal codebook** on the time axis—each kernel covers a temporal region and aggregates observations within it by affinity.
    - The mask mechanism ensures zero-padded positions do not participate in aggregation; only real observations contribute.
    - The compressed length $K$ is independent of the original length $L$, fundamentally resolving CPA's sequence length explosion.
    - Learnable bandwidths allow kernels to adaptively adjust their coverage—narrow kernels for fine-grained modeling in dense regions, wide kernels for smoothing in sparse regions.

3. **Frequency-domain Linear Attention (FLA)**: TKA compresses each variable to $\mathbf{z}^n \in \mathbb{R}^d$, which are concatenated into $\mathbf{Z} \in \mathbb{R}^{N \times d}$. FLA blocks model cross-variable dependencies in the frequency domain:

   rFFT is first applied to $\mathbf{Z}$ to obtain frequency-domain representations $\mathbf{C} \in \mathbb{R}^{N \times 2d_f}$, followed by multi-head attention on $\mathbf{C}$, and irFFT back to the time domain. The key innovation is approximating the softmax kernel via **Random Fourier Features (RFF)** to achieve linear-complexity attention:

    $\phi(\mathbf{x}) = \frac{1}{\sqrt{R}} [\cos(\mathbf{\Omega}^\top \mathbf{x} + \mathbf{b}), \sin(\mathbf{\Omega}^\top \mathbf{x} + \mathbf{b})] \in \mathbb{R}^R$

    $\mathbf{O}^{(h)} = \frac{\phi(\mathbf{Q}^{(h)})(\phi(\mathbf{K}^{(h)})^\top \mathbf{V}^{(h)})}{\phi(\mathbf{Q}^{(h)})(\phi(\mathbf{K}^{(h)})^\top)}$

   Multiple FLA blocks are stacked, each containing attention + FFN + residual connections.

   **Design Motivation**:
    - Frequency-domain transformation more naturally captures periodic and global information, and the $O(Nd\log d)$ complexity of rFFT/irFFT is far lower than full time-domain attention.
    - RFF linearization: standard softmax attention is $O(N^2)$ in the number of variables $N$, which is prohibitive for large $N$; RFF approximation reduces complexity to $O(NR)$.
    - Synergy with CPA: since CPA already aligns all variables to a unified time axis, FLA can directly exchange information across all variables—something graph-based models cannot achieve.

### Loss & Training

$$\mathcal{L} = \frac{1}{N} \sum_{n=1}^{N} \frac{1}{Q_n} \sum_{j=1}^{Q_n} (\hat{x}_j^n - x_j^n)^2$$

Standard MSE loss with the Adam optimizer. The output layer is a 3-layer MLP that concatenates the variable representation with the query temporal embedding to predict a scalar: $\hat{x}_j^n = \text{MLP}(\mathbf{H}^n \oplus \text{TE}(q_j^n))$.

**Computational Complexity**: Total complexity $\Omega = N[(4d+3K)L + Kd + (Q+3)d^2 + 2d(\log d + R)]$, linear in both $L$ and $N$. After TKA compression, FLA and the output layer are entirely independent of sequence length $L$.

## Key Experimental Results

### Main Results

KAFNet is evaluated against 23 baselines on four IMTS datasets: PhysioNet (41 variables), MIMIC (96 variables), Human Activity (12 variables), and USHCN (5 variables):

| Method | PhysioNet MAE(×10⁻²) | MIMIC MSE(×10⁻²) | Human Activity MSE(×10⁻³) | USHCN MAE(×10⁻¹) | Avg. Rank |
|------|----------------------|-------------------|---------------------------|-------------------|----------|
| **KAFNet** | **3.52** | **1.59** | **2.54** | **2.99** | **1.6** |
| tPatchGNN | 3.72 | 1.69 | 2.66 | 3.08 | 2.4 |
| GraFITi* | 3.73 | 1.71 | 2.73 | 3.09 | 3.8 |
| tPatchGNN* | 3.89 | 1.71 | 2.76 | 3.09 | 5.4 |
| Warpformer | 4.21 | 1.73 | 2.79 | 3.23 | 6.9 |
| TimeCHEAT* | 3.89 | 1.70 | 4.06 | 3.10 | 6.8 |
| DLinear | 15.52 | 4.90 | 4.03 | 3.88 | 20.9 |

KAFNet achieves an average rank of 1.6, significantly outperforming all baselines (Friedman + Nemenyi test, $\alpha=0.05$).

### Ablation Study

| Configuration | PhysioNet MSE(×10⁻³) | MIMIC MSE(×10⁻²) | Human Activity MSE(×10⁻³) | USHCN MSE(×10⁻¹) |
|------|----------------------|-------------------|---------------------------|-------------------|
| KAFNet (Full) | **5.88** | **1.59** | **2.54** | **4.98** |
| w/o CPA | 6.21 | 1.69 | 2.70 | 5.04 |
| w/o Pre-Conv | 6.42 | 1.62 | 2.66 | 5.06 |
| w/o T-Norm | 6.37 | 1.73 | 2.66 | 5.14 |
| w/o TKA | 6.95 | 1.74 | 4.21 | 5.07 |
| w/o FLA | 6.26 | 1.79 | 2.71 | 5.23 |
| w/o FLA & w/ SA | 6.08 | 1.67 | 2.57 | 5.20 |

Efficiency comparison (MIMIC dataset):

| Metric | KAFNet | tPatchGNN | GraFITi | TimeCHEAT | HyperIMTS |
|------|--------|-----------|---------|-----------|-----------|
| Parameters | **5K** | 36K | 180K+ | 100K+ | 50K+ |
| FLOPs | **0.38B** | Billions | Billions | Billions | Billions |

KAFNet achieves a 7.2× reduction in parameters and an 8.4× speedup in training and inference.

### Key Findings

1. **CPA is indispensable**: Removing CPA leads to consistent degradation across all datasets, confirming its critical role in mitigating inter-series asynchrony.
2. **TKA is crucial for high-dimensional IMTS**: Without TKA, MSE on Human Activity surges from 2.54 to 4.21 (+66%), as uncompressed long sequences severely impair downstream modeling.
3. **FLA outperforms standard attention**: Replacing FLA with standard softmax attention degrades performance; FLA's frequency-domain transformation combined with RFF approximation achieves superior efficiency and expressiveness.
4. **FLA attention maps exhibit greater dynamic range**: Visualization shows FLA attention scores span nearly the full color scale, whereas standard attention concentrates in a narrow low-value band, indicating that FLA more precisely amplifies or suppresses cross-variable dependencies.

## Highlights & Insights

- **Successful validation of a counter-mainstream argument**: "Revitalizing CPA" is a bold position in an IMTS field dominated by graph models, and the experimental results provide strong support.
- **TKA as a "soft temporal codebook"**: Gaussian kernel aggregation is conceptually analogous to soft attention but significantly more lightweight, and is generalizable to other scenarios requiring irregular temporal modeling.
- **Extreme efficiency**: Surpassing graph models tens of times larger with only 5K parameters demonstrates that the right inductive bias outweighs sheer parameter count.
- **Transferability of Pre-Conv and T-Norm**: These architecture-agnostic designs can be directly adopted by other IMTS models.

## Limitations & Future Work

1. **Limited to forecasting**: Classification, interpolation, and anomaly detection tasks are not evaluated.
2. **Limited dataset domains**: The 4 datasets cover healthcare, biomechanics, and climate, lacking larger-scale traffic and energy scenarios.
3. **Kernel count requires tuning**: Excessive kernels lead to excessive overlap, requiring a trade-off between accuracy and efficiency.
4. **Future directions**: Extension to IMTS classification, interpolation, and anomaly detection; evaluation on large-scale traffic/energy IMTS benchmarks.

## Related Work & Insights

- **tPatchGNN**: The primary baseline. tPatchGNN bypasses CPA via patching, but rigid patches may distort local temporal patterns; KAFNet retains CPA and applies TKA compression, better preserving global alignment information.
- **FiLM / FNet family**: Frequency-domain transformations have been successfully applied in NLP and vision; this paper introduces them into cross-variable modeling for IMTS.
- **Linear attention**: Methods such as Performer reduce attention complexity via kernel approximation; this paper adopts RFF approximation operating in the frequency domain, representing an effective combination.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — The counter-intuitive "revitalize CPA" argument combined with the TKA+FLA three-module design is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ — 23 baselines, comprehensive ablations, efficiency analysis, and attention visualization; dataset coverage could be broader.
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear argumentation, compelling motivation, and well-crafted figures.
- Value: ⭐⭐⭐⭐⭐ — Provides paradigm-shifting evidence in IMTS forecasting with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Learning Recursive Multi-Scale Representations for Irregular Multivariate Time Series Forecasting](../../ICLR2026/time_series/learning_recursive_multi-scale_representations_for_irregular_multivariate_time_s.md)
- [\[AAAI 2026\] HN-MVTS: HyperNetwork-based Multivariate Time Series Forecasting](hn-mvts_hypernetwork-based_multivariate_time_series_forecasting.md)
- [\[AAAI 2026\] Transparent Networks for Multivariate Time Series](transparent_networks_for_multivariate_time_series.md)
- [\[NeurIPS 2025\] Time-IMM: A Dataset and Benchmark for Irregular Multimodal Multivariate Time Series](../../NeurIPS2025/time_series/time-imm_a_dataset_and_benchmark_for_irregular_multimodal_multivariate_time_seri.md)
- [\[AAAI 2026\] Counterfactual Explainable AI (XAI) Method for Deep Learning-Based Multivariate Time Series Classification](counterfactual_explainable_ai_xai_method_for_deep_learning-based_multivariate_ti.md)

</div>

<!-- RELATED:END -->
