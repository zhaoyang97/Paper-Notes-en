---
title: >-
  [Paper Note] TSGDiff: Rethinking Synthetic Time Series Generation from a Pure Graph Perspective
description: >-
  [AAAI 2026][Image Generation][time series generation] This paper proposes TSGDiff, the first framework to rethink time series generation from a purely graph-based perspective. Time series are represented as dynamic graph…
tags:
  - "AAAI 2026"
  - "Image Generation"
  - "time series generation"
  - "graph neural networks"
  - "diffusion models"
  - "Fourier transform"
  - "topological fidelity"
date: 2026-05-08
content_hash: 13ef8c0bc80ccf5f
---

# TSGDiff: Rethinking Synthetic Time Series Generation from a Pure Graph Perspective

**Conference**: AAAI 2026
**arXiv**: [2511.12174](https://arxiv.org/abs/2511.12174)
**Code**: [TSGDiff](https://github.com/jvaeylee/TSGDiff)
**Area**: image_generation (actually time series generation)
**Keywords**: time series generation, graph neural networks, diffusion models, Fourier transform, topological fidelity

## TL;DR

This paper proposes TSGDiff, the first framework to rethink time series generation from a purely graph-based perspective. Time series are represented as dynamic graphs constructed from Fourier spectral features, diffusion modeling is performed in the graph latent space, and a novel Topo-FID metric is introduced to evaluate the structural fidelity of generated time series.

## Background & Motivation

1. **Background**: Multivariate time series generation is in broad demand across energy management, financial forecasting, and medical monitoring. Existing approaches include GAN-based methods (TimeGAN), VAE-based methods (TimeVAE), and diffusion-based methods (Diffusion-TS, CSDI).
2. **Limitations of Prior Work**: (a) Traditional generative models struggle to effectively capture complex spatio-temporal dependencies among variables; (b) they operate under Euclidean space assumptions and cannot represent the inherent topological and structural features of time series data; (c) methods such as Diffusion-TS employ an encoder-decoder transformer that handles spatial and temporal information separately, limiting the modeling of complex inter-variable dependencies; (d) graph-based methods have demonstrated strong performance in forecasting tasks but remain largely unexplored for generation.
3. **Key Challenge**: Time series exhibit rich temporal dependency structures (short-, medium-, and long-term periodicity), yet existing generative methods operate solely in the raw data domain without modeling these dependencies from a structural perspective.
4. **Goal**: To unify the modeling of temporal dependencies and structural relationships in time series from a graph perspective, enabling high-fidelity synthetic time series generation.
5. **Key Insight**: Treating time steps as graph nodes, constructing edges via Fourier spectral features to encode periodic dependencies, and performing diffusion modeling in the graph latent space.
6. **Core Idea**: Fourier transforms are used to discover periodic patterns in time series for graph construction; diffusion-based generation is then conducted in the graph latent representation space, with a topological fidelity metric employed to assess generation quality.

## Method

### Overall Architecture

TSGDiff proceeds in three stages: (1) **Graph Construction and Encoding** — time series are segmented via sliding windows and instance-normalized, Fourier transforms extract periodicity information to build dynamic graphs, and a GNN encoder produces latent representations; (2) **Latent Diffusion Modeling** — a diffusion–denoising process is executed in the latent graph space; (3) **Graph Decoding** — denoised latent representations are decoded back into synthetic time series.

### Key Designs

1. **Fourier-based Graph Construction**:
    - **Function**: Converts time series into graph structures that encode periodic dependencies.
    - **Mechanism**: A discrete Fourier transform $\tilde{X}_i^{(p)} = \sum_{n=0}^{w-1} \tilde{x}_{i+n} \cdot e^{-j\frac{2\pi}{w}pn}$ is applied to each window segment $\tilde{X}_i$; the top-3 frequencies with the highest amplitudes are identified and converted to their corresponding periods. Nodes represent time steps; edges connect adjacent time steps and periodic neighbors. The adjacency matrix entry $\mathbf{A}_{ij}=1$ indicates a temporal or periodic connection.
    - **Design Motivation**: The Fourier spectrum naturally reveals the periodic structure of time series; defining edges via periodicity simultaneously encodes short-, medium-, and long-term dependency patterns.

2. **Graph VAE Encoder-Decoder**:
    - **Function**: Encodes graph structures into latent representations and reconstructs them from latent vectors.
    - **Mechanism**: The encoder stacks multiple GCN layers (graph convolution $y' = \text{Mish}(\text{BN}((W\mathbf{A}\tilde{X}+b)^T)^T)$ with residual connections), applies mean pooling, and outputs $\mu$ and $\log\sigma$; a latent vector is sampled via the reparameterization trick as $z = \mu + \epsilon \odot \exp(\log\sigma / 2)$. The decoder recovers node features through fully connected layers with Tanh activation. A KL divergence loss $\mathcal{L}_{KL}$ constrains the latent distribution toward a standard normal.
    - **Design Motivation**: GCN naturally aggregates information using the adjacency-matrix-defined graph structure; the VAE framework provides a regularized latent space.

3. **Latent Diffusion + Topo-FID Metric**:
    - **Function**: Performs generative modeling in the latent graph space; evaluates the structural fidelity of generated time series.
    - **Mechanism**: The forward process gradually adds noise as $z_k = \sqrt{\alpha_k} \cdot z + \sqrt{1-\alpha_k} \cdot \epsilon$; the reverse process uses an MLP network to predict denoising, taking $\text{concat}(z_k, t_{emb})$ as input. Topo-FID is defined as $\alpha \cdot S_{edit} + (1-\alpha) \cdot S_{entropy}$, where $S_{edit}$ measures adjacency matrix discrepancy and $S_{entropy}$ compares structural entropy differences in node degree distributions.
    - **Design Motivation**: Performing diffusion in the latent space is more efficient than in the raw data domain and preserves structural consistency; conventional metrics cannot capture the structural characteristics of time series, and Topo-FID fills this gap from a graph perspective.

### Loss & Training

The total loss function is:
$$\mathcal{L} = \mathcal{L}_{recon} + \beta \mathcal{L}_{KL} + \gamma \mathcal{L}_{denoising} + \delta \mathcal{L}_{Fourier}$$

- $\mathcal{L}_{recon}$: MSE reconstruction loss between input and decoder output.
- $\mathcal{L}_{KL}$: KL divergence regularization ($\beta=0.2$).
- $\mathcal{L}_{denoising}$: Diffusion denoising loss $\|\epsilon - \epsilon_\theta(z_k, k)\|^2$ ($\gamma=1$).
- $\mathcal{L}_{Fourier}$: Frequency-domain consistency loss $\|\text{FFT}(x_0) - \text{FFT}(\hat{x}_0)\|^2$ ($\delta=1$).

Training configuration: batch size 128, learning rate 0.01, 500 epochs, sliding window size 48, stride 1.

## Key Experimental Results

### Main Results

| Dataset | Metric | TSGDiff | Diffusion-TS | TimeGAN | Gain |
|--------|------|------|----------|------|------|
| ETTh | Topo-FID↑ | **0.986** | 0.864 | 0.798 | +12.2% |
| Weather | Context-FID↓ | **0.353** | 1.161 | 2.420 | -69.6% |
| EEG | Discriminative↓ | **0.301** | 0.492 | 0.391 | -23.0% |
| ETTh | Predictive↓ | **0.020** | 0.026 | 0.030 | -23.1% |
| Stocks | Correlational↓ | **0.026** | 0.027 | 0.061 | -3.7% |
| Wind | Topo-FID↑ | **0.869** | 0.819 | 0.824 | +5.0% |

The average Discriminative score improves by 50% across 6 datasets.

### Ablation Study

| Configuration | Topo-FID↑ | Context-FID↓ | Discriminative↓ | Note |
|------|---------|---------|---------|------|
| w/o KL | 0.787 | 44.467 | 0.500 | Unconstrained latent space collapses |
| w/o Denoising | 0.908 | 3.922 | 0.432 | Limited generative capacity |
| w/o Fourier | 0.883 | 0.407 | 0.061 | Loss of frequency-domain information |
| Full (TSGDiff) | **0.986** | **0.224** | **0.056** | All components synergistically optimal |

### Key Findings

- The KL divergence loss is critical — removing it causes Context-FID to surge from 0.224 to 44.467, indicating complete latent space collapse.
- The Fourier loss has a significant impact on Topo-FID (0.986 → 0.883), demonstrating that frequency-domain constraints are essential for preserving structural fidelity.
- Kernel density estimation visualizations show that TSGDiff-generated distributions more closely match the real data distribution than those of Diffusion-TS.

## Highlights & Insights

- **Perspective Innovation**: This is the first work to approach time series generation from a purely graph-based perspective, modeling temporal dependencies as graph edges — a meaningful paradigm shift.
- **Natural Fusion of Fourier and Graph**: Using Fourier spectral analysis to discover periodicity for constructing graph edges is more physically meaningful than manually defined or fully connected graphs.
- **Novel Topo-FID Metric**: Conventional metrics (FID, Discriminative score, etc.) cannot assess the structural properties of time series; Topo-FID addresses this gap.
- **Unified Framework**: The three-stage design of graph construction + graph VAE + latent diffusion unifies structural modeling and the generative process within a single framework.

## Limitations & Future Work

- Graph construction considers only the top-3 frequency periodicities, potentially overlooking important non-periodic or nonlinear dependencies.
- Fourier transforms are applied independently per variable for edge construction, leaving cross-variable dependencies unmodeled explicitly (nodes represent time steps rather than variables).
- The latent diffusion network uses a simple MLP, which may limit the modeling capacity for complex latent distributions.
- Topo-FID still relies on pairwise comparison with real data, which may not fully capture generation diversity.
- The sliding window size is fixed at 48; sequences at different temporal scales may require adaptive window configurations.

## Related Work & Insights

- **vs. Diffusion-TS**: Diffusion-TS performs diffusion in the raw data domain with seasonal-trend decomposition, processing spatial and temporal information separately; TSGDiff unifies modeling in the graph latent space, enabling better capture of structural dependencies.
- **vs. TimeGAN**: TimeGAN captures temporal dynamics via joint optimization of supervised and adversarial objectives, but is prone to mode collapse; TSGDiff's graph VAE + diffusion design avoids this issue.
- **vs. FourierGNN (forecasting)**: FourierGNN combines Fourier and graph methods for forecasting tasks but has not been applied to generation; TSGDiff successfully extends this idea to the generative setting.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The graph-based perspective on time series generation is novel, and the Topo-FID metric constitutes a valuable contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation across 6 datasets, 5 metric categories, 4 baselines, ablation studies, and visualizations.
- **Writing Quality**: ⭐⭐⭐⭐ Architecture diagrams are clear and mathematical notation is well-defined, though some derivations could benefit from greater detail.
- **Value**: ⭐⭐⭐⭐ Introduces a new modeling paradigm and evaluation metric for time series generation with strong inspirational value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] SimDiff: Simpler Yet Better Diffusion Model for Time Series Point Forecasting](simdiff_simpler_yet_better_diffusion_model_for_time_series_point_forecasting.md)
- [\[ICLR 2026\] Conditionally Whitened Generative Models for Probabilistic Time Series Forecasting](../../ICLR2026/image_generation/conditionally_whitened_generative_models_for_probabilistic_time_series_forecasti.md)
- [\[NeurIPS 2025\] A Diffusion Model for Regular Time Series Generation from Irregular Data with Completion and Masking](../../NeurIPS2025/image_generation/a_diffusion_model_for_regular_time_series_generation_from_irregular_data_with_co.md)
- [\[AAAI 2026\] LongT2IBench: A Benchmark for Evaluating Long Text-to-Image Generation with Graph-structured Annotations](longt2ibench_a_benchmark_for_evaluating_long_text-to-image_generation_with_graph.md)
- [\[AAAI 2026\] UNSEEN: Enhancing Dataset Pruning from a Generalization Perspective](unseen_enhancing_dataset_pruning_from_a_generalization_perspective.md)

</div>

<!-- RELATED:END -->
