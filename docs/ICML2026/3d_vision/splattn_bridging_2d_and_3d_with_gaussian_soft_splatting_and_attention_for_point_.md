---
title: >-
  [Paper Note] SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion
description: >-
  [ICML 2026][3D Vision][Point Cloud Completion] This work identifies that in multimodal point cloud completion, "hard projection of 3D points directly onto 2D grids" leads to a support set with Lebesgue measure zero and g…
tags:
  - "ICML 2026"
  - "3D Vision"
  - "Point Cloud Completion"
  - "Differentiable Gaussian Splatting"
  - "Cross-Modal Entropy Collapse"
  - "Multimodal Learning Theory"
  - "KITTI Counter-factual"
date: 2026-05-08
content_hash: 8369fd92159b3397
---

# SplAttN: Bridging 2D and 3D with Gaussian Soft Splatting and Attention for Point Cloud Completion

**Conference**: ICML 2026  
**arXiv**: [2605.01466](https://arxiv.org/abs/2605.01466)  
**Code**: https://github.com/zay002/SplAttN (available)  
**Area**: 3D Vision / Point Cloud Completion / Multimodal Fusion  
**Keywords**: Point Cloud Completion, Differentiable Gaussian Splatting, Cross-Modal Entropy Collapse, Multimodal Learning Theory, KITTI Counter-factual

## TL;DR
This work identifies that in multimodal point cloud completion, "hard projection of 3D points directly onto 2D grids" leads to a support set with Lebesgue measure zero and gradients truncated by Dirac delta (termed Cross-Modal Entropy Collapse). The authors replace hard projection with differentiable Gaussian Soft Splatting for continuous density estimation, and employ a hybrid encoder combining local EdgeConv and global Transformer, along with a global-local decoder. The method achieves SOTA on PCN/ShapeNet-55/34, and counter-factual evaluation on KITTI demonstrates that baselines actually degenerate into "unimodal template retrievers."

## Background & Motivation

**Background**: Point cloud completion has evolved from early pure geometric methods (PCN, FoldingNet) to multimodal paradigms (SVDFormer, GeoFormer, etc.) that incorporate 2D images as semantic priors. These methods project sparse 3D partial point clouds onto the image plane to query visual features, which are then injected into the completion network. Multimodal learning theory (Lu, 2023) guarantees that as long as "modality heterogeneity" and "modality connection" are satisfied, the generalization bound can be improved by $O(\sqrt{n})$.

**Limitations of Prior Work**: The authors observe that although current SOTA multimodal point cloud completion methods achieve impressive scores on PCN/ShapeNet, the Connection aspect is actually not realized. The reason: projecting sparse 3D point clouds via $\pi: \mathbb{R}^3 \to \mathbb{R}^2$ onto the image plane yields a support set $\mathcal{S}_{hard} = \{\pi(p)\}$ that is a discrete set with Lebesgue measure $\mu(\mathcal{S}_{hard}) = 0$, occupying virtually no pixels.

**Key Challenge**: Under hard projection, the conditional density $P_{hard}(v|\mathcal{P}_{in}) = \tfrac{1}{N}\sum_p \delta(v - \pi(p))$ is a sum of Dirac deltas. By the chain rule, $\nabla_p \mathcal{L} = \tfrac{\partial \mathcal{L}}{\partial v} \cdot \tfrac{\partial v}{\partial \pi(p)} \cdot \nabla_p \delta(v - \pi(p))$, and the derivative of the Dirac delta is almost everywhere zero, preventing visual supervision gradients from flowing back to geometric point positions. Visual features are "attached" to discrete locations with negligible area, so the encoder cannot access dense semantics, and the model degenerates into a "geometric template retriever." The authors term this phenomenon Cross-Modal Entropy Collapse.

**Goal**: (1) Provide a measure-theoretic and information-theoretic formal explanation for the failure of hard projection; (2) Design a lossless projection alternative with nonzero measure support to enable gradient flow; (3) Develop geometric tokens that can actively query visual features; (4) Propose a counter-factual evaluation protocol to verify that the model truly utilizes visual information.

**Key Insight**: Since the problem is the support set having measure zero, each projected point is expanded into a Gaussian disk, so the support set $\mathcal{S}_{soft} = \bigcup_p \{v: \|v - \pi(p)\| < 3\sigma\}$ has positive measure, the density becomes continuous, and gradients are naturally nonzero. This echoes the success of differentiable rendering (Softmax Splatting, 3DGS)—using differentiable density estimation instead of hard sampling.

**Core Idea**: Replace hard projection with differentiable Gaussian Soft Splatting, and use a hybrid tokenization of EdgeConv (for local curvature) and Transformer (for global topology) to actively query the continuous visual density field, followed by a global-local decoder for coarse-to-fine generation.

## Method

### Overall Architecture

The input is a sparse partial point cloud $\mathcal{P}_{in} = \{p_i\}_{i=1}^N \subset \mathbb{R}^3$ and the corresponding RGB image $\mathcal{I}$; the output is the completed dense point cloud $\mathcal{P}_{out}$. The overall structure is dual-branch with a hierarchical decoder: (a) The GS-Bridge branch uses EdgeConv to extract local geometric tokens, passes them through a Transformer to obtain global geometric queries $\mathcal{F}_{geo}$, and applies Gaussian Soft Splatting to convert the visual feature map into a continuous density field $\mathcal{V}$. Cross-attention enables $\mathcal{F}_{geo}$ to actively query $\mathcal{V}$, yielding fused global features $\mathcal{F}_g$; (b) In parallel, the Local Encoder uses EdgeConv and Multi-Head Self-Attention to obtain topology-aware local features $\mathcal{F}_l$; (c) The Global-Local Decoder predicts a skeleton $\mathcal{P}_0$ from $\mathcal{F}_g$ via MLP (injecting input priors via $\mathcal{P}_{in}$-Merge), then hierarchically upsamples $\mathcal{P}_0 \to \mathcal{P}_1 \to \mathcal{P}_2$, with each upsampling stage employing Structure Self-Attention for geometric consistency and Cross-Attention to inject $\mathcal{F}_l$ for detail refinement.

### Key Designs

1. **Gaussian Soft Splatting as a Replacement for Hard Projection**:

    - **Function**: Diffuses discrete projected points $\pi(p)$ into continuous Gaussian disks, so that visual features on the image plane are defined for every sub-pixel query $\mathbf{q}$.
    - **Mechanism**: Defines soft density $P_{soft}(v|\mathcal{P}_{in}) = \tfrac{1}{N}\sum_p \alpha_p \mathcal{G}(v; \pi(p), \sigma)$; for any query $\mathbf{q}$, aggregates features as $\mathcal{V}(\mathbf{q}) = \tfrac{\sum_{k \in \mathcal{N}(\mathbf{q})} w_k(\mathbf{q}) f_k}{\sum_k w_k + \epsilon}$, where weights are spatial Gaussian kernel × inverse depth: $w_k(\mathbf{q}) = \exp(-\tfrac{\|\mathbf{u}_k - \mathbf{q}\|^2}{2\sigma^2}) \cdot (z_k + \epsilon)^{-1}$. The former acts as a low-pass filter, the latter as a soft Z-buffer for occlusion approximation. $f_k$ is instantiated using normalized 3D coordinates as pseudo-color.
    - **Design Motivation**: By measure subadditivity, $\mu(\mathcal{S}_{soft}) > 0$, so the density field has positive measure support, and the gradient $\nabla_{\mathbf{u}} \mathcal{L}$ is nonzero, allowing backpropagation to update geometric coordinates. The Gaussian tail ensures that even slight misalignments produce gradient signals, fundamentally addressing Entropy Collapse. The soft Z-buffer avoids the non-differentiability of hard z-buffering.

2. **Hybrid Geometric Tokenization with EdgeConv + Transformer**:

    - **Function**: Generates geometric queries $\mathcal{F}_{geo}$ that capture both local curvature and global topology, enabling the model to represent thin structures (e.g., chair legs) and reason about global invariants such as symmetry and holes.
    - **Mechanism**: EdgeConv dynamically constructs a k-NN graph on $\mathcal{P}_{in}$, $\mathbf{h}_i = \max_{j \in \mathcal{N}(i)} \phi_\theta(p_i, p_j - p_i)$, where the max-aggregation approximates the Laplace-Beltrami operator, thus capturing tangent spaces and mean curvature. This is followed by a Transformer encoder, where self-attention models global message passing for holes and symmetry.
    - **Design Motivation**: Purely local operators are blind to global topology, while pure Transformers lose local precision. The hybrid architecture achieves dual geometric invariance—local isometry and global homeomorphism—which is essential for GS-Bridge to query visual features at the correct granularity.

3. **Active Cross-Modal Alignment (Active Attention) + Global-Local Decoding**:

    - **Function**: Enables geometric tokens to actively query semantic information from the visual density field $\mathcal{V}$ (rather than passive concatenation), and at each decoder upsampling stage, uses Structure Self-Attention and Cross-Attention to inject local features for detail refinement.
    - **Mechanism**: Cross-modal alignment is performed as $\mathcal{F}_g = \mathcal{F}_{geo} + \mathrm{Softmax}(\tfrac{(\mathcal{F}_{geo} \mathbf{W}_Q)(\mathcal{V} \mathbf{W}_K)^\top}{\sqrt{d}})(\mathcal{V} \mathbf{W}_V)$, equivalent to "differentiable dictionary querying." The decoder uses Chamfer Distance as a local reconstruction uncertainty proxy, projecting this geometric error into a high-dimensional embedding so that self-attention densifies features in high-entropy (missing) regions; cross-attention then injects high-curvature local information from $\mathcal{F}_l$.
    - **Design Motivation**: Active querying allows the model to selectively absorb semantic priors and suppress background noise, while the structure-aware decoder focuses generation on missing regions rather than uniform global generation, which is the key inductive bias of hierarchical upsampling.

### Loss & Training

Uses Weighted Arc-CD: $\mathcal{L}_{\mathrm{warc}}(X, Y; \lambda) = \lambda \cdot \mathrm{arccosh}(1 + \mathcal{L}_{\mathrm{CD}}(X, Y))$, where the nonlinearity of arccosh naturally compresses outliers while preserving detail sensitivity. The three stage losses are summed equally: $\mathcal{L}_{total} = \mathcal{L}_{\mathrm{warc}}(\mathcal{P}_0, \mathbf{P}_{gt}) + \sum_{k=1}^{2} \mathcal{L}_{\mathrm{warc}}(\mathcal{P}_k, \mathbf{P}_{gt})$. Implemented in PyTorch with AdamW and one-cycle cosine scheduling, trained on 4 × RTX 4090, Gaussian kernel size = 4.

## Key Experimental Results

### Main Results

| Dataset | Metric | Ours SplAttN | SVDFormer | GeoFormer | AdaPoinTr |
|---------|--------|--------------|-----------|-----------|-----------|
| PCN | CD-Avg ↓ ($\times 10^3$) | **6.36** | 6.54 | 6.42 | 6.53 |
| PCN | F1 ↑ | **0.854** | 0.841 | 0.853 | 0.845 |
| PCN | DCD ↓ | **0.523** | 0.536 | 0.526 | – |
| ShapeNet-55 | CD-Avg ↓ ($\times 10^3$) | **0.77** | 0.82 | – | – |
| ShapeNet-55 | F1 ↑ | **0.520** | 0.444 | – | – |
| ShapeNet-34 (seen) | CD-Avg ↓ | **0.65** | 0.75 | – | 0.73 |
| ShapeNet-34 (21 unseen) | CD-Avg ↓ | **1.22** | 1.28 | – | 1.23 |
| ShapeNet-34 (21 unseen) | F1 ↑ | **0.481** | 0.427 | – | 0.416 |

### Ablation Study

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| Full SplAttN | PCN CD 6.36, ShapeNet-55 F1 0.520 | Complete method |
| Hard projection replaces GSS | Significant degradation | Causal evidence for "hard projection → entropy collapse" |
| EdgeConv only, no Transformer | Global topology damaged | Validates necessity of hybrid tokens |
| Passive concat replaces Active Attention | Cross-modal dependency drops | Validates importance of "active query" |

(Detailed ablation numbers in Section 4.3 of the paper; authors emphasize that removing any single component triggers performance regression.)

### Key Findings

- **SOTA across 6 benchmarks**: PCN/ShapeNet-55/34 (seen and unseen) and KITTI real-world scenarios. F1 improvement on unseen classes is especially notable (0.481 vs 0.427 for SVDFormer), confirming the generalization benefit after truly establishing Connection.
- **KITTI counter-factual evaluation is a highlight**: The authors directly run models trained on PCN on 2401 real car instances from KITTI, proposing the Semantic Consistency Score—removing the visual input (feeding a null image) and observing output changes. Baselines show almost no change (indicating degeneration into "geometry → template" unimodal retrievers), while SplAttN outputs change significantly (indicating true visual involvement in reconstruction). This turns "we use vision" from a claim into a quantifiable metric.
- **Significant improvement on thin structures (chair legs, lamps)**: Qualitative results show SplAttN recovers thin structures much better than SOTA, consistent with the design motivation of "hybrid encoder capturing both local curvature and global topology."
- **DCD improvement exceeds CD**: DCD is more sensitive to density distribution, indicating SplAttN not only brings points closer on average, but also better matches the true manifold's density.

## Highlights & Insights

- Elevates the engineering phenomenon of "hard projection failure" to a measure-theoretic proposition: "Lebesgue measure zero leads to Dirac delta gradient truncation"—mapping deep learning phenomena to measure theory is rare and provides a clean theoretical foundation for the work.
- The "KITTI counter-factual evaluation" is the most valuable methodological contribution: it exposes that many multimodal point cloud completion methods do not truly use visual information. This stress test paradigm can be extended to any multimodal fusion task (VLM, 3D Detection, etc.).
- After the rise of Gaussian Splatting in 3D rendering, the authors cleverly reverse its core idea (differentiable density estimation) back to the 2D projection plane.
- The soft Z-buffer uses the $1/(z + \epsilon)$ term to maintain differentiability while approximating occlusion—a practical and effective engineering trick.

## Limitations & Future Work

- The Gaussian kernel size $\sigma$ is fixed (kernel size = 4), not learnable or adaptive; in scenes with large density variation (e.g., near and far objects), a fixed $\sigma$ may be either too blurry or fail to cover enough area.
- The quality of visual features (RGB) is assumed to be high; if the input image has severe occlusion, noise, or modality mismatch (e.g., LiDAR + nighttime RGB), soft splatting may amplify noise.
- KITTI counter-factual evaluation is only conducted on the "car" class, not across multiple classes. The Semantic Consistency Score threshold of 0.7 is empirical.
- The entire architecture is heavier at inference than pure geometric methods (GS-Bridge + Transformer + Decoder); the paper does not provide latency comparisons. Real-time applications (e.g., robotic perception) require further pruning/distillation.
- The theoretical PMI analysis is placed in Appendix §C.1, with the main text only providing an explanation at the density estimation level. If the PMI lower bound could be linked to actual reward/accuracy, the theory would be more robust.

## Related Work & Insights

- **vs SVDFormer (Zhu et al. 2023b) / GeoFormer (Yu et al. 2024)**: These also use "3D point queries visual" but rely on hard projection, so the models degenerate into unimodal baselines; SplAttN uses Gaussian Splatting to repair Connection, enabling effective gradient paths with the same query paradigm.
- **vs Softmax Splatting (Niklaus & Liu 2020)**: Originally for video frame interpolation, this work adapts its idea to point cloud-image projection.
- **vs 3DGS (Kerbl et al. 2023)**: 3DGS uses Gaussian rendering for images, while SplAttN reverses this by projecting image features into 3D query space via Gaussians—a "reverse use" of rendering tools.
- **vs Pure Geometric methods (PoinTr, SeedFormer)**: Pure geometric methods do not require images and already achieve CD of 6.7~6.9 on PCN; SplAttN achieves 6.36, indicating that the marginal benefit of introducing vision is real, but only if Connection is effective—precisely the core argument of the paper.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The formalization of entropy collapse from a measure-theoretic + Dirac delta perspective is unprecedented in point cloud completion; KITTI counter-factual evaluation is also a paradigm-level contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ PCN + ShapeNet-55/34 (seen+unseen) + KITTI real-world scenarios, complete benchmarks; lacks inference latency and $\sigma$ sensitivity analysis.
- Writing Quality: ⭐⭐⭐⭐ Mathematical notation is clear, Figures 1/2/4 are information-dense; the "Cross-Modal Entropy Collapse" motivation is compelling, though the multimodal learning theory citation is somewhat lengthy.
- Value: ⭐⭐⭐⭐ Not only proposes a new method, but also exposes hidden bugs in SOTA (pseudo-multimodal), likely to advance evaluation standards for multimodal 3D tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Rethinking Multimodal Point Cloud Completion: A Completion-by-Correction Perspective](../../AAAI2026/3d_vision/rethinking_multimodal_point_cloud_completion_a_completion-by-correction_perspect.md)
- [\[AAAI 2026\] DAPointMamba: Domain Adaptive Point Mamba for Point Cloud Completion](../../AAAI2026/3d_vision/dapointmamba_domain_adaptive_point_mamba_for_point_cloud_completion.md)
- [\[AAAI 2026\] DANCE: Density-Agnostic and Class-Aware Network for Point Cloud Completion](../../AAAI2026/3d_vision/dance_density-agnostic_and_class-aware_network_for_point_cloud_completion.md)
- [\[AAAI 2026\] Simba: Towards High-Fidelity and Geometrically-Consistent Point Cloud Completion via Transformation Diffusion](../../AAAI2026/3d_vision/simba_towards_high-fidelity_and_geometrically-consistent_point_cloud_completion_.md)
- [\[ICCV 2025\] Revisiting Point Cloud Completion: Are We Ready For The Real-World?](../../ICCV2025/3d_vision/revisiting_point_cloud_completion_are_we_ready_for_the_real-world.md)

</div>

<!-- RELATED:END -->
