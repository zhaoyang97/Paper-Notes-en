---
title: >-
  [Paper Note] TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling
description: >-
  [AAAI 2026 Oral][Spiking Neural Networks] This paper proposes Topographic Deep Spiking Neural Networks (TDSNNs), which introduce a Spatiotemporal Constraint (STC) loss to successfully replicate the hierarchical topographic organization of the primate visual cortex from V1 to IT in deep SNNs, achieving zero accuracy degradation on ImageNet (top-1) while substantially outperforming existing topographic ANNs in brain similarity.
tags:
  - "AAAI 2026 Oral"
  - "Spiking Neural Networks"
  - "Topographic Organization"
  - "Visual Cortex Modeling"
  - "Spatiotemporal Constraints"
  - "Biological Plausibility"
date: 2026-05-08
content_hash: 9be0f266cb933a60
---

# TDSNNs: Competitive Topographic Deep Spiking Neural Networks for Visual Cortex Modeling

**Conference**: AAAI 2026 Oral  
**arXiv**: [2508.04270](https://arxiv.org/abs/2508.04270)  
**Code**: None  
**Area**: Spiking Neural Networks / Computational Neuroscience
**Keywords**: Spiking Neural Networks, Topographic Organization, Visual Cortex Modeling, Spatiotemporal Constraints, Biological Plausibility

## TL;DR

This paper proposes Topographic Deep Spiking Neural Networks (TDSNNs), which introduce a Spatiotemporal Constraint (STC) loss to successfully replicate the hierarchical topographic organization of the primate visual cortex from V1 to IT in deep SNNs, achieving zero accuracy degradation on ImageNet (top-1) while substantially outperforming existing topographic ANNs in brain similarity.

## Background & Motivation

**Background**: The primate visual cortex exhibits topographic organization—neurons with similar functional tuning cluster spatially, ranging from orientation/spatial-frequency/color selectivity in V1 to category selectivity (e.g., spatial clustering of face and body regions) in IT. Deep learning models such as TopoNet and TDANN have been applied to model this topographic structure by incorporating wiring-cost auxiliary losses to induce topographic features.

**Limitations of Prior Work**: Existing topographic ANN models suffer from two critical issues: (1) significant performance degradation on classification tasks (TopoNet drops 3% on ImageNet; LLCNN-G drops up to 16.57%); and (2) complete neglect of the temporal dimension—temporal dynamics are a fundamental characteristic of biological visual processing, yet ANNs cannot inherently capture temporal information.

**Key Challenge**: Temporal dynamics are central to biological neural systems, yet existing topographic models are either ANNs that ignore time, or SNNs limited to shallow architectures (e.g., the two-layer SESNN), making it impossible to simultaneously achieve hierarchical topographic organization and high task performance in deep networks.

**Goal**: (1) How to induce hierarchical topographic organization from V1 to IT in deep SNNs? (2) How to leverage the temporal dynamics of SNNs to mitigate performance degradation caused by topographic constraints? (3) How does topographic organization alter the temporal information processing mechanisms of SNNs?

**Key Insight**: SNNs inherently possess spike-based temporal dynamics. The authors design the STC loss to constrain both long-timescale (firing rate) and short-timescale (spike synchrony) responses, exploiting the temporal coding advantages of SNNs to compensate for the performance cost of topographic constraints.

**Core Idea**: Apply a Spatiotemporal Constraint (STC) loss to jointly optimize spatial topology and temporal synchrony in deep SNNs, achieving visual cortex topographic modeling with zero performance degradation.

## Method

### Overall Architecture

TDSNNs are constructed in three stages: (1) each layer's neurons are mapped onto a virtual 2D cortical sheet, assigning a physical coordinate to each LIF neuron; (2) a pre-optimization step rearranges neuron positions so that functionally similar neurons are spatially proximal; (3) the network is trained from scratch using a joint objective combining task loss and STC loss. The input is an image and the output is a classification result, while each layer spontaneously develops topographic features analogous to V1 (orientation, spatial frequency, color selectivity) and IT (category selectivity).

### Key Designs

1. **Cortical Sheet Mapping**:

    - Function: Non-uniformly embeds neurons of each SNN layer into a 2D physical space to simulate the spatial layout of the biological visual cortex.
    - Mechanism: For an SNN layer of dimension $(C, H, W)$, each unit $u_{c,h',w'}$ is assigned a unique coordinate $(x, y)$ on a cortical sheet of size $h \times w$ (in mm) via an injective mapping $\mathcal{M}$. The cortical sheet size for each layer is set according to the corresponding visual area (e.g., 36.75 mm for V1, 70.0 mm for IT), with neighborhood widths adjusted accordingly.
    - Design Motivation: Provides a spatial distance foundation for the subsequent STC loss while maintaining scale correspondence with the biological visual cortex.

2. **Spatiotemporal Constraint (STC) Loss**:

    - Function: Encourages spatially neighboring LIF neurons to exhibit similar response patterns at both long timescales (firing rate) and short timescales (spike synchrony).
    - Mechanism: STC consists of two components. The long-timescale loss is $\mathcal{L}_L = \frac{1}{2}(1 - P(\mathbf{r}, \mathbf{d}))$, where $\mathbf{r}$ is a vector of Pearson correlation coefficients of firing rates between neuron pairs and $\mathbf{d}$ is a vector of inverse spatial distances. The short-timescale loss is $\mathcal{L}_S = \frac{1}{2}(1 - P(\mathbf{r_{CCG}}, \mathbf{d}))$, which computes spike-timing synchrony $r_{CCG}(i,j)$ via a cross-correlogram (CCG) within a time window $[-W, W]$ with autocorrelation normalization. The total loss is $\mathcal{L} = \mathcal{L}_{task} + \frac{1}{M}\sum_{k,m}[\alpha \mathcal{L}_L + \beta \mathcal{L}_S]$.
    - Design Motivation: The biological visual cortex simultaneously employs long-timescale rate-based representations and short-timescale spike-synchrony-based representations. Using only the long-timescale constraint ($\beta=0$) effectively reduces the SNN to a rate-coding regime similar to ANNs; $\mathcal{L}_S$ acts as a spike-timing regularizer that substantially enhances temporal coding capacity.

3. **Neuron Position Pre-optimization**:

    - Function: Establishes a preliminary spatial organization for neurons on the cortical sheet prior to formal training.
    - Mechanism: An auxiliary SNN is first pretrained with BPTT, and its responses to sinusoidal grating stimuli are generated. Neuron positions are then refined by randomly swapping positions (500 swap attempts per neighborhood × 20,000 independent samples) to bring functionally similar neurons closer. The pretrained weights are discarded after position initialization.
    - Design Motivation: Due to parameter sharing in convolutional layers, neurons in large-scale SNN layers cannot spontaneously develop topographic structure—multiple units sharing the same filter parameters cause model updates to affect multiple neurons simultaneously.

### Loss & Training

The total loss is the weighted sum of the cross-entropy task loss and the STC loss, with coefficients $\alpha$ and $\beta$ controlling the strength of topographic constraints. The STC loss is approximated by randomly sampling $M=10$ fixed-size neuron clusters per layer at each step to reduce computational overhead. The time constant in the CCG is set proportional to the total timestep count $T$ (i.e., $\tau = T/2 - 1$ or $T/2$). End-to-end training is performed via BPTT with surrogate gradients. TSResnet18 is trained on ImageNet for 300 epochs using the AdamW optimizer (base learning rate 5e-4, cosine decay).

## Key Experimental Results

### Main Results

| Metric | TSResnet18 | SResnet18 (non-topo) | TopoNet (ANN) | TDANN (ANN) |
|--------|-----------|---------------------|--------------|------------|
| ImageNet Top-1 (%) | 58.34 (α50β50) / 58.72 (α10β90) | 58.49 | ~55.5 (−3%) | — |
| V1 BrainScore | 0.6845 | 0.6823 | 0.7116 | 0.6932 |
| IT BrainScore | **0.7127** | 0.7102 | 0.5723 | 0.4259 |
| V4 BrainScore | 0.3886 | **0.3970** | 0.2923 | 0.2792 |

### Ablation Study

| Configuration | ImageNet Acc (%) | V1 Smoothness | Notes |
|--------------|-----------------|--------------|-------|
| SResnet18 (non-topo) | 58.49 | 0.5555 | Baseline |
| TSResnet18 α10-β10 | 58.53 | 0.6839 | Topo + mild constraint |
| TSResnet18 α10-β90 | **58.72** | 0.6991 | Highest accuracy |
| TSResnet18 α50-β0 | 58.21 | 0.7550 | No short-timescale constraint |
| TSResnet18 α50-β50 | 58.34 | **0.7674** | Highest smoothness |

### Key Findings

- The short-timescale loss $\mathcal{L}_S$ is critical: its inclusion improves smoothness from 0.755 to 0.7674 and accuracy from 58.21% to 58.34%—it not only enhances topographic quality but also improves classification performance.
- TDSNNs also exhibit topographic gains on CIFAR-100 (non-topo 73.01% → topo 73.97%) and show no accuracy degradation on Spikformer.
- The spatial overlap between face- and body-selective regions in the IT layer is 0.63 for TSResnet18 vs. only 0.15 for SResnet18, consistent with the face–body co-localization observed in the primate visual cortex.
- Robustness experiments show that TSResnet18 outperforms non-topographic SResnet18 under four attack types (e.g., PGD: 10.7% vs. 9.97%), indicating that topographic organization enhances decision robustness.
- Fisher information analysis reveals a topology-driven information hierarchy: early layers (V1/V2) stably preserve signal fidelity, V4 substantially amplifies discriminative features, and IT reduces Fisher information for stable encoding.

## Highlights & Insights

- **SNN temporal coding advantage compensates for topographic cost**: This is the paper's central insight. Introducing topographic constraints in ANNs inevitably degrades performance by restricting spatial degrees of freedom; SNNs compensate for this spatial constraint through the additional degrees of freedom in the temporal dimension, achieving zero accuracy loss.
- **Dual-timescale design of the STC loss is elegant**: By unifying rate coding and temporal coding within a single loss function—combining long-timescale functional clustering with short-timescale synchrony constraints—the design directly corresponds to the two information encoding modes observed in biological neural systems.
- **Topology-induced reshaping of the information hierarchy**: Topographic organization is found to primarily reshape deep connectivity (V4 and IT) rather than shallow layers, consistent with the biological visual system's property of progressively increasing topographic complexity from low-level to high-level areas.

## Limitations & Future Work

- Validation is limited to ResNet18, CORnet, and Spikformer; extension to larger architectures (e.g., ResNet50, ViT-Large) is not explored, with computational cost being the primary bottleneck (BPTT training of deep SNNs is highly resource-intensive).
- Although LIF neurons are efficient, more complex models such as Hodgkin–Huxley or FIF neurons may provide greater biological fidelity.
- Only feedforward and local lateral connections are considered; long-range connections and the diversity of excitatory–inhibitory neuron populations are absent.
- The number of timesteps is limited (feedforward SNNs use only 4 steps; recurrent SNNs use only 10), precluding exploration of temporal dynamics over longer time windows.

## Related Work & Insights

- **vs. TopoNet (deb2025toponets)**: TopoNet balances topology and performance via neural pruning but still drops 3% on ImageNet; TDSNNs eliminate performance degradation entirely through temporal coding and substantially outperform TopoNet in IT BrainScore (0.71 vs. 0.57).
- **vs. SESNN (zhong2024emergence)**: SESNN reproduces V1 orientation preference maps only in two-layer networks, with topography disappearing in deeper layers; TDSNNs are the first to maintain topographic organization across all visual hierarchical levels (V1 to IT).
- **vs. TDANN (margalit2024unifying)**: As a representative topographic ANN, TDANN achieves substantially lower BrainScores than TDSNNs in V2, V4, and IT, highlighting the importance of temporal dynamics for brain similarity.

## Rating

- Novelty: ⭐⭐⭐⭐ — First work to achieve complete hierarchical visual cortex topographic modeling in deep SNNs; the dual-timescale STC design has theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers V1/IT topographic analysis, BrainScore, performance–topology trade-offs, robustness, and Fisher information across multiple evaluation dimensions.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured with thorough biological background; some notation could be simplified.
- Value: ⭐⭐⭐⭐ — Provides a new perspective for the intersection of computational neuroscience and deep learning, though engineering applicability is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Training Deep Normalization-Free Spiking Neural Networks with Lateral Inhibition](../../ICLR2026/others/training_deep_normalization-free_spiking_neural_networks_with_lateral_inhibition.md)
- [\[AAAI 2026\] I2E: Real-Time Image-to-Event Conversion for High-Performance Spiking Neural Networks](i2e_real-time_image-to-event_conversion_for_high-performance_spiking_neural_netw.md)
- [\[ICLR 2026\] Breaking Gradient Temporal Collinearity for Robust Spiking Neural Networks](../../ICLR2026/others/breaking_gradient_temporal_collinearity_for_robust_spiking_neural_networks.md)
- [\[ICLR 2026\] Beyond Linear Processing: Dendritic Bilinear Integration in Spiking Neural Networks](../../ICLR2026/others/beyond_linear_processing_dendritic_bilinear_integration_in_spiking_neural_networ.md)
- [\[ICLR 2026\] Online Pseudo-Zeroth-Order Training of Neuromorphic Spiking Neural Networks](../../ICLR2026/others/online_pseudo-zeroth-order_training_of_neuromorphic_spiking_neural_networks.md)

</div>

<!-- RELATED:END -->
