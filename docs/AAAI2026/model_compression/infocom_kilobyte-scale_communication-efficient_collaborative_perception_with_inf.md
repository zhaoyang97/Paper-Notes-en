---
title: >-
  [Paper Note] InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression
description: >-
  [AAAI 2026][Model Compression][Collaborative Perception] This paper proposes InfoCom, a framework that applies an extended information bottleneck (IB) principle to compress the communication payload of collaborative perc…
tags:
  - "AAAI 2026"
  - "Model Compression"
  - "Collaborative Perception"
  - "Communication Efficiency"
  - "Information Bottleneck"
  - "Feature Compression"
  - "3D Object Detection"
date: 2026-05-08
content_hash: 5810cdf48de9a264
---

# InfoCom: Kilobyte-Scale Communication-Efficient Collaborative Perception with Information-Aware Feature Compression

**Conference**: AAAI 2026
**arXiv**: [2512.10305](https://arxiv.org/abs/2512.10305)  
**Code**: [GitHub](https://weiquanmin.github.io/infocom)  
**Area**: Model Compression
**Keywords**: Collaborative Perception, Communication Efficiency, Information Bottleneck, Feature Compression, 3D Object Detection

## TL;DR

This paper proposes InfoCom, a framework that applies an extended information bottleneck (IB) principle to compress the communication payload of collaborative perception from the MB scale to the KB scale—a 440× reduction compared to Where2comm—while maintaining near-lossless perception performance. The framework consists of three core modules: information-aware encoding, sparse mask generation, and multi-scale decoding.

## Background & Motivation

Collaborative perception compensates for the limitations of single-agent perception (occlusion, long range, etc.) through multi-agent information sharing, and is a critical enabler of autonomous driving safety. However, existing collaborative perception methods face a fundamental communication–performance trade-off.

Existing communication-efficient methods fall into two categories: (1) **feature selection methods** (e.g., Where2comm), which selectively transmit salient features but still operate at the MB scale due to high feature dimensionality; and (2) **feature compression methods** (e.g., ERMVP), which map features to a lower-dimensional space, achieving some reduction but still far exceeding practical network constraints. The root cause is that these methods all assume MB-level bandwidth availability, whereas the average throughput of 5G V2X networks is only 3.5 MB/s and can drop below 0.4 MB/s, rendering MB-level communication unreliable in practice.

More fundamentally, existing methods lack a **theoretical analysis** of the communication–performance trade-off and rely solely on heuristic design. Motivated by an information-theoretic perspective, this paper proposes an **information purification paradigm**: rather than operating in the feature space, the method directly extracts the minimal sufficient task-relevant information using the IB principle. However, the direct application of the standard IB is constrained by the data processing inequality $I(Z;Y) \leq I(X;Y)$, which creates an inherent tension between extreme compression and high-accuracy perception. InfoCom overcomes this limitation by extending the IB Markov chain and introducing a sparse mask as auxiliary information.

## Method

### Overall Architecture

InfoCom is a plug-and-play communication-efficient collaborative perception framework that requires only replacing the communication layer of an existing collaborative system and adding an IB regularization term to the training loss. The pipeline proceeds as follows: each agent extracts intermediate BEV features $Z$ via a local encoder → information-aware encoding compresses $Z$ into an ultra-low-dimensional message $E$ → sparse mask generation identifies spatial cues $M$ → the pair $\{E, M\}$ (KB-scale) is transmitted → the receiver reconstructs actionable BEV features via multi-scale decoding → a fusion network aggregates multi-view information → 3D detection results are produced.

### Key Designs

1. **Information-Aware Encoding (IAE)**:

    - *Function*: Compresses high-dimensional intermediate features $Z \in \mathbb{R}^{N \times C \times H \times W}$ into ultra-low-dimensional information-aware features $E \in \mathbb{R}^{N \times D}$, where $D \ll C \times H \times W$.
    - *Mechanism*: The standard IB Markov chain $Y \to X \to Z$ is extended to $Y \to X \to Z \to (E, M)$, decoupling spatial cues as auxiliary information $M$ so that $E$ focuses on retaining perception-critical information. The optimization objective is $E, M = \arg\min_{E,M} -I(E,M;Y) + \beta I(E,M;Z)$. The IAEncoder outputs Gaussian parameters $(\mu_i, \sigma_i)$, from which $E_i = \mu_i + \sigma_i \odot \epsilon_i$ is sampled via the reparameterization trick.
    - *Design Motivation*: The standard IB can only trade off between sufficiency and minimality and cannot achieve extreme compression and high accuracy simultaneously. By introducing an ultra-low-dimensional feature space and an auxiliary mask, the inherent tension in the IB is decomposed into two sub-problems.

2. **Sparse Mask Generation (SMG)**:

    - *Function*: Provides critical spatial prior information at negligible communication cost to compensate for information loss under extreme compression.
    - *Mechanism*: A spatial importance mask $M_i \in \mathbb{R}^{H \times W}$ is generated via multi-scale convolutions ($3 \times 3$, $5 \times 5$, $7 \times 7$) and a projection layer, followed by two-step joint compression post-processing: (1) *filtering*—retaining only the top-$k$ salient locations, $k = \lfloor \alpha \cdot HW \rfloor$, $\alpha = 0.1$; (2) *quantization*—uniformly quantizing to $b = 4$ bit precision. Gradients are propagated via the straight-through estimator (STE).
    - *Design Motivation*: The data processing inequality implies that a higher compression ratio $D/(C \times H \times W)$ increases the risk of task-relevant information loss. However, extreme compression of $E$ frees up bandwidth for transmitting auxiliary spatial cues. Experiments confirm that only 10% of spatial locations provide the majority of the benefit, and high-precision representation is unnecessary.

3. **Multi-Scale Decoding (MSD)**:

    - *Function*: Progressively reconstructs actionable BEV features from the heavily compressed message $\{E, M^q\}$ at the receiver.
    - *Mechanism*: A three-step pipeline: (1) *feature initialization*—$E$ is expanded into a low-resolution feature map $F_{init}^0 \in \mathbb{R}^{C^0 \times H^0 \times W^0}$ via fully connected layers and transposed convolutions; (2) *mask-guided modulation*—the downsampled mask is element-wise multiplied with the initial features, $F^0 = F_{init}^0 \odot M^0$, directing reconstruction toward task-critical regions; (3) *multi-scale reconstruction*—cascaded decoding blocks progressively upsample, doubling resolution and halving channels at each step, reaching the target resolution after $K$ iterations.
    - *Design Motivation*: Unlike simple feature reconstruction, MSD focuses on recovering perceptual information rather than the complete feature map; mask guidance concentrates reconstruction resources on critical regions.

### Loss & Training

The total loss is $\mathcal{L} = \mathcal{L}_{detect} + \beta \text{KL}(p(E|Z) \| r(E))$, where the detection loss $\mathcal{L}_{detect}$ implicitly maximizes $I(E,M;Y)$, and the KL divergence term has a closed-form solution under the Gaussian prior $r(E) = \mathcal{N}(0, I)$, controlling $I(E;Z)$. Joint optimization of both terms achieves noise suppression, which is theoretically guaranteed by Lemma 1: $I(E,M;Y_N) \leq I(E,M;Z) - I(E,M;Y)$.

## Key Experimental Results

### Main Results

| Dataset | Method | Bandwidth | AP@50 | AP@70 |
|---------|--------|-----------|-------|-------|
| OPV2V | Standard Colla. | 34.375 MB | 0.9653 | 0.9229 |
| OPV2V | Where2comm | 3.439 MB | 0.9463 | 0.8820 |
| OPV2V | ERMVP | 0.741 MB | 0.9557 | 0.9127 |
| OPV2V | **InfoCom** | **7.875 KB** | **0.9650** | **0.9202** |
| V2XSet | Standard Colla. | 34.375 MB | 0.9212 | 0.8426 |
| V2XSet | ERMVP | / | OOM | OOM |
| V2XSet | **InfoCom** | **7.875 KB** | **0.9273** | **0.8488** |
| DAIR-V2X | Standard Colla. | 24.609 MB | 0.7843 | 0.6353 |
| DAIR-V2X | ERMVP | 0.531 MB | 0.7791 | 0.6324 |
| DAIR-V2X | **InfoCom** | **5.922 KB** | **0.7789** | **0.6385** |

### Ablation Study

| Configuration | Variant | Mean AP | Note |
|---------------|---------|---------|------|
| Full InfoCom | Full | 0.9518 | Baseline |
| IAE | Simple Encoder | 0.9320 | Simplified encoder, −2% |
| SMG | Simple Generator | 0.9379 | Simplified mask generation, −1.4% |
| Quantization | w/o STE | 0.8845 | No straight-through estimator, −6.7% (severe) |
| MSD | w/o Mask | 0.8839 | No mask guidance, −6.8% (severe) |
| MSD | w/o Multi-Scale Rec. | 0.9439 | Single-scale reconstruction, −0.8% |

### Key Findings

- InfoCom reduces communication from MB to KB (7.875 KB vs. Where2comm's 3.439 MB, a 440× reduction) while AP@50 drops by only 0.13 percentage points.
- On V2XSet, ERMVP encounters OOM, whereas InfoCom surpasses standard collaborative performance (AP@50: 0.9273 vs. 0.9212).
- When integrated into weaker backbones (AttFuse, MKD-Cooper), InfoCom improves the original model performance by up to 1.27%, demonstrating that the information purification mechanism effectively suppresses noise.
- Mask guidance and STE are the most critical components (removal causes ~6.8% performance degradation), confirming that spatial priors are indispensable for information recovery under extreme compression.
- Performance gains plateau at retention rate $\alpha > 0.1$ (improvement < 0.2%), and AP variation after 4-bit quantization is less than 0.18%, corroborating the inherent sparsity of spatial cues.

## Highlights & Insights

- The paper reframes the communication efficiency problem in collaborative perception from an information-theoretic perspective, establishing a theoretical foundation for the communication–performance trade-off rather than relying solely on engineering heuristics.
- The information purification paradigm represents a conceptual breakthrough: instead of compressing or selecting features (operating in the feature space), it directly extracts minimal sufficient information (operating in the information space).
- The extended IB Markov chain $Y \to X \to Z \to (E, M)$ elegantly decouples compression and spatial priors, enabling independent optimization of each.
- The plug-and-play architecture is of high practical value: replacing only the communication layer enables compatibility with existing collaborative perception models, lowering the deployment barrier.
- Lemma 1's noise suppression bound $I(E,M;Y_N) \leq I(E,M;Z) - I(E,M;Y)$ provides a theoretical explanation for the observation that information compression can actually improve performance.

## Limitations & Future Work

- The computational overhead is approximately 2× that of Where2comm (as shown in Fig. 4(a)); although transmission time is substantially reduced, the improvement in end-to-end latency depends on specific network conditions.
- The IAEncoder adopts a simple residual block design (targeting resource-constrained agents); a stronger encoder may further improve performance.
- Validation is limited to 3D object detection; extension to other collaborative perception tasks such as occupancy prediction and motion forecasting has not been explored.
- The theoretical analysis relies on a Gaussian prior assumption, and actual feature distributions may deviate from this assumption.
- Extreme compression (KB scale) is advantageous in latency-sensitive scenarios, but whether it underperforms MB-level methods when bandwidth is abundant remains unclear.
- The STE-based gradient estimation for quantization may affect training stability; training convergence is not discussed in the paper.

## Related Work & Insights

- Where2comm (Hu et al.) selects salient information via spatial importance weighting and is a representative feature selection method; ERMVP achieves state-of-the-art communication efficiency through spatial filtering and clustering.
- Information bottleneck theory (Tishby et al.) provides a mathematical framework for representation learning; this paper is the first to systematically apply it to communication optimization in collaborative perception.
- CoAlign (Lu et al.) serves as the default collaborative perception backbone in this work, employing multi-scale features.
- *Insight*: In communication-constrained settings, "transmitting information" is more efficient than "transmitting features"—a principle generalizable to broader domains such as federated learning and distributed inference.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ (The information purification paradigm is a conceptual breakthrough; KB-scale communication efficiency represents an order-of-magnitude leap.)
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ (Three datasets, multiple backbones, comprehensive ablation studies and visualization analyses.)
- **Writing Quality**: ⭐⭐⭐⭐ (Theory and experiments are tightly integrated, though the dense notation raises the reading barrier slightly.)
- **Value**: ⭐⭐⭐⭐⭐ (Addresses the communication bottleneck in real-world deployment with dual contributions in theory and practice.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Sharp Eyes and Memory for VideoLLMs: Information-Aware Visual Token Pruning for Efficient and Reliable VideoLLM Reasoning](sharp_eyes_and_memory_for_videollms_information-aware_visual_token_pruning_for_e.md)
- [\[ACL 2026\] Efficient Learned Data Compression via Dual-Stream Feature Decoupling](../../ACL2026/model_compression/efficient_learned_data_compression_via_dual-stream_feature_decoupling.md)
- [\[AAAI 2026\] Distillation Dynamics: Towards Understanding Feature-Based Distillation in Vision Transformers](distillation_dynamics_towards_understanding_feature-based_di.md)
- [\[AAAI 2026\] Distilling Cross-Modal Knowledge via Feature Disentanglement](distilling_cross-modal_knowledge_via_feature_disentanglement.md)
- [\[ICLR 2026\] COMI: Coarse-to-fine Context Compression via Marginal Information Gain](../../ICLR2026/model_compression/comi_coarse-to-fine_context_compression_via_marginal_information_gain.md)

</div>

<!-- RELATED:END -->
