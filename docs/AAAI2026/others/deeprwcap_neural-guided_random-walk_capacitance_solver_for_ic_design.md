---
title: >-
  [Paper Note] DeepRWCap: Neural-Guided Random-Walk Capacitance Solver for IC Design
description: >-
  [AAAI 2026][Capacitance Extraction] This paper proposes DeepRWCap, a machine-learning-guided random-walk capacitance solver that accelerates multi-dielectric capacitance extraction in IC design via a two-stage neural network architecture for transition kernel prediction, achieving an average error of 1.24% and 23% speedup across 10 industrial test cases.
tags:
  - "AAAI 2026"
  - "Capacitance Extraction"
  - "Random Walk"
  - "CNN"
  - "IC Design"
  - "EDA"
date: 2026-05-08
content_hash: b407e92705638545
---

# DeepRWCap: Neural-Guided Random-Walk Capacitance Solver for IC Design

**Conference**: AAAI 2026
**arXiv**: [2511.06831](https://arxiv.org/abs/2511.06831)  
**Code**: [github.com/THU-numbda/deepRWCap](https://github.com/THU-numbda/deepRWCap)  
**Area**: Other
**Keywords**: Capacitance Extraction, Random Walk, CNN, IC Design, EDA

## TL;DR

This paper proposes DeepRWCap, a machine-learning-guided random-walk capacitance solver that accelerates multi-dielectric capacitance extraction in IC design via a two-stage neural network architecture for transition kernel prediction, achieving an average error of 1.24% and 23% speedup across 10 industrial test cases.

## Background & Motivation

Parasitic capacitance extraction is a critical step in integrated circuit (IC) design verification, requiring analysis of physical layouts to ensure timing, power, and signal integrity requirements are met. As semiconductor technology advances toward more complex 3D integration (FinFET → Gate-All-Around → CFET), capacitance extraction faces significant computational challenges.

**Limitations of existing methods**:

**Finite Difference Method (FDM)**: High accuracy but poor scalability.

**Pattern Matching Methods**: Computationally efficient but lack accuracy guarantees and rely on expert knowledge.

**Floating Random Walk (FRW)**: A scalable stochastic framework, but computing the transition kernel in multi-dielectric domains is extremely expensive.

**Pure Learning Methods (CNN-Cap, GNN-Cap)**: Require retraining for each process node.

**Hybrid Methods (GE-CNN)**: Suffer from information bottlenecks; CPU-only implementations are 12× slower than vanilla FRW.

The core problem is that FRW methods, when handling dense structures in modern semiconductors, must perform unbiased sampling over transition domains containing multiple high-contrast dielectric materials at each step—a computationally prohibitive process. DeepRWCap is motivated by replacing this expensive numerical solve with a **compact neural network** while preserving the statistical unbiasedness of the computation.

## Method

### Overall Architecture

DeepRWCap builds upon the Floating Random Walk (FRW) framework. The FRW method uses a cube to "jump" through the problem domain; at each step, a transition cube is constructed at the current position, and the Poisson kernel is computed to determine the transition probability for the next step.

**Core Idea**: Decompose transition kernel prediction into **face selection** (choosing which face to jump to) and **intra-face kernel prediction** (determining the probability distribution on the selected face), leveraging cube symmetry to reduce learning redundancy.

### Key Designs

1. **Two-Stage Prediction Architecture**:

   Directly predicting a $6 \times N \times N$ Poisson kernel introduces substantial cross-face redundancy. DeepRWCap decomposes prediction into two stages:

   - **Face Selector $\mathcal{F}_\theta$**: A 3D convolutional network that predicts a categorical distribution $\mathbf{F} \in \mathbb{R}^6$ over six faces, where $\mathbf{F}_i = \sum_{j,k} (\mathbf{p}_\alpha)_{i,j,k}$. Four stride-2 3D convolutional layers progressively downsample the input; the output is normalized via softmax. The training loss is KL divergence.

   - **Face Predictor $\mathcal{G}_\theta$**: A 2D depthwise separable convolution network that predicts the conditional probability distribution on the selected face. Depthwise convolutions capture spatial patterns within each layer, while pointwise convolutions model inter-layer interactions. ReLU is used to ensure non-negativity, and L1 normalization enforces a valid probability distribution.

   **Design Motivation**: The Poisson kernel decays rapidly with distance from the surface, making it natural to decompose the 3D problem into face selection plus 2D intra-face prediction.

2. **Gradient Kernel Prediction and Symmetry Exploitation**:

   The **first step** of the random walk requires special transition quantities: weight value $w_\alpha$, sign distribution $s_\alpha$, and gradient kernel $g_\alpha$. Only the $z$-component gradient kernel is learned; **other components are derived via cube symmetry** (by rotating and reflecting the input).

   Notably, the symmetry group of the gradient kernel is only a subgroup of the full cube symmetry, requiring **at least two dedicated face predictors**:
   - **Tangential face**: High symmetry, relatively easy to predict.
   - **Normal face**: Exhibits a bimodal structure (positive at one end, negative at the other), an order of magnitude more difficult to predict, requiring a deeper network.

   The face selector extends its output dimension from 6 to 7, simultaneously predicting the face distribution and the weight value, using a combined loss:
   $$\mathcal{L}_{grad\text{-}face\text{-}select} = D_{KL}[\mathbf{F}^\nabla \| \text{softmax}(\mathcal{F}_\theta(\mathcal{X})_{1:6})] + \lambda |w_\alpha - \mathcal{F}_\theta(\mathcal{X})_7|^2$$

3. **High-Throughput GPU Inference Engine**:

    - **Asynchronous producer-consumer architecture**: Walker threads (producers) generate sampling tasks; sampler threads (consumers) execute GPU inference.
    - **Multi-instance model deployment**: One Poisson solver instance is deployed per 2 walker threads; one gradient solver is shared.
    - **Custom CUDA kernels**: Voxelization is performed directly on the GPU; TensorRT FP16 compilation accelerates inference.
    - Compact structural descriptions are transmitted rather than voxelized data, reducing GPU memory transfer overhead.

### Loss & Training

- **Data Generation**: 100,000 random dielectric configurations are generated via the block-wise generation procedure in Algorithm 1, simulating low-$\kappa$ ($U(2,10)$, 80% probability) and high-$\kappa$ ($U(10,80)$, 20% probability) materials found in real ICs.
- **Training Configuration**: AdamW optimizer, cosine annealing learning rate schedule ($10^{-3}$ → $5\times10^{-6}$), 20-epoch warmup, 200 total epochs, batch size = 16.
- Grid-based positional encoding (Grid PE) provides spatial context to the face predictor by appending $(x, y)$ coordinate channels.
- Ground-truth labels are generated by an FDM solver; data generation takes 1.7 hours and training takes 12.3 hours.

## Key Experimental Results

### Main Results

10 industrial test cases (12–55 nm process nodes), with Raphael commercial solver as ground truth:

| Case | Node (nm) | FRW-FDM Error | GE-CNN Error | FRW-AGF Error | Microwalk Error | **DeepRWCap Error** |
|------|-----------|---------------|--------------|---------------|-----------------|---------------------|
| 1 | 16 | 0.4±0.2% | 4.9±0.2% | 0.8±0.1% | 0.6±0.2% | **1.2±0.1%** |
| 4 | 28 | 1.8±0.4% | 6.6±0.3% | 1.4±0.4% | 0.5±0.4% | **0.7±0.3%** |
| 9 | 12 | 1.5±0.8% | 22.9±0.7% | 17.0±1.0% | 0.9±0.5% | **1.1±0.9%** |
| 10 | 12 | 0.6±0.6% | 27.1±0.8% | 23.9±1.3% | 0.6±0.4% | **1.2±0.9%** |
| **Average** | — | — | — | 5.18±7.81% | — | **1.24±0.53%** |

Speedup: 23% average speedup over Microwalk ($1.23\times$, $p=0.024$); 49% average speedup on complex designs (>10 s).

### Ablation Study

Face predictor architecture ablation (single-face Poisson kernel prediction):

| Architecture | Params | FLOPs | L2 Error (%) | KL Divergence |
|---|---|---|---|---|
| MLP (3 layers × 2048) | 34.4M | 34.4M | 7.83 | 0.0133 |
| 3D Conv | 19.7K | 20.3M | 24.15 | 0.0298 |
| 2D Conv | 4.28K | 2.31M | 13.80 | 0.0125 |
| GE-CNN + GMM | 0.43M | 0.65M | 26.63 | 0.0403 |
| DS Conv | 1.37K | 0.82M | 12.15 | 0.0083 |
| **DS Conv + Grid PE** | **1.40K** | **0.84M** | **3.93** | **0.0021** |

### Key Findings

1. **Depthwise separable convolutions achieve optimal accuracy with very few parameters (1.40K)**, far outperforming MLP and GE-CNN models with millions of parameters.
2. **Grid PE positional encoding is critical**: KL divergence drops from 0.0083 to 0.0021.
3. **3D convolutions are ill-suited for this task**: Despite being intuitively appropriate for volumetric data, they exhibit low computational efficiency and poor accuracy.
4. **GE-CNN fails on high-contrast dielectrics (Cases 9 and 10)**: Errors reach 22–27%.
5. **AGF exhibits unstable accuracy**: Average error of 5.18% with a standard deviation of 7.81%, versus DeepRWCap's 1.24±0.53%.
6. **TensorRT + CUDA optimization is essential**: Poisson prediction latency is reduced from ~0.7 ms to substantially lower levels.

## Highlights & Insights

- **Elegant integration of physics and learning**: Rather than replacing the random-walk framework, a CNN accelerates only the most expensive component—transition kernel computation—preserving the statistical unbiasedness of FRW.
- **Thorough symmetry exploitation**: Cube symmetry is leveraged to reduce learning redundancy (only the $z$-component is learned); the face selection + face prediction decomposition also stems from symmetry analysis.
- **Extreme model efficiency**: The core face predictor has only 1.40K parameters and 0.84M FLOPs, suitable for high-frequency invocation.
- **Cross-node generalization**: Trained on procedurally generated dielectric configurations, validated on industrial designs across 12–55 nm nodes, demonstrating strong cross-node generalization.
- **High engineering completeness**: A full GPU inference engine, producer-consumer scheduling, and multi-instance deployment are all implemented.

## Limitations & Future Work

1. **Self-capacitance only**: The paper focuses on self-capacitance estimation; coupling capacitance requires further validation.
2. **Synthetic training data**: Procedurally generated dielectric configurations may not fully cover all real-world process scenarios.
3. **Single-GPU constraint**: The current implementation targets a single RTX 4090; larger-scale designs may require multi-GPU support.
4. **Gradient kernel normal-face accuracy has room for improvement**: Validation loss is an order of magnitude higher than for tangential faces.
5. **Fixed cube discretization resolution ($N=23$)**: Higher resolution may improve accuracy but increases computational cost.

## Related Work & Insights

DeepRWCap sits at the intersection of EDA (Electronic Design Automation) and machine learning. Compared to purely learned surrogate approaches (CNN-Cap, GNN-Cap), the hybrid method retains physical interpretability and cross-node transferability. Compared to prior hybrid methods (GE-CNN), DeepRWCap achieves practically meaningful speedups through superior architecture design and GPU optimization. The approach offers broader inspiration for other scientific computing problems requiring accelerated Monte Carlo sampling, such as radiative transfer and diffusion equations.

## Rating

- Novelty: ⭐⭐⭐⭐ (Creative two-stage architecture design and symmetry exploitation)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (10 industrial cases, cross-node validation, complete architecture ablation)
- Writing Quality: ⭐⭐⭐⭐⭐ (Mathematically rigorous, system-level completeness, sufficient engineering detail)
- Value: ⭐⭐⭐⭐ (Addresses real EDA pain points with strong industrial application potential)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Fields to Random Trees](../../ICLR2026/others/from_fields_to_random_trees.md)
- [\[ICLR 2026\] An Information-Theoretic Framework For Optimizing Experimental Design To Distinguish Probabilistic Neural Codes](../../ICLR2026/others/an_information-theoretic_framework_for_optimizing_experimental_design_to_disting.md)
- [\[ICLR 2026\] IC-Custom: Diverse Image Customization via In-Context Learning](../../ICLR2026/others/ic-custom_diverse_image_customization_via_in-context_learning.md)
- [\[AAAI 2026\] Controllable Financial Market Generation with Diffusion Guided Meta Agent](controllable_financial_market_generation_with_diffusion_guided_meta_agent.md)
- [\[AAAI 2026\] GDBA Revisited: Unleashing the Power of Guided Local Search for Distributed Constraint Optimization](gdba_revisited_unleashing_the_power_of_guided_local_search_for_distributed_const.md)

</div>

<!-- RELATED:END -->
