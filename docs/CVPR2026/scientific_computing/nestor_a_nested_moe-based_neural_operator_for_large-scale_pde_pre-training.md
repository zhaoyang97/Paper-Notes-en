---
title: >-
  [Paper Note] NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training
description: >-
  [CVPR 2026][Scientific Computing][Neural Operator] NESTOR, a nested MoE-based neural operator, is proposed to capture global features across different PDE types via image-level MoE and local spatial correlations within p…
tags:
  - "CVPR 2026"
  - "Scientific Computing"
  - "Neural Operator"
  - "Mixture of Experts (MoE)"
  - "PDE Solving"
  - "Large-Scale Pre-Training"
  - "Fourier Attention"
date: 2026-05-08
content_hash: 7945e90f13805f23
---

# NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training

**Conference**: CVPR 2026
**arXiv**: [2602.22059](https://arxiv.org/abs/2602.22059)  
**Code**: [Available](https://github.com/Event-AHU/OpenFusion)  
**Area**: Scientific Computing
**Keywords**: Neural Operator, Mixture of Experts (MoE), PDE Solving, Large-Scale Pre-Training, Fourier Attention

## TL;DR

NESTOR, a nested MoE-based neural operator, is proposed to capture global features across different PDE types via image-level MoE and local spatial correlations within physical fields via token-level Sub-MoE. The model is pre-trained on 12 PDE datasets and effectively transferred to downstream tasks.

## Background & Motivation

Partial differential equations (PDEs) are widely used in physics, fluid mechanics, and related fields. Traditional numerical solvers (FEM, FDM) incur high computational costs, and neural operators (FNO, DeepONet, etc.) enable fast inference by learning mappings in function spaces. However, two core challenges persist:

**Scarcity of training data**: PDE training data typically requires expensive experiments or numerical simulations.

**Limitations of monolithic architectures**: Existing large-scale PDE pre-training approaches (e.g., DPOT, MPP) rely on a single unified network, which struggles to simultaneously handle:
- **Macro-level heterogeneity across PDEs**: Different equations exhibit vastly different dynamical mechanisms, boundary conditions, and variable dimensions.
- **Micro-level heterogeneity within PDEs**: A single equation's physical field contains complex spatiotemporal local correlations.

The core insight is that the diversity and complexity of PDE systems require specialized expert networks for different inputs, rather than a single network handling everything. The routing mechanism of MoE naturally fits this need; however, a single-layer MoE can only distinguish equation types and fails to capture intra-equation regional heterogeneity.

## Method

### Overall Architecture

NESTOR adopts an autoregressive prediction framework: given the most recent $T$ frames of PDE states $u_{t-T+1:t}$, the model predicts the next frame $u_{t+1}$. The overall pipeline is:

1. **Patch embedding + spatiotemporal encoding**: Input is divided into patches and mapped to a latent space.
2. **Nested MoE module** (core): image-level MoE selects global experts → each expert internally contains a token-level Sub-MoE.
3. **Output head**: Predicts the next PDE state frame.

### Key Designs

#### 1. Spatio-Temporal Encoding

**Function**: Encodes multi-frame PDE inputs into a unified latent representation.

**Mechanism**: Input $x \in \mathbb{R}^{B \times C \times H \times W}$ is divided into non-overlapping patches $X_p \in \mathbb{R}^{B \times N \times C \times P_H \times P_W}$. After linear projection and positional encoding, the representation becomes $X \in \mathbb{R}^{B \times N \times D}$. It is then rearranged into $X \in \mathbb{R}^{B \times X \times Y \times T \times C}$, and compressed along the temporal dimension via learnable weight matrices:

$$Y = \sum_{t=1}^{T} W_t X_t, \quad Y \in \mathbb{R}^{B \times X \times Y \times C_{\text{out}}}$$

**Design Motivation**: Different PDEs may have varying numbers of input frames; temporal compression ensures a unified input dimension for downstream modules.

#### 2. Image-level MoE (Global Expert Selection)

**Function**: Dynamically selects the most suitable expert network for the current PDE type based on the global features of the input sample.

**Mechanism**: A Top-$k$ routing strategy is employed. Global average pooling is applied to obtain $\bar{x}_b \in \mathbb{R}^C$, which is passed through a linear layer to produce expert scores, followed by softmax normalization and selection of the top-$k$ experts:

$$s_b = \bar{x}_b W^\top + b, \quad p_b = \text{softmax}(s_b)$$

$$w_{b,i} = \frac{p_{b,i}}{\sum_{j \in \mathcal{I}_b} p_{b,j}}, \quad i \in \mathcal{I}_b$$

Architecture configuration: 6 non-shared experts + 1 shared expert; the gating network activates 2 non-shared experts per forward pass.

**Expert Design**:
- **Shared expert**: AFNO (Adaptive Fourier Neural Operator), capturing global low-frequency spatial features in the frequency domain via FFT → complex convolution → IFFT.
- **Non-shared experts**: Flash Attention, capturing fine-grained spatiotemporal features. Q/K/V attention is followed by a Sub-MoE.

**Design Motivation**: Experimental analysis shows that different experts exhibit significant preferences for different PDE types (e.g., Expert 0+1 favors Navier–Stokes equations, Expert 2+3 favors shallow water equations), demonstrating that image-level routing effectively achieves functional specialization.

#### 3. Token-level Sub-MoE (Local Expert Selection)

**Function**: Within each image-level expert, selects the most appropriate local expert for each token (spatial position).

**Mechanism**: Replaces the FFN layer in Flash Attention. The same Top-$k$ routing is applied, but at token granularity rather than image granularity. Each expert is a standard MLP:

$$\text{ExpertMLP}(x) = W_2 \sigma(W_1 x + b_1) + b_2$$

where $W_1 \in \mathbb{R}^{C \times (rC)}$, $W_2 \in \mathbb{R}^{(rC) \times C}$, $r$ is the MLP ratio, and the activation function is GELU.

The configuration is again 6 non-shared + 1 shared expert with Top-2 activation.

**Design Motivation**: Visualization analysis shows that different token-level experts exhibit spatially region-specific activation patterns, validating their ability to capture local correlations within physical fields.

#### 4. Nested Architecture: "Macro Classification – Micro Partitioning" Mechanism

**Function**: The image-level MoE and token-level Sub-MoE form a hierarchical collaboration.

**Mechanism**:
- **Macro level**: The image-level MoE adaptively selects expert combinations based on PDE type (e.g., Expert 0+1 for NS equations, Expert 2+3 for SWE).
- **Micro level**: The token-level Sub-MoE, within each selected expert, further identifies spatial regional features of the physical field.

This nested design yields a total parameter count of 83M, with only 13M active parameters (activation ratio: 16.67%), achieving a balance between large capacity and low computational cost.

### Loss & Training

The total loss consists of three components:

$$\mathcal{L} = \mathcal{L}_2 + \alpha \mathcal{L}_{\text{aux}_1} + \beta \mathcal{L}_{\text{aux}_2}$$

- **Primary task loss** $\mathcal{L}_2$: L2 relative error (L2RE), $\mathcal{L}_2 = \frac{\|\hat{y}_i^{(c)} - y_i^{(c)}\|_2}{\|y_i^{(c)}\|_2}$
- **Image-level load balancing loss** $\mathcal{L}_{\text{aux}_1}$: Prevents uneven expert allocation.
- **Token-level load balancing loss** $\mathcal{L}_{\text{aux}_2}$: Same purpose as above.

The load balancing loss is uniformly defined as $\mathcal{L}_{\text{aux}} = E \sum_{i=1}^{E} p_i \cdot f_i$, where $p_i$ is the average routing probability and $f_i$ is the actual token allocation fraction.

Training strategy: Small-scale noise is injected into input frames to enhance robustness, following the denoising pre-training strategy from DPOT.

## Key Experimental Results

### Main Results

Pre-training and fine-tuning results across 12 PDE datasets (L2RE↓):

| Model | Active Params | FNO-ν 1e-5 | FNO-ν 1e-4 | FNO-ν 1e-3 | PDEBench Avg(1) | PDEBench Avg(0.1) | DR | SWE | CFDBench |
|---|---|---|---|---|---|---|---|---|---|
| FNO | 0.5M | 0.116 | 0.092 | 0.016 | 0.130 | 0.153 | 0.032 | 0.009 | 0.027 |
| DPOT-T (pre-trained) | 7M | 0.098 | 0.061 | 0.010 | 0.029 | 0.018 | 0.032 | 0.006 | 0.010 |
| **Ours (pre-trained)** | 13M | 0.120 | 0.095 | **0.009** | **0.027** | **0.016** | 0.031 | **0.005** | 0.011 |
| DPOT-FT500 | 7M | 0.052 | 0.037 | 0.006 | 0.015 | 0.016 | 0.015 | 0.002 | 0.004 |
| **Ours-FT500** | 13M | **0.051** | **0.022** | **0.004** | **0.011** | **0.010** | **0.012** | 0.003 | **0.004** |

After fine-tuning for 500 epochs, the proposed method achieves state-of-the-art performance on 9 out of 12 tasks, with globally optimal results on 10 out of 12.

### Ablation Study

Ablation results on six PDEBench sub-tasks (FT-500, Avg L2RE↓):

| Method | Avg L2RE | Performance Drop |
|---|---|---|
| **Full Model** | **0.0173** | - |
| w/o Sub-MoE | 0.0197 | +0.0024 |
| w/o Load Balancing Loss | 0.0178 | +0.0005 |
| FlashAttn + AFNO Direct Addition | 0.0196 | +0.0023 |

### Key Findings

1. **Sub-MoE contributes most**: Removing it increases error by 0.0024, validating the importance of token-level fine-grained expert selection.
2. **MoE fusion outperforms simple addition**: Replacing MoE-based fusion of AFNO and FlashAttn with direct addition increases error by 0.0023, demonstrating the superiority of routing over fixed fusion.
3. **More experts is not always better**: 6 non-shared experts achieves the best average performance under FT-500; 12 experts leads to degraded performance due to optimization difficulties.
4. **Pre-training data scale has a positive effect**: Pre-training on 12 datasets yields a lower Avg L2RE (0.0208) than pre-training on 3 datasets (0.0234).
5. **Significant transfer to downstream tasks**: On a 512×512 high-resolution turbulence task, fine-tuning improves accuracy by 47.3%.
6. **Activation efficiency**: Only 13M out of 83M total parameters are activated (16.67%), far below MoE-POT-T's 56.67%.

## Highlights & Insights

1. **The nested MoE design has clear physical correspondence**: image-level → specialization by PDE type; token-level → specialization by spatial region within the physical field. The "macro classification – micro partitioning" structure offers strong interpretability.
2. **Heterogeneous expert design**: The shared expert uses AFNO for global frequency-domain features, while non-shared experts use Flash Attention for local spatial features — the two are complementary rather than redundant.
3. **Thorough interpretability analysis**: Expert activation frequency statistics (Table 5) and token-level spatial activation heatmaps (Figure 5) clearly demonstrate functional differentiation within the MoE.
4. **Large capacity at low cost**: With 83M total parameters and only a 16.67% activation rate, the design provides an efficient scaling strategy for PDE neural operators.

## Limitations & Future Work

1. **Suboptimal pre-training performance on some datasets**: On FNO-ν 1e-5 and 1e-4, the pre-trained model underperforms DPOT, suggesting that the nested MoE may overfit to specific experts under limited data.
2. **Weaker performance on NS-cond and PDE Arena-NS**: On these two datasets, Ours-FT500 is roughly on par with or slightly worse than DPOT-FT500.
3. Only 2D PDEs are evaluated; scalability to 3D PDEs remains unverified.
4. The number of experts (6) and active experts (2) are manually configured, with no adaptive mechanism.
5. The load balancing loss offers limited benefit (only 0.0005 improvement); more effective expert balancing strategies merit exploration.
6. Incorporating physics-constrained losses (e.g., PDE residual loss) could further improve physical consistency.

## Rating

⭐⭐⭐⭐ 4/5

Introducing nested MoE into PDE neural operators is a meaningful innovation. The "macro classification – micro partitioning" design intuition is clear and well-validated experimentally. Achieving state-of-the-art on 10 out of 12 benchmarks is convincing. Points are deducted because the contribution is primarily an engineering combination of existing components (AFNO + Flash Attention + two-level routing), with no fundamentally new techniques; additionally, while activating only 13M of 83M parameters is computationally efficient, the total memory footprint remains substantial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](../../ICLR2026/scientific_computing/one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)
- [\[AAAI 2026\] PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](../../AAAI2026/scientific_computing/physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)
- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/scientific_computing/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[ICLR 2026\] Astral: Training Physics-Informed Neural Networks with Error Majorants](../../ICLR2026/scientific_computing/astral_training_physics-informed_neural_networks_with_error_majorants.md)
- [\[NeurIPS 2025\] Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](../../NeurIPS2025/scientific_computing/enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)

</div>

<!-- RELATED:END -->
