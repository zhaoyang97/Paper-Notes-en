---
title: >-
  [Paper Note] NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training
description: >-
  [CVPR 2026][Physics & Scientific Computing][Paper Note] Ours proposes NESTOR, a nested MoE neural operator. It captures global features of different PDE types through image-level MoE and local correlations within physical fields through token-level Sub-MoE. It achieves large-scale pre-training across 12 PDE datasets and effectively transfers to downstream tasks.
tags:
  - CVPR 2026
  - Physics & Scientific Computing
date: 2026-05-08
content_hash: 0be62cefdb0d793c
---
# NESTOR: A Nested MOE-based Neural Operator for Large-Scale PDE Pre-Training

**Conference**: CVPR 2026  
**arXiv**: [2602.22059](https://arxiv.org/abs/2602.22059)  
**Code**: [Yes](https://github.com/Event-AHU/OpenFusion)  
**Area**: Scientific Computing  
**Keywords**: Neural Operator, Mixture of Experts (MoE), PDE Solving, Large-Scale Pre-training, Fourier Attention  

## TL;DR

Ours proposes NESTOR, a nested MoE neural operator. It captures global features of different PDE types through image-level MoE and local correlations within physical fields through token-level Sub-MoE. It achieves large-scale pre-training across 12 PDE datasets and effectively transfers to downstream tasks.

## Background & Motivation

Partial Differential Equations (PDEs) are widely used in physics and fluid mechanics. Traditional numerical methods (FEM, FDM) are computationally expensive. Neural operators (FNO, DeepONet, etc.) achieve fast inference by learning mappings between function spaces but face two core challenges:

**Data Scarcity**: PDE training data typically requires expensive experiments or numerical simulations.

**Limitations of Single Architectures**: Existing large-scale PDE pre-training models (e.g., DPOT, MPP) use monolithic network architectures, making it difficult to simultaneously handle:
   - **Macro differences between PDEs**: Dynamics, boundary conditions, and variable dimensions vary significantly across different equations.
   - **Micro heterogeneity within PDEs**: Complex spatio-temporal local correlations exist within the physical fields of the same equation.

**Key Insight**: The diversity and complexity of PDE systems require different expert networks specialized for different inputs, rather than a "one network fits all" approach. MoE routing mechanisms are naturally suited for this, but single-layer MoE can only distinguish equation types and cannot capture regional heterogeneity within the same equation.

## Method

### Overall Architecture

NESTOR aims to solve the dilemma of using "one network for a dozen PDEs." Since different equations have massive dynamical differences and the same equation has complex spatial local correlations, a single architecture struggles to balance both. its solution is to frame the task as autoregressive prediction—taking the recent $T$ frames of PDE states $u_{t-T+1:t}$ as input to predict the next frame $u_{t+1}$—and inserting a two-layer nested expert system in the middle.

The data flow is as follows: the input is first partitioned into patches for spatio-temporal encoding and compressed into a uniform latent representation. It then enters the core nested MoE. The outer image-level MoE selects several non-shared experts suitable for the current PDE type based on global features of the entire image. Within each selected non-shared expert, a token-level Sub-MoE selects local experts for each spatial position. The selected experts are fused using routing weights and finally reconstructed into the next frame of the physical field by the output head. The outer layer handles "which type of equation this is," while the inner layer handles "which expert to use for this region"—this hierarchical structure is the source of the "Nested" in NESTOR.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Input: Recent T frames of PDE states u(t−T+1:t)"] --> B["Spatio-temporal Encoding<br/>Patching + Positional Encoding + Time-weighted Sum<br/>Compressed to uniform latent representation Y"]
    B --> C{"Image-level MoE Router<br/>Global Average Pooling Score → Top-2 Selection"}
    C -->|Shared Expert (Always Active)| D["AFNO Shared Expert<br/>FFT → Spectral Complex Conv → IFFT<br/>Captures global low-frequency structure"]
    C -->|Non-shared Experts (Activated by PDE type)| E
    subgraph E["Nested Collaboration (Non-shared: Flash Attention + Sub-MoE)"]
        direction TB
        E1["Flash Attention<br/>Captures fine-grained spatio-temporal features"] --> E2["Token-level Sub-MoE<br/>Per-token routing → MLP Local Experts<br/>Captures regional local correlations"]
    end
    D --> F["Weighted Fusion by Routing Weights w"]
    E --> F
    F --> G["Output Head<br/>Reconstructs next frame u(t+1)"]
```

### Key Designs

**1. Spatio-temporal Encoding: Compressing multi-frame inputs into a uniform dimension**

The number of history frames $T$ provided by different PDE datasets is inconsistent, but subsequent modules require fixed-dimension inputs. Therefore, the temporal axis is flattened first. The input $x \in \mathbb{R}^{B \times C \times H \times W}$ is divided into non-overlapping patches $X_p \in \mathbb{R}^{B \times N \times C \times P_H \times P_W}$, which are mapped linearly and added with positional encoding to obtain $X \in \mathbb{R}^{B \times N \times D}$. These are rearranged into $X \in \mathbb{R}^{B \times X \times Y \times T \times C}$, and a set of learnable weights is used for a weighted sum along the temporal dimension to collapse $T$ frames into a single frame representation:

$$Y = \sum_{t=1}^{T} W_t X_t, \quad Y \in \mathbb{R}^{B \times X \times Y \times C_{\text{out}}}$$

This ensures the output is of a standard specification regardless of the input frame count, allowing the expert system to handle multiple datasets uniformly.

**2. Image-level MoE: Determining "which type of equation" and routing experts based on global features**

Macro differences between PDEs require specialized handling. The outer MoE serves this "classification-routing" role. It performs global average pooling on the input features to obtain a sample-level representation $\bar{x}_b \in \mathbb{R}^C$, which is passed through a linear layer and softmax to select the Top-$k$ highest-scoring experts. Their probabilities are normalized into fusion weights:

$$s_b = \bar{x}_b W^\top + b, \quad p_b = \text{softmax}(s_b)$$

$$w_{b,i} = \frac{p_{b,i}}{\sum_{j \in \mathcal{I}_b} p_{b,j}}, \quad i \in \mathcal{I}_b$$

The expert pool consists of 6 non-shared experts and 1 shared expert, with the gating activating 2 non-shared experts per pass. The two types of experts are designed to be heterogeneous and complementary: the shared expert is AFNO (Adaptive Fourier Neural Operator), which uses FFT → Spectral Complex Conv → IFFT to capture global low-frequency spatial structures; non-shared experts use Flash Attention to capture fine-grained features, followed by the Sub-MoE described below. Experiments show spontaneous specialization—Expert 0+1 prefer NS equations, while Expert 2+3 prefer Shallow Water Equations (SWE).

**3. Token-level Sub-MoE: Selecting local experts per spatial position within selected experts**

The outer MoE alone cannot handle heterogeneity within different regions of the same equation. Therefore, the routing is pushed down a level. Sub-MoE replaces the standard FFN in Flash Attention, with routing granularity shifting from the whole image to individual tokens (spatial positions), also using Top-$k$ selection. Each local expert is a standard MLP:

$$\text{ExpertMLP}(x) = W_2 \sigma(W_1 x + b_1) + b_2$$

Where $W_1 \in \mathbb{R}^{C \times (rC)}$, $W_2 \in \mathbb{R}^{(rC) \times C}$, and $r$ is the MLP ratio with GELU activation. The configuration also uses 6 non-shared and 1 shared expert with Top-2 activation. Visualizations reveal region-specific activation patterns in space, corresponding to local correlation differences in the physical field (e.g., vortex regions vs. stable regions).

**4. Nested Collaboration: Macro classification + Micro partitioning for large capacity with low activation**

The two layers function hierarchically: "outer layer determines the category, inner layer partitions the regions." Image-level MoE selects the expert combination based on PDE type (e.g., NS activates Expert 0+1), and within those experts, token-level Sub-MoE further identifies spatial features. This nesting allows the total parameters to reach 83M (sufficient capacity for diverse equations), while only 13M are activated during each forward pass (16.67% activation rate), achieving a compromise between "large model capacity and small model computation" through sparse activation.

### Loss & Training

The total loss consists of three parts:

$$\mathcal{L} = \mathcal{L}_2 + \alpha \mathcal{L}_{\text{aux}_1} + \beta \mathcal{L}_{\text{aux}_2}$$

- **Main Task Loss** $\mathcal{L}_2$: L2 Relative Error (L2RE), $\mathcal{L}_2 = \frac{\|\hat{y}_i^{(c)} - y_i^{(c)}\|_2}{\|y_i^{(c)}\|_2}$
- **Image-level Load Balancing Loss** $\mathcal{L}_{\text{aux}_1}$: Prevents uneven expert assignment.
- **Token-level Load Balancing Loss** $\mathcal{L}_{\text{aux}_2}$: Same as above.

The load balancing loss is defined as $\mathcal{L}_{\text{aux}} = E \sum_{i=1}^{E} p_i \cdot f_i$, where $p_i$ is the average routing probability and $f_i$ is the actual token distribution ratio.

Training Strategy: Small-scale noise is injected into input frames to enhance robustness (following the denoising pre-training strategy of DPOT).

## Key Experimental Results

### Main Results

Pre-training and fine-tuning results on 12 PDE datasets (L2RE↓):

| Model | Activated Params | FNO-ν 1e-5 | FNO-ν 1e-4 | FNO-ν 1e-3 | PDEBench Avg(1) | PDEBench Avg(0.1) | DR | SWE | CFDBench |
|---|---|---|---|---|---|---|---|---|---|
| FNO | 0.5M | 0.116 | 0.092 | 0.016 | 0.130 | 0.153 | 0.032 | 0.009 | 0.027 |
| DPOT-T (Pre-train) | 7M | 0.098 | 0.061 | 0.010 | 0.029 | 0.018 | 0.032 | 0.006 | 0.010 |
| **Ours (Pre-train)** | 13M | 0.120 | 0.095 | **0.009** | **0.027** | **0.016** | 0.031 | **0.005** | 0.011 |
| DPOT-FT500 | 7M | 0.052 | 0.037 | 0.006 | 0.015 | 0.016 | 0.015 | 0.002 | 0.004 |
| **Ours-FT500** | 13M | **0.051** | **0.022** | **0.004** | **0.011** | **0.010** | **0.012** | 0.003 | **0.004** |

After 500 epochs of fine-tuning, Ours achieves SOTA in 9 out of 12 tasks, and overall best performance in 10/12.

### Ablation Study

Ablations on six PDEBench sub-tasks (FT-500, Avg L2RE↓):

| Method | Avg L2RE | Gain |
|---|---|---|
| **Full Model** | **0.0173** | - |
| w/o Sub-MoE | 0.0197 | +0.0024 |
| w/o Aux Loss | 0.0178 | +0.0005 |
| FlashAttn + AFNO Direct Addition | 0.0196 | +0.0023 |

### Key Findings

1. **Sub-MoE provides the largest contribution**: Removing it increases error by 0.0024, validating the importance of fine-grained token-level expert selection.
2. **MoE fusion outperforms simple addition**: Replacing MoE fusion with direct addition increases error by 0.0023, proving the routing mechanism is superior to fixed fusion.
3. **More experts are not always better**: 6 non-shared experts achieve the best average performance in FT-500; 12 experts lead to performance degradation due to optimization difficulties.
4. **Pre-training data volume has a positive impact**: Pre-training on 12 datasets (0.0208 Avg L2RE) outperforms 3 datasets (0.0234).
5. **Significant transfer to downstream tasks**: Precision improved by 47.3% after fine-tuning on a $512 \times 512$ high-resolution turbulence task.
6. **Activation Efficiency**: Only 13M of 83M total parameters are activated (16.67%), significantly lower than MoE-POT-T's 56.67%.

## Highlights & Insights

1. **Nested MoE design has clear physical correspondence**: Image-level → division by PDE type, token-level → division by spatial region. This "macro classification-micro partitioning" offers good interpretability.
2. **Heterogeneous Expert Design**: Shared experts use AFNO (spectral global features) while non-shared experts use Flash Attention (spatial local features), ensuring complementarity rather than redundancy.
3. **Rigorous Interpretability Analysis**: Expert activation frequency statistics and token-level spatial heatmaps clearly demonstrate functional differentiation of the MoE.
4. **Large Capacity, Low Cost**: Total parameters of 83M but only 16.67% activation rate provide a path for efficient scaling of PDE neural operators.

## Limitations & Future Work

1. **Suboptimal Pre-training on some datasets**: Pre-training performance on FNO-ν 1e-5 and 1e-4 is inferior to DPOT, suggesting nested MoE may overfit to specific experts when data is limited.
2. **Weaker performance on NS-cond and PDE Arena-NS**: Ours-FT500 is roughly equal to or slightly worse than DPOT-FT500 on these datasets.
3. Scalability to 3D PDEs was not validated.
4. The number of experts (6) and activation (2) are manually set, lacking an adaptive mechanism.
5. Load balancing loss contribution is small (only 0.0005); more effective expert balancing strategies could be explored.
6. Physical constraint losses (e.g., PDE residual loss) could be integrated to further improve physical consistency.

## Rating

⭐⭐⭐⭐ 4/5

Introducing nested MoE into PDE neural operators is a significant innovation. The "macro classification-micro partitioning" design intuition is clear and well-validated experimentally. Achieving SOTA results in 10 out of 12 benchmarks is convincing. The deduction is because the innovation is primarily an engineering combination of MoE architectures (AFNO + Flash Attention + dual-layer routing), where components are existing technologies. Additionally, while the 13M activation is efficient, the 83M total memory overhead remains substantial.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PhysicsCorrect: A Training-Free Approach for Stable Neural PDE Simulations](../../AAAI2026/physics/physicscorrect_a_training-free_approach_for_stable_neural_pde_simulations.md)
- [\[ICLR 2026\] One Operator to Rule Them All? On Boundary-Indexed Operator Families in Neural PDE Solvers](../../ICLR2026/physics/one_operator_to_rule_them_all_on_boundary-indexed_operator_families_in_neural_pd.md)
- [\[ICML 2026\] Topology-Preserving Neural Operator Learning via Hodge Decomposition](../../ICML2026/physics/topology-preserving_neural_operator_learning_via_hodge_decomposition.md)
- [\[ICLR 2026\] DRIFT-Net: A Spectral--Coupled Neural Operator for PDEs Learning](../../ICLR2026/physics/drift-net_a_spectral--coupled_neural_operator_for_pdes_learning.md)
- [\[NeurIPS 2025\] Enforcing Governing Equation Constraints in Neural PDE Solvers via Training-free Projections](../../NeurIPS2025/physics/enforcing_governing_equation_constraints_in_neural_pde_solvers_via_training-free.md)

</div>

<!-- RELATED:END -->
