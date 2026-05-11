---
title: >-
  [Paper Note] Mamba-3: Improved Sequence Modeling using State Space Principles
description: >-
  [ICLR 2026][Video Understanding][State Space Models] Three core improvements are proposed from an SSM perspective: exponential-trapezoidal discretization, complex-valued state spaces…
tags:
  - "ICLR 2026"
  - "Video Understanding"
  - "State Space Models"
  - "Mamba"
  - "Sequence Modeling"
  - "Inference Efficiency"
  - "MIMO"
date: 2026-05-08
content_hash: 33cc8b469308a1c7
---

# Mamba-3: Improved Sequence Modeling using State Space Principles

**Conference**: ICLR 2026
**arXiv**: [2603.15569](https://arxiv.org/abs/2603.15569)
**Code**: [Available](https://github.com/state-spaces/mamba)
**Area**: Video Understanding
**Keywords**: State Space Models, Mamba, Sequence Modeling, Inference Efficiency, MIMO

## TL;DR

Three core improvements are proposed from an SSM perspective: exponential-trapezoidal discretization, complex-valued state spaces, and multi-input multi-output (MIMO) formulation. These advances significantly improve model quality and state-tracking capability without increasing decoding latency, pushing the performance–efficiency Pareto frontier forward.

## Background & Motivation

Test-time compute has become a key driver of LLM performance, with techniques such as chain-of-thought reasoning and iterative refinement placing inference efficiency at the center of model design. Although Transformers remain the industry standard, they are constrained by:
- **Quadratic computational complexity**: self-attention mechanism
- **Linear memory requirements**: KV cache grows linearly with sequence length

Sub-quadratic models (SSMs, linear attention) offer constant memory and linear computation, yet suffer from three major shortcomings:

**Limited expressivity**: Mamba-2 sacrifices some expressivity for training speed, underperforming Mamba-1

**Lack of state-tracking capability**: unable to solve simple tasks such as parity

**Poor hardware efficiency**: arithmetic intensity during decoding is only ~2.5 ops/byte, leaving most hardware idle

## Method

### Overall Architecture

Mamba-3 introduces three SSM-principled core improvements over Mamba-2, along with several architectural refinements. The overall architecture follows the Llama style, interleaving Mamba-3 blocks and SwiGLU MLP blocks with pre-norm.

### Key Designs

#### 1. Exponential-Trapezoidal Discretization

**Background**: Discretizing continuous-time SSMs into recurrences. The discretization used in Mamba-1/2 lacked theoretical justification.

**Contributions**:
- Formalizes the heuristic discretization of Mamba-1/2 as an "exponential-Euler" method (first-order approximation, error $O(\Delta_t^2)$)
- Proposes an "exponential-trapezoidal" method (second-order approximation, error $O(\Delta_t^3)$)

Exponential-trapezoidal recurrence:

$$\mathbf{h}_t = e^{\Delta_t A_t}\mathbf{h}_{t-1} + (1-\lambda_t)\Delta_t e^{\Delta_t A_t}\mathbf{B}_{t-1}x_{t-1} + \lambda_t\Delta_t\mathbf{B}_t x_t$$

where $\lambda_t \in [0,1]$ is a data-dependent scalar. Setting $\lambda_t=1$ recovers Mamba-2's Euler method; $\lambda_t=\frac{1}{2}$ yields the classical trapezoidal rule.

**Equivalent convolution perspective**: This recurrence is equivalent to applying a width-2 data-dependent convolution to the state input $\mathbf{B}_t x_t$ before entering the linear recurrence — fundamentally different from the standard short convolution externally applied in Mamba, as it operates inside the recurrence core.

**Parallel form**: Via the SSD framework, the structured mask $\mathbf{L}$ corresponding to the new recurrence is the product of a 1-semiseparable matrix and a 2-band matrix (a special 2-semiseparable matrix), supporting efficient parallel matrix-multiplication computation.

#### 2. Complex-Valued SSM

**Design Motivation**: The transition matrix eigenvalues of real-valued SSMs (e.g., Mamba-2) are restricted to real numbers, preventing the representation of "rotational" dynamics — for example, parity can be expressed using a rotation matrix $\mathbf{R}(\pi x_t)$.

**Method**: Extends the underlying SSM parameters to complex values. A key equivalence is established:

A complex-valued SSM after discretization is equivalent to a real-valued SSM augmented with data-dependent rotary embeddings (RoPE):

$$\mathbf{h}_t = e^{\Delta_t A_t}\mathbf{R}_t\mathbf{h}_{t-1} + \Delta_t\mathbf{B}_t x_t$$

where $\mathbf{R}_t$ is a block-diagonal rotation matrix whose angles are produced by data projections.

**RoPE trick**: By cumulatively applying rotation matrices to the B/C projections (analogous to Q/K in attention), complex-valued SSMs can be implemented efficiently with minimal computational overhead. This establishes a theoretical connection between complex-valued SSMs and data-dependent RoPE.

#### 3. Multi-Input Multi-Output (MIMO) SSM

**Design Motivation**: The arithmetic intensity of SSM decoding is extremely low. Standard SISO arithmetic intensity is ~2.5 ops/byte, whereas H100 matmul peak is ~295 ops/byte, indicating that SSM decoding is severely memory-bound.

**SISO→MIMO conversion**:
- Expand $\mathbf{B}_t \in \mathbb{R}^N \to \mathbb{R}^{N \times R}$
- Expand $\mathbf{x}_t \in \mathbb{R}^P \to \mathbb{R}^{P \times R}$
- The outer product $\mathbf{B}_t\mathbf{x}_t^\top$ becomes a matrix multiplication (leveraging tensor cores)

Effect: FLOPs increase by $R\times$ but wall-clock latency remains nearly unchanged (as compute overlaps with memory I/O), raising arithmetic intensity from $\Theta(1)$ to $\Theta(R)$.

**Training**: MIMO decomposes into $R^2$ parallel SISO calls. By adjusting chunk size to $C_{\text{MIMO}} = \frac{1}{R}C_{\text{SISO}}$, total FLOPs increase by only $R\times$ (rather than $R^2$).

**Parameter matching**: The additional parameters from MIMO are compensated by reducing MLP width (only a 6.6% reduction in the 1.5B model).

#### 4. Architectural Refinements

- **BC normalization**: RMSNorm added after B/C projections (analogous to QKNorm in Transformers), enabling removal of the post-gate RMSNorm
- **B/C bias**: Learnable head-specific biases are added, providing a data-independent component (convolution-like behavior)
- **Removal of short convolution**: The combination of exponential-trapezoidal discretization and B/C bias renders the existing short convolution entirely removable

### Loss & Training

- Standard language modeling training: 100B FineWeb-Edu tokens, Llama-3.1 tokenizer, 2K context
- Identical training protocol across all scales for fair comparison
- MIMO rank $R=4$; parameter count matched by reducing MLP width

## Key Experimental Results

### Main Results

1.5B parameter models trained on 100B FineWeb-Edu tokens, average accuracy across 8 downstream tasks:

| Model | FW-Edu ppl↓ | Downstream Avg Acc↑ |
|------|:-:|:-:|
| Transformer-1.5B | 10.51 | 55.4 |
| GDN-1.5B | 10.45 | 55.8 |
| Mamba-2-1.5B | 10.47 | 55.7 |
| **Mamba-3-SISO-1.5B** | **10.35** | **56.4** |
| **Mamba-3-MIMO-1.5B** | **10.24** | **57.6** |

Mamba-3 SISO improves over the second-best model GDN by +0.6 points; MIMO yields a further +1.2 points, for a total gain of +1.8 points. MIMO also improves PPL by 0.11.

| Model | 180M | 440M | 880M | 1.5B |
|------|:-:|:-:|:-:|:-:|
| Mamba-2 | 42.9 | 49.6 | 53.4 | 55.7 |
| Mamba-3 SISO | 43.4 | 49.8 | 54.4 | 56.4 |
| Mamba-3 MIMO | 43.5 | 51.0 | 55.3 | 57.6 |

Mamba-3 outperforms the baseline across all model scales.

### Ablation Study

**Component ablation** (440M scale):

| Variant | PPL↓ |
|------|:-:|
| Mamba-3 - bias - trap | 16.68 |
| Mamba-3 - bias | 16.49 |
| Mamba-3 | 15.72 |
| Mamba-3 + conv | 15.85 |

B/C bias and trapezoidal discretization act synergistically, making the short convolution optional — adding convolution actually increases PPL.

**State-tracking tasks**:

| Model | Parity↑ | Arithmetic (no parens)↑ | Arithmetic (with parens)↑ |
|------|:-:|:-:|:-:|
| Mamba-2 | 0.90 | 47.81 | 0.88 |
| Mamba-3 (w/o RoPE) | 2.27 | 1.49 | 0.72 |
| Mamba-3 (w/ Std. RoPE) | 1.56 | 20.70 | 2.62 |
| **Mamba-3** | **100.00** | **98.51** | **87.75** |
| GDN [-1,1] | 100.00 | 99.25 | 93.50 |

Data-dependent RoPE is the key to state tracking: Mamba-2 fails entirely (near random), while Mamba-3 solves Parity near-perfectly.

**State size experiment**: Mamba-3 MIMO achieves at $d_{\text{state}}=64$ the same PPL that Mamba-2 achieves at $d_{\text{state}}=128$, effectively reaching the same quality at half the latency.

### Key Findings

- Empirical results are better when $\lambda_t$ in exponential-trapezoidal discretization is not constrained to $\frac{1}{2}$
- Under BF16 precision, the Mamba-3 SISO kernel is actually faster than Mamba-2 and GDN (0.156ms vs. 0.203ms vs. 0.257ms)
- MIMO with $R=4$ increases training time by only ~2×, while decoding latency remains nearly unchanged
- Hybrid models (Mamba-3 + NoPE attention, 5:1 ratio) significantly outperform pure linear models on retrieval tasks

## Highlights & Insights

1. **Unification through the SSM perspective**: All three improvements arise naturally from the continuous-time SSM viewpoint — conclusions that would be difficult to reach from a linear attention or test-time regression perspective
2. **Substantial theoretical contribution**: The paper provides the first rigorous proof that Mamba-1/2 discretization constitutes an "exponential-Euler" method, and derives a superior trapezoidal generalization
3. **Complex-valued SSM → RoPE**: The established equivalence between complex-valued SSMs and data-dependent RoPE unifies two independently developed research directions
4. **Practical value of MIMO**: Increasing computation without increasing latency perfectly exploits the hardware idle capacity during decoding

## Limitations & Future Work

- Pure linear models remain noticeably weaker than Transformers on semi-structured/unstructured data extraction (SWDE, FDA)
- The choice of normalization type and placement in hybrid models remains unclear, with competing trade-offs
- Validation is limited to scales ≤1.5B and 100B tokens; large-scale results remain to be confirmed
- Theoretical guidance for selecting the optimal MIMO rank $R$ is still lacking
- Long-context extrapolation requires additional RMSNorm layers, increasing architectural complexity

## Related Work & Insights

- Competitive with Gated DeltaNet (GDN) in performance but methodologically distinct (SSM discretization vs. delta rule)
- The RoPE equivalence of complex-valued SSMs may inspire new positional encoding designs in Transformers
- The arithmetic intensity optimization strategy of MIMO can be generalized to other memory-bound computation scenarios
- Already adopted by large-scale hybrid models including NVIDIA Nemotron and Alibaba Qwen3, validating industrial feasibility

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — All three improvements carry theoretical novelty; the complex-valued SSM–RoPE equivalence is particularly elegant
- Technical Depth: ⭐⭐⭐⭐⭐ — Full-stack coverage from continuous-time ODEs to discrete recurrences to efficient kernel implementation
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Four scales + synthetic tasks + retrieval tasks + kernel performance benchmarks
- Value: ⭐⭐⭐⭐⭐ — Open-sourced and already adopted by industry

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] MambaMia: State-Space Hierarchical Compression for Hour-Long Video Understanding in Large Multimodal Models](../../AAAI2026/video_understanding/state-space_hierarchical_compression_with_gated_attention_an.md)
- [\[ICLR 2026\] The Expressive Limits of Diagonal SSMs for State-Tracking](the_expressive_limits_of_diagonal_ssms_for_state-tracking.md)
- [\[NeurIPS 2025\] Structured Sparse Transition Matrices to Enable State Tracking in State-Space Models](../../NeurIPS2025/video_understanding/structured_sparse_transition_matrices_to_enable_state_tracking_in_state-space_mo.md)
- [\[NeurIPS 2025\] PASS: Path-Selective State Space Model for Event-Based Recognition](../../NeurIPS2025/video_understanding/pass_path-selective_state_space_model_for_event-based_recognition.md)
- [\[CVPR 2026\] HieraMamba: Video Temporal Grounding via Hierarchical Anchor-Mamba Pooling](../../CVPR2026/video_understanding/hieramamba_video_temporal_grounding_via_hierarchical_anchor-mamba_pooling.md)

</div>

<!-- RELATED:END -->
