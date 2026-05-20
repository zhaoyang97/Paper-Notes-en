---
title: >-
  [Paper Note] Out of the Shadows: Exploring a Latent Space for Neural Network Verification
description: >-
  [ICLR 2026][neural network verification] By interpreting zonotopes as "shadows" (projections) of high-dimensional hypercubes, this paper identifies that the input set and output enclosure share a common latent space. Bui…
tags:
  - "ICLR 2026"
  - "neural network verification"
  - "latent space"
  - "zonotope"
  - "branch-and-bound"
  - "reachability analysis"
date: 2026-05-08
content_hash: 0966a530973d1769
---

# Out of the Shadows: Exploring a Latent Space for Neural Network Verification

**Conference**: ICLR 2026
**arXiv**: [2505.17854](https://arxiv.org/abs/2505.17854)  
**Code**: [CORA toolbox](https://cora.in.tum.de/) (MATLAB implementation)  
**Area**: Neural Network Verification / Formal Methods
**Keywords**: neural network verification, latent space, zonotope, branch-and-bound, reachability analysis

## TL;DR

By interpreting zonotopes as "shadows" (projections) of high-dimensional hypercubes, this paper identifies that the input set and output enclosure share a common latent space. Building on this insight, it proposes a specification-driven input refinement method that back-propagates unsafe output constraints into the input space to prune subproblems, reducing branch-and-bound subproblem counts by 60–65%. All operations are matrix-based, enabling efficient GPU acceleration. The method achieves competitive performance with top-tier tools such as α-β-CROWN across eight VNN-COMP'24 benchmarks.

## Background & Motivation

**Background**: Neural networks are widely deployed in safety-critical domains such as autonomous driving and airborne collision avoidance systems, yet they remain sensitive to small input perturbations (adversarial examples). Formal verification before deployment is therefore essential—it must prove that all outputs within a given input set satisfy the safety specification. Mainstream verification approaches fall into two categories: (1) encoding the verification problem for SMT/MIP solvers (e.g., Reluplex, Marabou); (2) using reachability analysis or abstract interpretation to conservatively over-approximate the network's output set (e.g., DeepPoly/CROWN, zonotope propagation).

**Limitations of Prior Work**: The fundamental bottleneck of reachability-based methods is **conservatism**—over-approximation of nonlinear activations (e.g., ReLU) causes the output enclosure to be overly large, frequently yielding inconclusive safety verdicts. To improve precision, branch-and-bound (B&B) recursively partitions the problem into smaller subproblems, but the subproblem count grows **exponentially** with the number of splits. Furthermore, operations relied upon by many state-of-the-art methods (e.g., linear programming in α-β-CROWN) are difficult to batch-process efficiently on GPUs.

**Key Challenge**: A fundamental precision–efficiency trade-off—reducing over-approximation conservatism requires more branching, yet an explosion in subproblem count renders computation infeasible. Existing methods focus primarily on "smarter branching strategies," overlooking the fact that **most inputs are already safe and need not be branched at all**.

**Goal**: How can safe input regions be excluded prior to branching, thereby reducing the number of subproblems that must be processed at the source?

**Key Insight**: The authors identify an elegant geometric observation—a zonotope $Z = c + G\beta$ ($\beta \in [-1,1]^q$) is fundamentally the "shadow" (projection) of a $q$-dimensional hypercube onto a lower-dimensional space via the matrix $G$. When a zonotope propagates through an affine layer of a neural network, only the generator matrix $G$ changes; the domain of $\beta$ (i.e., the hypercube) remains invariant. Non-linear layers (ReLU) introduce additional error generators via linear approximation, extending the dimension of $\beta$ while preserving the first $q_0$ factors shared with the input. Consequently, **the input zonotope and output zonotope are distinct projections of the same high-dimensional hypercube**, naturally linked through a shared latent space.

**Core Idea**: Exploit the shared latent space between input and output zonotopes to back-project unsafe output constraints into input-space constraints, iteratively trimming the input set so that it covers only those inputs that may lead to unsafe outputs.

## Method

### Overall Architecture

Given a neural network $\Phi$, an input set $X$, and an unsafe output set $U$, the algorithm aims to prove $\Phi(X) \cap U = \emptyset$ (safe) or find a counterexample $x \in X$ such that $\Phi(x) \in U$ (unsafe). The overall pipeline is a branch-and-bound loop: (1) propagate a zonotope to enclose the output set → (2) check for intersection with the unsafe set → (3) if inconclusive, attempt to generate a counterexample via the latent space → (4) if falsification fails, refine the input set using the latent space and branch → repeat. The core contributions lie in steps (3) and (4).

### Key Designs

1. **Latent Space and "Shadow" Theory**

    - **Function**: Establish a mathematical link between the input and output spaces, enabling bidirectional transfer of constraints.
    - **Mechanism**: A zonotope $Z = \langle c, G \rangle_Z$ is the image of the hypercube $B^q = [-1,1]^q$ under the affine map $c + G\beta$. When propagated through an affine layer $W_k h + b_k$, the zonotope becomes $\langle W_k c + b_k, W_k G \rangle_Z$—the generator matrix changes but the domain of $\beta$ does not. Non-linear layers (ReLU) introduce additional approximate error generators, extending the dimension of $\beta$ while preserving the first $q_0$ factors shared with the input. Consequently, for any input $x$ and corresponding output $y$, there exists a common $\beta$ such that $x = c_x + G_x \beta_{[q_0]}$ and $y = c_y + G_y \beta_{[q_\kappa]}$.
    - **Design Motivation**: This property is a natural byproduct of zonotope propagation that has previously been overlooked. This paper is the first to systematically exploit it to establish a bridge between the input and output spaces.

2. **Specification-Driven Input Refinement (Proposition 2)**

    - **Function**: Convert unsafe output constraints $U = \{y \mid Ay \leq b\}$ into constraints on $\beta$ in the input space, $C\beta \leq d$, yielding a refined input set $X|_{C\leq d} \subseteq X$.
    - **Mechanism**: Since $y = c_y + G_y \beta$, the unsafe condition $Ay \leq b$ can be rewritten as a linear constraint on $\beta$. Because the dimension of $\beta$ satisfies $q_\kappa > q_0$ (due to additional error generators from ReLU approximation), taking worst-case upper bounds over the extra dimensions yields constraints involving only the first $q_0$ factors: $C = AG_{y(\cdot,[q_0])}$, $d = b - Ac_y + |AG_{y(\cdot,[q_\kappa]\setminus[q_0])}| \mathbf{1}$. This defines a constrained zonotope $X|_{C\leq d}$ that strictly encloses all inputs that may produce unsafe outputs, while excluding safe inputs. The process is iterative: the refined set is re-propagated to obtain a tighter output enclosure, which is then used for further refinement, until verification succeeds or no further progress can be made.
    - **Design Motivation**: Conventional methods branch "blindly" across the input space; this approach instead performs "targeted" pruning guided by the output specification, substantially reducing the number of branches required.

3. **Latent-Space-Based Set Falsification**

    - **Function**: Efficiently search for counterexamples using the latent space when verification is inconclusive.
    - **Mechanism**: For each half-space normal vector $A_{(i,\cdot)}$ of the unsafe set, compute the boundary factor $\tilde{\beta}_i = \text{sign}(A_{(i,\cdot)} G_y)$ and the corresponding input $\tilde{x} = c_x + G_x \tilde{\beta}_{[q_0]}$. If $\Phi(\tilde{x}) \in U$, a counterexample has been found. This approach requires no gradient computation and constitutes a purely set-theoretic adversarial attack.
    - **Design Motivation**: Compared to gradient-based methods such as FGSM, generating counterexamples directly from the set geometry is more efficient and avoids gradient vanishing issues.

### Loss & Training

This paper presents a deterministic formal verification method; no training is involved. All operations—affine mapping of zonotopes, Minkowski sums, constraint projection, and branching—can be implemented using matrix multiplications and element-wise operations, without solving any optimization problem or linear program. This allows the entire algorithm to process multiple subproblems simultaneously in a batched fashion, fully exploiting GPU parallelism. The method is implemented in TUM's CORA toolbox (MATLAB).

## Key Experimental Results

### Main Results

The method is evaluated against the top-5 tools on all 8 standardized benchmarks from VNN-COMP'24 (the international neural network verification competition). The metric is solve rate (%Solved = fraction of instances verified or falsified).

| Benchmark | CORA (Ours) | α-β-CROWN | NeuralSAT | PyRAT | Marabou | nnenum |
|-----------|------------|-----------|-----------|-------|---------|--------|
| acasxu | **99.5%** (138/47) | 100.0% (139/47) | 98.9% (138/46) | 98.9% (137/47) | 96.2% (134/45) | 99.5% (139/46) |
| collins-rul-cnn | **100.0%** (30/32) | 100.0% (30/32) | 100.0% (30/32) | 93.5% (30/28) | 100.0% (30/32) | 100.0% (30/32) |
| cora | 81.1% (18/128) | **87.8%** (24/134) | 87.2% (23/134) | 83.3% (22/128) | 86.7% (22/134) | 14.4% (20/6) |
| dist-shift | **98.6%** (63/8) | 98.6% (63/8) | 98.6% (63/8) | 98.6% (63/8) | 95.8% (62/7) | – |
| linearizenn | **100.0%** (59/1) | 100.0% (59/1) | 100.0% (59/1) | 100.0% (59/1) | 100.0% (59/1) | 98.3% (59/0) |
| meta-room | 97.0% (90/7) | **98.0%** (91/7) | 98.0% (91/7) | 97.0% (91/6) | 53.0% (46/7) | 46.0% (44/2) |
| safenlp | 89.2% (311/652) | **100.0%** (421/659) | 90.5% (327/650) | 79.9% (277/586) | 62.5% (300/375) | 89.2% (321/642) |
| tllverify-bench | **100.0%** (15/17) | 100.0% (15/17) | 100.0% (15/17) | 100.0% (15/17) | 93.8% (13/17) | 56.2% (2/16) |

CORA matches the best performance on 4 out of 8 benchmarks and trails α-β-CROWN by a small margin overall (primarily on safenlp).

### Ablation Study

**Effect of Input Refinement** (ablation of the core contribution):

| Configuration | acasxu Avg. Subproblems↓ | acasxu Max Time↓ | safenlp Avg. Subproblems↓ | safenlp Max Time↓ | safenlp Solve Rate↑ |
|---------------|--------------------------|------------------|---------------------------|-------------------|---------------------|
| Without input refinement | 1611.2 (max 133438) | 32.5s | 4401.8 (max 293304) | 19.9s | 80.8% |
| **With input refinement** | **651.5** (max 36134) | **15.1s** | **1529.2** (max 114065) | **12.4s** | **89.2%** |
| Reduction | **59.6%** | **53.5%** | **65.3%** | **37.7%** | +8.4pp |

**GPU Acceleration Effect**:

| Computation Mode | acasxu Max Time | acasxu Solve Rate | safenlp Max Time | safenlp Solve Rate |
|-----------------|-----------------|-------------------|------------------|--------------------|
| CPU (batch=1) | 40.1s | 95.7% | 14.7s | 83.1% |
| GPU (batch=128) | 7.3s | 99.5% | 2.0s | 86.8% |
| GPU (batch=1024) | **6.9s** | **99.5%** | **1.5s** | **89.2%** |

GPU speedup: 82.8% on acasxu and 89.8% on safenlp.

**Falsification Method Comparison** (safenlp, within the first 50 iterations):

| Method | Instances Falsified | Fraction |
|--------|---------------------|----------|
| FGSM | 257 | 23.8% |
| Ours (set falsification) | **647** | **59.9%** |

The proposed set falsification method finds **60.3% more counterexamples** than FGSM without requiring any gradient computation.

### Key Findings

- **Input refinement is the dominant contributor**: on safenlp, the subproblem count drops from 4,402 to 1,529 (−65.3%), directly yielding an 8.4 percentage-point improvement in solve rate.
- **GPU acceleration is significant but requires a pure-matrix implementation**: conventional methods incorporating LP solvers cannot be efficiently batched on GPUs; the all-matrix design of this work achieves nearly 10× speedup at batch size 1,024.
- **Set falsification substantially outperforms FGSM**: exploiting the latent space geometry to locate boundary points directly is more stable and efficient than gradient-based methods.
- **Near-perfect performance on small networks (acasxu)**, trailing α-β-CROWN by only 0.5%; the gap widens on large-scale benchmarks (safenlp, −10.8%), primarily due to the accumulation of zonotope conservatism in deeper networks.

## Highlights & Insights

- **Elegance of the "shadow" insight**: Reinterpreting zonotope propagation as "distinct projections of the same hypercube onto different spaces" is remarkably concise; the entire input refinement method follows naturally from this observation without any ad hoc design. This approach of discovering new properties within existing mathematical structures is instructive.
- **"Specification-driven" pruning vs. "blind" branching**: Conventional B&B branches uniformly across the entire input space, whereas this method first uses the output specification to prune safe regions before branching—a "reduce first, then expand" strategy that generalizes to other search and optimization problems.
- **Engineering value of the all-matrix design**: Deliberately constraining all operations to matrix arithmetic sacrifices some flexibility but in return delivers out-of-the-box GPU acceleration and batch-processing capability—a significant practical contribution given that GPU support is largely absent in existing verification tools.

## Limitations & Future Work

- **Accumulation of zonotope conservatism in deeper networks**: Each ReLU layer introduces additional approximate error generators, causing both the latent space dimension and the degree of conservatism to grow simultaneously. This is the primary reason for the performance gap relative to α-β-CROWN on safenlp (a deeper network). Tighter bounding techniques such as DeepPoly could be integrated to mitigate this.
- **Restricted to element-wise activation functions**: Although the paper claims support for arbitrary element-wise activations (sigmoid, tanh), experiments are conducted exclusively on ReLU networks; the practical effectiveness on other activations remains unvalidated.
- **MATLAB implementation limits ecosystem integration**: CORA is a MATLAB toolbox, which impedes integration with the mainstream Python/PyTorch ecosystem. Competing Python-based tools such as α-β-CROWN benefit from natural engineering advantages in this regard.
- **Convergence of iterative refinement**: The paper does not discuss the convergence rate or termination criteria of iterative input refinement; diminishing returns may occur in certain pathological cases.
- **No joint treatment of multiple output constraints**: The current method processes each half-space of the unsafe set independently, without accounting for joint relationships among half-spaces.

## Related Work & Insights

- **vs. α-β-CROWN**: α-β-CROWN employs DeepPoly/CROWN linear relaxation with gradient-optimized bound parameters and neuron splitting. This paper uses zonotope propagation with latent-space input refinement and input splitting. α-β-CROWN achieves superior results on large-scale problems (safenlp: 100% vs. 89.2%), but this work is nearly on par for small networks (acasxu) while being methodologically simpler.
- **vs. Marabou/NeuralSAT**: These tools are based on SMT solvers—exact but slow. This paper avoids expensive symbolic reasoning via reachability analysis and latent-space techniques, with a particularly pronounced speed advantage on GPU (meta-room: 97% vs. Marabou's 53%).
- **vs. nnenum**: Both belong to the reachability analysis camp (nnenum uses star sets), but nnenum performs inconsistently across multiple benchmarks. Input refinement combined with GPU acceleration substantially improves the competitiveness of the zonotope-based approach.
- The latent-space methodology is transferable to related problems such as **preimage computation** and **forward reachable set propagation**—the paper already provides a closed-form preimage enclosure in Proposition 2, with potential future applications in safety region annotation and controller verification.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The "shadow"-based input refinement is a genuinely novel contribution, though the core building blocks (zonotopes, B&B) are established techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — All 8 VNN-COMP'24 benchmarks with 3 sets of ablation studies and comprehensive tool comparisons.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The "shadow" metaphor is woven consistently throughout the paper; figures are intuitive and mathematical derivations are rigorous and well-presented.
- **Value**: ⭐⭐⭐⭐ — Opens a new direction for the formal verification community, though its direct impact on the mainstream ML community is limited.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] HypeVPR: Exploring Hyperbolic Space for Perspective to Equirectangular Visual Place Recognition](../../CVPR2026/others/hypevpr_exploring_hyperbolic_space_for_perspective_to_equirectangular_visual_pla.md)
- [\[AAAI 2026\] Learning Compact Latent Space for Representing Neural Signed Distance Functions with High-fidelity Geometry Details](../../AAAI2026/others/learning_compact_latent_space_for_representing_neural_signed_distance_functions_.md)
- [\[ICLR 2026\] Latent Equivariant Operators for Robust Object Recognition: Promises and Challenges](latent_equivariant_operators_for_robust_object_recognition_promises_and_challeng.md)
- [\[ICLR 2026\] Latent Fourier Transform](latent_fourier_transform.md)
- [\[ICLR 2026\] A Single Architecture for Representing Invariance Under Any Space Group](a_single_architecture_for_representing_invariance_under_any_space_group.md)

</div>

<!-- RELATED:END -->
