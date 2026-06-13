---
title: >-
  [Paper Note] DisjunctiveNet: Neural Symbolic Learning via Differentiable Convexified Optimization Layers
description: >-
  [ICML2026][Disjunctive Constraints] This work expresses "input-dependent if-then logical rules" as disjunctive constraints of the union of polyhedra. By convexifying Conjunctive Normal Form (CNF) into the convex hull of…
tags:
  - "ICML2026"
  - "Disjunctive Constraints"
  - "Convex Hull Relaxation"
  - "Differentiable LP Layers"
  - "Hard Constraint Satisfaction"
  - "Input-dependent Rules"
date: 2026-05-08
content_hash: 66bc6281c6fd7e9e
---

# DisjunctiveNet: Neural Symbolic Learning via Differentiable Convexified Optimization Layers

**Conference**: ICML2026  
**arXiv**: [2605.30456](https://arxiv.org/abs/2605.30456)  
**Code**: https://github.com/li-group/DisjunctiveNet.jl  
**Area**: Neuro-symbolic Learning / Differentiable Optimization Layers  
**Keywords**: Disjunctive Constraints, Convex Hull Relaxation, Differentiable LP Layers, Hard Constraint Satisfaction, Input-dependent Rules

## TL;DR
This work expresses "input-dependent if-then logical rules" as disjunctive constraints of the union of polyhedra. By convexifying Conjunctive Normal Form (CNF) into the convex hull of Disjunctive Normal Form (DNF) through a sequence of basic steps, the authors derive a differentiable LP projection layer. Neural network outputs passing through this layer exactly satisfy original MILP-level constraints during both training and inference.

## Background & Motivation
**Background**: Scientific and engineering problems often exhibit two simultaneous characteristics: extreme data sparsity and rich domain knowledge (physical laws, safety constraints, expert heuristics). This knowledge is typically expressed as propositional logic combined with linear inequalities, such as "if $x$ is in state $A$, then output $y$ must satisfy a specific linear inequality." The mainstream approach to integrating such knowledge into neural networks is neuro-symbolic learning.

**Limitations of Prior Work**: Existing methods generally fall into three categories, all with significant drawbacks: soft penalty methods (adding rule violation costs to the loss) do not guarantee feasibility and have hard-to-tune penalty coefficients; specialized architectures (e.g., MultiplexNet) can only encode input-independent global rules; and post-processing methods require non-differentiable ILP decoding during inference. While differentiable optimization layers (e.g., OptNet by Amos & Kolter, CVXPYlayer) achieve end-to-end hard constraints for continuous convex sets, they fail when encountering logical operators (e.g., $\lor$, $\Rightarrow$) due to non-convex and disconnected feasible regions.

**Key Challenge**: MILP is highly expressive, but its optimal solutions are non-differentiable with respect to parameters. Conversely, continuous convex relaxations are differentiable but typically heuristic, offering no guarantee of exactly satisfying the original constraints. Constructing a layer that is both differentiable and capable of exactly satisfying constraints while maintaining MILP/QF-LRA level expressivity is the fundamental difficulty.

**Goal**: (i) Provide a unified constraint format for "input-dependent logic + linear inequalities"; (ii) Construct a corresponding differentiable projection layer where LP vertex solutions exactly satisfy the original non-convex constraints; (iii) Provide a tunable complexity-tightness trade-off between CNF and DNF.

**Key Insight**: The authors borrow from Balas' (2018) Disjunctive Programming theory. Rules are formulated as a union (disjunction) of polyhedra. A "basic step" sequence is used to progressively convexify the Conjunctive Normal Form (CNF) into a Disjunctive Normal Form (DNF). The convex hull of a DNF in a lifted variable space can be represented exactly by an extended formulation LP.

**Core Idea**: Rewrite the "intersection of unions of polyhedra" into a convex LP via DNF expansion and the extended formulation method. An $\ell_1$ epigraph projection is then employed to ensure that vertex solutions satisfy the original constraints, achieving both differentiability and exactness.

## Method

### Overall Architecture
The input $x \in \mathcal{X}$ is fed into a backbone network $f_\theta$ to produce an unconstrained prediction $\hat{y} = f_\theta(x)$. Subsequently, an $\ell_1$ projection is performed: $y^\star(x) \in \arg\min_{y \in \mathcal{F}(x)} \|y - \hat{y}\|_1$, mapping $\hat{y}$ back to the feasible set $\mathcal{F}(x) = \bigcap_{r \in \mathcal{R}(x)} \mathcal{C}_r(x)$ defined by all active rules. Each rule $r$ takes the form $\mathbb{I}[x \in \mathcal{A}_r] \Rightarrow \mathbb{I}[y \in \mathcal{C}_r(x)]$, where $\mathcal{C}_r(x) = \bigcup_{j=1}^{m_r} \{y: A_{rj}(x) y \le b_{rj}(x)\}$ is the union of $m_r$ (potentially input-dependent) polyhedra. The paper proves this constraint class is as expressive as MILP and QF-LRA, covering most practical scenarios. The projection is solved via an extended formulation LP, with a backward pass using implicit differentiation of KKT conditions (via CVXPYlayer / DiffOpt.jl). Quadratic regularization is added to the LP during training to ensure strong convexity and mitigate solution discontinuities caused by parameters appearing in the constraint RHS.

### Key Designs

1.  **$\ell_1$ epigraph + global constraints within each disjunct**:
    *   **Function**: Incorporate the projection objective $\|y - \hat{y}\|_1$ and the global feasible set $\mathcal{G}(x)$ into each individual polyhedral slice $\mathcal{S}_{rj}$, resulting in the extended set $\widehat{\mathcal{S}}_{rj}(x; \hat{y}) = \{(y, \eta): A_{rj}(x) y \le b_{rj}(x), -\eta \le y - \hat{y} \le \eta, y \in \mathcal{G}(x)\}$.
    *   **Mechanism**: $\ell_1$ is chosen over $\ell_2$ because $\|y - \hat{y}\|_1$ possesses a standard epigraph form $\min \mathbf{1}^\top \eta$ s.t. $-\eta \le y - \hat{y} \le \eta$, which can be expressed purely linearly. This preserves the LP structure of the projection problem, which is a prerequisite for the exactness theorem stating that the DNF convex hull equals the original constraint’s convex hull.
    *   **Design Motivation**: If the epigraph were placed outside the convexification process, the "projection direction" would be convexified in the original $y$-space, causing LP vertex solutions to potentially fall outside the original disjuncts, thus losing the exact fulfillment guarantee.

2.  **Basic Step Sequence: Convexification from CNF to DNF Hull**:
    *   **Function**: Define the CNF relaxation $\widetilde{\mathcal{F}}_{\mathrm{CNF}} = \bigcap_r \mathrm{conv}(\widehat{\mathcal{C}}_r)$ (convex hull of each rule separately, then intersected) and the DNF relaxation $\widetilde{\mathcal{F}}_{\mathrm{DNF}} = \mathrm{conv}(\bigcup_{k \in \Pi(x)} \bigcap_r \widehat{\mathcal{S}}_{r,k_r})$ (intersect disjuncts first, then take the convex hull), along with intermediate pDNF forms.
    *   **Mechanism**: Theorem 3.6 proves the DNF relaxation exactly matches the convex hull of the original non-convex set $\widehat{\mathcal{F}}$, and LP vertex solutions must correspond to an original disjunct combination, thus exactly satisfying all rules. CNF relaxations are strictly larger, and their vertices may be infeasible. A basic step moves one rule from CNF to DNF form: $\widetilde{\mathcal{F}}_{\mathrm{pDNF}}(\mathcal{R}_C, \mathcal{R}_D) \to \widetilde{\mathcal{F}}_{\mathrm{pDNF}}(\mathcal{R}_C \setminus \{r'\}, \mathcal{R}_D \cup \{r'\})$, creating a monotonic chain $\widetilde{\mathcal{F}}_{\mathrm{DNF}} \subseteq \widetilde{\mathcal{F}}_{\mathrm{pDNF}} \subseteq \widetilde{\mathcal{F}}_{\mathrm{CNF}}$.
    *   **Design Motivation**: CNF models scale linearly with the number of rules but have loose relaxations; DNF is tight but the number of disjunct combinations $\prod_r m_r$ grows exponentially. Basic steps allow for a trade-off. In practice, DNF expansion for only a few strongly interacting rules yields accuracy close to the full DNF.

3.  **Extended Formulation LP + Implicit Differentiation**:
    *   **Function**: Use the extended formulation from Proposition 3.3 to write the "convex hull of a union of polyhedra" as an LP. This involves introducing variable copies $w_j$ for each disjunct and convex combination weights $\lambda_j$, with constraints $A_j w_j \le \lambda_j b_j$, $w = \sum_j w_j$, $\sum_j \lambda_j = 1$, and $\lambda_j \ge 0$.
    *   **Mechanism**: Epigraph constraints $\eta_k \ge y_k - \lambda_k \hat{y}$ and $\eta_k \ge \lambda_k \hat{y} - y_k$ ensure $\hat{y}$ only appears on the LP's RHS. The forward pass uses an LP solver (Simplex naturally returns vertex solutions), while the backward pass utilizes KKT conditions and implicit differentiation for $\partial y^\star / \partial \hat{y}$. To avoid solution jumps caused by parameters in the constraint LHS, a small quadratic regularization $\mu \|y\|^2$ is added during the backward pass to differentiate it as a strongly convex QP.
    *   **Design Motivation**: The extended formulation avoids binary variables compared to Big-M MILP encoding. The LP structure remains differentiable, and the Simplex vertex property aligns with Theorem 3.6 to achieve both differentiability and exact satisfaction.

### Loss & Training
The total loss consists of the base task loss (MSE for synthetic tasks, cross-entropy for scRNA) plus end-to-end backpropagation through the projection layer. All projection models are initialized from a pre-trained base NN and fine-tuned with the projection layer, allowing the model to focus on "incorporating rules into predictions" rather than "re-learning the base task." Fixed hyperparameters are used across methods, and results are averaged over 3 random seeds.

## Key Experimental Results

### Main Results
Two tasks: (i) Synthetic cooling control (continuous actions subject to global constraints + multiple simultaneously active safety/operational disjunctive rules; oracle is a QP); (ii) Single-cell RNA sequencing (scRNA-seq) classification (marker-gene rules such as "if gene X is high, then the cell is of type Y"). Metrics: MSE / macro-F1 + CSAT (Constraint Satisfaction Rate, calculated only for samples with consistent rules).

| Task | Setting | Metric | Base NN | Penalty (soft) | Fine-Pen | CNF | DNF |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Synthetic Cooling (n=500) | OOD MSE | ↓ | High | Slightly Better | Slightly Better | Significant Drop | **Lowest** |
| Synthetic Cooling (n=500) | OOD CSAT | ↑ | Low | Medium | Medium | High | **100%** |
| Synthetic Cooling (n=500) | IID CSAT | ↑ | Relatively Low | Relatively High | Relatively High | High | **100%** |
| scRNA-seq | macro-F1 / CSAT | ↑ | Baseline | Limited Gain | Limited Gain | Significant Gain | **Best, 100% Rule** |

Key Observation: DNF achieves 100% CSAT on both IID and OOD test sets. CNF is close but not guaranteed. Soft penalties (including fine-pen initialized from pre-training) cannot reliably satisfy constraints even with the same starting point. This validates that the hard projection layer provides not just feasibility but a strong inductive bias, which is particularly effective under OOD conditions.

### Ablation Study
Basic Step Sequence (Sequential Convexification): Moving progressively from CNF by incorporating more rules into the DNF expansion (7 rules total, 0→7).

| Configuration | OOD CSAT | OOD MSE | LP Scale |
| :--- | :--- | :--- | :--- |
| CNF (0 DNF rules) | Low (OOD drop) | High | Small |
| pDNF (1-3 DNF rules) | Monotonic Increase | Monotonic Decrease | Medium |
| pDNF (4-6 DNF rules) | Near DNF | Near DNF | Large |
| DNF (All 7 rules) | **Highest (100%)** | **Lowest** | Largest |
| Computational Overhead (per sample) | CNF 25.03 ms / DNF 28.62 ms | LP Vars: CNF 37.5 / DNF 44.4 | Constraints: CNF 137 / DNF 160 |

### Key Findings
- Sequential convexification leads to monotonic improvements in CSAT and MSE. It quickly approaches DNF performance after a few basic steps, suggesting that the logic of most rules can be captured by expanding only a small subset of strongly interacting rules.
- The DNF projection layer reduces OOD MSE significantly compared to the Base NN, while fine-pen (soft penalty fine-tuning) struggles to catch up. This demonstrates the qualitative advantage of hard constraints as an inductive bias over soft constraints.
- While the number of DNF disjunct combinations is theoretically exponential, many combinations are infeasible or inactive in practice, making the LP scale much smaller than worst-case estimates. The inference time increase (~28 ms vs <1μs) is acceptable for many applications.

## Highlights & Insights
- Utilizing Disjunctive Programming from Operations Research to provide an "end-to-end differentiable hard constraint layer" for neural networks is a significant cross-disciplinary contribution, bringing MILP/QF-LRA expressivity into differentiable frameworks.
- Integrating the $\ell_1$ epigraph into each disjunct *before* convexification is a subtle but critical technical step. It ensures the LP vertex always corresponds to an original disjunct, solving the conflict between "convexification" and "exact satisfaction."
- The pDNF basic step approach is highly engineering-friendly. It maintains the DNF accuracy ceiling while allowing users to select how many rules to expand based on computational budgets.

## Limitations & Future Work
- The number of DNF disjuncts grows exponentially with the number of active rules $|\mathcal{R}(x)|$. Although pruning helps, scalability remains a concern for very large rule sets. The "optimal ordering of basic steps" is also currently an open problem.
- Solving an LP for every sample increases latency by 4-5 orders of magnitude compared to a base NN (25-28 ms vs <1μs), which may be prohibitive for real-time control or massive-scale inference.
- Experiments focused on synthetic control and scRNA-seq with moderate rule counts; scalability to higher-dimensional tasks like interpretable constraints in vision or language remains to be verified.
- The use of strong convexity during the backward pass to handle solution discontinuities is more of an effective engineering trick than a rigorous theoretical solution.

## Related Work & Insights
- **vs. OptNet / CVXPYlayer**: While they handle convex constraints in differentiable layers, this work extends differentiability to non-convex logical/MILP-level constraints using DNF convex hulls and extended formulations.
- **vs. MultiplexNet**: MultiplexNet uses disjunctive constraints but is limited to input-independent global rules and relies on variational inference. This work supports input-dependent rules and guarantees exact satisfaction of all active rules.
- **vs. Soft Penalty / Semantic Loss**: Soft penalties fail to guarantee feasibility and degrade OOD. This work uses hard projections for superior OOD inductive bias.
- **vs. SATNet / LP Relaxation**: These methods use heuristic convex relaxations without satisfaction guarantees. This work utilizes the tightest convex relaxation (convex hull) and Simplex vertices to provide mathematical guarantees of satisfaction.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to systematically apply the "basic step sequence" to differentiable constraint layers, bridging Operations Research and Neuro-symbolic AI.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers synthetic and real biological tasks with multiple baselines and OOD testing, though restricted to julia implementations and moderate rule scales.
- **Writing Quality**: ⭐⭐⭐⭐ Clear presentation of theorems and propositions. Figures 1 and 2 provide excellent geometric intuition for CNF/DNF/lifted projections.
- **Value**: ⭐⭐⭐⭐ Highly practical for scientific/engineering scenarios requiring hard rules (control, biology, compliance). The release of DisjunctiveNet.jl lowers the barrier for adoption.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Scalable GPU-Accelerated Euler Characteristic Curves: Optimization and Differentiable Learning for PyTorch](../../NeurIPS2025/others/scalable_gpu-accelerated_euler_characteristic_curves_optimization_and_differenti.md)
- [\[NeurIPS 2025\] Exact Learning of Arithmetic with Differentiable Agents](../../NeurIPS2025/others/exact_learning_of_arithmetic_with_differentiable_agents.md)
- [\[ICML 2026\] Theoretical Analysis of Sparse Optimization with Reparameterization, Weight Decay, and Adaptive Learning Rate](theoretical_analysis_of_sparse_optimization_with_reparameterization_weight_decay.md)
- [\[AAAI 2026\] DS-ATGO: Dual-Stage Synergistic Learning via Forward Adaptive Threshold and Backward Gradient Optimization for Spiking Neural Networks](../../AAAI2026/others/ds-atgo_dual-stage_synergistic_learning_via_forward_adaptive_threshold_and_backw.md)
- [\[CVPR 2026\] DiffBMP: Differentiable Rendering with Bitmap Primitives](../../CVPR2026/others/diffbmp_differentiable_rendering_with_bitmap_primitives.md)

</div>

<!-- RELATED:END -->
