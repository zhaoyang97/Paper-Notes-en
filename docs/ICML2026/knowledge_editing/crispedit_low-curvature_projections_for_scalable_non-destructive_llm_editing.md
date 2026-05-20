---
title: >-
  [Paper Note] CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing
description: >-
  [ICML 2026][Knowledge Editing][Gauss-Newton Hessian] Formulates LLM editing as "minimize edit loss s.t. capability loss unchanged" constrained optimization…
tags:
  - "ICML 2026"
  - "Knowledge Editing"
  - "Gauss-Newton Hessian"
  - "K-FAC"
  - "Bregman divergence"
  - "Matrix-Free Projection"
  - "Capability Preservation"
date: 2026-05-08
content_hash: c5a256dc17613ff6
---

# CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing

**Conference**: ICML 2026  
**arXiv**: [2602.15823](https://arxiv.org/abs/2602.15823)  
**Code**: https://github.com/zarifikram/CrispEdit  
**Area**: Model Editing / LLM Knowledge Update / Second-Order Optimization  
**Keywords**: Gauss-Newton Hessian, K-FAC, Bregman divergence, Matrix-Free Projection, Capability Preservation

## TL;DR
Formulates LLM editing as "minimize edit loss s.t. capability loss unchanged" constrained optimization, converts it via Bregman divergence equivalence to low-curvature subspace projection using the Gauss-Newton Hessian, and leverages K-FAC plus a Kronecker basis trick that avoids explicit projector construction. This enables 3000 edits on A40 in 6 minutes, while keeping LLaMA-3-8B's average drop on MMLU/IFEval/ARC-C/TruthfulQA/GSM8K under 1%, significantly outperforming AlphaEdit / MEMIT / fine-tuning.

## Background & Motivation
**Background**: LLM knowledge becomes outdated (new facts, events); full retraining is too costly. Model editing injects new facts or removes harmful behaviors by updating a small set of weights, serving as a practical alternative for model updates. Representative methods like ROME / MEMIT locate "knowledge MLP layers" for least-squares updates; AlphaEdit / Adam-NSCL project updates onto the null space of activation covariance; LoRA / FT directly fine-tune a subset of parameters.

**Limitations of Prior Work**: Methods with high edit success often "silently" degrade general capabilities (akin to reward hacking): MEMIT on LLaMA-3-8B with 3000 ZsRE edits drops MMLU from 69.5 to 22.9, GSM8K to 0; AlphaEdit is better but still drops MMLU to 52.7, GSM8K to 45.5. Such issues are invisible in "teacher-forced" evaluation and must be assessed with autoregressive generation (yang-etal-2025-mirage). Existing methods rely on heuristics like "where knowledge is stored" or "activation covariance null space," which are strong assumptions and only indirectly related to capability preservation.

**Key Challenge**: Achieving both "successful edits" and "no degradation of general capability" is equivalent to finding a direction in high-dimensional parameter space that reduces edit loss while barely affecting capability loss—a hard-constrained quadratic program, previously infeasible at LLM scale ($10^{10}$ parameters).

**Goal**: (1) Formalize editing as constrained optimization without Lagrangian relaxation; (2) Replace heuristics with geometric quantities directly tied to capability preservation; (3) Make second-order methods practical for billion-parameter transformers (both memory and runtime).

**Key Insight**: Neural network loss landscapes are highly anisotropic (most Hessian eigenvalues are small), so "moving along low-curvature directions" barely affects capability loss. The second-order Taylor expansion of Bregman divergence equals the Gauss-Newton Hessian, not requiring the base model to be at a stationary point—more realistic than standard Hessian assumptions. K-FAC plus Kronecker basis enables matrix-free GNH projection.

**Core Idea**: Project the edit gradient onto the "γ-approximate null space of capability loss," defined by the Gauss-Newton Hessian, and implement the projection using K-FAC's $A_{l-1} \otimes S_l$ Kronecker decomposition plus a Hadamard mask, achieving $O(d_{\text{in}}^2 + d_{\text{out}}^2)$ memory without explicit projector construction.

## Method

### Overall Architecture
Given base parameters $\theta_0$, capability reference set $\mathcal{D}_{\text{cap}}$ (default: WikiText), and edit set $\mathcal{D}_{\text{edit}}$.  
Stage 1 (precompute, once): For each editable layer $l$, run forward pass on $\mathcal{D}_{\text{cap}}$ to collect K-FAC factors $A_{l-1} = \mathbb{E}[a_{l-1} a_{l-1}^\top]$ and $S_l = \mathbb{E}[g_l g_l^\top]$, perform SVD to get $U_{\text{in}}, U_{\text{out}}, \Lambda_{\text{in}}, \Lambda_{\text{out}}$, and compute mask $M_{ij} = \mathbb{1}[\lambda_i^{\text{out}} \lambda_j^{\text{in}} \le \lambda_\gamma]$.  
Stage 2 (edit training): For each edit batch, compute gradient $Q_l$, project via $Q_l^{\text{proj}} = U_{\text{out}}((U_{\text{out}}^\top Q_l U_{\text{in}}) \odot M) U_{\text{in}}^\top$, and update with PGD, never explicitly constructing the $d_{\text{in}} d_{\text{out}} \times d_{\text{in}} d_{\text{out}}$ projector.  
Stage 3 (optional, sequential editing): Accumulate K-FAC factors online, treating previous edits as new "capability" constraints.

### Key Designs
1. **Bregman Divergence Constraint → Gauss-Newton Hessian**:

    - **Function**: Expresses the "capability loss nearly unchanged" hard constraint as a quadratic form independent of base model convergence, addressing the issue that $\nabla \mathcal{L}_{\text{cap}}(\theta_0) = 0$ does not hold for standard Hessian derivations.
    - **Mechanism**: Defines $\mathsf{d}^{\text{Breg}}_{\ell, y}(f_\theta(x), f_{\theta_0}(x)) = \ell(f_\theta(x), y) - \ell(f_{\theta_0}(x), y) - \langle \nabla \ell(f_{\theta_0}(x), y), f_\theta(x) - f_{\theta_0}(x) \rangle$; its second-order Taylor expansion in $\theta$ naturally nullifies the linear term, yielding $\mathsf{d}^{\text{Breg}} \approx \frac{1}{2} (\theta-\theta_0)^\top G_{\text{cap}} (\theta-\theta_0)$, where $G_{\text{cap}} = \mathbb{E}[J^\top H_{\hat y} J]$ is the Gauss-Newton Hessian. For softmax + cross-entropy, GNH equals the Fisher; K-FAC is a natural approximation.
    - **Design Motivation**: Previous AlphaEdit / Adam-NSCL project onto the null space of activation covariance $K_{\text{cap}}$; Proposition 1 shows $\mathsf{Null}(K_{\text{cap}}^l) \subseteq \mathsf{Null}(G_{\text{cap}}^l)$—i.e., activation covariance null space is a subset of GNH null space, making AlphaEdit an overly conservative special case of CrispEdit. GNH provides a larger feasible direction set, enabling broader edits without harming capability.

2. **K-FAC + Matrix-Free Kronecker Projection**:

    - **Function**: Reduces the memory for low-curvature projection in billion-parameter transformers from $O(d_{\text{in}}^2 d_{\text{out}}^2)$ to $O(d_{\text{in}}^2 + d_{\text{out}}^2)$, without explicit projector construction.
    - **Mechanism**: K-FAC block-diagonalizes GNH by layer, $G_{\text{cap}}^l \approx A_{l-1} \otimes S_l$. Kronecker product eigenvalues are products of the two sides, so $\lambda_{ij} = \lambda_i^{\text{out}} \cdot \lambda_j^{\text{in}}$. For a weight gradient matrix $Q_l$, the projected gradient is $Q_l^{\text{proj}} = U_{\text{out}}((U_{\text{out}}^\top Q_l U_{\text{in}}) \odot M) U_{\text{in}}^\top$, where $M$ is a binary mask retaining only low-curvature (low product eigenvalue) directions. The entire operation requires only 3 matrix multiplies + 1 Hadamard product, with no large projector constructed.
    - **Design Motivation**: Even with K-FAC, explicitly storing a $d_{\text{in}} d_{\text{out}} \times d_{\text{in}} d_{\text{out}}$ projector for LLaMA-3-8B's MLP (4096 × 14336) would require ~3.4TB—impractical. Matrix-free reduces storage to $d_{\text{in}}^2 + d_{\text{out}}^2 \approx 200$M scale.

3. **Sequential Editing: CrispEdit-Seq**:

    - **Function**: Maintains K-FAC sufficient statistics online, treating each new edit as a hard constraint on both "base capability + past edits," mitigating catastrophic forgetting in continual editing.
    - **Mechanism**: Maintains accumulated factors $\{A_{\text{acc}}^{l-1}, S_{\text{acc}}^l\}$; after each round $k$ of edits, merges new edit K-FAC factors via streaming average, and recalculates the projection mask for the next round using the updated accumulated factors. No need to retain historical edit data, suitable for privacy-sensitive scenarios.
    - **Design Motivation**: In natural sequential editing, a series of edits is akin to continual learning and prone to forgetting earlier edits. CrispEdit-Seq incorporates edited data's "capability" into K-FAC factors, forcing subsequent edits to preserve them, while storing only $O(d_{\text{in}}^2 + d_{\text{out}}^2)$ statistics.

### Loss & Training
Constraint: $\min_\theta \mathcal{L}_{\text{edit}}(\theta)$ s.t. $(\theta - \theta_0)^\top G_{\text{cap}} (\theta - \theta_0) \le \varepsilon$. In practice, uses projected gradient descent (PGD) with K-FAC projection, once per epoch. The energy threshold $\gamma \in (0, 1)$ controls projection aggressiveness (paper searches $\gamma = 1 - 10^{-k}, k \in [1/10, 7]$). K-FAC factors are precomputed and cached on $\mathcal{D}_{\text{cap}}$, reusable for subsequent 3000 edits; on LLaMA-3-8B, the full 3000-edit process takes only 4–6 minutes (with cached projector).

## Key Experimental Results

### Main Results
LeNet-5 (MNIST → Fashion-MNIST) controlled experiments first verify: PGD projected onto Hessian low-curvature subspace achieves the best pre-train/fine-tune trade-off, with K-FAC and EK-FAC close behind, far outperforming activation covariance (Adam-NSCL heuristic)—empirically supporting Proposition 1.

LLaMA-3-8B-Instruct on ZsRE / CounterFact / WikiBigEdit with 3000 edits, using WILD (autoregressive) to evaluate edit reliability/generalization, and 5 base benchmarks (MMLU / IFEval / TruthfulQA / ARC-C / GSM8K) for capability preservation:

| Dataset | Method | Edit Rel (QA Context) | Edit Gen (No Context) | MMLU | GSM8K | Time |
|---------|--------|----------------------|----------------------|------|-------|------|
| ZsRE | base | 2.1 | 2.1 | 69.5 | 73.5 | – |
| ZsRE | MEMIT | 0.1 | 0.1 | **22.9** | **0.0** | 9h27m |
| ZsRE | AlphaEdit | 70.1 | 39.4 | 52.7 | 45.5 | 7h19m |
| ZsRE | LocBF-FT | 69.5 | 22.1 | 69.5 | 75.5 | 22m |
| ZsRE | **CrispEdit** | **80.5** | **50.9** | **69.5** | **76.0** | **4m6s** |
| CounterFact | AlphaEdit | 74.9 | 44.1 | 47.4 | 37.5 | 5h56m |
| CounterFact | **CrispEdit** | **79.4** | 32.4 | **69.3** | **76.5** | **3m17s** |

CrispEdit achieves both the highest edit success rate and nearly zero capability drop, while being 100× faster than AlphaEdit.

### Ablation Study

| Configuration | Pre-train Acc | Fine-tune Acc | Notes |
|---------------|--------------|---------------|-------|
| Hessian (gold) | 99% (maintained) | High | Control baseline, computable on LeNet |
| GNH (Bregman) | ≈ Hessian | ≈ Hessian | Bregman as Hessian replacement is nearly lossless |
| K-FAC | Slightly below GNH | ≈ GNH | Block-diag approximation effective |
| EK-FAC (CrispEdit) | ≈ K-FAC | ≈ K-FAC | Comparable to K-FAC |
| Adam-NSCL (activation covariance) | Poor | Poor | Consistent with Prop 1: heuristic is overly conservative |

### Key Findings
- **AlphaEdit is a strict special case of CrispEdit** (Proposition 1): $\mathsf{Null}(K_{\text{cap}}^l) \subseteq \mathsf{Null}(G_{\text{cap}}^l)$, explaining why AlphaEdit's conservativeness drops MMLU by 17 points, while CrispEdit edits more freely without harming capability.
- Autoregressive (WILD) evaluation reveals "teacher-forced evaluation inflation": MEMIT appears effective on traditional ROME-style metrics, but on WILD, GSM8K drops to 0.0.
- With cached K-FAC, editing cost drops from "hours" to "minutes," making productization feasible; 3000 edits on A40 in 6 min.
- LoRA / FT / FT Sequential suffer the most capability drop in sequential settings (LoRA Sequential GSM8K 0.0), while CrispEdit-Seq preserves 73–74.

## Highlights & Insights
- **Bregman divergence → GNH is a beautiful theoretical substitution**: Resolves the impracticality of second-order methods requiring base convergence to a stationary point, opening new avenues for all Hessian-based LLM editing/fine-tuning/continual learning work.
- **Proposition 1 unifies the AlphaEdit / Adam-NSCL lineage**: Clearly states "these methods are special cases of ours," providing theoretical unification and explaining experimental gaps—such "framework" work is highly citable.
- **Matrix-free Kronecker projection**: A numerical linear algebra trick, but the memory/speed gains (3.4TB → 200MB, hours → minutes) are decisive engineering breakthroughs; this technique is directly transferable to any K-FAC application (second-order training, curvature regularization, etc.).
- **Autoregressive (WILD) evaluation**: The paper adopts yang-etal-2025-mirage's true generation evaluation, exposing many seemingly SOTA methods as "teacher-forced illusions"; this is a valuable lesson for those evaluating editing.

## Limitations & Future Work
- The authors acknowledge K-FAC is a block-diagonal approximation, ignoring inter-layer coupling; when editing across multiple layers, approximation may lose accuracy. The paper uses EK-FAC to mitigate but not fully resolve this.
- The choice of "capability reference set" $\mathcal{D}_{\text{cap}}$ is crucial—if it mismatches the target benchmark distribution, the projection may not preserve the relevant capability. The paper uses WikiText as a general corpus, but reasoning-heavy tasks like GSM8K may require a reasoning-focused calibration set.
- Only validated on LLaMA-3-8B and Qwen-2.5-1.5B, not tested on 70B+ models; K-FAC factor size still grows with $d^2$, so further compression is needed for larger models (especially MoE).
- $\gamma$ is a key hyperparameter (energy threshold), requiring task-specific tuning; the paper searches $1 - 10^{-k}$, but does not provide "zero-cost selection of $\gamma$ for new tasks."
- Sequential editing with CrispEdit-Seq still shows some generalization drop (ZsRE: 80.5 → 71.1), indicating streaming K-FAC accumulation is not fully lossless.

## Related Work & Insights
- **vs AlphaEdit / Adam-NSCL**: Both project onto capability null space, but use activation covariance $K_{\text{cap}}$; Proposition 1 proves this is an overly strict special case of CrispEdit. Experimentally, MMLU differs by 17 points, and CrispEdit achieves higher edit success, revealing "conservativeness" is not always "safety."
- **vs MEMIT / ROME**: Both perform "locate + edit," but under autoregressive evaluation, MMLU drops catastrophically (22.9 vs 69.5); CrispEdit does not rely on "knowledge localization" assumptions, making it more broadly applicable.
- **vs LoRA / FT**: Fine-tuning methods collapse in sequential editing (LoRA Sequential GSM8K 0.0) due to lack of explicit capability preservation constraints; CrispEdit implements constraints as projectors, complementing FT.
- **vs UltraEdit**: UltraEdit is faster (3 min), but edit success is only 20.0; CrispEdit achieves 80.5 in 4 min, dominating the time-quality Pareto frontier.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Bregman → GNH substitution + matrix-free Kronecker projection are clear innovations; Proposition 1 subsumes prior methods as special cases.
- Experimental Thoroughness: ⭐⭐⭐⭐ 2 bases × 3 edit datasets × 5 capability benchmarks × autoregressive evaluation, including sequential and small-scale controls; lacks 70B+ validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Figure 2 geometric intuition, Proposition 1 rigorous proof, Algorithm 1/2 pseudocode, and clear experimental tables; concise writing.
- Value: ⭐⭐⭐⭐⭐ Provides a truly practical solution for "productized model editing" (4 min, 1% drop), unifies multiple heuristic editing methods, with high academic and engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Low-Rank Curvature for Zeroth-Order Optimization in LLM Fine-Tuning](../../AAAI2026/llm_evaluation/low-rank_curvature_for_zeroth-order_optimization_in_llm_fine-tuning.md)
- [\[AAAI 2026\] LLM-as-a-Judge for Scalable Test Coverage Evaluation](../../AAAI2026/llm_evaluation/llm-as-a-judge_for_scalable_test_coverage_evaluation_accuracy_operational_reliab.md)
- [\[ICML 2026\] DoLQ: 用 LLM 做定性 + 定量评估发现常微分方程](discovering_ordinary_differential_equations_with_llm-based_qualitative_and_quant.md)
- [\[ICML 2026\] Reasoning Is Not Free: Robust Adaptive Cost-Efficient Routing for LLM-as-a-Judge](reasoning_is_not_free_robust_adaptive_cost-efficient_routing_for_llm-as-a-judge.md)
- [\[ICLR 2026\] Non-Clashing Teaching in Graphs: Algorithms, Complexity, and Bounds](../../ICLR2026/llm_evaluation/non-clashing_teaching_in_graphs_algorithms_complexity_and_bounds.md)

</div>

<!-- RELATED:END -->
