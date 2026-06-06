---
title: >-
  [Paper Note] CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing
description: >-
  [ICML 2026][Knowledge Editing][Gauss-Newton Hessian] The authors formulate LLM editing as a constrained optimization problem: "minimize edit loss s.t. capability loss remains unchanged." This is equivalently transformed…
tags:
  - "ICML 2026"
  - "Knowledge Editing"
  - "Gauss-Newton Hessian"
  - "K-FAC"
  - "Bregman divergence"
  - "Matrix-free projection"
  - "Capability preservation"
date: 2026-05-08
content_hash: 6d11ba42705ac7b0
---

# CrispEdit: Low-Curvature Projections for Scalable Non-Destructive LLM Editing

**Conference**: ICML 2026  
**arXiv**: [2602.15823](https://arxiv.org/abs/2602.15823)  
**Code**: https://github.com/zarifikram/CrispEdit  
**Area**: Model Editing / LLM Knowledge Update / Second-order Optimization  
**Keywords**: Gauss-Newton Hessian, K-FAC, Bregman divergence, Matrix-free projection, Capability preservation

## TL;DR
The authors formulate LLM editing as a constrained optimization problem: "minimize edit loss s.t. capability loss remains unchanged." This is equivalently transformed into a low-curvature subspace projection of the Gauss-Newton Hessian (GNH) using Bregman divergence. By leveraging K-FAC and a matrix-free Kronecker basis technique that avoids explicit construction of the projection matrix, 3,000 edits are completed in 6 minutes on an A40. Meanwhile, the average performance degradation across MMLU/IFEval/ARC-C/TruthfulQA/GSM8K for LLaMA-3-8B is kept under 1%, significantly outperforming AlphaEdit, MEMIT, and fine-tuning.

## Background & Motivation
**Background**: LLM knowledge becomes outdated (new facts, new events), and full retraining is prohibitively expensive. Model editing serves as a practical alternative by updating a small amount of weights to inject new facts or remove harmful behaviors. Representative methods like ROME and MEMIT identify "knowledge-storing MLP layers" for least-squares updates; AlphaEdit and Adam-NSCL project updates onto the null space of activation covariance; LoRA and FT directly fine-tune a subset of parameters.

**Limitations of Prior Work**: Methods with high edit success rates often "quietly" destroy general capabilities (akin to reward hacking). For example, after 3,000 ZsRE edits on LLaMA-3-8B, MEMIT causes MMLU to plummet from 69.5 to 22.9 and GSM8K to 0. These issues are often invisible in "teacher-forced" evaluations and must be assessed via autoregressive generation (yang-etal-2025-mirage). Furthermore, existing methods rely on heuristics such as "knowledge location" or "activation covariance null space," which have strong assumptions and are only indirectly related to capability preservation.

**Key Challenge**: Achieving both "successful editing" and "preservation of general capability" is equivalent to finding a direction in a high-dimensional parameter space that minimizes edit loss without perturbing capability loss. This is a hard-constrained quadratic programming problem that was previously infeasible at the scale of LLMs ($10^{10}$ parameters).

**Goal**: (1) Formalize editing as constrained optimization without Lagrangian relaxation; (2) Replace heuristics with geometric quantities directly linked to capability preservation; (3) Make second-order methods feasible for billion-parameter Transformers in terms of both memory and time.

**Key Insight**: The authors observe that the loss landscape of neural networks is highly anisotropic (most Hessian eigenvalues are very small); thus, moving along low-curvature directions barely affects capability loss. They also note that a second-order Taylor expansion of the Bregman divergence is exactly equal to the Gauss-Newton Hessian. This does not require the base model to have converged to a stationary point, making it more realistic than standard Hessian assumptions. Finally, K-FAC and Kronecker basis techniques are used to make the GNH projection matrix-free.

**Core Idea**: Edit gradients are projected into the "$\gamma$-approximate null space of capability loss," which is defined by the Gauss-Newton Hessian. This is implemented via $A_{l-1} \otimes S_l$ Kronecker decomposition from K-FAC combined with a Hadamard mask, achieving matrix-free projection with $O(d_{\text{in}}^2 + d_{\text{out}}^2)$ memory complexity.

## Method

### Overall Architecture
Given base parameters $\theta_0$, a capability reference set $\mathcal{D}_{\text{cap}}$ (default WikiText), and an edit set $\mathcal{D}_{\text{edit}}$. Stage 1 (Precomputation, one-time): For each layer $l$ to be edited, collect K-FAC factors $A_{l-1} = \mathbb{E}[a_{l-1} a_{l-1}^\top]$ and $S_l = \mathbb{E}[g_l g_l^\top]$ over $\mathcal{D}_{\text{cap}}$. Perform SVD to obtain $U_{\text{in}}, U_{\text{out}}, \Lambda_{\text{in}}, \Lambda_{\text{out}}$, and compute the mask $M_{ij} = \mathbb{1}[\lambda_i^{\text{out}} \lambda_j^{\text{in}} \le \lambda_\gamma]$. Stage 2 (Edit Training): Compute the gradient $Q_l$ for an edit batch, apply the projection $Q_l^{\text{proj}} = U_{\text{out}}((U_{\text{out}}^\top Q_l U_{\text{in}}) \odot M) U_{\text{in}}^\top$, and perform a PGD update. Crucially, no $d_{\text{in}} d_{\text{out}} \times d_{\text{in}} d_{\text{out}}$ projection matrix is explicitly constructed. Stage 3 (Optional, Sequential Editing): Online accumulation of K-FAC factors, treating previous edits as new capability constraints.

### Key Designs
1.  **Bregman Divergence Constraint $\rightarrow$ Gauss-Newton Hessian**:
    - **Function**: Formulates the hard constraint of "near-constant capability loss" as a quadratic form that does not depend on whether the base model has converged, addressing the issue in standard Hessian derivations where $\nabla \mathcal{L}_{\text{cap}}(\theta_0) = 0$ does not hold.
    - **Mechanism**: Define $\mathsf{d}^{\text{Breg}}_{\ell, y}(f_\theta(x), f_{\theta_0}(x)) = \ell(f_\theta(x), y) - \ell(f_{\theta_0}(x), y) - \langle \nabla \ell(f_{\theta_0}(x), y), f_\theta(x) - f_{\theta_0}(x) \rangle$. Its second-order Taylor expansion with respect to $\theta$ has a vanishing linear term, yielding $\mathsf{d}^{\text{Breg}} \approx \frac{1}{2} (\theta-\theta_0)^\top G_{\text{cap}} (\theta-\theta_0)$, where $G_{\text{cap}} = \mathbb{E}[J^\top H_{\hat y} J]$ is the Gauss-Newton Hessian. Under softmax and cross-entropy, GNH is equivalent to Fisher, and K-FAC provides a natural approximation.
    - **Design Motivation**: Previous methods like AlphaEdit/Adam-NSCL project onto the null space of activation covariance $K_{\text{cap}}$. Proposition 1 in the paper proves that $\mathsf{Null}(K_{\text{cap}}^l) \subseteq \mathsf{Null}(G_{\text{cap}}^l)$, meaning the activation covariance null space is a subset of the GNH null space. AlphaEdit is thus an overly conservative special case of CrispEdit. GNH provides a larger set of feasible directions, allowing for more flexible editing without damaging capability.

2.  **K-FAC + Matrix-free Kronecker Projection**:
    - **Function**: Reduces the memory for low-curvature projection in billion-parameter Transformers from $O(d_{\text{in}}^2 d_{\text{out}}^2)$ to $O(d_{\text{in}}^2 + d_{\text{out}}^2)$ without explicit projection matrix construction.
    - **Mechanism**: K-FAC diagonalizes GNH by layer blocks, where $G_{\text{cap}}^l \approx A_{l-1} \otimes S_l$. The eigenvalues of a Kronecker product are the products of the eigenvalues of the factors: $\lambda_{ij} = \lambda_i^{\text{out}} \cdot \lambda_j^{\text{in}}$. For a weight gradient matrix $Q_l$, the projected gradient is $Q_l^{\text{proj}} = U_{\text{out}}((U_{\text{out}}^\top Q_l U_{\text{in}}) \odot M) U_{\text{in}}^\top$, where $M$ is a binary mask preserving only low-curvature (low product eigenvalue) components. This requires only three matrix multiplications and one Hadamard product.
    - **Design Motivation**: Even with K-FAC, explicitly storing $d_{\text{in}} d_{\text{out}} \times d_{\text{in}} d_{\text{out}}$ projectors for LLaMA-3-8B MLPs (4096 $\times$ 14336) would require ~3.4TB, which is infeasible. Matrix-free techniques reduce storage to the order of $d_{\text{in}}^2 + d_{\text{out}}^2 \approx 200\text{M}$.

3.  **Sequential Editing (CrispEdit-Seq)**:
    - **Function**: Maintains K-FAC sufficient statistics online, allowing each new round of editing to treat "base model capability + historical edits" as hard constraints, mitigating catastrophic forgetting in continual editing.
    - **Mechanism**: Maintains accumulated factors $\{A_{\text{acc}}^{l-1}, S_{\text{acc}}^l\}$. After each round $k$ of editing, K-FAC factors from the new edit are merged via streaming average. The projection mask for the next round is recomputed from the updated cumulative factors. This does not require retaining historical edit data.
    - **Design Motivation**: In sequential editing, a series of edits acts as "continual learning," which is prone to forgetting. CrispEdit-Seq encodes the "capability" of edited data into the K-FAC factors, forcing subsequent edits to preserve them while only storing $O(d_{\text{in}}^2 + d_{\text{out}}^2)$ statistics.

### Loss & Training
Constraint: $\min_\theta \mathcal{L}_{\text{edit}}(\theta)$ s.t. $(\theta - \theta_0)^\top G_{\text{cap}} (\theta - \theta_0) \le \varepsilon$. In practice, Projected Gradient Descent (PGD) with K-FAC projection is used once per epoch. The energy threshold $\gamma \in (0, 1)$ controls projection aggressiveness (searched as $\gamma = 1 - 10^{-k}, k \in [1/10, 7]$). K-FAC factors are precomputed once on $\mathcal{D}_{\text{cap}}$ and cached; 3,000 edits on LLaMA-3-8B take only 4–6 minutes using the cached projector.

## Key Experimental Results

### Main Results
A controlled experiment on LeNet-5 (MNIST $\rightarrow$ Fashion-MNIST) verified that projecting into the Hessian low-curvature subspace yields the best pre-train/fine-tune trade-off, followed closely by K-FAC and EK-FAC, outperforming activation covariance (Adam-NSCL heuristic), empirically supporting Proposition 1.

For LLaMA-3-8B-Instruct with 3,000 edits on ZsRE / CounterFact / WikiBigEdit, edit reliability/generalization were evaluated using WILD (autoregressive). Capability preservation was evaluated on five benchmarks:

| Dataset | Method | Edit Rel (QA Context) | Edit Gen (No Context) | MMLU | GSM8K | Time |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ZsRE | base | 2.1 | 2.1 | 69.5 | 73.5 | – |
| ZsRE | MEMIT | 0.1 | 0.1 | **22.9** | **0.0** | 9h27m |
| ZsRE | AlphaEdit | 70.1 | 39.4 | 52.7 | 45.5 | 7h19m |
| ZsRE | LocBF-FT | 69.5 | 22.1 | 69.5 | 75.5 | 22m |
| ZsRE | **Ours** | **80.5** | **50.9** | **69.5** | **76.0** | **4m6s** |
| CounterFact | AlphaEdit | 74.9 | 44.1 | 47.4 | 37.5 | 5h56m |
| CounterFact | **Ours** | **79.4** | 32.4 | **69.3** | **76.5** | **3m17s** |

CrispEdit achieves the highest edit success rate with almost zero capability drop, while being 100$\times$ faster than AlphaEdit.

### Ablation Study

| Configuration | Pre-train Acc | Fine-tune Acc | Note |
| :--- | :--- | :--- | :--- |
| Hessian (gold) | 99% (Maintained) | High | Control baseline for LeNet |
| GNH (Bregman) | $\approx$ Hessian | $\approx$ Hessian | Bregman replacement is near lossless |
| K-FAC | Slightly < GNH | $\approx$ GNH | Block-diag approximation is effective |
| EK-FAC (Ours) | $\approx$ K-FAC | $\approx$ K-FAC | Comparable to K-FAC |
| Adam-NSCL (Activation Cov) | Poorer | Poorer | Consistent with Prop 1: overly conservative |

### Key Findings
- **AlphaEdit is a strict subcase of CrispEdit** (Proposition 1): $\mathsf{Null}(K_{\text{cap}}^l) \subseteq \mathsf{Null}(G_{\text{cap}}^l)$. This explains why AlphaEdit drops 17 MMLU points due to over-conservatism, while CrispEdit edits freely in a wider range of directions to maintain performance.
- Autoregressive evaluation (WILD) reveals that "teacher-forced evaluation scores are inflated." MEMIT looks acceptable on traditional ROME metrics but scores 0.0 on GSM8K in WILD.
- After caching K-FAC, editing costs drop from hours to minutes, enabling production readiness; 3,000 edits in 6 mins on an A40.
- LoRA / FT / FT Sequential suffer the most in sequential settings (LoRA Sequential GSM8K 0.0), whereas CrispEdit-Seq maintains 73–74.

## Highlights & Insights
- **Bregman Divergence $\rightarrow$ GNH is an elegant theoretical replacement**: It bypasses the impractical requirement for base models to converge to a stationary point, opening new avenues for Hessian-based LLM editing, fine-tuning, and continual learning.
- **Proposition 1 unifies the AlphaEdit / Adam-NSCL lineage**: By demonstrating that these methods are special cases, the paper provides theoretical unity and explains the experimental performance gap. Such "framework" contributions are highly valuable.
- **Matrix-free Kronecker Projection**: While a numerical linear algebra trick, the resulting memory/speed gains (3.4TB $\rightarrow$ 200MB, hours $\rightarrow$ minutes) are engineering breakthroughs. This technique is transferable to any K-FAC application (second-order training, curvature regularization, etc.).
- **Autoregressive (WILD) Evaluation**: Utilizing realistic generation evaluation (yang-etal-2025-mirage) exposes many SOTA methods as "teacher-forced illusions." This is a crucial lesson for future editing research.

## Limitations & Future Work
- The authors acknowledge K-FAC is a block-diagonal approximation that ignores cross-layer coupling. Accuracy might degrade for edits across many layers; EK-FAC mitigates but does not fully solve this.
- Choice of the "capability reference set" $\mathcal{D}_{\text{cap}}$ is critical. For tasks like GSM8K, reasoning-heavy calibration sets might be superior to general corpora like WikiText.
- Validation was limited to LLaMA-3-8B and Qwen-2.5-1.5B; 70B+ models were not tested. K-FAC factor scale still grows with $d^2$, requiring further compression for massive models or MoEs.
- $\gamma$ (energy threshold) is a key hyperparameter requiring task-specific tuning. The paper lacks a "zero-shot" method for selecting $\gamma$ on new tasks.
- CrispEdit-Seq still sees some generalization drop (80.5 $\rightarrow$ 71.1 on ZsRE), indicating that streaming K-FAC accumulation is not yet perfectly lossless.

## Related Work & Insights
- **vs AlphaEdit / Adam-NSCL**: Both use null space projections but rely on activation covariance $K_{\text{cap}}$. Proposition 1 proves this is a restrictive special case of CrispEdit, explaining the 17-point MMLU gap.
- **vs MEMIT / ROME**: Both use "locate + edit," but suffer catastrophic MMLU drops (22.9 vs 69.5) under autoregressive evaluation. CrispEdit doesn't rely on "knowledge localization" assumptions.
- **vs LoRA / FT**: Fine-tuning methods collapse in sequential settings (LoRA Sequential GSM8K 0.0) without explicit capability constraints. CrispEdit is complementary to FT.
- **vs UltraEdit**: UltraEdit is faster (3 min) but has a success rate of only 20.0. CrispEdit achieves 80.5 in 4 min, dominating the time-quality Pareto front.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Bregman $\rightarrow$ GNH replacement + matrix-free Kronecker projection is original; Proposition 1 unifies prior work.
- Experimental Thoroughness: ⭐⭐⭐⭐ 2 bases $\times$ 3 edit datasets $\times$ 5 capability benchmarks $\times$ autoregressive evaluation, including sequential settings. Lacks 70B+ validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Geometric intuition in Figure 2, rigorous proof for Proposition 1, clear algorithms, and well-structured tables.
- Value: ⭐⭐⭐⭐⭐ Provides a production-ready solution (4 min, 1% drop) and unifies heuristic methods into a single framework. High academic and engineering value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] From Backward Spreading to Forward Replay: Revisiting Target Construction in LLM Parameter Editing](from_backward_spreading_to_forward_replay_revisiting_target_construction_in_llm_.md)
- [\[ACL 2026\] CLaRE-ty Amid Chaos: Quantifying Representational Entanglement to Predict Ripple Effects in LLM Editing](../../ACL2026/knowledge_editing/clare-ty_amid_chaos_quantifying_representational_entanglement_to_predict_ripple_.md)
- [\[ICML 2026\] Reverse-Engineering Model Editing on Language Models](reverse-engineering_model_editing_on_language_models.md)
- [\[ICML 2026\] AnyEdit++: Adaptive Long-Form Knowledge Editing via Bayesian Surprise](anyedit_adaptive_long-form_knowledge_editing_via_bayesian_surprise.md)
- [\[ICML 2026\] Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs](do_text_edits_generalize_to_visual_generation_benchmarking_cross-modal_knowledge.md)

</div>

<!-- RELATED:END -->
