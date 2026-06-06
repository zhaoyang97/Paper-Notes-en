---
title: >-
  [Paper Note] GIST: Targeted Data Selection for Instruction Tuning via Gradient Subspace Projection
description: >-
  [ICML 2026][LLM Alignment][targeted data selection] GIST formulates "instruction tuning data selection for target tasks" as gradient subspace alignment. It proves that methods like LESS…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "targeted data selection"
  - "instruction tuning"
  - "LoRA"
  - "gradient subspace"
  - "SVD"
date: 2026-05-08
content_hash: c4dfa660e6c77810
---

# GIST: Targeted Data Selection for Instruction Tuning via Gradient Subspace Projection

**Conference**: ICML 2026  
**arXiv**: [2602.18584](https://arxiv.org/abs/2602.18584)  
**Code**: https://github.com/GuanghuiMin/GIST  
**Area**: Data Selection / LLM Instruction Tuning / Optimization Geometry  
**Keywords**: targeted data selection, instruction tuning, LoRA, gradient subspace, SVD

## TL;DR
GIST formulates "instruction tuning data selection for target tasks" as gradient subspace alignment. It proves that methods like LESS, which use Adam states as a diagonal preconditioner, fail on LoRA due to cross-parameter coupling and low-rank task subspaces. Instead, GIST extracts task-specific low-rank subspaces via SVD on validation gradients and selects samples using cosine similarity. It matches or exceeds LESS on MMLU/TydiQA/BBH using only 0.29% of the storage and 25% of the computation time.

## Background & Motivation

**Background**: Instruction tuning is the mainstream for LLM alignment. However, increasing research (e.g., LIMA, AlpaGasus) suggests that data quality is more important than quantity, highlighting "less is more" and stimulating automated data selection research. Targeted instruction tuning further aims to select data that maximizes performance on a specific target task under a limited budget. Existing methods fall into three categories: hard example mining (based on loss/PPL), similarity-based (embedding retrieval), and optimizer-based (using Adam states as a diagonal preconditioner, represented by LESS).

**Limitations of Prior Work**: Using Adam’s diagonal preconditioner as a surrogate for update geometry (like in LESS) is a scalable choice for the LLM era but suffers from structural failure under PEFT (specifically LoRA): (1) The bilinear nature of LoRA $W = W_0 + BA$ introduces cross-block curvature (inherent coupling between $A$ and $B$), which diagonal preconditioners cannot represent; (2) Validation gradients in LLMs exhibit a low-rank structure (attaining 95% variance with rank 150), which axis-aligned diagonals completely ignore in favor of the rotated low-rank subspace. Together, these mean the diagonal is not only a poor approximation but is also systematically biased.

**Key Challenge**: Sample efficiency requires PEFT, yet PEFT introduces parameter coupling. Diagonal approximations cannot represent this coupling, while full Hessian computation is intractable. A non-diagonal but tractable surrogate is needed in between.

**Goal**: To build a data scoring framework that is (a) non-diagonal, (b) robust to cross-parameter coupling in LoRA, and (c) computationally scalable for LLM sizes.

**Key Insight**: Observing that task-specific update directions are concentrated in a low-dimensional subspace (rank 150 captures 95% variance), the full Hessian is unnecessary. One only needs to extract a task-relevant subspace projector $\boldsymbol{\Pi}$ via SVD of validation gradients and use cosine alignment for scoring. This naturally captures coupling while remaining scalable.

**Core Idea**: (1) Perform SVD on the validation gradient matrix $\mathbf{G}_{\text{val}}$ to obtain top-$r$ left singular vectors $\mathbf{U}_r$; (2) Use the projector $\boldsymbol{\Pi} = \mathbf{U}_r^\top$ to project candidate gradients into the task subspace; (3) Use cosine similarity within the subspace for scoring, taking the maximum over multiple target examples for the final score; (4) Select the top-$k$ samples. This entire process uses 0.29% of the storage and 25% of the compute required by LESS.

## Method

### Overall Architecture

The method consists of three steps: (1) Lightweight Warmup: Sample 5% of the candidate dataset $\mathcal{D}'$ and perform LoRA fine-tuning for 1 epoch to reach checkpoint $\boldsymbol{\theta}_t$ (allowing the model to enter a stable basin); (2) Spectral Filtering: Calculate LoRA gradients for each validation sample, stack them into $\mathbf{G}_{\text{val},t} \in \mathbb{R}^{d \times |\mathcal{D}_{\text{val}}|}$, perform SVD to get top-$r$ left singular vectors $\mathbf{U}_r$, and define the projector $\boldsymbol{\Pi} = \mathbf{U}_r^\top$; (3) Geometric Scoring: For each candidate $\boldsymbol{z}_i$, compute $\boldsymbol{\Pi} \mathbf{g}_{i,t}$, calculate cosine similarity with each target's $\boldsymbol{\Pi} \mathbf{g}_{\text{val},t}^{(j)}$, take the maximum as the FinalScore, and select the top-$k$ samples.

### Key Designs

1.  **Unified Theoretical Framework: Data selection as Hessian-preconditioned gradient alignment**:
    - **Function**: Views hard example mining, similarity-based, and optimizer-based methods as different approximations of a common objective, revealing the inherent flaws of diagonal methods.
    - **Mechanism**: From first-order Taylor expansion and Hessian preconditioning, we obtain $\Delta \mathcal{L}_{\text{val}}(\boldsymbol{z}) \approx -\eta \nabla_{\boldsymbol{\theta}} \mathcal{L}(\mathcal{D}_{\text{val}})^\top \mathbf{H}_{\text{val}}^\dagger \nabla_{\boldsymbol{\theta}} \ell(\boldsymbol{z})$. Thus, the selection objective is $\max_{S} \nabla \mathcal{L}_{\text{val}}^\top \mathbf{H}_{\text{val}}^\dagger \nabla \mathcal{L}(S)$ (Theorem 3.1). Hard mining assumes a constant cosine angle and selects by gradient norm; similarity-based methods replace the parameter-space metric with a representation kernel; LESS uses diagonal Adam states to approximate $\mathbf{H}_{\text{val}}^\dagger$, assuming coordinate independence. Ours systematically proves that diagonals cannot work effectively under LoRA.
    - **Design Motivation**: Previously, each method was motivated independently. Ours uses a unified framework to reveal they are all surrogates for $\mathbf{H}_{\text{val}}^\dagger$, making "why the diagonal is insufficient" an analytically tractable problem.

2.  **Theorem on LoRA-induced cross-block curvature**:
    - **Function**: Theoretically proves that the Hessian of LoRA parameters must contain off-diagonal terms, which diagonal preconditioners inevitably miss.
    - **Mechanism**: Theorem 3.2 shows that for LoRA $W = W_0 + BA$, the mixed second derivative $\frac{\partial^2 \mathcal{L}}{\partial B_{ik'} \partial A_{kj}} = \langle \mathbf{H}_W [B_{:k} e_j^\top], e_i A_{k':} \rangle_F + \delta_{kk'} (\mathbf{G}_W)_{ij}$ contains an explicit cross-block term. Specifically, when $k = k'$, the term $(\mathbf{G}_W)_{ij}$ arises directly from the bilinear parameterization. Even if $\mathbf{H}_W$ is diagonal, the LoRA parameter Hessian has cross-block components. The distance between a diagonal preconditioner $\mathbf{D}$ and the true Hessian with off-diagonal component $\rho$ is $\|\mathbf{H} - \mathbf{D}\|_F^2 \ge 2\rho^2$, representing an irreducible error floor.
    - **Design Motivation**: Previous critiques of diagonal methods were empirical; Ours provides a rigorous algebraic proof that "diagonals are structurally inadequate for LoRA," providing a theoretical basis for the necessity of non-diagonal methods.

3.  **Spectral filtering: Extracting low-rank task subspaces to replace full Hessians**:
    - **Function**: Captures cross-parameter coupling without requiring the full Hessian.
    - **Mechanism**: Defines a gradient covariance proxy $\widehat{\mathbf{F}}_{\text{val}} = \mathbf{G}_{\text{val}} \mathbf{G}_{\text{val}}^\top$, which is PSD and non-diagonal. Theorem 3.3 proves that under an NLL objective (where the Hessian has a Gauss-Newton decomposition), the principal angle between the top-$r$ eigenspace of $\widehat{\mathbf{F}}$ and the true Hessian's top-$r$ eigenspace is $\le C \varepsilon_t$, where $\varepsilon_t$ measures residual curvature and proxy mismatch. Once the loss enters a low-value basin, $\varepsilon_t$ is small and the subspace approximation is tight. The Eckart-Young-Mirsky theorem ensures that the SVD-based projector $\boldsymbol{\Pi} = \mathbf{U}_r^\top$ is the optimal rank-$r$ reconstruction. This projector is explicitly non-diagonal and encodes rotation, capturing coupled directions.
    - **Design Motivation**: Full Hessian is intractable, and diagonal methods miss coupling. Low-rank SVD is a tractable compromise supported by the empirical fact that "task gradients themselves possess a low-rank structure." Eigenspace stability theorems provide theoretical support for why a short warmup is sufficient.

### Multi-Task Aggregation

For multi-target tasks $\{\boldsymbol{z}_{\text{val}}^{(1)}, \dots, \boldsymbol{z}_{\text{val}}^{(M)}\}$, each candidate $\boldsymbol{z}_i$ computes cosine similarity with every target, taking the maximum (Maximum Relevance strategy): $\text{FinalScore}(\boldsymbol{z}_i) = \max_j \text{Sim}_t(\boldsymbol{z}_i, \boldsymbol{z}_{\text{val}}^{(j)})$. The reasoning is that averaging dilutes specialist candidates (e.g., a math sample might be buried if averaged with a coding target); taking the maximum preserves the value of specialists.

## Key Experimental Results

### Main Results: MMLU/TydiQA/BBH (k=5%, Llama2-7B)

GIST performs on par with or exceeds LESS, but with significant efficiency gains:

| Metric | LESS | **GIST** |
|--------|------|----------|
| Storage | baseline | **0.29% of LESS** |
| Compute Time | baseline | **25% of LESS** |
| Performance | baseline | match or better |

### Spectral Analysis (Figure 1)

SVD of the validation gradient matrix for Llama2-7B on MMLU shows:
- Rank 150 captures 95% of the explained variance (rapid spectral decay).
- Most variance is concentrated in a low-dimensional subspace.
- This validates the core empirical assumption that "task gradient is intrinsically low-rank."

### Key Findings

- **0.29% Storage + 25% Computation**: GIST only stores low-rank projectors and cosine scores after SVD, whereas LESS must store full candidate gradient features. This storage gap represents orders of magnitude in savings for large candidate pools (270K).
- **Inherent Failure of Diagonals under PEFT**: Theorem 3.2 and Eq. 10 establish an irreducible error of $\|\mathbf{H} - \mathbf{D}\|_F^2 \ge 2\rho^2$, proving that LESS has systematic errors on LoRA. GIST corrects this using a rotation-capable projector.
- **Minimal Warmup Required**: A 5% candidate sample and 1 epoch of LoRA warmup are sufficient to bring the loss into a stable basin, allowing eigenspace stability to yield a meaningful task signal in the SVD subspace.
- **Max Aggregation over Mean**: Using max-cosine for multi-target tasks ensures that specialist candidates (useful for only one target) are retained, which is crucial for multi-task instruction tuning.
- **Stability Across Benchmarks**: Ours achieves gains across MMLU (multiple choice), TydiQA (extractive span), and BBH (generative reasoning), demonstrating the robustness of the method.

## Highlights & Insights

- **Unified Method Taxonomy + Proof of Diagonal Flaws**: By framing hard mining, similarity, and optimizer-based methods as different surrogates for $\mathbf{H}_{\text{val}}^\dagger$, Ours provides a unified geometric view of data selection—a significant methodological contribution.
- **LoRA Cross-Block Curvature Theorem**: Theorem 3.2 uses algebra to prove that diagonals under PEFT inevitably have a $\rho^2$-irreducible error, serving as a theoretical refutation of LESS-style methods.
- **Empirical Factor of Low-Rank Subspaces + Theoretical Stability**: Figure 1's empirical evidence and Theorem 3.3's theory make the use of SVD a solid surrogate both empirically and theoretically.
- **Elegant Application of Eckart-Young-Mirsky**: $\boldsymbol{\Pi} = \mathbf{U}_r^\top$ is not a random projector but the optimal rank-$r$ choice; SVD locks in this optimality.
- **Practicality of 0.29% Storage**: Storing features for 270K candidates × $d$-dimensional LoRA gradients requires gigabytes. GIST only stores $r$-dimensional projectors and scores, making data selection feasible on massive pools from an engineering perspective.
- **Utility of Max Aggregation**: This is consistent with LESS's multi-task aggregation but with stricter reasoning, offering a reproducible recipe for multi-task instruction tuning.

## Limitations & Future Work

- **Choice of Rank $r$**: The experiment uses a heuristic of "capturing 95% variance," but the optimal $r$ may vary by task, and automatic selection of $r$ has not been explored.
- **Warmup Cost**: Training for 1 epoch on 5% of 270K samples still requires 13.5K LoRA training steps, which may be costly for extremely large pools.
- **NLL Assumption**: Theorem 3.3 is rigorous under the NLL objective; theoretical analysis for other losses (e.g., contrastive, RL) requires separate investigation.
- **Static $\boldsymbol{\Pi}$**: The projector is computed once at the warmup checkpoint; dynamically updating the projector across the training trajectory might improve performance but increases cost.
- **Scale of LLMs**: The advantage of GIST has not been verified on models larger than Llama2-7B (e.g., 70B, Mistral, Gemma).
- **Minor Gains on TydiQA**: Compared to MMLU/BBH, the gain on TydiQA is smaller (only matching LESS), possibly because gradient structures for extractive tasks differ from generative ones.

## Related Work & Insights

- **vs LESS (Xia et al. 2024)**: Direct competitor. LESS uses an Adam diagonal preconditioner. GIST proves diagonals are insufficient for LoRA and adopts an SVD subspace instead. GIST matches/exceeds performance with significantly optimized storage and computation.
- **vs RDS / RDS+**: Similarity-based methods use representation kernels, but kernels often mismatch the parameter-space metric. GIST operates directly in the gradient parameter space.
- **vs LIMA / AlpaGasus**: Inspired by work highlighting data quality, but LIMA uses manual selection while GIST automates the process.
- **vs Influence Functions (Koh & Liang 2017)**: The classical theoretical framework; Ours uses low-rank SVD to make influence-style scoring feasible for LLMs.
- **vs Gradient Coverage / Coresets**: Classical data selection approaches; Ours provides LLM-specific subspace alignment.
- **Insights**: (1) Any work using Hessian-preconditioned scoring on LLMs/PEFT should revisit the diagonal assumption; (2) Low-rank SVD on validation gradients is a universal tool for task subspaces, potentially applicable to prompt selection, active learning, or curriculum learning; (3) Max aggregation > mean aggregation is an underrated practical insight for multi-task settings.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Combining unified selection methods, rigorous LoRA cross-block curvature proofs, and SVD subspace projectors makes the methodology comprehensive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid results on MMLU/TydiQA/BBH plus storage/compute and spectral analysis; missing cross-model scale validation and dynamic projector ablation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear algebraic derivations for Theorems 3.1-3.3 and Eq. 10. The Figure 1 spectral analysis is highly motivating. The paper is mathematically rigorous yet intuitive.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses practical pain points in LLM instruction tuning data selection. The 100× storage savings and 4× speedup make GIST feasible for production pipelines. Open-source code lowers the barrier to entry.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] SPARD: Defending Harmful Fine-Tuning Attack via Safety Projection with Relevance-Diversity Data Selection](spard_defending_harmful_fine-tuning_attack_via_safety_projection_with_relevance-.md)
- [\[AAAI 2026\] Importance-Aware Data Selection for Efficient LLM Instruction Tuning](../../AAAI2026/llm_alignment/importance-aware_data_selection_for_efficient_llm_instruction_tuning.md)
- [\[NeurIPS 2025\] T-SHIRT: Token-Selective Hierarchical Data Selection for Instruction Tuning](../../NeurIPS2025/llm_alignment/t-shirt_token-selective_hierarchical_data_selection_for_instruction_tuning.md)
- [\[ACL 2026\] What Makes Good Instruction-Tuning Data? An In-Context Learning Perspective](../../ACL2026/llm_alignment/what_makes_good_instruction-tuning_data_an_in-context_learning_perspective.md)
- [\[NeurIPS 2025\] Improving Data Efficiency for LLM Reinforcement Fine-tuning Through Difficulty-targeted Online Data Selection and Rollout Replay](../../NeurIPS2025/llm_alignment/improving_data_efficiency_for_llm_reinforcement_fine-tuning_through_difficulty-t.md)

</div>

<!-- RELATED:END -->
