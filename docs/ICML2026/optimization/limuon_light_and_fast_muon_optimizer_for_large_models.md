---
title: >-
  [Paper Note] LiMuon: Light and Fast Muon Optimizer for Large Models
description: >-
  [ICML 2026][Optimization][Muon] LiMuon integrates STORM-style momentum variance reduction and Randomized SVD (RSVD) into the Muon optimizer. It compresses the momentum of matrix parameters from $m \times n$ to $(m+n)\hat…
tags:
  - "ICML 2026"
  - "Optimization"
  - "Muon"
  - "STORM Variance Reduction"
  - "Randomized SVD"
  - "Low-rank Momentum"
  - "Generalized Smoothness"
  - "Newton-Schulz"
date: 2026-05-08
content_hash: b58458e3f90ca032
---

# LiMuon: Light and Fast Muon Optimizer for Large Models

**Conference**: ICML 2026  
**arXiv**: [2509.14562](https://arxiv.org/abs/2509.14562)  
**Code**: TBD  
**Area**: Large Model Optimizers / Variance Reduction / Randomized SVD  
**Keywords**: Muon, STORM Variance Reduction, Randomized SVD, Low-rank Momentum, Generalized Smoothness, Newton-Schulz

## TL;DR
LiMuon integrates STORM-style momentum variance reduction and Randomized SVD (RSVD) into the Muon optimizer. It compresses the momentum of matrix parameters from $m \times n$ to $(m+n)\hat{r}$ and reduces the SFO complexity for finding $\epsilon$-stationary points from $\mathcal{O}(\epsilon^{-4})$ to $\mathcal{O}(\epsilon^{-3})$. It achieves lower perplexity/higher accuracy with smaller memory footprints across Mamba-130M, Qwen2.5-0.5B, and ViT.

## Background & Motivation

**Background**: While Adam/AdamW remain dominant for large models, optimizers specifically leveraging the "parameter as matrix/tensor" structure (Shampoo, Muon) have shown higher sample efficiency potential. Muon (Jordan et al., 2024) performs an orthogonalization on the momentum $B_t = \mu B_{t-1} + G_t$ before the update—equivalent to performing SVD $B_t = U \Sigma V^\top$ and using $O_t = U V^\top$ as the update direction. In practice, Newton-Schulz polynomial iterations are used for approximation, showing competitiveness across various LLMs.

**Limitations of Prior Work**: Existing Muon-based works (Shen 2025, SCG, Gluon, GGNC, Muon++, SUMO, etc.) share a common weakness: they either **maintain a sample complexity of $\mathcal{O}(\epsilon^{-4})$** (SCG, Gluon, GGNC, SUMO) or **retain a full-rank state memory of $mn$** (Shen, Muon++). Only Muon++ (Sfyraki & Wang 2025) reduces complexity to $\mathcal{O}(\epsilon^{-3})$ via STORM, but at the cost of storing $mn$ variance-reduced momentum and relying on gradient clipping. In modern LLM layers where $m, n$ reach several thousands, the $mn$ optimizer state represents a significant memory bottleneck.

**Key Challenge**: Reducing sample complexity relies on recursive variance estimation like STORM (based on $M_{t-1}$), which structurally requires retaining full gradient information from the previous step—this naturally conflicts with reducing memory. While SUMO employs subspace projection for memory compression, it requires strong assumptions like bounded objective functions and maintains a complexity of $\mathcal{O}(\epsilon^{-4})$.

**Goal**: To find a Muon variant that **simultaneously** compresses state memory to $(m+n)\hat{r}$ and reduces SFO complexity to $\mathcal{O}(\epsilon^{-3})$, while remaining valid under weaker $(L_0, L_1)$ generalized smoothness conditions and compatible with Newton-Schulz approximations.

**Key Insight**: The authors observe that the $M_t$ stored in STORM estimation is itself a noisy momentum; theoretically, its "significant directions" are far fewer than $\min(m,n)$. Thus, one can recursively use only its low-rank approximation $\hat{M}_t = \hat{U}_t \hat{S}_t \hat{V}_t^\top$ (projected onto $\hat{r} + s$ columns using Halko's Randomized SVD + QR), storing only three small matrices.

**Core Idea**: Replace the original momentum in Muon with the combination of **"STORM recursion + RSVD low-rank compression"**. It is theoretically proven that the bias introduced by low-rank approximation does not degrade the convergence order, while practically saving memory and improving performance metrics.

## Method

### Overall Architecture
LiMuon follows Muon's two-stage process: in each step, it performs (approximate) orthogonalization on a momentum proxy $M_t$ to obtain direction $O_t$, then updates parameters via $W_{t+1} = W_t - \eta_t O_t$. The difference lies entirely in the momentum proxy itself—while Muon uses EMA momentum and Muon++ uses full-rank STORM estimation, LiMuon uses **low-rank STORM** estimation. The paper provides two options: Option #1 retains full-rank $M_t$ (no memory saving, for theoretical comparison), and Option #2 stores the low-rank triplet of $\hat{M}_t$ (recommended for practice). Both Exact-SVD and Newton-Schulz versions are provided.

### Key Designs

1.  **Low-rank STORM Momentum Estimator**:
    -   **Function**: Compresses the storage of variance-reduced momentum from $\mathcal{O}(mn)$ to $\mathcal{O}((m+n)\hat{r})$ while maintaining the $\mathcal{O}(\epsilon^{-3})$ complexity.
    -   **Mechanism**: The update for Option #2 is formulated as $M_{t+1} = \nabla f(W_{t+1}; \xi_{t+1}) + (1 - \beta_{t+1})\big(\hat{M}_t - \nabla f(W_t; \xi_{t+1})\big)$, where $\hat{M}_t = \hat{U}_t \hat{S}_t \hat{V}_t^\top$ is the RSVD low-rank approximation of the previous momentum. Note that the gradient difference $\nabla f(W_{t+1}; \xi_{t+1}) - \nabla f(W_t; \xi_{t+1})$ is still calculated in full-rank (as it is not stored in state), but the **state** retained across steps consists only of $\hat{U}_t \in \mathbb{R}^{m \times \hat{r}}, \hat{S}_t \in \mathbb{R}^{\hat{r} \times \hat{r}}, \hat{V}_t \in \mathbb{R}^{n \times \hat{r}}$, where $m\hat{r} + n\hat{r} + \hat{r}^2 \ll mn$.
    -   **Design Motivation**: Standard STORM requires keeping the previous momentum intact to maintain $\mathcal{O}(\epsilon^{-3})$ complexity, causing memory spikes. Compressing it to a top-$\hat{r}$ subspace using RSVD avoids losing primary directions while matching SUMO's memory efficiency and retaining $\mathcal{O}(\epsilon^{-3})$.

2.  **Practical Orthogonalization based on RSVD**:
    -   **Function**: Implements Muon's per-step SVD orthogonalization using RSVD to avoid the $\mathcal{O}(\min(m,n)^2 \max(m,n))$ cost of exact SVD.
    -   **Mechanism**: RSVD draws a Gaussian random matrix $\Omega \in \mathbb{R}^{n \times (\hat{r} + s)}$, computes $Y = A\Omega$, performs QR decomposition $Y = QR$, then performs exact SVD on the small matrix $B = Q^\top A$ to get $(\tilde{U}, \Sigma, V)$, and recovers $U = Q\tilde{U}$ (with $s \ge 2$ oversampling for stability). This single RSVD process serves both orthogonalization and low-rank momentum compression.
    -   **Design Motivation**: For modern LLM layers with thousands of dimensions, exact SVD at every step is prohibitively expensive. RSVD requires only one matrix multiplication and a small-scale SVD, making it naturally suited for noisy momentum objects with low effective rank.

3.  **Newton-Schulz Compatibility + Generalized Smoothness**:
    -   **Function**: Integrates LiMuon with the Newton-Schulz approximation commonly used in Muon deployments and extends convergence proofs to $(L_0, L_1)$ generalized smoothness.
    -   **Mechanism**: Algorithm 3 replaces the SVD in Algorithm 1 with Newton-Schulz polynomial iterations $X_j = p_\kappa(X_{j-1} X_{j-1}^\top) X_{j-1}$ (default $p_2(z) = 3.4445 - 4.7750z + 2.0315z^2$ for $q$ iterations). The paper proves that under polar approximation error $\varepsilon_q \in (0,1)$ and $\chi_q = 1/(1-\varepsilon_q)$, LiMuon's complexity is $\mathcal{O}(\chi_q^3 \epsilon^{-3})$, strictly superior to the $\mathcal{O}(\chi_q^4 \epsilon^{-4})$ of Muon-NS (Kim & Oh, 2026). Generalized smoothness $\|\nabla F(W) - \nabla F(W')\|_F^2 \le (L_0^2 + L_1^2 \|\nabla F(W)\|_F^2) \|W - W'\|_F^2$ is much weaker than Lipschitz, fitting LLM training reality better.
    -   **Design Motivation**: Pure SVD versions are primarily theoretical; industry users run NS. Extending $\mathcal{O}(\epsilon^{-3})$ to the NS version bridges the "practice-theory gap."

### Loss & Training
The objective is non-convex stochastic optimization $\min_{W \in \mathbb{R}^{m \times n}} \mathbb{E}_{\xi \sim \mathcal{D}}[f(W; \xi)]$, with the stopping criterion being an $\epsilon$-Frobenius / Nuclear norm stationary point. Hyperparameters mainly include step size $\eta_t$, momentum coefficient $\beta_t$, target rank $\hat{r}$, RSVD oversampling $s \ge 2$, and NS iteration count $q$. Theorem 4.7 provides that under $\eta = \mathcal{O}(T^{-2/3}), \beta = \mathcal{O}(T^{-2/3})$, the average gradient nuclear norm is $\le \mathcal{O}(T^{-1/3})$, leading to $T = \mathcal{O}(\epsilon^{-3})$. Notably, LiMuon **does not rely on gradient clipping**, removing a tunable parameter found in Muon++.

## Key Experimental Results

### Main Results
Experiments were conducted on NVIDIA A100-SXM4-80GB, with baselines including Adam / AdamW / Lion / SUMO / Muon / Muon++.

| Model / Dataset | Optimizer | Memory (GB) | Key Metrics | Remarks |
| :--- | :--- | :--- | :--- | :--- |
| Mamba-130M / WikiText-103 | AdamW | 22.92 | val ppl 266.43 | baseline |
| (5k steps, bs=64, seq=256) | Muon | 22.20 | val ppl 71.27 | Matrix orthogonalization improved |
| | Muon++ | 22.35 | val ppl 56.79 | STORM variance reduction |
| | **LiMuon (rank=8)** | **20.25** | val ppl 62.23 | 2 GB less memory, comparable to Muon++ |
| | **LiMuon (full)** | 22.80 | **val ppl 47.78** | Lowest ppl at same memory level |
| Qwen2.5-0.5B / MiniPile | Muon | 54.14 | val ppl 67.60 | – |
| (2k steps, bs=16, seq=1024) | Muon++ | 54.30 | val ppl 82.26 | STORM struggles on larger models |
| | **LiMuon (rank=16)** | 54.21 | val ppl **46.77** | Equal memory, halved ppl |
| | **LiMuon (full)** | 55.15 | val ppl **40.83** | Best across the board |
| ViT / Tiny-ImageNet | Muon | 5.50 | val top-1 47.87% | – |
| (10k steps, bs=128) | SUMO | 5.31 | val top-1 44.23% | Subspace method |
| | **LiMuon (rank=8)** | 5.28 | val top-1 46.75% | More efficient and accurate than SUMO |
| | **LiMuon (full)** | 5.53 | val top-1 **48.04%** | Highest accuracy in class |

### Ablation Study / Complexity Comparison

| Algorithm | SFO Complexity | State Memory | Generalized Smoothness | NS Compatible |
| :--- | :--- | :--- | :--- | :--- |
| Muon (Shen 2025) | $\mathcal{O}(\epsilon^{-4})$ | $mn$ | ✗ | – |
| Muon++ | $\mathcal{O}(\epsilon^{-3})$ | $mn$ | ✗ | – |
| SUMO | $\mathcal{O}(\epsilon^{-4})$ | $(m+n)\hat{r}$ | ✗ | – |
| Gluon / GGNC | $\mathcal{O}(\epsilon^{-4})$ | $mn$ | ✓ | – |
| Muon-NS (Kim & Oh 2026) | $\mathcal{O}(\chi_q^4 \epsilon^{-4})$ | $mn$ | – | ✓ |
| **LiMuon (Exact SVD)** | $\mathcal{O}(\epsilon^{-3})$ | $(m+n)\hat{r}$ | ✓ | – |
| **LiMuon (NS)** | $\mathcal{O}(\chi_q^3 \epsilon^{-3})$ | $(m+n)\hat{r}$ | ✓ | ✓ |

### Key Findings
- **Muon++ performed worse than Muon on Qwen2.5-0.5B** (val ppl 82.26 vs 67.60), suggesting that full-rank STORM in Muon++ becomes unstable as models scale; LiMuon's low-rank momentum is significantly ahead in the same setting, indicating that compression can improve stability.
- **rank=8 / 16 usually approaches full rank**: For Mamba and ViT, rank=8 matches Muon++, and rank=16 surpasses it. This implies that $\hat{r}$ does not need to be large to capture the majority of the benefits.
- **No gradient clipping required**: Compared to Muon++, it has one fewer hyperparameter, making it more engineering-friendly.

## Highlights & Insights
- **First to achieve both "Lower Complexity $\times$ Lower Memory"**: Previous works either prioritized complexity (Muon++) or memory (SUMO). LiMuon uses RSVD to stitch both together without relying on strong "bounded objective function" assumptions.
- **Explicit inclusion of Newton-Schulz error $\chi_q$ in complexity**: Elevating NS approximation from an "engineering hack" to a provable object provides theoretical completion for deployment.
- **Low-rank momentum does not "damage" performance**: Empirical results on LLMs show rank=8/16 matches or exceeds full-rank, providing intuitive evidence that optimizer momentum is inherently a low effective-rank object.

## Limitations & Future Work
- Experimental scale remains medium (Mamba-130M, Qwen2.5-0.5B, ViT-22M); actual savings for 100B+ parameter LLMs still need validation. Specifically, the anomaly of Muon++ on Qwen2.5-0.5B suggests stability differences might widen at larger scales.
- Performing RSVD at every step (even if cheap) increases wall-clock time; Table 5 compares per-step time for ViT but lacks a full end-to-end NS-only vs LiMuon-NS timetable for larger models.
- Target rank $\hat{r}$ is manually set; adaptive ranks (adjusting with training stage or layer spectrum) are an obvious next step.
- Current theory still assumes unbiased stochastic gradients + bounded variance, not covering the coupled analysis of noisy LR schedules, warmup, and weight decay.

## Related Work & Insights
- **vs Muon++ (Sfyraki & Wang 2025)**: Both use STORM to achieve $\mathcal{O}(\epsilon^{-3})$, but Muon++ requires $mn$ state memory and clipping; LiMuon solves memory and clipping needs simultaneously via low-rank momentum.
- **vs SUMO (Refael et al. 2025)**: Both compress memory to $(m+n)\hat{r}$, but SUMO remains $\mathcal{O}(\epsilon^{-4})$ and depends on bounded $F$. LiMuon offers better complexity and weaker assumptions.
- **vs Gluon / GGNC**: Those lines focus on analyzing Muon under $(L_0, L_1)$ generalized smoothness; this work adopts that framework but adds the dual upgrades of variance reduction and low-rank compression.
- **vs Shampoo / KFAC (Second-order methods)**: Different philosophy—second-order methods compress the rank of the preconditioner, while LiMuon compresses the rank of the momentum. Both might be orthogonal and combinable.

## Rating
- Novelty: ⭐⭐⭐⭐ Merging STORM + RSVD into Muon is a clear combinational innovation. The observation of "low-rank momentum" is the first systematic exploration for the Muon series; technically not radical but a correct insight.
- Experimental Thoroughness: ⭐⭐⭐ Cover three architectures (Mamba/Qwen/ViT) with thorough rank ablation, though missing massive model scales and detailed training time analysis. Theoretical tables are clear, but some experimental details (NS-only baseline on large models) are slightly lacking.
- Writing Quality: ⭐⭐⭐⭐ Clear algorithms, theorems, and tables. Assumptions and limitations are well-defined. Table 1's complexity comparison is an excellent example of clarifying prior work vs. current contributions.
- Value: ⭐⭐⭐⭐ Optimizer state is a major memory consumer in LLM training; "better complexity + lower state + no clipping" yields tangible benefits for deployment.

## Rating
- Novelty: TBD
- Experimental Thoroughness: TBD
- Writing Quality: TBD
- Value: TBD

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Pareto-Grid-Guided Large Language Models for Fast and High-Quality Heuristics Design in Multi-Objective Combinatorial Optimization](../../AAAI2026/optimization/pareto-grid-guided_large_language_models_for_fast_and_high-quality_heuristics_de.md)
- [\[ICLR 2026\] Convergence of Muon with Newton-Schulz](../../ICLR2026/optimization/convergence_of_muon_with_newton-schulz.md)
- [\[ICML 2026\] Memory-Efficient LLM Pretraining via Minimalist Optimizer Design](memory-efficient_llm_pretraining_via_minimalist_optimizer_design.md)
- [\[ICML 2026\] Learning a Zeroth-Order Optimizer for Fine-Tuning LLMs](learning_a_zeroth-order_optimizer_for_fine-tuning_llms.md)
- [\[NeurIPS 2025\] Doubly Robust Alignment for Large Language Models](../../NeurIPS2025/optimization/doubly_robust_alignment_for_large_language_models.md)

</div>

<!-- RELATED:END -->
