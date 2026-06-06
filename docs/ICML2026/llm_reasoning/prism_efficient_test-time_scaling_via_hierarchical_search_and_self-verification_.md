---
title: >-
  [Paper Note] Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models
description: >-
  [ICML 2026][LLM Reasoning][dLLM] The authors decompose the problem of "efficient test-time scaling for discrete diffusion language models (dLLM)" into three components: allocating computation along a hierarchical timelin…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "dLLM"
  - "test-time scaling"
  - "hierarchical trajectory search"
  - "self-verification"
  - "partial remask"
date: 2026-05-08
content_hash: 47999fb89518cfbc
---

# Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.01842](https://arxiv.org/abs/2602.01842)  
**Code**: https://github.com/viiika/Prism  
**Area**: LLM Inference / Test-Time Scaling / Discrete Diffusion Language Models  
**Keywords**: dLLM, test-time scaling, hierarchical trajectory search, self-verification, partial remask

## TL;DR
The authors decompose the problem of "efficient test-time scaling for discrete diffusion language models (dLLM)" into three components: allocating computation along a hierarchical timeline of "exploration → progressive pruning → refinement" (HTS), using partial remask for local branching to preserve high-confidence "logic skeletons," and treating the dLLM itself as a Yes/No verifier (SVF). Ultimately, on four math/code benchmarks and three dLLMs, they achieve comparable or better accuracy than best-of-$N$ with far fewer NFE.

## Background & Motivation

**Background**: Test-time scaling (TTS) has become a mainstream tool for enhancing LLM reasoning—chain-of-thought, self-consistency, best-of-$N$, PRM-guided search, etc., are almost all built on autoregressive (AR) decoding: expanding a search tree left-to-right, with little chance to revise once a prefix is fixed. Recently, discrete diffusion language models (dLLM, e.g., LLaDA 8B, Dream 7B, LLaDA 2.0-mini) have emerged, which start from a fully [MASK]ed sequence, denoise in parallel at each step, and have bidirectional context visibility, making them seemingly better suited for planning and self-correction.

**Limitations of Prior Work**: Directly applying AR-era TTS to dLLM faces two issues. (1) dLLM decoding steps are typically locked to sequence length (one step per token), unlike image diffusion which can finish in 10–50 steps, so "length scaling" offers little room; only "width scaling"—running multiple trajectories in parallel—remains. (2) Naively running best-of-$N$ requires $O(NT)$ function evaluations (NFE) for $N$ trajectories and $T$ denoising steps; adding an external PRM/ORM verifier further increases GPU memory and computation. Schedule integration methods like HEX are helpful but still require running all trajectories to completion.

**Key Challenge**: dLLM's parallel denoising leads to dynamics fundamentally different from AR—entropy is high early on and the logic skeleton forms mid-to-late. Spreading compute evenly across all trajectories and time steps means paying full price for every unclear draft in the high-entropy early phase and wasting GPU on already stable trajectories later. Meanwhile, AR-era PRMs are trained on well-formed prefixes and are not calibrated for dLLM's partially masked states.

**Goal**: Decompose into three tasks—(i) allocate trajectory count *non-uniformly* across $T$ denoising steps; (ii) increase *local* diversity without resampling or discarding already formed structures; (iii) provide a reliable scoring signal for partially masked states without external PRMs.

**Key Insight**: The authors observe that dLLM entropy peaks in the early-mid phase and collapses to a logic skeleton later, while best-of-$N$ delays scoring to the end, causing waste. Instead, they propose coarse filtering in the mid-phase and using the dLLM itself for Yes/No prompt-based scoring (reusing a forward pass plus one token cost).

**Core Idea**: By combining "Hierarchical Trajectory Search (HTS) + partial remask local branching + Self-Verified Feedback (SVF)," Prism reduces dLLM TTS complexity from $O(NT)$ to near-linear $O(N+KT)$, where $K\ll N$ is the final refinement width.

## Method

### Overall Architecture
Prism designs a three-stage denoising pipeline for dLLM, using two hyperparameters $W=[w_{\min},w_{\max}]$ to define a "pruning window," corresponding to thresholds $T_p=\lceil w_{\max} T\rceil$ and $T_r=\lceil w_{\min} T\rceil$. Denoising proceeds from $t=T$ down to $t=1$: (1) **Exploration** ($T_p<t\le T$) runs $N$ trajectories for a brief warm-up without pruning; (2) **Thinning** ($T_r<t\le T_p$) shrinks the active pool geometrically as $W_t=\max(\lfloor N\cdot d^{-(T_p-t)}\rfloor, K)$, and every $i$ steps, performs "scoring → select top-$S$ → for each survivor, generate $b_t=\lceil W_{t-1}/S\rceil$ children via partial remask"; (3) **Refinement** ($1\le t\le T_r$) converges to width $K$, with final answers chosen by majority voting. All scoring uses the same dLLM on a specially constructed Yes/No verification prompt, requiring no external models.

### Key Designs

1. **Hierarchical Trajectory Search (HTS)**:

    - **Function**: Reduces best-of-$N$'s $O(NT)$ complexity to $O(N+KT)$, focusing compute on the critical "mid-phase logic skeleton formation" window.
    - **Mechanism**: The denoising schedule is split into three stages. Stage I (high noise) uses $N$ trajectories for random exploration without pruning (since $\hat{\mathbf{z}}_0$ is unstable and SVF unreliable, diversity is prioritized); Stage II (pruning window) prunes every $i$ steps by SVF scoring, retaining top-$S$ seeds, each spawning $b_t$ children via local branching, with the active pool shrinking geometrically as $W_t=\max(\lfloor Nd^{-(T_p-t)}\rfloor, K)$; Stage III stops pruning at width $K$, does pure denoising, and uses $\tau$-confidence threshold and "\boxed{} early stop" for acceleration. Total compute $C_{\mathrm{HTS}}=N(T-T_p)+\sum_{t=T_r+1}^{T_p}|\mathcal{P}_t|+KT_r\approx O(N+KT)$.
    - **Design Motivation**: dLLM entropy decreases monotonically with $t$—early phase is highly divergent, mid-phase logic skeleton emerges, late phase is highly convergent. Thus, "wide early, aggressive mid pruning, fine late refinement" matches its dynamics; geometric decay clears poor trajectories more aggressively than linear, so NFE grows little with $N$ when $K$ is fixed.

2. **Local Branching via Partial Remasking**:

    - **Function**: In the Thinning phase, creates "differentiated but not fully restarted" offspring, preventing top-$S$ from collapsing prematurely to the same local optimum.
    - **Mechanism**: For a survivor state $\mathbf{z}_t$, first obtain $\hat{\mathbf{z}}_0=\mathcal{C}_\theta(\mathbf{z}_t,c,t)$, then compute token-level uncertainty (e.g., entropy), retain high-confidence tokens as the "logic skeleton," and remask the low-confidence subset $\mathcal{I}_t\subseteq\{1,\dots,L\}$ to get $\mathbf{z}_t^{\exp}=\mathrm{Remask}(\mathbf{z}_t;\mathcal{I}_t)$ for further denoising. Each survivor randomly samples different $\mathcal{I}_t$ to generate multiple children.
    - **Design Motivation**: Restarting from $[m]^L$ discards all formed logic structure and wastes compute in the high-entropy early phase; fully inheriting $\mathbf{z}_t$ yields no diversity. Partial remask confines "exploration" to low-confidence positions, effectively "changing implementation details on an already formed solution skeleton," achieving both diversity and reuse.

3. **Self-Verified Feedback (SVF)**:

    - **Function**: Replaces external PRM/ORM, providing a trajectory ranking signal effective even for partially masked states, with minimal memory overhead.
    - **Mechanism**: For each trajectory $\mathbf{z}_t^{(i)}$, use argmax to obtain a complete draft $\hat{\mathbf{z}}_0^{(i)}$, insert it into a Yes/No verification prompt $\pi(c,\hat{\mathbf{z}}_0^{(i)})$, and extract the maximum logits $s_{\text{Yes}},s_{\text{No}}$ for Yes/No token sets from the dLLM. Define $\Phi_{\mathrm{SVF}}(\mathbf{z}_t^{(i)};c)=\exp(s_{\text{Yes}})/(\exp(s_{\text{Yes}})+\exp(s_{\text{No}}))$. SVF is triggered only during thinning, and sparsely at $i$-step intervals.
    - **Design Motivation**: Traditional PRMs are trained on clean prefixes and are not well-calibrated for dLLM's "partially masked" states. Letting the dLLM score itself leverages the same pretraining knowledge to judge if a complete draft "looks correct," and this scoring is insensitive to partial masks (since the evaluation target is the complete draft $\hat{\mathbf{z}}_0$). Reusing the dLLM also saves PRM memory; prefill plus one token decoding is much cheaper than a denoising step.

### Loss & Training
Prism is a purely inference-time method, does not modify dLLM weights, and trains no extra components, so there is no training loss. The dLLM itself is trained with the standard MDM ELBO objective: $\mathcal{L}(\theta)=\mathbb{E}[w(t)\sum_{i:z_{t,i}=m}(-\log\tilde p_\theta(z_{0,i}\mid\mathbf{z}_t,c,t))]$.

## Key Experimental Results

### Main Results
On four benchmarks (GSM8K, MATH500, HumanEval, MBPP) × three dLLMs (LLaDA 8B Instruct, Dream 7B Instruct, LLaDA 2.0-mini), Prism is compared to best-of-$N$ ($N\in\{4,8,16\}$). With $N=16$, $S=K/2$, and target widths $K\in\{2,4,8\}$, representative results (LLaDA 8B Instruct):

| Setting | GSM8K Acc / NFE | MATH500 / NFE | HumanEval / NFE | MBPP / NFE |
|---------|-----------------|---------------|-----------------|------------|
| $N=1$ Baseline | $67.58$ / $256$ | $26.40$ / $256$ | $54.88$ / $512$ | $21.80$ / $512$ |
| best-of-$16$ | $87.50$ / $4096$ | $38.00$ / $4096$ | $82.32$ / $8192$ | $35.20$ / $8192$ |
| Prism $K=2$ | $74.24$ / $283$ | $30.16$ / $334$ | $71.34$ / $549$ | $29.40$ / $561$ |
| Prism $K=4$ | $75.30$ / $509$ | $37.70$ / $622$ | $76.19$ / $1133$ | $32.40$ / $1196$ |
| Prism $K=8$ | $85.30$ / $1048$ | $42.80$ / $1304$ | $79.27$ / $2480$ | $38.20$ / $2576$ |

On MATH500, Prism $K=4$ achieves $37.70$ with about $622$ NFE, close to best-of-$16$'s $38.00$ but at only $\sim 1/7$ the NFE; on MBPP, Prism $K=8$'s $38.20$ even surpasses best-of-$16$'s $35.20$.

### Ablation Study

| Configuration | Key Metric | Notes |
|---------------|------------|-------|
| Full Prism (HTS+SVF+local branch) | See main table | All three components enabled |
| Remove HTS (best-of-$N$) | NFE increases by $N\times$ | Wastes compute on poor trajectories |
| Remove SVF (use PRM/external scoring) | Memory usage surges, scoring unreliable for partial masks | Shows dLLM self-scoring suffices |
| Remove local branch (restart from $[m]^L$ each time) | Early compute wasted | Logic skeleton lost |

### Key Findings
- "Focusing compute on the mid-phase" is the key insight distinguishing dLLM from AR models—AR models condition on fixed prefixes at each step and require broad sampling; dLLM states are highly ambiguous early and highly convergent late, so aggressive mid-phase pruning is meaningful.
- SVF is triggered far fewer times than NFE (one SVF is just prefill + one token decode), so even for Prism $K=8$ on GSM8K, only $33$ extra SVF calls are made, much less than $1048$ denoising steps.
- HTS with geometric decay ($d>1$) prunes trajectories more aggressively than linear decay, but local branching maintains diversity to offset collapse risk, so even $K=2$ can significantly outperform the $N=1$ baseline.

## Highlights & Insights
- "Letting the model score itself" is especially meaningful for dLLM—since dLLM already predicts tokens in parallel at each step, reusing the same forward pass to assess "reasonableness" adds almost no memory overhead, far more efficient than attaching a 7B-scale PRM. This "reuse base model as verifier" idea can be applied to KV cache retrieval, Mixture-of-Experts gating, and other scenarios.
- Local branching uses "high-confidence skeleton + low-confidence token remask" for local exploration, essentially "changing implementation details within the same solution mode," much more robust than restarting from scratch. This leverages dLLM's unique bidirectional context; AR models cannot "swap out only certain positions" in this way.
- The $O(N+KT)$ NFE formula directly reveals Prism's main leverage—breaking best-of-$N$'s multiplicative cost into an additive one, meaning early increases in $N$ add little cost, allowing very large $N$ for random exploration while only refining $K$ trajectories at the end.

## Limitations & Future Work
- HTS involves several key hyperparameters ($N,K,S,d,i,w_{\min},w_{\max}$); while practical defaults are provided, there is a lack of theoretical guidance, and cross-task/model transfer may require retuning.
- SVF assumes the dLLM can distinguish "looks correct" from "looks incorrect," but if the model systematically errs on certain problems (consistent hallucination), self-verification will also fail; a "counterfactual check" is missing.
- Only tested on math and code tasks with clear boxed answers or executable verification; for open-ended long-form generation (creative writing, long summaries), this voting + Yes/No verification paradigm is not applicable.
- Compared to SMC-style importance resampling in PG-DLM, Prism is heuristic; no convergence analysis is provided, and the upper bound on final answer quality via majority voting is unclear.

## Related Work & Insights
- **vs Best-of-$N$**: Runs $N$ trajectories for $T$ steps then selects, $O(NT)$ complexity. Prism prunes to $K$ in the late phase, $O(N+KT)$ complexity, achieving higher accuracy at the same NFE or $4\text{--}8\times$ fewer NFE at the same accuracy.
- **vs HEX (schedule integration)**: HEX augments diversity by combining multiple semi-AR block schedules but still runs all trajectories to completion. Prism's prune-branch mechanism is complementary and can theoretically be combined.
- **vs PG-DLM (SMC for dLLM)**: PG-DLM treats TTS as reward-tilted probabilistic inference with importance resampling and is analyzable; Prism uses SVF for heuristic ranking, top-$S$ hard pruning, and partial remask for local variation—more engineering-oriented but more efficient for verification-type reasoning tasks.
- **vs PRM-type work**: PRM is effective in the AR era but mismatched for partially masked states; Prism uses the same dLLM for Yes/No verification, naturally fitting dLLM's intermediate states.

## Rating
- Novelty: ⭐⭐⭐⭐ Each of the three components (HTS, local branch, SVF) has precedents, but their combination into an NFE-efficient TTS framework specifically for dLLM is the first in the literature.
- Experimental Thoroughness: ⭐⭐⭐⭐ 4 benchmarks × 3 dLLMs × 3 target widths, clear NFE accounting, and breakdown of SVF overhead.
- Writing Quality: ⭐⭐⭐⭐ Complete algorithm pseudocode, straightforward complexity analysis, and convincing "accuracy-NFE curve" in Figure 1; motivation clearly explains dLLM entropy dynamics.
- Value: ⭐⭐⭐⭐⭐ Open-source, plug-and-play, no need to retrain dLLM or external PRM, highly valuable for deployment by reasoning service providers.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](../../NeurIPS2025/llm_reasoning/rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[ICML 2026\] Break the Block: Dynamic-size Reasoning Blocks for Diffusion Large Language Models via Monotonic Entropy Descent with Reinforcement Learning](break_the_block_dynamic-size_reasoning_blocks_for_diffusion_large_language_model.md)
- [\[ICLR 2026\] Plan and Budget: Effective and Efficient Test-Time Scaling on Reasoning LLMs](../../ICLR2026/llm_reasoning/plan_and_budget_effective_and_efficient_test-time_scaling_on_reasoning_large_lan.md)
- [\[NeurIPS 2025\] Sampling-Efficient Test-Time Scaling: Self-Estimating the Best-of-N Sampling in Early Decoding](../../NeurIPS2025/llm_reasoning/sampling-efficient_test-time_scaling_self-estimating_the_best-of-n_sampling_in_e.md)

</div>

<!-- RELATED:END -->
