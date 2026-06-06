---
title: >-
  [Paper Note] Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models
description: >-
  [ICML 2026][LLM Reasoning][dLLM] The authors decompose the problem of "efficient test-time scaling for discrete diffusion language models (dLLMs)" into three components: hierarchical trajectory search (HTS) to allocate c…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "dLLM"
  - "test-time scaling"
  - "hierarchical trajectory search"
  - "self-verification"
  - "partial remasking"
date: 2026-05-08
content_hash: 8aae60354c5ec2a7
---

# Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models

**Conference**: ICML 2026  
**arXiv**: [2602.01842](https://arxiv.org/abs/2602.01842)  
**Code**: https://github.com/viiika/Prism  
**Area**: LLM Inference / Test-time Compute Scaling / Discrete Diffusion Language Models  
**Keywords**: dLLM, test-time scaling, hierarchical trajectory search, self-verification, partial remasking

## TL;DR
The authors decompose the problem of "efficient test-time scaling for discrete diffusion language models (dLLMs)" into three components: hierarchical trajectory search (HTS) to allocate computation via an "exploration $\rightarrow$ progressive thinning $\rightarrow$ refinement" schedule; local branching via partial remasking to preserve high-confidence "logic skeletons"; and self-verified feedback (SVF) treating the dLLM itself as a Yes/No verifier. Ultimately, Prism achieves comparable or superior accuracy to best-of-$N$ on four math/code benchmarks across three dLLMs with significantly fewer NFEs.

## Background & Motivation

**Background**: Test-time scaling (TTS) has become a primary tool for enhancing reasoning in LLMs. Techniques like chain-of-thought, self-consistency, best-of-$N$, and PRM-guided search almost exclusively rely on autoregressive (AR) decoding, where search trees expand from left to right, making it difficult to backtrack once a prefix is committed. In contrast, emerging discrete diffusion language models (dLLMs), such as LLaDA and Dream, start from a sequence of full [MASK] tokens and perform parallel denoising with bidirectional context, appearing more suitable for planning and self-correction.

**Limitations of Prior Work**: Directly applying AR-era TTS to dLLMs faces two specific issues: (1) The number of decoding steps in dLLMs is often locked to the sequence length (one step per token), unlike image diffusion which can be compressed to 10–50 steps, leaving little room for "length scaling"; thus, scaling is restricted to "width scaling" (running multiple trajectories). (2) Standard best-of-$N$ requires $O(NT)$ function evaluations (NFE) for $N$ trajectories and $T$ denoising steps. Adding an external PRM/ORM verifier further consumes significant GPU memory and compute. While schedule ensembles like HEX are useful, they still require completing all trajectories.

**Key Challenge**: The parallel denoising dynamics of dLLMs—characterized by high entropy in early stages followed by the formation of a "logic skeleton" in mid-to-late stages—differ fundamentally from AR models. Allocating compute uniformly across all trajectories and time steps is inefficient, as it pays "full price" for unclear drafts in early high-entropy stages and wastes GPU resources on stable, nearly-identical trajectories in the late stages. Furthermore, AR-based PRMs are trained on well-formed prefixes and are poorly calibrated for dLLM states where intermediate tokens are still [MASK].

**Goal**: To decompose the solution into: (i) *non-uniform* allocation of trajectory counts across $T$ denoising steps; (ii) increasing *local* diversity without full re-sampling or discarding formed structures; and (iii) providing a reliable scoring signal for partially masked states without external PRMs.

**Key Insight**: The authors observe that dLLM entropy is highest in early-to-mid stages and collapses into a logic skeleton later. Best-of-$N$ postpones scoring until the very end, which is highly wasteful. It is more efficient to perform coarse screening in the middle stages and use the dLLM's own Yes/No prompts for scoring (reusing one forward pass at the cost of a single token).

**Core Idea**: Utilize "Hierarchical Trajectory Search (HTS) + local branching via partial remasking + Self-Verified Feedback (SVF)" to compress the dLLM TTS complexity from $O(NT)$ to a near-linear $O(N+KT)$, where $K \ll N$ is the final refinement width.

## Method

### Overall Architecture
Prism implements a three-stage denoising pipeline for dLLMs, using two hyperparameters $W=[w_{\min},w_{\max}]$ to define a "pruning window" with thresholds $T_p=\lceil w_{\max} T\rceil$ and $T_r=\lceil w_{\min} T\rceil$. Denoising proceeds from $t=T$ to $t=1$: (1) **Exploration** ($T_p < t \le T$) performs a brief warmup with width $N$ without pruning. (2) **Thinning** ($T_r < t \le T_p$) shrinks the active pool via geometric decay $W_t=\max(\lfloor N\cdot d^{-(T_p-t)}\rfloor, K)$. Every $i$ steps, a loop of "scoring $\rightarrow$ top-$S$ selection $\rightarrow$ local branching" generates $b_t=\lceil W_{t-1}/S\rceil$ children for each survivor. (3) **Refinement** ($1 \le t \le T_r$) converges the active width to $K$, followed by majority voting for the final answer. All scores are derived from logit ratios of the same dLLM on specifically constructed Yes/No verification prompts.

### Key Designs

1.  **Hierarchical Trajectory Search (HTS)**:
    - **Function**: Compresses best-of-$N$ complexity from $O(NT)$ to $O(N+KT)$ by concentrating compute on the critical mid-stage window where logic skeletons form.
    - **Mechanism**: The denoising schedule is split into three segments. Stage I (high noise) uses $N$ trajectories for stochastic exploration without pruning (diversity is prioritized as $\hat{\mathbf{z}}_0$ is unstable and SVF is unreliable). Stage II (pruning window) performs SVF scoring every $i$ steps to keep top-$S$ seeds, with each seed producing $b_t$ children via local branching; the pool size follows geometric decay $W_t$. Stage III stops pruning once width reaches $K$ for pure denoising, incorporating a $\tau$-confidence threshold and "`\boxed{}` early stopping" for speed. Total computation is $C_{\mathrm{HTS}}=N(T-T_p)+\sum_{t=T_r+1}^{T_p}|\mathcal{P}_t|+KT_r\approx O(N+KT)$.
    - **Design Motivation**: dLLM entropy decreases monotonically with $t$—divergent early, forming logic mid-stage, and highly convergent late-stage. Thus, "maintaining width early, aggressive pruning mid-stage, and refinement late-stage" matches dLLM dynamics. Geometric decay removes poor trajectories more aggressively than linear decay, ensuring NFE scales minimally with $N$ for a fixed $K$.

2.  **Local Branching via Partial Remasking**:
    - **Function**: Generates "differentiated but not completely restarted" offspring during the Thinning stage to prevent the top-$S$ candidates from collapsing prematurely into a single local optimum.
    - **Mechanism**: For a survivor state $\mathbf{z}_t$, the model first obtains $\hat{\mathbf{z}}_0=\mathcal{C}_\theta(\mathbf{z}_t,c,t)$. Token-level uncertainty (e.g., entropy) is calculated; high-confidence tokens are retained as the "logic skeleton," while a low-confidence subset $\mathcal{I}_t\subseteq\{1,\dots,L\}$ is remasked via $\mathbf{z}_t^{\exp}=\mathrm{Remask}(\mathbf{z}_t;\mathcal{I}_t)$ before continuing denoising. Each survivor can produce multiple children by sampling different $\mathcal{I}_t$.
    - **Design Motivation**: Restarting from $[m]^L$ discards all formed logical structures and wastes compute. Conversely, inheriting $\mathbf{z}_t$ without modification lacks diversity. Partial remasking restricts exploration to low-confidence positions, essentially "trying a different implementation for an established solution skeleton," balancing diversity and reuse.

3.  **Self-Verified Feedback (SVF)**:
    - **Function**: Replaces external PRMs/ORMs by providing trajectory ranking signals that remain valid for partially masked states, with minimal memory overhead.
    - **Mechanism**: For each trajectory $\mathbf{z}_t^{(i)}$, an argmax pass yields a full draft $\hat{\mathbf{z}}_0^{(i)}$, which is then embedded into a Yes/No verification prompt $\pi(c,\hat{\mathbf{z}}_0^{(i)})$. The max logits $s_{\text{Yes}}, s_{\text{No}}$ for "Yes" and "No" token sets are extracted from the dLLM to define $\Phi_{\mathrm{SVF}}(\mathbf{z}_t^{(i)};c)=\exp(s_{\text{Yes}})/(\exp(s_{\text{Yes}})+\exp(s_{\text{No}}))$. SVF is triggered sparsely every $i$ steps during the thinning stage.
    - **Design Motivation**: Traditional PRMs are trained on clean prefixes and are not well-calibrated for dLLM's partially masked intermediate states. Letting the dLLM score itself uses its pre-trained knowledge to judge if a full draft "looks correct." This scoring is insensitive to partial masking because it evaluates the full draft $\hat{\mathbf{z}}_0$. Reusing the dLLM saves PRM memory, and the cost of prefill + 1 token decoding is much smaller than a full denoising step.

### Loss & Training
Prism is an inference-time method that does not modify dLLM weights or train additional components. The dLLM uses the standard MDM ELBO objective:
$$\mathcal{L}(\theta)=\mathbb{E}[w(t)\sum_{i:z_{t,i}=m}(-\log\tilde p_\theta(z_{0,i}\mid\mathbf{z}_t,c,t))]$$

## Key Experimental Results

### Main Results
Comparison with best-of-$N$ ($N\in\{4,8,16\}$) across 4 benchmarks (GSM8K, MATH500, HumanEval, MBPP) and 3 dLLMs (LLaDA 8B, Dream 7B, LLaDA 2.0-mini). With fixed $N=16, S=K/2$, and target width $K\in\{2,4,8\}$. Representative data (LLaDA 8B Instruct):

| Setting | GSM8K Acc / NFE | MATH500 / NFE | HumanEval / NFE | MBPP / NFE |
| :--- | :--- | :--- | :--- | :--- |
| $N=1$ baseline | $67.58$ / $256$ | $26.40$ / $256$ | $54.88$ / $512$ | $21.80$ / $512$ |
| best-of-$16$ | $87.50$ / $4096$ | $38.00$ / $4096$ | $82.32$ / $8192$ | $35.20$ / $8192$ |
| Prism $K=2$ | $74.24$ / $283$ | $30.16$ / $334$ | $71.34$ / $549$ | $29.40$ / $561$ |
| Prism $K=4$ | $75.30$ / $509$ | $37.70$ / $622$ | $76.19$ / $1133$ | $32.40$ / $1196$ |
| Prism $K=8$ | $85.30$ / $1048$ | $42.80$ / $1304$ | $79.27$ / $2480$ | $38.20$ / $2576$ |

Prism $K=4$ on MATH500 achieves $37.70$ with approx. $622$ NFE, nearly matching the $38.00$ of best-of-$16$ using only $\sim 1/7$ the NFE. On MBPP, Prism $K=8$ reaches $38.20$, surpassing best-of-$16$ ($35.20$).

### Ablation Study

| Configuration | Key Metric | Observation |
| :--- | :--- | :--- |
| Full Prism (HTS+SVF+local branch) | See main table | All components active |
| w/o HTS (best-of-$N$) | NFE scales $N\times$ | Resources wasted on poor trajectories |
| w/o SVF (external scoring) | Memory surge, miscalibrated scores | Internal scoring is sufficient |
| w/o local branch (restart from $[m]^L$) | Early compute wasted | Logic skeleton is lost |

### Key Findings
- Concentrating compute in the "middle stage" is a key insight for dLLMs compared to AR models. AR models require broad early sampling as they condition on fixed prefixes; dLLM states are highly fuzzy early and highly convergent late, making "aggressive mid-stage pruning" effective.
- SVF frequency is much lower than NFE (SVF is just prefill + 1 token decoding). Even for Prism $K=8$ on GSM8K, the $33$ SVF calls are negligible compared to $1048$ denoising steps.
- HTS with geometric decay ($d>1$) discards trajectories more aggressively than linear decay, but diversity maintained by local branching compensates for collapse risk, allowing $K=2$ to significantly outperform the $N=1$ baseline.

## Highlights & Insights
- "Internal self-scoring" is particularly meaningful for dLLMs. Since dLLMs perform parallel token prediction at each step, reusing the same forward pass to evaluate "reasonableness" costs almost no extra memory, making it far more efficient than a separate 7B-class PRM.
- Local branching via "high-confidence skeleton + low-confidence token remasking" performs exploration within the same "mode" of the solution space, which is more robust than restarting. This exploits the unique advantage of dLLM's bidirectional context—AR models cannot simply replace tokens at arbitrary positions.
- The $O(N+KT)$ complexity formula reveals Prism's primary leverage: by turning the multiplicative relationship of best-of-$N$ into an additive one, the early stages can afford a very large $N$ for exploration with almost no cost increase.

## Limitations & Future Work
- Prism involves several hyperparameters ($N, K, S, d, i, w_{\min}, w_{\max}$). While the paper provides practical defaults, theoretical guidance is lacking, and tuning may be required for different tasks/models.
- SVF assumes the dLLM can distinguish "looks correct" from "looks wrong." If the model consistently makes systematic errors (consistent hallucination), the self-verification will fail alongside the generation.
- Evaluation is limited to math and code tasks with clear boxed answers or executable verification. The voting + Yes/No paradigm may be less applicable to open-ended long-form generation (e.g., creative writing).
- Compared to SMC-style methods like PG-DLM, Prism is heuristic. The refinement upper bound when using majority voting is not fully characterized theoretically.

## Related Work & Insights
- **vs. Best-of-$N$**: Runs all $N$ trajectories for $T$ steps, complexity $O(NT)$. Prism compresses the late stage to $K$ trajectories ($O(N+KT)$), yielding higher accuracy at equal NFE or $4\text{--}8\times$ NFE reduction at equal accuracy.
- **vs. HEX (Schedule Ensemble)**: HEX improves diversity via semi-AR blocks but still completes all trajectories. Prism's pruning-branching is complementary and theoretically stackable with HEX.
- **vs. PG-DLM (SMC for dLLM)**: PG-DLM treats TTS as reward-tilted inference using importance resampling. Prism is more engineering-focused, using SVF for heuristic ranking and local branching for variation, which is more efficient for verification-heavy reasoning tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of HTS, local branching, and SVF as an NFE-efficient TTS framework specifically for dLLMs is a first in the literature.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive testing across 4 benchmarks, 3 dLLMs, and 3 target widths with a clear NFE breakdown.
- Writing Quality: ⭐⭐⭐⭐ Complete pseudo-code, straightforward complexity analysis, and persuasive "Accuracy-NFE" curves.
- Value: ⭐⭐⭐⭐⭐ Open-source, plug-and-play, and requires no fine-tuning or external models, offering high practical value for dLLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models](lookahead_sample_reward_guidance_for_test-time_scaling_of_diffusion_models.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[NeurIPS 2025\] Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling](../../NeurIPS2025/llm_reasoning/rethinking_optimal_verification_granularity_for_compute-efficient_test-time_scal.md)
- [\[ICML 2026\] Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models](stabilizing_recurrent_dynamics_for_test-time_scalable_latent_reasoning_in_looped.md)
- [\[ICML 2026\] UniScale: Adaptive Unified Inference Scaling via Online Joint Optimization of Model Routing and Test-time Scaling](uniscale_adaptive_unified_inference_scaling_via_online_joint_optimization_of_mod.md)

</div>

<!-- RELATED:END -->
