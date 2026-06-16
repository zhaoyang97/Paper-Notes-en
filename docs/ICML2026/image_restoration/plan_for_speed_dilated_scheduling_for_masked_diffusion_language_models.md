---
title: >-
  [Paper Note] Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models
description: >-
  [ICML 2026][Image Restoration][Masked Diffusion] This paper proposes the Dilated Unmasking Scheduler (DUS): it uses predefined "equidistant gaps" to determine the unmasking order independent of model confidence. This reduces the number of denoiser calls per block of $B$ tokens from $\mathcal O(B)$ to $\mathcal O(\log B)$, achieving a 5.8× wall-clock speedup on LLaDA
tags:
  - ICML 2026
  - Image Restoration
  - Masked Diffusion
  - Dilated Scheduling
  - Joint Entropy
  - Parallel Decoding
  - Inference-only
date: 2026-05-08
content_hash: 2f094a04157baed1
---
# Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models

**Conference**: ICML 2026  
**arXiv**: [2506.19037](https://arxiv.org/abs/2506.19037)  
**Code**: [github.com/omerlux/DUS](https://github.com/omerlux/DUS)  
**Area**: Discrete Diffusion Language Models / Inference Acceleration / Information-Theoretic Scheduling  
**Keywords**: Masked Diffusion, Dilated Scheduling, Joint Entropy, Parallel Decoding, Inference-only

## TL;DR
This paper proposes the Dilated Unmasking Scheduler (DUS): it uses predefined "equidistant gaps" to determine the unmasking order independent of model confidence. This reduces the number of denoiser calls per block of $B$ tokens from $\mathcal O(B)$ to $\mathcal O(\log B)$, achieving a 5.8× wall-clock speedup on LLaDA / Dream / DiffuCoder while outperforming confidence-based parallel planners in quality.

## Background & Motivation
**Background**: Masked Diffusion Language Models (MDLM, e.g., LLaDA-8B, Dream-7B, DiffuCoder-7B) allow for "any-order, parallel" decoding, theoretically breaking the $\mathcal O(G)$ latency of autoregressive models. However, high-quality generation typically still requires one denoiser call per token. Mainstream inference uses semi-autoregressive (semi-AR) blocking: the sequence is divided into blocks of length $B$, and multi-step denoising is performed within each block.

**Limitations of Prior Work**: Existing planners use per-token scores like confidence or entropy to select which masked positions to reveal. This results in "high-confidence tokens often being adjacent to each other," causing the process to fill tokens from left to right like autoregressive models, which renders parallelism ineffective. Alternatively, overly aggressive unmasking of many tokens at once ignores strong dependencies, leading to collapsed quality. Path Planning introduces external BERT scoring, trading quality for additional model calls.

**Key Challenge**: The true objective of parallel unmasking is to minimize $H(X_{\mathcal I_t}\mid\mathcal S_t)$ (joint conditional entropy), but planners can only optimize the factorizable $\sum_{i\in\mathcal I_t}H(X_i\mid\mathcal S_t)$ (sum of marginal entropies). The difference $\Delta(\mathcal I_t;\mathcal S_t):=\sum_{i\in\mathcal I_t}H(X_i\mid\mathcal S_t)-H(X_{\mathcal I_t}\mid\mathcal S_t)$ is always $\ge 0$, and this gap widens when the selected tokens are strongly correlated (usually spatially adjacent), causing parallel sampling to degrade into incoherent output.

**Goal**: (1) Design an unmasking schedule that requires no training, no external planner, and no denoiser scoring; (2) Reduce the number of denoiser calls from $\mathcal O(B)$ to $\mathcal O(\log B)$; (3) Maintain or improve task accuracy simultaneously.

**Key Insight**: From an information-theoretic perspective, to make the sum of marginal entropies approximate the joint entropy, one only needs to ensure that the tokens selected in parallel are as spatially dispersed as possible in the sequence. Under the fast-mixing VLMC assumption, the mutual information between tokens separated by distance $d$ decays exponentially as $C\rho^d$. Therefore, "explicitly increasing spatial intervals" directly tightens the entropy gap.

**Core Idea**: Use a predefined logarithmic dilated schedule—revealing positions at intervals of $B/a$ in the first round, $B/a^2$ in the second, and so on—to fill a block within $\lceil\log_a B\rceil$ rounds. The planner is entirely predetermined and does not depend on any model outputs.

## Method

### Overall Architecture
DUS addresses the problem of "how to unmask $B$ tokens in parallel without damaging quality" in MDLM semi-AR block decoding. It abstracts inference as an interaction between a "fixed denoiser $\mathcal D_\theta$ + optional planner $\mathcal P$": at each round $t$, the state $\mathcal S_t$ records unmasked tokens, the planner selects the set of indices $\mathcal I_t\subseteq\{b,\dots,b+B-1\}$ to be revealed, and the denoiser provides the distribution $p_\theta(X_i\mid\mathcal S_t)$ for each position. These are sampled in parallel to fill $\mathcal I_t$ before proceeding to the next round. DUS replaces dynamic confidence-based planners with a set of fully predefined schedules $\{\mathcal I_t\}_{t=1}^{R}$ independent of model output, filling the entire block within $R=\lceil\log_a B\rceil$ rounds and compressing calls per block from $\mathcal O(B)$ to $\mathcal O(\log B)$.

### Key Designs

**1. Entropy Gap Minimization Principle: Mathematizing the Impact of Parallel Unmasking**

The true objective parallel unmasking seeks to optimize is the joint conditional entropy $H(X_{\mathcal I_t}\mid\mathcal S_t)$, but planners can only practically optimize the factorizable sum of marginal entropies $\sum_{i\in\mathcal I_t}H(X_i\mid\mathcal S_t)$. The difference is the entropy gap $\Delta(\mathcal I_t;\mathcal S_t)$. This paper uses Lemma 3.3 to bound this relationship: for any grouping scheme, $H(X_{\mathcal B}\mid\mathcal S_1)\le\sum_t\mathcal L(\mathcal I_t;\mathcal S_1,X_{\mathcal I_{<t}})\le\sum_{i\in\mathcal B}H(X_i\mid\mathcal S_1)$. The left equality holds if and only if each group reveals only one token (degenerating to token-by-token AR), while the right corresponds to revealing all at once. Corollary 3.4 notes that $\Delta(\mathcal I_t;\mathcal S_t)\ge 0$ and planner design should explicitly minimize it. This step is significant because previous heuristics assumed per-token scores were equivalent to joint optimality, whereas this gap explains why confidence scheduling exhibits high scores but incoherent generation—selected high-score tokens are often adjacent and strongly correlated, enlarging the gap.

**2. Logarithmic Dilated Scheduling: Using Predefined Intervals to Force Separation**

Since the gap increases when selected tokens are spatially adjacent, DUS employs a predefined sequence of intervals to ensure positions revealed in each round are as dispersed as possible. Given block length $B$ and base $a>1$, the number of iterations is $R=\lceil\log_a B\rceil$. For round $t$, the step size is $s_t=\lfloor B/a^t\rfloor$, the selected set is $\mathcal I_t=\{k\in\{1,\dots,B\}\setminus\mathcal U_{t-1}\mid (k-1)\bmod s_t=0\}$, and the unmasked set is updated as $\mathcal U_t=\mathcal U_{t-1}\cup\mathcal I_t$. For example, with $B=8, a=2$, the block is filled in three rounds: $\mathcal I_1=\{1,5\}$, $\mathcal I_2=\{3,7\}$, and $\mathcal I_3=\{2,4,6,8\}$, requiring $\mathcal O(\log_a B)$ denoiser calls. This "early sparse, late dense" coarse-to-fine arrangement addresses both ends: in early rounds where tokens are revealed at long distances, mutual information is low under the fast-mixing assumption, meaning the gap is small and independent sampling approximates joint sampling; in late rounds where adjacent gaps are filled and correlations are strong, $\mathcal S_t$ already contains rich surrounding context, keeping per-token conditional entropy low and quality high.

**3. Exponential Convergence under Fast-Mixing: Why Dilated Scheduling Minimizes the Gap**

DUS is supported by a theoretical framework involving Variable Length Markov Chains (VLMC). Lemma 3.6 proves that stationary ergodic VLMCs with fast-mixing satisfy exponential decay of mutual information with distance: $I(X_i;X_{i+d},\dots,X_{i+(M+1)d})\le C\rho^d$. Combined with Lemma 3.5, this implies that if the distance between any two selected tokens is $\ge D_\varepsilon$, then $H(X_{i_1},\dots,X_{i_k}\mid\mathcal S_t)\ge\sum_j H(X_{i_j}\mid\mathcal S_t)-\varepsilon$, effectively compressing the entropy gap to $\le\varepsilon$. The authors treat fast-mixing VLMC as an analytical vehicle—while real text may not strictly satisfy it, DUS only requires the weaker condition that "tokens separated by distance $d$ are less correlated on average." Experiments on HumanEval and MATH500, which clearly violate fast-mixing, show stable acceleration, validating this relaxation.

### Loss & Training
DUS is entirely inference-only: no changes to the denoiser and no training of new modules. The schedule is statically computable. It also provides an optional skip heuristic—using the denoiser to calculate scores before each round and deferring uncertain positions that exceed a threshold to the next round (thresholds are scanned in Appendix B.2). Since the dilation concept is orthogonal to specific samplers, DUS can be applied directly alongside adaptive samplers like EB/CB as a "dilated-spacing post-filter."

## Key Experimental Results

### Main Results
Evaluation on 5 MDLM variants (LLaDA-B/I 8B, Dream-I 7B, DiffuCoder-B/I 7B), 7 benchmarks (GSM8K, MATH500, HumanEval, MBPP, BBH, MMLU-Pro, IFEval), with block size $B\in\{8,16,32,64\}$, comparing against self-confidence (Conf.):

| Model | Benchmark | Token-by-token (×1) | Conf. B=32 (×6.4) | DUS B=32 (×6.4) |
|------|-----------|---------------------|--------------------|------------------|
| LLaDA-I | GSM8K | 80.29 | 38.74 | 65.73 |
| LLaDA-I | MATH500 | 28.80 | 10.8 | 19.2 |
| LLaDA-I | HumanEval | 39.02 | 9.76 | 10.37 |
| LLaDA-I | MBPP | 39.4 | 14.0 | 23.2 |
| Dream-I | GSM8K | 77.10 | 27.60 | 44.66 |
| LLaDA-I | BBH | 53.89 | 44.26 | 50.93 |
| LLaDA-B | MMLU-Pro | 39.82 | 24.11 | 32.50 |

DUS outperforms Conf. across all tasks and block sizes. Notably, at high acceleration levels ($B=32/64$): on GSM8K, LLaDA-I with DUS retains 82% of single-step accuracy at 6.4× speed, while Conf. retains only 48%.

### Ablation Study
**Relationship between Block Size and Speedup Ratio** (DUS follows $R=\log_a B$, where speedup is strictly tied to NFE):

| B | DUS NFE/block | Speedup | LLaDA-I GSM8K | Conf. Equiv. |
|---|-----|--------|---------------|-----------|
| 8 | 3 | 2.7× | 73.24 | 69.22 |
| 16 | 4 | 4× | 70.66 | 61.41 |
| 32 | 5 | 6.4× | 65.73 | 38.74 |
| 64 | 6 | 10.7× | 57.09 | 18.73 |

Accuracy degradation for DUS is slow and predictable as $B$ increases; Conf. collapses at $B=64$ (−61 points), indicating that larger blocks exacerbate the issue of selecting strongly dependent tokens.

### Key Findings
- DUS is a "deterministic trade-off": selecting $B$ locks in the speed and quality loss upper bound, making it friendly for engineering deployment. The actual speedup and quality of Conf. are "expected values" that fluctuate with input.
- Coarse-to-fine is critical: the authors compared random/confidence planners with fixed $k$ against DUS-like incremental $k$ schedules. Fixed $k$ was consistently inferior to logarithmic growth, verifying that "early sparse, late dense" is necessary.
- DUS remains stable in code/math domains that violate fast-mixing. It maintains average token spacing 2-3 times greater than other planners, suggesting spacing is the primary factor and mutual information decay is a supporting theoretical explanation.
- Applying DUS as a post-filter for EB/CB samplers further improves their accuracy (Section 4.4), proving the dilation concept is orthogonal and additive.

## Highlights & Insights
- **Reforming Unmasking as an Information Theory Problem**: Moving beyond the "learn a planner" paradigm, this work uses entropy gaps to define the structure of a deterministic optimal solution—a beautiful example of "understanding before designing."
- **predefined > learned**: Independence from model confidence leads to robustness, interpretability, and predictability (users can precisely forecast NFE). This counter-intuitive insight is valuable for other scenarios.
- **Honest Handling of Theoretical-Empirical Mismatch**: The authors explicitly acknowledge that fast-mixing VLMC assumptions may fail in code or poetry but explain that DUS only requires "low average mutual information" at distance—this honest disclaimer makes the theory more applicable.
- **Compositional Nature of Post-filtering**: Orthogonal to EB/CB and KV-cache, allowing for cumulative improvements in industrial deployment.

## Limitations & Future Work
- VLMC assumptions, though relaxed, still matter: when text contains strong long-range dependencies (global variables in code, rhyming in poetry), the entropy gap becomes less controllable, increasing the accuracy drop for DUS.
- Accuracy still decreases with large $B$, even if slower than Conf.; there is no "free lunch" for acceleration.
- The schedule is manually crafted (base $a$ scanned between $\{2, 3, 4\}$) and has not been compared end-to-end with learnable schedules.
- Applicable only to masked diffusion; not directly applicable to score-matching discrete diffusion like SEDD.

## Related Work & Insights
- **vs Self-confidence / Self-entropy planner**: Conf. selects top-$k$ high-score tokens, but adjacent tokens are easily selected together, leading to a large entropy gap. DUS enforces spacing, keeping the gap near zero.
- **vs Path Planning P2 (peng2025)**: P2 uses BERT as an external planner, requiring extra model calls. DUS requires no scoring and is a purely static schedule.
- **vs Fast-dLLM / dKV-Cache**: KV-cache provides orthogonal memory optimization. DUS is a computation scheduling optimization; they are stackable.
- **vs Block Diffusion (arriola2025)**: semi-AR still uses $\mathcal O(B)$ calls within a block. DUS is an $\mathcal O(\log B)$ intra-block scheduler, serving as an optimal replacement for intra-block planners.
- **vs EB/CB sampler (ben-hamu2025/wu2025)**: These use entropy/confidence thresholds to dynamically select $k$ but do not control spatial distribution. DUS can act as a post-filter to improve them.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Information-theoretic gap analysis + dilated static scheduling; the framing is very fresh.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models × 7 benchmarks × 4 block sizes + post-filter validation + multiple ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear from motivation to theorems to pseudo-code to experimental organization.
- Value: ⭐⭐⭐⭐⭐ A new baseline for MDLM inference acceleration that can be directly adopted by any open-source diffusion LLM toolchain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Consistent Diffusion Language Models](consistent_diffusion_language_models.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)
- [\[ICML 2026\] Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](coevolutionary_continuous_discrete_diffusion_make_your_diffusion_language_model_.md)
- [\[ICML 2026\] Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models](early_decisions_matter_proximity_bias_and_initial_trajectory_shaping_in_non-auto.md)
- [\[ICML 2026\] Structured Diffusion Bridges: Inductive Bias for Denoising Diffusion Bridges](structured_diffusion_bridges_inductive_bias_for_denoising_diffusion_bridges.md)

</div>

<!-- RELATED:END -->
