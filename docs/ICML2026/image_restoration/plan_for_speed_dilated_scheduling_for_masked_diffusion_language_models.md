---
title: >-
  [Paper Note] Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models
description: >-
  [ICML 2026][Image Restoration][Masked Diffusion] This work proposes the Dilated Unmasking Scheduler (DUS): by using a "dilated, equidistant" predefined unmasking order that does not rely on model confidence…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "Masked Diffusion"
  - "Dilated Scheduling"
  - "Joint Entropy"
  - "Parallel Decoding"
  - "Inference-only"
date: 2026-05-08
content_hash: 5da246b945469897
---

# Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models

**Conference**: ICML 2026  
**arXiv**: [2506.19037](https://arxiv.org/abs/2506.19037)  
**Code**: [github.com/omerlux/DUS](https://github.com/omerlux/DUS)  
**Area**: Discrete Diffusion Language Models / Inference Acceleration / Information-Theoretic Scheduling  
**Keywords**: Masked Diffusion, Dilated Scheduling, Joint Entropy, Parallel Decoding, Inference-only

## TL;DR
This work proposes the Dilated Unmasking Scheduler (DUS): by using a "dilated, equidistant" predefined unmasking order that does not rely on model confidence, the number of denoiser calls per block of $B$ tokens is reduced from $\mathcal O(B)$ to $\mathcal O(\log B)$. On LLaDA / Dream / DiffuCoder, this achieves a 5.8× wall-clock speedup with quality surpassing confidence-based parallel planners.

## Background & Motivation
**Background**: Masked Diffusion Language Models (MDLMs, e.g., LLaDA-8B, Dream-7B, DiffuCoder-7B) allow "arbitrary-order, parallel" decoding, theoretically breaking the autoregressive $\mathcal O(G)$ latency. However, high-quality generation typically still requires one denoiser call per token. Mainstream inference uses semi-autoregressive (semi-AR) blocking: the sequence is divided into blocks of length $B$, with multi-step denoising within each block.

**Limitations of Prior Work**: Existing planners select which mask positions to reveal based on per-token scores such as confidence or entropy. As a result, "high-confidence tokens are often adjacent," leading to left-to-right, autoregressive-like filling, rendering parallelism ineffective. Alternatively, overly aggressive unmasking of large regions ignores strong token dependencies, causing quality collapse. Path Planning introduces external BERT scoring, trading extra model calls for quality.

**Key Challenge**: The true goal of parallel unmasking is to reduce $H(X_{\mathcal I_t}\mid\mathcal S_t)$ (joint conditional entropy), but planners can only optimize the decomposable $\sum_{i\in\mathcal I_t}H(X_i\mid\mathcal S_t)$ (sum of marginal entropies). The difference $\Delta(\mathcal I_t;\mathcal S_t):=\sum_{i\in\mathcal I_t}H(X_i\mid\mathcal S_t)-H(X_{\mathcal I_t}\mid\mathcal S_t)$ is always $\ge 0$, and grows when selected tokens are strongly correlated (typically spatially adjacent), causing parallel sampling to degenerate into incoherent outputs.

**Goal**: (1) Design an unmasking schedule that requires no training, no external planner, and no denoiser scoring; (2) Reduce denoiser calls from $\mathcal O(B)$ to $\mathcal O(\log B)$; (3) Maintain or improve task accuracy.

**Key Insight**: From an information-theoretic perspective, to make the sum of marginal entropies approximate the joint entropy, it suffices to ensure that tokens selected in parallel are as spatially dispersed as possible. Under the fast-mixing VLMC assumption, the mutual information between tokens $d$ apart decays exponentially as $C\rho^d$, so "explicitly increasing spatial distance" directly tightens the entropy gap.

**Core Idea**: Use a predefined logarithmic dilated schedule—first round unmasks positions at interval $B/a$, second round at $B/a^2$, etc.—to fill a block in $\lceil\log_a B\rceil$ rounds. The planner is entirely predetermined and does not depend on any model output.

## Method

### Overall Architecture
MDLM inference is abstracted as an interaction between a fixed denoiser $\mathcal D_\theta$ and an optional planner $\mathcal P$. At each round $t$, the state $\mathcal S_t$ contains already unmasked tokens; the planner selects an index set $\mathcal I_t\subseteq\{b,\dots,b+B-1\}$; the denoiser provides per-position distributions $p_\theta(X_i\mid\mathcal S_t)$, from which $\mathcal I_t$ is revealed in parallel; the next state is $\mathcal S_{t+1}$. DUS replaces confidence-based planners, providing a fixed schedule $\{\mathcal I_t\}_{t=1}^{R}$, completing the block in $R=\lceil\log_a B\rceil$ rounds. A skip heuristic (deferring uncertain tokens) can be optionally added.

### Key Designs

1. **Entropy Gap Minimization Principle**:

    - **Function**: Formalizes whether parallel unmasking harms quality, providing a quantitative target for scheduling.
    - **Mechanism**: Lemma 3.3 proves that for any grouping, $H(X_{\mathcal B}\mid\mathcal S_1)\le\sum_t\mathcal L(\mathcal I_t;\mathcal S_1,X_{\mathcal I_{<t}})\le\sum_{i\in\mathcal B}H(X_i\mid\mathcal S_1)$; equality on the left holds only for token-by-token AR, the right corresponds to full unmasking in one step. Corollary 3.4 introduces the entropy gap $\Delta(\mathcal I_t;\mathcal S_t)\ge 0$ and points out that planner design should explicitly minimize it.
    - **Design Motivation**: Previous work did not formalize "parallel scheduling" via information-theoretic gap; all heuristics assumed per-token scores and joint optimality were equivalent. This step reveals why confidence-based scheduling, despite high scores, produces incoherent generations.

2. **Logarithmic Dual Dilation Scheduling**:

    - **Function**: Uses a predefined interval sequence to ensure tokens unmasked in each round are as far apart as possible.
    - **Mechanism**: Given block length $B$ and base $a>1$, compute rounds $R=\lceil\log_a B\rceil$, step size at round $t$ is $s_t=\lfloor B/a^t\rfloor$; at round $t$, select $\mathcal I_t=\{k\in\{1,\dots,B\}\setminus\mathcal U_{t-1}\mid (k-1)\bmod s_t=0\}$, with $\mathcal U_t=\mathcal U_{t-1}\cup\mathcal I_t$. For example, $B=8,a=2$: $\mathcal I_1=\{1,5\},\mathcal I_2=\{3,7\},\mathcal I_3=\{2,4,6,8\}$, filling 8 tokens in 3 rounds. Complexity is $\mathcal O(\log_a B)$ denoiser calls.
    - **Design Motivation**: Early rounds unmask sparse, distant tokens—under the fast-mixing assumption, mutual information is low and entropy gap is small, so independent sampling approximates joint sampling. In later rounds, dense unmasking of remaining positions occurs when $\mathcal S_t$ already contains rich context, so per-token conditional entropy is low, maintaining quality.

3. **Theoretical Guarantee: Exponential Gap Decay under Fast-Mixing**:

    - **Function**: Under the VLMC model assumption, strictly proves that the entropy gap of dilation scheduling can be made arbitrarily small.
    - **Mechanism**: Lemma 3.6 shows that for fast-mixing stationary ergodic VLMC, $I(X_i;X_{i+d},\dots,X_{i+(M+1)d})\le C\rho^d$; Lemma 3.5 jointly yields that as long as selected tokens are pairwise at least $D_\varepsilon$ apart, $H(X_{i_1},\dots,X_{i_k}\mid\mathcal S_t)\ge\sum_j H(X_{i_j}\mid\mathcal S_t)-\varepsilon$, i.e., the difference between the sum of marginal entropies and joint entropy for parallel sampling is $\le\varepsilon$.
    - **Design Motivation**: Fast-mixing VLMC is only an analytical vehicle; the authors emphasize that real text may not satisfy this, but DUS only requires the weaker condition that "on average, tokens $d$ apart are not strongly correlated," and experiments confirm stable acceleration even in code/math domains (HumanEval/MATH500) that clearly violate fast mixing.

### Loss & Training
DUS is entirely inference-only, does not modify the denoiser, and does not train new modules. An optional skip heuristic: before each round, use the denoiser to score positions, and defer those exceeding a threshold to the next round (Appendix B.2). DUS can be stacked as a "dilated-spacing post-filter" on adaptive samplers like EB/CB.

## Key Experimental Results

### Main Results
Five MDLM variants (LLaDA-B/I 8B, Dream-I 7B, DiffuCoder-B/I 7B), seven benchmarks (GSM8K, MATH500, HumanEval, MBPP, BBH, MMLU-Pro, IFEval), block sizes $B\in\{8,16,32,64\}$, compared to self-confidence (Conf.):

| Model | Benchmark | Token-by-token (×1) | Conf. B=32 (×6.4) | DUS B=32 (×6.4) |
|-------|-----------|---------------------|--------------------|------------------|
| LLaDA-I | GSM8K | 80.29 | 38.74 | 65.73 |
| LLaDA-I | MATH500 | 28.80 | 10.8 | 19.2 |
| LLaDA-I | HumanEval | 39.02 | 9.76 | 10.37 |
| LLaDA-I | MBPP | 39.4 | 14.0 | 23.2 |
| Dream-I | GSM8K | 77.10 | 27.60 | 44.66 |
| LLaDA-I | BBH | 53.89 | 44.26 | 50.93 |
| LLaDA-B | MMLU-Pro | 39.82 | 24.11 | 32.50 |

DUS outperforms Conf. on all tasks and block sizes, especially at high acceleration ($B=32/64$): on GSM8K, LLaDA-I with DUS retains 82% of single-step accuracy at 6.4× speed, while Conf. retains only 48%.

### Ablation Study
**Predicted relationship between block size and speedup** (DUS uses $R=\log_a B$, so speedup is strictly tied to NFE):

| B | DUS NFE/block | Speedup | LLaDA-I GSM8K | Conf. counterpart |
|---|---------------|---------|---------------|------------------|
| 8 | 3 | 2.7× | 73.24 | 69.22 |
| 16 | 4 | 4× | 70.66 | 61.41 |
| 32 | 5 | 6.4× | 65.73 | 38.74 |
| 64 | 6 | 10.7× | 57.09 | 18.73 |

DUS accuracy drops slowly and predictably with $B$; Conf. collapses at $B=64$ (−61 points), indicating that larger blocks are increasingly harmed by strong token dependencies.

### Key Findings
- DUS is a "deterministic trade-off": selecting $B$ fixes both speed and the upper bound of quality loss, making it engineering-friendly; Conf.'s actual speedup and quality are input-dependent expectations.
- Coarse-to-fine is key: the authors compare fixed-$k$ random/confidence planners and DUS-like incremental $k$ schedules; fixed $k$ never matches logarithmic growth, confirming that "early sparse, late dense" is necessary.
- DUS remains stable even in code/math domains that violate fast-mixing, with average token spacing 2–3× that of other planners, indicating that spacing is the primary factor and mutual information decay is only a theoretical aid.
- Using DUS as a post-filter on EB/CB further improves their accuracy (Section 4.4), demonstrating that the dilation idea is orthogonal and additive.

## Highlights & Insights
- **Reframing unmasking order from an ML problem to an information-theoretic one**: Moves beyond "learning a planner" by using the information-theoretic gap to provide a deterministically optimal structure—a strong example of "understand first, then design."
- **Predefined > learned**: Completely independent of model confidence, yet robust, interpretable, and auditable (users can precisely predict NFE); this counterintuitive result is worth applying in other contexts.
- **Theoretical-empirical mismatch is honestly addressed**: The authors explicitly state that the VLMC fast-mixing assumption does not hold in code/poetry, but is empirically effective, and explain that spacing only requires "on average, low mutual information"—this honest disclaimer broadens theoretical applicability.
- **Post-filter composability**: Orthogonal to EB/CB/KV-cache, allowing for multiplicative gains in industrial deployment.

## Limitations & Future Work
- The VLMC assumption is a relaxed version but still present: when text contains strong long-range dependencies (e.g., global variable constraints in code, rhyming in poetry), the entropy gap becomes uncontrollable and DUS accuracy drops more.
- For large $B$, DUS accuracy still decreases, albeit more slowly than Conf.; there is no "free lunch" for acceleration.
- The schedule is entirely hand-crafted (base $a$ is swept over $\{2,3,4\}$), and has not yet been compared end-to-end with learnable schedules.
- Only applicable to masked diffusion; not directly applicable to score-matching discrete diffusion (e.g., SEDD).

## Related Work & Insights
- **vs Self-confidence / Self-entropy planner**: Conf. selects top-$k$ high-score tokens, which are often adjacent, leading to large entropy gaps; DUS enforces spacing, making the gap nearly zero.
- **vs Path Planning P2 (peng2025)**: P2 uses BERT as an external planner, requiring extra model calls; DUS requires no scoring, being purely static scheduling.
- **vs Fast-dLLM / dKV-Cache**: KV-cache is an orthogonal memory optimization, DUS is a compute scheduling optimization; they can be combined.
- **vs Block Diffusion (arriola2025)**: Semi-AR still requires $\mathcal O(B)$ calls within a block; DUS achieves $\mathcal O(\log B)$ scheduling within a block, serving as an optimal replacement for block-level planners.
- **vs EB/CB sampler (ben-hamu2025/wu2025)**: These use entropy/confidence thresholds to dynamically select $k$ but do not control spatial distribution; DUS as a post-filter can further improve them.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Information-theoretic gap analysis + dilated static scheduling; the overall framing is highly original
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models × 7 benchmarks × 4 block sizes + post-filter validation + multiple ablations
- Writing Quality: ⭐⭐⭐⭐⭐ Clear from motivation to theorems to pseudocode to experimental organization
- Value: ⭐⭐⭐⭐⭐ A new baseline for MDLM inference acceleration, directly adoptable by any open-source diffusion LLM toolchain

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Consistent Diffusion Language Models](consistent_diffusion_language_models.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)
- [\[ICML 2026\] Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models](early_decisions_matter_proximity_bias_and_initial_trajectory_shaping_in_non-auto.md)
- [\[ICML 2026\] Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](coevolutionary_continuous_discrete_diffusion_make_your_diffusion_language_model_.md)
- [\[ICML 2026\] DAPD: Dependency-Aware Parallel Decoding via Attention for Diffusion LLMs](dapd_dependency-aware_parallel_decoding_via_attention_for_diffusion_llms.md)

</div>

<!-- RELATED:END -->
