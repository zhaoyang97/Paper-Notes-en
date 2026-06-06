---
title: >-
  [Paper Note] Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models
description: >-
  [ICML 2026][Image Restoration][Masked Diffusion] This paper proposes the Dilated Unmasking Scheduler (DUS): it uses "equidistant gaps" to predefine an unmasking sequence independent of model confidence…
tags:
  - "ICML 2026"
  - "Image Restoration"
  - "Masked Diffusion"
  - "Dilated Scheduling"
  - "Joint Entropy"
  - "Parallel Decoding"
  - "Inference-only"
date: 2026-05-08
content_hash: 5cf84c4639ba4876
---

# Plan for Speed: Dilated Scheduling for Masked Diffusion Language Models

**Conference**: ICML 2026  
**arXiv**: [2506.19037](https://arxiv.org/abs/2506.19037)  
**Code**: [github.com/omerlux/DUS](https://github.com/omerlux/DUS)  
**Area**: Discrete Diffusion Language Models / Inference Acceleration / Information-theoretic Scheduling  
**Keywords**: Masked Diffusion, Dilated Scheduling, Joint Entropy, Parallel Decoding, Inference-only

## TL;DR
This paper proposes the Dilated Unmasking Scheduler (DUS): it uses "equidistant gaps" to predefine an unmasking sequence independent of model confidence, reducing the number of denoiser calls for each block of $B$ tokens from $\mathcal O(B)$ to $\mathcal O(\log B)$. It achieves a 5.8× wall-clock speedup on LLaDA, Dream, and DiffuCoder, while maintaining higher quality than confidence-based parallel planners.

## Background & Motivation
**Background**: Masked Diffusion Language Models (MDLM, such as LLaDA-8B, Dream-7B, DiffuCoder-7B) allow for "arbitrary-order, parallel" decoding, which theoretically can break the $\mathcal O(G)$ latency of autoregressive models. However, high-quality generation typically still requires one denoiser call per token. Mainstream inference utilizes semi-autoregressive (semi-AR) chunking: the sequence is divided into blocks of length $B$, and multi-step denoising is performed within each block.

**Limitations of Prior Work**: Existing planners use per-token scores like confidence or entropy to select which masked positions to reveal. This results in "high-confidence tokens usually being adjacent to each other," causing the model to fill tokens from left to right like an AR model, which renders parallelism ineffective. Conversely, being too aggressive and revealing large groups of tokens at once ignores strong dependencies between tokens, leading to a collapse in quality. Path Planning introduces external BERT scoring, trading quality for additional model calls.

**Key Challenge**: The true objective of parallel unmasking is to reduce the joint conditional entropy $H(X_{\mathcal I_t}\mid\mathcal S_t)$, but planners can only optimize the decomposable sum of marginal entropies $\sum_{i\in\mathcal I_t}H(X_i\mid\mathcal S_t)$. The difference $\Delta(\mathcal I_t;\mathcal S_t):=\sum_{i\in\mathcal I_t}H(X_i\mid\mathcal S_t)-H(X_{\mathcal I_t}\mid\mathcal S_t)$ is always $\ge 0$. This gap widens when selected tokens are strongly correlated (usually spatially adjacent), causing parallel sampling to degrade into incoherent output.

**Goal**: (1) Design an unmasking schedule that requires no training, no external planners, and no denoiser scoring; (2) reduce the number of denoiser calls from $\mathcal O(B)$ to $\mathcal O(\log B)$; (3) maintain or improve task accuracy simultaneously.

**Key Insight**: From an information-theoretic perspective, making the sum of marginal entropies approximate the joint entropy requires ensuring that tokens selected in parallel are as spatially dispersed as possible. Under the fast-mixing VLMC assumption, the mutual information between tokens at distance $d$ decays exponentially as $C\rho^d$. Therefore, "explicitly increasing spatial intervals" can directly tighten the entropy gap.

**Core Idea**: Use a predefined logarithmic dilated schedule—revealing positions at interval $B/a$ in the first round, $B/a^2$ in the second, and so on—to fill a block within $\lceil\log_a B\rceil$ rounds. The planner is entirely predetermined and does not depend on any model outputs.

## Method

### Overall Architecture
MDLM inference is abstracted as an interaction between a "fixed denoiser $\mathcal D_\theta$ + optional planner $\mathcal P$." In each round $t$, the state $\mathcal S_t$ contains already unmasked tokens. The planner selects an index set $\mathcal I_t\subseteq\{b,\dots,b+B-1\}$. The denoiser provides a per-position distribution $p_\theta(X_i\mid\mathcal S_t)$, from which tokens in $\mathcal I_t$ are sampled in parallel. This leads to the next state $\mathcal S_{t+1}$. DUS replaces confidence-based planners with a fixed $\{\mathcal I_t\}_{t=1}^{R}$ schedule, completing the entire block in $R=\lceil\log_a B\rceil$ rounds. A skip heuristic (delaying uncertain tokens) can also be applied.

### Key Designs

1.  **Entropy Gap Minimization Principle**:
    - **Function**: Mathematizes whether parallel unmasking harms quality, providing a quantitative target for scheduling.
    - **Mechanism**: Lemma 3.3 proves that for any grouping scheme $H(X_{\mathcal B}\mid\mathcal S_1)\le\sum_t\mathcal L(\mathcal I_t;\mathcal S_1,X_{\mathcal I_{<t}})\le\sum_{i\in\mathcal B}H(X_i\mid\mathcal S_1)$. The left equality is achieved only by token-by-token AR, while the right corresponds to revealing all tokens at once. Corollary 3.4 identifies the entropy gap $\Delta(\mathcal I_t;\mathcal S_t)\ge 0$ and points out that planner design should explicitly minimize it.
    - **Design Motivation**: Previously, no work formalized "parallel scheduling" using information-theoretic gaps; heuristics assumed that per-token scores were equivalent to joint optimality. This step reveals why confidence-based scheduling produces incoherent generation despite high scores.

2.  **Logarithmic Dual Dilation Scheduling**:
    - **Function**: Uses a predefined interval sequence to ensure that tokens revealed in each round are as far apart as possible.
    - **Mechanism**: Given block length $B$ and base $a>1$, the number of iterations is $R=\lceil\log_a B\rceil$. For round $t$, the step size is $s_t=\lfloor B/a^t\rfloor$. Round $t$ selects $\mathcal I_t=\{k\in\{1,\dots,B\}\setminus\mathcal U_{t-1}\mid (k-1)\bmod s_t=0\}$, where $\mathcal U_t=\mathcal U_{t-1}\cup\mathcal I_t$. For example, if $B=8, a=2$: $\mathcal I_1=\{1,5\}, \mathcal I_2=\{3,7\}, \mathcal I_3=\{2,4,6,8\}$. This fills 8 tokens in 3 rounds, requiring $\mathcal O(\log_a B)$ denoiser calls.
    - **Design Motivation**: Early sparse rounds reveal tokens at a distance, yielding small mutual information and entropy gaps under the fast-mixing assumption; thus, independent sampling approximates joint sampling. By the later dense rounds, although correlations between adjacent tokens are strong, $\mathcal S_t$ already contains rich surrounding context, resulting in low per-token conditional entropy and maintained quality.

3.  **Theoretical Guarantee: Exponential Gap Convergence under Fast-mixing**:
    - **Function**: Rigorously proves that the entropy gap of dilation scheduling can be arbitrarily small under VLMC model assumptions.
    - **Mechanism**: Lemma 3.6 proves that fast-mixing stable ergodic VLMC satisfies $I(X_i;X_{i+d},\dots,X_{i+(M+1)d})\le C\rho^d$. Lemma 3.5 combined with this shows that if selected tokens have pairwise distance $\ge D_\varepsilon$, then $H(X_{i_1},\dots,X_{i_k}\mid\mathcal S_t)\ge\sum_j H(X_{i_j}\mid\mathcal S_t)-\varepsilon$.
    - **Design Motivation**: While fast-mixing VLMC is used for analysis, the authors note that real text may not strictly satisfy it. DUS only requires the weaker condition that "average tokens at distance $d$ are not strongly correlated," which is empirically validated in code and math domains.

### Loss & Training
DUS is entirely inference-only, requiring no changes to the denoiser or training of new modules. An optional skip heuristic calculates scores using the denoiser before each round; uncertain positions exceeding a threshold are delayed to the next round. It can be superimposed on adaptive samplers like EB/CB as a "dilated-spacing post-filter."

## Key Experimental Results

### Main Results
Evaluation was conducted on 5 MDLM variants (LLaDA-B/I 8B, Dream-I 7B, DiffuCoder-B/I 7B) across 7 benchmarks (GSM8K, MATH500, HumanEval, MBPP, BBH, MMLU-Pro, IFEval), with block size $B\in\{8,16,32,64\}$, comparing against self-confidence (Conf.):

| Model | Benchmark | Token-by-token (×1) | Conf. B=32 (×6.4) | DUS B=32 (×6.4) |
|------|-----------|---------------------|--------------------|------------------|
| LLaDA-I | GSM8K | 80.29 | 38.74 | 65.73 |
| LLaDA-I | MATH500 | 28.80 | 10.8 | 19.2 |
| LLaDA-I | HumanEval | 39.02 | 9.76 | 10.37 |
| LLaDA-I | MBPP | 39.4 | 14.0 | 23.2 |
| Dream-I | GSM8K | 77.10 | 27.60 | 44.66 |
| LLaDA-I | BBH | 53.89 | 44.26 | 50.93 |
| LLaDA-B | MMLU-Pro | 39.82 | 24.11 | 32.50 |

DUS outperformed Conf. across all tasks and block sizes, especially at high acceleration ($B=32/64$). On GSM8K, LLaDA-I using DUS retained 82% of its single-step accuracy at 6.4× speed, whereas Conf. retained only 48%.

### Ablation Study
**Relationship between Block Size and Speedup Ratio** (DUS follows $R=\log_a B$, strictly tying speedup to NFE):

| B | DUS NFE/block | Speedup | LLaDA-I GSM8K | Conf. Equiv. |
|---|-----|--------|---------------|-----------|
| 8 | 3 | 2.7× | 73.24 | 69.22 |
| 16 | 4 | 4× | 70.66 | 61.41 |
| 32 | 5 | 6.4× | 65.73 | 38.74 |
| 64 | 6 | 10.7× | 57.09 | 18.73 |

The accuracy degradation of DUS is slow and predictable as $B$ increases; Conf. collapses at $B=64$, illustrating that it is severely harmed by strong token dependencies in large blocks.

### Key Findings
- DUS represents a "deterministic trade-off": selecting $B$ fixes the speed and quality loss upper bound, making it engineering-friendly.
- Coarse-to-fine is critical: comparing fixed-$k$ planners to DUS-like incremental $k$ scheduling, fixed $k$ is always inferior, validating the necessity of "early sparse, late dense" scheduling.
- DUS remains stable in code and math domains where fast-mixing is often violated.
- Applying DUS as a post-filter to EB/CB samplers further improves their accuracy, proving that the dilation concept is orthogonal and additive.

## Highlights & Insights
- **Reframing unmasking order as an information-theoretic problem**: Moving away from the "learning a planner" paradigm to find a deterministically optimal structure using information gaps.
- **Predefined > learned**: Independence from model confidence provides robustness and interpretability.
- **Honest treatment of theoretical-empirical mismatch**: The authors acknowledge that VLMC assumptions might not hold for all text types, but spacing only requires "low average mutual information" to work.
- **Post-filter additivity**: DUS is orthogonal to tools like EB/CB and KV-cache.

## Limitations & Future Work
- The entropy gap becomes difficult to control for text with strong long-range dependencies (e.g., global variable constraints in code).
- Quality still degrades at very large $B$; there is no "free lunch" for acceleration.
- The schedule is currently hand-crafted (scanning base $a \in \{2,3,4\}$) and has not been compared end-to-end with learnable schedules.
- It applies only to masked diffusion and is not directly applicable to score-matching discrete diffusion like SEDD.

## Related Work & Insights
- **vs Self-confidence / Self-entropy planner**: Conf. selects top-$k$ tokens, often choosing adjacent ones and creating large entropy gaps; DUS minimizes this via forced spacing.
- **vs Path Planning P2**: P2 uses BERT as an external planner, requiring extra model calls; DUS is purely static.
- **vs Fast-dLLM / dKV-Cache**: KV-cache is a memory optimization; DUS is a computation scheduling optimization.
- **vs Block Diffusion**: While semi-AR blocks usually use $\mathcal O(B)$ calls, DUS serves as an $\mathcal O(\log B)$ alternative for intra-block scheduling.
- **vs EB/CB sampler**: These use thresholds to select $k$ but do not control spatial distribution; DUS can enhance them as a post-filter.

## Rating
- Novelness: ⭐⭐⭐⭐⭐ Information-theoretic gap analysis + dilated static scheduling is a very fresh framing.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 models × 7 benchmarks × 4 block sizes + post-filter validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Extremely clear from motivation to theorems to results.
- Value: ⭐⭐⭐⭐⭐ A new baseline for MDLM inference acceleration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Consistent Diffusion Language Models](consistent_diffusion_language_models.md)
- [\[ICLR 2026\] Activation Steering for Masked Diffusion Language Models](../../ICLR2026/image_restoration/activation_steering_for_masked_diffusion_language_models.md)
- [\[ICML 2026\] Early Decisions Matter: Proximity Bias and Initial Trajectory Shaping in Non-Autoregressive Diffusion Language Models](early_decisions_matter_proximity_bias_and_initial_trajectory_shaping_in_non-auto.md)
- [\[ICML 2026\] Coevolutionary Continuous Discrete Diffusion: Make Your Diffusion Language Model a Latent Reasoner](coevolutionary_continuous_discrete_diffusion_make_your_diffusion_language_model_.md)
- [\[ICML 2026\] Image Restoration via Diffusion Models with Dynamic Resolution](image_restoration_via_diffusion_models_with_dynamic_resolution.md)

</div>

<!-- RELATED:END -->
