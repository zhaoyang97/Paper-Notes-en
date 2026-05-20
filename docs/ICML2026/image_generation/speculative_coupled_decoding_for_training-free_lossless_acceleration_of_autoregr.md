---
title: >-
  [Paper Note] Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation
description: >-
  [ICML 2026][Image Generation][Speculative Decoding] This work identifies that the root cause of limited acceleration in Speculative Jacobi Decoding (SJD) for autoregressive visual generation is the near-zero probability…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Speculative Decoding"
  - "Jacobi Iteration"
  - "Coupling"
  - "Autoregressive Image Generation"
  - "Lossless Acceleration"
date: 2026-05-08
content_hash: 8f4720880a61b13a
---

# Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation

**Conference**: ICML 2026  
**arXiv**: [2510.24211](https://arxiv.org/abs/2510.24211)  
**Code**: https://github.com/junhyukso/SCD (available)  
**Area**: Image Generation / Autoregressive Visual Models / Inference Acceleration  
**Keywords**: Speculative Decoding, Jacobi Iteration, Coupling, Autoregressive Image Generation, Lossless Acceleration

## TL;DR
This work identifies that the root cause of limited acceleration in Speculative Jacobi Decoding (SJD) for autoregressive visual generation is the near-zero probability of collision between draft tokens across consecutive iterations due to independent sampling. By simply replacing independent sampling with Maximal/Gumbel Coupling (a one-line modification, no extra training), image generation can be accelerated up to $4.2\times$ and video generation up to $13.6\times$, while strictly preserving the output distribution identical to original AR decoding.

## Background & Motivation

**Background**: Autoregressive (AR) modeling has become the mainstream paradigm for unified image, video, 3D, and audio generation (e.g., Lumina-mGPT, Janus-Pro, Cosmos-1-AR). However, generating a high-resolution image requires decoding thousands of tokens sequentially, resulting in severe inference latency. Speculative Decoding (SD) is the de facto standard for accelerating LLM text generation: a lightweight draft model predicts $L$ tokens, then the target model verifies them in parallel, with modified rejection sampling ensuring the output strictly follows the target distribution.

**Limitations of Prior Work**: Standard SD performs poorly on visual AR tasks—first, it requires training a separate draft model, which is costly; second, visual token distributions are very flat, leading to low draft hit rates. The recently proposed Speculative Jacobi Decoding (SJD) uses the distribution from the previous Jacobi verification round as the next draft, bypassing the need for a draft model and requiring no training. However, image acceleration is only about $\sim 2\times$, far below the $4\times$+ achieved by text SD.

**Key Challenge**: The authors observe that SJD’s acceptance rate $\beta_i^{(t)} = 1 - \mathcal{D}_{TV}(p^{(t)}, p^{(t-1)})$ is directly determined by the similarity of adjacent prefix contexts. The problem is: even if two draft distributions are close in probability space (small TV distance), the actual probability of collision between sampled token sequences ($\Pr[X^{(t)} = X^{(t-1)}]$) remains extremely low due to independent sampling—flat visual token distributions yield high Rényi-2 entropy, so the collision probability is upper-bounded by entropy and is nearly zero, with about 94% of positions changing each round. This "probability space proximity ≠ sample space proximity" disconnect causes SJD’s convergence trajectory to be highly unstable, with large and non-convergent acceptance rate fluctuations.

**Goal**: Without introducing extra models, modifying the target model, or compromising lossless properties, maximize SJD’s effective acceptance rate, thereby pushing the acceleration ratio of visual AR to match or even surpass that of text SD.

**Key Insight**: Since the root cause is "independent sampling suppresses collision probability due to entropy," simply avoid independent sampling—use Coupling tools from information theory to let adjacent rounds of draft sampling share randomness, maximizing $\Pr[X^{(t)} = X^{(t-1)}]$ while preserving marginal distributions. Since SD’s lossless property depends only on the marginal distribution of the draft, not their correlation, Coupling provides a "free" stability gain.

**Core Idea**: Replace the independent sampling line in SJD’s drafting stage with Maximal Coupling (equivalent to reusing MRS from verification) or Gumbel Sharing Coupling—a one-line code change, zero extra training or memory—raising the collision probability from near zero to close to the $1 - \mathcal{D}_{TV}$ upper bound.

## Method

### Overall Architecture

Input: any pretrained AR visual generation model $p_\theta$, target sequence length $N$, window length $L$; Output: sampled sequence $X$ strictly matching the original token-by-token AR sampling distribution. The pipeline follows SJD’s three-stage process: (1) Drafting: in parallel, sample a set of draft tokens for each position in the window from the previous round’s verified distribution $p^{(t-1)}$; (2) Evaluate: the target model computes new distributions $p^{(t+1)}_j = p_\theta(\cdot \mid X^t_{0:j-1})$ for these prefixes in parallel; (3) Verify: use MRS to sequentially verify, stopping at the first reject, committing previous tokens and inheriting the new sample at the reject position as the next draft. All SCD modifications are in (1): instead of "independently sampling from $p^t_j$," sample a pair $(X^t_j, X^{t-1}_j)$ from the Coupling joint distribution of $p^t_j, p^{t-1}_j$, and use $X^t_j$ as the draft. The marginal property of Coupling ensures $X^t_j$ still follows $p^t_j$, so the lossless property in verification is strictly preserved.

### Key Designs

1. **Maximal Coupling ($\pi_{MC}$)**:

    - **Function**: Given $p^{(t)}$ and $p^{(t-1)}$, constructs a joint distribution $\pi(x, y)$ such that $\Pr[X = Y]$ achieves the theoretical upper bound $1 - \mathcal{D}_{TV}(p^{(t)}, p^{(t-1)})$.
    - **Mechanism**: The authors observe that the modified rejection sampling (MRS) used in verification is itself maximal coupling—given $X \sim Q$, MRS outputs $Y$ such that $Y \sim P$ and $\Pr[Y = X] = 1 - \mathcal{D}_{TV}(P, Q)$. Thus, in drafting, simply reuse MRS: starting from the previous $X^{t-1}_j$, run $\texttt{MRS}(p^t_j, p^{t-1}_j, X^{t-1}_j)$ to obtain the new draft $X^t_j$. Algorithmically, this replaces SJD’s independent sample in line 5 with a single MRS call.
    - **Design Motivation**: The Coupling cost $C(\pi) = \Pr[X=Y]$ equals the token-level collision probability, directly determining the next round’s acceptance rate. $\pi_{MC}$ "greedily" pushes collision to the theoretical limit at each pair of adjacent iterations, maximizing 1-step acceptance; since lossless only depends on the marginal, the output distribution is strictly preserved.

2. **Gumbel Sharing Coupling ($\pi_{GS}$)**:

    - **Function**: Another Coupling implementation, where two rounds of categorical sampling share the same Gumbel noise $G$, so $X = \arg\max_i (\log P_i + g_i)$ and $Y = \arg\max_i (\log Q_i + g_i)$ are likely to select the same token when distributions are similar.
    - **Mechanism**: Based on the Gumbel-Max trick, the single-step collision lower bound is $C(\pi_{GS}) \ge (1 - \mathcal{D}_{TV})/(1 + \mathcal{D}_{TV})$, slightly lower than $\pi_{MC}$’s $1 - \mathcal{D}_{TV}$; but this lower bound holds for any pair of distributions, thus providing guarantees for multi-step iterations $\mathrm{Hamm}(t, t+N)$. Gumbel noise can be generated online using a global token index hash, incurring zero memory overhead.
    - **Design Motivation**: $\pi_{MC}$ is 1-step greedy optimal but lacks nontrivial guarantees for multiple steps—locally optimal but possibly unstable long-term. $\pi_{GS}$ provides long-term stability and performs better in tasks where drafts are easy to predict (e.g., video AR with highly similar adjacent frames, low-resolution images): the benefit of early draft tokens remaining unchanged outweighs continuous minor adjustments.

3. **Zero-Overhead Integration**:

    - **Function**: Integrates Drafting’s MRS and Verify’s MRS into the same loop, requiring no extra forward passes.
    - **Mechanism**: The authors observe that SCD (Alg. 3) line 5 $\texttt{MRS}(p^t_j, p^{t-1}_j, X^t_j)$ and line 10 $\texttt{MRS}(p^{t+1}_j, p^t_j, X^t_j)$ are the same operation—the next round’s $p^{t+1}$ is the previous round’s $p^t$. Thus, vectorizing the verification loop (without breaking) and recording the index of the first reject allows drafting and verification to be completed simultaneously.
    - **Design Motivation**: Keeps the per-NFE extra latency of $\pi_{MC}$ below 5% (empirically, vectorized MRS on Janus-Pro 7B takes only 1.5 ms vs Transformer forward 26–36 ms). The entire method adds only a few lines of code, zero parameters, and zero training.

### Loss & Training

Completely training-free. SCD is a pure inference-time algorithmic replacement. All logit post-processing (top-k, CFG) is applied before defining $p^{(t)}$, strictly preserving the lossless proof.

## Key Experimental Results

### Main Results

| Model / Dataset | Config | NFE ↓ | Latency ↓ | Speedup | FID ↓ | CLIP ↑ |
|-----------------|--------|-------|-----------|---------|-------|--------|
| Lumina-mGPT / MS-COCO | Vanilla AR | 2390 | 102.0 s | $1.0\times$ | 30.79 | 31.31 |
| Lumina-mGPT / MS-COCO | SJD ($L=64$) | 1036 | 43.0 s | $2.31\times$ | 30.81 | 31.31 |
| Lumina-mGPT / MS-COCO | + $\pi_{MC}$ ($L=64$) | 568 | 24.4 s | $\mathbf{4.21\times}$ | 30.83 | 31.37 |
| Lumina-mGPT / MS-COCO | + $\pi_{GS}$ ($L=64$) | 568 | 24.2 s | $\mathbf{4.21\times}$ | 30.90 | 31.37 |
| Janus-Pro 7B / MS-COCO | SJD ($L=32$) | 318 | 10.6 s | $1.25\times$ | 37.76 | – |
| Janus-Pro 7B / MS-COCO | + $\pi_{GS}$ ($L=32$) | 154 | 5.39 s | $\mathbf{2.45\times}$ | 37.49 | – |
| Cosmos-1-AR / Real-Estate-10k | Vanilla AR | 7680 | 157 s | $1.0\times$ | FVD 156.9 | – |
| Cosmos-1-AR / Real-Estate-10k | + $\pi_{GS}$ ($L=128$) | 564 | 13.6 s | $\mathbf{13.6\times}$ | FVD 152.4 | – |

### Ablation Study

| Config | NFE | Notes |
|--------|-----|-------|
| SJD baseline | 1036 | Independent sampling, ~94% tokens change each round |
| + $\pi_{MC}$ | 568 | Maximal coupling, maximizes single-step collision |
| + $\pi_{GS}$ | 568 | Gumbel coupling, guarantees long-range collision |
| Compared to lossy GSD ($G=10$) | 701 | GSD is faster but FID drops from 30.79 to 33.21 |
| Coupling strength $\alpha$ sweep | – | NFE decreases monotonically as $\alpha \to 1$, verifying causality |
| Window $L$ sweep | – | SJD stalls for $L\!>\!16$, SCD benefits monotonically from larger $L$ |

### Key Findings

- The speedup as a function of window $L$ is most illustrative: standard SJD’s speedup plateaus at $\sim 2.3\times$ for $L=16, 32, 64$, while SCD reaches $4.2\times$ at $L=64$. This shows SJD’s bottleneck is not window size, but acceptance rate suppressed by independent sampling; increasing the window only brings more rejects.
- Video AR achieves much higher speedup than images ($13.6\times$ vs $4.2\times$): strong temporal redundancy between adjacent frames makes draft prediction especially easy, and $\pi_{GS}$’s long-range stability is more beneficial with long windows.
- $\pi_{MC}$ is indeed better for 1-step Hamming, but $\pi_{GS}$ is more stable for 2/3-step—echoing "greedy 1-step optimal ≠ long-term optimal."
- The coupling strength $\alpha$ experiment provides clean causal evidence: as $\alpha$ increases from 0 to 1, token-level Hamming distance and NFE both decrease monotonically, validating the chain "increased collision → increased context stability → reduced NFE."

## Highlights & Insights

- The insight that "probability proximity ≠ sample proximity" is elegant: the failure of standard SJD is precisely attributed to the Rényi-2 entropy upper bound $C_{SJD} \le e^{-1/2 \cdot (H_2(p) + H_2(q))}$, explaining why visual AR (flat distribution, high entropy) is harder to accelerate than text. This mapping of engineering phenomena to information-theoretic quantities is reusable for any "similar distributions but inconsistent samples" problem.
- Reusing verification’s MRS as the drafting sampler is ingenious—drafting and verification share the same mathematical structure, making $\pi_{MC}$ nearly cost-free to implement and vectorizable in the same loop.
- $\pi_{GS}$ exposes the "1-step optimal vs long-term stability" trade-off, which may recur in other iterative inference acceleration methods (e.g., consistency models, Jacobi for LLMs), and is worth transferring.
- The paper exemplifies the "minimal change + rigorous proof + strong results" paradigm: one line of code, zero training, lossless, $4-13\times$ acceleration.

## Limitations & Future Work

- The speedup ceiling is determined by the target model’s own $\mathcal{D}_{TV}(p^{(t)}, p^{(t-1)})$—if the context changes too drastically (e.g., high-res generation with strong CFG), even $\pi_{MC}$ can only push collision to the TV distance upper bound; the authors also observe that larger CFG $\lambda$ weakens acceleration.
- No task-adaptive selection rule is provided for $\pi_{MC}$ vs $\pi_{GS}$; currently, the empirical guideline is "use $\pi_{GS}$ for video/low-res, $\pi_{MC}$ for high-res images."
- Only AR visual generation is tested; it remains unexplored whether this can transfer to AR audio/robotics token sequences. Given their flat distributions, potential gains should be substantial.
- The Coupling idea itself can be extended beyond Self-SD—for example, in Medusa’s multi-head SD, whether collision probability between heads can also be maximized using Coupling tools.

## Related Work & Insights

- **vs SJD (Teng et al., 2024)**: SJD uses the previous round’s distribution as draft, removing the draft model, but independent sampling suppresses collision, limiting speedup to $\sim 2\times$. This work replaces it with Coupling sampling in one line, pushing speedup to $4.2\times$ while preserving lossless property.
- **vs GSD (So et al., 2025)**: GSD is lossy SD, accelerating via "group acceptance," achieving $\sim 3.4\times$ speedup but FID degrades from 30.8 to 33.2. SCD is faster and strictly lossless.
- **vs Medusa (Cai et al., 2024)**: Medusa achieves $4\times$ speedup on text via multi-head trained drafts, but requires training and head correlation is hard to control. SCD achieves "no training + comparable speedup" on vision.
- **vs Judge Decoding (Bachmann et al., 2025)**: Uses a judge model to relax acceptance for acceleration, sacrificing lossless property. SCD retains lossless property.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introducing information-theoretic Coupling into SJD is highly novel, and a one-line change yields upper-bound-level improvement.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers Lumina-mGPT/Janus-Pro/Lumina-mGPT-2 images and Cosmos-1-AR video, with $\alpha$-coupling, multi-step Hamming, and CFG sweeps.
- Writing Quality: ⭐⭐⭐⭐⭐ Flows seamlessly from motivation to proof to algorithm, with clear proposition and algorithm numbering, and highly convincing trajectory visualizations in Figures 3, 4, and 5.
- Value: ⭐⭐⭐⭐⭐ Directly pluggable into existing AR visual generation pipelines, up to $13.6\times$ video acceleration, with high industrial value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation](../../AAAI2026/image_generation/annealed_relaxation_of_speculative_decoding_for_faster_autor.md)
- [\[ICCV 2025\] Grouped Speculative Decoding for Autoregressive Image Generation](../../ICCV2025/image_generation/grouped_speculative_decoding_for_autoregressive_image_generation.md)
- [\[CVPR 2026\] SJD-PAC: Accelerating Speculative Jacobi Decoding via Proactive Drafting and Adaptive Continuation](../../CVPR2026/image_generation/sjd-pac_accelerating_speculative_jacobi_decoding_via_proactive_drafting_and_adap.md)
- [\[ICLR 2026\] Autoregressive Image Generation with Randomized Parallel Decoding](../../ICLR2026/image_generation/autoregressive_image_generation_with_randomized_parallel_decoding.md)
- [\[ICML 2026\] Visual Implicit Autoregressive Modeling](visual_implicit_autoregressive_modeling.md)

</div>

<!-- RELATED:END -->
