---
title: >-
  [Paper Note] Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation
description: >-
  [ICML 2026][Image Generation][Speculative Decoding] This paper identifies that the root cause of the limited acceleration of Speculative Jacobi Decoding (SJD) in autoregressive visual generation is the near-zero collisio…
tags:
  - "ICML 2026"
  - "Image Generation"
  - "Speculative Decoding"
  - "Jacobi Iteration"
  - "Coupling"
  - "Autoregressive Image Generation"
  - "Lossless Acceleration"
date: 2026-05-08
content_hash: 56c94e8b4ddbeec2
---

# Speculative Coupled Decoding for Training-Free Lossless Acceleration of Autoregressive Visual Generation

**Conference**: ICML 2026  
**arXiv**: [2510.24211](https://arxiv.org/abs/2510.24211)  
**Code**: https://github.com/junhyukso/SCD (Available)  
**Area**: Image Generation / Autoregressive Visual Models / Inference Acceleration  
**Keywords**: Speculative Decoding, Jacobi Iteration, Coupling, Autoregressive Image Generation, Lossless Acceleration

## TL;DR
This paper identifies that the root cause of the limited acceleration of Speculative Jacobi Decoding (SJD) in autoregressive visual generation is the near-zero collision probability between draft tokens due to independent sampling across successive iterations. By simply replacing independent sampling with Maximal/Gumbel Coupling (a one-line modification with zero additional training), image generation is accelerated by up to $4.2\times$ and video generation by $13.6\times$, while strictly maintaining an output distribution identical to original AR decoding.

## Background & Motivation

**Background**: Autoregressive (AR) modeling has become the mainstream paradigm for unified image, video, 3D, and audio generation (e.g., Lumina-mGPT, Janus-Pro, Cosmos-1-AR). However, generating a high-resolution image requires serial decoding of thousands of tokens, leading to severe inference latency. Speculative Decoding (SD) is the de facto standard for accelerating text LLMs, where a cheap draft model predicts $L$ tokens followed by parallel verification by a target model using modified rejection sampling to ensure lossless output.

**Limitations of Prior Work**: Standard SD performs poorly on visual AR for two reasons: 1) it requires training a separate draft model, which is expensive; 2) visual token distributions are flat, leading to low draft hit rates. The recently proposed Speculative Jacobi Decoding (SJD) uses the distribution from the previous Jacobi iteration as the next draft, bypassing training. However, it only achieves $\sim 2\times$ acceleration for images, significantly lower than the $4\times$+ seen in text SD.

**Key Challenge**: The authors find that the SJD acceptance rate $\beta_i^{(t)} = 1 - \mathcal{D}_{TV}(p^{(t)}, p^{(t-1)})$ is determined by the similarity of prefix contexts. The problem is that even if two draft distributions are close in probability space (small TV distance), the actual collision probability ($\Pr[X^{(t)} = X^{(t-1)}]$) remains extremely low due to independent sampling. Flat visual token distributions cause Rényi-2 entropy to be very large, suppressing collision probability to near zero—approximately 94% of positions change every iteration. This "probability space proximity $\neq$ realization space proximity" gap makes the SJD convergence trajectory unstable, leading to fluctuating and non-convergent acceptance rates.

**Goal**: Push the effective acceptance rate of SJD to its limit and achieve acceleration ratios for visual AR comparable to or higher than text SD, without introducing extra models, modifying the target model, or breaking losslessness.

**Key Insight**: Instead of independent sampling, use Coupling tools from information theory to share randomness between draft samplings across iterations. This maximizes $\Pr[X^{(t)} = X^{(t-1)}]$ while ensuring each marginal distribution remains unchanged. Since the losslessness of SD depends only on the marginal distribution of the draft and is independent of correlations between drafts, Coupling provides a "free" gain in stability.

**Core Idea**: Replace the independent sampling line in the SJD drafting stage with Maximal Coupling (equivalent to reusing the Modified Rejection Sampler (MRS) from verification) or Gumbel Sharing Coupling. This one-line change, with zero extra training and zero extra memory, pushes the collision probability from near zero to the vicinity of the $1 - \mathcal{D}_{TV}$ upper bound.

## Method

### Overall Architecture

Input is any pretrained AR visual model $p_\theta$, target length $N$, and window length $L$. The output sequence $X$ has a distribution strictly equal to original token-by-token AR sampling. The system follows the three-stage SJD pipeline: (1) Drafting: Parallelly sample draft tokens for each position in the window from the distribution $p^{(t-1)}$ obtained in the previous iteration; (2) Evaluate: The target model calculates new distributions $p^{(t+1)}_j = p_\theta(\cdot \mid X^t_{0:j-1})$ in parallel; (3) Verify: Perform sequential verification using MRS, stopping at the first rejection, committing previous tokens, and inheriting the new sample from the rejected position for the next draft. SCD's modification is entirely in the sampler of step (1): replace "independent sampling from $p^t_j$" with "sampling a pair $(X^t_j, X^{t-1}_j)$ from the Coupled joint distribution of $p^t_j, p^{t-1}_j$ and taking $X^t_j$ as the draft." Marginal properties of Coupling ensure $X^t_j$ still follows $p^t_j$, preserving losslessness.

### Key Designs

1.  **Maximal Coupling ($\pi_{MC}$)**:
    *   **Function**: Given $p^{(t)}$ and $p^{(t-1)}$, constructs a joint distribution $\pi(x, y)$ such that $\Pr[X = Y]$ reaches the theoretical upper bound $1 - \mathcal{D}_{TV}(p^{(t)}, p^{(t-1)})$.
    *   **Mechanism**: The authors observe that the MRS used in verification is itself a maximal coupling—given $X \sim Q$, MRS outputs $Y$ such that $Y \sim P$ and $\Pr[Y = X] = 1 - \mathcal{D}_{TV}(P, Q)$. Thus, drafting simply reuses MRS: starting from $X^{t-1}_j$, it runs $\texttt{MRS}(p^t_j, p^{t-1}_j, X^{t-1}_j)$ to get the new draft $X^t_j$.
    *   **Design Motivation**: The coupling cost $C(\pi) = \Pr[X=Y]$ equals the token-level collision probability, which directly determines the next iteration's acceptance rate. $\pi_{MC}$ greedily pushes collisions to the limit, maximizing 1-step acceptance while strictly preserving the output distribution.

2.  **Gumbel Sharing Coupling ($\pi_{GS}$)**:
    *   **Function**: An alternative Coupling implementation that shares the same Gumbel noise $G$ across two categorical samplings, so $X = \arg\max_i (\log P_i + g_i)$ and $Y = \arg\max_i (\log Q_i + g_i)$ have a high probability of selecting the same token when distributions are similar.
    *   **Mechanism**: Based on the Gumbel-Max trick, the single-step collision lower bound is $C(\pi_{GS}) \ge (1 - \mathcal{D}_{TV})/(1 + \mathcal{D}_{TV})$, slightly lower than $\pi_{MC}$. However, this bound holds for any pair of distributions, providing guarantees for multi-step iterations $\mathrm{Hamm}(t, t+N)$.
    *   **Design Motivation**: While $\pi_{MC}$ is 1-step greedy optimal, it lacks non-trivial multi-step guarantees. $\pi_{GS}$ provides long-range stability, performing better in tasks where drafts are highly predictable (e.g., temporal redundancy in video AR or low-res images).

3.  **Zero-Overhead Implementation**:
    *   **Function**: Merges Drafting MRS and Verify MRS into a single loop without extra forwards.
    *   **Mechanism**: The authors observe that $\texttt{MRS}(p^t_j, p^{t-1}_j, X^t_j)$ in drafting and $\texttt{MRS}(p^{t+1}_j, p^t_j, X^t_j)$ in verification are the same operation across consecutive turns. By vectorizing the verification loop and identifying the first rejection index, drafting and verification are completed simultaneously.
    *   **Design Motivation**: Reduces the per-NFE overhead of $\pi_{MC}$ to < 5% (e.g., vectorized MRS takes 1.5 ms vs Transformer forward of 26-36 ms on Janus-Pro 7B).

### Loss & Training
Completely training-free. SCD is a pure inference-time algorithm replacement. All logit post-processing (top-k, CFG) is applied before defining $p^{(t)}$ to maintain the integrity of the lossless proof.

## Key Experimental Results

### Main Results

| Model / Dataset | Config | NFE ↓ | Latency ↓ | Gain | FID ↓ | CLIP ↑ |
|---------------|------|-------|--------|--------|-------|--------|
| Lumina-mGPT / MS-COCO | Vanilla AR | 2390 | 102.0 s | $1.0\times$ | 30.79 | 31.31 |
| Lumina-mGPT / MS-COCO | SJD ($L=64$) | 1036 | 43.0 s | $2.31\times$ | 30.81 | 31.31 |
| Lumina-mGPT / MS-COCO | + $\pi_{MC}$ ($L=64$) | 568 | 24.4 s | $\mathbf{4.21\times}$ | 30.83 | 31.37 |
| Lumina-mGPT / MS-COCO | + $\pi_{GS}$ ($L=64$) | 568 | 24.2 s | $\mathbf{4.21\times}$ | 30.90 | 31.37 |
| Janus-Pro 7B / MS-COCO | SJD ($L=32$) | 318 | 10.6 s | $1.25\times$ | 37.76 | – |
| Janus-Pro 7B / MS-COCO | + $\pi_{GS}$ ($L=32$) | 154 | 5.39 s | $\mathbf{2.45\times}$ | 37.49 | – |
| Cosmos-1-AR / Real-Estate-10k | Vanilla AR | 7680 | 157 s | $1.0\times$ | FVD 156.9 | – |
| Cosmos-1-AR / Real-Estate-10k | + $\pi_{GS}$ ($L=128$) | 564 | 13.6 s | $\mathbf{13.6\times}$ | FVD 152.4 | – |

### Ablation Study

| Config | NFE | Description |
|------|-----|------|
| SJD baseline | 1036 | Independent sampling; ~94% tokens change per round |
| + $\pi_{MC}$ | 568 | Maximal coupling; maximizes 1-step collision |
| + $\pi_{GS}$ | 568 | Gumbel coupling; guarantees long-range collision |
| vs. Lossy GSD ($G=10$) | 701 | GSD is faster but FID drops from 30.79 to 33.21 |
| Coupling strength $\alpha$ | – | NFE decreases monotonically as $\alpha \to 1$ |
| Window $L$ sweep | – | SJD plateaus after $L\!>\!16$; SCD benefits from larger $L$ |

### Key Findings
*   The relationship between acceleration and window size $L$ is revealing: standard SJD plateaus at $\sim 2.3\times$ regardless of $L$ ($16, 32, 64$), whereas SCD reaches $4.2\times$ at $L=64$. This proves SJD is bottlenecked by independent sampling capping the acceptance rate.
*   Video AR acceleration is much higher than image ($13.6\times$ vs $4.2\times$): strong temporal redundancy between frames makes draft prediction easy, where long-range stability of $\pi_{GS}$ is particularly effective.
*   Experimenting with coupling strength $\alpha$ provides clean causal evidence: as $\alpha$ increases from 0 to 1, both token-level Hamming distance and NFE decrease monotonically, validating the chain: "increased collision $\to$ increased context stability $\to$ decreased NFE."

## Highlights & Insights
*   The insight "probability proximity $\neq$ sample proximity" is powerful: the failure of standard SJD is precisely attributed to the Rényi-2 entropy upper bound $C_{SJD} \le e^{-1/2 \cdot (H_2(p) + H_2(q))}$, explaining why visual AR (flat distribution) is harder to accelerate than text.
*   The use of MRS as both a verification step and a drafting sampler is an elegant realization that the two share the same mathematical structure.
*   The paper exemplifies the paradigm of "small change + rigorous proof + strong results": a few lines of code, zero training, guaranteed losslessness, and $4-13\times$ acceleration.

## Limitations & Future Work
*   The acceleration ceiling is determined by the target model's $\mathcal{D}_{TV}(p^{(t)}, p^{(t-1)})$. If context changes too abruptly (e.g., high-resolution generation with strong CFG), even $\pi_{MC}$ is limited by this TV boundary.
*   The choice between $\pi_{MC}$ and $\pi_{GS}$ is currently empirical; no task-adaptive selection rule has been provided yet.
*   Exploration is limited to visual AR; potential gains in AR audio or robotics token sequences remain to be explored.

## Related Work & Insights
*   **vs. SJD (Teng et al., 2024)**: SJD uses the previous distribution as a draft but suffers from independent sampling limits. Ours uses Coupling to reach $4.2\times$ acceleration.
*   **vs. GSD (So et al., 2025)**: GSD is lossy and results in FID degradation. SCD is faster and strictly lossless.
*   **vs. Medusa (Cai et al., 2024)**: Medusa requires training extra heads for $4\times$ text acceleration. SCD achieves similar or better gains in vision with zero training.

## Rating
*   Novelty: ⭐⭐⭐⭐⭐ Introducing information-theoretic Coupling to SJD provides an upper-bound-level improvement with minimal changes.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of image (Lumina/Janus) and video (Cosmos) with detailed sweeps across $\alpha$, Hamming distance, and CFG.
*   Writing Quality: ⭐⭐⭐⭐⭐ Logical flow from motivation to proof and algorithm.
*   Value: ⭐⭐⭐⭐⭐ High industrial value due to easy plug-and-play capability and significant acceleration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Annealed Relaxation of Speculative Decoding for Faster Autoregressive Image Generation](../../AAAI2026/image_generation/annealed_relaxation_of_speculative_decoding_for_faster_autor.md)
- [\[ICCV 2025\] Grouped Speculative Decoding for Autoregressive Image Generation](../../ICCV2025/image_generation/grouped_speculative_decoding_for_autoregressive_image_generation.md)
- [\[ICML 2026\] DFlash: Block Diffusion for Flash Speculative Decoding](dflash_block_diffusion_for_flash_speculative_decoding.md)
- [\[CVPR 2026\] SJD-PAC: Accelerating Speculative Jacobi Decoding via Proactive Drafting and Adaptive Continuation](../../CVPR2026/image_generation/sjd-pac_accelerating_speculative_jacobi_decoding_via_proactive_drafting_and_adap.md)
- [\[CVPR 2026\] SparVAR: Exploring Sparsity in Visual Autoregressive Modeling for Training-Free Acceleration](../../CVPR2026/image_generation/sparvar_exploring_sparsity_in_visual_autoregressive_modeling_for_training-free_a.md)

</div>

<!-- RELATED:END -->
