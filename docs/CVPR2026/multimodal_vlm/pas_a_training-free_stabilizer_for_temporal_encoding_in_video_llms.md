---
title: >-
  [Paper Note] PAS: A Training-Free Stabilizer for Temporal Encoding in Video LLMs
description: >-
  [CVPR 2026][Multimodal VLM][Video LLM] PAS diagnoses the instability of temporal encoding in Video LLMs as "sampling an inverse Fourier temporal kernel with high-frequency ripples." It proposes training-free multi-head inverse phase smoothing—applying small, opposite temporal phase offsets to different attention head queries before standard aggregation. Thi
tags:
  - CVPR 2026
  - Multimodal VLM
  - Video LLM
  - M-RoPE
date: 2026-05-08
content_hash: 917ec28d46828278
---
# PAS: A Training-Free Stabilizer for Temporal Encoding in Video LLMs

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Sun_PAS_A_Training-Free_Stabilizer_for_Temporal_Encoding_in_Video_LLMs_CVPR_2026_paper.html)  
**Code**: https://github.com/Bowen-Sun-0728/PAS  
**Area**: Multimodal VLM / Video Understanding  
**Keywords**: Video LLM, M-RoPE, Temporal Encoding, Phase Smoothing, Training-Free

## TL;DR
PAS diagnoses the instability of temporal encoding in Video LLMs as "sampling an inverse Fourier temporal kernel with high-frequency ripples." It proposes training-free multi-head inverse phase smoothing—applying small, opposite temporal phase offsets to different attention head queries before standard aggregation. This effectively performs a controlled moving average to smooth out ripples, consistently improving performance across nine video benchmarks with near-zero additional overhead.

## Background & Motivation

**Background**: Video LLMs commonly generalize the Rotary Positional Encoding (RoPE), designed for text, into a multimodal version called M-RoPE. This allocates sets of frequency lines along the three dimensions of time (T), height (H), and width (W) to encode positions for video tokens.

**Limitations of Prior Work**: The authors observe a counter-intuitive vulnerability: minor variations in frame sampling (frame rate, sampling offset) can flip attention distributions. This causes keyframes that should be attended to to be suppressed while irrelevant frames are amplified, leading to downstream errors (Figure 1 in the paper provides real failure cases). This is not random noise but a structural issue inherent to temporal encoding.

**Key Challenge**: M-RoPE along the temporal axis is equivalent to providing a set of fixed temporal frequency lines, where each line contributes a periodic term to the attention logit. The temporal kernel $m(\Delta t)$ obtained by the inverse Fourier transform of these lines naturally contains ripples at the "frame scale" (an intrinsic property of the spectrum, not a side effect of sparse sampling). Ripples imply that adjacent frames are multiplied by vastly different modulation factors, making it uncontrollable whether attention is dominated by "content similarity" or by "which time interval it happens to fall into."

**Goal**: Smooth this temporal kernel without retraining, changing the token budget, or altering the positional encoding structure, ensuring attention returns to being content-driven and robust to small temporal perturbations.

**Key Insight**: Since the problem is "sampling the same rippled kernel at slightly different time lags," a natural solution is to "average over proximal time lags"—exactly the intuition behind using a moving average filter to suppress high-frequency ripples in signal processing.

**Core Idea**: Apply small, opposite temporal phase offsets to queries of different attention heads, so each head observes a slightly shifted version of the kernel. Standard multi-head aggregation then serves as a "temporal moving average," smoothing high-frequency ripples while preserving low-frequency trends.

## Method

### Overall Architecture
PAS (Phase Aggregated Smoothing) is an inference-time plugin. The core logic involves clarifying the problem from a Fourier perspective and then smoothing the kernel using a minimalist mechanism. The logical chain is: **M-RoPE rotated attention logit $\approx$ content inner product $\times$ temporal kernel $\mathrm{Re}\{m(\Delta t)\}$ (phase modulation perspective) $\to$ a smoother kernel leads to more stable attention regarding time lags (Lipschitz stability) $\to$ assigning opposite small phase offsets to heads and aggregating = controlled moving average on the kernel (smoothing ripples) $\to$ as long as Nyquist is satisfied, the recoverable spectrum of each head remains invariant under time shifts (changing "how to sample," not "what was encoded").**

Specifically for a forward pass: After calculating standard Q/K/V for each attention row, query heads are divided into $K$ phase groups. A gentle phase operator $\Gamma_{\delta_h}$ is applied to the **temporal half-dimension** of the queries in the $h$-th group (acting only on video tokens, leaving spatial dimensions untouched). Multi-head attention and aggregation then proceed as usual. The method introduces no new parameters, does not change tokenization, and does not increase the number of tokens. In benchmarks, the throughput of $(76.8\pm4.0)\times10^3$ tokens/s is indistinguishable from the $(77.2\pm3.1)\times10^3$ tokens/s of the original backbone.

> This method is a "single mechanism + theoretical analysis" rather than a multi-stage serial pipeline. Its core lies in frequency-domain matrix/phase operations. Since a framework diagram might not clarify the mechanism better than equations, no flowchart is provided.

### Key Designs

**1. Phase Modulation Perspective: Diagnosing temporal instability as "sampling a rippled temporal kernel"**

This is the diagnostic foundation of the paper and the prerequisite for the mechanism. RoPE rotates query/key in each 2D subspace with phase $e^{j\omega_i s}$. Defining the content term $C_i:=z_i w_i^*$ (independent of displacement), the rotated logit depends only on the relative displacement $\Delta$: $\langle\tilde q,\tilde k\rangle(\Delta)=\mathrm{Re}\big[\sum_i C_i e^{j\omega_i\Delta}\big]$. The authors prove (Theorem 1) that when the number of frequency lines $m$ is large and energy is approximately uniform, this sum concentrates toward its mean mode, allowing the logit to be approximated as a "pure content inner product" multiplied by a scalar temporal kernel:

$$\langle\tilde q,\tilde k\rangle(\Delta)\approx\langle q,k\rangle\cdot\mathrm{Re}\{m(\Delta)\},\qquad m(\Delta):=\tfrac{1}{m}\sum_{i=0}^{m-1}e^{j\omega_i\Delta}.$$

The key insight is that $m(\Delta)$ is an average of cosines, which is **intrinsically rippled** at the frame scale. Thus, tiny changes in time lag $\delta t$ can cause the modulation factor to swing drastically, turning content-driven attention into "timing-driven" attention. This step precisely characterizes the vague "Video LLM sensitivity to sampling" as "sampling an unsmooth kernel," pointing toward "smoothing the kernel" as the solution.

**2. Multi-Head Inverse Phase Smoothing: Smoothing ripples with controlled moving average**

To address the ripples exposed in Design 1, PAS does not modify the spectrum but leverages multi-head aggregation to perform "proximal lag averaging." By assigning small time shifts $\{\delta_h\}$ and normalized weights $\{a_h\}$ ($\sum_h a_h=1$) to $H$ heads, the effective modulation after aggregation becomes $m_{\mathrm{eff}}(\Delta t)=\sum_h a_h\, m(\Delta t+\delta_h)$. The authors prove (Theorem 3) that the mean square local variation of this weighted average does not exceed that of the original kernel, i.e., $V_\varepsilon(m_{\mathrm{eff}})\le V_\varepsilon(m)$, and is strictly smaller if phases are not all identical. In the frequency domain, this is equivalent to multiplying the line spectrum by an aggregation kernel $K(\omega)=\sum_h a_h e^{j\omega\alpha\delta_h}$, where the magnitude $|K(\omega)|\le1$. When head phases are sufficiently dispersed, it strictly attenuates non-zero frequencies—effectively suppressing high-frequency ripples while retaining low-frequency trends.

Combined with the Lipschitz stability in Theorem 2 (the smoother the kernel, the more the logit changes at most linearly with $\delta t$: $|A(\Delta t+\delta t)-A(\Delta t)|\le|\langle q,k\rangle|\,L_m|\delta t|$, where $L_m$ is the maximum local slope of the kernel), the causal chain is closed: **Multi-phase averaging $\Rightarrow$ Smoother $m_{\mathrm{eff}}$ $\Rightarrow$ Attention more robust to phase changes.** The "opposite/symmetric" offset design is intended to disperse phases sufficiently to minimize $|K(\omega)|$.

**3. Temporal Half-Dimension Only + Nyquist Fidelity: Ensuring "sampling method changes, encoded content remains"**

A natural concern is whether adding phase offsets destroys the positional semantics originally encoded by RoPE. The authors address this with a fundamental Fourier fact: time shifts only change the phase, not the magnitude spectrum. Theorem 4 proves that under a fixed window and satisfying the Nyquist bandlimit, after adding a time shift $\delta$ to a head, its $N$-point DFT is only modulated by a per-bin phase factor $e^{j\theta_k(\delta)}$, while the magnitude $|X_\delta[k]|=|X[k]|$ remains completely unchanged. Furthermore, as long as $\delta_{\max}-\delta_{\min}\le\Delta$, the discrete order of frames/bins between heads will not be scrambled. Therefore, smoothing only emerges "after standard multi-head aggregation," while the spectrum encoded by each individual head remains intact.

In engineering, this design results in three constraints: $\Gamma_\delta$ only acts on the **temporal half-dimension** (spatial encoding is untouched); it only applies to **video tokens** (using a right-aligned mask to lock the video span); the hook point is placed after the base positional encoding is applied to Q, and it is compatible with MHA/GQA (broadcasting offsets along head dimensions). In terms of overhead, PAS only performs a per-token linear transformation on the temporal half-dimension of Q, with an additional cost of $C_{\mathrm{PAS}}/C_{\mathrm{attn}}\le p_t S_v/S^2$ ($p_t$ being the proportion of the affected half-dimension), which is negligible when $S$ is in the hundreds or thousands.

### Loss & Training
PAS is a **training-free** inference-time plugin with no training targets or fine-tuning. The only "hyperparameters" are the number of phase groups $K$ and the offsets $\{\delta_h\}$ for each group (measured in bins, where $\phi=1.0$ equals a one-bin shift). The default configuration is minimalist: $K=2$, with offsets $[0, 0.5]$, providing plug-and-play capability for all backbones.

## Key Experimental Results

> Context: **bin** refers to a video token produced by the sampler+merger (merging adjacent frames along time); **matched token budget** means the total number of video tokens per video is aligned across all methods to ensure gains come from the mechanism rather than more tokens. Acc denotes classification accuracy, and Macro-F1 denotes the macro-averaged F1 score across categories.

### Main Results
The backbone is Qwen2.5-VL-7B-Instruct, compared against two training-free baselines, SlowFast-LLaVA and TS-LLaVA, under aligned token budgets, while also measuring the effect of stacking PAS on top of them.

| Dataset / Metric | Default backbone | SlowFast-LLaVA | TS-LLaVA | PAS (Ours) | SlowFast+PAS |
|--------------|------------------|----------------|----------|-----------|--------------|
| 20BN-Jester (Acc) | 16.0 | 14.9 | 15.4 | 18.3 | **19.6** |
| Kinetics-700 (Acc) | 44.9 | 45.1 | 44.7 | 48.2 | **49.8** |
| MVBench (Acc) | 67.2 | 69.2 | 67.8 | 69.5 | **71.0** |
| TempCompass (Overall) | 71.5 | 73.5 | 71.4 | 73.3 | **73.9** |
| EgoSchema (Acc) | 63.5 | 63.6 | 65.8 (TS) | 63.9 | 64.1 |
| MMBench-Video (0–3) | 1.71 | 1.76 | 1.70 | 1.78 | **1.81** |

PAS used alone consistently outperforms the backbone on action recognition (phase-sensitive) benchmarks, achieving SOTA among single models on 20BN-Jester and Kinetics-700. When stacked with SlowFast-LLaVA / TS-LLaVA, it further raises the performance ceiling, indicating it is synergistic with multi-rate or thumbnail-based approaches.

### Ablation Study
Since PAS has no "removable modules," the core verification comes from parameter scanning and sampling rate ablation—directly testing whether "gains truly come from smoothing the temporal kernel."

| Analysis Dimension | Setting | Key Finding |
|---------|------|---------|
| Offset Magnitude $\Delta$ ($K{=}2$) | Scan $0.0\!\to\!1.0$ | Consistent significant gains across three motion-intensive sets for $\Delta\in[0.3, 0.8]$, showing a broad plateau; $\Delta{\approx}0.5$ is a safe default across datasets. |
| Sampling Rate $r$ | Sampled/Total frames $\in[0,1]$ | Lower $r$ (sparse sampling) leads to larger Gain; at high $r$, results are statistically indistinguishable from the backbone—aligning with Theorem 2. |
| Nyquist Compliance | Breakfast subsampling (sub-Nyquist) | Under-sampling introduces aliasing which limits Gain, narrower improvement windows, and smaller absolute gains—aligning with Theorem 4. |
| Inference Overhead | A100 80G, matched seq | $(76.8\pm4.0)$ vs $(77.2\pm3.1)\times10^3$ tokens/s, statistically no difference. |

### Key Findings
- The largest Gain occurs in **sparse sampling / low frame rate** scenarios. In these cases, the temporal kernel is probed sparsely, the phase difference between adjacent bins is large, and ripples most severely damage attention. This is where the PAS moving average is most effective. As sampling density increases, the kernel is naturally smoothed, leaving less room for PAS.
- There is a **broad plateau** for the offset magnitude (0.3–0.8 are all effective), indicating the method is insensitive to hyperparameters and does not require per-dataset tuning. $K=2$ with $[0, 0.5]$ is a reusable default.
- The three theorems correspond one-to-one with the three sets of experiments (offset scan, sampling rate, Nyquist), creating a rare and tight alignment between theoretical prediction and experimental verification.

## Highlights & Insights
- **Elevating an engineering phenomenon to a provable frequency-domain problem**: Moving from "Video LLMs are sensitive to sampling" to "sampling an inverse Fourier kernel with ripples," and then to stability guarantees spanning three theorems. The diagnosis and solution create a very satisfying "Aha!" moment.
- **Zero-cost plug-and-play**: No retraining, no token changes, and no touching spatial encoding. By only adding phases to the temporal half-dimension of Q before normal aggregation, throughput remains virtually unchanged—making this "free" robustness upgrade very deployment-friendly.
- **Transferable insight**: Reinterpreting "multi-head aggregation" as a "moving average of the positional kernel" is an insight that could migrate to other long-sequence or multimodal scenarios using RoPE/relative positional encoding, wherever the positional kernel might contain ripples.

## Limitations & Future Work
- **Reliance on Nyquist assumption**: Theoretical guarantees (spectrum invariance, modified sampling only) rely on temporal sampling satisfying the Nyquist bandlimit. In cases of severe under-sampling (sub-Nyquist, e.g., Breakfast subsampling), fractional delays are no longer all-pass, guarantees fail, and gains are eaten by aliasing.
- **Biased gain scenarios**: The method provides the most benefit under sparse sampling/low frame rates. At dense sampling rates, there is no statistical difference from the backbone, meaning it offers limited help to settings that already employ high-frame-rate, strong temporal modeling.
- **Validation focused on 7B + Qwen2.5-VL**: Main experiments were conducted on a single backbone. Generalizability across model scales or different M-RoPE frequency allocation schemes requires further verification (⚠️ the paper did not provide large-scale results across different backbones).
- Future directions: Adaptively determining the offset magnitude and number of groups $K$ based on the estimated local kernel slope $L_m$ instead of using fixed values, or jointly optimizing with the sampling rate.

## Related Work & Insights
- **vs SlowFast-LLaVA / TS-LLaVA**: These methods combat sparse sampling at the input side by expanding temporal coverage using dual paths (Slow/Fast) or thumbnails + light sampling. PAS does not change inputs or add tokens; instead, it covers multiple temporal phases within a single forward pass inside the attention mechanism. They are orthogonal and can be stacked (as shown by the SlowFast+PAS Gain).
- **vs Inference-time multi-pass resampling**: Traditional approaches rely on averaging predictions from multiple resampling offsets, which doubles or triples latency. PAS compresses "multi-phase coverage" into the multi-head structure of a single forward pass with near-zero overhead.
- **vs Vision-specific 2D/3D RoPE designs**: Those lines focus on axis coupling and the expressiveness of frequency allocation. PAS focuses on the **robustness** of absolute temporal encoding, approaching the problem from the perspective of Fourier/phase stability.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Reinterpreting multi-head aggregation as a moving average on the positional kernel, supported by three rigorous theorems, is a very fresh perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Nine benchmarks with an offset/sampling rate ablation that perfectly matches the theory, though verification on a single 7B backbone leaves some questions about cross-model generalizability.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from diagnosis to theory to mechanism to experiment is clear, with tight correspondence between theorems and results.
- Value: ⭐⭐⭐⭐ Training-free, zero-overhead, and plug-and-play. It offers high value for deploying Video LLMs in low-frame-rate/sparse-sampling scenarios, though gains are limited in dense sampling contexts.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Pointing at Parts: Training-Free Few-Shot Grounding in Multimodal LLMs](pointing_at_parts_training-free_few-shot_grounding_in_multimodal_llms.md)
- [\[CVPR 2026\] TimeLens: Rethinking Video Temporal Grounding with Multimodal LLMs](timelens_rethinking_video_temporal_grounding_with_multimodal_llms.md)
- [\[CVPR 2026\] STiTch: Semantic Transition and Transportation in Collaboration for Training-Free Zero-Shot Composed Image Retrieval](stitch_semantic_transition_and_transportation_in_collaboration_for_training-free.md)
- [\[CVPR 2026\] DRS-GUI: Dynamic Region Search for Training-Free GUI Grounding](drs-gui_dynamic_region_search_for_training-free_gui_grounding.md)
- [\[NeurIPS 2025\] Training-free Online Video Step Grounding](../../NeurIPS2025/multimodal_vlm/training-free_online_video_step_grounding.md)

</div>

<!-- RELATED:END -->
