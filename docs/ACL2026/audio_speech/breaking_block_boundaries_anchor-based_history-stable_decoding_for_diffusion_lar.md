---
title: >-
  [Paper Note] Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models
description: >-
  [ACL 2026][Audio & Speech][diffusion language models] This paper proposes AHD (Anchor-based History-stable Decoding), a training-free plug-and-play dynamic decoding strategy that identifies cross-block stable tokens in d…
tags:
  - "ACL 2026"
  - "Audio & Speech"
  - "diffusion language models"
  - "semi-autoregressive decoding"
  - "cross-block stable tokens"
  - "dynamic anchor"
  - "inference acceleration"
date: 2026-05-08
content_hash: 7f050e4b21d1f1d0
---

# Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models

**Conference**: ACL 2026
**arXiv**: [2604.08964](https://arxiv.org/abs/2604.08964)
**Code**: [GitHub](https://github.com/zs1314/AHD)
**Area**: Audio & Speech
**Keywords**: diffusion language models, semi-autoregressive decoding, cross-block stable tokens, dynamic anchor, inference acceleration

## TL;DR
This paper proposes AHD (Anchor-based History-stable Decoding), a training-free plug-and-play dynamic decoding strategy that identifies cross-block stable tokens in diffusion LLMs by tracing historical trajectories via dynamic anchors, enabling early unlocking. On BBH, AHD reduces decoding steps by 80% while improving performance by 3.67%.

## Background & Motivation

**Background**: Diffusion large language models (dLLMs) such as LLaDA have emerged as strong alternatives to autoregressive LLMs. Semi-autoregressive (Semi-AR) decoding is widely adopted, partitioning the output sequence into multiple blocks decoded left-to-right, with diffusion-based iterative denoising applied within each block.

**Limitations of Prior Work**: Semi-AR decoding suffers from a severe "block boundary delay" problem — many tokens converge to their final values and remain stable before their designated block is reached, yet are forced to wait until that block's turn. The delayed decoding of these "cross-block stable tokens" wastes substantial decoding steps and degrades performance by suppressing the radiation effect within local regions.

**Key Challenge**: How can cross-block stable tokens be reliably identified? Existing approaches based on single-step confidence or entropy are unreliable: (1) already-stable tokens may still exhibit local fluctuations, causing false positives; (2) historical information is isolated in standard decoding, with each step's prediction depending only on the previous step.

**Goal**: To break the block boundary constraints of Semi-AR decoding by early-unlocking cross-block stable tokens, simultaneously improving both efficiency and generation quality.

**Key Insight**: Three key observations — (1) naive look-ahead decoding is unreliable due to local fluctuations; (2) token stability is strongly correlated with convergence trends (absolute stability tendency); (3) historical information remains isolated in standard decoding. This motivates the incorporation of historical trajectory information for global stability assessment.

**Core Idea**: At each decoding step, the current step serves as a dynamic anchor to trace back through a historical buffer and compute an anchored consistency score, capturing the absolute stability tendency of each token. Once stability is confirmed, the token is early-unlocked across block boundaries.

## Method

### Overall Architecture
Building on Semi-AR decoding, AHD divides the sequence into the current block $B_{current}^t$ and future blocks $B_{future}^t$. Within the current block, confidence-aware parallel decoding is applied. For future blocks, AHD maintains a historical buffer at each position and computes stability via dynamic anchor look-back. Tokens satisfying the stability criterion are early-unlocked and added to the decoding set.

### Key Designs

1. **Historical Buffer + Dynamic Anchor**:

    - Function: Maintains historical distribution trajectories for each future block position, enabling cross-step stability monitoring.
    - Mechanism: For each position $j$ in the future block, a historical buffer of length $H$ is maintained: $\mathcal{H}_j^t = \{P_j^{t-H+1}, ..., P_j^t\}$. Using the current step $P_j^t$ as the dynamic anchor, the anchored KL divergence is computed by looking back: $\delta_j^{t,\tau} = D_{KL}(P_{j,anchor}^t || P_j^{t-\tau})$.
    - Design Motivation: Single-step confidence and entropy are sensitive to local fluctuations, whereas anchor-based historical consistency provides a global perspective capable of capturing signals from the early stages of absolute stability trends.

2. **Anchored Consistency Score**:

    - Function: Aggregates stability evidence within the historical window to make reliable cross-block decoding decisions.
    - Mechanism: An exponentially decaying weighted sum is applied to the historical consistency sequence $\{\delta_j^{t,1}, ..., \delta_j^{t,H-1}\}$, yielding $D_j^t(acs) = \sum_{\tau=1}^{H-1} w_\tau \delta_j^{t,\tau}$, where $w_\tau = e^{-\lambda\tau}/Z$ assigns higher weight to more recent history. A token is deemed to have reached its absolute stability tendency when $D_j^t(acs) < \varepsilon$.
    - Design Motivation: Exponential decay weights balance sensitivity to recent changes and robustness to long-term trends; the threshold $\varepsilon$ controls the conservativeness of the unlocking criterion.

3. **Cross-block Early Unlocking**:

    - Function: Breaks block boundaries by decoding stable future-block tokens ahead of schedule.
    - Mechanism: The set of future-block positions satisfying the stability criterion, $G_f^t = \{j \mid j \in B_{future} \wedge D_j^t(acs) < \varepsilon\}$, is merged with the current block's decoding set $G_c^t$ to form $G_{unmasked}^t$, and the sequence is updated jointly.
    - Design Motivation: Stable tokens exhibit a "radiation effect" — once a token stabilizes, it accelerates the convergence of neighboring tokens. Early unlocking releases this radiation effect, simultaneously accelerating inference and improving generation quality.

### Loss & Training
AHD is a training-free plug-and-play method applied directly at inference time. Default hyperparameters: historical buffer length $H=6$, consistency threshold $\varepsilon=0.01$, and decay rate $\lambda$ controlling the weight distribution.

## Key Experimental Results

### Main Results (LLaDA-8B-Instruct)

| Task | Metric | AHD | Vanilla | Step Reduction |
|------|--------|-----|---------|----------------|
| BBH | Score↑ | 56.78 | 53.11 | 80% |
| HumanEval | Score↑ | 43.29 | 40.85 | 70% |
| MBPP | Score↑ | 31.20 | 29.20 | 74% |
| MMLU-Pro | Score↑ | 37.42 | 35.57 | 48% |
| Asdiv | Score↑ | 77.09 | 75.57 | 76% |

### Ablation Study

| Method | BBH Score | Step Reduction | Note |
|--------|-----------|----------------|------|
| Vanilla | 53.11 | 0% | Standard decoding |
| Fast-dLLM | 53.17 | 78% | Performance maintained but no gain |
| KLASS | 53.03 | 62% | Slight degradation |
| Saber | 52.88 | 66% | Performance degradation |
| AHD | 56.78 | 80% | Only method achieving simultaneous gains in performance and efficiency |

### Key Findings
- AHD is the only method that accelerates decoding while also improving performance; other acceleration strategies (Saber, KLASS) typically incur performance degradation.
- AHD is equally effective on LLaDA-1.5, yielding +1.55 on BBH with 78% step reduction, demonstrating generalizability.
- AHD extends successfully to vision-language (MMaDA) and audio-language (DIFFA) settings, confirming cross-modal applicability.

## Highlights & Insights
- **"Radiation Effect of Stable Tokens"**: Stable tokens are found to appear in clustered patterns, where the stabilization of one token accelerates the convergence of its neighbors. This insight is valuable for understanding the decoding dynamics of diffusion LLMs.
- **Counter-intuitive Finding — "Acceleration Yields Improvement"**: Early unlocking not only speeds up inference but also enhances generation quality by releasing the radiation effect, challenging the conventional speed–quality trade-off assumption.
- **Generalizability of the Anchor Look-back Mechanism**: The approach of assessing stability via historical trajectories is transferable to any iterative generation process (e.g., early determination of pixels in diffusion-based image generation).

## Limitations & Future Work
- Maintaining historical buffers introduces additional memory overhead, which may become a bottleneck for very long sequence generation.
- The threshold $\varepsilon$ and buffer length $H$ require tuning for different models and tasks.
- Validation has been conducted primarily on the LLaDA family; applicability to other dLLM architectures (e.g., MDLM) remains to be verified.
- The theoretical analysis assumes monotonic convergence of token stability, which may not hold in extreme cases.

## Related Work & Insights
- **vs. Fast-dLLM**: Fast-dLLM employs confidence thresholding for acceleration but achieves no performance gain; AHD leverages historical trajectory assessment to achieve both acceleration and improvement.
- **vs. Saber**: Saber uses a predictor for selective denoising but incurs performance degradation; AHD's dynamic anchor approach is more robust.
- **vs. PC-sampler**: PC-sampler modifies the sampling process without reducing the number of steps; AHD directly reduces steps by 70–80%.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The derivation chain from three observations to the dynamic anchor method is rigorous and natural; the finding that acceleration yields improvement is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 7 language benchmarks + 5 vision + 5 audio, two dLLM models, and comparisons against 5 baselines.
- Writing Quality: ⭐⭐⭐⭐⭐ The observation→insight→method narrative flows smoothly, with outstanding figure design (especially the heatmap analysis).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Temporal Contrastive Decoding: A Training-Free Method for Large Audio-Language Models](temporal_contrastive_decoding_a_training-free_method_for_large_audio-language_mo.md)
- [\[ACL 2026\] Do We Need Distinct Representations for Every Speech Token? Unveiling and Exploiting Redundancy in Large Speech Language Models](do_we_need_distinct_representations_for_every_speech_token_unveiling_and_exploit.md)
- [\[ACL 2026\] HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models](halluaudio_a_comprehensive_benchmark_for_hallucination_detection_in_large_audio-.md)
- [\[ACL 2026\] How Hypocritical Is Your LLM Judge? Listener–Speaker Asymmetries in the Pragmatic Competence of Large Language Models](how_hypocritical_is_your_llm_judge_listener-speaker_asymmetries_in_the_pragmatic.md)
- [\[ACL 2026\] SpeakerSleuth: Can Large Audio-Language Models Judge Speaker Consistency across Multi-turn Dialogues?](speakersleuth_can_large_audio-language_models_judge_speaker_consistency_across_m.md)

</div>

<!-- RELATED:END -->
