---
title: >-
  [Paper Note] Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models
description: >-
  [ACL 2026][LLM Efficiency][Diffusion Language Models] Ours proposes AHD (Anchor-based History-stable Decoding), a training-free plug-and-play dynamic decoding strategy. By backtracking historical trajectories via dynamic…
tags:
  - "ACL 2026"
  - "LLM Efficiency"
  - "Diffusion Language Models"
  - "Semi-autoregressive Decoding"
  - "Cross-block Stable Tokens"
  - "Dynamic Anchors"
  - "Inference Acceleration"
date: 2026-05-08
content_hash: 3623c89106b806cc
---

# Breaking Block Boundaries: Anchor-based History-stable Decoding for Diffusion Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.08964](https://arxiv.org/abs/2604.08964)  
**Code**: [GitHub](https://github.com/zs1314/AHD)  
**Area**: LLM Efficiency  
**Keywords**: Diffusion Language Models, Semi-autoregressive Decoding, Cross-block Stable Tokens, Dynamic Anchors, Inference Acceleration

## TL;DR
Ours proposes AHD (Anchor-based History-stable Decoding), a training-free plug-and-play dynamic decoding strategy. By backtracking historical trajectories via dynamic anchors to identify cross-block stable tokens in diffusion LLMs, it achieves early unlocking, reducing decoding steps by 80% on BBH while improving performance by 3.67%.

## Background & Motivation

**Background**: Diffusion Large Language Models (dLLMs) such as LLaDA have emerged as powerful alternatives to autoregressive LLMs. Semi-autoregressive (Semi-AR) decoding is widely adopted—partitioning the output sequence into multiple blocks for sequential decoding from left to right, with each block denoised through diffusion iterations.

**Limitations of Prior Work**: Semi-AR decoding suffers from a severe "block boundary delay" issue—many tokens converge to final values and remain stable before their corresponding block decoding starts, but are forced to wait for their specific block's turn. Delayed decoding of these "cross-block stable tokens" not only wastes numerous decoding steps but also reduces performance by suppressing the radiation effect in local regions.

**Key Challenge**: How to accurately identify cross-block stable tokens? Existing methods (single-step judgments based on confidence/entropy) are unreliable: (1) already stable tokens may still exhibit local fluctuations leading to misjudgments; (2) historical information is isolated in standard decoding, where each step's prediction depends only on the previous step.

**Goal**: Break the block boundary constraints of Semi-AR decoding and simultaneously improve efficiency and performance by early unlocking cross-block stable tokens.

**Key Insight**: Three critical observations—(1) Naive look-ahead decoding is unreliable due to local fluctuations; (2) token stability is highly correlated with convergence trends (absolute stability trends); (3) historical information is isolated in standard decoding. Therefore, historical trajectory information must be introduced to determine global stability.

**Core Idea**: Use the current decoding step as a dynamic anchor at each iteration, backtrack through a historical buffer to calculate the anchored consistency score, capture the absolute stability trend of tokens, and perform early cross-block decoding once stability is confirmed.

## Method

### Overall Architecture
Building upon Semi-AR decoding, AHD divides the sequence into a current block $B_{current}^t$ and future blocks $B_{future}^t$. Within the current block, confidence-aware parallel decoding is used. For future blocks, AHD maintains a historical buffer for each position and calculates stability via dynamic anchor backtracking. Tokens meeting the criteria are unlocked early and added to the decoding set.

### Key Designs

1.  **Historical Buffer + Dynamic Anchor**:

    - **Function**: Maintains historical distribution trajectories for each future block position to enable cross-step stability monitoring.
    - **Mechanism**: For each position $j$ in the future blocks, a historical buffer $\mathcal{H}_j^t = \{P_j^{t-H+1}, ..., P_j^t\}$ of length $H$ is maintained. Taking the current step $P_j^t$ as the dynamic anchor, the anchored KL divergence is calculated as $\delta_j^{t,\tau} = D_{KL}(P_{j,anchor}^t || P_j^{t-\tau})$.
    - **Design Motivation**: While single-step confidence/entropy is sensitive to local fluctuations, anchor-based historical consistency provides a global perspective, capturing signals at the early stages of an absolute stability trend.

2.  **Anchored Consistency Score**:

    - **Function**: Aggregates stability evidence within the historical window to make reliable cross-block decoding decisions.
    - **Mechanism**: An exponential decay weighted sum is applied to the historical consistency sequence $\{\delta_j^{t,1}, ..., \delta_j^{t,H-1}\}$ to obtain $D_j^t(acs) = \sum_{\tau=1}^{H-1} w_\tau \delta_j^{t,\tau}$, where $w_\tau = e^{-\lambda\tau}/Z$ assigns higher weights to recent history. A token is judged to have reached an absolute stability trend when $D_j^t(acs) < \varepsilon$.
    - **Design Motivation**: Exponential decay weights balance sensitivity to recent changes with the robustness of long-term trends, while the threshold $\varepsilon$ controls the conservatism of unlocking.

3.  **Cross-block Early Unlocking**:

    - **Function**: Breaks block boundaries to decode stable future block tokens ahead of schedule.
    - **Mechanism**: The set of future block positions meeting the stability condition $G_f^t = \{j | j \in B_{future} \wedge D_j^t(acs) < \varepsilon\}$ is merged with the current block's decoding set $G_c^t$ to form $G_{unmasked}^t$, and the sequence is updated collectively.
    - **Design Motivation**: Stable tokens exhibit a "radiation effect"—once a token stabilizes, it accelerates the convergence of neighboring tokens. Early unlocking releases this radiation effect, which not only accelerates inference but also enhances generation quality.

### Loss & Training
AHD is a training-free plug-and-play method applied directly during the inference stage. Default hyperparameters: historical buffer length $H=6$, consistency threshold $\varepsilon=0.01$, and decay rate $\lambda$ controlling the weight distribution.

## Key Experimental Results

### Main Results (LLaDA-8B-Instruct)

| Task | Metric | AHD | Vanilla | Step Reduction |
|------|------|-----|---------|----------|
| BBH | Score↑ | 56.78 | 53.11 | 80% |
| HumanEval | Score↑ | 43.29 | 40.85 | 70% |
| MBPP | Score↑ | 31.20 | 29.20 | 74% |
| MMLU-Pro | Score↑ | 37.42 | 35.57 | 48% |
| Asdiv | Score↑ | 77.09 | 75.57 | 76% |

### Ablation Study

| Method | BBH Score | Step Reduction | Description |
|------|-----------|----------|------|
| Vanilla | 53.11 | 0% | Standard Decoding |
| Fast-dLLM | 53.17 | 78% | Comparable performance but no gain |
| KLASS | 53.03 | 62% | Slight decrease |
| Saber | 52.88 | 66% | Performance drop |
| AHD | 56.78 | 80% | Only method to improve both performance and efficiency |

### Key Findings
- AHD is the only method capable of improving performance while accelerating; other acceleration strategies (Saber, KLASS) often lead to performance degradation.
- Equally effective on LLaDA-1.5, with a BBH improvement of +1.55 and a step reduction of 78%, proving the method's generality.
- Extension to vision-language (MMaDA) and audio-language (DIFFA) domains remains effective, demonstrating cross-modal applicability.

## Highlights & Insights
- **"Radiation Effect of Stable Tokens"**: Discovered that stable tokens appear in clusters; the stabilization of one token accelerates the convergence of its neighbors. This insight is valuable for understanding the decoding dynamics of diffusion LLMs.
- **Counter-intuitive "Acceleration as Improvement"**: Early unlocking not only speeds up inference but also improves generation quality by releasing the radiation effect. This challenges the common assumption of a "speed vs. quality trade-off."
- **Generality of Anchor Backtracking**: This history-based trajectory method for determining stability can be transferred to any iterative generation process (e.g., pixel-level early determination in diffusion image generation).

## Limitations & Future Work
- Maintaining a historical buffer increases memory overhead, which may become a bottleneck for ultra-long sequence generation.
- The threshold $\varepsilon$ and buffer length $H$ require tuning for different models or tasks.
- Currently validated primarily on the LLaDA series; applicability to other dLLM architectures (e.g., MDLM) remains to be verified.
- Theoretical analysis assumes monotonic convergence of token stability, which may not hold in extreme cases.

## Related Work & Insights
- **vs Fast-dLLM**: Fast-dLLM uses confidence thresholds for acceleration but results in stagnant performance; AHD achieves a win-win in acceleration and improvement through historical trajectory assessment.
- **vs Saber**: Saber uses a predictor for selective denoising but leads to performance drops; AHD's dynamic anchor approach is more robust.
- **vs PC-sampler**: PC-sampler modifies the sampling process without reducing steps; AHD directly reduces steps by 70-80%.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The derivation chain from three insights to the dynamic anchor method is rigorous and natural; the "acceleration as improvement" discovery is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 language benchmarks + 5 vision + 5 audio, two dLLM models, and 5 baseline comparisons.
- Writing Quality: ⭐⭐⭐⭐⭐ The narrative from observation to insight to method is fluid, with excellent chart design (especially the heatmap analysis).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CreditDecoding: Accelerating Parallel Decoding in Diffusion Large Language Models with Trace Credit](creditdecoding_accelerating_parallel_decoding_in_diffusion_large_language_models.md)
- [\[ACL 2026\] Lizard: An Efficient Linearization Framework for Large Language Models](lizard_an_efficient_linearization_framework_for_large_language_models.md)
- [\[ACL 2026\] Are Large Language Models Economically Viable for Industry Deployment?](are_large_language_models_economically_viable_for_industry_deployment.md)
- [\[ICML 2026\] TEAM: Temporal-Spatial Consistency Guided Expert Activation for MoE Diffusion Language Model Acceleration](../../ICML2026/llm_efficiency/team_temporal-spatial_consistency_guided_expert_activation_for_moe_diffusion_lan.md)
- [\[ACL 2026\] Tandem: Riding Together with Large and Small Language Models for Efficient Reasoning](tandem_riding_together_with_large_and_small_language_models_for_efficient_reason.md)

</div>

<!-- RELATED:END -->
