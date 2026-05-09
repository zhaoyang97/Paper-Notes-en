---
title: >-
  [Paper Note] Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play
description: >-
  [Multimodal VLM] This paper proposes Vision-Zero, the first annotation-free gamified self-play framework for VLMs. By casting visual reasoning as a "Who is the Spy?"-style game and combining it with the Iterative-SPO training algorithm, Vision-Zero achieves scalable self-improvement and surpasses SOTA methods trained on human-annotated data across reasoning, chart understanding, and vision-centric tasks.
tags:
  - Multimodal VLM
date: 2026-05-08
content_hash: 10e21f7932f4c4a3
---

# Vision-Zero: Scalable VLM Self-Improvement via Strategic Gamified Self-Play

- **Conference**: ICLR 2026
- **arXiv**: [2509.25541](https://arxiv.org/abs/2509.25541)
- **Code**: [GitHub](https://github.com/Vision-Zero-AI/Vision-Zero)
- **Area**: Multimodal VLM
- **Keywords**: VLM, Self-Play, Reinforcement Learning, Zero-Shot, Gamification, Self-Improvement

## TL;DR

This paper proposes Vision-Zero, the first annotation-free gamified self-play framework for VLMs. By casting visual reasoning as a "Who is the Spy?"-style game and combining it with the Iterative-SPO training algorithm, Vision-Zero achieves scalable self-improvement and surpasses SOTA methods trained on human-annotated data across reasoning, chart understanding, and vision-centric tasks.

## Background & Motivation

Current VLM training faces two core bottlenecks:

**Data scarcity**: Multimodal annotation is prohibitively expensive (COCO Attributes: $60,480 for 200K objects; Ego4D: >250K annotation hours).

**Knowledge ceiling**: Model capability is bounded by the upper limit of human annotations, preventing the discovery of strategies beyond human experience.

**Self-play** has demonstrated the ability to break knowledge ceilings in domains such as Go (AlphaGo) and esports (OpenAI Five). However, extending self-play to VLMs is non-trivial: the game environment must simultaneously handle visual and language modalities while satisfying requirements of skill alignment, scalable difficulty, diversity, and low data demand.

**Vision-Zero's design philosophy** draws inspiration from the social deduction game "Who is the Spy?": civilians observe a real image while the spy receives a blank input, and the two sides engage in interactive strategic gameplay, enabling the model to autonomously generate its own training data.

## Method

### Game Environment

**Roles**: $n_c$ civilians (observing the real image $I_c$) + 1 spy (receiving a blank image $I_s$).

**Two-stage gameplay**:

**Clue Stage**:
- Each player provides a language clue based on their role and observation.
- The spy must infer the hidden image content solely from civilian clues and disguise its identity.
- Civilians must provide accurate clues while minimizing information leakage to the spy.

**Decision Stage**:
- Civilians analyze all clues along with their own image and vote to identify the spy.
- The spy does not participate in voting.
- Players may respond with "n/a" to indicate uncertainty.

### Annotation-Free, Domain-Agnostic Data Input

Training requires only arbitrary images. Three data types are validated experimentally:
- **CLEVR data**: 2,000 automatically rendered images (4–6 random objects each).
- **Chart data**: 1,000 ChartQA images.
- **Real-world data**: 1,000 ImgEdit images.

### Iterative Self-Play Policy Optimization (Iterative-SPO)

**Clue Stage — Self-Play Optimization**:

**Zero-sum rewards**:

$$r_s^{clue} = -\beta(v_s - \bar{v}_c), \quad r_{c_j}^{clue} = \frac{\beta}{n_c}(v_s - \bar{v}_c) - \lambda(v_{c_j} - \bar{v}_c)$$

The sum of spy and civilian rewards is zero; players receiving more votes obtain lower rewards.

**Role Advantage Estimation (RAE)**: mitigates win-rate imbalance caused by information asymmetry:

$$A_k^{clue} = r_k^{clue} - b_k, \quad b_s = \alpha b_s + (1-\alpha) r_s^{clue}$$

**Clue stage objective**:

$$\mathcal{L}^{clue}(\theta) = -\mathbb{E}\left[\frac{1}{n}\sum_{k \in \mathcal{K}} A_k^{clue} \log \pi_\theta^k(u_k | I_k, h)\right] + \tau_{clue} \cdot D_{KL}(\pi_\theta^k \| \pi_{ref}^k)$$

**Decision Stage — RLVR Optimization**:

**Discrete rewards**: correctly identifying the spy yields +1; answering n/a yields −0.5; incorrect answers yield −1.

**Group-normalized GRPO objective**:

$$\mathcal{L}^{dec}(\theta) = -\mathbb{E}\left[\frac{1}{n_c}\sum_{i=1}^{n_c} A_{c_i}^{dec} \log q_\theta(\hat{s}_{c_i} | H)\right] + \tau_{dec} \cdot D_{KL}(q_\theta \| q_{ref})$$

**Alternating training**: stages are switched via hysteresis thresholds:
- Decision → Clue: when $\bar{acc}_t \geq \tau_{acc}^\uparrow$ and $\bar{na}_t \leq \tau_{na}^\downarrow$ (the spy is too easily identified; clue-stage difficulty is increased).
- Clue → Decision: when $1 - \bar{acc}_t \geq \tau_{err}^\uparrow$ or $\bar{na}_t \geq \tau_{na}^\uparrow$ (the spy is too difficult to identify; decision-stage training is reinforced).

### Advantages

1. **Domain-agnostic**: gameplay exploits inter-image differences without depending on specific image types.
2. **Simultaneous multi-capability enhancement**: reasoning, spatial understanding, visual understanding, and OCR.
3. **Extremely low cost**: no human annotation required; data can be rapidly generated using ChatGPT/NanoBanana.

## Key Experimental Results

### Reasoning and Mathematics Tasks

| Method | MathVista | MathVision | WeMath | MathVerse | LogicVista | Avg |
|------|-----------|------------|--------|-----------|------------|-----|
| Qwen2.5-VL-7B | 68.2 | 25.4 | 36.1 | 49.0 | 47.2 | 41.1 |
| MM-Eureka-7B | 73.0 | 26.9 | 36.2 | 50.3 | 42.9 | 42.9 |
| ViGaL-S+R | 71.9 | 27.5 | 36.9 | 52.4 | 46.5 | 43.0 |
| **VZ (CLEVR)** | 72.2 | **28.4** | **39.2** | **53.2** | **49.8** | **44.3** |
| **VZ (Real)** | **73.1** | 28.5 | **40.1** | 52.1 | 50.8 | **44.5** |

Using only unannotated data, Vision-Zero surpasses all baselines that rely on human-annotated data.

### Chart Understanding and Vision-Centric Tasks

Vision-Zero (Chart) achieves substantial gains on chart benchmarks such as ChartXiV and FunctionQA, with additional improvements on vision-centric tasks including MMVP and BLINK.

### Training Dynamics

- The civilian win rate against the spy increases steadily throughout training.
- Clue length (token count) grows over training, indicating that the model learns to describe and reason more elaborately.
- Iterative-SPO effectively prevents the premature convergence associated with naive self-play.

### Ablation Study

| Ablation | MathVista | MathVision |
|------|-----------|------------|
| Clue stage only | 70.8 | 27.1 |
| Decision stage only | 71.5 | 27.6 |
| Iterative-SPO | **73.1** | **28.5** |

Alternating training substantially outperforms single-stage training.

### Comparison with Gobang

On MathVision, Vision-Zero yields a +3% improvement (over 100 rounds) while Gobang shows no gain, demonstrating the superior generalization of visual reasoning games.

## Highlights & Insights

1. **Zero human involvement**: no human annotation or feedback is required at any stage.
2. **Domain-agnostic inputs**: CLEVR, chart, and natural images are all effective.
3. **Theoretically elegant Iterative-SPO**: alternating self-play and RLVR avoids local equilibria.
4. **Surpassing annotated baselines**: an annotation-free method outperforms SOTA trained on expensive human-labeled data.
5. **Simultaneous multi-capability gains**: improvements span reasoning, chart understanding, and vision-centric tasks.

## Limitations & Future Work

1. The number of roles in each game ($n_c + 1$) is fixed; more complex multi-role settings remain unexplored.
2. Whether the strategic space of the "Who is the Spy?" game sufficiently covers all visual reasoning capabilities is unclear.
3. The spy receives a blank image rather than a visually similar one, deviating from the original game design.
4. The hysteresis threshold hyperparameters in Iterative-SPO require manual tuning.
5. Gains on certain vision-centric tasks (e.g., RealWorldQA) remain limited.

## Related Work & Insights

- **LLM self-play**: SPIRAL (Liu et al., 2025) enhances reasoning via board games; Absolute Zero (Zhao et al., 2025) achieves SOTA on mathematics and coding.
- **VLM post-training**: R1-OneVision, MM-Eureka, and VLAA-Thinker employ RLVR with human annotations.
- **Gamified VLM training**: ViGaL (Xie et al., 2025) uses snake and rotation games but requires game data collection.
- **Self-play theory**: AlphaGo (Silver et al., 2017), TD-Gammon (Tesauro, 1995).

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — First annotation-free gamified self-play framework for VLMs.
- **Practicality**: ⭐⭐⭐⭐⭐ — Minimal cost, domain-agnostic, and plug-and-play.
- **Clarity**: ⭐⭐⭐⭐ — Framework is well-structured, though notation-heavy.
- **Significance**: ⭐⭐⭐⭐⭐ — Opens a new paradigm for VLM self-evolution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](../../AAAI2026/multimodal_vlm/plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[ICCV 2025\] SC-Captioner: Improving Image Captioning with Self-Correction by Reinforcement Learning](../../ICCV2025/multimodal_vlm/sc-captioner_improving_image_captioning_with_self-correction_by_reinforcement_le.md)
- [\[ICLR 2026\] VLM-SubtleBench: How Far Are VLMs from Human-Level Subtle Comparative Reasoning?](vlm-subtlebench_how_far_are_vlms_from_human-level_subtle_comparative_reasoning.md)
- [\[ICLR 2026\] WebDS: An End-to-End Benchmark for Web-based Data Science](webds_an_end-to-end_benchmark_for_web-based_data_science.md)
- [\[ICLR 2026\] VisJudge-Bench: Aesthetics and Quality Assessment of Visualizations](visjudge-bench_aesthetics_and_quality_assessment_of_visualizations.md)

</div>

<!-- RELATED:END -->
