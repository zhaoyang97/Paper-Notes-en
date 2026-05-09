---
title: >-
  [Paper Note] Reinforce to Learn, Elect to Reason: A Dual Paradigm for Video Reasoning
description: >-
  [CVPR 2026][Reinforcement Learning][Video Reasoning] This paper proposes RLER, a dual-paradigm framework in which the training stage employs GRPO with three novel rewards (Frame-sensitive, Think-transparency, Anti-repetition) to teach the model to generate structured evidence, while the inference stage uses a training-free orchestrator to perform evidence-consistency-based weighted election and self-checking across multiple candidates. RLER comprehensively outperforms open-source and RL-based LMMs on 8 video benchmarks with an average gain of 6.3%, requiring only approximately 3.1 candidates on average.
tags:
  - CVPR 2026
  - Reinforcement Learning
  - Video Reasoning
  - Evidence-Driven
  - Multi-Candidate Election
  - Test-Time Reasoning
date: 2026-05-08
content_hash: f86a37c78f0ed8f9
---

# Reinforce to Learn, Elect to Reason: A Dual Paradigm for Video Reasoning

**Conference**: CVPR 2026
**arXiv**: [2604.04379](https://arxiv.org/abs/2604.04379)
**Code**: N/A
**Area**: Video Understanding / Multimodal Reasoning / Reinforcement Learning
**Keywords**: Video Reasoning, Reinforcement Learning, Evidence-Driven, Multi-Candidate Election, Test-Time Reasoning

## TL;DR

This paper proposes RLER, a dual-paradigm framework in which the training stage employs GRPO with three novel rewards (Frame-sensitive, Think-transparency, Anti-repetition) to teach the model to generate structured evidence, while the inference stage uses a training-free orchestrator to perform evidence-consistency-based weighted election and self-checking across multiple candidates. RLER comprehensively outperforms open-source and RL-based LMMs on 8 video benchmarks with an average gain of 6.3%, requiring only approximately 3.1 candidates on average.

## Background & Motivation

1. **Background**: Large multimodal models (LMMs) have achieved remarkable progress in video understanding, yet reasoning remains a "single-pass" process — answers are generated without verifying whether the underlying reasoning is grounded in valid evidence. Even SOTA models are susceptible to minor perturbations (video changes or paraphrase variations).
2. **Limitations of Prior Work**:
    - Existing RL-based training methods (e.g., Video-R1, VideoChat-R1) improve reasoning capability but rarely cross-check evidence consistency across multiple reasoning trajectories.
    - Test-time scaling methods (best-of-N, beam search) provide diverse candidates but lack systematic evidence-based arbitration.
    - Chain-of-thought (CoT) reasoning is seldom systematically grounded against key frames and relations.
3. **Key Challenge**: Existing methods can demonstrate that a model "can reason," but cannot verify that it "reasoned with correct evidence."
4. **Goal**: To shift video reasoning from "answer-driven" to "evidence-driven" — training the model to emit structured, machine-checkable evidence signals, and at inference time electing reliable answers via evidence consistency.
5. **Key Insight**: Decouple "being able to think" from "thinking correctly" — training is responsible for shaping and enhancing reasoning capability (potentiation), while inference is responsible for guaranteeing reliability through evidence-based election.
6. **Core Idea**: During training, reward shaping encourages the model to generate outputs containing key-frame references and structured reasoning; during inference, evidence-weighted election selects the most reliable answer from multiple candidates.

## Method

### Overall Architecture

RLER consists of two symmetric stages. **RLER-Training** is built on GRPO and uses five reward signals to shape evidence-centric structured outputs (with `<think>`, `<answer>`, and `<keyframes>` tags). **RLER-Inference** generates a small number of diverse candidates via varied inputs, parses their structure, computes evidence scores, performs weighted election with outlier removal, applies an adaptive budget, and concludes with a referee self-check, forming a complete evidence-driven decision loop. The backbone model is Qwen2.5-VL-7B-Instruct; only the language model parameters are updated (LoRA r=8).

### Key Designs

1. **Three Novel Training Rewards**:

    - **Function**: To teach the model to generate structured, machine-parsable evidence signals.
    - **Mechanism**:
        - **Frame-sensitive Reward**: Encourages the model to cite specific key frames. The valid-frame score is defined as $s_{fs}(o) = \text{clip}(\frac{|K(o)| - E(o)}{1 + |K(o)|}, 0, 1)$, where $K(o)$ is the set of valid frame indices and $E(o)$ counts invalid indices. The reward is gated by answer correctness and format validity.
        - **Think-transparency Reward**: Encourages reasoning chains of moderate length — neither too short nor too verbose. A unimodal curve $\sin^2(\pi \tilde{L}(o))$ assigns maximum reward at intermediate lengths.
        - **Anti-repetition Reward**: Suppresses low-information-density repetition via n-gram deduplication. $R_{ar}(o) = -\rho(o)$, where $\rho$ denotes the repetition rate.
    - **Design Motivation**: The three rewards precisely correspond to three core requirements of video reasoning — "where to look," "how much to say," and "how to say it" — and the training signals directly align with the scoring dimensions used at inference time.

2. **Evidence-Aligned Reasoning Orchestrator (RLER-Inference)**:

    - **Function**: To improve answer reliability and interpretability without scaling up the model.
    - **Mechanism**: (1) Diverse candidates are generated via a temperature grid {0.2, 0.7, 0.9}, five-crop augmentation, and brightness perturbation; (2) Each candidate is parsed into $(a_i, K_i, z_i, c_i)$ (answer, key frames, reasoning, confidence); (3) A composite evidence score is computed as $S_i = \frac{1}{4}(s_{fs}(o_i) + \tau(o_i) + (1-\rho(o_i)) + c_i)$; (4) Evidence-weighted election computes the evidence intersection (Jaccard overlap) among supporters of each answer, removes outliers, and performs weighted voting; (5) Early stopping is triggered when the leading margin exceeds threshold $\delta=0.08$ and average confidence exceeds $\gamma=0.4$; (6) Referee self-check: a one-shot evidence-sufficiency verification that triggers re-weighting if necessary.
    - **Design Motivation**: Single-pass inference cannot guarantee evidence quality; however, structured outputs combined with evidence-consistency election substantially improve reliability at low computational cost (averaging only 3.1 candidates per question).

3. **Training–Inference Symmetric Design**:

    - **Function**: To establish a symmetric closed loop between training rewards and inference scoring.
    - **Mechanism**: The Frame-sensitive Reward, Think-transparency Reward, and Anti-repetition Reward from the training stage correspond directly to the $s_{fs}$, $\tau$, and $1-\rho$ scores at inference. The capabilities shaped during training are precisely the dimensions evaluated during inference.
    - **Design Motivation**: To prevent a training–inference disconnect and ensure that capabilities learned during training (key-frame citation, moderate reasoning length, repetition avoidance) are directly leveraged at inference time.

### Loss & Training

- Based on GRPO with group size G=4, clip $\epsilon=0.2$, KL coefficient $\beta=0.04$
- Reward weights: $w_{acc}=0.1, w_{fmt}=0.1, w_{fs}=0.2, w_{tt}=0.3, w_{ar}=0.3$
- LoRA (r=8, α=16), AdamW (lr=1e-5), trained for 2 epochs on Video-R1-260k
- Vision encoder and projection layers are frozen; only language model parameters are updated
- 16-frame uniform sampling during training, 32-frame standard inference, 1fps sub-sampling for long videos

## Key Experimental Results

### Main Results

| Benchmark | Type | Qwen2.5-VL-7B | Video-R1 | VideoChat-R1.5 | **RLER** |
|-----------|------|---------------|----------|----------------|----------|
| VSIBench | Video Reasoning | 37.4 | 35.8 | - | **43.3** |
| VideoMMMU | Video Reasoning | 47.4 | 52.3 | 51.4 | **54.2** |
| VideoMME | General | 65.1 | 59.3 | 67.1 | **68.5** |
| TempCompass | General | 69.2 | 73.2 | - | **76.2** |
| MVBench | General | 67.5 | 63.9 | 70.6 | **72.9** |
| LVBench | Long Video | 42.0 | - | 48.4 | **50.7** |
| LongVideoBench | Long Video | 56.0 | - | - | **63.0** |

### Ablation Study

**Training Reward Ablation**

| Configuration | VSIBench | VideoMMMU |
|---------------|----------|-----------|
| Full RLER | **43.3** | **54.2** |
| w/o frame-sensitive | 41.0 (-2.3) | 52.1 (-2.1) |
| w/o think-transparency | 41.9 (-1.4) | 52.7 (-1.5) |
| w/o anti-repetition | 42.1 (-1.2) | 53.0 (-1.2) |
| w/o RLER-Inference | 41.7 (-1.6) | 52.5 (-1.7) |
| w/o GRPO (SFT) | 39.2 (-4.1) | 49.8 (-4.4) |

**Inference Component Ablation (MVBench / LVBench)**

| Configuration | MVBench | LVBench | Avg K |
|---------------|---------|---------|-------|
| Full RLER | **72.9** | **50.7** | 3.1 |
| w/o diversity input | 69.1 | 46.5 | 1.0 |

### Key Findings

- **Frame-sensitive Reward contributes most to reasoning-intensive tasks**: a 2.3% drop is observed on VSIBench, which requires precise spatial reasoning and cross-frame association.
- **The gap between GRPO and SFT is substantial**: SFT learns formatting but lacks the deep reasoning capability elicited by reward signals, resulting in a 4.1–4.4% deficit.
- **Training alone (w/o RLER-Inference) is already highly competitive**: scores of 41.7/52.5 already surpass most RL-based LMMs, demonstrating that reward shaping alone enhances reasoning ability.
- **Only 3.1 candidates per question on average** suffice to achieve full performance gains (upper bound K=8), owing to the efficient early-stopping mechanism.
- An **"Aha moment"** emerges during training — the model spontaneously identifies flaws in its own reasoning and initiates self-correction (e.g., "Wait. Let me re-evaluate.").

## Highlights & Insights

- The **evidence-driven training–inference closed loop** is the central innovation: the same set of dimensions (frame citation, transparency, information density) serves as reward signals during training and as scoring criteria during inference — an elegant and principled design.
- **Surpassing GPT-4o on VideoMME (68.5 vs. 67.9)** is a notable result, demonstrating that a 7B open-source model can outperform a closed-source large model.
- **Evidence-weighted election vs. simple majority voting**: ablations show a significant performance drop when evidence weights are removed, confirming that evidence-based election is more effective than naive voting.
- **The emergence of "Aha moments"** suggests that RL training not only improves output quality but also changes the model's reasoning behavior patterns — a phenomenon that warrants deeper investigation.

## Limitations & Future Work

- Multi-candidate generation at inference time increases computational cost (although averaging 3.1 candidates is already efficient, it remains a concern for latency-sensitive scenarios).
- Validation is conducted only on Qwen2.5-VL-7B; larger models and alternative LMM architectures have not been tested.
- The referee self-check is performed only once; iterative verification could be explored.
- Key-frame citation is currently weakly supervised (at the reward level) without using frame-level annotations.
- The early-stopping thresholds $\delta=0.08$ and $\gamma=0.4$ are shared across all benchmarks; adaptive adjustment for datasets of varying difficulty may yield further improvements.

## Related Work & Insights

- **vs. Video-R1**: Video-R1 introduces RL training for video but relies on single-pass inference. RLER adds finer-grained evidence rewards on the training side and an election mechanism on the inference side, achieving comprehensive improvements (VSIBench: 43.3 vs. 35.8).
- **vs. VideoChat-R1.5**: Both adopt RL training combined with test-time scaling, but RLER's evidence-aligned election is more effective than simple beam search or best-of-N.
- **vs. test-time scaling**: Conventional methods (best-of-N, MCTS) lack video-specific evidence scoring mechanisms; RLER's frame citation and evidence-consistency scoring fill this gap.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The evidence-driven training–inference symmetric closed-loop paradigm is an entirely novel framework; the three rewards are precisely targeted and mutually complementary.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Comprehensive evaluation across 8 benchmarks with separate ablations for training and inference components, supplemented by auxiliary metrics (EGS, TI, RR) for quantifying evidence quality.
- Writing Quality: ⭐⭐⭐⭐ The framework is clearly presented with complete derivations, though the paper is lengthy and requires careful reading.
- Value: ⭐⭐⭐⭐⭐ A systematic solution is proposed for the reliability problem in video reasoning; the practical value of a 7B model surpassing GPT-4o is significant.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] How LLMs Learn to Reason: A Complex Network Perspective](../../ICLR2026/reinforcement_learning/how_llms_learn_to_reason_a_complex_network_perspective.md)
- [\[CVPR 2026\] CCCaption: Dual-Reward Reinforcement Learning for Complete and Correct Image Captioning](cccaption_dual-reward_reinforcement_learning_for_complete_and_correct_image_capt.md)
- [\[ICLR 2026\] ExGRPO: Learning to Reason from Experience](../../ICLR2026/reinforcement_learning/exgrpo_learning_to_reason_from_experience.md)
- [\[NeurIPS 2025\] Modulation of Temporal Decision-Making in a Deep Reinforcement Learning Agent under the Dual-Task Paradigm](../../NeurIPS2025/reinforcement_learning/modulation_of_temporal_decision-making_in_a_deep_reinforcement_learning_agent_un.md)
- [\[ICLR 2026\] Dual Goal Representations](../../ICLR2026/reinforcement_learning/dual_goal_representations.md)

</div>

<!-- RELATED:END -->
