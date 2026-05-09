---
title: >-
  [Paper Note] SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models
description: >-
  [Reinforcement Learning] This paper proposes the SPELL framework, in which a single LLM simultaneously assumes three roles—question generator, responder, and verifier—engaging in self-play reinforcement learning without human annotation to continuously improve long-context reasoning, achieving consistent performance gains across 6 long-context benchmarks.
tags:
  - Reinforcement Learning
date: 2026-05-08
content_hash: 0c13a3ba97c62c5e
---

# SPELL: Self-Play Reinforcement Learning for Evolving Long-Context Language Models

## Metadata
- **Conference**: ICLR 2026
- **arXiv**: [2509.23863](https://arxiv.org/abs/2509.23863)
- **Code**: [https://github.com/Tongyi-Zhiwen/Qwen-Doc](https://github.com/Tongyi-Zhiwen/Qwen-Doc)
- **Area**: Reinforcement Learning
- **Keywords**: self-play RL, long-context reasoning, GRPO, LLM, verifiable rewards, curriculum learning

## TL;DR
This paper proposes the SPELL framework, in which a single LLM simultaneously assumes three roles—question generator, responder, and verifier—engaging in self-play reinforcement learning without human annotation to continuously improve long-context reasoning, achieving consistent performance gains across 6 long-context benchmarks.

## Background & Motivation
- **Challenges in long-context reasoning**: RLVR (RL with Verifiable Rewards) has achieved success in domains with clear correctness criteria such as mathematics and code, but long-context reasoning faces two major bottlenecks:
  1. Human annotation is costly and unreliable (human accuracy on LongBench-V2 is only 25.1%).
  2. Programmatically verifiable reward signals are lacking.
- **Challenges of self-play**: Answers may be semantically correct but expressed differently, making string matching and majority voting unreliable.
- **Mechanism**: The model must not only generate and answer questions but also learn to verify its own responses.

## Method

### Overall Architecture: Triangular Self-Play Loop
A single policy $\pi_\theta$ dynamically assumes three roles:

1. **❓ Question Generator $\pi_\theta^{\text{que}}$**: Generates question-answer pairs $(q, a)$ from documents.
2. **📝 Responder $\pi_\theta^{\text{res}}$**: Answers questions given the document, producing $G$ independent samples.
3. **✅ Verifier $\pi_\theta^{\text{ver}}$**: Judges whether the responder's output is semantically equivalent to the reference answer.

### Key Design 1: Automatic Curriculum Learning
- **History memory $\mathcal{H}$**: Stores the most recent $L=3$ solved question-answer pairs and their corresponding documents.
- **Progressive context expansion**: Each iteration expands the document scope, requiring questions to span more documents.
- **Redundancy avoidance**: Solved question-answer pairs are used as conditions to encourage the generator to produce harder questions.

### Key Design 2: Role-Specific Rewards

**Verifier reward** (self-consistency):
$$r_{i,j}^{\text{ver}} = \mathbb{I}(v_{i,j} = v_i^{\text{ver}})$$
Each judgment receives a positive reward if it agrees with the majority vote.

**Responder reward** (rule-based + verifier fusion):
$$r_i^{\text{res}} = \max(\mathcal{R}_{\text{rule}}(y_i, a), v_i^{\text{ver}})$$
The maximum of rule-based matching (CEM) and verifier voting is taken to mitigate false-negative noise.

**Question generator reward** (Gaussian difficulty control):
$$r^{\text{que}} = \exp\left(-\frac{(\bar{r}^{\text{res}} - 0.5)^2}{2 \cdot (0.5/3)^2}\right)$$
A Gaussian function centered at a responder success rate of 0.5 ensures that questions that are too easy or too hard receive no reward. Questions with formatting errors or those not grounded in the document are penalized.

### Key Design 3: Dynamic Role Sampling and Unified Update
- Raw samples are severely imbalanced: 1 generator sample vs. $G$ responder samples vs. $G^2$ verifier samples.
- Responder: retains groups with reward variance $> 0$; Generator: equal positive/negative sampling; Verifier: sub-sampled to those consistent with majority vote.
- The final training set is reduced to approximately $1/G$ of the original size.

**Unified policy update** (GRPO):

$$\mathcal{J}_{\text{GRPO}}(\theta) = \mathcal{J}_{\text{GRPO}}^{\text{que}}(\theta) + \mathcal{J}_{\text{GRPO}}^{\text{res}}(\theta) + \mathcal{J}_{\text{GRPO}}^{\text{ver}}(\theta)$$

All three roles share a single policy and are jointly optimized before proceeding to the next round of self-play.

## Key Experimental Results

### Main Results: 12 Open-Source LLMs × 6 Benchmarks (16K Context)

| Model | Base Avg. | +RLVR Avg. | +SPELL Avg. | SPELL Gain |
|------|----------|-----------|------------|----------|
| Qwen2.5-7B (base) | 26.7 | 40.8 | **40.6** | +13.9 |
| Qwen2.5-14B (base) | 37.3 | — | **51.7** | +14.4 |
| Qwen2.5-32B (base) | — | — | — | +9.1 |
| Qwen2.5-7B-Instruct | — | — | — | +9.0 |
| R1-Distill-Qwen-14B | — | — | — | +3.4 |
| Qwen3-30B-A3B-Thinking | — | — | — | +2.0 |

> Base models trained with SPELL even surpass same-scale instruct models (which require large amounts of human-annotated data).

### Ablation Study: Contribution of Each Component

| Ablation Setting | DocMath | Frames | LB-MQA | LB-V2 | Avg. |
|---------|--------|--------|--------|-------|------|
| SPELL (full) | Best | Best | Best | Best | Best |
| w/o Verifier (CEM only) | Drop | Drop | Drop | — | Drop |
| w/o Curriculum Learning | Drop | Drop | Drop | — | Drop |
| w/o Generator Difficulty Reward | Drop | Drop | — | — | Moderate drop |
| Static RLVR (DeepSeek-R1 data) | Weaker than SPELL | — | — | — | Weaker than SPELL |

### Key Findings
1. SPELL yields consistent improvements across base, instruct, and reasoning model types, as well as both dense and MoE architectures.
2. Training base models with SPELL surpasses instruct models, indicating that self-play is more sample-efficient than human annotation.
3. SPELL's dynamic curriculum outperforms static RLVR using DeepSeek-R1-synthesized data.
4. Self-consistency training of the verifier provides reliable semantic rewards, compensating for the limitations of rule-based matching.
5. Qwen3-30B-A3B-Thinking surpasses Gemini-2.5-pro on pass@4.

## Highlights & Insights
- **Fully unsupervised**: Requires no human annotation or data generated by external models.
- **Unified three-role policy**: A single model simultaneously learns to generate, answer, and verify, yielding an elegant formulation.
- **Automatic curriculum**: Questions naturally become harder and contexts naturally become longer as training progresses.
- **Scalability**: Performance gains are observed even on strong reasoning models (Qwen3-30B Thinking), suggesting the approach has not yet reached a ceiling.

## Limitations & Future Work
- Self-consistency training of the verifier may be inaccurate in certain cases (e.g., semantically ambiguous answers).
- The maximum input length is limited to 16K tokens, which may be restrictive for very long documents (>100K tokens).
- The question generator may produce hallucinated questions outside the document scope, despite the grounding filter.
- Joint training of three roles incurs higher computational cost than standard SFT or single-role RL.

## Related Work & Insights
- **Long-context RL**: Wan et al. (2025) first extended RLVR to long-context settings.
- **Self-play**: Absolute Zero (Zhao et al., 2025) generates single-turn programming tasks; SPAG (Cheng et al., 2024) applies adversarial taboo games.
- **RLVR**: DeepSeek-R1 (Guo et al., 2025), GRPO (Shao et al., 2024).
- **Long-context models**: LongBench (Bai et al., 2024), Frames (Krishna et al., 2025).

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — The three-role self-play combined with automatic curriculum for long-context RL represents a distinctive approach.
- **Theoretical Depth**: ⭐⭐⭐ — Primarily a methodological contribution; theoretical analysis is limited.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 12 models × 6 benchmarks × detailed ablations.
- **Value**: ⭐⭐⭐⭐⭐ — Improves long-context capabilities without any annotated data, offering strong practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SPIRAL: Self-Play on Zero-Sum Games Incentivizes Reasoning via Multi-Agent Multi-Turn Reinforcement Learning](spiral_self-play_on_zero-sum_games_incentivizes_reasoning_via_multi-agent_multi-.md)
- [\[ICLR 2026\] Solving Parameter-Robust Avoid Problems with Unknown Feasibility using Reinforcement Learning](solving_parameter-robust_avoid_problems_with_unknown_feasibility_using_reinforce.md)
- [\[ICLR 2026\] Shop-R1: Rewarding LLMs to Simulate Human Behavior in Online Shopping via Reinforcement Learning](shop-r1_rewarding_llms_to_simulate_human_behavior_in_online_shopping_via_reinfor.md)
- [\[ICLR 2026\] Sample-efficient and Scalable Exploration in Continuous-Time RL](sample-efficient_and_scalable_exploration_in_continuous-time_rl.md)
- [\[ICLR 2026\] Virne: A Comprehensive Benchmark for RL-based Network Resource Allocation in NFV](virne_a_comprehensive_benchmark_for_rl-based_network_resource_allocation_in_nfv.md)

</div>

<!-- RELATED:END -->
