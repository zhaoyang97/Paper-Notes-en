---
title: >-
  [Paper Note] Note 5: ReSearch — Learning to Reason with Search
description: >-
  [NeurIPS 2025][Reinforcement Learning][Reasoning-search integration] ReSearch embeds search operations as first-class primitives within reasoning chains and leverages GRPO reinforcement learning to automatically learn when and how to search—without any supervision on intermediate reasoning steps—achieving an average relative improvement of 15.81% over baselines on multi-hop QA benchmarks.
tags:
  - "NeurIPS 2025"
  - "Reinforcement Learning"
  - "Reasoning-search integration"
  - "GRPO"
  - "multi-step retrieval augmentation"
  - "unsupervised reasoning"
date: 2026-05-08
content_hash: 50f70fda0b112df5
---

# Note 5: ReSearch — Learning to Reason with Search

**Conference**: NeurIPS 2025
**arXiv**: [2503.19470](https://arxiv.org/abs/2503.19470)  
**Code**: [GitHub](https://github.com/Agent-RL/ReSearch)  
**Area**: Other
**Keywords**: Reasoning-search integration, GRPO, multi-step retrieval augmentation, unsupervised reasoning

## TL;DR
ReSearch embeds search operations as first-class primitives within reasoning chains and leverages GRPO reinforcement learning to automatically learn when and how to search—without any supervision on intermediate reasoning steps—achieving an average relative improvement of 15.81% over baselines on multi-hop QA benchmarks.

## Background & Motivation
**The multi-step RAG dilemma**: Existing approaches rely on manually designed heuristics or labor-intensive annotations of reasoning steps, making them difficult to scale to complex questions.

**Adaptive limitations of information retrieval**: Intelligent decisions about "when to search and what to search for" typically depend on static rules or imperfect heuristics.

**RL opportunity**: DeepSeek-R1 has demonstrated that pure RL can elicit complex reasoning, yet it remains unclear how to effectively integrate external search in an unsupervised setting.

**Core Problem**: Can an LLM be trained to autonomously interleave search operations within its reasoning process, without annotated reasoning chains?

## Method

### Overall Architecture
**Unified three-component integration**: Reasoning (`<think>`), search (`<search>`), and retrieved results (`<result>`) coexist within a single unified chain:
$$\text{Rollout}: <think>...\text{reasoning}...</think> <search>\text{query}</search> <result>\text{retrieved result}</result> <think>...\text{continued reasoning}...</think> <answer>\text{final answer}</answer>$$

### Key Designs
**1. GRPO Reinforcement Learning**:
Unlike PPO, GRPO avoids an explicit critic network by estimating the baseline from a group of rollouts:
$$\mathcal{J}(\theta) = \mathbb{E}_{x,\{y_i\}^G}\left[\min\left(\frac{\pi_\theta(y_i|x)}{\pi_{old}(y_i|x)}A_i, \text{clip}(...)\right)-\beta\mathbb{D}_{KL}(\pi_\theta||\pi_{ref})\right]$$

where the advantage $A_i = (r_i-\text{mean}(r_j))/\text{std}(r_j)$ is normalized within each group.

**2. Search-Aware Rollout**:
During GRPO sampling, generation halts upon encountering `</search>`, triggering an actual search call:
- The search tool returns results, which are wrapped in `<result>...</result>`.
- Generation resumes for the next reasoning segment, conditioned on the retrieved results.
- This process repeats until EOS.

Strategy: Retrieved result tokens are masked and excluded from backpropagation; only the model-generated search queries and reasoning tokens are optimized.

**3. Simple yet Effective Reward Function**:
$$r = \begin{cases} \text{F1}(a_{pred}, a_{gt}) & \text{if F1}>0\\ 0.1 & \text{if F1}=0 \text{ and format correct}\\ 0 & \text{otherwise} \end{cases}$$

- **Answer reward**: F1 score (tolerant of variability in open-domain retrieval outputs).
- **Format reward**: Verifies the integrity of special tags and $\boxed{}$ formatting.

## Key Experimental Results

### 7B Model — Multi-hop QA Performance

| Model | HotpotQA EM% | HotpotQA LJ% | 2Wiki EM% | MuSiQue EM% | Bamboogle EM% |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Baselines** | | | | | |
| No RAG | 19.18 | 30.64 | 25.76 | 3.76 | 10.40 |
| Naive RAG | 31.90 | 49.59 | 25.78 | 6.21 | 20.80 |
| Iter-RetGen | 34.36 | 52.22 | 27.92 | 8.69 | 21.60 |
| **ReSearch** | | | | | |
| ReSearch-7B | **40.57** | **60.26** | **44.67** | **21.68** | **43.20** |
| Rel. gain vs. best baseline | +18.1% | +15.5% | +59.9% | +149% | **+100%** |

### 32B Model — Generalization

| Dataset | ReSearch-32B EM | ReSearch-32B LJ | Rel. vs. Iter-RetGen |
|:---:|:---:|:---:|:---:|
| HotpotQA | 45.16% | 66.32% | +17.2% |
| 2WikiMultiHop | 51.23% | 58.94% | +68.5% |
| MuSiQue (train set) | 25.39% | 38.22% | +192% |
| Bamboogle (unseen) | 48.31% | 60.58% | +122% |

### Key Findings
1. **Strong generalization**: Trained solely on MuSiQue, the model performs remarkably on three other benchmarks (including unseen Bamboogle at +122%), demonstrating that the acquired reasoning-and-search capability transfers universally.
2. **Exponential gains without annotation**: Absolute improvements of up to 149% (MuSiQue) are achieved without any supervision on intermediate reasoning steps, confirming that reward signals alone suffice to elicit complex multi-step reasoning.
3. **Emergent reflection and self-correction**: Analysis reveals that reflection and self-correction behaviors emerge naturally during RL training without explicit programming.

## Highlights & Insights
1. **Conceptual innovation**: Treating search as a first-class citizen within the reasoning chain breaks the conventional paradigm of keeping RAG and reasoning separate.
2. **Minimal supervision**: Only final-answer rewards and format feedback are required to induce automatic coordination between multi-step reasoning and search decisions.
3. **Generalization strength**: Outstanding cross-dataset transfer indicates that the model learns a general reasoning-plus-search capability rather than task-specific patterns.
4. **Experimental rigor**: Evaluation is conducted in an open-domain retrieval setting (live Wikipedia), mitigating risks of data leakage and overfitting.

## Limitations & Future Work
1. The search loop depends on the quality of Wikipedia content; performance on other knowledge bases remains unknown.
2. Multi-hop reasoning is tested up to approximately 3 steps; behavior under highly complex scenarios (5+ hops) is unclear.
3. Detailed ablations of critical hyperparameters (e.g., search count limits, result truncation length) are absent.

## Related Work & Insights
- Reasoning models and RL scaling (o1 / DeepSeek-R1 / OpenReasonerZero)
- Multi-step RAG reasoning methods (Iter-RetGen, IRCoT) and standard RAG
- Reinforcement learning for sequential decision-making and tool use

## Rating
⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Training Language Models to Reason Efficiently](training_language_models_to_reason_efficiently.md)
- [\[ICLR 2026\] ExGRPO: Learning to Reason from Experience](../../ICLR2026/reinforcement_learning/exgrpo_learning_to_reason_from_experience.md)
- [\[NeurIPS 2025\] Reinforcement Learning for Long-Horizon Multi-Turn Search Agents](reinforcement_learning_for_long-horizon_multi-turn_search_agents.md)
- [\[NeurIPS 2025\] TensorRL-QAS: Reinforcement Learning with Tensor Networks for Improved Quantum Architecture Search](tensorrl-qas_reinforcement_learning_with_tensor_networks_for_improved_quantum_ar.md)
- [\[ICLR 2026\] Learning to Reason Efficiently with Discounted Reinforcement Learning](../../ICLR2026/reinforcement_learning/learning_to_reason_efficiently_with_discounted_reinforcement_learning.md)

</div>

<!-- RELATED:END -->
