---
title: >-
  [Paper Note] TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards
description: >-
  [ACL 2026][LLM Safety][Multi-turn Jailbreak Attack] This paper models automated multi-turn jailbreak attacks as a multi-turn reinforcement learning problem and proposes TROJail. By utilizing two heuristic process rewards…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Multi-turn Jailbreak Attack"
  - "Trajectory-Level Optimization"
  - "Process Rewards"
  - "Reinforcement Learning"
  - "Red Teaming"
date: 2026-05-08
content_hash: bb10a6b4ccbbe30d
---

# TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards

**Conference**: ACL 2026  
**arXiv**: [2512.07761](https://arxiv.org/abs/2512.07761)  
**Code**: [GitHub](https://github.com/xxiqiao/TROJail)  
**Area**: AI Safety / LLM Reasoning  
**Keywords**: Multi-turn Jailbreak Attack, Trajectory-Level Optimization, Process Rewards, Reinforcement Learning, Red Teaming

## TL;DR

This paper models automated multi-turn jailbreak attacks as a multi-turn reinforcement learning problem and proposes TROJail. By utilizing two heuristic process rewards (over-harm penalization and semantic relevance progression), it alleviates the sparse supervision issue inherent in outcome rewards, significantly improving attack success rates across multiple models and benchmarks.

## Background & Motivation

**Background**: LLMs face safety threats from jailbreak attacks. Multi-turn jailbreak attacks have gained attention as they reflect real-world interaction scenarios. Existing training-based methods use DPO or rejection sampling fine-tuning to optimize the attacker LLM independently at each turn.

**Limitations of Prior Work**: (1) Turn-wise optimization is short-sighted—it maximizes the immediate harmfulness of each response but fails to learn long-term attack strategies across turns; (2) Early prompts that appear harmless but are strategically crucial are undervalued because they do not trigger immediate harmful responses; (3) Tuning-free methods rely on manually designed strategies, require extensive trials, and are prone to failure when the victim model deviates from expectations.

**Key Challenge**: Trajectory-level optimization is a natural solution, but relying solely on the harmfulness of the final response as an outcome reward faces a severe sparse supervision problem—the attacker cannot infer how intermediate prompts contribute to the final attack success.

**Goal**: Design richer intermediate feedback signals to estimate the utility of intermediate prompts, thereby supporting the learning of long-term attack strategies.

**Key Insight**: Controlled experiments reveal two empirical patterns—(1) Moderately harmful intermediate prompts are the most effective; excessively harmful ones trigger refusal mechanisms and lead to failure; (2) Semantic relevance of responses in successful trajectories increases progressively, whereas failed trajectories do not exhibit this pattern.

**Core Idea**: Introduce two process rewards—over-harm penalization $r_{h_1}$ and semantic relevance progression $r_{h_2}$—into a multi-turn GRPO framework. These are integrated into advantage estimation to provide fine-grained training signals for intermediate prompts.

## Method

### Overall Architecture

TROJail is based on multi-turn GRPO. For each harmful prompt $x_0$, the attacker $\pi_\theta$ interacts with the victim model $\pi_\phi$ for up to $T$ turns to generate $G$ trajectories. The outcome reward $r_o$ represents the harmfulness of the final response. Two process rewards $r_{h_1}$ and $r_{h_2}$ evaluate the utility of intermediate prompts. The final advantage $\hat{A}_{i,t} = \hat{A}_{i,t}^o + \lambda \hat{A}_{i,t}^h$ combines outcome and process advantages, optimizing $\pi_\theta$ via a PPO-style clipped objective.

### Key Designs

1.  **Over-Harm Penalization ($r_{h_1}$)**:
    - **Function**: Prevents intermediate prompts from being too harmful and triggering the victim model's refusal mechanism.
    - **Mechanism**: If an intermediate response triggers a refusal, $r_{h_1} = 0$; otherwise, it equals the harmfulness of the direct response $r(x_0, y_t)$. This encourages the attacker to maintain a moderate level of malice—advancing the attack without alerting the model.
    - **Design Motivation**: Controlled experiments show an inverted U-shaped relationship between intermediate prompt harmfulness and final attack success—moderate harmfulness is optimal, while excessive harmfulness leads to a sharp drop in outcome rewards.

2.  **Semantic Relevance Progression ($r_{h_2}$)**:
    - **Function**: Encourages intermediate responses to gradually guide the model toward target harmful content.
    - **Mechanism**: Calculates the cosine similarity between the sentence embeddings of the intermediate response and the original harmful prompt, weighted by the turn ratio: $r_{h_2}(x_t) = \frac{t}{|\tau|} \cdot \text{cosine}(e(x_0), e(y_t))$. Later turns receive higher weights, encouraging steadily increasing semantic alignment.
    - **Design Motivation**: In successful trajectories, semantic relevance increases steadily, while harmfulness rewards surge only in the final turn—semantic relevance provides a more reliable and gradual intermediate feedback signal.

3.  **Process Advantage Estimation and Integration**:
    - **Function**: Integrates process rewards into the advantage estimation of trajectory-level optimization.
    - **Mechanism**: Normalized process advantages $\hat{A}_{i,t}^h = \sum_{s=t}^{|\tau_i|} \frac{r_h(x_{i,s}) - \text{mean}(\mathcal{D}_h)}{\text{std}(\mathcal{D}_h)}$ are calculated for the set of heuristic rewards $\mathcal{D}_h$ across all trajectories and turns, using prefix sums to accumulate future rewards. The final advantage is $\hat{A}_{i,t} = \hat{A}_{i,t}^o + \lambda \hat{A}_{i,t}^h$.
    - **Design Motivation**: Outcome advantages provide global guidance while process advantages provide local guidance—the two are complementary, both optimizing the final objective and providing gradient signals for intermediate steps.

### Loss & Training

A PPO-style clipped objective for multi-turn GRPO is used with KL regularization. The attacker is based on Qwen2.5-3B-Instruct. Victim models include Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B, Mistral-7B, etc.

## Key Experimental Results

### Main Results

**Comparison of Average Attack Success Rate (ASR) across models**

| Method | Type | Average ASR |
|------|------|---------|
| ActorAttack | Tuning-free multi-turn | ~60% |
| HARM | Training-based turn-wise | ~58% |
| Siren (DPO) | Training-based turn-wise | ~65% |
| **TROJail** | Training-based trajectory-level | **~72%** |

### Ablation Study

**Ablation of Process Rewards**

| Configuration | Description |
|------|------|
| w/o Both process rewards | Degenerates to pure MT-GRPO; ASR drops significantly |
| w/o Over-harm penalization | Attacker tends to generate overly aggressive prompts, triggering more refusals |
| w/o Semantic progression | Intermediate turns easily deviate from the target harmful content |

### Key Findings

- TROJail consistently outperforms turn-wise optimization methods across all victim models and benchmarks.
- Both process rewards contribute substantially to performance, but semantic progression is more critical for long trajectories.
- Controlled experiments validate the inverted U-shaped relationship of over-harmfulness—intermediate prompts at L3-L4 levels are most effective.
- Trajectory visualization shows that TROJail learns a "paving first, triggering later" long-term strategy pattern.

## Highlights & Insights

- The discovery of two empirical patterns is the cornerstone of the paper—quantifying intermediate prompt utility through carefully designed controlled experiments.
- Modeling multi-turn jailbreaking as a multi-turn RL problem is natural and elegant, and the design of process rewards is supported by both theory and empirical evidence.
- Although the study focuses on attacks, its findings directly serve defense—only by understanding attack strategies can better safety mechanisms be designed.

## Limitations & Future Work

- Judgment of attack success relies on external harmfulness evaluators, which may themselves be imperfect.
- Evaluated only on victim models in the 7-9B range; larger or newer models were not tested.
- Process rewards are heuristic designs; better intermediate feedback signals may exist.
- Ethical considerations—public disclosure of attack methods could be misused; responsible disclosure is required.

## Related Work & Insights

- **vs Siren/MTSA (DPO Turn-wise Optimization)**: The latter optimizes each turn independently and cannot learn cross-turn strategies; TROJail’s trajectory-level optimization discovers long-term "paving-then-triggering" patterns.
- **vs ActorAttack (Tuning-free Multi-turn)**: The latter relies on preset strategies and easily collapses when the victim model deviates from expectations; TROJail automatically learns strategies via RL.
- **vs MT-GRPO**: Pure outcome rewards face sparse supervision; TROJail’s process rewards provide critical intermediate guidance.

## Rating

- Novelty: ⭐⭐⭐⭐ The approach of modeling multi-turn jailbreaks as multi-turn RL and designing process rewards is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 victim models × 3 benchmarks + controlled experiments + detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear logic from empirical patterns to method design.
- Value: ⭐⭐⭐⭐ Significantly advances LLM safety research with insights for both attack and defense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Multi-component Causal Tracing in Large Language Models](multi-component_causal_tracing_in_large_language_models.md)
- [\[ACL 2026\] SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models](sharedrequest_privacy-preserving_model-agnostic_inference_for_large_language_mod.md)
- [\[ICML 2026\] PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection](../../ICML2026/llm_safety/prpo_paragraph-level_policy_optimization_for_vision-language_deepfake_detection.md)
- [\[ACL 2026\] Instant Personalized Large Language Model Adaptation via Hypernetwork](instant_personalized_large_language_model_adaptation_via_hypernetwork.md)
- [\[ICML 2026\] MultiBreak: A Scalable and Diverse Multi-turn Jailbreak Benchmark for Evaluating LLM Safety](../../ICML2026/llm_safety/multibreak_a_scalable_and_diverse_multi-turn_jailbreak_benchmark_for_evaluating_.md)

</div>

<!-- RELATED:END -->
