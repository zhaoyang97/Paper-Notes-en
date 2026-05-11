---
title: >-
  [Paper Note] TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards
description: >-
  [ACL 2026][LLM Reasoning][Multi-turn jailbreak attacks] This paper frames automated multi-turn jailbreak attacks as a multi-turn reinforcement learning problem and proposes TROJail…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "Multi-turn jailbreak attacks"
  - "trajectory-level optimization"
  - "process rewards"
  - "reinforcement learning"
  - "red-teaming"
date: 2026-05-08
content_hash: 94a7c9b36686d045
---

# TROJail: Trajectory-Level Optimization for Multi-Turn Large Language Model Jailbreaks with Process Rewards

**Conference**: ACL 2026
**arXiv**: [2512.07761](https://arxiv.org/abs/2512.07761)
**Code**: [GitHub](https://github.com/xxiqiao/TROJail)
**Area**: AI Safety / LLM Reasoning
**Keywords**: Multi-turn jailbreak attacks, trajectory-level optimization, process rewards, reinforcement learning, red-teaming

## TL;DR

This paper frames automated multi-turn jailbreak attacks as a multi-turn reinforcement learning problem and proposes TROJail, which introduces two heuristic process rewards—over-harm penalization and semantic relevance progression—to alleviate the sparse supervision problem of outcome-only rewards, achieving substantial improvements in attack success rate across multiple models and benchmarks.

## Background & Motivation

**State of the Field**: LLMs face security threats from jailbreak attacks. Multi-turn jailbreak attacks have attracted increasing attention as they better reflect real-world interaction scenarios. Existing training-based methods use DPO or rejection-sampling fine-tuning to optimize the attacker LLM independently at each turn.

**Limitations of Prior Work**: (1) Turn-by-turn optimization is myopic—maximizing the harmfulness of the direct response at each turn prevents the model from learning long-term attack strategies across turns; (2) early prompts that appear benign but are strategically critical are undervalued because they do not elicit immediately harmful responses; (3) training-free methods rely on manually designed strategies, require extensive trial and error, and tend to collapse when victim models deviate from expected behavior.

**Root Cause**: Trajectory-level optimization is the natural solution, but relying solely on the harmfulness of the final response as an outcome reward introduces severe sparse supervision—the attacker cannot infer how intermediate prompts contribute to the ultimate attack success.

**Paper Goals**: Design richer intermediate feedback signals to estimate the utility of intermediate prompts, thereby enabling the learning of long-term attack strategies.

**Starting Point**: Controlled experiments reveal two empirical patterns: (1) moderately harmful intermediate prompts are most effective, while overly harmful ones trigger refusal mechanisms and ultimately fail; (2) semantic relevance of responses increases progressively along successful trajectories, a pattern absent in failed trajectories.

**Core Idea**: Introduce two process rewards within a multi-turn GRPO framework—over-harm penalization $r_{h_1}$ and semantic relevance progression $r_{h_2}$—and integrate them into advantage estimation to provide fine-grained training signals for intermediate prompts.

## Method

### Overall Architecture

TROJail builds upon multi-turn GRPO. For each harmful prompt $x_0$, the attacker $\pi_\theta$ interacts with the victim model $\pi_\phi$ for up to $T$ turns to generate $G$ trajectories. The outcome reward $r_o$ measures the harmfulness of the final response. Two process rewards $r_{h_1}$ and $r_{h_2}$ evaluate the utility of intermediate prompts. The final advantage $\hat{A}_{i,t} = \hat{A}_{i,t}^o + \lambda \hat{A}_{i,t}^h$ combines outcome and process advantages, and $\pi_\theta$ is optimized via a PPO-style clipped objective.

### Key Designs

1. **Over-Harm Penalization ($r_{h_1}$)**:

    - **Function**: Prevents intermediate prompts from being so harmful that they trigger the victim model's refusal mechanism.
    - **Mechanism**: If the intermediate response triggers a refusal, $r_{h_1} = 0$; otherwise it equals the harmfulness of the direct response $r(x_0, y_t)$. This encourages the attacker to maintain a moderate level of malice—advancing the attack without prematurely alarming the victim.
    - **Design Motivation**: Controlled experiments show an inverted-U relationship between intermediate prompt harmfulness and final attack success—moderate harmfulness is optimal, while excessive harmfulness causes a sharp drop in outcome reward.

2. **Semantic Relevance Progression ($r_{h_2}$)**:

    - **Function**: Encourages intermediate responses to gradually guide the conversation toward the target harmful content.
    - **Mechanism**: Computes the cosine similarity between the sentence embeddings of the intermediate response and the original harmful prompt, weighted proportionally by turn index: $r_{h_2}(x_t) = \frac{t}{|\tau|} \cdot \text{cosine}(e(x_0), e(y_t))$. Later turns receive higher weights, encouraging steadily increasing semantic alignment.
    - **Design Motivation**: Semantic relevance increases smoothly along successful trajectories, whereas the harmfulness reward spikes only at the final turn—semantic relevance thus provides a more reliable and gradual intermediate feedback signal.

3. **Process Advantage Estimation and Integration**:

    - **Function**: Integrates process rewards into advantage estimation for trajectory-level optimization.
    - **Mechanism**: Normalized process advantages $\hat{A}_{i,t}^h = \sum_{s=t}^{|\tau_i|} \frac{r_h(x_{i,s}) - \text{mean}(\mathcal{D}_h)}{\text{std}(\mathcal{D}_h)}$ are computed over the set of heuristic rewards $\mathcal{D}_h$ across all trajectories and turns, accumulating future rewards via prefix sums. The final advantage is $\hat{A}_{i,t} = \hat{A}_{i,t}^o + \lambda \hat{A}_{i,t}^h$.
    - **Design Motivation**: Outcome advantages provide global guidance while process advantages provide local supervision—the two are complementary, jointly optimizing the final objective while supplying gradient signals for intermediate steps.

### Loss & Training

A PPO-style clipped objective with KL regularization under the multi-turn GRPO framework is employed. The attacker is based on Qwen2.5-3B-Instruct. Victim models include Llama-3.1-8B, Qwen2.5-7B, Gemma-2-9B, and Mistral-7B, among others.

## Key Experimental Results

### Main Results

**Cross-Model Average Attack Success Rate (ASR) Comparison**

| Method | Type | Avg. ASR |
|--------|------|----------|
| ActorAttack | Training-free multi-turn | ~60% |
| HARM | Training-based turn-by-turn | ~58% |
| Siren (DPO) | Training-based turn-by-turn | ~65% |
| **TROJail** | Training-based trajectory-level | **~72%** |

### Ablation Study

**Process Reward Ablation**

| Configuration | Description |
|---------------|-------------|
| w/o both process rewards | Degrades to pure MT-GRPO; ASR drops significantly |
| w/o over-harm penalization | Attacker tends to generate overly aggressive prompts, triggering more refusals |
| w/o semantic progression | Intermediate turns easily drift away from the target harmful content |

### Key Findings

- TROJail consistently outperforms turn-by-turn optimization methods across all victim models and benchmarks.
- Both process rewards contribute comparably to performance, though semantic progression becomes more critical in longer trajectories.
- Controlled experiments confirm the inverted-U relationship for over-harm—intermediate prompts at harmfulness levels L3–L4 are most effective.
- Trajectory visualizations reveal that TROJail learns a long-term strategy pattern of "priming before triggering."

## Highlights & Insights

- The discovery of two empirical patterns serves as the cornerstone of the paper—the utility of intermediate prompts is quantified through carefully designed controlled experiments.
- Framing multi-turn jailbreaking as a multi-turn RL problem is a natural and elegant perspective; the process reward design is supported by both theoretical motivation and empirical evidence.
- Although the work focuses on attacks, its findings directly inform defense—understanding attack strategies is a prerequisite for designing better safety mechanisms.

## Limitations & Future Work

- Evaluation of attack success relies on an external harmfulness scorer, which may itself be imperfect.
- Experiments are conducted only on 7–9B-scale victim models; larger or newer models are not evaluated.
- The process rewards are heuristically designed; superior intermediate feedback signals may exist.
- Ethical considerations—public disclosure of attack methods may be misused, necessitating responsible disclosure practices.

## Related Work & Insights

- **vs. Siren/MTSA (DPO turn-by-turn optimization)**: The latter optimizes each turn independently and cannot learn cross-turn strategies; TROJail's trajectory-level optimization discovers long-term patterns of "priming before triggering."
- **vs. ActorAttack (training-free multi-turn)**: The latter relies on predefined strategies and tends to collapse when victim models deviate from expectations; TROJail automatically learns strategies via RL.
- **vs. MT-GRPO**: Pure outcome rewards suffer from sparse supervision; TROJail's process rewards provide essential intermediate guidance.

## Rating

- Novelty: ⭐⭐⭐⭐ — Framing multi-turn jailbreaking as multi-turn RL and designing process rewards is a novel contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — 4 victim models × 3 benchmarks, plus controlled experiments and detailed ablations.
- Writing Quality: ⭐⭐⭐⭐ — The logical flow from empirical patterns to method design is clear and well-structured.
- Value: ⭐⭐⭐⭐ — Makes an important contribution to LLM safety research with implications for both attack and defense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Process Reward Models Meet Planning: Generating Precise and Scalable Datasets for Step-Level Rewards](process_reward_models_meet_planning_generating_precise_and_scalable_datasets_for.md)
- [\[ACL 2026\] Dissecting Failure Dynamics in Large Language Model Reasoning](dissecting_failure_dynamics_in_large_language_model_reasoning.md)
- [\[ACL 2026\] Challenging the Boundaries of Reasoning: An Olympiad-Level Math Benchmark for Large Language Models](challenging_the_boundaries_of_reasoning_an_olympiad-level_math_benchmark_for_lar.md)
- [\[ACL 2026\] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization](cipo_counterfactual_unlearning_for_large_reasoning_models_through_iterative_pref.md)
- [\[ACL 2026\] JTPRO: A Joint Tool-Prompt Reflective Optimization Framework for Language Agents](jtpro_a_joint_tool-prompt_reflective_optimization_framework_for_language_agents.md)

</div>

<!-- RELATED:END -->
