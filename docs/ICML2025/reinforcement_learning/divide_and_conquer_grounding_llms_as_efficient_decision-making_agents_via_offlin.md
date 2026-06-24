---
title: >-
  [Paper Note] Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning
description: >-
  [ICML 2025][Reinforcement Learning][Hierarchical Reinforcement Learning] GLIDER introduces a parameter-efficient hierarchical structure where a high-level policy learns abstract step-by-step plans to guide a low-level controller. By decomposing complex long-horizon decision-making into coherent Chain-of-Thought (CoT) reasoning subtasks via offline hierarchical RL, it achieves consistent performance improvements and stronger generalization capabilities on ScienceWorld and ALFW…
tags:
  - "ICML 2025"
  - "Reinforcement Learning"
  - "Hierarchical Reinforcement Learning"
  - "Offline RL"
  - "LLM Decision-Making"
  - "Divide-and-Conquer"
  - "Temporal Abstraction"
date: 2026-05-08
content_hash: f68a623361b2caae
---

# Divide and Conquer: Grounding LLMs as Efficient Decision-Making Agents via Offline Hierarchical Reinforcement Learning

**Conference**: ICML 2025  
**arXiv**: [2505.19761](https://arxiv.org/abs/2505.19761)  
**Code**: [https://github.com/NJU-RL/GLIDER](https://github.com/NJU-RL/GLIDER)  
**Area**: Reinforcement Learning / LLM Agents  
**Keywords**: Hierarchical Reinforcement Learning, Offline RL, LLM Decision-Making, Divide-and-Conquer, Temporal Abstraction

## TL;DR

GLIDER introduces a parameter-efficient hierarchical structure where a high-level policy learns abstract step-by-step plans to guide a low-level controller. By decomposing complex long-horizon decision-making into coherent Chain-of-Thought (CoT) reasoning subtasks via offline hierarchical RL, it achieves consistent performance improvements and stronger generalization capabilities on ScienceWorld and ALFWorld.

## Background & Motivation

**Background**: LLMs exhibit strong reasoning capabilities but perform poorly in long-horizon decision-making tasks. Such tasks require agents to execute a sequence of coherent actions to accomplish complex goals, such as scientific experiments (ScienceWorld) or household chores (ALFWorld).

**Limitations of Prior Work**: LLMs as decision-making agents face two core challenges: (1) **insufficient exploration**: in sparse-reward scenarios, it is hard for LLMs to find the correct action sequence through random trials; (2) **difficult long-term credit assignment**: LLMs struggle to determine which actions in a long sequence contribute most to the final success.

**Key Challenge**: The strength of LLMs lies in high-level reasoning and planning, but direct step-by-step low-level decision-making by LLMs is highly inefficient, as they must grasp the overall task context while simultaneously managing concrete execution details. The challenge is how to leverage LLMs for what they do best—high-level reasoning—while efficiently executing low-level actions.

**Goal**: To design a hierarchical framework where the LLM is solely responsible for high-level planning ("what to do"), and a low-level controller handles physical execution ("how to do it"), with both trained cooperatively via offline RL.

**Key Insight**: Drawing inspiration from the divide-and-conquer strategy—decomposing complex long-horizon tasks into a sequence of subtasks (high-level), with each subtask executed by a specialized controller (low-level). The high-level policy generates abstract plans, and the low-level policy executes specific actions.

**Core Idea**: By introducing a parameter-efficient hierarchical structure into LLM policies, the high-level policy learns to output abstract step-by-step plans (similar to CoT), while the low-level controller is supervised to learn the execution of these plans. This provides flexible temporal abstractions to enhance exploration and learning in long-horizon tasks.

## Method

### Overall Architecture

GLIDER (**G**rounding **L**anguage Models as Eff**I**cient **D**ecision-Making Agents via Offline Hi**E**rarchical **R**einforcement Learning) is a two-layer structure:

```text
Task Description + Environment State
       ↓
  [High-Level Policy]
       ↓ Output: Abstract subgoals/step-by-step plans
  [Low-Level Controller]
       ↓ Output: Concrete action sequences
     Environment Interaction
```

Training consists of three stages: SFT $\to$ Offline RL (ORL) $\to$ Offline-to-Online (O2O) Adaptation.

### Key Designs

1. **High-Level Policy**:

    - **Function**: Receives task descriptions and the current environment state, and outputs abstract step-by-step plans.
    - **Mechanism**: The high-level policy is essentially a CoT reasoning process—decomposing complex tasks into an ordered sequence of subgoal steps. For example, in a "heat water" task, the high-level policy might output: "1. Find a container $\to$ 2. Fill it with water $\to$ 3. Place it on the stove $\to$ 4. Turn on the heat."
    - **Design Motivation**: LLMs naturally excel at high-level reasoning and planning. By keeping the high-level policy focused solely on "planning" rather than "execution," the model can fully leverage the LLM's language comprehension and world knowledge. This separation also significantly reduces the search space.

2. **Low-Level Controller**:

    - **Function**: Receives the subgoals generated by the high-level policy and outputs the corresponding concrete action sequences to accomplish those subgoals.
    - **Mechanism**: The low-level controller is trained to master **task-agnostic** execution skills—learning generic skills like "how to find an object" or "how to manipulate items," rather than solutions to specific tasks.
    - **Design Motivation**: Task-agnostic low-level skills provide two main benefits: (1) high reusability, allowing different tasks to share the same low-level skills; (2) rapid adaptation in non-stationary environments, as the low-level execution logic remains unchanged and only the high-level plans need adjustment.

3. **Offline Hierarchical RL Training (Three Stages)**:

    - **SFT Stage**: Uses expert trajectories from offline datasets to conduct supervised fine-tuning on both high-level and low-level modules, establishing basic planning and execution abilities.
    - **ORL Stage** (Offline RL): Employs offline RL algorithms such as AWAC (Advantage Weighted Actor-Critic) to further optimize the policies. It trains on collected interaction data, enabling the high-level policy to generate better subgoals and the low-level controller to execute more efficiently.
    - **O2O Stage** (Offline-to-Online): Deploys the offline-trained policy to new environments for online adaptation, leveraging the transferability of low-level skills to quickly adapt to non-stationary environments.
    - **Design Motivation**: The three-stage training progresses step-by-step: SFT builds the foundation $\to$ ORL optimizes policies $\to$ O2O adapts to new environments. Offline RL avoids the high computational and safety costs of directly training in the environment from scratch.

4. **Temporal Abstraction**:

    - **Function**: The high-level policy does not need to make decisions at every step; instead, it outputs subgoals at a lower frequency, with each subgoal spanning multiple low-level steps.
    - **Mechanism**: The high-level policy outputs a new subgoal every $k$ steps ($k$ is flexible, depending on the complexity of the subtask), while the low-level controller executes continuously during this interval. This mechanism is similar to the Options Framework.
    - **Design Motivation**: Temporal abstraction significantly reduces the decision-making space of the high-level policy (compressing it from hundreds of steps to just a few subgoals), thereby dramatically improving exploration efficiency and long-term credit assignment.

### Loss & Training

- **SFT Stage**: Standard cross-entropy loss, where the high-level policy learns subgoal generation and the low-level controller learns action execution.
- **ORL Stage**: AWAC-style weighted policy learning using the advantage function to weight "good" transitions within the dataset:

$$\mathcal{L}_{\text{AWAC}} = -\mathbb{E}_{(s,a) \sim \mathcal{D}} \left[ \frac{\exp(A(s,a)/\lambda)}{Z(s)} \log \pi_\theta(a|s) \right]$$

where $A(s,a)$ is the advantage, and $\lambda$ is a temperature parameter.

- **Parameter Efficiency**: Employs parameter-efficient fine-tuning (PEFT) techniques such as LoRA to avoid the high overhead of full-parameter training.

## Key Experimental Results

### Main Results

**ScienceWorld** (Scientific experiment simulation environment):

| Method | Average Score | Gain | Notes |
|------|--------|------|------|
| SayCan (Direct LLM Decision-Making) | Low | - | Non-hierarchical LLM |
| ReAct | Medium | - | Prompt-based |
| Flat RL (No hierarchy) | Medium | - | Standard offline RL |
| **GLIDER** | **Highest** | Consistent gain | Cooperative high/low-level hierarchy |

**ALFWorld** (Household task environment):

| Method | Success Rate | Gain | Notes |
|------|--------|------|------|
| Flat BC (Behavioral Cloning) | Baseline | - | Non-hierarchical |
| Flat RL | Medium | - | Offline RL without hierarchy |
| **GLIDER** | **Highest** | Consistent gain | High-level planning + Low-level execution |

### Ablation Study

| Configuration | Key Metric | Notes |
|------|---------|------|
| GLIDER (Full) | Best | High-level + Low-level + ORL |
| SFT-only (No ORL) | Second Best | Lacks RL optimization |
| Non-hierarchical (Flat) | Weak | Single policy handles everything |
| Fixed High-Level (Frozen) | Weak | No high-level updates limits adaptability |
| No O2O Adaptation | Poor Generalization | Fails to adapt to new environments |
| Removed Temporal Abstraction | Poor Exploration | High-level decisions at every step lead to search space explosion |

### Key Findings

1. **Crucial Role of Hierarchical Structure**: Removing the hierarchical structure (Flat baseline) leads to a significant performance drop, especially in long-horizon tasks, proving that the divide-and-conquer strategy is paramount for LLM decision-making.
2. **Offline RL Superiority over SFT**: The ORL stage further enhances policy quality on top of SFT, indicating that offline RL can learn superior decision-making strategies even from sub-optimal datasets.
3. **Generalization Capability**: GLIDER demonstrates outstanding transfer capabilities in the O2O phase—the task-agnostic low-level skills can be directly reused in new environments, requiring only the fine-tuning of high-level planning.
4. **Benefits of Temporal Abstraction**: Outputting subgoals at a lower frequency via the high-level policy drastically reduces the search space and eases long-term credit assignment difficulties.

## Highlights & Insights

- **Natural Synergy Between Hierarchical Design and LLMs**: LLMs are proficient in high-level reasoning and planning. Confining them to high-level decisions while offloading execution details to a specialized module optimizes computational efficiency. This "division of labor" is far more efficient than forcing LLMs to compute every base-level operation.
- **Chain-of-Thought as Planning**: The sequence of subgoals output by the high-level policy is inherently a Chain-of-Thought (CoT) process—breaking down a complex problem into an ordered set of sub-steps. This bridges the gap between "LLM reasoning" and "RL decision-making."
- **Practicality of Offline Training**: A paradigm fully reliant on offline training + lightweight online adaptation is significantly more practical than pure online RL from scratch, mitigating sample inefficiency challenges.

## Limitations & Future Work

1. Evaluation is restricted to the ScienceWorld and ALFWorld benchmarks; environment complexity remains somewhat bounded.
2. Automatically determining the granularity of high-level subgoals (rather than relying on manual design) remains an open challenge.
3. The impact of offline dataset quality and coverage on GLIDER's performance has not been exhaustively analyzed.
4. A direct comparison with advanced online RL baselines (e.g., PPO) is currently lacking.
5. Validation in more complex real-world robotics or web-browser task environments is an essential next step.

## Related Work & Insights

- **Options Framework / HAM**: Classic hierarchical RL theories, which GLIDER elegantly integrates with LLMs.
- **SayCan / Inner Monologue**: LLM-driven robotic decision-making but without using training via reinforcement learning.
- **ReAct**: A prompt-based reasoning-action framework that GLIDER further optimizes through reinforcement learning.
- **AWAC**: An offline RL algorithm that GLIDER extends to a hierarchical paradigm.
- **Insights**: The key to utilizing LLMs as decision-making agents might not be asking them to "do everything," but rather positioning them at the correct level of abstraction—focusing on planning rather than physical execution.

## Rating
- Novelty: ⭐⭐⭐⭐ Naturally marries hierarchical RL with LLM decision-making, offering intuitive yet substantial design innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Detailed ablations in two benchmark environments; however, evaluations could span more diverse environments.
- Writing Quality: ⭐⭐⭐⭐ Solid methodology exposition and comprehensive motivational arguments.
- Value: ⭐⭐⭐⭐ Offers a coherent, viable, and effective hierarchical RL framework for LLM-as-Agent paradigms.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] transitive rl value learning via divide and conquer](../../ICLR2026/reinforcement_learning/transitive_rl_value_learning_via_divide_and_conquer.md)
- [\[ICML 2025\] Counterfactual Effect Decomposition in Multi-Agent Sequential Decision Making](counterfactual_effect_decomposition_in_multi-agent_sequential_decision_making.md)
- [\[NeurIPS 2025\] Structured Reinforcement Learning for Combinatorial Decision-Making](../../NeurIPS2025/reinforcement_learning/structured_reinforcement_learning_for_combinatorial_decision-making.md)
- [\[ICML 2025\] Hierarchical Reinforcement Learning with Targeted Causal Interventions](hierarchical_reinforcement_learning_with_targeted_causal_interventions.md)
- [\[ICML 2025\] Enhancing Decision-Making of Large Language Models via Actor-Critic](enhancing_decision-making_of_large_language_models_via_actor-critic.md)

</div>

<!-- RELATED:END -->
