---
title: >-
  [Paper Note] Adaptive Theory of Mind for LLM-based Multi-Agent Coordination
description: >-
  [AAAI 2026][Multi-Agent][Theory of Mind] This paper proposes the Adaptive Theory of Mind agent (A-ToM), which formulates ToM order alignment as an online expert advice problem. By employing Follow-the-Leader (FTL) or Hed…
tags:
  - "AAAI 2026"
  - "Multi-Agent"
  - "Theory of Mind"
  - "LLM Multi-Agent"
  - "Zero-Shot Coordination"
  - "Online Learning"
  - "Expert Advice"
date: 2026-05-08
content_hash: 2f0c8f3815cc5227
---

# Adaptive Theory of Mind for LLM-based Multi-Agent Coordination

**Conference**: AAAI 2026
**arXiv**: [2603.16264](https://arxiv.org/abs/2603.16264)  
**Code**: [https://github.com/ChunjiangMonkey/Adaptive-ToM](https://github.com/ChunjiangMonkey/Adaptive-ToM)  
**Area**: Robotics
**Keywords**: Theory of Mind, LLM Multi-Agent, Zero-Shot Coordination, Online Learning, Expert Advice

## TL;DR

This paper proposes the Adaptive Theory of Mind agent (A-ToM), which formulates ToM order alignment as an online expert advice problem. By employing Follow-the-Leader (FTL) or Hedge algorithms to estimate a partner's ToM order in real time and dynamically adjust its own reasoning depth, A-ToM achieves robust zero-shot multi-agent coordination across four task categories, including repeated matrix games, grid navigation, and Overcooked.

## Background & Motivation

**Zero-shot coordination is a core challenge in multi-agent systems**: In domains such as autonomous driving and swarm robotics, agents must collaborate with unseen partners without the opportunity for joint training or prior communication.

**LLMs are naturally suited for zero-shot coordination**: Large language models possess strong decision-making and generalization capabilities, enabling deployment without task-specific training, and have been widely adopted for building zero-shot cooperative agents.

**ToM is introduced to model the behavior of others**: Theory of Mind (ToM) enables agents to reason about the beliefs, intentions, and desires of others. Explicit ToM modules have become a key component in multi-agent LLM architectures.

**Higher-order ToM is not always beneficial**: Prior work has observed that increasing the ToM order can degrade cooperative performance, attributing this to insufficient high-order reasoning capacity in LLMs or over-reasoning.

**This paper identifies a deeper cause — ToM order misalignment**: A ToM-$k$ agent assumes its partner operates at order $(k{-}1)$. When the two agents' ToM orders are mismatched, this leads to under- or over-reasoning, severely impairing coordination (e.g., two vehicles swerving to the same side when approaching each other).

**An adaptive ToM mechanism is therefore necessary**: Rather than relying on a fixed ToM order, agents should estimate their partner's ToM order in real time and align with it.

## Method

### Overall Architecture

The A-ToM agent maintains a set of hypothesis agents at different ToM orders $\{\pi_j^{(k)}\}_{k \in \{0,1,2\}}$. During interaction, an online learning algorithm (FTL or Hedge) dynamically selects the best prediction based on the historical prediction accuracy of each hypothesis agent, thereby estimating the partner's ToM order and selecting a coordinated action. The core pipeline is: (1) each hypothesis agent generates a candidate action; (2) one predicted action is selected based on historical accuracy; (3) a response action coordinated with the predicted action is chosen; (4) the partner's true action is observed and the prediction accuracy of each hypothesis agent is updated.

### Key Designs

**Design 1: Recursive Modeling of ToM Orders**

- **Function**: Defines the recursive decision process for ToM-0 through ToM-$k$ agents. ToM-0 acts solely based on the environment state; ToM-$k$ ($k>0$) first uses a $(k{-}1)$-order hypothesis agent to predict the partner's action, then selects the best response.
- **Mechanism**: $\pi_i^{(k)}(s, b_i^{(k)}) = \arg\max_{a \in \mathcal{A}_i} Q_{\boldsymbol{\pi}}(s, a_j^{\text{pred}}, a)$, where $b_i^{(k)} = a_j^{\text{pred}} = \pi_j^{(k-1)}(s, b_j^{(k-1)})$.
- **Design Motivation**: The recursive structure of ToM implies that a ToM-$k$ agent naturally aligns best with a ToM-$(k{\pm}1)$ partner. This "alignment" relationship is key to successful coordination. The order is capped at $k \leq 2$, as empirical cognitive science suggests humans typically reason only up to second-order ToM.

**Design 2: Framing ToM Alignment as an Online Expert Advice Problem**

- **Function**: Each hypothesis agent corresponding to a ToM order is treated as an "expert," and an online learning algorithm selects the best expert.
- **Mechanism**: The FTL algorithm selects the expert with the minimum cumulative loss ($\hat{k} = \arg\min_{k} L^{(k)}$), with a regret bound of $\mathcal{O}(\log T)$; the Hedge algorithm maintains a soft weight distribution ($w^{(k)} \leftarrow w^{(k)} \cdot \exp(-\eta \cdot \ell^{(k)})$), with a regret bound of $\mathcal{O}(\sqrt{T \log N})$.
- **Design Motivation**: The coordination problem in policy space is reduced to the ToM order space, leveraging LLMs' strength in abstract reasoning rather than requiring them to handle low-level fine-grained details. FTL suits stable settings with a fixed partner ToM order; Hedge's exploration capacity makes it more appropriate for non-stationary settings (e.g., A-ToM self-play).

**Design 3: Four-Module LLM Agent Architecture**

- **Function**: Each agent consists of a state encoding module, a ToM module, a decision module, and an action controller.
- **Mechanism**: The state encoder converts structured environment states into natural language; the ToM module recursively constructs hypothesis agents to predict partner behavior; the decision module integrates state descriptions and predicted actions to output a decision; the action controller maps natural language actions to executable commands.
- **Design Motivation**: Following the two-stage paradigm of "first predict partner behavior, then decide accordingly," the architecture fully exploits LLM reasoning capabilities while omitting output verification components to reduce inference latency.

## Key Experimental Results

Experiments use LLaMA-3.3-70B-Instruct (temperature=0.1), with 30 independent repetitions per configuration; means are reported.

**Table 1: Cooperative Performance of Fixed ToM Order Agents**

| Alignment State | Agent Pairing | Memory-1 (Point↑) | Memory-N (Point↑) | Game 1 (Time↓) | Game 2 (Time↓) | Overcooked (Time↓) |
|---|---|---|---|---|---|---|
| Misaligned | ToM-0 vs ToM-0 | 0.00 | 11.67 | 30.00 | 30.00 | 100.00 |
| Misaligned | ToM-1 vs ToM-1 | 0.00 | 0.00 | 23.37 | 30.00 | 96.40 |
| Misaligned | ToM-2 vs ToM-2 | 0.00 | 12.33 | 29.67 | 30.00 | 83.50 |
| **Aligned** | **ToM-0 vs ToM-1** | **75.00** | **75.00** | **6.00** | **8.13** | **44.17** |
| **Aligned** | **ToM-1 vs ToM-2** | **75.00** | **75.00** | **6.10** | **7.10** | **48.90** |

**Table 2: Cooperative Performance of A-ToM Agents**

| Algorithm | Agent Pairing | Memory-1 (Point↑) | Memory-N (Point↑) | Game 1 (Time↓) | Game 2 (Time↓) | Overcooked (Time↓) |
|---|---|---|---|---|---|---|
| FTL | A-ToM vs ToM-0 | 75.00 | 75.00 | 6.03 | 7.00 | 45.33 |
| FTL | A-ToM vs ToM-1 | 70.00 | 70.00 | 7.80 | 10.53 | 52.17 |
| FTL | A-ToM vs ToM-2 | 75.00 | 75.00 | 6.03 | 7.17 | 45.30 |
| FTL | A-ToM vs A-ToM | 0.00 | 0.00 | 20.90 | 27.23 | 51.17 |
| Hedge | A-ToM vs ToM-0 | 72.83 | 72.50 | 6.00 | 8.07 | 47.47 |
| Hedge | A-ToM vs ToM-1 | 70.00 | 70.17 | 7.87 | 10.07 | 57.60 |
| Hedge | A-ToM vs ToM-2 | 73.00 | 72.50 | 6.27 | 8.17 | 46.00 |
| Hedge | A-ToM vs A-ToM | 68.17 | 64.33 | 7.60 | 8.57 | 50.53 |

## Key Findings

1. **ToM alignment is the critical factor for successful cooperation**: Same-order ToM self-play (e.g., ToM-0 vs ToM-0) fails almost completely across all tasks, whereas adjacent-order alignment (e.g., ToM-0 vs ToM-1) achieves optimal performance.
2. **A-ToM robustly aligns with diverse partners**: Regardless of whether the partner operates at ToM-0, 1, or 2, A-ToM achieves performance close to that of the optimal aligned pairing.
3. **FTL and Hedge each have their advantages**: FTL performs slightly better in settings with a fixed partner ToM order (faster convergence), while Hedge is substantially stronger in A-ToM self-play (better exploration; 68.17 vs 0.00 in Memory-1).
4. **A-ToM generalizes to non-LLM partners**: When cooperating with planning- or RL-based agents such as Greedy and PBT, A-ToM tends to interpret them as ToM-0 and outperforms all fixed-order ToM agents.
5. **The importance of ToM alignment is modulated by two factors**: The smaller the optimal action space (2 actions > 3 actions) and the more rational the agent (lower temperature), the more harmful ToM misalignment becomes, and the greater the value of A-ToM.

## Highlights & Insights

- **Novel insight**: This work is the first to explicitly identify ToM order misalignment as the underlying cause of LLM multi-agent cooperation failures, moving beyond the simplistic explanation of "insufficient high-order ToM capability."
- **Elegant methodology**: The coordination problem in policy space is reduced to a three-element ToM order space, leveraging mature online learning theory (the expert advice framework) to obtain theoretical guarantees.
- **Comprehensive experiments**: Four structurally distinct cooperative task categories (matrix games, grid navigation, cooking) are covered, with additional analysis of non-LLM partners and boundary conditions of ToM alignment.
- **Open source**: Complete code and reproducible experiments based on LLaMA-70B are provided.

## Limitations & Future Work

1. **Only ToM orders up to 2 are considered**: Although grounded in cognitive science, the adaptive alignment effects of higher-order ToM remain unexplored.
2. **FTL self-play fails completely**: When both agents are A-ToM (FTL), coordination degrades severely, indicating that algorithm selection places strong constraints on practical deployment scenarios.
3. **Limited task scale**: All experiments involve two-agent settings; the complexity of ToM order alignment with more than two agents has not been validated.
4. **Dependence on LLM reasoning quality**: The recursive reasoning in the ToM module relies entirely on the instruction-following capability of the underlying LLM; different models may yield varying ToM estimation quality.
5. **Performance gap in Overcooked**: In complex tasks such as Overcooked, the gap between A-ToM and the optimal aligned pairing is larger, suggesting higher noise in ToM order estimation under large action spaces.

## Related Work & Insights

- **LLM agents**: LLM-driven decision-making agents have demonstrated strong capabilities in robotic control, GUI operation, and open-world games; multiple LLM agents can also collaborate on large-scale tasks such as software development and social simulation.
- **ToM-augmented multi-agent coordination**: Explicitly equipping AI agents with ToM enables inference of others' hidden states, improves communication efficiency, and enhances coordination. However, higher-order ToM is not always effective; this paper further identifies the cause as ToM order misalignment.
- **Online learning and expert advice**: FTL and Hedge are classical algorithms for the expert advice problem, suited to stationary and non-stationary settings respectively. This paper is the first to apply them to ToM order estimation.

## Rating

- **Novelty**: ⭐⭐⭐⭐ First to formalize the ToM misalignment problem and propose an adaptive solution; the insight is valuable, though the methodological framework is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Broad coverage across four task types with analysis of non-LLM partners and boundary conditions, but limited to two-agent settings.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Problem formulation is rigorous, recursive definitions are clear, and case analyses are intuitive.
- **Value**: ⭐⭐⭐⭐ Offers important implications for the design of LLM multi-agent systems, though the ToM order assumptions in real-world deployment scenarios require further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/multi_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[ICML 2026\] CoOT: Learning to Coordinate In-Context with Coordination Transformers](../../ICML2026/multi_agent/coot_learning_to_coordinate_in-context_with_coordination_transformers.md)
- [\[ACL 2026\] SILO-BENCH: A Scalable Environment for Evaluating Distributed Coordination in Multi-Agent LLM Systems](../../ACL2026/multi_agent/silo-bench_a_scalable_environment_for_evaluating_distributed_coordination_in_mul.md)
- [\[ICLR 2026\] HAMLET: A Hierarchical and Adaptive Multi-Agent Framework for Live Embodied Theatre](../../ICLR2026/multi_agent/hamlet_a_hierarchical_and_adaptive_multi-agent_framework_for_live_embodied_theat.md)
- [\[AAAI 2026\] Scalable and Accurate Graph Reasoning with LLM-Based Multi-Agents](scalable_and_accurate_graph_reasoning_with_llm-based_multi-agents.md)

</div>

<!-- RELATED:END -->
