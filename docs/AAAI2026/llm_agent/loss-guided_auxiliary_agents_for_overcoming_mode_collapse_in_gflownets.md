---
title: >-
  [Paper Note] Loss-Guided Auxiliary Agents for Overcoming Mode Collapse in GFlowNets
description: >-
  [AAAI 2026][LLM Agent][GFlowNet] This paper proposes LGGFN (Loss-Guided GFlowNets), in which the exploration of an auxiliary GFlowNet is directly driven by the training loss of the primary GFlowNet. The auxiliary agent's reward is defined as $R_{aux}(x) = R(x) + \lambda \cdot L_{main}(x)$, prioritizing regions where the primary model is least well-understood. On grid, sequence generation, and Bayesian structure learning tasks, LGGFN discovers 40× more unique modes and reduces…
tags:
  - "AAAI 2026"
  - "LLM Agent"
  - "GFlowNet"
  - "mode collapse"
  - "auxiliary agent"
  - "loss-guided"
  - "diversity sampling"
date: 2026-05-08
content_hash: 89f67ad4a0848b42
---

# Loss-Guided Auxiliary Agents for Overcoming Mode Collapse in GFlowNets

**Conference**: AAAI 2026
**arXiv**: [2505.15251](https://arxiv.org/abs/2505.15251)  
**Code**: Available  
**Area**: Generative Flow Networks / Exploration
**Keywords**: GFlowNet, mode collapse, auxiliary agent, loss-guided, diversity sampling

## TL;DR
This paper proposes LGGFN (Loss-Guided GFlowNets), in which the exploration of an auxiliary GFlowNet is directly driven by the training loss of the primary GFlowNet. The auxiliary agent's reward is defined as $R_{aux}(x) = R(x) + \lambda \cdot L_{main}(x)$, prioritizing regions where the primary model is least well-understood. On grid, sequence generation, and Bayesian structure learning tasks, LGGFN discovers 40× more unique modes and reduces exploration error by 99%.

## Background & Motivation

**Background**: GFlowNets are designed to sample from multimodal distributions proportional to a reward function—rather than merely finding the optimum—theoretically avoiding mode collapse. In practice, however, on-policy training still suffers from mode collapse, as the model is attracted to high-reward modes discovered early in training.

**Limitations of Prior Work**:
   - Existing exploration techniques rely on heuristic novelty signals (e.g., state counting, RND), which are decoupled from the model's actual learning state.
   - Novelty signals may direct exploration toward irrelevant regions (novel but unimportant).
   - There is no mechanism that leverages the model's own training signal to guide exploration.

**Key Challenge**: GFlowNets require broad exploration to learn the complete distribution, yet on-policy sampling is dominated by high-reward regions—a mechanism is needed to redirect sampling toward regions the model has not yet learned well.

**Goal**: Use the primary model's training loss as the exploration signal for an auxiliary agent, realizing a strategy of "explore where you understand least."

**Key Insight**: High training loss regions = regions poorly understood by the primary model → the auxiliary agent prioritizes exploration there → the collected samples are fed back to train the primary model.

**Core Idea**: Auxiliary GFlowNet reward = primary model reward + primary model loss → directed exploration of weak regions.

## Method

### Overall Architecture
A dual-GFlowNet architecture consisting of a primary agent (learning the target distribution) and an auxiliary agent (learning a loss-weighted distribution of the primary agent). The primary agent's training data is the union of on-policy samples and auxiliary agent samples. The auxiliary agent periodically updates its reward to reflect the current weaknesses of the primary agent.

### Key Designs

1. **Loss-Guided Auxiliary Reward**:

    - **Function**: Directs the auxiliary agent to prioritize regions where the primary agent incurs high loss.
    - **Mechanism**: $R_{aux}(x) = R(x) + \lambda \cdot L_{main}(x)$, where $L_{main}$ is the primary GFlowNet's training loss on trajectory $x$.
    - **Design Motivation**: High loss = primary model uncertainty/poor understanding → the region most in need of additional data. This is more direct than novelty signals, as it explicitly measures "learned poorly" rather than "visited or not."
    - **Implementation Details**: The loss can be trajectory-based TB loss, transition-based FM loss, or sub-trajectory loss, making the method agnostic to the specific training objective. $\lambda$ is set to keep $R_{aux}$ on the same scale as $R_{main}$, preventing training instability.

2. **Mixed Training Strategy**:

    - **Function**: Enables the primary agent to learn from two sources.
    - **Mechanism**: Training batch = $\alpha \cdot$ on-policy samples $+ (1-\alpha) \cdot$ auxiliary agent samples. $\alpha$ controls the exploration–exploitation trade-off.
    - **Design Motivation**: Purely auxiliary sampling may deviate too far from the reward distribution; mixing maintains stable distribution learning.

3. **Exploitation of Neural Network Generalization**:

    - **Function**: Leverages the generalization capacity of neural networks to propagate the loss signal across neighboring states.
    - **Mechanism**: High loss on a given trajectory → through network generalization, neighboring trajectories also exhibit higher loss → auxiliary agent exploration is region-wide rather than point-wise.
    - **Design Motivation**: This is more efficient than state counting—a single informative exploration step covers an entire "uncertain region."

### Loss & Training
- Primary agent: Standard GFlowNet training loss (TB/DB/SubTB).
- Auxiliary agent: TB loss with reward $= R + \lambda L_{main}$.

## Key Experimental Results

### Main Results

| Task | LGGFN | Best Baseline | Gain |
|------|-------|---------------|------|
| Hypergrid 128×128 L1↓ | **0.83±0.21** | 0.92±0.36 (AdaTeachers) | 10% |
| Sequence Generation Unique Modes↑ | **40× more** | Baseline | 40× |
| Sequence Generation Exploration Error↓ | **99% reduction** | Baseline | 99% |
| Bayesian Structure Learning | **+10%** | Baseline | 10% |

### Ablation Study

| Configuration | Performance |
|---------------|-------------|
| On-policy only (no auxiliary) | Severe mode collapse |
| Random exploration (unguided auxiliary) | Moderate improvement |
| Novelty-guided | Improvement but unstable |
| **Loss-guided (LGGFN)** | **Best and most stable** |

### Key Findings
- **Loss signal > novelty signal**: Loss directly measures "learned poorly," which is more precise than "visited or not."
- **40× more modes** (sequence generation): Demonstrates that LGGFN genuinely resolves mode collapse rather than merely improving approximation accuracy.
- **"Loss decay" of the auxiliary agent**: As the primary model improves, the auxiliary agent's exploration direction naturally shifts—an adaptive mechanism.

## Highlights & Insights
- The idea of **directly driving exploration via training loss** is simple yet powerful—"explore where you understand least" is the most natural exploration strategy.
- The approach has transfer value for any generative model requiring diversity sampling (e.g., drug design, materials discovery).
- The dual-model framework of GFlowNet + auxiliary agent is generalizable to other RL settings.

## Limitations & Future Work
- The $\lambda$ hyperparameter of the auxiliary agent requires tuning.
- The dual-GFlowNet architecture increases computational and memory overhead.
- Validation is limited to discrete-space tasks.

## Related Work & Insights
- **vs. Adaptive Teachers (Jain et al.)**: Uses a teacher distribution for guidance. LGGFN is more precise by directly leveraging the loss signal.
- **vs. RND / Curiosity**: Novelty-driven exploration focuses on state visitation frequency; LGGFN's loss-driven approach more accurately targets regions where the model has genuinely failed to learn.
- **vs. Diverse RL**: Diversity RL typically uses intrinsic rewards or information-theoretic regularization, whereas LGGFN's explicit auxiliary-agent guidance is more controllable and interpretable.
- Loss-guided exploration can be extended to diversity generation in RLHF and may also be applied to adversarial example mining.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The concept of loss-guided exploration is concise and powerful; directing the primary agent's exploration via high-loss regions through an auxiliary agent is a highly natural decoupled design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Three task categories (hypergrid, molecules, sequence generation), multiple baselines, and complete ablations; cross-domain validation is persuasive.
- **Writing Quality**: ⭐⭐⭐⭐ The motivation chain is clear, and the logical derivation from the mode collapse problem to the loss-guided solution is well-structured.
- **Value**: ⭐⭐⭐⭐ Directly applicable to multimodal sampling and molecular design; represents an important advance for the GFlowNet community in addressing mode collapse.

## Additional Notes
- The loss-guided auxiliary agent design is not limited to GFlowNets and can be generalized to any generative model scenario requiring diversity-driven exploration.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] GTR: Guided Thought Reinforcement Prevents Thought Collapse in RL-based VLM Agent Training](../../ICCV2025/llm_agent/gtr_guided_thought_reinforcement_prevents_thought_collapse_i.md)
- [\[AAAI 2026\] AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search](agentswift_efficient_llm_agent_design_via_value-guided_hierarchical_search.md)
- [\[AAAI 2026\] Verification-Guided Context Optimization for Tool Calling via Hierarchical LLMs-as-editors](verification-guided_context_optimization_for_tool_calling_via_hierarchical_llms-.md)
- [\[ICLR 2026\] Expanding the Capability Frontier of LLM Agents with ZPD-Guided Data Synthesis](../../ICLR2026/llm_agent/expanding_the_capability_frontier_of_llm_agents_with_zpd-guided_data_synthesis.md)
- [\[ICLR 2026\] WebArbiter: A Principle-Guided Reasoning Process Reward Model for Web Agents](../../ICLR2026/llm_agent/webarbiter_a_principle-guided_reasoning_process_reward_model_for_web_agents.md)

</div>

<!-- RELATED:END -->
