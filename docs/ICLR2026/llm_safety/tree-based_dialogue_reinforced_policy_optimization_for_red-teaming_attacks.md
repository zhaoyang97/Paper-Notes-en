---
title: >-
  [Paper Note] Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks (DialTree)
description: >-
  [ICLR 2026][LLM Safety][Multi-turn jailbreak] This paper proposes DialTree, which frames multi-turn red-teaming as a goal-oriented dialogue policy optimization problem. By employing tree-structured rollouts with quality-based pruning to explore the attack trajectory space, combined with an adaptive mask to prevent format forgetting, DialTree achieves an average ASR of 81.5% across 12 target models—44.2% higher than the previous SOTA—and attains 71% ASR even on Claude-4-Sonnet.
tags:
  - ICLR 2026
  - LLM Safety
  - Multi-turn jailbreak
  - red-teaming
  - reinforcement learning
  - tree search
  - dialogue policy optimization
date: 2026-05-08
content_hash: d9c7752ed6ed6b4c
---

# Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks (DialTree)

**Conference**: ICLR 2026  
**arXiv**: [2510.02286](https://arxiv.org/abs/2510.02286)  
**Code**: None  
**Area**: AI Safety / Red-Teaming  
**Keywords**: Multi-turn jailbreak, red-teaming, reinforcement learning, tree search, dialogue policy optimization

## TL;DR
This paper proposes DialTree, which frames multi-turn red-teaming as a goal-oriented dialogue policy optimization problem. By employing tree-structured rollouts with quality-based pruning to explore the attack trajectory space, combined with an adaptive mask to prevent format forgetting, DialTree achieves an average ASR of 81.5% across 12 target models—44.2% higher than the previous SOTA—and attains 71% ASR even on Claude-4-Sonnet.

## Background & Motivation

**Background**: Red-teaming is a critical technique for discovering safety vulnerabilities in LLMs. Existing methods fall into single-turn attacks (GCG/PAIR/TAP) and multi-turn attacks (MTSA/ActorAttack/X-Teaming). Research has shown that multi-turn attacks are far more effective than single-turn ones, as they can gradually erode safety boundaries.

**Limitations of Prior Work**:
   - Existing multi-turn methods rely on hand-crafted heuristics or templates and cannot learn long-horizon adaptive strategies.
   - The state space of multi-turn dialogue grows exponentially, making effective exploration with standard RL methods difficult.
   - Jailbreak rewards come from imperfect proxy models (non-verifiable rewards), leading to unstable guidance signals.
   - Format-following capability undergoes catastrophic forgetting during RL training.

**Key Challenge**: The dialogue space for multi-turn attacks is enormous, while effective attack strategies are sparse and difficult to discover.

**Goal**: How to efficiently explore the multi-turn attack space, learn long-horizon dialogue strategies, and stabilize RL training.

**Key Insight**: Framing red-teaming as goal-oriented strategic dialogue, using tree search for structured exploration and adaptive masking for training stabilization.

**Core Idea**: Tree rollout + pruning = structured exploration of the multi-turn attack space; adaptive mask = protecting format tokens from being negatively updated by RL gradients.

## Method

### Overall Architecture
Given an attack goal $g$, the attacker model $\pi_\theta$ generates chain-of-thought (CoT) reasoning and an attack query, which is sent to the target model $\pi_{\text{tgt}}$ for a response. The system then evaluates whether a jailbreak has succeeded before proceeding to the next turn. Training consists of two phases: (1) Cold-Start SFT to initialize format adherence and basic attack capability; and (2) DialTree RL to optimize the attack policy via tree search + GRPO.

### Key Designs

1. **Dialogue Tree Rollout + Quality Pruning**:

    - **Function**: Expands $n$ candidate action branches from each dialogue state to form a tree structure for exploring the attack space.
    - **Mechanism**: Starting from the initial state $s_0 = (g, \emptyset)$, the model samples $n$ distinct (CoT, query) pairs at each turn for every active state. Each query is sent to the target model to obtain a response, forming new states. Three pruning criteria are applied: format validation (discarding outputs without CoT or query), topic consistency (discarding outputs that deviate from the goal), and branch limit (retaining at most $w$ nodes per turn).
    - **Design Motivation**: Standard GRPO samples independent trajectories and cannot compare the effects of different actions under a shared dialogue context; the tree structure enables exploration of multiple directions from the same state.
    - **Default Parameters**: $T_{\max}=5$ turns, branching factor $n=4$, group size $G=32$.

2. **Adaptive Mask Mechanism**:

    - **Function**: Selectively protects format tokens from gradient updates during RL training.
    - **Mechanism**: Defines an adaptive mask $M_t^{(i)} = 1 - \mathbb{I}((T_t^{(i)} \in \mathcal{V}_{\text{fmt}}) \land (A^{(i)} < 0))$. When the trajectory advantage $A < 0$, format tokens are masked to shield them from penalty gradients; when $A \geq 0$, format tokens are updated normally to reinforce correct formatting.
    - **Design Motivation**: RL training was found to cause catastrophic forgetting of format-following capability—malformed outputs surge from near 0% to over 70%. The root cause is that penalty gradients from negative-advantage trajectories inadvertently penalize correct format tokens.
    - **vs. Static Mask**: A static mask always protects format tokens, but is less effective than the adaptive variant, since format tokens in positive-advantage trajectories also need to be updated to reinforce correct formatting.

3. **Red-Teaming Reward Design**:

    - **Function**: Uses the HarmAug-Guard safety classifier to assess whether each trajectory achieves a successful jailbreak.
    - **Mechanism**: $R = 1$ if any turn's (query, response) pair in the dialogue is judged harmful ($r_\phi(g; q_t, r_t) > 0.5$); otherwise $R = 0$. The binary reward is simple yet effective.
    - A held-out GPT-4o judge is used for evaluation, distinct from the HarmAug-Guard used during training, to avoid reward hacking.

### Loss & Training
- **SFT phase**: 397 manually curated red-teaming dialogues with CoT annotations.
- **RL phase**: Dialogue GRPO, computing group-relative advantages over trajectories collected via tree rollout. 500 training goals sampled from AdvBench/DangerousQA/CatQA.
- **Attacker model**: Llama-3.1-8B-Instruct; **target model** (during training): Llama-3.2-1B-Instruct.
- **Key insight**: Training uses only a 1B target model, yet the learned attack strategies transfer to large models such as GPT-4o and Claude-4.

## Key Experimental Results

### Main Results: Attack Success Rate (ASR@1, HarmBench)

| Method | GPT-4o | Claude-4-Sonnet | Grok-4 | o3-mini | Llama-3.3-70B | Avg (12 models) |
|--------|--------|-----------------|--------|---------|---------------|-----------------|
| GCG | 12.5 | 0 | 1.0 | 0 | 8.5 | 12.4 |
| PAIR | 18.0 | 2.5 | 8.5 | 11.5 | 25.5 | 17.6 |
| X-Teaming | 48.0 | 9.5 | 10.5 | 19.0 | 50.0 | 37.3 |
| **DialTree** | **86.0** | **71.0** | **75.0** | **86.5** | **89.5** | **81.5** |

### Ablation Study: Effect of Adaptive Mask

| Mask Strategy | Training Stability | Malformed Trajectory Rate (step 40) | Reward Trend |
|---------------|-------------------|--------------------------------------|--------------|
| No Mask | Training collapse | ~100% | Approaches 0 |
| Static Mask | Partially mitigated | ~100% (after step 60) | Slow decline |
| **Adaptive Mask** | **Stable** | **<50%** | **Steady increase** |

### Key Findings
- **Remarkable cross-model transfer**: Trained only on a 1B model, DialTree achieves 71% ASR on Claude-4-Sonnet (widely regarded as the safest model), far exceeding the best competing method at 26%.
- **Tree search contributes substantially**: Compared to standard rollout without tree search, the tree-structured approach yields significant ASR gains.
- **Adaptive mask is critical**: Without masking, training collapses within 40 steps; the adaptive mask is the only approach that sustains stable training.
- **High data efficiency**: Only 397 SFT samples and 500 RL training goals are sufficient to produce a powerful attacker.
- **Multi-turn vastly outperforms single-turn**: Multi-turn average ASR 81.5% vs. best single-turn 33.8%.

## Highlights & Insights
- **Red-teaming as strategic dialogue**: Reframing jailbreaking as a goal-oriented dialogue decision problem, rather than simple prompt optimization, enables long-horizon strategic planning.
- **Discovery and resolution of format forgetting**: Catastrophic forgetting of format-following capability during RL training is a widespread but underappreciated issue. The adaptive mask identifies the root cause (format tokens penalized by negative-advantage gradients) and proposes an elegant solution that is transferable to any RL training scenario requiring structured output formats.
- **Small-model training → large-model transfer**: Training with a 1B target model yields attack strategies that remain effective against GPT-4o/Claude-level models at inference time, indicating strong cross-model transferability of attack strategies—a serious warning for defenders.

## Limitations & Future Work
- **Absence of a defensive perspective**: The paper focuses exclusively on attacks and does not explore how the discovered vulnerabilities could inform improved defenses.
- **Reward model reliability**: HarmAug-Guard as a proxy reward may have blind spots, potentially leading to reward hacking.
- **Computational overhead**: Tree search combined with multi-turn interaction incurs high rollout costs.
- **Future directions**: Combining with ReSA's Answer-Then-Check defense strategy to evaluate DialTree's effectiveness against reasoning-augmented defenses would be a natural next step.

## Related Work & Insights
- **vs. X-Teaming**: X-Teaming uses multi-agent collaborative planning for multi-turn attacks (37.3% ASR), whereas DialTree uses a single agent with tree-search RL (81.5% ASR), demonstrating that learned policy optimization outperforms heuristic planning.
- **vs. PAIR/TAP**: These methods iteratively optimize single-turn prompts; DialTree extends the paradigm to multi-turn dialogue policy space, achieving a dramatic performance leap.
- **vs. ActorAttack**: ActorAttack guides attacks through semantically related entities; DialTree directly learns dialogue policies, offering greater flexibility and superior performance.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The framework combining tree-search RL with adaptive masking is novel; the discovery of format forgetting has independent value.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Evaluated on 12 target models (including GPT-4o/Claude-4/Grok-4), 8 baselines, with detailed ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Problem formulation is clear, though some notation could be simplified.
- **Value**: ⭐⭐⭐⭐⭐ Significant implications for understanding multi-turn safety vulnerabilities in LLMs and for improving defenses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] On the Sample Complexity of Differentially Private Policy Optimization](../../NeurIPS2025/llm_safety/on_the_sample_complexity_of_differentially_private_policy_optimization.md)
- [\[ICLR 2026\] OFMU: Optimization-Driven Framework for Machine Unlearning](ofmu_optimization-driven_framework_for_machine_unlearning.md)
- [\[ICLR 2026\] Membership Inference Attacks Against Fine-tuned Diffusion Language Models (SAMA)](membership_inference_attacks_against_fine-tuned_diffusion_language_models.md)
- [\[NeurIPS 2025\] Teaming LLMs to Detect and Mitigate Hallucinations](../../NeurIPS2025/llm_safety/teaming_llms_to_detect_and_mitigate_hallucinations.md)
- [\[ICLR 2026\] Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning](veritas_generalizable_deepfake_detection_via_pattern-aware_reasoning.md)

</div>

<!-- RELATED:END -->
