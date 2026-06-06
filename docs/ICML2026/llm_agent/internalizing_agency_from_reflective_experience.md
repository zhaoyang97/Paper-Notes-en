---
title: >-
  [Paper Note] Internalizing Agency from Reflective Experience
description: >-
  [ICML 2026][LLM Agent][agentic LLM] This paper proposes the LEAFE framework, which enables LLM agents to generate "failure $\rightarrow$ rollback $\rightarrow$ correction $\rightarrow$ success" experience data by reflect…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "agentic LLM"
  - "reflective experience"
  - "backtracking exploration"
  - "experience distillation"
  - "Pass@k"
date: 2026-05-08
content_hash: 881ebe0ed39f571b
---

# Internalizing Agency from Reflective Experience

**Conference**: ICML 2026  
**arXiv**: [2603.16843](https://arxiv.org/abs/2603.16843)  
**Code**: Undisclosed  
**Area**: LLM Agent / Long-horizon Interaction Training  
**Keywords**: agentic LLM, reflective experience, backtracking exploration, experience distillation, Pass@k

## TL;DR
This paper proposes the LEAFE framework, which enables LLM agents to generate "failure $\rightarrow$ rollback $\rightarrow$ correction $\rightarrow$ success" experience data by reflecting on failed trajectories. It then uses SFT to distill feedback-grounded recovery capabilities, improving Pass@128 by up to 14% on long-horizon tasks such as CodeContests, WebShop, and ALFWorld, significantly outperforming outcome-driven RL like GRPO.

## Background & Motivation

**Background**: LLMs are transitioning from passive responders to autonomous agents. Common post-training methods involve RL with verifiable rewards (RLVR / GRPO), which use multi-turn sampling, a scalar reward at the end of the episode, and policy gradients to increase the probability of successful trajectories.

**Limitations of Prior Work**: In long-horizon interaction scenarios, the information density of end-of-episode scalar rewards is extremely low. Most rollouts fail to receive a reward, and updates are dominated by a few samples that are already successful; the model merely learns to stabilize what it "already knows how to do." Although the environment provides rich feedback at every step (error messages, state transitions, compilation errors), this is often compressed into 0/1 signals and discarded. This results in distribution sharpening: Pass@1 increases while Pass@1024 remains stagnant or decreases.

**Key Challenge**: Distribution sharpening and agency internalization are distinct phenomena. To genuinely expand the set of solvable problems, a model must learn "where the current trajectory failed and how to fix it," whereas outcome-driven training only teaches that "this trajectory is overall good."

**Goal**: Internalize the recovery procedure—identifying key decision points, rolling back at those points, and making targeted corrections based on environment feedback—into the model weights, rather than relying on best-of-$k$ retries or Tree-of-Thoughts external searches during inference.

**Key Insight**: Instead of only rewarding successful trajectories, one should explicitly create failure cases, locate errors, and supervise correction actions. Environment feedback is not compressed into a scalar but structured into natural language "diagnoses + repair instructions" (experience summary) to serve as training supervision.

**Core Idea**: Generate experimental trajectories of "failure $\rightarrow$ rollback $\rightarrow$ correction $\rightarrow$ success" through reflection, and then use SFT to distill post-rollback corrective actions, thereby embedding recovery agency into the weights.

## Method

### Overall Architecture
LEAFE consists of two stages: (1) Tree-Based Experience Generation with Rollback: Rollouts are performed using the base policy $\pi_\theta$. Every few steps, a reflection module determines if the trajectory is deviating. If so, it generates a rollback point $\tau$ and an experience summary $e$ (containing failure diagnosis and repair suggestions), restarts from $\tau$, and branches out one or more corrective actions. (2) Experience Distillation: Corrective actions at $\tau$ are extracted from all "eventually successful" correction trajectories as target tokens for SFT. This enables the model to output corrections under similar failure signals during inference without explicitly providing the experience summary.

### Key Designs

1.  **Tree-Based Experience Generation with Rollback**:
    - **Function**: Expands single-trajectory rollouts into a "failure detection + counterfactual correction" search tree, allowing a failed trace to derive multiple "successful after correction" traces for training data.
    - **Mechanism**: Under the ReAct paradigm, the state at each time step $t$ is $h_t=(o_0, a_0, \ldots, o_t)$, and actions $a_t \sim \pi_\theta(\cdot|h_t, q)$. Reflection prompts are inserted periodically for the model to decide on a rollback. If triggered, it outputs (i) a rollback point $\tau$ and (ii) a natural language experience summary $e$. New actions are then sampled via $\pi_\theta(\cdot|h_\tau, q, e)$ to create branches. One original rollout can derive multiple (failure $\rightarrow$ rollback $\rightarrow$ fix $\rightarrow$ success) triplets.
    - **Design Motivation**: Scalar rewards cannot pinpoint "which step went wrong," but LLMs possess the ability to read environment feedback and locate errors. Externalizing this in-context ability into explicit rollback and correction signals provides much higher information density than GRPO's group-relative rewards.

2.  **Experience to Policy Distillation**:
    - **Function**: Distills the generated corrective actions into the model weights, allowing the model to naturally make corrections at deployment without experience prompts.
    - **Mechanism**: For each (failure $\rightarrow$ rollback $\rightarrow$ fix $\rightarrow$ success) trajectory, the corrected sub-trajectory generated with experience prompts is intercepted from $\tau$. SFT data $(h_\tau, a^{\rm fix}_\tau, \ldots, o_T)$ is constructed. **During training, the experience summary is not provided; the model is trained to directly produce the sequence of corrective actions following $h_\tau$.** Consequently, during testing, the model can endogenously switch to a correction mode when encountering similar failure patterns.
    - **Design Motivation**: Either the model runs expensive reflection and retries for every inference (high deployment cost), or this capability is stored in the weights. The key to distillation is conditioning on $h_\tau$ without $e$ to predict the fix, forcing the model to infer the necessary correction from the environment feedback itself.

3.  **Contrast with GRPO: Sparse Rewards vs. Decision-Level Supervision**:
    - **Function**: Provides decision-level reflect $\rightarrow$ revise supervision under the same interaction budget, rather than episode-level scalar ratings.
    - **Mechanism**: GRPO calculates group-relative advantage $\hat{A}_i = (r_i - \bar{r})/\sigma_r$ for $G$ traces of the same prompt for policy gradient, essentially weighting the entire trace. LEAFE provides token-by-token supervision on "what should be output after rollback." The former encourages exploitation of known success patterns, while the latter pushes the behavioral distribution into new regions.
    - **Design Motivation**: On long-horizon tasks, GRPO tends to sharpen onto a few patterns the base model already knows in its long tail, failing to improve Pass\@$k$ for large $k$. Explicit error-correction supervision expands coverage.

### Loss & Training
Stage 1 solely uses base policy self-sampling and reflection for data generation with no gradient updates. Stage 2 uses standard SFT cross-entropy: $\mathcal{L} = -\sum_t \log \pi_\theta(a^{\rm fix}_t|h_t)$. The loss is calculated only on the corrected action tokens, not the environment feedback tokens. Hyperparameters such as reflection frequency and rollback budget are detailed in the appendix.

## Key Experimental Results

### Main Results
Evaluation on 5 agentic benchmarks: CodeContests (program synthesis with execution feedback), WebShop (shopping agent), ALFWorld (household agent), ScienceWorld (scientific exploration), and Sokoban (box-pushing). All methods are evaluated under a fixed interaction budget.

| Task | Metric | Base | GRPO | Early Exp. | Ours (LEAFE) | Rel. Strongest Baseline |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| CodeContests | Pass@1 | base | Slight gain | Slight gain | Significant gain | Gain |
| Average (Long-horizon) | Pass@128 | base | ≈base | + | +14% | +14% |
| General | Pass@1 | base | + | + | ++ | Consistently leading |

### Ablation Study

| Configuration | Pass@1 | Pass@128 | Description |
| :--- | :--- | :--- | :--- |
| Base | Low | Low | No post-training |
| GRPO (outcome RL) | Mid-High | ≈Base | Typical sharpening |
| Early Experience (no rollback) | Mid | Mid | Distills success traces only, no recovery signal |
| LEAFE w/o rollback | Mid | Mid | No tree-based branching, degrades to linear SFT |
| LEAFE w/o experience summary | Mid-High | Mid-High | Fix actions only, no diagnostic explanation |
| **Full LEAFE** | **High** | **High (+14%)** | Complete framework |

### Key Findings
- Large $k$ reveals the true difference: While GRPO improves Pass@1, its Pass@128 remains stagnant. LEAFE excels at large $k$, indicating that RLVR merely sharpens high-frequency patterns in the existing support set without expanding coverage.
- Experience summaries and rollback are synergistic: Removing either significantly decreases Pass@128, proving that "diagnosis + corrective action" together form effective decision-level supervision.
- The model trained by LEAFE triggers corrections internally even if not prompted to "reflect" during inference, verifying that agency is internalized into the weights.
- Higher data efficiency than Early Experience: With the same SFT sample volume, LEAFE utilizes failed traces, producing multiple successful sub-trajectories per failure on average.

## Highlights & Insights
- The distinction between "distribution sharpening vs. agency internalization" is a clear conceptual split that helps the community move beyond the "Pass@1 is enough" evaluation pitfall.
- Structuring environment feedback into "natural language diagnosis + repair suggestions" is a reusable pattern applicable to any scenario where the environment reports errors (tool use, code agents, web agents).
- Distilling without feeding the experience summary during training—a "train-time auxiliary, inference-time self-consistency" design—forces the model to derive correction logic from environmental signals, leading to better generalization than simply using experience as a prompt.

## Limitations & Future Work
- Reflection frequency and rollback budget are hyperparameters that may require tuning for different tasks; automatically determining when to reflect remains an open problem.
- The reflection module depends on the base policy's self-assessment capability; if the base is too weak (e.g., small models under 7B), it may fail to recognize failures.
- Experiments focused on scenarios with rich feedback and verifiers; further validation is needed for sparse feedback (e.g., open dialogue) or delayed feedback scenarios.
- There is no direct computational cost comparison with best-of-$N$ + self-reflection (e.g., Reflection); the real cost difference during deployment needs systematic measurement.

## Related Work & Insights
- **vs. GRPO / DeepSeek-R1 style RLVR**: Like these, LEAFE performs post-training on LLMs but replaces scalar rewards with structured reflection; LEAFE improves for large $k$ where GRPO does not.
- **vs. Early Experience**: Early Experience uses reflective trajectories but lacks rollback branching, meaning it only distills successful traces without utilizing failure signals. LEAFE uses failed traces as a data source.
- **vs. Reflexion / Tree-of-Thoughts**: Those methods keep reflection/tree search at inference time, requiring multiple calls; LEAFE internalizes this agency into the weights for single-pass inference.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of "reflection-generated data + distillation" is not entirely original, but the framework of explicit rollback + decision-level supervision and the Pass@k perspective are clear contributions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 5 agentic benchmarks (coding/web/household/science/Sokoban), though lacks a direct cost comparison with inference-time methods like Reflexion.
- **Writing Quality**: ⭐⭐⭐⭐ The "sharpening vs. internalization" narrative is clear, and the Pass@k curves are highly persuasive.
- **Value**: ⭐⭐⭐⭐ Provides a new paradigm for agentic LLM post-training that is complementary to RLVR, with low deployment costs and high reusability across tool/code agent scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] EvolveR: Self-Evolving LLM Agents through an Experience-Driven Lifecycle](evolver_self-evolving_llm_agents_through_an_experience-driven_lifecycle.md)
- [\[ICML 2026\] Skill-Pro: Learning Reusable Skills from Experience via Non-Parametric PPO for LLM Agents](skill-pro_learning_reusable_skills_from_experience_via_non-parametric_ppo_for_ll.md)
- [\[ACL 2026\] ExpSeek: Self-Triggered Experience Seeking for Web Agents](../../ACL2026/llm_agent/expseek_self-triggered_experience_seeking_for_web_agents.md)
- [\[ACL 2026\] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](../../ACL2026/llm_agent/mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)
- [\[ACL 2026\] From Storage to Experience: A Survey on the Evolution of LLM Agent Memory Mechanisms](../../ACL2026/llm_agent/from_storage_to_experience_a_survey_on_the_evolution_of_llm_agent_memory_mechani.md)

</div>

<!-- RELATED:END -->
