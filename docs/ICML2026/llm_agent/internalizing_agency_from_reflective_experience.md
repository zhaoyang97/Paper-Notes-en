---
title: >-
  [Paper Note] Internalizing Agency from Reflective Experience
description: >-
  [ICML 2026][LLM Agent][agentic LLM] This paper proposes the LEAFE framework, enabling LLM agents to generate "failure→rollback→correction→success" experience data by reflecting on failed trajectories…
tags:
  - "ICML 2026"
  - "LLM Agent"
  - "agentic LLM"
  - "reflective experience"
  - "rollback exploration"
  - "experience distillation"
  - "Pass@k"
date: 2026-05-08
content_hash: c5b20f4e33925e58
---

# Internalizing Agency from Reflective Experience

**Conference**: ICML 2026  
**arXiv**: [2603.16843](https://arxiv.org/abs/2603.16843)  
**Code**: Not released  
**Area**: LLM Agent / Long-horizon Interaction Training  
**Keywords**: agentic LLM, reflective experience, rollback exploration, experience distillation, Pass@k

## TL;DR
This paper proposes the LEAFE framework, enabling LLM agents to generate "failure→rollback→correction→success" experience data by reflecting on failed trajectories, and then distilling feedback-grounded recovery ability via SFT. On long-horizon tasks such as CodeContests, WebShop, and ALFWorld, Pass@128 is improved by up to 14%, significantly outperforming outcome-driven RL methods like GRPO.

## Background & Motivation

**Background**: LLMs are shifting from passive answering to autonomous agents. The common post-training approach is RL with verifiable rewards (RLVR / GRPO)—multiple rollouts, a scalar reward at the end, and policy gradients to increase the probability of successful trajectories.

**Limitations of Prior Work**: In long-horizon interactive scenarios, terminal scalar rewards are extremely sparse. Most rollouts receive no reward, and updates are dominated by the few already successful samples, so the model only learns to "do what it already knows" more reliably. The environment actually provides rich feedback at every step (error messages, state transitions, compilation errors), but all are compressed into a 0/1 signal and discarded. The result is distribution sharpening: Pass@1 increases, but Pass@1024 remains unchanged or even drops.

**Key Challenge**: Distribution sharpening and agency internalization are distinct. To truly expand the set of problems the model can solve, it must learn "my trajectory failed, where did it go wrong, how to fix it," whereas outcome-driven training only teaches "this trajectory as a whole is good."

**Goal**: Internalize the recovery process of "identify key decision points → rollback at that point → use environment feedback for targeted correction" into model weights, rather than relying on best-of-$k$ retries or Tree-of-Thoughts external search at inference.

**Key Insight**: Instead of only rewarding entire successful trajectories, explicitly create failure cases, locate error points, and supervise corrective actions. Environment feedback is no longer compressed into a scalar but structured as natural language "diagnosis + repair instructions" (experience summary) for training supervision.

**Core Idea**: Use reflection to generate experimental trajectories of "failure→rollback→correction→success," then distill post-rollback corrective actions via SFT, thereby encoding recovery agency into the weights.

## Method

### Overall Architecture
LEAFE consists of two stages: (1) Tree-Based Experience Generation with Rollback: Roll out trajectories on the base policy $\pi_\theta$, periodically triggering reflection. The reflection module determines if the current trajectory is deviating; if so, it outputs a rollback point $\tau$ and an experience summary (including failure diagnosis and repair suggestions). From $\tau$, restart and branch out one or more corrective actions. (2) Experience Distillation: From all "eventually successful" corrected trajectories, extract the "corrective action at $\tau$" as the target token for SFT, so that at inference, even without explicit experience, the model can output corresponding corrections when encountering similar failure signals.

### Key Designs

1. **Tree-based Reflective Rollback Experience Generation**:

    - **Function**: Expands a single trajectory rollout into a "failure detection + counterfactual correction" search tree, allowing one failed trace to generate multiple "corrected and successful" traces as training data.
    - **Mechanism**: Under the ReAct paradigm, the state at each time step $t$ is $h_t=(o_0,a_0,\ldots,o_t)$, with action $a_t\sim\pi_\theta(\cdot|h_t,q)$. Every few steps, a reflection prompt is inserted, and the model itself decides whether to rollback; if so, it outputs (i) rollback point $\tau$, (ii) natural language experience summary $e$ (including "where it went wrong + how to fix"). Then, $\pi_\theta(\cdot|h_\tau,q,e)$ samples new actions to branch. One original rollout can yield multiple (failure→rollback→fix→success) triplets.
    - **Design Motivation**: Scalar rewards cannot identify "which step went wrong," but LLMs inherently have the ability to read environment feedback and locate errors. Externalizing this in-context ability into explicit rollback + correction training signals provides much denser information than GRPO's group-relative reward.

2. **Experience-to-Policy Distillation**:

    - **Function**: Distills the corrective actions generated in the first stage into model weights, so that at deployment, the model can naturally make corrections without experience prompts.
    - **Mechanism**: For each (failure→rollback→fix→success) trajectory, extract the "corrected sub-trajectory with experience prompt" from the rollback point $\tau$, constructing SFT data $(h_\tau, a^{\rm fix}_\tau,\ldots,o_T)$. **During training, experience is not provided; the model is required to output the corrective action sequence directly after $h_\tau$**. Thus, at test time, the model can autonomously switch to correction mode when encountering similar failure patterns, without external reflection prompts.
    - **Design Motivation**: Either the model runs expensive reflection + retries at every inference (high deployment cost), or this ability is stored in the weights. The key to distillation is "conditioning on $h_\tau$ without experience to predict corrective actions," forcing the model to infer corrections from environment feedback alone.

3. **Comparison with GRPO: Sparse Rewards vs Decision-level Supervision**:

    - **Function**: Provides decision-level reflect→revise supervision under the same interaction budget, rather than episode-level scalar scoring.
    - **Mechanism**: GRPO computes group-relative advantage $\hat{A}_i=(r_i-\bar{r})/\sigma_r$ for $G$ traces of the same prompt, then applies policy gradient, essentially weighting entire traces. LEAFE directly supervises "what should be output after rollback" at the token level. The former encourages exploitation of known successful modes, while the latter pushes the behavior distribution into new regions.
    - **Design Motivation**: On long-horizon tasks, GRPO tends to sharpen only the few modes already present in the base model's long tail, with Pass@k for large $k$ unchanged; explicit correction supervision expands coverage.

### Loss & Training

Stage 1 uses only base policy self-sampling and reflection to generate data, with no gradient updates; Stage 2 uses standard SFT cross-entropy $\mathcal{L}=-\sum_t \log \pi_\theta(a^{\rm fix}_t|h_t)$, with loss computed only on corrected action tokens, not environment feedback tokens. Hyperparameters such as reflection frequency and rollback budget are detailed in the appendix.

## Key Experimental Results

### Main Results
Five agentic benchmarks: CodeContests (program synthesis + execution feedback), WebShop (shopping agent), ALFWorld (household agent), ScienceWorld (scientific exploration), Sokoban (box-pushing). All methods are evaluated under a fixed interaction budget.

| Task | Metric | Base | GRPO | Early Exp. | LEAFE | Gain over strongest baseline |
|--------|------|------|----------|---|---|---|
| CodeContests | Pass@1 | base | slight ↑ | slight ↑ | significant ↑ | Gain |
| Long-horizon Avg | Pass@128 | base | ≈base | + | +14% | +14% |
| General | Pass@1 | base | + | + | ++ | consistently leads |

### Ablation Study

| Configuration | Pass@1 | Pass@128 | Notes |
|------|---------|---------|------|
| Base | Low | Low | No post-training |
| GRPO (outcome RL) | Medium-High | ≈Base | Typical sharpening |
| Early Experience (no rollback) | Medium | Medium | Only distills successful trajectories, no recovery signal |
| LEAFE w/o rollback | Medium | Medium | Removes tree branching, degrades to linear SFT |
| LEAFE w/o experience summary | Medium-High | Medium-High | Only "correction actions," no diagnostic explanation |
| **Full LEAFE** | **High** | **High (+14%)** | Complete framework |

### Key Findings
- Large $k$ reveals true differences: GRPO also improves Pass@1, but Pass@128 is almost unchanged, while LEAFE excels—indicating RLVR only sharpens high-frequency modes in the existing support set without expanding coverage.
- Experience summary and rollback are synergistic: Removing either causes a significant drop in Pass@128, showing that "diagnosis + corrective action" together constitute decision-level supervision.
- Models trained with LEAFE spontaneously trigger corrections internally at inference, even without being prompted to "reflect"—demonstrating that agency is truly internalized into the weights.
- Higher data efficiency than Early Experience: With the same SFT sample size, LEAFE's rollback construction utilizes "failure traces" as well, with each failure producing multiple successful sub-trajectories on average.

## Highlights & Insights
- The distinction between "distribution sharpening vs agency internalization" is very clear and helps the community avoid the "Pass@1 improvement is enough" evaluation trap.
- Structuring environment feedback as "natural language diagnosis + repair suggestions" is a reusable pattern: applicable to any scenario where the environment provides error signals, such as tool use, code agents, and web agents.
- Training with experience but not providing it at inference—this "training-assisted, inference-consistent" design forces the model to derive correction logic from environment signals itself, outperforming simple experience prompting for generalization.

## Limitations & Future Work
- Reflection trigger frequency and rollback budget are hyperparameters and may require retuning for different tasks; automatically deciding when to reflect remains an open problem.
- The reflection module relies on the base policy's self-assessment ability; if the base is too weak (e.g., sub-7B models), it may not recognize failures at all.
- Experiments are mainly in scenarios with rich environment feedback and verifiers; further validation is needed in sparse or delayed feedback settings (e.g., open-domain dialogue).
- No direct computational cost comparison with best-of-$N$ + self-reflection methods (e.g., Reflexion); systematic measurement of real deployment costs is needed.

## Related Work & Insights
- **vs GRPO / DeepSeek-R1 style RLVR**: This work also conducts post-training on LLMs, but uses structured reflection instead of scalar rewards; GRPO does not improve large $k$, whereas this method does.
- **vs Early Experience**: Early Experience also uses reflective trajectories but does not perform rollback branching, thus only distilling successful traces without utilizing failure signals; this work further uses failure traces as data sources.
- **vs Reflexion / Tree-of-Thoughts**: Those methods keep reflection/tree search at inference, requiring multiple calls each time; this work internalizes such agency into the weights, enabling single-pass inference.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of "reflection-generated data + distillation into weights" is not entirely original, but the explicit rollback + decision-level supervision framework and Pass@k perspective are clear contributions.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five agentic benchmarks covering coding/web/household/science/box-pushing, but lacks direct cost comparison with inference-time methods like Reflexion.
- Writing Quality: ⭐⭐⭐⭐ The "sharpening vs internalization" narrative is clear, and Pass@k curves are highly convincing.
- Value: ⭐⭐⭐⭐ Provides a complementary new paradigm to RLVR for agentic LLM post-training, with low deployment cost and reusable methods for tool/code agent scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] IDER: IDempotent Experience Replay for Reliable Continual Learning](../../ICLR2026/model_compression/ider_idempotent_experience_replay_for_reliable_continual_learning.md)
- [\[ACL 2026\] Mem²Evolve: Towards Self-Evolving Agents via Co-Evolutionary Capability Expansion and Experience Distillation](../../ACL2026/model_compression/mem2evolve_towards_self-evolving_agents_via_co-evolutionary_capability_expansion.md)
- [\[AAAI 2026\] SafeSieve: From Heuristics to Experience in Progressive Pruning for LLM-based Multi-Agent Communication](../../AAAI2026/model_compression/safesieve_from_heuristics_to_experience_in_progressive_pruning_for_llm-based_mul.md)
- [\[ACL 2026\] MAGEO: From Experience to Skill — Multi-Agent Generative Engine Optimization via Reusable Strategy Learning](../../ACL2026/model_compression/from_experience_to_skill_multi-agent_generative_engine_optimization_via_reusable.md)
- [\[ICML 2026\] QHyer: Q-conditioned Hybrid Attention-mamba Transformer for Offline Goal-conditioned RL](qhyer_q-conditioned_hybrid_attention-mamba_transformer_for_offline_goal-conditio.md)

</div>

<!-- RELATED:END -->
