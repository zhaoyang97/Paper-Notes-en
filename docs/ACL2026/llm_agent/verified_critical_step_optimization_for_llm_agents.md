---
title: >-
  [Paper Note] Verified Critical Step Optimization for LLM Agents
description: >-
  [ACL 2026][LLM Agent][Critical Step Optimization] CSO identifies "verified critical steps" from an agent's own failed trajectories—specific points where an alternative action leads to task success. by constructing DPO pr…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Critical Step Optimization"
  - "DPO"
  - "Process Reward Model"
  - "Credit Assignment"
date: 2026-05-08
content_hash: e47ad87ff4d4da87
---

# Verified Critical Step Optimization for LLM Agents

**Conference**: ACL 2026 Findings  
**arXiv**: [2602.03412](https://arxiv.org/abs/2602.03412)  
**Code**: https://github.com/kiaia/CSO; https://github.com/Tencent/CognitiveKernel-Pro  
**Area**: llm_agent  
**Keywords**: LLM Agent, Critical Step Optimization, DPO, Process Reward Model, Credit Assignment  

## TL;DR
CSO identifies "verified critical steps" from an agent's own failed trajectories—specific points where an alternative action leads to task success. by constructing DPO preference pairs exclusively at these critical decision points, it enhances the post-training performance of long-horizon LLM agents using fewer and more reliable supervisory signals.

## Background & Motivation
**Background**: LLM agents are increasingly handling long-horizon tasks such as web searching, tool calling, file manipulation, and multi-step information synthesis. The standard post-training pipeline involves SFT on high-quality trajectories followed by RL or preference optimization to enhance execution capability. Unlike chat models, agent outputs consist of trajectories composed of alternating states, actions, and observations.

**Limitations of Prior Work**: Trajectory-level methods apply success/failure rewards to the entire sequence, which often penalizes reasonable steps in failed trajectories or reinforces accidental errors in successful ones. While step-level methods are more granular, they typically rely on PRM score estimates for every step, causing PRM noise to amplify in long-horizon tasks. Monte Carlo-based step rewards require rollouts from every intermediate state, which is prohibitively expensive.

**Key Challenge**: Not every step in an agent's trajectory is equally valuable for learning. Many steps are merely sequential executions or data transfers. The actual success or failure is determined by a few "bifurcation points," such as choosing the right tool, formulating a search query, or extracting evidence. Post-training requires precise credit assignment without modeling every step identically.

**Goal**: The authors aim to find a method positioned between trajectory-level DPO and expensive online RL: learning only from steps verified to change the final outcome. This avoids the coarse rewards of full trajectories while bypassing the need to trust PRM estimates for every step.

**Key Insight**: Drawing inspiration from the observation in RLVR that "a few high-entropy tokens drive effective learning," this paper treats critical actions in long-horizon agents as sparse learning locations. Starting from failed trajectories of the current policy, the method uses an expert to provide alternative candidate actions and employs outcome verification to determine if these actions truly flip a failure branch into a success branch.

**Core Idea**: First, use a PRM to efficiently filter candidate critical steps where the "policy action is poor but the expert action is good." Then, verify the outcome by continuing the rollout from the expert alternative until the task ends. Only branches that are verified as successful are used to construct DPO preference pairs.

## Method
The core of CSO is not providing more accurate rewards for every step, but changing how training data is constructed. It treats agent post-training as a process of "locating critical errors from failures": allowing the current policy to execute tasks and collect failed trajectories; at each potential decision point, an expert model generates alternative actions; a PRM filters candidate critical points; and finally, the expert action is grafted onto the original state for the policy to complete. Only successful branches are considered verified critical steps, used to form preference pairs where the "expert alternative is superior to the original policy action."

### Overall Architecture
The paper formalizes an agent trajectory as $\tau=(s_1,a_1,o_1,\ldots,s_T,a_T,o_T)$, where $s_t$ includes the task and history, $a_t$ is the action, $o_t$ is the observation, and the outcome $y\in\{0,1\}$ indicates success. The model starts as a basic policy $\pi_\theta$ after SFT, which still fails at certain critical decision points.

CSO consists of six steps: 1. Deploy the current policy to collect failed trajectories; 2. Sample $K=5$ alternative actions using an expert model at each step of the failed trajectories; 3. Score both the policy action and expert alternatives using a PRM; 4. Identify candidate critical steps where $r^{policy}_t<\gamma_{low}$ and $max_j r^{expert}_{t,j}>\gamma_{high}$ (with $\gamma_{low}=0.45, \gamma_{high}=0.65$); 5. Conduct branch rollouts for high-scoring expert alternatives where the policy completes the task; 6. Retain only the successful branches to construct $(s_t,a_t^+,a_t^-)$ preference pairs for DPO training.

### Key Designs
1. **Locating policy weaknesses from failed trajectories**:
	- **Function**: Focuses training data on the state distribution where the current policy actually makes mistakes.
	- **Mechanism**: Instead of generalizing from expert success trajectories, CSO executes the policy on training tasks and retains trajectories $\mathcal{T}_{fail}$ where the outcome is failure. All candidate critical steps originate from these failures.
	- **Design Motivation**: Learning only from expert demos might involve actions beyond the model's capacity; learning only from success trajectories makes it difficult to identify specific weaknesses. Failed trajectories provide semi-on-policy coverage, targeting areas the model needs to correct.

2. **Dual filtering with PRM screening + outcome verification**:
	- **Function**: Identifies decision points truly worth preference construction without expensive Monte Carlo sampling for every step.
	- **Mechanism**: The PRM serves as a candidate filter rather than a final reward. It identifies steps where the original action is low-rated and at least one expert alternative is high-rated. Subsequently, the system continues the rollout and uses final task correctness to verify the branch.
	- **Design Motivation**: PRM estimates are noisy; training directly on PRM scores can mislead the model. However, avoiding PRM entirely results in excessive verification costs. CSO uses PRM for recall and outcome verification for precision.

3. **Verified critical step-level DPO**:
	- **Function**: Applies the learning signal only to local actions that change the task outcome.
	- **Mechanism**: For each verified critical step, a preference pair $(s_t,a_t^+,a_t^-)$ is constructed, where $a_t^+$ is the expert action that led to success and $a_t^-$ is the original policy action. The DPO objective is: $L_{CSO}=-E\log\sigma(\beta\log\frac{\pi_\theta(a_t^+|s_t)}{\pi_{ref}(a_t^+|s_t)}-\beta\log\frac{\pi_\theta(a_t^-|s_t)}{\pi_{ref}(a_t^-|s_t)})$.
	- **Design Motivation**: Trajectory-level DPO compares full successful/failed trajectories, resulting in coarse credit assignment. CSO targets critical actions, reducing interference from irrelevant tokens.

### Loss & Training
The base model is CK-Pro-8B, an agent policy based on Qwen3-8B SFT, running in the Cognitive Kernel Pro framework. Training data is generated from 47K SFT task trajectories by collecting failures. Claude-3.7-Sonnet is used as both the expert model and the PRM (using rubric-based prompts). DPO training uses LlamaFactory with $\beta=0.5$. The framework supports iterative training, where the policy is updated, new failures are collected, and the previous policy serves as the reference for up to 2 rounds.

## Key Experimental Results

### Main Results
Experiments were conducted on GAIA-Text-103 and XBench-DeepSearch2505. GAIA-Text-103 is a text-based subset of GAIA (L1/L2/L3); XBench-DeepSearch involves complex search and synthesis tasks. Evaluation uses an LLM judge to check correctness against gold answers.

| Model/Method | GAIA L1 | GAIA L2 | GAIA L3 | GAIA All | XBench Score |
|-----------|---------|---------|---------|----------|--------------|
| GPT-4.1 | 56.4 | 44.2 | 16.7 | 45.6 | 27.0 |
| Claude-3.7-Sonnet | 76.9 | 57.7 | 33.3 | 62.1 | 41.0 |
| Qwen3-8B | 35.9 | 13.5 | 0.0 | 20.4 | 7.0 |
| CK-Pro-8B (SFT) | 46.2 | 34.6 | 8.3 | 35.9 | 23.0 |
| CK-Pro-8B + ETO | 51.2 | 36.5 | 8.3 | 38.9 | 22.0 |
| CK-Pro-8B + RFT | 51.2 | 28.8 | 8.3 | 34.9 | 20.0 |
| CK-Pro-8B + Step-DPO | 53.3 | 34.6 | 8.3 | 38.9 | 25.0 |
| CK-Pro-8B + IPR | 56.4 | 42.3 | 16.7 | 44.6 | 24.0 |
| CK-Pro-8B + CSO | 61.5 | 48.1 | 16.7 | 49.5 | 29.0 |

### Ablation Study

| Configuration | GAIA-Text | Sample Count/Cost | Description |
|------|-----------|------------|------|
| Expert Success + Expert Failure | 46.6 | Same critical step set | Compares expert's own success/failure; less aligned with policy weakness |
| Policy Success + Policy Failure | 42.7 | Same critical step set | Limited policy success quality leads to weaker signals |
| Expert Success + Policy Failure | 49.5 | Same critical step set | Optimal: high-quality positive, negative from real policy failure |
| PRM + Verification | 49.5 | 671 preference pairs | Best performance with the fewest samples |
| w/o PRM | 48.5 | 1,967 preference pairs | Similar performance but ~3x verification cost |
| w/o Verification | 43.6 | 4,126 preference pairs | PRM-only noise is high, causing performance drop |

| Analysis Item | Result | Meaning |
|--------|------|------|
| Branch candidates $k=3$ | GAIA 46.6, XBench 26.0 | Insufficient exploration |
| Branch candidates $k=5$ | GAIA 49.6, XBench 29.0 | Optimal cost-benefit balance |
| More branch candidates | GAIA 49.6, XBench 28.0 | Saturated gains, increased cost |
| PRM (Claude-3.7-Sonnet) | CSO 61.5, Step-level BoN 56.2 | CSO outperforms direct PRM action selection |
| PRM (GPT-4.1) | CSO 53.3, Step-level BoN 48.7 | PRM quality matters, but verification mitigates noise |
| Extra tokens per round | CSO ~168M, Step-DPO ~141M, ETO ~212M | 19% higher than Step-DPO but much lower than ETO |

### Key Findings
- CSO improved GAIA-Text-103 score from SFT (35.9) to 49.5 (~37% relative gain) and XBench from 23.0 to 29.0 (~26% relative gain).
- The 8B open-source agent with CSO reached 49.5 on GAIA All, surpassing GPT-4.1 (45.6), demonstrating that critical step post-training significantly amplifies the capabilities of smaller model agents.
- IPR also utilizes outcome-grounded step-level signals but propagates the result to more steps. CSO's focus on sparse verified critical steps resulted in a 5.0 GAIA All point lead over IPR.
- Error analysis shows broad distribution of critical steps: Tool errors (26.1%), Reasoning (25.1%), Others (24.1%), Understanding (13.0%), and Extraction (11.7%). This confirms CSO identifies semantic decision points rather than fixed positions.

## Highlights & Insights
- The strongest aspect of the paper is demoting the PRM from "direct supervisor" to "candidate recaller." This mirrors a high-recall first stage in retrieval: allowing PRM noise as long as it is filtered by real outcome verification.
- Starting from failed trajectories is inherently suited for agents. Errors are often bound to specific frameworks or tool formats; fixing the policy where it specifically fails is more precise than generalized learning from expert successes.
- The "reachability" design is crucial: subsequent steps in branch rollouts are executed by the policy itself. This ensures the positive sample is not an unattainable expert trajectory, but one the policy can actually complete given a better starting point.
- The CSO philosophy can be transferred to code, web, or scientific agents: dense rewards are unnecessary if the few outcome-changing tool calls or queries are identified.

## Limitations & Future Work
- Outcome verification requires executing branches until task completion, which is time-consuming in complex online environments. While token costs are manageable, real-time tool costs may be higher.
- Main experiments rely on Claude-3.7-Sonnet as the expert/PRM. Although GPT-4.1 and Qwen3-235B show improvements, the strongest results depend on closed models.
- The method requires a reliable final correctness judge. For open-ended or subjective tasks without gold answers, defining a verified outcome is challenging.
- The PRM and policy are not jointly trained; as the policy evolves, the PRM rubrics may require dynamic adaptation.

## Related Work & Insights
- **vs Trajectory-level ETO/DPO**: ETO performs preference learning on whole trajectories with coarse credit assignment; CSO focuses on verified critical steps to avoid penalizing irrelevant actions.
- **vs Step-DPO / AgentRPM**: Step-level methods rely on PRM scores for every step, which are granular but noisy. CSO uses PRM for candidates and outcomes for confirmation.
- **vs IPR**: IPR uses outcome verification for step signals but may pollute non-critical steps; CSO uses a triple condition (low policy score + high expert score + branch success) to keep signals sparse.
- **vs Online RLVR**: RLVR is closer to the policy distribution but suffers from high rollout costs and sparse rewards; CSO achieves stable data efficiency via offline/semi-online DPO.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The verified critical step as a training unit is clear, combining PRM, branch rollout, and DPO innovatively.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Main experiments, data source ablations, and PRM analyses are comprehensive; open-ended task validation is still needed.
- Writing Quality: ⭐⭐⭐⭐☆ Problem decomposition and method flow are clear; some dependency on external agent frameworks makes replication difficult.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for credit assignment in long-horizon agents, particularly for systems requiring tool integration and deep search.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)
- [\[ACL 2026\] SEARL: Joint Optimization of Policy and Tool Graph Memory for Self-Evolving Agents](searl_joint_optimization_of_policy_and_tool_graph_memory_for_self-evolving_agent.md)
- [\[ACL 2026\] Agent-GWO: Collaborative Agents for Dynamic Prompt Optimization in Large Language Models](agent-gwo_collaborative_agents_for_dynamic_prompt_optimization_in_large_language.md)
- [\[AAAI 2026\] DEPO: Dual-Efficiency Preference Optimization for LLM Agents](../../AAAI2026/llm_agent/depo_dual-efficiency_preference_optimization_for_llm_agents.md)
- [\[ACL 2026\] BAPO: Boundary-Aware Policy Optimization for Reliable Agentic Search](bapo_boundary-aware_policy_optimization_for_reliable_agentic_search.md)

</div>

<!-- RELATED:END -->
