---
title: >-
  [Paper Note] Temp-R1: A Unified Autonomous Agent for Complex Temporal KGQA via Reverse Curriculum Reinforcement Learning
description: >-
  [ACL 2026][LLM Agent][Temporal KGQA] Temp-R1 transforms Temporal Knowledge Graph Question Answering (TKGQA) from a manually designed fixed prompt workflow into an autonomous agent trainable via reinforcement learning. By…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Temporal KGQA"
  - "Autonomous Agent"
  - "GRPO"
  - "Reverse Curriculum"
  - "Tool Use"
date: 2026-05-08
content_hash: e69c7adfd0128d33
---

# Temp-R1: A Unified Autonomous Agent for Complex Temporal KGQA via Reverse Curriculum Reinforcement Learning

**Conference**: ACL 2026  
**arXiv**: [2601.18296](https://arxiv.org/abs/2601.18296)  
**Code**: https://github.com/zjukg/Temp-R1  
**Area**: LLM Agent / Temporal KGQA  
**Keywords**: Temporal KGQA, Autonomous Agent, GRPO, Reverse Curriculum, Tool Use

## TL;DR
Temp-R1 transforms Temporal Knowledge Graph Question Answering (TKGQA) from a manually designed fixed prompt workflow into an autonomous agent trainable via reinforcement learning. By employing explicit internal actions, SFT cold-start, GRPO, and a "hard-first" reverse curriculum, it outperforms several strong baselines driven by GPT-4o/DeepSeek-V3 on an 8B open-source model.

## Background & Motivation
**Background**: Temporal Knowledge Graph Question Answering (TKGQA) requires models to answer questions based on fact quadruplets containing timestamps. This involves retrieving entity relations while handling sequential orders, time intervals, multi-hop dependencies, and answer granularity. Recent LLM methods typically decompose the task into fixed modules like decomposer, planner, retriever, and generator, linked by multiple carefully designed prompts.

**Limitations of Prior Work**: While fixed workflows perform well in the short term, they suffer from high costs and poor flexibility. On one hand, they frequently rely on closed-source APIs like GPT-4o or DeepSeek-V3, where multi-round calls for complex questions escalate costs. On the other hand, manually prescribed processes restrict the model's exploration paths, making it difficult to dynamically adjust reasoning steps for different question types.

**Key Challenge**: TKGQA necessitates open-ended, multi-step, variable-length temporal reasoning, yet existing methods lock the reasoning process within fixed prompt templates. Although standard ReAct-style agents possess tool-calling capabilities, they tend to cram all internal reasoning into a single `<think>` tag, leading to heavy cognitive load under complex temporal constraints.

**Goal**: The authors aim to train a small open-source model that can autonomously determine when to plan, retrieve, filter temporal constraints, rank facts, and output answers, while avoiding being misled by simple questions during early RL training.

**Key Insight**: The paper formulates TKGQA as an MDP: the state consists of the original question and the historical interaction trajectory, while actions include both external `<search>` and internal `<plan>`, `<filter>`, and `<rank>`, terminating with `<answer>`. Consequently, temporal reasoning is no longer a hard-coded prompt process but an action sequence learnable by a policy model.

**Core Idea**: Use a small number of high-quality trajectories to teach the model valid action formats, then train with GRPO under verifiable answer rewards, utilizing a "hard-first" reverse curriculum to compel the model to learn complex tool chains before migrating to simple problems.

## Method
The core of Temp-R1 is not a new retriever, but a shift in the control paradigm of TKGQA. Traditional methods are usually "decompose-then-retrieve-then-generate" pipelines where the model only fills content into preset slots; Temp-R1 allows the model to choose actions itself. It can first plan the question type and temporal constraints, then search the TKG, subsequently filter facts that do not meet temporal conditions, rank facts by time if necessary, and finally provide an answer. This process preserves the interpretability of tool use while allowing different questions to follow different trajectories.

### Overall Architecture
The paper defines a temporal KG where facts are $(s,p,o,t)$ quadruplets ($s/o$ are entities, $p$ is a relation, $t$ is a timestamp). TKGQA aims to infer entity or time answers based on a question $q$ and relevant temporal facts.

In the agent MDP, the state is $s_t=(q,h_t)$, where $h_t$ records actions and observations up to the current step. The action space is divided into internal and external actions. Internal actions include `<plan>`, `<filter>`, and `<rank>`, responsible for initial planning, filtering facts by semantic relations/temporal constraints, and temporal sorting, respectively. External actions include `<search>`, which calls the TKG retriever, and `<answer>`, which terminates the episode. The training pipeline consists of three steps: rollout to form executable trajectories, SFT cold-start to learn formats and basic policies, and finally RL optimization using GRPO with a reverse curriculum.

### Key Designs
1. **Autonomous Rollout Loop with Expanded Action Space**:
	- **Function**: Decomposes complex temporal reasoning into observable and optimizable action sequences.
	- **Mechanism**: Every response must start with `<plan>`. The model can alternate between `<search>` to fetch facts, `<filter>` to process temporal/semantic constraints, and `<rank>` to sort candidate facts by time, eventually outputting the final answer in `<answer>`.
	- **Design Motivation**: A single `<think>` tag causes all reasoning to occur implicitly, which leads to confusion with complex temporal constraints; explicit actions allow RL to separately reward tool calling, filtering, and ranking strategies.

2. **SFT Cold-Start + Masked Loss**:
	- **Function**: Prepares the base model to learn valid label formats and basic tool chains before entering RL exploration.
	- **Mechanism**: The authors use GPT-4o to construct approximately 1,000 high-quality $(q,\tau_{gold})$ trajectories, filtering for structural correctness and semantic answer accuracy. During SFT, cross-entropy is calculated only for tokens generated by the agent, while system prompts, user questions, and retrieval observations are masked.
	- **Design Motivation**: If RL is applied directly from a base model, the model tends to output invalid labels, meaningless searches, or chaotic reasoning; cold-start provides a stable initial policy to avoid KL explosion and training collapse.

3. **GRPO and Reverse Curriculum Learning**:
	- **Function**: Encourages the discovery of more flexible temporal reasoning strategies using verifiable answer rewards and prevents shortcuts caused by simple questions.
	- **Mechanism**: A group of trajectories is sampled for each question. A binary terminal reward is assigned based on whether the answer matches the gold answer. Advantages are normalized using the group mean and standard deviation to optimize the GRPO objective with KL regularization. The training data schedule follows a hard-first approach: early stages involve only complex multi-hop/multiple-constraint questions, with simple questions added only after a threshold is met.
	- **Design Motivation**: A standard easy-first curriculum would lead the model to learn shortcuts like `<search>`→`<answer>` early on, failing to actively combine actions like `<filter>`/`<rank>` for complex problems. Training on difficult problems first forces the agent to master advanced tool chains.

### Loss & Training
The SFT phase optimizes masked cross-entropy, applying loss only on tokens the model should generate. The RL phase uses GRPO, with the trajectory ratio $\rho_i(\theta)=\frac{\pi_\theta(\tau_i|q)}{\pi_{old}(\tau_i|q)}$ and advantage $\hat A_i=\frac{r_i-mean(\{r_k\})}{std(\{r_k\})+\eta}$, where the reward $r_i$ is a binary signal for exact answer matches. Implementation-wise, the authors fine-tuned Llama3.1-8B-Instruct using an E5 retriever similar to Search-R1, with GRPO trained on approximately 9% of the unlabeled QA pairs from the MultiTQ training set.

## Key Experimental Results

### Main Results
MultiTQ is the primary experiment. Temp-R1 is compared against embedding methods, prompt-based LLM workflows, and fine-tuning-based LLM methods using the Hits@1 metric.

| Method Type | Method | Overall | Multiple | Single | Entity | Time |
|----------|------|---------|----------|--------|--------|------|
| TKG embedding | EmbedKGQA | 0.206 | 0.134 | 0.235 | 0.290 | 0.001 |
| TKG embedding | MultiQA | 0.293 | 0.159 | 0.347 | 0.349 | 0.157 |
| Prompt-based LLM | TempAgent | 0.702 | 0.316 | 0.857 | 0.624 | 0.870 |
| Prompt-based LLM | RTQA | 0.765 | 0.424 | 0.902 | 0.692 | 0.942 |
| FineTune-based LLM | TimeR4 | 0.728 | 0.335 | 0.887 | 0.639 | 0.945 |
| FineTune-based LLM | PoK | 0.779 | 0.409 | 0.929 | 0.696 | 0.962 |
| Temp-R1 | Temp-R1 | 0.780 | 0.550 | 0.888 | 0.714 | 0.969 |

In terms of overall scores, Temp-R1 is slightly higher than PoK (0.780 vs. 0.779). Importantly, it reaches 0.550 on complex multiple questions, significantly outperforming PoK (0.409), RTQA (0.424), and MemoTime (0.459). The 19.8% improvement emphasized in the abstract primarily comes from the complex question category, indicating that autonomous action sequences effectively mitigate the rigidity of fixed workflows under complex constraints.

| Dataset | Method | Overall | Simple | Medium | Complex | Note |
|--------|------|---------|--------|--------|---------|------|
| TimelineKGQA-Cron | RAG Baseline | 0.235 | 0.704 | 0.092 | 0.009 | Hits on simple, fails on complex |
| TimelineKGQA-Cron | GPT-4o | 0.206 | 0.069 | 0.130 | 0.376 | Zero-shot closed-source unfit for TKGQA |
| TimelineKGQA-Cron | PoK | 0.651 | 0.737 | 0.539 | 0.683 | Strong FT baseline |
| TimelineKGQA-Cron | Temp-R1 | 0.705 | 0.960 | 0.486 | 0.672 | Best overall, far ahead in simple |
| TimelineKGQA-ICEWS Actor | GPT-4o | 0.113 | 0.051 | 0.035 | 0.353 | Poor OOD performance |
| TimelineKGQA-ICEWS Actor | PoK | 0.602 | 0.744 | 0.456 | 0.578 | Strong OOD baseline |
| TimelineKGQA-ICEWS Actor | Temp-R1 | 0.642 | 0.866 | 0.388 | 0.595 | Leads in OOD overall |

TimelineKGQA results demonstrate that Temp-R1 does not just overfit MultiTQ. Particularly in the out-of-domain (OOD) ICEWS Actor setting, its overall score of 0.642 exceeds PoK (0.602), while GPT-4o only achieved 0.113, showing that small open-source agents trained via task-specific RL can be more stable than general-purpose closed-source models.

### Ablation Study
The authors progressively removed internal actions, the reverse curriculum, and the SFT cold-start on MultiTQ.

| Configuration | Overall | Multiple | Single | Entity | Time | Main Conclusion |
|------|---------|----------|--------|--------|------|----------|
| Temp-R1 | 0.780 | 0.550 | 0.888 | 0.714 | 0.969 | Full Model |
| w/o internal actions | 0.620 | 0.388 | 0.729 | 0.563 | 0.783 | Reasoning burden returns to implicit `<think>` |
| w/o Reverse CL | 0.556 | 0.143 | 0.750 | 0.447 | 0.868 | Largest drop, complex questions fail |
| w/o SFT | 0.582 | 0.325 | 0.703 | 0.536 | 0.713 | Difficulty in learning stable formats/tools |

### Key Findings
- Reverse curriculum is the most critical training strategy. Removing it dropped the overall score from 0.780 to 0.556 and the Multiple category from 0.550 to 0.143, indicating that easy-first or mixed training causes the model to fall into simple retrieval shortcuts.
- Internal actions are capability carriers, not just format fillers. Removing `<filter>`/`<rank>` still allowed searching, but temporal constraint processing significantly deteriorated.
- Model scale remains important. Qwen 7B reached a peak accuracy of ~0.790, while the 1.5B version only reached 0.532. However, both Llama and Qwen architectures benefited consistently, showing the framework is not dependent on a specific backbone.
- Trajectory complexity adaptively increases with question difficulty. In complex questions, `<think>` tags increased on average from 1.36 to 2.93 and `<search>` from 1.33 to 1.92, meeting agentic expectations for difficult tasks.

## Highlights & Insights
- The most valuable contribution of this paper is transforming the TKGQA prompt pipeline into a policy learning problem. Instead of hard-coding modules, it allows the model to learn when to use which internal action.
- Hard-first reverse curriculum is highly suitable for tool-using agents. Many agent tasks have a structure where "simple samples can be solved by shortcuts, but complex samples require tool combinations." Training on hard problems first mitigates shortcut learning.
- While simple, binary answer rewards are ideal for verifiable tasks like KGQA. They avoid reward model costs and allow GRPO to directly optimize final answer correctness.

## Limitations & Future Work
- The largest experiment used an 8B model; 14B or larger models were not verified. While larger models might further improve complex temporal reasoning, training costs and trajectory lengths would also increase.
- Whether the reverse curriculum is applicable to non-TKGQA tasks remains unproven. For tasks without clear easy/hard boundaries or those with high noise in hard samples, hard-first might lead to early training instability.
- Rewards only consider the final answer, failing to distinguish between "correct process but wrong answer format" and "wrong process with a lucky guess." Future work could incorporate trajectory-level rewards.
- The current retriever and datasets are fixed. Real-world open environments with incomplete TKGs, conflicting facts, and inconsistent temporal granularity will be more complex.

## Related Work & Insights
- **vs TempAgent / MemoTime / RTQA**: These rely on manually designed decomposition and generation, whereas Temp-R1 learns dynamic trajectories through RL, offering greater flexibility.
- **vs Search-R1**: Search-R1 proved RL + search is effective for QA; Temp-R1 extends this to TKGQA and introduces temporal-specific actions like `<filter>`/`<rank>`.
- **vs PoK / TimeR4**: PoK and TimeR4 are strong fine-tuning baselines for TKGQA. Temp-R1 trains a complete agent policy rather than just fine-tuning answer generation, resulting in more stable OOD performance.
- **Insight**: For verifiable structured reasoning tasks, explicit "task-specific intermediate operations" can be made into action tokens. Small-scale SFT cold-start followed by RL provides a better approach than cramming all logic into natural language chain-of-thought.

## Rating
- Novelty: ⭐⭐⭐⭐ Systematically combines GRPO, autonomous tool calling, and reverse curriculum for TKGQA with high task adaptability.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes main results, OOD, ablation, backbone/scale, and trajectory analysis; however, lacks validation on larger models and noisy real-world KGs.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with rich tables, though some training details in the appendix require cross-referencing for reproduction.
- Value: ⭐⭐⭐⭐ Valuable reference for both temporal KGQA and verifiable agent training, particularly the transferable experience of hard-first curriculum.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Hierarchical Reinforcement Learning with Augmented Step-Level Transitions for LLM Agents](hierarchical_reinforcement_learning_with_augmented_step-level_transitions_for_ll.md)
- [\[ACL 2026\] SOLAR-RL: Semi-Online Long-horizon Assignment Reinforcement Learning](solar-rl_semi-online_long-horizon_assignment_reinforcement_learning.md)
- [\[ACL 2026\] Robust Tool Use via Fission-GRPO: Learning to Recover from Execution Errors](robust_tool_use_via_fission-grpo_learning_to_recover_from_execution_errors.md)
- [\[AAAI 2026\] MoralReason: Generalizable Moral Decision Alignment For LLM Agents Using Reasoning-Level Reinforcement Learning](../../AAAI2026/llm_agent/moralreason_generalizable_moral_decision_alignment_for_llm_agents_using_reasonin.md)
- [\[ICLR 2026\] Reducing Belief Deviation in Reinforcement Learning for Active Reasoning of LLM Agents](../../ICLR2026/llm_agent/reducing_belief_deviation_in_reinforcement_learning_for_active_reasoning.md)

</div>

<!-- RELATED:END -->
