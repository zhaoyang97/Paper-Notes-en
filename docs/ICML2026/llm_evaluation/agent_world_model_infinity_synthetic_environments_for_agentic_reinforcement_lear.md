---
title: >-
  [Paper Note] Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning
description: >-
  [ICML2026][LLM Evaluation][Agent Environment Synthesis] Ours proposes Agent World Model, a fully synthetic pipeline from scenarios, tasks, databases, and MCP tool interfaces to verifiers. It generates 1…
tags:
  - "ICML2026"
  - "LLM Evaluation"
  - "Agent Environment Synthesis"
  - "Tool-use"
  - "MCP"
  - "Reinforcement Learning"
  - "Executable World Model"
date: 2026-05-08
content_hash: f3855170412e61d3
---

# Agent World Model: Infinity Synthetic Environments for Agentic Reinforcement Learning

**Conference**: ICML2026  
**arXiv**: [2602.10090](https://arxiv.org/abs/2602.10090)  
**Code**: https://github.com/Snowflake-Labs/agent-world-model  
**Area**: LLM Agent / Reinforcement Learning  
**Keywords**: Agent Environment Synthesis, Tool-use, MCP, Reinforcement Learning, Executable World Model  

## TL;DR
Ours proposes Agent World Model, a fully synthetic pipeline from scenarios, tasks, databases, and MCP tool interfaces to verifiers. It generates 1,000 executable database-driven environments and uses them to train tool-calling agents, achieving stronger out-of-distribution generalization on BFCLv3, $\tau^2$-bench, and MCP-Universe.

## Background & Motivation

**Background**: LLM Agents have become capable of multi-turn tool calling, web operations, and complex task planning. However, the training bottleneck for these agents is increasingly shifting from "models cannot call tools" to a "lack of sufficient, resettable, parallelizable, and verifiable interactive environments." Existing benchmarks are often small-scale, real APIs are difficult to reproduce stably, and LLM-simulated environments, while easy to generate, produce hallucinatory state transitions.

**Limitations of Prior Work**: Agentic RL requires thousands of interactions; environments must support concurrent instances, reliable resets, state consistency, and automated rewards. Real services usually do not open the APIs required for training and cannot withstand large-scale trial-and-error. Manual environments such as $\tau^2$-bench or TheMCPCompany contain only a few scenarios. LLM simulators require model calls at every step, which is both expensive and prone to self-contradictory state updates.

**Key Challenge**: Tool-calling agents requires real executable environments to learn long-term interactions, but real-world environments cannot be scaled, and pure LLM simulations are not reliable enough. While training data synthesis is abundant, the real deficiency lies in the synthesis of the "environment itself."

**Goal**: Ours aims to construct an open environment synthesis pipeline that can automatically generate a large number of executable environments with database states, tool interfaces, and task verifiers from a small set of scenario seeds, proving that these environments can be directly used for large-scale online RL.

**Key Insight**: This paper treats the agent environment as a software system rather than delegating it to an LLM for step-by-step simulation. An executable application typically consists of requirements, a database, interfaces, backend code, and tests/verification. By synthesizing these components sequentially, a "programmatic world model" is obtained where state transitions are determined by code and SQL constraints.

**Core Idea**: Use a software engineering pipeline to synthesize database-driven MCP environments, transforming the world model from a neural predictor into an executable code sandbox, followed by large-scale agentic RL within these sandboxes.

## Method

### Overall Architecture

Agent World Model formalizes each environment as a POMDP. The database defines the state space $\mathcal{S}_{E_i}$, the MCP tool interface defines the action space $\mathcal{A}_{E_i}$, observation space $\mathcal{O}_{E_i}$, and transition function $T_{E_i}$. Each user task $\tau$ corresponds to a reward function $R_\tau$. The Agent can only interact through unified MCP tools and cannot directly modify the database.

The pipeline starts from 100 popular website/application seeds, first expanding into scenarios suitable for CRUD operations, and then generating 10 user tasks for each scenario. Tasks are not by-products but functional requirements for subsequent database schemas, sample data, tool interfaces, and verifiers. After this, the LLM sequentially synthesizes the SQLite database, sample data, interface specifications, Python MCP service code, and task verification functions. Each step executes the generated result; if the code or SQL fails, the error summary is fed back to the LLM for self-correction, with up to 5 retries.

The generated environments are used for online GRPO training. During training, 1,024 isolated environment instances are launched at each step, each with an independent SQLite database copy. After rollout, a "code verification signal + LLM-as-a-Judge" gives a status of Completed, Partially Completed, Agent Error, or Environment Error, which is then mapped to a reward.

### Key Designs

1. **Requirement-Driven Environment Synthesis Chain**:

	- **Function**: Automatically generate executable tool environments from high-level scenarios instead of just generating tasks or trajectories.
	- **Mechanism**: Screen for stateful, CRUD-suitable scenarios first, then generate specific user tasks; tasks in turn constrain which tables the database needs, which endpoints the interfaces need, and which preconditions the sample data must satisfy. SQLite is used for the database, interfaces are exposed as tools via MCP, and backend code is responsible for reading and writing states.
	- **Design Motivation**: If an LLM is directly asked to write a complete environment, inconsistencies between schemas, tools, and tasks often occur. Synthesizing in the order of "Task $\rightarrow$ Database $\rightarrow$ Interface $\rightarrow$ Code" essentially aligns each component using requirements.

2. **Code-Augmented Verification and Hybrid Rewards**:

	- **Function**: Provide reliable task rewards for synthetic environments used in RL.
	- **Mechanism**: Each task generates a verification function that compares the database state before and after execution, extracting changed records, expected results, and diagnostic signals. The final judgment is completed by LLM-as-a-Judge combining structured verification signals and the Agent's trajectory. Rewards are set to 1.0 for Completed, 0.1 for Partially Completed, and 0 otherwise; format errors result in early termination and a -1 reward.
	- **Design Motivation**: Pure code verification is too brittle and may misjudge due to environmental flaws; pure LLM judgment lacks state grounding. Combining the two reduces reward noise while maintaining automation.

3. **History-Aligned Training for Tool-Calling RL**:

	- **Function**: Reduce distribution mismatch between full history during training and truncated history during inference.
	- **Mechanism**: In actual deployment, Agents often use a sliding window to retain the most recent rounds of history. In GRPO optimization, trajectories are also split by window $w=3$, so that the loss for each action $a_t$ is only conditioned on the truncated history $h_t^{trunc}$ rather than forward-passing the entire long trajectory at once.
	- **Design Motivation**: Multi-turn Agent trajectories are long, and it is impossible to retain infinite context during inference. If training always sees the full history, the model will learn information that is unavailable at deployment. Incorporating history management into the training objective improves stability and generalization.

### Loss & Training

Training employs GRPO. For each task, a group of rollouts is sampled, advantages $A^{(k)}=(R^{(k)}-\bar{R})/\sigma_R$ are calculated based on group rewards, and the log-probability of each action under the truncated history is optimized. In implementation, Qwen3 thinking 4B, 8B, and 14B are chosen as Agents. The training subset includes 526 AWM environments and 3,315 tasks, with a maximum of 96 optimization steps, a learning rate of $7\times10^{-7}$, a batch size of 64, and 16 rollouts per task, totaling 1,024 parallel environment instances per step.

To unify different environments, the Agent only sees two meta-tools: `list_tools` to list tools in the current MCP environment, and `call_tool` to call a specific tool by name and JSON parameters. Format validation is added during training: `list_tools` must be called once first, tool names and parameters must be valid, and inference messages must conform to the Qwen3 tool-calling format. Format errors result in immediate termination to avoid wasting environment steps on invalid trajectories.

## Key Experimental Results

### Main Results

| Benchmark | Model | Base | Simulator | EnvScaler | AWM | Main Conclusion |
|-----------|------|------|-----------|-----------|-----|----------|
| BFCLv3 Overall | Qwen3-4B | 54.92 | 55.52 | 54.06 | 64.50 | AWM significantly improves overall function-calling capabilities |
| BFCLv3 Overall | Qwen3-8B | 53.83 | 52.53 | 36.83 | 65.94 | 12.11 point Gain relative to Base on 8B |
| BFCLv3 Overall | Qwen3-14B | 61.25 | 67.68 | - | 70.18 | Still outperforms LLM simulation training on larger models |
| $\tau^2$-bench Pass@1 | Qwen3-8B | 26.44 | 31.30 | 39.39 | 33.45 | AWM exceeds Base and Simulator despite not targeting dialog tasks |
| MCP-Universe Overall | Qwen3-14B | 8.38 | 10.62 | - | 12.29 | Best generalization on real MCP server-type tasks |

### Ablation Study

| Analysis Item | Setting | Key Metric | Description |
|------|------|---------|------|
| Synthesis Scale | 1000 Envs, 10000 Tasks | Avg 35.1 tools, 1984.7 LOC, 18.5 DB tables | Environment complexity is far higher than toy tasks, suitable for multi-turn tool training |
| Synthesis Success Rate | GPT-5 Generation | DB 88.3%, Data 88.2%, Code 86.8%; Avg 1.13 fixes | Execution feedback fixes most shallow generation errors |
| Complexity Binning | BFCLv3 / $\tau^2$ 8B | BFCLv3 Simple: Base 53.6 $\rightarrow$ AWM 80.3; Med: 60.0 $\rightarrow$ 75.3; Hard: 43.9 $\rightarrow$ 45.0 | AWM yields the largest gains for simple and medium multi-tool tasks; hard tasks are still limited by base model capacity |
| Verification Strategy | LLM-only / Code-only / Augmented | 8B BFCLv3: 55.46 / 60.00 / 65.94; $\tau^2$ P@1: 26.44 / 29.59 / 33.45 | Code-augmented Judge outperforms both pure LLM and pure code verification |
| History Alignment | 4B w/ HL vs w/o HL | Aligned: BFCLv3 64.50 vs 55.35; $\tau^2$ P@1 22.57 vs 15.92 | Using the same history truncation during training is significantly better |

### Key Findings
- AWM provides the most stable improvements on BFCLv3, indicating that code-driven, multi-tool, and multi-state synthetic environments translate well to function-calling benchmarks.
- On $\tau^2$-bench, AWM is not as high as EnvScaler's 8B Pass@1, but it does not regress on BFCLv3 and MCP-Universe simultaneously; the authors suggest EnvScaler relies on existing task sets that may be closer to the $\tau^2$ distribution.
- The overall scores on MCP-Universe remain low, indicating that real MCP tasks are difficult; however, AWM brings visible improvements in subcategories like Finance, Location, and Browser, proving it hasn't only learned formats internal to synthetic environments.
- Environment quality analysis shows that AWM's Task Feasibility, Data Alignment, and Toolset Completeness are all higher than those of EnvScaler, though many auto-generated code bugs remain. The key is not zero bugs but a significantly lower proportion of blocked tasks, preventing the RL process from being polluted by large numbers of non-executable tasks.

## Highlights & Insights
- The most valuable contribution of the paper is grounding "world models" in executable software environments rather than letting neural networks or LLMs imagine state transitions step-by-step. For tool Agents, databases and backend code are inherently the world dynamics; this modeling perspective is very natural.
- Requirement-driven synthesis is critical. Having tasks before schemas and tools avoids many common problems in synthetic environments: tools that are numerous but unused by tasks, tasks that mention entities not present in the database, or interface returns inconsistent with verifiers.
- Code-augmented Judge is a pragmatic compromise. Synthesis environment verifiers cannot be perfect, especially given boundary bugs in auto-generated code; letting an LLM reference database diffs and trajectories for final judgment is more suitable for RL rewards than relying solely on one side.
- History-aligned training reminds us of a frequently overlooked issue: context management in Agent frameworks is not just an inference trick; it changes the distribution of information seen by the policy and should be integrated into the training loop.

## Limitations & Future Work
- AWM mainly covers CRUD-type, database-driven applications and is not suitable for tasks with strong UI dependencies, real-time data, complex visual interactions, or those requiring real external service states.
- Synthetic environments still contain code bugs, with a non-negligible proportion of sampled environments containing bugs. Although blocked tasks are fewer than in EnvScaler, long-term RL may still learn biases caused by environmental flaws.
- Synthesis and generation costs are relatively high. The paper uses the GPT-5 generation pipeline and Judge; while Claude and Qwen3.5 generators were evaluated, fully replacing closed-source models with open ones requires more validation.
- Task verification relies on LLM-as-a-Judge, and rewards may still be affected by Judge bias. Future work could explore more provable state-diff specifications, automated test generation, and few-shot calibration via human audit.
- There is still a semantic gap between AWM environments and real web/enterprise systems. In the future, real API documents, open-source application backends, or synthetic UIs could be combined to form a multimodal Agent training environment spanning database tools to visual operations.

## Related Work & Insights
- **vs LLM Simulated Environments**: LLM simulators generate state transitions at every step, which is hallucination-prone and expensive; AWM uses Python code and SQLite to execute transitions, which is more stable and suitable for large-scale RL.
- **vs EnvScaler**: EnvScaler also generates programming environments but relies on existing task sets and has a scale of 191 environments; AWM synthesizes 1,000 environments and 35,062 tools starting from scenario names, with SQL-backed state consistency.
- **vs $\tau^2$-bench / TheMCPCompany**: These benchmarks are more like evaluation environments with limited quantities and heavy reliance on manual design; the goal of AWM is a training environment pool, emphasizing parallel instances, automatic resets, and scalable generation.
- **vs ToolLLM / Gorilla etc.**: These methods primarily synthesize API documents, calling trajectories, or supervised data; AWM synthesizes full environments that allow Agents to learn through trial and error and receive rewards.
- **vs Traditional Model-Based RL World Models**: Traditional world models learn environment dynamics; AWM does not learn neural dynamics but automatically generates programmatic dynamics, sacrificing some realism for controllability and scalability.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Environment synthesis has parallel works, but the complete open pipeline spanning tasks, SQL, MCP, verifiers, and agentic RL is very thorough.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Includes three major OOD benchmarks, three model scales, synthesis quality, complexity, verification strategies, history truncation, and scaling curves, providing very strong evidence.
- Writing Quality: ⭐⭐⭐⭐☆ The methodology structure is clear, and engineering details are sufficient; a slight disadvantage is that some core quality issues are scattered in the appendix.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for Agent RL training infrastructure, especially for research and open-source replication requiring large numbers of resettable tool environments.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Beyond Trajectory-Level Attribution: Graph-Based Credit Assignment for Agentic Reinforcement Learning](beyond_trajectory-level_attribution_graph-based_credit_assignment_for_agentic_re.md)
- [\[ICML 2026\] HiPER: Hierarchical Reinforcement Learning with Explicit Credit Assignment for Large Language Model Agents](hiper_hierarchical_reinforcement_learning_with_explicit_credit_assignment_for_la.md)
- [\[ICML 2026\] Multi$^2$: Hierarchical Multi-Agent Decision-Making with LLM-Based Agents in Interactive Environments](multi2_hierarchical_multi-agent_decision-making_with_llm-based_agents_in_interac.md)
- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](../../ACL2026/llm_evaluation/multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)
- [\[ICLR 2026\] UIS-Digger: Towards Comprehensive Research Agent Systems for Real-world Unindexed Information Seeking](../../ICLR2026/llm_evaluation/uis-digger_towards_comprehensive_research_agent_systems_for_real-world_unindexed.md)

</div>

<!-- RELATED:END -->
