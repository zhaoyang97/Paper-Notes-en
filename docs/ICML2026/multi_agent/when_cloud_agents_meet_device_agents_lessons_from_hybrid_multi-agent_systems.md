---
title: >-
  [Paper Note] When Cloud Agents Meet Device Agents: Lessons from Hybrid Multi-Agent Systems
description: >-
  [ICML2026][Multi-Agent][Hybrid Multi-Agent] This paper systematically investigates hybrid multi-agent systems composed of a cloud-based GPT-4o supervisor and on-device Qwen3 executors. It finds that PEVR and EVA offer di…
tags:
  - "ICML2026"
  - "Multi-Agent"
  - "Hybrid Multi-Agent"
  - "Cloud Models"
  - "On-device Models"
  - "Agent Evaluation"
  - "Context Efficiency"
date: 2026-05-08
content_hash: 4a22cb427b32d06e
---

# When Cloud Agents Meet Device Agents: Lessons from Hybrid Multi-Agent Systems

**Conference**: ICML2026  
**arXiv**: [2605.30102](https://arxiv.org/abs/2605.30102)  
**Code**: None  
**Area**: multi_agent  
**Keywords**: Hybrid Multi-Agent, Cloud Models, On-device Models, Agent Evaluation, Context Efficiency

## TL;DR
This paper systematically investigates hybrid multi-agent systems composed of a cloud-based GPT-4o supervisor and on-device Qwen3 executors. It finds that PEVR and EVA offer distinct advantages for UI assistance and deep search, respectively; more cloud intervention is not necessarily better; and context resets combined with summarization can significantly improve costs and KV-cache pressure for long-duration tasks on-device.

## Background & Motivation
**Background**: LLM agents are evolving from short dialogues to long-horizon task execution, requiring goal decomposition, tool calling, state maintenance, and multi-step actions within an environment. The most powerful frontier LLMs are typically deployed in the cloud, offering high capability but significant token costs. Smaller SLMs can run on phones or laptops, providing lower costs and better privacy but limited long-context and complex reasoning capabilities.

**Limitations of Prior Work**: Hybrid AI often selects between large and small models via a router or escalates to a large model when a small model fails. However, in agent scenarios, computation involves more than just "which model answers"; it also concerns who plans, executes, verifies, and when to restart the context. Existing systems are mostly designed ad-hoc for single tasks and lack systematic comparisons across tasks and cost dimensions.

**Key Challenge**: Theoretically, more intervention from cloud models provides stronger supervision but increases API costs and may frequently interrupt on-device execution. Continual on-device execution saves money but leads to KV-cache growth, context corruption, and error accumulation. The challenge for hybrid MAS is finding a Pareto balance between accuracy, cloud expenses, and on-device energy consumption.

**Goal**: The authors aim to adapt representative multi-agent architectures into cloud-edge hybrid versions and systematically evaluate the impact of model labor division, supervision frequency, restart methods, and context management on performance and cost across three task types: HotpotQA, FanOutQA, and AppWorld.

**Key Insight**: The paper treats the cloud GPT-4o as an intermittent Supervisor and the on-device Qwen3 (4B/8B/14B/32B) as a long-term Executor. In this way, the token-intensive ReAct execution remains on-device, while the expensive cloud model intervenes only for planning, verification, or advice.

**Core Idea**: View hybrid agent design as a multi-agent role assignment problem rather than simple model routing. By comparing the PEVR and EVA structures, the authors identify the boundaries where "planned supervision" and "advisory summarization" are respectively suitable under different tasks.

## Method
The paper does not propose a new agent benchmark but performs systematic experiments around the design space of hybrid MAS. Key variables include the choice between PEVR or EVA architectures, the size of the on-device Qwen3 Executor, the frequency of cloud Supervisor verification, and whether to re-plan or provide advice with a context reset after an execution failure.

### Overall Architecture
The system consists of two roles. The Executor is a small on-device model responsible for the continuous ReAct loop: generating reasoning/actions based on the current task, context, and available tools, calling the environment, and collecting observations. The Supervisor is cloud-based GPT-4o, which does not directly execute every tool but periodically reviews the trajectory to decide if intervention is needed. Users control the cloud intervention frequency via a verification interval; a smaller interval results in more frequent cloud calls and higher costs.

Experiments cover three task families of increasing difficulty. HotpotQA is short-range multi-hop QA (reporting ROUGE-1 F1); FanOutQA is long-range fan-out information aggregation (reporting ROUGE-1 F1); AppWorld is a stateful API environment (reporting Test Pass Ratio and Task Success). Efficiency is reported through cloud API costs in USD, estimated on-device energy consumption, and maximum KV-cache footprint during long tasks.

### Key Designs
1. **PEVR: Plan-Execute-Verify-Replan**:

	- **Function**: Allows the cloud Supervisor to provide an explicit plan first and check if the on-device Executor deviates during execution.
	- **Mechanism**: The Supervisor generates a natural language plan based on the user query; the Executor follows the plan; every $T_v$ steps, the Supervisor performs verification based on the current plan and trajectory. If intervention is deemed necessary, it outputs a revised plan, and the Executor continues under the updated plan.
	- **Design Motivation**: In stateful UI/API tasks like AppWorld, early incorrect actions may be irreversible, making explicit plans, control flows, and tool sequences crucial. PEVR concentrates cloud capabilities on planning and correction.

2. **EVA: Execute-Verify-Advise**:

	- **Function**: Allows the on-device Executor to execute autonomously first, with the cloud Supervisor providing summaries and advice only when progress appears abnormal.
	- **Mechanism**: EVA has no initial plan; the Executor performs ReAct directly. The Supervisor periodically judges intervention needs based on the query and trajectory. Instead of rewriting a plan, it generates a summary of past actions and subsequent advice, then clears the Executor's old context.
	- **Design Motivation**: Deep search tasks like FanOutQA/HotpotQA rely more on exploration and information aggregation; frequent re-planning can interrupt search trajectories. Lightweight advice plus summarization is better suited for maintaining long-term search direction.

3. **Joint Evaluation of Cost and Context Efficiency**:

	- **Function**: Evaluates not only accuracy but also cloud costs, on-device energy, and KV-cache growth.
	- **Mechanism**: Cloud costs are accumulated based on GPT-4o API pricing; device energy is estimated based on model inference. Context efficiency is measured by the maximum KV-cache footprint. Interventions in PEVR/EVA trigger context resets or summarization, limiting long-context growth on-device.
	- **Design Motivation**: The value of hybrid MAS is not just being "more accurate than on-device" but being "cheaper than pure cloud" and "able to complete long tasks within device memory limits."

### Loss & Training
This paper does not train models; all experiments are inference-time system design comparisons. The cloud model is fixed as GPT-4o, and on-device models include Qwen3 4B, 8B, 14B, and 32B. HotpotQA uses a maximum of 10 ReAct turns with verification intervals of 1/2/3/5; FanOutQA uses 20 turns with intervals of 1/2/3/5/10; AppWorld uses 40 turns with intervals of 1/2/4/8/16. Qwen3 32B uses fp8 KV-cache and weight quantization for experimentation on a single A100.

## Key Experimental Results

### Main Results
The primary conclusions are drawn from Figure 2 and Table 2. The table below focuses on the comparison of "who acts as Executor and who acts as Supervisor," with scores representing the best accuracy setting for the respective task and costs reflecting cloud API expenses (lower is better).

| Configuration | AppWorld Task Success | AppWorld Cost | FanOutQA ROUGE-1 F1 | FanOutQA Cost | Conclusion |
|------|----------------------|---------------|---------------------|---------------|------|
| GPT-4o Monolithic Cloud | 0.25 | 0.37 | 0.14 | 0.19 | Accurate but high cost |
| Qwen 32B Executor + GPT-4o Supervisor | 0.21 | 0.09 | 0.23 | 0.11 | Device exec + Cloud superv outperforms pure cloud on FanOutQA and is cheaper |
| Qwen 14B Executor + GPT-4o Supervisor | 0.19 | 0.08 | 0.12 | 0.04 | Low cost, capability limited by on-device model |
| Qwen 8B Executor + GPT-4o Supervisor | 0.16 | 0.08 | 0.09 | 0.04 | Still outperforms some monolithic on-device configs |
| Qwen 4B Executor + GPT-4o Supervisor | 0.11 | 0.13 | 0.06 | 0.04 | Small model execution becomes the bottleneck |
| GPT-4o Executor + Qwen 32B Supervisor | 0.25 | 0.67 | 0.14 | 0.17 | Cloud exec + Device superv is more expensive and not more accurate |
| Qwen 32B Monolithic On-device | 0.07 | 0.00 | 0.15 | 0.00 | Cheap but significantly insufficient for AppWorld |

### Ablation Study
The paper analyzes verifier misjudgment, architectural role reversal, and context efficiency. Two tables directly supporting the design conclusions are listed below.

| Benchmark | Verifier | False Negative | False Positive | Explanation |
|-----------|----------|----------------|----------------|------|
| AppWorld | Plan-based PEVR | 5.3% | 6.2% | More active intervention; plan deviation is a valid signal |
| AppWorld | Query-based EVA | 6.0% | 1.9% | More conservative, but lacks help from an initial plan |
| FanOutQA | Plan-based PEVR | 8.4% | 7.7% | Excessive restarts hurt long-range search |
| FanOutQA | Query-based EVA | 14.8% | 6.1% | More misses, but fewer execution interruptions |

| Max Turns | Architecture | Qwen3 8B Task Success | Qwen3 8B KV (GB) | Qwen3 32B Task Success | Qwen3 32B KV (GB) |
|-----------|--------------|-----------------------|------------------|------------------------|-------------------|
| 20 | Monolithic | 0.00 | 3.52 | 0.05 | 6.59 |
| 20 | PEVR | 0.07 | 3.34 | 0.18 | 6.53 |
| 40 | Monolithic | 0.02 | 4.82 | 0.07 | 11.34 |
| 40 | PEVR | 0.09 | 3.65 | 0.16 | 6.98 |
| 80 | Monolithic | 0.00 | 5.17 | 0.09 | 13.12 |
| 80 | PEVR | 0.11 | 3.82 | 0.19 | 7.90 |

### Key Findings
- Hybrid MAS can find configurations that are more accurate than monolithic on-device models and cheaper than monolithic cloud models, though the optimal point varies by task.
- PEVR is better suited for AppWorld because stateful environments require initial plans and executable re-planning; EVA is more suited for FanOutQA/HotpotQA, as frequent search restarts significantly harm performance.
- More cloud supervision does not equate to higher accuracy. Overly short verification intervals increase costs and may interrupt on-device execution through excessive intervention.
- Larger on-device models are generally better, but hybrid architectures completed some tasks that neither pure-cloud nor pure-device models could, suggesting collaboration is not a simple union of routers.
- The context resets and summarization in PEVR/EVA limit KV-cache growth. In an 80-turn AppWorld scenario, Qwen3 32B under PEVR used 7.90GB of KV-cache, whereas the monolithic on-device version reached 13.12GB.

## Highlights & Insights
- The paper avoids simplifying hybrid agents into "ask the big model if the small one fails" and instead carefully decomposes roles like planning, execution, verification, and advice, analyzing their impact across different tasks.
- The finding that "more cloud intervention can be worse" is a useful engineering insight. If a strong model frequently restarts a long-range search, it may destroy accumulated intermediate states.
- The differences between PEVR and EVA suggest that agent architectures should be task-adaptive: stateful tasks need executable plans, while open information search needs to preserve exploration trajectories with moderate summarization.
- The context efficiency data is highly practical. The bottleneck for many on-device agents is not single-step inference but KV-cache explosion from long trajectories; periodic resets in MAS conveniently mitigate this.

## Limitations & Future Work
- The cloud model is fixed as GPT-4o and on-device models as the Qwen3 series; different model families, context lengths, or stronger local models might change the optimal architecture.
- The paper covers HotpotQA, FanOutQA, and AppWorld but does not address coding agents, robotic control, browser GUIs, or real-world mobile deployments.
- Energy consumption is estimated rather than measured on real devices; latency, thermal control, memory bandwidth, and system scheduling on actual phones/laptops will introduce additional constraints.
- The current verification interval is manually tuned; a more natural direction involves learning a dynamic supervisor policy to decide when to request the cloud based on uncertainty and task state.

## Related Work & Insights
- **vs. Monolithic Cloud Agent**: Cloud models are powerful but costly and do not necessarily cover all tasks hybrid systems can solve; hybrid MAS can enter similar or better Pareto regions at lower costs.
- **vs. Monolithic On-device Agent**: On-device models are cheap but often fail in long-range stateful tasks like AppWorld; cloud planning/verification provides critical error correction.
- **vs. Model Routing**: Routing selects one model per query, whereas this study shows MAS can complete tasks that both pure-cloud and pure-device models fail, indicating that role collaboration creates new behaviors.
- **vs. AgentFlow / Advisor-style MAS**: PEVR is close to a planner-executor-verify-replan loop, while EVA is closer to an advisor with resets. The contribution lies in comparing them within a unified cloud-edge cost framework.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Hybrid cloud/device agents are not a new concept, but systematically decomposing PEVR/EVA relative to cost, energy, and context efficiency is highly valuable.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Solid analysis across three benchmarks, multiple Qwen3 sizes, and supervision intervals; real-device measurements and more task domains are still needed.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure and effective mechanism analysis. Figure 2 is information-dense and requires subsequent tables for full comprehension.
- Value: ⭐⭐⭐⭐⭐ Extremely practical for deploying on-device agents and cloud-assisted systems, particularly the design principles of "minimal cloud supervision + context management."

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Latent Agents: A Post-Training Procedure for Internalized Multi-Agent Debate](../../ACL2026/multi_agent/latent_agents_a_post-training_procedure_for_internalized_multi-agent_debate.md)
- [\[ACL 2026\] HACHIMI: Scalable and Controllable Student Persona Generation via Orchestrated Agents](../../ACL2026/multi_agent/hachimi_scalable_and_controllable_student_persona_generation_via_orchestrated_ag.md)
- [\[ICML 2026\] EngiAgent: Fully Connected Coordination of LLM Agents for Solving Open-ended Engineering Problems with Feasible Solutions](engiagent_fully_connected_coordination_of_llm_agents_for_solving_open-ended_engi.md)
- [\[ACL 2026\] ATLAS: Adaptive Trading with LLM AgentS Through Dynamic Prompt Optimization and Multi-Agent Coordination](../../ACL2026/multi_agent/atlas_adaptive_trading_with_llm_agents_through_dynamic_prompt_optimization_and_m.md)
- [\[ACL 2026\] When Identity Skews Debate: Anonymization for Bias-Reduced Multi-Agent Reasoning](../../ACL2026/multi_agent/when_identity_skews_debate_anonymization_for_bias-reduced_multi-agent_reasoning.md)

</div>

<!-- RELATED:END -->
