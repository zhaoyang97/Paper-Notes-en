---
title: >-
  [Paper Note] Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks
description: >-
  [ACL2026][LLM Agent][Long-term memory] Shopping Companion constructs an e-commerce task benchmark featuring long-term user preference memory and a real-world product database. It employs a two-stage agent optimized with…
tags:
  - "ACL2026"
  - "LLM Agent"
  - "Long-term memory"
  - "E-commerce Agent"
  - "Preference Identification"
  - "Tool Calling"
  - "Reinforcement Learning"
date: 2026-05-08
content_hash: 98b52f6f9256b49c
---

# Shopping Companion: A Memory-Augmented LLM Agent for Real-World E-Commerce Tasks

**Conference**: ACL2026  
**arXiv**: [2603.14864](https://arxiv.org/abs/2603.14864)  
**Code**: Not disclosed  
**Area**: LLM Agent / E-commerce Recommendation  
**Keywords**: Long-term memory, E-commerce Agent, Preference Identification, Tool Calling, Reinforcement Learning

## TL;DR
Shopping Companion constructs an e-commerce task benchmark featuring long-term user preference memory and a real-world product database. It employs a two-stage agent optimized with dual rewards and tool-level rewards to improve preference identification and product recommendation, allowing a 4B model to approach the performance of powerful closed-source models.

## Background & Motivation
**Background**: E-commerce LLM agents need to complete tasks such as product recommendation, budget constraints, and bundle deals. Unlike general Q&A, a shopping assistant must query product databases, compare attributes, satisfy hard constraints, and understand implicit preferences regarding brands, sizes, styles, and budgets expressed by the user across multiple past conversation sessions.

**Limitations of Prior Work**: Existing benchmarks often only cover single-session product searches or evaluate long-term memory via Q&A without applying it to real downstream tasks. WebShop lacks long-term memory, LongMemEval does not evaluate shopping tasks, and ShoppingBench/ShopSimulator do not integrate cross-session preference memory, real product retrieval, and user intervention into an end-to-end setting.

**Key Challenge**: Preference identification and shopping execution are coupled in real-world scenarios. Treating long-term memory retrieval as an upstream module and product recommendation as a downstream module leads to preference extraction errors that cannot be corrected by task feedback. Conversely, an end-to-end approach often causes the model to get lost in long histories and massive product databases.

**Goal**: The authors aim to establish an e-commerce agent benchmark capable of simultaneously evaluating long-term memory, real product constraints, and multi-turn user intervention. They also seek to train a lightweight model that exceeds strong open-source models and rivals closed-source models in preference capture and final recommendation success rates.

**Key Insight**: The paper models the shopping task as a partially observable MDP, where the state consists of the dialogue context, the long-term memory database, and the current shopping instruction. Success requires recommendations to satisfy both explicit requirements and implicit preferences found in memory.

**Core Idea**: Decompose the agent into "Preference Identification" and "Shopping Execution" stages. Then, use stage-level dual rewards to evaluate stage objectives and tool-wise rewards to evaluate intermediate tool calls, thereby improving sparse credit assignment in multi-turn tool interactions.

## Method

### Overall Architecture
The paper first constructs a large-scale shopping simulation environment containing 1,298,797 real products, long-term session histories, natural language shopping instructions, and two types of tasks: Single Product Recommendation and Add-on Deals. Each sample contains 15-50 historical turns, where target preferences are buried among irrelevant conversations, similar to a "needle-in-a-haystack" setup.

The Shopping Companion agent operates in two stages. Stage 1 is Preference Identification: the agent calls memory search, view, and summarize tools to extract implicit preferences relevant to the current instruction and presents them for user confirmation. Stage 2 is Shopping Assistance: based on confirmed preferences, the agent calls product search and view tools, iteratively checking requirements, budgets, and preference constraints to output formatted recommendation products.

### Key Designs
1. **Long-term preference memory benchmark**:
    - **Function**: Integrates "remembering user preferences" into the evaluation of real shopping success rates rather than as a standalone Q&A memory task.
    - **Mechanism**: The authors sample products and deal combinations from a real product database, use GPT-5 to generate user instructions and dialogue sessions containing implicit preferences, and then mix in irrelevant sessions in a LongMemEval style. This results in 1,000 instructions (800 for training, 200 for testing), each requiring preference evidence retrieval from history to map to product constraints.
    - **Design Motivation**: Users rarely repeat all preferences in a current request; assistants must find evidence in history. If the benchmark does not evaluate downstream recommendation results, the memory module might only optimize for text recall rather than task success.

2. **Two-stage agentic framework**:
    - **Function**: Reduces the coupling difficulty between long-history preference extraction and product constraint solving, while allowing for user intervention.
    - **Mechanism**: Stage 1 focuses solely on identifying preferences from long-term memory (e.g., brand exclusions, size history, color/material preferences), which are then displayed for user confirmation. Stage 2 treats confirmed preferences as hard or soft constraints, using product retrieval tools to find candidates and verify if they meet the instruction and historical preferences.
    - **Design Motivation**: A one-stage end-to-end agent is prone to continuing with erroneous searches if preference extraction fails. The two-stage structure provides an intermediate checkable state where low-granularity or high-granularity user feedback can correct preferences.

3. **Dual Reward + Tool-wise Reward**:
    - **Function**: Addresses the sparse credit assignment problem for multi-turn tool agents that only receive final success signals.
    - **Mechanism**: Stage 1 reward measures query relevance, preference attribute matching, and identified product counts for add-on deals. Stage 2 reward evaluates if the recommendation is extractable, satisfies instruction requirements, and matches preferences/budget/quantity constraints. Tool-wise reward assigns scores to each memory search, memory view, product search, and product view, checking if they hit "gold" preference sessions or products. The final reward is the sum of stage, tool, and format rewards.
    - **Design Motivation**: Failures in final recommendations can stem from memory search errors, product search errors, insufficient views, or formatting errors. Tool-level rewards move feedback into intermediate actions, making it easier for RL to learn "what to search, what to look at, and when to stop."

### Loss & Training
Training involves SFT and RL. SFT uses GPT-4.1 rejection sampling to obtain 2,948 successful step-level trajectories. LLaMA-Factory is used for LoRA fine-tuning on Qwen3-4B-Thinking-2507, with a rank of 64 targeting q/k/v/o projections. RL is based on VeRL and GRPO, with 8 rollouts per sample, a maximum output length of 32,768, up to 20 assistant turns, batch size 16, mini-batch size 8, temperature 0.6, top-k 20, top-p 0.95, and a learning rate of $1\times10^{-6}$ over approximately 2.6 epochs.

## Key Experimental Results

### Main Results
The primary metrics include Preference Identification Accuracy (Acc.) for Stage 1 and Final Recommendation Success Rate (Succ.) for Stage 2. Below are the results on the test set.

| Category | Model | Single Acc | Single Succ | Add-on Acc | Add-on Succ | Avg Acc | Avg Succ |
|------|------|------------|-------------|------------|-------------|---------|----------|
| Closed | GPT-5 | 82.0 | 75.0 | 66.0 | 54.0 | 74.0 | 64.5 |
| Closed | GPT-4.1 | 88.0 | 78.0 | 39.0 | 24.0 | 63.5 | 51.0 |
| Closed | GPT-4o | 79.0 | 72.0 | 41.0 | 26.0 | 60.0 | 49.0 |
| Closed | Qwen3-Max | 80.0 | 72.0 | 35.0 | 24.0 | 57.5 | 48.0 |
| Open | Qwen3-Next-80B-A3B | 63.0 | 57.0 | 29.0 | 18.0 | 46.0 | 37.5 |
| Open | Qwen3-30B-A3B | 60.0 | 53.0 | 21.0 | 13.0 | 40.5 | 33.0 |
| Open | Qwen3-4B | 49.0 | 44.0 | 11.0 | 6.0 | 30.0 | 25.0 |
| Ours | Qwen3-4B-LoRA | 82.0 | 72.0 | 42.0 | 31.0 | 62.0 | 51.5 |
| Ours | + Dual-reward RL | 89.0 | 81.0 | 50.0 | 38.0 | 69.5 | 59.5 |
| Ours | + Dual & Tool-wise Reward | 90.0 | 84.0 | 55.0 | 43.0 | 72.5 | 63.5 |

The final model achieved an average success rate of 63.5%, close to GPT-5's 64.5%, and significantly outperformed all open-source baselines. Single Product success reached 84.0%, but Add-on Deals remain at 43.0%, indicating higher difficulty in bundle-related tasks.

### Ablation Study

| Strategy | Single Succ | Add-on Succ | Avg Succ | Description |
|------|-------------|-------------|----------|------|
| Oracle | 85.0 | 73.0 | 79.0 | Gold preference evidence provided directly |
| One-Stage | 73.0 | 32.0 | 52.5 | Preference ID and execution combined |
| Two-Stage None | 75.0 | 55.0 | 65.0 | Explicit stages without extra user prompts |
| Two-Stage Low Hint | 78.0 | 59.0 | 68.5 | User points out missing/incorrect info |
| Two-Stage High Hint | 80.0 | 60.0 | 70.0 | User specifies missing dimensions without values |

| Training Strategy | Avg. Turns | Avg. Tool Calls | Avg. Resp. Length | Conclusion |
|----------|----------|--------------|--------------|------|
| Dual-Reward | 9.82 | 9.17 | 10485.39 | Longer trajectories with only stage rewards |
| Dual & Tool-wise | 8.89 | 8.47 | 10068.83 | Tool-level rewards make retrieval more focused |

### Key Findings
- Closed-source models performed strongly on single-product tasks, but success rates for add-on deals dropped significantly (GPT-5 at 54.0%). Multi-product combinations, budgets, and preference matching create combinatorial optimization challenges.
- The 4B model improved from 25.0 Succ to 51.5 after LoRA, rose to 59.5 with dual-reward RL, and reached 63.5 with tool-wise rewards, showing that training signals are progressively aligning with task goals.
- The two-stage structure is a major contributor. One-Stage achieved only 32.0 on add-on deals, while Two-Stage None reached 55.0, proving that explicitly organizing preferences significantly eases the search burden.
- User intervention yields stable gains, though a 9-point gap remains between High Hint and Oracle, indicating bottlenecks exist in product retrieval and multi-constraint decision-making as well as preference identification.

## Highlights & Insights
- The paper progresses long-term memory evaluation from "recalling facts" to "improving task success." This is closer to real agent products than simple Q&A benchmarks.
- The two-stage design is practical: confirming preferences before execution enhances controllability and provides a natural point for error correction.
- Tool-wise reward is the most valuable engineering contribution. By converting gold sessions and gold products into intermediate action feedback, it solves the problem of not knowing why a multi-turn agent failed.
- Results demonstrate that lightweight models can compete with large closed-source models in vertical scenarios when benchmarks, tools, and rewards are sufficiently tailored to the task.

## Limitations & Future Work
- Add-on Deals remain difficult (43.0% success), showing room for improvement in handling budgets, multi-product compatibility, and preference combinations.
- Tool-wise rewards are tied to a specific toolset and product database; transferring to other domains (travel, medical, etc.) requires redesigning gold actions and reward servers.
- The benchmark uses LLM-synthesized instructions/sessions; while human-verified, real user preferences may be more ambiguous, contradictory, and dynamic.
- Long-term memory involves privacy and fairness; real deployment requires user-controllable memory and avoidance of inferring sensitive attributes or inducement of consumption.
- Code is not public, and reproducing the full environment requires the product database, retrieval index, tool protocols, and reward server.

## Related Work & Insights
- **vs WebShop**: WebShop focuses on web shopping operations but lacks cross-session preferences; Shopping Companion makes history a prerequisite for success.
- **vs LongMemEval**: Evaluates memory Q&A, while this paper embeds memory into a recommendation task to measure the value of memory for downstream action.
- **vs ShopSimulator**: ShopSimulator features interaction but more static preferences; this paper uses persistent memory and user confirmation, mimicking real assistants.
- **vs Agentic Memory**: While Agentic Memory optimizes memory strategy, Shopping Companion applies this to e-commerce tools and extends tool-level rewards to search.

## Rating
- Novelty: ⭐⭐⭐⭐ Good integration of long-term memory, real product databases, and user intervention.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong baselines and comprehensive ablations, though lacks real-user studies.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and reward design; some protocol details are scattered.
- Value: ⭐⭐⭐⭐⭐ High reference value for vertical agent benchmarks and tool-level RL.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](../../ICLR2026/llm_agent/exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [\[ACL 2026\] MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools](mcp-flow_facilitating_llm_agents_to_master_real-world_diverse_and_scaling_mcp_to.md)
- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](../../ICLR2026/llm_agent/openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [\[ACL 2026\] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](agencybench_benchmarking_the_frontiers_of_autonomous_agents_in_1m-token_real-wor.md)
- [\[AAAI 2026\] D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies](../../AAAI2026/llm_agent/d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ICLR 2026\] Exploratory Memory-Augmented LLM Agent via Hybrid On- and Off-Policy Optimization](../../ICLR2026/llm_agent/exploratory_memory-augmented_llm_agent_via_hybrid_on-_and_off-policy_optimizatio.md)
- [\[ACL 2026\] MCP-Flow: Facilitating LLM Agents to Master Real-World, Diverse and Scaling MCP Tools](mcp-flow_facilitating_llm_agents_to_master_real-world_diverse_and_scaling_mcp_to.md)
- [\[ICLR 2026\] OpenAgentSafety: A Comprehensive Framework for Evaluating Real-World AI Agent Safety](../../ICLR2026/llm_agent/openagentsafety_a_comprehensive_framework_for_evaluating_real-world_ai_agent_saf.md)
- [\[ACL 2026\] AgencyBench: Benchmarking the Frontiers of Autonomous Agents in 1M-Token Real-World Contexts](agencybench_benchmarking_the_frontiers_of_autonomous_agents_in_1m-token_real-wor.md)
- [\[AAAI 2026\] D-GARA: A Dynamic Benchmarking Framework for GUI Agent Robustness in Real-World Anomalies](../../AAAI2026/llm_agent/d-gara_a_dynamic_benchmarking_framework_for_gui_agent_robust.md)

</div>

<!-- RELATED:END -->
