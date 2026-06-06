---
title: >-
  [Paper Note] ACON: Optimizing Context Compression for Long-horizon LLM Agents
description: >-
  [ICML2026][LLM Agent][Context Compression] Acon utilizes failure trajectory contrast to optimize natural language compression guidelines…
tags:
  - "ICML2026"
  - "LLM Agent"
  - "Context Compression"
  - "Prompt Optimization"
  - "Trajectory Contrast"
  - "Compression Distillation"
date: 2026-05-08
content_hash: 67dca0a5634270a1
---

# ACON: Optimizing Context Compression for Long-horizon LLM Agents

**Conference**: ICML2026  
**arXiv**: [2510.00615](https://arxiv.org/abs/2510.00615)  
**Code**: https://github.com/microsoft/acon  
**Area**: LLM Agent / Context Compression / Long-horizon Tasks  
**Keywords**: LLM Agent, Context Compression, Prompt Optimization, Trajectory Contrast, Compression Distillation  

## TL;DR
Acon utilizes failure trajectory contrast to optimize natural language compression guidelines, compressing both agent history and observation contexts. It reduces peak tokens by 26% to 54% on AppWorld, OfficeBench, and multi-objective QA while maintaining or improving task success rates for long-horizon tasks.

## Background & Motivation
**Background**: LLM agents have been deployed for multi-step tasks such as office automation, application operations, and search-based QA. Unlike single-turn QA, agents must continuously store observations, actions, tool outputs, and intermediate states, as every subsequent decision depends on this interaction history.

**Limitations of Prior Work**: The context of long-horizon agents grows continuously, leading to two primary issues. First, Transformer inference and KV cache costs increase with context length, making memory and latency uncontrollable. Second, as excessive obsolete or irrelevant information accumulates in long contexts, models are more easily distracted by noise, which significantly degrades survival rates, especially for smaller models.

**Key Challenge**: Context compression must be both "aggressive" and "accurate." If only truncation or generic summarization is applied, critical states such as file paths, API parameters, account statuses, and constraints in tool returns are easily lost. If too much is retained, costs are not effectively reduced. Furthermore, many proprietary LLMs do not support gradient updates, and RL-based compression strategies require expensive rollouts.

**Goal**: The authors aim to construct a model-agnostic compression framework that automatically learns compression rules for different environments without modifying agent weights, ensuring compressed contexts are concise yet retain necessary states for task completion.

**Key Insight**: The observation is that compression failures leave strong diagnostic signals at the trajectory level: if a task succeeds with full context but fails with compressed context, it indicates the compressor missed critical states. Providing this contrast to an LLM for analysis can generate natural language feedback to update compression prompts.

**Core Idea**: Instead of fine-tuning the agent, Acon iteratively optimizes "what to retain and what to delete" guidelines in the natural language space and distills the knowledge into small-scale compressors using successful trajectories.

## Method
Acon can be viewed as a context management layer inserted between the agent and the environment. The agent still makes decisions using the original ReAct or benchmark-specified tool formats; Acon invokes the compressor only when the history or observation exceeds a threshold to rewrite long text into a shorter, high-density state summary. The key is not a hand-written generic summary prompt, but rather the automatic improvement of this prompt using success/failure trajectories from training tasks.

### Overall Architecture
The input consists of a long-horizon agent benchmark, a fixed agent LLM, a fixed system prompt, and a batch of training tasks. At each time step, the agent sees history $h_{t-1}$ and the latest observation $o_t$. If the history length exceeds threshold $T_{hist}$, Acon uses compressor $f(h_t;\phi,P_{hist})$ to generate a compressed history; if the observation exceeds threshold $T_{obs}$, it uses $f(o_t,h_{t-1};\phi,P_{obs})$ to generate a compressed observation. The compressed context replaces the original for the next decision step.

The training phase first runs tasks with initial guidelines to collect a "contrastive subset" where the full context succeeded but the compressed context failed. An optimizer LLM analyzes the original context, the compressed version, and the failure information to determine what was missed and which states should be preserved. A subsequent update prompt aggregates feedback to update the compression guidelines. The first stage, "utility maximization," focuses on success rate; the second stage, "compression maximization," analyzes successful trajectories to remove redundant info. Finally, the optimized LLM compressor generates input-output pairs to fine-tune small models (e.g., Qwen3, Phi) via LoRA.

### Key Designs
1.  **Separated History and Observation Compression**:
    - **Function**: Simultaneously manages the growth of agent history and excessively long single-step environment observations.
    - **Mechanism**: History compression triggers only when $|h_t|>T_{hist}$, while observation compression triggers when $|o_t|>T_{obs}$. History compression focuses on cross-step states and future constraints; observation compression identifies valid fields in current tool returns, referencing history for relevance.
    - **Design Motivation**: Token explosion stems not only from history but also from large tables, emails, or files returned by tool calls. Optimizing these separately outperforms a generalized summary prompt.

2.  **Compression Guideline Optimization via Failure Contrast**:
    - **Function**: Converts sparse task success/failure rewards into readable natural language optimization signals.
    - **Mechanism**: For tasks that succeed with full context but fail when compressed, the LLM compares $H$ and $H'$ to identify missing information. Feedback is aggregated into a new guideline $P^{(1)}$. Multiple candidates are generated and evaluated to select the best one.
    - **Design Motivation**: Direct RL optimization requires excessive rollouts and cannot update API LLMs. Natural language prompt optimization allows fast iteration on few training tasks and is compatible with any agent model.

3.  **Compression Maximization and Compressor Distillation**:
    - **Function**: Further reduces context length and compressor invocation costs while maintaining success rates.
    - **Mechanism**: The utility step ensures critical states are kept; the compression step analyzes which content was actually used in successful trajectories and removes redundancies. High-quality outputs from the teacher LLM are then used to fine-tune a student model using cross-entropy loss.
    - **Design Motivation**: Pursuit of success rate alone leads to conservative summaries; pursuit of brevity leads to state loss. Redundancy is handled separately from utility, and distillation avoids expensive LLM calls at every step.

### Loss & Training
The objective is defined as $\max_\psi E[R(s_T(\psi))]-\lambda E[C(H'(\psi))]$, where $R$ is the task reward and $C$ is the cost. No gradient updates are performed on the agent; instead, textual feedback updates the guidelines. For distillation, standard next-token cross-entropy is used on $(x,y)$ pairs: $x=h_t, y=h'_t$ for history and $x=(h_{t-1},o_t), y=o'_t$ for observations.

## Key Experimental Results

### Main Results
Core results across AppWorld, OfficeBench, and 8-objective QA show the trade-off between performance and peak tokens.

| Benchmark / Setting | Method | Main Results | Steps | Peak tokens | Dependency | Note |
|:--- |:--- |:--- |:--- |:--- |:--- |:--- |
| AppWorld / history / gpt-4.1 | No compression | 56.0 Acc | 16.14 | 9.93K | 5.96M | Upper bound, highest cost |
| AppWorld / history / gpt-4.1 | Prompting | 43.5 Acc | 24.01 | 6.93K | 5.29M | Standard compression drops success |
| AppWorld / history / gpt-4.1 | Acon UT | 51.2 Acc | 20.92 | 7.17K | 4.49M | Token reduction with stability |
| AppWorld / history / gpt-4.1 | Acon UT+CO | 56.5 Acc | 22.82 | 7.33K | 4.69M | Matches/exceeds full, -26% peak tokens |
| AppWorld / observation / gpt-4.1 | Prompting | 42.3 Acc | 17.38 | 6.58K | 4.09M | Observation compression loses info |
| AppWorld / observation / gpt-4.1 | Acon UT+CO | 53.6 Acc | 18.12 | 7.43K | 4.93M | Significantly higher success rate |

| Benchmark | Method Category | Main Results | Conclusion |
|:--- |:--- |:--- |:--- |
| OfficeBench | history compression | Acon reduces peak context by ~30%, Acc > 74% | UT is more stable for precise tasks |
| 8-objective QA | history compression | Acon exceeds no compression in EM/F1; peak tokens -54.5% | Removing redundancy improves focus |
| Small agent Qwen3-14B | distilled compressor | AppWorld 25.6% -> 33.9%, QA EM 0.158 -> 0.23 | Compression mitigates interference |
| Compressor cost | gpt-4.1-mini / Qwen3 | Cost drops from $0.045 to $0.0004 | Distillation drastically reduces overhead |

### Ablation Study

| Ablation Dimension | Configuration | AppWorld Avg Acc | Conclusion |
|:--- |:--- |:--- |:--- |
| Prompt optimizer | o3 + contrastive feedback | 51.2 | Default setup is optimal |
| Prompt optimizer | o3, no contrastive | 50.6 (-0.6) | Failure-only is inferior to contrast |
| Prompt optimizer | gpt-4.1 + contrastive | 47.6 (-3.6) | Weaker optimizers still function but yield lower quality |

| Efficiency Setting | API cost / task | Latency / task | Note |
|:--- |:--- |:--- |:--- |
| No Compression | $0.331 | 73.24s | High token cost |
| Acon history | $0.285 | 87.68s | Lower API cost, higher latency |
| Acon observation | $0.272 | 101.92s | Lowest cost, highest latency |

### Key Findings
- The primary benefit of Acon is making the context "task-relevant" rather than just saving tokens.
- UT and CO represent a selectable trade-off. Environments like AppWorld benefit from both, while precision-sensitive tasks like OfficeBench favor UT.
- Compressors can be localized. Distilled small models retain over 95% of teacher performance, practically zeroing compression costs while introducing some latency.
- Smaller agents benefit significantly as compression removes distractors, allowing models like Qwen3-14B to make correct decisions in long-horizon tasks.

## Highlights & Insights
- This work transforms "how to write compression prompts" from manual rules into optimizable objects, with signals derived from actual agent failures.
- The failure contrast design is highly practical. The difference between a successful full-context run and a failed compressed run pinpoint the compressor's errors more effectively than final rewards.
- Acon is not coupled to agent weights, making it realistic for API-based, closed-source, or already deployed agent systems.
- Compression acts as "denoising." The fact that some compressed settings outperform full context suggests that "more information" is not always better for reasoning.

## Limitations & Future Work
- Acon requires task rollouts for training, which depends on reproducible environments or benchmarks.
- Compression introduces additional latency. Real-world systems may require asynchronous compression or even smaller models to optimize wall-clock time.
- Optimization quality is bottlenecked by the strength of the optimizer LLM.
- The current scope is limited to textual tool environments, excluding multimodal agents, GUI browsers, or large-scale codebases.
- High-fidelity tasks might suffer from "Compression Maximization" (CO) deleting subtle facts, requiring task-specific conservative strategies.

## Related Work & Insights
- **vs FIFO / Retrieval**: These lack environmental knowledge of "what state to keep"; Acon learns this via failure contrast.
- **vs LLMLingua / generic prompting**: Generic methods do not necessarily understand agent action consequences; Acon is trajectory-aware.
- **vs ReSum / RL-based**: These often require updating the agent; Acon is model-agnostic.
- **vs KV cache compression**: Acon operates at the semantic level; the two are complementary as Acon reduces input redundancy and interference.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Optimizing compression via trajectory contrast addresses a core agent pain point.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers diverse benchmarks and distillation, though could expand to open environments.
- Writing Quality: ⭐⭐⭐⭐☆ Clear methodology, though some symbol definitions are dense.
- Value: ⭐⭐⭐⭐⭐ Directly improves deployment cost, noise robustness, and utility of small model agents.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] OCR-Memory: Optical Context Retrieval for Long-Horizon Agent Memory](../../ACL2026/llm_agent/ocr-memory_optical_context_retrieval_for_long-horizon_agent_memory.md)
- [\[ICLR 2026\] Solving the Granularity Mismatch: Hierarchical Preference Learning for Long-Horizon LLM Agents](../../ICLR2026/llm_agent/solving_the_granularity_mismatch_hierarchical_preference_learning_for_long-horiz.md)
- [\[ICLR 2026\] Harnessing Uncertainty: Entropy-Modulated Policy Gradients for Long-Horizon LLM Agents](../../ICLR2026/llm_agent/harnessing_uncertainty_entropy-modulated_policy_gradients_for_long-horizon_llm_a.md)
- [\[AAAI 2026\] When Refusals Fail: Unstable Safety Mechanisms in Long-Context LLM Agents](../../AAAI2026/llm_agent/when_refusals_fail_unstable_safety_mechanisms_in_long-context_llm_agents.md)
- [\[ICML 2026\] Agent-Omit: Adaptive Context Omission for Efficient LLM Agents](agent-omit_adaptive_context_omission_for_efficient_llm_agents.md)

</div>

<!-- RELATED:END -->
