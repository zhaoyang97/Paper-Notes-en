---
title: >-
  [Paper Note] Polaris: A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair
description: >-
  [ACL2026][LLM Agent][Gödel Agent] Polaris transforms the recursive self-improvement of Gödel Agents into a "failure analysis → experience abstraction → minimal code patch → execution verification" policy repair loop suit…
tags:
  - "ACL2026"
  - "LLM Agent"
  - "Gödel Agent"
  - "Small Language Models"
  - "Self-improvement"
  - "Experience Abstraction"
  - "Policy Repair"
date: 2026-05-08
content_hash: d6f486b60cd6a602
---

# Polaris: A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair

**Conference**: ACL2026  
**arXiv**: [2603.23129](https://arxiv.org/abs/2603.23129)  
**Code**: No public repository address provided in the paper cache  
**Area**: llm_agent  
**Keywords**: Gödel Agent, Small Language Models, Self-improvement, Experience Abstraction, Policy Repair

## TL;DR
Polaris transforms the recursive self-improvement of Gödel Agents into a "failure analysis → experience abstraction → minimal code patch → execution verification" policy repair loop suitable for 7B/8B small models. It enables small models to achieve explainable and persistently reusable policy-level improvements across MGSM, DROP, GPQA, and LitBench.

## Background & Motivation
**Background**: Self-improvement for language agents generally follows two paths: response-level reflection and rewriting (e.g., ReAct, Reflexion, Self-Refine, CRITIC), or modifying model parameters and knowledge representations (e.g., model editing). Gödel Agent provides a third path: treating the agent policy as an explicit, inspectable, and modifiable program object, allowing the agent to modify its own policy based on execution trajectories.

**Limitations of Prior Work**: The original Gödel Agent approach places high demands on context window and tool-calling capabilities. When attempting to port it to Qwen2.5-7B-Instruct, the authors found that the agent easily suffers from OOM (Out of Memory) due to retaining too many validation samples, historical tool calls, and debugging trajectories. Conversely, shortening the context leads to tool-calling hallucinations, repetitive ineffective repairs, and non-targeted behavior.

**Key Challenge**: Recursive self-improvement requires sufficiently rich failure experience to generate transferable policy updates. However, small models have limited context, memory, and meta-reasoning capabilities, making them unable to carry large-scale historical trajectories. This paper addresses the core problem of balancing "sufficiently abstracted experience" with "sufficiently small context."

**Goal**: Polaris attempts to enable policy-level self-improvement for small language models. It does not fine-tune parameters or merely correct individual answers; instead, it compresses a small number of failure samples into reusable strategies and generates minimal code patches. These patches are written back to the current policy after passing syntax and execution checks.

**Key Insight**: The authors view failure samples as "experience" but do not directly feed all failure trajectories into the context. Instead, they first abstract them into diagnoses, revisions, and preventions before synthesizing a small number of new strategies. This preserves the explainability of failures while reducing the risk of small models getting lost in long contexts.

**Core Idea**: Use experience abstraction to compress several instance-level failures into policy-level repair strategies, then use minimal verifiable patches to complete recursive self-improvement within the capacity of small models.

## Method
The goal of Polaris is not to make the model "rethink the answer," but to let the agent modify how it solves problems next time. It inherits the self-inspect, interact, self-update, and continue-improve framework of Gödel Agents but replaces the original repair policy module with a more conservative experience-abstracted repairer. The entire system operates around a mutable policy $\pi$. First, the current policy is run on a validation set to identify failure samples. Next, the causes of failure are analyzed, repair strategies are abstracted, and code patches are generated. Finally, if the patch passes execution checks, the new policy serves as the starting point for the next iteration.

### Overall Architecture
The input includes the current agent policy, task goals, failure samples from the validation set, and agent memory; the output is a modified policy. In each round, the current policy is evaluated to collect a set of failed tasks $T=\{\tau_i\}$. Polaris then performs failure analysis on each failed task to obtain a structured reflection $A_i = (\text{diagnosis}_i, \text{revision}_i, \text{prevention}_i)$. These are compressed into one or two transferable strategies $\delta_j$ through strategy synthesis. Subsequently, patch generation creates a code patch modifying only the necessary lines. Finally, patch integration checks if the patch is executable (retrying up to 3 times on failure), updating the policy via runtime code mutation upon success.

### Key Designs
1.  **Failure Analysis as Explainable Experience Units**:
    - **Function**: Converts each failure sample into diagnostic and reusable repair evidence.
    - **Mechanism**: Failure samples include not just inputs and reference answers, but also reasoning trajectories and predicted outputs. `AnalyzeFailure` requires the model to write a fault diagnosis, the policy logic needing modification, and a prevention rule to avoid similar errors. The resulting $A_i$ is not a vague reflection but a repair record oriented toward the policy code.
    - **Design Motivation**: Small models tend to mistake surface features of a single case for general rules. Separating diagnosis, revision, and prevention forces the model to distinguish between "what went wrong this time" and "how the policy should change."

2.  **Experience-Abstracted Strategy Synthesis**:
    - **Function**: Compresses multiple instance-level reflections into a few general strategies to control context growth.
    - **Mechanism**: `StrategySynthesis` reads the collection of reflections from the current round and existing strategies in memory to synthesize one or two new strategies (e.g., finer task decomposition, numerical normalization, output format checks, context-specific verification, or control flow adjustments). The prompt requires new strategies to avoid duplication with historical ones.
    - **Design Motivation**: The original Gödel Agent retains massive historical trajectories, quickly overwhelming small models. Polaris replaces raw trajectories with abstract strategies, essentially distilling experience so that subsequent repairs can reuse "patterns" rather than repeating "cases."

3.  **Minimal Patches & Conservative Integration**:
    - **Function**: Ensures policy updates are auditable, rollable, and executable.
    - **Mechanism**: `PatchGeneration` generates only the minimal code patch required to implement the strategy, without explanatory text. During integration, a temporary policy is updated and subjected to syntax and execution checks. If it fails, it retries up to 3 times. Patches and contexts that fail persistently are written to memory rather than being forcibly applied.
    - **Design Motivation**: Self-modifying code risks "regressive repairs." Minimal patches reduce side effects, execution checks prevent syntax errors from entering the policy, and memory records maintain traceability for failed repairs.

### Loss & Training
Polaris involves no parameter training or gradient loss; the optimization goal is to improve performance on validation/test tasks through iterative policy repair. In experiments, Qwen2.5-7B-Instruct was run on two 32GB V100 GPUs, with each independent run allowed to evolve autonomously for 10 hours. MGSM and DROP used 50 validation and 250 test samples; GPQA used 20/100; LitBench used 20/250. A key hyperparameter is the number of failure samples $N$ used for reflection. The paper compares $N=3$ and $N=5$ and compresses the historical tool calls in memory from 10 (in the original Gödel Agent) to 6 to reduce context and memory pressure.

## Key Experimental Results

### Main Results
Directly migrating the original Gödel Agent to 7B small models is nearly impossible. Even with reduced historical tool messages, $k=5$ failed entirely, and $k=3$ only saw one successful run on DROP. Polaris's experience abstraction significantly improved the success rate, especially on MGSM, GPQA, and LitBench.

| Method/Setting | MGSM successful | DROP successful | GPQA successful | LitBench successful | Main Observation |
|:---|:---:|:---:|:---:|:---:|:---|
| Gödel Agent, k=3 | 0/5 | 1/5 | 0/5 | 0/5 | Prone to repetition and tool-call hallucinations after shortening context |
| Gödel Agent, k=5 | 0/5 | 0/5 | 0/5 | 0/5 | All unsuccessful, primarily due to OOM |
| Polaris, N=3 | 5/10 | 3/10 | 4/10 | 6/10 | SLMs can complete multiple rounds of policy repair |
| Polaris, N=5 | 4/10 | 3/10 | 5/10 | 5/10 | More generalized abstraction but higher context pressure |

In successful runs, Polaris achieved non-monotonic but consistent best-so-far improvements relative to the CoT-SC baseline. The paper emphasizes not expecting monotonic increases in every step, as code-level policy mutation is a discrete search; in deployment, the champion policy should be held, replacing it only when a challenger is stably better.

| Setting | MGSM Gain | DROP Gain | GPQA Gain | LitBench Gain |
|:---|:---:|:---:|:---:|:---:|
| Polaris N=3 | +4.0% | +3.9% | +9.0% | +8.8% |
| Polaris N=5 | +3.6% | +5.7% | +9.0% | +5.2% |
| Qwen3-8B N=3 | 4/5 successful | 2/5 successful | 3/5 successful | 4/5 successful |
| devstral-small-2 N=3 | 2/5 successful | 2/5 successful | 2/5 successful | 4/5 successful |

### Ablation Study
The paper does not perform traditional module-removal ablation but instead analyzes the necessity of designs by comparing the original Gödel Agent, different $N$, different base models, and failure modes.

| Analysis Object | Key Findings | Notes |
|:---|:---|:---|
| Original Gödel Agent | k=5 all failed, k=3 only 1/5 successful on DROP | Truncating context alone does not solve SLM self-improvement issues |
| Experience Abstraction | Polaris N=3: 18/40 successful across four tasks | Small failure samples can be compressed into executable policy repair strategies |
| N=3 vs N=5 | N=3 more volatile, N=5 more general but still fails | Trade-off between reflection depth and context burden |
| Qwen3-8B | Higher success rate, but thinking mode increases token/time overhead | Stronger SLM capabilities improve Polaris stability, but budget remains key |
| devstral-small-2 | OOM significantly reduced; few failures from early bad updates | More compute/stronger base shifts failure from resource issues to algorithmic stability |

### Key Findings
- Experience abstraction is central: retaining raw failure trajectories blows up the context, while retaining only abstract strategies allows repairs to transfer to unseen instances.
- Polaris's gains are usually non-monotonic, but the best-so-far policy can exceed the initial policy and CoT-SC, resembling anytime search rather than a standard training curve.
- DROP is more prone to failure than MGSM/GPQA/LitBench because its longer context more easily triggers OOM, redundant evaluations, and tool-calling errors.
- The bottleneck for SLMs is not a single ability but the system stability constituted by meta-reasoning, tool calling, context management, and patch execution.

## Highlights & Insights
- This paper anchors "agent self-improvement" in inspectable policy patches rather than staying at verbal feedback. Once written into the policy, strategies can be reused in subsequent samples—something response-level self-correction cannot achieve.
- Experience abstraction is a memory compression form well-suited for small models: storing generalized strategies that guide code modification instead of long logs. This is valuable for any resource-constrained agent.
- The minimal patch design makes self-modification more like automated program repair than open-ended rewriting. it reduces the risk of the model breaking the entire agent in one go and improves feasibility for human auditing.
- The honest presentation of unsuccessful runs is valuable. While many self-improving agent papers show only successful trajectories, Polaris uses failure classification to illustrate that real-world deployment first hits tool calling, memory, and error recovery limits.

## Limitations & Future Work
- Polaris still relies on manually designed prompt templates for failure analysis, strategy synthesis, patch generation, and integration; whether these templates can be auto-generated or migrated to new agent architectures remains unresolved.
- Execution checks mainly guarantee syntax and runtime availability, not necessarily performance improvement. A "bad" patch might still lead the policy toward low-quality trajectories if it happens to be executable.
- The proportion of unsuccessful runs is still significant, particularly on DROP; actual deployment requires rollback, checkpointing, champion-challenger mechanisms, and stronger static verification.
- Tasks are still within fixed benchmark distributions, which is not equivalent to true open-ended learning. The paper acknowledges it is closer to bounded self-improvement than autonomous learning in a long-term open environment.
- SLMs have limited meta-reasoning depth; abstractions of diverse failure samples may be overly generalized. Future work could combine external memory, retrieval-based strategy libraries, or verifiers to enhance repair quality.

## Related Work & Insights
- **vs Gödel Agent**: Gödel Agent demonstrated the possibility of policy self-modification but is better suited for strong models with large contexts; Polaris targets small models using experience abstraction and conservative patches to control resources.
- **vs Reflexion / Self-Refine**: Reflexion-style methods typically use feedback for the next answer, with updates remaining in verbal memory; Polaris generates persistent policy patches that change the agent's problem-solving program.
- **vs model editing**: Model editing modifies weights or representations, with opaque effects that are hard to audit; Polaris modifies explicit policy code, making changes clear but limited by the policy's expressiveness.
- **vs automated program repair (APR)**: APR repairs external programs, while Polaris repairs the agent's own policy. Both emphasize local patches and verification. This intersection deserves further exploration, such as adding test generation and patch ranking.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Combines Gödel Agent, experience abstraction, and minimal code patches for SLM self-improvement with a distinctive problem focus.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Covers four tasks, multiple base models, and failure mode analysis, though it lacks rigorous module-level ablation and long-term deployment validation.
- Writing Quality: ⭐⭐⭐⭐☆ The framework and failure analysis are clear; the appendix provides ample trajectories and prompts, though some notation is dense.
- Value: ⭐⭐⭐⭐☆ Insights into resource-constrained agents, runtime adaptation, and auditable self-improvement are significant, though it remains a step away from stable production systems.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Lightweight LLM Agent Memory with Small Language Models](lightweight_llm_agent_memory_with_small_language_models.md)
- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)
- [\[ACL 2026\] Meta-Tool: Efficient Few-Shot Tool Adaptation for Small Language Models](meta-tool_efficient_few-shot_tool_adaptation_for_small_language_models.md)
- [\[ACL 2026\] CLAG: Adaptive Memory Organization via Agent-Driven Clustering for Small Language Model Agents](clag_adaptive_memory_organization_via_agent-driven_clustering_for_small_language.md)
- [\[ICML 2026\] Scaling Small Agents Through Strategy Auctions](../../ICML2026/llm_agent/scaling_small_agents_through_strategy_auctions.md)

</div>

<!-- RELATED:END -->
