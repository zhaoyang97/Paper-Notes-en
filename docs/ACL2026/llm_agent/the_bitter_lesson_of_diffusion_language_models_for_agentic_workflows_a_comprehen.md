---
title: >-
  [Paper Note] The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive Reality Check
description: >-
  [ACL2026][LLM Agent][Diffusion Language Models] This paper systematically evaluates the performance of diffusion Language Models (dLLMs) in embodied and tool-calling agents. It discovers that despite their potential for…
tags:
  - "ACL2026"
  - "LLM Agent"
  - "Diffusion Language Models"
  - "Agents"
  - "Tool Calling"
  - "Long-term Planning"
  - "DiffuAgent"
date: 2026-05-08
content_hash: 940ef9aba42139a0
---

# The Bitter Lesson of Diffusion Language Models for Agentic Workflows: A Comprehensive Reality Check

**Conference**: ACL2026  
**arXiv**: [2601.12979](https://arxiv.org/abs/2601.12979)  
**Code**: None  
**Area**: Agent / Diffusion Language Models  
**Keywords**: Diffusion Language Models, Agents, Tool Calling, Long-term Planning, DiffuAgent  

## TL;DR
This paper systematically evaluates the performance of diffusion Language Models (dLLMs) in embodied and tool-calling agents. It discovers that despite their potential for speed due to parallel decoding, they lag significantly behind Autoregressive (AR) LLMs in long-term causal planning and strict format generation. The authors introduce DiffuAgent to demonstrate that dLLMs are better suited as auxiliary non-causal modules for memory compression and tool filtering.

## Background & Motivation
**Background**: LLM agents have become the dominant paradigm in ReAct-style embodied tasks, tool calling, and interactive decision-making. Typical systems place the language model at the center of a loop, reading historical trajectories, environmental feedback, or tool descriptions to generate the next thought and action. While powerful, this process is naturally limited by AR decoding, where latency and inference costs scale with multi-round task length.

**Limitations of Prior Work**: dLLMs update multiple tokens simultaneously through parallel denoising, seemingly breaking the token-by-token generation bottleneck. However, agent scenarios require more than "fast text generation"; they demand that models adapt plans based on latest feedback, maintain state consistency across interactions, and output perfectly valid JSON or function call formats. While existing dLLMs perform close to same-scale LLMs on general benchmarks, their suitability as agent backbones lacks systematic verification.

**Key Challenge**: The advantage of dLLMs stems from non-autoregressive parallel generation, whereas critical agent capabilities often rely on causal, step-by-step, and error-correctable decision-making. Embodied tasks require strong constraints from new observations to the next action; tool calling requires precise placement of brackets, field names, and parameters. Bidirectional attention and fuzzy intermediate states during parallel denoising may weaken these functionalities.

**Goal**: The authors aim to answer three questions: first, whether current dLLMs can directly serve as backbones for embodied and tool-calling agents; second, whether their failures are accidental fluctuations or systematic patterns; and third, whether dLLMs can undertake auxiliary cognitive roles in agentic workflows if they cannot serve as backbones.

**Key Insight**: Instead of looking at single-turn QA, the paper selects ALFWorld, ScienceWorld, and BabyAI from AgentBoard along with BFCL-v3 tool-calling tasks to observe behavior in real multi-round closed loops. This perspective is vital because the short-text generation capabilities of dLLMs do not expose agent-level deficiencies such as "hitting the same wall repeatedly" or execution failures caused by a single character offset in formats.

**Core Idea**: Through unified experiments, dLLMs are placed as both agent backbones and auxiliary modules to prove that "diffusion-style parallel generation is suitable for non-causal compression and filtering, but not for the causal planning and symbolic precision required by current agents."

## Method
The method does not propose a single training algorithm but constructs a reality check and modular diagnostic framework. The authors first treat dLLMs as complete agent backbones, replacing AR LLMs like Qwen-8B and Ministral-8B. Subsequently, they propose DiffuAgent, which decomposes dLLMs into modules for memory, verification, tool selection, and format editing to analyze their utility in specific cognitive roles.

### Overall Architecture
The workflow is divided into two layers.

The first layer is the backbone reality check. Embodied tasks use the ReAct process where, at step $t$, the model generates intermediate thoughts $q_t$ and actions $a_t$ based on action/observation history $e_{1:t-1}$ and task description $u_{task}$. Tool-calling tasks involve outputting a set of structured calls $\mathcal{C}=\{(\tau_i,\alpha_i)\}$ based on user requests and tool libraries. In these processes, Qwen-8B and Ministral-8B are compared against LLaDA-8B, Dream-7B, FdLLM-7B, and DVar-8B.

The second layer is the DiffuAgent modular evaluation. Instead of controlling the whole loop, dLLMs are embedded as pluggable auxiliary cognitive modules. The embodied side includes a pre-hoc memory and a post-hoc early-exit verifier; the tool-calling side includes a pre-hoc tool selector and a post-hoc tool-call editor. This distinguishes between "failure as a backbone" and "unavailability of a specific local capability."

### Key Designs
1. **Backbone-level Failure Diagnostics**:
    - **Function**: Directly measures long-term planning and tool-calling capabilities when dLLMs serve as full agent controllers.
    - **Mechanism**: In AgentBoard, success rate and progress rate are measured per round. In BFCL-v3, the evaluation checks function names, parameters, structure, and multi-round state. The authors also track "retry loops"—repeating the same action for more than three consecutive steps—to capture the dLLM behavior of fixating on old plans.
    - **Design Motivation**: Average accuracy alone fails to identify if dLLMs suffer from "weak capability" or "interaction mechanism failure." Comparing embodied tasks with tool calling exposes both non-causal planning failures and fuzzy format generation.

2. **Modular Role Splitting in DiffuAgent**:
    - **Function**: Downgrades dLLMs from "full-process decision-makers" to auxiliary modules to observe their value in non-causal tasks.
    - **Mechanism**: A memory module compresses the past trajectory every $k_{mem}=5$ steps. An early-exit verifier determines if the trajectory is stuck in a loop every $k_{earlyexit}=5$ steps. A tool selector filters a subset of relevant tools, and a tool-call editor attempts to correct corrupted formats.
    - **Design Motivation**: Agent workflows consist of heterogeneous capabilities. dLLMs might struggle with causal action selection but could excel at summarization, filtering, or redundancy judgment due to parallel global modeling.

3. **Supplementary Validation of Failure Mechanisms**:
    - **Function**: Rules out explanations that failures are due to suboptimal decoding or can be fixed with simple format repairers.
    - **Mechanism**: The authors test dLLM decoding optimizations like APD, D2F, and DCD, alongside AR self-refine, periodic AR feedback, and light schema guardrails.
    - **Design Motivation**: To ensure fairness, supplementary validations show that while optimizations can alleviate local metrics, they do not resolve the bottlenecks in long-term causality and symbolic precision.

### Loss & Training
No new dLLMs were trained, and no new supervised losses were proposed. The focus is on inference-only evaluation: AR models are deployed via vLLM, and dLLMs are reproduced using Fast-dLLM/FastAPI, running on NVIDIA A800 80GB GPUs. Tasks use ReAct prompts and BFCL uses OpenAI API style templates to ensure comparisons focus on native agent behavior.

## Key Experimental Results

### Main Results
The results for embodied tasks are stark: AR Qwen-8B achieves a 45.0% average success rate, while the best dLLM, LLaDA-8B, reaches only 7.5%, and DVar-8B drops to 2.0%. The progress rate follows the same trend, indicating dLLMs struggle even with intermediate sub-goals.

| Model | ALFWorld SR | ScienceWorld SR | BabyAI SR | Avg. SR | Avg. Progress |
|------|----------------|---------------------|---------------|------------|------------|
| Qwen-8B | 76.1 | 26.7 | 32.1 | 45.0 | 62.1 |
| Ministral-8B | 45.5 | 13.3 | 36.6 | 31.8 | 54.9 |
| LLaDA-8B | 5.2 | 1.1 | 16.1 | 7.5 | 16.4 |
| Dream-7B | 0.7 | 0.6 | 8.9 | 3.4 | 8.7 |
| FdLLM-7B | 3.3 | 0.7 | 5.4 | 3.1 | 8.9 |
| DVar-8B | 0.7 | 0.0 | 5.4 | 2.0 | 8.9 |

Tool calling experiments show a similar gap. While Qwen-8B reaches a 57.8% overall success rate on BFCL, DVar-8B (the strongest dLLM) only hits 28.0%. Notably, in multi-turn tool calling, all dLLMs scored 0.0%, highlighting their inability to maintain state and format stability in interactive protocols.

| Model | Non-Live | Single-Turn Live Avg. | Multi-Turn Avg. | Hallucination Rel. | Hallucination Irrel. | Overall |
|------|----------|-----------------------|-----------------|--------------------|----------------------|---------|
| Qwen-8B | 87.5 | 78.0 | 12.5 | 94.4 | 68.0 | 57.8 |
| Ministral-8B | 49.8 | 60.0 | 4.0 | 66.7 | 58.0 | 39.5 |
| LLaDA-8B | 23.0 | 11.6 | 0.0 | 66.7 | 56.0 | 19.4 |
| Dream-7B | 4.2 | 1.5 | 0.0 | 27.8 | 77.0 | 13.6 |
| FdLLM-7B | 1.2 | 0.0 | 0.0 | 5.6 | 99.0 | 15.0 |
| DVar-8B | 35.0 | 34.1 | 0.0 | 44.4 | 63.0 | 28.0 |

### Ablation Study
DiffuAgent results provide more nuance. As backbones, dLLMs are weak; however, as memory modules, they significantly aid AR models. For Qwen-8B, adding LLaDA-8B memory improves SR from 28.4% to 40.5%, outperforming AR-based memory. This suggests dLLM global compression is effective for "history summarization."

| Agent | Memory Module | Avg. SR | Avg. Progress | Note |
|-------|-------------|------------|------------|------|
| Qwen-8B | w/o | 28.4 | 50.6 | Only recent interaction |
| Qwen-8B | Qwen-8B | 34.9 | 54.8 | AR memory gains |
| Qwen-8B | LLaDA-8B | 40.5 | 59.6 | Strongest memory setting |
| Qwen-8B | Dream-7B | 38.9 | 57.3 | Better than AR memory |
| Qwen-8B | FdLLM-7B | 35.6 | 54.8 | Subtle gains |
| Qwen-8B | DVar-8B | 37.3 | 56.8 | Consistently better than w/o |

For tool calling, the selector role is more suitable for dLLMs than the editor role. DVar-8B as a selector maintains some efficacy (11.5% success), but as an editor, success drops to 0.0%.

| Main Agent | Selector | Editor | BFCL Multi-Turn Avg. | Interpretation |
|----------|----------|--------|----------------------|------|
| Qwen-8B | Qwen-8B | - | 18.5 | AR selector is strongest |
| Qwen-8B | LLaDA-8B | - | 12.5 | dLLM filters relevant tools |
| Qwen-8B | Dream-7B | - | 13.0 | Selector performance near LLaDA |
| Qwen-8B | - | LLaDA-8B | 16.0 | LLaDA as editor is functional |
| Qwen-8B | - | FdLLM-7B | 2.0 | Format editing fails |
| Qwen-8B | DVar-8B | DVar-8B | 0.0 | Combined modules fail |

### Key Findings
- Efficiency advantages do not translate into agent success. High throughput models like FdLLM-7B fail to progress significantly in tasks.
- Embodied failure is characterized by the "retry loop," where models repeat actions despite environmental feedback.
- Tool-calling failure is primarily due to schema and parameter errors. dLLMs often identify the correct tool but produce invalid syntax.
- dLLMs are better for non-causal roles like memory compression and tool filtering rather than backbone decision-making.

## Highlights & Insights
- The paper's greatest value is placing the "diffusion LMs are faster" narrative into the agent loop. Speed is meaningful only if actions are valid; otherwise, high throughput just produces invalid trajectories faster.
- The modular design of DiffuAgent provides a refined conclusion: dLLMs are not useless for agents, but they shouldn't be the causal control core in the current paradigm.
- The concepts of "non-causal" vs. "fuzzy" failures are highly explanatory for embodied stalemates and schema collapses, respectively.

## Limitations & Future Work
- The coverage of dLLMs and benchmarks is limited (e.g., no web agents or software engineering tasks).
- Experiments are primarily inference-only, leaving training-based質变 (qualitative leaps) unexplored.
- The workflow still relies on AR LLMs as the main loop controller.
- Future work should focus on diffusion-native agents that incorporate causal states and grammar constraints directly into the denoising process.

## Related Work & Insights
- **vs. LLaDA / Dream**: While those works emphasize general performance, Ours demonstrates that general benchmarks do not guarantee agent suitability.
- **vs. ReAct / AgentBoard**: This study extends the evaluation from AR LLMs to dLLMs, specifically diagnosing repetition and error propagation.
- **vs. BFCL**: Attributes failures to symbolic precision issues under diffusion noise rather than just model scale.
- **vs. Agent Memory / Verifiers**: Suggests these auxiliary roles are the optimal home for dLLMs due to their global compression strengths.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Timely systematic check of dLLMs in agents.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Solid across main and modular tests, though task variety could be broader.
- Writing Quality: ⭐⭐⭐⭐☆ Clear argumentation with a strong "bitter lesson" theme.
- Value: ⭐⭐⭐⭐⭐ Highly influential for both diffusion and agent communities, clarifying the correct roles for dLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Towards GUI Agents: Vision-Language Diffusion Models for GUI Grounding](../../CVPR2026/llm_agent/towards_gui_agents_vision-language_diffusion_models_for_gui_grounding.md)
- [\[ACL 2026\] Dynamic Generation of Multi-LLM Agents Communication Topologies with Graph Diffusion Models](dynamic_generation_of_multi-llm_agents_communication_topologies_with_graph_diffu.md)
- [\[ACL 2026\] Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments](feedback-driven_tool-use_improvements_in_large_language_models_via_automated_bui.md)
- [\[ACL 2026\] Don't Adapt Small Language Models for Tools; Adapt Tool Schemas to the Models](don39t_adapt_small_language_models_for_tools_adapt_tool_schemas_to_the_models.md)
- [\[ACL 2026\] Polaris: A Gödel Agent Framework for Small Language Models through Experience-Abstracted Policy Repair](polaris_a_gödel_agent_framework_for_small_language_models_through_experience-abs.md)

</div>

<!-- RELATED:END -->
