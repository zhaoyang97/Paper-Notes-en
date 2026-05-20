---
title: >-
  [Paper Note] PerfGuard: A Performance-Aware Agent for Visual Content Generation
description: >-
  [ICLR 2026][LLM Agent][Tool Selection] This paper proposes PerfGuard, a performance-aware agent framework for visual content generation. It replaces textual descriptions with a multi-dimensional scoring matrix to model t…
tags:
  - "ICLR 2026"
  - "LLM Agent"
  - "Tool Selection"
  - "Performance Boundary Modeling"
  - "Visual Generation"
  - "AIGC"
  - "Preference Optimization"
date: 2026-05-08
content_hash: f6ba9836568c29a1
---

# PerfGuard: A Performance-Aware Agent for Visual Content Generation

**Conference**: ICLR 2026
**arXiv**: [2601.22571](https://arxiv.org/abs/2601.22571)  
**Code**: [GitHub](https://github.com/FelixChan9527/PerfGuard)  
**Area**: AI Agent / Visual Content Generation
**Keywords**: LLM Agent, Tool Selection, Performance Boundary Modeling, Visual Generation, AIGC, Preference Optimization

## TL;DR
This paper proposes PerfGuard, a performance-aware agent framework for visual content generation. It replaces textual descriptions with a multi-dimensional scoring matrix to model tool performance boundaries (PASM), employs Adaptive Preference Updating (APU) to dynamically calibrate deviations between theoretical rankings and actual execution outcomes, and introduces Capability-Aligned Planning Optimization (CAPO) to guide the Planner in generating subtasks aligned with tool capabilities. PerfGuard comprehensively outperforms SOTA methods such as GenArtist and T2I-Copilot on image generation and editing tasks.

## Background & Motivation

**Background**: LLM-driven agents have become capable of automated task processing via reasoning and tool invocation, with multi-tool coordination systems such as CompAgent and GenArtist emerging in the visual content generation (AIGC) domain.

**The Problem of Idealized Assumptions**: Existing work broadly assumes that tool invocations always succeed, lacking systematic evaluation of actual tool execution success rates. This uncertainty in tool selection directly undermines the overall accuracy of agent planning and decision-making.

**Limitations of Textual Descriptions**: Current systems rely on generic textual descriptions to define tool capabilities (e.g., "capable of generating images semantically aligned with text"), which fails to distinguish fine-grained performance differences across models and cannot support precise tool matching.

**Absence of Performance Boundaries**: Taking text-to-image generation as an example, FLUX, SD3, and DALL·E 3 exhibit significant performance differences across dimensions such as color, shape, texture, and spatial relationships, yet agents cannot perceive these differences, introducing uncertainty into planning and execution.

**Insufficiency of Static Evaluation**: Even when benchmark scores are available, pre-established performance boundaries may deviate from actual task execution outcomes, necessitating dynamic adjustment based on real-world usage feedback.

**Disconnect Between Planning and Tools**: Existing methods do not account for actual tool performance capabilities during task planning. The Planner may generate subtasks that tools cannot complete with high quality, highlighting the need to integrate performance awareness into the planning process.

## Method

### Overall Architecture: Four-Role Agent System

PerfGuard is built upon a standardized agent system comprising four core roles:
- **Analyst**: Parses multimodal inputs and generates task summary $\tau^*$, target image semantics $s^*$, and evaluation objective $g$.
- **Planner**: Uses $\tau^*$, $s^*$, and tool performance profiles $\mathcal{B}$ to decompose tasks into subtasks $u_t$.
- **Worker**: Selects appropriate tools from the tool library to execute subtasks and produces image outputs $o_t$.
- **Self-Evaluator**: Assesses alignment between $o_t$ and objective $g$ across multiple dimensions; feedback is used for iterative optimization.

### Key Design 1: Performance-Aware Selection Modeling (PASM)

**Tool Performance Boundary Definition**: A multi-dimensional scoring system is constructed:
- Image generation tools: Evaluated across 7 dimensions based on T2I-CompBench (color, shape, texture, 2D spatial, 3D spatial, non-spatial semantics, numeracy).
- Image editing tools: Evaluated across 7 dimensions based on ImgEdit-Bench (addition, removal, replacement, attribute change, motion change, style transfer, background replacement).

**Performance-Driven Selection**: The Worker generates a preference weight vector based on subtask characteristics and multiplies it with the normalized performance matrix to obtain a tool suitability ranking:

$$\mathcal{W}_{task} = \pi_{\text{Worker}}(u_t, \mathcal{B}, \mathcal{D})$$

$$S_{tools} = \mathcal{W}_{task} \cdot \text{Normalize}(M_p)^\top$$

$$\mathcal{R} = \text{argsort}(S_{tools}, \text{descending})$$

where $M_p \in \mathbb{R}^{d \times l}$ is the tool performance boundary matrix ($d$ dimensions × $l$ tools), and $\mathcal{W}_{task} \in \mathbb{R}^{1 \times d}$ is the task preference weight vector.

### Key Design 2: Adaptive Preference Updating (APU)

An explore–exploit strategy is introduced: the top-$m$ high-scoring tools are selected alongside $n$ randomly sampled tools. After execution, the theoretical ranking is compared with the actual ranking, and the performance matrix is adaptively updated:

$$M_p^{\text{new}} = \text{Normalize}\big(M_p + \mathcal{W}_{task} \cdot \eta \cdot \Delta\big)$$

$$\Delta = \frac{\mathcal{R}_{theory} - \mathcal{R}_{actual}}{m+n}$$

- When a tool's actual performance exceeds theoretical expectations, its performance boundary score is increased; otherwise, it is decreased.
- $\eta$ is the update step size; experiments show that $\eta = 0.13$ achieves the optimal balance.
- Newly added tools are initialized with the average score of similar tools, ensuring they are not overlooked.

### Key Design 3: Capability-Aligned Planning Optimization (CAPO)

Step-aware Preference Optimization (SPO) is extended to the agent planning domain:
- At each step, $k$ candidate subtasks $\{u_t^1, u_t^2, \ldots, u_t^k\}$ are generated.
- The Self-Evaluator assesses the execution result of each subtask and selects winning/losing samples.
- Optimization objective (DPO variant):

$$\mathcal{L}(\theta) = -\mathbb{E}\Big[\log\sigma\Big(\alpha\big(\log\frac{p_\theta(u_t^w | \tau^*, s^*, \mathcal{B}, h_{t-1})}{p_{\text{ref}}(u_t^w | \tau^*, s^*, \mathcal{B}, h_{t-1})} - \log\frac{p_\theta(u_t^l | \tau^*, s^*, \mathcal{B}, h_{t-1})}{p_{\text{ref}}(u_t^l | \tau^*, s^*, \mathcal{B}, h_{t-1})}\big)\Big)\Big]$$

- A memory retrieval mechanism is integrated: CLIP similarity is used to retrieve historically successful task sequences as contextual guidance.

### Additional Key Design Highlights
- **Dual-granularity evaluator**: Global semantics $g^{global}$ and local semantics $g^{local}_i$ are combined with weighted aggregation for comprehensive evaluation.
- **Explore–exploit strategy in APU**: $\beta k$ candidates are retrieved from historical experience, while $(1-\beta)k$ are randomly generated, balancing exploitation and exploration.
- **Direct reuse of benchmark scores for the performance matrix**: Scores are adopted directly from T2I-CompBench and ImgEdit-Bench, reducing evaluation overhead.

## Key Experimental Results

### Basic Image Generation (T2I-CompBench)

| Method | Type | Color↑ | Shape↑ | Texture↑ | Spatial↑ | Non-Spatial↑ | Complex↑ |
|--------|------|--------|--------|----------|----------|-------------|----------|
| FLUX | Diffusion | 0.7407 | 0.5718 | 0.6922 | 0.2863 | 0.3127 | 0.3771 |
| SD3 | Diffusion | 0.8132 | 0.5885 | 0.7334 | 0.3200 | 0.3140 | 0.3703 |
| GoT | CoT | 0.4793 | 0.3668 | 0.4327 | 0.2238 | 0.3053 | 0.3255 |
| T2I-R1 | CoT | 0.8130 | 0.5852 | 0.7243 | 0.3378 | 0.3090 | 0.3993 |
| GenArtist | Agent | 0.8482 | 0.6948 | 0.7709 | 0.5437 | 0.3346 | 0.4499 |
| T2I-Copilot | Agent | 0.8039 | 0.6120 | 0.7604 | 0.3228 | 0.3379 | 0.3985 |
| **PerfGuard** | **Agent** | **0.8753** | **0.7366** | **0.8148** | **0.6120** | **0.3754** | **0.5007** |

### Advanced Image Generation (OneIG-Bench)

| Method | Type | Alignment↑ | Text↑ | Reasoning↑ | Style↑ |
|--------|------|-----------|-------|-----------|--------|
| FLUX | Diffusion | 0.786 | 0.523 | 0.253 | 0.368 |
| SD3 | Diffusion | 0.801 | 0.648 | 0.279 | 0.361 |
| T2I-R1 | CoT | 0.793 | 0.662 | 0.297 | 0.370 |
| T2I-Copilot | Agent | 0.821 | 0.679 | 0.318 | 0.386 |
| **PerfGuard** | **Agent** | **0.834** | **0.684** | **0.350** | **0.395** |

### Complex Image Editing (Complex-Edit Level-3)

| Method | IF↑ | PQ↑ | IP↑ | Overall↑ |
|--------|-----|-----|-----|----------|
| AnySD | 4.13 | 7.14 | 9.08 | 6.78 |
| Step1X_Edit | 7.95 | 8.66 | 7.70 | 8.10 |
| GenArtist | 6.14 | 7.24 | 6.19 | 6.52 |
| OmniGen | 7.52 | 8.86 | 8.01 | 8.13 |
| **PerfGuard** | **8.95** | **9.02** | **8.56** | **8.84** |

## Key Findings

1. **Textual descriptions offer almost no discriminative power for tool selection**: Relying solely on textual descriptions results in error rates as high as 77.8% (QWen3-14B) and 72.2% even with GPT-4o. The multi-dimensional performance scoring matrix reduces the error rate to 30.5%, and further to 14.2% with APU.

2. **PASM is the core contribution**: Ablation results show that introducing PASM yields +3.42% on the Color dimension and +5.7% on the Texture dimension, demonstrating that performance boundary modeling fundamentally improves tool selection accuracy.

3. **APU delivers significant adaptive gains**: The Complex metric improves from 0.4412 to 0.4738, as real execution feedback calibrates theoretical deviations and enables the performance matrix to more accurately reflect actual task requirements.

4. **CAPO endows the Planner with tool-aware capability**: The trained Planner can perceive tool performance boundaries and understand how operation ordering affects outcomes (e.g., editing the background first reduces the success rate of subsequent steps).

5. **The choice of update step size $\eta$ is critical**: $\eta = 0.1$ converges too slowly; $\eta = 0.15$ is fast early on but oscillates significantly later; $\eta = 0.13$ achieves the optimal error rate of 14.2% at step 800, balancing convergence speed and stability.

6. **Token efficiency advantage**: As the number of tools grows from 10 to 200, token consumption for conventional text-based methods increases catastrophically, whereas PerfGuard's performance-driven selection is unaffected by tool count, making it suitable for large-scale agent tool management.

## Highlights & Insights

- **Performance boundaries → quantifiable tool capability profiles**: Tool capabilities are transformed from vague textual descriptions into precise multi-dimensional numerical matrices, shifting tool selection from "guessing" to "computation"—a significant step toward the engineering of agent systems.
- **Closed-loop self-correction**: APU establishes a closed loop of "theoretical prediction → actual execution → deviation feedback → matrix update," enabling the system to continuously self-improve during deployment without relying on fixed benchmarks.
- **Transfer of SPO from image generation to agent planning**: Step-aware Preference Optimization is extended from the denoising process of diffusion models to the autoregressive planning process of agents, demonstrating the potential of preference optimization in agent decision-making.
- **Scalability validation**: Token consumption experiments confirm that PerfGuard remains efficient with large tool libraries (200+ tools), pointing toward a viable tool management solution for future agent ecosystems.

## Limitations & Future Work

- **Performance matrix depends on existing benchmarks**: Scores are adopted directly from T2I-CompBench and ImgEdit-Bench; tools in new domains or without established benchmarks require additional evaluation.
- **APU convergence requires sufficient samples**: Optimal performance is reached only after 800 steps, and infrequently used tools may receive insufficient updates.
- **The tool set caps the performance ceiling**: PerfGuard's advantage on Alignment and Text metrics is modest, constrained by the inherent generation capabilities of the tool set itself.
- **Inference overhead**: Generating $k$ candidate subtasks and evaluating each at every step increases computational cost compared to single-pass planning, despite reduced tool selection time.
- **Scope limited to visual generation and editing**: Effectiveness on other agent task categories (code generation, data analysis, etc.) has not yet been validated.

## Related Work & Insights

### vs. GenArtist (NeurIPS 2024)
GenArtist similarly employs multimodal LLMs to coordinate generation and editing tools, but lacks a performance-aware tool selection strategy, relying instead on detailed textual descriptions. Reasoning time increases significantly as the number of tools grows. PerfGuard's performance matrix with quantitative selection outperforms GenArtist on both tool selection time and accuracy (Complex: 0.4499 → 0.5007).

### vs. T2I-Copilot
T2I-Copilot achieves semantic decomposition through multi-agent collaboration but operates with a fixed tool set, limiting tool diversity and causing fine-grained details to be missed (e.g., spiral galaxies, green glasses). PerfGuard's performance-aware selection automatically matches the optimal tool, improving Reasoning from 0.318 to 0.350.

### vs. CLOVA (CVPR 2024)
CLOVA improves tool success rates through self-reflection and prompt tuning, but still operates at the individual tool level without modeling cross-tool performance comparisons. PerfGuard systematically models tool selection at a broader level, offering a more comprehensive solution.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of performance boundary modeling, adaptive updating, and planning optimization is novel, though individual components offer limited technical innovation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three benchmarks, ablation studies, efficiency analysis, and tool error rate analysis provide comprehensive coverage.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear and mathematical notation is well-formulated, though some descriptions are verbose.
- Value: ⭐⭐⭐⭐ Provides a viable engineering solution for tool selection in agent systems with strong practical applicability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] SceneAssistant: A Visual Feedback Agent for Open-Vocabulary 3D Scene Generation](../../CVPR2026/llm_agent/sceneassistant_a_visual_feedback_agent_for_openvoc.md)
- [\[ICLR 2026\] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents](simuhome_a_temporal-_and_environment-aware_benchmark_for_smart_home_llm_agents.md)
- [\[AAAI 2026\] COVR: Collaborative Optimization of VLMs and RL Agent for Visual-Based Control](../../AAAI2026/llm_agent/covrcollaborative_optimization_of_vlms_and_rl_agent_for_visu.md)
- [\[ICLR 2026\] AgentSynth: Scalable Task Generation for Generalist Computer-Use Agents](agentsynth_scalable_task_generation_for_generalist_computer-use_agents.md)
- [\[AAAI 2026\] History-Aware Reasoning for GUI Agents](../../AAAI2026/llm_agent/history-aware_reasoning_for_gui_agents.md)

</div>

<!-- RELATED:END -->
