---
title: >-
  [Paper Note] ViLoMem: Agentic Learner with Grow-and-Refine Multimodal Semantic Memory
description: >-
  [CVPR 2026][LLM Agent][grow-and-refine] ViLoMem equips Multimodal Large Language Models (MLLMs) with a "visual stream + logic stream" dual-channel semantic memory. This allows the agent, upon failing a task, to **attribute, store, and retrieve** perception errors and reasoning errors separately. By using a grow-and-refine incremental update strategy to avoid
tags:
  - CVPR 2026
  - LLM Agent
  - grow-and-refine
date: 2026-05-08
content_hash: 98f61c61b7843e3e
---
# ViLoMem: Agentic Learner with Grow-and-Refine Multimodal Semantic Memory

**Conference**: CVPR 2026  
**Paper**: [CVF Open Access](https://openaccess.thecvf.com/content/CVPR2026/html/Bo_ViLoMem_Agentic_Learner_with_Grow-and-Refine_Multimodal_Semantic_Memory_CVPR_2026_paper.html)  
**Code**: Project Page https://weihao-bo.github.io/ViLoMeo-page/ (Repository not explicitly open-sourced)  
**Area**: Agent / Multimodal Memory  
**Keywords**: Multimodal Memory, Agent Self-Improvement, Visual Perception Errors, Grow-and-Refine, Lifelong Learning

## TL;DR
ViLoMem equips Multimodal Large Language Models (MLLMs) with a "visual stream + logic stream" dual-channel semantic memory. This allows the agent, upon failing a task, to **attribute, store, and retrieve** perception errors and reasoning errors separately. By using a grow-and-refine incremental update strategy to avoid forgetting, it consistently improves pass@1 across six multimodal reasoning benchmarks and significantly reduces repetitive mistakes.

## Background & Motivation

**Background**: MLLMs have become powerful in single-example reasoning, yet they solve each problem "de novo"—solving independently, repeatedly deriving the same conclusions, and prone to repeated mistakes. To alleviate this, "memory-augmented agents" have emerged, storing past successful/failed trajectories for subsequent query and reuse (e.g., Dynamic Cheatsheet, ACE).

**Limitations of Prior Work**: The authors identify three fatal flaws in current memory agents. First is **brevity bias**: memories become shorter through repeated rewriting, gradually eroding critical domain knowledge. Second is the **single modality**: even for multimodal tasks, what is recorded is only a textual "high-level logic summary," completely losing visual grounding and perceptual cues. Third is being **logic-centric**: existing frameworks focus almost exclusively on reasoning patterns, neglecting the visual dimension.

**Key Challenge**: The authors disprove the assumption that "logic memory is sufficient" through ablation statistics—across six benchmarks, the **proportion of visual errors (59%–93%) remained higher than logic errors** (Fig. 4). In multimodal mathematical problems, perceptual errors regarding figures often exceed logic reasoning errors, and these visual errors **cascade**: misinterpreting a figure directly leads to downstream logical hallucinations. Therefore, recording only logic without vision is akin to blocking the downstream while letting the upstream overflow.

**Key Insight**: The authors draw inspiration from the human brain's "hub-and-spoke semantic memory" architecture—visual semantic associations reside in the inferior temporal/perirhinal cortex (visual spoke), while abstract reasoning rules reside in the temporoparietal cortex (logic spoke), integrated by the Anterior Temporal Lobe (ATL) as a hub. Semantic memory is inherently **multimodal and bifurcated**.

**Core Idea**: Replace single-stream logic memory with **dual-stream memory**—modeling "visual distraction patterns" and "logical hallucination errors" as two distinct structured schemas. These are coordinated through unified retrieval and incremental accumulation following the grow-and-refine principle, learning from both successes and failures.

## Method

### Overall Architecture

ViLoMem is a **plug-and-play** dual-stream memory plugin centered around a closed-loop **Memory Cycle**: given a multimodal input $x_i=(I_i, q_i)$ (image + question), the system maintains two memory banks—logic memory $\mathcal{M}^L$ (pure text reasoning principles) and visual memory $\mathcal{M}^V$ (visual principles + paired source images).

One cycle consists of four steps: **Retrieval** (parallel retrieval of related memories $R^L_i, R^V_i$) → **Utilization** (feeding them to the Solver to generate a candidate answer $\tilde{y}_i$) → **Verifier** (validating against ground truth $y_i$) → Once an error is detected ($\tilde{y}_i \ne y_i$), **Generation** is triggered to update both memory banks $\mathcal{M}^L_{i+1}, \mathcal{M}^V_{i+1}$ in parallel. This allows the agent to progressively improve through self-correction without manual supervision, with errors automatically attributed to the corresponding stream.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400}}}%%
flowchart TD
    A["Input (Image + Question)"] --> B["Dual-Stream Retrieval<br/>2-Stage Visual + Rich Logic Query"]
    B --> C["Solver Answer utilizing Memory"]
    C --> D["Verifier checks against Answer"]
    D -->|Correct| A
    D -->|Incorrect (Trigger)| E["Dual-Stream Memory Generation<br/>Error Attribution + Merge/New"]
    E --> F["Dual-Stream Semantic Memory Bank<br/>Visual Stream ‖ Logic Stream"]
    F --> B
```

### Key Designs

**1. Dual-Stream Semantic Memory: Separating Perception and Reasoning Errors**

Addressing the pain point that "single-stream logic memory loses vision and causes cascading errors," ViLoMem establishes two independent modality-specific memory banks. Logic memory $\mathcal{M}^L=\{m^L_1,\dots\}$ stores textual reasoning principles only; visual memory $\mathcal{M}^V=\{(m^V_1, I^V_1),\dots\}$ stores pairs of "visual principles + source images"—this step is crucial as visual knowledge is only meaningful when tied to specific images. Two different models share the work during generation: a LLM handles logic error attribution (formula misuse, logical fallacies, etc.), while a MLLM handles visual error attribution (object confusion, missing visual symbols, spatial misjudgment, etc.). This "bifurcation" directly corresponds to the division between visual and logic spokes in the brain's hub-and-spoke architecture. Ablations prove these streams capture **complementary rather than redundant** error patterns.

**2. Grow-and-Refine Memory Generation: Combating Brevity Bias via Similarity Gating**

To solve "detail erosion caused by repeated rewriting," memory generation employs **similarity filtering** before deciding to merge or create new entries. On the visual side, one MLLM call produces error indicators and correction principles $(e^V_i, g^V_i)=\text{AnalyzeGenerate}^V(I_i, q_i, \tilde{y}_i, y_i)$. Textual similarity $s^V_j=\text{Sim}(\phi_T(g^V_i), \phi_T(m^V_j))$ is calculated before storage. If the maximum similarity exceeds threshold $\tau^V$, a merge is triggered, replacing entry $j^*$ with a fused version:

$$\mathcal{M}^V_{i+1}=\mathcal{M}^V_i \setminus \{(m^V_{j^*}, I^V_{j^*})\} \cup \{(\text{Merge}^V(m^V_{j^*}, g^V_i), I^V_{j^*})\}$$

Otherwise, a new entry is added: $\mathcal{M}^V_{i+1}=\mathcal{M}^V_i \cup \{(g^V_i, I_i)\}$. The logic side follows a similar logic but includes a gate—updating only if the error is classified as $e^L_i=\text{Logical}$ and the principle is non-empty:

$$\mathcal{M}^L_{i+1}=\begin{cases} \mathcal{M}^L_i \setminus \{m^L_{j^*}\} \cup \{m^L_{\text{new}}\} & s^L_{j^*} > \tau^L \\ \mathcal{M}^L_i \cup \{g^L_i\} & s^L_{j^*} \le \tau^L \\ \mathcal{M}^L_i & \text{Otherwise} \end{cases}$$

This "merge if similar, add if distinct" strategy prevents infinite bank expansion while preserving details, avoiding catastrophic forgetting inherent in simple iterative rewriting.

**3. Dual-Stream Retrieval: "Where to Look" for Vision, "Which Rule" for Logic**

For visual retrieval, ViLoMem uses a **two-stage "multimodal-to-text" pipeline**. First, multimodal embeddings calculate visual similarity $s^M_j=\text{Sim}(\phi_M(I_i), \phi_M(I^V_j))$ between the query and bank images to recall top-$k^M$ candidates. Since image similarity alone is insufficient, the second stage uses a "question-aware query" $\tilde{q}_i$ to rerank candidates based on text $s^T_j=\text{Sim}(\phi_T(\tilde{q}_i), \phi_T(m^V_j))$, filtering by $\tau^V$ to get top-$k^V$. This ensures retrieved visual memories are both visually similar and **semantically relevant**, precisely identifying "visual trap regions" for specific questions. Furthermore, retrieved visual patterns generate a **question-aware attention heatmap**, highlighting historically error-prone regions as auxiliary input to the solver.

The logic stream utilizes pure text semantic matching, but with **analysis-before-retrieval**: instead of matching the raw question, the LLM extracts the subject area and core concepts $a_i=\text{Analyze}^L(q_i, \tilde{y}_i)$ to form a rich query $\tilde{q}_i=[q_i; a_i]$. This precisely locates the problem type and selects relevant logic schemas without being misled by surface-level lexical similarity.

**4. Closed-loop Solver–Verifier Self-Correction: Unsupervised Error Attribution**

Final generation is $\tilde{y}_i=\text{Gen}(I_i, q_i, R^L_i, R^V_i)$, where the solver consumes raw input, logic principles (structured reasoning), and visual principles (explicit perceptual priors). The cycle is driven by the Verifier: it compares against ground truth, filters redundant/invalid trajectories, only activates memory generation upon error, and automatically attributes the failure to the visual or logic stream—requiring no manual labeling. This closed loop allows the agent to transfer knowledge across domains and progressively suppress repetitive errors.

## Key Experimental Results

### Main Results

Evaluated across six multimodal reasoning benchmarks (MMMU / MathVista / MathVision / HallusionBench / MMStar / RealWorldQA), utilizing GPT-4.1, Qwen3-VL-235B, and Qwen3-VL-8B across three configurations (baseline / step-by-step reasoning / +ViLoMem). Selected results (pass@1 accuracy):

| Model / Configuration | MMMU | MathVista | MathVision | HallusionBench | RealWorldQA |
|---|---|---|---|---|---|
| GPT-4.1 (step) | 74.16 | 74.27 | 47.47 | 74.44 | 72.03 |
| GPT-4.1 (+ViLoMem) | **77.26** | **76.88** | **53.95** | **75.29** | **74.38** |
| Qwen3-VL-235B (step) | 75.97 | 83.66 | 62.17 | 74.58 | 78.66 |
| Qwen3-VL-235B (+ViLoMem) | **79.40** | **84.98** | **62.83** | **75.21** | 77.22 |
| Qwen3-VL-8B (step) | 65.52 | 77.80 | 48.35 | 73.08 | 70.85 |
| Qwen3-VL-8B (+ViLoMem) | **69.90** | **77.87** | **49.34** | **73.19** | **73.59** |

Mathematical reasoning saw the largest gains (GPT-4.1 +6.48 on MathVision, +2.61 on MathVista), validating that math problems rely heavily on visual grounding. The small 8B model improved significantly on MMMU (+4.38) and RealWorldQA (+2.74), showing structured memory can compensate for limited parameter capacity.

### Ablation Study

Validating dual-stream necessity using GPT-4.1 on MMMU and MathVista:

| Configuration | MMMU | MathVista | Description |
|---|---|---|---|
| GPT-4.1 (step) | 74.16 | 74.27 | CoT only |
| w/o logic memory | 76.64 | 75.59 | Logic stream removed |
| w/o visual memory | 76.88 | 75.66 | Visual stream removed |
| + ViLoMem | 77.26 | 76.88 | Full dual-stream |
| + ViLoMem & attention | **78.21** | 76.87 | With attention heatmap |

Removing either stream causes a drop, proving the two streams are complementary. Removing logic memory had a larger impact on MathVista (high frequency of formulaic errors), while removing visual memory impacted both benchmarks consistently (prevalence of visual distraction).

### Key Findings

- **Vision is the Primary Bottleneck**: Across six benchmarks, visual errors account for 59%–93% of generated memories, yet both streams contribute equally during retrieval (Fig. 4), suggesting visual memory generation is frequent while logic memory has high reuse rates.
- **Cross-Model Distillation**: Utilizing memories generated by stronger models for weaker models (Table 3), the 8B model outperformed its own self-generated memory (MMMU +1.36, MathVista +1.33). This indicates strong models' error patterns encode higher-quality generalization strategies.
- **Scalability without Forgetting**: In stress tests on WeMath (Table 5), accuracy increased monotonically from 72.53 to 74.58 as memory scale grew from 15k to 150k tokens. Progressive cross-benchmark memory (74.58) even outperformed direct WeMath-specific memory (73.85), proving abstract patterns transfer to unseen problems.
- **Heterogeneity in Cross-Domain Transfer**: MathVision/RealWorldQA benefit from shared memory (both rely on spatial reasoning), but MathVista and HallusionBench (diagrams vs natural images) see conflicts when domain gaps are large (Table 4), justifying the choice of separate bank maintenance.

## Highlights & Insights
- **Solid observation on "Visual error cascades into logical hallucination"**: The authors quantified visual errors at 59%–93%, providing a rigorous argument for why visual memory must be independently modeled.
- **Visual memory stores "Principles + Source Image" pairs rather than just text**, which is the fundamental difference from logic-centric methods. Visual knowledge loses meaning without images; paired storage and two-stage retrieval solve the non-trivial problem of visual memory lookup.
- **Question-aware attention heatmaps** transform "where to look" into an explicit, transferable spatial prior, effectively highlighting historical pitfalls for the solver.
- **Grow-and-Refine similarity gating** is a practical trick against brevity bias: merging only when similar and adding when distinct preserves details while controlling growth, a mechanism applicable to any long-term memory system.

## Limitations & Future Work
- **Incomplete dual-stream decoupling**: If the solver has a strong textual bias and ignores the image, the trajectory lacks visual information, preventing effective visual memory generation; conversely, poor visual descriptions make it hard to distinguish visual from logic errors.
- **Dependency on strong Verifiers and Ground Truth**: The cycle relies on ground truth to trigger generation, which works for benchmarks but remains an open problem for open-domain scenarios without standard answers.
- **Limited gains on knowledge-intensive tasks**: Improvements on MMMU (fact retrieval) were modest, as the memory mechanism helps less with "memorization-type" knowledge.
- **High inference overhead**: The pipeline relies on multiple external models (235B for generation, Qwen2.5 for embeddings), necessitating a trade-off between performance and latency/cost in deployment.

## Related Work & Insights
- **vs Context Engineering (ReAct / Reflexion / TextGrad)**: These modify prompts without weight changes, but context is transient and suffers from brevity bias. ViLoMem uses persistent dual-stream banks to solve the "transience" flaw.
- **vs Long-term Memory Agents (Dynamic Cheatsheet / ACE)**: These store success/failure strategies but remain logic-centric and unimodal, losing visual grounding. ViLoMem's core contribution is the visual stream, proving it as the primary bottleneck.
- **vs Human Hub-and-Spoke Semantic Memory**: This work maps a cognitive architecture (ATL hub + modality spokes) to an AI system. The "cognitive science to engineering" mapping is highly insightful.

## Rating
- Novelty: ⭐⭐⭐⭐ (Dual-stream multimodal memory + paired visual storage + attention feedback is a clear advancement in memory agents).
- Experimental Thoroughness: ⭐⭐⭐⭐ (6 benchmarks across 3 models plus extensive ablation/transfer analysis; lacks end-to-end cost analysis).
- Writing Quality: ⭐⭐⭐⭐ (Well-argued motivation regarding cascading errors and clear framework diagrams).
- Value: ⭐⭐⭐⭐ (Plug-and-play, training-free, and supports cross-model distillation; a useful paradigm for continuous learning from errors).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning](worldmm_dynamic_multimodal_memory_agent_for_long_video_reasoning.md)
- [\[ICML 2026\] HawkesLLM: Semantic Uncertainty Propagation in Agentic Text Simulation](../../ICML2026/llm_agent/hawkesllm_semantic_uncertainty_propagation_in_agentic_text_simulation.md)
- [\[CVPR 2026\] Experience Transfer for Multimodal LLM Agents in Minecraft Game](experience_transfer_for_multimodal_llm_agents_in_minecraft_game.md)
- [\[ECCV 2024\] VideoAgent: A Memory-augmented Multimodal Agent for Video Understanding](../../ECCV2024/llm_agent/videoagent_a_memory-augmented_multimodal_agent_for_video_understanding.md)
- [\[CVPR 2026\] ReFAct: Empowering Multimodal Web Agents with Visual and Context Focusing](refact_empowering_multimodal_web_agents_with_visual_and_context_focusing.md)

</div>

<!-- RELATED:END -->
