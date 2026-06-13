---
title: >-
  [Paper Note] HierVA: Hierarchical Visual Agent — Managing Contexts in Joint Image-Text Space for Advanced Chart Reasoning
description: >-
  [ACL 2026][Multimodal VLM][chart QA] HierVA utilizes a "manager–worker" bilayer multimodal agent to manage both image and text contexts during chart reasoning through a disciplined "acquire–limit–distill" approach. Witho…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "chart QA"
  - "hierarchical agent"
  - "multimodal context"
  - "zoom-in"
  - "training-free"
date: 2026-05-08
content_hash: 36e975cd6d9e75e1
---

# HierVA: Hierarchical Visual Agent — Managing Contexts in Joint Image-Text Space for Advanced Chart Reasoning

**Conference**: ACL 2026  
**arXiv**: [2605.04304](https://arxiv.org/abs/2605.04304)  
**Code**: TBD  
**Area**: Multimodal VLM / Agent / Chart Reasoning  
**Keywords**: chart QA, hierarchical agent, multimodal context, zoom-in, training-free

## TL;DR
HierVA utilizes a "manager–worker" bilayer multimodal agent to manage both image and text contexts during chart reasoning through a disciplined "acquire–limit–distill" approach. Without training, it consistently outperforms strong baselines like CoT and "thinking with images" on complex chart reasoning benchmarks such as CharXiv.

## Background & Motivation

**Background**: Chart Question Answering (Chart QA) is a core capability for research assistants and document understanding systems. While CoT enables MLLMs to perform explicit reasoning, the recent "thinking with images" paradigm (OpenAI 2025, Lai 2025, Zheng 2025, etc.) allows models to iteratively acquire additional visual evidence (such as zoom-in crops) during the reasoning process, integrating visual details into the reasoning trace.

**Limitations of Prior Work**: MLLMs have achieved 90%+ accuracy on single-image, single-step tasks (ChartQA-style). however, they struggle with complex charts involving multiple subplots and multi-step reasoning (CharXiv reasoning split). Existing methods fail because CoT becomes distracted by irrelevant elements in the global image, and "thinking with images" continuously appends each zoom-in crop to the context. This leads to **monotonic context growth**, where images consume tokens and intermediate steps accumulate, diluting global reference information.

**Key Challenge**: Complex chart reasoning is essentially a hybrid image–text task requiring the simultaneous retention of "small-scale details" and "multi-step intermediate results." However, LLM contexts are finite, and the intermixing of text and image information causes mutual dilution, leading to chaos in later steps.

**Goal**: To establish disciplined context management for multimodal reasoning without training any models—acquiring necessary details while promptly distilling unnecessary intermediate products.

**Key Insight**: Transposing the same discipline used for managing text reasoning traces (plan distillation, scope, summarize) onto visual contexts, enforced by a **manager-worker hierarchical architecture**.

**Core Idea**: A manager maintains a refined global context, while multiple workers perform tasks in isolated local contexts. Each zoom-in or calculation result only returns a distilled summary to the manager.

## Method

### Overall Architecture
Input: Chart $I_0$ + natural language question $q$; Output: Answer $a$. The Manager maintains a global context $C_M = \{q, I_0, \text{refined plan}, \text{distilled summaries}\}$, while Workers use isolated local contexts $C_{W_t} = \{\text{task instr}, \text{optional skill}, \text{single image}\}$. The Manager operates via the control loop in Algorithm 1: it first performs two-stage planning (coarse plan then refinement, retaining only the refined version), then loops through [Termination judgment → CreateNextTask → ExecuteWorker → Append distilled summary to $C_M$] until a final $\boxed{}$ answer is provided.

### Key Designs

1. **Manager–Worker Hierarchy + Encapsulation**:

    - **Function**: Completely decouples global planning from local execution.
    - **Mechanism**: The Manager only observes its own $C_M$. After a Worker completes a task, its internal reasoning trace is not appended to $C_M$; instead, only a "single-sentence fact" or an "image crop handle" is returned as a distilled result. Each Worker receives only a single image (original or crop) and minimal task instructions, forcing a scoped evidence approach.
    - **Design Motivation**: Directly counteracts monotonic context growth. Prior "thinking with images" methods dumped all worker deliberations into the primary context; HierVA uses an abstraction barrier to block local noise.

2. **Adaptive Zoom as an Explicit Action**:

    - **Function**: Allows the manager to acquire high-resolution evidence on demand during reasoning.
    - **Mechanism**: Abstracts zoom into an image-expected task. Upon receiving this task, the worker calls a zoom-in tool, specifies a bounding box, and returns a cropped and resized high-resolution image. The Manager’s action space is a binary choice: (a) request new visual evidence (zoom) or (b) request text facts (reading values, comparisons, calculations).
    - **Design Motivation**: Small text, ticks, and legends in charts are unreliable at global scales and must be enlarged as needed. Elevating zoom to a first-class action is far more controllable than merely suggesting "look closer" in a prompt.

3. **Skill Routing + Triple Context Distillation**:

    - **Function**: Enables workers to utilize specialized skills (e.g., code tools) at appropriate times without polluting the global context.
    - **Mechanism**: Maintains a compact skill library $\mathcal{S}$, where each skill is a brief markdown procedural description. The Manager selects a set of relevant skills for each task, performing **just-in-time** injection into the worker's system prompt; the Manager never sees the skill content itself. This is paired with triple distillation: (1) Plan distillation retains only the final refined plan; (2) Worker encapsulation isolates worker traces; (3) Result distillation compresses worker answers into single sentences or crop handles before appending.
    - **Design Motivation**: The naive approach of stuffing all possible skills into the base prompt causes immediate context bloating. Just-in-time injection retains capability without polluting the main reasoning line.

### Loss & Training
The method is training-free, involving no parameter updates. All improvements stem from prompt orchestration design, reusing the same base MLLM (Qwen3VL-A22B in experiments).

## Key Experimental Results

### Main Results: CharXiv reasoning split
HierVA is compared against Direct / CoT / CoT-Plan / Thinking w/ Images baselines using Qwen3VL-A22B as the base model. Metrics include overall Acc and sub-types: Extr / First / Read / RevR / Comp / Freq, along with Peak Token #.

| Method | Image Tools | Skills | CharXiv-All | Peak Tok # |
|------|-------------|--------|-------------|------------|
| Direct | — | — | 45.7 | 702 |
| CoT | — | — | 62.1 | 1926 |
| CoT-Plan | — | — | 62.4 | 1947 |
| Thinking w/ Images | zoom | — | (See original table) | — |
| **HierVA (Ours)** | zoom + code | ✓ | **Consistently outperforms all baselines** | Controlled growth |

On ChartQA + synthetic multi-subplot charts (sp#1→sp#6): As the number of subplots increases, all methods show a performance drop, but HierVA only drops by **1.5%**, compared to 2.6% for CoT, 3.8% for CoT-Plan, and 5.4% for Direct.

| Method | ChartQA | sp#1 | sp#2 | sp#4 | sp#6 |
|------|---------|------|------|------|------|
| Direct | 88.9 | 88.5 | 86.5 | 84.8 | 83.1 |
| CoT | 90.2 | 90.1 | 89.2 | 88.4 | 87.5 |
| Thinking w/ Images | 89.9 | 89.7 | 88.5 | 87.4 | 87.5 |
| **HierVA** | 89.9 | 89.7 | 89.2 | 88.5 | **88.2** |

### Ablation Study

| Configuration | Key Effect | Description |
|------|---------|------|
| Full HierVA | Best | Manager + worker + triple distillation |
| w/o Hierarchical Architecture | Significant Drop | Degenerates into thinking-with-images |
| w/o Scoped Visual Context | Moderate Drop | Workers are distracted by the full image |
| w/o Distilled Context | Moderate Drop | Manager context bloats, leading to multi-step reasoning failure |

### Key Findings
- Context management provides minimal gain for simple questions (ChartQA-style single-step retrieval), where HierVA performs on par with thinking-with-images. The advantage is truly realized on complex charts requiring multi-step reasoning.
- The relative advantage of HierVA becomes more pronounced as chart complexity (number of subplots) increases, confirming that long-chain reasoning is the primary test of context management.
- The three distillation mechanisms are complementary; removing any single one leads to a performance drop, indicating that plan, worker, and result stages are all sources of context bloating.

## Highlights & Insights
- **Transferring "Text Context Management" to "Image Context Management"**: A core insight is that image and text are homogeneous at the token-budget level, making distillation, scope, and encapsulation equally applicable. This framing itself is valuable.
- **Just-in-Time Skill Injection**: Compared to the traditional approach of writing all skills into the system prompt, this on-demand injection design is applicable to all tool-use agents.
- **Zoom as a First-Class Action**: Modeling zoom explicitly as a typed action, rather than burying "look closer" inside CoT, makes scheduling and debugging more controllable.

## Limitations & Future Work
- Completely dependent on the instruction-following capability of the base MLLM, which may fail for smaller models.
- Peak token counts remain high, requiring further compression for latency-sensitive scenarios.
- Evaluation is limited to CharXiv reasoning split + ChartQA + synthetic charts; it has not yet been validated on more diverse visual documents like dashboards, maps, or flowcharts.
- Lacks a learned skill selection strategy, relying on manager prompt heuristics, which may limit cross-domain generalization.

## Related Work & Insights
- **vs CoT / CoT-Plan**: Pure text CoT cannot see details, and CoT-Plan cannot dynamically acquire evidence; HierVA incorporates images into the working memory.
- **vs Thinking w/ Images** (Zheng 2025): Both allow zooming, but HierVA uses hierarchy and distillation to prevent monotonic context growth, serving as a structural upgrade.
- **vs ReAct / Tool Agents**: Similar underlying logic (agent + tool), but HierVA emphasizes the overlooked aspect of multimodal context management.

## Rating
- Novelty: ⭐⭐⭐⭐ The manager-worker + distillation framing provides a clear new perspective in multimodal agents.
- Experimental Thoroughness: ⭐⭐⭐ Primarily focused on CharXiv + ChartQA; the synthetic subplot experiment is clever but benchmark coverage could be broader.
- Writing Quality: ⭐⭐⭐⭐ Motivation and design principles are tightly linked, with triple distillation addressing the core problem.
- Value: ⭐⭐⭐⭐ Training-free and plug-and-play, making it directly applicable to chart assistant applications.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] TEMA: Anchor the Image, Follow the Text for Multi-Modification Composed Image Retrieval](tema_anchor_the_image_follow_the_text_for_multi-modification_composed_image_retr.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[ACL 2026\] Learning More from Less: Exploiting Counterfactuals for Data-Efficient Chart Understanding](learning_more_from_less_exploiting_counterfactuals_for_data-efficient_chart_unde.md)
- [\[CVPR 2026\] KEC: Hierarchical Textual Knowledge for Enhanced Image Clustering](../../CVPR2026/multimodal_vlm/kec_hierarchical_textual_knowledge_clustering.md)
- [\[ACL 2026\] HiPrune: Hierarchical Attention for Efficient Token Pruning in Vision-Language Models](hiprune_hierarchical_attention_for_efficient_token_pruning_in_vision-language_mo.md)

</div>

<!-- RELATED:END -->
