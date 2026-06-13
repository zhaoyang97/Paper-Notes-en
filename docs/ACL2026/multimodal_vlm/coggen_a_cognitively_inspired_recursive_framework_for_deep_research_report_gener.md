---
title: >-
  [Paper Note] CogGen: A Cognitively Inspired Recursive Framework for Deep Research Report Generation
description: >-
  [ACL 2026][Multimodal VLM][Deep research reports] CogGen proposes a multi-agent recursive framework simulating the human cognitive writing process. It implements global restructuring through macro-cognitive loops…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Deep research reports"
  - "recursive writing framework"
  - "multimodal fusion"
  - "cognitive load assessment"
  - "multi-agent"
date: 2026-05-08
content_hash: 46512bd09ac0bb13
---

# CogGen: A Cognitively Inspired Recursive Framework for Deep Research Report Generation

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.17072](https://arxiv.org/abs/2604.17072)  
**Code**: [GitHub](https://github.com/NJUNLP/CogGen)  
**Area**: Multimodal VLM  
**Keywords**: Deep research reports, recursive writing framework, multimodal fusion, cognitive load assessment, multi-agent

## TL;DR
CogGen proposes a multi-agent recursive framework simulating the human cognitive writing process. It implements global restructuring through macro-cognitive loops, parallel chapter refinement through micro-cognitive cycles, and semantic-level text-chart collaborative planning via Abstract Visual Representation (AVR). It reaches human expert levels on the OWID benchmark and outperforms Gemini Deep Research.

## Background & Motivation

**Background**: Automated deep research report generation is a frontier application for LLMs. Existing solutions consist of single-agent systems (e.g., Gemini Deep Research) and multi-agent frameworks (e.g., STORM, Co-STORM), all of which follow linear predefined workflows.

**Limitations of Prior Work**: Linear workflows cannot backtrace or modify content once generated—when downstream findings invalidate upstream organizational logic, "reverse restructuring" is impossible. Additionally, the generation of text and charts is typically asynchronous and decoupled, resulting in charts acting as mere illustrations rather than organic components of the argumentation.

**Key Challenge**: Writing by experts is a non-linear recursive process (plan → write → review → restructure → rewrite), whereas existing AI writing frameworks are linear forward processes, failing to achieve global consistency across chapters and deep text-chart synergy.

**Goal**: Construct a recursive report generation framework that supports global restructuring and multimodal semantic-level collaboration.

**Key Insight**: Design a framework based on Flower & Hayes' cognitive process theory of writing and Cognitive Offloading theory.

**Core Idea**: A hierarchical recursive architecture (Macro-loop for global restructuring + Micro-cycle for chapter refinement) and Abstract Visual Representation (AVR) to decouple chart generation from reasoning.

## Method

### Overall Architecture
CogGen consists of three peer cognitive agents: Planner (retrieval + structural planning), Writer (text writing + visual intent definition), and Reviewer (real-time monitoring + post-evaluation). The macro-cognitive loop recurses at the global report level: plan → write → review → feedback → re-plan. The micro-cognitive cycle executes a Search-Replan-Write loop in parallel at the chapter level.

### Key Designs

1.  **Macro-Cognitive Loop**:

    - **Function**: Enables global reverse restructuring to solve the "forward-locking" problem of linear workflows.
    - **Mechanism**: Treats the outline $\mathcal{O}$ as a mutable object rather than a fixed plan. In each iteration, the Planner generates the outline → Writer generates chapter drafts in parallel → Reviewer evaluates the complete draft and generates feedback $\Delta^{(t)}$ → Planner revises the outline $\mathcal{O}^{(t+1)} = A_p(Q, \{\mathcal{O}^{(t)}, \Delta^{(t)}\}|K)$. A strict monotonic improvement constraint is designed to accept updates only when the Reviewer verifies distinct quality gains, preventing infinite oscillation.
    - **Design Motivation**: Human writing is recursive—completing the latter half often leads to modifying the organizational logic of the first half. This ability is crucial for generating high-quality long documents.

2.  **Micro-Cognitive Cycle**:

    - **Function**: Generates chapter content in parallel while ensuring cross-chapter consistency.
    - **Mechanism**: Multiple threads execute "Search → Re-plan → Write" cycles in parallel, with each thread using the global outline $\mathcal{O}^{(t)}$ as a read-only constraint. Chapter-specific retrieval results are stored in local thread caches. Cross-chapter conflicts are not resolved locally but deferred to the Reviewer for unified arbitration in the macro-loop (Deferred Update Policy), avoiding context oscillation issues seen in serial revisions.
    - **Design Motivation**: Parallel generation improves efficiency, but needs to resolve recursive modification traps where modifying one section to accommodate another leads to a cycle of endless updates.

3.  **Abstract Visual Representation (AVR)**:

    - **Function**: Achieves semantic-level collaborative planning of text and charts rather than post-hoc insertion.
    - **Mechanism**: The Writer generates structured semantic descriptions (Title, Chart_Type, X/Y_Axis, Data_Source, Purpose) instead of executable code. A Renderer Agent translates these semantic intents into ECharts/Mermaid code and renders them in a headless browser. This allows the Writer to iteratively modify the visual plan like "semantic tokens" without dealing with pixel-level details.
    - **Design Motivation**: Based on Cognitive Offloading theory—separating visual design decisions from writing reasoning reduces the Writer's cognitive load, allowing focus on narrative logic.

### Loss & Training
CogGen is a purely inference-time framework and involves no training. It utilizes GPT-4.1 as the backbone for each agent and GPT-4.1-Mini for search expansion, with a temperature of 0.5.

## Key Experimental Results

### Main Results

| Dataset | Method | Avg Score | Organization | Depth | Alignment | Synergy |
|---------|--------|-----------|--------------|-------|-----------|---------|
| OWID | Human Expert (Ref) | 0.4997 | 0.4986 | 0.5000 | 0.5000 | 0.5000 |
| OWID | CogGen | 0.4992 | 0.4972 | 0.5813 | 0.4806 | 0.4326 |
| OWID | WriteHere | 0.4502 | 0.4912 | 0.5503 | 0.3846 | 0.3312 |
| OWID | STORM | 0.3205 | 0.4253 | 0.4443 | 0.1675 | 0.1667 |
| WildSeek | Gemini DR (Ref) | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| WildSeek | CogGen | 0.5341 | 0.5389 | 0.5000 | 0.5544 | 0.5437 |

### Ablation Study

| Configuration | Avg Score | Description |
|---------------|-----------|-------------|
| GPT-4.1 + Search (No Framework) | 0.4119 | Single-agent baseline |
| CogGen w/o Review | 0.4681 | Significant drop after removing Reviewer |
| CogGen Two-stage (No Native Multimodal) | 0.4904 | Separate generation of text and charts |
| CogGen Full | 0.4994 | Synergy of all components |

### Key Findings
- CogGen reaches human expert levels on OWID (0.4992 vs 0.4997) and outperforms Gemini Deep Research on WildSeek (0.5341 vs 0.5000).
- Multimodal alignment and synergy (D4, D5) are the core advantages of CogGen over STORM/Co-STORM (score gap exceeds 0.3).
- Removing the Reviewer leads to the largest performance decline, indicating the review-feedback loop is essential for quality assurance.
- AVR provides significant improvements in data accuracy compared to FDV (Direct Code Generation).

## Highlights & Insights
- The **Macro-Micro Dual-layer Recursive** design accurately simulates the non-linear characteristics of human writing—the ability to revisit and restructure the outline after finishing the full text is key to surpassing linear systems.
- The **Deferred Update Policy** cleverly addresses site context oscillation in parallel generation—by leaving conflicts to the global reviewer rather than local modification, recursive modification traps are avoided.
- AVR decouples "what to display" from "how to draw," which can be generalized to any scenario requiring collaborative text-code generation.

## Limitations & Future Work
- Dependency on closed-source models such as GPT-4.1 results in high costs and low reproducibility.
- The convergence speed of recursive loops is not fully analyzed; actual generation times may be long.
- While theoretically grounded, the evaluation framework CLEF depends on GPT-5 as an evaluator, introducing potential evaluation bias.
- Supports only static text and charts, without interactive visualizations.

## Related Work & Insights
- **vs STORM/Co-STORM**: These use multi-perspective QA and collaborative writing but lack global restructuring capabilities.
- **vs WriteHere**: Supports recursive decomposition but remains a forward-generation process, unable to modify previously generated content in reverse.
- **vs Gemini Deep Research**: Commercial systems are still limited by fixed frameworks in the writing execution phase; CogGen outperforms its output quality on WildSeek.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovative systematic application of cognitive writing theory in AI report generation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two datasets, multiple baseline comparisons, detailed ablations, and human evaluation.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear theoretical motivation, excellent illustrations, and fluid narrative.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] WeatherSyn: An Instruction Tuning MLLM For Weather Forecasting Report Generation](../../ICML2026/multimodal_vlm/weathersyn_an_instruction_tuning_mllm_for_weather_forecasting_report_generation.md)
- [\[AAAI 2026\] PET2Rep: Towards Vision-Language Model-Driven Automated Radiology Report Generation for Positron Emission Tomography](../../AAAI2026/multimodal_vlm/pet2rep_towards_vision-language_model-drived_automated_radiology_report_generati.md)
- [\[ICML 2026\] Deep Pre-Alignment for VLMs](../../ICML2026/multimodal_vlm/deep_pre-alignment_for_vlms.md)
- [\[ACL 2026\] UniversalRAG: Retrieval-Augmented Generation for Multimodal Corpora](universalrag_retrieval-augmented_generation_over_corpora_of_diverse_modalities_a.md)
- [\[CVPR 2026\] Recursive Think-Answer Process for LLMs and VLMs](../../CVPR2026/multimodal_vlm/recursive_think-answer_process_for_llms_and_vlms.md)

</div>

<!-- RELATED:END -->
