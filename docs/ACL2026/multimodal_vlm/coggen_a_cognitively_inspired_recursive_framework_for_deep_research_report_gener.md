---
title: >-
  [Paper Note] CogGen: A Cognitively Inspired Recursive Framework for Deep Research Report Generation
description: >-
  [ACL 2026][Multimodal VLM][Deep research report] CogGen proposes a multi-agent recursive framework that simulates the human cognitive writing process. It achieves global restructuring via a macro-cognitive loop…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Deep research report"
  - "recursive writing framework"
  - "multimodal fusion"
  - "cognitive load assessment"
  - "multi-agent"
date: 2026-05-08
content_hash: 5313d205119a6ac3
---

# CogGen: A Cognitively Inspired Recursive Framework for Deep Research Report Generation

**Conference**: ACL 2026
**arXiv**: [2604.17072](https://arxiv.org/abs/2604.17072)  
**Code**: [GitHub](https://github.com/NJUNLP/CogGen)  
**Area**: Multimodal VLM
**Keywords**: Deep research report, recursive writing framework, multimodal fusion, cognitive load assessment, multi-agent

## TL;DR
CogGen proposes a multi-agent recursive framework that simulates the human cognitive writing process. It achieves global restructuring via a macro-cognitive loop, parallel section refinement via a micro-cognitive cycle, and semantic-level text–chart co-planning via Abstract Visual Representation (AVR). On the OWID benchmark, CogGen reaches human expert-level performance and surpasses Gemini Deep Research.

## Background & Motivation

**Background**: Automated deep research report generation is a frontier application of LLMs. Existing approaches include single-agent systems (e.g., Gemini Deep Research) and multi-agent frameworks (e.g., STORM, Co-STORM), both of which follow linear, predefined workflows.

**Limitations of Prior Work**: Linear workflows preclude retroactive revision once content has been generated — when downstream findings contradict upstream organizational logic, "backward restructuring" is impossible. Furthermore, text and chart generation are typically asynchronous and decoupled, reducing figures to mere illustrations rather than integral components of the argument.

**Key Challenge**: Expert writing is a non-linear, recursive process (plan → write → review → restructure → rewrite), whereas existing AI writing frameworks are linear and forward-only, unable to achieve cross-section global consistency or deep text–chart integration.

**Goal**: To construct a recursive report generation framework that supports global restructuring and multimodal semantic-level co-planning.

**Key Insight**: The framework is grounded in Flower & Hayes' cognitive process theory of writing and the theory of Cognitive Offloading.

**Core Idea**: A hierarchical recursive architecture (macro-loop for global restructuring + micro-loop for section refinement) combined with Abstract Visual Representation that decouples chart generation from the reasoning process.

## Method

### Overall Architecture
CogGen consists of three peer cognitive agents: a Planner (retrieval and structural planning), a Writer (text composition and visual intent definition), and a Reviewer (real-time monitoring and post-hoc evaluation). The macro-cognitive loop operates recursively at the global report level: plan → write → review → feedback → re-plan. The micro-cognitive cycle executes a Search–Replan–Write loop in parallel at the section level.

### Key Designs

1. **Macro-Cognitive Loop**:

    - **Function**: Enables global backward restructuring, resolving the "forward locking" problem of linear workflows.
    - **Mechanism**: The outline $\mathcal{O}$ is treated as a mutable object rather than a fixed plan. In each iteration, the Planner generates an outline → the Writer produces section drafts in parallel → the Reviewer evaluates the complete draft and generates feedback $\Delta^{(t)}$ → the Planner revises the outline as $\mathcal{O}^{(t+1)} = A_p(Q, \{\mathcal{O}^{(t)}, \Delta^{(t)}\}|K)$. A strict monotonic improvement constraint is enforced: updates are accepted only when the Reviewer verifies a clear quality gain, preventing infinite oscillation.
    - **Design Motivation**: Human writing is recursive — completing the latter half of a document often prompts revision of the organizational logic in the earlier sections. This capacity is essential for generating high-quality long-form documents.

2. **Micro-Cognitive Cycle**:

    - **Function**: Generates section content in parallel while maintaining cross-section consistency.
    - **Mechanism**: A Search–Replan–Write loop is executed across multiple threads in parallel. Each thread treats the global outline $\mathcal{O}^{(t)}$ as a read-only constraint and maintains section-specific retrieval results in a thread-local cache. Cross-section conflicts are not resolved locally; instead, they are deferred to the Reviewer for unified arbitration within the macro-loop (Deferred Update Policy), avoiding the context oscillation problem associated with serial revision.
    - **Design Motivation**: Parallel generation improves efficiency, but it must avoid the recursive revision trap — modifying Sec. 1 to accommodate findings from Sec. 5, which in turn necessitates updating Sec. 5.

3. **Abstract Visual Representation (AVR)**:

    - **Function**: Enables semantic-level co-planning of text and charts, rather than appending figures after the fact.
    - **Mechanism**: The Writer produces structured semantic descriptions (Title, Chart\_Type, X/Y\_Axis, Data\_Source, Purpose) rather than executable code. A Renderer Agent translates these semantic intents into ECharts/Mermaid code and renders them in a headless browser. This allows the Writer to iteratively refine visual plans as if manipulating "semantic tokens," without dealing with pixel-level details.
    - **Design Motivation**: Grounded in Cognitive Offloading theory — separating visual design decisions from writing reasoning reduces the Writer's cognitive load and allows it to focus on narrative logic.

### Loss & Training
CogGen is a purely inference-time framework and involves no training. GPT-4.1 serves as the backbone for all agents, GPT-4.1-Mini handles search expansion, and temperature is set to 0.5.

## Key Experimental Results

### Main Results

| Dataset | Method | Avg. | Organization | Depth | Alignment | Coherence |
|--------|------|--------|------|------|------|------|
| OWID | Human Expert (reference) | 0.4997 | 0.4986 | 0.5000 | 0.5000 | 0.5000 |
| OWID | CogGen | 0.4992 | 0.4972 | 0.5813 | 0.4806 | 0.4326 |
| OWID | WriteHere | 0.4502 | 0.4912 | 0.5503 | 0.3846 | 0.3312 |
| OWID | STORM | 0.3205 | 0.4253 | 0.4443 | 0.1675 | 0.1667 |
| WildSeek | Gemini DR (reference) | 0.5000 | 0.5000 | 0.5000 | 0.5000 | 0.5000 |
| WildSeek | CogGen | 0.5341 | 0.5389 | 0.5000 | 0.5544 | 0.5437 |

### Ablation Study

| Configuration | Avg. | Notes |
|------|--------|------|
| GPT-4.1 + Search (no framework) | 0.4119 | Single-agent baseline |
| CogGen w/o review | 0.4681 | Removing Reviewer leads to significant quality drop |
| CogGen two-stage (no native multimodal) | 0.4904 | Text and charts generated separately |
| CogGen (full) | 0.4994 | All components integrated |

### Key Findings
- CogGen reaches human expert-level performance on OWID (0.4992 vs. 0.4997) and surpasses Gemini Deep Research on WildSeek (0.5341 vs. 0.5000).
- Multimodal alignment and coherence (D4, D5) represent CogGen's core advantage over STORM/Co-STORM, with score gaps exceeding 0.3.
- Removing the Reviewer produces the largest performance drop, confirming that the review–feedback loop is central to quality assurance.
- AVR demonstrates a significant improvement in data accuracy compared to FDV (direct code generation).

## Highlights & Insights
- The **macro–micro dual-loop recursive** design precisely simulates the non-linear nature of human writing — the ability to restructure the outline after completing the full draft is the key differentiator from linear systems.
- The **Deferred Update Policy** elegantly resolves context oscillation in parallel generation by delegating conflict resolution to a global reviewer rather than applying local fixes, thereby avoiding the recursive revision trap.
- AVR decouples "what to show" from "how to render it," a principle generalizable to any scenario requiring text–code co-generation.

## Limitations & Future Work
- The framework depends on closed-source models such as GPT-4.1, resulting in high cost and limited reproducibility.
- The convergence speed of recursive loops is not sufficiently analyzed, and actual generation time may be substantial.
- The CLEF evaluation framework, while theoretically grounded, relies on GPT-5 as the evaluator, introducing potential evaluation bias.
- Only static text and charts are supported; interactive visualizations are not accommodated.

## Related Work & Insights
- **vs. STORM/Co-STORM**: These frameworks support multi-perspective QA and collaborative writing but lack global restructuring capability.
- **vs. WriteHere**: Supports recursive decomposition but remains forward-only in generation, unable to revise already-generated content in reverse.
- **vs. Gemini Deep Research**: This commercial system is still constrained by a fixed framework at the writing execution stage; CogGen surpasses its output quality on the WildSeek benchmark.

## Rating
- **Novelty**: ⭐⭐⭐⭐ The systematic application of cognitive writing theory to AI report generation is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Two datasets, multiple baselines, detailed ablations, and human evaluation validation.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical motivation is clear, figures are excellent, and the narrative is fluid.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] PET2Rep: Towards Vision-Language Model-Driven Automated Radiology Report Generation for Positron Emission Tomography](../../AAAI2026/multimodal_vlm/pet2rep_towards_vision-language_model-drived_automated_radiology_report_generati.md)
- [\[CVPR 2026\] Recursive Think-Answer Process for LLMs and VLMs](../../CVPR2026/multimodal_vlm/recursive_think-answer_process_for_llms_and_vlms.md)
- [\[ACL 2026\] GAMBIT: A Gamified Jailbreak Framework for Multimodal Large Language Models](gambit_a_gamified_jailbreak_framework_for_multimodal_large_language_models.md)
- [\[ICCV 2025\] Evading Data Provenance in Deep Neural Networks](../../ICCV2025/multimodal_vlm/evading_data_provenance_in_deep_neural_networks.md)
- [\[ACL 2026\] FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models](finesteer_a_unified_framework_for_fine-grained_inference-time_steering_in_large_.md)

</div>

<!-- RELATED:END -->
