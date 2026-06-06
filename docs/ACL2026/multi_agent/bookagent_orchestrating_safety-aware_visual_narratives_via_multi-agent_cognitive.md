---
title: >-
  [Paper Note] BookAgent: Orchestrating Safety-Aware Visual Narratives via Multi-Agent Cognitive Calibration
description: >-
  [ACL 2026][Multi-Agent][Picture book generation] BookAgent is a safety-aware multi-agent framework that generates high-quality, character-consistent…
tags:
  - "ACL 2026"
  - "Multi-Agent"
  - "Picture book generation"
  - "multi-agent collaboration"
  - "safety alignment"
  - "cross-frame consistency"
  - "visual storytelling"
date: 2026-05-08
content_hash: 8f4b92217fa16e27
---

# BookAgent: Orchestrating Safety-Aware Visual Narratives via Multi-Agent Cognitive Calibration

**Conference**: ACL 2026  
**arXiv**: [2604.16541](https://arxiv.org/abs/2604.16541)  
**Code**: [https://github.com/bogao-code/BookAgent](https://github.com/bogao-code/BookAgent)  
**Area**: Image Generation  
**Keywords**: Picture book generation, multi-agent collaboration, safety alignment, cross-frame consistency, visual storytelling

## TL;DR
BookAgent is a safety-aware multi-agent framework that generates high-quality, character-consistent, and content-safe picture book stories end-to-end from user drafts through a three-stage closed-loop architecture: **Value-Aligned Storyboard (VAS) + Iterative Cross-modal Refinement (ICR) + Temporal Cognitive Calibration (TCC)**.

## Background & Motivation

**Background**: Large generative models have made significant progress in text and image generation, but automatic picture book generation remains an open challenge. Existing methods decouple story visualization into independent stages (fixing the storyline first, then generating images page-by-page), lacking holistic multi-modal alignment.

**Limitations of Prior Work**: (1) Weak cross-modal alignment—visual content rarely provides structured feedback to revise scripts, leading to insufficient bidirectional alignment; (2) Poor global consistency—character appearance drift, missing props, and broken causal relationships in long sequences; (3) Child safety is not integrated—current safety methods are mostly post-hoc filters, not embedded in narrative planning and global consistency checks.

**Key Challenge**: A unified system is needed to simultaneously address cross-modal alignment, long-range consistency, and domain safety, whereas existing methods can only handle one at a time.

**Goal**: To construct an end-to-end picture book synthesis system that generates both scripts and illustrations from user drafts, ensuring page-level alignment, global character consistency, and compliance with child safety standards.

**Key Insight**: View picture book generation as a **collaborative cognitive process** rather than a pipeline—multiple specialized agents (Reviewers, Directors, Safety Auditors, etc.) collaborate through closed-loop feedback.

**Core Idea**: A three-stage hierarchical workflow—VAS ensures a safe narrative blueprint, ICR ensures single-page quality, and TCC ensures cross-page global consistency.

## Method

### Overall Architecture
Picture book synthesis is formalized as a constrained optimization problem: maximize text-image faithfulness $\alpha$, character identity consistency $\eta$, and global sequence coherence $\beta$, subject to the constraint that all text and images pass safety audits $\mathcal{S}_T=1, \mathcal{S}_I=1$. The system comprises 10 specialized agents (see Table 1) collaborating via the VAS→ICR→TCC workflow.

### Key Designs

1.  **Value-Aligned Storyboard (VAS)**:
    - **Function**: Ensures narrative safety and establishes visual anchors before visualization.
    - **Mechanism**: Reviewer-Refiner rewrites user drafts into a $K$-page structured story $\hat{x}$, verified by the Text Safety Auditor; the Character Extractor identifies $\le 5$ main characters and visual descriptors; the Character Sheet Renderer produces reference images with neutral backgrounds for each character to serve as ground truth for identity verification; the Page Planner breaks the story into page-by-page plans.
    - **Design Motivation**: Safety auditing in the pre-generation stage elevates safety from "passive post-hoc filtering" to "active planning constraint," while character reference sheets provide reliable anchors for subsequent inter-frame consistency.

2.  **Iterative Cross-modal Refinement (ICR)**:
    - **Function**: Ensures single-page text-image alignment and character consistency through a "generate-verify-revise" loop.
    - **Mechanism**: A budgeted loop is executed for each page: (1) Retrieval of relevant character reference images $\mathcal{R}_i$ for conditional image generation $y_i^{(r)}$; (2) Frame Director scores text-image faithfulness $\alpha_i^{(r)}$, and Identity Director checks character consistency $\eta_i^{(r)}$; (3) If the safety audit fails, safety negative constraints are added; otherwise, semantic/identity feedback is fused to revise the prompt $p_i^{(r+1)}$. Local memory $\mathcal{M}_i$ accumulates historical constraints to prevent regression.
    - **Design Motivation**: Single-pass diffusion models cannot guarantee compliance with complex constraints (e.g., precise button counts). Iterative verification-revision transforms generation from static sampling into dynamic self-correction.

3.  **Temporal Cognitive Calibration (TCC)**:
    - **Function**: Detects and repairs global inconsistencies across pages.
    - **Mechanism**: The Sequence Director performs a global audit of the complete sequence $\mathcal{B}^{(m)}$, outputting a consistency score $\beta^{(m)}$, global critique $\Gamma^{(m)}$, and indices of problematic pages $\mathcal{I}^{(m)}$. If $\beta^{(m)} < \tau_\beta$, selective repair is performed only on problematic pages (re-entering ICR with global context constraints) until convergence.
    - **Design Motivation**: Relying solely on local historical conditioning (e.g., previous page as context) cannot prevent long-range appearance drift; global auditing + selective repair shifts the paradigm from linear autoregressive accumulation to holistic temporal reasoning.

### Loss & Training
No training; multi-agent collaboration during inference only. Uses Google Gemini 3.0 for reasoning and Nano-Banana for generation. All methods are compared under the same prompt protocol and generation settings.

## Key Experimental Results

### Main Results

| Method | Text-Image Alignment (1-5) | Cross-frame Character Consistency (1-5) | Safety (1-5) |
| :--- | :--- | :--- | :--- |
| **BookAgent** | **4.6** | **4.7** | **4.8** |
| StoryGPT-V | 3.1 | 2.4 | 4.5 |
| MovieAgent | 2.8 | 2.1 | 3.6 |
| StoryGen | 2.5 | 1.9 | 4.4 |

### Ablation Study

| Config | Text-Image Alignment | Cross-frame Consistency | Safety | Note |
| :--- | :--- | :--- | :--- | :--- |
| Baseline (No VAS/ICR/TCC) | 2.7 | 2.0 | 4.2 | |
| + VAS | 2.8 | 2.1 | **4.8** | Safety significantly improved |
| + VAS + ICR | 4.6 | 3.0 | 4.8 | Text-image alignment significantly improved |
| + VAS + ICR + TCC | **4.6** | **4.7** | **4.8** | Cross-frame consistency significantly improved |

### Key Findings
- ICR is critical for text-image alignment (2.8 → 4.6), proving that single-pass generation cannot satisfy complex constraints.
- TCC is essential for cross-frame consistency (3.0 → 4.7), proving that local conditioning is insufficient for maintaining long-range coherence.
- VAS improves safety from 4.2 to 4.8, showing that safety auditing in the pre-planning stage is more effective than post-filtering.
- In a parent user study, BookAgent received the highest preference ratings; improved long-range consistency made stories easier for children to comprehend.

## Highlights & Insights
- **The design paradigm of "establish anchors first, then refine iteratively"** is highly instructive: character reference images serve as consistency anchors, and all subsequent generation and verification are benchmarked against them, avoiding the root cause of autoregressive drift.
- **Selective repair** (fixing only problematic pages instead of regenerating the whole sequence) provides a good balance between efficiency and quality.
- The **hierarchical safety design**, with safety audits deeply embedded in various stages (VAS text audit, ICR image audit, TCC global audit), serves as a paradigm for safety-aware systems.

## Limitations & Future Work
- Dependence on proprietary models like Gemini 3.0 and Nano-Banana limits open-source reproducibility.
- Iterative refinement and global calibration introduce significant inference costs (potentially multiple generation-verification loops per page).
- Evaluation is primarily based on LLM-as-a-judge automatic scoring, with a small-scale human evaluation.
- Longest tests involve 20 pages; consistency maintenance for longer picture books (e.g., 50+ pages) remains unverified.

## Related Work & Insights
- **vs MovieAgent (Wu et al., 2025)**: Shares a hierarchical multi-agent paradigm, but BookAgent adds safety auditing and global calibration for cross-frame consistency, significantly outperforming it across all metrics.
- **vs StoryGPT-V**: The latter uses LLMs to align character descriptions with diffusion models but remains a unidirectional generation pipeline; BookAgent achieves bidirectional alignment through closed-loop feedback.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of end-to-end picture book synthesis, hierarchical safety, and temporal calibration is a novel system configuration.
- Experimental Thoroughness: ⭐⭐⭐ Evaluation relies heavily on qualitative analysis and LLM judges, lacking large-scale automated metrics.
- Writing Quality: ⭐⭐⭐⭐ The system design is clear and formalization is rigorous, though excessive formulas slightly impact readability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] MASFactory: A Graph-centric Framework for Orchestrating LLM-Based Multi-Agent Systems with Vibe Graphing](masfactory_a_graph-centric_framework_for_orchestrating_llm-based_multi-agent_sys.md)
- [\[ACL 2026\] AgenticEval: Toward Agentic and Self-Evolving Safety Evaluation of Large Language Models](agenticeval_toward_agentic_and_self-evolving_safety_evaluation_of_large_language.md)
- [\[ICML 2026\] RADAR: Redundancy-Aware Diffusion for Multi-Agent Communication Structure Generation](../../ICML2026/multi_agent/radar_redundancy-aware_diffusion_for_multi-agent_communication_structure_generat.md)
- [\[ACL 2026\] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs](topology_matters_measuring_memory_leakage_in_multi-agent_llms.md)
- [\[ACL 2026\] when identity skews debate anonymization for bias-reduced multi-agent reasoning](when_identity_skews_debate_anonymization_for_bias-reduced_multi-agent_reasoning.md)

</div>

<!-- RELATED:END -->
