---
title: >-
  [Paper Note] BookAgent: Orchestrating Safety-Aware Visual Narratives via Multi-Agent Cognitive Calibration
description: >-
  [ACL 2026][Image Generation][Picture book generation] BookAgent is a safety-aware multi-agent framework that generates high-quality, character-consistent…
tags:
  - "ACL 2026"
  - "Image Generation"
  - "Picture book generation"
  - "multi-agent collaboration"
  - "safety alignment"
  - "cross-frame consistency"
  - "visual narratives"
date: 2026-05-08
content_hash: 32aa36fda9d79647
---

# BookAgent: Orchestrating Safety-Aware Visual Narratives via Multi-Agent Cognitive Calibration

**Conference**: ACL 2026  
**arXiv**: [2604.16541](https://arxiv.org/abs/2604.16541)  
**Code**: [https://github.com/bogao-code/BookAgent](https://github.com/bogao-code/BookAgent)  
**Area**: Image Generation  
**Keywords**: Picture book generation, multi-agent collaboration, safety alignment, cross-frame consistency, visual narratives

## TL;DR
BookAgent is a safety-aware multi-agent framework that generates high-quality, character-consistent, and content-safe picture books end-to-end from user drafts through a three-stage closed-loop architecture: **Value-Aligned Storyboard (VAS) + Iterative Cross-Modal Refinement (ICR) + Temporal Cognitive Calibration (TCC)**.

## Background & Motivation

**Background**: Large generative models have achieved remarkable progress in text and image generation, but automatic picture book generation remains an open challenge. Existing methods decompose story visualization into independent stages (first fixing the storyline, then generating images page by page), lacking holistic multimodal alignment.

**Limitations of Prior Work**: (1) Weak cross-modal alignment—visual content rarely provides structured feedback to correct scripts, resulting in insufficient bidirectional alignment; (2) Poor global consistency—character appearance drift, missing props, and broken causal relationships in long sequence generation; (3) Children's safety not integrated—existing safety methods are mostly post-hoc filtering, not embedded in narrative planning and global consistency checking.

**Key Challenge**: A unified system is needed to simultaneously address cross-modal alignment, long-range consistency, and domain safety, but existing methods can only handle one of these problems separately.

**Goal**: Build an end-to-end picture book synthesis system that generates both scripts and illustrations from user drafts, ensuring page-level alignment, global character consistency, and child safety compliance.

**Key Insight**: Treat picture book generation as a **collaborative cognitive process** rather than a pipeline—multiple specialized agents (reviewers, directors, safety auditors, etc.) collaborate through closed-loop feedback.

**Core Idea**: Three-stage hierarchical workflow—VAS ensures a safe narrative blueprint, ICR ensures single-page quality, and TCC ensures cross-page global consistency.

## Method

### Overall Architecture
Picture book synthesis is formalized as a constrained optimization problem: maximize text-image fidelity $\alpha$, character identity consistency $\eta$, and global sequence coherence $\beta$, subject to all text and images passing safety audits $\mathcal{S}_T=1, \mathcal{S}_I=1$. The system contains 10 specialized agents (see Table 1) collaborating through a three-stage workflow: VAS→ICR→TCC.

### Key Designs

1. **Value-Aligned Storyboard (VAS)**:

    - Function: Ensure narrative safety and establish visual anchors before visualization
    - Mechanism: Reviewer-Refiner rewrites user drafts into K-page structured stories $\hat{x}$, validated by text safety auditors; Character Extractor extracts ≤5 main characters with visual descriptors; Character Sheet Renderer generates reference images with neutral backgrounds for each character as ground truth for subsequent identity verification; Page Planner decomposes the story into page-by-page plans
    - Design Motivation: Pre-generation safety audits elevate safety from "passive post-filtering" to "proactive planning constraints"; character reference images provide reliable anchors for subsequent inter-frame consistency

2. **Iterative Cross-Modal Refinement (ICR)**:

    - Function: Ensure single-page text-image alignment and character consistency through "generate-validate-revise" closed-loop
    - Mechanism: Execute budgeted loops per page: (1) Retrieve relevant character references $\mathcal{R}_i$, conditionally generate image $y_i^{(r)}$; (2) Frame Director scores text-image fidelity $\alpha_i^{(r)}$, Identity Director checks character consistency $\eta_i^{(r)}$; (3) If safety audit fails, add safety negative constraints; otherwise fuse semantic/identity feedback to revise prompt $p_i^{(r+1)}$. Local memory $\mathcal{M}_i$ accumulates historical constraints to prevent regression
    - Design Motivation: Single-shot diffusion models cannot guarantee complex constraints (e.g., exact button count); iterative validation-revision transforms generation from static sampling to dynamic self-correction

3. **Temporal Cognitive Calibration (TCC)**:

    - Function: Detect and repair cross-page global inconsistencies
    - Mechanism: Sequence Director performs global audit on complete sequence $\mathcal{B}^{(m)}$, outputs consistency score $\beta^{(m)}$, global critique $\Gamma^{(m)}$, and problematic page indices $\mathcal{I}^{(m)}$. If $\beta^{(m)} < \tau_\beta$, only perform selective repair on problematic pages (re-entering ICR with global context constraints), iterating until convergence
    - Design Motivation: Local historical conditioning alone (e.g., previous page as context) cannot prevent long-range appearance drift; global audit + selective repair elevates the paradigm from linear autoregressive accumulation to holistic temporal reasoning

### Loss & Training
Training-free, purely inference-time multi-agent collaboration. Uses Google Gemini 3.0 for reasoning and Nano-Banana for generation. All methods compared under identical prompting protocols and generation settings.

## Key Experimental Results

### Main Results

| Method | Text-Image Consistency (1-5) | Cross-Frame Character Consistency (1-5) | Safety (1-5) |
|--------|------|------|------|
| BookAgent | **4.6** | **4.7** | **4.8** |
| StoryGPT-V | 3.1 | 2.4 | 4.5 |
| MovieAgent | 2.8 | 2.1 | 3.6 |
| StoryGen | 2.5 | 1.9 | 4.4 |

### Ablation Study

| Config | Text-Image Consistency | Cross-Frame Consistency | Safety | Note |
|------|---------|------|------|------|
| Baseline (no VAS/ICR/TCC) | 2.7 | 2.0 | 4.2 | |
| + VAS | 2.8 | 2.1 | **4.8** | Significant safety improvement |
| + VAS + ICR | 4.6 | 3.0 | 4.8 | Significant text-image consistency improvement |
| + VAS + ICR + TCC | **4.6** | **4.7** | **4.8** | Significant cross-frame consistency improvement |

### Key Findings
- ICR is critical for text-image consistency (2.8→4.6), proving single-shot generation fundamentally cannot satisfy complex constraints
- TCC is critical for cross-frame consistency (3.0→4.7), proving local conditioning is insufficient to maintain long-range consistency
- VAS improves safety from 4.2 to 4.8; pre-planning safety audits are more effective than post-filtering
- In parent user studies, BookAgent received the highest preference scores; improved long-range consistency makes stories more comprehensible to children

## Highlights & Insights
- The **"establish anchors first, then iteratively refine"** design paradigm is highly valuable: character reference images serve as consistency anchors, with all subsequent generation and validation based on these benchmarks, avoiding the root cause of autoregressive drift
- **Selective repair** (repairing only problematic pages rather than regenerating the entire sequence) is a good trade-off between efficiency and quality
- The layered safety design with safety audits deeply embedded in each stage (VAS text audit, ICR image audit, TCC global audit) can serve as a paradigm for safety-aware systems

## Limitations & Future Work
- Relies on commercial models like Gemini 3.0 and Nano-Banana, limiting open-source reproducibility
- Iterative refinement and global calibration introduce significant inference costs (potentially multiple generation-validation loops per page)
- Evaluation primarily based on automated LLM judge scoring, with limited human evaluation scale
- Longest test is 20 pages; consistency maintenance for longer books (e.g., 50+ pages) remains unverified

## Related Work & Insights
- **vs MovieAgent (Wu et al., 2025)**: Shares hierarchical multi-agent paradigm, but BookAgent adds safety audits and global calibration for cross-frame consistency, significantly outperforming on all metrics
- **vs StoryGPT-V**: The latter uses LLMs to align character descriptions with diffusion models, but remains a unidirectional generation pipeline; BookAgent achieves bidirectional alignment through closed-loop feedback

## Rating
- Novelty: ⭐⭐⭐⭐ End-to-end picture book synthesis + layered safety + temporal calibration is a novel system combination
- Experimental Thoroughness: ⭐⭐⭐ Evaluation mainly qualitative and LLM judge-based, lacking large-scale automated metrics
- Writing Quality: ⭐⭐⭐⭐ Clear system design, rigorous formalization, though excessive formulas impact readability

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Multi-agent Coordination via Flow Matching](../../ICLR2026/image_generation/multi-agent_coordination_via_flow_matching.md)
- [\[CVPR 2026\] coDrawAgents: A Multi-Agent Dialogue Framework for Compositional Image Generation](../../CVPR2026/image_generation/codrawagents_a_multiagent_dialogue_framework_for_c.md)
- [\[CVPR 2026\] When Safety Collides: Resolving Multi-Category Harmful Conflicts in Text-to-Image Diffusion via Adaptive Safety Guidance](../../CVPR2026/image_generation/when_safety_collides_resolving_multi-category_harmful_conflicts_in_text-to-image.md)
- [\[ICLR 2026\] MAC-AMP: A Closed-Loop Multi-Agent Collaboration System for Multi-Objective Antimicrobial Peptide Design](../../ICLR2026/image_generation/mac-amp_a_closed-loop_multi-agent_collaboration_system_for_multi-objective_antim.md)
- [\[AAAI 2026\] Conditional Diffusion Model for Multi-Agent Dynamic Task Decomposition](../../AAAI2026/image_generation/conditional_diffusion_model_for_multi-agent_dynamic_task_dec.md)

</div>

<!-- RELATED:END -->
