---
title: >-
  [Paper Note] Foundation Models for Scientific Discovery: From Paradigm Enhancement to Paradigm Transition
description: >-
  [NeurIPS 2025][Self-Supervised Learning][foundation models] This paper proposes a three-stage framework (meta-scientific integration → hybrid human-AI co-creation → autonomous scientific discovery) to characterize how fo…
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "foundation models"
  - "scientific paradigms"
  - "autonomous scientific discovery"
  - "human-AI collaboration"
  - "AI for Science"
date: 2026-05-08
content_hash: 632217b4b17277ef
---

# Foundation Models for Scientific Discovery: From Paradigm Enhancement to Paradigm Transition

**Conference**: NeurIPS 2025
**arXiv**: [2510.15280](https://arxiv.org/abs/2510.15280)  
**Code**: [GitHub](https://github.com/usail-hkust/Awesome-Foundation-Models-for-Scientific-Discovery)  
**Area**: Scientific Discovery / Foundation Model Survey
**Keywords**: foundation models, scientific paradigms, autonomous scientific discovery, human-AI collaboration, AI for Science

## TL;DR

This paper proposes a three-stage framework (meta-scientific integration → hybrid human-AI co-creation → autonomous scientific discovery) to characterize how foundation models are driving a transition in scientific paradigms from tool-based enhancement toward paradigm-level transformation. It also provides a systematic survey of FM integration across the four classical scientific paradigms: experimental, theoretical, computational, and data-driven.

## Background & Motivation

- **Background**: Scientific discovery has historically undergone four major paradigm shifts — experiment-driven (16th–17th century), theory-driven (18th–19th century), computation-driven (mid-20th century), and data-driven (21st century). Foundation models (FMs) such as GPT-4, AlphaFold, and DeepSeek are now reshaping the scientific landscape.

- **Limitations of Prior Work**: Scientific problems increasingly exhibit emergent behavior, open-endedness, and irreducible complexity that challenge all four prior paradigms. The experimental paradigm is limited by the difficulty of directly manipulating large-scale or complex systems; the theoretical paradigm faces a growing gap between theoretical complexity and experimental testability; the computational paradigm relies on simplifying assumptions; and the data-driven paradigm struggles with causal inference and interpretability.

- **Key Challenge**: A fundamental tension exists as to whether FMs are merely powerful tools that accelerate existing scientific methodology or whether they are catalyzing an entirely new scientific paradigm. Proponents argue that FMs restructure the logic of discovery and lower barriers to entry; skeptics contend that FMs remain conventional tools at their core.

- **Goal**: To clarify the positioning of FMs in scientific discovery and to propose a systematic framework for understanding how FMs evolve from tools to cognitive agents.

- **Key Insight**: The authors adopt the lens of scientific paradigm evolution to construct a progressive three-stage framework.

- **Core Idea**: FMs are catalyzing a fifth-stage transition in scientific paradigms — from human-led discovery toward discovery in which machine intelligence participates or takes the lead.

## Method

### Overall Architecture

The paper proposes a three-stage framework for FM-driven scientific evolution, characterized along five dimensions (paradigm definition, FM role, task scope, autonomy, and scientific impact):

| Dimension | Meta-Scientific Integration | Hybrid Human-AI Co-Creation | Autonomous Scientific Discovery |
|-----------|----------------------------|-----------------------------|---------------------------------|
| FM Role | Back-end tool | Co-creator | Autonomous agent |
| Task Scope | Task enhancer | Full-cycle tasks | End-to-end, self-directed |
| Autonomy | Low | Moderate | High |
| Impact | Efficiency gains | Labor redistribution | Refoundation of science |

### Key Designs

1. **Stage 1 — Meta-Scientific Integration**:

    - FMs serve as intelligent infrastructure, augmenting but not transforming scientific practice.
    - Role: back-end coordinator, automating data preprocessing, literature retrieval, and method matching.
    - Bridges previously siloed components (sensor data ↔ simulation models, experimental planning ↔ prior knowledge).
    - Key characteristics: low autonomy, requiring continuous human oversight, instrumental rather than epistemological.
    - Analogy: improves scientific throughput, but the core of reasoning and knowledge production remains human-driven.

2. **Stage 2 — Hybrid Human-AI Co-Creation**:

    - FMs transition from passive infrastructure to active collaborators in the research workflow.
    - Participate in research question generation, hypothesis structuring, and experimental planning.
    - Exhibit moderate autonomy: capable of generating ideas, selecting methods, and adjusting workflows based on feedback.
    - Reconfigures the division of cognitive labor: FMs handle literature synthesis, multi-step reasoning, and combinatorial experimental planning, while humans focus on judgment, creativity, and strategic framing.

3. **Stage 3 — Autonomous Scientific Discovery**:

    - FMs operate as autonomous agents, executing complete scientific cycles with minimal human intervention.
    - Autonomously formulate research questions, generate hypotheses, select methods, execute experiments, and interpret results.
    - Key distinction: driven by internal objectives and feedback mechanisms rather than passively responding to human input.
    - Represents the fifth scientific paradigm: discovery no longer exclusively human-driven, but emerging from autonomous machine reasoning.
    - Case in point: AI Scientist has demonstrated an end-to-end research pipeline.

4. **FM Integration across the Four Classical Paradigms**:

    - **Experimental Paradigm**: FMs serve as Bayesian optimization priors to accelerate molecular and materials discovery; FMs generate instrument control scripts; multimodal agents are embedded in robotic control.
    - **Theoretical Paradigm**: Knowledge-graph-guided hypothesis generation (KG-CoI); coupling with symbolic logic systems for hypothesis verification (Logic-LM); assistance with theorem proving (LeanCopilot, DeepSeekProver).
    - **Computational Paradigm**: Symbolic discovery (LLM-SR, FunSearch); latent operator learning (PROSE-PDE); neural operators for accelerating PDE solving (GraphCast for weather forecasting).
    - **Data-Driven Paradigm**: Scientific knowledge discovery (DNABERT, MoLFormer, ClimaX); predictive inference (AlphaFold, ESMFold, RFdiffusion).

### Loss & Training

This paper is a survey and position paper and does not involve specific training strategies. The primary contribution lies in the conceptual framework and systematic taxonomy.

## Key Experimental Results

### Main Results

This is a position paper and contains no original experimental data. Arguments are supported through a systematic review of existing work.

### Ablation Study

Not applicable.

### Key Findings

The paper systematically identifies four major risk dimensions:

| Risk Dimension | Description |
|----------------|-------------|
| Bias & Epistemic Equity | FMs inherit biases from training data, potentially leading to epistemic homogenization and the marginalization of underrepresented knowledge domains. |
| Hallucination & Scientific Misinformation | FMs may generate plausible-sounding but unverified scientific claims; the hazard scales with increasing autonomy. |
| Reproducibility & Transparency | End-to-end FM decision processes are opaque, threatening scientific reproducibility. |
| Authorship & Accountability | The shift of FMs from tools to collaborators raises questions of intellectual property and ethical attribution. |

Three major future directions are identified: embodied scientific agents, closed-loop scientific autonomy, and continual learning and generalization.

## Highlights & Insights

- **Elegant framework design**: The three-stage progressive framework (tool → collaborator → autonomous agent) is clear and persuasive.
- **Comprehensive coverage**: The analysis spans a matrix of four classical scientific paradigms × three stages of FM integration.
- **Pragmatic risk analysis**: Beyond discussing technical capabilities, the paper seriously engages with core scientific values such as bias, hallucination, and reproducibility.
- **Cross-paradigm integration**: The paper discusses how FMs break down traditional paradigm boundaries and enable cross-paradigm workflow coordination.
- **Concrete case analyses**: Examples such as PROSE-FD and Coscientist illustrate real progress in cross-paradigm integration.

## Limitations & Future Work

- As a position paper, the arguments are partly speculative and lack rigorous empirical validation.
- The claim of a "fifth paradigm" may be overly optimistic; current FMs remain far from genuine autonomous scientific discovery.
- Coverage of mathematics and formal sciences is less thorough than that of experimental sciences.
- Specific failure cases of FMs (e.g., AlphaFold's limitations on particular proteins) are not discussed in depth.
- Evaluation of large closed-source industrial models (e.g., GPT-4, Gemini) is constrained by their lack of reproducibility.

## Related Work & Insights

- **AlphaFold** (Jumper et al., 2021): A landmark case resolving the protein folding challenge.
- **FunSearch** (Romera-Paredes et al., 2024): Autonomously discovers mathematical conjectures and algorithms.
- **Coscientist** (Boiko et al., 2023): A chemical experimentation system that translates research objectives into machine-executable protocols.
- **AI Scientist**: Demonstrates automation of a complete research pipeline.
- Provides the AI4Science community with a valuable conceptual map and reference for future directions.

## Rating

- Novelty: ⭐⭐⭐⭐ — The three-stage framework is original and integrates a fragmented discourse into a coherent narrative.
- Experimental Thoroughness: ⭐⭐⭐ — No original experiments as a survey paper, but literature coverage is broad.
- Writing Quality: ⭐⭐⭐⭐⭐ — Well-structured, logically rigorous argumentation with polished figures and tables.
- Value: ⭐⭐⭐⭐ — Provides an important intellectual framework for the scientific community to understand the transformative role of FMs.

---

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Implicit Modeling for Transferability Estimation of Vision Foundation Models](implicit_modeling_for_transferability_estimation_of_vision_foundation_models.md)
- [\[NeurIPS 2025\] Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models](mitra_mixed_synthetic_priors_for_enhancing_tabular_foundation_models.md)
- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[AAAI 2026\] Robust Tabular Foundation Models](../../AAAI2026/self_supervised/robust_tabular_foundation_models.md)
- [\[ICCV 2025\] LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models](../../ICCV2025/self_supervised/loftup_learning_a_coordinatebased_feature_upsampler_for_visi.md)

</div>

<!-- RELATED:END -->
