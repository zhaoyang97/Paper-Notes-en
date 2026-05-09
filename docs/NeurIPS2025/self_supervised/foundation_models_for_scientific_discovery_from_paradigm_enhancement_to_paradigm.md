---
title: >-
  [Paper Note] Foundation Models for Scientific Discovery: From Paradigm Enhancement to Paradigm Transition
description: >-
  [NeurIPS 2025][Self-Supervised Learning][foundation models] This paper proposes a three-stage framework for FM-driven scientific discovery — meta-scientific integration, hybrid human-AI co-creation, and autonomous scientific discovery — and systematically surveys FM applications across the four classical paradigms (experimental, theoretical, computational, and data-driven), arguing that FMs are catalyzing a fifth scientific paradigm.
tags:
  - NeurIPS 2025
  - Self-Supervised Learning
  - foundation models
  - scientific discovery
  - paradigm shift
  - autonomous discovery
  - human-AI co-creation
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

# Foundation Models for Scientific Discovery: From Paradigm Enhancement to Paradigm Transition

**Conference**: NeurIPS 2025
**arXiv**: [2510.15280](https://arxiv.org/abs/2510.15280)
**Authors**: Fan Liu, Jindong Han, Tengfei Lyu, Weijia Zhang, Zhe-Rui Yang, Lu Dai, Cancheng Liu, Hao Liu (HKUST(GZ) & HKUST)
**Code**: [GitHub](https://github.com/usail-hkust/Awesome-Foundation-Models-for-Scientific-Discovery)
**Area**: Self-Supervised Learning / Foundation Models / AI for Science
**Keywords**: foundation models, scientific discovery, paradigm shift, autonomous discovery, human-AI co-creation

## TL;DR

This paper proposes a three-stage framework for FM-driven scientific discovery — meta-scientific integration, hybrid human-AI co-creation, and autonomous scientific discovery — and systematically surveys FM applications across the four classical paradigms (experimental, theoretical, computational, and data-driven), arguing that FMs are catalyzing a fifth scientific paradigm.

## Background & Motivation

Scientific discovery has undergone four paradigm shifts: experiment-driven (16th–17th century, Galileo, Boyle) → theory-driven (18th–19th century, Newton, Maxwell, Einstein) → computation-driven (mid-20th century, numerical simulation) → data-driven (21st century, statistical and deep learning). Each paradigm transition not only introduced new tools but also redefined the epistemological foundations of science.

Contemporary science, however, faces increasingly complex challenges:
- **Emergence and irreducibility**: Problems such as consciousness, protein folding pathways, and social polarization resist reductionist modeling.
- **Combinatorial explosion**: Candidate spaces in drug discovery and materials design render exhaustive search infeasible.
- **Theory–data gap**: The exponential growth of experimental and observational data far outpaces humanity's capacity to synthesize unified theories.
- **Assumption limitations**: Existing computational models rely on simplifying assumptions of linearity, stationarity, and equilibrium, which are inconsistent with the dynamic, nonlinear nature of real-world systems.

Foundation models (FMs) — such as GPT-4, AlphaFold, and DeepSeek — offer a response to these challenges. Trained on large-scale and diverse data, they exhibit general-purpose capabilities that transfer across tasks. AlphaFold resolved the long-standing protein folding problem; FunSearch autonomously proposed and validated new mathematical conjectures. These developments raise a central question: **Are FMs merely enhancing existing scientific methods, or are they catalyzing an entirely new scientific paradigm?**

This paper takes a clear stance: FMs are not only improving individual components of the scientific process but are reshaping the underlying paradigm of scientific discovery itself.

## Core Framework: Three-Stage Evolution

The authors propose a three-stage framework for FM-driven scientific discovery, characterizing a progressive transition from tool to collaborator to autonomous agent:

### Stage 1: Meta-Scientific Integration

| Dimension | Characteristics |
|-----------|----------------|
| FM Role | Back-end tool |
| Task Scope | Task enhancer |
| Autonomy | Low |
| Scientific Impact | Efficiency gains |

FMs operate as intelligent infrastructure, augmenting but not transforming scientific practice. Their core value lies in workflow automation (data preprocessing, literature retrieval, method matching), cross-component integration (connecting sensor data with simulation models, experimental planning with prior knowledge), and improvements in reproducibility and efficiency. At this stage, FMs are **instrumental rather than epistemological** — executing tasks within established paradigms without altering their logical structure. Humans remain the primary agents of reasoning and knowledge production.

### Stage 2: Hybrid Human-AI Co-Creation

| Dimension | Characteristics |
|-----------|----------------|
| FM Role | Co-creator |
| Task Scope | Full-cycle tasks |
| Autonomy | Moderate |
| Scientific Impact | Shift in labor distribution |

FMs transition from passive infrastructure to active collaborators in scientific workflows. They participate in research question generation, hypothesis structuring, and experimental planning, and in some cases execute end-to-end tasks. FMs exhibit moderate autonomy — generating ideas, selecting methods, and adapting workflows based on feedback within bounded research environments — while still relying on humans for problem framing and ethical guidance. This stage redistributes cognitive labor in science: FMs handle literature synthesis, multi-step reasoning, and combinatorial experimental planning, while humans focus on judgment, creativity, and strategic framing.

### Stage 3: Autonomous Scientific Discovery

| Dimension | Characteristics |
|-----------|----------------|
| FM Role | Autonomous agent |
| Task Scope | End-to-end, self-directed |
| Autonomy | High |
| Scientific Impact | Refoundation of science |

FMs move beyond collaboration to evolve into autonomous agents capable of scientific discovery with minimal human intervention. They autonomously formulate research questions, generate hypotheses, select methods, execute experiments or simulations, interpret results, and update their internal models based on feedback. Rather than reactive tools triggered by human input, FMs act as epistemic agents — contributing original insights, challenging existing theories, and shaping the direction of scientific discourse. If fully realized, this would mark a **fifth scientific paradigm** in which discovery is no longer exclusively human-driven but can emerge from the autonomous reasoning of machine intelligence.

## FM Integration across the Four Classical Paradigms

### Experiment-Driven Paradigm

**Experimental Design**: Classical Bayesian optimization and active learning are limited by sparse priors and poor generalizability. FMs encode domain knowledge and guide searches over optimal configurations — for example, acting as priors or feature extractors within BO pipelines for molecular and materials discovery to accelerate convergence, or bypassing surrogate modeling by directly maximizing mutual information to improve data efficiency.

**Physical Experiment Execution**: Laboratory experiments require coordinated planning, perception, and control. FMs increasingly serve as unified interfaces and planners — generating Python control scripts that translate user objectives into executable experimental protocols, orchestrating modular agents for structured reaction planning, embedding into robotic control for language-guided physical manipulation, and integrating multimodal inputs (vision and speech) for real-time interaction and error correction.

### Theory-Driven Paradigm

**Hypothesis Generation**: FMs facilitate systematic hypothesis generation by synthesizing large-scale corpora and structured priors. Knowledge-graph-guided approaches direct hypothesis construction through ontological concept paths to enhance novelty and verifiability; physics-guided FMs embed physical laws directly into the generation process to ensure consistency with known dynamics.

**Theory Verification and Formal Reasoning**: FMs coupled with symbolic logic systems support deductive reasoning, consistency checking, and falsifiability analysis. Logic-LM couples LLMs with symbolic solvers in a feedback loop to improve formal rigor; LeanCopilot and DeepSeekProver demonstrate the ability of pretrained models to assist proof construction and verification at scale.

### Computation-Driven Paradigm

**Building Executable Scientific Models**: FMs support three modeling modalities: symbolic, implicit, and differentiable. For symbolic discovery, LLM-SR converts diverse inputs into equation skeletons for subsequent refinement, while FunSearch frames program synthesis as a language-guided search task to discover new algorithms. When explicit equations are unavailable, PROSE-PDE jointly predicts system dynamics and infers underlying governing laws; DiffusionPDE trains generative priors over coefficient-solution pairs to sample posteriors from sparse data.

**Solving and Inverting Scientific Equations**: Neural operators learn continuous mappings — from forcing terms to PDE solutions — that generalize across grid resolutions. GraphCast surpasses traditional numerical weather prediction models at reduced computational cost. PDE-Refiner iteratively corrects coarse solver outputs, reducing errors without re-running full simulations.

### Data-Driven Paradigm

**Scientific Knowledge Discovery**: FMs compress large corpora into structured representations supporting cross-modal reasoning. DNABERT identifies functional DNA elements from sequences; MoLFormer learns SMILES embeddings enabling zero-shot molecular candidate retrieval; ClimaX fuses diverse climate inputs to learn unified spatiotemporal representations; Galactica transforms millions of papers into a queryable knowledge base.

**Predictive Scientific Reasoning**: FMs reframe prediction tasks as generative modeling. GraphCast and Pangu-Weather learn the latent dynamics of reanalysis data for global weather forecasting; DiffusionSat generates high-resolution satellite imagery to bridge observational gaps; AlphaFold 2 and ESMFold predict protein structures at near-experimental accuracy; RFdiffusion and MatterGen design novel protein folds and inorganic crystals, respectively.

### Cross-Paradigm Integration

FMs serve as integration engines bridging the experimental, theoretical, computational, and data-driven paradigms. PROSE-FD jointly trains symbolic equation templates and spatial field data within a multimodal Transformer, enabling generalization across operating conditions in fluid dynamics. Latent neural operators (LNOs) encode physical operators into geometry-agnostic, resolution-invariant latent spaces. Coscientist translates high-level research objectives into machine-executable protocols, controls robotic synthesis, and adapts subsequent actions based on results — realizing end-to-end scientific workflow orchestration.

## Risks and Challenges

### Bias and Epistemic Equity

FMs inherit biases from training data — overrepresenting mainstream paradigms, Western institutions, and high-citation authors. As FMs transition from tools to co-creators to autonomous agents, these biases shift from passively reflecting existing inequities to actively shaping the scientific agenda. For example, in global health modeling, FMs trained predominantly on English-language literature may systematically prioritize diseases prevalent in Western contexts while neglecting pressing health challenges in developing regions.

### Hallucination and Scientific Misinformation

FMs are fundamentally data-driven pattern recognizers rather than faithful reasoners. As their role shifts from task enhancement to autonomous hypothesis generation, the risk of producing plausible-sounding but unverified claims grows substantially. In biomedicine, FMs may propose novel mechanisms lacking experimental grounding; in physics, they may generate mathematically elegant but physically invalid formulas.

### Reproducibility and Transparency

When FMs take end-to-end control of experimental design, simulation execution, and result interpretation, their decision processes are often opaque. Without access to intermediate reasoning steps, model assumptions, or versioning information, it becomes difficult to replicate or verify findings. Addressing this requires transparent logging of reasoning steps, versioned model checkpoints, and open science practices that preserve the traceability of FM-driven workflows.

### Authorship, Accountability, and Scientific Ethics

When FMs generate core hypotheses or experimental designs, should they be listed as co-authors? Who bears responsibility when their outputs lead to harm or scientific error? Governance frameworks are needed to distinguish mechanical from creative contributions, alongside mandatory transparency disclosure mechanisms and systems for tracking the downstream impact of AI-generated outputs.

## Future Directions

### Embodied Scientific Agents

Anchoring FMs in the physical world — deployed in laboratory robots, automated instruments, and digital twin environments. By coupling language-based reasoning with real-world perception and control, these agents will plan experiments, interact with physical systems, and iteratively optimize processes. Key challenges include integrating high-level task planning with low-level control, robustness under real-world uncertainty, and safety and interpretability in dynamic experimental environments.

### Closed-Loop Scientific Autonomy

Moving from open-loop systems (where FMs assist with discrete steps while humans decide next actions) to closed-loop systems (where FMs continuously formulate hypotheses, design and execute experiments, analyze results, and update internal models based on feedback). Progress has been made through reinforcement-learning-based planning (CycleResearcher), planning-as-reasoning, and neurosymbolic agents. Key challenges include ensuring robustness to noisy observations, adaptability to shifting objectives, and alignment with scientific validity rather than mere reward maximization.

### Continual Learning and Generalization

FMs must evolve from static systems to continual learners capable of accumulating and refining knowledge over time. Core challenges include catastrophic forgetting and domain drift. Promising directions include parameter-efficient online adaptation, memory-augmented architectures, and modular lifelong learning frameworks. Advancing these mechanisms will enable FMs to incrementally build cross-domain representations and facilitate analogical reasoning across scientific contexts.

## Assessment

### Strengths

1. **Clear and compelling framework**: The three-stage evolutionary framework (meta-scientific integration → human-AI co-creation → autonomous discovery) provides a systematic perspective for understanding the relationship between FMs and scientific discovery, combining descriptive and predictive power.
2. **Comprehensive coverage**: The paper systematically surveys FM integration across the four classical scientific paradigms (experimental, theoretical, computational, and data-driven) and discusses cross-paradigm integration.
3. **Mature risk awareness**: Rather than uncritically celebrating FM capabilities, the paper seriously addresses four categories of risk — bias, hallucination, reproducibility, and scientific ethics — reflecting a responsible stance.
4. **Clear positioning**: The paper explicitly argues that FMs are catalyzing a fifth scientific paradigm rather than serving merely as efficiency tools, providing the community with a concrete thesis open to debate.

### Weaknesses

1. **Lack of quantitative evidence**: As a position paper, the three-stage delineation rests primarily on qualitative argumentation, without operationalizable metrics or empirical case studies to measure the degree of paradigm shift.
2. **Blurry stage boundaries**: The boundaries between meta-scientific integration and human-AI co-creation, and between human-AI co-creation and autonomous discovery, are insufficiently defined; real-world systems may span multiple stages simultaneously.
3. **Insufficient critical engagement**: The case for a "fifth paradigm" lacks deep, direct engagement with opposing views (e.g., critiques by Wolfram).
4. **Limited technical depth**: Discussion of specific FM methods is largely survey-level, lacking in-depth technical analysis or a unified comparative framework.
5. **Insufficient justification for the feasibility of autonomous scientific discovery**: Current systems (e.g., AI Scientist) remain significantly short of genuine autonomous discovery, and the paper does not adequately examine this gap.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Implicit Modeling for Transferability Estimation of Vision Foundation Models](implicit_modeling_for_transferability_estimation_of_vision_foundation_models.md)
- [\[NeurIPS 2025\] Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models](mitra_mixed_synthetic_priors_for_enhancing_tabular_foundation_models.md)
- [\[NeurIPS 2025\] SEAL: Semantic-Aware Hierarchical Learning for Generalized Category Discovery](seal_semantic-aware_hierarchical_learning_for_generalized_category_discovery.md)
- [\[AAAI 2026\] Robust Tabular Foundation Models](../../AAAI2026/self_supervised/robust_tabular_foundation_models.md)
- [\[NeurIPS 2025\] Uncertainty-Guided Model Selection for Tabular Foundation Models in Biomolecule Efficacy Prediction](uncertainty-guided_model_selection_for_tabular_foundation_models_in_biomolecule_.md)

</div>

<!-- RELATED:END -->
