---
title: >-
  [Paper Note] MOOSE-Copilot: A Web-Based Interactive Assistant for Unified Exploratory and Fine-Grained Scientific Hypothesis Discovery
description: >-
  [ACL 2026][LLM Agent][Scientific Hypothesis Discovery] MOOSE-Copilot unifies divergent scientific idea exploration and convergent fine-grained hypothesis refinement into a visual Human-AI collaborative system…
tags:
  - "ACL 2026"
  - "LLM Agent"
  - "Scientific Hypothesis Discovery"
  - "Human-AI Collaboration"
  - "Exploration-Exploitation"
  - "Interactive Agents"
  - "MOOSE-Chem"
date: 2026-05-08
content_hash: 18ba99d8daec6cee
---

# MOOSE-Copilot: A Web-Based Interactive Assistant for Unified Exploratory and Fine-Grained Scientific Hypothesis Discovery

**Conference**: ACL 2026  
**arXiv**: [2605.29475](https://arxiv.org/abs/2605.29475)  
**Code**: https://moosedemo.com (Demo site; cache does not provide GitHub link)  
**Area**: LLM Agent / Scientific Discovery  
**Keywords**: Scientific Hypothesis Discovery, Human-AI Collaboration, Exploration-Exploitation, Interactive Agents, MOOSE-Chem  

## TL;DR
MOOSE-Copilot unifies divergent scientific idea exploration and convergent fine-grained hypothesis refinement into a visual Human-AI collaborative system, significantly enhancing hypothesis discovery through three types of explicit human signals: initial blueprints, stage routing, and feedback.

## Background & Motivation
**Background**: LLMs have been applied to scientific workflows including hypothesis generation, experimental design, paper writing, and peer review assistance. "Scientific hypothesis discovery" sits at the early stage of the research process, directly influencing subsequent experimental directions and potential value. Existing automated systems are broadly categorized into two types: those focusing on divergent exploration, generating diverse high-level ideas from background research; and those focusing on fine-grained optimization, refining a starting concept with methods, experiments, and execution details.

**Limitations of Prior Work**: These two categories of systems are typically treated as independent tasks. Exploratory systems expand directions but often produce coarse and non-specific outputs; fine-grained systems polish ideas into executable plans but rely on a pre-selected starting point. Crucially, many agent workflows run autonomously, leaving domain experts to filter results post-hoc without the ability to provide timely course corrections.

**Key Challenge**: Scientific discovery requires both exploration and exploitation. Searching across the joint space of high-level inspiration and fine-grained experimental details fully autonomously leads to massive combinatorial explosion. However, relying solely on manual control loses the LLM's advantages in large-scale generation and local refinement.

**Goal**: The authors aim to establish a unified framework that connects the exploratory search of MOOSE-Chem with the fine-grained refinement of MOOSE-Chem2, while allowing human experts to inject directional information at critical nodes to decide where to start, when to switch to refinement, and how to regenerate based on feedback.

**Key Insight**: The paper formalizes human-in-the-loop not as a UI feature, but as a Human-AI Interaction Interface (HAII) protocol. Human inputs are modeled as routing operators and constraint signals during the search process, used to prune search spaces, specify granularity transitions, and correct current trajectories.

**Core Idea**: Utilize structured human signals to decompose the joint "divergent search + convergent optimization" space into a controllable Human-AI collaborative search trajectory.

## Method
The core of MOOSE-Copilot is not a new single-stage generator, but rather the integration of existing MOOSE-Chem and MOOSE-Chem2 into a unified state machine equipped with a visual interactive interface. The left side handles exploration, expanding a hypothesis tree from backgrounds and inspiration corpora; the right side handles exploitation, refining a coarse-grained node into a complete research plan. The user acts as a navigator, deciding which nodes warrant further exploration, which should be drilled down for refinement, and which results require regeneration with feedback.

### Overall Architecture
Inputs include a research problem, optional literature surveys, optional inspiration knowledge corpora, and the user's initial blueprint. The system first combines background $b$ and inspiration knowledge $i_j$ via MOOSE-Chem to incrementally generate a hypothesis tree; each path represents a sequence of inspiration-driven updates. Subsequently, users can select a hypothesis node from the tree and route it to MOOSE-Chem2. MOOSE-Chem2 then utilizes a hierarchical search strategy, starting from high-level corrections and converging toward methodological details and experimental designs.

The output is not a single answer, but a search trajectory with history. The interface provides an input page, a tree view, a ranking page, and a feedback page: the tree view shows how hypotheses evolve from different inspirations; the ranking page displays LLM self-evaluation scores; the feedback page allows users to choose between continued exploration or entering fine-grained refinement, and to input directional feedback.

### Key Designs
1. **Exploration-Exploitation Unified State Machine**:
	- **Function**: Integrates the divergent exploration of MOOSE-Chem and the fine-grained refinement of MOOSE-Chem2 into a single workflow.
	- **Mechanism**: The exploration stage approximates $P(h \mid b)$ by iteratively selecting inspirations to update intermediate hypotheses; the refinement stage treats the initial hypothesis $h_0$ as an object for optimization, correcting it across different abstraction layers to make it more executable and consistent.
	- **Design Motivation**: Independent exploration tends to yield coarse ideas, while independent refinement lacks a mechanism for starting point selection; the unified framework creates an exploration-exploitation loop.

2. **Three Types of HAII Guidance Signals**:
	- **Function**: Allows human experts to influence the search process in explicit, reusable ways.
	- **Mechanism**: $f_{init}$ acts as an initial blueprint to constrain the root node search boundary; $f_{route}$ acts as inter-stage routing, letting users decide whether to drill down from conceptual space $\mathcal{C}$ to execution space $\mathcal{E}$ or return to exploration; $f_{dir}$ acts as intra-stage feedback, incorporating critiques into the context to trigger regenerative generation.
	- **Design Motivation**: Automated systems struggle to judge when to continue diverging versus when to converge, whereas domain experts can rapidly identify promising branches and erroneous directions.

3. **Interactive Tree Interface**:
	- **Function**: Lowers the barrier to using command-line agent tools and ensures the hypothesis evolution process is traceable.
	- **Mechanism**: Each hypothesis is a node in the tree; users can visually inspect paths, compare rankings, select nodes, and input feedback before deciding the next call to MOOSE1 or MOOSE2.
	- **Design Motivation**: Scientists require not just automated answers, but an understanding of how ideas were generated, where they branched, and why a specific node is worth pursuing.

### Loss & Training
This paper does not train new large models; it primarily evaluates the system's responsiveness to human guidance signals. The experiments use oracle-simulated evaluation: an oracle LLM has access to ground-truth fine-grained hypotheses but can only generate directional critiques without leaking answers; node selection is also simulated via oracle ranking to represent high-quality expert routing. This aims to establish a performance upper bound under structured expert signals rather than estimating average user performance.

## Key Experimental Results

### Main Results
Experiments were conducted on TOMATO-Chem2, containing research problems, literature surveys, and fine-grained hypothesis annotations from 51 top-tier papers. The metric is the recall of ground-truth elements in the generated hypotheses.

| Method | Main Setup | Recall | Search Steps |
|------|----------|--------|--------------|
| baseline_MC | MOOSE-Chem exploration only | 11.44% | N/A |
| baseline_MC2 | MOOSE-Chem2 refinement only | 10.33% | 478.6 |
| MC_with_hint | MOOSE-Chem + initial blueprint | 15.37% | N/A |
| MC_with_feedback_with_hint | Initial blueprint + oracle ranking + feedback + MOOSE-Chem | 16.93% | N/A |
| MC2_with_MC_input_oracle_rank | Oracle node selection after exploration -> MOOSE-Chem2 | 18.26% | 336.6 |
| MC2_with_feedback_oracle_rank | Oracle node selection + 1 feedback refinement | 21.98% | 166.1 |
| MC2_with_strong_feedback_x4_oracle_rank | Oracle node selection + 4 strong feedback refinements | 26.96% | 90.1 |

### Ablation Study
| Guidance Signal | Comparison Setup | Observation |
|----------|----------|------|
| Initial Blueprint | MC 11.44% vs MC_with_hint 15.37% | Initial constraints significantly narrow the search range and improve exploration quality. |
| Stage Routing | Self-ranking into MC2 (12.74%) vs Oracle-ranking into MC2 (18.26%) | The choice of which node to drill down largely determines the ceiling of subsequent refinement. |
| Directional Feedback | Standard feedback x1 (21.98%) vs Strong feedback x4 (26.96%) | Stronger, clearer feedback consistently drives up recall and reduces search steps. |
| Pure Autonomous Refinement | baseline_MC2 at 10.33% with 478.6 steps | Without human routing and feedback, fine-grained search is costly and less effective. |

### Key Findings
- The initial blueprint restricts exploration to more reasonable starting points, preventing the system from searching blindly in an oversized conceptual space.
- Routing is highly impactful: transitioning from MOOSE-Chem to MOOSE-Chem2 via self-ranked nodes yields 12.74%, whereas oracle-selected nodes reach 18.26%.
- Feedback does more than refine linguistic expression; it alters the search trajectory during the refinement stage. Strong feedback x4 achieved the highest recall (26.96%) while reducing search steps to 90.1.

## Highlights & Insights
- The most valuable contribution is formalizing "human involvement" into three control signals within the search, rather than a vague "human-in-the-loop" concept. This abstraction allows future systems to compare the marginal contributions of different human signals.
- The paper effectively distinguishes between exploratory and fine-grained hypothesis discovery. While many research agent papers only show the final idea, MOOSE-Copilot emphasizes the evolutionary path from coarse to fine.
- The tree interface is highly appropriate for scientific scenarios, as researchers typically do not accept a single answer at once but repeatedly compare, backtrack, and drill down across multiple branches.
- The significance of oracle-simulated evaluation lies in testing the system's upper bound: if expert signals are good enough, can the framework handle it? Results indicate it can, but also suggest that real-world user studies remain necessary.

## Limitations & Future Work
- The authors explicitly acknowledge that the system has not yet integrated automated experimental execution; thus, the closed loop from "hypothesis generation" to "experimental falsification" is incomplete.
- The system does not utilize post-training methods specifically for scientific hypothesis discovery; generation quality still depends on the underlying LLM and existing MOOSE modules.
- Current evaluation uses an oracle to simulate high-quality expert signals, which proves protocol effectiveness but does not directly represent the cost, cognitive load, or feedback quality of real scientists.
- Future work could connect experimental execution, literature retrieval, failure case back-propagation, and hypothesis ranking, so that $f_{dir}$ originates from real experimental results rather than just human text.

## Related Work & Insights
- **vs MOOSE-Chem**: MOOSE-Chem models exploratory discovery as inspiration-driven search. This paper retains that capacity but adds human blueprints, routing, and subsequent refinement.
- **vs MOOSE-Chem2**: MOOSE-Chem2 focuses on refining an initial hypothesis layer by layer. This paper addresses where that initial hypothesis comes from, when to enter refinement, and how to regenerate based on feedback.
- **vs IdeaSynth / NOVA / LLM-SR**: While these systems emphasize automatic generation or iterative ideas, MOOSE-Copilot prioritizes interaction protocols and visual controllability.
- **Insights**: For research agents, "controllable intermediate states" may be more important than "end-to-end automation." Abstracting user operations into evaluable signals is key to designing the next generation of scientific discovery systems.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Formalizing exploration and refinement into an HAII protocol is highly distinctive, though the generation modules are largely reused from the MOOSE series.
- Experimental Thoroughness: ⭐⭐⭐☆☆ Logical ablations are present, but the reliance on oracle-simulated settings means real user experiments are lacking.
- Writing Quality: ⭐⭐⭐⭐☆ The correspondence between motivation, protocol, and interface is clear, making the design rationale easy to follow.
- Value: ⭐⭐⭐⭐☆ Provides direct reference value for the human-AI collaborative design of research agents, particularly suited for complex scientific discovery workflows.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] SR-Scientist: Scientific Equation Discovery With Agentic AI](../../ICLR2026/llm_agent/sr-scientist_scientific_equation_discovery_with_agentic_ai.md)
- [\[ICLR 2026\] NewtonBench: Benchmarking Generalizable Scientific Law Discovery in LLM Agents](../../ICLR2026/llm_agent/newtonbench_benchmarking_generalizable_scientific_law_discovery_in_llm_agents.md)
- [\[ACL 2026\] Mina: A Multilingual LLM-Powered Legal Assistant Agent for Bangladesh](mina_a_multilingual_llm-powered_legal_assistant_agent_for_bangladesh_for_empower.md)
- [\[ACL 2026\] Temp-R1: A Unified Autonomous Agent for Complex Temporal KGQA via Reverse Curriculum Reinforcement Learning](temp-r1_a_unified_autonomous_agent_for_complex_temporal_kgqa_via_reverse_curricu.md)
- [\[ACL 2026\] FAMA: Failure-Aware Meta-Agentic Framework for Open-Source LLMs in Interactive Tool Use Environments](fama_failure-aware_meta-agentic_framework_for_open-source_llms_in_interactive_to.md)

</div>

<!-- RELATED:END -->
