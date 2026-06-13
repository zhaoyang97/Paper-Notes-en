---
title: >-
  [Paper Note] Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation
description: >-
  [ACL 2026][Text Generation][Narrative Generation] This paper proposes the PLOTTER framework, which is the first to shift narrative planning from textual representations to graph structures (Event Graph + Character Graph)…
tags:
  - "ACL 2026"
  - "Text Generation"
  - "Narrative Generation"
  - "Graph-based Reasoning"
  - "Event Graph"
  - "Character Graph"
  - "Multi-agent Iterative Optimization"
date: 2026-05-08
content_hash: 795ff65b95ab6756
---

# Planning Beyond Text: Graph-based Reasoning for Complex Narrative Generation

**Conference**: ACL 2026  
**arXiv**: [2604.21253](https://arxiv.org/abs/2604.21253)  
**Code**: None  
**Area**: LLM Efficiency  
**Keywords**: Narrative Generation, Graph-based Reasoning, Event Graph, Character Graph, Multi-agent Iterative Optimization

## TL;DR
This paper proposes the PLOTTER framework, which is the first to shift narrative planning from textual representations to graph structures (Event Graph + Character Graph). Through a multi-agent Evaluate-Plan-Revise iterative loop, it diagnoses and repairs narrative defects on the graph topology, significantly outperforming existing methods in dimensions such as narrativity, characterization, and dramatic tension.

## Background & Motivation

**Background**: LLMs are capable of generating fluent text. Research on long-form narrative generation has evolved along two lines: outline-based planning (e.g., hierarchical outline generation in Re3, DOC, DOME) and role-playing-based planning (e.g., multi-agent simulation in HoLLMwood, IBSEN).

**Limitations of Prior Work**: (1) Outline-based methods operate sequentially, where early logical errors cascade downstream, and rigid outlines limit the flexibility of complex revisions; (2) Role-playing methods perform well in stylistic diversity and dialogue richness, but coordination relies on unstructured natural language, which is prone to semantic drift and instruction misunderstanding in long contexts; (3) Both approaches fail to maintain global narrative coherence, contextual logical consistency, and smooth character development—often producing scripts that are monotonous or structurally fragmented.

**Key Challenge**: Narrative planning directly on textual representations is inherently inefficient—it lacks explicit modeling of plot dependencies, preventing the system from effectively reasoning about the underlying causal network and character-event evolutionary relationships, which eventually limits the ability to generate rigorous narrative structures.

**Goal**: Transform script generation from a sequential planning problem into a dynamic graph generation and refinement problem, achieving causal-level diagnosis and repair via iterative editing on the graph topology.

**Key Insight**: Drawing from classical narratology theories (Barthes' Logic of Action, Moretti's Character Network Theory), the narrative's causal skeleton and character social dynamics are explicitly represented using graph structures.

**Core Idea**: Perform narrative planning on graph structures rather than text—resolving causal breaks and character inconsistencies through atomic editing operations on the Event Graph and Character Graph.

## Method

### Overall Architecture
PLOTTER consists of three stages: (1) Graph-structured Script Planning—generating an initial Event Graph $G_e$ and Character Graph $G_c$ from a premise $P$; (2) Iterative Graph Refinement—a multi-agent jury diagnoses graph topological issues and constrains a graph editor to perform atomic repair operations, looping for up to $K$ rounds; (3) Graph-Guided Script Synthesis—serializing the event graph and character descriptions for a state-aware generator to produce the final script scene by scene.

### Key Designs

1.  **Dual-Graph Narrative Representation (Event Graph + Character Graph)**:
    - **Function**: Explicitly models the narrative's causal skeleton and character relationship network.
    - **Mechanism**: In the Event Graph $G_e = (V_e, E_e)$, each node represents a plot event with attributes including event description, narrative stage (rising action-climax, etc.), and time index; directed edges encode narrative relationship labels $\rho(e) \in \{\text{Causal}, \text{Foreshadowing}, \text{Suspense}\}$. In the Character Graph $G_c = (V_c, E_c)$, each node encodes multi-dimensional character attributes (core personality, internal conflict, external goals, hidden secrets), and edges represent evolutionary relationships (conflict/cooperation/emotional/secret).
    - **Design Motivation**: Textual outlines cannot explicitly capture foreshadowing and suspense relationships between non-adjacent events, nor can they model the dynamic evolution of character relationships; graph structures make these long-range dependencies editable first-class citizens.

2.  **Multi-Agent Jury + Constrained Graph Editor (Evaluate-Plan-Revise Loop)**:
    - **Function**: Systematically diagnoses structural defects in the narrative graph and executes constraint-compliant repairs.
    - **Mechanism**: Three specialized critic agents execute in a fixed order: Theme Critic (detects thematic drift and insufficient exposition) → Character Critic (detects character flattening, lack of motivation, sudden attitude shifts) → Plot Critic (detects causal breaks, logical contradictions, missing foreshadowing). Each agent outputs a structured issue list $\mathcal{I}_i$. Cross-agent validation ensures only edits with consistent support are executed. The constrained graph editor maps issues to atomic editing operations (e.g., Add-Plot-Bridge, Revise-Event) and validates them under two symbolic constraints: (1) Causal Rationality $\mathcal{K}_C$—the causal subgraph must remain a DAG (no temporal loops); (2) Narrative Completeness $\mathcal{K}_N$—all nodes must be reachable from the start and have a path to the end.
    - **Design Motivation**: Text-level reviews are prone to ambiguity and semantic drift; symbolic-level constraint validation is deterministic (independent of the LLM), preventing structurally invalid edits from propagating.

3.  **Graph-Guided Progressive Script Synthesis**:
    - **Function**: Transforms optimized symbolic graphs into coherent long-form script text.
    - **Mechanism**: First, the event graph is serialized into a hierarchical event plan $\mathcal{T}_h$ via deterministic depth-first traversal (prioritizing suspense successors and preserving foreshadowing clues). Simultaneously, character graph nodes are expanded into detailed character profiles. Then, all scene beats are generated at once, and finally, a state-aware generator expands them scene-by-scene—conditioned on event relationship types (suspense/conflict, etc.), character profiles, and the rolling narrative state $M_i$.
    - **Design Motivation**: Deterministic serialization ensures that the graph's causal topology is not destroyed during texturization; state-aware memory prevents referential breaks in long-range generation.

### Loss & Training
Training-free—PLOTTER is a pure inference-time framework, using existing LLMs (GPT-4.1, DeepSeek-R1, Qwen3) as the backbone. Evaluation is conducted using GPT-4.1 for pairwise comparison + human evaluation.

## Key Experimental Results

### Main Results (GPT-4.1 Backbone, Pairwise Win Rate)

| Dimension | vs LLM-Plan-Write | vs Dramatron | vs DOC |
| :--- | :--- | :--- | :--- |
| Narrative (Script) | 72% | 74% | 92% |
| Thematic (Script) | 100% | 90% | 86% |
| Characterization (Script) | 100% | 76% | 92% |
| Dramatic Engagement (Script) | 96% | 72% | 92% |
| Premise Fidelity (Script) | 40% | 14% | 44% |

### Ablation Study

| Configuration | Effect Description |
| :--- | :--- |
| w/o Character module | Largest drop in character and dramatic dimensions |
| w/o Plot module | Largest drop in narrative dimension |
| w/o Theme module | Noticeable drop in thematic dimension but minor overall impact |
| Single module vs Full | Full module win rate >80%, far exceeding the sum of single modules—"1+1>2" synergy (+29% Storyline, +34% Script) |
| K=3 Iterations (Default) | Distinct-2=0.793, Self-BLEU=0.017, optimal balance point |
| K=5 Iterations | Quality degradation (Distinct-2=0.640), edit success rate drops to 0.83 |

### Key Findings
- PLOTTER defeats all baselines with an overwhelming advantage in four dimensions: narrative, thematic, character, and dramatic (win rates 72-100%)—the only slight weakness is premise fidelity.
- Strong synergy exists among the three critic agents—improvements from any single agent are limited, but the win rate jumps by 29-34% when they collaborate, validating the necessity of cross-dimensional joint optimization.
- The default K=3 iterations is the optimal choice—excessive iterations (K=5) lead to a decrease in edit success rate and quality degradation.
- Human evaluation is highly consistent with LLM evaluation (Cohen's κ = 0.834), enhancing the credibility of the conclusions.
- The cost per script is 1.68 USD (K=3) and 0.36 USD in budget mode (K=1), making the computational burden manageable.

## Highlights & Insights
- The **paradigm shift in narrative planning from text to graphs** is the core contribution—graph structures turn causal reasoning, foreshadowing relationships, and character dynamics into editable symbolic objects rather than vague textual hints. This aligns with the "design data structures before writing algorithms" philosophy in software engineering.
- The deterministic validation of DAG and connectivity constraints is elegant—symbolic checks independent of the LLM ensure structural validity and avoid the propagation of unreliable LLM critiques.
- The case study on the "Trinity of Action" repair strategy (Why-Who-How triple bridging) is compelling—demonstrating that complex narrative breaks require multi-layered causal chain repairs rather than simple text polishing.

## Limitations & Future Work
- Premise Fidelity is a clear weakness—iterative graph refinement may deviate from the original premise.
- Highly dependent on the capabilities of the backbone LLM—the advantage over Dramatron is smaller on DeepSeek-R1 than on GPT-4.1.
- The evaluation dataset contains only 50 premises; although it covers 9 genres, the sample size per genre is limited.
- It has not been compared with newer baselines (e.g., StoryWriter) under equivalent conditions.
- Computational costs are relatively high (523k tokens/script); optimization is needed for large-scale applications.

## Related Work & Insights
- **vs DOC (Yang et al., 2023)**: DOC uses static hierarchical textual outlines as constraints, while Ours uses dynamic editable graph structures—the flexibility of graphs allows for non-linear revisions instead of linear outline rewriting.
- **vs Dramatron (Mirowski et al., 2023)**: Dramatron is based on role-play and free-text coordination; the lack of a shared symbolic state leads to instruction misunderstanding. PLOTTER's graph structure provides a shared, verifiable state.
- **vs R2 (Lin et al., 2025)**: R2 extracts static graphs from full source texts as generation references, while PLOTTER performs dynamic planning and iterative editing on the graph—the former is passive reference, the latter is active reasoning.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Graph-based narrative planning + symbolic constraint editing + multi-agent synergy is a brand-new paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐ 3 LLM backbones, 3 baselines, pairwise + human + objective metrics, but the data scale is small.
- Writing Quality: ⭐⭐⭐⭐⭐ Vivid case studies, clear method description, and solid theoretical motivation.
- Value: ⭐⭐⭐⭐ Fundamentally advances the methodology of long-form narrative generation; graph-based planning ideas are transferable to other long-term planning tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Beyond Isolated Words: Diffusion Brush for Handwritten Text-Line Generation](../../ICCV2025/nlp_generation/beyond_isolated_words_diffusion_brush_for_handwritten_text-line_generation.md)
- [\[ACL 2026\] Adaptive Planning for Multi-Attribute Controllable Summarization with Monte Carlo Tree Search](adaptive_planning_for_multi-attribute_controllable_summarization_with_monte_carl.md)
- [\[ACL 2026\] Frankentext: Stitching Random Text Fragments into Long-Form Narratives](frankentext_stitching_random_text_fragments_into_long-form_narratives.md)
- [\[AAAI 2026\] Structured Language Generation Model: Loss Calibration and Formatted Decoding for Efficient Text](../../AAAI2026/nlp_generation/structured_language_generation_model_loss_calibration_and_formatted_decoding_for.md)
- [\[ACL 2026\] Right at My Level: A Unified Multilingual Framework for Proficiency-Aware Text Simplification](right_at_my_level_a_unified_multilingual_framework_for_proficiency-aware_text_si.md)

</div>

<!-- RELATED:END -->
