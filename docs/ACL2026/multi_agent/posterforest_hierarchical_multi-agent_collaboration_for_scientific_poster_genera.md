---
title: >-
  [Paper Note] PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation
description: >-
  [ACL2026][Multi-Agent][Scientific poster generation] PosterForest utilizes a Poster Tree as an intermediate representation that simultaneously encodes paper hierarchical semantics and poster spatial layout. Content…
tags:
  - "ACL2026"
  - "Multi-Agent"
  - "Scientific poster generation"
  - "hierarchical document understanding"
  - "multi-agent collaboration"
  - "layout planning"
  - "Poster Tree"
date: 2026-05-08
content_hash: 502a3bd91d5cb018
---

# PosterForest: Hierarchical Multi-Agent Collaboration for Scientific Poster Generation

**Conference**: ACL2026  
**arXiv**: [2508.21720](https://arxiv.org/abs/2508.21720)  
**Code**: https://github.com/kaist-cvml/poster-forest  
**Area**: nlp_generation  
**Keywords**: Scientific poster generation, hierarchical document understanding, multi-agent collaboration, layout planning, Poster Tree

## TL;DR
PosterForest utilizes a Poster Tree as an intermediate representation that simultaneously encodes paper hierarchical semantics and poster spatial layout. Content, Layout, and Feedback Agents recursively collaborate to optimize the generation in a training-free manner. It achieves a 59.2% overall preference in human evaluations, significantly outperforming P2P and Paper2Poster.

## Background & Motivation
**Background**: Scientific papers are becoming increasingly long and complex, making posters an essential medium for rapid technical communication. Early automated poster generation methods relied on rule-based extraction and heuristic typesetting, while recent methods like P2P and Paper2Poster introduced LLM/MLLM multi-agent pipelines for parsing, summarization, layout, and rendering.

**Limitations of Prior Work**: Existing SPG methods often treat papers as linear text or fixed section-to-panel mappings, lacking hierarchical modeling of reference relationships between sections, subsections, paragraphs, and figures/tables. Content and layout are frequently optimized separately—panels are determined before filling text and images—leading to misplaced tables, excessively long paragraphs, improper image scaling, or fractured logical flow.

**Key Challenge**: Scientific poster generation must compress information without destroying paper logic while achieving visual balance without losing key experimental figures. A single agent or sequential pipeline struggles to simultaneously handle the competing goals of content fidelity, layout efficiency, and visual coherence.

**Goal**: The authors aim to construct a framework that requires no additional training, preserves the document's hierarchical structure, and jointly optimizes content and layout, ensuring generated posters are both informative and visually organized.

**Key Insight**: The paper proposes the Poster Tree as a unified intermediate representation: each node possesses both semantic and spatial attributes. The tree structure is inherited from the paper's hierarchy and mapped to the poster canvas via a layout tree. In this way, when an agent modifies a node, it considers parent constraints, child content, and global feedback.

**Core Idea**: Transform scientific poster generation from "linear summarization followed by typesetting" to "recursive collaborative optimization on a hierarchical semantic-spatial tree."

## Method
The core of PosterForest is to construct a tree and then refine it. Instead of forcing an LLM to output HTML or images in one shot, it first parses the paper into a Raw Doc Tree, followed by pruning, merging, and asset matching to obtain a Content Tree. A Layout Tree is then initialized based on content hierarchy and merged with the Content Tree to form the Poster Tree. Finally, multiple agents perform both local and global refinement for a maximum of `K=2` iterations before final rendering.

### Overall Architecture
The input is a paper PDF or document. The Parser Agent extracts nodes such as title, section, subsection, paragraph, figure, and table to form the Raw Doc Tree. An MLLM acts as the Refinement Agent to prune the tree, merge redundant nodes, compress long text, and preserve figure references to generate the Content Tree. Layout initialization divides the canvas into row/column/panel hierarchies based on content statistics, creating the Layout Tree. A Merge operation aligns semantic nodes with spatial nodes to form the Poster Tree. The system then performs refined rendering through tree-level iterations.

### Key Designs
1.  **Poster Tree Unified Representation**:
    - **Function**: Merges "what to display" and "where to place it" into a single hierarchical structure.
    - **Mechanism**: The Raw Doc Tree records the original structure, the Content Tree retains refined semantic nodes $c=(t,s)$ (where $t$ is the type and $s$ is the content/asset), and the Layout Tree records spatial nodes $l=(r,x)$ (where $r$ is the spatial type and $x$ is attributes like position and scale). The Poster Tree merges these so each node contains both content and layout.
    - **Design Motivation**: Poster errors often stem from the decoupling of content and spatial structures. The unified tree ensures the system knows that "Table 1 belongs to the Experiments subtree" and its current panel allocation, reducing misplacements.

2.  **Node-level Content/Layout Collaboration**:
    - **Function**: Simultaneously adjusts text density and spatial proportions at local nodes.
    - **Mechanism**: The tree is traversed from root to leaves. The Layout Agent optimizes region ratio, alignment, and spatial distribution for layout nodes based on parent information and descendants. The Content Agent adjusts text abstraction and redundancy for content nodes based on parent layout constraints. This can be formalized as $l_i^{t+1}=A_{layout}(l_i^t, P(l_i^t), D(l_i^t))$ and $c_i^{t+1}=A_{content}(c_i^t, P(c_i^t), D(P(c_i^t)))$.
    - **Design Motivation**: Modifying only content leads to unbalanced layouts, while modifying only layout results in text overflow or improper image scaling. Synchronous read-write operations enable integrated content compression and spatial allocation.

3.  **Global Feedback-Driven Tree-level Iteration**:
    - **Function**: Subjects local modifications to a global visual quality check.
    - **Mechanism**: After each node-level traversal, the system renders the current Poster Tree. An MLLM Feedback Agent evaluates visual organization, structure, and balance, outputting structured feedback and a binary signal to continue or stop. If continuing, the next traversal incorporates this global feedback.
    - **Design Motivation**: Local nodes lack a global aesthetic perspective. The Feedback Agent acts as a poster reviewer, preventing the poster from becoming crowded or fragmented despite local rationality.

### Loss & Training
PosterForest is a training-free framework with no gradient-based objectives. It relies on Docling, MLLM, and APIs for parsing, summarization, and evaluation. "Optimization" occurs via prompt-driven iterative modifications. GPT-4o is used to ensure baseline consistency. To avoid aesthetic bias from color or fonts, these elements are unified across methods. Tree-level iterations are capped at `K=2`.

## Key Experimental Results

### Main Results
Quantitative evaluation was conducted on 100 paper-poster pairs from the Paper2Poster benchmark; qualitative and user studies were conducted on 15 additional pairs from AI conferences (NeurIPS, CVPR, ACL). MLLM-as-a-Judge scored aspects from 1-5.

| Method | Training-free | Aesthetic Avg.↑ | Information Avg.↑ | Overall↑ | Key Observation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Original Paper | - | 3.58 | 4.22 | 3.90 | Most complete but not a poster |
| GT Poster | - | 3.56 | 3.98 | 3.77 | Human quality upper bound |
| 4o-HTML | Yes | 3.36 | 3.68 | 3.52 | End-to-end HTML is functional but lacks structure |
| P2P-4o | No | 3.91 | 3.94 | 3.72 | Strong aesthetics, limited flow |
| PosterAgent-4o | No | 3.58 | 3.86 | 3.72 | Stable specialized agent baseline |
| PosterForest-Qwen | Yes | 3.62 | 3.82 | 3.72 | Comparable to strong baselines with open-weights |
| PosterForest-4o | Yes | 3.65 | 3.87 | 3.76 | Best training-free performance, near GT |

Human evaluations showed a stronger preference for PosterForest across 25 AI graduate students.

| Method | Content preference↑ | Aesthetics preference↑ | Structure preference↑ | Overall preference↑ |
| :--- | :--- | :--- | :--- | :--- |
| 4o-HTML | 2.0% | 1.6% | 2.4% | 1.6% |
| P2P | 9.2% | 21.2% | 13.2% | 12.0% |
| Paper2Poster | 32.8% | 24.0% | 24.8% | 27.2% |
| PosterForest | 56.0% | 53.2% | 59.6% | 59.2% |

### Ablation Study
Ablations focused on the hierarchical Content Tree and the joint application of Content/Layout Agents.

| Configuration | Key Phenomenon | Description |
| :--- | :--- | :--- |
| w/o Hierarchical Content Tree | Disorder in sections/subsections | Results tables might appear in Introduction |
| w/ Hierarchical Content Tree | Better logical/spatial coherence | Related content is grouped; clear reader path |
| Only Content Agent | Reduced redundancy | Better fit for panels, but layout remains unbalanced |
| Only Layout Agent | Neater spatial organization | Image scaling and text overflow issues persist |
| Both Agents | Simultaneous improvement | Appropriate info density and visual harmony |

### Key Findings
- PosterForest's primary advantage is not just a marginal score lead, but a significant human preference, indicating structure and completeness are vital for real readers.
- The gap between MLLM judge and human preference suggests automated metrics still struggle to fully capture the reading experience.
- Hierarchical structure is critical for scientific documents to prevent mislabeling experimental tables or conclusion text.
- Being training-free is a practical highlight, allowing deployment to new domains without retraining instruction models.

## Highlights & Insights
- The Poster Tree is a natural yet effective intermediate representation that combines tree-like info compression with 2D layout.
- The multi-agent setup matches the real design workflow: content editor, layout designer, and reviewer. This role decomposition is more valid than general "agent discussion."
- Training-free approaches eliminate maintenance costs for tasks where styles and data distributions shift rapidly.
- Jointly optimizing visual harmony and content fidelity in the same loop—rather than a "summarize then typeset" pipeline—could benefit slides or technical report generation.

## Limitations & Future Work
- Content density is not always optimal; some areas may have underutilized space or insufficient detail.
- Quantitative evaluation still relies on MLLM judges, which don't perfectly align with human preference.
- System performance is bound by the accuracy of the parser and asset matching.
- Unified styling limits the diversity of design for different brandings or specific conference templates.
- The Feedback Agent does not yet model fine-grained design preferences like visual focus points or audience-specific emphasis.

## Related Work & Insights
- **vs P2P**: P2P uses instruction tuning for collaboration; PosterForest uses a tree structure and iterations to inject hierarchy.
- **vs Paper2Poster**: Paper2Poster follows a more sequential process, whereas PosterForest emphasizes joint content/layout modification on a unified tree.
- **vs Generic GPT-4o-HTML**: Direct generation is simpler but often ignores the specific hierarchical structure of scientific documents.
- **Insights for Doc Gen**: For slides or posters, constructing "Semantic Tree + Layout Tree" as an intermediate state for agent editing is a robust strategy.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The combination of Poster Tree and hierarchical multi-agent collaboration is highly suited for this task.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes automated/human evaluation and ablations, though more quantitative ablation stats would be better.
- Writing Quality: ⭐⭐⭐⭐☆ Visuals are clear and the mechanism is easy to follow.
- Value: ⭐⭐⭐⭐☆ Extremely practical for academic communication and training-free deployment scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] EvoSci: A Bio-Inspired Multi-Agent Framework for the Evolution of Scientific Discovery](evosci_a_bio-inspired_multi-agent_framework_for_the_evolution_of_scientific_disc.md)
- [\[ACL 2026\] ConSensus: Multi-Agent Collaboration for Multimodal Sensing](consensus_multi-agent_collaboration_for_multimodal_sensing.md)
- [\[ACL 2026\] Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration](scaling_external_knowledge_input_beyond_context_windows_of_llms_via_multi-agent_.md)
- [\[ACL 2026\] RoadMapper: A Multi-Agent System for Roadmap Generation of Solving Complex Research Problems](roadmapper_a_multi-agent_system_for_roadmap_generation_of_solving_complex_resear.md)
- [\[ACL 2026\] LLM-Based Human-Agent Collaboration and Interaction Systems: A Survey](llm-based_human-agent_collaboration_and_interaction_systems_a_survey.md)

</div>

<!-- RELATED:END -->
