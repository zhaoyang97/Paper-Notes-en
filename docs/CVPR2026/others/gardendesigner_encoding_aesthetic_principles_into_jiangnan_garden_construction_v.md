---
title: >-
  [Paper Note] GardenDesigner: Encoding Aesthetic Principles into Jiangnan Garden Construction via a Chain of Agents
description: >-
  [CVPR 2026][Jiangnan garden] This paper proposes GardenDesigner, a framework that encodes the aesthetic principles of Jiangnan gardens into computable constraints through a chain of agents (terrain distribution → road ge…
tags:
  - "CVPR 2026"
  - "Jiangnan garden"
  - "chain of agents"
  - "procedural modeling"
  - "aesthetic constraints"
  - "layout optimization"
date: 2026-05-08
content_hash: a59164f009aac8d0
---

# GardenDesigner: Encoding Aesthetic Principles into Jiangnan Garden Construction via a Chain of Agents

**Conference**: CVPR 2026  
**arXiv**: [2604.01777](https://arxiv.org/abs/2604.01777)  
**Code**: [https://github.com/monad-cube/GardenDesigner](https://github.com/monad-cube/GardenDesigner)  
**Area**: Scene Generation / Cultural Heritage  
**Keywords**: Jiangnan garden, chain of agents, procedural modeling, aesthetic constraints, layout optimization

## TL;DR
This paper proposes GardenDesigner, a framework that encodes the aesthetic principles of Jiangnan gardens into computable constraints through a chain of agents (terrain distribution → road generation → asset selection → layout optimization). Combined with the expert-annotated GardenVerse dataset, the framework enables non-expert users to automatically construct aesthetically compliant Jiangnan gardens from text input within one minute.

## Background & Motivation
1. **Background**: Jiangnan gardens represent a major school of classical Chinese garden design with significant application potential in digital tourism, film, and game production. Traditional digital modeling of such gardens relies heavily on expert knowledge, typically requiring 3–4 designers and 3–4 weeks to complete.
2. **Limitations of Prior Work**: Existing learning-based scene generation methods are constrained by training data domains and exhibit limited generalization; procedural modeling methods combined with LLMs/VLMs primarily focus on indoor spaces or unstructured natural scenes, and cannot handle the intricate spatial composition characteristic of Jiangnan gardens.
3. **Key Challenge**: Jiangnan gardens pose three unique challenges: (1) complex water-centric terrain and spatial layout, (2) abstract aesthetic principles that are difficult to encode as computational constraints, and (3) the absence of culturally annotated garden datasets.
4. **Goal**: To transform the implicit aesthetic rules of Jiangnan gardens—water-centricity, winding paths leading to secluded spots, symbolic miniaturization, and asymmetric balance—into an optimizable procedural generation pipeline.
5. **Key Insight**: Decompose garden construction into four sequentially dependent subtasks, executed step by step by a chain of agents, each embedding relevant aesthetic constraints.
6. **Core Idea**: Drive procedural modeling through a chain of LLM agents, encoding aesthetic principles as fitness functions for genetic algorithms and loss functions for layout optimization.

## Method

### Overall Architecture
The input is a user text description and the output is a complete 3D Jiangnan garden. The pipeline consists of two major modules: (1) **Hierarchical Garden Composition**, which generates terrain and roads; and (2) **Knowledge-Embedded Asset Arrangement**, which selects assets and optimizes their layout. Four agents execute in a chained sequence: terrain distribution $\mathcal{A}_T$ → road generation $\mathcal{A}_R$ → asset selection $\mathcal{A}_S$ → layout optimization $\mathcal{A}_C$, where the output of each agent serves as the input to the next.

### Key Designs

1. **Genetic Algorithm-Driven Terrain Distribution Agent**

    - **Function**: Generates water-centric terrain distributions from text instructions.
    - **Mechanism**: A genetic algorithm on a 2D grid classifies terrain into four categories: Outside, Waterbody, Land, and Ground. The LLM converts user text into genetic algorithm parameters (presence, quantity, coverage rate, and per-region coverage rate), after which crossover, mutation, and evolution operations generate the terrain. The key innovation is a **water-centric fitness function** $L_{\text{terrain}} = f \cdot \max(1 - \frac{\sum c(T,(x_i,y_i))}{\phi}, 0)$, which ensures the water body serves as the spatial organizing core of the garden.
    - **Design Motivation**: Conventional procedural terrain algorithms cannot capture the water-centric spatial logic of Jiangnan gardens, and tend to produce scattered ponds and unnatural terrain configurations.

2. **Exploratory Road Generation Agent**

    - **Function**: Generates a road system on the terrain that conforms to the principle of "winding paths leading to secluded spots."
    - **Mechanism**: The agent first extracts parameters from user instructions (number of entrances, key points, main road width, and road complexity), then selects optimal paths along grid boundaries via a scoring mechanism. The scoring rules reflect three aesthetic-principle-derived requirements: roads should reach most areas, preferentially follow boundaries, and avoid both excessive curvature and excessive straightness. This is formalized as $R = \mathcal{A}_R(\mathcal{S}(T, e_{i,j}), U, K_{\text{global}})$.
    - **Design Motivation**: Existing path generation methods pursue geometric efficiency or uniform coverage, overlooking the exploratory path design principle of "a new scene at every step" inherent to Jiangnan gardens.

3. **Knowledge-Guided Asset Retrieval and Aesthetics-Constrained Layout Optimization**

    - **Function**: Selects culturally appropriate assets from the GardenVerse dataset and optimizes their placement according to aesthetic constraints.
    - **Mechanism**: The asset selection agent $\mathcal{A}_S$ constructs a vector store from expert-annotated garden knowledge (visual attributes, spatial combinations, suitable seasons, etc.) and uses an LLM to query it for culturally consistent asset selection per region. The layout optimization agent $\mathcal{A}_C$ defines five categories of optimization losses: Global (edge/central positioning), Position (surrounding/backing relationships), Distance (near/far spacing), Alignment, and Rotation. The final loss is $\mathcal{L}_{\text{opt}} = \lambda_1 \mathcal{L}_{\text{glo}} + \lambda_2 \mathcal{L}_{\text{pos}} + \lambda_3 \mathcal{L}_{\text{dis}} + \lambda_4 \mathcal{L}_{\text{ali}} + \lambda_5 \mathcal{L}_{\text{rot}}$, with a feasible layout found via depth-first search.
    - **Design Motivation**: General-purpose LLMs lack domain knowledge of garden design and cannot reason about the cultural associations among architecture, plants, and rockery; conventional retrieval and constraint methods cannot capture the culturally implicit spatial logic.

### Loss & Training
Terrain generation employs a water-centric fitness function. Layout optimization uses a weighted combination of five spatial constraint losses with weights $\lambda = \{2.0, 0.5, 1.8, 0.5, 0.5\}$. GPT-5 serves as the LLM backbone, and Unity is used as the visualization and interaction platform.

## Key Experimental Results

### Main Results

| Method | Path-S ↑ | Class-Div | FD | CLIP-S ↑ |
|------|----------|-----------|-----|----------|
| Liu et al. (baseline) | 0 | 21.8±1.6 | 1.42±0.1 | 27.4±0.1 |
| **GardenDesigner** | **8.1±2.5** | **68.3±5.6** | **1.36±0.1** | **27.6±0.1** |

| Method | CLIP-A ↑ | VLM-S ↑ | QA-Quality ↑ |
|------|----------|---------|--------------|
| Liu et al. | 52.9±1.0 | 24.9±1.2 | 43.8±2.5 |
| **GardenDesigner** | **54.2±2.0** | **32.5±2.3** | **53.8±3.1** |

### Ablation Study

| Configuration | FD | CLIP-S ↑ | VLM-S ↑ |
|------|-----|----------|---------|
| GardenDesigner w/o Asset Arrange. | 1.27±0.1 | 27.4±0.1 | 31.6±1.1 |
| **Full GardenDesigner** | **1.36±0.1** | **27.6±0.1** | **32.5±2.3** |

### Key Findings
- Path-S improves from 0 to 8.1, indicating that the baseline entirely fails to produce reasonable road–building relationships, whereas GardenDesigner's roads successfully connect to key scenic spots.
- Asset diversity (Class-Div) increases by more than threefold (21.8→68.3), expanding from 26 to 71 asset categories.
- FD=1.36 approximates the fractal dimension range of real Jiangnan gardens (1.123–1.329), indicating a more naturalistic spatial structure.
- In human evaluation, 11 garden experts and 32 general users both preferred GardenDesigner across all dimensions, with a particularly strong preference on the cultural atmosphere dimension.
- The inclusion of the GardenVerse dataset alone substantially improved baseline quality, underscoring the importance of high-quality domain-specific datasets.

## Highlights & Insights
- **The chain-of-agents decomposition design is particularly elegant**: decomposing complex garden construction into four subtasks with clear dependency relationships, each with well-defined inputs and outputs, effectively leverages LLMs' language understanding while ensuring spatial constraint precision through procedural algorithms.
- **Computationalization of aesthetic principles**: translating abstract concepts such as "water-centricity" and "winding paths to secluded spots" into fitness functions and loss functions demonstrates a compelling paradigm of bridging humanistic knowledge and mathematical optimization, transferable to other cultural heritage digitization contexts.
- **Expert annotation in GardenVerse**: beyond basic metadata, annotations include garden domain knowledge (suitable seasons, cultural context, etc.), providing LLMs with essential domain-specific supplementary knowledge.

## Limitations & Future Work
- Reliance on the 132 assets in GardenVerse limits diversity and makes comprehensive coverage of all Jiangnan garden elements difficult.
- Evaluation metrics are primarily based on VLM scores and human assessments, lacking quantitative measures for professional garden design criteria such as spatial accessibility and sightline analysis.
- Errors in the chained agents propagate downstream—unreasonable terrain generation will adversely affect all subsequent steps.
- The framework currently targets only the Jiangnan garden style; its extensibility to other garden traditions (e.g., imperial Chinese gardens, Japanese gardens) remains to be validated.

## Related Work & Insights
- **vs. Liu et al. (LLM for landscape)**: Their approach applies LLMs to general landscape generation but lacks garden-specific knowledge and cultural constraints, resulting in layouts with large empty areas. GardenDesigner addresses this through expert knowledge embedding and aesthetic loss functions.
- **vs. Infinigen**: Infinigen focuses on procedural generation of natural scenes without cultural constraints. The chain-of-agents + aesthetic encoding paradigm of GardenDesigner is generalizable to other culturally specific scene generation tasks.
- This paper demonstrates the feasibility of **computationalizing humanistic knowledge**, inspiring reflection on how expert knowledge from other domains can be encoded as optimizable constraints.

## Rating
- Novelty: ⭐⭐⭐⭐ The framing of Jiangnan garden aesthetic principles as a computational framework is distinctly original, though the technical contributions primarily combine existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Includes quantitative comparisons, human evaluation, and ablation studies, though only a single baseline is used.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with commendable formalization of aesthetic principles.
- Value: ⭐⭐⭐⭐ Cultural heritage digitization is an important research direction, and the GardenVerse dataset holds independent value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] A Differentiable Model of Supply-Chain Shocks](../../NeurIPS2025/others/a_differentiable_model_of_supply-chain_shocks.md)
- [\[AAAI 2026\] More Than Irrational: Modeling Belief-Biased Agents](../../AAAI2026/others/more_than_irrational_modeling_belief-biased_agents.md)
- [\[NeurIPS 2025\] Exact Learning of Arithmetic with Differentiable Agents](../../NeurIPS2025/others/exact_learning_of_arithmetic_with_differentiable_agents.md)
- [\[ICML 2026\] Markov Chain Monte Carlo without Evaluating the Target: An Auxiliary Variable Approach](../../ICML2026/others/markov_chain_monte_carlo_without_evaluating_the_target_an_auxiliary_variable_app.md)
- [\[AAAI 2026\] Symbolic Planning and Multi-Agent Path Finding in Extremely Dense Environments with Unassigned Agents](../../AAAI2026/others/symbolic_planning_and_multi-agent_path_finding_in_extremely_dense_environments_w.md)

</div>

<!-- RELATED:END -->
