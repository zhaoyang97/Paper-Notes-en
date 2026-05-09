---
title: >-
  [Paper Note] Text-to-Code Generation for Modular Building Layouts in Building Information Modeling
description: >-
  [NeurIPS 2025][Text-to-Code] This paper proposes Text2MBL, a framework that translates natural language descriptions into executable BIM code (rather than coordinate sequences). Through an object-oriented code architecture and LLM fine-tuning, it enables automatic generation of modular building layouts, achieving 10%+ IoU improvement in geometric consistency over coordinate-driven methods.
tags:
  - NeurIPS 2025
  - Text-to-Code
  - BIM
  - Modular Building Layout
  - Code Generation
  - LLM Fine-tuning
date: 2026-05-08
content_hash: d27a2caf4ba96821
---

# Text-to-Code Generation for Modular Building Layouts in Building Information Modeling

**Conference**: NeurIPS 2025  
**arXiv**: [2509.23713](https://arxiv.org/abs/2509.23713)  
**Code**: [GitHub](https://github.com/CI3LAB/Text2MBL)  
**Area**: NLP Understanding / Code Generation / BIM Automation  
**Keywords**: Text-to-Code, BIM, Modular Building Layout, Code Generation, LLM Fine-tuning

## TL;DR
This paper proposes Text2MBL, a framework that translates natural language descriptions into executable BIM code (rather than coordinate sequences). Through an object-oriented code architecture and LLM fine-tuning, it enables automatic generation of modular building layouts, achieving 10%+ IoU improvement in geometric consistency over coordinate-driven methods.

## Background & Motivation

**Background**: Modular construction prefabricates standardized 3D volumetric units in factories for on-site assembly, representing a growing trend in the construction industry. BIM (Building Information Modeling) is a core digital tool throughout the building lifecycle, encoding semantically rich structural information. Text-driven design has recently emerged as a new research direction.

**Limitations of Prior Work**:
   - Existing text-to-layout methods (e.g., Tell2Design, Text2BIM) output coordinate sequences or pixel images; coordinate-based methods suffer sharp performance degradation as the number of components increases, since LLMs must perform complex spatial geometric reasoning.
   - Image outputs require post-processing (raster-to-vector conversion) to be converted into BIM format, which is prone to spatial conflicts and semantic inconsistencies.
   - The three-level hierarchy unique to modular construction (Module → Unit → Room) is far more complex than conventional layouts, which existing methods cannot handle.
   - An output format compatibility gap exists between conceptual design and construction workflows.

**Key Challenge**: LLMs excel at sequence generation but are poor at spatial geometric reasoning; modular construction demands precise hierarchical constraints and geometric consistency — a representation is needed that reduces spatial reasoning to sequence understanding.

**Goal**: (1) Design a code-driven MBL representation that allows LLMs to generate high-level semantic action sequences rather than low-level coordinates; (2) construct the first text–BIM code paired dataset; (3) demonstrate that the code paradigm significantly outperforms the coordinate paradigm.

**Key Insight**: Leveraging BIM's inherently object-oriented nature, layout generation is abstracted as a series of code operations (creating modules, splitting, merging, assigning rooms, etc.), each encapsulating the underlying geometric logic. This allows LLMs to focus on semantic understanding rather than coordinate derivation.

**Core Idea**: Use object-oriented code as an intermediate representation, reformulating the text-to-BIM-layout problem as a sequence-to-sequence code generation task.

## Method

### Overall Architecture
The input is a natural language architectural description (e.g., "two modules; the northern module contains a bedroom and a living room..."), and the output is a C# code sequence directly executable in Autodesk Revit. The code executes step by step: create modules → define units → assign rooms → place architectural elements such as doors and windows.

### Key Designs

1. **Object-Oriented Code Architecture (Module/Unit/Room/Utils)**:

    - Function: Defines four core classes that encapsulate the hierarchical structure and operations of modular buildings.
    - Mechanism: `Module` supports creation via absolute coordinates and relative positions, as well as splitting and merging; `Unit` is composed of a set of Modules; `Room` can be assigned based on module/unit, direction, corner, relative position, and other modes; `Utils` provides static methods for door/window creation, splitting, and merging.
    - Design Motivation: Complex geometric reasoning (collision detection, concave polygon handling, etc.) is encapsulated at the lower level, so LLMs only need to invoke high-level interfaces. For example, "create a kitchen on the south side of module 1" requires only a single line: `Room kitchen = new Room(name: "Kitchen", module: m1, direction: "south", dimension: 1800)`.
    - Key Advantage: **Relative positioning** replaces absolute coordinates — new modules are positioned relative to existing ones, eliminating the need for the LLM to compute absolute coordinates.

2. **Concept Formalization**:

    - Function: Defines MBL as a 6-tuple $L = \langle \mathcal{M}, \mathcal{U}, \mathcal{R}, \mathcal{E}, \mathcal{A}, \mathcal{C} \rangle$.
    - Mechanism: The three-level hierarchy of modules, units, and rooms must satisfy nesting constraints: each unit is fully contained within the union of modules, and each room is nested within a unit. In addition to adjacency and connectivity relations, a **conjoint** relation is introduced to represent the co-existence of rooms within the same module.
    - Design Motivation: Provides a mathematical foundation for the code architecture and ensures that generated layouts satisfy the physical constraints of modular construction.

3. **Data Collection and Synthetic Data Augmentation**:

    - Function: Collects 198 real modular building designs from the Hong Kong Architectural Services Department, each annotated with executable BIM code and two types of textual descriptions.
    - Mechanism: Three data augmentation strategies are employed: (a) partial synthesis — generating descriptions from existing code using templates or GPT; (b) full synthesis — generating new code from grammar rules and then generating descriptions. Training on a mix of original and synthetic data outperforms using synthetic data alone.
    - Two code styles: named arguments (explicit parameter names) and positional arguments (parameter passing by position), forming $\mathcal{D} = \{(d_i^a, d_i^b, c_i^{\text{name}}, c_i^{\text{pos}})\}$.
    - Design Motivation: MBL domain data is extremely scarce (only 198 samples), making synthetic augmentation a necessary measure.

4. **LLM Fine-tuning**:

    - Function: Fine-tunes the Qwen2.5 model family (0.5B–7B, including vanilla/Coder/Math variants).
    - Mechanism: The task is modeled as conditional code generation $\hat{c} = \arg\max_c p_\theta(c | d)$, with LoRA fine-tuning.
    - Three evaluation dimensions: **executable validity** (compile rate + pass rate), **semantic fidelity** (instance F1 + argument F1), and **geometric consistency** (IoU).

## Key Experimental Results

### Main Results: Code vs. Coordinates

| Model | Output Format | Overall IoU | Module IoU | Unit IoU | Room IoU |
|-------|--------------|-------------|------------|----------|----------|
| Qwen2.5-1.5B | Coordinates | 78.13% | 86.98% | 80.27% | 69.85% |
| Qwen2.5-1.5B | Code | 91.65% | 95.47% | 93.79% | 86.79% |
| Qwen2.5-7B | Coordinates | 85.38% | 90.35% | 90.39% | 77.33% |
| Qwen2.5-7B | Code | 95.83% | 98.51% | 98.11% | 91.64% |
| Qwen2.5-Coder-7B | Coordinates | 84.64% | 90.12% | 88.18% | 77.53% |
| Qwen2.5-Coder-7B | Code | **98.43%** | **98.78%** | **99.19%** | **97.37%** |

### Ablation Study

| Configuration | Key Finding |
|---------------|-------------|
| Increasing number of components | Coordinate method IoU drops sharply; code method is more robust |
| Only 10% original data + synthetic augmentation | Large performance gain, confirming effectiveness of synthetic data |
| Named vs. positional arguments | Named arguments yield higher compile rates under abstract instructions (more robust) |
| Coder vs. vanilla vs. Math | Coder series performs best overall; Math ranks second |

### Key Findings
- **Code comprehensively outperforms coordinates**: Code-driven generation significantly surpasses coordinate-driven generation in IoU across all model sizes (+10–15%), with the advantage growing as the number of components increases.
- **Relative positioning is the key**: The code architecture eliminates the need for absolute coordinate computation through relative position descriptions, which directly explains why LLMs perform better with code than with coordinates.
- **Semantic fidelity exceeds pass rate**: Models are better at understanding and extracting parameter information (high argument F1), but producing perfectly executable code remains challenging (lower pass rate), suggesting that argument extraction could serve as a post-processing repair mechanism.
- **Synthetic data augmentation is effective**: Under extremely low-data settings (10% original data), both template-based and model-based synthesis augmentation significantly improve performance.
- **Qwen2.5-Coder-7B achieves the best results**: Reaching 98.43% IoU, approaching near-perfect performance.

## Highlights & Insights
- **The paradigm of code as intermediate representation is highly elegant**: Reformulating spatial reasoning as semantic code generation leverages the LLM's code generation capability rather than geometric reasoning ability — an approach generalizable to CAD, circuit design, and other domains.
- **Relative positioning to eliminate absolute coordinates**: Modules are described using "south"/"east" + offset rather than (x, y) coordinates, substantially reducing the reasoning burden on LLMs.
- **Rigorous concept formalization**: The hierarchical structure of modular construction is formalized as a mathematical definition, providing clear theoretical grounding for the code architecture.

## Limitations & Future Work
- **Extremely small dataset**: Only 198 real designs; generalizability remains to be validated.
- **Limited to parametric design**: The current framework requires users to provide precise spatial/geometric parameters and does not support vague or high-level conceptual descriptions (e.g., "a spacious living room").
- **Limited variety of architectural elements**: The same wall, floor, and door families are used throughout, with no consideration of variations in material, width, etc.
- **No user study conducted**: Evaluation of actual designer experience is absent.
- **Pass rate still has room for improvement**: Although IoU is high, code compilation pass rates under certain settings still require post-processing repair.

## Related Work & Insights
- **vs. Text2BIM/Tell2Design**: These methods output coordinate sequences, requiring LLMs to perform spatial reasoning; Text2MBL outputs code, encapsulating spatial reasoning at the code level, with a performance gap of 10%+ IoU.
- **vs. Text2CAD**: A similar idea applied to CAD modeling, but without addressing the hierarchical constraints and semantically rich BIM format specific to the architectural domain.
- **The trend of code as intermediate representation**: This work aligns with AutomaTikZ (scientific vector graphics) and DiagrammerGPT (open-domain diagrams), where code/symbolic grammar serves as an interpretable intermediate representation — a common direction in design automation.

## Rating
- Novelty: ⭐⭐⭐⭐ The code-driven paradigm replacing coordinates is a first in the architectural domain; the BIM code architecture design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model, multi-scale comparisons are comprehensive, but the small dataset size limits the generalizability of conclusions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Embedding Alignment in Code Generation for Audio](embedding_alignment_in_code_generation_for_audio.md)
- [\[ICCV 2025\] TikZero: Zero-Shot Text-Guided Graphics Program Synthesis](../../ICCV2025/code_intelligence/tikzero_zero-shot_text-guided_graphics_program_synthesis.md)
- [\[NeurIPS 2025\] MaintainCoder: Maintainable Code Generation Under Dynamic Requirements](maintaincoder_maintainable_code_generation_under_dynamic_requirements.md)
- [\[NeurIPS 2025\] QiMeng-SALV: Signal-Aware Learning for Verilog Code Generation](qimeng-salv_signal-aware_learning_for_verilog_code_generation.md)
- [\[NeurIPS 2025\] Table2LaTeX-RL: High-Fidelity LaTeX Code Generation from Table Images via Reinforced Multimodal Language Models](table2latex-rl_high-fidelity_latex_code_generation_from_table_images_via_reinfor.md)

</div>

<!-- RELATED:END -->
