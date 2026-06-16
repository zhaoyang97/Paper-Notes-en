---
title: >-
  [Paper Note] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning
description: >-
  [ACL 2026][Multimodal VLM][executable intermediate] This survey proposes two complementary perspectives: the Perception–Alignment–Reasoning (PAR) process framework and the Answer–Process–Executable (APE) evaluation framework. It systematically organizes three major task families—geometry, chart/table, and visual word problems—mapping existing methods and benchmarks onto
tags:
  - ACL 2026
  - Multimodal VLM
  - executable intermediate
date: 2026-05-08
content_hash: 2352f7d658a3e27e
---
# A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning

**Conference**: ACL 2026  
**arXiv**: [2603.08291](https://arxiv.org/abs/2603.08291)  
**Code**: Awesome Multimodal Mathematical Reasoning (GitHub repository, link provided in the paper)  
**Area**: Multimodal VLM / Multimodal Mathematical Reasoning / Survey  
**Keywords**: PAR framework, APE evaluation, geometric reasoning, chart/table reasoning, executable intermediate

## TL;DR
This survey proposes two complementary perspectives: the Perception–Alignment–Reasoning (PAR) process framework and the Answer–Process–Executable (APE) evaluation framework. It systematically organizes three major task families—geometry, chart/table, and visual word problems—mapping existing methods and benchmarks onto these coordinates, making it the first process-centric multimodal mathematical reasoning survey.

## Background & Motivation
**Background**: LLMs have approached SOTA in symbolic and arithmetic reasoning, but practical mathematical problems are often multimodal (diagrams, tables, geometric figures, coordinate graphs, mixed documents). While numerous datasets and methods for Multimodal Mathematical Reasoning (MMR) have emerged, a unified perspective linking "perception, alignment, reasoning, and evaluation" is missing.

**Limitations of Prior Work**: (1) Previous MMR surveys (e.g., Yan et al. 2024) are mostly benchmark catalogs or MLLM role classifications (Reasoner / Enhancer / Planner), which are horizontal; (2) most evaluations only consider the final answer, failing to distinguish between correct guesses based on shortcuts and truly correct reasoning; (3) different methods use varying DSLs, alignment methods, and reasoning paradigms, making horizontal comparison difficult.

**Key Challenge**: MMR differs fundamentally from text-only mathematical reasoning. Multimodal coupling causes errors in perception, alignment, and reasoning to propagate across layers. A single metric cannot pinpoint the failure stage; a process-centric perspective is required to diagnose "at which step the failure occurred."

**Goal**: Organize MMR around four fundamental questions: (1) **what to extract** from multimodal input; (2) how to **represent and align** text/visual information; (3) how to **perform reasoning** (CoT / program-aided / tool use); (4) how to **evaluate** the correctness of the entire reasoning process.

**Key Insight**: Model "methods" and "evaluations" using two three-stage frameworks, PAR and APE respectively, allowing methodological contributions and evaluation objectives to be mapped onto these coordinates for easy horizontal comparison and diagnosis.

**Core Idea**: The "PAR + APE" dual framework—PAR describes the three stages of processing multimodal input into a correct answer, while APE describes the levels of testing these stages. The intersection of both constitutes a unified map for MMR research.

## Method

### Overall Architecture

Rather than proposing a new method, this survey provides a map for the MMR field. PAR (Perception–Alignment–Reasoning) characterizes the three steps a method takes to process multimodal input into an answer. APE (Answer–Process–Executable) characterizes the levels at which evaluation tests these steps. These two main axes intersect to form a 2D coordinate system, onto which three task families (geometry, chart/table, visual word problems) and existing methods/benchmarks are mapped.

### Key Designs

**1. PAR Process Framework: Decomposing "Method" into Perception → Alignment → Reasoning**

PAR answers what steps a method must take. **Perception** extracts mathematical facts $\mathcal{F}$ from input $X \subseteq \{T, D, C, I\}$ (Text, Diagram, Chart, Image) across three levels: low-level primitives (points, lines), structural relationships (parallelism, row-column correspondence), and quantitative attributes (lengths, angles, values). **Alignment** maps these facts to symbolic or executable representations, such as geometry DSL, constraint sets, proof sketches, chart/table operators, SQL, or program-of-thought traces. **Reasoning** performs interpretable and verifiable derivation on the aligned representations via CoT, tree/graph of thought, RL, tool use, or process feedback. These stages are serially dependent: errors in perception propagate to alignment, and alignment errors pollute reasoning.

**2. APE Evaluation Framework: Decomposing "Evaluation" into Answer → Process → Executable**

APE is dual to PAR and identifies which stage of capability is being tested. **Answer-level** only checks final answer accuracy (exact match / numeric tolerance), which is simple but mixes all error sources, failing to distinguish correct guesses from correct logic. **Process-level** checks the validity of intermediate reasoning steps and visual grounding consistency (e.g., step types in MM-MATH, step judges in MPBench). **Executable-level** is most rigorous, directly running programs, verifying proofs, or checking constraints to assess the fidelity of alignment and reasoning (e.g., programs in GeoQA+, formal proofs in FormalGeo).

**3. Three Task Families: Describing Geometry, Chart/Table, and Visual MWP via PAR**

This survey uses PAR to provide a unified language for different tasks. **Geometry Problems** are formalized as $f: (T, D) \mapsto y$, requiring the identification of spatial relationships and grounding text to diagrams. Methods range from symbolic provers (GEOS) to hybrid pipelines (E-GPS) and LMMs (GeoGPT4V). **Chart and Table Problems** are formalized as $f: (C, Q) \mapsto a$, requiring axis/legend/table identification followed by numeric or logical reasoning. Methods evolve from symbolic parsing (PlotQA) to instruction-tuned LMMs (ChartLlama). **Visual Math Word Problems** are formalized as $f: (I, Q) \mapsto a$, involving object counting and attribute reasoning. Methods progress from symbolic perception (Patch-TRM) to LMM CoT.

**4. Alignment: Four Perspectives on "Bridging" Perception and Reasoning**

Alignment is the current bottleneck of MMR. The survey categorizes bridging methods into four types: **Executable intermediates** (Inter-GPS, R1-OneVision) convert visual content into DSL/programs for direct verification. **Symbolic-Neural Hybrids** (AlphaGeometry) pair neural perception with symbolic reasoning engines. **Cross-modal Alignment Frameworks** (BLIP-2, LLaVA) pursue stable vision-language coupling, often using curriculum designs. **Pre-training & Fine-tuning Enablers** (Geo170K, MathV360K) inject alignment capabilities via large-scale priors and task-specific supervision.

**5. Reasoning Paradigms: Four Approaches to "How to Reason"**

Reasoning methods are categorized into four paradigms: **Deliberate chains** use explicit thinking (CoT, ToT/GoT, VReST). **RL-based reasoning** is growing fastest, involving reward mechanisms (step-wise rewards in R1-VL, VisualPRM) and search algorithms (GRPO in DeepSeek-R1, MCTS in Mulberry). **Tool-augmented** (Toolformer, ToRA, Chameleon) outsource symbolic steps to solvers or code. **Process feedback & verification** (MM-PRM) use verifiers to score intermediate steps.

**6. Mapping APE Evaluation Levels to Specific Benchmarks**

The survey classifies benchmarks by APE levels. **Answer-level** benchmarks are most numerous (ChartQA, FinQA, IconQA). **Process-level** includes MM-MATH, MathVerse, and CHAMP. **Executable-level** includes GeoQA+, FormalGeo, and Inter-GPS. **Comprehensive** benchmarks span multiple levels, such as MathVista, MATH-V, and MM-PRM. The dominance of Answer-level benchmarks highlights a critical need for more Process and Executable-level evaluation.

## Key Experimental Results

### Benchmark Overview (Excerpt from Table 1, organized by APE & PAR)

| Benchmark | Year | Eval Level | PAR Stage | Key Contribution |
|-----------|------|------------|-----------|------------------|
| ChartQA | 2022 | Answer | P+R | Real charts + logic/numeric QA |
| FinQA | 2021 | Answer | A+R | Hybrid table/text + gold programs |
| MM-MATH | 2024 | Process | R | Step type + error label |
| MathVerse | 2024 | Process | All | Diagram perturbation + CoT step scoring |
| GeoQA+ | 2022 | Executable | A+R | Executable geometry programs |
| FormalGeo | 2024 | Executable | A+R | Olympiad-level formal proofs |
| MathVista | 2024 | Comprehensive | All | Comprehensive suite of 28 sub-collections |
| MM-PRM | 2025 | Comprehensive | All | Real K-12 multimodal QA |

### Dataset Scale (Excerpt from Table 2)

| Task Family | Representative Dataset | Scale | Key Features |
|-------------|-------------------------|-------|--------------|
| Geometry | Geometry3K | 3,002 | Dense formal language |
| Geometry | GeoQA / GeoQA+ | 5,010+ | Executable program supervised |
| Geometry | Geo170K | ~170K | Large-scale pre-training data |
| Chart/Table | ChartQA | 32.7K | Visual + logic QA |
| Chart/Table | FinQA | 8,281 | Hybrid table+text numeric |
| Visual MWP | IconQA | 107,439 | Diverse formats |
| Visual MWP | MathVista | 6,000+ | Merger of 28 suites |

### Key Findings
- Most benchmarks remain at the Answer-level. Process-level and Executable-level evaluations are underrepresented, meaning evaluations are "held hostage" by final answer accuracy, failing to expose intermediate reasoning errors.
- Geometry tasks have the highest proportion of executable benchmarks due to their natural support for formal proving; support in chart/table and visual MWP is relatively weak.
- RL-based reasoning paradigms are growing fastest (e.g., R1-VL, Vision-R1), with process reward models becoming a new hotspot.
- The lack of a unified DSL for alignment is the biggest bottleneck: geometry uses Inter-GPS DSL, charts use SQL, and MWP uses natural language.

## Highlights & Insights
- **PAR × APE is a contribution itself**: Moves beyond simple categorization to provide a 2D coordinate system for horizontal comparison and gap identification.
- **Evaluation-Method Alignment**: Explicitly links evaluation levels to PAR stages (e.g., Executable with Alignment), driving future benchmark design.
- **Failure Cause Attribution**: Emphasizes that perception errors propagate to alignment and reasoning, aiding in diagnostic analysis of MLLM failures.
- **Cross-task Normalization**: Provides a unified language for geometry, charts, and word problems, paving the way for cross-task universal modeling.
- **Pragmatic Future Directions**: Identifies unified DSLs, lightweight reward models, adaptive reasoning depth, and process reward + symbolic verifiers as key next steps.

## Limitations & Future Work
- The PAR/APE framework may have classification ambiguities in hybrid cases.
- Coverage of 2026 work is limited due to review cycles.
- Absolute numerical comparisons are limited as the survey relies on cited values rather than uniform reproduction.
- Efficiency dimensions (latency, VRAM) are not discussed in detail.
- Multilingual MMR (beyond CMM-Math) is insufficient.
- Multi-agent multimodal mathematical reasoning is not covered extensively.

## Related Work & Insights
- **vs Yan et al. (2024)**: That work focuses on MLLM roles and cataloging; Ours proposes the PAR/APE process framework, which is more systematic.
- **vs Ahn et al. (2024)**: Focused on text-only math (CoT/RL); Ours focuses on specific multimodal challenges like perception and alignment.
- **vs Li et al. (2025)**: Their "Perception-Reason-Think-Plan" is general; Ours is more compact (3 stages) and bound to math DSLs.
- **Insights for Idea Generation**: (1) Developing a unified DSL across task families is high-value; (2) Combining PRMs with symbolic verifiers is the primary direction for RL agents in math; (3) Process-level benchmarks are still scarce.

## Rating
- **Novelty**: ⭐⭐⭐⭐ New framework (PAR + APE) providing a process-view attribution tool.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 30+ benchmarks and 100+ references across three task families.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure and dense tables, though some sub-sections are slightly catalog-like.
- **Value**: ⭐⭐⭐⭐⭐ Essential map for new researchers; provides clear directions for evaluation and reward model design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)
- [\[ACL 2026\] Decoding Scientific Experimental Images: The SPUR Benchmark for Perception, Understanding, and Reasoning](decoding_scientific_experimental_images_the_spur_benchmark_for_perception_unders.md)
- [\[ACL 2026\] Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization](addressing_overthinking_in_large_vision-language_models_via_gated_perception-rea.md)
- [\[ACL 2025\] The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights](../../ACL2025/multimodal_vlm/the_role_of_visual_modality_in_multimodal_mathematical_reasoning_challenges_and_.md)
- [\[ACL 2026\] ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding](chemvlr_prioritizing_reasoning_in_perception_for_chemical_vision-language_unders.md)

</div>

<!-- RELATED:END -->
