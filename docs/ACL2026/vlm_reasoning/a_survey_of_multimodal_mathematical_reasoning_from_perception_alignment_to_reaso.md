---
title: >-
  [Paper Note] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning
description: >-
  [ACL 2026][vlm_reasoning][executable intermediate] This survey proposes a complementary perspective consisting of the Perception–Alignment–Reasoning (PAR) process framework and the Answer–Process–Executable (APE) evaluation framework. It systematically organizes three task families—geometry, chart/table, and visual word problems—mapping existing methods and benchmarks
tags:
  - ACL 2026
  - vlm_reasoning
  - executable intermediate
date: 2026-05-08
content_hash: a95b301e4576e2ab
---
# A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning

**Conference**: ACL 2026  
**arXiv**: [2603.08291](https://arxiv.org/abs/2603.08291)  
**Code**: Awesome Multimodal Mathematical Reasoning (GitHub repository, link provided in the paper)  
**Area**: Multimodal VLM / Multimodal Mathematical Reasoning / Survey  
**Keywords**: PAR framework, APE evaluation, geometric reasoning, chart/table reasoning, executable intermediate

## TL;DR
This survey proposes a complementary perspective consisting of the Perception–Alignment–Reasoning (PAR) process framework and the Answer–Process–Executable (APE) evaluation framework. It systematically organizes three task families—geometry, chart/table, and visual word problems—mapping existing methods and benchmarks onto these two coordinate axes. It represents the first process-centric survey on multimodal mathematical reasoning.

## Background & Motivation
**Background**: LLMs have approached SOTA in symbolic and arithmetic reasoning, but real-world mathematical problems are often multimodal (containing diagrams, tables, geometric figures, coordinate plots, and mixed documents). While a large number of Multimodal Mathematical Reasoning (MMR) datasets and methods have emerged, a unified perspective linking "perception, alignment, reasoning, and evaluation" is still missing.

**Limitations of Prior Work**: (1) Previous MMR surveys (e.g., Yan et al. 2024) mostly focused on benchmark cataloging or MLLM role classification (Reasoner / Enhancer / Planner), which is relatively horizontal. (2) Most evaluations only consider the final answer, failing to distinguish between "correct guesses," "shortcuts," and "correct reasoning." (3) Methods utilize various DSLs, alignment strategies, and reasoning paradigms, making horizontal comparisons difficult.

**Key Challenge**: MMR differs fundamentally from pure text mathematical reasoning. Multimodal coupling allows errors in perception, alignment, and reasoning to propagate across stages; a single metric cannot pinpoint the specific failure point. A process-centric perspective is required to diagnose exactly where the failure occurs.

**Goal**: Organize MMR around four fundamental questions: (1) **what to extract** from multimodal inputs; (2) how to **represent and align** textual and visual information; (3) how to **perform reasoning** (CoT / program-aided / tool use); (4) how to **evaluate** the correctness of the entire reasoning process.

**Key Insight**: Model total "methods" and "evaluations" using two three-stage frameworks, PAR and APE respectively, allowing methodological contributions and evaluation objectives to be mapped onto these coordinates for easier horizontal comparison and diagnosis.

**Core Idea**: The "PAR + APE" dual framework. PAR describes the three stages a method undergoes to process multimodal input into a correct answer, while APE describes the levels of examination an evaluation performs on these stages. Their intersection forms a unified map for MMR research.

## Method

### Overall Architecture

Rather than proposing a new method, this survey maps the entire Multimodal Mathematical Reasoning (MMR) field. It uses PAR (Perception–Alignment–Reasoning) to characterize the three steps a method takes to process multimodal input into an answer, and APE (Answer–Process–Executable) to categorize the levels of evaluation. These two main axes form a two-dimensional coordinate system onto which three major task families (geometry, chart/table, and visual word problems), as well as existing methods and benchmarks, are mapped.

### Key Designs

**1. PAR Process Framework: Decomposing "Method" into Perception → Alignment → Reasoning**

PAR answers the question: "What steps must a method take to transform multimodal input into a correct answer?" **Perception** is responsible for extracting mathematical facts $\mathcal{F}$ from inputs $X \subseteq \{T, D, C, I\}$ (Text/Diagram/Chart/Image), deepening through three layers: low-level primitives (points, lines, axes, objects) → structural relations (incidence, parallelism, row-column correspondence) → quantitative attributes (lengths, angles, values, units). **Alignment** maps these facts to symbolic or executable representations, such as geometry DSLs, constraint sets, proof sketches, chart/table operators, SQL, or program-of-thought traces. **Reasoning** performs interpretable and verifiable derivation on the aligned representations using means like CoT, tree/graph of thought, RL, tool use, or process feedback. These stages are serially dependent: perception errors propagate to alignment, and alignment errors contaminate reasoning. Thus, diagnosing an MLLM failure requires locating the specific stage in the PAR process.

**2. APE Evaluation Framework: Decomposing "Evaluation" into Answer → Process → Executable**

APE is dual to PAR, answering: "Which stage of capability is the evaluation actually testing?" **Answer-level** only considers final answer accuracy (exact match / numeric tolerance), which is simple to implement but conflates all error sources, failing to distinguish correct reasoning from shortcuts or guessing. **Process-level** checks the validity of intermediate reasoning steps and visual grounding consistency (e.g., step types in MM-MATH, step judging in MPBench, and diagram perturbation scoring in MathVerse). **Executable-level** is the most rigorous, evaluating the faithfulness of alignment and reasoning by directly running programs, verifying proofs, or checking constraints (e.g., GeoQA+ programs, FormalGeo formal proofs, E-GPS solvers). The survey explicitly links Process-level to the Reasoning stage and Executable-level to the Alignment stage.

**3. Three Task Families: Describing Geometry, Charts/Tables, and Visual Word Problems via PAR**

Traditional surveys treat these tasks separately; this paper uses PAR to provide a unified language—what perception extracts, what DSL alignment uses, and whether reasoning follows CoT or tools. **Geometry Problems** are formalized as $f: (T, D) \mapsto y$, requiring the identification of points/lines/angles and spatial relations. Methods range from symbolic provers (GEOS) to hybrid pipelines (E-GPS) and LMMs (GeoGPT4V). **Chart and Table Problems** are formalized as $f: (C, Q) \mapsto a$, requiring the identification of axes/legends/rows/columns followed by numeric or logical reasoning. Methods range from symbolic parsing (PlotQA) to instruction-tuned LMMs (ChartQA-X). **Visual Math Word Problems** are formalized as $f: (I, Q) \mapsto a$, involving object counting and attribute reasoning. Methods range from symbolic perception (Patch-TRM) to LMM CoT.

**4. Alignment Perspectives: Four Routes to "Bridging" Perception and Reasoning**

Alignment is the current primary bottleneck in MMR due to the lack of a unified DSL. The survey categorizes bridging methods into four types: **Executable intermediates** (Inter-GPS, R1-OneVision) convert visual content into DSL/programs for direct verification. **Symbolic-Neural Hybrids** (AlphaGeometry) pair neural perception with symbolic reasoning engines. **Cross-modal Alignment Frameworks** (BLIP-2, LLaVA, Math-PUMA) pursue stable vision-language coupling through progressive or curriculum designs. **Pre-training & Fine-tuning Enablers** (Geo170K, MAmmoTH-VL) use large-scale alignment priors and task-specific supervision to inject alignment capabilities via data.

**5. Reasoning Paradigms: Four Approaches to "How to Reason" Post-Alignment**

Reasoning methods are categorized into four paradigms: **Deliberate chains** use explicit thinking chains (CoT, ToT/GoT, VisuoThink, VReST). **RL-based reasoning** is the fastest-growing area, including reward mechanisms (R1-VL step-wise rewards, VisualPRM, MM-Eureka rule-based RL) and search algorithms (DeepSeek-R1 GRPO, Vision-R1, VL-Rethinker, AlphaProof formal RL). **Tool-augmented** (Toolformer, ToRA, Chameleon) offloads symbolic steps to solvers or code. **Process feedback & verification** (VisualPRM, MM-PRM) utilizes PRMs or verifiers to score intermediate steps.

**6. Mapping APE Evaluation Levels to Specific Benchmarks**

Mapping the APE levels to benchmarks reveals what evaluations are currently "captured" by. **Answer-level** benchmarks are the most numerous (ChartQA, FinQA, IconQA). **Process-level** includes MM-MATH, MathVerse, and CHAMP. **Executable-level** includes GeoQA+, FormalGeo, and E-GPS. A **Comprehensive** category spans multiple levels (MathVista, MATH-V, MM-PRM). The survey warns that the vast majority are Answer-level, with insufficient Process and Executable benchmarks.

## Key Experimental Results

### Benchmark Panorama (Excerpt from Table 1, organized by APE dimension + PAR stage)

| Benchmark | Year (Venue) | Eval Level | PAR Stage | Key Contribution |
|-----------|----------|------------|-----------|----------|
| ChartQA | 2022 (ACL Find.) | Answer | P+R | Real charts + logic/numeric QA |
| FinQA | 2021 (EMNLP) | Answer | A+R | Hybrid table/text + gold programs |
| MM-MATH | 2024 (EMNLP Find.) | Process | R | Step type + error label |
| MathVerse | 2024 (ECCV) | Process | All | Diagram perturbation + CoT step scoring |
| GeoQA+ | 2022 (COLING) | Executable | A+R | Executable geometry programs |
| FormalGeo | 2024 (MATH-AI) | Executable | A+R | Olympiad-level formal proofs |
| MathVista | 2024 (ICLR) | Comprehensive | All | Comprehensive suite of 28 sub-collections |
| MATH-V | 2024 (NeurIPS) | Comprehensive | All | Difficulty-calibrated visual math |
| MM-PRM | 2025 (arXiv) | Comprehensive | All | Real K-12 multimodal QA |

### Dataset Scale (Excerpt from Table 2)

| Task Family | Representative Dataset | Scale | Key Features |
|--------|------------|------|----------|
| Geometry | Geometry3K | 3,002 problems | Dense formal language |
| Geometry | GeoQA / GeoQA+ | 5,010+ | Executable program supervised |
| Geometry | Geo170K | ~170K img-cap + QA | Large-scale geometry pre-training |
| Chart/Table | ChartQA | 9.6K human + 23.1K gen | Visual + logical QA |
| Chart/Table | FinQA | 8,281 | Hybrid table + text numeric |
| Chart/Table | DocMath-Eval | 4,000 | Includes gold programs |
| Visual MWP | IconQA | 107,439 | Multiple formats |
| Visual MWP | MV-MATH | 2,009 multi-image | Cross-image dependency reasoning |
| Visual MWP | MathVista | 6,000+ | Merger of 28 suites |

### Key Findings
- Most benchmarks remain at the Answer-level; the low proportion of Process-level and Executable-level evaluation means failures are "kidnapped" by final accuracy without exposing intermediate errors.
- Geometry tasks have the highest executable ratio (formal geometry naturally supports prove/check), while chart/table and visual MWP have relatively weak executable support.
- RL-based reasoning is growing fastest (with over ten papers like R1-VL and Vision-R1 in 2024–2025), and process reward models have become a new hotspot.
- The lack of a unified DSL for alignment is the biggest current bottleneck: geometry uses Inter-GPS DSL, charts use SQL/PoT, and word problems use natural language.

## Highlights & Insights
- **The PAR × APE dual framework is a contribution in itself**: While many surveys only provide categorization, this work builds a "process × level" two-dimensional coordinate system to facilitate horizontal comparison and identify research gaps.
- **Focus on aligning evaluation with method stages**: By explicitly linking Process-level evaluation to Reasoning and Executable-level evaluation to Alignment, it emphasizes that "evaluation should test the capability of its intended stage," which can drive future benchmark design.
- **Failure cause attribution perspective**: The survey repeatedly emphasizes how errors propagate through the PAR stages, reminding researchers to perform attribution rather than generalized conclusions about reasoning capability.
- **Task family normalization**: Geometry, charts, and word problems are discussed using the same PAR language, paving the way for cross-task unified modeling.
- **Pragmatic future directions**: The authors identify unified DSLs, lightweight reward models, adaptive reasoning depth, and the combination of process rewards with symbolic verifiers as key next steps.

## Limitations & Future Work
- As a survey, classification in the PAR/APE framework can be ambiguous for hybrid cases; finer sub-categorization may be needed.
- Benchmark coverage is capped at 2025 (NeurIPS/arXiv), missing some papers currently in review for 2026.
- Most experimental figures rely on citations from original papers rather than centralized reproduction, limiting the absolute comparability of numerical values.
- Efficiency dimensions (latency, VRAM) are not discussed in detail; engineering deployment of MMR requires considering costs alongside accuracy.
- Multi-language MMR support is insufficient; global math education scenarios require more diverse evaluations.
- Multi-agent multimodal mathematical reasoning (e.g., multi-agent geometry provers) is not covered extensively.

## Related Work & Insights
- **vs. Yan et al. (2024) MMR survey**: That work focuses on MLLM roles and benchmark cataloging; this paper proposes a more systematic PAR/APE process framework.
- **vs. Ahn et al. (2024) Text-only math reasoning survey**: Text-based surveys focus on CoT/RL/verification; this paper focuses on multimodal-specific challenges (perception, cross-modal alignment).
- **vs. Li et al. (2025) Perception-Reason-Think-Plan**: The latter splits multimodal reasoning into 4 actions; this paper's PAR is more compact and bound to mathematical DSLs, making it more suitable for the math sub-domain.
- **vs. Lu et al. (2023d) Deep learning for math reasoning survey**: An earlier survey covering mostly pre-2022 literature; this paper covers the 2023–2025 explosion of LMM/RL/tool-use.
- **Inspiration for idea generation**: (1) A unified DSL across task families is a high-value open problem; (2) Hybrid rewards (Process Reward Model + Symbolic Verifier) are a primary direction for RL agents; (3) The scarcity of process-level benchmarks represents a significant opportunity for dataset contribution.

## Rating
- Novelty: ⭐⭐⭐⭐ New framework (PAR + APE), provides a process-based attribution tool rather than just a literature catalog.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 30+ benchmarks and 100+ references, expanding on three task families, four alignment types, and four reasoning types.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with dense tables and consistent PAR/APE segmentation.
- Value: ⭐⭐⭐⭐⭐ A must-read map for researchers entering the MMR field; provides clear directions for evaluation, reward model design, and unified modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)
- [\[ACL 2025\] The Role of Visual Modality in Multimodal Mathematical Reasoning: Challenges and Insights](../../ACL2025/vlm_reasoning/the_role_of_visual_modality_in_multimodal_mathematical_reasoning_challenges_and_.md)
- [\[ACL 2025\] MathCoder-VL: Bridging Vision and Code for Enhanced Multimodal Mathematical Reasoning](../../ACL2025/vlm_reasoning/mathcoder-vl_bridging_vision_and_code_for_enhanced_multimodal_mathematical_reaso.md)
- [\[ACL 2026\] Decoding Scientific Experimental Images: The SPUR Benchmark for Perception, Understanding, and Reasoning](decoding_scientific_experimental_images_the_spur_benchmark_for_perception_unders.md)
- [\[ACL 2026\] Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization](addressing_overthinking_in_large_vision-language_models_via_gated_perception-rea.md)

</div>

<!-- RELATED:END -->
