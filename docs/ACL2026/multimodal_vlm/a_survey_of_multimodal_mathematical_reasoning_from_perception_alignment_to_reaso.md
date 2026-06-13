---
title: >-
  [Paper Note] A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning
description: >-
  [ACL 2026][Multimodal VLM][PAR Framework] This survey proposes two complementary perspectives: the Perception–Alignment–Reasoning (PAR) process framework and the Answer–Process–Executable (APE) evaluation framework. It s…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "PAR Framework"
  - "APE Evaluation"
  - "Geometric Reasoning"
  - "Chart/Table Reasoning"
  - "executable intermediate"
date: 2026-05-08
content_hash: d276418c9d39c30a
---

# A Survey of Multimodal Mathematical Reasoning: From Perception, Alignment to Reasoning

**Conference**: ACL 2026  
**arXiv**: [2603.08291](https://arxiv.org/abs/2603.08291)  
**Code**: Awesome Multimodal Mathematical Reasoning（GitHub repository, link provided in the paper）  
**Area**: Multimodal VLM / Multimodal Mathematical Reasoning / Survey  
**Keywords**: PAR Framework, APE Evaluation, Geometric Reasoning, Chart/Table Reasoning, executable intermediate

## TL;DR
This survey proposes two complementary perspectives: the Perception–Alignment–Reasoning (PAR) process framework and the Answer–Process–Executable (APE) evaluation framework. It systematically organizes three major task families—geometry, chart/table, and visual math word problems—mapping existing methods and benchmarks onto these two coordinate axes, making it the first process-centric survey for multimodal mathematical reasoning.

## Background & Motivation
**Background**: LLMs have approached SOTA in symbolic and arithmetic reasoning, but real-world mathematical problems are often multimodal (diagrams, tables, geometric figures, coordinate plots, mixed documents). A large number of datasets and methods for Multimodal Mathematical Reasoning (MMR) have emerged, but there is a lack of a unified perspective to connect "perception / alignment / reasoning / evaluation."

**Limitations of Prior Work**: (1) Previous MMR surveys (e.g., Yan et al. 2024) mostly focus on benchmark cataloging or MLLM role classification (Reasoner / Enhancer / Planner), which is relatively horizontal; (2) Most evaluations only consider the final answer, failing to distinguish between "guessing correctly, relying on shortcuts, or reasoning correctly"; (3) Different methods use various DSLs, alignment methods, and reasoning paradigms, making horizontal comparison difficult.

**Key Challenge**: MMR is fundamentally different from text-only mathematical reasoning. Multimodal coupling allows perception errors, alignment errors, and reasoning errors to propagate through layers. Single metrics cannot locate the point of failure. A process-centric perspective is required to diagnose "at which step the failure occurred."

**Goal**: Organize MMR around four fundamental questions: (1) **what to extract** from multimodal inputs; (2) how to **represent and align** textual/visual information; (3) how to **perform reasoning** (CoT / program-aided / tool use); (4) how to **evaluate** the correctness of the entire reasoning process.

**Key Insight**: Modeling "methods" and "evaluation" using two three-stage frameworks, PAR and APE respectively, allows methodological contributions and evaluation objectives to be mapped onto these coordinates for easier horizontal comparison and diagnosis.

**Core Idea**: The twin "PAR + APE" frameworks—PAR describes the three stages of how methods process multimodal input into correct answers, while APE describes the verification of these stages at different levels. Their intersection constitutes a unified roadmap for MMR research.

## Method

### Survey structure and Taxonomy

#### Primary Framework: PAR (Methodology)
- **Perception**: Extract mathematical facts $\mathcal{F}$ from $X \subseteq \{T, D, C, I\}$ (text/diagram/chart/image), consisting of three levels: low-level primitives (points, lines, axes, objects) $\rightarrow$ structural relations (incidence, parallelism, row-column) $\rightarrow$ quantitative attributes (length, angle, numerical value, unit).
- **Alignment**: Map perceived facts to symbolic or executable representations (geometry DSL, constraint sets, proof sketches, chart/table operators, SQL, program-of-thought traces).
- **Reasoning**: Perform interpretable/verifiable reasoning on aligned representations (CoT, tree/graph of thought, RL, tool use, process feedback).

#### Primary Framework: APE (Evaluation)
- **Answer**: Checks only the final answer accuracy (exact match / numeric tolerance)—scalable but obscures error sources.
- **Process**: Checks the validity of intermediate reasoning steps and visual grounding consistency (e.g., MM-MATH step types, MPBench step judge, CHAMP concept labels, MathVerse diagram perturbation scores).
- **Executable**: Directly evaluates the faithfulness of alignment and reasoning by running programs, verifying proofs, or checking constraints (e.g., GeoQA+ programs, FormalGeo formal proofs, E-GPS solvers, WikiSQL execution).

#### Secondary Taxonomy: Three Task Families (Each sliced by PAR)
1. **Geometry Problems**: $f: (T, D) \mapsto y$, requiring identification of points, lines, angles, spatial relationships, and grounding text to geometric figures. Methods evolve from symbolic provers (GEOS) $\rightarrow$ neural VLMs $\rightarrow$ hybrid pipelines (E-GPS / Pi-GPS) $\rightarrow$ LMMs (G-LLaVA / GeoGPT4V / GEOX). Benchmarks: Geometry3K, GeoQA/+, PGDP5K, PGPS9K, FormalGeo7K.
2. **Chart and Table Problems**: $f: (C, Q) \mapsto a$, requiring identification of axes, legends, rows, and columns for numeric/logic reasoning. Methods evolve from symbolic parsing (DVQA, PlotQA) $\rightarrow$ neural VLMs (Pix2Struct) $\rightarrow$ instruction-tuned LMMs (ChartLlama, ChartQA-X). Benchmarks: PlotQA, ChartQA(Pro), CharXiv, FinQA, TAT-QA, MultiHiertt, DocMath-Eval, WikiSQL.
3. **Visual Math Word Problems**: $f: (I, Q) \mapsto a$, involving object counting, attribute reasoning, and cross-image coreference. Methods evolve from symbolic perception (Patch-TRM) $\rightarrow$ neural multimodal $\rightarrow$ LMM CoT. Benchmarks: IconQA, CLEVR-Math, TABMWP, MV-MATH, MathVista, MATH-V, Math2Visual.

#### Secondary Taxonomy: Four Alignment Perspectives
1. **Executable intermediates** (Inter-GPS, E-GPS, Pi-GPS, R1-OneVision): Convert visual content into DSL / programs / SQL for executable verification.
2. **Symbolic-Neural Hybrids** (GeoGen, MathCoder-VL, AlphaGeometry): Neural perception combined with symbolic reasoning engines.
3. **Cross-modal Alignment Frameworks** (BLIP-2, LLaVA, Math-PUMA, VCAR, TVC, VIC): Stable vision-language coupling, including progressive/curriculum designs.
4. **Pre-training & Fine-tuning Enablers** (Geo170K, SynthGeo228K, Math-LLaVA, MAVIS, MultiMath-300K, MAmmoTH-VL, MathV360K): Large-scale alignment priors + task-specific supervision.

#### Secondary Taxonomy: Four Reasoning Paradigms
1. **Deliberate chains**: CoT (LLaVA-CoT), TVC continuous visual conditioning, VIC text-first planning, AtomThink atomic decomposition, advancing to ToT / GoT / AGoT, VisuoThink, VReST (MCTS + self-reward).
2. **RL-based reasoning**: Reward mechanisms (R1-VL step-wise reward, VisualPRM, MM-PRM + MCTS, MM-Eureka rule-based RL) + search algorithms (DeepSeek-R1 GRPO, Vision-R1, Mulberry MCTS, Skywork R1V2 MPO+GRPO, VL-Rethinker, FAST, Think-or-Not?, VLAA-Thinking, VLM-R3, MAYE, SoTA-with-Less, AlphaProof formal RL).
3. **Tool-augmented** (Toolformer, ToRA, COPRA, MM-REACT, Visual Sketchpad, Pi-GPS, Chameleon, MathCoder-VL): Outsource symbolic steps to solvers/code.
4. **Process feedback & verification** (VisualPRM, MM-PRM, TVC continuous visual, VIC late fusion): Score intermediate steps using PRMs / verifiers.

#### Secondary Taxonomy: APE Evaluation
- **Answer-level**: ChartQA, PlotQA, FigureQA, IconQA, CLEVR-Math, FinQA, TAT-QA.
- **Process-level**: MM-MATH, MPBench, ErrorRadar, Sherlock, We-Math, MathVerse, CHAMP, PolyMATH.
- **Executable-level**: GeoQA+, FormalGeo, Inter-GPS, E-GPS, Pi-GPS.
- **Comprehensive**: MathVista, MATH-V, OlympiadBench, MathScape, CMM-Math, Children's Olympiads, MM-PRM.

## Key Experimental Results

### Benchmark Panorama (Excerpt from Table 1, organized by APE levels + PAR stages)

| Benchmark | Year (Venue) | Eval Level | PAR Stage | Key Contributions |
|-----------|----------|------------|-----------|----------|
| ChartQA | 2022 (ACL Findings) | Answer | P+R | Real charts + logic/numeric QA |
| FinQA | 2021 (EMNLP) | Answer | A+R | Hybrid table/text + gold programs |
| MM-MATH | 2024 (EMNLP Findings) | Process | R | Step type + error label |
| MathVerse | 2024 (ECCV) | Process | All | Diagram perturbation + CoT step scoring |
| GeoQA+ | 2022 (COLING) | Executable | A+R | Executable geometry programs |
| FormalGeo | 2024 (MATH-AI) | Executable | A+R | Olympiad-level formal proofs |
| MathVista | 2024 (ICLR) | Comprehensive | All | Suite of 28 merged sub-collections |
| MATH-V | 2024 (NeurIPS) | Comprehensive | All | Difficulty-calibrated visual math |
| MM-PRM | 2025 (arXiv) | Comprehensive | All | Real K-12 multimodal QA |

### Dataset Scales (Excerpt from Table 2)

| Task Family | Representative Datasets | Scale | Key Features |
|--------|------------|------|----------|
| Geometry | Geometry3K | 3,002 problems | Dense formal language |
| Geometry | GeoQA / GeoQA+ | 5,010+ | Executable program supervised |
| Geometry | Geo170K | ~170K image-caption + QA | Large-scale geometry pre-train |
| Chart/Table | ChartQA | 9.6K human + 23.1K synth | Visual + logical QA |
| Chart/Table | FinQA | 8,281 | Hybrid table + text numeric |
| Chart/Table | DocMath-Eval | 4,000 | Includes gold programs |
| Visual MWP | IconQA | 107,439 | Multi-format |
| Visual MWP | MV-MATH | 2,009 multi-image | Cross-image dependency reasoning |
| Visual MWP | MathVista | 6,000+ | 28 suites merged |

### Key Findings
- Most benchmarks still remain at the Answer-level, with Process-level and Executable-level accounting for a low percentage—evaluation is being hijacked by "final answer accuracy," failing to expose intermediate reasoning errors.
- Geometry tasks have the highest proportion of executable components (formal geometry naturally supports prove/check), while executable support for chart/table and visual MWP is relatively weak.
- In reasoning paradigms, RL-based methods are growing fastest (over ten papers between 2024–2025 including R1-VL, VisualPRM, MM-PRM, Vision-R1, Mulberry), and process reward models have become a new hotspot.
- The lack of a unified DSL for Alignment is currently the biggest bottleneck: geometry uses Inter-GPS DSL, chart uses SQL/PoT, and word problems use natural language, with no cross-task shared alignment foundation.

## Highlights & Insights
- **The PAR × APE dual framework is a contribution of the survey itself**: Unlike many surveys that only perform classification, this work constructs a two-dimensional coordinate system of "method process × evaluation level," allowing every work to find its place for horizontal comparison and gap identification.
- **Using "how evaluation aligns with method stages" as the main thread**: Explicitly linking Process-level evaluation to the Reasoning stage and Executable evaluation to the Alignment stage emphasizes that "evaluation should test the capability of a specific stage," which directly drives future benchmark design.
- **Failure cause attribution perspective**: The survey repeatedly emphasizes that "Perception errors propagate to Alignment, and Alignment errors pollute Reasoning," reminding readers to attribute MLLM failures according to the three PAR stages rather than vaguely stating "the model cannot reason."
- **Cross-task normalization**: Geometry / chart / word problems are often discussed separately in traditional surveys; this paper describes them using the same language (what perception extracts, what DSL alignment uses, whether reasoning uses CoT or tools), paving the way for unified cross-task modeling.
- **Pragmatic future directions**: The authors explicitly point out that unified DSL, lightweight reward models, adaptive reasoning depth, and process reward + symbolic verifiers are the key technologies for the next step; education, accessibility, and AR-VR are potential applications.

## Limitations & Future Work
- As a survey, the PAR/APE framework may have blurry classifications in certain boundary cases (e.g., hybrid methods might span multiple alignment perspectives), requiring finer sub-classification in the future.
- Benchmark tracking is current up to 2025 NeurIPS / arXiv, with a few 2026 works under submission/review not covered.
- Most experimental figures rely on citations from original papers without unified reproduction, limiting the absolute numerical comparability across papers.
- Efficiency dimensions (inference latency, VRAM) are not discussed in detail; engineering deployment of MMR methods requires considering cost alongside accuracy, left for future surveys.
- Insufficient coverage of Chinese/multilingual MMR (only CMM-Math), whereas globalized mathematics education scenarios need more evaluation.
- **Multi-agent** multimodal mathematical reasoning (e.g., multi-agent geometry provers) is not covered, which is a new hotspot for 2025–2026.

## Related Work & Insights
- **vs Yan et al. (2024) MMR survey**: That work focuses on MLLM roles (Reasoner / Enhancer / Planner) and benchmark cataloging; this work proposes the PAR/APE process framework, which is more systematic.
- **vs Ahn et al. (2024) text-only math reasoning survey**: Textual math reasoning surveys focus on CoT / RL / verification; this work focuses on challenges unique to multimodality (perception, cross-modal alignment).
- **vs Li et al. (2025) Perception-Reason-Think-Plan**: The latter splits multimodal reasoning into 4 actions; this paper's PAR is more compact (3 stages) and bound to specific math DSLs, making it more suitable for the math sub-domain.
- **vs Lu et al. (2023d) survey of deep learning for math reasoning**: An earlier survey primarily covering literature before 2022; this work completes the 2023–2025 explosive period of LMM / RL / tool-use.
- **Insights for idea generation**: (1) Unified DSL across geometry/chart/word problems is a high-value open problem; (2) Hybrid rewards consisting of Process Reward Models + Symbolic Verifiers are the main developmental direction for RL agents in mathematics; (3) APE's process-level benchmarks are still scarce and can serve as contribution points for new datasets.

## Rating
- Novelty: ⭐⭐⭐⭐ New framework (PAR + APE dual coordinates), not just a literature catalog, providing an attribution tool from a process perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 30+ benchmarks and 100+ method citations, expanding on three task families, four types of alignment, and four types of reasoning.
- Writing Quality: ⭐⭐⭐⭐ Clear structure, dense tables, consistent PAR/APE segmentation; some subsections are slightly catalog-heavy and could be further refined.
- Value: ⭐⭐⭐⭐⭐ A must-read roadmap for researchers newly entering the MMR field; provides clear directions for evaluation design, reward model design, and unified cross-task modeling.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)
- [\[ACL 2026\] ChemVLR: Prioritizing Reasoning in Perception for Chemical Vision-Language Understanding](chemvlr_prioritizing_reasoning_in_perception_for_chemical_vision-language_unders.md)
- [\[ACL 2026\] Decoding Scientific Experimental Images: The SPUR Benchmark for Perception, Understanding, and Reasoning](decoding_scientific_experimental_images_the_spur_benchmark_for_perception_unders.md)
- [\[ACL 2026\] Addressing Overthinking in Large Vision-Language Models via Gated Perception-Reasoning Optimization](addressing_overthinking_in_large_vision-language_models_via_gated_perception-rea.md)
- [\[ACL 2026\] TRACE: Unleashing Spatial Reasoning in Multimodal Large Language Models via Textual Representation Guided Reasoning](unleashing_spatial_reasoning_in_multimodal_large_language_models_via_textual_rep.md)

</div>

<!-- RELATED:END -->
