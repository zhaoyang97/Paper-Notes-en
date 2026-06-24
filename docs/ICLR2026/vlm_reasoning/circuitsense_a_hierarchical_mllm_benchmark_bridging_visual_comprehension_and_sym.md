---
title: >-
  [Paper Note] CircuitSense: A Hierarchical MLLM Benchmark Bridging Visual Comprehension and Symbolic Reasoning in Engineering Design Process
description: >-
  [ICLR2026][VLM Reasoning][Circuit Understanding] CircuitSense establishes the first MLLM benchmark organized by engineering abstraction levels, emphasizing the derivation of symbolic equations from circuit schematics. Using 8,006 problems (human-curated + synthetically generated), it systematically evaluates 8 MLLMs, revealing a fundamental gap where closed-source models exceed 85% in perception tasks but plummet below 19% in symbolic derivation.
tags:
  - "ICLR2026"
  - "VLM Reasoning"
  - "Circuit Understanding"
  - "Visual-to-Symbolic Reasoning"
  - "MLLM Evaluation"
  - "Hierarchical Benchmark"
  - "Synthetic Data Generation"
date: 2026-05-08
content_hash: 0ea6eca3d66f7b66
---

# CircuitSense: A Hierarchical MLLM Benchmark Bridging Visual Comprehension and Symbolic Reasoning in Engineering Design Process

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=WEuJlEJmX8](https://openreview.net/forum?id=WEuJlEJmX8)  
**Code**: [Project Page](https://circuitsense-benchmark.github.io)  
**Area**: Multimodal VLM / Visual Reasoning / Datasets & Benchmarks  
**Keywords**: Circuit Understanding, Visual-to-Symbolic Reasoning, MLLM Evaluation, Hierarchical Benchmark, Synthetic Data Generation

## TL;DR
CircuitSense establishes the first MLLM benchmark organized by engineering abstraction levels, emphasizing the derivation of symbolic equations from circuit schematics. Using 8,006 problems (human-curated + synthetically generated), it systematically evaluates 8 MLLMs, revealing a fundamental gap where closed-source models exceed 85% in perception tasks but plummet below 19% in symbolic derivation.

## Background & Motivation
**Background**: The essence of engineering design is translating visual representations (circuit schematics, optical layouts, block diagrams) into precise mathematical models. Electrical engineers convert circuit diagrams into symbolic transfer functions to analyze noise, stability, and sensitivity. This "visual-to-math" translation capability determines engineering success. While MLLMs perform strongly on natural image tasks, a critical question remains: can they extract mathematical models from technical drawings?

**Limitations of Prior Work**: Existing visual circuit benchmarks (CIRCUIT, EEE-Bench, MMCircuitEval, AMSbench) focus almost entirely on shallow "recognition" tasks—identifying component types, answering basic multiple-choice questions, or performing simple numerical calculations. The core capability defining "circuit understanding"—extracting mathematical relationships from visual topology that are self-consistent across multiple system levels—has never been tested. Furthermore, no existing benchmark evaluates the hierarchical reasoning ability to switch between block diagrams and circuit schematics.

**Key Challenge**: Equation derivation serves as the watershed between "true engineering understanding" and "pattern matching." Without examining the symbolic derivation process, it is impossible to determine whether a model truly understands the circuit or merely remembers visual patterns. Consequently, one cannot judge if they can genuinely assist human designers in catching catastrophic failures (oscillation, instability, excessive noise) before expensive fabrication.

**Goal**: To build a benchmark capable of decoupling "visual-to-symbolic" abilities and evaluating them step-by-step according to engineering abstraction levels, specifically focusing on the overlooked capability of deriving symbolic equations from visual input.

**Key Insight**: The authors select analog circuits as the entry point because their design workflow (topology creation → device sizing → layout design) is naturally hierarchical, and catastrophic failures often only emerge at the end of verification. Thus, early mathematical analysis is highly valuable. Simultaneously, "synthetic generation + symbolic ground-truth" is introduced to fundamentally solve the data contamination issue.

**Core Idea**: Construct the benchmark using a "dual-axis (task category × six abstraction levels) + verifiable symbolic ground-truth synthetic generation pipeline." This decouples perception, analysis, and design capabilities, using **symbolic derivation accuracy** as the key metric for engineering proficiency.

## Method

### Overall Architecture
CircuitSense is not a model but a benchmark for evaluating visual circuit understanding, organizing 8,006 problems along two orthogonal axes. The first axis is **Task Category**, corresponding to three stages of the engineering workflow: Perception (890 problems)—counting components, identifying connections, classifying functions; Analysis (7,043 problems, the majority)—directly testing the extraction of mathematical models from visuals; and Design (157 problems)—mapping specifications back to circuit implementations. The second axis comprises **Six Abstraction Levels**, ranging from Level 0 Resistor Networks (1,777) to Level 5 System-level Block Diagrams (228). The intersection of these axes allows for precise localization of where the "visual-to-math translation" breaks down as complexity increases.

Data sources follow a dual-track: 2,986 problems are human-curated from authoritative textbooks (Gray, Razavi, Allen & Holberg, etc.) and university course repositories (Toronto ECE331, Georgia Tech ECE6412/ECE3050), ensuring breadth and pedagogical validity. 5,020 problems are produced by a **hierarchical synthetic generation pipeline**, each with verifiable symbolic ground truths, specifically addressing the issues of textbook contamination and the lack of systematic equation derivation testing. This synthetic pipeline includes a circuit schematic generator and a block diagram generator, covering component-level depth and system-level breadth, respectively.

### Key Designs

**1. Dual-Axis Hierarchical Task Organization: Testing Visual-to-Math Capabilities Layer-by-Layer**

Addressing the limitation that existing benchmarks only perform shallow recognition at a single level, CircuitSense places questions into a "3 Task Categories × 6 Abstraction Levels" grid. For Perception (Component Detection 200 / Connection Identification 200 / Function Classification 406), the benchmark verifies if the model possesses the foundational visual understanding required for subsequent analysis. The Analysis axis is subdivided into frequency response, transient response, transfer function analysis, small-signal analysis, CMR & PSRR, noise & jitter, and power & energy. Design tasks progress from the schematic level (63) to the block diagram level (56) and finally to hierarchical design (38) requiring cross-layer coordination. This 2D segmentation provides direct feedback: if a model scores >85% in perception but drops to single digits in symbolic derivation, the grid pinpoints the failure at the "visual parsing → symbolic reasoning" jump rather than a generic "model failure."

**2. Circuit Schematic Synthetic Generator: Guaranteed Symbolic Ground Truth via Grid Sampling + Lcapy**

To solve data contamination and provide verifiable equation answers, the authors extend the MAPS framework by synthesizing circuits on an $m \times n$ grid. Grid dimensions are sampled from discrete probability distributions to ensure topological diversity. 18 component types are supported (passive R/L/C, independent sources, four types of controlled sources VCVS/VCCS/CCVS/CCCS, and ideal op-amps). Op-amps are treated as template sub-circuits (input resistance + feedback network + high-gain VCVS) and placed randomly. Electrical validity is ensured via multiple constraints: eliminating dangling nodes, ensuring at least degree-2 connectivity for all nodes, and exactly one voltage source for reference. The topology is translated into a SPICE netlist and validated through three stages: topological check, SPICE simulation (DC operating point and AC response), and, most critically, symbolic analysis. Using Lcapy's Modified Nodal Analysis (MNA), the ground-truth transfer function $H(s)=V_{out}(s)/V_{in}(s)$ and nodal equations are extracted. Since answers are "computed" rather than "copied," models cannot rely on memorization, exposing their true derivation capabilities.

**3. Block Diagram Generator: Symbolic Transfer Functions via Mason’s Gain Formula**

Higher abstraction levels (Level 4-5) require system-level block diagrams. The authors designed a two-stage generator. First, a primary signal path is laid along a horizontal axis with $n \in [\tau_b, \tau_e]$ components (transfer function blocks + summation nodes). Then, $n_{fb} \in [0, \tau_{fb}]$ feedback loops and $n_{ff} \in [0, \tau_{ff}]$ feedforward paths are systematically added. The overall transfer function is solved symbolically using Mason’s Gain Formula by identifying all forward paths $P_k$, loops $L_i$, and non-touching loop combinations to calculate the system determinant:
$$\Delta = 1 - \sum L_i + \sum L_i L_j - \sum L_i L_j L_k + \dots$$
The final transfer function is:
$$H(s) = \frac{\sum P_k \Delta_k}{\Delta}$$
where $\Delta_k$ is the cofactor for path $P_k$. This method ensures verifiable symbolic ground truths for system-level problems, bridging component depth and system breadth.

**4. Symbolic Equivalence Evaluation Pipeline: Mathematical Equivalence Over String Matching**

Symbolic evaluation faces the challenge of algebraic equivalence (e.g., $H(s)=1/(RCs+1)$ vs. $H(s)=(1/RC)/(s+1/RC)$). String matching would lead to false negatives. The authors implement rigorous symbolic comparison using SymPy: predicted and ground-truth equations are parsed into expression trees, algebraically simplified, and verified via symbolic subtraction. If symbolic comparison is computationally infeasible, numerical verification is performed at 100 random complex frequency points. For open-ended questions, Gemini-2.5-Flash generates distractors to support multiple-choice evaluation, while open-ended numerical questions are judged by Gemini-2.5-Flash for mathematical equivalence (including unit conversion). Design tasks are validated via simulation using Ngspice with the Skywater 130nm PDK.

## Key Experimental Results

Tests were conducted on 8 SOTA MLLMs: Gemini-2.5-Pro, Claude-Sonnet-4, GPT-4o, GPT-4o-mini, InternVL3-78B, Qwen2.5-VL-72B, GLM-4.5V, and Gemma-3-27B.

### Main Results

The Gap between Perception vs. Analysis (excerpt from original Table 2 / Table 4):

| Model | Perception: Connection (%) | Perception: Function (%) | Analysis: Transfer Fn (%) | Analysis: Transient (%) |
|------|------|------|------|------|
| Gemini-2.5-Pro | 100 | 95 | 38 | 13 |
| Claude-Sonnet-4 | 88 | 86 | 23 | 9 |
| GPT-4o | 70 | 95 | 16 | 6 |
| GLM-4.5V | 78 | 26 | 14 | 4 |
| InternVL3-78B | 76 | 12 | 8 | 3 |

Closed-source models generally achieve >85% accuracy in perception, proving that vision is not the primary bottleneck. However, performance drops below 19% for symbolic derivation, exposing a fundamental fault line between visual parsing and symbolic reasoning.

For Design tasks: All models perform significantly better at block-diagram level design (30.91–67.27%) than schematic level design (7.01–36.38%), indicating that models are better at manipulating abstract functional blocks than implementing specifications with discrete components.

### Ablation Study

Systematic degradation from Curated (with options) vs. Synthetic (requires derivation) (excerpt from Table 5 / Table 6):

| Configuration | Gemini-2.5-Pro | Claude-Sonnet-4 | Note |
|------|------|------|------|
| Curated - Multiple Choice (%) | 80.71 | 69.67 | Allows elimination |
| Curated - Open-ended (%) | 70.32 | 34.76 | Removed scaffold |
| Synthetic - Symbolic Derivation (%) | 19.06 | 6.29 | No options/memory |

Gemini-2.5-Pro drops 61 percentage points from multiple-choice to synthetic derivation. This monotonic collapse confirms that most models rely on option elimination and pattern matching rather than genuine mathematical reasoning.

### Key Findings
- **Inverted Difficulty Paradox**: Models perform better on "traditionally harder" tasks like Noise & Jitter (up to 90%) than "basic" Transient Response (3-13%), as synthetic problems are concentrated in the latter, revealing that models memorize textbook solutions rather than understanding the underlying math.
- **Failure Point Localization**: Breaking 100 transfer function derivations into 6 sub-steps revealed that while total impedance calculation was 55% correct, **output impedance derivation was only 8%**. This 47% drop is the primary bottleneck. 
- **Capability Specialization**: Different models excel at different abstraction levels (e.g., Claude excels at Level 3 transistors while Gemini excels at Level 4 block diagrams), suggesting specialized rather than uniform understanding.

## Highlights & Insights
- **Operationalizing "Engineering Understanding"**: The authors argue and verify that "symbolic equation derivation" is the prerequisite and bottleneck for engineering capability.
- **Solving Data Contamination via Symbolic Truth**: Using Lcapy (MNA) and Mason’s Gain Formula to compute verifiable equations forces models to move beyond memorization.
- **Fine-grained Failure Diagnosis**: Dissecting derivations into sub-steps to pinpoint where reasoning fails (e.g., output impedance) provides a diagnostic paradigm that can be migrated to other multi-step reasoning tasks.

## Limitations & Future Work
- **Domain Focus**: The weight is on analog/electronic circuits. Whether these "visual-to-math" conclusions generalize to optical or mechanical engineering requires further validation.
- **Synthetic Distribution Skew**: Synthetic problems are concentrated in specific categories like transient response, which creates the "inverted difficulty" observed in cross-category comparisons.
- **Reliance on LLM Judges**: Using Gemini-2.5-Flash for distractor generation and open-ended scoring may introduce biases or caps based on the evaluator model’s own limits.
- **Future Directions**: Extending the pipeline to other engineering drawings and designing specialized training data or reasoning scaffolds to address specifically identified bottlenecks like impedance derivation.

## Related Work & Insights
- **vs. Visual Math Benchmarks (MathVista, ScienceQA)**: These often rely on knowledge-centric pattern recognition and ignore the "visual translation to formal symbolic equations" core to engineering.
- **vs. Previous Circuit Benchmarks (CIRCUIT, EEE-Bench, MMCircuitEval)**: Prior works either cover single abstraction levels or focus on conceptual multiple-choice questions without hierarchical organization or symbolic derivation testing.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First benchmark organized by engineering hierarchy emphasizing visual-to-symbolic derivation.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 8 SOTA models across 3 tasks and 6 levels with fine-grained failure analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear logic, though the inverted difficulty results require careful reading to avoid misinterpretation.
- Value: ⭐⭐⭐⭐⭐ Powerfully diagnoses the "strong perception, weak symbolic reasoning" gap in MLLMs for engineering.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Hierarchical Process Reward Models are Symbolic Vision Learners](../../CVPR2026/vlm_reasoning/hierarchical_process_reward_models_are_symbolic_vision_learners.md)
- [\[ICLR 2026\] FlowGen: Synthesizing Diverse Flowcharts to Enhance and Benchmark MLLM Reasoning](flowgen_synthesizing_diverse_flowcharts_to_enhance_and_benchmark_mllm_reasoning.md)
- [\[ICLR 2026\] Ref-Adv: Exploring MLLM Visual Reasoning in Referring Expression Tasks](ref-adv_exploring_mllm_visual_reasoning_in_referring_expression_tasks.md)
- [\[ICLR 2026\] VisualPRM400K: An Effective Dataset for Training Multimodal Process Reward Models](visualprm400k_an_effective_dataset_for_training_multimodal_process_reward_models.md)
- [\[ICLR 2026\] MMR-V: What's Left Unsaid? A Benchmark for Multimodal Deep Reasoning in Videos](mmr-v_whats_left_unsaid_a_benchmark_for_multimodal_deep_reasoning_in_videos.md)

</div>

<!-- RELATED:END -->
