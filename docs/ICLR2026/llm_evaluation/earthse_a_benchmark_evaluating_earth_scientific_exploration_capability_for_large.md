---
title: >-
  [Paper Note] EARTHSE: A Benchmark for Earth Science Knowledge Exploration
description: >-
  [ICLR 2026][LLM Evaluation][Earth Science Benchmark] EARTHSE constructs a three-layer progressive benchmark (Breadth QA → Professional QA → Open Conversation) from 100,000 Earth science papers. It covers 5 major Earth spheres, 114 subfields, and 11 task categories. The benchmark systematically evaluates LLMs across basic knowledge and scientific exploration dimensions, revealing significant shortcomings of existing LLMs in domain depth and open-ended scientific thinking.
tags:
  - "ICLR 2026"
  - "LLM Evaluation"
  - "Earth Science Benchmark"
  - "Scientific Exploration"
  - "Multi-task"
  - "Open-ended Dialogue"
date: 2026-05-08
content_hash: df0bed62836e60cb
---

# EARTHSE: A Benchmark for Earth Science Knowledge Exploration

**Conference**: ICLR 2026  
**Paper**: [OpenReview](https://openreview.net/forum?id=jyYE06FL8G)  
**Code**: https://huggingface.co/ai-earth  
**Area**: LLM Evaluation  
**Keywords**: Earth Science Benchmark, LLM Evaluation, Scientific Exploration, Multi-task, Open-ended Dialogue

## TL;DR

EARTHSE constructs a three-layer progressive benchmark (Breadth QA → Professional QA → Open Conversation) from 100,000 Earth science papers. It covers 5 major Earth spheres, 114 subfields, and 11 task categories. The benchmark systematically evaluates LLMs across basic knowledge and scientific exploration dimensions, revealing significant shortcomings of existing LLMs in domain depth and open-ended scientific thinking.

## Background & Motivation

**Background**: Although LLMs perform well on general scientific QA (ScienceQA, MMLU, etc.), these benchmarks either lack sufficient depth (failing to comprehensively cover the 114 subfields across Earth's five major spheres) or focus on single subfields (e.g., climate, oceanography), failing to provide a holistic assessment of Earth science as a cross-disciplinary, multi-scale complex system.

**Limitations of Prior Work**: (1) Existing scientific benchmarks almost exclusively adopt QA formats, which only examine factual recall and reasoning over known knowledge, failing to evaluate the ability of LLMs to conduct open-ended scientific exploration—including identifying methodological limitations, proposing improvement ideas, and cross-domain analogical innovation. (2) The complexity of Earth science (multi-dimensional data, deep concept trees, computational intensity) is significantly simplified in general benchmarks. LLMs may appear proficient at answering questions but expose numerous vulnerabilities when facing professional calculations, terminology definitions, and tool selection. (3) Evaluation metrics are often singular (usually just accuracy), failing to reflect the diversity and depth of scientific thinking.

**Key Challenge**: The core value of Earth science lies in scientific exploration rather than knowledge recitation, yet existing tools only evaluate the latter. This creates an illusion: LLMs achieve 85%+ accuracy on general benchmarks but fail repeatedly on real Earth science problems.

**Goal**: (1) Construct a domain-specific and sufficiently deep benchmark for Earth science covering the entire field; (2) Simultaneously evaluate basic knowledge and high-level scientific exploration capabilities; (3) Introduce open-ended dialogues to simulate the process of real scientific discovery; (4) Design a new metric (SES) to measure both answer quality and cognitive diversity.

**Key Insight**: This paper observes a universal pattern in real scientific research—"critiquing the limitations of existing methods → proposing improvements." This iterative self-consistency process can be formalized as a recurrence formula: $(M^{i+1}, L^{i+1}) = \text{LLM}(M^i, L^i)$, where $M$ represents the method and $L$ represents its limitations. A model with true potential for scientific discovery must maintain consistency and progression within this closed loop.

**Core Idea**: Utilize a three-layer progressive dataset (Breadth → Depth → Exploration) to separately evaluate different capabilities, replace single-round QA with open-ended dialogue to capture the true nature of scientific thinking, and use a composite metric (SES = retention × diversity) to simultaneously evaluate answer quality and cognitive diversity.

## Method

### Overall Architecture

The construction pipeline of EARTHSE starts from 100,000 papers and builds three datasets via three parallel routes. First, papers are classified into Earth's five major spheres (biosphere, lithosphere, atmosphere, hydrosphere, cryosphere) using abstracts and keywords. Three progressive subsets are screened based on publication quality and citation counts: the Base set ($P_{\text{base}}$, 100k papers), the High-quality set ($P_{hj}$, 10k papers), and the High-citation set ($P_{hc}$, 1k papers). Two different construction pipelines are then used: the QA pipeline generates two stepped QA datasets (Earth-Iron, Earth-Silver), while the Dialogue pipeline extracts structured scientific exploration patterns from high-citation papers to generate an open-ended dialogue dataset (Earth-Gold). Both pipelines include automated cleaning (format checking, rule-based filtering) and manual verification (expert multi-dimensional scoring), resulting in three benchmarks that are both independent and collaborative.

### Key Designs

**1. Three-layer Progressive Dataset Architecture: Basic Breadth → Professional Depth → Exploration Dimension**

Earth-Iron (Basic Layer) contains 4,133 questions across 114 subfields and 11 basic scientific tasks (terminology explanation, knowledge QA, fact verification, logical analysis, relation extraction, calculation, tool selection, literature citation, dataset recommendation, experimental design, code generation), emphasizing broad mastery of basic Earth science concepts. Earth-Silver (Depth Layer) uses papers from high-impact journals to generate harder and more professional questions, with a significant increase in difficulty (average MC accuracy is only 54% vs. 89% for Iron). Earth-Gold (Exploration Layer) moves beyond single-round QA to open-ended multi-round dialogues, requiring models to demonstrate scientific thinking by "critiquing limitations of existing methods → proposing improvement plans → reflecting on the limitations of new plans." The three datasets correspond to three stages of scientific capability: memorizing facts → deep understanding → innovative thinking.

**2. Automated Structured Extraction + Multi-dimensional Manual Verification Quality Control**

For Earth-Iron/Silver QA generation, a small LLM first analyzes paper abstracts to select the most appropriate task from 11 categories (e.g., papers with many numerical results are prioritized for calculation tasks). Then, GPT-4 generates QA pairs given paper content and task prompts. Critically, the generation process mandates a complete Chain-of-Thought (CoT) derivation. This CoT serves as both part of the answer and "evidence" for subsequent data cleaning—if the CoT is contradictory or fails to support the final answer, LLM hallucinations are automatically detected. Cleaning involves two stages: rule-based cleaning (format, missing options, non-standard answers) → semantic cleaning (multiple correct options, irrelevant answers). Additionally, difficulty filtering is performed using mainstream LLMs: questions with accuracy > 80% are removed (too easy), while those between 60%-80% are judged by experts. For Earth-Gold dialogue generation, quadruplets ($M_0, L_0, M_1, L_1$)—existing method, its limitation, new method, and its limitation—are structurally extracted from paper sections like related work, motivation, methods, and discussion. GPT-4 then generates two-round dialogues: the first round summarizes current methods and critiques limitations; the second round proposes improvements and objectively assesses new constraints. Finally, domain experts score based on three dimensions: (a) information density, (b) method quality, and (c) logical consistency.

**3. Scientific Exploration Score (SES): Measuring Both Answer Quality and Thinking Diversity**

Since open-ended questions cannot be scored with a single "correct answer," EARTHSE innovatively defines SES to evaluate scientific exploration capability. The core idea is for the model to generate $M$ diverse responses (temperature 0.6) and calculate two dimensions:

- **Retention rate** $r$: GPT-4 ranks the $M$ generated answers and the reference answer based on "depth of reflection and innovation." If the reference answer is at position $i$ in the ranking, then $r = \frac{i-1}{M}$. If all generated answers are worse than the reference, $r=0$; if the reference is ranked first, $r=1$. This measures the "quality retention" of the model's answers.

- **Diversity** $d$: Use sentence-transformers to generate embedding vectors $v_i$ for each response, calculate the mean vector $\bar{v} = \frac{1}{M}\sum_i v_i$, and determine the average cosine similarity $\bar{s} = \frac{1}{M}\sum_i \cos(v_i, \bar{v})$. Lower similarity indicates higher diversity, thus $d = \frac{1}{\bar{s}}$. This measures the "divergent thinking" of the model on the same problem.

- **Combined SES Metric**: $\text{SES} = r \times d$ (with $\bar{s}$ normalized to $[0.9, 1]$ for adjustment, i.e., $\text{SES} = \frac{r}{10 \times (\bar{s} - 0.9)}$). A higher SES indicates answers that are both high-quality (high retention) and cognitively diverse (high diversity), reflecting the qualities of a scientist who "thinks deeply and views problems from multiple perspectives."

**4. Multi-format QA Design + CoT Guidance Strategy**

Earth-Iron/Silver QA includes four formats: Multiple Choice (MC), Fill-in-the-Blank (FIB), True/False (TF), and Free Response (FR). MC/TF/FIB are scored by Accuracy (Acc). FR uses two metrics: (a) Win Rate (WR)—using GPT-4 to compare model answers and reference answers across "relevance, scientific rigor, and specificity"; (b) Semantic Similarity (SS)—using sentence-transformers for cosine similarity. Furthermore, the paper finds that FIB accuracy is particularly low (avg. 11%). Experiments show that providing preliminary CoT steps as hints during inference significantly boosts accuracy (from 12.8% at 0 steps → 21.6% at 1 step → 45.6% at 3 steps), offering new insights for "inference-time scaling."

## Key Experimental Results

### Dataset Composition and Basic Statistics

| Dataset | Questions | Source Paper Set | Spheres | Subfields | Task Types | Format |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Earth-Iron | 4,133 | $P_{\text{base}}$ (100k) | 5 | 114 | 11 Basic Tasks | MC/FIB/TF/FR |
| Earth-Silver | ~2,000 | $P_{hj}$ (10k) | 5 | 114 | 11 Basic Tasks | MC/FIB/TF/FR |
| Earth-Gold | ~1,000 | $P_{hc}$ (1k high-citation) | 5 | - | Scientific Exploration | 2-round Open |

### Main Results: 11 Mainstream LLMs on Earth-Iron

| Model | MC Acc↑ | TF Acc↑ | FIB Acc↑ | FR-WR↑ | FR-SS↑ |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Llama-3.1-8B | 59.41% | 74.36% | 2.52% | 13.70% | 0.76 |
| Llama-3.1-70B | 91.56% | 87.91% | 6.63% | 61.85% | 0.80 |
| Qwen-2.5-72B | 92.42% | 86.26% | 11.96% | 92.05% | 0.79 |
| DeepSeek-V3 | 93.40% | 81.14% | 18.99% | 97.60% | 0.81 |
| GPT-4o | 93.28% | 88.28% | 19.12% | 82.00% | 0.81 |
| Gemini-1.5-Flash | 90.83% | 75.82% | 13.65% | 95.60% | 0.79 |
| Gemini-2.0-Flash | 92.67% | 87.55% | 14.69% | 77.10% | 0.77 |
| Gemini-2.5-Flash | 93.15% | 77.84% | 17.02% | 95.81% | 0.75 |
| Claude-3.5-Haiku | 91.08% | 83.52% | 12.48% | 12.05% | 0.79 |
| Claude-3.7-Sonnet | 94.01% | 61.90% | 20.68% | 75.00% | 0.80 |
| Grok-3 | 93.03% | 88.64% | 21.85% | 98.70% | 0.81 |
| **Average** | **89.53%** | **81.20%** | **14.50%** | **72.86%** | **0.78** |

### Ablation Study: Difficulty Gap Between Earth-Iron and Earth-Silver

| Dataset | MC Acc | TF Acc | FIB Acc | FR-WR | Difficulty Characteristics |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Earth-Iron | 89.53% | 81.20% | 14.50% | 72.86% | Basic, Broad, Relatively Easy |
| Earth-Silver | 54.40% | 59.38% | 11.06% | 41.15% | Professional, Deep, Significantly Difficult |
| **Drop** | **-35.13%** | **-21.82%** | **-3.44%** | **-31.71%** | Silver performance is barely better than random |

### Earth-Gold Open Dialogue Evaluation: SES Metric Comparison

| Model | Retention↑ | Diversity↑ | SES↑ | Feature |
| :--- | :--- | :--- | :--- | :--- |
| Llama-3.1-8B | 8.00% | 3.98 | 0.33 | Low quality, diverse but poor |
| DeepSeek-V3 | 38.00% | 1.69 | 0.66 | Moderate quality, singular thinking |
| Gemini-2.5-Flash | 50.56% | 2.70 | **1.37** | **SOTA**, balances quality & diversity |
| Grok-3 | 17.22% | 1.53 | 0.27 | Poor performance |
| **Average** | **22.58%** | **2.02** | **0.48** | Most models retention < 50%, lack diversity |

### Key Findings

1.  **Multiple choice is deceptively simple; Fill-in-the-blank exposes gaps**: MC avg. is 89%, but FIB is only 14.5%, suggesting LLM understanding of Earth science concepts remains at the "identification" rather than "recall/calculation" stage.
2.  **Difficulty cliff in Earth-Silver**: MC drops from 89% to 54% (only 29 percentage points above random guessing). This reflects that even top models struggle with professional-grade Earth science problems, as the domain's depth exceeds the coverage in general training data.
3.  **Calculation capability is a common weakness**: Nearly all models fail significantly on questions involving multi-step formulas, unit conversions, and numerical derivation. Errors often include "correct formula but wrong variables" or "forgetting unit conversion."
4.  **Extreme lack of open-ended exploration capability**: On Earth-Gold, over 75% of model responses are inferior to the reference (retention < 50%). More critically, diversity is low—responses to the same question are highly similar (avg. $\bar{s}$ near 1), contradicting the multi-perspective thinking required for scientific exploration.
5.  **The power of CoT guidance**: Providing preliminary CoT steps during inference triples FIB accuracy from 12.8% to 45.6%. This suggests that designing better inference-time scaffolds is as important as increasing model scale.

## Highlights & Insights

-   **Elegance of the Three-layer Progressive Benchmark**: Separating capabilities into Breadth (Iron) → Depth (Silver) → Exploration (Gold) prevents a single "Earth science performance" score from masking fundamental differences. This framework serves as a transferable "benchmark design template."
-   **Shift from Single-round QA to Open Dialogue**: Real scientific discovery is an iterative process of questioning and refinement. Earth-Gold formalizes this loop via the recurrence $(M^{i+1}, L^{i+1}) = \text{LLM}(M^i, L^i)$, establishing a new paradigm for scientific thinking evaluation.
-   **SES Composite Metric Design**: Using a multiplicative combination of retention (quality) × diversity (variability) severely penalizes models that are "good but inflexible." This align well with the character requirements of a scientist.
-   **CoT as a Dual-purpose Quality Control Tool**: Using CoT not just as part of the answer but as an "evidence chain" for automated error detection is a strategy transferable to any benchmark generation task.
-   **Balanced Coverage of the Five Earth Spheres**: Strict balance (approx. 20% each) across the spheres ensures multi-domain fairness and prevents data overload from specific sub-fields.

## Limitations & Future Work

-   **Limitations of Prior Work noted by authors**: The 11 tasks are not yet integrated into complex "task chains" (e.g., Design Experiment → Analyze Data → Explain Phenomenon), which exceeds the current benchmark's scope.
-   **Ours Findings on Limitations**:
    -   Earth-Gold only features two rounds of dialogue; real research requires more. 1,000 papers might not yield sufficient coverage.
    -   SES calculation relies on GPT-4 rankings (subjectivity risk) and sentence-transformer similarities (may miss subtle semantic differences).
    -   Extremely low FIB accuracy (11%) might reflect excessively high design difficulty rather than just model capability gaps.
    -   Failure to distinguish "missing Earth science knowledge" from "general reasoning failure."
-   **Goal for Improvements**:
    -   Expand Earth-Gold to 5-10 rounds to simulate comprehensive paper writing.
    -   Introduce "Self-consistency" metrics: check if $M^{i+1}$ actually solves $L^i$.
    -   Establish specialized CoT-guided protocols for FIB and calculations to test the "scaffold vs. no scaffold" ceiling.
    -   Incorporate multimodal questions (charts, temporal series), crucial for spatial Earth science data.

## Related Work & Insights

-   **Comparison with ScienceQA/MMLU-Pro/SciBench**: These broad benchmarks lack Earth science depth. EARTHSE sacrifices breadth for depth, employing dense sampling across 114 subfields. Key Insight: **Detailed segmentation is sometimes more valuable than broad coverage.**
-   **Comparison with OceanBench/ClimaQA/GeoBench**: While domain-specific, these are limited to QA. EARTHSE adds the "Open Exploration" layer, redefining benchmark dimensions.
-   **Comparison with Automated Discovery Systems (e.g., AlphaFold)**: These are tool systems for end-to-end discovery. EARTHSE is a benchmark to diagnose the status quo of "scientific thinking," facilitating the development of future tool systems.

## Rating

-   **Novelty**: ⭐⭐⭐⭐⭐ — First benchmark covering "Basic + Depth + Exploration" in Earth science; SES and open-dialogue paradigm are innovative.
-   **Experimental Thoroughness**: ⭐⭐⭐⭐☆ — Systematic evaluation of 11 LLMs with multi-dimensional analysis; minor desire for deeper ablation on CoT deprivation.
-   **Writing Quality**: ⭐⭐⭐⭐☆ — High information density and clear structure, though motivation for some technical details (e.g., SES normalization) could be more thorough.
-   **Value**: ⭐⭐⭐⭐⭐ — Direct diagnostic value for LLMs in science; the three-layer design is a transferable paradigm for AI-aided research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] In-Context Learning for Pure Exploration](in-context_learning_for_pure_exploration.md)
- [\[ICLR 2026\] Towards Self-Evolving Agent Benchmarks: Validatable Agent Trajectory via Test-Time Exploration](towards_self-evolving_agent_benchmarks_validatable_agent_trajectory_via_test-tim.md)
- [\[ICLR 2026\] CatalystBench: A Comprehensive Multi-Task Benchmark for Advancing Language Models in Catalysis Science](catalystbench_a_comprehensive_multi-task_benchmark_for_advancing_language_models.md)
- [\[ICLR 2026\] DARE-bench: Evaluating Modeling and Instruction Fidelity of LLMs in Data Science](dare-bench_evaluating_modeling_and_instruction_fidelity_of_llms_in_data_science.md)
- [\[ICLR 2026\] Sci2Pol: Evaluating and Fine-tuning LLMs' "Science-to-Policy Brief" Generation Capabilities](sci2pol_evaluating_and_fine-tuning_llms_on_scientific-to-policy_brief_generation.md)

</div>

<!-- RELATED:END -->
