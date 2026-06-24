---
title: >-
  [Paper Note] CatalystBench: A Comprehensive Multi-Task Benchmark for Advancing Language Models in Catalysis Science
description: >-
  [ICLR2026][LLM Evaluation][Catalysis Science] This paper constructs the first multi-task benchmark for catalysis science, CatalystBench, which unifies theoretical computational data and experimental literature into 8 tasks covering the "full process of catalyst design." It proposes Multi-head Full-task Fine-tuning (MFT) to decouple classification, regression, and generation heads. The resulting CatalystLLM outperforms strong baselines like GPT-4.1 on most tasks…
tags:
  - "ICLR2026"
  - "LLM Evaluation"
  - "Catalysis Science"
  - "Multi-task benchmark"
  - "Domain fine-tuning"
  - "Multi-head architecture"
  - "Structure-activity relationship"
date: 2026-05-08
content_hash: c8bc9dcb708265f9
---

# CatalystBench: A Comprehensive Multi-Task Benchmark for Advancing Language Models in Catalysis Science

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=tCFYwPdmT4](https://openreview.net/forum?id=tCFYwPdmT4)  
**Code**: To be confirmed (The paper commits to open-sourcing CatalystBench and CatalystLLM)  
**Area**: LLM Evaluation / Science LLMs / Catalytic Materials  
**Keywords**: Catalysis Science, Multi-task benchmark, Domain fine-tuning, Multi-head architecture, Structure-activity relationship

## TL;DR
This paper constructs the first multi-task benchmark for catalysis science, CatalystBench, which unifies theoretical computational data and experimental literature into 8 tasks covering the "full process of catalyst design." It proposes Multi-head Full-task Fine-tuning (MFT) to decouple classification, regression, and generation heads. The resulting CatalystLLM outperforms strong baselines like GPT-4.1 on most tasks, achieving an average improvement of 12.44% over single-task baselines.

## Background & Motivation
**Background**: Catalyst discovery is a cornerstone of chemical engineering and sustainable energy but is highly dependent on experience and domain knowledge. In the recent "AI for Science" wave, LLMs have demonstrated potential in fields like bioengineering, materials discovery, and process optimization, leading to interest in their application to catalyst design.

**Limitations of Prior Work**: The catalysis field lacks a benchmark that reflects real-world R&D workflows. Existing chemistry or materials benchmarks (ChemBench, MaScQA, SCIBENCH, ChemCoTBench, LLM4Mat-Bench, MatTools, etc.) either only test "textbook-style" molecular level knowledge or evaluate restricted problem-solving abilities. They rarely incorporate complementary knowledge such as **synthesis conditions, stability, and structure-activity trends from experimental literature**. For catalysis, this is crucial as real catalyst surfaces often deviate significantly from the ideal models assumed in theoretical calculations.

**Key Challenge**: Real catalyst R&D is a **multi-modal, multi-stage** workflow requiring "precise numerical regression (e.g., adsorption energy, Faraday efficiency) + category judgment (control method classification) + open-ended mechanism reasoning (understanding control schemes)" simultaneously. The properties of the loss landscapes for these three types of objectives differ drastically. Models trained with homogeneous tasks or a unified output format suffer from mutual interference due to task heterogeneity, making it difficult to maintain accuracy across the entire spectrum. Another challenge is the **siloing of knowledge sources**: high-fidelity theoretical descriptors from DFT calculations and synthesis/performance data recorded in experimental literature have long existed as isolated islands without a unified framework.

**Goal**: (1) Construct a benchmark that unifies theoretical data and experimental literature into "task-oriented Q&A," where each task corresponds to a stage in the catalyst design lifecycle; (2) Design a domain fine-tuning method capable of handling heterogeneous qualitative and quantitative tasks without performance degradation.

**Key Insight**: The authors abstract catalyst R&D into three fundamental capabilities—**Understanding, Reasoning, and Explaining**—and derive 8 real-world tasks. Observing that task heterogeneity stems from "output space heterogeneity," they address this by **decoupling the output space at the architectural level**.

**Core Idea**: Use a "unified multi-task benchmark + multi-head full-task fine-tuning (shared backbone + decoupled classification/regression/generation heads)" to solve the lack of evaluation tools for catalysis LLMs and the problem of multi-task training interference.

## Method

### Overall Architecture
The paper follows two main threads: **constructing a benchmark** by organizing dispersed catalysis data into 8 tasks covering the full process, and **building a model** by proposing the MFT fine-tuning strategy to train and validate CatalystLLM.

Regarding the benchmark: Centered on "Understanding-Reasoning-Explaining," data was collected from 8 public theoretical simulation datasets and scientific literature. Theoretical data (containing descriptors like SMILES, adsorption sites) was converted into "instruction + input + output" natural language tasks via prompt templates. Experimental literature was processed using SciQAG to guide GPT-4o in converting synthesis/control texts into high-quality Q&A pairs, followed by expert annotation and filtering. The final 8 tasks are grouped into four output types: information extraction (ME, SE), text classification (RMC), numerical regression (FEP, AP, d-CP, FP), and semantic generation (RSC).

Regarding the model: ChemLLM-7B was chosen as the base (due to its prior chemical domain instruction tuning). The authors compared 4 fine-tuning paradigms: Single-Task (ST), Multi-head Single-Task (MST), Full-Task (FT), and Multi-head Full-Task (MFT). MFT is the proposed solution, utilizing three decoupled output heads attached to a shared backbone. A "task type token" appended to the input determines which head to activate, allowing for shared cross-task features while isolating heterogeneous outputs.

### Key Designs

**1. "Full-Process" task system with 3-layer abilities × 8 tasks: Covering the catalyst design loop**

The problem with existing benchmarks is that they only test isolated segments. Catalyst R&D is a closed loop from literature review and experimental analysis to scheme design. The authors split catalysis capability into Understanding, Reasoning, and Explaining, implemented as 8 tasks: Material Extraction (ME), Synthesis Extraction (SE), Regulation Method Classification (RMC), Faradaic Efficiency Prediction (FEP), Adsorption Energy Prediction (AP), d-Center Prediction (d-CP), Formation energy Prediction (FP), and Regulation Scheme Comprehension (RSC). Each corresponds to a real-world design stage. Task scales were kept balanced to reflect real-world constraints where high-fidelity data is expensive to obtain.

**2. Dual-source construction with theory + experiment: Bridging knowledge silos**

This is the core differentiator from pure theoretical benchmarks. Theoretical data was filtered for descriptors (e.g., SMILES strings, formulas like Ti18Pd54, sites like (2 1 0)) and converted into quantifiable tasks. Experimental data used the SciQAG framework to guide GPT-4o in interpreting control scheme texts, followed by domain expert review. Evaluation focused on higher-order "Explanation" capabilities. Multi-stage quality control was applied to ensure semantic accuracy and diversity, unifying high-fidelity theoretical descriptors with experimental synthesis conditions.

**3. Multi-head Full-task Fine-tuning (MFT): Decoupling output spaces via task tokens**

MFT addresses the interference between numerical regression, category judgment, and open generation. It **decouples the output space** by appending a task type token $\tau$ to the input sequence. This token determines which specialized head is activated using the contextual representation $h[t_{n-1}]$ of the last non-padding token from the shared ChemLLM backbone.
- Classification tasks use a task-specific MLP head.
- Regression tasks use a head optimized via Mean Squared Error (MSE).
- Generation tasks use the original `lm_head`.
The backbone learns jointly across all tasks while output modalities are isolated, using a weighted composite loss to balance contributions. This preserves cross-domain feature learning while avoiding interference between disparate loss landscapes.

**4. Multi-dimensional evaluation protocol for heterogeneous tasks: Specific metrics and dual-track scoring**

The authors report Accuracy and Balanced F1 for classification and extraction; MAE and $R^2$ for regression. For non-factual semantic Q&A (RSC), they designed a triple-track protocol: Sentence-level semantic similarity (STS), scores from GPT-4o and DeepSeek-R1 (on rationality, accuracy, usability), and final determination by domain experts on a random subset. This revealed that while top general models produce fluent, logical answers (high LLM score), they often suffer from "scientific hallucinations," which only experts can identify.

### Loss & Training
Fine-tuning used LoRA (rank 8, scale 16.0, dropout 0.1 on all linear modules) with the AdamW optimizer, a 5e-5 learning rate, and linear decay with warmup. Speed was enhanced via bf16 mixed precision and Flash Attention-2. The MFT total loss is a weighted composite of language modeling loss (generation), cross-entropy (classification), and MSE (regression).

## Key Experimental Results

### Main Results
Factual tasks (7 tasks / 12 metrics):

| Model | RMC ACC↑ | FEP $R^2$↑ | AP $R^2$↑ | d-CP $R^2$↑ | FP $R^2$↑ |
|------|----------|-----------|-----------|-------------|-----------|
| GPT-4.1 (Closed-source) | 0.75 | 0.56 | 0.61 | 0.59 | 0.65 |
| ChemLLM (Domain Base) | 0.52 | 0.45 | 0.63 | 0.54 | 0.64 |
| Darwin1.5 (Domain Model) | 0.50 | 0.44 | 0.59 | 0.54 | 0.68 |
| **CatalystLLM (Ours)** | **0.81** | **0.73** | **0.81** | **0.73** | **0.80** |

CatalystLLM achieved SOTA on all factual tasks, with particularly significant gains in regression (e.g., FEP $R^2$ increased from 0.45 to 0.73). General models like GPT-4.1 performed well in extraction/classification but lagged significantly in numerical regression, indicating that current general LLMs struggle with deep scientific reasoning and quantitative calculations.

Semantic comprehension tasks (RSC):

| Model | STS↑ | GPT-4o Score↑ | DeepSeek-R1 Score↑ | Expert Score @100↑ |
|------|------|-----------|------------------|--------------|
| GPT-4.1 | 0.72 | 0.85 | 0.88 | 0.56 |
| DeepSeek-v3 | 0.73 | 0.84 | 0.86 | 0.57 |
| **CatalystLLM** | **0.79** | 0.82 | 0.86 | **0.75** |

CatalystLLM led in STS and Expert scores. Notably, its LLM-based scores were not the highest, confirming that general models provide "beautiful" answers that contain domain hallucinations detectable only by experts.

### Ablation Study

| Configuration | Avg. Gain vs. ST Baseline | Note |
|--------------|---------------------------|------|
| Multi-task Only (MT/FT) | +9.24% | Gains from shared features in multi-tasking |
| Task-specific Heads Only (MST) | +5.13% | Limited gain without multi-tasking |
| **MFT (Ours, Multi-head + Full-task)** | **+12.44%** | Optimal combination |

- **Task Synergy**: Removing a task category caused performance drops across related tasks. Strong intra-group synergy was observed (e.g., removing RMC degraded FEP/RSC).
- **Input Format**: Structured complete inputs outperformed unstructured SMILES strings. Structured knowledge allows models to learn internal atomic connections and potential structure-activity relationships.

### Key Findings
- **Multi-task + Multi-head is essential**: Neither multi-tasking (+9.24%) nor decoupled heads (+5.13%) alone match the combination (+12.44%), suggesting that decoupled heads require multi-task training to be effective.
- **Regression is a weakness for General LLMs**: While general LLMs compete in extraction/classification, the gap in numerical regression is vast, marking it as the primary future direction for catalysis LLMs.
- **Domain fine-tuning builds "Knowledge Models"**: Synergies suggest models are not just memorizing patterns but are building a holistic catalysis knowledge model across chemical formulas and natural language.
- **Divergence between Expert and LLM scores**: High LLM scores for general models despite low expert scores highlight scientific hallucinations, cautioning against "LLM-as-a-judge" in professional scientific scenarios.

## Highlights & Insights
- **"Full-process taskification" is a transferable benchmark design**: Splitting a domain lifecycle into stages is more effective than textbook multiple-choice questions for evoking true capability. This paradigm applies to batteries, drugs, and alloys.
- **Task type token routing is a simple yet effective trick**: Appending $\tau$ at the end allows switching heads on a shared backbone with near-zero inference cost while separating heterogeneous loss landscapes.
- **Triple-track evaluation exposes "evaluator hallucination"**: Comparing STS, model scores, and expert scores side-by-side explicitly shows the departure of LLM judges from professional accuracy.
- **Honest disclosure of general model strengths**: The authors admit GPT-4.1's superiority in some extraction tasks (SE), which strengthens the credibility of their argument regarding the value of specialized models in regression.

## Limitations & Future Work
- **Knowledge scope constraints**: Capabilities are limited by CatalystBench coverage; accuracy drops for out-of-distribution materials.
- **Model scale**: 7B is a trade-off; larger models might show stronger reasoning but scaling behavior was not tested here.
- **Literature bias**: Inclusion of published literature may amplify focus on "hot" topics and ignore niche systems.
- **Evaluation dependence**: Data construction and semantic evaluation still utilize closed-source models (GPT-4o), risking circular dependence and bias.
- **Future Work**: Plan to expand more catalysis systems, enhance downstream reasoning, and embed the model into a "prediction-synthesis-characterization" closed-loop framework.

## Related Work & Insights
- **vs. Knowledge-based Chemistry Benchmarks (ChemBench / MaScQA)**: These use textbook/exam questions; CatalystBench focuses on experimental synthesis and real design stages.
- **vs. Prediction/Reasoning Benchmarks (LLM4Mat-Bench / ChemCoTBench)**: CatalystBench uniquely tests quantitative regression, qualitative classification, and open explanation in a unified framework.
- **vs. Domain LLMs (ChemLLM / Darwin1.5)**: Previous models often used unified output styles with limited regression gains; CatalystLLM’s multi-head architecture handles mixed tasks without mutual interference.

## Rating
- Novelty: ⭐⭐⭐⭐ First multi-task benchmark for catalysis science + systematic validation of MFT in the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Coverage of 8 tasks, 12 metrics, multiple baselines, and ablation studies; scaling behaviors not explored.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and structure; minor reliance on appendices for detail.
- Value: ⭐⭐⭐⭐ Provides a rare evaluation framework and open-source model for catalysis; clear guidance on regression weaknesses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](../../ACL2025/llm_evaluation/seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [\[ICLR 2026\] EARTHSE: A Benchmark for Earth Science Knowledge Exploration](earthse_a_benchmark_evaluating_earth_scientific_exploration_capability_for_large.md)
- [\[ICLR 2026\] AutoCodeBench: Large Language Models are Automatic Code Benchmark Generators](autocodebench_large_language_models_are_automatic_code_benchmark_generators.md)
- [\[ICLR 2026\] Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective](can_vision_language_models_assess_graphic_design_aesthetics_a_benchmark_evaluati.md)
- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](../../ACL2025/llm_evaluation/mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)

</div>

<!-- RELATED:END -->
## Related Papers

- [\[ACL 2025\] SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science](../../ACL2025/llm_evaluation/seedbench_a_multi-task_benchmark_for_evaluating_large_language_models_in_seed_sc.md)
- [\[ICLR 2026\] Can Vision–Language Models Assess Graphic Design Aesthetics? A Benchmark, Evaluation, and Dataset Perspective](can_vision_language_models_assess_graphic_design_aesthetics_a_benchmark_evaluati.md)
- [\[ACL 2025\] McBE: A Multi-task Chinese Bias Evaluation Benchmark for Large Language Models](../../ACL2025/llm_evaluation/mcbe_a_multi-task_chinese_bias_evaluation_benchmark_for_large_language_models.md)
- [\[ICLR 2026\] AnesSuite: A Comprehensive Benchmark and Dataset Suite for Anesthesiology Reasoning](anessuite_a_comprehensive_benchmark_and_dataset_suite_for_anesthesiology_reasoni.md)
- [\[ICLR 2026\] DeepResearch Bench: A Comprehensive Benchmark for Deep Research Agents](deepresearch_bench_a_comprehensive_benchmark_for_deep_research_agents.md)

</div>

<!-- RELATED:END -->
