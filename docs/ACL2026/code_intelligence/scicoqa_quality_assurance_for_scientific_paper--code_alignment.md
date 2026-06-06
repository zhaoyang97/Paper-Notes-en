---
title: >-
  [Paper Note] SciCoQA: Quality Assurance for Scientific Paper–Code Alignment
description: >-
  [ACL 2026][Code Intelligence][Paper-Code Difference Detection] This paper proposes SciCoQA, the first benchmark dataset for detecting differences between scientific papers and their code implementations. It contains 635…
tags:
  - "ACL 2026"
  - "Code Intelligence"
  - "Paper-Code Difference Detection"
  - "Scientific Reproducibility"
  - "Cross-modal Verification"
  - "LLM Evaluation"
  - "Quality Assurance"
date: 2026-05-08
content_hash: d276881086dec7ec
---

# SciCoQA: Quality Assurance for Scientific Paper–Code Alignment

**Conference**: ACL 2026  
**arXiv**: [2601.12910](https://arxiv.org/abs/2601.12910)  
**Code**: [https://github.com/ukplab/scicoqa](https://github.com/ukplab/scicoqa)  
**Area**: Scientific Reproducibility / Paper-Code Alignment Verification  
**Keywords**: Paper-Code Difference Detection, Scientific Reproducibility, Cross-modal Verification, LLM Evaluation, Quality Assurance

## TL;DR

This paper proposes SciCoQA, the first benchmark dataset for detecting differences between scientific papers and their code implementations. It contains 635 difference instances (92 real + 543 synthetic). After evaluating 22 LLMs, the study finds that the strongest model only detects 46.7% of real differences, revealing a critical capability gap in automated scientific quality assurance.

## Background & Motivation

**Background**: The scientific reproducibility crisis continues to plague academia. While publishing code and data has become a consensus, code availability does not equate to consistency between the code and the paper's description. In practice, implementation details often deviate from paper descriptions—ranging from "mathiness" (where equations merely simulate technical depth while actual gains come from undocumented tricks) to implementation differences in evaluation metrics (e.g., different implementations of BLEU scores rendering scientific comparisons invalid).

**Limitations of Prior Work**: (1) Paper-code inconsistencies are typically discovered only during reproduction attempts, wasting significant resources and eroding scientific trust; (2) Reviewers already face severe time pressure, making meticulous code review impractical; (3) With the rise of automated systems like "AI Scientists" (generating ideas, code, and papers automatically), manual review is increasingly unfeasible—an automatically generated codebase might run perfectly and perform well, but the implemented method may differ entirely from the paper's description.

**Key Challenge**: Scientific output is growing exponentially due to automation, yet the ability to verify the faithfulness of papers and code still relies entirely on manual effort. Existing evaluations (such as manual rubrics in PaperBench, general LLM-as-a-judge, or "does the code run") cannot reliably detect semantic differences between papers and code.

**Goal**: Build the first paper-code difference detection benchmark to systematically evaluate whether LLMs can automatically discover semantic inconsistencies between scientific papers and code.

**Key Insight**: Start from "natural discoveries" in the reproducibility community—utilizing user-reported paper-code differences in GitHub Issues and reproduction reports from Reproducibility Challenges as sources of real differences, then expanding data to computational science fields beyond CS via a synthetic generation pipeline.

**Core Idea**: Define paper-code difference detection as a cross-modal verification task (text vs. code). Construct a structured dataset including a difference type taxonomy (Difference/Paper Omission/Code Omission) and an impact category taxonomy (Algorithm/Model/Loss/Evaluation/Data/Training) to evaluate the long-context cross-modal reasoning capabilities of LLMs.

## Method

### Overall Architecture

The construction of SciCoQA follows two paths: real data collection and synthetic data generation. Real data is obtained from GitHub Issues (59 differences from 10,636 issues across 1,890 repositories after automated + manual filtering) and reproducibility papers (65 differences from 171 reproduction reports via GPT-5 extraction + manual verification), followed by dual verification and standardized description generation using Gemini 3.1 Pro and GPT-5, resulting in 92 real differences. Synthetic data consists of 543 differences across 204 repositories where GPT-5 generated code modifications, covering CS as well as physics and statistics.

### Key Designs

1.  **Strict Definition and Three-Type Classification of Paper-Code Differences**:
    *   Function: Establishes clear boundaries for difference definitions to exclude irrelevant noise.
    *   Mechanism: A difference is defined as a "semantic conflict between the paper's description of the scientific method and the code implementation, making the code unable to faithfully reproduce the reported method." It is categorized into three types: Difference (logic differs, e.g., L1 vs. L2 regularization), Paper Omission (code contains key components not described in the paper), and Code Omission (steps described in the paper are missing in the code). Specifically excluded are bugs (unrelated to paper description), hyperparameter differences resolvable via CLI/config, and standard engineering practices (e.g., noise addition for numerical stability).
    *   Design Motivation: Without a clear definition, consistent annotation standards cannot be built. Excluding engineering details and bugs ensures the dataset focuses on semantic differences affecting scientific validity.

2.  **Six-Category Impact Taxonomy**:
    *   Function: Describes which part of the research pipeline the difference affects.
    *   Mechanism: Defines six categories: Algorithm (step order/operations/core logic), Model (architecture/weight initialization), Loss (loss definitions/terms), Evaluation (evaluation logic/metrics), Data (data usage/preprocessing/augmentation), and Training (learning process/scheduling/optimization). In real data, Algorithm (25%) and Loss (24%) dominate; in synthetic data, Algorithm (26%) and Model (21%) dominate.
    *   Design Motivation: Knowing the type (what) and the scope of impact (where) helps in understanding the severity and detection difficulty of different inconsistencies.

3.  **Synthetic Data Generation Pipeline**:
    *   Function: Scales the dataset from CS/AI to other computational scientific fields such as physics, statistics, and quantitative biology.
    *   Mechanism: Samples 204 repositories linked to arXiv papers with permissive licenses from GitHub. GPT-5 generates 5 code diffs per repository based on paper and code (constrained by difference definitions). At most 3 modifications per repository that do not manipulate the same file and allow exact matching to original code are sampled. The correlation of detection rates between real and synthetic data reaches $r = 0.94$, validating synthetic data as a reliable proxy for model ranking.
    *   Design Motivation: Real differences are naturally scarce and limited to CS/AI. The synthetic pipeline addresses bottlenecks in data scale and domain coverage while generating data not present in model pre-training corpora to combat data contamination.

### Loss & Training

This work evaluates benchmarks and does not involve model training. Evaluation uses LLM-as-Judge (GPT-OSS 20B reasoning model), achieving an F1 of 87.5% on 1,039 manually annotated samples. Models are prompted to generate a list of differences, which are then parsed into independent differences for matching evaluation.

## Key Experimental Results

### Main Results

| Model | Real Data Recall | Synthetic Data Recall | Average Recall |
| :--- | :--- | :--- | :--- |
| Gemini 3.1 Pro | 46.7% | — | — |
| GPT-5 Mini | 46.7% | — | — |
| GPT-5 | — | 70.0% | — |
| Nemotron 49B | — | — | 23.9% |
| Qwen3 30B Coder | — | — | 23.5% |

| Model | Precision | Recall | F1 |
| :--- | :--- | :--- | :--- |
| GPT-5 | 88.0 | 51.2 | 64.7 |
| Gemini 2.5 Pro | 94.6 | 41.1 | 57.3 |
| GPT-OSS 20B | 69.9 | 55.8 | 62.1 |

### Ablation Study

| Input Condition | Real Data | Synthetic Data | Description |
| :--- | :--- | :--- | :--- |
| Paper + Code | Baseline | Baseline | Complete input |
| Code Only | -19.2pp (rel. ↓48.3%) | -16.3pp (rel. ↓30.8%) | Paper provides necessary cross-modal signals |

### Key Findings

*   **Recall is the core bottleneck**: The strongest model detects only 46.7% of real differences. Precision (88-94.6%) is significantly higher than recall, indicating that models are "mostly correct in what they find, but miss too much."
*   **Paper Omission is hardest to detect**: Inconsistencies where code contains components not described in the paper are the most difficult to find (differences from GitHub are 71.4% "Difference" and thus easier; reproduction reports are 50% "Paper Omission" and harder) because there is no anchor in the paper for comparison.
*   **Long context severely degrades performance**: Detection rates consistently drop as paper + code token counts increase. Median input is 56,903 tokens, with 73/276 papers exceeding 100k tokens.
*   **Data contamination has a significant impact**: Models perform better on papers published before their pre-training cutoff. Detection rates are lowest for 2025 papers, suggesting models benefit from specific papers and code seen during pre-training.
*   **Paper is a necessary input**: Performance drops significantly for all models when the paper is removed (rel. drop of 48.3% on real data), confirming the cross-modal nature of the task.
*   **Code-specific models do not hold an advantage**: GPT-5 Codex performed worse than GPT-5 Mini, indicating that this task requires a combination of code understanding and natural language reasoning, where general instruction-following is more useful.

## Highlights & Insights

*   **Filling a Critical Gap**: Formally defines paper-code alignment detection as a benchmarkable NLP task for the first time, possessing high practical significance in the era of scientific automation.
*   **Complementary Real and Synthetic Design**: Real data ensures realism (from actual user reports and reproduction efforts), while synthetic data addresses scarcity and domain coverage. The high correlation ($r=0.94$) between their detection rates validates the design.
*   **Insight on "High Precision, Low Recall"**: In verification scenarios, low recall is most damaging—missed differences provide a false sense of security, whereas false positives can be filtered by humans. This serves as a critical warning for deploying automated verification systems.
*   **Natural Testbed for Data Contamination**: Analyzing detection rates by publication year cleverly reveals data contamination issues, while the synthetic pipeline provides a solution for generating uncontaminated data.

## Limitations & Future Work

*   Real data is heavily biased toward CS/AI; non-CS domains only have synthetic data, and error distributions may differ from reality.
*   Synthetic differences are generated by GPT-5, and since GPT-5 is also an evaluated model, self-preference bias may exist ($r$ increases from 0.94 to 0.98 when GPT-5 is removed).
*   The dataset size is relatively small (635 differences), reflecting a trade-off between quality and scale.
*   Difference definitions exclude bugs and hyperparameter issues, not covering the full spectrum of software engineering defects in research code.
*   Future work needs to expand collection channels for real non-CS data and develop specialized models for paper-code verification.

## Related Work & Insights

*   **vs. Bianchi et al. (2025)**: That work detects inconsistencies within paper text; SciCoQA extends this to cross-modal inconsistencies between paper and code.
*   **vs. PaperBench**: PaperBench uses manual rubrics to verify code implementation correctness, which is high-cost and non-scalable; SciCoQA provides an automatically evaluatable benchmark.
*   **vs. CCI (Code-Comment Inconsistency)**: CCI handles function-level code-comment inconsistencies; SciCoQA requires global semantic alignment across an entire paper and a multi-file codebase, making it significantly more challenging.
*   **vs. ProcessBench/ErrorRadar**: These benchmarks detect errors in reasoning chains; SciCoQA detects cross-modal semantic differences between paper descriptions and code implementations.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First to formalize paper-code alignment verification; precise and highly relevant problem definition.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ 22 models, multi-dimensional analysis (type/source/length/year/ablation/precision verification); rigorous experimental design.
*   Writing Quality: ⭐⭐⭐⭐⭐ Insightful motivation, strict difference definitions, and progressive experimental analysis.
*   Value: ⭐⭐⭐⭐⭐ Provides infrastructure-level contributions for verifying paper-code faithfulness in the era of scientific automation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] CodeRL+: Improving Code Generation via Reinforcement with Execution Semantics Alignment](coderl_improving_code_generation_via_reinforcement_with_execution_semantics_alig.md)
- [\[ACL 2026\] QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions](qaq_bidirectional_semantic_coherence_for_selecting_high-quality_synthetic_code_i.md)
- [\[ACL 2026\] CodeDistiller: Automatically Generating Code Libraries for Scientific Coding Agents](codedistiller_automatically_generating_code_libraries_for_scientific_coding_agen.md)
- [\[ICLR 2026\] Paper2Code: Automating Code Generation from Scientific Papers in Machine Learning](../../ICLR2026/code_intelligence/paper2code_automating_code_generation_from_scientific_papers_in_machine_learning.md)
- [\[NeurIPS 2025\] Embedding Alignment in Code Generation for Audio](../../NeurIPS2025/code_intelligence/embedding_alignment_in_code_generation_for_audio.md)

</div>

<!-- RELATED:END -->
