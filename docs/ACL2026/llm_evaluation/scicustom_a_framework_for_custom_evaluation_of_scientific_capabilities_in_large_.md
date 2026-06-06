---
title: >-
  [Paper Note] SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models
description: >-
  [ACL2026][LLM Evaluation][Scientific LLM Evaluation] SciCustom decomposes scientific evaluation requirements into reusable ontological knowledge units. It automatically constructs domain-specific benchmarks through a tag…
tags:
  - "ACL2026"
  - "LLM Evaluation"
  - "Scientific LLM Evaluation"
  - "Custom Benchmark"
  - "Ontology"
  - "Knowledge Units"
  - "Ranking Consistency"
date: 2026-05-08
content_hash: ff66b2f955313129
---

# SciCustom: A Framework for Custom Evaluation of Scientific Capabilities in Large Language Models

**Conference**: ACL2026  
**arXiv**: [2605.19357](https://arxiv.org/abs/2605.19357)  
**Code**: https://github.com/yjwtheonly/SciCustom  
**Area**: LLM Evaluation / Scientific Capability Assessment / Automated Benchmark Construction  
**Keywords**: Scientific LLM Evaluation, Custom Benchmark, Ontology, Knowledge Units, Ranking Consistency

## TL;DR
SciCustom decomposes scientific evaluation requirements into reusable ontological knowledge units. It automatically constructs domain-specific benchmarks through a tagger, multi-model voting, binary-search relevance filtering, and proxy subset selection, achieving the highest Spearman rank consistency across 10/11 chemistry and medical subtasks.

## Background & Motivation
**Background**: LLMs are being utilized in scientific research for literature understanding, experimental hypothesis generation, and medical/chemical QA. Users generally prioritize a model's performance in specific application scenarios—such as technical chemistry, drug discovery, or clinical knowledge—rather than its average score on a broad scientific benchmark.

**Limitations of Prior Work**: General benchmarks like GPQA, MMLU-Pro, or SimpleQA do not necessarily predict model performance on niche scientific tasks. Manually customized benchmarks are costly and slow to iterate, while direct synthesis of questions by LLMs may lack grounded validity, failing to guarantee that scientific facts originate from reliable data.

**Key Challenge**: Scientific tasks are both highly interdisciplinary and require factual grounding. Constructing a benchmark from scratch for every new requirement leads to redundant labor, yet simple semantic retrieval or question synthesis fails to precisely capture "exactly which scientific knowledge units are required by this specific need."

**Goal**: The authors aim to build a framework capable of automatically generating application-specific scientific benchmarks based on user requirements without expert annotation or purely synthetic questions. The generated benchmarks should replicate the model rankings produced by expert-curated benchmarks across 10 different LLMs.

**Key Insight**: SciCustom assumes that complex scientific applications can be approximated as combinations of fine-grained knowledge units. By pre-mapping large-scale scientific QA data to these knowledge units, these units can be reused to dynamically construct benchmarks when new requirements arrive.

**Core Idea**: Organize data using scientific ontologies offline, then retrieve, filter, and sample grounded data using requirement-related knowledge units online before converting them into multiple-choice questions (MCQs) for efficient evaluation.

## Method
SciCustom operates in two stages: offline indexing and online construction. In the offline stage, knowledge units of moderate granularity are extracted from scientific ontologies, and a tagger is trained to map large-scale scientific QA to these units. In the online stage, the system receives user requirements, identifies relevant units via multi-model voting, and generates a compact yet discriminative benchmark using binary filtering and proxy selection.

### Overall Architecture
The input consists of a user evaluation requirement $r$ and a large scientific corpus $\mathcal{D}$, and the output is the corresponding benchmark $\mathcal{B}_r$. The framework first converts an ontology DAG of 227 scientific sub-disciplines into 642 knowledge units. A tagger then assigns knowledge unit labels to each data point. When a user submits a requirement, multiple LLMs vote to rank the relevance of knowledge units, selecting target units. The system retrieves candidate data from the tagged corpus, employs binary search to find a relevance cutoff, samples a representative proxy subset, and finally converts the original QA into automatically evaluable MCQs.

### Key Designs
1.  **Ontology-grounded Knowledge Units**:
    - **Function**: Provide a reusable and combinable semantic skeleton for scientific capability evaluation.
    - **Mechanism**: The authors integrate multiple authoritative scientific ontologies into a DAG of 227 scientific sub-disciplines. DFS is used to traverse each node, where an LLM judges whether the node granularity is coarse, moderate, or fine: coarse nodes are expanded, moderate nodes are collected as knowledge units, and fine nodes are pruned. This results in 641 scientific units plus one "Non-Scientific" unit, totaling 642 units.
    - **Design Motivation**: Concepts that are too coarse fail to distinguish niche capabilities, while those too fine are difficult to reuse. Granularity similar to "textbook chapter headings" balances interpretability and composability.

2.  **Tagger and Voting-based Unit Selection**:
    - **Function**: Organize large-scale scientific data into a knowledge space and translate natural language requirements into sets of target units.
    - **Mechanism**: Tagger training data consists of two parts: synthetic queries generated by an LLM from 1 to 5 sampled units combined with descendant keywords, and real scientific instruction data annotated with units by an LLM. Online, several heterogeneous LLMs independently rank candidate units; the system takes the average rank and selects the top-$K_1$ as $\mathcal{T}_r$.
    - **Design Motivation**: An offline tagger allows the same scientific data to be reused for different requirements. Multi-model consensus reduces bias from a single LLM's interpretation of the requirement.

3.  **Binary Search Filtering and Proxy Subset Selection**:
    - **Function**: Identify evaluation samples from massive candidates that are both relevant and discriminative.
    - **Mechanism**: Candidate data are ranked by the size of their intersection with target units and average rank. Since judging relevance session-by-session with an LLM is costly, SciCustom assumes ranked relevance generally decreases and uses binary search to find the last position still judged relevant by a majority of models, reducing oracle judgments to $O(\log(|\mathcal{D}'_r|))$. Subsequently, hardness scores, quality scores, and embedding clustering are used to select a proxy subset whose distribution mirrors the full set.
    - **Design Motivation**: The most relevant samples are not necessarily the most discriminative; they may represent high-frequency textbook knowledge. Binary cutoff and representative sampling help avoid "ceiling effects" caused by overly canonical questions.

### Loss & Training
The paper does not introduce complex new losses; the primary training target is the tagger. Specifically, the authors fine-tune LLaMA-3-8B as the tagging model using 50,000 synthetic scientific queries and 30,000 real scientific queries over 22 epochs with a learning rate of $2e^{-5}$ on 8 NVIDIA A100 GPUs. The final scientific corpus is aggregated from SciRIFF, SciInstruct, Mol-Instruct, MultiMedQA, SciEval, MMLU-Pro, GPQA, IfBench, and SimpleQA, totaling 2,000,367 instances. Data overlapping with expert ground-truth benchmarks is filtered to prevent leakage.

## Key Experimental Results

### Main Results
The primary metric is the Spearman/Kendall correlation between the rankings of 10 LLMs on SciCustom-generated benchmarks and their rankings on expert ground-truth benchmarks. The Spearman results are listed below.

| Domain Task | GPQA | MMLU / MedQA | GPT-5 synthetic | Embedding | SciCustom |
|----------|------|--------------|-----------------|-----------|-----------|
| Analytical chemistry | 0.61 | 0.21 | -0.11 | -0.34 | 0.86 |
| Inorganic chemistry | 0.52 | 0.27 | 0.05 | -0.59 | 0.67 |
| Material science | 0.21 | -0.61 | -0.04 | -0.39 | 0.42 |
| Organic chemistry | 0.72 | 0.21 | 0.38 | 0.11 | 0.89 |
| Physical chemistry | 0.21 | 0.52 | 0.24 | -0.73 | 0.74 |
| Technical chemistry | 0.03 | 0.31 | -0.07 | -0.41 | 0.86 |
| Virology | -0.11 | 0.44 | 0.25 | 0.18 | 0.55 |
| Human aging | -0.10 | 0.62 | 0.20 | 0.21 | 0.49 |
| Medical genetics | -0.09 | 0.35 | 0.09 | -0.21 | 0.42 |
| Anatomy | 0.48 | -0.19 | 0.11 | -0.32 | 0.62 |
| Nutrition | 0.18 | 0.45 | 0.52 | 0.27 | 0.78 |

### Ablation Study
Ablation analysis indicates that ontology-driven units, binary-search cutoff, and subset selection are all indispensable.

| Configuration | Key Metric | Description |
|------|----------|------|
| SciCustom | Optimal Spearman in 10/11 tasks | MedQA scored 0.62 on Human Aging, higher than SciCustom's 0.49 |
| SciCustom Top-1 Selection | 8/11 consistent with ground-truth | Helps users select the best model for specific needs |
| Tagger | Macro F1 75.2%, Micro F1 78.6% | Evaluated on 1,000 unseen compositional queries |
| Human Evaluation | Correctness 0.92, Relevance 0.70 | 50 chemistry questions, labeled by 3 AI4Chemistry masters |
| Greedy Search | Virology 0.21, Anatomy 0.24, Nutrition 0.31 | Simply picking top relevant samples lacks discrimination |
| SciCustom Binary Strategy | Virology 0.55, Anatomy 0.62, Nutrition 0.78 | Closer to expert rankings than greedy search |

### Key Findings
- Rankings on general benchmarks often diverge from, or even negatively correlate with, specialized tasks; high scores on GPQA or MMLU do not directly imply strength in niche scientific capabilities.
- Fully synthetic GPT-5 benchmarks show unstable performance, and embedding baselines are weak, indicating that grounded data and ontology structures are both essential.
- In Material Science, Spearman correlation reached only 0.42. However, the authors found that the correlation between two expert benchmarks in this field was only $\rho=0.31, \tau_b=0.22$, reflecting significant discrepancies in existing evaluation protocols for this domain.
- A Pericyclic Reaction case study demonstrates that SciCustom can locate relevant knowledge anchors like Cyclization, Aromatic hydrocarbon, and Ring compound even for niche requirements lacking existing benchmarks.

## Highlights & Insights
- This paper shifts "custom evaluation" from a problem of question generation to one of knowledge organization. The genuine difficulty lies not in having an LLM write questions, but in knowing which grounded data represents a specific scientific capability.
- The choice of knowledge unit granularity is insightful: units too coarse do not distinguish capabilities, while those too fine are not reusable. Intermediate granularity allows new requirements to be composed of multiple units.
- The counter-intuitive benefit of binary search is notable. The most relevant samples might be too common to distinguish models; expanding the cutoff appropriately yields a more challenging benchmark with better ranking resolution.
- Using model ranking consistency as the primary metric aligns better with the actual purpose of a benchmark: users ultimately care about model selection rather than the aesthetic quality of individual questions.

## Limitations & Future Work
- Current ontologies are primarily derived from OBO, BioPortal, and OLS, which skew toward biomedicine and chemistry; mathematics and theoretical physics are not yet included.
- SciCustom depends on the coverage of the source scientific corpus $\mathcal{D}$. High-quality benchmarks may be hard to construct for low-resource knowledge units due to data sparsity.
- The healthcare subset is currently restricted to text QA evaluation. The authors emphasize it does not involve actionable biological threats, but additional governance is needed if applied to sensitive experimental design or clinical decision-making.
- Relevance ranking does not strictly satisfy monotonicity. While experiments show binary search outperforms greedy search, there is room for further theoretical analysis.

## Related Work & Insights
- **vs GPQA / MMLU-Pro / MedQA**: These are fixed benchmarks suitable for general capability assessment; SciCustom dynamically constructs benchmarks for user needs, making it better for application-specific selection.
- **vs Pure LLM Synthetic Benchmarks**: Synthetic baselines lack grounded data and show unstable ranking consistency; SciCustom uses real scientific QA as a foundation to avoid "hallucinated" questions.
- **vs Embedding Retrieval**: Semantic vector retrieval finds surface-level relevant data but lacks scientific ontological structure, making it difficult to capture interdisciplinary compositional requirements.
- **Insight**: Domain evaluation systems should first establish a "knowledge unit index layer," building benchmark generation, model selection, and capability diagnosis upon a shared, interpretable knowledge space.

## Rating
- Novelty: ⭐⭐⭐⭐☆ The combination of ontological knowledge units and custom evaluation is solid with a clear problem definition.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Includes chemistry, medicine, case studies, taggers, human evaluation, and ablations, though verification across more scientific categories is still needed.
- Writing Quality: ⭐⭐⭐⭐☆ Well-structured with reasonable primary metrics and clear explanations of the motivation and limitations of binary search.
- Value: ⭐⭐⭐⭐⭐ Directly valuable for scientific LLM selection, automated domain benchmark construction, and application-specific capability diagnosis.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Capabilities and Evaluation Biases of Large Language Models in Classical Chinese Poetry Generation: A Case Study on Tang Poetry](capabilities_and_evaluation_biases_of_large_language_models_in_classical_chinese.md)
- [\[ACL 2026\] Reward Modeling for Scientific Writing Evaluation](reward_modeling_for_scientific_writing_evaluation.md)
- [\[ACL 2026\] Zero-shot Large Language Models for Automatic Readability Assessment](zero-shot_large_language_models_for_automatic_readability_assessment.md)
- [\[ACL 2026\] NovBench: Evaluating Large Language Models on Academic Paper Novelty Assessment](novbench_evaluating_large_language_models_on_academic_paper_novelty_assessment.md)
- [\[ACL 2026\] Question Difficulty Estimation for Large Language Models via Answer Plausibility Scoring](question_difficulty_estimation_for_large_language_models_via_answer_plausibility.md)

</div>

<!-- RELATED:END -->
