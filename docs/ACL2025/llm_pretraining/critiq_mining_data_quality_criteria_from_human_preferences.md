---
title: >-
  [Paper Note] CritiQ: Mining Data Quality Criteria from Human Preferences
description: >-
  [ACL 2025][LLM Pretraining][Data Selection] CritiQ proposes an automatic data quality criteria mining method based on agent collaboration. With only about 30 human preference annotation pairs, it can automatically discover interpretable data quality criteria and train a scorer for efficient data selection, significantly improving the downstream performance of Llama 3.1 in code, math, and logic domains.
tags:
  - "ACL 2025"
  - "LLM Pretraining"
  - "Data Selection"
  - "Quality Criteria Mining"
  - "Human Preferences"
  - "Agent Collaboration"
  - "Interpretability"
date: 2026-05-08
content_hash: e469ece3293e7dc4
---

# CritiQ: Mining Data Quality Criteria from Human Preferences

**Conference**: ACL 2025  
**arXiv**: [2502.19279](https://arxiv.org/abs/2502.19279)  
**Code**: [https://github.com/KYLN24/CritiQ](https://github.com/KYLN24/CritiQ)  
**Area**: Data Quality / LLM Training  
**Keywords**: Data Selection, Quality Criteria Mining, Human Preferences, Agent Collaboration, Interpretability

## TL;DR

CritiQ proposes an automatic data quality criteria mining method based on agent collaboration. With only about 30 human preference annotation pairs, it can automatically discover interpretable data quality criteria and train a scorer for efficient data selection, significantly improving the downstream performance of Llama 3.1 in code, math, and logic domains.

## Background & Motivation

**Background**: High-quality data is crucial for language model performance and is one of the core factors determining the upper bound of model capabilities. Current mainstream data selection methods include human-designed heuristic rules (such as length filtering and deduplication), perplexity-based filtering using existing models, training classifiers to distinguish data quality, and using elaborately designed prompts to prompt LLMs for quality assessment.

**Limitations of Prior Work**: These methods each have distinct limitations. Heuristic rules require extensive expert experience and are difficult to generalize to new domains; perplexity-based methods depend on the distribution of pre-existing models, potentially introducing circular bias; classifier-based methods require large amounts of human annotation and lack transparency in criteria; prompt-engineering-based approaches rely on the trial-and-error experience of engineers, and the evaluation criteria remain implicit and uninterpretable. Crucially, the data quality criteria produced by these methods are either non-existent (black-box classifiers) or non-reusable (prompt-specific), preventing human experts from reviewing and accumulating them.

**Key Challenge**: Data selection requires explicit and interpretable quality criteria. However, existing methods either lack explicit criteria (perplexity, classifiers) or rely on fixed, non-evolving criteria (handcrafted rules), lacking a mechanism that can automatically discover and iteratively optimize quality criteria from a small amount of human feedback.

**Goal**: To design a system that can automatically mine interpretable and reusable verbal data quality criteria from an extremely small amount of human preference annotations (~30 pairs) and convert these criteria into efficient data selection tools.

**Key Insight**: The authors observe that when judging data quality, humans rely on an implicit system of criteria (such as code readability, logical rigor, completeness of problem-solving steps). Although these criteria are difficult to exhaustively list at once, they can be progressively mined and refined by analyzing human preference choices.

**Core Idea**: To use a multi-agent collaborative system (CritiQ Flow) to iteratively mine quality criteria from human preference pairs—where a Manager Agent proposes and evolves criteria hypotheses, and Worker Agents validate the effectiveness of the criteria by performing pairwise judgments, continuously optimizing the set of criteria through multiple rounds of iteration.

## Method

### Overall Architecture

The methodology of CritiQ consists of two main stages. The first stage is **CritiQ Flow**: taking a small set of human preference annotation pairs (about 30 pairs, with each pair comprising two data samples and a human judgment on which is better) as input, it mines a set of verbalized quality criteria through iterative multi-agent collaboration. The second stage is **CritiQ Scorer**: using the mined criteria, it labels a large volume of data via agents and then trains a lightweight scoring model to perform efficient quality scoring and selection on massive datasets.

### Key Designs

1. **CritiQ Flow — Multi-Agent Quality Criteria Mining**:

    - **Function**: Automatically discovers and iteratively optimizes data quality criteria from a small number of human preference pairs.
    - **Mechanism**: The system consists of one Manager Agent and multiple Worker Agents. The Manager Agent is responsible for reflecting on current judgment results and proposing new quality criteria hypotheses based on them (e.g., "code should have clear variable naming", "mathematical reasoning steps should be complete"). The Worker Agents are responsible for making pairwise judgments on human preference pairs based on the given set of criteria. In each iteration, the Worker's judgments are compared with human annotations, and the Manager analyzes the deficiencies of the criteria based on error cases, adding new criteria or modifying existing ones. This process repeats until the criteria set achieves satisfactory accuracy on the validation set.
    - **Design Motivation**: It is difficult for a single Agent to simultaneously propose and validate criteria. Splitting the roles into a Manager and Workers creates a "hypothesis-verification" scientific discovery loop, systematically exploring the criteria space.

2. **Knowledge Base Boosting**:

    - **Function**: Accelerates the convergence of CritiQ Flow using quality criteria discovered in prior work.
    - **Mechanism**: Known quality criteria are extracted from prior research papers related to data quality (such as studies on QuRating) to construct a structured knowledge base. During the initialization of CritiQ Flow, the Manager Agent can refer to existing criteria in the knowledge base as a starting point instead of exploring completely from scratch. The knowledge base is formatted in JSON, containing criteria names and descriptions.
    - **Design Motivation**: Mining criteria entirely from scratch is inefficient and may miss validated, important dimensions. Utilizing accumulated domain knowledge can significantly improve efficiency and quality of the criteria.

3. **CritiQ Scorer — Efficient Quality Scorer**:

    - **Function**: Translates verbalized criteria into a numerical scoring tool applicable to large-scale datasets.
    - **Mechanism**: First, the criteria mined by CritiQ Flow are used to guide Worker Agents in labeling a large number of data points (through pairwise comparison), generating labeled data with quality ranking information. Then, a scorer in the form of a reward model is trained on models like Qwen2, which takes a data sample as input and outputs a quality score. The training employs a standard pairwise ranking loss. The trained Scorer can rapidly score massive datasets and perform Gumbel distribution sampling based on scores to select subsets of training data.
    - **Design Motivation**: Although LLM Agent annotation is accurate, it is slow and costly. Training a lightweight Scorer preserves criteria accuracy while enabling large-scale data selection.

### Loss & Training

CritiQ Scorer is trained using a pairwise ranking loss: given a pair of data samples $(x_w, x_l)$ (where $x_w$ is judged to be of higher quality), the model learns to output $f(x_w) > f(x_l)$. Specifically, a margin-based loss function is adopted. Training is conducted with distributed training via DeepSpeed ZeRO-2, parallelized across 8 GPUs, with a learning rate of 2e-5, trained for 3 epochs, and a warmup ratio of 0.2.

## Key Experimental Results

### Main Results

The effectiveness of data selection is verified across three domains: Code, Math, and Logic. A continually trained Llama 3.1 model is evaluated to assess the quality of selected data.

| Domain | Method | Human Preference Accuracy | Downstream Task Gain |
|------|------|---------------|-------------|
| Code | Random Sampling | — | Baseline |
| Code | Perplexity | 61.2% | +1.3% |
| Code | Classifier | 67.8% | +2.1% |
| Code | CritiQ | 82.5% | +4.7% |
| Math | Random Sampling | — | Baseline |
| Math | Perplexity | 58.9% | +0.8% |
| Math | CritiQ | 79.3% | +3.9% |
| Logic | Random Sampling | — | Baseline |
| Logic | CritiQ | 80.1% | +3.2% |

### Ablation Study

| Configuration | Code Accuracy | Math Accuracy | Description |
|------|------------|------------|------|
| Full CritiQ Flow | 82.5% | 79.3% | Complete system |
| w/o Knowledge Base | 76.8% | 73.5% | Removes knowledge base boosting, mining from scratch |
| w/o Reflection Mechanism | 74.2% | 71.8% | Manager does not perform error analysis or criteria correction |
| w/o Majority Voting | 78.1% | 75.6% | Worker uses only single judgment without voting |
| Fixed Criteria (No Iteration) | 70.5% | 67.2% | Uses only initial criteria without iterative optimization |

### Key Findings

- Knowledge base boosting contributes approximately 5-6% improvement in accuracy, showing that domain prior-knowledge is valuable for criteria mining. However, even without the knowledge base, CritiQ still significantly outperforms perplexity and classifier methods.
- The reflection mechanism is the second largest contributing factor; the Manager Agent's ability to adjust criteria based on error cases is crucial.
- Criteria show interesting evolutionary patterns during the iterative process: starting with broad criteria in the early stages (e.g., "code should be correct") and gradually refining into more specific criteria (e.g., "code should have error handling mechanisms", "variable naming should be semantic").
- Achieving 80%+ human preference prediction accuracy using only about 30 annotation pairs demonstrates extremely high data efficiency.
- The mined criteria are interpretable and reusable, making them directly available for human expert review and editing.

## Highlights & Insights

- **High-quality criteria mining at extremely low annotation cost**: Automatically discovering effective quality criteria requires only about 30 human preference annotation pairs, a cost significantly lower than the hundreds or thousands of annotations required for training classifiers. This dramatically lowers the barrier and cost for data selection.
- **Interpretability as a core advantage**: Unlike black-box methods like perplexity or classifiers, CritiQ produces quality criteria described in natural language, which can be directly read, understood, modified, and reused by humans, bringing exceptional value to production environments.
- **The "hypothesis-verification" paradigm of multi-agent collaboration**: The Manager-Worker design pattern can be extended to other areas requiring implicit rule discovery from limited feedback, such as automatic annotation guideline generation and evaluation criteria discovery.

## Limitations & Future Work

- Currently, validation has only been conducted in three relatively structured domains (Code, Math, and Logic); its effectiveness in more open-ended domains (such as creative writing or dialogue) has not been tested.
- The iterative optimization of CritiQ Flow relies on powerful LLMs (like GPT-4) acting as Manager and Worker, leading to non-trivial API call costs.
- Knowledge base construction still requires manual extraction of criteria from previous literature, which is not yet fully automated.
- The coverage of the criteria depends on the diversity of the initial preference annotation pairs; if the pairs do not cover all quality dimensions, important criteria might be missed.

## Related Work & Insights

- **vs QuRating**: QuRating uses LLMs to score data on predefined quality dimensions, which are manually designed. CritiQ's criteria are automatically mined, making them more flexible and capable of discovering dimensions that humans might overlook.
- **vs DSIR/DsDm**: These methods use perplexity or distribution matching for data selection, representing black-box numerical methods. CritiQ outputs interpretable language criteria, holding fundamental advantages in transparency and adjustability.
- **vs AlpaGasus/LIMA**: These works select small subsets of high-quality data to train models, but their selection criteria remain implicit. CritiQ can provide automated, interpretable selection criteria for such tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The idea of using multi-agent collaboration to automatically mine data quality criteria is highly novel, elevating data selection from "judging quality" to "discovering criteria".
- Experimental Thoroughness: ⭐⭐⭐⭐ Complemented by validation across three domains, along with detailed ablation and criteria evolution analyses, the experimental design is highly complete.
- Writing Quality: ⭐⭐⭐⭐ The methodology description is clear, the logic chain of motivation is robust, and the charts and tables are well-designed.
- Value: ⭐⭐⭐⭐⭐ It resolves the core pain point of uninterpretable criteria in data selection. Working efficiently with only 30 pairs of annotations gives it extraordinary practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Predict Training Data Quality via Its Geometry in Metric Space](../../NeurIPS2025/llm_pretraining/predict_training_data_quality_via_its_geometry_in_metric_space.md)
- [\[AAAI 2026\] ELSPR: Evaluator LLM Training Data Self-Purification on Non-Transitive Preferences](../../AAAI2026/llm_pretraining/elspr_evaluator_llm_training_data_self-purification_on_non-transitive_preference.md)
- [\[ACL 2025\] Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning](data_whisperer_data_selection.md)
- [\[ICLR 2026\] GneissWeb: Preparing High Quality Data for LLMs at Scale](../../ICLR2026/llm_pretraining/gneissweb_preparing_high_quality_data_for_llms_at_scale.md)
- [\[ACL 2025\] Data-Constrained Synthesis of Training Data for De-Identification](data-constrained_synthesis_of_training_data_for_de-identification.md)

</div>

<!-- RELATED:END -->
