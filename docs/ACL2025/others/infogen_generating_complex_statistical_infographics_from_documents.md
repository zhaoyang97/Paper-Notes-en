---
title: >-
  [Paper Note] Infogen: Generating Complex Statistical Infographics from Documents
description: >-
  [ACL 2025][Infographics Generation] The Infogen framework is proposed to transform textual documents into complex statistical infographics (combinations of multiple subplots) using a two-stage design: first generating structured intermediate metadata with a fine-tuned LLM, and then iteratively generating final infographic code using an LLM code generator and a feedback module.
tags:
  - "ACL 2025"
  - "Infographics Generation"
  - "Data Visualization"
  - "LLM Code Generation"
  - "Metadata"
  - "Multi-Subplot Alignment"
date: 2026-05-08
content_hash: 3b2f726333dbfc75
---

# Infogen: Generating Complex Statistical Infographics from Documents

**Conference**: ACL 2025  
**arXiv**: [2507.20046](https://arxiv.org/abs/2507.20046)  
**Code**: None (samples of the Infodat dataset are public)  
**Area**: Other  
**Keywords**: Infographics Generation, Data Visualization, LLM Code Generation, Metadata, Multi-Subplot Alignment

## TL;DR

The Infogen framework is proposed to transform textual documents into complex statistical infographics (combinations of multiple subplots) using a two-stage design: first generating structured intermediate metadata with a fine-tuned LLM, and then iteratively generating final infographic code using an LLM code generator and a feedback module.

## Background & Motivation

Statistical infographics are powerful tools for translating complex data into intuitive visualizations. Existing AI-driven visualization methods (e.g., LIDA, ChartGPT) primarily focus on generating single simple charts (bar, line, etc.) from structured data (CSV/tables). However, in real-world scenarios, users often need to generate complex statistical infographics containing multiple subplots (e.g., bar chart + pie chart + line chart) from unstructured text documents.

Such tasks present several challenges:
1. Identifying and extracting statistical data from long text.
2. Determining the number, type, and content of subplots.
3. Arranging multiple subplots into a visually coherent, overall layout.

The authors argue that direct text-to-infographic generation yields low-quality results. Introducing structured intermediate metadata as a bridge can significantly improve generation quality.

## Method

### Overall Architecture

Infogen consists of two main modules:
1. **Metadata Generation Module**: Converts text documents into structured metadata $M = f(T)$
2. **Code Generation Module**: Converts metadata into executable Python code $C = g(M)$

The entire pipeline is $C = g(f(T))$

### Key Designs

1. **Metadata Definition**: Includes the title of the infographic, a text summary, and detailed information for each subplot—chart type (line, bar, pie, etc.), axis labels, data points, alignment, position, font, background color, etc. Metadata serves as the blueprint guiding the final code generation.

2. **Three-Stage Metadata Generation**:

    - **QLoRA Fine-tuning**: Fine-tunes three LLMs—Qwen-2 Large (72B), LLAMA 3 (70B), and Phi-3 Medium—using QLoRA to optimize the cross-entropy loss.
    - **DPO Alignment**: Generates two metadata outputs (with different temperatures) from the fine-tuned models for each data point. GPT-3.5 Turbo ranks them to create a synthetic preference dataset, and the models are further fine-tuned using DPO loss.
    - **Ranker LLM**: Uses fine-tuned LLAMA 3 (70B) to evaluate outputs from the three DPO models and select the most accurate result, resolving the hallucination issue inherent in single-model generation.

3. **Dual-Module Code Generation**:

    - **Coder Module**: Uses GPT-4o to translate metadata into Python code (using Plotly/Plotnine libraries) via in-context learning. This includes subplot configuration, data integration, and layout composition.
    - **Feedback Module**: Reviews the generated code for consistency with the metadata, checking data mapping, subplot attributes, and layout coherence, and provides revision suggestions. It refines the code iteratively for up to 5 rounds.

### Loss & Training

- Fine-tuning Phase: Standard cross-entropy loss $\mathcal{L} = -\sum_{i=1}^{N} y_i \log(\hat{y}_i)$
- DPO Phase: DPO loss on the synthetic preference dataset, using GPT-3.5 Turbo as the preference annotator.
- Training Hyperparameters: batch size=2, gradient accumulation=4, lr=2e-4, AdamW 8-bit, A100 80GB.

## Key Experimental Results

### Main Results

| Model | Subplot Accuracy | RSE | Title Rouge-L | Chart Type Accuracy | Statistical Accuracy |
|------|-----------|-----|------------|--------------|-----------|
| Infogen (large) | **74.69** | **1.80** | **0.56** | 84.23 | **89.56** |
| GPT-4o 20-shot | 56.73 | 2.06 | 0.36 | 72.10 | 87.77 |
| Phi3 QLoRA large DPO | 72.11 | 1.96 | 0.56 | 83.03 | 89.44 |
| LLAMA3 QLoRA large DPO | 68.65 | 2.05 | 0.55 | 82.98 | 88.27 |
| In-context merge | 65.57 | 2.24 | 0.51 | 83.46 | 88.79 |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Prompting vs Fine-tuning | Subplot Accuracy 56.73→72+ | Fine-tuning is significantly superior to few-shot prompting |
| Without DPO vs With DPO | Phi3: 63.46→72.11 | DPO alignment brings a significant improvement |
| Small vs Large LLM | Subplot Accuracy 37~54→63~72 | Large models consistently outperform small models |
| Infogen small vs large | 59.2→74.69 | Multi-LLM collaboration + large models yields substantial benefits |
| BM25 clustering selection vs Random | 55.76→57.69 | Curated selection brings a modest but consistent improvement |

Human Evaluation (5-point scale):

| Model | Readability | Visual Appeal | Accurate Data Alignment |
|------|--------|-----------|-------------|
| Infogen | **4.1** | **3.8** | **4.1** |
| Phi3 (DPO) | 3.7 | 3.2 | 3.4 |
| GPT-4o (20-shot) | 3.4 | 2.8 | 2.4 |

### Key Findings

1. Fine-tuned LLMs consistently outperform prompting methods, including GPT-4o few-shot.
2. DPO alignment brings substantial improvements for all evaluated models, even when preference data is synthesized by GPT-4o.
3. Multi-LLM collaboration (Ranker selecting the best) performs better than a single model, demonstrating the advantage of an ensemble.
4. The Feedback module successfully resolves text overlapping, layout misalignment, and other issues in the code.
5. The Infogen framework outperforms individual models even when using smaller LLMs.

## Highlights & Insights

- **Intermediate Metadata as a Bridge**: This is the core insight of this paper. Directly translating text to code is overly challenging; using metadata as a structured intermediate representation simplifies the task, analogous to "content planning."
- **Three-Stage Progressive Optimization**: The progression of Fine-tuning → DPO → Ranker is a clever engineering design.
- **Iterative Refinement with Feedback Module**: Borrowing the concept of code debugging, this mirrors the currently popular LLM agent self-correction paradigm.
- **Dataset Construction Methodology**: The semi-automated synthetic data pipeline (GPT-4o generation + manual verification) serves as a valuable reference.

## Limitations & Future Work

1. The Infodat dataset is limited in scale (3,463 samples), which may impact domain generalization.
2. There is still room for improvement in subplot alignment, as misalignment can lead to the loss of key structural details in the data.
3. Context-based custom selection of templates is not supported.
4. The code generation module relies heavily on GPT-4o's in-context learning, which incurs high costs.
5. The evaluation metrics are custom-designed and have not yet been widely validated by the research community.

## Related Work & Insights

- While LIDA and ChartGPT handle simple chart generation, Infogen extends the problem to complex multi-subplot infographics.
- The approach of using metadata as an intermediate representation draws inspiration from findings in instruction tuning where "explicit instructions improve quality."
- The dual-agent framework in DataNarrative shares a similar philosophy with Infogen's coder+feedback module design.
- Future work can extend this approach to vertical domains such as healthcare and finance.

## Rating

- Novelty: ⭐⭐⭐• Formulates the text-to-complex-infographics task for the first time; intermediate metadata design is highly reasonable.
- Experimental Thoroughness: ⭐⭐⭐• Comprehensive evaluation covering automatic metrics, human evaluation, and qualitative analysis.
- Writing Quality: ⭐⭐⭐• Structural diagrams are clear, though some details are scattered in the appendix.
- Value: ⭐⭐⭐• High practical value; the dataset and task definition contribute solid benchmarks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Statistical Deficiency for Task Inclusion Estimation](statistical_deficiency_task_inclusion.md)
- [\[ACL 2025\] Generating Synthetic Relational Tabular Data via Structural Causal Models](generating_synthetic_relational_tabular_data_via_structural_causal_models.md)
- [\[ACL 2025\] Generating Plausible Distractors for Multiple-Choice Questions via Student Choice Prediction](distractor_gen_multiple_choice.md)
- [\[NeurIPS 2025\] Statistical Inference Under Performativity](../../NeurIPS2025/others/statistical_inference_under_performativity.md)
- [\[NeurIPS 2025\] Robust Sampling for Active Statistical Inference](../../NeurIPS2025/others/robust_sampling_for_active_statistical_inference.md)

</div>

<!-- RELATED:END -->
