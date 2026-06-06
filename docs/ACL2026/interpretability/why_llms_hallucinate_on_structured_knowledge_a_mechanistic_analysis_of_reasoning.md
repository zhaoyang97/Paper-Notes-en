---
title: >-
  [Paper Note] Why LLMs Hallucinate on Structured Knowledge: A Mechanistic Analysis of the Reasoning Process
description: >-
  [ACL 2026][Interpretability][Hallucination Detection] The paper reveals the internal failure mechanisms of LLMs when processing linearized structured knowledge through two mechanistic metrics (Structural Shortcut Relianc…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "Hallucination Detection"
  - "Structured Knowledge"
  - "Attention Mechanism"
  - "Feed-Forward Networks"
  - "Knowledge Reasoning"
date: 2026-05-08
content_hash: f2074bdeba1a7d1c
---

# Why LLMs Hallucinate on Structured Knowledge: A Mechanistic Analysis of the Reasoning Process

**Conference**: ACL 2026  
**arXiv**: [2605.26362](https://arxiv.org/abs/2605.26362)  
**Code**: https://github.com/ShanghaoLi0913/struhall-mechanism  
**Area**: Mechanistic Interpretability  
**Keywords**: Hallucination Detection, Structured Knowledge, Attention Mechanism, Feed-Forward Networks, Knowledge Reasoning

## TL;DR

The paper reveals the internal failure mechanisms of LLMs when processing linearized structured knowledge through two mechanistic metrics (Structural Shortcut Reliance, SSR, and Semantic Alignment Score, SAS) and constructs a lightweight hallucination detector based on these signals.

## Background & Motivation

**Background**: Modern RAG frameworks and LLM reasoning systems commonly use linearization strategies to handle structured knowledge. Knowledge graphs are transformed into triple sequences, and tables are flattened into natural language text, as the Transformer architecture is inherently based on sequential token representation operations.

**Limitations of Prior Work**: A critical issue is that LLMs still frequently generate hallucinated answers even when provided with sufficient and accurate structured knowledge as context. Existing literature focuses mostly on external intervention methods (retrieval augmentation, prompt engineering) but lacks a deep understanding of the underlying mechanisms—why do models ignore explicitly provided structured knowledge?

**Key Challenge**: The linearization process breaks the explicit relational constraints of structured data, leading to the internal inability of the model to utilize this knowledge correctly. The inductive bias of Transformers tends to model sequential structures in natural language but adapts poorly to artificially flattened knowledge structures. When the context contains both critical evidence and distracting information, models often lean towards fast "shortcuts" rather than complete reasoning.

**Goal**: To decouple the model's external evidence utilization from internal parametric memory through mechanistic interpretability methods, discovering the systematic internal dynamics of hallucination generation.

**Key Insight**: The authors start from the two core functional modules of Transformers—Self-Attention heads (selectively attending to input subsets) and Feed-Forward Networks (storing and integrating knowledge). The hypothesis is that hallucinations originate from a systematic imbalance between attention allocation and semantic evidence integration.

**Core Idea**: Introduce two diagnostic metrics to quantify this imbalance, thereby transforming black-box phenomena into interpretable mechanistic signals.

## Method

### Overall Architecture

The paper adopts a causal analysis framework, performing forward passes under frozen model parameters while collecting attention weights and hidden representations of intermediate layers. For each generated answer, SSR and SAS metrics are calculated, followed by statistical analysis and visualization to investigate the correlation between these metrics and hallucination labels. Finally, a simple XGBoost classifier is trained based on these two metrics as a hallucination detector.

### Key Designs

1.  **Structural Shortcut Reliance (SSR)**:

    - **Function**: Quantifies the extent to which the model over-relies on minimal structural cues when processing linearized structured knowledge. The core idea is to divide input tokens into two categories: ① Structural Shortcut (SS)—the minimal set of paths connecting the question to the answer; ② Contextual Prompts ($\bar{S}$)—other knowledge providing relational context and global constraints.
    - **Mechanism**: For each answer token position, the quality of attention assigned to SS versus $\bar{S}$ by each attention head in each layer is calculated. SSR is defined as the average difference: $\text{SSR}=\frac{1}{L \cdot H \cdot |A|}\sum_{l=1}^{L}\sum_{h=1}^{H}\sum_{i \in A}(\alpha_{l,h,i,S}-\alpha_{l,h,i,\bar{S}})$. Here $\alpha_{l,h,i,S} = \sum_{j \in S}\alpha_{l,h,i,j}$ is the sum of attention weights from answer position $i$ to all positions in SS. SSR ranges between [-1, 1], where positive values indicate excessive reliance on shortcuts.
    - **Design Motivation**: Intuitively, if a model only focuses on the shortest path and ignores surrounding evidence, it cannot perform complete factual verification. This concentration of attention implies the model has adopted a "shortcut learning" strategy, bypassing necessary semantic validation.

2.  **Semantic Alignment Score (SAS)**:

    - **Function**: Measures the degree of semantic alignment between the model's internal representations and the input structured knowledge during generation. Compared to attention, which only reflects information routing, this metric directly captures whether representations within the FFN are correctly "grounded" by the knowledge.
    - **Mechanism**: First, define the Supporting Context Set (SCS), starting from the core structural prompt SS and including its 1-hop neighboring triples. For each generated answer token, its hidden representation $\mathbf{h}_t$ at the penultimate layer is extracted, and the cosine similarity with the encoding $\mathbf{g}_i$ of each knowledge unit $U_i$ in the SCS is calculated, taking the maximum value: $\text{SAS}(y_t)=\max_{U_i \in \mathcal{E}}\cos(\mathbf{h}_t, \mathbf{g}_i)$. Sentence-level SAS is the average similarity across all answer tokens. Values close to 1 indicate the representation is well-grounded by knowledge, while values close to -1 indicate the representation has drifted toward parametric memory.
    - **Design Motivation**: When linearization weakens the semantic scaffolding, the FFN is easily dominated by parametric priors learned during training, causing generated representations to deviate from the evidence. This is the root cause of knowledge-driven hallucinations.

3.  **Complementarity & Four-Quadrant Analysis**:

    - **Function**: SSR and SAS capture two independent dimensions of hallucinations: selective failure of attention allocation vs. semantic drift of the representation layer.
    - **Mechanism**: The Pearson correlation coefficient between the two is only -0.26, indicating complementary rather than redundant signals. Four-quadrant analysis divides the output space into four regions, each corresponding to different failure modes.
    - **Design Motivation**: The hallucination rate is lowest (5%) in Q2 (low SSR + high SAS), corresponding to a healthy state of "broad attention + strong semantic alignment"; Q3 (low SSR + low SAS) has the highest hallucination rate (22.2%), representing the most dangerous state of "scattered attention but failed semantic fusion"; Q4 (high SSR + low SAS) carries medium risk (10.9%), showing that attention concentration alone is insufficient to cause severe hallucinations without accompanying representation drift.

## Key Experimental Results

### Main Results

| Metric | Hallucinated Output | Truthful Output | t-statistic | p-value |
| :--- | :--- | :--- | :--- | :--- |
| SSR | 0.745 | 0.683 | -3.31 | <0.001 |
| SAS | 0.343 | 0.412 | 10.96 | <1e-26 |

**Key Findings**: Hallucinations and truthful outputs show statistically significant distributional differences in both SSR and SAS, confirming that both metrics are reliable discriminative signals. However, they act in opposite directions, reflecting the collaborative failure of attention and representation.

### Four-Quadrant and Cross-dataset Generalization

| Quadrant | Configuration | Hallucination Rate (1-hop) | Hallucination Rate (2-hop) | Hallucination Rate (Table) |
| :--- | :--- | :--- | :--- | :--- |
| Q1 | High SSR, High SAS | 9.5% | 36.4% | 84.1% |
| Q2 | Low SSR, High SAS | 5.0% | 14.8% | 80.9% |
| Q3 | Low SSR, Low SAS | 22.2% | 18.4% | 87.5% |
| Q4 | High SSR, Low SAS | 10.9% | 54.4% | 85.9% |

**Key Findings**: Although absolute hallucination rates vary with task complexity, the relative importance of SAS remains stable—the high SAS quadrant (Q2) consistently performs best. This suggests that semantic alignment is a more universal predictor of hallucinations, while SSR reflects task-dependent failure modes.

### Hallucination Detection Performance

Comparison of the SSR+SAS based detector against existing baselines on MetaQA-1hop:

- **Confidence-based methods** (Perplexity, Token Confidence): High recall but low precision, tending to over-predict hallucinations.
- **Semantic similarity methods** (BERTScore, Embedding Distance, NLI): Moderate performance, unable to distinguish effectively.
- **Ours (SSR + SAS)**: AUC=0.834, F1=0.539 on LLaMA2-7B; AUC=0.853, F1=0.461 on Qwen2.5-7B.

Advantages: No model fine-tuning required, calculated within a single forward pass, and logically interpretable (failure causes correspond to specific mechanisms).

## Highlights & Insights

- **Leap from Observation to Mechanism**: Traditional work stays at the descriptive level of "LLMs hallucinate," whereas this paper goes deeper to reveal parallel failure trajectories: attention over-concentration and representation drift.
- **Discovery of Metric Complementarity**: Although SSR and SAS are weakly correlated, they capture different failure modes. This suggests that when designing multi-angle diagnostic tools, independence between signals should be prioritized over simple stacking.
- **Theory-to-Application Closed Loop**: Mechanistic discoveries are directly translated into deployable detectors without retraining, which is valuable for resource-constrained scenarios.
- **Cross-format Generalization**: The same framework generalizes from graphs to tables without modification, indicating it diagnoses universal failure modes of Transformers processing *any* linearized structured knowledge.
- **"Insufficiency of Minimal Path" Insight**: The Q4 region still produces hallucinations despite focused attention, confirming that merely finding the shortest reasoning path is insufficient—the model must truly understand and fuse this path in the representation layer.

## Limitations & Future Work

- **Model Type Constraints**: Only decoder-only models were analyzed; encoder-decoder or specialized graph encoders might exhibit different characteristics.
- **Linearization Paradigm Constraints**: The study assumes knowledge must be converted into sequential token strings. These mechanisms might not apply if graph structural representations are maintained directly inside the model.
- **Insufficient Causal Evidence**: The current analysis is a correlation study and has not verified whether SSR/SAS are the true causes of hallucinations through intervention experiments.
- **Incomplete Scale Coverage**: Experiments are limited to 7B models; whether larger models (above 70B) behave consistently remains to be verified.
- **Future Directions**: Targeted improvement strategies could be explored, such as explicitly penalizing high SSR-low SAS configurations during training or dynamically adjusting attention bias during inference for more uniform distribution.

## Related Work & Insights

- **vs. Traditional Hallucination Detection (Perplexity, Self-consistency)**: Traditional methods rely on model output statistics, while Ours starts from internal mechanisms, making it better at locating root problems.
- **vs. Other Interpretability Work**: Previous studies often focused on single components (e.g., attention visualization). This paper emphasizes the *interactive failure* of multiple components—complete diagnosis requires monitoring both attention and representation.
- **Inspiration from KGQA**: The knowledge graph community has long known that "the shortest path may not uniquely determine an answer," but the LLM community lacked this awareness. Ours fills this gap.
- **Inspiration**: This framework can be migrated to other tasks involving structured knowledge (e.g., table reasoning, code understanding) by simply redefining the extraction rules for core structural prompts.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ The internal mechanism of diagnostics using complementary dual metrics is an original perspective; joint analysis of attention and representation dimensions is a first.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Validated across 1-hop/2-hop/4-hop and graph/table settings with detailed ablation. However, lacks causal intervention experiments (correlation-only).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Logical flow from phenomena to mechanisms to applications forms a complete story. Formula notation is accurate and tables have high information density.
- **Value**: ⭐⭐⭐⭐⭐ Provides both theoretical insight (understanding hallucination mechanisms) and practical tools (deployable detectors), inspiring future work on hallucination mitigation and LLM reliability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation](understanding_new-knowledge-induced_factual_hallucinations_in_llms_analysis_and_.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Mechanistic Interpretability of Large-Scale Counting in LLMs through a System-2 Strategy](mechanistic_interpretability_of_large-scale_counting_in_llms_through_a_system-2_.md)
- [\[CVPR 2026\] Why Does It Look There? Structured Explanations for Image Classification](../../CVPR2026/interpretability/why_does_it_look_there_structured_explanations_for_image_classification.md)
- [\[ICML 2026\] Understanding LoRA as Knowledge Memory: An Empirical Analysis](../../ICML2026/interpretability/understanding_lora_as_knowledge_memory_an_empirical_analysis.md)

</div>

<!-- RELATED:END -->
