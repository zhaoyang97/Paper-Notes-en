---
title: >-
  [Paper Note] Retrieve to Explain: Evidence-driven Predictions for Explainable Drug Target Identification
description: >-
  [ACL 2025][Computational Biology][Retrieval-augmented prediction] Proposes R2E (Retrieve to Explain), a retrieval-based framework that scores and ranks candidate answers by retrieving evidence from a literature corpus and faithfully attributes predictions to supporting evidence using Shapley values, outperforming genetics and GPT-4 baselines in drug target identification tasks.
tags:
  - "ACL 2025"
  - "Computational Biology"
  - "Retrieval-augmented prediction"
  - "Shapley value explanation"
  - "Drug target identification"
  - "Evidence attribution"
  - "Clinical trial prediction"
date: 2026-05-08
content_hash: 747990da7614ca8a
---

# Retrieve to Explain: Evidence-driven Predictions for Explainable Drug Target Identification

**Conference**: ACL 2025  
**arXiv**: [2402.04068](https://arxiv.org/abs/2402.04068)  
**Area**: Computational Biology  
**Keywords**: Retrieval-augmented prediction, Shapley value explanation, Drug target identification, Evidence attribution, Clinical trial prediction

## TL;DR

Proposes R2E (Retrieve to Explain), a retrieval-based framework that scores and ranks candidate answers by retrieving evidence from a literature corpus and faithfully attributes predictions to supporting evidence using Shapley values, outperforming genetics and GPT-4 baselines in drug target identification tasks.

## Background & Motivation

In high-stake scientific discovery scenarios (such as drug target identification), model predictions must be explainable so that human experts can review the supporting evidence before taking action. Although existing language models can answer factual questions, they lack the ability to **quantitatively and faithfully** compare the credibility of answers when facing complex scientific questions with multiple plausible answers and varying evidence strengths.

Key challenges faced by drug target identification:
- **Extremely high cost of failure**: About half of the drugs fail to show efficacy in human trials, typically due to ineffective target selection.
- **Human-in-the-loop**: Experts need to inspect the evidence and reasoning process behind each prediction.
- **Limitations of Prior Work**: Knowledge graph methods require constructing KGs, while parametric language models lack transparent evidence attribution.

## Method

### Overall Architecture

R2E consists of two modules: **Retriever** and **Reasoner**.

1. **Masked Entity-Linked Corpus**: Constructs an entity-linked corpus, masking 19,176 protein-coding gene entities across 160 million biomedical literature sentences.
2. **Retriever**: A Masked Language Model (MLM)-based text encoder that builds independent FAISS search indexes for each candidate entity. Given a user's cloze-style query, it retrieves the $k=64$ evidence paragraphs most similar to the query.
3. **Reasoner**: Fuses query-evidence pair embeddings using a convolutional layer, aggregates all evidence via a Set Transformer, and outputs binary classification probabilities through a linear layer and a sigmoid activation function.

### Key Designs

**1. Answer masking mechanism**: Masks the answer entity in both queries and evidence, forcing the model to make indirect inferences based solely on evidence content, ensuring predictions are based entirely on evidence rather than surface features of entity names.

**2. Shapley-value evidence attribution**: Since the feature space consists of the $k$ retrieved evidence passages (a relatively small feature space), permutation sampling can be used to efficiently approximate Shapley values, quantitatively attributing each prediction score to individual evidence. During training, random substitution with a `NULL` embedding (with a dropout rate sampled from $\text{Uniform}(0,1)$) is introduced, serving both as regularization and enabling the model to robustly handle missing features.

**3. Frequency bias correction**: Introduces a parameter $c \in [0, 1]$ to control frequency bias, where $c=0$ performs no correction and $c=1$ makes the ranking reflect the Pointwise Mutual Information (PMI) between the query and the answer. Experiments adopt $c=0.5$ (selected via the validation set).

**4. Multimodal evidence templating**: Converts non-textual data (such as genetic association data) into natural language sentences using simple templates (e.g., `"[MASK] is genetically associated with {MeSH name}"`), which are directly integrated into the retrieval corpus without retraining, making it easy to integrate diverse evidence sources.

**5. Closed-loop evidence auditing**: The explainability of R2E allows the use of LLMs (e.g., GPT-4) to automatically audit evidence with high Shapley values, filtering out irrelevant false positive evidence, thereby further improving prediction performance.

## Key Experimental Results

### Main Results

**Task 1: Gene Description Facts**

| Metric | FREQ | MCS | MLM | R2E-uncor | R2E-cor |
|------|------|------|------|-----------|---------|
| MRR | <0.001 | 0.176 | 0.167 | 0.202 | **0.260** |
| Mean Rank | 8252 | 1776 | 2208 | 937 | **599** |
| Hits@10 | <0.001 | 0.309 | 0.296 | 0.349 | **0.434** |
| Hits@200 | 0.013 | 0.622 | 0.590 | 0.701 | **0.776** |

**Task 2: Clinical Trial Outcomes**

| Model | Evidence Source | AUROC |
|------|----------|-------|
| Genetics baseline | Genetic data | 0.545 |
| FREQ | Literature | 0.561 |
| MCS | Literature | 0.623 |
| MLM | Literature | 0.630 |
| R2E-uncor | Genetic data | 0.579 |
| R2E-uncor | Literature | 0.629 |
| R2E-cor | Literature | 0.632 |
| R2E-cor | Literature + Genetics | 0.633 |
| R2E-audit | Literature + Genetics | **0.638** |

Dataset size: 1,449 successful + 4,222 failed clinical trials.

### Key Findings

1. **Using only the same genetic data, R2E significantly outperforms the genetics baseline** (AUROC 0.579 vs 0.545, $p<0.001$), demonstrating that representing gene-trait associations in natural language improves generalization ability.
2. **R2E significantly outperforms the few-shot CoT GPT-4 baseline**, which is not only highly computationally expensive but also sacrifices faithful explainability.
3. **Shapley values are highly consistent with GPT-4 relevance annotations**: AUROC = 0.824 for Gene Description Facts, AUROC = 0.665 for Clinical Trial Outcomes.
4. **Agreement rate between GPT-4 and human expert annotations**: 71.5% for Gene Description Facts, 82.2% for Clinical Trial Outcomes.
5. **Evidence auditing further improves performance**: R2E-audit shows significant improvement compared to R2E-cor ($p=0.004$).

## Highlights & Insights

- **The "retrieval-as-feature" design philosophy**: Treating retrieved evidence itself as the model's feature space allows traditional feature attribution methods (Shapley values) to naturally become evidence attribution methods, yielding an elegant system-level unification.
- **Indirect reasoning capability**: R2E can reason using indirect evidence—a target never directly associated with a disease might score highly because it regulates mechanisms related to that disease.
- **Templated multimodal fusion**: Converting structured data into natural language templates is an extremely concise way to achieve multimodal fusion without modifying the model architecture or retraining.
- **Auditable closed-loop system**: The explainability of R2E makes it an auditable system where GPT-4 or human experts can review and filter evidence, forming a virtuous cycle of "prediction $\rightarrow$ explanation $\rightarrow$ auditing $\rightarrow$ refinement".

## Limitations & Future Work

1. **Inference overhead**: Each answer requires independent vector search and forward propagation. The computational cost grows linearly with the size of the answer set (which can be mitigated by parallelization).
2. **Corpus dependency**: Performance is sensitive to the completeness of the underlying corpus, although explainability helps expose corpus biases.
3. **No task-specific fine-tuning**: Directly uses a general query template and is not fine-tuned for specific downstream tasks like clinical trials.
4. **Context window limits**: Currently, only sentence-level evidence is used; extending to paragraph-level evidence might further improve performance.

## Related Work & Insights

- **Retrieval-Augmented Language Models**: kNN-LM, Fusion-in-Decoder (FiD)—R2E distinguishes itself by being fully retrieval-based with explainability as its core design objective.
- **Explainability and Data Attribution**: SHAP, Data Shapley, SimplEx—R2E extends feature attribution to evidence at inference time.
- **Hypothesis Generation**: Knowledge graph methods (Paliwal 2020, Aliper 2023), word2vec materials science prediction (Tshitoyan 2019)—R2E operates directly on text without constructing KGs.
- **Drug Discovery**: Genetic clinical trial prediction (Nelson 2015, Minikel 2024), Graph Neural Networks (Aliper 2023).

## Rating

- **Novelty**: ⭐⭐⭐⭐ — Treating retrieval itself as the feature space to achieve faithful evidence attribution. The concept of "retrieval as explanation" is novel.
- **Technical Depth**: ⭐⭐⭐⭐ — The architectural design is thoroughly considered (masking mechanism, `NULL` embedding regularization, frequency bias correction, closed-loop evidence auditing).
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three public benchmarks, comparisons with multiple baselines, GPT-4 + human expert validation for explainability, and rich ablation studies.
- **Value**: ⭐⭐⭐⭐ — Demonstrates effectiveness in outperforming general industry methods in the high-stake field of drug target identification.
- **Overall Recommendation**: ⭐⭐⭐⭐ — A work that elegantly unifies retrieval, prediction, and explainability, offering key insights for high-stake AI-assisted decision-making.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Site4Drug: Predicting Drug-Binding Target Sites with an AI Agent](../../ICML2026/computational_biology/site4drug_predicting_drug-binding_target_sites_with_an_ai_agent.md)
- [\[ACL 2025\] LADDER: Language Driven Slice Discovery and Error Rectification in Vision Classifiers](ladder_language-driven_slice_discovery_and_error_rectification_in_vision_classif.md)
- [\[ICLR 2026\] GRAM-DTI: Adaptive Multimodal Representation Learning for Drug-Target Interaction Prediction](../../ICLR2026/computational_biology/gram-dti_adaptive_multimodal_representation_learning_for_drugtarget_interaction_.md)
- [\[AAAI 2026\] Constrained Best Arm Identification with Tests for Feasibility](../../AAAI2026/computational_biology/constrained_best_arm_identification_with_tests_for_feasibility.md)
- [\[NeurIPS 2025\] GFlowNets for Learning Better Drug-Drug Interaction Representations](../../NeurIPS2025/computational_biology/gflownets_for_learning_better_drug-drug_interaction_representations.md)

</div>

<!-- RELATED:END -->
