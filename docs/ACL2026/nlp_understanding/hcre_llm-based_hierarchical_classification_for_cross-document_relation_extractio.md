---
title: >-
  [Paper Note] HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction
description: >-
  [ACL 2026][NLP Understanding][Cross-Document Relation Extraction] The HCRE model is proposed to transform cross-document relation extraction from direct classification over a large relation set into layer-by-layer hierar…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Cross-Document Relation Extraction"
  - "Hierarchical Classification"
  - "Large Language Models"
  - "Error Propagation Mitigation"
  - "Predict-then-Verify Strategy"
date: 2026-05-08
content_hash: 7f471df840efc004
---

# HCRE: LLM-based Hierarchical Classification for Cross-Document Relation Extraction

**Conference**: ACL 2026  
**arXiv**: [2604.07937](https://arxiv.org/abs/2604.07937)  
**Code**: [https://github.com/XMUDeepLIT/HCRE](https://github.com/XMUDeepLIT/HCRE)  
**Area**: NLP Understanding  
**Keywords**: Cross-Document Relation Extraction, Hierarchical Classification, Large Language Models, Error Propagation Mitigation, Predict-then-Verify Strategy

## TL;DR
The HCRE model is proposed to transform cross-document relation extraction from direct classification over a large relation set into layer-by-layer hierarchical classification by constructing a hierarchical relation tree. A Predict-then-Verify reasoning strategy is designed to mitigate inter-layer error propagation, significantly outperforming SLM and LLM baselines on the CodRED dataset.

## Background & Motivation

**Background**: Cross-document relation extraction (RE) aims to identify relations between entities distributed across different documents. More than half of the relational facts in Wikidata span multiple documents. Existing methods primarily adopt the "Small Language Model (SLM) + Classifier" paradigm.

**Limitations of Prior Work**: Constraints in the language understanding capabilities of SLMs hinder further improvements in cross-document RE. Preliminary experiments by the authors found that directly applying LLMs to cross-document RE yields unsatisfactory results, even underperforming strong SLM baselines. In-depth analysis reveals the root cause to be the excessive number of predefined relations (277 in CodRED): (1) Numerous semantically similar relations are difficult to distinguish; (2) Enumerating all relations leads to excessively long inputs, distracting the LLM's attention from key document information.

**Key Challenge**: LLMs possess powerful language understanding capabilities but cannot effectively handle large-scale relation option sets, while SLMs can handle the sets but lack sufficient understanding.

**Goal**: Reduce the number of relation options the LLM must consider during each inference step while avoiding the error propagation issues inherent in hierarchical classification.

**Key Insight**: Preliminary experiments demonstrate that reducing the number of relation options significantly improves LLM performance (see Figure 4), which inspired the design of hierarchical classification.

**Core Idea**: Construct a hierarchical relation tree to allow the LLM to perform top-down reasoning for the target relation layer by layer, considering only a small number of options at each level. Simultaneously, a Predict-then-Verify strategy is used to mitigate inter-layer error propagation through multi-视角 verification.

## Method

### Overall Architecture
HCRE consists of two core components: (1) An LLM $\mathcal{M}_1$ used for relation prediction; (2) A hierarchical relation tree constructed from the predefined relation set. Guided by the tree, the LLM performs classification layer by layer until reaching a leaf node, which represents the target relation. During inference, a Predict-then-Verify strategy is used to enhance reliability at each layer.

### Key Designs

1.  **Hierarchical Relation Tree Construction**:

    - **Function**: Organizes 277 predefined relations into a tree structure to reduce the number of options for classification at each layer.
    - **Mechanism**: An advanced LLM (e.g., GPT-4o) is used to build the tree layer by layer. First, partition criteria $C_l$ (e.g., "by domain") are generated for each layer. Then, relations under each current node are grouped according to the criteria to generate and name child nodes. The second layer is specifically designed with two nodes: "Valid Relation" and "No Valid Relation," explicitly separating positive samples from NA to alleviate label imbalance. This process proceeds recursively until the maximum depth $L$ is reached.
    - **Design Motivation**: The number of child nodes for each parent node is far smaller than the total number of relations, effectively reducing the LLM's classification difficulty.

2.  **Predict-then-Verify Reasoning Strategy**:

    - **Function**: Mitigates error propagation at each tree level through multi-view verification.
    - **Mechanism**: Divided into two steps—Prediction step: The LLM selects the optimal node $\hat{r}_{1st}$ and the sub-optimal node $\hat{r}_{2nd}$ from the current layer's option set $\mathcal{R}_l$. Verification step: $\hat{r}_{1st}$ and $\hat{r}_{2nd}$ are replaced by their respective child nodes to construct three verification option sets $\mathcal{R}_l^{v_1}, \mathcal{R}_l^{v_2}, \mathcal{R}_l^{v_3}$. The LLM is then asked to choose the optimal node for each verification set. If more than half of the auxiliary verification nodes are semantically consistent with $\hat{r}_{1st}$, the prediction is confirmed; otherwise, $\hat{r}_{1st}$ is removed and the process repeats.
    - **Design Motivation**: Verification option sets provide finer-grained semantic information (sub-node level), helping the LLM discern subtle differences between optimal and sub-optimal choices.

3.  **Training Data Construction**:

    - **Function**: Constructs training samples for hierarchical classification and verification steps.
    - **Mechanism**: For an original training sample $(x, \mathcal{R}, r)$, the path from the root to the leaf is identified and expanded into $L-1$ layer-by-layer training samples to form $\mathcal{D}_1$. For each sample in $\mathcal{D}_1$, the verification step is simulated to generate three verification samples to form $\mathcal{D}_2$. The LLM is fine-tuned on the merged dataset $\mathcal{D}_1 \cup \mathcal{D}_2$.
    - **Design Motivation**: Explicitly training the verification capability allows the LLM to learn how to effectively utilize fine-grained information for verification during inference.

### Loss & Training
Standard language model fine-tuning loss (cross-entropy) is used for supervised fine-tuning on $\mathcal{D}_1 \cup \mathcal{D}_2$.

## Key Experimental Results

### Main Results
Results on the CodRED dataset:

| Model | Closed micro F1 | Closed binary F1 | Open micro F1 | Open binary F1 |
|------|----------------|-----------------|---------------|----------------|
| ECRIM (RoBERTa) | 42.54 | 49.47 | 23.39 | 27.60 |
| NEPD (RoBERTa) | 42.96 | 52.67 | 30.12 | 37.04 |
| Vanilla LLaMA | 38.14 | 41.43 | 15.19 | 17.00 |
| **HCRE (LLaMA)** | **45.35** | **58.19** | **34.91** | **49.33** |

### Ablation Study

| Configuration | micro F1 | binary F1 | Description |
|------|---------|----------|------|
| Full HCRE | 45.35 | 58.19 | Full model |
| w/o multi-view | 39.37 | 49.63 | Using only a single verification set |
| w/o PtV | 37.66 | 47.28 | Removing the Predict-then-Verify strategy |
| w/o LTC | 43.18 | 56.60 | Directly generating the tree instead of layer-by-layer construction |
| w/o HRT | 38.14 | 41.43 | No hierarchical tree, direct classification |

### Key Findings
- Compared to the strongest SLM baseline (NEPD), HCRE improves micro F1 by 2.39 and binary F1 by 5.52 in the closed setting.
- The improvement is even more significant in the open setting (binary F1 jumping from 37.04 to 49.33), indicating that hierarchical classification is more effective for long documents.
- The Predict-then-Verify strategy is the most critical component: removing it causes micro F1 to drop by 7.69 and binary F1 by 10.91.
- Multi-view verification (3 sets vs. 1 set) brings an additional 1.71 micro F1 improvement.
- Error propagation analysis shows that the PtV strategy effectively reduces the error propagation rate at every layer.

## Highlights & Insights
- Preliminary experiments reveal that "excessive relation options" is the root cause of poor LLM performance in cross-document RE. This finding has general guiding significance—LLMs may face similar issues in any large-scale label classification task.
- The Predict-then-Verify strategy, which constructs verification sets by "replacing parent nodes with sub-nodes," is ingeniously designed. It essentially uses finer-grained information to validate coarse-grained judgments.
- Evaluation metric analysis (maximum F1 overestimating performance, P@K sensitivity to data scale) provides valuable methodological suggestions for the community.

## Limitations & Future Work
- Tree construction relies on GPT-4o, which introduces dependency on tree quality and higher costs.
- The verification steps increase inference overhead (multiple LLM calls required per layer).
- Experiments were only validated on the CodRED dataset; generalizability needs further confirmation.
- Future work could explore adaptive tree depth or dynamic adjustment of verification intensity to balance efficiency and accuracy.

## Related Work & Insights
- **vs NEPD**: NEPD focuses on long-distance dependency modeling but is still limited by the upper bound of SLM capabilities. HCRE breaks this limitation by leveraging the strong language understanding of LLMs.
- **vs Hierarchical Text Classification (HTC) methods**: Traditional HTC methods (DFS-L, BFS-L) lack verification mechanisms and suffer from severe error propagation in cross-document RE.
- **vs Vanilla LLM**: Preliminary experiments clearly show that directly using LLMs to handle large relation sets performs poorly; hierarchy is necessary.

## Rating
- Novelty: ⭐⭐⭐⭐ The combination of hierarchical classification and the Predict-then-Verify strategy is novel and practical.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablations are thorough, and the initial motivation analysis is persuasive.
- Writing Quality: ⭐⭐⭐⭐⭐ The logical chain from problem discovery to analysis to solution is very clear.
- Value: ⭐⭐⭐⭐ Provides practical guidance for LLM applications in large-scale classification tasks.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] LexRel: Benchmarking Legal Relation Extraction for Chinese Civil Cases](lexrel_benchmarking_legal_relation_extraction_for_chinese_civil_cases.md)
- [\[ACL 2026\] Beyond Chunking: Discourse-Aware Hierarchical Retrieval for Long Document Question Answering](beyond_chunking_discourse-aware_hierarchical_retrieval_for_long_document_questio.md)
- [\[ACL 2026\] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)
- [\[ACL 2026\] MSMO-ABSA: Multi-Scale and Multi-Objective Optimization for Cross-Lingual Aspect-Based Sentiment Analysis](msmo-absa_multi-scale_and_multi-objective_optimization_for_cross-lingual_aspect-.md)

</div>

<!-- RELATED:END -->
