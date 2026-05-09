---
title: >-
  [Paper Note] Aligning Language Models with Real-time Knowledge Editing
description: >-
  [ACL 2026][Knowledge Editing][Real-time Knowledge Editing] This paper introduces CRAFT (a continuously updated Chinese financial knowledge editing dataset) and KEDAS (a knowledge editing alignment paradigm based on diverse edit augmentation and self-adaptive inference), addressing the inability of existing knowledge editing methods to balance edit success rate, locality, and portability in real-time scenarios.
tags:
  - ACL 2026
  - Knowledge Editing
  - Real-time Knowledge Editing
  - Knowledge Alignment
  - Dataset Contamination
  - Diverse Augmentation
  - Self-Adaptive Inference
date: 2025-04-17
content_hash: f37b76ed1e42eee6
---

# Aligning Language Models with Real-time Knowledge Editing

**Conference**: ACL 2026  
**arXiv**: [2508.01302](https://arxiv.org/abs/2508.01302)  
**Code**: [GitHub](https://github.com/JamyDon/CRAFT-KEDAS)  
**Area**: Knowledge Editing  
**Keywords**: Real-time Knowledge Editing, Knowledge Alignment, Dataset Contamination, Diverse Augmentation, Self-adaptive Inference

## TL;DR

This paper introduces CRAFT (a continuously updated Chinese financial knowledge editing dataset) and KEDAS (a knowledge editing alignment paradigm based on diverse edit augmentation and self-adaptive inference), addressing the problem that existing knowledge editing methods cannot simultaneously achieve high editing success rate, locality, and portability in real-time scenarios.

## Background & Motivation

**Background**: Knowledge editing aims to efficiently modify outdated knowledge in language models without full retraining. However, mainstream evaluation datasets (ZsRE, MQuAKE, RippleEdits) are static and cannot be updated once published.

**Limitations of Prior Work**: (1) Static datasets suffer from serious data leakage — most knowledge has already been seen by LMs during pre-training, leading to unfair evaluation; (2) WikiBigEdit is real-time but requires processing hundreds of GB of Wiki data with severe sparsity; (3) Existing methods struggle to balance editing success rate, locality, and portability.

**Key Challenge**: Parameter modification methods (e.g., ROME, WISE) suffer severe model degradation during sequential editing; retrieval methods (e.g., IKE, EREN) exhibit unstable performance due to lack of alignment; alignment methods (e.g., LTE) have poor locality due to overfitting.

**Goal**: Build a continuously updated, contamination-free real-time knowledge editing dataset, and propose a knowledge editing method that performs well across all metrics.

**Key Insight**: Leverage publicly available Chinese official financial statistics (continuously updated and unseen by LMs) for dataset construction, and redefine knowledge editing as an LM alignment problem.

**Core Idea**: Through one-time offline alignment (LoRA fine-tuning) to endow the LM with knowledge editing capability, then at inference time use self-adaptive routing to decide whether to use the original model or the aligned model, fundamentally solving the locality problem.

## Method

### Overall Architecture

KEDAS has two stages: (1) Offline alignment — LoRA fine-tuning on knowledge editing format data to endow the LM with the ability to update answers using edit prompts; (2) Online editing — storing new knowledge in diverse forms in memory, with intelligent retrieval and self-adaptive model path selection during inference.

### Key Designs

1. **CRAFT Dataset**:

    - Function: Provide a continuously updated, contamination-free real-time knowledge editing evaluation benchmark
    - Mechanism: Leverages publicly available Chinese official financial and statistical data (e.g., GDP, population), designing paired edits as compositional reasoning tests. Supports alias portability, temporal locality, and common-sense locality evaluation
    - Design Motivation: Official statistical data is continuously updated, guaranteeing data freshness; paired edit design tests the model's ability to integrate multiple edits (compositional portability)

2. **Diverse Edit Augmentation**:

    - Function: Enhance coverage and retrieval robustness of edit memory
    - Mechanism: Each edit is stored in multiple expression forms — including original QA pairs, paraphrased versions, alias versions, etc., increasing retrieval hit rate
    - Design Motivation: User queries may express the same knowledge need in various ways; a single-form edit may lead to retrieval failure

3. **Self-adaptive Inference Path**:

    - Function: Dynamically select whether to activate the LoRA-aligned model at inference time
    - Mechanism: A filter-augmented intelligent retriever determines whether the query is related to any edit. If related, it is processed through the LoRA-aligned model (with edit context provided); if unrelated, the original LM is used directly
    - Design Motivation: Fundamentally solves the locality problem — completely unmodified model behavior for unrelated queries, avoiding knowledge forgetting caused by overfitting

### Loss & Training

Alignment stage uses LoRA fine-tuning, with training data containing both in-scope and out-of-scope queries. After one-time alignment, subsequent edits only operate on memory without parameter modifications.

## Key Experimental Results

### Main Results

| Method | Edit Success | Locality | Portability | Overall |
|------|----------|--------|---------|------|
| ROME (Parameter Modification) | High→Degraded | Poor | Poor | Unbalanced |
| IKE (Retrieval) | Medium | Medium | Medium | Unstable |
| LTE (Alignment) | High | Poor (Overfitting) | Medium | Unbalanced |
| KEDAS (Ours) | High | High | High | **Excellent across all** |

### Ablation Study

| Config | Metric | Note |
|------|---------|------|
| Data leakage analysis | CRAFT exposure rate ≈ 0 | Traditional datasets largely seen by LMs |
| Remove diverse augmentation | Retrieval recall decreases | Multi-form storage improves robustness |
| Remove self-adaptive inference | Locality drops | Routing mechanism is key to locality guarantee |

### Key Findings
- Knowledge leakage in existing datasets is severe — exposure rates of 5 LMs on traditional datasets far exceed CRAFT
- Parameter modification methods like ROME rapidly degrade during sequential editing, failing to meet real-time editing requirements
- KEDAS significantly outperforms all baselines on both CRAFT and traditional datasets, achieving balanced performance across all metrics for the first time

## Highlights & Insights
- Revealing the data leakage problem carries warning significance for the knowledge editing field — evaluation results may not reflect true capabilities
- The "align once, edit forever" paradigm elegantly separates alignment cost from editing flexibility
- Self-adaptive inference path cleverly resolves the edit-vs-no-edit trade-off — editing without parameter modification

## Limitations & Future Work
- CRAFT currently covers only Chinese and the financial/statistical domain; generalization to other languages and domains requires further validation
- Self-adaptive retriever quality is the system performance bottleneck
- Memory management efficiency at extremely large scales (e.g., millions of edits) is not discussed
- Future directions include more efficient alignment strategies and cross-lingual real-time editing

## Related Work & Insights
- **vs ROME/MEMIT**: Parameter modification methods degrade during sequential editing; KEDAS avoids this through external memory
- **vs LTE**: Both are alignment methods, but KEDAS resolves LTE's overfitting problem through self-adaptive inference paths
- **vs RAG**: KEDAS not only retrieves edits but also aligns the LM's ability to utilize edits, making it more effective than pure RAG

## Rating
- Novelty: ⭐⭐⭐⭐ Both CRAFT dataset and KEDAS paradigm are innovative
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual validation on CRAFT + traditional datasets, data leakage analysis
- Writing Quality: ⭐⭐⭐⭐ Problem definition is clear, method description is systematic
- Value: ⭐⭐⭐⭐⭐ Dual contribution of dataset and methodology to the knowledge editing field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](../../NeurIPS2025/knowledge_editing/uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)
- [\[AAAI 2026\] Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)](../../AAAI2026/knowledge_editing/multiplicative_orthogonal_sequential_editing_for_language_models.md)
- [\[NeurIPS 2025\] KScope: A Framework for Characterizing the Knowledge Status of Language Models](../../NeurIPS2025/knowledge_editing/kscope_a_framework_for_characterizing_the_knowledge_status_of_language_models.md)
- [\[ACL 2026\] EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing](evoedit_evolving_null-space_alignment_for_robust_and_efficient_knowledge_editing.md)
- [\[ICLR 2026\] When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations](../../ICLR2026/knowledge_editing/when_large_multimodal_models_confront_evolving_knowledge_challenges_and_explorat.md)

</div>

<!-- RELATED:END -->
---
title: >-
  [Paper Note] Aligning Language Models with Real-time Knowledge Editing
description: >-
  [ACL 2026][Knowledge Editing][Real-time Editing] CRAFT dataset (continuously updated Chinese financial knowledge) and KEDAS paradigm (diverse edit augmentation + self-adaptive inference) for balanced knowledge editing performance.
tags:
  - ACL 2026
  - Real-time Knowledge Editing
  - Knowledge Editing
  - Dataset Contamination
  - Diverse Augmentation
  - Self-Adaptive Inference
date: 2025-04-17
content_hash: f37b76ed1e42eee6
---

# Aligning Language Models with Real-time Knowledge Editing

**Conference**: ACL 2026  
**arXiv**: [2508.01302](https://arxiv.org/abs/2508.01302)  
**Code**: [GitHub](https://github.com/JamyDon/CRAFT-KEDAS)  
**Area**: Knowledge Editing  
**Keywords**: Real-time Knowledge Editing, Knowledge Alignment, Dataset Contamination, Diverse Augmentation, Self-Adaptive Inference

## TL;DR

This paper introduces CRAFT (a continuously updated Chinese financial knowledge editing dataset) and KEDAS (a knowledge editing alignment paradigm based on diverse edit augmentation and self-adaptive inference), addressing the inability of existing knowledge editing methods to balance edit success rate, locality, and portability in real-time scenarios.

## Background & Motivation

**Background**: Knowledge editing aims to efficiently modify outdated knowledge in language models without full retraining. However, mainstream evaluation datasets (ZsRE, MQuAKE, RippleEdits) are static and cannot be updated once published.

**Limitations of Prior Work**: (1) Static datasets suffer from severe data leakage — most knowledge has already been seen by LMs during pre-training, leading to unfair evaluation; (2) WikiBigEdit, while real-time, requires processing hundreds of GB of Wiki data with severe sparsity; (3) Existing methods struggle to balance edit success rate, locality, and portability.

**Key Challenge**: Parameter modification methods (e.g., ROME, WISE) suffer severe model degradation during continuous editing; retrieval methods (e.g., IKE, EREN) exhibit unstable performance due to lack of alignment; alignment methods (e.g., LTE) show poor locality due to overfitting.

**Goal**: Build a continuously updated, contamination-free real-time knowledge editing dataset and propose a knowledge editing method that performs uniformly well across all metrics.

**Key Insight**: Leverage publicly available Chinese official financial statistical data (continuously updated and unseen by LMs) to construct the dataset, and redefine knowledge editing as an LM alignment problem.

**Core Idea**: Through one-time offline alignment (LoRA fine-tuning) to endow the LM with knowledge editing capabilities, then use self-adaptive routing during inference to decide whether to use the original or aligned model, fundamentally solving the locality problem.

## Method

### Overall Architecture

KEDAS has two stages: (1) Offline alignment — fine-tuning the LM with LoRA on knowledge editing format data to endow it with the ability to update answers using edit prompts; (2) Online editing — storing new knowledge in diverse forms in memory, with intelligent retrieval and self-adaptive model path selection during inference.

### Key Designs

1. **CRAFT Dataset**:

    - Function: Provide a continuously updated, contamination-free real-time knowledge editing evaluation benchmark
    - Mechanism: Leverages publicly available Chinese official financial and statistical data (e.g., GDP, population), designing paired edits as compositional reasoning tests. Supports alias portability, temporal locality, and commonsense locality evaluation
    - Design Motivation: Official statistical data is continuously updated, guaranteeing data freshness; paired edit design tests the model's ability to integrate multiple edits (composite portability)

2. **Diverse Edit Augmentation**:

    - Function: Enhance edit memory coverage and retrieval robustness
    - Mechanism: Store each edit in multiple expression forms — including original QA pairs, paraphrased versions, alias versions, etc., increasing retrieval hit rate
    - Design Motivation: User queries may express the same knowledge need in various ways; single-form edits may cause retrieval failures

3. **Self-Adaptive Inference Path**:

    - Function: Dynamically select whether to activate the LoRA-aligned model during inference
    - Mechanism: A filter-augmented intelligent retriever determines whether the query is related to any edit. If related, the query is processed through the LoRA-aligned model (providing edit context); if unrelated, the original LM is used directly
    - Design Motivation: Fundamentally solves the locality problem — model behavior is completely unmodified for unrelated queries, avoiding knowledge forgetting caused by overfitting

### Loss & Training

The alignment stage uses LoRA fine-tuning with training data containing both in-scope and out-of-scope edit queries. After one-time alignment, subsequent edits only operate on memory without parameter modification.

## Key Experimental Results

### Main Results

| Method | Edit Success Rate | Locality | Portability | Overall |
|------|----------|--------|---------|------|
| ROME (Parameter Mod.) | High→Degraded | Poor | Poor | Imbalanced |
| IKE (Retrieval) | Medium | Medium | Medium | Unstable |
| LTE (Alignment) | High | Poor (Overfitting) | Medium | Imbalanced |
| KEDAS (Ours) | High | High | High | **Excellent across all** |

### Ablation Study

| Config | Metric | Note |
|------|---------|------|
| Data leakage analysis | CRAFT exposure rate ≈ 0 | Most traditional datasets already seen by LMs |
| Remove diverse augmentation | Retrieval recall decreased | Multi-form storage improves robustness |
| Remove self-adaptive inference | Locality decreased | Routing mechanism is key to locality preservation |

### Key Findings
- Knowledge leakage in existing datasets is severe — exposure rates of 5 LMs on traditional datasets are far higher than on CRAFT
- Parameter modification methods like ROME degrade rapidly during continuous editing, unable to meet real-time editing demands
- KEDAS significantly outperforms all baselines on both CRAFT and traditional datasets, achieving balanced performance across all metrics for the first time

## Highlights & Insights
- The revelation of data leakage problems has cautionary significance for the knowledge editing field — evaluation results may not reflect true capabilities
- The "align once, edit for life" paradigm elegantly separates alignment cost from editing flexibility
- The self-adaptive inference path cleverly resolves the edit/no-edit trade-off — editing without parameter modification

## Limitations & Future Work
- CRAFT currently covers only Chinese and financial/statistical domains; generalization to other languages and domains requires further validation
- Self-adaptive retriever quality is the system performance bottleneck
- Memory management efficiency for extremely large-scale edits (e.g., millions) is not discussed
- Future directions include more efficient alignment strategies and cross-lingual real-time editing

## Related Work & Insights
- **vs ROME/MEMIT**: Parameter modification methods degrade during continuous editing; KEDAS avoids this through external memory
- **vs LTE**: Both are alignment methods; KEDAS solves LTE's overfitting problem through self-adaptive inference path
- **vs RAG**: KEDAS not only retrieves edits but also aligns the LM's ability to utilize edits, making it more effective than pure RAG

## Rating
- Novelty: ⭐⭐⭐⭐ Both the CRAFT dataset and KEDAS paradigm are innovative
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual validation on CRAFT + traditional datasets, data leakage analysis
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition, systematic method description
- Value: ⭐⭐⭐⭐⭐ Dual contribution of dataset and methodology to the knowledge editing field

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](../../NeurIPS2025/knowledge_editing/uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)
- [\[AAAI 2026\] Multiplicative Orthogonal Sequential Editing for Language Models (MOSE)](../../AAAI2026/knowledge_editing/multiplicative_orthogonal_sequential_editing_for_language_models.md)
- [\[NeurIPS 2025\] KScope: A Framework for Characterizing the Knowledge Status of Language Models](../../NeurIPS2025/knowledge_editing/kscope_a_framework_for_characterizing_the_knowledge_status_of_language_models.md)
- [\[ACL 2026\] EvoEdit: Evolving Null-space Alignment for Robust and Efficient Knowledge Editing](evoedit_evolving_null-space_alignment_for_robust_and_efficient_knowledge_editing.md)
- [\[ICLR 2026\] When Large Multimodal Models Confront Evolving Knowledge: Challenges and Explorations](../../ICLR2026/knowledge_editing/when_large_multimodal_models_confront_evolving_knowledge_challenges_and_explorat.md)

</div>

<!-- RELATED:END -->
