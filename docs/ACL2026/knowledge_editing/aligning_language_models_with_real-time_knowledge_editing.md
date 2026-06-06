---
title: >-
  [Paper Note] Aligning Language Models with Real-time Knowledge Editing
description: >-
  [ACL 2026][Knowledge Editing][Real-time knowledge editing] This paper introduces CRAFT (a continuously updated Chinese Financial Knowledge Editing dataset) and KEDAS (a Knowledge Editing alignment paradigm based on Diver…
tags:
  - "ACL 2026"
  - "Knowledge Editing"
  - "Real-time knowledge editing"
  - "Knowledge alignment"
  - "Data contamination"
  - "Diverse augmentation"
  - "Adaptive inference"
date: 2026-05-08
content_hash: b0c129e787d67669
---

# Aligning Language Models with Real-time Knowledge Editing

**Conference**: ACL 2026  
**arXiv**: [2508.01302](https://arxiv.org/abs/2508.01302)  
**Code**: [GitHub](https://github.com/JamyDon/CRAFT-KEDAS)  
**Area**: Knowledge Editing  
**Keywords**: Real-time knowledge editing, Knowledge alignment, Data contamination, Diverse augmentation, Adaptive inference

## TL;DR

This paper introduces CRAFT (a continuously updated Chinese Financial Knowledge Editing dataset) and KEDAS (a Knowledge Editing alignment paradigm based on Diverse augmentation and Adaptive Selection) to solve the difficulty of balancing success rate, locality, and portability in real-time knowledge editing scenarios.

## Background & Motivation

**Background**: Knowledge editing aims to efficiently modify outdated knowledge in LMs without full retraining. However, mainstream evaluation datasets (ZsRE, MQuAKE, RippleEdits) are static and cannot be updated once released.

**Limitations of Prior Work**: (1) Static datasets suffer from severe data leakage—most knowledge has been seen by LMs during pre-training, leading to unfair evaluation; (2) WikiBigEdit is real-time but requires processing hundreds of gigabytes of Wiki data and suffers from high sparsity; (3) Existing methods struggle to balance editing success, locality, and portability.

**Key Challenge**: Parameter modification methods (e.g., ROME, WISE) suffer from severe model degradation during continuous editing; retrieval-based methods (e.g., IKE, EREN) show unstable performance due to a lack of alignment; alignment methods (e.g., LTE) exhibit poor locality due to overfitting.

**Goal**: Construct a continuously updated, contamination-free real-time knowledge editing dataset and propose a method achieving balanced performance across all metrics.

**Key Insight**: Leveraging publicly available Chinese official financial statistics (which are continuously updated and unseen by LMs) to build the dataset, while redefining knowledge editing as an LM alignment problem.

**Core Idea**: Empower the LM with knowledge editing capabilities through one-time offline alignment (LoRA fine-tuning), and then fundamentally solve the locality problem during inference by using adaptive routing to decide between the original model and the aligned model.

## Method

### Overall Architecture

KEDAS consists of two stages: (1) Offline Alignment—fine-tuning the LM with LoRA on knowledge editing format data to enable it to update answers using editing prompts; (2) Online Editing—storing new knowledge in memory in diverse forms, followed by intelligent retrieval and adaptive model path selection during inference.

### Key Designs

1. **CRAFT Dataset**:

    - **Function**: Provides a continuously updated, contamination-free benchmark for real-time knowledge editing.
    - **Mechanism**: Utilizes Chinese official financial and statistical data (e.g., GDP, population), designed as "paired edits" to serve as compositional reasoning tests. It supports evaluations for alias portability, temporal locality, and common-sense locality.
    - **Design Motivation**: Official statistics updated regularly ensure data freshness; the paired edits design tests the model's ability to integrate multiple edits (compositional portability).

2. **Diverse Edit Augmentation**:

    - **Function**: Enhances the coverage and retrieval robustness of the edit memory.
    - **Mechanism**: Stores each edit in multiple formats—including the original QA pairs, paraphrased versions, and alias versions—to increase retrieval hit rates.
    - **Design Motivation**: User queries may express the same knowledge requirement in various ways; a single-form edit might lead to retrieval failure.

3. **Self-adaptive Inference**:

    - **Function**: Dynamically selects whether to activate the LoRA-aligned model during inference.
    - **Mechanism**: A filtered intelligent retriever determines if a query is relevant to any stored edits. If relevant, it is processed by the LoRA-aligned model (with edit context); otherwise, it uses the original LM directly.
    - **Design Motivation**: Fundamentally solves the locality issue—model behavior remains unchanged for irrelevant queries, avoiding knowledge forgetting caused by overfitting.

### Loss & Training

The alignment phase uses LoRA fine-tuning, with training data containing both in-scope and out-of-scope queries. Once one-time alignment is completed, subsequent edits only manipulate the memory without modifying parameters.

## Key Experimental Results

### Main Results

| Method | Success Rate | Locality | Portability | Overall |
|------|----------|--------|---------|------|
| ROME (Param Mod) | High→Degrades | Poor | Poor | Unbalanced |
| IKE (Retrieval) | Medium | Medium | Medium | Unstable |
| LTE (Alignment) | High | Poor (Overfit) | Medium | Unbalanced |
| KEDAS (Ours) | High | High | High | **Superior Overall** |

### Ablation Study

| Configuration | Key Metric | Description |
|------|---------|------|
| Leakage Analysis | CRAFT exposure ≈ 0 | Most traditional datasets have been seen by LMs |
| Remove Diverse Augmentation | Lower Recall | Multi-format storage improves robustness |
| Remove Adaptive Inference | Lower Locality | Routing mechanism is key to locality guarantee |

### Key Findings
- Knowledge leakage in existing datasets is severe—the exposure rates of 5 LMs on traditional datasets are much higher than on CRAFT.
- Parameter modification methods like ROME degrade rapidly during continuous editing, failing to meet real-time requirements.
- KEDAS significantly outperforms all baselines on both CRAFT and traditional datasets, achieving a balance across all metrics for the first time.

## Highlights & Insights
- The revelation of data leakage issues serves as a warning to the knowledge editing field—evaluation results may not reflect true capabilities.
- The "one-time alignment, lifetime editing" paradigm elegantly decouples alignment costs from editing flexibility.
- The adaptive inference path cleverly solves the trade-off between editing and non-editing—enabling editing without parameter modification.

## Limitations & Future Work
- CRAFT currently only covers Chinese and financial/statistical domains; generalization to other languages and fields needs further validation.
- The quality of the adaptive retriever is a bottleneck for system performance.
- Efficiency of memory management for extremely large-scale editing (e.g., millions) has not been discussed.
- Future work could explore more efficient alignment strategies and cross-lingual real-time editing.

## Related Work & Insights
- **vs ROME/MEMIT**: Parameter modification methods degrade during continuous editing; KEDAS avoids this via external memory.
- **vs LTE**: Both are alignment methods, but KEDAS solves the overfitting problem of LTE through its adaptive inference path.
- **vs RAG**: KEDAS not only retrieves edits but also aligns the LM's ability to utilize them, proving more effective than pure RAG.

## Rating
- Novelty: ⭐⭐⭐⭐ Innovation in both the CRAFT dataset and the KEDAS paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Dual validation on CRAFT and traditional datasets, plus comprehensive data leakage analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear problem definition and systematic methodological description.
- Value: ⭐⭐⭐⭐⭐ Provides dual contributions to the field in terms of both datasets and methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The Model Agreed, But Didn't Learn: Diagnosing Surface Compliance in Large Language Models](the_model_agreed_but_didn39t_learn_diagnosing_surface_compliance_in_large_langua.md)
- [\[ICML 2026\] The Labyrinth and the Thread: Rethinking Regularizations in Sequential Knowledge Editing for Large Language Models](../../ICML2026/knowledge_editing/the_labyrinth_and_the_thread_rethinking_regularizations_in_sequential_knowledge_.md)
- [\[ICML 2026\] Reverse-Engineering Model Editing on Language Models](../../ICML2026/knowledge_editing/reverse-engineering_model_editing_on_language_models.md)
- [\[NeurIPS 2025\] UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models](../../NeurIPS2025/knowledge_editing/uniedit_a_unified_knowledge_editing_benchmark_for_large_language_models.md)
- [\[ICML 2026\] Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence](../../ICML2026/knowledge_editing/revisiting_parameter-based_knowledge_editing_in_large_language_models_theoretica.md)

</div>

<!-- RELATED:END -->
