---
title: >-
  [Paper Note] DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition
description: >-
  [ACL 2026][LLM Evaluation][Zero-shot NER] DiZiNER simulates the pilot annotation workflow in human labeling pipelines by employing multiple heterogeneous LLMs as annotators and a supervisor LLM to analyze inter-model disagreements and iteratively refine task instructions. The method achieves zero-shot state-of-the-art on 14 out of 18 NER benchmarks, with an average improvement of +8.0 F1, and surpasses its own supervisor model, GPT-4o mini, without any parameter updates.
tags:
  - ACL 2026
  - LLM Evaluation
  - Zero-shot NER
  - Disagreement-guided
  - Instruction Refinement
  - Pilot Annotation Simulation
  - Multi-model Ensemble
date: 2026-05-08
content_hash: b12b56bd60b0c728
---

# DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition

**Conference**: ACL 2026  
**arXiv**: [2604.15866](https://arxiv.org/abs/2604.15866)  
**Code**: [https://github.com/SiunKim/diziner-ner/](https://github.com/SiunKim/diziner-ner/)  
**Area**: LLM Evaluation  
**Keywords**: Zero-shot NER, Disagreement-guided, Instruction Refinement, Pilot Annotation Simulation, Multi-model Ensemble

## TL;DR

DiZiNER simulates the pilot annotation workflow in human labeling pipelines by employing multiple heterogeneous LLMs as annotators and a supervisor LLM to analyze inter-model disagreements and iteratively refine task instructions. The method achieves zero-shot state-of-the-art on 14 out of 18 NER benchmarks, with an average improvement of +8.0 F1, and surpasses its own supervisor model, GPT-4o mini, without any parameter updates.

## Background & Motivation

**Background**: Large language models (LLMs) have demonstrated remarkable progress on named entity recognition (NER) through zero-shot and few-shot learning. However, state-of-the-art NER systems still rely heavily on human-annotated data, and a substantial performance gap remains between zero-shot approaches and supervised fine-tuning methods (approximately −32.0 F1 on average).

**Limitations of Prior Work**: LLMs exhibit persistent systematic error patterns in NER, falling into three main categories: (1) difficulty following complex annotation guidelines; (2) ambiguity in entity boundary detection; and (3) frequent entity type confusion. Existing solutions such as instruction tuning, open NER frameworks, and large-scale synthetic data generation have achieved partial improvements, but the gap with supervised methods remains significant.

**Key Challenge**: Existing zero-shot NER methods lack an effective mechanism to systematically identify and correct systematic error patterns in LLM annotations. Single-model instruction optimization is constrained by the model's own biases and cannot transcend its inherent capability limitations.

**Goal**: To design a zero-shot NER framework that requires no parameter updates, automatically discovers and corrects systematic annotation errors in LLMs, and narrows the performance gap between zero-shot and supervised approaches.

**Key Insight**: The authors observe that error patterns in LLM-based NER bear strong resemblance to annotation inconsistencies encountered in the early stages of human labeling. In human annotation, the pilot annotation workflow—where multiple annotators label independently, a supervisor analyzes disagreements, and guidelines are updated accordingly—has proven effective in resolving such issues.

**Core Idea**: Multiple heterogeneous LLMs are used to simulate annotators, while a stronger LLM simulates the supervisor. By analyzing inter-model disagreements to iteratively refine NER task instructions, the framework continuously improves zero-shot NER performance without any parameter updates.

## Method

### Overall Architecture

DiZiNER adopts an iterative pilot annotation simulation framework consisting of three core stages: (1) **independent cross-annotation** — multiple heterogeneous LLMs independently annotate the same set of documents for NER; (2) **disagreement analysis** — high-disagreement regions (hotspot spans) are identified and disagreement patterns are quantified and categorized; (3) **instruction refinement** — a supervisor model iteratively refines both universal instructions and model-specific instructions based on disagreement reports. The inputs are NER task definitions (entity types and examples), and the outputs are iteratively refined, high-quality NER annotations.

### Key Designs

1. **Heterogeneous Annotator Pool and Independent Cross-Annotation**:

    - **Function**: Multiple LLMs from different development teams and architectures are employed as independent annotators to perform NER on the same documents.
    - **Mechanism**: Eight open-source LLMs are used (including mistral-small3.2:24b, gpt-oss:20b, phi4:14b, and qwen3:14b), sourced from different organizations with distinct training data and optimization pipelines. Each iteration samples 25 documents, and all annotators label independently according to their respective task configurations $\Theta_k^{(t)} = (\Sigma, C^{(t)}, R_k^{(t)}, G^{(t)})$. Annotation results are converted from span-level to BIO sequence representation to facilitate token-level comparative analysis.
    - **Design Motivation**: Heterogeneity ensures that annotator errors are independent of one another, preventing correlated errors from producing spuriously high agreement and thereby making disagreement signals more informative.

2. **Multi-dimensional Disagreement Analysis and Hotspot Identification**:

    - **Function**: Precisely locates text regions with high inter-annotator disagreement and quantifies disagreements into structured reports.
    - **Mechanism**: Model weights are first computed based on pairwise inter-model F1 scores, and consensus labels are obtained via weighted majority voting. Three complementary token-level disagreement metrics are then computed: label conflict $D_{\text{conf}}$ (dispersion of BIO labels), type confusion $D_{\text{type}}$ (entity type disagreement), and boundary uncertainty $U_{\text{bnd}}$ (entity boundary consistency). The final disagreement score is the maximum of the three; tokens in the top 20% are marked as high-disagreement regions, and adjacent high-disagreement tokens are merged into hotspot spans.
    - **Design Motivation**: Different types of disagreement point to different annotation problems (boundary issues vs. type confusion vs. entity judgment), and multi-dimensional metrics ensure no category of systematic error is overlooked.

3. **Four-stage Instruction Refinement**:

    - **Function**: The supervisor model systematically refines task instructions based on disagreement documents and the instructions from the previous iteration.
    - **Mechanism**: Refinement proceeds in four stages — (1) *disagreement pattern analysis*: recurring disagreement patterns in hotspots are identified and their root causes inferred; (2) *model-specific diagnosis*: targeted adjustments are formulated for residual errors in non-elite models; (3) *guideline integration and conflict resolution*: new and existing instructions are merged, with conflicts resolved based on the ultimate task objective; (4) *hierarchical organization*: refined instructions are restructured into a hierarchical format, with general rules taking precedence over specific ones. GPT-4o mini serves as the supervisor model.
    - **Design Motivation**: The staged refinement pipeline ensures that instruction updates are systematic and controllable, while hierarchical organization improves instruction readability and LLM compliance.

### Loss & Training

DiZiNER involves no parameter training and relies entirely on iterative instruction refinement. Each iteration processes 25 sampled documents, with a maximum of 5 optimization cycles. The optimal configuration is selected via inter-model pairwise agreement (strict span F1) — since agreement correlates strongly with NER performance (Pearson correlation up to 0.922), the best iteration–model combination can be reliably identified without any labeled data. Three parameter configuration groups are explored experimentally to ensure consistency across benchmarks.

## Key Experimental Results

### Main Results

| Method | CrossNER Avg. | 13-benchmark Avg. | vs. Best Zero-shot | vs. Supervised Gap |
|--------|--------------|-------------------|--------------------|--------------------|
| B2NER (Prev. SOTA) | 75.3 | — | — | −32.0 |
| GPT-4o mini (Supervisor) | 69.3 | 62.3 | — | — |
| DiZiNER (Ours) | 75.7 | 68.4 | +11.1 | −20.9 |

DiZiNER achieves zero-shot SOTA on 14 out of 18 benchmarks, surpassing the GPT-4o mini supervisor by +5.0 to +6.4 F1 on average.

### Ablation Study

| Ablation | Impact |
|----------|--------|
| Remove final task objective | F1 drops from 77.6 to 71.9 |
| Heterogeneous vs. same-family model pool | Heterogeneous pool outperforms by 1.7–3.7 F1 |
| Annotator count 4 → 8 | F1 improves from 73.1 to 75.5 |
| Annotator count > 12 | Performance degrades (consensus noise) |
| Using gold annotations | Marginal gain of only +0.3 F1 |
| Optimal document set size | 15–25 samples |

### Key Findings

- Inter-model agreement correlates strongly with NER performance and can serve as a label-free quality indicator.
- Heterogeneous model pools (≤24B) consistently outperform pools composed of larger models from the same family.
- Gold-annotated data provides minimal benefit to the framework, indicating that disagreement-guided refinement alone is sufficiently effective.
- The average optimization cost per benchmark is only $40.1 (inference: $1.90/round + supervision: $0.77/round).

## Highlights & Insights

- The paper makes a compelling and practically grounded transfer of the pilot annotation methodology from human annotation to the LLM setting.
- The framework surpasses its own supervisor model without any parameter updates, demonstrating that disagreement signals alone carry information that exceeds the capacity of a single model.
- Inter-model agreement as a label-free performance proxy provides a viable quality monitoring mechanism for real-world deployment.
- The extremely low cost ($40 per benchmark) makes large-scale application feasible.

## Limitations & Future Work

- A gap of approximately −20.9 F1 remains between zero-shot and supervised methods, which has not been fully closed.
- The framework has some dependency on the supervisor model's capability, and performance varies across different supervisor models.
- The fixed 20% threshold may lead to over-correction, with some benchmarks exhibiting performance degradation after peaking early.
- The small document set size (25 samples) may limit coverage for more complex tasks.

## Related Work & Insights

- Unlike instruction tuning approaches such as InstructUIE and GoLLIE, DiZiNER is entirely training-free.
- DiZiNER is complementary to encoder-based methods such as UniversalNER and GLiNER, which focus on inference efficiency.
- Unlike self-iterative methods such as EvoPrompt, which rely on self-generated pseudo-examples, DiZiNER leverages inter-model disagreements as a stronger signal.
- Insight: Multi-model disagreement signals may play a similar role in other information extraction tasks such as relation extraction and event extraction.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Systematically transfers the pilot annotation methodology to zero-shot NER with LLMs; conceptually original and thoroughly executed.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 18 benchmarks, multiple ablations, cost analysis, and robustness validation; extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Framework description is clear and mathematical notation is well-defined, though some sections are dense.
- **Value**: ⭐⭐⭐⭐⭐ Provides a low-cost, training-free, high-performance zero-shot NER solution with strong practical utility.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] A Conditional Probability Framework for Compositional Zero-shot Learning](../../ICCV2025/llm_evaluation/a_conditional_probability_framework_for_compositional_zero-shot_learning.md)
- [\[AAAI 2026\] GranAlign: Granularity-Aware Alignment Framework for Zero-Shot Video Moment Retrieval](../../AAAI2026/llm_evaluation/granalign_granularity-aware_alignment_framework_for_zero-shot_video_moment_retri.md)
- [\[NeurIPS 2025\] Benchmarking Large Language Models for Zero-Shot and Few-Shot Phishing URL Detection](../../NeurIPS2025/llm_evaluation/benchmarking_large_language_models_for_zero-shot_and_few-shot_phishing_url_detec.md)
- [\[NeurIPS 2025\] Unlocking Transfer Learning for Open-World Few-Shot Recognition](../../NeurIPS2025/llm_evaluation/unlocking_transfer_learning_for_open-world_few-shot_recognition.md)
- [\[ACL 2026\] HiGMem: A Hierarchical and LLM-Guided Memory System for Long-Term Conversational Agents](higmem_a_hierarchical_and_llm-guided_memory_system_for_long-term_conversational_.md)

<!-- RELATED:END -->
