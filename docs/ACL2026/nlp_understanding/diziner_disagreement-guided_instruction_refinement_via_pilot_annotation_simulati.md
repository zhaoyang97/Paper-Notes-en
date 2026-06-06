---
title: >-
  [Paper Note] DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition
description: >-
  [ACL 2026][NLP Understanding][Zero-shot NER] By simulating the "pilot annotation" workflow in human labeling, DiZiNER utilizes multiple heterogeneous LLMs as annotators and a supervisor LLM to analyze inter-model disagre…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Zero-shot NER"
  - "Disagreement-guided"
  - "Instruction Refinement"
  - "Pilot Annotation Simulation"
  - "Multi-model Ensemble"
date: 2026-05-08
content_hash: 03c7b0a992a15271
---

# DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition

**Conference**: ACL 2026  
**arXiv**: [2604.15866](https://arxiv.org/abs/2604.15866)  
**Code**: [https://github.com/SiunKim/diziner-ner/](https://github.com/SiunKim/diziner-ner/)  
**Area**: LLM Evaluation  
**Keywords**: Zero-shot NER, Disagreement-guided, Instruction Refinement, Pilot Annotation Simulation, Multi-model Ensemble

## TL;DR

By simulating the "pilot annotation" workflow in human labeling, DiZiNER utilizes multiple heterogeneous LLMs as annotators and a supervisor LLM to analyze inter-model disagreements and iteratively refine task instructions. It achieves zero-shot SOTA on 14 out of 18 NER benchmarks, with an average improvement of +8.0 F1, even surpassing its supervisor, GPT-5 mini.

## Background & Motivation

**Background**: Large Language Models (LLMs) have made significant progress in Named Entity Recognition (NER) through zero-shot and few-shot learning. However, current state-of-the-art NER systems still heavily rely on human-annotated data, leaving a substantial performance gap between zero-shot methods and supervised fine-tuning methods (averaging approximately -32.0 F1).

**Limitations of Prior Work**: LLMs exhibit persistent systematic error patterns in NER tasks, primarily categorized into three types: (1) difficulty in following complex annotation guidelines; (2) ambiguity in entity boundary detection; and (3) frequent confusion between entity types. Existing solutions such as instruction fine-tuning, open-ended NER frameworks, and large-scale synthetic data generation have provided improvements but still lag significantly behind supervised methods.

**Key Challenge**: Existing zero-shot NER methods lack an effective mechanism to systematically discover and correct LLM annotation error patterns. Instruction refinement for a single model is limited by the model's own biases and cannot transcend its inherent capabilities.

**Goal**: To design a zero-shot NER framework that requires no parameter updates, capable of automatically discovering and correcting systematic errors in LLM annotations to bridge the performance gap between zero-shot and supervised methods.

**Key Insight**: The authors observed that the NER error patterns of LLMs are highly similar to the annotation inconsistencies found in the early stages of human labeling. In human annotation, these issues are effectively addressed through a "pilot annotation" process—where multiple annotators label independently, a supervisor analyzes disagreements, and guidelines are updated accordingly.

**Core Idea**: Use multiple heterogeneous LLMs to simulate annotators and a stronger LLM to simulate a supervisor. By analyzing inter-model disagreements to iteratively refine NER task instructions, the framework continuously improves zero-shot NER performance without any parameter updates.

## Method

### Overall Architecture

DiZiNER adopts an iterative pilot annotation simulation framework. The overall pipeline consists of three core phases: (1) Independent Cross-Annotation—multiple heterogeneous LLMs independently annotate the same set of documents for NER; (2) Disagreement Analysis—identifying high-disagreement regions (hotspot spans), quantifying and classifying annotation disagreement patterns; and (3) Instruction Refinement—the supervisor model iteratively refines general and model-specific instructions based on disagreement reports. The input is the NER task definition (entity types, examples), and the output is the high-quality NER annotation result after iterative refinement.

### Key Designs

1.  **Heterogeneous Annotator Pool & Independent Cross-Annotation**:
    - **Function**: Utilizes multiple LLMs from different development teams and architectures as independent annotators to perform NER on the same documents.
    - **Mechanism**: Employs 8 open-source LLMs (including mistral-small3.2:24b, gpt-oss:20b, phi4:14b, qwen3:14b, etc.). These models originate from different organizations and possess varied training data and optimization processes. In each iteration, 25 samples are drawn from the document set, and all annotators label them independently based on their respective task configurations $\Theta_k^{(t)} = (\Sigma, C^{(t)}, R_k^{(t)}, G^{(t)})$. Annotation results are converted from span-level to BIO sequence representations for token-level comparative analysis.
    - **Design Motivation**: Heterogeneity ensures that errors among annotators are mutually independent, avoiding correlated errors that lead to false high consistency, thereby making disagreement signals more informative.

2.  **Multi-dimensional Disagreement Analysis & Hotspot Identification**:
    - **Function**: Precisely locates text regions where high disagreement exists among annotators and quantifies these disagreements into structured reports.
    - **Mechanism**: First, model weights are calculated based on pairwise inter-model F1 scores, and consensus labels are obtained through weighted majority voting. Then, three complementary token-level disagreement metrics are calculated: label conflict $D_{\text{conf}}$ (dispersion of BIO labels), type confusion $D_{\text{type}}$ (disagreement in entity types), and boundary uncertainty $U_{\text{bnd}}$ (consistency of entity boundaries). The final disagreement score is the maximum of the three. Tokens in the top 20% are marked as high-disagreement regions, and adjacent tokens are merged into hotspot spans.
    - **Design Motivation**: Different types of disagreements point to different annotation problems (boundary issues vs. type confusion vs. entity detection). Multi-dimensional metrics ensure no systematic error types are missed.

3.  **Four-stage Instruction Refinement**:
    - **Function**: The supervisor model systematically optimizes task instructions based on disagreement documents and instructions from the previous round.
    - **Mechanism**: Refinement is divided into four stages: (1) Disagreement Pattern Analysis: identifying recurring disagreement patterns in hotspots and inferring root causes; (2) Model-specific Diagnosis: formulating targeted adjustments for residual errors in non-elite models; (3) Guideline Integration & Conflict Resolution: merging new and old instructions and resolving conflicts based on the final task objective; (4) Hierarchical Organization: reorganizing refined instructions into a hierarchical structure where general rules take precedence over specific rules. GPT-5 mini is used as the supervisor model.
    - **Design Motivation**: The staged refinement process ensures that instruction updates are systematic and controllable, while hierarchical organization improves readability and LLM instruction-following.

### Loss & Training

DiZiNER involves no parameter training and is entirely based on iterative instruction refinement. Each iteration processes 25 document samples, with a maximum of 5 optimization loops. The optimal configuration is selected via pairwise inter-model consistency (strict span F1). Since consistency is strongly correlated with NER performance (correlation coefficient as high as 0.922), the best "iteration-model" combination can be reliably selected without labeled data. The experiments explored three sets of parameter configurations to ensure consistency across benchmarks.

## Key Experimental Results

### Main Results

| Method | CrossNER Avg | 13 Benchmarks Avg | Gap with Best Zero-shot | Gap with Supervised |
| :--- | :--- | :--- | :--- | :--- |
| B2NER (Prev. SOTA) | 75.3 | - | - | -32.0 |
| GPT-5 mini (Supervisor) | 69.3 | 62.3 | - | - |
| Ours (DiZiNER) | 75.7 | 68.4 | +11.1 | -20.9 |

Ours achieved zero-shot SOTA on 14 out of 18 benchmarks, outperforming the GPT-5 mini supervisor by an average of +5.0 to +6.4 F1.

### Ablation Study

| Ablation Item | Impact |
| :--- | :--- |
| Remove final task objective | F1 dropped from 77.6 to 71.9 |
| Heterogeneous vs. Homogeneous pool | Heterogeneous pool performed 1.7-3.7 F1 better |
| Annotator count 4 → 8 | F1 increased from 73.1 to 75.5 |
| Annotator count > 12 | Performance degraded (consensus noise) |
| Using gold-labeled data | Only marginal improvement of +0.3 F1 |
| Optimal document set size | 15-25 samples |

### Key Findings

- Inter-model consistency is strongly correlated with NER performance and serves as a label-free quality metric.
- Heterogeneous model pools (≤24B) consistently outperform large model pools of the same family.
- Gold-labeled data provides minimal help to the framework, indicating that disagreement guidance itself is sufficiently effective.
- The average optimization cost per benchmark is only $40.1 ($1.90/round for inference + $0.77/round for supervision).

## Highlights & Insights

- The framework cleverly transfers the mature "pilot annotation" methodology from the human annotation field to the LLM context; this analogy is profound and practical.
- It surpasses the supervisor model without any parameter updates, proving that disagreement signals contain information far exceeding the capability ceiling of a single model.
- Inter-model consistency as a label-free performance proxy offers a feasible solution for quality monitoring in real-world deployment.
- The extremely low cost ($40 per benchmark) makes large-scale application possible.

## Limitations & Future Work

- A gap of approximately -20.9 F1 still exists between zero-shot and supervised methods, which has not been fully bridged.
- The framework has some dependence on the supervisor model's capabilities; performance varies with different supervisor models.
- The fixed 20% threshold may lead to over-correction, where performance in some benchmarks peaks early and then declines.
- The small document set size (25 samples) may limit coverage of complex tasks.

## Related Work & Insights

- Unlike instruction fine-tuning methods like InstructUIE or GoLLIE, DiZiNER is entirely training-free.
- It is complementary to encoder-based methods like UniversalNER and GLiNER, which focus on inference efficiency.
- Self-iterative methods like EvoPrompt use self-generated pseudo-samples, whereas DiZiNER leverages inter-model disagreement as a stronger signal.
- **Insight**: Multi-model disagreement signals may play a similar role in more Information Extraction (IE) tasks, such as Relation Extraction and Event Extraction.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Systematically transfers pilot annotation methodology to zero-shot NER for LLMs; the concept is novel and execution is complete.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 18 benchmarks, multiple ablations, cost analysis, and robustness validation; experiments are extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Framework description is clear and mathematical notation is standardized, though some details are quite dense.
- **Value**: ⭐⭐⭐⭐⭐ Provides a low-cost, training-free, high-performance zero-shot NER solution with high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SAM-NER: Semantic Archetype Mediation for Zero-Shot Named Entity Recognition](sam-ner_semantic_archetype_mediation_for_zero-shot_named_entity_recognition.md)
- [\[ACL 2026\] Reasoning-Based Refinement of Unsupervised Text Clusters with LLMs](reasoning-based_refinement_of_unsupervised_text_clusters_with_llms.md)
- [\[ACL 2026\] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)
- [\[ACL 2026\] Refining and Reusing Annotation Guidelines for LLM Annotation](refining_and_reusing_annotation_guidelines_for_llm_annotation.md)
- [\[ACL 2026\] MADE: A Living Benchmark for Multi-Label Text Classification with Uncertainty Quantification](made_a_living_benchmark_for_multi-label_text_classification_with_uncertainty_qua.md)

</div>

<!-- RELATED:END -->
