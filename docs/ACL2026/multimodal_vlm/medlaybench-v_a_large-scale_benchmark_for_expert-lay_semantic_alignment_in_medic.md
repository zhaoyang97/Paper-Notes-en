---
title: >-
  [Paper Note] MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models
description: >-
  [ACL 2026][Multimodal VLM][Medical Vision-Language Models] This paper introduces MedLayBench-V, the first large-scale multimodal medical expert-lay semantic alignment benchmark (79…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Medical Vision-Language Models"
  - "Expert-Lay Semantic Alignment"
  - "Medical Text Simplification"
  - "UMLS"
  - "Multimodal Benchmark"
date: 2026-05-08
content_hash: 0b82e6f83eca5a67
---

# MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models

**Conference**: ACL 2026 Oral Findings  
**arXiv**: [2604.05738](https://arxiv.org/abs/2604.05738)  
**Code**: [GitHub](https://github.com/) (Available via Project Page)  
**Area**: Multimodal VLM / Medical NLP  
**Keywords**: Medical Vision-Language Models, Expert-Lay Semantic Alignment, Medical Text Simplification, UMLS, Multimodal Benchmark

## TL;DR

This paper introduces MedLayBench-V, the first large-scale multimodal medical expert-lay semantic alignment benchmark (79,793 image-text pairs). Through a Structured Concept-Grounded Refinement (SCGR) pipeline, professional radiology reports are converted into layperson descriptions. This ensures clinical semantic fidelity while reducing reading difficulty from graduate-school to high-school levels. Zero-shot retrieval experiments demonstrate that lay descriptions result in less than 1% performance loss.

## Background & Motivation

**Background**: Medical Vision-Language Models (Med-VLMs) have reached expert-level proficiency in diagnostic image interpretation, yet they are primarily trained on professional literature, outputting technical clinical terminology. Research on Medical Lay Language Generation (MLLG) in the text domain is relatively mature, with shared tasks like BioLaySumm driving developments in medical text simplification.

**Limitations of Prior Work**: (1) Existing multimodal medical datasets (e.g., ROCOv2, PMC-OA) consist entirely of professional-grade reports without layperson annotations; (2) Direct use of LLMs for lay description generation poses hallucination risks—approximately 6-7% of simplified reports contain factual errors or omissions of key information; (3) Traditional n-gram metrics (BLEU, ROUGE) naturally penalize vocabulary substitution and are unsuitable for evaluating expert-to-lay translation quality.

**Key Challenge**: Layperson communication capabilities in the text domain have not yet permeated multimodal systems—VLMs can encode visual features into technical terms like "Pneumothorax" but lack training data to learn the corresponding lay expression "collapsed lung."

**Goal**: To construct the first multimodal medical dual-domain benchmark (expert + layperson) to support the training and evaluation of Med-VLMs capable of bridging the communication gap between clinical experts and patients.

**Key Insight**: Following the practice of utilizing structured medical knowledge to enhance summary relevance in the text domain, this work extends the approach to the multimodal domain. Semantic fidelity during layperson translation is ensured via UMLS ontology mapping and NER entity constraints.

**Core Idea**: Explicitly decouple semantic extraction from stylistic rewriting—first extract semantic constraints using UMLS CUI mapping and NER, then perform layperson rewriting with an LLM under these constraints to achieve controllable language simplification while preventing hallucinations.

## Method

### Overall Architecture

The SCGR pipeline consists of three stages: (1) Concept-Knowledge Alignment—extracting a semantic constraint set $C$ from expert reports; (2) Knowledge-Constrained Refinement—synthesizing a lay draft based on constraints and refining it with an LLM; (3) LLM Refinement—using Llama-3.1-8B-Instruct to optimize grammar and fluency while maintaining semantic equivalence. The input consists of expert-level image-text pairs from the ROCOv2 dataset ($T_{exp}$), and the output is the lay version ($T_{lay}$).

### Key Designs

1.  **Dual-layer Semantic Constraint Extraction (Concept-Knowledge Alignment)**:
    *   **Function**: Establish a semantic bridge from expert reports to lay descriptions, ensuring diagnostic information is fully preserved.
    *   **Mechanism**: At the macro level, the UMLS Metathesaurus API maps clinical terms to CUIs (e.g., C0040405 → "CTPA"), forming the ontology constraint set $C_{onto}$. At the micro level, a SciSpacy NER model extracts quantitative attributes and spatial descriptors (e.g., lesion size), forming the entity constraint set $C_{ent}$. The final constraint set is $C = C_{onto} \cup C_{ent}$.
    *   **Design Motivation**: Simple CUI mapping misses numerical and spatial details, while pure NER lacks high-level semantic anchoring. The dual-layer combination ensures both core pathological concepts and critical quantitative information are captured.

2.  **Constraint-Guided Layperson Refinement (Knowledge-Constrained Refinement)**:
    *   **Function**: Reduce language complexity from graduate-school to high-school level while strictly maintaining diagnostic precision.
    *   **Mechanism**: First, the MedlinePlus vocabulary in UMLS is queried for patient-friendly definitions, and deterministic dictionary substitution generates an initial lay draft $T_{draft}$ (reliable vocabulary, potentially rough grammar). Then, Llama-3.1-8B-Instruct refines this under structured prompt constraints—including the original text $T_{exp}$ (factual anchor), constraint set $C$ (hallucination prevention), and initial draft $T_{draft}$ (vocabulary guidance).
    *   **Design Motivation**: Llama-3.1-8B was chosen over larger models because the structured constraints handle semantic fidelity; the LLM only needs to optimize grammatical fluency, a task for which a smaller model is competent and efficient for ~80K samples.

3.  **Multi-dimensional Quality Validation System**:
    *   **Function**: Ensure layperson results meet standards across relevance, readability, and clinical correctness.
    *   **Mechanism**: Relevance is measured by surface similarity (BLEU-4/ROUGE-L/METEOR); readability is assessed via difficulty indices (FKGL, CLI) and LENS (a learnable metric for text simplification); clinical correctness is detected using RaTEScore and GREEN for hallucinations and factual errors. Human evaluation was conducted by two radiologists and one lay reader on a 5-point scale.
    *   **Design Motivation**: Effective MLLG evaluation requires simultaneous consideration of visual anchoring, factual correctness, and lay accessibility; a single metric cannot reflect quality comprehensively.

### Loss & Training

The SCGR pipeline is a data construction method and does not involve end-to-end training. Llama-3.1-8B-Instruct is used in inference mode without fine-tuning. Downstream experiments utilize a zero-shot retrieval protocol for evaluation.

## Key Experimental Results

### Main Results

**Zero-shot Image-Text Retrieval Performance (Recall@1, %)**

| Model | Image→Text (Expert / Layman) | Text→Image (Expert / Layman) |
| :--- | :--- | :--- |
| BiomedCLIP | 31.06 / 30.70 | 32.50 / 32.07 |
| PMC-CLIP | 28.98 / 28.38 | 30.90 / 30.24 |
| BMC-CLIP | 22.69 / 22.42 | 23.04 / 23.21 |
| PubMedCLIP | 4.61 / 4.26 | 4.85 / 4.71 |
| OpenCLIP-Huge | 3.33 / 3.44 | 5.17 / 5.15 |
| OpenAI-CLIP | 1.23 / 1.08 | 1.57 / 1.54 |

### Ablation Study

| SCGR Config | CUI | MedlinePlus | LLM | Average R@1 |
| :--- | :--- | :--- | :--- | :--- |
| LLM Only | ✗ | ✗ | ✓ | 1.96 |
| LLM + CUI | ✓ | ✗ | ✓ | 2.08 |
| SCGR (Full) | ✓ | ✓ | ✓ | 11.26 |
| Expert (Original)| — | — | — | 11.44 |

### Key Findings

*   The decrease in retrieval performance after layperson conversion is minimal—BiomedCLIP’s I2T R@1 dropped only from 31.06% to 30.70%, proving SCGR successfully preserves core diagnostic semantics.
*   Removing structured constraints (LLM Only) caused R@1 to plummet by 83% (from 11.44 to 1.96), confirming that constraint-to-guidance is critical for preventing hallucinations.
*   Readability indices (FKGL) improved from 13.10 to 10.35, with vocabulary size reduced by 46.1%, showing significant gains in accessibility.
*   Human evaluation scores exceeded 4.5/5.0 across four dimensions, with factual correctness and completeness reaching 4.86.
*   Medical-domain VLMs significantly outperformed general VLMs (BiomedCLIP R@1 ~31% vs. OpenAI-CLIP ~1%), emphasizing the importance of domain adaptation.

## Highlights & Insights

*   The explicit decoupling of semantic extraction and stylistic rewriting is a core innovation—ensuring "what to say" before determining "how to say it" fundamentally avoids common hallucination issues in end-to-end generation. This approach is transferable to any task requiring semantic invariance across different expression styles.
*   Utilizing MedlinePlus as a bridge for layperson translation is both authoritative and practical—the patient education vocabulary maintained by the NLM serves as a natural "expert-to-lay" mapping dictionary, which is more reliable than training a model to learn these mappings.
*   Ablation studies clearly demonstrate that CUI extraction is a necessary condition, but the true restoration of performance comes from the knowledge-constrained refinement provided by MedlinePlus.

## Limitations & Future Work

*   Dependency on synthetic data—lay descriptions are generated by LLMs rather than written by humans, which may lack the linguistic nuances found in real patient communication.
*   English-only coverage—multilingual medical layperson communication needs remain unaddressed.
*   Inheritance of modality imbalance issues from ROCOv2.
*   Future work could expand to more complex downstream tasks like Visual Question Answering (VQA) and Report Generation to further expose gaps in expert-lay representation alignment.

## Related Work & Insights

*   **vs. BioLaySumm**: While BioLaySumm is a text-only shared task for lay summarization, MedLayBench-V is the first multimodal version, adding a visual anchoring dimension.
*   **vs. Layman's RRG**: Limited to a single modality (Chest X-ray) with small data volume; MedLayBench-V covers 7 modalities with 80K samples.
*   **vs. End-to-End LLM Simplification**: Direct simplification using LLMs results in an error rate of 6-7%; SCGR minimizes hallucinations through structured constraints.

## Rating

*   Novelty: ⭐⭐⭐⭐⭐ First multimodal medical expert-lay alignment benchmark; SCGR pipeline design is ingenious.
*   Experimental Thoroughness: ⭐⭐⭐⭐ Zero-shot retrieval with 8 models + ablation + human evaluation, though lacks fine-tuning experiments.
*   Writing Quality: ⭐⭐⭐⭐⭐ Rigorous structure, clear motivation, and convincing ablations.
*   Value: ⭐⭐⭐⭐⭐ Fills a critical resource gap for patient-centered multimodal medical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] ChartDiff: A Large-Scale Benchmark for Comprehending Pairs of Charts](chartdiff_a_large-scale_benchmark_for_comprehending_pairs_of_charts.md)
- [\[ACL 2026\] Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models](cross-cultural_expert-level_art_critique_evaluation_with_vision-language_models.md)
- [\[ACL 2026\] Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models](doc-pp_document_policy_preservation_benchmark_for_large_vision-language_models.md)
- [\[ACL 2026\] MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models](mmerror_a_benchmark_for_erroneous_reasoning_in_vision-language_models.md)
- [\[ACL 2026\] More Than Meets the Eye: Measuring the Semiotic Gap in Vision-Language Models via Semantic Anchorage](more_than_meets_the_eye_measuring_the_semiotic_gap_in_vision-language_models_via.md)

</div>

<!-- RELATED:END -->
