---
title: >-
  [Paper Note] MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models
description: >-
  [ACL 2026][Multimodal VLM][Medical vision-language models] This paper presents MedLayBench-V, the first large-scale multimodal medical expert-lay semantic alignment benchmark (79…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Medical vision-language models"
  - "expert-lay semantic alignment"
  - "medical text simplification"
  - "UMLS"
  - "multimodal benchmark"
date: 2026-05-08
content_hash: d6f03fffe320cf70
---

# MedLayBench-V: A Large-Scale Benchmark for Expert-Lay Semantic Alignment in Medical Vision Language Models

**Conference**: ACL 2026
**arXiv**: [2604.05738](https://arxiv.org/abs/2604.05738)  
**Code**: [GitHub](https://github.com/) (Project Page available)  
**Area**: Multimodal VLM / Medical NLP
**Keywords**: Medical vision-language models, expert-lay semantic alignment, medical text simplification, UMLS, multimodal benchmark

## TL;DR

This paper presents MedLayBench-V, the first large-scale multimodal medical expert-lay semantic alignment benchmark (79,793 image-text pairs). Through a Structured Concept-Grounded Refinement (SCGR) pipeline, professional radiology reports are transformed into lay descriptions, reducing reading difficulty from graduate level to high school level while preserving clinical semantic fidelity. Zero-shot retrieval experiments demonstrate that lay descriptions incur less than 1% performance degradation.

## Background & Motivation

**Background**: Medical vision-language models (Med-VLMs) have achieved expert-level performance in diagnostic image interpretation, yet are predominantly trained on professional literature and produce outputs dominated by clinical terminology. In the text domain, medical lay language generation (MLLG) is relatively mature, with shared tasks such as BioLaySumm driving advances in medical text simplification.

**Limitations of Prior Work**: (1) Existing multimodal medical datasets (e.g., ROCOv2, PMC-OA) consist entirely of professional-level reports with no lay annotations; (2) directly generating lay descriptions with LLMs introduces hallucination risks—approximately 6–7% of simplified reports contain factual errors or critical omissions; (3) conventional n-gram metrics (BLEU, ROUGE) inherently penalize lexical substitution and are ill-suited for evaluating expert-to-lay translation quality.

**Key Challenge**: The lay language capability established in the text domain has not penetrated multimodal systems—VLMs can encode visual features into technical terms such as "Pneumothorax," but lack training data to learn the corresponding lay expression "collapsed lung."

**Goal**: To construct the first multimodal medical bilingual-domain benchmark (expert + lay) that supports training and evaluation of Med-VLMs capable of bridging the communication gap between clinical experts and patients.

**Key Insight**: Drawing on the practice of leveraging structured medical knowledge to enhance summarization relevance in the text domain, this work extends that approach to the multimodal setting, ensuring semantic fidelity of lay descriptions through UMLS ontology mapping and NER entity constraints.

**Core Idea**: Explicitly decouple semantic extraction from stylistic rewriting—first extract semantic constraints via UMLS CUI mapping and NER, then perform lay rewriting under those constraints using an LLM—thereby enabling controllable language simplification while preventing hallucinations.

## Method

### Overall Architecture

The SCGR pipeline comprises three stages: (1) **Concept-Knowledge Alignment**—extracting a semantic constraint set $C$ from expert reports; (2) **Knowledge-Constrained Refinement**—synthesizing a lay draft under the constraints and refining it with an LLM; (3) **LLM Refinement**—using Llama-3.1-8B-Instruct to optimize grammar and fluency while preserving semantic equivalence. The input consists of expert-level image-text pairs ($T_{exp}$) from the ROCOv2 dataset; the output is the corresponding lay version ($T_{lay}$).

### Key Designs

1. **Two-Level Semantic Constraint Extraction (Concept-Knowledge Alignment)**

    - **Function**: Establishes a semantic bridge from expert reports to lay descriptions, ensuring complete preservation of diagnostic information.
    - **Mechanism**: At the macro level, the UMLS Metathesaurus API maps clinical terms to CUIs (e.g., C0040405 → "CTPA"), yielding an ontology constraint set $C_{onto}$. At the micro level, the SciSpacy NER model extracts quantitative attributes and spatial descriptors (e.g., lesion size), yielding an entity constraint set $C_{ent}$. The final constraint set is $C = C_{onto} \cup C_{ent}$.
    - **Design Motivation**: Pure CUI mapping misses numerical and spatial details, while pure NER lacks high-level semantic anchoring. The two-level combination ensures that both core pathological concepts and key quantitative information are captured.

2. **Constraint-Guided Lay Rewriting (Knowledge-Constrained Refinement)**

    - **Function**: Reduces linguistic complexity from graduate to high school level while strictly preserving diagnostic accuracy.
    - **Mechanism**: MedlinePlus patient-friendly definitions are first retrieved from UMLS; deterministic dictionary substitution generates an initial lay draft $T_{draft}$ (potentially grammatically rough but lexically reliable). Llama-3.1-8B-Instruct then refines the draft under a structured prompt containing the original report $T_{exp}$ (factual anchoring), the constraint set $C$ (hallucination prevention), and the initial draft $T_{draft}$ (lexical guidance).
    - **Design Motivation**: Llama-3.1-8B is chosen over larger models because structured constraints handle semantic fidelity; the LLM need only optimize grammatical fluency, a task manageable by a smaller model at the scale of ~80K samples.

3. **Multi-Dimensional Quality Validation**

    - **Function**: Ensures that lay outputs meet standards across three dimensions: relevance, readability, and clinical correctness.
    - **Mechanism**: Relevance is measured by BLEU-4/ROUGE-L/METEOR for surface similarity; readability is assessed by FKGL, CLI, and LENS (a learnable metric designed for text simplification); clinical correctness is evaluated by RaTEScore and GREEN for hallucination and factual error detection. Human evaluation is conducted by two radiologists and one lay reader on a 5-point scale.
    - **Design Motivation**: Effective MLLG evaluation must jointly consider visual grounding, factual correctness, and lay accessibility; no single metric can adequately capture all dimensions.

### Loss & Training

The SCGR pipeline is a data construction method and does not involve end-to-end training. Llama-3.1-8B-Instruct is used in inference mode without fine-tuning. Downstream experiments adopt a zero-shot retrieval protocol for evaluation.

## Key Experimental Results

### Main Results

**Zero-Shot Image-Text Retrieval Performance (Recall@1, %)**

| Model | Image→Text (Expert / Layman) | Text→Image (Expert / Layman) |
|---|---|---|
| BiomedCLIP | 31.06 / 30.70 | 32.50 / 32.07 |
| PMC-CLIP | 28.98 / 28.38 | 30.90 / 30.24 |
| BMC-CLIP | 22.69 / 22.42 | 23.04 / 23.21 |
| PubMedCLIP | 4.61 / 4.26 | 4.85 / 4.71 |
| OpenCLIP-Huge | 3.33 / 3.44 | 5.17 / 5.15 |
| OpenAI-CLIP | 1.23 / 1.08 | 1.57 / 1.54 |

### Ablation Study

| SCGR Configuration | CUI | MedlinePlus | LLM | Avg. R@1 |
|---|---|---|---|---|
| LLM Only | ✗ | ✗ | ✓ | 1.96 |
| LLM + CUI | ✓ | ✗ | ✓ | 2.08 |
| SCGR (Full) | ✓ | ✓ | ✓ | 11.26 |
| Expert (Original) | — | — | — | 11.44 |

### Key Findings

- Retrieval performance degradation after lay simplification is minimal—BiomedCLIP I2T R@1 drops only from 31.06% to 30.70%, confirming that SCGR successfully preserves core diagnostic semantics.
- Removing structured constraints (LLM Only) causes R@1 to plummet by 83% (from 11.44 to 1.96), demonstrating that constraint guidance is critical for hallucination prevention.
- The readability metric FKGL decreases from 13.10 to 10.35, with vocabulary size reduced by 46.1%, indicating substantial improvement in readability.
- Human evaluation scores exceed 4.5/5.0 across all four dimensions, with factual correctness and completeness reaching 4.86.
- Domain-adapted Med-VLMs substantially outperform general-purpose VLMs (BiomedCLIP R@1 ~31% vs. OpenAI-CLIP ~1%), underscoring the importance of domain adaptation.

## Highlights & Insights

- The explicit decoupling of semantic extraction from stylistic rewriting is the core innovation—determining *what to say* before addressing *how to say it* fundamentally mitigates the hallucination problems common in end-to-end generation. This principle is transferable to any task requiring preservation of semantics under stylistic transformation.
- Using MedlinePlus as a lay simplification bridge is both authoritative and practical—NLM's patient education vocabulary is inherently an "expert→lay" mapping dictionary, making direct utilization more reliable than training a model to learn the mapping.
- The ablation study clearly shows that CUI extraction is only a necessary condition; the substantive performance recovery stems from the knowledge-constrained refinement via MedlinePlus.

## Limitations & Future Work

- Reliance on synthetic data—lay descriptions are generated by an LLM rather than authored by humans, potentially lacking the linguistic nuances present in authentic patient communication.
- Coverage is limited to English; multilingual medical lay language needs remain unaddressed.
- The modality imbalance inherited from ROCOv2 is not corrected.
- Future work may extend the benchmark to more complex downstream tasks such as visual question answering and report generation to more fully expose expert-lay representational alignment gaps.

## Related Work & Insights

- **vs. BioLaySumm**: BioLaySumm is a text-only lay simplification shared task; MedLayBench-V is the first multimodal counterpart, adding a visual grounding dimension.
- **vs. Layman's RRG**: Limited to chest X-rays as a single modality with small data scale; MedLayBench-V covers 7 modalities with ~80K samples.
- **vs. End-to-End LLM Simplification**: Direct LLM simplification incurs a 6–7% factual error rate; SCGR minimizes hallucinations through structured constraints.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First multimodal medical expert-lay alignment benchmark; the SCGR pipeline design is elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Zero-shot retrieval across 8 models, ablation study, and human evaluation; fine-tuning experiments are absent.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Rigorous structure, clear motivation, and convincing ablations.
- **Value**: ⭐⭐⭐⭐⭐ Fills a critical resource gap for patient-centered multimodal medical AI.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Doc-PP: Document Policy Preservation Benchmark for Large Vision-Language Models](doc-pp_document_policy_preservation_benchmark_for_large_vision-language_models.md)
- [\[ACL 2026\] MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models](mmerror_a_benchmark_for_erroneous_reasoning_in_vision-language_models.md)
- [\[CVPR 2026\] Uncertainty-guided Compositional Alignment with Part-to-Whole Semantic Representativeness in Hyperbolic Vision-Language Models](../../CVPR2026/multimodal_vlm/uncertainty-guided_compositional_alignment_with_part-to-whole_semantic_represent.md)
- [\[ACL 2026\] More Than Meets the Eye: Measuring the Semiotic Gap in Vision-Language Models via Semantic Anchorage](more_than_meets_the_eye_measuring_the_semiotic_gap_in_vision-language_models_via.md)
- [\[CVPR 2026\] MoDES: Accelerating Mixture-of-Experts Multimodal Large Language Models via Dynamic Expert Skipping](../../CVPR2026/multimodal_vlm/modes_accelerating_mixture-of-experts_multimodal_large_language_models_via_dynam.md)

</div>

<!-- RELATED:END -->
