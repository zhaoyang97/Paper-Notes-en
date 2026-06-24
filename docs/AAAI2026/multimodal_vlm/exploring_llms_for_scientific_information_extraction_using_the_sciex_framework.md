---
title: >-
  [Paper Note] Exploring LLMs for Scientific Information Extraction using the SciEx Framework
description: >-
  [AAAI 2026][Multimodal VLM][Scientific Information Extraction] This paper proposes SciEx, a modular and composable scientific information extraction framework that decouples PDF parsing, multimodal retrieval, schema-guided extraction, and cross-document aggregation into independent components. The framework evaluates the extraction capabilities of GPT-4o and Gemini-2.5-Flash on a dataset of 143 papers spanning medicine and environmental science…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "Scientific Information Extraction"
  - "LLM"
  - "RAG"
  - "Multimodal Reasoning"
  - "Modular Framework"
date: 2026-05-08
content_hash: a52815ed4afa1a6f
---

# Exploring LLMs for Scientific Information Extraction using the SciEx Framework

**Conference**: AAAI 2026
**arXiv**: [2512.10004](https://arxiv.org/abs/2512.10004)  
**Code**: None  
**Area**: Multimodal / Information Extraction
**Keywords**: Scientific Information Extraction, LLM, RAG, Multimodal Reasoning, Modular Framework

## TL;DR
This paper proposes SciEx, a modular and composable scientific information extraction framework that decouples PDF parsing, multimodal retrieval, schema-guided extraction, and cross-document aggregation into independent components. The framework evaluates the extraction capabilities of GPT-4o and Gemini-2.5-Flash on a dataset of 143 papers spanning medicine and environmental science, revealing systematic deficiencies in current LLMs with respect to cross-modal reasoning, numerical precision, and domain generalization.

## Background & Motivation
Scientific information extraction requires compiling structured knowledge—experimental parameters, relationships, and results—from free-text papers, which constitutes a critical step in the digitization of scientific research. Although LLMs demonstrate strong performance on general NLP tasks, they face unique challenges in scientific information extraction:

**Multimodal dispersion**: Scientific knowledge is distributed across text, tables, and figures, requiring cross-modal reasoning to capture dependencies among methods, results, and interpretations. Traditional extractors based on local or sentence-level context cannot effectively aggregate cross-document dependencies.

**Terminological and unit inconsistency**: The same concept may be expressed in multiple ways (e.g., "SARS-CoV-2 persistence" vs. "COVID-19 virus viability"), and numerical units are often non-uniform (molarity vs. ppm), violating the assumptions of schema-constrained extractors.

**Long-document dependencies**: Evidence is scattered across different sections of a paper (methods, results, supplementary materials) and frequently exceeds model context windows.

**High schema variability**: Different research objectives may require different data schemas, making it costly to redesign or fine-tune systems accordingly.

The central positioning of this paper is not to outperform a specific state-of-the-art method, but rather to **provide an honest capability assessment**: to what extent can current LLMs perform scientific IE, and where are the bottlenecks?

## Method

### Overall Architecture
SciEx adopts a Map-Reduce-style distributed processing pipeline: a PDF Extractor combined with a REV module performs the Map operation independently on each paper (applying identical extraction logic), while a Schema Aggregator serves as the Reduce operation to consolidate cross-document outputs. The framework comprises four core modules: PDF preprocessing → Schema processing → Retrieval-Extraction-Verification (REV) → Aggregation and conflict resolution.

### Key Designs
1. **PDF Preprocessing and Multimodal Knowledge Base**:

    - Function: Parses scientific PDFs into a structured multimodal database.
    - Mechanism: Docling is used to extract text and images; text is segmented into semantically coherent chunks. Scientific figures are filtered via a VLM binary classifier (distinguishing scientific figures from decorative ones), with each figure paired with its caption (original or VLM-generated). A VLM further parses axis labels, legends, and data points into structured JSON. All pages are additionally saved as full-page images to support joint reasoning.
    - Design Motivation: A large portion of critical data in scientific papers resides in figures; purely text-based extraction discards important information. The multimodal knowledge base provides a unified indexing foundation for downstream RAG retrieval.

2. **Schema Module**:

    - Function: Defines the structured representation of extraction targets.
    - Mechanism: Supports two modes — (1) explicit schema definition, where users specify attributes and data types; and (2) implicit schema description, where users provide natural-language instructions and an LLM automatically generates the corresponding structured schema.
    - Design Motivation: Accommodates both domain experts (who require precise constraints) and general users (who prefer natural-language descriptions). Schema-guided extraction ensures consistency across papers.

3. **Retrieval-Extraction-Verification (REV) Loop Module**:

    - Function: Iteratively discovers, extracts, and verifies relevant information.
    - Mechanism: A three-step closed loop — (1) *Retrieval*: using the schema as a query blueprint, vector-based semantic search retrieves the top-k relevant evidence segments (text, tables, figure JSON) from the multimodal database; (2) *Extraction*: the LLM performs schema-guided structured extraction over the retrieved evidence, annotating each extracted element with provenance metadata (document ID, chunk index, figure reference); (3) *Verification*: extracted results undergo self-verification, with missing or uncertain fields triggering targeted follow-up queries that re-enter the retrieval stage.
    - Design Motivation: Single-pass extraction is often incomplete; the closed-loop iterative mechanism ensures semantic consistency and empirical completeness of the output.

4. **Aggregation and Conflict Resolution Module**:

    - Function: Merges extraction results from multiple papers into a unified, schema-consistent representation.
    - Mechanism: Results are grouped by shared entities or experimental conditions → unit normalization → LLM-driven terminology normalization (mapping variants to canonical schema terms) → hierarchical conflict resolution (cross-model ensemble and consistency voting). Genuinely missing information is explicitly marked as null.
    - Design Motivation: The use of different terminology and units to describe the same concept across papers is a core challenge in scientific IE; automated normalization and conflict resolution are necessary components of any practical system.

### Loss & Training
SciEx is a purely prompt-driven RAG framework and involves no model training. GPT-4o and Gemini-2.5-Flash serve as the extraction models, with each retrieval round returning the top-5 relevant chunks. Automatic prompt optimization via DSPy is also supported. Evaluation employs a bipartite matching algorithm with row-level matching to align ground truth with extracted results.

## Key Experimental Results

### Main Results

| Dataset | Model | Precision | Recall | F1-score | Accuracy |
|---------|-------|-----------|--------|----------|----------|
| CFS | Gemini-2.5-Flash | 0.169 | 0.273 | 0.175 | 0.507 |
| CFS | GPT-4o | 0.241 | 0.355 | 0.248 | 0.512 |
| UV | Gemini-2.5-Flash | 0.199 | 0.468 | 0.237 | 0.329 |
| UV | GPT-4o | 0.279 | 0.609 | 0.331 | 0.467 |
| VD | Gemini-2.5-Flash | 0.284 | 0.382 | 0.297 | 0.556 |
| VD | GPT-4o | 0.333 | 0.476 | 0.380 | 0.580 |

### Ablation Study

| Analysis Dimension | Key Findings | Notes |
|-------------------|--------------|-------|
| Cross-dataset | Simpler datasets (UV, VD) outperform complex ones (CFS) | CFS requires integrating information across multiple tables and figures |
| Cross-model | GPT-4o consistently outperforms Gemini-2.5-Flash | Avg. Precision 0.26 vs. 0.22; Recall 0.48 vs. 0.37 |
| Recall > Precision | Consistent across both models | Extracted content is relevant but includes a large number of extraneous data points |
| Post-localization accuracy | Moderate (0.5–0.6) | Once a record is correctly located, field-level extraction is relatively reliable |

### Key Findings
- **Performance is far from production-ready**: Even after extensive prompt optimization and RAG augmentation, the best F1 is only 0.38 (VD + GPT-4o), well below the threshold for reliable deployment.
- **Figure parsing is the primary bottleneck**: Low resolution in older PDFs, truncated axes, and overlapping curves introduce visual complexity that severely degrades numerical precision.
- **Cross-sentence and cross-paragraph reasoning is weak**: When multiple entities appear in close proximity, LLMs frequently misattribute experimental conditions.
- **Table structural diversity is difficult to handle**: Nested headers, merged cells, and cross-table variable associations cause schema mismatches.
- **Simple tasks (direct reading from a single figure or table) are handled reasonably well, while complex tasks (multi-source integration) degrade substantially.**

## Highlights & Insights
- The modularity and composability of the framework design represent a genuine engineering contribution — any component can be independently replaced or upgraded.
- Diverging from the mainstream "chasing SOTA" paradigm, the paper honestly exposes the limitations of LLMs in scientific IE, offering greater constructive value to the community.
- The manually annotated dataset of 143 papers was labeled by domain PhD students, ensuring high quality and broad coverage.
- The error analysis section is exceptionally detailed and practical, providing clear directions for future improvement.
- The Map-Reduce analogy makes the system architecture intuitive and easy to understand.

## Limitations & Future Work
- Overall performance is low, particularly in terms of Precision, indicating a need for better relevance filtering mechanisms.
- No comparison with fine-tuned models is provided; all inference is conducted in a zero- or few-shot setting.
- All three datasets are drawn from medicine and environmental science; generalizability to other STEM fields (e.g., physics, chemistry, materials science) remains unknown.
- The stopping conditions for the iterative REV module (maximum rounds or confidence thresholds) lack systematic investigation.
- The potential benefit of multi-LLM ensemble approaches for improving Precision has not been explored.
- The quality of the figure VLM classifier and JSON-based parsing itself warrants validation.

## Related Work & Insights
- SciEx shares conceptual similarities with ChatExtract (conversational iterative extraction) but places greater emphasis on modularity and multimodal integration.
- The approach in SciDaSynth—combining automated extraction with human verification—is worth adopting in SciEx; introducing human-in-the-loop collaboration under low-precision conditions may represent a practical compromise.
- For researchers conducting systematic reviews or meta-analyses, SciEx provides a valuable (if imperfect) automated starting point.
- The taxonomy of error types (parsing quality, cross-sentence reasoning, table structure, figure numerics) defines clear research challenges for both the NLP and document AI communities.

## Rating
- Novelty: ⭐⭐⭐ (The framework design is sound, but the components are largely combinations of existing techniques.)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Three datasets, two models, and detailed error analysis, though comparisons with fine-tuned methods are absent.)
- Writing Quality: ⭐⭐⭐⭐ (Clear structure; error analysis is particularly valuable.)
- Value: ⭐⭐⭐⭐ (Honestly delineates the capability boundaries of LLMs in scientific IE, offering practical guidance.)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] LiveWeb-IE: A Benchmark For Online Web Information Extraction](../../ICLR2026/multimodal_vlm/liveweb-ie_a_benchmark_for_online_web_information_extraction.md)
- [\[CVPR 2025\] Relation-Rich Visual Document Generator for Visual Information Extraction](../../CVPR2025/multimodal_vlm/relation-rich_visual_document_generator_for_visual_information_extraction.md)
- [\[ACL 2025\] Table Understanding and (Multimodal) LLMs: A Cross-Domain Case Study on Scientific Tables](../../ACL2025/multimodal_vlm/table_understanding_and_multimodal_llms_a_cross-domain_case_study_on_scientific_.md)
- [\[AAAI 2026\] Knowledge Completes the Vision: A Multimodal Entity-aware Retrieval-Augmented Generation Framework for News Image Captioning](knowledge_completes_the_vision_a_multimodal_entity-aware_retrieval-augmented_gen.md)
- [\[AAAI 2026\] Information Theoretic Optimal Surveillance for Epidemic Prevalence in Networks](information_theoretic_optimal_surveillance_for_epidemic_prevalence_in_networks.md)

</div>

<!-- RELATED:END -->
