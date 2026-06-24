---
title: >-
  [Paper Note] Pharos-ESG: A Framework for Multimodal Parsing, Contextual Narration, and Hierarchical Labeling of ESG Reports
description: >-
  [AAAI 2026][Multimodal VLM][ESG report parsing] This paper proposes Pharos-ESG, a unified framework for structured parsing of ESG reports via four core modules: layout-flow-based reading order modeling, table-of-contents (ToC) anchor-guided hierarchical reconstruction, context-aware multimodal image-to-text conversion, and multi-level financial label prediction. The framework achieves an F1 of 93.59, ROKT of 0.92, and TBTA of 92.46% in comprehensive evaluation…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "ESG report parsing"
  - "document understanding"
  - "reading order modeling"
  - "hierarchical structure reconstruction"
  - "multi-level label prediction"
date: 2026-05-08
content_hash: 8fd92d5adeac61c9
---

# Pharos-ESG: A Framework for Multimodal Parsing, Contextual Narration, and Hierarchical Labeling of ESG Reports

**Conference**: AAAI 2026
**arXiv**: [2511.16417](https://arxiv.org/abs/2511.16417)  
**Code**: [https://github.com/liucun-zy/Pharos-ESG](https://github.com/liucun-zy/Pharos-ESG)  
**Area**: Multimodal VLM
**Keywords**: ESG report parsing, document understanding, reading order modeling, hierarchical structure reconstruction, multi-level label prediction

## TL;DR
This paper proposes Pharos-ESG, a unified framework for structured parsing of ESG reports via four core modules: layout-flow-based reading order modeling, table-of-contents (ToC) anchor-guided hierarchical reconstruction, context-aware multimodal image-to-text conversion, and multi-level financial label prediction. The framework achieves an F1 of 93.59, ROKT of 0.92, and TBTA of 92.46% in comprehensive evaluation, substantially outperforming baselines such as MinerU, GPT-4o, and Gemini 2.5 Pro. The authors also release Aurora-ESG, the first large-scale public ESG report dataset comprising over 24K reports.

## Background & Motivation

**Background**: ESG (Environmental, Social, and Governance) principles are reshaping global financial governance, transitioning from voluntary disclosure to mandatory reporting and emerging as a critical infrastructure connecting corporations, investors, and regulators. ESG reports serve as the primary medium for assessing corporate ESG performance.

**Limitations of Prior Work**: ESG reports are typically published as visually dense, long-form PDFs, presenting two key technical challenges:

**Disordered reading order**: Report layouts are highly heterogeneous, with text, tables, and figures interleaved in slide-like formats, exhibiting inconsistencies even in ostensibly structured sections such as the table of contents.

**Implicit hierarchical structure**: Reports commonly exceed 50 pages and lack standardized structural indicators such as numbered headings or consistent formatting, making hierarchical organization difficult to recover.

**Why existing approaches are insufficient**:
- **Indirect proxies**: Financial research is compelled to rely on simple disclosure metrics, small-scale case studies, or third-party ratings, bypassing the rich semantics of the reports themselves.
- **General-purpose document parsers** (MinerU, Docling, Marker): Designed for regularly structured formats such as academic papers and legal contracts, these tools suffer severe performance degradation on the irregular layouts of ESG reports.
- **General-purpose multimodal models** (GPT-4o, Gemini): Prone to hallucination and computationally expensive when reconstructing implicit hierarchies in lengthy, weakly structured documents.

**Key Insight**: This work designs a unified framework tailored to the specific characteristics of ESG reports, achieving structured parsing along four dimensions—reading order → hierarchical structure → multimodal content → financial labels. It also releases the first large-scale public ESG report dataset to address the gap in data resources.

## Method

### Overall Architecture

Pharos-ESG comprises four core modules: (a) layout-flow-based reading order modeling; (b) ToC anchor-guided structural reconstruction; (c) visual-element-to-natural-language contextual conversion; and (d) multi-level ESG/GRI/sentiment label annotation.

### Key Designs

1. **Reading Order Modeling**

    - **Function**: After extracting multimodal elements from ESG reports, the global sequence ordering problem is reformulated as a pairwise successor classification task.
    - **Block content encoding**: Each page is represented as a set of content blocks $\mathcal{D}_p = \{(w_i, b_i, c_i, p)\}$, comprising content, bounding box, block type, and page number.
    - **Multimodal feature construction**: For each ordered block pair $(i,j)$, a feature vector $\varphi_{ij}$ is constructed, integrating semantic (LayoutLMv3 encoding), spatial (centroid offset, IoU, distance), and categorical signals.
    - **Relation prediction**: A Relation-Aware Transformer (RAT) predicts via cross-attention whether block $j$ directly follows block $i$, computing the successor probability $s_{ij} = \sigma(\mathbf{W} \cdot \text{Transformer}(\varphi_{ij}) + \mathbf{b})$.
    - **Topological sorting**: A directed weighted graph is constructed, and a topological sort yields a globally consistent, acyclic reading sequence.
    - **Design Motivation**: The slide-like layout of ESG reports invalidates conventional top-to-bottom, left-to-right rules, necessitating a learned approach to model complex reading flows.

2. **ToC-Guided Hierarchical Structure Reconstruction**

    - **RAP (Region-Aware Prompting)**: A visual prompting strategy that leverages color similarity, spatial proximity, and textual adjacency to guide MLLMs in inferring implicit hierarchies. It comprises four components: cross-region entry aggregation, context-aware label enrichment, region-based hierarchy inference, and multi-line merging.
    - **ALIGN (Anchor-based Linguistic Indexing for Granular Navigation)**: A multi-stage alignment algorithm with three phases:
        - Exact matching: character-level matching to identify high-confidence anchors.
        - Fuzzy/containment matching: Levenshtein similarity and substring containment to extend coverage.
        - Context-aware insertion: LLM reasoning to resolve unmatched headings (summarize paragraph → assess whether an overview heading is missing → identify improvement locations → select the optimal insertion point).
    - **Design Motivation**: ToC formats in ESG reports are highly varied and hierarchies are implicit; conventional approaches relying on visual cues such as font size and indentation are unreliable.

3. **Image-to-Text Conversion**

    - **Hierarchy-guided multimodal aggregation**: Target images are integrated with surrounding content into coherent multimodal clusters, guided by nearby section headings while preserving reading order.
    - **Contextualized image caption generation**: A two-stage process — structured semantic modeling (encoding hierarchical path and element declarations) followed by multimodal embedding and semantic generation (ViT extracts visual features → Q-Former projects features → Qwen2.5-VL-Instruct generates captions).
    - **Design Motivation**: Charts and images in ESG reports require contextual grounding for accurate interpretation; isolated descriptions discard critical contextual information.

4. **MLPDH: Multi-Level Financial Label Prediction with Dynamic Hierarchy**

    - **Ternary embedding**: Textual semantics (the [CLS] token from Chinese-RoBERTa-wwm-ext) + hierarchical context (GRU-encoded heading path) + global reading order position.
    - **Hierarchical attention**: Stacked attention layers propagate hierarchical signals: $\mathbf{v}^{(h)}_{blk} = \text{softmax}(\frac{(\mathbf{W}_q \mathbf{e}_{blk})^\top (\mathbf{W}_k \mathbf{v}^{(h-1)}_{ref})}{\sqrt{d}}) \cdot \mathbf{W}_v \mathbf{v}^{(h-1)}_{ref}$
    - **Hierarchical consistency constraint**: Parent–child label dependency penalty jointly optimized with BCE loss.
    - **Label structure**: Three-level prediction spanning ESGN categories → GRI indicators → sentiment labels.
    - **Design Motivation**: ESG analysis requires multi-dimensional annotation; hierarchical structural information is closely correlated with financial labels.

## Key Experimental Results

### Main Results

| Method | Type | Parsing F1↑ | ROKT↑ | TBTA(%)↑ |
|------|------|---------|-------|----------|
| Marker | Dedicated parser | 39.88 | 0.34 | 3.79 |
| MinerU | Dedicated parser | 76.89 | 0.82 | 6.94 |
| Textin | Dedicated parser | 82.55 | 0.80 | 9.68 |
| GPT-4o | General multimodal | 65.17 | 0.75 | 43.55 |
| Gemini 2.5 Pro | General multimodal | 87.50 | 0.75 | 64.30 |
| **Pharos-ESG** | **Ours** | **93.59** | **0.92** | **92.46** |

### Ablation Study

| Config | Component | F1↑ | Note |
|------|------|-----|------|
| Config 1 | None | 76.95 | Comparable to general-purpose parsers |
| Config 2 | +GP-based ToC | 78.79 | +1.84; limited effect of general prompting |
| Config 3 | +RAP replacing GP | 83.57 | +4.78; RAP significantly outperforms general prompting |
| Config 4 | +Reading order modeling | 88.14 | +4.57; large recall gain from reading order |
| Config 5 | +ALIGN first two stages | 90.05 | +1.91; gain from exact/fuzzy matching |
| Config 6 | +Full ALIGN | **93.59** | +3.54; context-aware insertion resolves unmatched headings |

**Multi-level label prediction**:

| Method | ESGN F1 | GRI F1 | Sentiment F1 | Macro-F1 | HLA |
|------|---------|--------|--------|----------|-----|
| SVM+TF-IDF | 72.14 | 61.59 | 68.31 | 67.35 | - |
| BERT-base | 80.21 | 72.30 | 77.61 | 76.71 | 81.31 |
| HMCN | 82.70 | 76.86 | 79.07 | 79.54 | 88.15 |
| **MLPDH** | **85.62** | **84.23** | **89.11** | **86.32** | **94.78** |

**Cross-market generalization**:

| Market | Parsing F1 | ROKT | TBTA | Macro-F1 |
|------|--------|------|------|----------|
| China A-share | 92.04 | 0.92 | 92.46 | 86.32 |
| Hong Kong | 89.05 | 0.88 | 89.50 | 87.20 |
| United States | **94.30** | **0.94** | **94.80** | **87.60** |

### Key Findings

1. **Dedicated parsers nearly completely fail on the TBTA task** (< 20%), as heuristics based on font size and indentation are unreliable in ESG reports.
2. **The RAP strategy yields substantial gains**: Compared to general prompting (GP), RAP achieves average improvements of +9.89% / +12.02% / +14.51% on CC/RC/HC metrics.
3. **Reading order modeling is strongly correlated with structural extraction quality**: ROKT exhibits a positive correlation with parsing metrics.
4. **U.S. market reports are parsed most accurately**: Their formats are more standardized, hierarchies are clearer, and segmentation is more consistent.
5. **The ternary embedding and hierarchical constraints in MLPDH are critical**: BERT-base achieves an HLA of only 81.31; the absence of cross-level constraints leads to parent–child label inconsistencies.

## Highlights & Insights

1. **End-to-end document understanding for a vertical domain**: Rather than a general-purpose parser, the framework is customized to the characteristics of ESG reports—an approach with reference value for other specialized document types.
2. **RAP visual prompting strategy**: Color, spatial, and textual adjacency cues are leveraged to guide MLLMs in understanding implicit hierarchies, with potential generalization to other weakly structured documents.
3. **ALIGN three-stage alignment algorithm**: The progressive design moves from exact matching to fuzzy matching to LLM-based reasoning, handling alignment cases of increasing difficulty.
4. **Aurora-ESG dataset contribution**: Over 24K reports, 8M+ content blocks, covering three markets (China, U.S., Hong Kong), addressing a significant data resource gap in the field.
5. **Comprehensive financial annotation schema**: Three-level labels spanning ESGN → GRI → sentiment directly serve downstream financial analysis needs.

## Limitations & Future Work

1. **Evaluation is primarily focused on Chinese ESG reports**: Although cross-market generalization is tested, the training set of 50 Chinese reports is relatively small.
2. **Computational resource requirements**: The complete pipeline involves multiple models (LayoutLMv3, Qwen2.5-VL, etc.), resulting in non-trivial deployment costs.
3. **ALIGN's dependence on LLMs**: The context-aware insertion stage relies on LLM reasoning, which may produce errors in complex cases.
4. **Dataset quality**: Aurora-ESG is automatically generated by the framework and may contain systematic biases.
5. **Scalability**: The transferability of the method to other types of long documents (e.g., annual reports, legal documents) outside the ESG domain has not been validated.

## Related Work & Insights

- **Document parsing**: From PubLayNet to DocLayNet, layout analysis has progressively extended from regularly structured to irregularly structured documents; ESG reports represent a new frontier of challenge.
- **Cross-modal models such as LayoutLMv3**: Effective for semantic alignment but constrained in simultaneously understanding spatially dispersed yet semantically related elements in ESG reports.
- **Long document understanding**: The attention degradation problem (recognition failure beyond 20 pages) is exacerbated in ESG reports.
- Insight: Domain-specific document understanding may require customized "domain knowledge injection" strategies; general-purpose multimodal large models cannot resolve all challenges.
- The ToC as a structural "anchor" for document understanding is a generalizable and promising design principle.

## Rating
- Novelty: ⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Multimodal DeepResearcher: Generating Text-Chart Interleaved Reports From Scratch with Agentic Framework](multimodal_deepresearcher_generating_text-chart_interleaved_.md)
- [\[ACL 2026\] SlideAgent: Hierarchical Agentic Framework for Multi-Page Visual Document Understanding](../../ACL2026/multimodal_vlm/slideagent_hierarchical_agentic_framework_for_multi-page_visual_document_underst.md)
- [\[AAAI 2026\] Plug-and-Play Clarifier: A Zero-Shot Multimodal Framework for Egocentric Intent Disambiguation](plug-and-play_clarifier_a_zero-shot_multimodal_framework_for_egocentric_intent_d.md)
- [\[ICML 2026\] FlowNar: Scalable Streaming Narration for Long-Form Videos](../../ICML2026/multimodal_vlm/flownar_scalable_streaming_narration_for_long-form_videos.md)
- [\[AAAI 2026\] PlantTraitNet: An Uncertainty-Aware Multimodal Framework for Global-Scale Plant Trait Inference from Citizen Science Data](planttraitnet_an_uncertainty-aware_multimodal_framework_for_global-scale_plant_t.md)

</div>

<!-- RELATED:END -->
