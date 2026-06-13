---
title: >-
  [Paper Note] MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems
description: >-
  [ACL 2026][Multimodal VLM][Industry Classification] This paper proposes MONETA, the first multimodal industry classification benchmark combining text (websites, Wikipedia, Wikidata) and geospatial data (OpenStreetMap…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Industry Classification"
  - "Geographic Information"
  - "Multimodal LLM"
  - "Multi-Agent"
  - "OpenStreetMap"
date: 2026-05-08
content_hash: 43d2e9082f238363
---

# MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems

**Conference**: ACL 2026  
**arXiv**: [2604.07956](https://arxiv.org/abs/2604.07956)  
**Code**: [GitHub](https://github.com/trusthlt/Moneta)  
**Area**: Remote Sensing / Multimodal Understanding  
**Keywords**: Industry Classification, Geographic Information, Multimodal LLM, Multi-Agent, OpenStreetMap

## TL;DR

This paper proposes MONETA, the first multimodal industry classification benchmark combining text (websites, Wikipedia, Wikidata) and geospatial data (OpenStreetMap, satellite imagery). It designs two training-free pipelines—Zero-Shot and Multi-Turn Multi-Agent—achieving 62.10%-74.10% accuracy across 20 NACE industry categories using open-source and closed-source MLLMs, with the multi-turn design providing up to 22.80% gain.

## Background & Motivation

**Background**: Industry classification schemes (e.g., NACE, ISIC, GICS) are core components of public and corporate databases. Existing automated classification methods primarily rely on text (company descriptions, financial reports, websites) and typically require fine-tuning models.

**Limitations of Prior Work**: (1) Pure text methods are inapplicable to newly established or small enterprises that lack public textual information; (2) Fine-tuned models require massive data collection and cannot easily transfer across classification schemes; (3) Geospatial information (e.g., geographic location and surrounding environment) contains strong industrial clues but has never been systematically utilized.

**Key Challenge**: The economic activities of enterprises are highly correlated with their spatial locations (factories in industrial zones, banks on commercial streets), yet existing industry classification completely ignores this spatial-economic association.

**Goal**: To construct the first multimodal industry classification benchmark and explore whether MLLMs can utilize geospatial information for industry classification.

**Key Insight**: Treat OpenStreetMap and satellite imagery as complementary information sources alongside text. Use a multi-agent architecture to allow specialized agents to extract clues from different modalities, which are then synthesized by a decision agent.

**Core Idea**: Multimodal resources + Multi-agent + Training-free—each resource has a specialized agent to extract economic activity clues, and a decision agent synthesizes all clues for classification, requiring no training throughout the process.

## Method

### Overall Architecture

The MONETA framework includes two pipelines: (1) Zero-Shot—inputting all available resources into the MLLM at once to generate a classification; (2) Multi-Turn—divided into two stages: a clue extraction stage where each resource is processed by an independent MLLM agent to generate economic activity clues, and a decision stage where a decision agent synthesizes all clues and the entity name for the final classification.

### Key Designs

1.  **NACE-to-OSM Mapping Construction**:
    - **Function**: Establish the correspondence between industry classification schemes and geographic data.
    - **Mechanism**: First, Gemini generates OSM tags from the NACE official guide RDF/XML, followed by manual review and iterative correction using GPT/Gemini to obtain a verified OSM tag list for each NACE section. European OSM data is then queried by tags and filtered for quality (name, address, external links).
    - **Design Motivation**: This mapping did not previously exist and serves as the bridge connecting economic activities with spatial data; manual review ensures mapping quality.

2.  **Multi-Turn Multi-Agent Pipeline**:
    - **Function**: Enables independent extraction and analysis of clues from each information source.
    - **Mechanism**: Specialized clue extraction agents are designed for each resource (OSM, satellite imagery, Wikidata, Wikipedia, websites) to generate free text containing economic activity keywords. The decision agent receives all intermediate clues, entity names, and NACE section descriptions to perform the final classification.
    - **Design Motivation**: Single-turn inference often causes MLLMs to confuse different modalities; modality-specific extraction followed by synthesis better aligns with human expert review processes.

3.  **Clue Analysis Methodology (Frequency Vector)**:
    - **Function**: Quantify the contribution and correctness of intermediate agents toward the final prediction.
    - **Mechanism**: Keywords extracted by each agent are grouped by NACE section to form normalized frequency vectors. By selecting indices corresponding to ground truth and predicted labels, correctness and validity vectors are constructed respectively. Correctness measures the correlation between clues and ground truth, while validity measures the influence of clues on the final prediction.
    - **Design Motivation**: There is a need to understand individual agent contributions in multi-agent systems—identifying which resources provide correct clues and which mislead the final decision.

### Loss & Training

A fully training-free framework. Evaluations were conducted on open-source models such as InternVL 2.5/3, Llava 1.6, and QwenVL 2.5, as well as closed-source models like Gemini 2.5 and GPT-5.

## Key Experimental Results

### Main Results

**Zero-Shot classification accuracy under different input configurations (selected models)**

| Model | No extra input | Satellite | External Text | All inputs |
| :--- | :--- | :--- | :--- | :--- |
| InternVL 2.5-38B | 46.30 | 49.80 | 58.40 | 60.10 |
| InternVL 3-78B | 43.40 | 47.80 | 60.40 | 58.80 |
| QwenVL 2.5-72B | — | — | — | ~62% |

### Ablation Study

**Multi-Turn vs. Zero-Shot Gain**

| Configuration | Description |
| :--- | :--- |
| Multi-Turn + Context Rich + Explanation | Max Gain +22.80% |
| Expanded prompt (incl. NACE descriptions) | Significant improvement over simple prompts |
| Satellite imagery | Limited effect alone, but yields gain when combined with text |

### Key Findings

- External textual resources (websites/Wiki) contribute the most to classification, while satellite imagery shows limited effectiveness when used alone.
- The Multi-Turn Multi-Agent pipeline consistently outperforms the zero-shot pipeline, with a maximum improvement of 22.80%.
- Classification with explanations (JSON output including reasoning) yields higher accuracy than pure label output.
- Closed-source models (GPT-5, Gemini 2.5) reach ~74%, significantly outperforming open-source models.
- Clue analysis reveals that OSM and websites have the highest validity, while satellite imagery has lower correctness but complements textual data.

## Highlights & Insights

- Introduces geospatial information to industry classification for the first time—opening a new cross-disciplinary research direction.
- The NACE-to-OSM mapping is a valuable research output in itself, reusable for subsequent work.
- The frequency vector clue analysis method provides a quantitative tool for evaluating intermediate steps in multi-agent systems.

## Limitations & Future Work

- The benchmark scale of 1,000 samples is relatively small, with only 50 samples per class.
- The resolution and coverage of geospatial resources vary by region.
- Finer-grained NACE classifications (e.g., 88 divisions or 272 groups) were not explored.
- The contribution of satellite imagery is limited, possibly requiring higher resolution or better visual understanding capabilities.

## Related Work & Insights

- **vs. Text-only Industry Classification (Kühnemann et al.)**: The latter uses only website text, whereas Ours introduces the geospatial modality.
- **vs. Remote Sensing Classification (UC Merced, AID)**: The latter performs land-use classification rather than enterprise-level industry classification.
- **vs. Fine-tuning methods**: Fine-tuning requires large amounts of labeled data and is limited to a single classification scheme; the training-free framework in Ours is more adaptable.

## Rating

- Novelty: ⭐⭐⭐⭐ First multimodal industry classification benchmark; the combination of geospatial + text is novel.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models × multiple configurations × multiple pipelines + novel clue analysis method.
- Writing Quality: ⭐⭐⭐⭐ Research questions are clear, and the dataset construction process is detailed.
- Value: ⭐⭐⭐⭐ Open-source benchmark and mapping are significant for driving subsequent research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Why Keep Your Doubts to Yourself? Trading Visual Uncertainties in Multi-Agent Bandit Systems](../../ICLR2026/multimodal_vlm/why_keep_your_doubts_to_yourself_trading_visual_uncertainties_in_multi-agent_ban.md)
- [\[ACL 2026\] GeoArena: Evaluating Open-World Geographic Reasoning in Large Vision-Language Models](geoarena_evaluating_open-world_geographic_reasoning_in_large_vision-language_mod.md)
- [\[ACL 2026\] From Verbatim to Gist: Distilling Pyramidal Multimodal Memory via Semantic Information Bottleneck](from_verbatim_to_gist_distilling_pyramidal_multimodal_memory_via_semantic_inform.md)
- [\[ICLR 2026\] Multimodal Classification via Total Correlation Maximization](../../ICLR2026/multimodal_vlm/multimodal_classification_via_total_correlation_maximization.md)
- [\[AAAI 2026\] Large Language Models Meet Extreme Multi-label Classification: Scaling and Multi-modal Framework](../../AAAI2026/multimodal_vlm/large_language_models_meet_extreme_multi-label_classification_scaling_and_multi-.md)

</div>

<!-- RELATED:END -->
