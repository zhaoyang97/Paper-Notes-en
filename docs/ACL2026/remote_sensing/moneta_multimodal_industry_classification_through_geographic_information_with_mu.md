---
title: >-
  [Paper Note] MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems
description: >-
  [ACL 2026][Remote Sensing][Industry Classification] The paper proposes MONETA, the first multimodal industry classification benchmark combining text (websites, Wikipedia, Wikidata) and geospatial data (OpenStreetMap, satellite imagery), with zero-shot and multi-turn multi-agent training-free pipelines using open-source and proprietary MLLMs achieving 62.10%-74.10% accuracy on 20-class NACE industry classification, with multi-turn design improving up to 22.80%.
tags:
  - ACL 2026
  - Remote Sensing
  - Industry Classification
  - Geographic Information
  - Multimodal LLM
  - Multi-Agent
  - OpenStreetMap
date: 2025-05-08
content_hash: b2310b81944550a2
---

# MONETA: Multimodal Industry Classification through Geographic Information with Multi Agent Systems

**Conference**: ACL 2026  
**arXiv**: [2604.07956](https://arxiv.org/abs/2604.07956)  
**Code**: [GitHub](https://github.com/trusthlt/Moneta)  
**Area**: Remote Sensing / Multimodal Understanding  
**Keywords**: Industry Classification, Geographic Information, Multimodal LLM, Multi-Agent, OpenStreetMap

## TL;DR

The paper proposes MONETA, the first multimodal industry classification benchmark combining text (websites, Wikipedia, Wikidata) and geospatial data (OpenStreetMap, satellite imagery), with zero-shot and multi-turn multi-agent training-free pipelines using open-source and proprietary MLLMs achieving 62.10%-74.10% accuracy on 20-class NACE industry classification, with multi-turn design improving up to 22.80%.

## Background & Motivation

**Background**: Industry classification schemes (e.g., NACE, ISIC, GICS) are core components of public and enterprise databases. Existing automated classification methods primarily rely on text (company descriptions, financial reports, websites) and typically require model fine-tuning.

**Limitations of Prior Work**: (1) Text-only methods are inapplicable for newly established or small enterprises that may lack public text information; (2) Fine-tuned models require extensive data collection and cannot transfer across classification schemes; (3) Geospatial information (e.g., company locations and surroundings) contains strong industry cues but has never been systematically utilized.

**Key Challenge**: Business activities are highly correlated with spatial locations (factories in industrial zones, banks on commercial streets), but existing industry classification completely ignores this spatial-economic correlation.

**Goal**: Construct the first multimodal industry classification benchmark and explore whether MLLMs can utilize geospatial information for industry classification.

**Key Insight**: Use OpenStreetMap and satellite imagery as complementary information sources alongside text, with a multi-agent architecture where different modality cues are extracted by specialized agents and then synthesized by a decision agent.

**Core Idea**: Multimodal resources + multi-agent + training-free—each resource is processed by a specialized agent extracting economic activity cues, and a decision agent synthesizes all cues for classification, entirely without training.

## Method

### Overall Architecture

MONETA has two pipelines: (1) Zero-Shot—feeds all available resources to an MLLM at once for direct classification; (2) Multi-Turn—two stages: a clue extraction stage where each resource is processed by an independent MLLM agent to generate economic activity cues, and a decision stage where a decision agent synthesizes all cues and entity names for final classification.

### Key Designs

1. **NACE-to-OSM Mapping Construction**:

    - Function: Establishes correspondence between industry classification schemes and geographic data
    - Mechanism: First uses Gemini to generate OSM tags from NACE official guide RDF/XML, then iteratively corrects through human review and GPT/Gemini, producing validated OSM tag lists for each NACE section. Then queries European OSM data by tags and applies quality filters (name, address, external links)
    - Design Motivation: This mapping did not previously exist and serves as the bridge connecting economic activities to spatial data; human review ensures mapping quality

2. **Multi-Turn Multi-Agent Pipeline**:

    - Function: Allows independent extraction and analysis of cues from each information source
    - Mechanism: Designs specialized clue extraction agents for each resource (OSM, satellite images, Wikidata, Wikipedia, websites), generating free-text with economic activity keywords. The decision agent receives all intermediate cues, entity names, and NACE section descriptions for final classification
    - Design Motivation: Single-pass inference makes MLLMs prone to confusion when processing multiple modalities simultaneously; extracting by modality then synthesizing better matches human expert review workflows

3. **Clue Analysis Methodology (Frequency Vectors)**:

    - Function: Quantifies intermediate agent contributions and correctness toward final predictions
    - Mechanism: Groups each agent's extracted keywords by NACE section into normalized frequency vectors. Constructs correctness vectors and effectiveness vectors by selecting indices corresponding to ground-truth and predicted labels respectively. Correctness measures clue relevance to true labels; effectiveness measures clue influence on final predictions
    - Design Motivation: Understanding each agent's contribution in multi-agent systems is essential—identifying which resources provided correct cues and which misled final decisions

### Loss & Training

Fully training-free framework. Evaluated InternVL 2.5/3, Llava 1.6, QwenVL 2.5 and other open-source models, as well as Gemini 2.5 and GPT-5 proprietary models.

## Key Experimental Results

### Main Results

**Zero-Shot classification accuracy under different input configurations (selected models)**

| Model | No extra input | Satellite | External text | All inputs |
|-------|---------------|-----------|--------------|------------|
| InternVL 2.5-38B | 46.30 | 49.80 | 58.40 | 60.10 |
| InternVL 3-78B | 43.40 | 47.80 | 60.40 | 58.80 |
| QwenVL 2.5-72B | — | — | — | ~62% |

### Ablation Study

**Multi-Turn vs Zero-Shot improvement**

| Config | Note |
|--------|------|
| Multi-turn + context-rich + explanation | Maximum improvement +22.80% |
| Extended prompt (with NACE descriptions) | Significantly better than simple prompt |
| Satellite imagery | Limited alone, but gains when combined with text |

### Key Findings

- External text resources (websites/Wikipedia) contribute the most to classification; satellite imagery alone has limited effectiveness
- Multi-turn multi-agent pipeline consistently outperforms zero-shot pipeline, with maximum improvement of 22.80%
- Classification with explanations (JSON output with reasoning) achieves higher accuracy than label-only output
- Proprietary models (GPT-5, Gemini 2.5) reach ~74%, significantly outperforming open-source models
- Clue analysis reveals OSM and websites have the highest effectiveness; satellite imagery has lower correctness but complements text

## Highlights & Insights

- The first introduction of geospatial information into industry classification opens a new cross-domain research direction
- The NACE-to-OSM mapping itself is a valuable research artifact reusable by subsequent work
- The frequency vector clue analysis method provides a quantitative tool for evaluating intermediate steps in multi-agent systems

## Limitations & Future Work

- The 1,000-sample benchmark is relatively small, with only 50 samples per class
- Geospatial resource resolution and coverage vary by region
- Finer-grained NACE classification (e.g., 88 divisions or 272 groups) has not been explored
- Satellite imagery contribution is limited; higher resolution or better visual understanding capabilities may be needed

## Related Work & Insights

- **vs Text-only industry classification (Kühnemann et al.)**: The latter uses only website text; this paper introduces geospatial modalities
- **vs Remote sensing classification (UC Merced, AID)**: The latter performs land use classification rather than enterprise-level industry classification
- **vs Fine-tuning methods**: Fine-tuning requires extensive labeled data and is limited to a single classification scheme; this paper's training-free framework has greater adaptability

## Rating

- Novelty: ⭐⭐⭐⭐ First multimodal industry classification benchmark; geospatial + text combination is novel
- Experimental Thoroughness: ⭐⭐⭐⭐ Multi-model × multi-config × multi-pipeline + novel clue analysis method
- Writing Quality: ⭐⭐⭐⭐ Clear research question, detailed dataset construction process
- Recommendation: ⭐⭐⭐⭐ The open-source benchmark and mapping have significant value for subsequent research

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] GeoMMBench and GeoMMAgent: Toward Expert-Level Multimodal Intelligence in Geoscience and Remote Sensing](../../CVPR2026/remote_sensing/geommbench_and_geommagent_toward_expert_level_multimodal_intelligence_in_geoscience_and_remote_sensing.md)
- [\[AAAI 2026\] Perceive, Act and Correct: Confidence Is Not Enough for Hyperspectral Classification](../../AAAI2026/remote_sensing/perceive_act_and_correct_confidence_is_not_enough_for_hyperspectral_classificati.md)
- [\[ICLR 2026\] Earth-Agent: Unlocking the Full Landscape of Earth Observation with Agents](../../ICLR2026/remote_sensing/earth-agent_unlocking_the_full_landscape_of_earth_observation_with_agents.md)
- [\[NeurIPS 2025\] OrbitZoo: Real Orbital Systems Challenges for Reinforcement Learning](../../NeurIPS2025/remote_sensing/orbitzoo_real_orbital_systems_challenges_for_reinforcement_learning.md)
- [\[ICCV 2025\] Information-Bottleneck Driven Binary Neural Network for Change Detection](../../ICCV2025/remote_sensing/information-bottleneck_driven_binary_neural_network_for_change_detection.md)

<!-- RELATED:END -->
