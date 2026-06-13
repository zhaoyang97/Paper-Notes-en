---
title: >-
  [Paper Note] LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval
description: >-
  [AAAI 2026][Multi-Agent][Multi-Agent Framework] This paper proposes LLandMark, a modular multi-agent framework that achieves landmark-aware multimodal interactive video retrieval through landmark knowledge augmentation…
tags:
  - "AAAI 2026"
  - "Multi-Agent"
  - "Multi-Agent Framework"
  - "Landmark-Aware"
  - "Multimodal Video Retrieval"
  - "Vietnamese Scenes"
  - "CLIP"
  - "OCR"
date: 2026-05-08
content_hash: b5f3a735d58c1cf7
---

# LLandMark: A Multi-Agent Framework for Landmark-Aware Multimodal Interactive Video Retrieval

**Conference**: AAAI 2026  
**arXiv**: [2603.02888](https://arxiv.org/abs/2603.02888)  
**Code**: None  
**Area**: LLM Agents  
**Keywords**: Multi-Agent Framework, Landmark-Aware, Multimodal Video Retrieval, Vietnamese Scenes, CLIP, OCR  

## TL;DR

This paper proposes LLandMark, a modular multi-agent framework that achieves landmark-aware multimodal interactive video retrieval through landmark knowledge augmentation, LLM-assisted image retrieval, and OCR refinement modules, achieving a total score of 77.40/88 in the Vietnamese large-scale video retrieval challenge (HCMAIC 2025).

## Background & Motivation

**Background**: Retrieving information from large-scale heterogeneous video corpora faces challenges including multilingual variation, noisy text extraction, and cross-modal reasoning. Existing systems such as MAVEN introduce agent-based frameworks but with limited planning capabilities, while RAG-based fusion approaches encounter scalability difficulties.

**Limitations of Prior Work**: Most existing systems neglect spatial and cultural context, particularly landmark reasoning. In Vietnamese queries, landmark names cannot be directly matched to corresponding visual features via CLIP text embeddings, due to insufficient text-visual associations for Vietnamese landmarks in CLIP's training data.

**Key Challenge**: Textual semantic retrieval relies on lexical matching, yet a significant semantic gap exists between the textual name of a landmark and its visual appearance. For example, the visual features of "St. Joseph's Cathedral" correspond to "twin square bell towers, dark gray stone, Gothic architecture."

**Key Insight**: The paper designs a landmark knowledge agent that reformulates landmark names into detailed visual-descriptive prompts to bridge the semantic gap in CLIP's embedding space, while also introducing a fully automated LLM-assisted image-to-image retrieval pipeline.

## Method

### Overall Architecture

LLandMark consists of four tightly integrated stages: (1) query parsing and planning, (2) landmark knowledge augmentation, (3) parallel multimodal retrieval, and (4) reranking and answer synthesis. The system is built upon CLIP ConvNeXt-XXLarge for visual embeddings, Milvus as the vector database, and Elasticsearch for text retrieval.

### Key Designs

1. **Query Parsing and Planning Agent**

    - Analyzes user query intent (Vietnamese/English) and constructs a weighted search plan (SearchPlan).
    - Translates queries for semantic search into descriptive English to maximize alignment with the CLIP embedding space.
    - Retains Vietnamese landmark names and proper nouns for ASR/OCR search to ensure precise matching.
    - Automatically detects landmark entities and flags them for special augmentation in the subsequent stage.

2. **Landmark Knowledge Augmentation Agent**

    - Maintains a Vietnamese landmark knowledge base, with each landmark containing detailed visual and architectural attribute descriptions.
    - Replaces landmark names in semantic queries with rich descriptive prompts.
    - Example: "St. Joseph's Cathedral" is reformulated as "Twin square bell towers, dark gray stone, Gothic architecture."
    - This reformulation bridges the semantic gap in CLIP's visual embedding space.

3. **LLM-Assisted Landmark Image-to-Image Retrieval**

    - Fully automated pipeline: Gemini 2.5 Flash detects landmarks and generates image search queries.
    - Reference images are retrieved via Google Custom Search API, encoded with CLIP, and used for vector search in Milvus.
    - Eliminates the limitation of traditional image retrieval that requires users to manually provide reference images.
    - Reduces query ambiguity by using real landmark images as query anchors.

4. **OCR Refinement Module**

    - PaddleOCR extracts text from video frames but handles Vietnamese diacritics poorly.
    - Text is first normalized to a diacritic-free form to preserve lexical integrity.
    - Gemini 2.5 Flash combined with LlamaIndex is then used to automatically restore diacritics and correct spelling.

### Loss & Training

Multimodal fusion scoring is computed as a weighted average. Temporal retrieval uses minimum-score aggregation to ensure high relevance of retrieved videos across all query steps. Preprocessing employs TransNetV2 for shot segmentation, with 3 representative keyframes selected per shot (at percentiles 0.15, 0.5, and 0.85).

## Key Experimental Results

### Main Results (HCMAIC 2025 Qualification Round)

| Round | Score | Full Score | Rate |
|-------|-------|------------|------|
| Round 1 | 20.00 | 23.00 | 87.0% |
| Round 2 | 28.20 | 30.00 | 94.0% |
| Round 3 | 29.20 | 35.00 | 83.4% |
| **Total** | **77.40** | **88.00** | **87.9%** |

### Ablation Study (Qualitative Comparison)

| Query | Baseline CLIP Retrieval | LLandMark | Effect |
|-------|------------------------|-----------|--------|
| "Ba Dinh Pier night scene with glowing lights" | Returns irrelevant results | Correctly retrieves target video | Landmark augmentation effective |
| "Ben Thanh Market" | Misidentified as a generic market | Accurately identified and matched | Image retrieval pipeline effective |

### Key Findings

- Ranked in the top 56 among 680+ registered teams (approximately top 8%), validating the framework's competitiveness in large-scale evaluation.
- Landmark knowledge augmentation is significantly effective on culturally specific queries; landmark queries that baseline CLIP failed entirely were successfully retrieved.
- OCR refinement is critical for Vietnamese text processing, as diacritic errors in raw outputs severely degrade downstream retrieval quality.
- A score rate of 83.4% in Round 3 (the most complex round) demonstrates the system's robustness on high-difficulty tasks.

## Highlights & Insights

- **Reformulating landmarks into visual descriptions** is the most central contribution, converting the semantic gap problem into a text rewriting problem.
- **The fully automated image retrieval pipeline** eliminates the user burden of manually sourcing reference images.
- **The modular multi-agent design** allows each component to be upgraded independently, providing excellent engineering scalability.

## Limitations & Future Work

- The landmark knowledge base is currently curated manually, limiting coverage to known Vietnamese landmarks.
- The system relies on the Google Custom Search API for reference image retrieval, incurring API costs and rate limitations.
- Evaluation is conducted solely on the HCMAIC 2025 challenge dataset; generalization to other cultural contexts remains unverified.
- System latency and throughput metrics are not reported.

## Related Work & Insights

| Aspect | MAVEN (Predecessor System) | LLandMark |
|--------|---------------------------|-----------|
| Planning Capability | Limited agent planning | Query parsing + weighted search plan |
| Landmark Handling | No dedicated mechanism | Knowledge augmentation + visual reformulation |
| OCR Accuracy | DeepSolo + PARSeq | PaddleOCR + Gemini post-correction |
| Image Retrieval | Requires manual image input | Fully automated LLM-assisted pipeline |

vs. **General CLIP Retrieval Systems**: CLIP exhibits training data bias in culturally specific scenarios; LLandMark compensates for this limitation through a knowledge augmentation layer.

## Rating

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Novelty | ⭐⭐⭐⭐ | Landmark-aware augmentation combined with an LLM-assisted image retrieval pipeline is a novel and practical combination |
| Technical Depth | ⭐⭐⭐ | Individual technical components are mature; innovation lies at the system integration level |
| Experimental Thoroughness | ⭐⭐⭐ | Only challenge leaderboard rankings and qualitative comparisons are provided; systematic ablation studies are lacking |
| Practical Value | ⭐⭐⭐⭐ | Clear application prospects in tourism guidance, cultural heritage retrieval, and related scenarios |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] COACH: Collaborative Agents for Contextual Highlighting -- A Multi-Agent Framework for Sports Video Analysis](coach_collaborative_agents_for_contextual_highlighting_--_a_multi-agent_framewor.md)
- [\[AAAI 2026\] KDR-Agent: A Multi-Agent LLM Framework for Multi-Domain Low-Resource In-Context NER via Knowledge Retrieval](a_multi-agent_llm_framework_for_multi-domain_low-resource_in-context_ner_via_kno.md)
- [\[AAAI 2026\] BAMAS: Structuring Budget-Aware Multi-Agent Systems](bamas_structuring_budget-aware_multi-agent_systems.md)
- [\[AAAI 2026\] Conversational Learning Diagnosis via Reasoning Multi-Turn Interactive Learning](conversational_learning_diagnosis_via_reasoning_multi-turn_interactive_learning.md)
- [\[ACL 2026\] ConSensus: Multi-Agent Collaboration for Multimodal Sensing](../../ACL2026/multi_agent/consensus_multi-agent_collaboration_for_multimodal_sensing.md)

</div>

<!-- RELATED:END -->
