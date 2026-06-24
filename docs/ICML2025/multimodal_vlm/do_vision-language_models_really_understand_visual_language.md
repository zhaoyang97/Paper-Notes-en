---
title: >-
  [Paper Note] Do Vision-Language Models Really Understand Visual Language?
description: >-
  [ICML 2025][Multimodal VLM][Visual Language Understanding] This work systematically evaluates the diagram understanding capabilities of large vision-language models (LVLMs) by constructing a comprehensive test suite (including synthetic and real-world diagrams). It reveals that while models can identify entities, their understanding of relationships is extremely limited; their seemingly excellent performance in diagram reasoning actually stems from utilizing background knowle…
tags:
  - "ICML 2025"
  - "Multimodal VLM"
  - "Visual Language Understanding"
  - "Diagram Understanding"
  - "LVLM Evaluation"
  - "Relational Reasoning"
  - "Background Knowledge Shortcut"
date: 2026-05-08
content_hash: 8f0cb7c1768bda36
---

# Do Vision-Language Models Really Understand Visual Language?

**Conference**: ICML 2025  
**arXiv**: [2410.00193](https://arxiv.org/abs/2410.00193)  
**Code**: None  
**Area**: Multimodal VLM  
**Keywords**: Visual Language Understanding, Diagram Understanding, LVLM Evaluation, Relational Reasoning, Background Knowledge Shortcut

## TL;DR
This work systematically evaluates the diagram understanding capabilities of large vision-language models (LVLMs) by constructing a comprehensive test suite (including synthetic and real-world diagrams). It reveals that while models can identify entities, their understanding of relationships is extremely limited; their seemingly excellent performance in diagram reasoning actually stems from utilizing background knowledge as a shortcut.

## Background & Motivation

**Background**: Visual Language is a communication system that conveys information through symbols, shapes, and spatial arrangements. Diagrams are a typical example, depicting complex concepts and their relationships in image format. Recently, Large Vision-Language Models (LVLMs) have achieved rapid progress in multimodal understanding, and many studies claim that models are already capable of handling complex diagram reasoning tasks.

**Limitations of Prior Work**: The symbolic nature of diagrams requires models to simultaneously understand: (a) concept entity recognition (e.g., node labels, text content), and (b) relationships of entities (e.g., hierarchical structure, causal relationships, chronological order). However, existing evaluations primarily focus on the correctness of the final answer, failing to distinguish whether the models "truly" understand the diagram structure or are utilizing alternative cues.

**Key Challenge**: The high scores of models on diagram reasoning benchmarks vs. whether they truly understand the structured information in visual language. There exists a major **evaluation loophole**: if models can guess relationships between entities by identifying entity names and utilizing world knowledge gained from pre-training, then high scores do not represent true diagram understanding capabilities.

**Goal**: To systematically evaluate the "true" diagram understanding capabilities of LVLMs, specifically distinguishing between **entity recognition capability** and **relationship reasoning capability**, and to reveal the "shortcut" role of background knowledge in model performance.

**Key Insight**: Designing synthetic diagrams (with complete control over entities and relations) and real-world diagrams, constructing multi-type questions (entity recognition, relational reasoning), and using controlled experiments to **remove background knowledge cues** to discern the true capabilities of models.

**Core Idea**: The diagram reasoning capability of LVLMs is an "illusion" — primarily relying on background knowledge shortcuts rather than a true understanding of visual structures.

## Method

### Overall Architecture
This work is an **evaluation study** rather than proposing a new model. The core contribution lies in designing a comprehensive test suite, including:

- **Input**: Synthetic diagrams + real-world cross-domain diagrams
- **Evaluation Dimensions**: Entity recognition, relationship reasoning
- **Control Design**: Contrastive conditions with/without background knowledge cues
- **Output**: Performance analysis of each LVLM across different dimensions

### Key Designs

1. **Synthetic Diagram Generation**:

    - Generating diagrams automatically, offering precise control over nodes (entities) and edges (relationships)
    - Utilizing various diagram types: flowcharts, tree structures, entity-relationship diagrams, etc.
    - Flexibly replacing entity labels (e.g., using meaningless random characters to replace real conceptual names), thereby eliminating the interference of background knowledge
    - **Design Motivation**: Synthetic diagrams enable fully controlled experiments. In contrast, entities and relationships in real-world diagrams might have appeared in the pre-training data, making it impossible to disentangle "understanding" from "memorization."

2. **Multi-level Question Design**:

    - **Entity-level Questions**: "What nodes are in the diagram?", "What is a certain label?" — testing visual recognition capability.
    - **Relationship-level Questions**: "What is the relationship between A and B?", "Who is whose parent/predecessor?" — testing spatial/structural understanding capability.
    - **Reasoning-level Questions**: Compound questions requiring multi-step reasoning, such as "How many intermediate nodes are passed to go from A to C?"
    - **Design Motivation**: Vertically assessing "seeing" vs. "understanding" to reveal the true limits of model capabilities.

3. **Background Knowledge Elimination Experiment (Debiasing Experiment)**:

    - For the same diagram structure, divided into two groups:
        - **With Semantic Labels**: Utilizing real concept names (e.g., "cell division" -> "DNA replication" -> "mitosis")
        - **Without Semantic Labels**: Replacing concepts with random strings (e.g., "XYZ" -> "ABC" -> "MNP")
    - Comparing the accuracy of relationship reasoning under both conditions.
    - **Design Motivation**: A significant drop in relationship reasoning performance without semantic labels indicates the model relies on background knowledge rather than understanding the visual structure.

### Loss & Training
This work does not involve model training and is purely an evaluation study. The evaluated models include major LVLMs such as GPT-4V, Gemini Pro Vision, LLaVA, and InstructBLIP.

## Key Experimental Results

### Main Results

| Evaluation Dimension | GPT-4V | Gemini Pro | LLaVA-1.5 | InstructBLIP |
|----------------------|--------|-----------|-----------|--------------|
| Entity Recognition (Semantic) | ~85% | ~78% | ~65% | ~60% |
| Relationship Reasoning (Semantic) | ~72% | ~65% | ~45% | ~40% |
| Relationship Reasoning (Non-semantic) | ~35% | ~30% | ~20% | ~18% |
| Performance Drop | -37pp | -35pp | -25pp | -22pp |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Semantic Labels + Entity Questions | High Accuracy (~80%+) | Models excel at text recognition |
| Semantic Labels + Relationship Questions | Medium Accuracy (~60-70%) | Partially relies on background knowledge |
| Non-semantic Labels + Relationship Questions | Low Accuracy (~20-35%) | Reveals that true diagram understanding is extremely weak |
| Synthetic Diagrams vs. Real Diagrams | Synthetic diagrams are slightly lower | Real diagrams offer more leverageable background knowledge |
| Simple Relations vs. Complex Relations | High on simple, low on complex | Multi-step reasoning capability is even poorer |

### Key Findings
1. **Huge Gap Between Entity Recognition and Relationship Reasoning**: Models can reasonably identify entities in diagrams (thanks to powerful OCR and object recognition capabilities), but their understanding of spatial/logical relationships between entities is extremely weak.
2. **Background Knowledge Dominates Relational Reasoning**: When semantic labels are removed, relationship reasoning accuracy plunges by 30-40 percentage points, confirming that the models rely heavily on world knowledge rather than visual cues.
3. **Model Scale Cannot Compensate for Deficiencies**: Even the strongest model, GPT-4V, achieves only ~35% in relationship reasoning under non-semantic conditions, indicating this is not a simple model capacity issue.

## Highlights & Insights
- **Unveiling a critical "illusion"**: The seemingly robust diagram understanding capability is actually a "shortcut" of background knowledge, sounding an alarm for downstream applications (such as scientific literature understanding and automated report generation) that rely on LVLMs for diagram analysis.
- **Simple and effective experimental design**: Clear disentanglement of the two capabilities is achieved at minimal cost through a controlled setup of synthetic diagrams with semantic vs. non-semantic labels.
- **Contribution to evaluation methodologies**: Reminding the research community to control confounding variables (background knowledge) when evaluating diagram understanding; otherwise, benchmark scores may overestimate model capabilities.

## Limitations & Future Work
- The evaluation scope could be further extended to wider ranges of visual language (e.g., maps, musical notation, circuit diagrams, etc.).
- The concrete mechanism of the background knowledge shortcut is not analyzed in depth — i.e., which internal layers / attention heads of the model utilize background knowledge.
- No improvement solutions are proposed to enhance the "true" diagram understanding capabilities of the models.
- Synthetic diagrams might be oversimplified, leaving a gap with real-world complex diagrams.

## Related Work & Insights
- Echoes research on "hallucinations": model outputs appear correct but are not based on the correct reasoning process.
- Inspires future evaluation designs to include "debiased / de-confounded" conditions to avoid illusions of high scores.
- For tasks requiring true structured understanding (e.g., scientific diagram analysis), more specialized architectural designs may be necessary.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematically reveal the "illusion" nature of LVLM diagram understanding, with clever experimental design.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models, various diagram types, with controlled experiments, offering comprehensive quantitative analysis.
- Writing Quality: ⭐⭐⭐⭐ Clear argumentative logic and powerful conclusions.
- Value: ⭐⭐⭐⭐ Plays an important corrective role in community cognition, providing a significant warning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Vision-Language Models Do Not Understand Negation](../../CVPR2025/multimodal_vlm/vision-language_models_do_not_understand_negation.md)
- [\[ACL 2025\] Can Vision Language Models Understand Mimed Actions?](../../ACL2025/multimodal_vlm/can_vision_language_models_understand_mimed_actions.md)
- [\[ACL 2025\] NegVQA: Can Vision Language Models Understand Negation?](../../ACL2025/multimodal_vlm/negvqa_can_vision_language_models_understand_negation.md)
- [\[ACL 2025\] Can Multimodal Large Language Models Understand Spatial Relations?](../../ACL2025/multimodal_vlm/spatialmqa_mllm_spatial_relations.md)
- [\[CVPR 2025\] Words or Vision: Do Vision-Language Models Have Blind Faith in Text?](../../CVPR2025/multimodal_vlm/words_or_vision_do_vision-language_models_have_blind_faith_in_text.md)

</div>

<!-- RELATED:END -->
