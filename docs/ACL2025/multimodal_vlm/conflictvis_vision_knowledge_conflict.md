---
title: >-
  [Paper Note] Insight Over Sight: Exploring the Vision-Knowledge Conflicts in Multimodal LLMs
description: >-
  [ACL 2025][Multimodal VLM][Vision-Knowledge Conflict] This paper presents the first systematic study of commonsense-level vision-knowledge conflicts in MLLMs. It proposes an automated framework to construct the ConflictVis benchmark (374 images + 1122 QA pairs), finding that MLLMs over-rely on parametric knowledge in approximately 20% of conflict scenarios (particularly in Yes-No and action-related questions). Additionally, a Focus-on-Vision prompting strategy is proposed to…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Vision-Knowledge Conflict"
  - "MLLM Hallucination"
  - "Counter-Commonsense Benchmark"
  - "Memorization Ratio"
  - "Focus-on-Vision Prompt"
date: 2026-05-08
content_hash: d3b144ba9ac86f9c
---

# Insight Over Sight: Exploring the Vision-Knowledge Conflicts in Multimodal LLMs

**Conference**: ACL 2025  
**arXiv**: [2410.08145](https://arxiv.org/abs/2410.08145)  
**Code**: None (benchmark data provided)  
**Area**: Multimodal VLM  
**Keywords**: Vision-Knowledge Conflict, MLLM Hallucination, Counter-Commonsense Benchmark, Memorization Ratio, Focus-on-Vision Prompt

## TL;DR
This paper presents the first systematic study of commonsense-level vision-knowledge conflicts in MLLMs. It proposes an automated framework to construct the ConflictVis benchmark (374 images + 1122 QA pairs), finding that MLLMs over-rely on parametric knowledge in approximately 20% of conflict scenarios (particularly in Yes-No and action-related questions). Additionally, a Focus-on-Vision prompting strategy is proposed to mitigate this issue.

## Background & Motivation

**Background**: MLLMs (e.g., GPT-4o, LLaVA) perform exceptionally well in tasks like image captioning and VQA by integrating visual encoders with LLMs. However, the inherent knowledge conflict in LLMs (parametric knowledge vs. external information) evolves into a new form in multimodal scenarios—namely, the conflict between visual inputs and the model's intrinsic commonsense knowledge.

**Limitations of Prior Work**:
   - Existing studies on visual-knowledge conflicts lack systematic evaluation: HallusionBench only uses manually edited infographics, AutoHallusion focuses solely on object existence/spatial relationships, and PhD relies on manual collection.
   - There is a lack of an automated pipeline for generating large-scale-conflict samples.
   - The classification of conflicts is not fine-grained enough (failing to distinguish between action and scene conflicts).
   - Question types are limited (mostly restricted to Yes-No questions).

**Key Challenge**: When visual information contradicts the LLM's commonsense knowledge (e.g., "a waitress signing a bill in the kitchen" instead of "washing dishes"), the model tends to ignore visual info and rely on its intrinsic knowledge to answer. This is fundamentally an issue of under-utilizing visual information in MLLMs, serving as a primary source of hallucination.

**Goal**:
   - How to automatically construct high-quality counter-commonsense visual benchmarks.
   - What are the performance patterns of MLLMs across different conflict types and question formats.
   - Whether existing mitigation methods (VCD, PAI, CoT) are effective, and if there are better strategies.

**Key Insight**: Leveraging Normalized Pointwise Mutual Information (NPMI) to automatically discover low-cooccurrence $\langle \text{Subject, Action, Place} \rangle$ triples as counter-commonsense scenarios, combined with text-to-image models to generate corresponding images, forming an automated benchmark construction pipeline.

**Core Idea**: Utilizing NPMI cooccurrence statistics to automatically construct counter-commonsense triples, generate images, and formulate multi-type QAs, thereby systematically evaluating the "memorization" behavior of MLLMs under vision-knowledge conflicts.

## Method

### Overall Architecture
**Input Corpus** (OMCS Commonsense Dataset) $\rightarrow$ **Knowledge Component Extraction** (extracting Subject/Action/Place phrases) $\rightarrow$ **Counter-commonsense Query Construction** (NPMI filtering for high-cooccurrence context + low-cooccurrence target) $\rightarrow$ **Multimodal Input Generation** (DALL·E 3 image generation + template-based generation of three types of QA: Yes-No/MC/OE) $\rightarrow$ **Human Quality Control** $\rightarrow$ **ConflictVis Benchmark** (374 images, 1122 QAs)

### Key Designs

1. **Automated Counter-commonsense Query Construction**:

    - **Function**: Automatically discovers triple scenarios that contradict common sense.
    - **Mechanism**:
        - Extract high-frequency Subject (100), Action (150), and Place (150) phrases from the OMCS corpus.
        - Measure co-occurrence relations between components using NPMI: $\text{NPMI}(C_X; C_Y) = \frac{\text{PMI}(C_X; C_Y)}{-\log_2 P(C_X, C_Y)}$
        - **High-cooccurrence Context**: Select top-$K$ pairs of (Subject, Place) or (Subject, Action) with the highest NPMI as the "normal background".
        - **Low-cooccurrence Target**: Given the context, select the top-$M$ Action/Place phrases with the lowest NPMI as the counter-commonsense elements.
        - Estimate co-occurrence probability $P(\cdot)$ using an LLM (Vicuna-13B).
    - **Design Motivation**: NPMI normalization avoids bias from high-frequency words, and the automated approach offers better scalability than manual construction. High-cooccurrence context ensures the scene itself is "normal", with only one unusual element.

2. **Multi-type Question Generation**:

    - **Function**: Generates three question types for each counter-commonsense scenario.
    - **Design**:
        - **Yes-No**: "Is the waitress in the kitchen signing a bill?" — directly presents the counter-commonsense statement.
        - **Multiple-Choice**: The correct option is the counter-commonsense action/scene, while distractor options are commonsense-aligned.
        - **Open-Ended**: "What is the waitress doing in the kitchen?" — requires free-form generation from the model.
    - **Design Motivation**: Different question types impose varying levels of adversarial pressure on the model. Yes-No questions most directly trigger a commonsense-based negative reaction.

3. **Memorization Ratio (MR)**:

    - **Function**: Quantifies the model's reliance on parametric knowledge.
    - **Core Formula**: $MR = \frac{P_K}{P_K + P_V}$
        - $P_K$: Answers consistent with the no-image condition (reliance on knowledge).
        - $P_V$: Answers consistent with the visual information (reliance on vision).
    - It is an elegant causal analysis method that classifies behaviors by comparing answers under image vs. no-image conditions.

4. **Focus-on-Vision (FoV) Prompting Strategy**:

    - **Function**: A simple yet effective mitigation strategy.
    - **Implementation**: Appending "Please focus on the visual information." after the textual query.
    - **Design Motivation**: Since the root cause is the under-utilization of visual information, the most direct approach is explicitly prompting the model to focus on vision.

### Evaluation Setup
- 9 MLLMs: LLaVA (8B/13B/34B), BLIP-2 (12.1B/13B), Qwen-VL (9.6B), GPT-4o, Claude-3.5-Sonnet
- Metrics: Accuracy, MR (Memorization Ratio, lower is better)

## Key Experimental Results

### Main Results (ConflictVis Accuracy)

| Model | Yes-No | Multiple-Choice | Open-Ended | Avg Acc |
|------|--------|-----------------|------------|---------|
| BLIP-2-12B | 39.3 | — | — | — |
| LLaVA-1.5-13B | 70.6 | 88.0 | 82.9 | 80.5 |
| LLaVA-NeXT-34B | 73.3 | 92.5 | 88.0 | 84.6 |
| Qwen-VL-Chat | 69.8 | 80.5 | 89.3 | 79.9 |
| GPT-4o | 74.9 | 97.1 | 97.9 | 89.9 |
| Claude-3.5-Sonnet | 56.4 | — | — | — |

All models perform **significantly worse** on Yes-No compared to MC and OE.

### Mitigation Method Comparison (LLaVA-1.5-13B)

| Method | Yes-No | MC | OE | Avg |
|------|--------|-----|-----|-----|
| Baseline | 70.6 | 88.0 | 82.9 | 80.5 |
| +VCD | 72.7 | 89.3 | 84.2 | 82.1 |
| +PAI | **85.6** | 88.8 | 86.1 | **86.8** |
| +VR (CoT) | 38.0 ↓↓ | 89.8 | 76.7 | 68.2 |
| +FoV (Ours) | 82.9 | 89.0 | 81.8 | 84.6 |

| Method (LLaVA-NeXT-34B) | Yes-No | MC | OE | Avg |
|------|--------|-----|-----|-----|
| Baseline | 73.3 | 92.5 | 88.0 | 84.6 |
| +VR (CoT) | 43.6 ↓↓ | 87.2 | 72.5 | 67.7 |
| +FoV | **85.8** | 92.5 | **89.8** | **89.4** |

### Key Findings
- **Approximately 20% of responses over-rely on parametric knowledge**, ignoring visual information.
- **Yes-No questions are most likely to trigger knowledge overriding**: Claude-3.5-Sonnet achieves an MR of 43.6%, because Yes-No questions directly present counter-commonsense statements, triggering a negative reaction from the model.
- **Action conflicts are more difficult to resolve than scene conflicts**: Action accuracy is 73.9% vs. scene accuracy of 85.2%, with an MR of 23.8% vs. 13.4%. This is because scenes provide richer contextual clues for inference, whereas actions rely on fine-grained visual details.
- **CoT reasoning is unexpectedly harmful**: Prompting the model to "reason step-by-step" intensifies its reliance on prior knowledge. This occurs because the generated rationale continually reinforces commonsense priors, leading to self-contradiction or refusal to answer.
- **FoV is simple and effective**: Simply appending "Please focus on the visual information." improves LLaVA-NeXT-34B's accuracy from 84.6% to 89.4%.
- Input-output relevancy analysis reveals that in failure cases, the model pays significantly higher attention to text tokens than to image tokens.

## Highlights & Insights
- **The NPMI-driven counter-commonsense generation framework is highly scalable**: New conflict types and QA formats can be flexibly defined, enabling automated benchmark construction as long as domain corpora are available. The concept of this framework can be generalized to other counterfactual evaluation scenarios.
- **That CoT backfires in conflict scenarios is a significant counter-intuitive finding**: Reasoning chains amplify the reliance on parametric knowledge. This poses a valuable challenge to the prevailing belief in the "omnipotence of reasoning"—when the premise itself is "counter-commonsense", the reasoning chain instead steers the model in the wrong direction.
- **The image vs. no-image comparison design of the MR metric** is an elegant causal inference approach that accurately quantifies the actual impact of visual information on answers.

## Limitations & Future Work
- ConflictVis consists of only 374 images and 1122 QAs, which is relatively small and may suffer from limited domain coverage.
- The counter-commonsense images are generated by DALL·E 3, which may present quality issues (e.g., object distortion) and requires intensive human filtering.
- Probability estimation relies on a single model (Vicuna-13B), which may introduce model-specific biases into the benchmark.
- The root cause of the under-utilization of visual information has not been analyzed in-depth (e.g., is it an issue with the visual encoder or cross-modal fusion?).
- Although the FoV prompt is effective, it is overly simple. More sophisticated prompting strategies, such as visual chain-of-thought, remain unexplored.

## Related Work & Insights
- **vs. HallusionBench**: HallusionBench focuses on factual conflicts in infographics, whereas ConflictVis focuses on natural commonsense conflicts, covering a broader range of scenarios and question types.
- **vs. AutoHallusion**: AutoHallusion only addresses object existence and spatial relationships via Yes-No questions, whereas ConflictVis covers the three dimensions of $\langle \text{Subject, Action, Place} \rangle$ alongside three question types.
- **vs. PhD**: PhD relies on manual collection and is not scalable, whereas ConflictVis's NPMI framework supports automated large-scale construction.

## Rating
- Novelty: ⭐⭐⭐⭐ The systematic study of vision-knowledge conflicts addresses a critical gap in the MLLM field, and the automated benchmark construction framework is innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 9 models, 3 question types, 2 conflict categories, and multiple mitigation methods with granular analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivational logic, rich figures (the relevancy map analysis is particularly impressive), and vivid case studies.
- Value: ⭐⭐⭐⭐ Uncovers critical flaws in MLLMs (e.g., that CoT can be harmful), offering valuable insights for trustworthy AI research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Hidden in Plain Sight: Evaluation of the Deception Detection Capabilities of LLMs in Multimodal Settings](hidden_in_plain_sight_evaluation_of_the_deception_detection_capabilities_of_llms.md)
- [\[ACL 2025\] Exploring Compositional Generalization of Multimodal LLMs for Medical Imaging](exploring_compositional_generalization_of_multimodal_llms_for_medical_imaging.md)
- [\[ICCV 2025\] MM-Spatial: Exploring 3D Spatial Understanding in Multimodal LLMs](../../ICCV2025/multimodal_vlm/mm-spatial_exploring_3d_spatial_understanding_in_multimodal_llms.md)
- [\[ICCV 2025\] Are They the Same? Exploring Visual Correspondence Shortcomings of Multimodal LLMs](../../ICCV2025/multimodal_vlm/are_they_the_same_exploring_visual_correspondence_shortcomings_of_multimodal_llm.md)
- [\[ACL 2026\] When Seeing Overrides Knowing: Disentangling Knowledge Conflicts in Vision-Language Models](../../ACL2026/multimodal_vlm/when_seeing_overrides_knowing_disentangling_knowledge_conflicts_in_vision-langua.md)

</div>

<!-- RELATED:END -->
