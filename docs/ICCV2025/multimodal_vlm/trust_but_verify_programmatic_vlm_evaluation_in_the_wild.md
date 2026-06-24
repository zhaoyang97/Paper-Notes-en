---
title: >-
  [Paper Note] Trust but Verify: Programmatic VLM Evaluation in the Wild
description: >-
  [ICCV 2025][Multimodal VLM][VLM Evaluation] This paper proposes the PROVE (Programmatic VLM Evaluation) evaluation paradigm. By constructing high-fidelity scene graphs from ultra-detailed image descriptions and utilizing LLMs to generate programmatically verifiable open-ended visual question-answering pairs, it simultaneously evaluates both the **helpfulness** and **truthfulness** of VLM responses within a unified scene graph framework, revealing that current models struggle…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "VLM Evaluation"
  - "Hallucination Detection"
  - "Scene Graph"
  - "Programmatic Verification"
  - "Open-ended QA"
  - "Helpfulness-Truthfulness Trade-off"
date: 2026-05-08
content_hash: ced8555d4f591190
---

# Trust but Verify: Programmatic VLM Evaluation in the Wild

**Conference**: ICCV 2025  
**arXiv**: [2410.13121](https://arxiv.org/abs/2410.13121)  
**Code**: [Project Page](https://prove-explorer.netlify.app/)  
**Area**: Multimodal VLMs  
**Keywords**: VLM Evaluation, Hallucination Detection, Scene Graph, Programmatic Verification, Open-ended QA, Helpfulness-Truthfulness Trade-off

## TL;DR

This paper proposes the PROVE (Programmatic VLM Evaluation) evaluation paradigm. By constructing high-fidelity scene graphs from ultra-detailed image descriptions and utilizing LLMs to generate programmatically verifiable open-ended visual question-answering pairs, it simultaneously evaluates both the **helpfulness** and **truthfulness** of VLM responses within a unified scene graph framework, revealing that current models struggle to achieve a good balance between the two.

## Background & Motivation

### Core Problem

VLMs frequently generate plausible-sounding but actually incorrect responses (hallucinations). However, existing evaluation methods have obvious limitations:

**Discriminative benchmarks** (e.g., POPE): They only test binary existence questions ("Is there a person in the image?"), failing to simulate real-world usage scenarios.

**Generative benchmarks** (e.g., CHAIR, MMHal): They evaluate open-ended responses but rely on external LLM scoring, and the context provided during judgment is often insufficient to verify all claims.

**Limitations of LLM-as-a-judge**: The lack of clear scoring criteria and high sensitivity to prompt variations lead to inconsistent and arbitrary grading.

### Specific Example

When a VLM answers "there are four Labradoodles in the image," both claims `<quantity==4>` and `<breed==Labradoodle>` need to be verified. However, if the LLM judge only receives a brief description ("four puppies on a light blue carpet"), it cannot complete the full verification.

### Motivation

An evaluation method is needed that can test open-ended question answering (close to real-world usage) while reliably and interpretably evaluating response quality. The key lies in: high recall of scene descriptions + programmatic verification + unified evaluation framework.

## Method

### Overall Architecture

PROVE consists of two main components: **Dataset Construction** and **Programmatic Evaluation**.

#### 1. Dataset Construction Pipeline

```text
Ultra-detailed image descriptions (DOCCI) -> Scene graph construction -> LLM-generated QA pairs + verification programs -> Filtering -> 10.5K high-quality QA pairs
```

**Step 1: Scene Graph Representation Construction**
- It uses 5K image-description pairs from the DOCCI test set (average description length of 136 words, far exceeding competing datasets).
- Entity-attribute-relation triplets are extracted from the descriptions to construct a directed graph $g(\mathcal{C})$.
- The scene graph is implemented as a Python class, offering APIs to query entities, attributes, relations, and to extract subgraphs.

**Step 2: Verifiable QA Pair Generation**
- GPT-4o is used to generate 10-15 diverse and challenging open-ended QA pairs for each image.
- A corresponding Python verification program is simultaneously generated, which can be executed on the scene graph objects to verify the correctness of the QA pairs.

**Step 3: Dual Filtering**
- **Programmatic Filtering**: The verification program is executed, discarding QA pairs that result in program execution failure (18.3% ) or return incorrect answers (9.8%).
- **Text Filtering**: Low-quality QA pairs are excluded—including those that are trivial/ambiguous/incomplete (determined by LLMs), not entailed by the image (using a visual entailment model), contain sensitive words, or are semantically redundant (using SemDeDup).

Ultimately, approximately 50% of the QA pairs are retained, yielding a total of **10.5K** high-quality samples.

#### 2. Programmatic Evaluation Method

Given the model's answer $\hat{\mathcal{A}} = m_\theta(\mathcal{Q}, \mathcal{I})$, evaluation is conducted along two dimensions:

**Helpfulness score (hscore)**—The recall of the model's answer relative to the ground truth (GT) answer's scene graph:

$$\text{hscore}(\hat{\mathcal{A}}) = \frac{\sum_{t \in g(\mathcal{A}) - g(\mathcal{Q})} \max_{t' \in g(\hat{\mathcal{A}})} \text{sim}(t, t')}{|g(\mathcal{A)} - g(\mathcal{Q})|}$$

- Scene graph tuples are extracted from both the GT answer and the model's answer.
- Premise tuples already present in the question are excluded.
- The average of the maximum cosine similarities from the GT tuples to the answer tuples is computed.

**Truthfulness score (tscore)**—The precision of the answer tuples relative to the complete scene:

$$\text{tscore}(\hat{\mathcal{A}}) = \frac{\sum_{t' \in g(\hat{\mathcal{A}})} \max\left(\max_{t \in g(\mathcal{C})} \text{sim}(t', t),\ p(\mathcal{I} \models t')\right)}{|g(\hat{\mathcal{A}})|}$$

- It not only matches against the scene graph of the full description but also utilizes a visual entailment model to check the image itself.
- This reduces false positive hallucination detections caused by incomplete descriptions.

### Key Designs

1. **Dual Verification Sources**: The tscore simultaneously leverages the textual scene graph and visual entailment, as even highly detailed descriptions cannot cover all aspects of an image.
2. **Decoupling of hscore and tscore**: The two are not necessarily positively correlated—a response can be helpful but not entirely truthful (containing hallucinations), or truthful but not helpful enough.
3. **Interpretability**: The approach is based on concrete scoring criteria from scene graph matching rather than opaque LLM scoring.

## Key Experimental Results

### Main Results

VLM helpfulness-truthfulness trade-off on PROVE

| Model | Parameters | hscore ↑ | tscore ↑ | Average ↑ |
|------|--------|----------|----------|--------|
| Qwen2-VL | 2B | 69.36 | 80.64 | 75.00 |
| InternVL2 | 2B | 73.96 | 79.51 | 76.74 |
| Phi-3.5-Vision | 4B | 73.35 | 82.27 | 77.81 |
| LLaVA-1.5 | 7B | 72.67 | **82.58** | 77.62 |
| LLaVA-Next | 7B | 74.28 | 80.03 | 77.15 |
| InternVL2 | 8B | 74.55 | 80.56 | 77.56 |
| Pixtral | 12B | 73.34 | 82.43 | 77.88 |
| LLaVA-1.5 | 13B | 72.46 | 82.40 | 77.43 |
| InternVL2 | 26B | 74.63 | 79.23 | 76.93 |
| Claude-3.5-Sonnet† | - | 71.06 | 77.31 | 74.19 |
| GPT-4o-mini† | - | 73.18 | 79.24 | 76.21 |
| Gemini-1.5-Flash† | - | 72.73 | 81.74 | 77.23 |
| **GPT-4o†** | - | **76.53** | 80.92 | **78.72** |
| Oracle* | - | 82.84 | 85.59 | 84.22 |

### Ablation Study

| Analysis Dimension | Findings |
|---------|------|
| hscore vs tscore Correlation | The average linear correlation across models is only **0.03**, indicating almost no correlation. |
| Model Size vs. Truthfulness | For InternVL2 (2B -> 8B -> 26B), hscore increases but tscore does not necessarily improve. |
| LLaVA Series Comparison | The LLaVA-1.5 series achieves the overall best tscore but has a lower hscore. |
| Human Evaluation - QA Quality | 95.9% of the questions are judged as relevant, and 98.2% of the answers are judged as correct. |
| Human Evaluation - Metric Correlation | hscore correlates with human judgment at 0.81, and tscore correlates at 0.45. |
| Oracle vs. Best Model | The Oracle scores 84.22 on average vs. GPT-4o at 78.72, showing significant room for improvement. |

### Key Findings

1. **Very few models achieve a good balance between the two**: Only GPT-4o, Phi-3.5-Vision, and Pixtral exhibit balanced performance.
2. **Highly-ranked models do not necessarily yield high truthfulness**: Claude-3.5-Sonnet and InternVL2-26B rank high on aggregated leaderboards, but their tscores lag behind the simpler LLaVA-1.5.
3. **Models fail in different ways**: GPT-4o's errors are "milder" (e.g., reading 3 out of 6 letters correctly), whereas LLaVA's errors are more severe (e.g., reading only 1 correctly); GPT-4o generates more descriptive answers, improving its hscore.
4. **Commonly hallucinated objects**: Common objects such as tree, building, wall, sign, etc.

## Highlights & Insights

1. **Paradigm Innovation**: Programmatic verification is introduced to open-ended VLM evaluation for the first time, achieving high reliability through a closed-loop pipeline of "QA generation -> programmatic verification -> scene graph evaluation."
2. **Revealing Critical Trade-offs**: The weak correlation (0.03) between helpfulness and truthfulness demonstrates that improvements in recent models primarily enhance helpfulness rather than truthfulness.
3. **Scalability of Evaluation**: The entire benchmark construction process is fully automated, allowing easy scaling to larger image-description sources.
4. **Quantified Counter-intuitive Findings**: Larger and newer models are not necessarily more truthful, challenging the naive assumption that "scaling = better."

## Limitations & Future Work

1. **Recall Cost**: To ensure high precision, approximately 50% of the QA pairs are filtered out, which may exclude certain hard-to-verify question types.
2. **Description Coverage**: Even high-recall descriptions cannot capture all aspects of an image, potentially missing some hallucinations.
3. **Dependency on External Models**: Text embeddings (Sentence-BERT), scene-graph extraction, and visual entailment (OFA) tools may introduce their respective errors.
4. **Untested Mitigation Methods**: Hallucination mitigation strategies such as fine-tuning, preference optimization, and training-free decoding have not been evaluated on PROVE.

## Related Work & Insights

- **Difference from CHAIR**: CHAIR only evaluates the precision/recall of objects in descriptions and is limited to image captioning templates; PROVE supports arbitrary open-ended questions.
- **Difference from MMHal-Bench**: MMHal relies on a pipeline of off-the-shelf models that introduce noise, and its GPT-4 scoring can penalize correct answers due to a lack of context.
- **Difference from GAVIE**: GAVIE relies on dense descriptions and bounding boxes, where questions focus heavily on local regions and spatial relations, leading to unnatural response qualities.
- **Inspirations for Future Work**: Integrating agentic VLMs (capable of planning, reasoning, and self-reflection) could be a direction for simultaneously improving both hscore and tscore.

## Rating ⭐⭐⭐⭐

**Novelty**: ⭐⭐⭐⭐⭐ — The paradigm of programmatic verification + scene graph evaluation is novel and highly influential.  
**Utility**: ⭐⭐⭐⭐ — Provides a scalable automated evaluation pipeline.  
**Experimental Thoroughness**: ⭐⭐⭐⭐ — Covers 14 models, including human evaluation verification.  
**Writing Quality**: ⭐⭐⭐⭐ — Clear motivation, adequate examples, and rigorous presentation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Reading Recognition in the Wild](../../NeurIPS2025/multimodal_vlm/reading_recognition_in_the_wild.md)
- [\[ICCV 2025\] SimpleVQA: Multimodal Factuality Evaluation for Multimodal Large Language Models](simplevqa_multimodal_factuality_evaluation_for_multimodal_large_language_models.md)
- [\[ICCV 2025\] Fine-Grained Evaluation of Large Vision-Language Models in Autonomous Driving](fine-grained_evaluation_of_large_vision-language_models_in_autonomous_driving.md)
- [\[CVPR 2025\] VisionZip: Longer is Better but Not Necessary in Vision Language Models](../../CVPR2025/multimodal_vlm/visionzip_longer_is_better_but_not_necessary_in_vision_language_models.md)
- [\[ICCV 2025\] GEOBench-VLM: Benchmarking Vision-Language Models for Geospatial Tasks](geobench-vlm_benchmarking_vision-language_models_for_geospatial_tasks.md)

</div>

<!-- RELATED:END -->
