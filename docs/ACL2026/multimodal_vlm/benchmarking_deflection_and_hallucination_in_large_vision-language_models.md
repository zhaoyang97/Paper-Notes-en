---
title: >-
  [Paper Note] Benchmarking Deflection and Hallucination in Large Vision-Language Models
description: >-
  [ACL 2026][Multimodal VLM][Vision-Language Models] This paper introduces VLM-DeflectionBench, a multimodal benchmark containing 2,775 samples…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Vision-Language Models"
  - "Hallucination Detection"
  - "Deflection Evaluation"
  - "Knowledge Question Answering"
  - "Retrieval-Augmented Generation"
date: 2026-05-08
content_hash: 1fb2cf40c5ca245a
---

# Benchmarking Deflection and Hallucination in Large Vision-Language Models

**Conference**: ACL 2026  
**arXiv**: [2604.12033](https://arxiv.org/abs/2604.12033)  
**Code**: Yes (public after publication)  
**Area**: Multimodal VLM  
**Keywords**: Vision-Language Models, Hallucination Detection, Deflection Evaluation, Knowledge Question Answering, Retrieval-Augmented Generation

## TL;DR
This paper introduces VLM-DeflectionBench, a multimodal benchmark containing 2,775 samples, which systematically evaluates the deflection vs. hallucination behavior of Large Vision-Language Models (LVLMs) when evidence is insufficient or misleading across four evaluation scenarios (Parameterized/Oracle/Realistic/Adversarial). Experiments across 20 SOTA LVLMs reveal that almost all models fail to reliably deflect under noisy evidence.

## Background & Motivation

**Background**: Large Vision-Language Models (LVLMs) increasingly rely on retrieval augmentation to answer knowledge-intensive multimodal questions. Existing KB-VQA benchmarks (e.g., OK-VQA, InfoSeek, E-VQA) primarily evaluate accuracy when correct evidence is retrieved.

**Limitations of Prior Work**: (1) **Neglect of evidence conflict**: Existing benchmarks do not consider contradictions between visual and textual evidence, nor do they consider what a model should do when retrieved knowledge is incomplete. (2) **Rapid obsolescence**: As LVLM training sets expand, many questions originally requiring retrieval can now be answered via parameterized knowledge, causing benchmarks to lose discriminative power. (3) **Failure to distinguish failure modes**: Current metrics only measure "correctness" without distinguishing between "incorrect answers" (hallucination) and "refusal to answer" (deflection)—where deflection is the preferred failure mode when evidence is insufficient.

**Key Challenge**: A reliable RAG system should deflect rather than hallucinate when evidence is insufficient, but no benchmark currently evaluates this behavior systematically.

**Goal**: Construct a dynamically updatable benchmark specifically to evaluate LVLM hallucination vs. deflection behaviors under different knowledge conditions.

**Key Insight**: Design four complementary scenarios to decouple parameterized memory from retrieval robustness—ranging from no evidence to perfect evidence, mixed evidence, and pure distractors.

**Core Idea**: Use a dynamic filtering pipeline to maintain benchmark difficulty (filtering out samples answerable via parameters) and a four-scenario evaluation protocol to separately assess what the model knows and how it behaves when it does not know.

## Method

### Overall Architecture
A three-stage construction pipeline: Stage I uses multiple gating models to filter out samples answerable through parameters → Stage II mines textual and visual distractors for the retained samples → Stage III quality control (ensuring solvability and distractor validity). The final benchmark contains 2,775 samples, each equipped with gold standard evidence and distractors.

### Key Designs

1.  **Dynamic Parameterized Filtering (Stage I)**:
    - **Function**: Ensures that questions in the benchmark truly require external retrieval to answer.
    - **Mechanism**: Four powerful gating models (Gemma3-27B, Qwen-2.5-VL-32B, InternVL3-38B, VL-Rethinker-72B) attempt to answer each question without external knowledge. Only samples that all models fail to answer correctly are retained. GPT-4o serves as the judge.
    - **Design Motivation**: As model capabilities increase, questions requiring retrieval may become answerable via parameters. Dynamic filtering allows the benchmark to be updated over time by using stronger gating models.

2.  **Four-Scenario Evaluation Protocol**:
    - **Function**: Decouples parameterized knowledge from retrieval robustness and explicitly evaluates deflection behavior.
    - **Mechanism**: **Parameterized Scenario** (no external knowledge): Validates filtering effectiveness; expected accuracy is near zero. **Oracle Scenario** (gold evidence only): Tests maximum capability given perfect evidence. **Realistic Scenario** (mixed gold and distractors): Simulates real-world retrieval. **Adversarial Scenario** (distractors only): Expects the model to deflect rather than hallucinate. Three metrics are reported: accuracy, deflection rate, and hallucination rate.
    - **Design Motivation**: A single scenario cannot reveal the full behavioral profile of a model. These four scenarios cover the spectrum from ideal to worst-case knowledge conditions.

3.  **Multimodal Distractor Mining (Stage II)**:
    - **Function**: Provides high-quality textual and visual distractors for each sample.
    - **Mechanism**: Textual distractors are retrieved via EVA-CLIP from Wikipedia (top-10 pages) and reranked using Contriever; visual distractors are retrieved from an image index (top-10 similar non-gold images). Each sample has at least 5 distractors.
    - **Design Motivation**: Noise is inevitable in real retrieval scenarios; high-quality distractors test the ability of the model to distinguish relevant from irrelevant evidence.

## Key Experimental Results

### Main Results (Four-scenario evaluation of 20 LVLMs, representative models selected)

| Model | Oracle Acc↑ | Oracle Hall↓ | Realistic Acc | Realistic Hall↓ | Adversarial Defl↑ | Adversarial Hall↓ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Ovis2-34B | 66.5 | 27.8 | 49.1 | 43.3 | 38.7 | 58.1 |
| GPT-5 | 73.1 | 12.6 | 59.5 | 25.5 | 61.2 | 34.7 |
| Claude-Opus-4 | 49.1 | 9.2 | 32.1 | 8.5 | **88.3** | **11.1** |
| Gemini-2.5-Pro | 59.8 | 13.9 | 51.0 | 20.5 | 76.1 | 22.2 |
| Qwen-2.5-VL-32B | 61.0 | 33.9 | 45.2 | 49.5 | 13.7 | **83.9** |
| Mistral-Small-3.1 | 42.6 | 10.3 | 23.5 | 14.9 | 83.8 | 15.6 |

### Key Findings
- **No model performs in a balanced manner across all scenarios**: Claude over-deflects (Oracle accuracy only 49.1%), Qwen is overconfident (83.9% hallucination in Adversarial scenario), and Mistral also over-deflects.
- **Hallucinations remain severe even with gold evidence**: LLaVA-OneVision maintains a 41.6% hallucination rate in the Oracle scenario, indicating that grounding, rather than retrieval, is a major bottleneck.
- **Accuracy typically drops by 10-20 percentage points in Realistic scenarios**: Distractors frequently mislead models.
- **GPT-5 shows high accuracy in the Parameterized scenario (23.7%)**: This likely reflects training set contamination.
- **Open-source models rarely deflect in Adversarial scenarios**: Most deflection rates are below 35%, with models tending to fabricate answers.
- **A fundamental trade-off exists between deflection and accuracy**: High-deflection models (like Claude) sacrifice Oracle accuracy.

## Highlights & Insights
- The **"Dynamic Filtering" concept** is crucial—benchmarks should evolve with models, otherwise they become obsolete. The VLM-DeflectionBench pipeline maintains relevance by updating gating models.
- The **Four-Scenario Evaluation Protocol** reveals behavioral patterns invisible to a single accuracy metric. For instance, while Claude might score lower on traditional benchmarks due to high deflection, it is the most suitable choice for safety-critical applications.
- **Distinguishing "Hallucination vs. Deflection"** is vital for RAG deployment—in high-risk fields like medicine or law, generating unsupported answers is far more dangerous than deflecting.

## Limitations & Future Work
- The benchmark relies on GPT-4o as a judge, which may introduce evaluation bias.
- The scale of 2,775 samples is relatively limited; some modal combinations have fewer samples.
- The "Strict RAG" assumption (all incorrect answers count as hallucinations) is simplified—in reality, errors might stem from misreading evidence rather than fabrication.
- Strategies for training models to improve deflection were not explored (evaluation only).
- Deflection behavior in multi-turn interactions was not considered.
- Distractor difficulty is not graded; different difficulty levels may trigger different behaviors.

## Related Work & Insights
- **vs. MRAG-Bench**: MRAG-Bench includes visual evidence but does not evaluate deflection and hallucination. VLM-DeflectionBench is the first to systematically evaluate both in KB-VQA.
- **vs. HaloQuest/AMBER**: These benchmarks focus only on visual hallucination without retrieval-augmentation scenarios. VLM-DeflectionBench evaluates within a RAG framework.
- **vs. SimpleQA/GaRaGe**: These are text-only hallucination evaluations that cannot capture visual-textual evidence conflicts.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First benchmark to systematically evaluate deflection in multimodal RAG; unique four-scenario design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 20 models (open and closed source); human verification $\kappa=0.91$.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear motivation, rigorous experimental design, insightful findings.
- Value: ⭐⭐⭐⭐⭐ Proposes a new paradigm for RAG reliability assessment with direct guidance for deployment decisions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)
- [\[ACL 2026\] Topology-Aware Layer Pruning for Large Vision-Language Models](topology-aware_layer_pruning_for_large_vision-language_models.md)
- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)
- [\[ACL 2026\] Efficient Inference for Large Vision-Language Models: Bottlenecks, Techniques, and Prospects](efficient_inference_for_large_vision-language_models_bottlenecks_techniques_and_.md)
- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)

</div>

<!-- RELATED:END -->
