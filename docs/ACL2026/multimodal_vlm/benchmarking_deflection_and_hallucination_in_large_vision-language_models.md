---
title: >-
  [Paper Note] Benchmarking Deflection and Hallucination in Large Vision-Language Models
description: >-
  [ACL 2026][Multimodal VLM][Vision-language models] This paper proposes VLM-DeflectionBench, a multimodal benchmark comprising 2,775 samples that systematically evaluates the deflection vs. hallucination behavior of large vision-language models (LVLMs) under insufficient or misleading evidence, through four evaluation scenarios (Parametric / Oracle / Realistic / Adversarial). Experiments covering 20 state-of-the-art LVLMs reveal that virtually no model can reliably deflect under noisy evidence.
tags:
  - ACL 2026
  - Multimodal VLM
  - Vision-language models
  - hallucination detection
  - deflection evaluation
  - knowledge-based VQA
  - retrieval-augmented generation
date: 2026-05-08
content_hash: b28dc90cea5542a9
---

# Benchmarking Deflection and Hallucination in Large Vision-Language Models

**Conference**: ACL 2026
**arXiv**: [2604.12033](https://arxiv.org/abs/2604.12033)
**Code**: Available (to be released upon publication)
**Area**: Multimodal VLM
**Keywords**: Vision-language models, hallucination detection, deflection evaluation, knowledge-based VQA, retrieval-augmented generation

## TL;DR
This paper proposes VLM-DeflectionBench, a multimodal benchmark comprising 2,775 samples that systematically evaluates the deflection vs. hallucination behavior of large vision-language models (LVLMs) under insufficient or misleading evidence, through four evaluation scenarios (Parametric / Oracle / Realistic / Adversarial). Experiments covering 20 state-of-the-art LVLMs reveal that virtually no model can reliably deflect under noisy evidence.

## Background & Motivation

**Background**: Large vision-language models increasingly rely on retrieval augmentation to answer knowledge-intensive multimodal questions. Existing KB-VQA benchmarks (e.g., OK-VQA, InfoSeek, E-VQA) primarily evaluate accuracy when correct evidence is retrieved.

**Limitations of Prior Work**: (1) **Evidence conflicts are ignored**: Existing benchmarks do not account for contradictions between visual and textual evidence, nor do they address what a model should do when retrieved knowledge is incomplete. (2) **Rapid obsolescence**: As LVLM training sets expand, many questions that previously required retrieval can now be answered directly via parametric knowledge, rendering benchmarks less discriminative. (3) **Failure modes are conflated**: These benchmarks only measure correctness without distinguishing between erroneous answers (hallucination) and refusals to answer (deflection)—yet deflection is a more desirable failure mode when evidence is insufficient.

**Key Challenge**: Reliable RAG systems should deflect rather than fabricate when evidence is insufficient, yet no benchmark systematically evaluates this behavior.

**Goal**: Construct a dynamically updatable benchmark specifically designed to evaluate LVLM hallucination vs. deflection behavior under varying knowledge conditions.

**Key Insight**: Four complementary scenarios are designed to disentangle parametric memory from retrieval robustness—ranging from no evidence to perfect evidence to mixed evidence to pure distractors.

**Core Idea**: A dynamic filtering pipeline maintains benchmark difficulty by excluding parametrically answerable samples, while a four-scenario evaluation protocol separately assesses what models know and how they behave when they do not know.

## Method

### Overall Architecture
A three-stage construction pipeline: Stage I applies multiple gating models to filter out parametrically answerable samples → Stage II mines textual and visual distractors for the retained samples → Stage III performs quality control (ensuring answerability and distractor validity). The final benchmark contains 2,775 samples, each accompanied by gold-standard evidence and distractors.

### Key Designs

1. **Dynamic Parametric Filtering (Stage I)**:

    - **Function**: Ensures that questions in the benchmark genuinely require external retrieval.
    - **Mechanism**: Four powerful gating models (Gemma3-27B, Qwen-2.5-VL-32B, InternVL3-38B, VL-Rethinker-72B) attempt to answer each question without external knowledge. Only samples that none of the models answer correctly are retained. GPT-4o serves as the judge.
    - **Design Motivation**: As model capabilities improve, questions previously requiring retrieval may become parametrically answerable. Dynamic filtering allows the benchmark to be updated over time by substituting stronger gating models, preserving evaluation validity.

2. **Four-Scenario Evaluation Protocol**:

    - **Function**: Disentangles parametric knowledge from retrieval robustness and explicitly evaluates deflection behavior.
    - **Mechanism**: **Parametric scenario** (no external knowledge): validates filtering effectiveness; near-zero accuracy is expected. **Oracle scenario** (gold-standard evidence only): measures maximum capability given perfect evidence. **Realistic scenario** (gold-standard evidence mixed with distractors): simulates real-world retrieval results. **Adversarial scenario** (distractors only): models are expected to deflect rather than hallucinate. Each scenario reports three metrics: accuracy, deflection rate, and hallucination rate.
    - **Design Motivation**: A single scenario cannot reveal a complete picture of model behavior. The four scenarios span the spectrum from ideal to worst-case knowledge conditions.

3. **Multimodal Distractor Mining (Stage II)**:

    - **Function**: Provides high-quality textual and visual distractors for each sample.
    - **Mechanism**: Textual distractors are retrieved from a Wikipedia index using EVA-CLIP to obtain the top-10 relevant pages, which are then chunked and reranked with Contriever. Visual distractors are retrieved from an image index as the top-10 similar non-gold images. Each sample is guaranteed at least five distractors.
    - **Design Motivation**: Noise is inevitable in real-world retrieval settings; high-quality distractors test a model's ability to distinguish relevant from irrelevant evidence.

## Key Experimental Results

### Main Results (Four-scenario evaluation of 20 LVLMs; representative models shown)

| Model | Oracle Acc↑ | Oracle Hall↓ | Realistic Acc | Realistic Hall↓ | Adversarial Defl↑ | Adversarial Hall↓ |
|---|---|---|---|---|---|---|
| Ovis2-34B | 66.5 | 27.8 | 49.1 | 43.3 | 38.7 | 58.1 |
| GPT-5 | 73.1 | 12.6 | 59.5 | 25.5 | 61.2 | 34.7 |
| Claude-Opus-4 | 49.1 | 9.2 | 32.1 | 8.5 | **88.3** | **11.1** |
| Gemini-2.5-Pro | 59.8 | 13.9 | 51.0 | 20.5 | 76.1 | 22.2 |
| Qwen-2.5-VL-32B | 61.0 | 33.9 | 45.2 | 49.5 | 13.7 | **83.9** |
| Mistral-Small-3.1 | 42.6 | 10.3 | 23.5 | 14.9 | 83.8 | 15.6 |

### Key Findings
- **No model performs consistently across all scenarios**: Claude over-deflects (Oracle accuracy only 49.1%), Qwen is overconfident (adversarial hallucination rate 83.9%), and Mistral also over-deflects.
- **Hallucination remains severe even with gold-standard evidence**: LLaVA-OneVision still exhibits a 41.6% hallucination rate in the Oracle scenario, indicating that grounding rather than retrieval is the primary bottleneck.
- **Accuracy drops by 10–20 percentage points in the Realistic scenario**: Distractors frequently mislead models.
- **GPT-5 exhibits elevated accuracy in the Parametric scenario (23.7%)**: This may reflect training data contamination.
- **Open-source models rarely deflect in the Adversarial scenario**: Most deflection rates fall below 35%, with models tending to fabricate answers.
- **A fundamental trade-off exists between deflection and accuracy**: Models with high deflection rates (e.g., Claude) sacrifice Oracle accuracy.

## Highlights & Insights
- The **dynamic filtering philosophy** is critically important—benchmarks should evolve alongside models; otherwise they quickly become obsolete. VLM-DeflectionBench's pipeline can maintain timeliness by replacing gating models as stronger ones become available.
- The **four-scenario evaluation protocol** reveals behavioral patterns invisible to a single accuracy metric. For instance, Claude may score low on traditional benchmarks due to its high deflection rate, yet it may be the most suitable choice in safety-critical deployment contexts.
- The **distinction between hallucination and deflection** is essential for RAG system deployment—in high-stakes domains such as medicine and law, generating unsupported answers is far more dangerous than deflecting.

## Limitations & Future Work
- The benchmark relies solely on GPT-4o as the judge, which may introduce evaluation bias.
- The scale of 2,775 samples is relatively modest, with limited coverage of certain modality combinations.
- The "strict RAG" assumption—treating all incorrect answers as hallucinations—is an oversimplification; in practice, errors may stem from misreading evidence rather than confabulation.
- Training models to improve deflection capability is not explored; the work is purely evaluative.
- Deflection behavior in multi-turn interactions is not considered.
- Distractor difficulty is not stratified; distractors of varying difficulty may elicit qualitatively different behaviors.

## Related Work & Insights
- **vs. MRAG-Bench**: MRAG-Bench incorporates visual evidence but does not evaluate deflection or hallucination. VLM-DeflectionBench is the first to systematically assess both behaviors in the KB-VQA setting.
- **vs. HaloQuest/AMBER**: These benchmarks focus exclusively on visual hallucination and do not involve retrieval-augmented scenarios. VLM-DeflectionBench evaluates within a RAG framework.
- **vs. SimpleQA/GaRaGe**: These are text-only hallucination evaluations that cannot capture visual-textual evidence conflicts.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First benchmark to systematically evaluate deflection behavior in multimodal RAG; the four-scenario design is highly original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Covers 20 models (open- and closed-source) with human validation at κ=0.91.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Motivation is clear, experimental design is rigorous, and findings are insightful.
- **Value**: ⭐⭐⭐⭐⭐ Introduces a new paradigm for evaluating the reliability of RAG systems with direct implications for deployment decisions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] OMIBench: Benchmarking Olympiad-Level Multi-Image Reasoning in Large Vision-Language Models](omibench_benchmarking_olympiad-level_multi-image_reasoning_in_large_vision-langu.md)
- [\[ACL 2026\] Topology-Aware Layer Pruning for Large Vision-Language Models](topology-aware_layer_pruning_for_large_vision-language_models.md)
- [\[ACL 2026\] ErrorRadar: Benchmarking Complex Mathematical Reasoning of Multimodal Large Language Models Via Error Detection](errorradar_benchmarking_complex_mathematical_reasoning_of_multimodal_large_langu.md)
- [\[ACL 2026\] Efficient Inference for Large Vision-Language Models: Bottlenecks, Techniques, and Prospects](efficient_inference_for_large_vision-language_models_bottlenecks_techniques_and_.md)
- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)

</div>

<!-- RELATED:END -->
