---
title: >-
  [Paper Note] TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection
description: >-
  [CVPR 2026][LLM Safety][DeepFake Detection] This paper proposes TriDF — the first benchmark that comprehensively evaluates interpretable DeepFake detection across three dimensions: Perception, Detection…
tags:
  - "CVPR 2026"
  - "LLM Safety"
  - "DeepFake Detection"
  - "Interpretable Detection"
  - "Multimodal Large Language Models"
  - "Hallucination Evaluation"
  - "Artifact Taxonomy"
date: 2026-05-08
content_hash: 164693f666a6eb34
---

# TriDF: Evaluating Perception, Detection, and Hallucination for Interpretable DeepFake Detection

**Conference**: CVPR 2026
**arXiv**: [2512.10652](https://arxiv.org/abs/2512.10652)  
**Code**: [https://j1anglin.github.io/TriDF/](https://j1anglin.github.io/TriDF/)  
**Area**: Diffusion Models / Security Detection
**Keywords**: DeepFake Detection, Interpretable Detection, Multimodal Large Language Models, Hallucination Evaluation, Artifact Taxonomy

## TL;DR
This paper proposes TriDF — the first benchmark that comprehensively evaluates interpretable DeepFake detection across three dimensions: Perception, Detection, and Hallucination. It comprises 55K high-quality samples covering 16 DeepFake types and 3 modalities, and reveals a triadic coupling relationship in which accurate perception is a prerequisite for reliable detection, yet hallucination can severely undermine decision-making.

## Background & Motivation

1. **Background**: With the rapid advancement of generative models, DeepFake detection has evolved from a simple binary classification task toward requiring interpretability — not only determining whether content is fake, but also explaining why. Multimodal large language models (MLLMs) are increasingly employed for interpretable DeepFake detection.

2. **Limitations of Prior Work**:
    - **Coarse-grained annotations in existing datasets**: Datasets such as FF++ and DFDC provide only binary labels, making it impossible to evaluate interpretability.
    - **Incomplete coverage in existing benchmarks**: DD-VQA covers only 4 forgery types, FakeBench only 1, and LOKI only 3; most support only the image modality, lacking cross-modal coverage.
    - **Absence of hallucination evaluation**: MLLMs may produce "hallucinations" when generating explanations — providing rationales for artifacts that do not exist. This is particularly dangerous in DeepFake detection, as fabricated explanations can mislead judgments. No existing benchmark addresses this aspect.
    - **Using MLLMs to judge MLLMs**: Many benchmarks rely on GPT-4o to evaluate the outputs of other models, introducing self-preference bias.

3. **Key Challenge**: Interpretable DeepFake detection requires models to simultaneously possess three capabilities — perceiving artifacts, correctly detecting fakes, and generating reliable explanations — yet no unified framework exists to evaluate these three capabilities and their interdependencies.

4. **Goal**: To construct a comprehensive benchmark for interpretable DeepFake detection that jointly evaluates perception, detection, and hallucination, and to reveal the coupling relationships among them.

5. **Key Insight**: Starting from a human-annotated fine-grained artifact taxonomy, the paper establishes quantifiable perception evaluation; pairs real and fake samples to support hallucination detection; and covers three modalities (image/video/audio) and 16 DeepFake types.

6. **Core Idea**: Perception, Detection, and Hallucination form an inseparable triad for interpretable DeepFake detection. TriDF is the first unified benchmark to evaluate all three simultaneously.

## Method

### Overall Architecture
The construction of TriDF proceeds in two phases: (1) **Data generation and annotation** — collecting face-related data from public datasets, generating real-fake pairs using 16 DeepFake techniques, performing quality control, and conducting human annotation of fine-grained artifacts; (2) **Evaluation** — designing three question types (True-False Questions, Multiple-Choice Questions, and Open-Ended Questions), feeding them into MLLMs, and assessing perception, detection, and hallucination using the proposed metric system.

### Key Designs

1. **Fine-Grained Artifact Taxonomy**:

    - **Function**: Establishes a standardized classification framework for DeepFake artifacts, providing reliable human-annotated ground truth for perception evaluation.
    - **Mechanism**: Artifacts are divided into two major categories — **quality artifacts** (blur, noise, flickering, etc., detectable via traditional image processing) and **semantic artifacts** (anatomical inconsistencies, object integrity defects, unnatural rhythm, etc., requiring commonsense reasoning). Quality artifacts are further localized to specific regions (e.g., nose, limbs, background), enabling systematic evaluation of MLLMs' localization ability. This taxonomy is human-annotated, avoiding the bias inherent in MLLM self-evaluation.
    - **Design Motivation**: Prior benchmarks lack a standardized artifact annotation framework, and relying on MLLM-generated explanations as ground truth is unreliable. Human-annotated fine-grained artifacts provide an objective baseline for perception evaluation.

2. **Three-Dimensional Evaluation Framework (Perception / Detection / Hallucination)**:

    - **Function**: Comprehensively evaluates MLLMs' interpretable DeepFake detection capability from three complementary perspectives.
    - **Mechanism**:
        - **Perception**: Uses only forged samples; tests models' artifact recognition ability via TFQ/MCQ/OEQ-A, covering artifact identification and localization. MCQ includes a "none of the above" option and allows multiple selections to increase difficulty.
        - **Detection**: Uses both real and fake samples; prompts models via OEQ-B to first provide a real/fake judgment and then list detected artifacts; evaluated using Accuracy and Cover.
        - **Hallucination**: Identifies fabricated artifacts — those that do not exist — from OEQ-A and OEQ-B responses; evaluated using CHAIR, Hal, and F0.5. When the mapped artifact list has length zero, or when the model classifies a fake sample as real, CHAIR is set to 1 as a penalty.
    - **Design Motivation**: The three dimensions are inseparable — accurate perception is the foundation of detection, but even with correct perception, hallucination can undermine the final decision. Evaluating only one or two dimensions cannot provide a comprehensive understanding of model capability.

3. **Data Generation and Quality Control**:

    - **Function**: Generates 55K high-quality real-fake paired samples covering 16 DeepFake types and 3 modalities.
    - **Mechanism**: Real face data is collected from 30+ public datasets; forged samples are generated using 50+ specialized generative models (GANs, SD, DiT, commercial APIs, etc.). The 16 DeepFake types are categorized into partial manipulation (face swapping, attribute editing, lip synchronization, face reenactment, full-body manipulation, subject-driven editing, voice conversion) and full synthesis (audio-driven talking head, identity-preserving generation, text-to-human image/video, etc.). Each type is generated by at least 3 different models to ensure generator diversity. Automatic quality filtering is applied using authenticity and consistency metrics.
    - **Design Motivation**: One-to-one real-fake pairing enables more precise artifact annotation (by comparing against the real sample) and supports hallucination evaluation (checking whether models fabricate artifacts for real samples).

### Evaluation Metrics
- **Perception / Detection**: Accuracy (TFQ); weighted scoring for MCQ (correct: $+1/K$, incorrect: $-1/(M-K)$); Cover (coverage rate of correctly identified artifacts in OEQ).
- **Hallucination**: CHAIR (proportion of fabricated artifacts), Hal (proportion of responses containing hallucinations), F0.5 (precision-weighted composite score).
- An external lightweight LLM (Gemini 2.5 Flash-Lite) is used for artifact mapping, avoiding the "using MLLMs to judge MLLMs" bias.

## Key Experimental Results

### Main Results — Perception Evaluation (TFQ)

| MLLM | Image TFQ Avg | Video TFQ Avg | Total Avg | Rank |
|------|---------------|---------------|-----------|------|
| GPT-5 | 63.36% | 57.02% | 60.19% | 1 |
| Gemini 2.5-Pro | 61.69% | 57.58% | 59.63% | 3 |
| Qwen3-VL-30B | 61.04% | 58.65% | 59.85% | 2 |
| Claude Sonnet 4.5 | 53.57% | 51.05% | 52.31% | 14 |
| InternVL3_5-8B | 53.69% | 54.03% | 53.86% | 7 |

### Main Results — Detection + Hallucination Evaluation (Type-B OEQ)

| MLLM | Image Acc | Image Cover↑ | Image CHAIR↓ | Image F0.5↑ |
|------|-----------|--------------|--------------|-------------|
| Qwen3-Omni-30B | 0.6942 | 0.4143 | 0.6701 | 0.3381 |
| Qwen3-VL-30B | 0.6894 | 0.3661 | 0.7137 | 0.2388 |
| InternVL2_5-38B | 0.5747 | 0.2306 | 0.8066 | 0.1971 |
| GPT-5 | — | — | — | — |

### Key Findings
- **Perception is the foundation of detection**: Models with stronger perception ability (higher TFQ/MCQ rankings) tend to achieve better detection performance, though this is not a sufficient condition.
- **Hallucination severely undermines decision-making**: Even models with strong perception capabilities exhibit unstable detection performance when hallucination rates are high. Most MLLMs achieve CHAIR > 0.5, meaning more than half of all responses contain fabricated artifacts.
- **Open-source vs. closed-source gap**: GPT-5 ranks first in perception, while Qwen3-VL-30B performs best among open-source models. Claude Sonnet 4.5, despite a low perception ranking (14th), achieves the highest MCQ score (0.21), suggesting higher reasoning precision.
- **Video is harder than image**: Nearly all models perform worse on the video modality, indicating that temporal artifact recognition remains a challenge.
- **Hallucination is pervasive and severe**: For most models, Type-A OEQ Hal > 0.9, meaning over 90% of responses contain at least one hallucinated artifact — posing a serious threat to the trustworthiness of interpretable detection.

## Highlights & Insights
- **Completeness of the triadic framework**: Prior benchmarks focus only on detection accuracy or explanation quality in isolation. TriDF is the first to unify perception, detection, and hallucination into a single framework, revealing the inseparable relationship among the three. This framework design is transferable to other domains requiring explainable AI.
- **Human-annotated artifact taxonomy**: This approach avoids the circular bias of MLLM self-evaluation and establishes an objective baseline for perception assessment. The two-level design separating quality artifacts from semantic artifacts is well-structured.
- **Timely introduction of hallucination evaluation**: The hallucination problem of MLLMs in DeepFake detection has been entirely overlooked in prior work. The finding of Hal > 0.9 demonstrates that current interpretable detection is far from reliable.

## Limitations & Future Work
- Artifact annotation relies on human labor, incurring high costs and potential inter-annotator inconsistency.
- Although the 55K sample scale is substantial, distribution across the 16 DeepFake types may be uneven, with insufficient samples for rare types.
- The evaluation framework is primarily designed for static detection scenarios and does not account for interactive detection settings (e.g., follow-up queries for details).
- The Cover metric only measures coverage rate, not the precision or detail of the descriptions provided.
- No remediation strategies or methods for mitigating hallucination are proposed.

## Related Work & Insights
- **vs. FakeBench**: FakeBench covers only 1 DeepFake type and provides no hallucination evaluation; TriDF covers 16 types with a complete hallucination evaluation framework.
- **vs. LOKI**: LOKI supports multiple modalities but covers only 3 DeepFake types and lacks a human-annotated artifact taxonomy.
- **vs. Forensics-Bench**: Covers 10 DeepFake types with 63K samples but provides no perception or hallucination evaluation.
- **vs. DD-VQA**: Pioneered the VQA-style detection evaluation but covers only 4 types; TriDF offers substantially broader coverage.

## Rating
- Novelty: ⭐⭐⭐⭐ The three-dimensional evaluation framework and hallucination evaluation are genuine contributions, though the core deliverable is a benchmark rather than a method.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 16 DeepFake types, 51 generators, and evaluates 18 MLLMs.
- Writing Quality: ⭐⭐⭐⭐ Structure is clear, but the excessive number of tables dilutes the presentation; core insights could be made more prominent.
- Value: ⭐⭐⭐⭐⭐ Fills an important gap in the evaluation of interpretable DeepFake detection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Veritas: Generalizable Deepfake Detection via Pattern-Aware Reasoning](../../ICLR2026/llm_safety/veritas_generalizable_deepfake_detection_via_pattern-aware_reasoning.md)
- [\[CVPR 2026\] Pixels Don't Lie (But Your Detector Might): Bootstrapping MLLM-as-a-Judge for Trustworthy Deepfake Detection and Reasoning Supervision](pixels_dont_lie_but_your_detector_might_bootstrapping_mllm-as-a-judge_for_trustw.md)
- [\[ICML 2026\] PRPO: Paragraph-level Policy Optimization for Vision-Language Deepfake Detection](../../ICML2026/llm_safety/prpo_paragraph-level_policy_optimization_for_vision-language_deepfake_detection.md)
- [\[ACL 2026\] Enhancing Hallucination Detection via Future Context](../../ACL2026/llm_safety/enhancing_hallucination_detection_via_future_context.md)
- [\[ICLR 2026\] VeriTrail: Closed-Domain Hallucination Detection with Traceability](../../ICLR2026/llm_safety/veritrail_closed-domain_hallucination_detection_with_traceable_evidence_synthes.md)

</div>

<!-- RELATED:END -->
