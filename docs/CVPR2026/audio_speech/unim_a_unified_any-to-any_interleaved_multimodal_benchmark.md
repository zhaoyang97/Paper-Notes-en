---
title: >-
  [Paper Note] UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark
description: >-
  [CVPR 2026][Audio & Speech][multimodal benchmark] This paper proposes UniM, the first unified any-to-any interleaved multimodal benchmark (31K samples, 7 modalities, 30 domains), accompanied by a three-dimensional evaluation suite and an agentic baseline UniMA based on traceable evidence reasoning, revealing critical deficiencies of existing MLLMs under the interleaved multimodal paradigm.
tags:
  - CVPR 2026
  - "Audio & Speech"
  - multimodal benchmark
  - any-to-any
  - interleaved multimodal
  - evaluation suite
  - agentic model
date: 2026-05-08
content_hash: 040181549655782b
---

# UniM: A Unified Any-to-Any Interleaved Multimodal Benchmark

**Conference**: CVPR 2026
**arXiv**: [2603.05075](https://arxiv.org/abs/2603.05075)
**Code**: Available ([Project Page](https://any2any-mllm.github.io/unim))
**Area**: Audio/Speech (Multimodal Benchmark)
**Keywords**: multimodal benchmark, any-to-any, interleaved multimodal, evaluation suite, agentic model

## TL;DR

This paper proposes UniM, the first unified any-to-any interleaved multimodal benchmark (31K samples, 7 modalities, 30 domains), accompanied by a three-dimensional evaluation suite and an agentic baseline UniMA based on traceable evidence reasoning, revealing critical deficiencies of existing MLLMs under the interleaved multimodal paradigm.

## Background & Motivation

### 1. State of the Field
Multimodal large language models (MLLMs) have rapidly evolved from early vision-language understanding to unified frameworks that simultaneously support understanding and generation (e.g., NExT-GPT, AnyGPT, MIO). Interleaved multimodal learning has become a core capability for next-generation systems.

### 2. Limitations of Prior Work
Existing interleaved multimodal benchmarks (MMIE, CoMM, ISG-Bench, OpenING, etc.) suffer from three critical shortcomings:

- **Narrow modality coverage**: Limited to text and image only, unable to evaluate broader modality combinations such as audio, video, documents, code, and 3D.
- **Single-capability evaluation**: Each data instance tests only a single capability, failing to reflect compound reasoning that interweaves multiple capabilities in real-world scenarios.
- **Insufficient domain diversity**: Concentrated in general domains, neglecting professional scenarios such as natural science and social science.

### 3. Root Cause
Model capabilities have expanded to any-to-any multimodal conversion, yet a systematic evaluation benchmark to match this development is absent — existing benchmarks lag far behind in evaluation dimensions, modality coverage, and difficulty gradation.

### 4. Paper Goals
To construct a unified interleaved multimodal benchmark that simultaneously covers **multiple modalities** (7), **multiple domains** (30), **multiple capabilities** (multi-task per instance), and **multiple difficulty levels** (3 tiers), along with a compatible evaluation methodology and baseline model.

### 5. Starting Point
Starting from real-world data (public datasets, social media, knowledge bases such as Wikipedia and YouTube), the paper constructs a large-scale interleaved multimodal dataset in open-ended QA format, where both inputs and outputs are interleaved sequences of arbitrary modalities.

### 6. Core Idea
Three main contributions: (1) UniM dataset — the first unified any-to-any interleaved multimodal benchmark; (2) UniM evaluation suite — a three-dimensional assessment covering semantic correctness, structural integrity, and interleaved coherence; (3) UniMA — an agentic baseline model based on traceable evidence reasoning.

## Method

### Overall Architecture

UniM adopts an open-ended QA format where inputs and outputs are interleaved sequences of arbitrary modality combinations, with non-textual content represented by placeholder tags (e.g., `<<image1>>`, `<<video2>>`). The dataset comprises 31,026 high-quality instances spanning 7 modalities (text, image, audio, video, document, code, 3D) across 30 domains (three major categories: natural science, social science, and general domain), partitioned by rule into three difficulty levels: Easy, Medium, and Hard.

### Key Designs

#### 1. Three-Dimensional Evaluation Suite

Traditional metrics (e.g., accuracy) are inadequate for open-ended multimodal generation. The paper designs three complementary evaluation dimensions:

**Semantic Quality and Correctness Score (SQCS)**:
- **Function**: Evaluates semantic alignment and perceptual quality of generated content.
- **Mechanism**: All modality outputs are converted to caption-like text representations; LLM-as-Judge evaluates semantic correctness (SC); modality-specific reference-free quality assessment (GQ) is designed accordingly.
- **Formula**: $\text{SQCS} = \text{SC} \cdot (\eta^{\text{SQCS}} + (1 - \eta^{\text{SQCS}}) \cdot \text{GQ})$, where $\eta^{\text{SQCS}} = 0.7$

**Response Structure Integrity (StS/LeS)**:
- **Function**: Evaluates whether the model adheres to the modality type and quantity requirements defined by the task.
- **Mechanism**: StS (Strict Structure Score) requires exact match in both modality type and placeholder count; LeS (Lenient Structure Score) only requires consistent modality type coverage.
- **Design Motivation**: Decouples structural compliance from semantic correctness to independently measure instruction-following ability.

**Interleaved Coherence Score (ICS)**:
- **Function**: Evaluates cross-modal logical coherence and stylistic consistency.
- **Mechanism**: $\text{ICS} = \eta^{\text{ICS}} \cdot \text{HC} + (1 - \eta^{\text{ICS}}) \cdot \text{SH}$, where HC measures cross-modal semantic-structural consistency, SH measures writing style/visual aesthetic consistency, and $\eta^{\text{ICS}} = 0.8$.

#### 2. Supporting Rate Correction

- **Function**: Distinguishes between a model's absolute and relative capabilities.
- **Mechanism**: A supporting rate $\tau$ is introduced as a conditional correction, $\mathcal{X}^{rel} = \tau \cdot \mathcal{X}^{abs}$, mitigating evaluation bias caused by models not supporting certain modalities.

#### 3. UniMA Agentic Baseline Model

**Receiving Module**: Converts non-textual modalities into task-conditioned dense captions (TCDC), forming a unified text space.

**Traceable Evidence Reasoning Module (TER)**: The core reasoning engine, operating through a four-step Structured Evidence Reasoning Chain (SERC):
- Step 1: Generate TCDC and a rewritten question → improve semantic correctness.
- Step 2: Determine whether data analysis is involved → invoke a code interpreter to generate a data report.
- Step 3: Organize modal content, textual content, and tool lists → improve SQCS, ICS, and StS/LeS respectively.
- Step 4: Integrate all evidence to generate a final report draft.

Key mechanisms: a **Checker** detects factual and logical errors in the report; a **Judger** performs backtracking and corrective reasoning; a reliable traceable reasoning process is achieved through an iterative "generate → check → backtrack → regenerate" cycle.

**Generating Module**: Produces interleaved multimodal output based on the verified final report.

### Loss & Training

UniMA is constructed as an agentic pipeline rather than an end-to-end trained model, integrating dedicated multimodal encoders/decoders. Its core relies on the structured reasoning process of TER rather than gradient optimization. In the evaluation suite, $\eta^{\text{SQCS}} = 0.7$ and $\eta^{\text{ICS}} = 0.8$ are determined by optimal alignment with human evaluation.

## Key Experimental Results

### Main Results

**Table 1: Semantic Quality and Correctness Score (SQCS) and Supporting Rate**

| Domain | Model | SC | GQ | SQCS_abs | τ | SQCS_rel |
|--------|-------|---:|---:|---:|---:|---:|
| Natural Science | AnyGPT | 13.7 | 37.9 | 11.1 | 90.4 | 10.7 |
| Natural Science | NExT-GPT | 8.4 | 23.4 | 6.2 | 62.0 | 2.9 |
| Natural Science | MIO | 19.7 | 29.1 | 15.9 | 59.2 | 10.0 |
| Natural Science | **UniMA** | **59.8** | **79.7** | **57.3** | **100** | **57.3** |
| Social Science | AnyGPT | 18.0 | 23.8 | 15.5 | 94.7 | 14.7 |
| Social Science | NExT-GPT | 16.8 | 31.9 | 13.3 | 89.0 | 10.8 |
| Social Science | MIO | 25.2 | 32.8 | 21.4 | 80.8 | 16.1 |
| Social Science | **UniMA** | **76.2** | **81.0** | **72.7** | **100** | **72.7** |
| General Domain | **UniMA** | **64.7** | **83.6** | **62.2** | **100** | **62.2** |

**Table 2: Interleaved Coherence Score (ICS)**

| Domain | Model | HC | SH | ICS_abs | ICS_rel |
|--------|-------|---:|---:|---:|---:|
| Natural Science | AnyGPT | 39.9 | 46.3 | 41.8 | 38.5 |
| Natural Science | NExT-GPT | 23.5 | 26.1 | 24.9 | 16.3 |
| Natural Science | MIO | 49.4 | 63.7 | 52.1 | 31.8 |
| Natural Science | **UniMA** | **68.4** | **71.9** | **69.1** | **69.1** |
| Social Science | AnyGPT | 31.3 | 35.3 | 32.1 | 29.2 |
| Social Science | MIO | 46.3 | 55.0 | 51.6 | 42.0 |
| Social Science | **UniMA** | **73.1** | **76.5** | **73.8** | **73.8** |
| General Domain | MIO | 68.3 | 77.7 | 60.0 | 45.7 |
| General Domain | **UniMA** | **68.7** | **74.3** | **69.8** | **69.8** |

### Ablation Study

**Table 3: UniMA Ablation Study**

| Configuration | SQCS | ICS | StS | LeS |
|--------------|---:|---:|---:|---:|
| UniMA (Full) | 85.1 | 63.4 | 52.7 | 82.6 |
| w/o TER | 72.9 (-12.2) | 56.6 (-6.8) | 16.4 (-36.3) | 21.8 (-60.8) |
| w/o TCDC | 78.4 (-6.7) | 57.7 (-5.7) | 46.2 (-6.5) | 82.1 (-0.5) |
| w/o Verification | 72.9 (-12.2) | 54.7 (-8.7) | 38.3 (-14.4) | 66.8 (-15.8) |

Key findings: removing TER causes the largest drops in StS/LeS (−36.3/−60.8), demonstrating that traceable reasoning is critical for structural integrity; removing the verification submodule leads to across-the-board degradation, indicating that the check–backtrack–regenerate mechanism is indispensable for reliable output.

### Key Findings

1. **Existing models perform extremely poorly**: Baseline model SQCS is mostly below 20%; NExT-GPT and MIO StS/LeS are mostly below 5%, demonstrating that existing MLLMs fall far short of the requirements for interleaved multimodal learning.
2. **Supporting rate severely constrains relative performance**: AnyGPT general domain StS drops from 12.5% to 9.8% (rel); MIO natural science SQCS drops from 15.9% to 10.0% (rel) — incomplete modality support is a core bottleneck.
3. **Significant domain variation**: Social science achieves the highest SQCS (common concepts + descriptive reasoning); general domain achieves the highest ICS (open-domain data better matches training distribution); natural science performs worst (requires precise terminology and structured logic).
4. **UniMA leads by a large margin**: StS/LeS is 2–6× higher than AnyGPT and 15–40× higher than NExT-GPT/MIO.
5. **Difficulty sensitivity**: Only UniMA exhibits a performance gradient consistent with task difficulty; baseline models already fail on the simplest tasks and cannot differentiate task complexity.

## Highlights & Insights

- **High value in problem formulation**: The paper is the first to systematically define "any-to-any interleaved multimodal learning" and provide a complete evaluation framework, filling the assessment gap across 7 modalities, 30 domains, and multiple difficulty levels.
- **Elegant evaluation suite design**: The three dimensions of SQCS/StS-LeS/ICS decouple semantics, structure, and coherence; Pearson correlation with human evaluation reaches 0.974/0.960.
- **Supporting rate correction** ($\tau$) fairly handles incomplete modality support across models, accounting for both absolute and relative capabilities.
- **TER module design**: Traceable evidence combined with check–backtrack mechanisms effectively improves structured output quality within an agentic framework.

## Limitations & Future Work

- UniMA is fundamentally an agentic pipeline (multi-module integration) rather than an end-to-end unified model; its advantages partly stem from engineering integration rather than breakthroughs in model capability.
- Among the 7 modalities, code (2.6%) and 3D (1.4%) are severely underrepresented, raising questions about the representativeness of evaluation for these modality types.
- The evaluation relies heavily on LLM-as-Judge, introducing biases inherent to the evaluator model itself.
- The absence of a human performance baseline makes it difficult to assess what UniMA's ~60% SQCS represents in absolute terms.
- A portion of the data expansion uses GPT-5-mini to generate candidate instances, potentially introducing synthetic data bias.

## Related Work & Insights

- **Comparison with MMIE/CoMM**: UniM expands modalities from 2 to 7, domains from ~10 to 30, and interleaved combinations from 3–4 to 41, representing an order-of-magnitude leap.
- **Relationship with NExT-GPT/AnyGPT**: These models serve as evaluated baselines; experiments expose their critical limitations in interleaved settings.
- **Inspiration from TER**: The traceable evidence reasoning chain yields significant gains on complex multimodal tasks; the iterative "generate → check → backtrack → regenerate" paradigm is worth adopting.
- **Evaluation methodology inspiration**: Decomposing multimodal evaluation into three orthogonal dimensions — semantics, structure, and coherence — is more informative than a single metric.

## Rating

⭐⭐⭐⭐ An important benchmark contribution that systematically defines and evaluates any-to-any interleaved multimodal learning for the first time, with large data scale and thoughtful evaluation design. However, the UniMA baseline is engineering-heavy, and the core contributions lie in the dataset and evaluation methodology rather than model innovation.

<!-- RELATED:START -->

## Related Papers

- [\[AAAI 2026\] Cross-Space Synergy: A Unified Framework for Multimodal Emotion Recognition in Conversation](../../AAAI2026/audio_speech/cross-space_synergy_a_unified_framework_for_multimodal_emotion_recognition_in_co.md)
- [\[ACL 2026\] MSU-Bench: Musical Score Understanding Benchmark](../../ACL2026/audio_speech/musical_score_understanding_benchmark_evaluating_large_language_models39_compreh.md)
- [\[CVPR 2026\] Tri-Subspaces Disentanglement for Multimodal Sentiment Analysis](tri-subspaces_disentanglement_for_multimodal_sentiment_analysis.md)
- [\[AAAI 2026\] USE: A Unified Model for Universal Sound Separation and Extraction](../../AAAI2026/audio_speech/use_a_unified_model_for_universal_sound_separation_and_extraction.md)
- [\[ACL 2026\] Beyond Transcription: Unified Audio Schema for Perception-Aware AudioLLMs](../../ACL2026/audio_speech/beyond_transcription_unified_audio_schema_for_perception-aware_audiollms.md)

<!-- RELATED:END -->
