---
title: >-
  [Paper Note] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection
description: >-
  [ACL 2026][Medical Imaging][Cognitive Distortion Detection] This paper proposes decomposing utterances into Emotion–Logic–Behavior (ELB) components and leveraging LLM reasoning to generate multiple cognitive distortion i…
tags:
  - "ACL 2026"
  - "Medical Imaging"
  - "Cognitive Distortion Detection"
  - "Multiple Instance Learning"
  - "LLM Reasoning"
  - "Psychological Decomposition"
  - "Gated Attention"
date: 2026-05-08
content_hash: d316dfb63dc9129e
---

# Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection

**Conference**: ACL 2026
**arXiv**: [2509.17292](https://arxiv.org/abs/2509.17292)
**Code**: [GitHub](https://github.com/cocoboldongle/MVACD)
**Area**: Medical Imaging
**Keywords**: Cognitive Distortion Detection, Multiple Instance Learning, LLM Reasoning, Psychological Decomposition, Gated Attention

## TL;DR

This paper proposes decomposing utterances into Emotion–Logic–Behavior (ELB) components and leveraging LLM reasoning to generate multiple cognitive distortion instances, which are subsequently aggregated via a multi-view gated attention MIL framework for bag-level classification. The approach outperforms direct LLM inference baselines on both the Korean (KoACD) and English (Therapist QA) datasets.

## Background & Motivation

**Background**: Cognitive distortions—such as all-or-nothing thinking, overgeneralization, and personalization—are closely associated with mental health conditions including anxiety and depression. Automatically detecting cognitive distortions is an important task in mental health NLP. Recent work has applied LLMs to this task; for instance, the DoT framework employs structured prompting to improve interpretability.

**Limitations of Prior Work**: (1) Most methods treat utterances as unstructured single inputs and return holistic predictions, neglecting the fact that different cognitive distortions may originate from distinct psychological dimensions (emotional, logical, or behavioral); (2) Multiple cognitive distortions frequently co-occur within a single utterance, yet semantic similarity among distortion types leads to low inter-annotator agreement; (3) Direct LLM inference remains insufficiently accurate—GPT-4o achieves an F1 of only 0.325 on KoACD.

**Key Challenge**: Cognitive distortion detection must simultaneously address two problems: (1) precisely localizing distorted expressions across distinct psychological dimensions of an utterance (emotion, logic, behavior), and (2) aggregating multiple potentially co-occurring distortion instances for a final judgment. Existing methods either perform only holistic classification or rely solely on LLM reasoning, both of which are insufficient.

**Goal**: (1) Structurally decompose utterances into psychologically grounded components (ELB) to provide richer reasoning context; (2) model each LLM-inferred distortion instance as an instance in MIL, enabling fine-grained expression-level classification.

**Key Insight**: The paper combines the cognitive triangle from Cognitive Behavioral Therapy (CBT) to decompose utterances into Emotion–Logic–Behavior components, leverages LLM reasoning to generate multiple distortion candidate instances (each with a type, text span, and salience score), and applies a supervised MIL framework for bag-level classification.

**Core Idea**: Cognitive distortion detection is formulated as a multiple instance learning problem—an utterance serves as a bag, each distortion expression inferred by an LLM constitutes an instance, and multi-view gated attention aggregates instance-level features for final prediction.

## Method

### Overall Architecture

The pipeline consists of three stages: (1) **ELB Extraction**: GPT-4 is prompted zero-shot to decompose each utterance into emotion, logic, and behavior components; (2) **LLM Multi-Instance Reasoning**: Three LLMs (GPT-4o, Gemini 2.0 Flash, Claude 3.7 Sonnet) independently process ELB-augmented utterances and infer multiple cognitive distortion instances (type + text + salience score); (3) **MIL Classification**: All instances are aggregated into a bag, instance weights are computed via multi-view gated attention, and the resulting representation is fused with the original utterance embedding for softmax classification.

### Key Designs

1. **ELB Psychological Decomposition**:

    - **Function**: Decomposes unstructured utterances into three psychologically grounded components, providing finer-grained context for LLM reasoning.
    - **Mechanism**: Grounded in the CBT cognitive triangle (Beck, 1979), the framework replaces "thought" with "logic" to emphasize inferential reasoning. GPT-4 zero-shot prompting independently generates the emotion, logic, and behavior components for each utterance. These components, together with the original text, serve as input to downstream LLM reasoning.
    - **Design Motivation**: Cognitive distortions may originate from different psychological dimensions—"I can't do anything right" primarily involves logic (overgeneralization), whereas "it must be my fault" involves personalization (emotion + behavior). Decomposition enables LLMs to more precisely locate the source of distortions.

2. **LLM Multi-Instance Reasoning and Salience Scoring**:

    - **Function**: Infers multiple cognitive distortion candidate instances from an utterance, each comprising a type, associated text, and salience score.
    - **Mechanism**: Three LLMs independently process each ELB-augmented utterance and output $N$ instances $x_i = (\text{type}_i, \text{text}_i, s_i)$. Salience scores $s_i$ are directly assigned by the LLMs to reflect the relative importance of each instance. All instances are merged into a single bag, and scores are normalized as $\hat{p}_i = s_i / \sum_j s_j$.
    - **Design Motivation**: A single LLM may miss certain distortion types; multi-LLM ensemble improves coverage. Salience scores inject LLM "confidence" into the downstream classification model.

3. **Multi-View Gated Attention MIL**:

    - **Function**: Aggregates instance-level features from multiple perspectives and fuses global utterance information for final prediction.
    - **Mechanism**: Each instance embedding is weighted via gated attention: $h_i = \sigma(W_g \cdot x_i) \cdot \tanh(W_f \cdot x_i) \cdot s_i$. $K$ independent views each compute attention weights, and their average yields $h_\text{multi}$. This is concatenated with the transformed original utterance embedding $z'$, then projected through a linear layer and ReLU to obtain the final bag representation.
    - **Design Motivation**: Single-view attention may focus on only a subset of instances; multi-view design improves coverage of relevant instances. Fusing the original utterance embedding preserves global context that instance-level reasoning may overlook.

### Loss & Training

Standard multi-class cross-entropy loss is used. The learning rate decays linearly from 0.0005 to 0.00001, with early stopping triggered after 10 epochs without improvement on validation loss. Instances are encoded into 384-dimensional vectors using all-MiniLM-L12-v2. All experiments are repeated 10 times and reported as mean ± standard deviation.

## Key Experimental Results

### Main Results

| Method | KoACD Val F1 | KoACD Test F1 | Therapist QA Val F1 | Therapist QA Test F1 |
|---|---|---|---|---|
| Baseline (w/o ELB, w/o Salience) | 0.504 | 0.473 | 0.410 | 0.340 |
| ELB only | 0.519 | 0.483 | 0.438 | 0.378 |
| Salience only | 0.518 | 0.486 | 0.428 | 0.360 |
| **ELB + Salience** | **0.529** | **0.505** | **0.460** | **0.394** |
| GPT-4o (direct inference) | — | 0.325 | — | 0.332 |
| DoT (GPT-4) | — | 0.346 | — | — |

### Ablation Study

| Analysis Dimension | Results |
|---|---|
| Effect of ELB | Reduces missing rate from 10.89% to 8.93%, improving label coverage |
| Per-type F1 | "Should statements" achieves the highest F1 (0.852); "emotional reasoning" the lowest (0.297) |
| LLM baselines | Direct LLM inference F1 for all three LLMs falls below the MIL framework |

### Key Findings

- The combination of ELB decomposition and salience scoring achieves the best performance; both components contribute independently, with ELB contributing more.
- Distortion types with high semantic ambiguity (e.g., emotional reasoning, overgeneralization) yield lower F1, consistently across datasets.
- The proposed framework (0.505/0.394) substantially outperforms GPT-4o direct inference (0.325/0.332) and DoT (0.346).
- "Should statements" achieves an F1 of 0.852 on the Korean dataset but only 0.460 on English, revealing significant cross-lingual stylistic differences.

## Highlights & Insights

- Integrating the CBT cognitive triangle into an NLP pipeline represents an elegant fusion of psychological theory and technical methodology—ELB decomposition aligns the model's reasoning process more closely with clinical practice.
- The MIL framework enables instance-level attribution of predictions, providing attribution-based interpretability.
- Multi-LLM ensemble reasoning mitigates blind spots of individual models and improves distortion type coverage.

## Limitations & Future Work

- ELB components have not been independently validated by psychological experts; extraction errors may propagate to downstream stages.
- Substantial imbalance exists in instance counts across distortion types ("jumping to conclusions" 19.5% vs. "disqualifying the positive" 2.9%), potentially biasing attention toward high-frequency types.
- Reliance on proprietary LLMs (GPT-4, Claude) limits portability and raises privacy concerns.
- The framework does not produce natural language explanations; interpretability is limited to the attribution level.

## Related Work & Insights

- **vs. DoT (Chen et al.)**: DoT improves LLM interpretability via structured prompting but still operates under a single-input single-output paradigm; this paper decomposes reasoning outputs into multiple instances and aggregates them via MIL.
- **vs. conventional MIL in NLP**: Prior MIL applications in NLP define instances at the sentence or paragraph level; this paper is the first to treat LLM-inferred expressions as instances.
- **vs. zero-shot LLM detection**: Direct LLM inference F1 falls substantially below the supervised MIL framework, demonstrating that pure LLM reasoning still lags behind on fine-grained classification tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The combination of psychological theory, LLM reasoning, and MIL is novel, though each individual component relies on existing techniques.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Bilingual evaluation and ablation analysis are thorough, but dataset scale is relatively small.
- **Writing Quality**: ⭐⭐⭐⭐ The framework is described clearly, though some details are relegated to the appendix.
- **Value**: ⭐⭐⭐⭐ Provides a more fine-grained detection paradigm for mental health NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Fair Lung Disease Diagnosis from Chest CT via Gender-Adversarial Attention Multiple Instance Learning](../../CVPR2026/medical_imaging/fair_lung_disease_diagnosis_from_chest_ct_via_gend.md)
- [\[ACL 2026\] Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)
- [\[CVPR 2026\] MIL-PF: Multiple Instance Learning on Precomputed Features for Mammography Classification](../../CVPR2026/medical_imaging/milpf_multiple_instance_learning_on_precomputed_fe.md)
- [\[ACL 2026\] Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised Reinforcement Learning Approach](eliciting_medical_reasoning_with_knowledge-enhanced_data_synthesis_a_semi-superv.md)
- [\[CVPR 2026\] Every Error has Its Magnitude: Asymmetric Mistake Severity Training for Multiclass Multiple Instance Learning](../../CVPR2026/medical_imaging/every_error_has_its_magnitude_asymmetric_mistake_severity_training_for_multiclas.md)

</div>

<!-- RELATED:END -->
