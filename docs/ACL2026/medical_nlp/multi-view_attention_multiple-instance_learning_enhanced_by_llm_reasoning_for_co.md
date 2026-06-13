---
title: >-
  [Paper Note] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection
description: >-
  [ACL 2026][Medical NLP][Cognitive distortion detection] This paper proposes decomposing utterances into Emotion-Logic-Behavior (ELB) components and utilizing LLMs to reason about multiple cognitive distortion instances.…
tags:
  - "ACL 2026"
  - "Medical NLP"
  - "Cognitive distortion detection"
  - "multiple-instance learning"
  - "LLM reasoning"
  - "psychological decomposition"
  - "gated attention"
date: 2026-05-08
content_hash: 8fdcd2fa7bb5b9cc
---

# Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection

**Conference**: ACL 2026  
**arXiv**: [2509.17292](https://arxiv.org/abs/2509.17292)  
**Code**: [GitHub](https://github.com/cocoboldongle/MVACD)  
**Area**: Medical Imaging  
**Keywords**: Cognitive distortion detection, multiple-instance learning, LLM reasoning, psychological decomposition, gated attention

## TL;DR

This paper proposes decomposing utterances into Emotion-Logic-Behavior (ELB) components and utilizing LLMs to reason about multiple cognitive distortion instances. These are then processed through a multi-view gated attention MIL framework for bag-level classification. Ours outperforms direct LLM reasoning baselines on both Korean (KoACD) and English (Therapist QA) datasets.

## Background & Motivation

**Background**: Cognitive distortions (e.g., all-or-nothing thinking, overgeneralization, personalization) are closely related to mental health disorders such as anxiety and depression. Automated detection of these distortions is a critical task in mental health NLP. Recently, LLMs have been applied to this task, such as the DoT framework which uses structured prompts to improve interpretability.

**Limitations of Prior Work**: (1) Most methods treat utterances as single unstructured inputs and return holistic predictions, ignoring that different cognitive distortions may stem from different psychological dimensions (Emotion/Logic/Behavior) of the utterance; (2) Multiple cognitive distortions often co-occur in a single utterance, but semantic similarity between types leads to low inter-expert annotation agreement; (3) The accuracy of direct LLM reasoning is insufficient—GPT-4o achieves an F1 of only 0.325 on KoACD.

**Key Challenge**: Cognitive distortion detection requires addressing two problems simultaneously: precisely locating distorted expressions within different psychological dimensions (Emotion, Logic, Behavior) of an utterance, and aggregating multiple potentially co-existing distortion instances to make a final judgment. Existing methods either perform only holistic classification or only LLM reasoning, both of which are inadequate.

**Goal**: (1) Decompose utterances into structured psychologically-grounded components (ELB) to provide richer reasoning context; (2) Model each distortion instance reasoned by the LLM as an instance in MIL to achieve fine-grained expression-level classification.

**Key Insight**: This work combines the Cognitive Triad theory from CBT (Cognitive Behavioral Therapy) to decompose utterances into Emotion-Logic-Behavior. It leverages the reasoning capabilities of LLMs to generate multiple candidate distortion instances (including type, text segments, and salience scores), and then utilizes an MIL framework for supervised bag-level classification.

**Core Idea**: Model cognitive distortion detection as a multiple-instance learning problem—the utterance is the bag, and each distorted expression reasoned by the LLM is an instance. A multi-view gated attention mechanism is used to aggregate instance-level features for the final prediction.

## Method

### Overall Architecture

The process consists of three steps: (1) **ELB Extraction**: Use GPT-4 with zero-shot prompting to decompose each utterance into Emotion, Logic, and Behavior components; (2) **LLM Multi-instance Reasoning**: Three LLMs (GPT-4o, Gemini 2.0 Flash, Claude 3.7 Sonnet) independently process the ELB-enhanced utterances to reason about multiple cognitive distortion instances (type + text + salience score); (3) **MIL Classification**: All instances are aggregated into a bag. Instance weights are calculated through a multi-view gated attention mechanism. After fusing with the original utterance embedding, softmax classification is performed.

### Key Designs

1.  **ELB Psychological Decomposition**:
    - **Function**: Decomposes unstructured utterances into three fundamental psychological components to provide finer-grained context for LLM reasoning.
    - **Mechanism**: Based on the CBT cognitive triad (Beck, 1979), "Logic" replaces "Thoughts" to emphasize the nature of reasoning. Emotion, Logic, and Behavior components are generated independently for each utterance using GPT-4 zero-shot prompting. These components, along with the original text, serve as input for downstream LLM reasoning.
    - **Design Motivation**: Cognitive distortions may originate from different psychological dimensions—"I can’t do anything right" primarily involves logic (overgeneralization), while "It must be my fault" involves personalization (emotion + behavior). Decomposition allows LLMs to locate the source of distortion more accurately.

2.  **LLM Multi-instance Reasoning and Salience Scoring**:
    - **Function**: Infers multiple candidate cognitive distortion instances from an utterance, each containing a type, relevant text, and a salience score.
    - **Mechanism**: Three LLMs independently process each ELB-enhanced utterance, each outputting $N$ instances $x_i = (\text{type}_i, \text{text}_i, s_i)$. The salience score $s_i$ is assigned directly by the LLM, reflecting the relative importance of that instance. All instances are merged into a single bag, and scores are normalized as $\hat{p}_i = s_i / \sum_j s_j$.
    - **Design Motivation**: A single LLM might miss certain distortion types; multi-LLM integration improves coverage. Salience scores inject the "confidence" of the LLM into the downstream classification model.

3.  **Multi-View Gated Attention MIL**:
    - **Function**: Aggregates instance-level features from multiple perspectives and fuses global utterance information for the final prediction.
    - **Mechanism**: Each instance embedding is processed via gated attention $h_i = \sigma(W_g \cdot x_i) \cdot \tanh(W_f \cdot x_i) \cdot s_i$ to calculate weights. $K$ independent views each compute attention, and the average $h_\text{multi}$ is taken. This is then concatenated with the transformed original utterance embedding $z'$, and the final bag representation is obtained through linear projection and ReLU.
    - **Design Motivation**: Single-view attention might only focus on a subset of instances; the multi-view design can cover more relevant instances. Integrating the original utterance embedding preserves global context that might be missed during instance-level reasoning.

### Loss & Training

Standard multi-class cross-entropy loss is used. The learning rate linearly decays from 0.0005 to 0.00001, with early stopping if the validation loss does not improve for 10 epochs. Instances are encoded into 384-dimensional vectors using all-MiniLM-L12-v2. All experiments were repeated 10 times, reporting mean ± standard deviation.

## Key Experimental Results

### Main Results

| Method | KoACD Val F1 | KoACD Test F1 | Therapist QA Val F1 | Therapist QA Test F1 |
|------|-------------|-------------|-------------------|-------------------|
| Baseline (No ELB, No Salience) | 0.504 | 0.473 | 0.410 | 0.340 |
| ELB only | 0.519 | 0.483 | 0.438 | 0.378 |
| Salience only | 0.518 | 0.486 | 0.428 | 0.360 |
| **ELB + Salience** | **0.529** | **0.505** | **0.460** | **0.394** |
| GPT-4o (Direct Reasoning) | - | 0.325 | - | 0.332 |
| DoT (GPT-4) | - | 0.346 | - | - |

### Ablation Study

| Analysis Dimension | Result |
|----------|------|
| ELB Effect | Reduced missing rate from 10.89% to 8.93%, improving label coverage |
| Per-type F1 | "Should Statements" highest (0.852), "Emotional Reasoning" lowest (0.297) |
| LLM Baselines | Direct reasoning F1 for all three LLMs was lower than the MIL framework |

### Key Findings

- The combination of ELB decomposition and salience scores yields the best results; both contribute independently, but ELB contributes more significantly.
- Distortion types with high semantic ambiguity (e.g., emotional reasoning, overgeneralization) show lower F1 scores, which is consistent across datasets.
- Ours (0.505/0.394) significantly outperforms direct GPT-4o reasoning (0.325/0.332) and DoT (0.346).
- "Should Statements" achieve an F1 as high as 0.852 on Korean data but only 0.460 on English—indicating significant differences in linguistic styles.

## Highlights & Insights

- Integrating the CBT cognitive triad into an NLP pipeline represents an elegant fusion of psychological theory and technical methodology—ELB decomposition aligns the model's reasoning process more closely with clinical practice.
- The introduction of the MIL framework allows the model to track the source of predictions at the instance level, providing attribution-based interpretability.
- Multi-LLM ensemble reasoning avoids the blind spots of a single model and improves the coverage of distortion types.

## Limitations & Future Work

- ELB components were not independently validated by psychological experts; extraction errors may propagate downstream.
- There is a significant imbalance in the number of instances for different distortion types ("Jumping to Conclusions" 19.5% vs. "Discounting the Positive" 2.9%), which may bias attention toward high-frequency types.
- Reliance on commercial LLMs (GPT-4, Claude) limits portability and privacy protection.
- Natural language explanations are not provided—interpretability is limited to the attribution level.

## Related Work & Insights

- **vs DoT (Chen et al.)**: DoT uses structured prompts to enhance LLM interpretability but remains a single-input single-output system; ours decomposes reasoning results into multiple instances and aggregates them via MIL.
- **vs Traditional MIL-NLP**: Previous MIL applications in NLP defined instances at the sentence or paragraph level; this work is the first to treat expressions reasoned by LLMs as instances.
- **vs Zero-shot LLM Detection**: Direct LLM reasoning F1 is far lower than the supervised MIL framework, indicating that pure LLM reasoning still lags in fine-grained classification tasks.

## Rating

- Novelty: ⭐⭐⭐⭐ The combination of psychological theory, LLM, and MIL is novel, although individual components use existing technologies.
- Experimental Thoroughness: ⭐⭐⭐⭐ Bilingual evaluation and ablation analysis are comprehensive, though the data scale is relatively small.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear, though some details are relegated to the appendix.
- Value: ⭐⭐⭐⭐ Provides a more refined detection paradigm for mental health NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Eliciting Medical Reasoning with Knowledge-enhanced Data Synthesis: A Semi-Supervised RL Approach](eliciting_medical_reasoning_with_knowledge-enhanced_data_synthesis_a_semi-superv.md)
- [\[ACL 2026\] BioHiCL: Hierarchical Multi-Label Contrastive Learning for Biomedical Retrieval with MeSH Labels](biohicl_hierarchical_multi-label_contrastive_learning_for_biomedical_retrieval_w.md)
- [\[ACL 2026\] CURE-Med: Curriculum-Informed Reinforcement Learning for Multilingual Medical Reasoning](cure-med_curriculum-informed_reinforcement_learning_for_multilingual_medical_rea.md)
- [\[ACL 2026\] MultiDx: A Multi-Source Knowledge Integration Framework towards Diagnostic Reasoning](multidx_a_multi-source_knowledge_integration_framework_towards_diagnostic_reason.md)
- [\[ACL 2026\] Beyond the Individual: Virtualizing Multi-Disciplinary Reasoning for Clinical Intake via Collaborative Agents](beyond_the_individual_virtualizing_multi-disciplinary_reasoning_for_clinical_int.md)

</div>

<!-- RELATED:END -->
