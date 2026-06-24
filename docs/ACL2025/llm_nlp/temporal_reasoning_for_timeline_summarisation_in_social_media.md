---
title: >-
  [Paper Note] Temporal Reasoning for Timeline Summarisation in Social Media
description: >-
  [ACL 2025][LLM (Other)][Temporal Reasoning] This paper proposes enhancing the temporal reasoning capabilities of LLMs by constructing a new narrative temporal reasoning dataset, NarrativeReason. It transfers temporal reasoning knowledge to smaller models via a knowledge distillation framework, while training them to perform timeline summarisation. This approach achieves state-of-the-art performance and significantly reduces hallucinations in cross-domain mental health summari…
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Temporal Reasoning"
  - "Timeline Summarisation"
  - "Knowledge Distillation"
  - "Social Media"
  - "Mental Health"
date: 2026-05-08
content_hash: cd09744c0bd44354
---

# Temporal Reasoning for Timeline Summarisation in Social Media

**Conference**: ACL 2025  
**arXiv**: [2501.00152](https://arxiv.org/abs/2501.00152)  
**Code**: None  
**Area**: Others  
**Keywords**: Temporal Reasoning, Timeline Summarisation, Knowledge Distillation, Social Media, Mental Health

## TL;DR

This paper proposes enhancing the temporal reasoning capabilities of LLMs by constructing a new narrative temporal reasoning dataset, NarrativeReason. It transfers temporal reasoning knowledge to smaller models via a knowledge distillation framework, while training them to perform timeline summarisation. This approach achieves state-of-the-art performance and significantly reduces hallucinations in cross-domain mental health summarisation tasks.

## Background & Motivation

1. **Background**: Timeline Summarisation requires extracting event sequences and generating coherent summaries from long texts (e.g., sequences of social media posts). Pre-existing methods usually construct event graphs or cluster event timelines to identify related events.

2. **Limitations of Prior Work**: Mental health-related posts on social media present unique challenges: events lack explicit timestamps, requiring contextual inference of chronological order; mental state events are difficult to identify; and models are highly prone to hallucination. Existing temporal reasoning research mainly focuses on improving the temporal reasoning capabilities of LLMs themselves, without exploring how to apply them to improve downstream tasks.

3. **Key Challenge**: A clear link exists between temporal reasoning and timeline summarisation (temporal reasoning helps maintain temporal consistency and correct event ordering), but current research treats them in isolation. Moreover, existing temporal reasoning datasets primarily handle pairwise event relations rather than complex temporal relations among multiple events in a narrative.

4. **Goal**: How to effectively transfer enhanced temporal reasoning capability to downstream timeline summarisation tasks? How to maintain performance in cross-domain scenarios (trained on the news domain, tested on the mental health domain)?

5. **Key Insight**: The authors hypothesize that existing pairwise temporal reasoning datasets cause models to learn shortcuts rather than genuinely understand temporal relationships. Therefore, they construct a narrative-based multi-event temporal reasoning dataset and bridge temporal reasoning and summarisation tasks via knowledge distillation.

6. **Core Idea**: Enhance large models with narrative-level temporal reasoning data, then inject temporal understanding capabilities into the summary generation process of smaller models through knowledge distillation.

## Method

### Overall Architecture

The method consists of two stages. Stage 1: Fine-tune a large model (Teacher, e.g., LLaMA-3) on the newly constructed NarrativeReason dataset to enhance its temporal reasoning capabilities. Stage 2: Freeze the Teacher's parameters, fine-tune a small model (Student, Phi-3-mini) on the news timeline summarisation dataset, and games-simultaneously acquire temporal reasoning knowledge from the Teacher through knowledge distillation. Finally, summarisation quality is evaluated on social media timelines in the mental health domain.

### Key Designs

1. **NarrativeReason Dataset Construction**:

    - **Function**: Provide multi-event narrative-level temporal reasoning training data.
    - **Mechanism**: Reconstructed from the NarrativeTime dataset by extracting verb-triggered event triples (e.g., `<Indonesian stock market value, fall, 12%>`), then building temporal relationship QA pairs for all events in the narrative. A total of 19,614 temporal relation QA pairs are generated from 668 events extracted from 30 articles. The question format is "Based on the story, determine the temporal relationship between Event A and Event B (BEFORE/AFTER/INCLUDES/SIMULTANEOUS)".
    - **Design Motivation**: Existing datasets (such as TEMPLAMA) only handle pairwise events, which can allow models to "cheat" by memorizing high-frequency answers. Multi-event sequence reasoning requires identifying patterns, dependencies, and causal chains, which is closer to the needs of actual summarisation tasks.

2. **Three Knowledge Distillation Strategies**:

    - **Function**: Transfer temporal reasoning knowledge from the Teacher to the Student.
    - **Mechanism**: Three complementary distillation methods are adopted: (1) NST (Neuron Selectivity Transfer): using MMD to match the neuron activation pattern distributions of the Teacher's and Student's final hidden layers; (2) CRD (Contrastive Representation Distillation): maximizing the mutual information between the Teacher's and Student's representations via contrastive learning; (3) PRT (Probabilistic Knowledge Transfer): using KL divergence to match the conditional probability distributions of the Teacher's and Student's output logits, employing a cosine similarity kernel function.
    - **Design Motivation**: Different distillation strategies capture knowledge at different levels. Experiments show that the NST+PRT combination performs best because both focus on structural consistency, whereas CRD focuses on instance discrimination, which is less suited for temporal reasoning tasks.

3. **Cross-Domain Mental Health Summarisation Generation**:

    - **Function**: Validate the transferability of temporal reasoning capability.
    - **Mechanism**: The Student is trained on news-domain summarisation data but directly applied to TalkLife data in the mental health domain. Summaries are generated across three clinical concept dimensions (Diagnosis, Interpersonal, Moments of Change) using the format defined by Song et al. (2024).
    - **Design Motivation**: The cross-domain setting better validates the transferability of temporal reasoning as a general capability rather than simple in-domain overfitting.

### Loss & Training

The total loss of the Student model is composed of the language modeling loss $L_{language}$ (used for next token prediction in timeline summarisation) and the knowledge distillation losses ($L_{PKT}$, $L_{MMD^2}$, $L_{CRD}$). The Teacher is fine-tuned via SFT using LoRA.

## Key Experimental Results

### Main Results

| Model Configuration | FC (Fact Consistency) | EA (Evidence Alignment) |
|---------|-------|-------|
| P-Phi (NST&PRT) | **0.438** | **0.973** |
| L-Phi (NST&PRT) | 0.424 | 0.971 |
| Phi_ICL | 0.412 | 0.965 |
| TH-VAE (Prev. SOTA) | 0.378 | 0.970 |
| LLaMA (zero-shot) | 0.372 | 0.956 |
| KD_origin | 0.332 | 0.967 |
| KD_timeline | 0.330 | 0.965 |
| Phi_joint | 0.238 | 0.941 |
| Phi_tl | 0.184 | 0.966 |
| Phi_temp | 0.141 | 0.895 |

### Ablation Study

| Configuration | FC | EA | Description |
|------|-----|-----|------|
| P-Phi (NST&PRT) | 0.438 | 0.973 | Best combination |
| P-Phi (PRT only) | 0.378 | 0.965 | PRT only |
| P-Phi (NST only) | 0.344 | 0.968 | NST only |
| P-Phi (CRD only) | 0.369 | 0.954 | CRD only |
| P-Phi (NST&CRD) | 0.397 | 0.969 | NST+CRD combination |

Human Evaluation (5-point Likert Scale):

| Evaluation Dimension | Phi | P-Phi | LLaMA | L-Phi |
|---------|-----|-------|-------|-------|
| Factual Consistency | 2.90 | 3.32 | 3.58 | **3.83** |
| Usefulness (Overall) | 2.60 | 3.13 | 3.17 | **3.48** |
| Diagnosis | 2.90 | 3.37 | 3.45 | **3.62** |
| Interpersonal | 2.95 | 3.00 | 3.40 | **3.51** |
| Moments of Change | 2.97 | 2.97 | 3.42 | **3.47** |

### Key Findings

- The NST+PRT combination performs best because both focus on structural consistency (distribution matching), whereas CRD focuses on instance discrimination, which is less suited for temporal reasoning tasks.
- Fine-tuning solely on temporal reasoning data (Phi_temp) severely degrades summarisation performance (FC=0.141), whereas introducing it via KD significantly improves it.
- Direct joint training (Phi_joint) also fails (FC=0.238), indicating that naive multi-task learning cannot effectively integrate these two capabilities.
- L-Phi (large Teacher LLaMA $\rightarrow$ small Student Phi) consistently outperforms P-Phi (small Teacher Phi $\rightarrow$ small Student Phi) in human evaluations, indicating that a larger Teacher is indeed beneficial.
- UMAP visualizations show that KD models exhibit more polysemantic activations, and CKA analysis indicates that KD models better maintain and refine input information.

## Highlights & Insights

- **Exquisite design of narrative-level temporal reasoning**: Reasoning about multi-event temporal structures in narrative contexts rather than simple pairwise relationships aligns better with actual summarisation requirements.
- **Deep analysis of why KD outperforms joint training**: UMAP and CKA analyses reveal how KD helps the model learn better representations—the KD model refines information layer-by-layer, whereas the joint model SFT saturates prematurely.
- **Cross-domain transfer validates generality of temporal reasoning**: Training on the news domain but remaining effective on the mental health domain demonstrates that temporal reasoning acts as a transferable foundational capability.

## Limitations & Future Work

- The model tends to provide specific DSM diagnoses (e.g., PTSD, bipolar disorder) rather than more cautious phrasing (such as "evidence suggests potential...").
- Summary content tends to be generic, lacking depth in personalized analysis.
- The NarrativeReason dataset is limited in size (30 articles) and could be expanded to a larger scale.
- Improvement in the MoC (Moments of Change) dimension is limited, potentially requiring specialized design.

## Related Work & Insights

- **vs TH-VAE (Song et al. 2024)**: TH-VAE first extracts mental health-related evidence using a VAE and then generates summaries, whereas this work directly uses annotated evidence to generate high-level summaries, focusing on validating the role of temporal reasoning.
- **vs TEMPREASON (Tan et al. 2023)**: TEMPREASON handles pairwise event relations, whereas the NarrativeReason dataset in this study handles narrative-level multi-event relations, which is better suited for downstream tasks.
- The paradigm of using knowledge distillation for cross-task capability transfer can be generalized to other "foundational capability $\rightarrow$ downstream task" scenarios.

## Rating

- Novelty: ⭐⭐⭐⭐ The framework design of injecting temporal reasoning into summarisation tasks via KD is relatively novel, though KD itself is a mature technology.
- Experimental Thoroughness: ⭐⭐⭐⭐ Thorough ablation with automatic evaluation + human evaluation + representation analysis, and a comprehensive comparison of three KD strategies.
- Writing Quality: ⭐⭐⭐⭐ Clear chain of motivation and intuitive methodology figures, though the math formula equations are somewhat verbose in LaTeX rendering.
- Value: ⭐⭐⭐⭐ The idea of using temporal reasoning to improve summarisation quality has general appeal, though it is only validated on a single test set (30 TalkLife timelines).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] TReMu: Towards Neuro-Symbolic Temporal Reasoning for LLM-Agents with Memory in Multi-Session Dialogues](tremu_towards_neuro-symbolic_temporal_reasoning_for_llm-agents_with_memory_in_mu.md)
- [\[ACL 2025\] SocialEval: Evaluating Social Intelligence of Large Language Models](socialeval_evaluating_social_intelligence_of_large_language_models.md)
- [\[ACL 2025\] Explicit and Implicit Data Augmentation for Social Event Detection](explicit_and_implicit_data_augmentation_for_social_event_detection.md)
- [\[ACL 2025\] MIRAGE: Exploring How Large Language Models Perform in Complex Social Interactive Environments](mirage_exploring_how_large_language_models_perform_in_complex_social_interactive.md)
- [\[ACL 2025\] ExpliCa: Evaluating Explicit Causal Reasoning in Large Language Models](explica_evaluating_explicit_causal_reasoning_in_large_language_models.md)

</div>

<!-- RELATED:END -->
