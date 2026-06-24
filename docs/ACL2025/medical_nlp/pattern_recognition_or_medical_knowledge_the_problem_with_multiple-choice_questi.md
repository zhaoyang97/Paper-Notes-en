---
title: >-
  [Paper Note] Pattern Recognition or Medical Knowledge? The Problem with Multiple-Choice Questions in Medicine
description: >-
  [ACL 2025][Medical LLM][Medical benchmarking] This paper constructs a medical multiple-choice question (MCQ) benchmark centered around a fictitious organ, "Glianorex," to reveal that LLMs predominantly rely on pattern recognition and test-taking heuristics rather than genuine clinical reasoning in medical MCQ tests. Models scored an average of 64% on entirely fictitious medical knowledge, whereas medical doctors scored only 27%.
tags:
  - "ACL 2025"
  - "Medical LLM"
  - "Medical benchmarking"
  - "multiple-choice question assessment"
  - "fictitious knowledge"
  - "pattern recognition"
  - "clinical reasoning"
date: 2026-05-08
content_hash: 875ca4c1f1a10c8f
---

# Pattern Recognition or Medical Knowledge? The Problem with Multiple-Choice Questions in Medicine

**Conference**: ACL 2025  
**arXiv**: [2406.02394](https://arxiv.org/abs/2406.02394)  
**Code**: [glianorex-gen](https://github.com/maximegmd/glianorex-gen)  
**Area**: Medical NLP  
**Keywords**: Medical benchmarking, multiple-choice question assessment, fictitious knowledge, pattern recognition, clinical reasoning

## TL;DR

This paper constructs a medical multiple-choice question (MCQ) benchmark centered around a fictitious organ, "Glianorex," to reveal that LLMs predominantly rely on pattern recognition and test-taking heuristics rather than genuine clinical reasoning in medical MCQ tests. Models scored an average of 64% on entirely fictitious medical knowledge, whereas medical doctors scored only 27%.

## Background & Motivation

**Background**: LLMs (such as ChatGPT, GPT-4) have demonstrated great potential in the medical domain. Multiple-choice benchmarks like USMLE, MedQA, MedMCQA, PubMedQA, and MMLU-Medical are commonly used to evaluate their medical capability. These benchmarks have become the standard gold standard for measuring medical AI capabilities.

**Limitations of Prior Work**: Multiple-choice evaluation has a fundamental flaw: it can severely overestimate the actual clinical understanding of LLMs. The MCQ format inherently contains numerous exploitable shallow cues (such as option length, linguistic patterns, and elimination strategies). LLMs can achieve high scores through pattern recognition and test-taking heuristics without genuinely understanding medical knowledge. For instance, Meerkat-7B improved its medical benchmark performance by 18.6% solely by training on synthetic MCQs, surpassing Meditron-7B which was pre-trained on a vast corpus of real medical literature.

**Key Challenge**: When training data potentially contains test questions or related knowledge, it is impossible to distinguish whether a model is "memorizing answers" or "performing genuine reasoning." This confounding of data leakage and pattern recognition raises serious doubts about the reliability of current evaluation results.

**Goal**: To design an evaluation method that completely decouples memorization from reasoning, thereby answering the core question: can MCQs effectively evaluate the clinical reasoning capabilities of LLMs?

**Key Insight**: If models still achieve high scores when tested on entirely fictitious medical knowledge that cannot possibly exist in their training data, it would prove that they rely on pattern recognition rather than genuine medical understanding.

**Core Idea**: Create a complete medical knowledge base (textbook + MCQs) centered around a fictitious organ called "Glianorex" (a fictitious gland located in the mediastinum that regulates mood) to serve as a zero-contamination benchmark for testing the true reasoning capabilities of LLMs.

## Method

### Overall Architecture

The entire experimental pipeline comprises three steps: (1) generating a fictitious medical textbook for Glianorex using GPT-4 (approximately 30,000 words each in English and French); (2) generating 264 MCQs based on the textbook content using GPT-4 (bilingual in English and French); and (3) evaluating multiple LLMs under a zero-shot setting and comparing their performance with that of medical doctors.

### Key Designs

1. **Fictitious Medical Knowledge Base Construction**:

    - **Function**: To build a complete and self-consistent corpus of fictitious medical knowledge, ensuring zero overlap with any real-world medical knowledge.
    - **Mechanism**: A fictitious gland named "Glianorex" is conceived, positioned in the mediastinum, which secretes two fictitious hormones, Equilibrion and Neurostabilin, to regulate mood. GPT-4 is leveraged to generate a full textbook top-down according to a predefined chapter structure (anatomy, physiology, biochemistry, pathology, diagnostic tools, etc.), maintaining consistency by sharing summaries of key configurations across chapters. The textbook is generated in both English (~31,000 words) and French (~37,000 words) versions.
    - **Design Motivation**: Completely fictitious knowledge guarantees that models cannot retrieve any relevant information from their training data, thereby cleanly isolating and testing their reasoning and pattern recognition capabilities.

2. **Multiple-Choice Question Generation and Quality Control**:

    - **Function**: To generate standardized, moderately difficult medical MCQs.
    - **Mechanism**: Structured prompts are designed to instruct GPT-4 to generate four-option MCQs requiring multi-step reasoning, using the textbook's table of contents and random paragraphs as context. Around 50% of the questions incorporate random gender and age parameters (12-90 years old) to introduce clinical scenario variations. A temperature of 1 is utilized, and questions are generated four times per paragraph to ensure diversity. French questions are translated from the English versions.
    - **Design Motivation**: Mimicking the USMLE style ensures that the question format aligns with existing medical benchmarks, facilitating direct comparison. The blend of clinical vignettes and knowledge recall questions mirrors the composition of actual licensing examinations.

3. **Multi-dimensional Evaluation Framework**:

    - **Function**: To analyze the question-answering behavior of models from multiple perspectives, going beyond simple accuracy.
    - **Mechanism**: Evaluation is performed using the `lm-evaluation-harness` framework in a zero-shot setting via log-likelihood methods. Beyond accuracy, the analysis includes: Cohen's d effect size analysis (to assess the significance of differences between models), binomial tests (compared to random guessing), answer distribution analysis (identifying which questions are answered correctly/incorrectly by most models), cross-lingual comparison (English vs. French), and ablation remains (medical fine-tuned models vs. base models).
    - **Design Motivation**: A single accuracy metric can be misleading; a deep dive into the models' answering patterns is required to understand the underlying heuristics.

### Evaluated Models

Fourteen models were evaluated, including:
- **Closed-source models**: GPT-3.5-turbo, GPT-4-turbo, GPT-4o
- **Open-source base models**: Yi-1.5-9B/34B, Mistral-7B, Mixtral-8x7B, Llama-3-8B/70B, Qwen1.5-7B/32B/110B
- **Medical fine-tuned models**: Internist.ai (base-7b-v0.2), Meerkat-7b

## Key Experimental Results

### Main Results

| Model | Overall Accuracy | English Accuracy | French Accuracy | vs. Random (25%) |
|------|-----------|-----------|-----------|----------------|
| GPT-4o | ~73% | ~76% | ~70% | p < 10⁻⁵⁴ |
| GPT-4-turbo | ~71% | ~74% | ~68% | p < 10⁻⁵⁴ |
| Llama-3-70B | ~68% | ~71% | ~65% | Significant |
| Qwen1.5-110B | ~68% | ~70% | ~66% | Significant |
| Yi-1.5-34B | ~67% | ~70% | ~64% | Significant |
| GPT-3.5-turbo | ~67% | ~70% | ~64% | Significant |
| Meerkat-7b (Medical Fine-tuned) | ~63% | ~68% | ~58% | Significant |
| Mistral-7B | ~63% | ~66% | ~60% | Significant |
| Medical Doctors | 27% | - | - | Near Random |
| **Model Average** | **~67%** | **~69.5%** | **~63.8%** | - |

### Ablation Study

| Dimension | Key Findings |
|----------|---------|
| Impact of Model Scale | Differences among base models of various scales/architectures are minimal (most Cohen's d values are close to 0). |
| Medical Fine-tuning Effect (EN) | Internist.ai and Meerkat perform slightly better than the base Mistral-7B. |
| Medical Fine-tuning Effect (FR) | No improvement or even degradation, indicating a lack of multilingual generalization in fine-tuning. |
| Maximum Effect Size | Meerkat vs GPT-4o: d=0.270 (small to medium effect). |
| Answer Distribution (EN) | Heavily right-skewed—most questions are answered correctly by most models. |
| Answer Distribution (FR) | Still skewed but to a lesser extent, indicating weakened reasoning heuristics in French. |
| Binomial Test | Even the lowest-scoring model significantly surpasses random probability (p < 10⁻⁵⁴). |

### Key Findings

- **Pattern Recognition Dominates**: On entirely fictitious knowledge, the average model score of 67% significantly exceeds random chance (25%), proving that models exploit shallow cues and test-taking heuristics rather than reasoning based on knowledge.
- **Doctors Score Low Instead**: Medical doctors scored only 27% (near random chance) on the fictitious content, showing that humans genuinely require domain knowledge to answer, whereas LLMs can bypass the knowledge to directly "guess correctly".
- **Minimal Differences Among Models**: Models across different architectures and scales exhibit highly similar performance, suggesting that MCQ-solving capability might be a general byproduct of LLM pre-training.
- **Cross-lingual Degradation**: Performance in French is consistently lower than in English (63.8% vs. 69.5%), implying that shallow linguistic pattern cues are weakened in non-English texts.
- **Limited Value of Medical Fine-tuning**: Fine-tuning shows marginal improvements in English and is ineffective in French, indicating that performance gains stem largely from adapting to the question format rather than acquiring genuine knowledge.

## Highlights & Insights

- **Methodological Innovation in Fictitious Knowledge as a Zero-Contamination Benchmark**: Constructing a complete knowledge system non-existent in any training data thoroughly resolves the confounding issues of "data leakage" and "memorization vs. reasoning." This experimental design approach can be transferred to other benchmarks requiring the evaluation of "genuine understanding" (such as law, finance, etc.).
- **Exquisite Conception of Glianorex**: Modeling a fictitious human organ—concrete enough to support a complete medical knowledge framework (anatomy, physiology, pathology) yet entirely absent from the training corpus. This "carefully designed fiction" balances the internal and external validity of the evaluation.
- **Challenging the Medical AI Evaluation Paradigm**: The results directly question the validity of MCQ-based medical AI benchmarks (such as MultiMedQA). This has profound implications for the field: if high scores do not represent true clinical capability, current narratives claiming "models passing medical licensing exams" may mislead both the medical community and the public.

## Limitations & Future Work

- The generated textbook may contain internal inconsistencies (lacking a comprehensive consistency check), potentially leading to multiple plausible answers for some questions.
- The sample size of 264 questions is relatively small, though it remains in the same order of magnitude as established medical benchmarks.
- Utilizing GPT-4 to generate MCQs may introduce implicit patterns that favor the GPT model family.
- Only the zero-shot setting was evaluated; the performance of few-shot prompting or fine-tuning on fictitious knowledge remains unexplored.
- The authors advocate for shifting medical AI evaluations toward clinically meaningful methods, such as open-ended question answering, clinical scenario simulations, and even clinical trials similar to those used for medical devices.

## Related Work & Insights

- **vs. MultiMedQA (Med-PaLM)**: MultiMedQA consolidates multiple MCQ benchmarks as evaluation standards. This study directly challenges the foundations of this paradigm, demonstrating that high MCQ scores do not equate to high clinical competence.
- **vs. Meerkat-7B**: Meerkat substantially boosted its benchmark scores through training on a small set of synthetic MCQs. The findings of this paper corroborate the same conclusion from an alternative angle: MCQ training enhances "test-taking mechanics" rather than actual "medical knowledge."
- **vs. Med-Gemini**: Google has already begun incorporating human physician evaluations in Med-Gemini. The discoveries of this study reinforce the necessity of this shift.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The concept of using fictitious knowledge as an evaluation tool is exceptionally clever, and the experimental design is elegant and persuasive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Strong model coverage and diverse analytical dimensions, though the sample size is relatively small and few-shot experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ The exposition is highly logical, thought-provoking, and offers a profound critique of medical AI evaluation.
- Value: ⭐⭐⭐⭐⭐ Provides fundamental skepticism and enlightenment regarding the paradigm of medical AI evaluation, warning the entire LLM benchmarking domain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Enhancing Medical Dialogue Generation through Knowledge Refinement and Dynamic Prompt Adjustment](enhancing_medical_dialogue_generation_through_knowledge_refinement_and_dynamic_p.md)
- [\[ACL 2025\] Radar: Enhancing Radiology Report Generation with Supplementary Knowledge Injection](radar_radiology_report_gen.md)
- [\[ICLR 2026\] Knowledgeable Language Models as Black-Box Optimizers for Personalized Medicine](../../ICLR2026/medical_nlp/knowledgeable_language_models_as_black-box_optimizers_for_personalized_medicine.md)
- [\[ICLR 2026\] Cancer-Myth: Evaluating Large Language Models on Patient Questions with False Presuppositions](../../ICLR2026/medical_nlp/cancer-myth_evaluating_large_language_models_on_patient_questions_with_false_pre.md)
- [\[ACL 2026\] Text-Attributed Knowledge Graph Enrichment with Large Language Models for Medical Concept Representation](../../ACL2026/medical_nlp/text-attributed_knowledge_graph_enrichment_with_large_language_models_for_medica.md)

</div>

<!-- RELATED:END -->
