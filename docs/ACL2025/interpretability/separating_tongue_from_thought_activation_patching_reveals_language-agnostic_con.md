---
title: >-
  [Paper Note] Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers
description: >-
  [ACL 2025][Interpretability][Multilingual representations] Through activation patching experiments, this work provides the first causal evidence demonstrating the existence of language-decoupled concept representations inside large language models. The model first determines the output language and then the concept, and averaging concept representations across languages not only preserves but actually improves translation accuracy.
tags:
  - "ACL 2025"
  - "Interpretability"
  - "Multilingual representations"
  - "language-agnostic concepts"
  - "activation patching"
  - "mechanistic interpretability"
  - "Transformer"
date: 2026-05-08
content_hash: ff91fd88db712da1
---

# Separating Tongue from Thought: Activation Patching Reveals Language-Agnostic Concept Representations in Transformers

**Conference**: ACL 2025  
**arXiv**: [2411.08745](https://arxiv.org/abs/2411.08745)  
**Code**: [https://github.com/Butanium/llm-lang-agnostic](https://github.com/Butanium/llm-lang-agnostic)  
**Area**: Interpretability  
**Keywords**: Multilingual representations, language-agnostic concepts, activation patching, mechanistic interpretability, Transformer

## TL;DR

Through activation patching experiments, this work provides the first causal evidence demonstrating the existence of language-decoupled concept representations inside large language models. The model first determines the output language and then the concept, and averaging concept representations across languages not only preserves but actually improves translation accuracy.

## Background & Motivation

Large Language Models (LLMs), despite being trained predominantly on English data, demonstrate surprising multilingual capabilities. A core question is: Do LLMs develop **language-agnostic, unified concept representations**? For instance, when processing the English "cat" and the French "chat", does the model map them to the same internal conceptual representation, or does it maintain separate language-specific representations?

Prior work has provided indirect evidence:
- Wendler et al. (2024) found that intermediate decoding always passes through English before translating to the target language.
- Instruction tuning and safety alignment on English alone can generalize to other languages.
- However, these are strictly **observational** pieces of evidence lacking causal demonstration.

The key insight of this paper is: If language and concept can vary independently (H1), then averaging concept representations across languages should preserve conceptual information; if they are entangled (H2), averaging will yield an inconsistent mixture, disrupting the model's translation capability. This opposing hypothesis setup provides a perfect verification framework for causal experiments.

## Method

### Overall Architecture

A series of carefully designed activation patching experiments are proposed to inject residual stream activations of a source prompt into a target prompt during the forward pass of a translation task. By observing the changes in probabilities of **four combinations** in the output distribution (source concept × source language, source concept × target language, target concept × source language, target concept × target language), the authors infer how language and concept information are encoded across different layers.

### Key Designs

1. **Exploratory Patching Experiment (Determining Timing)**:

    - Construct translation prompt pairs: source prompt TP(de→it, book) and target prompt TP(fr→zh, lemon).
    - Patch the residual stream at the last token at each layer.
    - **Key Finding**: A three-stage pattern: Layers 0-11 output target concept + target language; Layers 12-16 output target concept + source language; Layers 16-31 output source concept + source language.
    - **Interpretation**: The model first computes the output language (~Layer 12), and then determines the concept (~Layer 16).

2. **Further Evidence Experiment**:

    - Modify the patching position: instead of the last token of the prompt, patch the last token of the word to be translated.
    - Using TPconcept (the prompt truncated to the concept word), patch from this layer to all subsequent layers.
    - Successfully observe the combination of source concept + target language, verifying the feasibility of both H1 and H2.

3. **Disambiguation Experiment (Distinguishing H1 from H2)**:

    - **Core Idea**: Generate multiple source prompts with different input/output languages for the same concept.
    - Average the activations of these source prompts at the concept word position and inject the average into the target prompt.
    - H1 Prediction: Averaging preserves concept information (since $z_C$ is identical across languages), and translation should still succeed.
    - H2 Prediction: Averaging causes mutual interference among different language versions, and translation should fail.
    - **Result**: Averaging does not impair, but actually **improves** the translation accuracy of $P(C_S^{zh})$.
    - **Explanation**: Averaging acts like a "majority voting" mechanism, achieving concept denoising.

4. **Definition Generation Experiment (Validation on Multi-token Generation)**:

    - Design a new definition prompt template (DP), where the task is to describe a concept rather than translate it.
    - Inject cross-lingual averaged concept representations into the definition prompt to let the model generate natural language descriptions.
    - Use paraphrase-multilingual-mpnet-base-v2 to evaluate semantic similarity to BabelNet gold definitions.
    - Results show that the model successfully generates accurate definitions from cross-lingually averaged representations.

### Loss & Training

This is a purely analytical work and does not involve model training. All experiments use the inference process of pre-trained models with causal intervention through activation patching.

## Key Experimental Results

### Main Results

**Translation Probability Changes (Llama 2 7B, de→it Source, fr→zh Target)**:

| Patching Layer Range | Dominant Output | Meaning |
|-----------|---------|------|
| Layer 0-11 | $P(C_T^{zh})$ is the highest | Target concept + target language (Patching did not cover language/concept) |
| Layer 12-16 | $P(C_T^{it})$ is the highest | Target concept + source language (Language is covered, concept is not) |
| Layer 16-31 | $P(C_S^{it})$ is the highest | Source concept + source language (Both language and concept are covered) |

**Cross-lingual Average vs. Single Source (Translation Accuracy of Layers 0-15)**:

| Setup | $P(C_S^{zh})$ | Trend |
|------|--------------|------|
| Single source prompt patching | ~0.35 | Baseline |
| 5-language average patching | ~0.45 | Significant improvement |

### Ablation Study

| Variable | Finding |
|------|------|
| Patching Position: Last token vs. Concept token | Concept token patching yields source concept + target language outputs |
| Average vs. Single Source | Averaging **improves** rather than degrades translation accuracy |
| Different Source Language Combinations | Consistent effects |
| Randomized Source Prompts | Layers 0-11 indeed contain no task-specific information |

### Key Findings

- Language and concepts are encoded in **different layers** in the residual stream and can be **independently manipulated**.
- Cross-lingual averaging $\approx$ concept denoising, supporting H1 (language-agnostic representation).
- The same phenomenon is consistently observed in Llama 2 7B, Llama 2 70B, Llama 3 8B, Mistral 7B, Qwen 1.5 7B, Aya 23 8B, and Gemma 2 2B.
- Even models specifically trained for multilingual scenarios (Aya 23) employ language-agnostic conceptual representations.

## Highlights & Insights

- **Extremely Elegant Experimental Design**: Designing discriminative experiments based on differing predictions of two opposing hypotheses is a classic application of scientific methodology.
- **Counter-intuitive Finding**: Cross-lingual averaging does not disrupt concept representations; instead, it improves translation—providing strong causal evidence for language-agnostic representations.
- **Broad Generalizability**: 7 models with different architectures, scales, and training data show consistent patterns, indicating that language-agnostic representations are a universal characteristic of Transformers.
- **Definition Generation Experiment**: Extending from single-token prediction to multi-token generation reinforces the practical significance of the findings.
- **Interesting Contrast with Parallel Work**: Fierro et al. (2025) observed the opposite sequence—concept first, language second—in factual recall tasks, suggesting that different tasks might employ different processing pipelines.

## Limitations & Future Work

- Only simple concepts (word-level translation) are studied; complex or language-specific concepts (e.g., "Waldeinsamkeit") remain unexplored.
- Finer-grained probing is required to determine the extent to which concepts can specialize to specific languages.
- The experimental framework relies on few-shot translation prompt formats; whether this generalizes to other multilingual tasks (e.g., QA, summarization) remains to be seen.
- The degree of decoupling between concept and language—are they fully independent or still partially entangled in subtle ways?
- The study mainly investigates English-centric pre-trained models; whether the same conclusions hold for balanced multilingual training models warrants more validation.

## Related Work & Insights

- Built upon the logit lens observations of Wendler et al. (2024), but upgrading the analysis from observational to causal.
- Aligned with the cross-lingual representation research from the BERT era (Conneau et al., 2020; Pires et al., 2019), generalizing the conclusions to decoder-only architectures.
- Inspired by the activation patching methods of Variengien and Winsor (2023) and Ghandeharioun et al. (2024).
- **Implications for Multilingual Bias**: Bias might propagate through shared conceptual spaces—mitigating bias in English might automatically mitigate bias in other languages.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ First causal proof of the existence and utilization of language-agnostic concept representations; the disambiguation experiment design is extremely elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Validated across 7 models for generalizability, following a progressive flow of exploration, validation, disambiguation, and generation experiments.
- Writing Quality: ⭐⭐⭐⭐ Clearly defined concepts, rigorous experimental logic, and well-designed figures.
- Value: ⭐⭐⭐⭐⭐ Provides profound insights into the inner workings of multilingual LLMs, offering crucial guidance for cross-lingual transfer and bias research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] From Concepts to Components: Concept-Agnostic Attention Module Discovery in Transformers](../../ICLR2026/interpretability/from_concepts_to_components_concept-agnostic_attention_module_discovery_in_trans.md)
- [\[ACL 2025\] Reasoning Circuits in Language Models: A Mechanistic Interpretation of Syllogistic Inference](reasoning_circuits_in_language_models_a_mechanistic_interpretation_of_syllogisti.md)
- [\[ICCV 2025\] Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations](../../ICCV2025/interpretability/granular_concept_circuits_toward_a_fine-grained_circuit_discovery_for_concept_re.md)
- [\[ACL 2025\] An Empirical Study of Mechanistic Interpretability Approaches for Factual Recall](an_empirical_study_of_mechanistic_interpretability_approaches_for_factual_recall.md)
- [\[ACL 2025\] Mechanistic Interpretability of Emotion Inference in Large Language Models](mechanistic_interpretability_of_emotion_inference_in_large_language_models.md)

</div>

<!-- RELATED:END -->
