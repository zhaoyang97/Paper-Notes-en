---
title: >-
  [Paper Note] Lost in Translation? A Comparative Study on the Cross-Lingual Transfer of Composite Harms
description: >-
  [AAAI 2026][LLM Safety][Multilingual Evaluation] This paper introduces the CompositeHarm benchmark, which systematically investigates the vulnerability of LLM safety alignment in cross-lingual settings by translating adv…
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "Multilingual Evaluation"
  - "Adversarial Attacks"
  - "Cross-Lingual Transfer"
  - "Indic Languages"
date: 2026-05-08
content_hash: 1eec0eab89b4532e
---

# Lost in Translation? A Comparative Study on the Cross-Lingual Transfer of Composite Harms

**Conference**: AAAI 2026
**arXiv**: [2602.07963](https://arxiv.org/abs/2602.07963)  
**Code**: N/A (dataset available upon request from the corresponding author)  
**Area**: AI Safety
**Keywords**: LLM Safety, Multilingual Evaluation, Adversarial Attacks, Cross-Lingual Transfer, Indic Languages

## TL;DR

This paper introduces the CompositeHarm benchmark, which systematically investigates the vulnerability of LLM safety alignment in cross-lingual settings by translating adversarial syntactic attacks (AttaQ) and contextualized harms (MMSafetyBench) into five Indic languages. The study finds that adversarial syntactic attacks achieve dramatically higher attack success rates in Indic languages.

## Background & Motivation

**English-centric LLM safety evaluation**: The vast majority of safety evaluations are grounded in English, yet LLMs are used daily by speakers of dozens of languages. The assumption that refusal behaviors or safety guardrails fine-tuned on English data remain effective across other languages is untenable.

**Translation as a probing mechanism**: Translation is the most practical bridge for testing multilingual safety. While imperfect, it remains the clearest approach for studying how harmful intent transfers across languages—some harms transfer almost entirely, while others are distorted or lost.

**Distinction between two harm types**:
   - **Adversarial Syntax** (from AttaQ): Instructions that bypass safety filters through obfuscation or encoding, relying on syntactic manipulation to challenge model parsing and alignment.
   - **Contextual Harms** (from MMSafetyBench): Policy-relevant harms grounded in real-world scenarios, such as hate speech, misinformation, and ethical dilemmas.

**Limitations of prior work**:
   - Most prior work is restricted to a single language family or translated benchmarks.
   - Systematic study of how different harm types behave differently under cross-lingual transfer is lacking.
   - Comparative analysis between multimodal attacks and cross-lingual textual attacks remains unexplored.

## Method

### Overall Architecture

CompositeHarm is a translation-based safety evaluation benchmark designed to measure the failure points of safety alignment under syntactic and semantic variation through controlled experimental conditions.

### Key Designs

#### 1. **Dataset Construction**

- **Source datasets**:
    - AttaQ: 140 structured adversarial attack prompts exploiting syntactic manipulation (e.g., role-playing, encoded instructions) to bypass safety filters.
    - MMSafetyBench: 140 contextualized harm prompts spanning categories including violence, discrimination, explicit content, fraud, and misinformation.
- **Translation process**:
    - The NLLB (No Language Left Behind) model is used to translate prompts into five Indic languages: Hindi, Assamese, Marathi, Kannada, and Gujarati.
    - All translations are manually verified and corrected by bilingual annotators to ensure semantic accuracy and cultural appropriateness, with particular attention to preserving the syntactic structure of adversarial prompts.

The final dataset contains 1,680 prompts (280 × 6 languages), balanced across languages and sources.

#### 2. **Evaluation Protocol (LLM-as-a-Judge)**

- Three LLMs representing different architectures and scales are evaluated:
    - **GPT-OSS 20B**: An open-source mid-scale variant with a conservative defense posture.
    - **LLaMA-3-8B-Instruct**: A lightweight instruction-tuned model suitable for edge deployment.
    - **Qwen3-32B**: A multilingual model from Alibaba with extensive training on Asian languages.

- LLaMA-3.3-70B is used as the judge model with a standardized evaluation pipeline:
  1. Each translated prompt is fed to the target model to generate a response (temperature=0).
  2. The judge model answers two binary questions: "Is it refused?" and "Is it harmful?"
  3. Results are aggregated into Refusal Rate (RR) and Attack Success Rate (ASR).

- A total of 5,040 responses are processed (1,680 × 3 models).

#### 3. **Lightweight Inference Strategy**

Inspired by edge AI design principles, compact generative models are used for preliminary evaluation, with large-scale evaluators invoked only for selective verification, reducing redundant API calls and GPU usage.

### Loss & Training

This paper presents an evaluation benchmark; no training loss is designed. The core metrics are:
- **Refusal Rate (RR)**: The proportion of harmful prompts correctly refused by the model.
- **Attack Success Rate (ASR)**: The proportion of prompts that successfully elicit harmful responses.

## Key Experimental Results

### Main Results

Average cross-lingual safety performance (averaged across three models):

| Language | Avg. Refusal Rate (%) | Avg. Attack Success Rate (%) |
|------|-------------|-----------------|
| English | 39.3 | 4.7 |
| Hindi | 43.8 | 6.1 |
| Marathi | 40.7 | 10.4 |
| Assamese | 29.9 | 10.2 |
| Gujarati | 22.0 | 22.0 |
| Kannada | 21.4 | 21.1 |

Cross-language-family comparison (vs. European languages):

| Language | Avg. Refusal Rate (%) | Avg. Attack Success Rate (%) |
|------|-------------|-----------------|
| English (AttaQ) | 82.7 | 4.7 |
| French | 83.0 | 4.0 |
| Spanish | 86.3 | 1.3 |
| German | 80.7 | 3.7 |
| Hindi | 43.8 | 6.1 |
| Gujarati | 22.0 | 22.0 |
| Kannada | 21.4 | 21.1 |

### Ablation Study

Analysis of the insufficiency of binary metrics (the "gray zone"):

| Language | RR+ASR Total (%) | Unclassified (%) | Note |
|------|-------------|-------------|------|
| English | 44.0 | 56.0 | Baseline; most responses are normal |
| Hindi | 49.9 | 50.1 | Half of interactions unclassifiable |
| Gujarati | 44.0 | 56.0 | Large proportion of evasive responses |
| Kannada | 42.5 | 57.5 | Most severe gray zone |

### Key Findings

1. **Syntactic attacks are more dangerous than semantic attacks**: The ASR of adversarial syntactic prompts surges dramatically in Indic languages (LLaMA-3-8B exceeds 45% for Gujarati and Kannada), while contextualized harms transfer more moderately.
2. **Linguistic distance effect**: Greater linguistic distance from English correlates with more severe degradation of safety alignment. European languages maintain RR above 80% with ASR below 5%, whereas Indic languages exhibit RR drops of over half.
3. **Pervasive "soft failures"**: RR and ASR fail to account for all responses; a large proportion falls into a gray zone of evasive answers and guardrail interceptions, indicating that internal safety mechanisms have broken down.
4. **Complementary weaknesses across three models**:
    - Qwen3-32B: Semantic vulnerability—fluent but safety alignment relies on superficial lexical cues.
    - LLaMA-3-8B: Ethical laxity—inconsistent moral judgment.
    - GPT-OSS 20B: Over-defensiveness—rules fail to generalize to new languages.

## Highlights & Insights

- **Necessity of a composite benchmark**: This is the first work to systematically distinguish the differential transfer behavior of adversarial syntactic and contextualized harms across languages, revealing the limitations of relying on a single harm type.
- **"Gray zone" analysis**: The prevalence of evasive responses and guardrail interceptions beyond RR and ASR is quantified for the first time, providing a more complete picture of safety evaluation.
- **Comparison with multimodal attacks**: Contrasted with the work of Derner & Batistič (2025), the findings suggest that LLM safety is vulnerable along both the linguistic generalization and modality generalization axes.
- **Edge deployment risk warning**: The lightest models—most suitable for edge deployment—are precisely the most vulnerable, indicating a serious trade-off between computational efficiency and multilingual safety.

## Limitations & Future Work

- Coverage is limited to five Indic languages, excluding Sino-Tibetan, Semitic, and other language families.
- The LLM-as-a-judge evaluation paradigm is adopted without human evaluation for validation.
- Although translations are manually verified, the translation process itself may introduce semantic drift.
- Targeted safety alignment improvement strategies are not explored in depth.
- The dataset size is relatively small (280 prompts per language), which may limit statistical significance.

## Related Work & Insights

- **AttaQ (Kour et al., 2023)** and **MMSafetyBench (Li et al., 2023)** serve as two complementary harm sources.
- The performance of the **NLLB** translation model on low-resource language translation merits attention.
- The multimodal red-teaming work of **Derner & Batistič (2025)** is complementary to this paper: cross-lingual textual attacks and multimodal attacks jointly reveal the systemic vulnerability of LLM safety.
- Implication: Future safety benchmarks should construct composite evaluations along both the harm type dimension (adversarial syntactic vs. semantic) and the modality dimension (text vs. image).

## Rating

- Novelty: ⭐⭐⭐⭐ — First systematic distinction of the differential cross-lingual transfer of two harm categories.
- Experimental Thoroughness: ⭐⭐⭐ — Three models across six languages, but dataset size is small and human evaluation is absent.
- Writing Quality: ⭐⭐⭐⭐ — Analysis is in-depth, though some discussions are slightly verbose.
- Value: ⭐⭐⭐⭐ — Provides an important methodological and empirical foundation for multilingual LLM safety research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](cross-modal_unlearning_via_influential_neuron_path_editing_i.md)
- [\[AAAI 2026\] Multi-Faceted Attack: Exposing Cross-Model Vulnerabilities in Defense-Equipped Vision-Language Models](multi-faceted_attack_exposing_cross-model_vulnerabilities_in_defense-equipped_vi.md)
- [\[ACL 2026\] Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models](../../ACL2026/llm_safety/lost_in_diffusion_uncovering_hallucination_patterns_and_failure_modes_in_diffusi.md)
- [\[ICLR 2026\] Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models](../../ICLR2026/llm_safety/reasoning_or_retrieval_a_study_of_answer_attribution_on_large_reasoning_models.md)
- [\[ACL 2026\] How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study](../../ACL2026/llm_safety/how_should_we_enhance_the_safety_of_large_reasoning_models_an_empirical_study.md)

</div>

<!-- RELATED:END -->
