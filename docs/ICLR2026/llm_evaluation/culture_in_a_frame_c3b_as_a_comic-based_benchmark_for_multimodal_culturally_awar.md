---
title: >-
  [Paper Note] Culture In a Frame: C$^3$B as a Comic-Based Benchmark for Multimodal Culturally Awareness
description: >-
  [ICLR 2026][LLM Evaluation][MLLM] C3B (Comics Cross-Cultural Benchmark) utilizes 2,220 comic panels and 18,789 QA pairs to establish a task chain of three progressive difficulty levels: "Identifying Cultural Objects → Judging Cultural Conflicts → Cross-lingual Cultural Content Generation." It specifically evaluates the cultural awareness of Multimodal
tags:
  - ICLR 2026
  - LLM Evaluation
  - MLLM
date: 2026-05-08
content_hash: b3fe4ae907eeabda
---
# Culture In a Frame: C$^3$B as a Comic-Based Benchmark for Multimodal Culturally Awareness

**Conference**: ICLR2026  
**OpenReview**: [https://openreview.net/forum?id=jvPdTOSTVl](https://openreview.net/forum?id=jvPdTOSTVl)  
**Code**: https://c3b-benchmark.github.io/  
**Area**: Multimodal VLM / Evaluation Benchmark  
**Keywords**: Cultural Awareness, Comics, Multimodal Evaluation, Multilingual, MLLM

## TL;DR
C3B (Comics Cross-Cultural Benchmark) utilizes 2,220 comic panels and 18,789 QA pairs to establish a task chain of three progressive difficulty levels: "Identifying Cultural Objects → Judging Cultural Conflicts → Cross-lingual Cultural Content Generation." It specifically evaluates the cultural awareness of Multimodal Large Language Models (MLLMs). Evaluations across 11 open-source MLLMs demonstrate a significant performance gap compared to human levels.

## Background & Motivation

**Background**: Cultural awareness is becoming a core capability for MLLMs. It is widely observed that existing models perform well within Western cultural contexts but significantly underperform in non-Western contexts. To systematically measure this capability, the community has developed several benchmarks (CVQA, CulturalVQA, GIMMICK, ALM-bench, etc.).

**Limitations of Prior Work**: The authors identify three common shortcomings in existing benchmarks. First, they **rely almost entirely on real-world images**, where a single photo typically carries only one culture, resulting in low cultural density and tasks that are too simple for models. Second, **most adopt a simple "one-image-one-question" format**, failing to evaluate the same image from multiple cross-dimensions. Third, they are **mostly monolingual**, overlooking the complexity of "untranslated concepts across languages" where language itself serves as a cultural carrier.

**Key Challenge**: To make cultural awareness evaluation difficult and comprehensive, multiple cultures must be embedded in a single image, multiple task layers must be designed for the same image, and multilingualism must be integrated. However, real-world photography naturally struggles to depict "multiple cultures per frame," which is the fundamental reason existing benchmarks lack difficulty.

**Goal**: To create a cultural awareness benchmark that simultaneously satisfies the requirements of **multi-culturalism (multiple cultures per image), multi-tasking (multiple questions per sample), multilingualism, and progressive difficulty**, while providing baselines for the first batch of open-source MLLMs.

**Key Insight**: The authors shift the medium from real photography to **comics**. Comics depict fictional scenes and are not constrained by the "one-location-one-culture" reality. Thus, multiple cultures can be concentrated into a single frame, naturally creating complex contexts with high cultural density.

**Core Idea**: Utilizing "comics as the canvas + a three-level progressive task chain + multilingual generation" to maximize the difficulty and coverage of cultural awareness evaluation, exposing current MLLM weaknesses in underrepresented cultures and cultural conflict understanding.

## Method

### Overall Architecture

C3B produces a structured evaluation dataset: **2,220 comic panels + 18,789 QA pairs, covering 77 cultures**. The benchmark is organized around a progressive task chain:

1.  **Level 1 · Culture-aware Objects Extraction (Extraction@Culture)**: Tests basic visual recognition and fundamental cultural understanding. Includes Q1 (Identifying the background culture of the comic page, multi-choice) and Q2 (Selecting the option that precisely contains all cultural representative objects in the frame).
2.  **Level 2 · Cultural-conflict Objects Detection (Conflict@Culture)**: Tests cultural conflict understanding. Includes Q3 (Judging if a cultural conflict exists—defined as two distinct cultures appearing in one image) and Q4 (If a conflict exists, identifying which objects in Q2 conflict with the background culture in Q1).
3.  **Level 3 · Culturally-aligned Content Generation (Generation@Culture)**: Tests multilingual generation in a given multimodal cultural context, specifically through machine translation of Japanese comics into English, German, Russian, Spanish, and Thai (JA-EN/DE/RU/ES/TH).

The data comes from "two sets of images": Extraction and Conflict use 1,023 **automatically generated** comic panels (via a Doubao text-to-image pipeline), while Generation uses 1,197 **manually selected** authentic Japanese comic pages from Manga109. Annotation follows different pipelines per task. Evaluation uses Accuracy for recognition/conflict tasks, a composite CACC metric for Q4, and BLEU/COMET/BLEURT for translation tasks.

### Key Designs

**1. Comics as the Medium: Compressing multiple cultures into one frame to increase density**

This is the fundamental design distinguishing C3B from prior works, addressing the limitation that real photos are "too simple with only one culture." While real photos are tied to physical locations, a comic can depict a "German tourist in a Hawaiian hula skirt conflicting with a Maori warrior in a New Zealand cultural village" in one frame. To quantify this, the authors define three cultural diversity metrics:

$$\text{CDPI}(D) = \frac{1}{|D|}\sum_{i=1}^{n}\text{CultureInImage}(I_i)$$

$$\text{CBI}(D) = \text{CDPI}(D)\times N_{\text{cultures}},\qquad \text{CAD}(D)=\text{CDPI}(D)\times\log_2(N_{\text{cultures}}+1)$$

where CDPI is the average cultures per image, CBI scales density by the total number of cultures, and CAD applies a logarithmic scale to the total culture count to penalize artificially inflated culture lists. Results show C3B achieves a CDPI of **2.28** (CVQA/CulturalVQA/GIMMICK are all 1.00) and a CAD of 14.29, far exceeding GIMMICK’s 7.18.

**2. Three-Level Progressive Task Chain: From "Seeing" to "Understanding Conflict" to "Generation"**

Unlike benchmarks with one question per image, C3B designs **logically progressive** tasks for the same set of images. The first level requires "seeing and categorizing" cultural objects. The second level requires **judging if the juxtaposition of cultures constitutes a conflict** and precisely identifying conflict objects (requiring cross-cultural relational reasoning). The third level requires **generating in the target language** translation consistent with the frame's cultural context. Q4 specifically relies on answers from Q1 and Q2—only after correctly identifying the "background culture" and "cultural objects" can the model correctly point out the "conflicting object," thereby exposing weaknesses in long-chain reasoning.

**3. Multi-agent Automated Construction Pipeline: Quality control for generation, annotation, and translation**

To scale data production while maintaining quality, the authors built semi-automated pipelines. **Image Generation** uses the Doubao API: a model generates prompts based on "cultural conflict scene" instructions, followed by manual filtering and generation. **Extraction/Conflict Annotation**: A full list of cultures and artifacts is manually compiled. Q1 distractors are randomly selected, and Q2 distractors are created by perturbing the correct object list. Conflict annotation uses Deepseek-V3 for detection, followed by manual verification. **Generation Annotation** uses a Translator + Reviewer dual-agent system: the Reviewer provides feedback on consistency and cultural errors to the Translator for iterative refinement before manual finalization.

**4. CACC Composite Metric: Reflecting the reasoning chain in dependent tasks**

Since Q4 is built upon Q1 and Q2, using raw accuracy for Q4 hides instances where errors in earlier steps caused the failure. The authors designed a composite accuracy:

$$\text{CACC}(Q4) = a\cdot\text{ACC}(Q1) + b\cdot\text{ACC}(Q2) + c\cdot\text{ACC}(Q4)$$

Weighting $a=0.3,\ b=0.3,\ c=0.4$ ensures a model scores high only if it identifies the background, identifies the objects, and correctly identifies the conflict. This metric reflects comprehensive performance across the cultural reasoning chain.

## Key Experimental Results

### Main Results

11 open-source MLLMs were evaluated (SPHINX, Monkey, MiniGPT-v2, mPLUG-Owl3, LLaVA series, InternLM-XC2.5, Llama3.2, Qwen2.5-VL, InternVL2, etc.).

Recognition and Conflict Results (Representative models, %):

| Model | Q1 (Extraction) | Q2 (Extraction) | Q3 (Conflict) | Q4 ACC | Q4 CACC |
|------|------|------|------|------|------|
| LLaVA1.5-7B | 32.5 | 2.93 | 56.3 | 0.00 | 10.6 |
| LLaVA-NeXT | 16.5 | 39.8 | 0.88 | 0.00 | 16.9 |
| InternLM-XC2.5 | 46.0 | 50.9 | 68.5 | 1.94 | 29.8 |
| Llama3.2 | 46.0 | 59.0 | 44.9 | 2.76 | 32.6 |
| **Qwen2.5-VL** | **53.7** | **55.9** | 63.1 | 3.20 | **34.2** |
| InternVL2 | 46.0 | 50.9 | 68.5 | 0.01 | 29.1 |

Qwen2.5-VL is the strongest overall. However, **raw ACC for Q4 is extremely low for all models (mostly <4%, with LLaVA at 0.00)**, indicating that "identifying cultural conflict objects" is a near-total blind spot for current MLLMs.

Multilingual Generation (Generation@Culture, BLEU):

| Model | JA-EN | JA-DE | JA-RU | JA-ES | JA-TH |
|------|------|------|------|------|------|
| MiniGPT-v2 | 0.03 | 0.00 | 0.00 | 0.00 | 0.00 |
| **Qwen2.5-VL** | **13.2** | **12.0** | 8.74 | **14.5** | **9.72** |

All models performed **worst on JA-TH (Thai) and best on JA-EN (English)**, exposing performance drops in low-resource cultural context translation.

### Ablation Study

Analyzing Q4's dependency on Q1/Q2 answers (Average CACC of 11 models):

| Configuration | CACC | Note |
|------|------|------|
| Base Prompt | 19.231 | No Q1/Q2 answers provided |
| + Q1 Answer | 19.234 | Only Q1 answer injected (+0.003) |
| + Q1&Q2 Answer | 19.764 | Both injected (+0.533) |

While Q1/Q2 are moderately correlated with Q4 (0.56 and 0.51), injecting correct answers into the prompt yielded only marginal gains. This suggests that the correlation stems from model consistency across related questions rather than an ability to leverage previous logical steps for reasoning.

### Key Findings

- **Cultural conflict understanding is the biggest blind spot**: Models failed Q4 almost entirely, and providing ground truth answers did not resolve the issue, suggesting the bottleneck is reasoning, not just missing information.
- **Vast performance gaps in underrepresented cultures**: Representative cultures (Cambodia, Japan) are identified reliably, while error rates are significantly higher for others (Finland, Somalia).
- **Huge Human-AI Gap**: Humans achieved 100% accuracy on Q3 across all difficulty levels. Overall CACC for humans (Easy 74.7 / Mid 59.4 / Hard 50.0) far exceeds models (Easy 23.8 / Mid 18.6 / Hard 15.9).
- **Typical Failure Modes**: LLaVA-NeXT tends to describe images without answering ("Turn-a-deaf-ear"), LLaVA1.5 consistently picks "A" ("Take-a-shot-in-the-dark"), and general "stubbornness" in ignoring instructions.

## Highlights & Insights
- **"Changing the Medium" is the most clever strategy**: Instead of scaling data volume, the authors addressed the bottleneck of "low cultural density in real photos." Using comics to double cultural density naturally increases the difficulty—a transferable idea for benchmark design.
- **Progressive task chains + Dependent metrics** provide "diagnostic" capabilities: Using Q4's dependency on Q1/Q2 and CACC allows researchers to pinpoint whether a model fails at "recognition" or "conflict reasoning."
- **The "No Gain from Answers" ablation is valuable**: It distinguishes between "correlation" and "causal utility," reminding researchers that high consistency does not imply a model is actually utilizing prior information.

## Limitations & Future Work
- **Authenticity of synthetic comics**: Extraction/Conflict images are model-generated. While prompts were filtered, the "cultural authenticity" of AI-generated scenes might still harbor stereotypes or inaccuracies.
- **Narrow definition of conflict**: Juxtaposing two cultures as a "conflict" (co-occurrence conflict) is a simplification and might mislabel normal cross-cultural coexistence.
- **Generation limited to translation**: Using JA→X translation as a proxy for cultural content generation is limited in scope. BLEU is also a weak measure for cultural nuance.
- **Closed-source models missing**: The study lacks baselines for GPT-4o or Gemini.

## Related Work & Insights
- **vs CVQA / CulturalVQA**: These use real photos with single-culture images and single-task formats. C3B increases cultural density (CDPI from 1.0 to 2.28) and adds multilingual generation.
- **vs GIMMICK / ALM-bench**: Despite GIMMICK's 6 tasks and ALM-bench's 100+ languages, they still rely on real-world photos. C3B exceeds them in cultural density (CAD 14.29 vs 7.18).
- **vs Manga109 / CoMix**: Existing comic datasets focus on low-level visual tasks (OCR, speaker ID). C3B is the first to systematically apply comics to "cultural awareness" evaluation.

## Rating
- Novelty: ⭐⭐⭐⭐ Using comics to increase cultural density and the progressive CACC metric are highly original.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive model evaluation and human comparison, though closed-source models are absent.
- Writing Quality: ⭐⭐⭐⭐ Clear presentation of construction and metrics.
- Value: ⭐⭐⭐⭐ Effectively identifies blind spots in MLLM cultural reasoning.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

</div>

<!-- RELATED:END -->

## Related Papers

- [\[ACL 2025\] Access Denied Inc: The First Benchmark Environment for Sensitivity Awareness](../../ACL2025/llm_evaluation/access_denied_inc_the_first_benchmark_environment_for_sensitivity_awareness.md)
- [\[ICLR 2026\] Culture in Action: Evaluating Text-to-Image Models through Social Activities](culture_in_action_evaluating_text-to-image_models_through_social_activities.md)
- [\[ACL 2026\] Minos: A Multimodal Evaluation Model for Bidirectional Generation Between Image and Text](../../ACL2026/llm_evaluation/minos_a_multimodal_evaluation_model_for_bidirectional_generation_between_image_a.md)
- [\[ACL 2025\] SANSKRITI: A Comprehensive Benchmark for Evaluating Language Models' Knowledge of Indian Culture](../../ACL2025/llm_evaluation/sanskriti_a_comprehensive_benchmark_for_evaluating_language_models_knowledge_of_.md)
- [\[ACL 2026\] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge](../../ACL2026/llm_evaluation/multi-task_reinforcement_learning_for_enhanced_multimodal_llm-as-a-judge.md)

</div>

<!-- RELATED:END -->
