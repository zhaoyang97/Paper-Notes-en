---
title: >-
  [Paper Note] Verb Mirage: Unveiling and Assessing Verb Concept Hallucinations in Multimodal Large Language Models
description: >-
  [AAAI 2026][Multimodal VLM][verb hallucination] This paper presents the first systematic study of verb concept hallucinations in multimodal large language models (MLLMs), constructs a multi-dimensional benchmark…
tags:
  - "AAAI 2026"
  - "Multimodal VLM"
  - "verb hallucination"
  - "MLLM"
  - "hallucination evaluation"
  - "action understanding"
  - "fine-grained evaluation"
date: 2026-05-08
content_hash: 814fef455f3f8bbb
---

# Verb Mirage: Unveiling and Assessing Verb Concept Hallucinations in Multimodal Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2412.04939](https://arxiv.org/abs/2412.04939)  
**Code**: [github](https://github.com/davidwang200099/Verb_Mirage)  
**Area**: Multimodal VLM
**Keywords**: verb hallucination, MLLM, hallucination evaluation, action understanding, fine-grained evaluation

## TL;DR
This paper presents the first systematic study of verb concept hallucinations in multimodal large language models (MLLMs), constructs a multi-dimensional benchmark, demonstrates that existing hallucination mitigation methods are ineffective against verb hallucinations, and proposes a fine-tuning baseline enriched with verb knowledge that significantly alleviates verb hallucinations.

## Background & Motivation

MLLMs have achieved remarkable progress on tasks such as OCR, VQA, and image captioning, yet **hallucination** remains a central bottleneck constraining their reliability. Existing hallucination research and mitigation methods have **focused almost exclusively on object/noun concept hallucinations** — for example, POPE evaluates whether objects exist, and CHAIR assesses object hallucinations in generated captions.

However, **verb concepts** are critical for understanding human behavior: it is not sufficient to know what objects appear in an image; one must also understand what actions are taking place among them. Verb hallucination has long been neglected, primarily due to the following reasons:

**Dataset bias**: In commonly used MLLM pre-training corpora, nouns outnumber verbs by a factor of 4–10 (as shown in Figure 4(a)), causing models to understand nouns far better than verbs.

**Evaluation gap**: No benchmark specifically targeting verb hallucination exists.

**Intuitive misconception**: It is commonly assumed that resolving object hallucination would automatically resolve verb hallucination — a claim this paper refutes.

The paper's core starting point is that **verb hallucination and object hallucination are fundamentally distinct problems**. Methods designed to mitigate object hallucination are ineffective against verb hallucination and may even exacerbate it, revealing deep-seated deficiencies in the semantic understanding of MLLMs.

## Method

### Overall Architecture

The work consists of three components: (1) constructing a verb hallucination evaluation benchmark; (2) systematically probing and analyzing verb hallucination phenomena from multiple dimensions; and (3) proposing a fine-tuning baseline with enriched verb knowledge as a mitigation approach.

### Key Designs

#### 1. **Multi-Dimensional Verb Hallucination Benchmark Construction**

A first-of-its-kind verb hallucination benchmark is constructed based on the HICO and CharadesEgo datasets, requiring no additional manual annotation. The benchmark covers two question formats:

- **Yes/No (YN) questions**: e.g., "Is there a person holding a cup in the image?"
- **Multiple-choice (MC) questions**: given one correct verb and three distractor verbs, evaluated using circular evaluation.

The core design principle is to **vary the verb while keeping the object constant**, thereby isolating verb hallucination from object hallucination. For example, if an image depicts a person holding a cup, the benchmark asks both "Is someone holding a cup?" and "Is someone washing a cup?"

#### 2. **Multi-Angle Probing Design**

Verb hallucination is systematically probed across three dimensions:

**(a) Query condition probing**:
- **Question format**: MC vs. YN — models perform better on MC but still exhibit substantial hallucination.
- **Object association**: Comparing "Is someone holding a cup?" vs. "Is someone holding something?" reveals that MLLMs rely heavily on object references to understand verbs.

**(b) Image condition probing**:
- **Image quality**: Salt-and-pepper noise is added (affecting 75% of pixels); the resulting visual degradation has a **substantially greater** impact on verb understanding than on object understanding (as measured by significant differences in Cohen's Kappa agreement).
- **Viewpoint difference**: Using CharadesEgo to compare first-person (ego) and third-person (exo) perspectives, MLLMs show a marked decline in verb comprehension under the egocentric viewpoint.

**(c) Semantic condition probing**:
- **Rare vs. common verbs**: Models tend to reject existing rare verbs while accepting non-existent common verbs.
- **Content ambiguity**: Verb hallucinations are more severe in crowded, occluded, or person–object size-imbalanced scenes.

#### 3. **In-Depth Analysis of Model Behavior**

Using LLaVA V1.5 as a case study, verb hallucination is analyzed from two perspectives — visual-language interaction and token uncertainty:

- **Attention to key image regions**: Models exhibiting hallucination attend less to key regions, but the gap is modest — even correct attention does not guarantee correct verb semantic understanding.
- **Visual token attention**: Unlike object hallucination, attending more to visual tokens **does not** eliminate verb hallucination, which explains the failure of the OPERA method.
- **Token uncertainty**: Hallucinated answers are typically generated with low probability, and models tend to answer "Yes" with high confidence.
- **mAP vs. Accuracy analysis**: Although LLaVA V1.5 achieves low accuracy (52.16), its mAP (68.41) exceeds that of HICO fine-tuned CLIP (60.45), indicating that **one source of verb hallucination is token calibration error** rather than a complete absence of verb understanding.

### Loss & Training

The baseline mitigation method fine-tunes LLaVA V1.5 with LoRA using 60K instruction-tuning samples constructed from the Pangea dataset. Pangea organizes heterogeneous action datasets and establishes a mapping from action labels to 290 abstract verb nodes in VerbNet, covering 280 out of 290 nodes and encompassing a broad range of verb semantics.

## Key Experimental Results

### Main Results

**Verb hallucination performance of various models on YN and MC tasks**:

| Model | YN+obj acc | YN+obj prec | YN+obj recall | MC+obj acc | MC verb-only acc |
|-------|-----------|-------------|---------------|-----------|-----------------|
| Qwen2-VL-7B | 75.51 | 58.37 | 93.75 | 71.47 | 65.31 |
| MiniCPM-Llama3-V2.5 | 80.91 | 66.83 | 85.41 | 66.39 | 60.77 |
| LLaVA V1.5 | 52.16 | 40.99 | 97.35 | 57.37 | 51.00 |
| Molmo-7B-D | 59.16 | 44.91 | 96.63 | 60.64 | 56.78 |

**Key finding**: All models exhibit extremely high recall (85–97%) but very low precision (40–67%), indicating that models tend to answer "Yes" regardless of whether the verb is present.

**Comparison of mitigation methods** (baseline: LLaVA V1.5):

| Method | YN+obj acc | YN+obj F1 | MC+obj acc | MC verb-only acc |
|--------|-----------|----------|-----------|-----------------|
| LLaVA V1.5 | 52.16 | 57.69 | 57.37 | 51.00 |
| OPERA | 42.46 | 53.69 | 57.28 | 51.13 |
| VCD | 52.38 | 58.04 | 54.26 | 48.94 |
| Haloquest | 70.57 | 64.89 | 55.20 | 47.45 |
| **Ours (Pangea FT)** | **78.48** | **68.13** | **61.73** | **60.79** |

### Ablation Study

| Configuration | YN+obj acc | MC verb-only acc | Note |
|---------------|-----------|-----------------|------|
| OPERA | 42.46 | 51.13 | Penalizing summary token attention worsens performance |
| VCD | 52.38 | 48.94 | Language priors cannot be easily removed (KL=0 for 18.6K/20K samples) |
| Nullu | 51.99 | 53.17 | Model layers do not form reliable truth/hallucination distinction for verbs |
| REVERIE | 40.67 | 41.32 | Training set lacks sufficient verb knowledge |
| Ours | 78.48 | 60.79 | Fine-tuning with enriched verb knowledge is effective |

**Ablation on image quality impact**:

| Model | YN acc (no noise) | YN acc (with noise) | Error consistency (Cohen's κ) |
|-------|------------------|--------------------|-----------------------------|
| MiniCPM | 79.14 | 67.40 | 26.12 (poor) |
| Qwen-VL-Chat | 79.24 | 66.64 | 38.47 (fair) |
| LLaVA V1.5 | 59.16 | 51.29 | 73.85 (good but low baseline) |

### Key Findings

1. **Verb hallucination is pervasive and severe**: All SOTA MLLMs perform poorly on verb understanding, even those achieving high scores on object hallucination benchmarks (POPE).
2. **Verb understanding relies heavily on object references**: MC accuracy drops substantially when object references are removed.
3. **Existing object hallucination mitigation methods are ineffective for verb hallucinations**: OPERA, VCD, Nullu, and similar methods all fail.
4. **Models share similar biases**: Ensembling three models does not yield significant improvement.
5. **Verb hallucination stems from token calibration error** rather than a complete failure to understand verbs: mAP is high while accuracy is low.
6. **Visual degradation affects verb understanding far more than object understanding**.

## Highlights & Insights

1. **Pioneering problem formulation**: The paper is the first to study verb hallucination independently from object hallucination, filling an important gap in the field.
2. **Comprehensive evaluation dimensions**: Systematic analysis across query conditions, image conditions, semantic conditions, and model behavior, with rigorous experimental design.
3. **Valuable deep insight**: The finding that "more visual attention ≠ fewer verb hallucinations" challenges the core assumption underlying OPERA and related methods.
4. **The mAP vs. accuracy analysis** pinpoints miscalibration as the true source of hallucination, providing a clear direction for future research.
5. **Pangea fine-tuning experiments** demonstrate that enriching verb knowledge effectively mitigates hallucination without substantially degrading other capabilities.

## Limitations & Future Work

1. The proposed mitigation method is a baseline; its performance remains far from satisfactory (MC verb-only acc of only 60.79%).
2. Whether effective training-free methods for verb hallucination exist remains an open question.
3. Evaluation is limited to static image scenarios; verb understanding in video is not addressed.
4. Only open-source 7B-scale models are tested; larger-scale models are not thoroughly evaluated.
5. Fine-tuning data is limited to 60K samples; larger-scale and more diverse verb data may yield further improvements.

## Related Work & Insights

- **POPE** (Li et al., 2023): A pioneer in object hallucination evaluation; this paper extends the concept to verb concepts.
- **OPERA** (Huang et al., 2024): Mitigates hallucination by penalizing summary token attention; shown here to be ineffective for verbs.
- **VCD** (Leng et al., 2024): A contrastive decoding method; language priors are found to be deeply entrenched in verb understanding.
- **Pangea** (Li et al., 2024): Unifies heterogeneous action datasets and provides rich verb knowledge.
- Insight: Future work may incorporate verb hallucination evaluation into standard MLLM assessment pipelines; dedicated data and training strategies are required for verb understanding.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (First to define and systematically study verb hallucination)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Comprehensive evaluation across multiple dimensions, models, and conditions)
- Writing Quality: ⭐⭐⭐⭐ (Logically clear, though somewhat lengthy)
- Value: ⭐⭐⭐⭐⭐ (Reveals an important blind spot in MLLMs with significant implications for the community)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Concept-RuleNet: Grounded Multi-Agent Neurosymbolic Reasoning in Vision Language Models](concept-rulenet_grounded_multi-agent_neurosymbolic_reasoning.md)
- [\[AAAI 2026\] SDEval: Safety Dynamic Evaluation for Multimodal Large Language Models](sdeval_safety_dynamic_evaluation_for_multimodal_large_language_models.md)
- [\[NeurIPS 2025\] Seeing is Believing? Mitigating OCR Hallucinations in Multimodal Large Language Models](../../NeurIPS2025/multimodal_vlm/seeing_is_believing_mitigating_ocr_hallucinations_in_multimodal_large_language_m.md)
- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](../../ACL2026/multimodal_vlm/mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)
- [\[AAAI 2026\] VP-Bench: A Comprehensive Benchmark for Visual Prompting in Multimodal Large Language Models](vp-bench_a_comprehensive_benchmark_for_visual_prompting_in_m.md)

</div>

<!-- RELATED:END -->
