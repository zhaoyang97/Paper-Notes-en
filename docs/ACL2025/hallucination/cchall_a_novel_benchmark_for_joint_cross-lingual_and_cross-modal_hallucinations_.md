---
title: >-
  [Paper Note] CCHall: A Novel Benchmark for Joint Cross-Lingual and Cross-Modal Hallucinations Detection in Large Language Models
description: >-
  [ACL2025][Hallucination Detection][cross-lingual] This paper proposes the first **joint cross-lingual and cross-modal** hallucination detection benchmark, CCHall, covering 9 languages and 4 types of multimodal datasets. It systematically evaluates the hallucination performance of 6 mainstream MLLMs in joint scenarios, revealing that the F1 score of current models in this joint scenario is 10.9% lower than that of cross-modal alone, and 3.4% lower than that of cross-lingual al…
tags:
  - "ACL2025"
  - "Hallucination Detection"
  - "cross-lingual"
  - "cross-modal"
  - "benchmark"
  - "multimodal large models"
date: 2026-05-08
content_hash: 4f3b7d14d8a38ada
---

# CCHall: A Novel Benchmark for Joint Cross-Lingual and Cross-Modal Hallucinations Detection in Large Language Models

**Conference**: ACL2025  
**arXiv**: [2505.19108](https://arxiv.org/abs/2505.19108)  
**Code**: [GitHub](https://github.com/BRZ911/CCHall)  
**Area**: Hallucination Detection  
**Keywords**: Hallucination detection, cross-lingual, cross-modal, benchmark, multimodal large models

## TL;DR

This paper proposes the first **joint cross-lingual and cross-modal** hallucination detection benchmark, CCHall, covering 9 languages and 4 types of multimodal datasets. It systematically evaluates the hallucination performance of 6 mainstream MLLMs in joint scenarios, revealing that the F1 score of current models in this joint scenario is 10.9% lower than that of cross-modal alone, and 3.4% lower than that of cross-lingual alone. Additionally, two mitigation paths are proposed: multilingual prompting and external tool assistance.

## Background & Motivation

**Hallucination is a core obstacle for LLM deployment**: Large language models produce hallucinations in high-risk applications such as medical diagnosis, image captioning, and speech-to-text, severely hindering large-scale deployment.

**Cross-lingual hallucination has been studied but remains isolated**: Benchmarks such as mFACT, HalOmi, and MM-Eval only evaluate translation or summarization hallucinations in multilingual scenarios, without involving the visual modality.

**Cross-modal hallucination research is also isolated**: CHAIR, POPE, MHaluBench, HallusionBench, etc. only evaluate vision-language alignment in an English monolingual environment, ignoring the extra challenges brought by linguistic disparities.

**Joint scenarios are closer to the real world**: Practical applications often require simultaneously processing multilingual translation and multimodal alignment (e.g., international medical imaging reports), but currently, no benchmark covers this joint scenario.

**Joint scenarios are more challenging**: The superposition of linguistic and modal disparities amplifies the risk of hallucination. Models must align both images with text and multilingual queries, which is far more difficult than a single scenario.

**Lack of systematic evaluation and mitigation strategy analysis**: Existing works do not systematically compare the effectiveness of mitigation strategies such as CoT, SRO, VDGD, and HalluciMAD in joint cross-lingual and cross-modal scenarios.

## Method

### Overall Architecture: Four-stage Benchmark Construction Process

- **Function**: Construct a comprehensive detection benchmark that simultaneously covers both cross-lingual and cross-modal hallucinations.
- **Design Motivation**: Fill the gap in joint cross-lingual $\times$ cross-modal hallucination evaluation, providing MLLMs with a more realistic evaluation environment.
- **Implementation**: Executed sequentially in four stages: (1) Selection of source multimodal datasets; (2) Construction of cross-modal hallucination data; (3) Construction of cross-lingual hallucination data; (4) Assembly of the joint dataset.

### Key Designs 1: Cross-Modal Hallucination Data Construction

- **Function**: Inject semantically similar but non-existent entities into image captions to construct deceptive hallucination samples.
- **Design Motivation**: Simple, unrelated false entities are easy to detect, whereas semantically similar substitutions truly test the fine-grained visual reasoning capabilities of models.
- **Implementation**: Select object existence questions (VQA) from GQA and AMBER, and select image captions (IC) from XM3600 and xFlickr&Co, limiting each object to appear at most twice to reduce redundancy. Then, use Gemini-1.5-Pro to compare ground-truth answers with the images, and embed nouns that are semantically close but non-existent in the images to generate natural hallucinations. Randomly sample 900 instances from each subset, resulting in a total of 3,600 instances.

### Key Designs 2: Cross-Lingual Hallucination Data Construction

- **Function**: Translate English data into 9 languages with different resource levels and ensure quality through manual verification.
- **Design Motivation**: Languages with different resource levels show significant disparities in translation quality and model understanding capabilities. Comprehensive coverage is required to reveal the vulnerabilities of models during language transfer.
- **Implementation**: Group languages into high, medium, and low resources, selecting the three languages with the lowest translation error rates for each group: high-resource (fr/es/pt), medium-resource (cs/nl/sv), and low-resource (hr/cy/sw). Translate using Google Translate and shuffle the order to eliminate order bias. Pair a random target language with English as the anchor language. Finally, organize manual reviews to check whether the hallucination data meets the criteria and whether the translation accurately preserves the original meaning.

### Key Designs 3: Definition of Four Hallucination Combinations

- **Function**: Define four types of samples: no hallucination, cross-modal hallucination only, cross-lingual hallucination only, and joint cross-lingual $\times$ cross-modal hallucination.
- **Design Motivation**: Fine-grained categorization allows the evaluation to decouple the impact of different factors, facilitating a comparison of difficulty between joint and single scenarios.
- **Implementation**: Each sample contains an image, a question, and responses in two languages, labeled as a four-class classification task based on the presence and type of hallucination. Models are required to detect the hallucination types across four subsets: AMBER, GQA, xFlickr&Co, and XM3600.

## Key Experimental Results

### Table 1: Main Results — Acc/Macro-F1 (%) of Six MLLMs on CCHall

| Model | Method | AMBER Acc | GQA Acc | xFlickr Acc | XM3600 Acc | AVG Acc | AVG F1 |
|------|------|-----------|---------|-------------|------------|---------|--------|
| InternVL2-8B | Direct | 29.1 | 29.9 | 38.3 | 38.8 | 34.0 | 42.9 |
| Llama-3.2-11B | CoT | 32.0 | 34.3 | 43.6 | 46.4 | 39.1 | 46.8 |
| Qwen2-VL-7B | CoT | 38.6 | 33.9 | 48.3 | 48.4 | 42.3 | 46.7 |
| Pixtral-12B | HalluciMAD | 46.3 | 45.2 | 57.1 | 58.7 | 51.8 | 56.4 |
| Gemini-1.5-Flash | HalluciMAD | 52.2 | 59.0 | 61.6 | 63.7 | 59.1 | 61.0 |
| **GPT-4o** | **HalluciMAD** | **70.9** | **68.6** | **84.1** | **86.4** | **77.5** | **78.8** |

**Key Findings**:
- CCHall is highly challenging; the weakest model (InternVL2-8B Direct) achieves only 34.0% accuracy, while even the strongest combination (GPT-4o + HalluciMAD) only reaches 77.5%.
- Closed-source GPT-4o and Gemini-1.5-Flash significantly outperform open-source models. Among open-source models, Qwen2-VL-7B outperforms the larger Llama-3.2-11B, indicating that training strategies are more important than parameter size.
- Basic strategies (CoT/SRO) are more suitable for smaller models (<12B), whereas advanced strategies (VDGD/HalluciMAD) are more effective on strong models.

### Table 2: Analytical Experiments — Effects of Resource Levels, Resolution, and Response Length

| Analytical Dimension | Key Findings |
|----------|----------|
| Language Resource Level | High-resource languages achieve the highest detection accuracy, while low-resource languages (hr/cy/sw) show a significant drop. |
| Image Resolution | High Resolution > Low Resolution > No Image; visual information is crucial for reducing hallucinations. |
| Model Scale | InternVL: 8B > 4B > 2B; larger parameters yield better performance. |
| Response Length | Hallucination rates scale rapidly once exceeding 120 words; long outputs are less reliable. |
| Multilingual Prompting | On Gemini-1.5-Flash, En+SL improves by 4.8% on GQA compared to English-only (En). |
| External Tool (UniHD) | Improves performance by an additional average of 2.7% over HalluciMAD, proving the effectiveness of external verification. |

## Highlights & Insights

1. **Pioneering Joint Evaluation Dimension**: The first benchmark to cover both cross-lingual and cross-modal hallucinations simultaneously, filling an important gap.
2. **Meticulous Data Construction Process**: Using Gemini-1.5-Pro to generate semantically close hallucinated entities is more challenging than simple random substitution. The coverage of three-tier language resources ensures a comprehensive evaluation.
3. **Rich Analytical Dimensions**: Aside from evaluating model performance, it systematically explores various factors such as language resources, image resolution, response length, multilingual prompting, and external tools, providing practical guidelines for mitigating hallucinations.

## Limitations & Future Work

1. **Limited to Text and Image Modalities**: Modalities like audio/speech are not covered. As multimodal models expand to more modalities, the benchmark needs to be upgraded synchronously.
2. **Model-Generated Hallucination Data**: Although manually verified, hallucination samples generated by Gemini-1.5-Pro may still contain residual errors or distributional biases.
3. **Translation Quality Reliant on Google Translate**: The translation quality of low-resource languages is inherently poor, which may introduce non-hallucination translation noise.
4. **Simplistic Four-Class Setup**: Real-world hallucinations may lie on a continuous spectrum rather than discrete classes. Future work can introduce fine-grained hallucination severity annotations.
5. **Lack of Automated Hallucination Attribution**: While knowing a hallucination exists, it is unclear which part generates it. Future work could incorporate span-level annotations.

## Related Work & Insights

| Dimension | CCHall | HallusionBench (Guan et al. 2024) | POPE (Li et al. 2023) |
|------|--------|----------------------------------|----------------------|
| Cross-Lingual | ✔ 9 languages | ✘ English-only | ✘ English-only |
| Cross-Modal | ✔ VQA+IC | ✔ VQA | ✔ Object Existence |
| Joint Scenarios | ✔ | ✘ | ✘ |
| Data Scale | Large (3600 × Multilingual) | 346 images + 1129 questions | Relatively Large |
| Hallucination Type | 4 combinations | Visual reasoning | Binary classification |

- **vs MM-Eval (Son et al. 2024)**: MM-Eval covers 18 languages but only evaluates textual hallucinations, without involving cross-modal scenarios. CCHall simultaneously evaluates the interaction between visual and textual hallucinations.
- **vs XTRUST (Li et al. 2024)**: XTRUST covers 10 languages and multiple trust dimensions (hallucination, misinformation, fairness, etc.), but does not include cross-modal scenarios. CCHall focuses on joint cross-modal $\times$ cross-lingual hallucination detection.

## Rating

| Dimension | Rating |
|------|------|
| Novelty | ⭐⭐⭐⭐ |
| Experimental Thoroughness | ⭐⭐⭐⭐⭐ |
| Writing Quality | ⭐⭐⭐⭐ |
| Value | ⭐⭐⭐⭐ |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] MAD: Modality-Adaptive Decoding for Mitigating Cross-Modal Hallucinations in Multimodal Large Language Models](../../CVPR2026/hallucination/mad_modality-adaptive_decoding_for_mitigating_cross-modal_hallucinations_in_mult.md)
- [\[ACL 2025\] ReefKnot: A Comprehensive Benchmark for Relation Hallucination Evaluation, Analysis and Mitigation in Multimodal Large Language Models](reefknot_a_comprehensive_benchmark_for_relation_hallucination_evaluation_analysi.md)
- [\[ACL 2026\] HalluAudio: A Comprehensive Benchmark for Hallucination Detection in Large Audio-Language Models](../../ACL2026/hallucination/halluaudio_a_comprehensive_benchmark_for_hallucination_detection_in_large_audio-.md)
- [\[CVPR 2026\] Cross-Modal Attention Calibration for LVLM Hallucination Mitigation](../../CVPR2026/hallucination/cross-modal_attention_calibration_for_lvlm_hallucination_mitigation.md)
- [\[ACL 2025\] Retrieval Visual Contrastive Decoding to Mitigate Object Hallucinations in Large Vision-Language Models](retrieval_visual_contrastive_decoding_to_mitigate_object_hallucinations_in_large.md)

</div>

<!-- RELATED:END -->
