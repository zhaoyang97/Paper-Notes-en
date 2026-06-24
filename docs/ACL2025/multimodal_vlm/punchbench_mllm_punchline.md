---
title: >-
  [Paper Note] PunchBench: Benchmarking MLLMs in Multimodal Punchline Comprehension
description: >-
  [ACL 2025][Multimodal VLM][punchline comprehension] This paper proposes PunchBench, a multimodal humor/sarcasm comprehension benchmark containing 6,000 image-text pairs and 54,000 QA pairs. It eliminates language shortcuts through synonymous/antonymous caption generation, and proposes a Simple-to-Complex Chain-of-Question (SC-CoQ) strategy to consistently improve punchline comprehension capabilities across all models and question formats.
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "punchline comprehension"
  - "multimodal benchmark"
  - "humor"
  - "sarcasm"
  - "chain-of-question"
date: 2026-05-08
content_hash: aab29fe9c6e6bb83
---

# PunchBench: Benchmarking MLLMs in Multimodal Punchline Comprehension

**Conference**: ACL 2025  
**arXiv**: [2412.11906](https://arxiv.org/abs/2412.11906)  
**Code**: [https://github.com/OuyangKun10/PunchBench](https://github.com/OuyangKun10/PunchBench)  
**Area**: Multimodal VLM  
**Keywords**: punchline comprehension, multimodal benchmark, humor, sarcasm, chain-of-question

## TL;DR
This paper proposes PunchBench, a multimodal humor/sarcasm comprehension benchmark containing 6,000 image-text pairs and 54,000 QA pairs. It eliminates language shortcuts through synonymous/antonymous caption generation, and proposes a Simple-to-Complex Chain-of-Question (SC-CoQ) strategy to consistently improve punchline comprehension capabilities across all models and question formats.

## Background & Motivation
1. **Background**: Multimodal Large Language Models (MLLMs) have achieved significant progress in factual comprehension tasks such as visual question answering and image captioning. However, their ability to comprehend punchlines (humor/sarcasm) has not been fully evaluated.
2. **Limitations of Prior Work**: Existing punchline comprehension benchmarks suffer from three major limitations: (1) Language shortcuts—models can answer correctly solely relying on biased words or inconsistencies in the text without truly understanding the image-text interaction; (2) Single question format—only using one QA scheme fails to comprehensively evaluate model robustness; (3) Narrow content domains—only focusing on a single domain like cartoons, leading to insufficient coverage.
3. **Key Challenge**: Design flaws in existing benchmarks make it impossible to differentiate whether "the model truly understands the punchline" or "the model exploits data shortcuts," casting doubt on the validity of evaluation results.
4. **Goal**: (1) How to build an accurate and comprehensive benchmark that eliminates language shortcuts, covers multiple question formats, and spans multiple domains? (2) How large is the gap between MLLMs and humans in punchline comprehension? (3) How to improve the punchline comprehension capabilities of MLLMs?
5. **Key Insight**: Eliminate shortcut biases by generating modified captions via synonymous/antonymous replacement; simultaneously, design the SC-CoQ prompting strategy inspired by the "simple-to-complex" learning process in cognitive science.
6. **Core Idea**: Use synonymous/antonymous captions to eliminate evaluation shortcuts, and use a simple-to-complex question chain to enhance MLLMs' punchline comprehension.

## Method

### Overall Architecture
The construction of PunchBench consists of four steps: (1) Collect and manually annotate image-text pairs from existing datasets and multimedia platforms; (2) Generate synonymous and antonymous captions to eliminate shortcuts; (3) Construct multi-format instructions for a two-layer task (perception layer and reasoning layer); (4) Conduct manual quality checks. Building on this, the SC-CoQ strategy is proposed to improve model performance.

### Key Designs

1. **Synonymous & Antonymous Captions**:

    - **Function**: Eliminate text shortcuts that models might leverage.
    - **Mechanism**: Use `gpt-3.5-turbo` to perform word replacement on original captions (replacing emotional words, action words, etc., with synonyms/antonyms) to generate synonymous and antonymous captions. For captions containing semantic conflicts (e.g., "I'm so glad! What a disgusting day!"), first use the LLM to identify the conflicting parts and then process them separately. Synonymous captions retain the same punchline label as the original caption, while antonymous captions serve as contrasts.
    - **Design Motivation**: Experiments show that models like CogVLM2 can correctly determine if the original caption contains a punchline, but their performance drops significantly when faced with synonymous/antonymous variants, showing that models rely on specific vocabulary rather than true comprehension.

2. **Two-Level Multi-Format Task Design**:

    - **Function**: Comprehensively evaluate punchline comprehension from perception to reasoning through multiple perspectives.
    - **Mechanism**: The perception layer (Punchline Perception) includes Yes/No QA (determining the presence of a punchline), Matching QA (choosing which of two captions contains the punchline), and Multi-option QA (four-option comprehension); the reasoning layer (Punchline Reasoning) includes Yes/No QA (determining whether a reasoning sentence correctly explains the punchline), Matching QA (choosing the correct explanation), and Generation QA (free-form generation of explanations). Each format is paired with multiple instruction templates, and the option order is randomized.
    - **Design Motivation**: A single question format cannot provide a comprehensive evaluation—experiments indicate that a model might perform well on Yes/No QA but fail on Matching QA, exposing performance inconsistency.

3. **Simple-to-Complex Chain-of-Question (SC-CoQ)**:

    - **Function**: Gradually improve the punchline comprehension of MLLMs through a simple-to-complex sequence of questions.
    - **Mechanism**: Organizes a chain of questions from simple to complex, both within and across tasks. Specifically, the model is first asked to answer simple questions from the perception layer (such as Yes/No), and then gradually transitions to complex questions in the reasoning layer (such as generating explanations), utilizing the answers of the previous simple questions as context to assist in answering subsequent complex questions.
    - **Design Motivation**: Complex punchline comprehension can be decomposed into multiple sub-skills (recognizing the presence of a punchline $\to$ selecting the correct caption $\to$ explaining the reason). Mastering simple sub-skills first before advancing is more effective than directly facing complex questions.

## Key Experimental Results

### Main Results — Punchline Perception

| Model | Params | Yes/No (SC-CoQ) | Matching (SC-CoQ) | Multi-choice (SC-CoQ) |
|------|-------|-----------------|--------------------|-----------------------|
| GPT-4o | - | 80.7 | 67.9 | 53.1 |
| GPT-4V | - | 78.1 | 65.0 | 51.9 |
| Qwen2-VL-72B | 72B | 76.1 | 62.9 | 51.7 |
| Aria | 3.5B×8 | 74.5 | 63.6 | 50.8 |
| CogVLM2 | 19B | 71.3 | 60.8 | 46.3 |
| LLaVA | 7B | 64.8 | 57.1 | 39.1 |
| Human | - | **98.3** | **97.7** | **90.7** |

### SC-CoQ vs. Other Prompting Methods (GPT-4o Perception Yes/No)

| Method | Accuracy |
|------|----------|
| Zero-shot | 77.5 |
| CoT | 78.6 |
| 3-shot | 79.2 |
| **SC-CoQ** | **80.7** |

### Key Findings
- There is a huge gap between MLLMs and humans in punchline comprehension: the overall strongest model, GPT-4o, achieves 80.7% on Perception Yes/No, compared to humans at 98.3%; the gap is even wider on Multi-choice (53.1% vs 90.7%).
- SC-CoQ consistently outperforms zero-shot, CoT, and few-shot methods across all models and question formats, with $P\text{-value} < 0.01$, indicating statistical significance.
- Models experience a significant drop in performance when faced with synonymous/antonymous captions, confirming that language shortcuts indeed exist.
- Among open-source models, Qwen2-VL-72B and Aria perform the best, reaching performance close to GPT-4V; small models (2B-7B) score only slightly above random guess (25%) on multi-choice questions.
- The reasoning layer task (Generation QA) is the most challenging, with GPT-4o only scoring around 53%.

## Highlights & Insights
- Synonymous/antonymous captions are a clever design to eliminate textual shortcuts—they test the capability of "image-text interaction comprehension" more precisely than simply deleting text. This logic can be transferred to other multimodal benchmarks.
- The key insight of SC-CoQ is that punchline comprehension is a hierarchical ability: first perceiving existence, then locating key elements, and finally reasoning the cause. This simple-to-complex paradigm aligns better with cognitive processes than flat chain-of-thought.
- During the human quality checks of 500 instructions, only 1 was labeled as "unanswerable", proving the high quality of the dataset.

## Limitations & Future Work
- The dataset primarily covers English punchlines; cross-lingual/cross-cultural humor comprehension has not been incorporated.
- SC-CoQ increases reasoning steps and token consumption, which poses efficiency considerations for practical deployment.
- The evaluation of Generation QA relies on reference answer similarity, which might not fully capture diverse correct explanations.
- Fine-tuning strategies have not been explored—SC-CoQ only acts as an inference-time strategy; incorporating it during training might yield even better results.
- Although the scale of 6,000 image-text pairs is substantial, the domain distribution and difficulty distribution for each domain have not been detailed.
- The definition of punchlines (humor + sarcasm) might omit other rhetorical figures that require deep comprehension (e.g., irony, hyperbole).

## Related Work & Insights
- **vs MORE (Desai et al., 2022)**: Only focuses on the single task of sarcasm explanation, whereas PunchBench covers humor + sarcasm, perception + reasoning, and multiple question formats.
- **vs HUB (Hessel et al., 2023)**: Only focuses on humor in the cartoon domain, whereas PunchBench covers multiple multimedia domains like posts, cartoons, comments, and memes.
- **vs Chain-of-Thought**: CoT lets models reason freely, whereas SC-CoQ guides the reasoning direction through a structured question chain, which experiments prove to be more effective.
- PunchBench can serve as a long-term tracker for VLM evolution—as model capabilities grow, the upward trend in punchline comprehension can be monitored.

## Rating
- **Overall Evaluation**: Constructed a high-quality multimodal humor/sarcasm comprehension benchmark, with the SC-CoQ strategy showing practical value.
- **Novelty**: ⭐⭐⭐⭐ The design of using synonymous/antonymous captions to eliminate shortcuts and the SC-CoQ strategy are both novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation involving 12 models, 6 question formats, and 4 prompting methods.
- **Writing Quality**: ⭐⭐⭐⭐ Clear structure with rich examples and diagrams.
- **Value**: ⭐⭐⭐⭐ Fills a crucial gap in evaluating the punchline comprehension of MLLMs.

<!-- 数据规模: 6,000 image-caption pairs, 54,000 QA pairs, 4类多媒体域, 12个模型评测 -->

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension](unveiling_cultural_blind_spots_analyzing_the_limitations_of_mllms_in_procedural_.md)
- [\[ACL 2025\] MMINA: Benchmarking Multihop Multimodal Internet Agents](mmina_benchmarking_multihop_multimodal_internet_agents.md)
- [\[CVPR 2026\] WEAVE: Unleashing and Benchmarking the In-context Interleaved Comprehension and Generation](../../CVPR2026/multimodal_vlm/weave_unleashing_and_benchmarking_the_in-context_interleaved_comprehension_and_g.md)
- [\[ACL 2025\] MMSciBench: Benchmarking Language Models on Chinese Multimodal Scientific Problems](mmscibench_benchmarking_language_models_on_chinese_multimodal_scientific_problem.md)
- [\[ICCV 2025\] Instruction-Oriented Preference Alignment for Enhancing Multi-Modal Comprehension Capability of MLLMs](../../ICCV2025/multimodal_vlm/instruction-oriented_preference_alignment_for_enhancing_multi-modal_comprehensio.md)

</div>

<!-- RELATED:END -->
