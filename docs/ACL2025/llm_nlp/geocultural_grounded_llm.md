---
title: >-
  [Paper Note] Towards Geo-Culturally Grounded LLM Generations
description: >-
  [ACL 2025][LLM (Other)][Cultural Awareness] This paper systematically evaluates the impact of two RAG strategies—Knowledge Base grounding (KB grounding) and search grounding—on the cultural awareness capabilities of LLMs. It finds that search grounding significantly improves propositional cultural knowledge but exacerbates stereotype risks, and neither strategy improves cultural fluency in human evaluations.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Cultural Awareness"
  - "RAG"
  - "Search Grounding"
  - "Stereotype"
  - "Multicultural LLMs"
date: 2026-05-08
content_hash: df42ea50a84e4491
---

# Towards Geo-Culturally Grounded LLM Generations

**Conference**: ACL 2025  
**arXiv**: [2502.13497](https://arxiv.org/abs/2502.13497)  
**Code**: None  
**Area**: LLM / Cultural Consciousness and Fairness  
**Keywords**: Cultural Awareness, RAG, Search Grounding, Stereotype, Multicultural LLMs

## TL;DR

This paper systematically evaluates the impact of two RAG strategies—Knowledge Base grounding (KB grounding) and search grounding—on the cultural awareness capabilities of LLMs. It finds that search grounding significantly improves propositional cultural knowledge but exacerbates stereotype risks, and neither strategy improves cultural fluency in human evaluations.

## Background & Motivation

**Background**: LLMs exhibit significant gaps in global cultural awareness, tending to stereotype different cultures, simplify representations, and possess limited knowledge of non-Western cultures. The over-representative inclusion of specific cultures in training data and human feedback is a major cause.

**Limitations of Prior Work**: Strategies for improving LLM cultural awareness remain underexplored. Research on prompt engineering and model fine-tuning is limited, and it remains unclear whether Retrieval-Augmented Generation (RAG) using external knowledge can effectively improve cultural awareness capabilities.

**Key Challenge**: Retrieving cultural knowledge from external sources may improve factual knowledge, but biases (such as stereotypes) in the internet and knowledge bases might instead exacerbate cultural injustice.

**Goal**: Systematically evaluate the performance and risks of two strategies, KB grounding and search grounding, across multiple cultural dimensions.

**Key Insight**: Comprehensively test the two strategies on multiple multiple-choice cultural QA benchmarks (BLEnD, NormAd, SeeGULL) and open-ended human evaluations, distinguishing between "propositional cultural knowledge" and "cultural fluency."

**Core Idea**: Search grounding can improve LLMs' factual cultural knowledge but does not improve cultural fluency and exacerbates stereotypes—cultural awareness requires distinguishing between "knowing cultural facts" and "expressing oneself like a cultural insider."

## Method

### Overall Architecture

Two strategies: (1) KB Grounding: retrieves relevant text from a self-built cultural knowledge base (comprising 468k documents from CultureAtlas, Cube, CultureBank, and SeeGULL) using RAG to augment prompts; (2) Search Grounding: utilizes the Google Search API to retrieve relevant web content to augment prompts. These are tested on three LLMs (Gemini, GPT-4o-mini, OLMo2-7B).

### Key Designs

1. **Self-built Cultural Knowledge Base + Selective RAG**
    - **Function**: Compiles a knowledge base from four large-scale cultural data sources, supporting vector retrieval.
    - **Mechanism**: Retrieves top-5 relevant documents and selectively retains only $k$ truly relevant ones after assessing relevance via an LLM (selective RAG).
    - **Design Motivation**: Non-selective RAG may introduce irrelevant documents that distract weaker models (e.g., OLMo); selective filtering can alleviate this issue.

2. **Search Grounded Generation**
    - **Function**: Converts user prompts into search queries and retrieves relevant text from the internet.
    - **Mechanism**: Uses the end-to-end API of Google Vertex AI, leveraging the page-ranking capability of search engines to retrieve high-quality cultural information.
    - **Design Motivation**: The scale of the internet far exceeds that of any knowledge base, making it more likely to contain long-tail cultural information.

3. **Multi-dimensional Evaluation System**
    - **Function**: Evaluates by distinguishing between propositional cultural knowledge and cultural fluency.
    - **Mechanism**: Uses two multiple-choice benchmarks, BLEnD (everyday cultural knowledge) and NormAd (cultural norms), to evaluate knowledge; SeeGULL to evaluate stereotype avoidance; and human evaluation (9 evaluators per country across 10 countries) to evaluate cultural fluency.
    - **Design Motivation**: Relying solely on QA benchmarks cannot reflect whether an LLM truly understands a culture; human evaluation is required to detect cultural fluency in open-ended generation.

### Loss & Training

Does not involve model training. All methods are inference-time strategies (prompt augmentation) utilizing the API interfaces of existing models.

## Key Experimental Results

### Main Results

| Strategy | BLEnD Accuracy (↑) | NormAd-Country (↑) | Stereotype Avoidance (↑) |
|---|---|---|---|
| Gemini Vanilla | 60.3 (ETH) | ~47% | **Highest** |
| Gemini Search | **74.2 (ETH)** | **Highest** | Significant decline |
| Gemini KB (best) | 62.9 (ETH) | Medium | Close to vanilla |
| GPT Vanilla | Baseline | Baseline | Low |
| GPT KB (best) | Improved | Selective KB best | Low |
| OLMo Vanilla | Lowest | Lowest | Low |
| OLMo KB (non-sel.) | Decreased | Decreased | Improved instead |

### Ablation Study

| Experimental Condition | Results |
|---|---|
| Selective RAG vs Non-selective RAG | Weaker model (OLMo) benefits significantly from selective RAG, avoiding distraction from long documents |
| KB query with options vs without options | Retrieves more SeeGULL stereotypes when options are included (1266 vs 1156 questions) |
| Human Evaluation (ANOVA) | $F=0.18$, $p=0.827$, no significant difference among the three strategies |
| Search Grounding on Stereotypes | Retrieving stereotypic texts from the internet leads the model to select stereotypic answers |

### Key Findings

- Search grounding increases Gemini's accuracy on Ethiopia-related questions in BLEnD from 60.3% to 74.2%, but causes a significant regression in stereotype avoidance tests.
- Around 19% of CultureAtlas entries and 25% of CultureBank entries in the KB are about US culture, showing a data bias toward the West.
- Under non-selective KB grounding, OLMo unexpectedly improves in stereotype avoidance—because a massive amount of irrelevant text makes the model uncertain of the answer, leading it to select "uncertain".
- Human evaluation (100 prompts $\times$ 3 strategies $\times$ 3 generations $\times$ 90 evaluators) shows no strategy significantly improves cultural fluency ($p=0.827$).

## Highlights & Insights

- **Distinction Between Knowledge and Fluency**: This is the core insight of this paper. Propositional cultural knowledge (knowing facts) and cultural fluency (expressing like an insider) are two distinct dimensions, and RAG can only address the former.
- **Double-Edged Sword Effect of Search Grounding**: While the scale of the internet is vast, it contains stereotypes, and search ranking may amplify biases. This serves as a warning for all search-grounded LLM systems.
- **Unexpected Behavior of Weaker Models**: OLMo gets "confused" by irrelevant retrieved texts and unexpectedly performs better on specific tasks, revealing a non-linear interaction between RAG and model capability.

## Limitations & Future Work

- Only smaller versions of the three models were tested; larger models might perform differently.
- Human evaluation only covers 10 national cultures, lacking representation from regions like Africa, the Middle East, and South Asia.
- Search grounding was only implemented using the Gemini API; other models were not tested due to API limitations, which limits the generalizability of the findings.
- All experiments are restricted to English; cultural awareness issues are more complex in multilingual scenarios.
- Cultural-aware fine-tuning (such as CultureLLM and similar methods) was not attempted as a comparison.

## Related Work & Insights

- **BLEnD (Myung et al. 2024)**: A cross-cultural everyday knowledge benchmark with 24k English questions covering 10 countries.
- **NormAd (Rao et al. 2024)**: A benchmark for cultural norms and values, testing the acceptability of social behaviors.
- **SeeGULL (Jha et al. 2023)**: A stereotype benchmark, which this paper novelly employs to evaluate whether RAG introduces bias.
- **Insight**: Future work may require "cultural alignment" training (similar to safety alignment), rather than relying solely on retrieval augmentation to improve cultural awareness.

## Rating

- **Novelty**: ⭐⭐⭐ (The strategies themselves are not new, but the perspective of systematic evaluation in cultural scenarios is novel)
- **Experimental Thoroughness**: ⭐⭐⭐⭐ (3 models + 4 benchmarks + human evaluation, with rigorous statistical analysis)
- **Writing Quality**: ⭐⭐⭐⭐ (Clear structure, well-articulated findings, and in-depth discussion)
- **Value**: ⭐⭐⭐⭐ (The distinction between knowledge and fluency has important implications for multicultural LLM deployment)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Palm: A Culturally Inclusive and Linguistically Diverse Dataset for Arabic LLMs](palm_a_culturally_inclusive_and_linguistically_diverse_dataset_for_arabic_llms.md)
- [\[ACL 2025\] Big5-Chat: Shaping LLM Personalities Through Training on Human-Grounded Data](big5-chat_shaping_llm_personalities_through_training_on_human-grounded_data.md)
- [\[ACL 2025\] Detecting Referring Expressions in Visually Grounded Dialogue with Autoregressive Language Models](detecting_referring_expressions_in_visually_grounded_dialogue_with_autoregressiv.md)
- [\[ACL 2025\] Only a Little to the Left: A Theory-grounded Measure of Political Bias in LLMs](political_bias_theory_grounded.md)
- [\[ACL 2025\] A Large-Scale Real-World Evaluation of an LLM-Based Virtual Teaching Assistant](a_large-scale_real-world_evaluation_of_llm-based_virtual_teaching_assistant.md)

</div>

<!-- RELATED:END -->
