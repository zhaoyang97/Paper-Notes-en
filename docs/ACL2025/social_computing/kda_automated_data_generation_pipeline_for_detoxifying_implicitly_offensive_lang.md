---
title: >-
  [Paper Note] K/DA: Automated Data Generation Pipeline for Detoxifying Implicitly Offensive Language in Korean
description: >-
  [ACL 2025][Social Computing][Language detoxification] This paper proposes K/DA, an automated Korean offensive language parallel data generation pipeline. It retrieves trendy slang from online communities via RAG to augment neutral sentences into toxic variants, which are then filtered using a two-stage process (pair consistency + implicit offensiveness). This yields a high-quality dataset of 7.5K neutral-toxic pairs. Detoxification models trained on this dataset outperform th…
tags:
  - "ACL 2025"
  - "Social Computing"
  - "Language detoxification"
  - "implicitly offensive language"
  - "Korean text"
  - "RAG data generation"
  - "parallel dataset"
date: 2026-05-08
content_hash: 0271abcbc619fdcb
---

# K/DA: Automated Data Generation Pipeline for Detoxifying Implicitly Offensive Language in Korean

**Conference**: ACL 2025  
**arXiv**: [2506.13513](https://arxiv.org/abs/2506.13513)  
**Code**: None (Dataset released under CC BY-NC 4.0)  
**Area**: Social Computing  
**Keywords**: Language detoxification, implicitly offensive language, Korean text, RAG data generation, parallel dataset

## TL;DR

This paper proposes K/DA, an automated Korean offensive language parallel data generation pipeline. It retrieves trendy slang from online communities via RAG to augment neutral sentences into toxic variants, which are then filtered using a two-stage process (pair consistency + implicit offensiveness). This yields a high-quality dataset of 7.5K neutral-toxic pairs. Detoxification models trained on this dataset outperform those trained on human-annotated or translated datasets.

## Background & Motivation

**Background**: Language detoxification aims to convert offensive language into non-toxic versions while preserving the original meaning. The most direct approach to training a detoxification model is using parallel neutral-toxic datasets. Existing Korean offensive language datasets are constructed primarily through three methods: manual crawling and annotation, LLM-based generation, and translation from English.

**Limitations of Prior Work**: Each of the three existing methods has severe bottlenecks. (1) Manual crawling: Interactive content is fragmented, annotation consistency is poor, and constructing parallel datasets is highly expensive. (2) LLM generation: LLMs tend to generate offensive content that is irrelevant to the context and lean heavily toward explicit offensiveness (e.g., direct profanity), failing to produce implicit offensiveness. (3) Translation: Due to significant cultural differences between English and Korean, offensive nuances are severely lost during translation. More fundamentally, offensive language evolves rapidly—online communities constantly invent new implicit insults to bypass detection, rendering static datasets quickly obsolete.

**Key Challenge**: Implicitly offensive language (expressions containing sarcasm, bias, or contempt without explicit swear words) accounts for approximately 64% of real-life online comments and represents the core challenge of detoxification. However, existing methods struggle to generate this type of data automatically, as LLMs inherently lean toward explicit rather than implicit toxicity.

**Goal**: Design an automated pipeline capable of generating high-quality parallel datasets containing implicit offensiveness and up-to-date slang.

**Key Insight**: The authors introduce the concept of "trend-aligned slang", categorizing implicit offensiveness into three sub-types: (1) contempt and sarcasm, (2) community-specific slang, and (3) swearing variants (homophones, visually similar characters, and other circumvention methods). Utilizing RAG, current slang is retrieved from Korean online communities to enhance the generation capabilities of LLMs.

**Core Idea**: A two-stage pipeline: in the first stage, RAG (using a vector database built from 93k online comments) retrieves relevant slang to augment neutral sentences and generate toxic variants; in the second stage, LLMs act as filters to sequentially verify pair consistency and implicit offensiveness, weeding out low-quality generations.

## Method

### Overall Architecture

The K/DA pipeline takes a set of neutral sentences as input and outputs a neutral-toxic parallel dataset. The pipeline consists of two stages: (1) Slang Retrieval: For each neutral sentence, relevant content is retrieved from a slang vector database using varying retrieval numbers $n \in \{0, 3, 5, 7, 9\}$ to enhance prompts and generate toxic variants containing slang (generating 5 candidates per sentence). (2) Generation Filtering: All candidates are sequentially processed through pair consistency filtering and implicit offensiveness filtering, keeping only the high-quality outputs that pass both filters.

### Key Designs

1. **Multiple RAGs for Maximized Diversity**:

    - **Function**: Balance retrieval quality and generation diversity.
    - **Mechanism**: Traditional RAG uses a fixed retrieval quantity $n$. If $n$ is too small, useful information may be missed; if $n$ is too large, noise may be introduced. K/DA retrieves and generates using five configurations $n \in \{0, 3, 5, 7, 9\}$, passing all results to the filtering stage. $n=0$ (zero-retrieval, pure prompt-based generation) is also vital, as some neutral sentences handle topics that lack corresponding slang in the database. Experiments show that different $n$ values contribute relatively evenly to the optimal generations.
    - **Design Motivation**: To avoid the overhead of training an additional model to dynamically decide $n$ (such as Self-RAG), relying instead on the robustness of the filtering stage to handle noisy retrievals.

2. **Pair Consistency Filtering**:

    - **Function**: Ensure that the toxic variant conveys the same meaning as the original neutral sentence.
    - **Mechanism**: LLMs are prompted to determine the relationship type between the generated toxic and neutral sentences—either "context maintained" (consistent) or "context shifted" (inconsistent, such as conversational replies or irrelevant content). Instructed via a prompt with definitions of inconsistency types and a one-shot example, the LLM-based filter achieves an 86% agreement rate with human annotators.
    - **Design Motivation**: Addressing three common inconsistent generations: (1) LLM answering the neutral sentence as a question; (2) LLM introducing irrelevant slang causing semantic shift; (3) LLM simply paraphrasing instead of adding toxicity. The Context Shift filtering prompt performs the best, retaining 47.89% of the generated candidates.

3. **Implicit Offensiveness Filtering**:

    - **Function**: Ensure that the retained data possesses a sufficient level of implicit offensiveness.
    - **Mechanism**: A Derogatory Detection prompt is utilized to let LLMs judge if the generation matches the broad definition of implicit offensiveness (including gender/regional/political disparagement, community slang, homophonic variations of swear words). Non-offensive and explicitly profane generations are filtered out, keeping only implicit offensiveness. Retaining rate is 63.24%.
    - **Design Motivation**: While more fine-grained multi-class classification prompts (like Multi-meaning Relationship) yield the highest implicit offensiveness scores, their retention rate is too low (3.2%) to be practical. Derogatory Detection achieves the best balance between retention rate and quality.

### Loss & Training

The detoxification model is trained using simple instruction fine-tuning. Based on Ko-LLaMA3-Luxia-8B, the model uses the (toxic -> neutral) pairs from the K/DA dataset as training data, with a learning rate of 2e-4 and a batch size of 4, trained on two A100 GPUs.

## Key Experimental Results

### Main Results (Dataset Quality Comparison via G-Eval)

| Dataset | Overall O. ↑ | Implicit O. ↑ | Consistency ↑ |
|--------|-------------|--------------|--------------|
| K-OMG (LLM Generation) | 3.770 | 2.399 | 1.393 |
| BEEP (Human Crawled) | 2.300 | 2.206 | - |
| KODOLI (Human Annotated) | 3.293 | 2.554 | - |
| Translated CADD | 2.963 | 1.861 | 1.458 |
| **K/DA (Ours)** | **2.719** | **2.622** | **4.060** |

K/DA exhibits the lowest overall offensiveness but the highest implicit offensiveness, demonstrating that the dataset is successfully biased toward implicit toxicity. Additionally, its pair consistency drastically outperforms other datasets.

### Ablation Study (Detoxification Model Performance Tested on Ours)

| Training Data | Overall O. ↓ | Implicit O. ↓ | Consistency ↑ | Fluency ↑ |
|---------|-------------|--------------|--------------|----------|
| Vanilla LM (No training) | 1.677 | 1.603 | 3.263 | 2.916 |
| **K/DA (Ours)** | **1.145** | **1.156** | **3.553** | **3.027** |
| K-OMG | 1.657 | 1.608 | 3.227 | 2.995 |
| Translated CADD | 1.802 | 1.686 | 3.463 | 2.985 |

### Key Findings

- Models trained on K-OMG and CADD show no statistically significant difference in detoxification compared to the untrained Vanilla LM, suggesting that pair consistency is critical for training detoxification models—inconsistent pairs instead hinder the learning process.
- Models trained on K/DA improve on both its own test set and the KOLD dataset, though the improvement disappears on BEEP (the most challenging transfer setting). The authors attribute this to the limited coverage of neutral sentences, which can be mitigated by expanding the diversity of source neutral sentences.
- The K/DA pipeline is effective across languages (539 pairs generated for English) and robust across models (both Trillion-7B and Gemma2-9B produce competitive datasets).
- LLM filters show high agreement with human judgment (86% for pair consistency, 90% for implicit offensiveness), achieving even higher agreement under majority voting (97% and 94% respectively).

## Highlights & Insights

- **"Generate-then-filter" instead of "direct high-quality LLM generation"**: The division of labor for LLMs is cleverly designed—leveraging multiple RAG configurations for diversity during generation, and relying on LLM-as-a-judge for quality during filtering. This generate-then-filter paradigm largely converts LLM unpredictability into manageable classification decisions.
- **Proposing the concept of trend-aligned slang**: Categorizing implicit offensiveness into sarcasm, community slang, and swearing variants, this work points out that the latter two categories constitute 64% of real toxic language but are ignored by previous research. This conceptual framework is valuable for understanding the evolution of online hate speech.
- **Using RAG to retrieve slang from online communities**: Utilizing a vector database built on 93k crawled comments allows generated data to keep pace with slang dynamics, solving the issue of static datasets becoming quickly outdated.

## Limitations & Future Work

- **Dependency on Large Models**: Open-source LLM generation quality is inferior to GPT-4 Turbo, especially regarding pair consistency. Future work could fine-tune open-source models specifically for generation and filtering tasks.
- **Dataset Limited to Korean**: Although the pipeline is language-agnostic, the currently released dataset is mainly in Korean. The English subset contains only 539 pairs.
- **Limited Sources of Neutral Sentences**: The poor detoxification transfer performance on BEEP underscores the need to expand both the sources and diversity of the input neutral sentences.
- **Safety and Ethical Considerations**: The dataset contains real-world offensive content and is restricted to academic research purposes (CC BY-NC 4.0).

## Related Work & Insights

- **vs K-OMG (Shin et al., 2023)**: K-OMG also uses LLMs to generate Korean offensive data but lacks a filtering mechanism, leading to very low pair consistency (1.393 vs 4.060 for K/DA). This demonstrates the poor quality of unfiltered LLM-generated data.
- **vs Translated CADD**: The translation approach yields the worst implicit offensiveness (1.861) due to the loss of subtle nuances between English and Korean cultures. K/DA directly captures slang from Korean active forums, preserving local cultural characteristics.
- **vs ToxiGen (Hartvigsen et al., 2022)**: ToxiGen is a benchmark for English implicit toxicity. The English subset generated by K/DA outperforms ToxiGen in implicit offensiveness (2.269 vs 1.834), demonstrating the strength of the RAG + filtering pipeline.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The RAG + double-filtering automated parallel data generation pipeline is cleverly designed, and the concept of trend-aligned slang is novel.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Comprehensive evaluation involving G-Eval, human assessment, cross-lingual/cross-model experiments, and downstream detoxification tasks.
- **Writing Quality**: ⭐⭐⭐⭐ Clear logic, every design choice is supported by ablation studies, and the prompt designs are fully transparently disclosed.
- **Value**: ⭐⭐⭐⭐ High practical value for the Korean NLP community; the pipeline design can easily scale to other languages.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Is LLM an Overconfident Judge? Unveiling the Capabilities of LLMs in Detecting Offensive Language with Annotation Disagreement](is_llm_an_overconfident_judge_unveiling_the_capabilities_of_llms_in_detecting_of.md)
- [\[ACL 2026\] Synthia: Scalable Grounded Persona Generation from Social Media Data](../../ACL2026/social_computing/synthia_scalable_grounded_persona_generation_from_social_media_data.md)
- [\[ACL 2025\] Evaluation of LLM Vulnerabilities to Being Misused for Personalized Disinformation Generation](llm_personalized_disinformation.md)
- [\[NeurIPS 2025\] DATE-LM: Benchmarking Data Attribution Evaluation for Large Language Models](../../NeurIPS2025/social_computing/date-lm_benchmarking_data_attribution_evaluation_for_large_language_models.md)
- [\[NeurIPS 2025\] Precise Information Control in Long-Form Text Generation](../../NeurIPS2025/social_computing/precise_information_control_in_long-form_text_generation.md)

</div>

<!-- RELATED:END -->
