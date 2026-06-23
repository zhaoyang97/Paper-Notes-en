---
title: >-
  [Paper Note] VULCA-Bench: A Multicultural Vision-Language Benchmark for Evaluating Cultural Understanding
description: >-
  [ACL 2026][Multimodal VLM][Vision-Language Model] VULCA-Bench utilizes 8 cultural traditions, 7,410 image-bilingual expert critique pairs, and an L1-L5 five-layer cultural understanding framework to advance VLM evaluation from "seeing objects" to "understanding symbols, history, and aesthetic philosophy," revealing that existing models typically drop 31-40 percentage
tags:
  - ACL 2026
  - Multimodal VLM
  - Vision-Language Model
date: 2026-05-08
content_hash: 1ae1d98650d2cc68
---
# VULCA-Bench: A Multicultural Vision-Language Benchmark for Evaluating Cultural Understanding

**Conference**: ACL2026  
**arXiv**: [2601.07986](https://arxiv.org/abs/2601.07986)  
**Code**: https://github.com/yha9806/VULCA-Bench  
**Area**: Multimodal VLM  
**Keywords**: Multicultural evaluation, Vision-Language Models, Art criticism, Cultural understanding, Cross-cultural fairness

## TL;DR
VULCA-Bench utilizes 8 cultural traditions, 7,410 image-bilingual expert critique pairs, and an L1-L5 five-layer cultural understanding framework to advance VLM evaluation from "seeing objects" to "understanding symbols, history, and aesthetic philosophy," revealing that existing models typically drop 31-40 percentage points on high-level cultural reasoning.

## Background & Motivation
**Background**: Mainstream evaluations for multimodal VLMs have long focused on object recognition, scene description, VQA, hallucination detection, and chart/document QA. These benchmarks can measure L1-level visual perception and some factual QA, but they rarely require models to explain the cultural symbolism, historical genres, and aesthetic philosophies behind an image.

**Limitations of Prior Work**: While cultural datasets are emerging, many still use QA or recognition formats, which tend to compress cultural understanding into factual recall. Art-related datasets like WikiArt, OmniArt, and ArtEmis cover works and styles but lack expert-level critiques, cultural dimension annotations, and cross-cultural hierarchical diagnosis. Crucially, many existing datasets under-represent non-Western traditions; models may seem to "understand art" while only being familiar with Western visual vocabularies.

**Key Challenge**: Cultural understanding is not a single ability but a hierarchy of capabilities deepening from visual surfaces to philosophical interpretation. A model recognizing plum blossoms, ink strokes, and composition does not equate to it understanding the resilience symbolized by plum blossoms in Chinese painting, the "Four Gentlemen" tradition, or aesthetic concepts like "Qi-yun-sheng-dong" (rhythmic vitality) and "Yi-jing" (artistic conception). Existing benchmarks mix these levels, allowing a model's shallow visual capabilities to mask its deep cultural deficiencies.

**Goal**: The authors aim to construct a cross-cultural, reproducible, and diagnostic VLM cultural understanding benchmark. It must possess sufficient scale and distinguish between L1-L2 visual/technical analysis and L3-L5 symbolic, historical, and philosophical/aesthetic reasoning, while maintaining methodological fairness across different cultures.

**Key Insight**: The paper selects "art criticism" as the task vehicle because artistic images naturally embody visual forms, material techniques, cultural symbols, historical context, and aesthetic philosophy. Compared to multiple-choice or short QA, generative expert critiques more effectively expose whether a model can truly organize high-level cultural interpretations rather than just stating keywords.

**Core Idea**: Replace single visual QA metrics with "cross-cultural expert critiques + five-layer cultural understanding dimensions + balanced evaluation subsets," allowing VLM cultural understanding capabilities to be diagnosed by hierarchy, culture, and dimension.

## Method
The methodology essentially involves benchmark construction and validation. The authors first define a hierarchical framework for cultural understanding, then collect open artistic images across 8 cultural traditions, organize experts to write bilingual (Chinese-English) critiques, annotate cultural dimensions, and finally conduct pilot evaluations with several VLMs to verify if the dataset reveals high-level cultural understanding deficiencies.

### Overall Architecture
The input to VULCA-Bench is an artwork and its metadata, and the output is an expert critique covering five levels with explicit cultural dimension labels. The complete workflow involves four steps: first, collecting images and metadata from open museum collections; second, defining L1-L5 dimension tables for each cultural tradition; third, having experts from corresponding cultural backgrounds write critiques in Chinese and English and annotate `covered_dimensions`; fourth, using the Dimension Coverage Rate (DCR) to diagnose whether model-generated critiques cover these cultural dimensions.

The data covers 8 cultural traditions: Western, Chinese, Japanese, Korean, Islamic, Indian, Mural, and Hermitage. The full version contains 7,410 image-critique pairs with a total of 225 culture-specific dimensions. Simultaneously, it provides subsets such as Balanced, Balanced-Pilot, Gold, and Human for full evaluation, balanced fairness analysis, and manual calibration.

### Key Designs

**1. Five-layer Cultural Understanding Framework: Decomposing "cultural understanding" into diagnostic capability levels rather than a single score**

Many VLM benchmarks only ask "what is seen," mixing object recognition with philosophical/aesthetic interpretation into one score. Consequently, models can achieve high scores based on shallow visual capability, masking deep cultural weaknesses. Borrowing from Panofsky's iconographic method, VULCA-Bench splits capabilities into five levels: L1 Visual Perception, L2 Technical Analysis, L3 Cultural Symbolism, L4 Historical Context, and L5 Philosophical Aesthetics. L1-L2 can be completed through image observation and material/technique knowledge, whereas L3-L5 require true understanding of symbolic traditions, art history lineages, and native aesthetic theories. Identifying a plum blossom is one thing; understanding its symbolism of resilience in Chinese painting or the "Four Gentlemen" tradition is another. This stratification transforms evaluation results from a single score into a capability profile.

**2. Cultural Symmetry Principle: Evaluating each culture under the same protocol without forcing them into Western standards**

Forcing every culture to have identical dimensions would erase cultural differences, while allowing arbitrary definitions would prevent horizontal comparison. The authors compromise by seeking symmetry in schema and annotation protocols rather than sample counts: the 8 cultural traditions follow the same L1-L5 framework, quality thresholds, and expert review processes, but specific dimensions can reflect native theories—such as "Qi-yun" and "Yi-jing" in Chinese painting, *wabi-sabi* in Japanese art, or *rasa* in Indian art. Combined with the Balanced subset, smaller cultural categories are not overshadowed by larger ones like Western or Chinese traditions during horizontal comparisons.

**3. Bilingual Expert Critiques and DCR Diagnosis: Making cultural interpretation evaluable, trainable, and labeled**

Free-text critiques alone are difficult to reproduce for diagnosis. Therefore, each expert critique must meet standards and be structurally labeled: at least 150 characters for Chinese and 100 words for English, covering at least 70% of cultural dimensions, with `covered_dimensions` explicitly stored. During evaluation, the Dimension Coverage Rate (DCR) approximates how many cultural dimensions a model's critique touches. For culture $c$ and level $k$, it is calculated as:

$$DCR(c,k)=\frac{|D_k^c|}{|D_k|}$$

where $D_k^c$ is the set of dimensions hit by the model's critique and $D_k$ is the set of dimensions that should be covered at that level. The bilingual design preserves untranslatable terms like "Qi-yun" while ensuring accessibility for English readers; explicit dimension labels ensure the benchmark is reproducible and ready for use as supervision signals.

### Loss & Training
The paper does not propose a new training loss but rather evaluation metrics and data construction protocols. The core diagnostic metric is the Dimension Coverage Rate, estimated using keywords, synonym dictionaries, embedding similarity, and NLI verification. In the pilot, all models generated English critiques, and the authors reported L1-L2, L3-L5, hierarchical gaps, and overall DCR.

## Key Experimental Results

### Main Results
The pilot evaluation was conducted on the Balanced-Pilot subset, with 48 samples per culture (336 total samples across 7 cultures). Results were highly consistent: L1-L2 DCR was significantly higher than L3-L5 for all models, indicating they can describe visuals and techniques but struggle with deep cultural symbolism and philosophical aesthetics.

| Model | L1-L2 DCR | L3-L5 DCR | Hierarchical Gap ΔL | Total DCR |
|------|-----------|-----------|-----------|--------|
| Gemini-2.5-Pro | 89.2 | 58.1 | 31.1 | 72.4 |
| Qwen3-VL-235B | 85.6 | 54.3 | 31.3 | 68.7 |
| GPT-4o | 87.1 | 46.8 | 40.3 | 65.3 |
| Claude-Sonnet-4.5 | 84.3 | 48.2 | 36.1 | 64.8 |
| GLM-4V-Flash | 78.4 | 40.7 | 37.7 | 58.2 |

The dataset size and quality control are comprehensive. Instead of just a single collection, the authors provide multiple evaluation perspectives.

| Item | Value / Description | Meaning |
|------|-------------|------|
| Total Samples | 7,410 image-critique pairs | Supports aggregate benchmarking and training |
| Cultural Traditions | 8 | Covers Chinese, Western, Japanese, Korean, Islamic, Indian, Dunhuang Murals, Hermitage, etc. |
| Cultural Dimensions | 225 | ~25-30 dimensions per culture |
| Bilingual Completion | 100% | Every sample has both ZH and EN critiques |
| Cultural Fact Accuracy | 98% | Estimated via sampled expert audits |
| balanced-pilot | 336 samples, 7 cultures | Used for fair, low-cost pilot evaluation |

### Ablation Study
The paper lacks traditional ablation for model training but includes analyses of data quality, evaluation robustness, and few-shot diagnosis to prove the benchmark signal is not caused by sample length, random sampling, or proprietary embeddings.

| Analysis Item | Result | Description |
|--------|------|------|
| Balanced vs Full Ranking Consistency | Spearman ρ=0.94, 95% CI [0.87, 0.98] | Small balanced subset predicts full ranking well |
| DCR Correlation with Manual Dimensions | Pearson r=0.82 | DCR serves as a coarse-grained diagnostic signal |
| Keyword Hit Precision (Expert) | ~78% | Noisy but sufficient for dataset-level checks |
| OpenAI embedding vs BGE | Consistency 86% vs 84% | Conclusions do not depend on proprietary embeddings |
| Few-shot Prompting | DeepSeek-VL2 3-shot dropped 41.3%, GPT-4o dropped 15.5% | Expert critiques as examples do not necessarily improve understanding |

### Key Findings
- The most stable finding is the hierarchical gap: all models drop 31-40 percentage points from L1-L2 to L3-L5, showing "cultural depth" is not a natural byproduct of general visual perception.
- Error types concentrate in three areas: using cultural terms without visual evidence, applying modern historical concepts to early works, and confusing adjacent cultural traditions (e.g., misidentifying a Safavid Persian miniature as Mughal/Rajput style).
- Few-shot results are intriguing: providing culture-matched expert critiques as examples caused some models to degrade, possibly due to long-context attention dilution, imitation of format over reasoning, or expert templates restricting generative flexibility.

## Highlights & Insights
- The greatest value of this paper is not just "another art dataset," but framing cultural understanding as a hierarchical diagnostic problem. This avoids using a single total score to hide what the model truly cannot do.
- The Cultural Symmetry Principle is practical: it acknowledges naturally imbalanced sample counts across cultures but ensures fairness via identical protocols and balanced subsets.
- Expert critiques serve as both evaluation targets and trainable resources. For future cultural VLM fine-tuning, VULCA-Bench provides supervision signals with dimension labels.
- While DCR is coarse, it makes large-scale rapid diagnosis feasible; it can later be combined with LLM judges and human scores into a multi-layered evaluation system.

## Limitations & Future Work
- Western and Chinese data account for 82% of the total, reflecting the reality of museum digitization and expert resources, which increases estimated variance for minority cultures. Serious cross-cultural comparisons should prioritize the balanced subset and report confidence intervals.
- L5 philosophical aesthetics are naturally more subjective, with higher expert revision rates observed compared to L1-L2. Future work requires stronger psychometric calibration beyond dimension coverage.
- The bilingual design focuses on Chinese and English; native terminology preservation for Japanese, Korean, Arabic, Sanskrit/Hindi is still limited. A truly multicultural benchmark should expand to critiques in more native languages.
- DCR remains a coarse keyword/synonym-driven diagnosis that may miss implicit explanations or be fooled by surface-level terminology. A more robust direction involves expert-calibrated judge-based rubrics.

## Related Work & Insights
- **vs MME / SEED-Bench / POPE**: These benchmarks focus on visual perception, object hallucination, and general VQA. VULCA-Bench targets higher-level cultural symbolism and aesthetic philosophy in art.
- **vs CulturalBench / CulturalVQA / GIMMICK**: The latter are mostly QA or recognition tasks measuring cultural facts/biases; VULCA-Bench uses generative art criticism for open interpretative capabilities.
- **vs WikiArt / OmniArt / ArtEmis**: These highlight style, category, or emotion but lack cross-cultural expert critiques and hierarchical dimension labels. The insight: art understanding benchmarks need "expert explanation structures" rather than just images and labels.
- **Inspiration for Follow-up**: L1-L5 can be used as a training curriculum—first training for visual/technical grounding, then introducing RAG or knowledge graphs for L3-L5, and finally using expert judges for quality calibration of cultural interpretation.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Clear problem definition and benchmark value by organizing multicultural art critiques via a hierarchical framework.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Quality, pilot, and robustness analyses are comprehensive, though DCR is still coarse and the model evaluation is not yet a final leaderboard.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear structure and sufficient data construction details, though tables and appendices are numerous; the core protocol could be more concise.
- **Value**: ⭐⭐⭐⭐⭐ Directly applicable value for multimodal cultural understanding, cultural fairness, and artistic VLM evaluation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ENC-Bench: A Benchmark for Evaluating MLLMs in Electronic Navigational Chart Understanding](../../CVPR2026/multimodal_vlm/enc-bench_a_benchmark_for_evaluating_multimodal_large_language_models_in_electro.md)
- [\[ACL 2025\] Evaluating Visual and Cultural Interpretation: The K-Viscuit Benchmark with Human-VLM Collaboration](../../ACL2025/multimodal_vlm/evaluating_visual_and_cultural_interpretation_the_k-viscuit_benchmark_with_human.md)
- [\[ACL 2026\] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity](cartbench_evaluating_vision-language_models_on_chinese_art_understanding_interpr.md)
- [\[CVPR 2026\] Flat-Pack Bench: Evaluating Spatio-Temporal Understanding in Large Vision-Language Models through Furniture Assembly](../../CVPR2026/multimodal_vlm/flat-pack_bench_evaluating_spatio-temporal_understanding_in_large_vision-languag.md)
- [\[CVPR 2026\] VS-Bench: Evaluating VLMs for Strategic Abilities in Multi-Agent Environments](../../CVPR2026/multimodal_vlm/vs_bench_evaluating_vlms_for_strategic_abilities_in_multi_agent_environments.md)

</div>

<!-- RELATED:END -->
