---
title: >-
  [Paper Note] Almieyar-Oryx-BloomBench: A Bilingual Multimodal Benchmark for Cognitively Informed Evaluation of Vision-Language Models
description: >-
  [ACL2026][Multimodal VLM][Bloom's Taxonomy] BloomBench reconstructs VLM evaluation using Bloom's cognitive taxonomy, organizing 7,747 bilingual English-Arabic vision-language QA samples into 6 cognitive levels and 106 ta…
tags:
  - "ACL2026"
  - "Multimodal VLM"
  - "Bloom's Taxonomy"
  - "Multimodal Evaluation"
  - "English-Arabic Bilingual"
  - "Cognitive Diagnosis"
  - "Likelihood-based Scoring"
date: 2026-05-08
content_hash: f7f1d0b189c6fb2d
---

# Almieyar-Oryx-BloomBench: A Bilingual Multimodal Benchmark for Cognitively Informed Evaluation of Vision-Language Models

**Conference**: ACL2026 Findings  
**arXiv**: [2606.05531](https://arxiv.org/abs/2606.05531)  
**Code**: https://github.com/qcri/Almieyar-Oryx-BloomBench  
**Area**: Multimodal VLM / Evaluation Benchmark  
**Keywords**: Bloom's Taxonomy, Multimodal Evaluation, English-Arabic Bilingual, Cognitive Diagnosis, Likelihood-based Scoring  

## TL;DR
BloomBench reconstructs VLM evaluation using Bloom's cognitive taxonomy, organizing 7,747 bilingual English-Arabic vision-language QA samples into 6 cognitive levels and 106 task types. It reveals that high scores in current VLMs often mask significant weaknesses in factual recall, creative synthesis, and cross-lingual reasoning.

## Background & Motivation
**Background**: VLM evaluation has evolved from early VQA, image captioning, and hallucination detection to more comprehensive benchmarks like MMMU, MMT-Bench, and VLM2-Bench. The mainstream approach typically aggregates a large number of tasks into a single global score to compare model performance across multimodal knowledge, perception, reasoning, or localization.

**Limitations of Prior Work**: Although these benchmarks have widened in coverage, their diagnostic granularity remains insufficient. High scores in reading charts or answering multiple-choice questions do not equate to genuine human-like hierarchical cognitive abilities; instead, models may simply exploit task formats, statistical shortcuts, or common patterns in English-centric corpora. Furthermore, existing VLM benchmarks are significantly biased toward English, with inadequate coverage of non-English vision-language scenarios such as Arabic.

**Key Challenge**: VLM evaluation must simultaneously achieve scalability, automatic scoring, and interpretable diagnosis. However, pursuing large-scale unified scores often leads to the conflation of different cognitive abilities. The authors argue the concern is not just "how much the model answers correctly," but "at which level of the cognitive process the model succeeds or fails."

**Goal**: This paper aims to construct a cognitively-driven bilingual multimodal benchmark. It utilizes Bloom’s taxonomy to cover six levels—Remember, Understand, Apply, Analyze, Evaluate, and Create—while exposing cross-lingual generalization capabilities through English-Arabic bilingual questions. It further distinguishes explicit output correctness from internal confidence distributions using two scoring methods.

**Key Insight**: Bloom's taxonomy, derived from educational psychology, naturally decomposes cognitive processes into hierarchical levels of increasing depth. By mapping this framework to image-question-answer tasks, each sample is associated not only with a task type but also a cognitive level, allowing evaluation results to be interpreted as a "cognitive profile" of the model.

**Core Idea**: Replace loose task collections with Bloom's taxonomy to organize VLM evaluation. Use English-Arabic bilingualism and the RAE/LBS dual-scoring approach to simultaneously diagnose cognitive level differences, cross-lingual disparities, and confidence calibration gaps.

## Method
BloomBench proposes a framework for evaluation construction and analysis rather than a new model architecture. Its key lies in mapping the abstract cognitive taxonomy to executable multimodal multiple-choice questions (MCQs) and forming a closed loop through automated generation, translation, quality verification, and dual-score evaluation.

### Overall Architecture
The workflow consists of four steps. First, the BloomBench taxonomy is defined, splitting the six Bloom levels into 106 specific leaf task nodes. Second, the system generates visual scenarios for each leaf node, retrieves real images, and generates open-ended VQA based on the images. Third, open-ended VQA is converted into four-option MCQs and translated into Modern Standard Arabic. Fourth, quality is controlled via LLM-as-a-judge and human verification sampling, followed by running Regex-based Answer Extraction (RAE) and Likelihood-based Scoring (LBS) across multiple open-source/closed-source VLMs.

The input consists of an image, a question, and four candidate answers; the output includes diagnostic results decomposed by language, cognitive level, model family, model size, and scoring method. This design renders BloomBench more like a "cognitive health checkup" than a simple leaderboard.

### Key Designs
1. **Hierarchical Mapping from Bloom Taxonomy to VLM Tasks**:

	- Function: Decomposes multimodal capabilities into six levels—Remember, Understand, Apply, Analyze, Evaluate, and Create—further refined into 106 leaf tasks.
	- Mechanism: Lower levels cover object, attribute, activity, symbol, and text recognition; middle levels cover compositional understanding, knowledge application, basic logic, contextual reasoning, and table/chart analysis; higher levels cover consistency/safety/quality evaluation and constrained creative selection. This ensures each question corresponds to a specific cognitive operation.
	- Design Motivation: Traditional benchmarks often mix different difficulties and cognitive processes into a single score, making it difficult to discern if a model fails due to visual perception issues, inability to apply rules, or lack of creative synthesis. Hierarchical mapping enhances error interpretability.

2.  **Semi-automated Data Generation and Mixed Quality Verification**:

	- Function: Controls question quality, image relevance, and bilingual semantic consistency while maintaining scale.
	- Mechanism: Gemini 2.5 Pro generates culture-aware scenarios and image keywords for each taxonomy leaf node, followed by visual question generation based on real web images. Another instruction model converts open QA into four-option MCQs, including a deceptive distractor, which is then translated into Arabic. Quality control uses LLM-as-a-judge for filtering, followed by stratified sampling of 969 instances across 106 nodes; Gemini 3 Pro flagged 15 suspicious samples, and human review confirmed these as errors, resulting in a 98.45% quality rate.
	- Design Motivation: Full manual construction cannot cover 7,747 bilingual samples, while purely automated generation risks producing unanswerable questions, visual irrelevance, or translation drift. Stratified sampling combined with human review balances cost and reliability.

3.  **RAE and LBS Dual Evaluation Protocols**:

	- Function: Distinguishes whether a model "eventually says the correct option" from whether its "internal probability distribution truly supports the correct answer."
	- Mechanism: RAE extracts A/B/C/D from the model's free-form output. LBS calculates the length-normalized log probability of each candidate answer given the image and question, formulated as: $$\text{NormalizedScore}(C_i)=\frac{1}{k}\sum_{j=1}^{k}\log P(w_j|I,Q,w_{<j})$$, then selecting the candidate with the highest score.
	- Design Motivation: Many models can output the correct letter via prompting but remain unstable at the probability level. LBS exposes issues of confidence calibration and reasoning consistency hidden behind successful formatting.

### Loss & Training
BloomBench does not involve training new models, thus there is no optimization loss. The "training strategy" during the construction phase refers to evaluation protocol design: data generation uses prompt engineering and agentic pipelines, quality control uses LLM judge + stratified human verification, and model evaluation uses a zero-shot setting with decoding temperature set to 0. Metrics are primarily based on accuracy, reported in both micro and macro formats to prevent category imbalance from masking weaknesses.

## Key Experimental Results

### Main Results
BloomBench contains 7,747 bilingual English-Arabic image-question-answer samples across 106 task types. The sample distribution across the six cognitive levels is shown below.

| Cognitive Level | Samples | Evaluation Significance |
|----------|--------|----------|
| Remember | 2,948 | Basic recognition and recall of objects, attributes, symbols, text, etc. |
| Understand | 1,592 | Understanding of relations, compositional semantics, emotions, and visual paraphrasing |
| Apply | 499 | Applying math, science, and logic knowledge to visual scenarios |
| Analyze | 1,431 | Contextual reasoning, structured data analysis, identifying anomalous attributes |
| Evaluate | 592 | Judgments on consistency, safety, and image quality |
| Create | 685 | Identifying the most plausible creative synthesis under constraints |
| Total | 7,747 | 106 taxonomy leaf nodes, bilingual (English/Arabic) |

Overall results show that under RAE (explicit answers), Gemma4-31B performs best. However, model rankings change significantly under LBS, indicating that "outputting the answer" and "probabilistically believing the answer" are distinct.

| Model | Eng RAE Micro | Eng LBS Micro | Ara RAE Micro | Ara LBS Micro | Key Observation |
|------|---------------|---------------|---------------|---------------|----------|
| Qwen2-VL-7B | 0.854 | 0.421 | 0.773 | 0.326 | Decent RAE, but weak LBS confidence |
| Qwen2.5-VL-7B | 0.869 | 0.654 | 0.792 | 0.503 | One of the best LBS stabilities |
| Gemma3-27B | 0.883 | 0.336 | 0.859 | 0.440 | High RAE, but significant English LBS drop |
| Gemma4-31B | 0.898 | 0.430 | 0.876 | 0.397 | Best overall RAE, LBS still suboptimal |
| GPT-4o mini | 0.824 | N/A | 0.769 | N/A | Closed-source models do not support LBS |

### Ablation Study
The paper lacks model architecture ablations but provides two valuable diagnostic comparisons: the difference between RAE and LBS for the same model, and the coverage gap between the BloomBench taxonomy and existing MMMU.

| Analysis Item | Result | Explanation |
|--------|------|------|
| Quality Verification | 969 samples checked, 15 errors, quality rate 98.45% | Stratified coverage of 106 nodes; data is reliable |
| MMMU Coverage Map | Analyze accounts for 66.4%; Create + Evaluate < 1.1% | Existing benchmarks favor expert knowledge/analysis over complete cognitive coverage |
| Zero-coverage Nodes | 45 taxonomy leaf nodes have no samples in MMMU | Capabilities like Ambiguity Resolution, Toxicity Detection, and Dialogue Generation are missing |
| Qwen2.5-VL-7B Variance | Eng 0.869 RAE → 0.654 LBS | Relatively stable; output and confidence are consistent |
| Gemma3-27B Variance | Eng 0.883 RAE → 0.336 LBS | Exposes surface-level output strength but weak probability calibration |

### Key Findings
- English performance is generally superior to Arabic, but the gap is not merely a translation issue; LBS is also affected by Arabic tokenization fertility and non-English probability priors.
- Understand and Evaluate levels approach or exceed 0.88 under RAE, suggesting current VLMs are strong in discriminative visual semantic understanding.
- Apply, Create, and Remember levels expose deeper flaws under LBS, suggesting models may rely on semantic association rather than stable factual recall, procedural application, or creative synthesis.
- The Gemma3 series shows good cross-lingual RAE consistency, but larger models exhibit inverse scaling in LBS, suggesting stronger instruction tuning does not necessarily yield better probability calibration.

## Highlights & Insights
- The most significant value is transforming "multimodal evaluation" into "cognitive level diagnosis." This ensures that model failures are not just low scores but can be pinpointed to specific levels like basic memory, procedural application, structural analysis, or creative synthesis.
- The RAE/LBS dual metric is highly insightful: RAE reflects how human users perceive model output, while LBS investigates whether the model's internal state truly prioritizes the correct answer. Large discrepancies indicate the model may have learned formatting well without reliable confidence distributions.
- The bilingual design goes beyond "adding a language" to directly challenge the extrapolation assumptions of English-centric evaluation. The degradation of Create and Apply in Arabic suggests that cross-lingual transfer of higher-order cognitive abilities is far more fragile than basic semantic understanding.

## Limitations & Future Work
- The number of evaluated models is restricted by GPU and closed-source API costs; future work should include a larger model pool, particularly those with different training corpora and visual encoding architectures.
- All questions are multiple-choice for easy automated scoring, but this does not fully cover open-ended generation, multi-step reasoning, or real-world interactive tasks. Future versions could incorporate short-answer, fill-in-the-blank, long-chain generation, or adaptive difficulty.
- Despite high sampling quality, not all 7,747 samples were manually verified; automated benchmarks may still harbor localized image failures, option ambiguities, or translation nuances.
- LBS is not entirely fair across different languages due to tokenization differences, particularly for morphologically rich languages like Arabic, where length normalization may still retain systematic biases.

## Related Work & Insights
- **vs MMMU**: MMMU excels in expert domain knowledge and analysis, but when mapped to the BloomBench taxonomy, Analyze accounts for 66.4%, while Create/Evaluate aggregate to less than 1.1%. BloomBench offers more balanced cognitive coverage but currently relies primarily on MCQs.
- **vs MMT-Bench / VLM2-Bench**: These benchmarks expand task coverage and fine-grained visual capabilities but remain organized primarily as task collections. BloomBench differs by defining the cognitive framework first and then generating tasks, making results better suited for capability profiling.
- **vs Arabic VLM benchmarks (e.g., CAMEL-Bench)**: While Arabic-specific benchmarks emphasize linguistic and cultural coverage, BloomBench integrates English and Arabic into the same cognitive taxonomy to compare cross-lingual cognitive transfer via isomorphic tasks.
- **Insights**: When conducting VLM/MLLM evaluation, researchers should move beyond single leaderboards and report results decomposed by cognitive levels, languages, and scoring mechanisms. For data construction, one can actively augment data for weak levels like Apply and Create.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Systematically organizes bilingual multimodal evaluation using Bloom's Taxonomy with a clear diagnostic perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers multiple VLMs, two languages, and two scoring methods, though the model pool is limited by resources.
- Writing Quality: ⭐⭐⭐⭐ Well-structured with clear methodology and discussion; large tables may require some cross-referencing.
- Value: ⭐⭐⭐⭐⭐ Highly valuable for building more interpretable and inclusive VLM evaluations, especially for cross-lingual multimodal diagnostic research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] VIGNETTE: Socially Grounded Bias Evaluation for Vision-Language Models](vignette_socially_grounded_bias_evaluation_for_vision-language_models.md)
- [\[CVPR 2026\] CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods](../../CVPR2026/multimodal_vlm/crosshoi-bench_a_unified_benchmark_for_hoi_evaluation_across_vision-language_mod.md)
- [\[ACL 2026\] Can MLLMs Reason Beyond Language? VisReason: A Comprehensive Benchmark for Vision-Centric Reasoning](can_mllms_reason_beyond_language_visreason_a_comprehensive_benchmark_for_vision-.md)
- [\[ACL 2026\] MMErroR: A Benchmark for Erroneous Reasoning in Vision-Language Models](mmerror_a_benchmark_for_erroneous_reasoning_in_vision-language_models.md)
- [\[ACL 2026\] Cross-Cultural Expert-Level Art Critique Evaluation with Vision-Language Models](cross-cultural_expert-level_art_critique_evaluation_with_vision-language_models.md)

</div>

<!-- RELATED:END -->
