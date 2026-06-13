---
title: >-
  [Paper Note] FigSIM: A Dataset for Fine-grained Suicide Severity and Figurative Language in Suicide Memes
description: >-
  [ACL2026 Findings][Social Computing][suicide memes] FigSIM constructs the first fine-grained multimodal dataset for suicide-related memes, annotating figurative phenomena, suicide severity…
tags:
  - "ACL2026 Findings"
  - "Social Computing"
  - "suicide memes"
  - "fine-grained risk annotation"
  - "figurative language"
  - "multimodal moderation"
  - "mental health safety"
date: 2026-05-08
content_hash: c9da7142d0612eb0
---

# FigSIM: A Dataset for Fine-grained Suicide Severity and Figurative Language in Suicide Memes

**Conference**: ACL2026 Findings  
**arXiv**: [2606.02523](https://arxiv.org/abs/2606.02523)  
**Code**: https://github.com/LiuliuChen/FigSIM  
**Area**: Social Computing / Safety & Mental Health NLP  
**Keywords**: suicide memes, fine-grained risk annotation, figurative language, multimodal moderation, mental health safety  

## TL;DR
FigSIM constructs the first fine-grained multimodal dataset for suicide-related memes, annotating figurative phenomena, suicide severity, and suicide-related content. Using 16 types of models, it verifies that current models systematically underestimate risks involving metaphors, irony, and high severity.

## Background & Motivation
**Background**: Mental health research on social media has long focused on depression, anxiety, and suicide risk in textual posts, with some work starting to process images and memes. Platform content moderation often treats risky content as binary or coarse-grained safety categories, with mainstream models including text classifiers, vision models, multimodal models, and LLM moderation interfaces.

**Limitations of Prior Work**: Suicide-related memes are unique: they may involve seeking help, empathy, or coping, but can also contain distressing or potentially harmful content. The humor, irony, metaphors, and templated visual elements of memes can obscure true intent, making it difficult for both human and automated systems to judge severity. Existing datasets lack fine-grained annotations specifically for suicide memes and lack specialized evaluations of models' ability to understand "indirectly expressed risks."

**Key Challenge**: Content moderation needs to protect users, especially younger ones, from potentially harmful content; however, suicide-related expressions can also be cries for help and peer support. Crude removal or misjudgment may harm the expresser. To achieve finer moderation, it is essential to first understand the expressed severity, whether figurative language is used, and the presence of potentially harmful or protective factors.

**Goal**: The authors aim to establish a benchmark dataset to support research and moderation policy design, containing 1049 suicide-related memes, and evaluate models on three tasks: figurative language detection, suicide severity detection, and suicide-related content detection.

**Key Insight**: The paper combines mental health safety with meme understanding, providing five annotation dimensions instead of just a "harmful/harmless" label: figurative phenomenon, suicide severity, suicide-related content, modality, and context. Severity annotation refers to C-SSRS, and content annotation refers to online suicide communication guidelines such as #chatsafe and Mindframe.

**Core Idea**: To enable models to truly serve suicide-related meme moderation, one must simultaneously model fine-grained severity, indirect expressions like metaphors/irony, and context dependencies arising from visual-text complementarity.

## Method
FigSIM is a dataset and benchmark paper. Its core contribution is not a new model but a reusable evaluation framework across data collection, annotation systems, expert calibration, model baselines, and error analysis.

### Overall Architecture
Data was sourced from r/SuicideMeme, a subreddit dedicated to sharing suicide-related memes. The authors used the Pushshift Reddit API to collect submissions from April 2018 to December 2022, extracting image URLs and downloading them. Filtering rules included: OCR readability and English text, removal of near-duplicate memes, and retention of purely visual memes. After filtering, 1967 images were obtained; due to annotation resource constraints, 1050 were randomly sampled for annotation. One image was eventually removed due to quality control, resulting in 1049 data points.

The annotation scheme includes five categories: figurative phenomenon is multi-label (metaphor, pun/double meaning, irony/sarcasm, none); suicide severity is ordinal single-label (wish to be dead, suicide ideation, suicide planning, suicide attempts, suicide death, none); suicide-related content is multi-label (stylized method/location depiction, naturalistic depiction, protective factors, harmful factors, third-person description, none); modality (text, image, or complementary); and context (necessity of external background knowledge).

Model evaluation covers text, image, multimodal, and specialized mental-health meme models. Supervised models used a 60:20:20 stratified split (train: 633, val: 208, test: 208). CrossEntropyLoss was used for single-label tasks and BCEWithLogitsLoss for multi-label tasks, with class-reweighted losses and threshold adjustments for imbalance. MLLMs used zero-shot/few-shot prompting.

### Key Designs
1. **Annotation system combining clinical scales and online communication guidelines**:
	- **Function**: Deconstructs mental health risks in memes into actionable labels rather than remaining at a binary "suicide-related" level.
	- **Mechanism**: Severity labels draw from C-SSRS, adding "suicide death"; suicide-related content labels draw from #chatsafe and Mindframe to distinguish method/location depictions, protective factors, harmful factors, and third-person descriptions.
	- **Design Motivation**: The impact of suicide-related content depends on expression, severity, and context. Fine-grained labels help research which content requires different levels of moderation or intervention.

2. **Multi-round annotation process involving mental health experts**:
	- **Function**: Improves annotation consistency for sensitive tasks and confines subjective judgment within clear guidelines.
	- **Mechanism**: Two pilot rounds (50 images each) involved registered psychologists and researchers; consistency was checked via a calibration batch; for large-scale annotation, two independent annotators labeled each item, with a third annotator adjudicating disagreements.
	- **Design Motivation**: Figurative expression and suicide severity are highly subjective and cannot be left to uncalibrated crowdsourcing. Expert participation enhances conceptual clarity and ethical robustness.

3. **Multi-type model baselines and deviation analysis**:
	- **Function**: Systematically determines which models are more reliable for suicide meme understanding and reveals error types.
	- **Mechanism**: Evaluates BERT/MentalBERT/RoBERTa, ResNet/ViT/DINOv2, CLIP/BLIP-2, various MLLMs, and task-relevant models like Yadav et al. and M3H; subsequently analyzes per-label F1, severity over/underprediction, and context-required/modality-related errors.
	- **Design Motivation**: Reporting only average F1 obscures critical safety risks. For content moderation, underestimation of high severity is more concerning than false positives in low-severity cases.

### Loss & Training
In supervised baselines, single-label classification (suicide severity, modality/context) uses CrossEntropyLoss; multi-label tasks (figurative phenomenon, suicide-related content) use BCEWithLogitsLoss. To handle class imbalance, authors use class-reweighted losses and adjust decision thresholds for multi-label tasks. Data is split 60:20:20 (633/208/208) while maintaining distribution across annotation dimensions. MLLMs are evaluated without fine-tuning, using zero-shot/few-shot prompts based on label definitions and guidelines; few-shot prompts randomly select 6 examples per task covering all labels.

## Key Experimental Results

### Main Results
| Task | Best Model | Macro-F1 | Weighted-F1 | Observation |
|--------|------|------|----------|------|
| Figurative detection | Claude-sonnet-4-5 zero-shot | 70.21±0.82 | 80.69±0.42 | Irony/sarcasm is easiest to identify; pun and metaphor are harder |
| Suicide severity detection | Gemini-3-pro few-shot | 71.60±2.43 | 71.73±2.52 | Few-shot clearly benefits fine-grained severity |
| Suicide-related content detection | Gemini-3-pro zero-shot | 58.51±4.06 | 58.16±2.53 | Hardest among three tasks; harmful factors and naturalistic depiction have lower F1 |
| CLIP | Three tasks Macro-F1 | 62.06 / 49.97 / 33.70 | 80.21 / 52.10 / 43.75 | Shows some figurative capability but lacks safety semantics |
| M3H | Suicide severity Macro-F1 | 61.47±1.74 | 62.58±1.95 | Mental-health meme transfer is effective but ~10 points lower than best MLLM |

### Dataset & Annotation Quality
| Dimension | Metric | Value | Description |
|------|---------|------|------|
| Data Scale | Final Samples | 1049 | 1 sample removed from 1050 annotated samples due to QC failure |
| Split | Train / Val / Test | 633 / 208 / 208 | Stratified split |
| IAA: Suicide Severity | Cohen's κ | 0.65 | Substantial |
| IAA: Modality | Cohen's κ | 0.88 | Almost perfect |
| IAA: Context | Cohen's κ | 0.58 | Moderate |
| IAA: Figurative Phenomenon | Avg Cohen's κ | 0.56 | Moderate, reflecting subjectivity of metaphor/irony |
| IAA: Suicide-related Content | Avg Cohen's κ | 0.63 | Substantial |

### Key Findings
- Text models generally outperform image-only models in suicide severity and content detection, indicating that OCR text in memes carries significant risk information.
- MLLMs are overall strongest across three tasks, but open-source MLLMs and certain safety filtering mechanisms can lead to refusals or unstable performance.
- Few-shot prompting is not always effective: in figurative detection, it may decrease performance due to prompt sensitivity/subjectivity, but suicide severity benefits more from examples.
- The most critical safety issue is underprediction of high severity, especially when it is expressed through metaphors, irony, or visual puns.

## Highlights & Insights
- **Precise problem definition**: The paper avoids simplifying suicide memes into "suicide/non-suicide" and instead separates severity, figurative language, and content types, aligning better with real-world moderation decisions.
- **Prudent annotation process**: The involvement of psychologists, youth suicide prevention researchers, pilot rounds, adjudication, and ethical approvals mitigates risks of crude annotation.
- **Error analysis value**: Model underestimation of high-severity content is a core platform risk; the paper links this bias to figurative language, providing clear targets for future model design.
- **Moderation analysis reveals API blind spots**: Moderation models generally flag more as severity increases, but figurative memes often receive lower scores for the same severity, suggesting indirect expressions may bypass safety systems.

## Limitations & Future Work
- Data comes from a single subreddit; community culture, humor norms, and expression styles might not generalize to TikTok, X, Instagram, forums, or non-English communities.
- Annotation is based solely on meme content, assuming it reflects indirect self-expression unless clearly third-person; it is not a clinical judgment and cannot infer the creator's actual mental state.
- "Fine-grained" is relative to existing binary classifications; labels still cannot cover all nuances of suicide-related expression.
- The κ for figurative phenomenon and context is only moderate, indicating high task subjectivity. Future work could include explanation annotation, more expert adjudication, and cross-cultural comparisons.
- Data release only provides annotations and a retrieval script for privacy and compliance, but this may lead to issues like disappearing images or broken URLs for replication.

## Related Work & Insights
- **vs Hateful Memes / toxic meme benchmarks**: These focus on hate/offensive content and do not handle clinical severity of suicide risks; FigSIM's label system is better suited for mental health safety.
- **vs Text-based suicide risk detection**: Text models cannot handle visual templates, irony, or image-text complementarity; FigSIM proves these require multimodal and context-aware understanding.
- **vs mental-health meme model M3H**: M3H shows transferability in suicide severity but lags ~10 points behind the best models, indicating that knowledge from depression/anxiety memes cannot directly cover suicide-specific risks.
- **Key Insight**: Moderation models need explicit training on the mapping from "indirect expression to severity," rather than just identifying suicide-related words.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First fine-grained suicide meme dataset; task definitions and label systems are highly domain-specific.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Extensive baseline coverage and solid error analysis; however, the data source is limited, and cross-platform generalization remains to be verified.
- **Writing Quality**: ⭐⭐⭐⭐☆ Clear structure; ethics and limitations are handled with restraint; note that some model names refer to future systems given the timeline.
- **Value**: ⭐⭐⭐⭐⭐ High benchmark value for social platform safety, mental health NLP, and multimodal moderation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PSK@EEUCA 2026: Fine-Tuning Large Language Models with Synthetic Data Augmentation for Multi-Class Toxicity Detection in Gaming Chat](pskeeuca_2026_fine-tuning_large_language_models_with_synthetic_data_augmentation.md)
- [\[ACL 2026\] Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning](prompt-level_distillation_a_non-parametric_alternative_to_model_fine-tuning_for_.md)
- [\[ACL 2026\] Persona-E2: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events](persona-e2_a_human-grounded_dataset_for_personality-shaped_emotional_responses_t.md)
- [\[ACL 2026\] BITS Pilani at SemEval-2026 Task 9: Structured Supervised Fine-Tuning with DPO Refinement for Polarization Detection](bits_pilani_at_semeval-2026_task_9_structured_supervised_fine-tuning_with_dpo_re.md)
- [\[NeurIPS 2025\] AVerImaTeC: A Dataset for Automatic Verification of Image-Text Claims with Evidence from the Web](../../NeurIPS2025/social_computing/averimatec_a_dataset_for_automatic_verification_of_image-text_claims_with_eviden.md)

</div>

<!-- RELATED:END -->
