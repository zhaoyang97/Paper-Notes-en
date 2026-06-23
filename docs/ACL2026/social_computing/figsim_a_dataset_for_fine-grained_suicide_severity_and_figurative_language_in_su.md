---
title: >-
  [Paper Note] FigSIM: A Dataset for Fine-grained Suicide Severity and Figurative Language in Suicide Memes
description: >-
  [ACL 2026][Social Computing][suicide memes] FigSIM constructs the first fine-grained multimodal dataset for suicide-related memes, annotating figurative phenomena, suicide severity, and suicide-related content. Experiments with 16 types of models verify that current models systematically underestimate high-severity risks involving metaphors, irony, and sarcasm.
tags:
  - ACL 2026
  - Social Computing
  - suicide memes
date: 2026-05-08
content_hash: ecf5f185006cee82
---
# FigSIM: A Dataset for Fine-grained Suicide Severity and Figurative Language in Suicide Memes

**Conference**: ACL2026 Findings  
**arXiv**: [2606.02523](https://arxiv.org/abs/2606.02523)  
**Code**: https://github.com/LiuliuChen/FigSIM  
**Area**: Social Computing / Safety and Mental Health NLP  
**Keywords**: suicide memes, fine-grained risk annotation, figurative language, multimodal moderation, mental health safety  

## TL;DR
FigSIM constructs the first fine-grained multimodal dataset for suicide-related memes, annotating figurative phenomena, suicide severity, and suicide-related content. Experiments with 16 types of models verify that current models systematically underestimate high-severity risks involving metaphors, irony, and sarcasm.

## Background & Motivation
**Background**: Mental health research on social media has long focused on depression, anxiety, and suicide risk in textual posts, with recent work addressing images and memes. Platform content moderation typically treats risky content as binary or coarse-grained safety categories, utilizing text classifiers, vision models, multimodal models, and LLM moderation interfaces.

**Limitations of Prior Work**: Suicide-related memes are unique: they may represent help-seeking, empathy, or coping, but can also contain distressing or potentially harmful content. The humor, irony, metaphors, and templated visual elements of memes can obscure true intent, making it difficult for both human and automated systems to assess severity. Existing datasets lack fine-grained annotations for suicide memes and specialized evaluations of "indirectly expressed risk."

**Key Challenge**: Content moderation must protect users—especially youth—from potentially harmful content; however, suicide-related expressions may also serve as help-seeking and peer support. Blunt removal or misjudgment may harm the expresser. To achieve effective moderation, it is necessary to identify the severity of the expression, the presence of figurative language, and potential harmful or protective factors.

**Goal**: The authors aim to establish a benchmark dataset supporting research and moderation strategy design, containing 1049 suicide-related memes, evaluated across three tasks: figurative language detection, suicide severity detection, and suicide-related content detection.

**Key Insight**: The paper integrates mental health safety with meme understanding. Instead of binary "harmful/harmless" labels, five annotation dimensions are designed: figurative phenomenon, suicide severity, suicide-related content, modality, and context. Severity labels refer to the C-SSRS, while content labels refer to online suicide communication guidelines such as #chatsafe and Mindframe.

**Core Idea**: To enable models to effectively serve suicide-related meme moderation, they must simultaneously model fine-grained severity, indirect expressions (metaphor/irony), and context dependencies arising from visual-textual complementarity.

## Method
FigSIM is a dataset and benchmark paper. Its core methodology is not a new model architecture but a reproducible evaluation framework involving data collection, annotation systems, expert calibration, model baselines, and error analysis.

### Overall Architecture
Data was sourced from r/SuicideMeme, a subreddit dedicated to sharing suicide-related memes. Submissions from April 2018 to December 2022 were collected using the Pushshift Reddit API. Images were filtered based on OCR readability (English text), near-duplicate removal, and retention of visual memes. From 1,967 filtered images, 1,050 were randomly sampled for annotation due to resource constraints; one was removed during quality control, resulting in 1,049 samples.

The annotation scheme includes five categories: figurative phenomenon (multi-label: metaphor, pun/double meaning, irony/sarcasm, none); suicide severity (ordinal single-label: wish to be dead, suicide ideation, suicide planning, suicide attempts, suicide death, none); suicide-related content (multi-label: stylized method/location depiction, naturalistic depiction, protective factors, harmful factors, third-person description, none); modality (text, image, or complementary); and context (necessity of external background knowledge).

Model evaluation covered text, image, multimodal, and specialized mental-health meme models. Supervised models used a 60:20:20 stratified split (633 train, 208 val, 208 test). `CrossEntropyLoss` was used for single-label tasks and `BCEWithLogitsLoss` for multi-label tasks, incorporating class-reweighted losses and threshold tuning for imbalance. MLLMs were evaluated via zero-shot and few-shot prompting.

### Key Designs

**1. Annotation system combining clinical scales and online communication guidelines: Refining meme risk into actionable multidimensional labels.**
Compressing suicide-related content into binary categories loses critical information. Memes involving suicide can vary from help-seeking to induction, requiring different moderation actions. FigSIM adapts severity labels from the C-SSRS and content labels from #chatsafe and Mindframe. This explicitly encodes the judgment that content impact depends on expression, severity, and context.

**2. Multi-round annotation process involving mental health experts: Utilizing calibration and adjudication to stabilize highly subjective tasks.**
Figurative expression and suicide severity are highly subjective. Direct crowdsourcing without calibration yields noisy labels, which is undesirable for sensitive topics. The process included two pilot rounds with psychologists, a calibration batch for inter-annotator agreement (IAA), and a large-scale phase where two annotators labeled independently with a third acting as an adjudicator for disagreements.

**3. Multi-type model baselines and bias analysis: Evaluating safety and reliability through error structures rather than average scores.**
Average F1 scores can hide fatal risks, such as the underestimation of high-severity content. The study evaluated text-based (BERT/MentalBERT/RoBERTa), vision-based (ResNet/ViT/DINOv2), multimodal (CLIP/BLIP-2), and MLLM models. Analysis focused on per-label F1, over/underprediction of severity, and errors related to context or modality. This revealed a systematic underestimation of risk when figurative language is present.

### Loss & Training
Supervised baselines employed `CrossEntropyLoss` for single-label classification (severity, modality, context) and `BCEWithLogitsLoss` for multi-label tasks (figurative, content). Class-reweighted losses were used for imbalance, with optimized decision thresholds for multi-label tasks. MLLMs used zero-shot and few-shot prompts based on label definitions and guidelines, with 6 examples per task for few-shot.

## Key Experimental Results

### Main Results

| Task | Best Model | Macro-F1 | Weighted-F1 | Observation |
|--------|------|------|----------|------|
| Figurative detection | Claude-sonnet-4-5 zero-shot | 70.21±0.82 | 80.69±0.42 | Irony/sarcasm is easiest; pun and metaphor are harder. |
| Suicide severity detection | Gemini-3-pro few-shot | 71.60±2.43 | 71.73±2.52 | Few-shot significantly aids fine-grained severity. |
| Suicide-related content detection | Gemini-3-pro zero-shot | 58.51±4.06 | 58.16±2.53 | Hardest task; low F1 for harmful factors/naturalistic depiction. |
| CLIP | Three-task Macro-F1 | 62.06 / 49.97 / 33.70 | 80.21 / 52.10 / 43.75 | Some figurative capability but insufficient safety semantics. |
| M3H | Suicide severity Macro-F1 | 61.47±1.74 | 62.58±1.95 | Mental-health meme transfer is effective but ~10pts below MLLMs. |

### Dataset & Annotation Quality

| Dimension | Metric | Value | Note |
|------|---------|------|------|
| Scale | Final Samples | 1049 | 1 sample removed after QC. |
| Split | Train / Val / Test | 633 / 208 / 208 | Stratified split. |
| IAA: Suicide Severity | Cohen's κ | 0.65 | Substantial. |
| IAA: Modality | Cohen's κ | 0.88 | Almost perfect. |
| IAA: Context | Cohen's κ | 0.58 | Moderate. |
| IAA: Figurative Phenomenon | Avg Cohen's κ | 0.56 | Moderate (reflects subjectiveness). |
| IAA: Suicide-related Content | Avg Cohen's κ | 0.63 | Substantial. |

### Error Analysis & Moderation Findings

| Analysis Item | Value | Conclusion |
|------|------|------|
| Azure flag rate for "None" severity | 0.304 | Moderation models false alarm on low/no risk content. |
| Azure flag rate for "Suicide attempts" | 0.661 | Generally more sensitive as severity increases. |
| OpenAI flag rate for "Suicide attempts" | 0.575 | Follows severity trend but less sensitive than Azure. |
| Severity underprediction (Irony/Sarcasm) | absent: 0.429, present: 0.604 | Irony leads to more frequent underestimation. |
| Severity underprediction (Metaphor) | absent: 0.551, present: 0.833 | Metaphor samples are few but show strongest underestimation. |
| Context-required errors | severity 5/26, fig. 16/26, content 12/26 | External background significantly increases difficulty. |

### Key Findings
- Text models usually outperform image-only models in severity and content tasks, indicating OCR text carries substantial risk information.
- MLLMs are overall the strongest across tasks, though open-source versions and safety filters may cause refusal or unstable performance.
- Few-shot prompting is not always beneficial: in figurative detection, it may decrease performance due to prompt sensitivity, whereas severity benefits from examples.
- The critical safety issue is high-severity underprediction, particularly when severity is expressed via metaphor, irony, or visual puns.

## Highlights & Insights
- **Precise Problem Definition**: Separating severity, figurative language, and content types aligns better with actual moderation decision-making than binary classification.
- **Cautious Annotation for Sensitive Topics**: The involvement of psychologists and suicide prevention researchers, pilot rounds, and adjudication reduced risks of poor annotation.
- **Value of Error Analysis**: Identifying model underestimation of high-severity content as a core risk linked to figurative language provides a clear target for future model design.
- **Moderation Blind Spots**: While APIs are more sensitive to high severity, figurative memes receive lower scores at the same severity level, suggesting indirect expressions may bypass safety systems.

## Limitations & Future Work
- Data is from a single subreddit; community-specific humor and norms may not generalize to platforms like TikTok, X, or non-English communities.
- Annotations are based on meme content, assuming indirect self-expression; they do not represent clinical diagnoses or actual creator intent.
- "Fine-grained" labels still cannot capture every nuance of suicide-related expression.
- Moderate κ for figurative phenomenon and context indicates high task subjectivity; future work could include rationales or cross-cultural annotators.
- Data release limited to annotations and retrieval scripts to comply with privacy and platform policies, which may affect reproducibility if URLs expire.

## Related Work & Insights
- **Comparison to Hateful Memes**: Existing benchmarks focus on hate/toxicity and lack clinical severity for suicide risk; FigSIM is tailored for mental health safety.
- **Comparison to Text-based Detection**: Text-only models fail to capture visual templates and irony; FigSIM demonstrates the need for multimodal and context-aware understanding.
- **Comparison to Mental-health Meme Model (M3H)**: While M3H has transferability, it lags behind the best MLLMs by ~10 points, suggesting general depression/anxiety knowledge does not fully cover suicide-specific risks.
- **Insight**: Moderation models must be explicitly trained to map "indirect expressions to severity" rather than just recognizing suicide-related keywords.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First fine-grained suicide meme dataset with domain-specific labels.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Broad baseline coverage and solid error analysis; however, single-platform data limits generalization.
- Writing Quality: ⭐⭐⭐⭐☆ Clear structure with responsible ethics and limitations sections.
- Value: ⭐⭐⭐⭐⭐ High benchmark value for social platform safety, mental health NLP, and multimodal moderation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] PSK@EEUCA 2026: Fine-Tuning Large Language Models with Synthetic Data Augmentation for Multi-Class Toxicity Detection in Gaming Chat](pskeeuca_2026_fine-tuning_large_language_models_with_synthetic_data_augmentation.md)
- [\[ACL 2026\] Prompt-Level Distillation: A Non-Parametric Alternative to Model Fine-Tuning for Efficient Reasoning](prompt-level_distillation_a_non-parametric_alternative_to_model_fine-tuning_for_.md)
- [\[ACL 2026\] BITS Pilani at SemEval-2026 Task 9: Structured Supervised Fine-Tuning with DPO Refinement for Polarization Detection](bits_pilani_at_semeval-2026_task_9_structured_supervised_fine-tuning_with_dpo_re.md)
- [\[ACL 2026\] Persona-E2: A Human-Grounded Dataset for Personality-Shaped Emotional Responses to Textual Events](persona-e2_a_human-grounded_dataset_for_personality-shaped_emotional_responses_t.md)
- [\[CVPR 2025\] Project-Probe-Aggregate: Efficient Fine-Tuning for Group Robustness](../../CVPR2025/social_computing/project-probe-aggregate_efficient_fine-tuning_for_group_robustness.md)

</div>

<!-- RELATED:END -->
