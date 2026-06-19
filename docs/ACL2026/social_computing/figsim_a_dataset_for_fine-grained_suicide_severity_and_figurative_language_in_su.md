---
title: >-
  [Paper Note] FigSIM: A Dataset for Fine-grained Suicide Severity and Figurative Language in Suicide Memes
description: >-
  [ACL 2026][Social Computing][suicide memes] FigSIM constructs the first fine-grained multimodal dataset for suicide-related memes, annotating figurative phenomena, suicide severity, and suicide-related content. Evaluations across 16 model classes demonstrate that current models systematically underestimate risks involving irony, metaphors, and high-severity cont
tags:
  - ACL 2026
  - Social Computing
  - suicide memes
date: 2026-05-08
content_hash: 14b9820b73876478
---
# FigSIM: A Dataset for Fine-grained Suicide Severity and Figurative Language in Suicide Memes

**Conference**: ACL2026 Findings  
**arXiv**: [2606.02523](https://arxiv.org/abs/2606.02523)  
**Code**: https://github.com/LiuliuChen/FigSIM  
**Area**: Social Computing / Safety and Mental Health NLP  
**Keywords**: suicide memes, fine-grained risk annotation, figurative language, multimodal moderation, mental health safety  

## TL;DR
FigSIM constructs the first fine-grained multimodal dataset for suicide-related memes, annotating figurative phenomena, suicide severity, and suicide-related content. Evaluations across 16 model classes demonstrate that current models systematically underestimate risks involving irony, metaphors, and high-severity content.

## Background & Motivation
**Background**: Mental health research on social media has long focused on text-based posts regarding depression, anxiety, and suicide risk, with recent efforts expanding to images and memes. Platform moderation typically treats risky content as binary or coarse safety categories, utilizing text classifiers, vision models, multimodal models, and LLM moderation APIs.

**Limitations of Prior Work**: Suicide-related memes are unique: they may serve as calls for help, empathy, or coping mechanisms, yet can also contain distressing or potentially harmful content. Humor, irony, metaphors, and templated visual elements in memes obscure true intent, making severity assessment difficult for both humans and automated systems. Existing datasets lack fine-grained annotations for suicide memes and specialized evaluations of "indirectly expressed risk."

**Key Challenge**: Content moderation must protect users, particularly youth, from harmful content while recognizing that suicidal expressions can also involve help-seeking and peer support. Indiscriminate deletion or misjudgment can harm the expresser. Fine-grained moderation requires understanding severity, the presence of figurative language, and potential harmful or protective factors.

**Goal**: The authors aim to establish a benchmark dataset supporting research and moderation strategy design, containing 1049 suicide-related memes evaluated across three tasks: figurative language detection, suicide severity detection, and suicide-related content detection.

**Key Insight**: The paper integrates mental health safety with meme understanding, moving beyond "harmful/harmless" labels to design five annotation dimensions: figurative phenomenon, suicide severity, suicide-related content, modality, and context. Severity labels reference the C-SSRS, while content labels reference online communication guidelines such as #chatsafe and Mindframe.

**Core Idea**: Effective moderation of suicide memes necessitates the simultaneous modeling of fine-grained severity, indirect expressions like metaphors/irony, and context dependencies driven by visual-textual complementarity.

## Method
FigSIM is a dataset and benchmark paper. Its methodology focuses on establishing a reproducible evaluation framework through data collection, annotation schemes, expert calibration, model baselines, and error analysis.

### Overall Architecture
Data was sourced from r/SuicideMeme, a subreddit dedicated to suicide-related memes. Submissions from April 2018 to December 2022 were collected via the Pushshift Reddit API. Images were filtered for OCR readability, English text, and near-duplicate removal, while pure visual memes were retained. This yielded 1967 images, with 1050 randomly sampled for annotation (1049 remained after quality control).

The annotation scheme includes five categories: figurative phenomenon (multi-label: metaphor, pun/double meaning, irony/sarcasm, none); suicide severity (ordered single-label: wish to be dead, suicide ideation, suicide planning, suicide attempts, suicide death, none); suicide-related content (multi-label: stylized method/location depiction, naturalistic depiction, protective factors, harmful factors, third-person description, none); modality (text-only, image-only, or complementary); and context (necessity of external background knowledge).

Model evaluation covered text, image, multimodal, and specialized mental-health meme models. Supervised models used a $60:20:20$ stratified split. Single-label tasks used `CrossEntropyLoss`, while multi-label tasks used `BCEWithLogitsLoss`, with class-reweighted losses and threshold adjustments for imbalance. MLLMs were evaluated via zero-shot and few-shot prompting.

### Key Designs

**1. Multi-dimensional annotation combining clinical scales and online guidelines: Refining meme risk into actionable labels.**  
Collapsing suicide-related content into binary categories loses critical information. FigSIM adapts the clinical C-SSRS scale for severity and guidelines like #chatsafe and Mindframe for content (e.g., distinguishing stylized depictions from harmful factors). This allows research to pinpoint exactly which types of content require specific interventions.

**2. Multi-round annotation with mental health experts: Stabilizing high-subjectivity tasks.**  
Expert involvement included two pilot rounds and a calibration batch involving registered psychologists. Large-scale annotation involved two independent annotators and a third for adjudication. This ensures judgment remains within clinical guidelines, enhancing conceptual clarity and ethical robustness.

**3. Error and bias analysis: Evaluating reliability through error structures rather than average scores.**  
Average F1 scores can mask dangerous underestimations of high-severity content. The analysis decomposes per-label F1, over/underprediction of severity, and errors related to context or modality. This revealed that models systematically underestimate high severity when expressed through metaphors or irony.

### Loss & Training
Supervised baselines employed `CrossEntropyLoss` for single-label tasks (severity, modality, context) and `BCEWithLogitsLoss` for multi-label tasks (figurative, content). To handle class imbalance, class-reweighted losses were used alongside tuned decision thresholds. Data was split $60:20:20$ ($633/208/208$). MLLMs used zero/few-shot prompts based on label definitions; few-shot prompts included 6 random examples covering all labels.

## Key Experimental Results

### Main Results

| Task | Best Model | Macro-F1 | Weighted-F1 | Observation |
|:---|:---|:---|:---|:---|
| Figurative detection | Claude-sonnet-4-5 zero-shot | $70.21 \pm 0.82$ | $80.69 \pm 0.42$ | Irony/sarcasm easiest; pun and metaphor harder. |
| Suicide severity detection | Gemini-3-pro few-shot | $71.60 \pm 2.43$ | $71.73 \pm 2.52$ | Few-shot improves fine-grained severity. |
| Suicide-related content detection | Gemini-3-pro zero-shot | $58.51 \pm 4.06$ | $58.16 \pm 2.53$ | Hardest task; harmful factors and naturalistic depiction have low F1. |
| CLIP | Three-task Macro-F1 | $62.06 / 49.97 / 33.70$ | $80.21 / 52.10 / 43.75$ | Moderate figurative ability; lacks safety semantics. |
| M3H | Suicide severity Macro-F1 | $61.47 \pm 1.74$ | $62.58 \pm 1.95$ | Domain transfer works but lags ~10 points behind MLLMs. |

### Key Findings
- Text models generally outperform image-only models in severity and content detection, indicating that OCR text carries substantial risk information in memes.
- MLLMs are overall the strongest but suffer from inconsistent performance or refusal due to safety filters.
- The most critical safety failing is the underprediction of high-severity content, especially when severity is conveyed through metaphors, irony, or visual puns.

## Highlights & Insights
- **Precise Problem Definition**: Decoupling severity, figurative language, and content types aligns better with real-world moderation decision-making than binary classification.
- **Ethical Rigor**: The involvement of psychologists and youth suicide prevention researchers minimizes the risks associated with labeling sensitive topics.
- **Valuable Bias Analysis**: Identifying the correlation between figurative language and severity underestimation provides a clear target for improving future safety models.
- **Moderation Gaps**: While standard moderation APIs are generally more sensitive to higher severity, they score figurative memes lower for the same severity level, suggesting indirect expressions can bypass safety systems.

## Limitations & Future Work
- **Source Material**: Data is limited to one subreddit; cultural norms and humor styles may not generalize to platforms like TikTok, X, or non-English communities.
- **Proxy Interpretation**: Annotations reflect the meme content rather than clinical diagnoses or the creator's actual state.
- **Subjectivity**: The moderate Cohen's $\kappa$ for figurative language and context indicates the inherent difficulty and subjectivity of these tasks.
- **Distribution Strategy**: Releasing only annotations and retrieval scripts (for privacy/policy compliance) may lead to issues with link rot or image availability for future replication.

## Related Work & Insights
- **Comparison with Hateful Memes**: Unlike toxic content datasets, FigSIM focuses on clinical severity of self-harm risks.
- **Comparison with Unified Models**: Specialized models like M3H exhibit some transferability but fail to capture suicide-specific nuances, trailing behind best-in-class MLLMs.
- **Mechanism Insight**: Content moderation models must be trained to map "indirect expressions to severity levels" rather than simply flagging the presence of suicide-related keywords.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ 
- Experimental Thoroughness: ⭐⭐⭐⭐☆ 
- Writing Quality: ⭐⭐⭐⭐☆ 
- Value: ⭐⭐⭐⭐⭐ 

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
