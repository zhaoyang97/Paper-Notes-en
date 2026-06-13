---
title: >-
  [Paper Note] Beyond Screenshots: Evaluating VLMs' Understanding of UI Animations
description: >-
  [ACL 2026][Multimodal VLM][UI Animation] The authors construct AniMINT, the first UI animation evaluation benchmark containing 300 densely annotated animation videos with labels from 3 experts and 300 users. Systematic t…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "UI Animation"
  - "VLM Evaluation"
  - "AniMINT"
  - "Rhetorical Structure"
  - "Motion Blending"
date: 2026-05-08
content_hash: e9829c3c220302f1
---

# Beyond Screenshots: Evaluating VLMs' Understanding of UI Animations

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.26148](https://arxiv.org/abs/2604.26148)  
**Code**: <https://github.com/publicationacc/AniMINT>  
**Area**: Multimodal VLM / UI Understanding / Evaluation  
**Keywords**: UI Animation, VLM Evaluation, AniMINT, Rhetorical Structure, Motion Blending

## TL;DR
The authors construct AniMINT, the first UI animation evaluation benchmark containing 300 densely annotated animation videos with labels from 3 experts and 300 users. Systematic testing of 9 SOTA VLMs reveals that while basic motion effects can be identified, a significant gap remains between VLMs and humans in categorizing animation purposes and high-level semantic interpretation. Enhancing Gemini-2.5-Flash with the Motion-Context-Perceptual Cue (MCPC) framework simultaneously improves performance in both classification and interpretation.

## Background & Motivation

**Background**: UI agents (e.g., GPT-Operator, Mind2Web) require holistic perception of user interfaces. However, existing VLM research on UI understanding focuses almost exclusively on static screenshots, such as button identification, layout parsing, and UI semantics.

**Limitations of Prior Work**: In modern UIs, animations serve core communicative functions rather than mere decoration—such as a bouncing MacOS dock icon conveying notifications, a shaking password field indicating input errors, or loading animations suggesting status progress. This information is often captured only in motion and is completely missed by static frames. If VLM agents only perceive screenshots, they lose approximately 30-50% of the feedback channels between the user and the system.

**Key Challenge**: "The meaning of animation is in the motion, not the frames" ("motion that is drawn, not drawings that move"). However, VLM inputs are typically either single frames or sparsely sampled videos, a structure that inherently struggles to capture brief, spatially localized, and semantically abstract UI motions.

**Goal**: (1) Provide the first UI animation evaluation set covering mobile, web, and desktop platforms with three levels of annotation: motion effects, functional purpose, and semantic interpretation. (2) Systematically benchmark the performance limits of 9 mainstream VLMs. (3) Explore which signal enhancements (motion blending, context, or captions) significantly improve performance.

**Key Insight**: Leveraging existing UI/UX taxonomies (7 purposes $\times$ 7 basic motion effects) to build multi-level annotations. By recruiting 3 experts for purpose labels and 300 Prolific users to provide 10 independent natural language interpretations per animation, the work creates a dual expert-crowd perspective.

**Core Idea**: Designing the evaluation to align directly with the linguistic framework of the UI design community allows for testing both whether a VLM can perceive motion and whether it can understand why the animation exists as a human would.

## Method

### Overall Architecture
The work proceeds in two stages: (1) **AniMINT Dataset Construction**—300 UI animation videos (primarily mobile, sourced from Top 100 App Store/Google Play apps) with multi-level annotations (timestamp, ROI, interaction context, purpose category, and 10 independent semantic interpretations). (2) **VLM Systematic Evaluation & Enhancement Exploration**—Testing 9 VLMs across three RQs: identifying basic motion effects (RQ1), categorizing animation purposes (RQ2), and interpreting animation semantics (RQ3). Subsequently, the MCPC three-factor probe is used to locate bottlenecks and verify enhancement effects.

### Key Designs

1.  **Three-Tier AniMINT Annotation Protocol**:
    - **Function**: Supports evaluation at three granularities using the same animation segments: low-level motion recognition, mid-level purpose classification, and high-level semantic interpretation to identify VLM bottlenecks.
    - **Mechanism**: Each animation is standardized to 480p resolution with 10 fps resampling, using green bboxes to mark animation ROIs to reduce interference. Three UI/UX experts used majority voting to label one of 7 purposes (Transition, Demonstration, Guidance, Feedback, Visualization, Highlight, Aesthetic; consensus reached after discussion with Krippendorff $\alpha=0.78$). 300 Prolific users each labeled 10 videos, resulting in 10 independent semantic interpretations per video (3,000 total user responses). Videos were manually screened for harmful content before upload.
    - **Design Motivation**: A single label cannot capture the rich semantics of animation. Combining the expert perspective for purpose with the crowd perspective for semantics preserves professional judgment while reflecting the diversity of actual user understanding. 10 independent interpretations also allow for evaluating "semantic alignment distribution" rather than single-point comparisons.

2.  **Three Progressive RQs + GPT-judge Evaluation Protocol**:
    - **Function**: Decomposes the abstract question of "whether VLMs understand animation" into three independently quantifiable sub-problems. RQ1 tests perception using 7 categories of pure geometric motion effects (move, rotate, size, color, fade, blur, morph). RQ2 tests 7 purpose categories (reporting accuracy and macro F1). RQ3 tasks VLMs with generating free-text interpretations, compared against human responses for 0-5 semantic similarity.
    - **Mechanism**: RQ1 uses a stationary square with a single motion as a controlled stimulus, averaging results over 10 randomized option orderings per model. For RQ2/RQ3, VLMs receive animations accompanied by context (application/task), user input (action type), and green bbox ROI markers. RQ3 uses GPT-5-mini as a judge with a prompt that avoids length bias, comparing VLM output against a "consensus response" summarized from the 10 human responses.
    - **Design Motivation**: VLMs may differ drastically in their ability to "see motion," "classify purpose," and "write semantics." Separating these allows for precise bottleneck identification. GPT-judge with a standardized rubric (5=fully equivalent / 0=unrelated) is a best practice for capturing semantic alignment better than surface metrics like BLEU.

3.  **Motion-Context-Perceptual Cue (MCPC) Probes**:
    - **Function**: Decomposes "VLM animation perception" into three complementary signals: Motion blending (stacking the last 6 frames with decreasing opacity, inspired by Phosphor afterglow), Context (interaction context and user input), and Perceptual caption (textual description of the animation), testing their combinations on RQ2/RQ3.
    - **Mechanism**: Using Gemini-2.5-Flash as the backbone. The base setting provides only sampled frames. Combinations of M, C, and P are added incrementally, re-running RQ2 and RQ3 for each. Motion blending explicitly "draws" the trajectory into a single image to bypass inter-frame reasoning bottlenecks; context provides interaction scenarios; perceptual captions describe exactly what happened.
    - **Design Motivation**: Categorizes failure into three possibilities: "unseen motion," "seen but contextually misunderstood," or "unseen high-level semantics." Effective individual enhancements identify specific bottlenecks, while a combined optimal result suggests synergy between perception, context, and semantics.

### Loss & Training
This is a zero-shot evaluation paper and does not involve training models. All 9 VLMs were called via OpenRouter for closed-source models or local inference for open-source models using default temperatures. Context lengths varied from 64K (GLM-4.5V) to 1M (Gemini-2.5-Pro).

## Key Experimental Results

### Main Results: RQ2 Purpose Classification (Accuracy + Macro F1)

| Model | Accuracy | Macro F1 |
|------|----------|----------|
| Gemini-2.5-Pro | **0.64** | **0.55** |
| GPT-5 | 0.64 | 0.53 |
| GPT-o4-mini | 0.63 | 0.51 |
| GPT-o3 | 0.62 | 0.54 |
| Gemini-2.5-Flash | 0.61 | 0.53 |
| GPT-5-mini | 0.58 | 0.48 |
| Claude-Sonnet-4 | 0.57 | 0.46 |
| GLM-4.5V | 0.45 | 0.40 |
| Qwen2.5-VL-72B | 0.39 | 0.32 |

The strongest model only reached 0.64, showing a significant gap from human performance. Per-category recall: Feedback (0.69), Visualization (0.69), and Guidance (0.59) were high, while Highlight (0.24) and Aesthetic (0.16) performed poorly. VLMs excel at animations with strong functionality or clear text feedback but struggle with "subtle" animations meant for emotional or brand emphasis.

### RQ3 Semantic Interpretation Similarity (vs. Crowd Consensus, 0-5)

| Model | Mean | Std |
|------|------|-----|
| GPT-o3 | **3.47** | 0.91 |
| GPT-5 | 3.44 | 0.90 |
| Gemini-2.5-Pro | 3.40 | 0.90 |
| GPT-5-mini | 3.39 | 0.82 |
| Gemini-2.5-Flash | 3.31 | 0.95 |
| Claude-Sonnet-4 | 3.10 | 1.12 |
| Qwen2.5-VL-72B | 2.94 | 1.24 |
| GLM-4.5V | 2.71 | 1.47 |

Most models scored around 3, capturing the gist but often missing key details or drifting in direction.

### Ablation Study: MCPC (Gemini-2.5-Flash)

| Enhancement | RQ2 Acc | RQ2 F1 | RQ3 Mean | RQ3 Std |
|-------------|---------|--------|----------|---------|
| Base        | 0.59    | 0.47   | 3.15     | 1.09    |
| + Motion    | 0.52    | 0.41   | 3.08     | 1.07    |
| + Context   | 0.58    | 0.48   | 3.30     | 0.95    |
| + Perceptual| 0.57    | 0.45   | 3.50     | 0.89    |
| + M+P       | 0.53    | 0.40   | 3.48     | 0.86    |
| + C+P       | 0.55    | 0.46   | 3.48     | 0.77    |
| **+ M+C+P** | **0.61**| **0.52**| **3.52†**| **0.73**|

The combination of all three signals significantly outperformed any single or dual signal combination, confirming strong synergy between perception, context, and semantics.

### Key Findings
- **VLMs can see motion but cannot interpret it**: In RQ1, 5/9 models perfectly identified all 7 basic motion effects, but performance dropped significantly in RQ2/RQ3, indicating the bottleneck is not low-level perception.
- **Error Pattern 1: Over-reliance on the final static frame**: In a McDonald's animation (Aesthetic: logo bounce + "ba da ba"), 6 models misclassified it as Feedback because the final frame included the text "Your order is confirmed."
- **Error Pattern 2: Small ROI failures**: Average animation ROI was 24.3% in correct cases vs. 14.1% in incorrect ones (Mann-Whitney $p=0.03$). Models are frequently distracted by large surrounding elements when the ROI is small.
- **Error Pattern 3: Ignoring interaction context**: When a user's repeated failed swipes triggered a demonstration animation for the correct gesture, 8/9 models misclassified it as a Transition, failing to link the "failed action → instructional animation" sequence.
- **Subtle, fast animations are entirely missed**: 5/9 models reported "no animation" or hallucinated non-existent progress bars for password field "shake" animations.
- **Gemini-2.5-Pro exhibits hallucinations**: Occasionally concocts details like "translucent rounded objects" that do not exist, consistent with known VLM hallucination literature.

## Highlights & Insights
- The definition "motion that is drawn, not drawings that move" is precise—it explains why frames are insufficient and why video-level evaluation is a necessity for UI agents.
- The three-tier evaluation (motion effects → purpose → semantics) precisely locates bottlenecks. This methodology is exemplary; any "Does VLM understand X" question could follow this perception $\rightarrow$ classification $\rightarrow$ interpretation hierarchy.
- Motion blending, using the old "Phosphor afterglow" trick of stacking 6 frames with transparency, is a clever prompt engineering technique to compress dynamic video into a single image, bypassing VLM inter-frame reasoning limits.
- The "shallow persona detection" finding (models perceive the existence of animation but fail to distinguish purposes) mirrors findings in psychology-related VLM papers, suggesting a common failure mode in fine-grained distinction tasks.
- The dual expert-crowd annotation design is excellent—experts ensure taxonomic rigor while the crowd ensures semantic diversity (admitting multiple valid interpretations for one animation).

## Limitations & Future Work
- Data sources are primarily US-based apps with English interfaces, failing to cover cultural or linguistic differences (e.g., Arabic RTL, Asian stock market color conventions).
- Annotators were all native English speakers from the US, potentially biasing certain cross-cultural semantics.
- Only 9 SOTA VLMs were evaluated; small models (7-14B) were almost entirely incapable of handling UI animation tasks (due to context length or single-image limits) and were excluded from the main tables.
- MCPC probes were only tested on Gemini-2.5-Flash; transferability to other models remains unverified.
- ROIs were pre-defined with green bboxes; requiring VLMs to localize ROIs themselves in real-world deployments would be significantly more challenging.

## Related Work & Insights
- **vs. Rico / MONDAY / GUI World**: Those datasets contain UI screenshots or interaction recordings but were not designed for animation understanding. AniMINT is the first multi-level annotated set specifically focused on animation semantics.
- **vs. Mackamul 2025 / Dessart 2011**: Those UX studies primarily measure human perception of animation; this work introduces that taxonomy to VLM evaluation, bridging HCI and NLP.
- **vs. General VLM agent benchmarks (OSWorld / WebWalker)**: Those benchmarks test end-to-end task completion. This work tests fine-grained perception of UI elements. They are complementary—if an agent cannot understand animation feedback, end-to-end tasks are likely to fail without a diagnosable cause.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First systematic focus on UI animation understanding. New designs from taxonomy to motion blending and MCPC probes fill the evaluation gap between static screenshots and dynamic interactions.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Evaluates 9 SOTA VLMs across 3 RQs + MCPC ablation with 8 cue combinations. Detailed quantitative analysis of failure modes (e.g., Mann-Whitney tests for ROI).
- **Writing Quality**: ⭐⭐⭐⭐⭐ Uses vivid examples throughout (McDonald's logo, password shake, Android swipe guidance) to make abstract failure modes intuitive. The 3-RQ structure is exceptionally clear.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses a major blind spot for the UI agent era. The resource of 3,000 human interpretations and dual-perspective annotations is rare and will drive research in dynamic UI agents, accessibility, and animation generation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models](../../CVPR2026/multimodal_vlm/beyond_recognition_evaluating_visual_perspective_taking_in_vision_language_model.md)
- [\[ACL 2026\] Measuring What Matters Beyond Text: Evaluating Multimodal Summaries by Quality, Alignment, and Diversity (MM-Eval)](measuring_what_matters_beyond_text_evaluating_multimodal_summaries_by_quality_al.md)
- [\[ACL 2026\] CArtBench: Evaluating Vision-Language Models on Chinese Art Understanding, Interpretation, and Authenticity](cartbench_evaluating_vision-language_models_on_chinese_art_understanding_interpr.md)
- [\[CVPR 2026\] Think360: Evaluating the Width-centric Reasoning Capability of MLLMs Beyond Depth](../../CVPR2026/multimodal_vlm/think_360_evaluating_the_width-centric_reasoning_capability_of_mllms_beyond_dept.md)
- [\[ACL 2026\] Long Story Short: Disentangling Compositionality and Long-Caption Understanding in Contrastive VLMs](long_story_short_disentangling_compositionality_and_long-caption_understanding_i.md)

</div>

<!-- RELATED:END -->
