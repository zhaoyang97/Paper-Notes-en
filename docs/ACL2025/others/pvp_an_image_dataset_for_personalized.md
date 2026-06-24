---
title: >-
  [Paper Note] PVP: An Image Dataset for Personalized Visual Persuasion with Persuasion Strategies, Viewer Characteristics, and Persuasiveness Ratings
description: >-
  [ACL 2025][Visual Persuasion] Builds PVP (28,454 images, 596 behavioral messages, 9 persuasion strategies), the first large-scale dataset linking image persuasion strategies with the psychological traits (personality/values/moral foundations) of 2,521 annotators. It validates the critical role of psychological profiles in enhancing persuasion effects on two benchmark tasks: personalized persuasive image generation and automatic persuasiveness evaluation.
tags:
  - "ACL 2025"
  - "Visual Persuasion"
  - "Personalization"
  - "Persuasion Strategies"
  - "Psychological Traits"
  - "Dataset"
date: 2026-05-08
content_hash: 50fe54c22a6f278c
---

# PVP: An Image Dataset for Personalized Visual Persuasion with Persuasion Strategies, Viewer Characteristics, and Persuasiveness Ratings

**Conference**: ACL 2025  
**arXiv**: [2506.00481](https://arxiv.org/abs/2506.00481)  
**Code**: [https://github.com/holi-lab/PVP_Personalized_Visual_Persuasion](https://github.com/holi-lab/PVP_Personalized_Visual_Persuasion)  
**Area**: Others  
**Keywords**: Visual Persuasion, Personalization, Persuasion Strategies, Psychological Traits, Dataset

## TL;DR

Builds PVP (28,454 images, 596 behavioral messages, 9 persuasion strategies), the first large-scale dataset linking image persuasion strategies with the psychological traits (personality/values/moral foundations) of 2,521 annotators. It validates the critical role of psychological profiles in enhancing persuasion effects on two benchmark tasks: personalized persuasive image generation and automatic persuasiveness evaluation.

## Background & Motivation

**Background**: Visual persuasion influences cognition, emotion, and behavior using visual elements, which is critical in advertising, political communication, and public health. Existing datasets cover meme persuasion technique detection, advertisement understanding, and political image analysis, but mostly focus on classification/detection of "which persuasion techniques are contained in the image".

**Limitations of Prior Work**: Existing visual persuasion datasets suffer from four main limitations: (1) most lack persuasiveness ratings, making them unusable for training/evaluating persuasion systems; (2) they focus heavily on memes and semiotics, relying on deep cultural knowledge, which is unsuitable for immediate visual impact in daily scenarios; (3) they have narrow topic coverage, concentrating on controversial topics like politics, abortion, and guns, limiting generalizability; (4) they completely ignore the individual differences of viewers—the same image can have completely different persuasiveness for people with different psychological traits, yet no existing datasets collect viewer psychological profiles.

**Key Challenge**: Persuasion is inherently personalized; there is no "one-size-fits-all" approach. However, existing datasets only focus on the "image-side" strategy labels, lacking the "human-side" psychological traits, which hinders the development of personalized persuasion systems.

**Goal**: (1) Construct a large-scale dataset linking image persuasion strategies, persuasiveness ratings, and viewer psychological traits; (2) Define two downstream tasks—personalized persuasive image generation and automatic persuasiveness evaluation—and establish benchmarks for them.

**Key Insight**: Grounded in psychology and communication theories (Theory of Planned Behavior, Big Five personality traits, Schwartz values, Moral Foundations Theory), the authors design 9 persuasion strategies and comprehensively profile annotators' psychological traits in a theory-driven manner, providing the dataset with a solid theoretical foundation rather than a purely empirical one.

**Core Idea**: Construct the first large-scale dataset supporting personalized visual persuasion research by binding 9 theory-driven persuasion strategies with the three-dimensional psychological profiles of 2,521 annotators.

## Method

### Overall Architecture

The dataset construction pipeline consists of six steps: message (target behavior) generation $\to$ 9 persuasion strategies definition $\to$ premise generation $\to$ dual-source image collection (DALL-E generation + Google retrieval) $\to$ persuasiveness rating annotation $\to$ annotator psychological profile questionnaires. The input consists of 596 behavioral messages (covering 20 daily topics), and the output comprises 28,454 images with their corresponding multi-annotator ratings and psychological traits. Based on this dataset, two downstream tasks are proposed: (1) given a message and target viewer psychological traits, generate a personalized persuasive image; (2) given an image and target viewer psychological traits, automatically predict the persuasiveness score.

### Key Designs

1. **Theory-driven 9 persuasion strategies framework**:

    - Function: Provides 9 different persuasion angles of image coverage for each behavioral message, enabling a systematic analysis of strategy effects.
    - Mechanism: Based on the Theory of Planned Behavior and Argumentation Theory, 5 dimensions are defined: Perceived Persona (how others view you), Internal Emotion (your own emotional response), External Emotion (others' emotional response), Consequence (positive/negative consequences of the behavior), and Bandwagon (everyone is doing it). The first four categories are split into positive (gain-framed) and negative (loss-framed) versions, which, along with the positive-only bandwagon strategy, yield 9 strategies. Each message $\times$ 9 strategies $\times$ 3 premises $\times$ 2 image sources = 54 candidate images.
    - Design Motivation: It covers both cognitive dimensions (consequences, bandwagon) and emotional dimensions (internal/external emotions). The distinction between positive and negative framing directly leverages classic findings of framing effects in psychology, ensuring the systematism and theoretical completeness of the strategy space.

2. **Dual-source image collection and quality filtering**:

    - Function: Generates/retrieves visually accurate and highly diverse persuasive images for each premise.
    - Mechanism: GPT-4o is first used to convert each premise into a DALL-E generation prompt and a Google search query to retrieve one AI-generated image and one real retrieved image respectively. A double-filtering process (human + GPT) is then applied to verify whether each image accurately conveys the target premise, while filtering out Google images with excessive text (as visual persuasion should not rely on text). An average of 6 unqualified images are discarded per message, resulting in a final collection of 28,454 images.
    - Design Motivation: DALL-E images match premises more precisely (with slightly higher ratings), while Google images are more natural and realistic; combining both increases diversity and provides a contrast for analyzing the persuasiveness differences between AI-generated and real images.

3. **Multi-dimensional psychological profile annotation and balanced sampling**:

    - Function: Establishes complete psychological profiles for each annotator to correlate persuasiveness ratings with individual differences.
    - Mechanism: 2,521 annotators completed three standard psychological scales: BFI-10 (Big Five personality: Openness/Conscientiousness/Extraversion/Agreeableness/Neuroticism), PVQ-21 (Schwartz 10 values), and MFQ-30 (5 moral foundations). In addition, whether the annotator already practices the target behavior in daily life (Habit) is recorded. In terms of sampling, participants are balanced across gender $\times$ age group (20s/30s/40s/50s). Each annotator only annotated one message (evaluating up to 54 images), and each image is rated independently by 4 different annotators.
    - Design Motivation: A single personality scale is insufficient to capture individual differences—values and moral foundations influence persuasion receptivity from different perspective angles. The one-person-one-message design avoids annotator learning effects and fatigue bias, while the 4-annotator rating ensures statistical reliability.

### Loss & Training

For the evaluator task, the authors compare zero-shot inference of GPT-4o and GPT-4o-mini with fine-tuned LLaMA3-8B-Instruct. Fine-tuning uses QLoRA with the input of message + image description + psychological trait text, and the output of a persuasiveness score ranging from 0-10. For the generator task, LLaMA3 is fine-tuned only on high-scoring images (persuasiveness score > 8) to prevent low-quality samples from degrading generation quality. The generator outputs text prompts (image descriptions) instead of direct images to decouple the "understanding of persuasive features" from "image generation quality".

## Key Experimental Results

### Main Results: Persuasiveness Evaluator

| Model | Input Format | Spearman ↑ | Pearson ↑ | NDCG ↑ | RMSE ↓ |
|------|---------|:---:|:---:|:---:|:---:|
| GPT-4o | Image (Multimodal) | 0.19 | 0.19 | 0.39 | 3.90 |
| GPT-4o-mini | Image (Multimodal) | 0.13 | 0.11 | 0.35 | 4.01 |
| GPT-4o | Image description (Text-only) | 0.16 | 0.17 | 0.37 | 3.81 |
| GPT-4o-mini | Image description (Text-only) | 0.15 | 0.13 | 0.36 | 3.84 |
| LLaMA3-8B Zero-shot | Image description | 0.07 | 0.06 | 0.34 | 3.71 |
| **LLaMA3-8B Fine-tuned** | **Image description** | **0.25** | **0.25** | **0.42** | **3.40** |

### Ablation Study: Impact of Psychological Trait Types on Evaluator

| Psychological Trait Input | Spearman ↑ | Pearson ↑ | NDCG ↑ | RMSE ↓ |
|:---:|:---:|:---:|:---:|:---:|
| PVQ-21 (Values) | **0.25** | **0.25** | 0.42 | **3.40** |
| BFI-10 (Big Five) | 0.23 | 0.24 | 0.43 | 3.66 |
| MFQ-30 (Moral Foundations) | 0.25 | 0.24 | 0.42 | 3.48 |
| No Psychological Traits | 0.23 | 0.23 | 0.43 | 3.78 |

### Generator Performance

| Model | Avg. Persuasiveness Score ↑ | Std. Dev. |
|------|:---:|:---:|
| GPT-4o | 4.45 | 2.41 |
| GPT-4o-mini | 4.59 | 2.30 |
| **LLaMA3-8B Fine-tuned** | **4.77** | 2.37 |

### Key Findings
- **Psychological profiles are effective**: Incorporating PVQ-21 increases Spearman from 0.23 to 0.25 and reduces RMSE from 3.78 to 3.40. Personal values (PVQ) are the most effective for predicting persuasiveness, surpassing personality and moral foundations.
- **Fine-tuning small models outperforms zero-shot large models**: Fine-tuned LLaMA3-8B significantly outperforms both GPT-4o and GPT-4o-mini, indicating the high training value of the PVP dataset.
- **Negative internal emotional strategy is the most persuasive** (average score of 5.83) but is the most sensitive to personality: individuals with high neuroticism are more sensitive and receptive (correlation +0.57), while those with high conscientiousness show aversion (correlation -0.51).
- **Positive strategies are overall "safer"**: They show weaker correlation with personality traits, making them suitable for general persuasion targeted at unknown audiences.
- **Cognitive dissonance effect of existing habits**: Annotators who already perform the target behavior gave an average score of 5.0, compared to only 4.3 for those who do not, which likely stems from psychological mechanisms to avoid cognitive dissonance.
- **Main errors of the generator**: (1) Misalignment between the image description and the target message; (2) Failure to accurately comprehend the target psychological traits, especially the values dimension.

## Highlights & Insights
- **First to introduce "human-side" information in visual persuasion**: All prior datasets focused exclusively on the "image-side" (strategy labels, content analysis). PVP constructs multi-dimensional profiles of 2,521 individuals through three psychological scales, making personalized persuasion research genuinely feasible. This methodology can be transferred to any content generation task that needs to consider user differences.
- **Theory-driven strategy design is highly exemplary**: Grounded in the Theory of Planned Behavior and Argumentation Theory, the authors systematically derive a 5-dimension $\times$ positive/negative framing strategy space, ensuring coverage and interpretability instead of relying on arbitrary definitions.
- **The decoupled design of Generator = Text Prompt + Independent Image Model is ingenious**: Separating "understanding what kind of image is persuasive" from "the capability to generate high-quality images" allows a fairer evaluation and makes the methodology applicable to any image generation backend.

## Limitations & Future Work
- Annotators are solely South Korean, raising questions about the cross-cultural generalizability of the findings; the authors plan to expand to other cultural backgrounds.
- Self-reported persuasiveness ratings (0-10) are used instead of actual behavior changes, which may differ from real-world persuasion effects.
- The absolute correlation of the evaluator remains relatively low (best Spearman is only 0.25), presenting limited confidence as an automatic metric for the generator.
- DALL-E generated images often exhibit AI artifacts, which might reduce credibility and persuasive effects in real-world applications.
- Each image only has 4 annotator ratings, limiting statistical confidence when individual differences are high.

## Related Work & Insights
- **vs Hussain et al. (2017) ad dataset**: Focuses on identifying persuasion techniques in advertisements (semiotics, emotional appeals, etc.), but lacks persuasiveness ratings and viewer characteristics. PVP complements this with complete "human-side" multi-dimensional psychological profiles.
- **vs Liu et al. (2022)**: Contains persuasiveness ratings but only covers 3 controversial topics (abortion, immigration, guns) and lacks annotator psychological traits. PVP covers 20 daily life topics and provides three-dimensional psychological profiles.
- **vs Dimitrov et al. (2021) meme dataset**: Focuses on classifying persuasion techniques in memes (e.g., loaded language, name calling), which requires deep cultural background knowledge. PVP utilizes intuitively understandable situational images, offering wider applicability.

## Rating
- Novelty: ⭐⭐⭐⭐ First to systematically connect three-dimensional psychological traits with visual persuasion; the dataset design shows strong theoretical depth.
- Experimental Thoroughness: ⭐⭐⭐ The data analysis is detailed with detailed strategy $\times$ personality interaction analysis, but experiments on evaluation and generation models are preliminary and the best correlation remains low.
- Writing Quality: ⭐⭐⭐⭐ Clear theoretical framework, rigorous data construction pipeline, and well-explained psychological background.
- Value: ⭐⭐⭐⭐ Provides the first comprehensive resource for personalized visual persuasion research, offering highly reusable datasets.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Learning Visual Hierarchies in Hyperbolic Space for Image Retrieval](../../ICCV2025/others/learning_visual_hierarchies_in_hyperbolic_space_for_image_retrieval.md)
- [\[ACL 2025\] In Prospect and Retrospect: Reflective Memory Management for Long-term Personalized Dialogue Agents](in_prospect_and_retrospect_reflective_memory_management_for_long-term_personaliz.md)
- [\[ACL 2025\] Towards Text-Image Interleaved Retrieval](towards_text-image_interleaved_retrieval.md)
- [\[ACL 2025\] VAQUUM: Are Vague Quantifiers Grounded in Visual Data?](vaquum_are_vague_quantifiers_grounded_in_visual_data.md)
- [\[ACL 2025\] KodCode: A Diverse, Challenging, and Verifiable Synthetic Dataset for Coding](kodcode_a_diverse_challenging_and_verifiable_synthetic_dataset_for_coding.md)

</div>

<!-- RELATED:END -->
