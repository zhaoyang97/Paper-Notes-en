---
title: >-
  [Paper Note] LOTUS: A Leaderboard for Detailed Image Captioning from Quality to Societal Bias and User Preferences
description: >-
  [ACL 2025][Recommender Systems][Image Captioning Evaluation] Proposed the LOTUS leaderboard, which uniformly evaluates the detailed image captioning capabilities of Large Vision-Language Models across three dimensions: description quality (alignment, descriptiveness, language complexity), side effects (hallucinations, toxicity), and societal bias (gender, skin tone), while supporting customized evaluation based on user preferences.
tags:
  - "ACL 2025"
  - "Recommender Systems"
  - "Image Captioning Evaluation"
  - "Large Vision-Language Models"
  - "Societal Bias"
  - "User Preferences"
  - "Evaluation Benchmark"
date: 2026-05-08
content_hash: febb69fc131843c9
---

# LOTUS: A Leaderboard for Detailed Image Captioning from Quality to Societal Bias and User Preferences

**Conference**: ACL 2025  
**arXiv**: [2507.19362](https://arxiv.org/abs/2507.19362)  
**Code**: [https://huggingface.co/spaces/nvidia/lotus-vlm-bias-leaderboard](https://huggingface.co/spaces/nvidia/lotus-vlm-bias-leaderboard) (Yes, Leaderboard)  
**Area**: Recommendation Systems  
**Keywords**: Image Captioning Evaluation, Large Vision-Language Models, Societal Bias, User Preferences, Evaluation Benchmark

## TL;DR

Proposed the LOTUS leaderboard, which uniformly evaluates the detailed image captioning capabilities of Large Vision-Language Models across three dimensions: description quality (alignment, descriptiveness, language complexity), side effects (hallucinations, toxicity), and societal bias (gender, skin tone), while supporting customized evaluation based on user preferences.

## Background & Motivation

**Background**: With the emergence of Large Vision-Language Models (LVLMs) such as LLaVA, image captioning has shifted from generating concise headlines to generating detailed descriptions. This transition enhances visual semantic understanding but also poses new challenges for evaluation methods.

**Limitations of Prior Work**:
   - **Lack of a unified evaluation framework**: Existing studies evaluate single dimensions such as descriptiveness, alignment, and hallucination detection separately, lacking a standardized comprehensive framework, which makes cross-study comparison difficult.
   - **Lack of bias-aware evaluation**: Although research indicates that LVLMs often exhibit societal bias (e.g., gender bias), existing evaluation methods largely neglect these biases.
   - **Evaluation ignores user preferences**: The quality of detailed descriptions is highly subjective; some users prioritize description richness, while others value minimizing hallucination risks.

**Key Challenge**: How to simultaneously evaluate description quality, side-effect risks, and social fairness in a unified framework, while supporting the diverse needs of different users?

**Goal**: To establish a comprehensive, bias-aware, and user-preference-sensitive leaderboard for evaluating detailed image captioning.

**Key Insight**: Unifying multiple existing evaluation dimensions (alignment, descriptiveness, complexity, side effects) and innovatively incorporating societal bias evaluation and user-preference-oriented evaluation mechanisms.

**Core Idea**: Integrating quality, risk, and bias dimensions into a unified leaderboard to reveal the trade-off that more detailed descriptions lead to greater bias risk, allowing users to select the optimal model according to their needs.

## Method

### Overall Architecture

The LOTUS evaluation framework comprises four main dimensions:
1. **Alignment**: The degree of match between the description and the image content.
2. **Descriptiveness**: The level of detail in the description.
3. **Language Complexity**: The complexity of the sentence structure.
4. **Side Effects**: Hallucinations and harmful content.

In addition, there are two bias dimensions:
5. **Gender Bias**
6. **Skin Tone Bias**

### Key Designs

#### 1. Unified Comprehensive Evaluation

- **Alignment Metrics**:
    - CLIPScore: $\text{CLIPScore} = \max(0, \cos(\phi_I(I), \phi_T(y')))$, which measures the semantic similarity between image and description.
    - CapScore: GPT-4 scores, including similarity (CapScore$_S$) and alignment (CapScore$_A$), ranging from 0 to 1.

- **Descriptiveness Metrics**:
    - CLIP Recall@$k$: Whether the description can uniquely identify the corresponding image.
    - Noun/Verb Coverage: $\text{Noun Coverage} = \frac{|N(y) \cap N(y')|}{|N(y')|}$, which measures the coverage of key objects and actions.

- **Language Complexity**:
    - Syntactic Complexity: The maximum depth of the dependency tree.
    - Semantic Complexity: The number of nodes in the scene graph.

- **Side Effects**:
    - CHAIRs: $\text{CHAIR}_s = \frac{O_H}{O_T}$, the ratio of hallucinated objects.
    - FaithScore: $\text{FaithScore} = \frac{1}{K}\sum_{k=1}^K V(f_k, I)$, the faithfulness based on atomic facts.
    - NSFW vocabulary detection.

- **Design Motivation**: Multiple metrics are used for each dimension to enhance robustness, and N-avg consolidated scores are computed via Min-Max normalization.

#### 2. Bias-Aware Evaluation

- **Function**: Quantifying model performance differences across different demographic groups.
- **Mechanism**: Grouping the dataset by protected attributes (e.g., gender, skin tone), generating descriptions separately and calculating each metric. The degree of bias is defined as the absolute performance difference between groups.
- **Gender Bias**: $\mathcal{D}_g = \{(I,y,a) \in \mathcal{D} | a = g\}$, $g \in \{\text{woman}, \text{man}\}$
- **Skin Tone Bias**: Similarly, $a \in \{\text{darker-skin}, \text{lighter-skin}\}$
- **Linguistic Variance**: Comparing the impact of different prompting languages (English, Japanese, Chinese) on the generated results.
- **Design Motivation**: Revealing the unequal performance of models across different groups.

#### 3. User-Preference-Oriented Evaluation

- **Function**: Selecting evaluation dimensions according to different user profiles to provide customized model recommendations.
- **Three User Profiles**:
    - **Detail-Oriented Users**: Selecting {Alignment, Descriptiveness}.
    - **Risk-Conscious Users**: Selecting {Alignment, Side Effects, Gender Bias, Skin Tone Bias}.
    - **Accuracy-Oriented Users**: Selecting {Alignment, Side Effects}.
- **Preference Score**: The average of the N-avg scores of the selected dimensions.
- **Design Motivation**: There is no universally optimal model, only models that are optimal for specific needs.

### Loss & Training

LOTUS is an evaluation framework and does not involve model training. The five evaluated models all use their 7B parameter variants (MiniGPT-4, InstructBLIP, LLaVA-1.5, mPLUG-Owl2, Qwen2-VL) to ensure a fair comparison.

## Key Experimental Results

### Main Results

Unified evaluation on the COCO Karpathy test set (5000 images) (N-avg, normalized to 0-1):

| Model | Alignment | Descriptiveness | Complexity | Side Effects↑ |
|------|--------|--------|--------|---------|
| MiniGPT-4 | 0.19 | 0.22 | 0.38 | 0.18 |
| InstructBLIP | 0.18 | 0.40 | 0.41 | 0.66 |
| LLaVA-1.5 | 0.67 | 0.11 | 0.08 | 0.71 |
| mPLUG-Owl2 | 0.49 | 0.34 | 0.28 | 0.58 |
| **Qwen2-VL** | **0.82** | **1.00** | **1.00** | 0.46 |

Bias evaluation (gender bias, N-avg↑ indicates smaller bias is better):

| Model | Gender Bias N-avg | Skin Tone Bias N-avg |
|------|---------------|---------------|
| MiniGPT-4 | 0.51 | 0.55 |
| LLaVA-1.5 | 0.46 | **0.67** |
| Qwen2-VL | **0.63** | 0.50 |

### Ablation Study

User-preference-oriented evaluation results (Preference Score):

| User Profile | Best Model | Description |
|---------|---------|------|
| Detail-Oriented | Qwen2-VL | Descriptiveness and alignment far exceed other models. |
| Risk-Conscious | LLaVA-1.5 | Low hallucination risk, minimal skin tone bias. |
| Accuracy-Oriented | LLaVA-1.5 | Fewest side effects, most reliable. |

### Key Findings

1. **No "one-size-fits-all" model**: Qwen2-VL performs best in description quality (alignment 0.82, descriptiveness 1.00), but shows poor performance in side effects and skin tone bias.
2. **Trade-off between descriptiveness and bias**:
    - High descriptiveness → low gender bias (correlation coefficient -0.92): More detailed models tend to describe both genders more fairly.
    - High descriptiveness → high skin tone bias (correlation coefficient 0.94): More detailed models describe more dark skin-tone-related vocabulary.
3. **Negative correlation between gender bias and skin tone bias** (-0.55): Simultaneously reducing both types of bias is challenging.
4. **Significant linguistic variance**: Qwen2-VL exhibits the most substantial language difference (N-avg 0.28), with syntactic complexity variance reaching up to 90.8.
5. **Discrepancy in gendered word usage**: LLaVA-1.5 has the largest difference in the recall of gendered words for male and female images (|Δ|=6.1).
6. **Usage of racial terms**: Qwen2-VL uses racial-coded terms for darker-skin images at a significantly higher rate (7.0%) than for lighter-skin images (2.9%), with |Δ|=4.1.

## Highlights & Insights

1. **Revealing the trade-off of "more detailed description, greater bias"**: This is an important finding that alerts the community to pay attention to fairness when pursuing description quality.
2. **Practical value of user-preference evaluation**: Different users require different model recommendations, which is highly meaningful for practical deployment.
3. **Comprehensive and reproducible metric design**: Integrates multiple existing metrics, and the N-avg normalization scheme is simple and effective.
4. **In-depth bias analysis**: Not only quantifies the scale of bias but also analyzes the causes of bias (such as the increased usage of racial terms caused by higher descriptiveness).
5. **LOTUS as a continuously updated leaderboard**: Deployed on HuggingFace, allowing for the continuous evaluation of new models.

## Limitations & Future Work

1. Only five models were evaluated and limited to their 7B variants; larger/newer models (e.g., GPT-4V, Gemini) are not covered.
2. Societal bias is only considered in binary gender and binary skin tone, omitting more fine-grained or other dimensions (such as age, disability, etc.).
3. Bias metrics are based on performance differences, but the implications of the direction of these differences may require deeper discussion.
4. The representation of the COCO dataset may be insufficient; assessment of domain-specific images needs to be supplemented.
5. Linguistic variance experiments only cover three languages, requiring a broader multilingual evaluation.
6. No specific recommendations or methods for mitigating bias are provided.

## Related Work & Insights

- **DCScore, DetailCaps**: Evaluation methods focused on descriptiveness; LOTUS incorporates them into unified framework.
- **CHAIR, FaithScore**: Hallucinating evaluation metrics, serving as the side-effects dimension in LOTUS.
- **Zhao et al. (2021)**: Gender and skin tone annotations of COCO, providing a foundation for bias analysis.
- Insight: The "No one-size-fits-all" evaluation philosophy can be generalized to the evaluation of other generative tasks.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The combination of a unified framework, bias evaluation, and user preferences is novel, and the descriptiveness-bias trade-off is an important finding.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — The multi-dimensional evaluation is comprehensive, but the number of models is limited.
- **Writing Quality**: ⭐⭐⭐⭐ — Excellent visualization and well-structured.
- **Value**: ⭐⭐⭐⭐ — High value as a community benchmarking infrastructure, though missing improvement methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Aligning LLMs by Predicting Preferences from User Writing Samples](../../ICML2025/recommender/aligning_llms_by_predicting_preferences_from_user_writing_samples.md)
- [\[CVPR 2025\] FineVQ: Fine-Grained User Generated Content Video Quality Assessment](../../CVPR2025/recommender/finevq_fine-grained_user_generated_content_video_quality_assessment.md)
- [\[ICLR 2026\] More Than What Was Chosen: LLM-based Explainable Recommendation Beyond Noisy User Preferences](../../ICLR2026/recommender/more_than_what_was_chosen_llm-based_explainable_recommendation_beyond_noisy_user.md)
- [\[ACL 2026\] Quality Over Clicks: Intrinsic Quality-Driven Iterative RL for Cold-Start E-Commerce Query Suggestion](../../ACL2026/recommender/quality_over_clicks_intrinsic_quality-driven_iterative_reinforcement_learning_fo.md)
- [\[ICML 2025\] QuRe: Query-Relevant Retrieval through Hard Negative Sampling in Composed Image Retrieval](../../ICML2025/recommender/qure_query-relevant_retrieval_through_hard_negative_sampling_in_composed_image_r.md)

</div>

<!-- RELATED:END -->
