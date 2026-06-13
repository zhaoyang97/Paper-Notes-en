---
title: >-
  [Paper Note] Content Fuzzing for Escaping Information Cocoons on Social Media
description: >-
  [ACL 2026][Social Computing][Information Cocoons] The study proposes ContentFuzz, a confidence-guided fuzzing framework from the perspective of content creators. It utilizes LLMs to rewrite posts to modify machine-inferr…
tags:
  - "ACL 2026"
  - "Social Computing"
  - "Information Cocoons"
  - "Stance Detection"
  - "Fuzzing"
  - "Content Rewriting"
  - "Recommender Systems"
date: 2026-05-08
content_hash: 2b91d5d19df4ab4e
---

# Content Fuzzing for Escaping Information Cocoons on Social Media

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.05461](https://arxiv.org/abs/2604.05461)  
**Code**: None  
**Area**: Social Computing / Adversarial Learning  
**Keywords**: Information Cocoons, Stance Detection, Fuzzing, Content Rewriting, Recommender Systems

## TL;DR
The study proposes ContentFuzz, a confidence-guided fuzzing framework from the perspective of content creators. It utilizes LLMs to rewrite posts to modify machine-inferred stance labels while maintaining the human-interpreted meaning, thereby breaking through social media information cocoons.

## Background & Motivation

**Background**: Social media platforms employ stance detection as a critical signal in recommendation and ranking pipelines. Posts are primarily routed to audiences with identical viewpoints, which reduces cross-stance exposure, limits the dissemination of diverse opinions, and hinders constructive discussion.

**Limitations of Prior Work**: Existing methods for breaking information cocoons are primarily platform-side algorithmic interventions (e.g., diversity re-ranking). However, these are controlled by platforms; individual users and creators cannot modify recommendation algorithms nor observe how posts are filtered, ranked, and distributed. Creators lack proactive tools to expand their content reach.

**Key Challenge**: While users and creators need to increase cross-group exposure, they lack actionable technical means—the only element they control is the content itself.

**Goal**: From the creator’s perspective, the study explores how to break information cocoons through content rewriting—specifically identifying semantic-preserving rewrites that maintain human-interpreted stances while altering machine-classified stances.

**Key Insight**: Borrowing methodology from software fuzzing, the stance detection model is treated as a "System Under Test" (SUT). Input variants are iteratively discovered to flip the classification results.

**Core Idea**: Confidence feedback from the stance detection model guides the LLM to generate semantic-preserving rewrites. A decrease in confidence indicates that the rewrite is exploring areas near the classifier's decision boundary. High-iteration continues until the label flips or the budget is exhausted.

## Method

### Overall Architecture
Starting from an original post, ContentFuzz iteratively executes: seed selection $\rightarrow$ LLM-based mutation to generate candidate rewrites $\rightarrow$ execution of the stance detector to obtain confidence $\rightarrow$ retention of candidates that reduce confidence as future seeds $\rightarrow$ termination when a candidate changes the predicted stance or iterations are exhausted.

### Key Designs

1. **Confidence-guided Feedback**:

    - **Function**: Guides the LLM to generate rewrites that evolve in the "correct" direction (approaching the decision boundary).
    - **Mechanism**: After each mutation, a stance analyzer is run to obtain the predicted stance and confidence. If the new candidate’s confidence is lower than the seed's, it indicates the candidate is pushing the model away from its current decision; it is then added to the seed pool. If the stance label flips, success is returned immediately.
    - **Design Motivation**: Blind rewriting is inefficient. Confidence feedback provides a "temperature" signal—lower confidence denotes proximity to the decision boundary.

2. **Seed Scheduling Strategy**:

    - **Function**: Prioritizes the most promising seeds for the next round of mutation.
    - **Mechanism**: A seed pool is maintained and sorted by confidence. Seeds with lower confidence are closer to the decision boundary and deserve prioritized mutation. The strategy also considers the number of times a seed has been mutated to avoid over-exploiting a single seed.
    - **Design Motivation**: When computational resources are limited, focusing on the most promising search directions is critical.

3. **Semantic-preserving Mutation**:

    - **Function**: Generates rewrites that preserve the original meaning but potentially alter machine judgment.
    - **Mechanism**: LLMs (e.g., GPT-4) generate rewrites using carefully designed prompts that mandate the preservation of core viewpoints and attitudes while allowing modifications to phrasing, sentence structure, and rhetorical devices. Multiple candidates are generated to increase coverage.
    - **Design Motivation**: Unlike adversarial attacks, ContentFuzz requires that rewrites remain semantically identical to human readers—this is "escaping cocoons" rather than "deceiving classifiers."

### Loss & Training
ContentFuzz is an inference-time framework and does not require training. The optimization objective is to minimize the stance detector's confidence in the original label until a label flip occurs.

## Key Experimental Results

### Main Results

| Setting | Stance Model | Success Rate | Semantic Preservation | Fluency |
|------|---------|-------|---------|-------|
| English Dataset | BERT-based | High | Strong | High |
| English Dataset | LLM-based | High | Strong | High |
| Chinese Dataset | BERT-based | High | Strong | High |
| Cross-topic Transfer | Multi-model | Stable | Stable | Stable |

### Ablation Study

| Configuration | Performance | Description |
|------|------|------|
| W/o Confidence Feedback (Random) | Low Success Rate | Directionless exploration is highly inefficient |
| W/o Seed Scheduling (Uniform) | Decreased | Wastes resources on low-potential seeds |
| Full ContentFuzz | **Optimal** | Synergistic effect of feedback and scheduling |

### Key Findings
- ContentFuzz is effective across 3 datasets, 2 languages, and 4 stance detection models.
- Rewrites successfully flip machine stance labels while maintaining semantic integrity.
- Minor phrasing changes significantly impact stance detector outputs, revealing the vulnerability of these models.

## Highlights & Insights
- The **perspective shift** is the primary highlight: moving from "how platforms break cocoons" to "how creators break out" is an overlooked yet actionable direction.
- The **cross-domain transfer of fuzzing methodology** is ingenious—seamlessly applying core software testing concepts (iterative mutation, feedback guidance, seed scheduling) to NLP.
- It **reveals the vulnerability of stance detection models**—rewrites with unchanged semantics can flip predictions, questioning the reliability of recommendation systems.

## Limitations & Future Work
- Dependency on black-box/gray-box access to stance detection models—fully black-box recommendation systems may not provide confidence scores.
- Whether successful rewrites truly alter distribution decisions in recommendation algorithms has not been verified on real-world platforms.
- Potential misuse for public opinion manipulation requires the consideration of ethical boundaries.

## Related Work & Insights
- **vs. Adversarial Attacks**: Adversarial attacks seek minimal perturbations to flip labels; ContentFuzz seeks natural, semantic-preserving rewrites.
- **vs. Platform-side Intervention**: These are complementary—platforms control algorithms, while creators control content.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First framework for breaking information cocoons from the content side; unique perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across multiple languages and models.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and appropriate methodological analogies.
- Value: ⭐⭐⭐⭐ Dual value for information diversity and recommender system robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Synthia: Scalable Grounded Persona Generation from Social Media Data](synthia_scalable_grounded_persona_generation_from_social_media_data.md)
- [\[ACL 2026\] Bayesian Social Deduction with Graph-Informed Language Models](bayesian_social_deduction_with_graph-informed_language_models.md)
- [\[ACL 2026\] The Proxy Presumption: From Semantic Embeddings to Valid Social Measures](the_proxy_presumption_from_semantic_embeddings_to_valid_social_measures.md)
- [\[ACL 2026\] DIA-HARM: Dialectal Disparities in Harmful Content Detection Across 50 English Dialects](dia-harm_dialectal_disparities_in_harmful_content_detection_across_50_english_di.md)
- [\[NeurIPS 2025\] Precise Information Control in Long-Form Text Generation](../../NeurIPS2025/social_computing/precise_information_control_in_long-form_text_generation.md)

</div>

<!-- RELATED:END -->
