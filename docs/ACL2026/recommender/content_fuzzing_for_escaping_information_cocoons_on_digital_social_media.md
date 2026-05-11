---
title: >-
  [Paper Note] Content Fuzzing for Escaping Information Cocoons on Social Media
description: >-
  [ACL 2026][Recommender Systems][information cocoon] This paper proposes ContentFuzz, a confidence-guided fuzzing framework from the content creator's perspective. It leverages LLMs to rewrite posts such that the machine-…
tags:
  - "ACL 2026"
  - "Recommender Systems"
  - "information cocoon"
  - "stance detection"
  - "fuzzing"
  - "content rewriting"
date: 2026-05-08
content_hash: a8aec84feb5d836f
---

# Content Fuzzing for Escaping Information Cocoons on Social Media

**Conference**: ACL 2026
**arXiv**: [2604.05461](https://arxiv.org/abs/2604.05461)
**Code**: None
**Area**: Social Computing / Adversarial Learning
**Keywords**: information cocoon, stance detection, fuzzing, content rewriting, recommender systems

## TL;DR
This paper proposes ContentFuzz, a confidence-guided fuzzing framework from the content creator's perspective. It leverages LLMs to rewrite posts such that the machine-inferred stance label changes while the human-interpreted meaning remains unchanged, thereby breaking information cocoons on social media.

## Background & Motivation

**State of the Field**: Social media platforms employ stance detection as a key signal in recommendation and ranking pipelines, routing posts primarily to audiences sharing similar viewpoints and reducing cross-stance exposure. This restricts the reach of diverse opinions and hinders constructive discourse.

**Limitations of Prior Work**: Existing approaches to breaking information cocoons mainly focus on platform-side algorithmic interventions (e.g., diversity re-ranking). However, such methods are controlled by platforms; individual users and content creators cannot modify recommendation algorithms, nor can they observe how posts are filtered, ranked, or distributed. Creators lack tools to proactively expand their content's reach.

**Root Cause**: Users and creators have a genuine need to broaden cross-group exposure, yet lack actionable technical means to do so—the only variable they can control is the content itself.

**Paper Goals**: From the creator's perspective, this paper explores how information cocoons can be escaped through content rewriting—specifically, by finding semantics-preserving rewrites that maintain the human-perceived stance while altering the machine-classified stance.

**Starting Point**: Drawing on the fuzzing methodology from software testing, the paper treats stance detection models as the "system under test" and iteratively discovers input variants that flip the classification result.

**Core Idea**: The confidence scores from a stance detection model are used to guide an LLM in generating semantics-preserving rewrites. A decrease in confidence indicates that the rewrite is exploring the vicinity of the classifier's decision boundary; the process iterates until the label flips or the budget is exhausted.

## Method

### Overall Architecture
ContentFuzz starts from the original post and iteratively executes the following steps: select a seed → mutate via LLM to generate candidate rewrites → run the stance detector to obtain confidence scores → retain candidates that lower confidence as future seeds → repeat until a candidate changes the predicted stance label or the iteration budget is exhausted.

### Key Designs

1. **Confidence-Guided Feedback**:

    - Function: Directs the LLM to generate rewrites that evolve toward the "correct" direction (i.e., closer to the decision boundary).
    - Mechanism: After each mutation, a stance analyzer is run to obtain the predicted stance and confidence score. If a new candidate's confidence is lower than that of the seed, it indicates the rewrite is pushing the model away from its current decision; the candidate is added to the seed pool. If the stance label flips, a success is immediately returned.
    - Design Motivation: Blind rewriting is inefficient; confidence feedback provides a "temperature" signal—the lower the temperature, the closer to the decision boundary.

2. **Seed Scheduling Strategy**:

    - Function: Prioritizes the most promising seeds for the next round of mutation.
    - Mechanism: A seed pool is maintained and sorted by confidence score—seeds with lower confidence are closer to the decision boundary and thus more worthy of further mutation. The number of times each seed has already been mutated is also considered to avoid over-exploiting a single seed.
    - Design Motivation: When computational resources are limited, focusing on the most promising search directions is critical.

3. **Semantics-Preserving Mutation**:

    - Function: Generates rewrites that preserve the original meaning while potentially altering the machine's judgment.
    - Mechanism: An LLM (e.g., GPT-4) is prompted with carefully designed instructions to retain the core viewpoint and attitude while allowing modifications to surface features such as phrasing, sentence structure, and rhetorical devices. Multiple candidates are generated per iteration to broaden coverage.
    - Design Motivation: Unlike adversarial attacks, ContentFuzz requires that rewrites remain completely unchanged in meaning for human readers—the goal is to "escape the cocoon," not to "deceive the classifier."

### Loss & Training
ContentFuzz is an inference-time framework and requires no training. The optimization objective is to minimize the stance detector's confidence in the original label until the label flips.

## Key Experimental Results

### Main Results

| Setting | Stance Model | Success Rate | Semantic Preservation | Fluency |
|---|---|---|---|---|
| English dataset | BERT-based | High | Strong | High |
| English dataset | LLM-based | High | Strong | High |
| Chinese dataset | BERT-based | High | Strong | High |
| Cross-topic transfer | Multiple models | Stable | Stable | Stable |

### Ablation Study

| Configuration | Performance | Note |
|---|---|---|
| Without confidence feedback (random mutation) | Low success rate | Undirected exploration is highly inefficient |
| Without seed scheduling (uniform selection) | Degraded | Resources wasted on low-potential seeds |
| Full ContentFuzz | **Best** | Feedback and scheduling act synergistically |

### Key Findings
- ContentFuzz is effective across 3 datasets, 2 languages, and 4 stance detection models.
- Rewrites successfully flip machine stance labels while preserving semantic integrity.
- Minor phrasing changes can significantly alter stance detector outputs, revealing the fragility of these models.

## Highlights & Insights
- **Perspective shift** is the most significant contribution: the focus moves from "how platforms break cocoons" to "how creators break out," an overlooked yet practically actionable direction.
- **Cross-domain transfer of the fuzzing methodology** is elegant—the core ideas from software testing (iterative mutation + feedback guidance + seed scheduling) are seamlessly applied to an NLP setting.
- **Exposes the fragility of stance detection models**—semantics-preserving rewrites suffice to flip predictions, raising questions about the reliability of recommendation systems built upon them.

## Limitations & Future Work
- The framework relies on black-box or gray-box access to stance detection models; confidence scores may be unavailable in fully black-box recommendation systems.
- Whether successful rewrites actually alter the distribution decisions of real recommendation algorithms has not been validated on live platforms.
- The approach could be misused for opinion manipulation, necessitating careful consideration of ethical boundaries.

## Related Work & Insights
- **vs. Adversarial Attacks**: Adversarial attacks seek minimal perturbations to flip labels, whereas ContentFuzz seeks natural, semantics-preserving rewrites.
- **vs. Platform-Side Interventions**: The two are complementary—platforms control algorithms, while creators control content.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First content-side framework for escaping information cocoons; highly original perspective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across multiple languages and models.
- Writing Quality: ⭐⭐⭐⭐ Problem motivation is clear; methodological analogy is well-chosen.
- Value: ⭐⭐⭐⭐ Dual value for information diversity and recommendation system robustness.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] CollectiveKV: Decoupling and Sharing Collaborative Information in Sequential Recommendation](../../ICLR2026/recommender/collectivekv_decoupling_and_sharing_collaborative_information_in_sequential_reco.md)
- [\[AAAI 2026\] Generalization Bounds for Semi-supervised Matrix Completion with Distributional Side Information](../../AAAI2026/recommender/generalization_bounds_for_semi-supervised_matrix_completion_with_distributional_.md)
- [\[AAAI 2026\] FreqRec: Exploiting Inter-Session Information with Frequency-enhanced Dual-Path Networks for Sequential Recommendation](../../AAAI2026/recommender/exploiting_inter-session_information_with_frequency-enhanced_dual-path_networks_.md)
- [\[NeurIPS 2025\] Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation](../../NeurIPS2025/recommender/who_you_are_matters_bridging_topics_and_social_roles_via_llm-enhanced_logical_re.md)
- [\[NeurIPS 2025\] The Coming Crisis of Multi-Agent Misalignment: AI Alignment Must Be a Dynamic and Social Process](../../NeurIPS2025/recommender/the_coming_crisis_of_multi-agent_misalignment_ai_alignment_must_be_a_dynamic_and.md)

</div>

<!-- RELATED:END -->
