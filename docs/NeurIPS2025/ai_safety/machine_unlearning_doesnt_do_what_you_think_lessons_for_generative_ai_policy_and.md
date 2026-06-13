---
title: >-
  [Paper Note] Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy and Research
description: >-
  [NeurIPS 2025][AI Safety][Machine Unlearning] This paper systematically identifies five fundamental mismatches between machine unlearning techniques and policy objectives in the context of generative AI…
tags:
  - "NeurIPS 2025"
  - "AI Safety"
  - "Machine Unlearning"
  - "Generative AI Policy"
  - "Privacy Compliance"
  - "Copyright Protection"
  - "Safety Governance"
date: 2026-05-08
content_hash: 134a2ac27254b4d3
---

# Machine Unlearning Doesn't Do What You Think: Lessons for Generative AI Policy and Research

**Conference**: NeurIPS 2025 Oral  
**arXiv**: [2412.06966](https://arxiv.org/abs/2412.06966)  
**Code**: None  
**Area**: AI Safety / Policy Analysis
**Keywords**: Machine Unlearning, Generative AI Policy, Privacy Compliance, Copyright Protection, Safety Governance

## TL;DR

This paper systematically identifies five fundamental mismatches between machine unlearning techniques and policy objectives in the context of generative AI, arguing that machine unlearning cannot serve as a universal solution for privacy, copyright, or safety concerns, and provides a practical conceptual framework for both ML researchers and policymakers.

## Background & Motivation

**Background**: Machine unlearning was first proposed around 2016 in response to the "right to erasure" under Article 17 of the EU GDPR. With the rise of generative AI, it has been widely promoted as a panacea—capable of removing personal data, eliminating copyrighted content, and erasing dangerous knowledge such as bioweapon synthesis, as if "forgetting" unwanted information were sufficient to resolve these concerns.

**Limitations of Prior Work**: This prevailing understanding of machine unlearning is plagued by serious conceptual confusion. Researchers and policymakers alike conflate *removal*—eliminating the influence of specific training data from model parameters—with *suppression*—preventing specific content from appearing in model outputs. These two objectives differ fundamentally in both technical realization and legal interpretation. Moreover, numerous policy documents and media reports treat machine unlearning as a universal solution across privacy, copyright, and safety domains, overlooking its inherent technical limitations.

**Key Challenge**: Deleting information from an ML model is fundamentally unlike deleting a record from a database—model parameters are not directly interpretable, and the influence of specific data cannot be precisely localized or "removed." Furthermore, even if certain data is successfully removed from the training set, the model may still generate similar content through generalization. This gives rise to a fundamental gap between what unlearning methods can actually do and what policy expects them to do.

**Goal**: To construct an interdisciplinary analytical framework that systematically characterizes five technical mismatches in machine unlearning, and to analyze their specific manifestations and consequences under U.S. copyright law, privacy law, and safety policy.

**Key Insight**: The authors span ML, law, and policy (including Stanford, Google DeepMind, Microsoft Research, and Cornell Law School), and proceed from first principles to deconstruct each misalignment between technical capabilities and policy objectives.

**Core Idea**: Machine unlearning exhibits five fundamental mismatches in generative AI settings (removal ≠ suppression; removal does not guarantee output control; model ≠ output ≠ use), and therefore cannot serve as a singular, universal mechanism for legal compliance.

## Method

### Overall Architecture

This paper is an analytical rather than a conventional technical contribution. Its core framework proceeds as follows: first, it defines two distinct objectives of machine unlearning (removal and suppression); second, it surveys mainstream technical approaches; third, it distills five fundamental mismatches; and finally, it analyzes the implications of these mismatches across three policy domains—copyright, privacy, and safety.

### Key Designs

1. **Distinguishing Two Unlearning Objectives and Surveying Technical Methods**

    - *Function*: Decomposes the ambiguous concept of "machine unlearning" into two distinct objectives.
    - *Mechanism*: **Removal** refers to deleting specific samples from training data and retraining so that their influence is absent from model parameters. The gold standard is full retraining, which is prohibitively expensive; structural exact unlearning and approximate methods offer more efficient alternatives, each with their own limitations. **Suppression** refers to preventing specific content from appearing in generated outputs, achieved through model modification (e.g., RLHF fine-tuning) or system-level interventions (e.g., output filters). These two categories differ fundamentally in both technical mechanism and legal force.
    - *Design Motivation*: Policy discussions routinely conflate the two, leading to systematic misjudgment of technical capabilities.

2. **Five Fundamental Mismatches**

    - *Function*: Constitutes the core analytical framework of the paper.
    - *Mechanism*:
        - **Mismatch 1**: Output suppression does not substitute for training data removal—information persists in model parameters and may be extractable by adversaries.
        - **Mismatch 2**: Removing training data does not guarantee meaningful output suppression—even after removing all copyrighted Spider-Man images, the model may still generate similar imagery through generalization (confirmed by the CommonCanvas experiment).
        - **Mismatch 3**: The model is not equivalent to its outputs—via prompt injection, a model can combine latent knowledge to regenerate "forgotten" content.
        - **Mismatch 4**: Model outputs are not equivalent to how those outputs are used—seemingly benign outputs may be repurposed for harmful ends by downstream users, beyond the reach of technical controls.
        - **Mismatch 5**: Unlearning produces unintended side effects—removing specific information can inadvertently degrade model performance on unrelated tasks.
    - *Design Motivation*: Exposes the fundamental gap between technical methods and policy objectives.

3. **Analysis Across Three Policy Domains (Copyright, Privacy, Safety)**

    - *Function*: Concretizes abstract technical mismatches within real legal and policy contexts.
    - *Mechanism*: **Copyright**—the determination of "substantial similarity" is highly subjective and cannot be proceduralized; unlearning methods cannot distinguish between fair use and infringing outputs; and the scope of removal is difficult to bound (too broad damages model utility; too narrow fails to prevent infringing outputs). **Privacy**—data deletion requests require identifying all relevant training data (itself a difficult problem), and even after deletion, latent information may still allow the model to infer personal details. **Safety**—the boundaries of dangerous knowledge (e.g., bioweapon synthesis) are extremely blurry; combinations of high-school chemistry can yield toxic molecular formulas, and users can reintroduce forgotten knowledge through prompting.
    - *Design Motivation*: Enables both ML researchers and policymakers to understand the limitations of machine unlearning from within their respective domains.

## Key Experimental Results

### Main Results

This is an analytical paper with no conventional quantitative experiments, but it employs one key empirical case:

| Case | Phenomenon | Explanation |
|------|------------|-------------|
| CommonCanvas (training set with no copyrighted Spider-Man images) | Still generates Mickey Mouse-like imagery | The training set contained only Creative Commons-licensed personal photos (e.g., tourist photos at Disneyland), yet the model produced Mickey Mouse-like outputs through generalization |
| Shumailov et al. "ununlearning" phenomenon | Forgotten knowledge re-emerges via in-context prompting | Providing relevant context through prompts reverses the effect of unlearning |
| WMDP safety benchmark | Multiple-choice evaluation is insufficient | Fails to assess safety risks in open-ended reasoning scenarios |

### Mismatch Impact Analysis

| Policy Domain | Core Difficulty | Root Mismatch |
|---------------|-----------------|---------------|
| Copyright | Substantial similarity cannot be proceduralized; removal scope is indeterminate | Mismatch 2, 5 |
| Privacy | Training data identification is difficult; inferential leakage cannot be blocked | Mismatch 1, 2, 3 |
| Safety | Boundaries of dangerous knowledge are fuzzy; prompts can circumvent unlearning | Mismatch 2, 3, 4 |

### Key Findings

- The gold standard for machine unlearning (full retraining) is itself problematic: the choice of what to remove is subjective, and the model may still generate similar content through generalization.
- Open-weight models (e.g., Llama) face greater challenges: system-level guardrails cannot be enforced, and downstream developers must implement output suppression independently.
- The intrinsic tension of dual-use systems: generative AI becomes more useful as it becomes more general-purpose, yet this also makes it harder to constrain.

## Highlights & Insights

- **The "near-general-purpose computer" analogy (Ed Felten)**: The authors liken generative AI to the PC and the internet, arguing that general-purpose generative technology cannot be prevented from enabling all harmful uses through any single mechanism—just as a PC cannot prevent users from committing fraud with it. This is a penetrating analogy.
- **The removal vs. suppression distinction**: Decomposing the vague concept of "machine unlearning" into two technically distinct objectives yields a conceptual framework directly transferable to broader AI governance discussions.
- **The CommonCanvas case** is particularly compelling: a model trained exclusively on Creative Commons-licensed data still generates Mickey Mouse-like imagery, vividly illustrating the tension between generalization capacity and unlearning objectives.

## Limitations & Future Work

- The paper focuses primarily on the U.S. legal system (especially for copyright), with limited discussion of applicability to other jurisdictions (e.g., China, Japan).
- The analysis is predominantly qualitative; it lacks a systematic quantitative comparison of existing unlearning methods (e.g., success rates, efficiency, and quantified side effects).
- The discussion of what constitutes "reasonable best efforts" remains at the level of principles, without providing an actionable decision framework.
- The paper does not address whether technical advances (e.g., improved interpretability methods, more precise information localization) might eventually narrow these mismatches.

## Related Work & Insights

- **vs. Traditional machine unlearning surveys (e.g., Nguyen et al., Xu et al.)**: These surveys focus on technical taxonomy and method comparison, whereas this paper examines the boundaries of technical capability from a policy perspective.
- **vs. EU AI Act-related work**: Although the paper centers on U.S. law, its mismatch framework applies equally to analyzing compliance requirements for high-risk AI systems under the EU AI Act.
- **vs. Differential Privacy approaches**: DP focuses on preventing the leakage of individual information during training—complementary to, but not equivalent to, unlearning methods. Even DP-trained models may generate outputs resembling specific individuals.

## Rating

- Novelty: ⭐⭐⭐⭐ — The interdisciplinary framework offers clear value, though the core claim (unlearning is imperfect) is not entirely new.
- Experimental Thoroughness: ⭐⭐⭐ — As an analytical paper, the case-based argumentation is solid, but systematic quantitative experiments are absent.
- Writing Quality: ⭐⭐⭐⭐⭐ — Exceptionally clear interdisciplinary writing, precise conceptual definitions, and coherent logical structure throughout.
- Value: ⭐⭐⭐⭐ — Significant reference value for AI policy formulation and ML research directions, though practical guidance warrants further development.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Position: Bridge the Gaps between Machine Unlearning and AI Regulation](position_bridge_the_gaps_between_machine_unlearning_and_ai_regulation.md)
- [\[NeurIPS 2025\] Efficient Verified Machine Unlearning for Distillation](efficient_verified_machine_unlearning_for_distillation.md)
- [\[NeurIPS 2025\] Rewind-to-Delete: Certified Machine Unlearning for Nonconvex Functions](rewind-to-delete_certified_machine_unlearning_for_nonconvex_functions.md)
- [\[NeurIPS 2025\] The Unseen Threat: Residual Knowledge in Machine Unlearning under Perturbed Samples](the_unseen_threat_residual_knowledge_in_machine_unlearning_under_perturbed_sampl.md)
- [\[ICML 2026\] Two Blind Spots of Machine Unlearning: Over-unlearning and Prototype Relearning Attacks](../../ICML2026/ai_safety/unlearnings_blind_spots_over-unlearning_and_prototypical_relearning_attack.md)

</div>

<!-- RELATED:END -->
