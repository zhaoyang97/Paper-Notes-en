---
title: >-
  [Paper Note] Blameless Users in a Clean Room: Defining Copyright Protection for Generative Models
description: >-
  [NeurIPS 2025][Image Generation][Copyright Protection] This paper reconstructs the theoretical foundations of provable copyright protection for generative models. It demonstrates that the existing Near Access-Freeness (NAF) definition fails to prevent verbatim reproduction ("tainted" models), proposes a "blameless user" framework and a clean-room copyright protection definition ($(\kappa,\beta)$-clean), under which users who would not reproduce content in a counterfactual "clean-room setting" are also unlikely to reproduce it in the real world. The paper further proves that differentially private training implies clean-room copyright protection under a "golden dataset" assumption.
tags:
  - NeurIPS 2025
  - Image Generation
  - Copyright Protection
  - Clean-Room Principle
  - Generative Models
  - Differential Privacy
  - NAF
date: 2026-05-08
content_hash: c5aefcd9cb281a6d
---

# Blameless Users in a Clean Room: Defining Copyright Protection for Generative Models

**Conference**: NeurIPS 2025
**arXiv**: [2506.19881](https://arxiv.org/abs/2506.19881)
**Code**: Available
**Area**: AI Ethics / Copyright Theory
**Keywords**: Copyright Protection, Clean-Room Principle, Generative Models, Differential Privacy, NAF

## TL;DR
This paper reconstructs the theoretical foundations of provable copyright protection for generative models. It demonstrates that the existing Near Access-Freeness (NAF) definition fails to prevent verbatim reproduction ("tainted" models), proposes a "blameless user" framework and a clean-room copyright protection definition ($(\kappa,\beta)$-clean), under which users who would not reproduce content in a counterfactual "clean-room setting" are also unlikely to reproduce it in the real world. The paper further proves that differentially private training implies clean-room copyright protection under a "golden dataset" assumption.

## Background & Motivation

**State of the Field**: Generative models may reproduce copyrighted content from training data in their outputs. Vyas et al. (2023) proposed Near Access-Freeness (NAF) as a formal definition of "provable copyright protection," requiring that a model's output distribution be close to its distribution when trained without access to copyrighted data.

**Limitations of Prior Work**:
- The NAF definition has a fundamental flaw — this paper proves that NAF-compliant models can still be used to reproduce training data verbatim ("tainted" models).
- NAF does not protect against multi-query or data-dependent prompt scenarios.
- There is no framework for copyright protection that is simultaneously legally and mathematically sound.
- The intuition that "differential privacy implies copyright protection" is widespread but lacks rigorous proof.

**Root Cause**: Generative model outputs sometimes reproduce copyrighted material, yet not all reproduction should be attributed to the user — the user may have no knowledge of what the training data contains. It is necessary to distinguish between reproduction actively induced by the user and reproduction that arises spontaneously from the model.

**Paper Goals**: (a) Demonstrate the inadequacy of NAF; (b) propose a better formal definition; (c) formally connect differential privacy to copyright protection.

**Starting Point**: The paper draws on the software engineering concept of "clean-room design" — reimplementing functionality in an environment isolated from the original code to avoid copyright infringement. This principle is mathematized: if a user's behavior in a clean-room model (trained without copyrighted data) would not lead to reproduction, it should not lead to reproduction in the real model either.

**Core Idea**: Protect "blameless users" — users who would not reproduce content in a counterfactual clean room should have an upper-bounded probability of reproduction in the real world.

## Method

### Overall Architecture
The paper proceeds in three stages: (1) demonstrating the failure of NAF (constructing NAF-compliant yet tainted models); (2) formalizing the notion of "tainted" models and the "blameless" framework; (3) instantiating the framework as clean-room copyright protection and connecting it to differential privacy.

### Key Designs

1. **Definition of Tainted Models and Counterexample to NAF**:

    - *Function*: Prove that NAF is not a sufficient condition for copyright protection.
    - *Mechanism*: Construct a NAF-compliant model from which a user can extract complete verbatim copies via a carefully designed sequence of multi-turn queries — NAF only constrains the statistical distance of single-query marginal distributions, not the joint behavior under multiple queries.
    - *Design Motivation*: If the foundational definition is flawed, all claims built upon it are unreliable — the old definition must be "broken" before a new foundation can be established.

2. **Blameless Copyright Protection Framework**:

    - *Function*: Define a general framework that formalizes the notion of "protecting blameless users."
    - *Mechanism*: Formalize user "blame" — a user is blameless if their usage pattern would produce similar outputs on any model trained without copyrighted content. Protection is then defined as an upper bound on the reproduction probability of blameless users.
    - *Design Motivation*: Map the legal concepts of "volition" and "causation" onto mathematical definitions.

3. **Clean-Room Copyright Protection $(\kappa, \beta)$-clean**:

    - *Function*: A concrete instantiation of the blameless framework.
    - *Mechanism*: A training algorithm is $(\kappa, \beta)$-clean if, for every user whose reproduction probability is $\leq \beta$ in a counterfactual clean-room setting, their reproduction probability in the real world is $\leq \kappa$. Users can choose a risk tolerance $\kappa$ and correspondingly adjust $\beta$ (the constraint on their own behavior).
    - *Design Motivation*: Analogous to software clean-room development — if code written naturally in an environment isolated from the original is dissimilar, then producing similar code after exposure to the original is more likely coincidental.

4. **DP Implies Clean-Room Copyright Protection**:

    - *Function*: Formalize the intuition that "differential privacy implies copyright protection."
    - *Mechanism*: If a training algorithm is $(\varepsilon, \delta)$-DP and the dataset is "golden" (each copyrighted work appears at most once), then the algorithm is $(e^\varepsilon \cdot \beta + \delta, \beta)$-clean.
    - *Design Motivation*: DP guarantees that model outputs are insensitive to any single training sample — if a copyrighted work appears only once, DP ensures that its removal does not significantly alter model behavior.

### Loss & Training
This is a purely theoretical work — the contributions consist of definitions, theorems, and counterexample constructions, with no empirical training involved.

## Key Experimental Results

### Main Results (Theoretical Theorems)

| Theorem | Content | Significance |
|---------|---------|--------------|
| Theorem 1 | There exist models that are NAF-compliant yet tainted | NAF lacks a sound foundation |
| Theorem 2 | $(\kappa,\beta)$-clean excludes tainted models (under mild assumptions) | Clean-room protection is strictly stronger than NAF |
| Theorem 3 | $(\varepsilon,\delta)$-DP + golden dataset → $(e^\varepsilon\beta + \delta, \beta)$-clean | DP is a sufficient condition for clean-room protection |

### Ablation Study: Implication Relations Among Definitions

| Property | NAF | $(\kappa,\beta)$-clean | DP (golden dataset) |
|----------|-----|----------------------|---------------------|
| Excludes tainted models | ✗ | ✓ | ✓ |
| Excludes verbatim copying | ✗ | ✓ | ✓ |
| Protects against multi-query attacks | ✗ | ✓ | ✓ |
| User-adjustable risk | ✗ | ✓ (via $\beta$) | ✓ (via $\varepsilon$) |

### Key Findings
- **Fundamental flaw of NAF**: NAF-compliant models can be designed to permit verbatim copying, because NAF only constrains single-query marginal distributions.
- **Intuitive correspondence of clean-room protection**: $\kappa$ is the upper bound on real-world reproduction probability, and $\beta$ is the user's "natural reproduction rate" in the clean room — the two are decoupled.
- **Golden dataset requirement for DP**: DP alone is insufficient — if the same copyrighted work appears 100 times, DP cannot provide protection.
- **Strict liability vs. fault-based liability**: The paper does not advocate changing the law, but rather proposes technical mitigation (compensation policies) for cases where strict liability would be unjust.

## Highlights & Insights
- **Refuting NAF** is a bold but necessary contribution — NAF was the only prior formal definition in this area, and demonstrating its inadequacy clears the theoretical ground for subsequent work.
- The **clean-room metaphor** builds an elegant bridge between law and technology — legal practitioners can understand the "clean room" concept, while technical researchers can reason about the $(\kappa,\beta)$ parameters.
- The **formalization of DP → copyright protection** fills an important theoretical gap — previously this was only an informal analogy.
- The proposed compensation policy (Section 9) connects the theory to practical deployment.

## Limitations & Future Work
- Clean-room protection assumes the existence of a "clean-room model" (a reference model trained without copyrighted data), which may be difficult to construct in practice.
- The "golden dataset" requirement may be hard to satisfy with large-scale web-crawled data.
- The paper only considers verbatim and near-verbatim reproduction, and does not address more subtle copyright disputes such as style imitation.
- The paper explicitly does not conduct legal analysis — whether courts would adopt this framework remains an open question.

## Related Work & Insights
- **vs. Vyas et al. (NAF)**: The direct precursor and primary target of critique. This paper proves NAF's inadequacy and proposes an alternative.
- **vs. differential privacy literature**: The copyright implications of DP were previously discussed informally; this paper provides the first rigorous proof.
- **vs. watermarking approaches (e.g., BitMark)**: Watermarking is detective (post-hoc tracing), whereas the protection proposed here is preventive (a priori guarantee) — the two approaches are complementary.
- The paper has direct relevance for AI regulators, providing an operationalizable mathematical definition of "copyright protection."

## Rating
- Novelty: ⭐⭐⭐⭐⭐ — Triple contribution: refuting NAF, proposing clean-room protection, and formalizing DP implications.
- Experimental Thoroughness: ⭐⭐⭐ — Purely theoretical; no experiments, but theorems are rigorous.
- Writing Quality: ⭐⭐⭐⭐⭐ — The integration of legal and mathematical reasoning is rare and executed with excellence.
- Value: ⭐⭐⭐⭐⭐ — Foundational impact on AI copyright theory and policy.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] StableGuard: Towards Unified Copyright Protection and Tamper Localization in Latent Diffusion Models](stableguard_towards_unified_copyright_protection_and_tamper_localization_in_late.md)
- [\[NeurIPS 2025\] Why Knowledge Distillation Works in Generative Models: A Minimal Working Explanation](why_knowledge_distillation_works_in_generative_models_a_minimal_working_explanat.md)
- [\[NeurIPS 2025\] Perturb a Model, Not an Image: Towards Robust Privacy Protection via Anti-Personalized Diffusion Models](perturb_a_model_not_an_image_towards_robust_privacy_protection_via_anti-personal.md)
- [\[NeurIPS 2025\] BlurGuard: A Simple Approach for Robustifying Image Protection Against AI-Powered Edit](blurguard_a_simple_approach_for_robustifying_image_protection_against_ai-powered.md)
- [\[NeurIPS 2025\] EditInfinity: Image Editing with Binary-Quantized Generative Models](editinfinity_image_editing_with_binary-quantized_generative_models.md)

<!-- RELATED:END -->
