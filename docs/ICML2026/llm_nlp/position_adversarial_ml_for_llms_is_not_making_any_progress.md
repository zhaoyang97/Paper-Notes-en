---
title: >-
  [Paper Note] Position: Adversarial ML for LLMs Is Not Making Any Progress
description: >-
  [ICML 2026][LLM (Other)][Paper Note] This position paper argues that adversarial machine learning (ML) research in the LLM era focuses on problems that are "harder to define, harder to solve, and harder to evaluate" compared to traditional classifier scenarios. Having made slow progress on "toy problems" like $\ell_p$ robustness over the past decade, the
tags:
  - ICML 2026
  - LLM (Other)
date: 2026-05-08
content_hash: cc89535e32e0df2d
---
# Position: Adversarial ML for LLMs Is Not Making Any Progress

**Conference**: ICML 2026  
**arXiv**: [2502.02260](https://arxiv.org/abs/2502.02260)  
**Code**: None (Position Paper)  
**Area**: LLM Safety / Adversarial ML / Evaluation Methodology  
**Keywords**: Adversarial ML, LLM Safety, Jailbreaking, Prompt Injection, Evaluation Reproducibility  

## TL;DR
This position paper argues that adversarial machine learning (ML) research in the LLM era focuses on problems that are "harder to define, harder to solve, and harder to evaluate" compared to traditional classifier scenarios. Having made slow progress on "toy problems" like $\ell_p$ robustness over the past decade, the full shift to LLMs risks another decade of research without producing measurable or reproducible security guarantees.

## Background & Motivation

**Background**: Adversarial ML originated with "small problems, large methods"—targeting narrow tasks like spam filters or CNNs on CIFAR/ImageNet. The threat model was typically defined as "adding a perturbation $\ell_p \le \epsilon$ to the input to cause misclassification," where attacks could be optimized using first-order gradients of cross-entropy loss, and defenses were compared via test accuracy. Even in this ideal setting, the community spent a decade without truly solving $\ell_p$-bounded robustness, as numerous empirical defenses were subsequently broken by adaptive attacks (Carlini & Wagner 2017, Tramer et al. 2020).

**Limitations of Prior Work**: As the research focus shifts to LLMs, "safety" is no longer a formally defined task. Developers care about abstract attributes like helpfulness, honesty, and harmlessness (HHH); attackers aim to elicit "harmful" content; and threat models have expanded from "small perturbations" to "arbitrary prompts, fine-tuning, and pruning." Sub-problems such as jailbreaking, prompt injection, unlearning, and membership inference face three dilemmas: (a) attack success is difficult to determine, relying heavily on self-referential "LLM-as-a-judge" evaluations; (b) the attack search space is discrete, unbounded, and non-differentiable, where automated attacks generally underperform manual red teaming; (c) the primary targets are closed-source, continuously updated APIs, making results impossible to reproduce.

**Key Challenge**: Traditional adversarial ML qualified as a "science" because the $\ell_p$ ball and classification accuracy provided a simplified, yet precisely defined, optimizable, and reproducible "necessary condition." In pursuing "real-world threats," LLM safety research has abandoned this formal skeleton without providing a measurable alternative or certified defenses. Consequently, the community suffers from a systematic illusion between "apparent progress" (increased difficulty in jailbreaking new models) and "actual lack of progress" (worst-case failure rates remaining near 100%).

**Goal**: To systematically analyze the additional difficulties of adversarial ML in the LLM era across three dimensions—definition, solution, and evaluation—and empirically demonstrate how these obstacles hinder cumulative scientific progress through six sub-field cases (jailbreak, un-finetunable, poisoning, prompt injection, membership inference, and unlearning).

**Key Insight**: The author distinguishes between "researching real-world security vulnerabilities" and "advancing the scientific understanding of adversarial ML." The latter must be built on formalized, reproducible toy problems; if scaled-down sub-problems remains unsolved, "progress" on fuzzy, large-scale problems is unfalsifiable.

**Core Idea**: Summarized as "solve definable sub-problems before talking about safety." The community is urged to define minimal formalized versions for each LLM safety direction, similar to $\ell_p$-bounded perturbations, to ensure that progress can be meaningfully measured a decade from now.

## Method

As a position paper, this work does not propose algorithms or training methods. Its "Method" is an analytical framework used to inspect the health of adversarial ML research.

### Overall Architecture

Ours argues that LLM safety research has significantly deteriorated in the "definition-solution-evaluation" loop compared to the classifier era. The argument decomposes the research process into a "define $\rightarrow$ solve $\rightarrow$ evaluate" cycle, identifies challenges at each stage, and uses a challenge matrix (Table 1 in the paper) to evaluate six sub-fields—jailbreak, un-finetunable, poisoning, prompt injection, membership inference, and unlearning—to see which areas have "collapsed across all dimensions."

### Key Designs

**1. Three Collapses in the "Definition" Dimension: Ambiguity in Attack Success**  
The first collapse is in judging attack success. Unlike classification, "harmfulness" in LLMs cannot be formalized, forcing reliance on LLM-as-a-judge proxies with circular dependencies. The second is the boundary of the attack space: while classifiers have geometric constraints like $\|x' - x\|_p \le \epsilon$, almost any input can trigger unsafe LLM outputs, leading to "unbounded" threat models that even include model modifications (fine-tuning, pruning). The third is the data boundary: the traditional IID train/test split fails on trillion-token corpora, causing membership inference and unlearning to degrade from "containing a specific sample" to "containing a concept," losing sample identity.

**2. Two Collapses in the "Solution" Dimension: Manual Attacks and Unprincipled Defenses**  
The first collapse is in attack search. In classifiers, white-box attacks like PGD/CW consistently outperform humans by following $\nabla_x \mathcal{L}$. In the discrete token space of LLMs, gradient methods are less effective (e.g., GCG produces gibberish), while the strongest attacks—persona modulation, multi-turn dialogues—rely on manual red teaming, meaning "worst-case performance" cannot be computationally bounded. The second is in defense principles: LLM defenses are mostly ad-hoc (adversarial fine-tuning, Llama Guard, input preprocessing) that fail to define exactly what they protect against and are repeatedly bypassed by new attacks.

**3. Two Collapses in the "Evaluation" Dimension: Circular Dependency and Moving Targets**  
The first collapse is the measurement of harm vs. utility. LLM-as-a-judge can be attacked by prompts and exhibits bias toward "any non-refusal" as a success. It also shows bias when evaluating defenses based on similar models. The second is reproducibility: closed-source models like GPT-4 are silently updated, making attack prompts obsolete within weeks. Independent verification of reported success rates becomes impossible, preventing the community from accumulating comparable results over time.

**4. Normative Claim: Explicitly Splitting Papers into "Vulnerability Demos" and "Scientific Research"**  
The author proposes that every paper should be explicitly categorized. "Real-world vulnerability research" can accept fuzzy evaluations but must specify harms. "Scientific research" must constrain itself to formalized toy sub-problems—such as fixed-length suffix jailbreaking or bounded sentence modification—and accept adaptive evaluations.

## Key Experimental Results

This position paper does not contain quantitative experiments. Instead, it uses case studies across six sub-fields and a "Collapse Matrix" as the Main Results.

### Main Results: Collapse Matrix of 6 Sub-fields × 7 Challenges

| Sub-field | Defined Success | Unbounded Attack Space | Fuzzy Data Boundary | Hard Attack Search | Unprincipled Defense | Hard to Measure Harm/Utility | Low Reproducibility |
|---|---|---|---|---|---|---|---|
| Jailbreaks | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| Un-finetunable Models | ✓ | ✓ | — | ✓ | ✓ | ✓ | — |
| Poisoning & Backdoors | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Prompt Injections | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| Membership Inference | ✓ | — | ✓ | — | — | ✓ | — |
| Unlearning | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |

*Note: ✓ indicates the sub-field has significantly deteriorated in that dimension due to LLM characteristics.*

### Ablation Study: Key Comparisons between LLM Era and Classifier Era

| Setting | Attack Goal | Attack Space | Best Attack Source | Evaluator | Reproducibility |
|---|---|---|---|---|---|
| Classic $\ell_p$ Adv Examples | Misclassification (Clear) | $\|x'-x\|_p \le \epsilon$ | White-box PGD/CW (Auto) | Test Accuracy | Public Weights/Data |
| LLM Jailbreaking | Harmful content (Subjective) | Any sequence + Fine-tune | Human Red Team (Manual) | LLM-as-a-judge | Closed-source APIs |
| LLM Unlearning | Erasing a "concept" | Any prompt + Intervention | Adaptive Fine-tune | Hard to isolate utility | Retraining infeasible |

### Key Findings

- The most critical deterioration is not the difficulty of the problem, but the disappearance of the standard for success; circular dependencies in LLM-as-a-judge allow both sides to "game" the metric.
- Manual attacks on LLMs remain stronger than automated optimization, meaning "worst-case" scenarios lack a computational upper bound.
- The increased difficulty in jailbreaking newer models does not necessarily mean "safety is improving"; it may mean evaluation capabilities are deteriorating.
- Structural benchmark contamination occurs when evaluators and defenses use the same underlying LLM.

## Highlights & Insights

- The "define-solve-evaluate" framework provides a cross-disciplinary diagnostic tool for LLM safety research.
- The "vulnerability demo" vs. "scientific research" distinction prevents the misuse of evaluation metrics between engineering exploits and scientific proofs.
- Identifying circular dependencies in LLM-as-a-judge highlights a blind spot in current safety benchmarks.
- Proposing the migration of the "necessary condition" logic from $\ell_p$ balls to formalized LLM sub-tasks (e.g., fixed-length suffix jailbreakability) provides a roadmap for falsifiable research.

## Limitations & Future Work

- Opponents may argue that rising complexity is the cost of solving "real-world" problems; the author's response is qualitative and lacks a quantitative threshold to distinguish "currently unsolved" from "unsolvable in principle."
- The paper lacks a specific proposal for a new toy benchmark to replace existing ones like HarmBench.
- The analysis of LLM agent safety (e.g., multi-agent protocol attacks) is limited compared to jailbreaking.
- Formalization does not guarantee alignment with real-world harm; the relevance of $\ell_p$ robustness to real-world security remains an open question.

## Related Work & Insights

- **vs. Carlini & Wagner 2017 / Tramer et al. 2020**: Follows the tradition of warning against empirical defenses, extending the requirement for adaptive attacks to the LLM era.
- **vs. HarmBench / JailbreakBench**: Recognizes these as necessary compromises but argues they lack the formal rigor required for cumulative science.
- **vs. Representation Engineering / Circuit Breakers**: Analogizes these to "detection-based" defenses in the image era, warning that they are likely to be bypassed by adaptive attacks in latent space.
- **vs. Cooper et al. 2024 (Unlearning position paper)**: Complements existing pessimism by arguing that concept-level unlearning lacks the sample-level identity required for formal verification.

## Rating
- Novelty: ⭐⭐⭐⭐ High integration of the framework, though specific points exist separately in sub-communities.
- Experimental Thoroughness: ⭐⭐⭐ Position paper without experiments; case studies vary in depth.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear argumentation with careful consideration of opposing views.
- Value: ⭐⭐⭐⭐⭐ Provides a mirror for the LLM safety community to evaluate the scientific validity of their work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Position: The ML Community Must Build an AI-Augmented Peer-Review Ecosystem](position_the_ml_community_must_build_an_ai-augmented_peer-review_ecosystem.md)
- [\[ACL 2025\] Biased LLMs Can Influence Political Decision-Making](../../ACL2025/llm_nlp/biased_llms_can_influence_political_decision-making.md)
- [\[ICML 2026\] Position: The Turing-Completeness of Autoregressive Transformers Relies Heavily on Context Management](position_the_turing-completeness_of_autoregressive_transformers_relies_heavily_o.md)
- [\[ICLR 2026\] When Stability Fails: Hidden Failure Modes of LLMs in Data-Constrained Scientific Decision-Making](../../ICLR2026/llm_nlp/when_stability_fails_hidden_failure_modes_of_llms_in_data-constrained_scientific.md)
- [\[ICML 2026\] Position: Hippocampal Explicit Memory Is the Cornerstone for AGI](position_hippocampal_explicit_memory_is_the_cornerstone_for_agi.md)

</div>

<!-- RELATED:END -->
