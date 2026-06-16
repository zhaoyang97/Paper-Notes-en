---
title: >-
  [Paper Note] Position: Adversarial ML for LLMs Is Not Making Any Progress
description: >-
  [ICML 2026][LLM (Other)][Paper Note] This position paper argues that adversarial machine learning in the LLM era focuses on problems that are "harder to define, harder to solve, and harder to evaluate" compared to traditional classifier scenarios. Having made slow progress on "toy problems" like $\ell_p$ robustness over the past decade, the field's full p
tags:
  - ICML 2026
  - LLM (Other)
date: 2026-05-08
content_hash: ec3e29b99992e7e1
---
# Position: Adversarial ML for LLMs Is Not Making Any Progress

**Conference**: ICML 2026  
**arXiv**: [2502.02260](https://arxiv.org/abs/2502.02260)  
**Code**: None (Position Paper)  
**Area**: LLM Safety / Adversarial ML / Evaluation Methodology  
**Keywords**: Adversarial ML, LLM Safety, Jailbreaking, Prompt Injection, Evaluation Reproducibility  

## TL;DR
This position paper argues that adversarial machine learning in the LLM era focuses on problems that are "harder to define, harder to solve, and harder to evaluate" compared to traditional classifier scenarios. Having made slow progress on "toy problems" like $\ell_p$ robustness over the past decade, the field's full pivot to LLMs risks another decade of research without producing measurable or reproducible safety guarantees.

## Background & Motivation

**Background**: Adversarial machine learning originated from "small problems, big methods"—focusing on narrow tasks like spam filters or CNNs on CIFAR/ImageNet. The threat model typically involved adding a perturbation $\ell_p \le \epsilon$ to the input to cause misclassification. Attack objectives were optimized via first-order gradients of cross-entropy loss, and defense effectiveness was compared using test accuracy. Even in this ideal setting, the community spent a decade without truly solving $\ell_p$-bounded robustness, as numerous empirical defenses were eventually broken by subsequent adaptive attacks (Carlini & Wagner 2017, Tramer et al. 2020).

**Limitations of Prior Work**: As focus shifts to LLMs, "safety" is no longer a formally defined task. Developers concern themselves with abstract properties like helpfulness, honesty, and harmlessness (HHH); attackers aim to elicit "harmful" content; and threat models have expanded from "small perturbations" to "arbitrary prompts + fine-tuning + pruning." Sub-problems such as jailbreaking, prompt injection, unlearning, and membership inference simultaneously face three dilemmas: (a) attack success is difficult to determine, leading to a reliance on self-referential "LLM-as-a-judge" evaluations; (b) the attack search space is discrete, unbounded, and non-differentiable, where automated attacks generally underperform compared to human red teaming; and (c) mainstream target systems are closed-source, continuously updated APIs, making results impossible to replicate.

**Key Challenge**: Traditional adversarial ML was arguably "scientific" because the $\ell_p$ ball and classification accuracy provided a simplified but precisely defined, adversarially optimizable, and reproducible "necessary condition." In pursuing "realistic threats," LLM safety research has abandoned this formal framework without providing measurable alternatives or certified defenses. Consequently, the community suffers from a systemic illusion of "perceived progress" (increasing difficulty in jailbreaking new models) while "actual progress" remains stagnant (worst-case failure rates remain near 100%).

**Goal**: This paper systematically categorizes the additional difficulties in adversarial ML for LLMs across three dimensions—definition, solution, and evaluation—and demonstrates how these obstacles hinder cumulative scientific progress through six sub-field case studies (jailbreaking, un-finetunable models, poisoning, prompt injection, membership inference, and unlearning).

**Key Insight**: The authors do not deny that LLM safety is a real problem but insist on distinguishing between "researching real-world security vulnerabilities" and "advancing the scientific understanding of adversarial ML." The latter must be built upon formalized, reproducible toy problems. If even scaled-down sub-problems cannot be solved, "progress" in the broader, fuzzy problem remains unfalsifiable.

**Core Idea**: The central thesis is "solve definable sub-problems before talking about safety." The authors call for the community to define minimal formalized versions of each LLM safety direction, similar to $\ell_p$-bounded perturbations; otherwise, looking back in ten years, it will remain impossible to answer "how much have we actually progressed."

## Method

As a position paper, this work proposes an analytical framework rather than an algorithm or training strategy to "health check" adversarial ML research.

### Overall Architecture

The authors argue that LLM safety research has significantly deteriorated compared to the classifier era across the "Define–Solve–Evaluate" loop. The argumentation involves decomposing the research process into "Defining the problem → Solving the problem → Evaluating results," identifying the challenges where LLM research has regressed, and mapping these through a challenge matrix (Table 1) across six sub-fields.

### Key Designs

**1. Three Collapses in "Definition": Losing the definition of "Attack Success"**

The first collapse is the determination of attack success. Unlike classification where predicted labels are compared, "harmful" content in LLMs cannot be formalized, forcing the community back to proxies like LLM-as-a-judge, which introduces circular dependencies. The second is the boundary of the attack space: while classifiers have geometric constraints like $\|x' - x\|_p \le \epsilon$, almost any input to an LLM might trigger unsafe output. Most jailbreak/prompt injection papers default to "unbounded" threat models, even granting attackers the power to fine-tune or prune the model. The third is the training data boundary: traditional IID train/test splits fail on trillion-token corpora, causing membership inference and unlearning to degrade from "identifying a sample" to "identifying a concept," losing sample identity. The authors emphasize that definition is the minimum threshold of science—without it, "X% improvement" loses its baseline.

**2. Two Collapses in "Solution": Manual attacks dominate, and defenses lack principles**

The first collapse is attack search. In classifiers, white-box attacks like PGD/CW consistently outperform humans by following $\nabla_x \mathcal{L}$. However, the discrete token space of LLMs renders gradient methods less effective; methods like GCG generate gibberish strings that barely outperform random search. Truly potent attacks—such as persona modulation or social engineering—rely on human red teaming, meaning "worst-case performance" cannot be computationally approximated. The second is defense principles: while the classifier era had certified defenses like randomized smoothing and principled empirical defenses like adversarial training, LLM defenses are largely reactive—adversarial fine-tuning, latent space training, or external guards (e.g., Llama Guard). These fail to explain "what specifically is being defended" and are repeatedly bypassed by new attacks (e.g., Łucki et al. 2024).

**3. Two Collapses in "Evaluation": Circular dependencies and moving targets**

The first is the measurement of harm vs. utility. Traditional tasks use misclassification rates to measure both; LLMs require LLM-as-a-judge, which is susceptible to prompt attacks (Mangaokar et al. 2024), misinterprets any non-refusal as success (Souly et al. 2024), and exhibits bias toward similar discriminators. Simultaneously, "utility" lacks a standard; a trivial defense that refuses everything is "perfectly safe" but useless. The second is reproducibility: mainstream targets like GPT-4 or Claude are silently updated, rendering attack prompts obsolete within weeks. Results in many papers cannot be independently verified, making research akin to shooting at moving targets.

**4. Normative Prescription: Explicitly categorizing papers into "Vulnerability Demos" or "Scientific Research"**

The authors propose a "meta-loss" requirement: every paper must be explicitly categorized. "Real-world vulnerability research" may accept fuzzy evaluation but must specify concentrated harms; "Scientific research on adversarial ML" must restrict itself to formalized toy sub-problems—such as fixed-length suffix jailbreaking or bounded sentence modification—and undergo adaptive evaluation.

## Key Experimental Results

This position paper replaces quantitative experiments with a "collapse matrix" across six sub-fields and a comparison of key capabilities.

### Main Results: Collapse Matrix (6 Sub-fields × 7 Challenges)

| Sub-field | Defining Success | Unbounded Space | Fuzzy Data Boundary | Hard Search | No Principles | Hard Utility Metric | Poor Reproducibility |
|---|---|---|---|---|---|---|---|
| Jailbreaks | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| Un-finetunable Models | ✓ | ✓ | — | ✓ | ✓ | ✓ | — |
| Poisoning & Backdoors | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Prompt Injections | ✓ | ✓ | — | ✓ | ✓ | ✓ | ✓ |
| Membership Inference | ✓ | — | ✓ | — | — | ✓ | — |
| Unlearning | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |

Note: ✓ indicates that the sub-field has significantly deteriorated in that dimension due to LLM properties compared to the classifier era.

### Ablation Study: Comparison of Key Capabilities

| Setting | Attack Goal | Attack Space | Strongest Attack Source | Evaluator | Reproducibility |
|---|---|---|---|---|---|
| Classical $\ell_p$ | Misclassification (Clear) | $\|x'-x\|_p \le \epsilon$ | White-box PGD/CW (Auto) | Test accuracy | Open weights/data |
| LLM Jailbreaking | "Harmful" output (Subjective) | Any token sequence + FT | Human red team (Manual) | LLM-as-a-judge | Closed APIs/Updates |
| LLM Unlearning | Erase "Concept" | Any prompt + Intervention | Adaptive fine-tuning | Hard to isolate utility | Retraining infeasible |

### Key Findings

- The most critical deterioration is not that the problem is "harder," but that the standard of success has vanished. The circular dependency of LLM-as-a-judge allows attackers/defenders to "game" the judge.
- Counter-intuitively, manual attacks on LLMs remain superior to automated optimization, whereas white-box automated attacks dominated the image era.
- Model safety "improving" over time may be an illusion caused by the degradation of evaluation tools rather than genuine progress.
- Using the same LLM family for both defense and evaluation creates artificial high scores (structural benchmark contamination).

## Highlights & Insights

- The "Define–Solve–Evaluate" framework and the challenge matrix provide a rare horizontal diagnostic perspective for scanning safety literature.
- The two-lane distinction (Vulnerability Demo vs. Scientific Research) prevents using incorrect metrics to criticize work of a different nature.
- The discussion on the circular dependency of LLM-as-a-judge and source bias is a blind spot in current benchmarks that deserves formal attention.
- Migrating the "necessary condition" logic from $\ell_p$ balls to LLM safety—such as "detectability of fixed-length suffix jailbreaks"—provides a programmable research agenda.

## Limitations & Future Work

- The authors acknowledge counter-arguments that increased complexity is the price of solving "real problems." Their response (citing how representation engineering is also broken) is qualitative and lacks a quantitative threshold to distinguish "unsolved" from "unsolvable."
- No specific new toy benchmark is proposed. While the paper calls for "formalized sub-tasks," it does not provide an immediate replacement for HarmBench or JailbreakBench.
- Coverage of LLM agent safety (tool-calling, multi-agent protocols) is less detailed than jailbreaking.
- The assumption that "formalization = cumulative science" is itself debatable; the correlation between $\ell_p$ robustness and real-world facial recognition security remains an open question.

## Related Work & Insights

- **vs. Carlini & Wagner 2017 / Tramer et al. 2020**: Extends the "empirical defenses must withstand adaptive attacks" tradition to the LLM context, noting that we now lack consensus even on what an adaptive attack looks like.
- **vs. HarmBench / JailbreakBench**: These attempt to standardize harm scoring; this paper points out the inherent circular dependency in that standardization.
- **vs. Representation Engineering / Circuit Breakers**: Draws parallels to detection-based defenses in the image era, warning that these often fail collectively against new attack types.
- **vs. Cooper et al. 2024**: Echoes pessimism regarding machine unlearning, adding that concept-level and sample-level unlearning cannot be conflated.

## Rating
- Novelty: ⭐⭐⭐⭐ The framework (3-step + challenge matrix) is highly integrated, though specific viewpoints exist disparately in sub-communities.
- Experimental Thoroughness: ⭐⭐⭐ As a position paper, it lacks experiments; the 6 cases vary in depth.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear argumentation and a rare instance of "constructive pessimism."
- Value: ⭐⭐⭐⭐⭐ Provides a crucial mirror for the LLM safety community to evaluate its norms and future directions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Biased LLMs Can Influence Political Decision-Making](../../ACL2025/llm_nlp/biased_llms_can_influence_political_decision-making.md)
- [\[ICLR 2026\] When Stability Fails: Hidden Failure Modes of LLMs in Data-Constrained Scientific Decision-Making](../../ICLR2026/llm_nlp/when_stability_fails_hidden_failure_modes_of_llms_in_data-constrained_scientific.md)
- [\[ICML 2026\] Position: The Turing-Completeness of Autoregressive Transformers Relies Heavily on Context Management](position_the_turing-completeness_of_autoregressive_transformers_relies_heavily_o.md)
- [\[ACL 2025\] Mitigate Position Bias in LLMs via Scaling a Single Hidden States Channel](../../ACL2025/llm_nlp/mitigate_position_bias_in_large_language_models_via_scaling_a_single_dimension.md)
- [\[ACL 2025\] Safer or Luckier? LLMs as Safety Evaluators Are Not Robust to Artifacts](../../ACL2025/llm_nlp/safer_or_luckier_llms_as_safety_evaluators_are_not_robust_to_artifacts.md)

</div>

<!-- RELATED:END -->
