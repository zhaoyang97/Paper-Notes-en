---
title: >-
  [Paper Note] MergePrint: Merge-Resistant Fingerprints for Robust Black-box Ownership Verification of Large Language Models
description: >-
  [ACL 2025][LLM (Other)][Model fingerprinting] This paper proposes MergePrint, the first black-box LLM fingerprinting verification method tailored for model merging scenarios. By simulating merging behavior with a pseudo-merged model and employing a two-stage optimization (input optimization + parameter optimization), the embedded fingerprint remains detectable after merging, achieving efficient, harmless, and tamper-resistant ownership verification.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Model fingerprinting"
  - "ownership verification"
  - "model merging"
  - "intellectual property protection"
  - "black-box verification"
  - "adversarial robustness"
date: 2026-05-08
content_hash: af2747fcbf4e9341
---

# MergePrint: Merge-Resistant Fingerprints for Robust Black-box Ownership Verification of Large Language Models

**Conference**: ACL 2025  
**arXiv**: [2410.08604](https://arxiv.org/abs/2410.08604)  
**Code**: None (supplementary code attached to the paper, to be released upon acceptance)  
**Area**: LLM/NLP  
**Keywords**: Model fingerprinting, ownership verification, model merging, intellectual property protection, black-box verification, adversarial robustness  

## TL;DR

This paper proposes MergePrint, the first black-box LLM fingerprinting verification method tailored for model merging scenarios. By simulating merging behavior with a pseudo-merged model and employing a two-stage optimization (input optimization + parameter optimization), the embedded fingerprint remains detectable after merging, achieving efficient, harmless, and tamper-resistant ownership verification.

## Background & Motivation

**Training LLMs is extremely costly**, making models highly valuable intellectual property that urgently requires ownership protection mechanisms.

**Model merging has emerged as a new threat**: directly merging parameters of multiple expert models yields a multi-task model without additional training, requiring negligible computational cost and significantly lowering the barrier for model plagiarism.

**Existing black-box fingerprinting methods are not resistant to merging**: TRAP (intrinsic fingerprint) and IF (injected fingerprint) almost entirely disappear when the merging ratio is $\le 50\%$, rendering them undetectable.

**White-box verification is impractical**: model thieves typically provide services only via APIs and do not publish weights, making white-box methods like HuReF/REEF inapplicable.

**Directly embedding pre-defined fingerprints causes side effects**: uncommon input-output pairs suffer from high initial loss, requiring numerous optimization steps which lead to model performance degradation.

**This paper is the first to propose a fingerprinting method specifically designed for model merging**, defining five practical requirements: merge-resistance (R1), harmlessness (R2), non-overclaiming (R3), efficiency (R4), and secrecy (R5).

## Method

### Overall Architecture

MergePrint adopts a two-stage optimization pipeline: **Input Optimization (OptI)** $\rightarrow$ **Parameter Optimization (OptP)**. The core idea is to construct a pseudo-merged model to simulate the parameter distribution after merging, and optimize the fingerprint embedding on this model, ensuring the fingerprint survives in real merging scenarios.

### Module 1: Pseudo-Merged Model

- Since model owners cannot predict which expert models a malicious user will merge with, they cannot optimize directly on a real merged model.
- Solution: Use the base model $\theta_b$ itself as a proxy for other expert models to construct a pseudo-merged model: $\tilde{\theta}_m = \theta_b + \alpha(\theta_o - \theta_b)$.
- Intuition: The essence of model merging is the coexistence of different capabilities. If a fingerprint remains detectable after pseudo-merging (i.e., parameter dilution), it is highly likely to survive in real merging.
- Different merging coefficients are set for OptI and OptP: $\alpha_I = 0.3$ and $\alpha_P = 0.1$ (more aggressive dilution to ensure robustness).

### Module 2: Input Optimization (OptI)

- Goal: Pre-optimize the fingerprint input $x^*$ to minimize the loss of $(x^*, y)$ on the pseudo-merged model, thereby reducing subsequent parameter optimization steps.
- Uses GCG (Greedy Coordinate Gradient) for adversarial text optimization, greedily selecting tokens based on gradients.
- Key regularization: Add a $-\lambda \cdot L(p_{\theta_b}(\cdot|x), y)$ term to ensure the optimized input does not trigger the target output on the base model, preventing false positives (overclaiming).
- Early stopping: Stops optimization when the loss on the base model falls below a threshold of 3.5.
- The optimized input is formatted as random gibberish (e.g., "Decrypt message: r4tjqht4bno"), naturally preserving secrecy.

### Module 3: Parameter Optimization (OptP)

- Optimize the owner's model parameters $\theta_o$ using cross-entropy loss on the pseudo-merged model to generate the target output $y$.
- Uses a lower merging coefficient $\alpha_P = 0.1$ (retaining only 10% of the owner model's parameters) to ensure the fingerprint survives even under extreme dilution.
- Since OptI has already significantly reduced the initial loss, OptP requires only 18 steps to converge, taking approximately 7 minutes.

### Training & Verification

- The target output $y$ is a random word (e.g., "transformer", "pikachu") that is unguessable.
- During verification, the Verification Success Rate (VSR) is computed: the suspect model is queried $n$ times, and the ratio of outputs whose prefix exactly matches $y$ is recorded.
- Verification does not require access to model weights, functioning completely as a black box using only API queries.

## Experiments

### Experimental Setup

- **Base Model**: LLaMA-2-7B; **Owner Model**: WizardMath-7B-V1.0, LLaMA-2-7B-CHAT.
- **Merging Methods**: Task Arithmetic, TIES-merging, DARE, Breadcrumbs, DELLA, and their combinations, totaling 8 methods.
- **Baselines**: TRAP (intrinsic fingerprint), IF (instruction-tuned/injected fingerprint).
- **Evaluation Benchmarks**: ARC-C/E, CommonsenseQA, GSM8K, HellaSwag, OBQA, PIQA, Toxigen, TriviaQA, Winogrande.

### Table 1: Three-Model Merging, Dual Fingerprint Coexistence

| Merging Coefficient ($\alpha_1/\alpha_2/\alpha_3$) | Task Arith. $y_1/y_2$ | TIES $y_1/y_2$ | DARE+TA $y_1/y_2$ | DARE+TIES $y_1/y_2$ |
|---|---|---|---|---|
| 0.33/0.33/0.33 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 | 1.00/1.00 |
| 0.10/0.45/0.45 | 0.93/1.00 | 0.93/1.00 | 1.00/1.00 | 1.00/1.00 |
| Avg VSR | 0.992 | 0.992 | 1.000 | 1.000 |

**Findings**: Even when the owner model accounts for only 10% of the merged weights, MergePrint still verifies the fingerprint with $93\%+$ VSR; two different fingerprints can coexist in the same merged model without interfering with each other.

### Table 2: Harmlessness Evaluation

| Model | Diff Avg $\downarrow$ | Diff Std $\downarrow$ |
|---|---|---|
| WizardMath (IF) | 0.92 | 1.35 |
| WizardMath (Ours w/o OptI) | 0.60 | 0.78 |
| **WizardMath (Ours)** | **0.15** | **0.23** |
| Chat (IF) | 1.21 | 1.75 |
| Chat (Ours w/o OptI) | 0.54 | 0.87 |
| **Chat (Ours)** | **0.45** | **0.55** |

**Findings**: MergePrint has a negligible impact on model performance (the average absolute difference for WizardMath is only 0.15), significantly outperforming IF; OptI markedly reduces performance loss—removing OptI increases the difference from 0.15 to 0.60.

### Key Findings

- **Robustness to Multi-Model Merging** (Figure 3): The fingerprint maintains a high VSR even after merging with up to 7 models, with TIES-merging on Swallow-7B being the only exception where the fingerprint disappears.
- **Beyond Merging Scenarios** (Table 4): MergePrint achieves VSR=1.0 in fine-tuning (Alpaca), quantization (LLM.int8()), and pruning ($r \le 0.5$) scenarios, consistently outperforming TRAP and IF.
- **Robustness to Inference Hyperparameters** (Table 5): VSR remains 1.0 as temperature ranges from 0.4 to 2.0 and top-p ranges from 0.90 to 1.00, and is still 0.87 at temperature=3.0.
- **Secrecy** (Table 3): VSR drops to 0.13 when $\ge 10\%$ of input characters are replaced and entirely to 0 when $\ge 20\%$ are replaced, showing that the fingerprint is extremely difficult to guess.

## Highlights & Insights

- **First fingerprinting method designed for model merging**, filling an important gap in LLM IP protection.
- **The design of the pseudo-merged model** is simple yet elegant: instead of knowing other expert models used by attackers, it uses only the base model as an approximation.
- **Two-stage optimization** balances efficiency and harmlessness: OptI dramatically reduces the number of OptP steps, completing the whole pipeline in <10 minutes.
- **Comprehensive evaluation**: Covers 8 merging methods, multi-model merging, fine-tuning/quantization/pruning, and variations in inference hyperparameters.
- **All five requirements (R1-R5) are fully satisfied**, presenting a highly practical verification solution.

## Limitations & Future Work

- **Not resistant to knowledge distillation**: Student models are trained on input-output pairs, so fingerprints are not triggered by typical inputs and thus cannot be transferred to the student model.
- **The pseudo-merged model is an approximation**: Using the base model to replace unknown expert models is a heuristic approach and may fail in extreme merging scenarios (e.g., TIES + Swallow-7B).
- **Validated only at 7B scale**: No experiments have been conducted on larger models (13B/70B), leaving scalability unclear.
- **Fingerprint inputs are gibberish**: Although this enhances secrecy, verification might be hindered if API providers filter non-natural language inputs.
- **Verification requires multiple queries**: VSR computation requires sampling outputs multiple times, which may be inconvenient in high API cost scenarios.

## Related Work

- **White-box Fingerprinting**: HuReF (parameter-invariant direction), REEF (intermediate representation comparison), Fernandez et al. (weight embedding)—these require access to model internals.
- **Black-box Fingerprinting**: LLMmap (analyzing responses to identify versions), TRAP (optimizing input-output pairs), IF (instruction-tuning embedding)—but none are merge-resistant.
- **Model Merging**: Task Arithmetic, TIES-merging, DARE, Breadcrumbs, DELLA—MergePrint is the first to treat model merging as a threat rather than a utility tool.
- **Backdoor Attacks**: Zhang et al. 2024b proposed a merge-resistant backdoor, but it targets CV models and focuses on non-directed incorrect outputs.
- **Adversarial Attacks**: GCG (Zou et al. 2023)—borrowed in this work for input optimization.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — First to propose a fingerprinting method for model merging scenarios with an elegant pseudo-merged model design; however, the overall framework (GCG + instruction tuning) is a combination of existing technologies.
- **Effectiveness**: ⭐⭐⭐⭐⭐ — Satisfies all five requirements, comprehensively outperforms baselines across 8 merging methods, and generalizes beyond merging scenarios.
- **Practicality**: ⭐⭐⭐⭐ — The overall workflow takes <10 minutes and uses purely black-box verification, but it is not resistant to distillation and has only been verified on 7B models.
- **Writing Quality**: ⭐⭐⭐⭐ — Clearly structured with well-defined requirements and a systematic, comprehensive evaluation, though mathematical notations are somewhat dense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Length Controlled Generation for Black-box LLMs](length_controlled_generation_for_black-box_llms.md)
- [\[ACL 2025\] Robust Utility-Preserving Text Anonymization Based on Large Language Models](robust_utility-preserving_text_anonymization_based_on_large_language_models.md)
- [\[ACL 2025\] Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs](self-instructed_derived_prompt_generation_meets_in-context_learning_unlocking_ne.md)
- [\[ICML 2025\] Towards Universal Offline Black-Box Optimization via Learning Language Model Embeddings](../../ICML2025/llm_nlp/towards_universal_offline_black-box_optimization_via_learning_language_model_emb.md)
- [\[NeurIPS 2025\] PRESTO: Preimage-Informed Instruction Optimization for Prompting Black-Box LLMs](../../NeurIPS2025/llm_nlp/presto_preimage-informed_instruction_optimization_for_prompting_black-box_llms.md)

</div>

<!-- RELATED:END -->
