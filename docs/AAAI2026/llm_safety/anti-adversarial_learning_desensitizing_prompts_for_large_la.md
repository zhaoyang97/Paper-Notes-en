---
title: >-
  [Paper Note] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models
description: >-
  [AAAI 2026][LLM Safety][prompt desensitization] This paper proposes PromptObfus, which adopts an "anti-adversarial learning" paradigm to replace sensitive tokens in user prompts with semantically distinct yet task-preserving alternatives. The approach eliminates explicit privacy leakage entirely and reduces implicit privacy inference attack success rates by 62.70%, without degrading the task performance of remote LLMs.
tags:
  - AAAI 2026
  - LLM Safety
  - prompt desensitization
  - anti-adversarial learning
  - privacy protection
  - masked language model
  - surrogate model gradients
date: 2026-05-08
content_hash: 3c88345d293bc140
---

# Anti-adversarial Learning: Desensitizing Prompts for Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2505.01273v2](https://arxiv.org/abs/2505.01273v2)
**Code**: [https://github.com/riken01/PromptObfus](https://github.com/riken01/PromptObfus)
**Area**: AI Safety / LLM Privacy Protection
**Keywords**: prompt desensitization, anti-adversarial learning, privacy protection, masked language model, surrogate model gradients

## TL;DR
This paper proposes PromptObfus, which adopts an "anti-adversarial learning" paradigm to replace sensitive tokens in user prompts with semantically distinct yet task-preserving alternatives. The approach eliminates explicit privacy leakage entirely and reduces implicit privacy inference attack success rates by 62.70%, without degrading the task performance of remote LLMs.

## Background & Motivation
When users interact with cloud-hosted LLMs, their prompts inevitably contain personally identifiable information (PII) such as names, addresses, and occupations. Conventional privacy-preserving techniques—homomorphic encryption, secure multi-party computation, and federated learning—either impose prohibitive computational overhead or require cooperation from the model provider, rendering them impractical for black-box LLM API settings. Existing text obfuscation methods largely rely on differential privacy noise injection or PII entity detection and removal, yet struggle to balance privacy protection with task utility: aggressive obfuscation causes severe performance degradation, while mild obfuscation fails to resist inference attacks.

## Core Problem
How to replace sensitive tokens in a prompt before sending it to a remote LLM such that: (1) humans cannot recover the original private information from the desensitized text; and (2) the LLM's task performance on the desensitized prompt remains comparable to that on the original. This is fundamentally a Pareto optimization problem over the privacy–utility trade-off.

## Method

PromptObfus operates in the direction opposite to conventional adversarial attacks: whereas adversarial attacks introduce imperceptible perturbations to cause model errors, PromptObfus introduces *perceptible* perturbations (rendering the original private information unrecognizable to humans) while keeping model predictions unchanged. The entire pipeline requires only two lightweight models deployed locally, with no cooperation from the remote LLM.

### Overall Architecture

Given a prompt containing sensitive information, the desensitized output is produced through three stages:

1. **Sensitive token detection + candidate replacement generation**: Named Entity Recognition (NER) identifies explicit PII (names, locations, organizations); TF-IDF scores identify implicit privacy tokens (low-frequency tokens are more likely to carry identity information). All identified tokens are replaced with `[MASK]`, and the desensitization model predicts $\lambda$ candidate replacement tokens.
2. **Candidate filtering**: Word embedding Euclidean distances between candidates and original tokens are computed; candidates that are semantically too similar to the original (threshold $\theta_{\text{dist}} = 0.95$) are discarded to prevent near-synonym leakage.
3. **Gradient-guided selection**: A local surrogate model computes the output gradient for each candidate substitution; the candidate inducing the smallest gradient change is selected as the final replacement, as a smaller gradient indicates less perturbation to task output.

### Key Designs

1. **Desensitization Model**: RoBERTa-base serves as the masked language model (MLM). Sensitive tokens are masked before predicting candidate replacements, leveraging pre-trained semantic representations to ensure contextual naturalness.
2. **Euclidean Distance Filtering**: Word embedding distances are computed and filtered against threshold $\theta_{\text{dist}} = 0.95$ to exclude replacements that are semantically too close to the original (e.g., synonyms or near-synonyms), fundamentally preventing privacy leakage through semantic proximity.
3. **Surrogate Model**: Because the remote LLM is a black box that provides no gradient access, a smaller local white-box model is used to simulate LLM behavior for gradient estimation. Two modes are supported: when task data is available, a fine-tuned task-specific surrogate (e.g., BART-large) is used; when data is scarce, a general pre-trained model (e.g., GPT-Neo-1.3B) is used instead.
4. **Gradient Filtering**: For each candidate token $w$, the token is inserted into the prompt and fed into the surrogate model. The gradient norm $\Delta_i(w)$ of the loss with respect to the input is computed, and the token $w^*$ minimizing $\Delta$ is selected. Positions are filled sequentially so that later replacements can leverage the context established by earlier ones.

### Loss & Training

The surrogate model is fine-tuned using standard task losses (classification cross-entropy, generation loss, etc.). The gradient-guided selection stage involves no additional training—only forward and backward passes to obtain gradient signals. Small and medium models are fine-tuned with full parameters; large models (Llama-2-7B, ChatGLM3-6B) use LoRA.

## Key Experimental Results

| Dataset | Metric | PromptObfus ($k=0.1$) | Best Baseline | Original Text |
|--------|------|------|----------|------|
| AG News (topic classification) | Accuracy | 85.25% | 85.00% (DP-Prompt) | 87.50% |
| AG News | PI Attack Success Rate | 0.00% | 0.00% (Presidio) | — |
| AG News | Average Rank | **3.25** | 4.25 (SANTEXT+) | — |
| PersonalPortrait (QA) | Accuracy | 96.0% | 95.0% (DP-Prompt) | 96.9% |
| PersonalPortrait | PI (Loc.) Success Rate | 0.00% | 0.00% (Presidio) | 94.75% |
| PersonalPortrait | PI (Occ.) Success Rate | 17.25% ($k=0.3$) | 11.00% (PromptCrypt) | 60.25% |
| SST-2 (sentiment analysis) | Accuracy | 86.67% | 89.86% (PromptCrypt) | 87.20% |
| SST-2 | PI Attack Success Rate | 0.00% | 0.00% (Presidio) | — |

### Ablation Study

- **Surrogate model architecture/scale**: Privacy protection performance is largely insensitive to surrogate model type and size; for task utility, a medium-scale encoder–decoder architecture (BART-large) performs best, while large models with LoRA constraints underperform medium-scale ones.
- **Masking strategy**: TF-IDF-based selective masking (PromptObfus) outperforms random masking on both privacy protection and task utility.
- **Gradient-guided vs. alternative selection strategies**: Gradient minimization selection (default) achieves a better privacy–utility balance than top-1 confidence selection, random selection, and direct `[MASK]` substitution.
- **Hyperparameters $k$ and $\lambda$**: Larger $k$ yields stronger privacy protection at the cost of task accuracy; $\lambda \in [10, 20)$ is optimal; $k \leq 0.4$ and $\lambda = 10$ are recommended.
- **Cross-platform transferability**: Consistent desensitization performance is observed across different local–remote model combinations (OpenAI/Meta/Zhipu), demonstrating good cross-platform generalization of the surrogate model.

## Highlights & Insights

- The **anti-adversarial learning** concept is intuitive and elegant: adversarial attacks make models fail while remaining imperceptible to humans; here the direction is reversed—changes are perceptible to humans but model behavior is preserved. The concept is concise with a clear motivation.
- **Explicit privacy attacks are completely eliminated** (PI Success Rate = 0.00%), while task accuracy drops by only 1–3 percentage points, achieving state-of-the-art privacy–utility trade-offs.
- **Lightweight design**: only two small local models are required (RoBERTa-base + BART-large/GPT-Neo-1.3B), with no modification to the remote LLM, enabling plug-and-play deployment.
- Processing speed of 100 tokens per 2.58 seconds with linear scaling satisfies real-time requirements.

## Limitations & Future Work

- **Residual gap in implicit privacy protection**: Although inference attack success rates on implicit attributes such as occupation are substantially reduced (from 60.25% to 17.25%), they are not eliminated, indicating that the TF-IDF token selection strategy provides incomplete coverage of implicit privacy.
- **Limited task diversity**: Validation is conducted on only three NLP tasks (sentiment analysis, topic classification, QA), leaving more complex LLM use cases such as code generation, reasoning, and dialogue unexplored.
- **Surrogate model dependency**: The surrogate model must sufficiently approximate the behavior of the remote LLM; if the task is highly complex or the LLM's capability far exceeds that of the surrogate, gradient guidance may become unreliable.
- **The PersonalPortrait dataset is synthesized by GPT-4**, and the distribution of private text in real-world scenarios may be considerably more complex.
- Multi-turn cross-turn privacy inference attacks in conversational settings are not considered.

## Related Work & Insights

- **vs. Presidio**: Presidio handles only predefined PII patterns (dates, place names, etc.) and cannot protect implicit privacy (e.g., occupation); its rule-based nature limits flexibility. PromptObfus combines TF-IDF with NER to cover both explicit and implicit privacy.
- **vs. SANTEXT/SANTEXT+**: Differential privacy-based word substitution provides strong privacy guarantees but introduces excessive noise, causing task accuracy to collapse (55–61%) and severely degrading text readability. PromptObfus avoids this through gradient-guided precise token selection.
- **vs. DP-Prompt**: Rewriting entire prompts with an LLM preserves task utility reasonably well but barely eliminates PII (PI attack success rate of 96.25%), rendering privacy protection largely ineffective. PromptObfus substantially outperforms DP-Prompt in privacy protection.

### Broader Implications

- The anti-adversarial paradigm generalizes to other security settings. For instance, in the image domain, can one apply transformations that are unrecognizable to humans yet processable by models, for privacy-preserving visual representations? This shares conceptual connections with privacy-preserving 3D scene representation.
- The surrogate model + gradient-guided selection framework can be transferred to prompt injection defense: replacing critical tokens in malicious prompts with benign alternatives while preserving syntactic structure, enabling robustness studies of LLMs.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The anti-adversarial concept is novel and intuitive, though the underlying techniques (MLM + gradient-guided selection) are relatively standard.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three tasks, six baselines, and extensive ablations; task diversity could be broader.
- **Writing Quality**: ⭐⭐⭐⭐ — Clear structure with well-articulated problem motivation.
- **Value**: ⭐⭐⭐⭐ — High practical value; the lightweight, plug-and-play design achieves an excellent privacy–utility balance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models](auvic_adversarial_unlearning_of_visual_concepts_for_multi-mo.md)
- [\[AAAI 2026\] Gender Bias in Emotion Recognition by Large Language Models](gender_bias_in_emotion_recognition_by_large_language_models.md)
- [\[AAAI 2026\] SproutBench: A Benchmark for Safe and Ethical Large Language Models for Youth](sproutbench_a_benchmark_for_safe_and_ethical_large_language_models_for_youth.md)
- [\[AAAI 2026\] Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting](learning_from_the_undesirable_robust_adaptation_of_language_models_without_forge.md)
- [\[CVPR 2026\] Multi-Paradigm Collaborative Adversarial Attack Against Multi-Modal Large Language Models](../../CVPR2026/llm_safety/multi-paradigm_collaborative_adversarial_attack_against_multi-modal_large_langua.md)

</div>

<!-- RELATED:END -->
