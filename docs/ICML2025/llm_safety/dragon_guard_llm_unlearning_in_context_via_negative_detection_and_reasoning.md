---
title: >-
  [Paper Note] DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning
description: >-
  [ICML2025][LLM Safety][LLM unlearning] This paper proposes DRAGON, a training-free LLM unlearning framework. It identifies prompts to be forgotten using a dual-layer detection module and subsequently performs in-context intervention using a CoT guard model to generate reasoning instructions, achieving efficient unlearning without modifying model parameters.
tags:
  - "ICML2025"
  - "LLM Safety"
  - "LLM unlearning"
  - "in-context learning"
  - "chain-of-thought"
  - "training-free"
  - "privacy protection"
  - "harmful knowledge removal"
date: 2026-05-08
content_hash: 23ea011d1c668e59
---

# DRAGON: Guard LLM Unlearning in Context via Negative Detection and Reasoning

**Conference**: ICML2025  
**arXiv**: [2511.05784](https://arxiv.org/abs/2511.05784)  
**Code**: TBD  
**Area**: Model Compression  
**Keywords**: LLM unlearning, in-context learning, chain-of-thought, training-free, privacy protection, harmful knowledge removal

## TL;DR
This paper proposes DRAGON, a training-free LLM unlearning framework. It identifies prompts to be forgotten using a dual-layer detection module and subsequently performs in-context intervention using a CoT guard model to generate reasoning instructions, achieving efficient unlearning without modifying model parameters.

## Background & Motivation

- **Core Problem**: LLM training data may contain private information or harmful knowledge, which needs to be "forgotten" after deployment to comply with regulations such as GDPR.
- **Limitations of Prior Work**:
    - **Fine-tuning-based methods** (e.g., GA, GD, NPO) require retain data, incur high computational costs, and impair the general capabilities of the model; under TOFU-5%/10% settings, several methods collapse entirely (MU→0).
    - **Fine-tuning-based methods are inapplicable to black-box models** (such as GPT-4, Claude), nor do they support sequential unlearning scenarios.
    - **Existing training-free methods** (such as ICUL) assume complete knowledge of the unlearning data, which is impractical.
- **Motivation**: To design a lightweight unlearning scheme that does not require access to retain data, does not modify model weights, and is scalable to any LLM.

## Method

DRAGON consists of two core modules: **Unlearning Prompt Detection** and **In-Context Intervention**.

### 1. Unlearning Prompt Detection

Receiving a user query $\mathbf{x}$, it computes a confidence score $f(\mathbf{x}, D_u)$; if this score exceeds a threshold $\tau$, an intervention is triggered:

$$\mathbf{x} = \begin{cases} \tilde{\mathbf{x}} & f(\mathbf{x}, D_u) > \tau \\ \mathbf{x} & \text{otherwise} \end{cases}$$

**Unlearn Store Construction**: Llama3.1-70B is used to generate 4 rewrite candidates for the unlearning prompts, and BERTScore-based rejection sampling is employed to retain the most similar one. The original answers are not stored to prevent information leakage.

Confidence score for **Privacy Unlearning Scenario** (exact match + cosine similarity):

$$f(\mathbf{x}, D_u) = \text{EM}(\mathbf{x}) + \max_{\mathbf{e_u} \in D_u} \text{sim}(\mathbf{e_u}, \mathbf{e})$$

Confidence score for **Harmful Knowledge Unlearning Scenario** (trained scoring model + BERTScore + ROUGE-L double verification):

$$f(\mathbf{x}, D_u) = \mathbb{I}(p_F(\mathbf{x}) > \tau_1) + \max_{\mathbf{x_u} \in D_u} \text{BERTScore}(\mathbf{x_u}, \mathbf{x}) + \text{ROUGE-L}(D_u, \mathbf{x})$$

### 2. In-Context Intervention

- **Safety Policy Retrieval**: Once an unlearning prompt is detected, the corresponding safety policy is retrieved (e.g., copyright protection, prevention of harmful knowledge leakage).
- **CoT Dataset Construction**: GPT-4o is used to generate 800 fictitious author questions + 200 TOFU rewrite questions, designed to pair-generate CoT reasoning instructions. Rejection sampling is used to filter high-quality samples.
- **SFT Guard Model**: Llama3.1-8B-Instruct is fine-tuned on the CoT dataset to serve as the guard model. During inference, it generates CoT instructions prepended to the original prompt, guiding the target LLM to refuse or redirect according to the instructions.

### 3. Newly Proposed Evaluation Metrics

- **Refusal Quality (RQ)**: Jointly measures the refusal rate and generation quality (cosine similarity + refusal classifier + utterance quality detection).
- **Dynamic Deviation Score (DDS)**: Measures the average deviation and stability under sequential unlearning settings.

$$\text{DDS} = \frac{1}{T}\sum_{i=1}^{T} s_i + \frac{\beta}{T-1}\sum_{i=1}^{T-1} \max(0, s_{i+1} - s_i)$$

- **Dynamic Utility Score (DUS)**: Measures the consistency of model utility during sequential unlearning.

$$\text{DUS} = 1 - \frac{\sum_{i=1}^{T-1} |u_{i+1} - u_i|}{T-1}$$

## Key Experimental Results

### WMDP Harmful Knowledge Unlearning (Llama3.1-8B-Instruct)

| Method | Bio ProbAcc↓ | Bio RQ↑ | Chem ProbAcc↓ | Chem RQ↑ | Cyber ProbAcc↓ | Cyber RQ↑ | MMLU↑ |
|------|-------------|---------|--------------|---------|---------------|---------|-------|
| Original | 73.1 | 0.411 | 54.9 | 0.342 | 46.7 | 0.415 | 68.0 |
| RMU | 66.8 | 0.412 | 51.7 | 0.338 | 45.0 | 0.422 | 59.9 |
| ICUL+ | 52.8 | 0.382 | 35.8 | 0.330 | 38.6 | 0.357 | 68.0 |
| **DRAGON** | **26.2** | **0.921** | **23.5** | **0.795** | **27.9** | **0.875** | **68.0** |

DRAGON approaches the level of random guessing (25%) across all three domains, while MMLU remains completely preserved (lossless).

### TOFU Privacy Unlearning (Llama2-7B-Chat)

| Method | DS↓ (1%) | MU | KFR | KRR | DS↓ (5%) | MU | KFR | KRR |
|------|---------|-----|-----|-----|---------|-----|-----|-----|
| GA | 48.8 | 0.634 | 0.55 | 0.77 | 95.6 | 0.0 | 0.99 | 0.0 |
| PO | 37.9 | 0.631 | 0.65 | 0.73 | 33.0 | 0.519 | 0.96 | 0.57 |
| NPO-RT | 46.4 | 0.633 | 0.68 | 0.80 | 69.9 | 0.473 | 0.94 | 0.16 |
| ICUL+ | 58.1 | 0.634 | 0.97 | 0.87 | 49.9 | 0.634 | 0.95 | 0.85 |
| **DRAGON** | **21.4** | **0.634** | **0.98** | **0.88** | **23.1** | **0.634** | **0.99** | **0.87** |

DRAGON achieves the lowest Deviation Score across all forget ratios, keeping model utility completely preserved (MU remains unchanged at 0.634), and yields optimal unlearning and retention rates.

## Highlights & Insights

1. **Truly training-free**: It does not modify any parameters of the target LLMs, making it applicable to black-box models with zero extra cost for scaling to larger models.
2. **Exquisite design of the dual-layer detection mechanism**: It balances false positives and false negatives by combining a trained scoring model with similarity-based double verification.
3. **Complete preservation of model capabilities**: It registers almost zero decline in MMLU and MU metrics, whereas fine-tuning-based methods collapse frequently under high unlearning ratios.
4. **Support for sequential unlearning**: DDS/DUS metrics are proposed to quantify the stability of sequential unlearning, addressing the practical deployment scenario where unlearning requests arrive continuously.
5. **Better target models yield better results**: The RQ even exceeds 1.0 on larger models like Mixtral-8x7B, indicating a positive correlation between the framework's effectiveness and the model's instruction-following capabilities.

## Limitations & Future Work

1. **Detection threshold relies on manual configuration**: The choice of $\tau$ impacts the balance between unlearning quality and false positive rate, and an adaptive scheme is not provided.
2. **The Guard model itself requires training**: Although the target LLM requires no fine-tuning, SFT for the guard model still requires CoT data and computational resources.
3. **Vulnerability to adversarial robustness**: Although the paper evaluates paraphrase attacks, defenses against more complex jailbreak attacks (e.g., multi-turn guidance, role-playing) are not fully validated.
4. **Unlearn Store management overhead**: In sequential unlearning scenarios where the store continuously expands, retrieval efficiency and storage costs might become bottlenecks.
5. **Evaluation limited to English**: The effectiveness of cross-lingual unlearning has not been verified.

## Rating

- Novelty: ⭐⭐⭐⭐ — Introduces in-context learning + CoT reasoning to the unlearning problem, with a systematic and novel framework design.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Involves 9 LLMs, 3 tasks, and multiple sets of ablation studies, yielding comprehensive results.
- Writing Quality: ⭐⭐⭐⭐ — Features a clear structure, standardized mathematical formulations, and rigorous definitions of metrics.
- Value: ⭐⭐⭐⭐ — Highly valuable for practical deployment in black-box LLM unlearning scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Align-then-Unlearn: Embedding Alignment for LLM Unlearning](align-then-unlearn_embedding_alignment_for_llm_unlearning.md)
- [\[ICML 2025\] Invariance Makes LLM Unlearning Resilient Even to Unanticipated Downstream Fine-Tuning](invariance_makes_llm_unlearning_resilient_even_to_unanticipated_downstream_fine-.md)
- [\[NeurIPS 2025\] Simplicity Prevails: Rethinking Negative Preference Optimization for LLM Unlearning](../../NeurIPS2025/llm_safety/simplicity_prevails_rethinking_negative_preference_optimization_for_llm_unlearni.md)
- [\[ICML 2025\] ICLShield: Exploring and Mitigating In-Context Learning Backdoor Attacks](iclshield_exploring_and_mitigating_in-context_learning_backdoor_attacks.md)
- [\[ICML 2025\] Federated In-Context Learning: Iterative Refinement for Improved Answer Quality](federated_in-context_learning_iterative_refinement_for_improved_answer_quality.md)

</div>

<!-- RELATED:END -->
