---
title: >-
  [Paper Note] CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment
description: >-
  [NeurIPS 2025][LLM Safety][model protection] This paper proposes CoreGuard, which locks Transformer linear layer weights via row permutation and reduces TEE authorization to a single invocation through a column-permutation propagation protocol, protecting foundational capabilities of edge-deployed LLMs against model stealing attacks with negligible computational and communication overhead.
tags:
  - NeurIPS 2025
  - LLM Safety
  - model protection
  - edge deployment
  - TEE
  - permutation
  - model stealing
date: 2026-05-08
content_hash: 3692267d2c78d0f8
---

# CoreGuard: Safeguarding Foundational Capabilities of LLMs Against Model Stealing in Edge Deployment

**Conference**: NeurIPS 2025
**arXiv**: [2410.13903](https://arxiv.org/abs/2410.13903)
**Code**: Not released
**Area**: AI Security
**Keywords**: model protection, edge deployment, TEE, permutation, LLM safety, model stealing

## TL;DR

This paper proposes CoreGuard, which locks Transformer linear layer weights via row permutation and reduces TEE authorization to a single invocation through a column-permutation propagation protocol, protecting foundational capabilities of edge-deployed LLMs against model stealing attacks with negligible computational and communication overhead.

## Background & Motivation

**State of the Field**: Proprietary LLMs (e.g., ChatGPT, Claude) are increasingly deployed on edge devices for low-latency and privacy-preserving inference (e.g., Apple Intelligence deploys a 3B-parameter LLM on iOS).

**Limitations of Prior Work**:
   - **Watermarking** (passive defense): Only proves ownership; attackers can still abuse the model in unsupervised environments.
   - **Model encryption**: Models remain vulnerable to reverse-engineering attacks after runtime decryption.
   - **Full-model TEE protection** (TPTE): CPU enclave execution incurs ~50× slowdown, making it infeasible.
   - **Partial TEE execution** (PPTE, e.g., DarkneTZ): Protects too few parameters; attackers can reconstruct the model with as little as 1% of training data.
   - **Parameter shuffling** (PSP, e.g., ShadowNet): Requires TEE↔GPU transfer per linear layer; generating a single token on LLaMA3-8B incurs ~1.3 GB of transfer and ~1.3 seconds of latency.

**Root Cause**: Existing approaches face a fundamental tension between security and efficiency—sufficiently strong protection incurs unacceptable computational/communication overhead, while efficient solutions provide inadequate security. The threat of *foundational capability stealing* is particularly acute: attackers can fine-tune a locked model to exploit its generalization ability for new tasks.

**Paper Goals**: Design a plug-and-play protection scheme that allows a locked model to function correctly only under TEE authorization, with negligible computational and communication overhead.

**Starting Point**: Exploit the mathematical properties of permutation matrices ($\pi\pi^T = I$) to realize a "lock-and-key" mechanism, and propagate the authorization signal automatically through all subsequent layers via column-permutation propagation, eliminating repeated TEE invocations.

**Core Idea**: Row permutation locks the weights; column-permutation propagation carries the authorization signal, compressing TEE interactions from two per layer to only five rounds for the entire model.

## Method

### Overall Architecture

CoreGuard operates in two phases:

**Model Locking (pre-deployment)**: The linear layer weight matrices of a trained model are row-permuted via a *protection protocol*, rendering unauthorized inputs unable to produce correct outputs. Output processing layers are simultaneously column-permuted via a *propagation protocol* so that the authorization signal propagates automatically.

**Inference Authorization (post-deployment)**: The TEE performs a one-time column permutation on the input to the first locked layer; all subsequent layers then automatically receive correctly permuted inputs. A one-time pad (OTP) is used during authorization to conceal the permutation operation.

### Key Designs

1. **Protection Protocol**:

    - *Function*: Locks input processing layers (QKV projections, FFN input layer) via row permutation.
    - *Mechanism*: Given permutation matrix $\pi \in \{0,1\}^{d \times d}$, the weight is row-permuted as $W'_q = \pi^T W_q$. Only a column-permuted input $x\pi$ yields correct output: $x\pi \cdot \pi^T W_q = xW_q = Q$; any other input produces entirely incorrect results.
    - *Design Motivation*: Permutation does not alter parameter functionality but maps parameters to a new domain. The probability of guessing the correct $\pi$ is $1/(d!)$, which is computationally infeasible for $d=4096$.

2. **Propagation Protocol**:

    - *Function*: Automatically propagates the authorization signal from one layer to the next, eliminating per-layer TEE authorization.
    - *Mechanism*: Output processing layers ($W_o$, $W_n$, Add-Norm parameters) are column-permuted: $W'_n = W_n\pi$, $b'_n = b_n\pi$, $\gamma'_2 = \gamma_2\pi$, $\beta'_2 = \beta_2\pi$. This causes each layer's output to automatically take the column-permuted form $n' = mW_n\pi + b_n\pi = n\pi$. The overall Transformer layer satisfies $f_{w'}(x\pi) = f_w(x)\pi$, and the output $z\pi$ serves directly as the authorized input for the next layer.
    - *Design Motivation*: ShadowNet requires 2 TEE transfers per linear layer (448 total for LLaMA3-8B); the propagation protocol reduces this to a single initial authorization for the entire model.

3. **Inference Authorization and OTP Encryption**:

    - *Function*: Securely generates the column-permuted features for the initial authorization, preventing attackers from inferring $\pi$ via input–output comparison.
    - *Mechanism*: Four steps — (a) the FFN input layer computes $m$ normally; (b) the TEE encrypts and permutes $m' = (m + p)\pi$ using OTP; (c) the GPU processes the encrypted feature with pre-aligned weights $W'_n = \pi^T W_n$, yielding $n' = n + pW_n$; (d) the TEE decrypts $n'' = n' - pW_n = n$, applies Add-Norm, and outputs the column-permuted result $z\pi$.
    - *Design Motivation*: OTP ensures that each encryption produces a different result even for identical inputs. Permutation and encryption mutually obscure each other. The TEE executes only lightweight linear operations, with five TEE–GPU transfers in total.

4. **Authorization Position Selection**:

    - *Function*: Determines at which layer locking begins.
    - *Mechanism*: Locking is applied at the middle layers of the model, requiring an attacker to recover at least half the parameters to obtain a complete model.
    - *Design Motivation*: Locking only the first or last layer allows an attacker to retrain a single layer (analogous to prompt tuning or training a classification head); a middle position maximizes attack difficulty.

### Loss & Training

CoreGuard is a plug-and-play scheme that requires no additional training. The locking operation is a deterministic mathematical transformation (permutation matrix operations), guaranteeing that the authorized model is functionally identical to the original with zero accuracy loss.

## Key Experimental Results

### Main Results — Security Comparison (Model Stealing Attack Accuracy ↓)

| Model | Task | No-shield | DarkneTZ | ShadowNet | DTE | CoreGuard | Black-box |
|-------|------|:---------:|:--------:|:---------:|:---:|:---------:|:---------:|
| Qwen2-0.5B | GSM8k | 21.53 | 16.81 | 1.34 | 2.36 | **2.41** | 1.29 |
| Gemma2-2B | GSM8k | 40.94 | 37.07 | 10.81 | 4.56 | **3.91** | 1.74 |
| ChatGLM3-6B | GSM8k | 55.95 | 54.91 | 0.43 | 0.93 | **1.04** | 0.23 |
| LLaMA3-8B | GSM8k | 53.07 | 51.31 | 4.15 | 6.09 | **6.22** | 4.05 |
| Relative avg. | All | 9.58× | 8.43× | 1.09× | 1.18× | **1.17×** | 1.00× |

CoreGuard's security is nearly on par with the black-box upper bound (1.17× vs. 1.00×), far surpassing PPTE-based approaches (DarkneTZ: 8.43×). The unauthorized direct inference accuracy across all models is **0.00%**.

### Ablation Study — Efficiency Comparison

| Scheme | TEE Transfers/token | Extra FLOPs | Practicality |
|--------|:-------------------:|:-----------:|:------------:|
| ShadowNet | 448 (LLaMA3-8B) | High | ~1.3s per token |
| DTE (half-model TEE) | 0 | Extreme (~50×) | Infeasible |
| CoreGuard | **5** | **Negligible** | ✓ |

### Key Findings
- **Security equivalent to full TEE protection**: CoreGuard's 1.17× relative accuracy is nearly identical to DTE's 1.18× (which uses TEE to directly protect the same portion of parameters), demonstrating that permutation-based protection is equivalent in efficacy to direct TEE protection.
- **Zero accuracy loss**: The authorized model achieves accuracy identical to the original across four tasks (GSM8k, Spider, PubMedQA, SQuAD).
- **Attackers face an NP-hard problem**: Even with auxiliary information, recovering $\pi$ reduces to the Learning With Errors (LWE) problem.
- **PPTE schemes are broadly insecure**: DarkneTZ (8.43×), SOTER, and similar approaches that protect only a small subset of parameters provide near-zero defense under full-dataset attacks.
- **TPTE is completely ineffective**: NPLO (9.59×) is nearly equivalent to no protection (9.58×); protecting only task-specific adapters offers no meaningful defense for LLMs.

## Highlights & Insights
- **Mathematical elegance of the propagation protocol**: The orthogonality of permutation matrices ($\pi\pi^T = I$) enables the authorization signal to self-propagate across all Transformer layers. The lock (row permutation) and key (column permutation) cancel each other out exactly, with zero error.
- **Nested security of OTP and permutation**: OTP conceals the permutation and the permutation conceals the OTP; observing either operation in isolation is insufficient to infer the other, as security derives from their mutual concealment.
- **Plug-and-play design**: No model retraining or architectural modification is required; the scheme applies pure mathematical transformations to the weights and is directly applicable to any Transformer model.

## Limitations & Future Work
- Only fine-tuning attacks are considered; other threat vectors such as distillation attacks and side-channel attacks are not evaluated.
- The scheme assumes a fully secure TEE, whereas practical TEE implementations (e.g., Intel SGX) have been compromised multiple times.
- The permutation granularity is at the channel level ($d$ dimensions); whether finer-grained protection (e.g., sub-channel level) could further improve security remains unexplored.
- The offline precomputation of OTP noise $pW_n$ introduces storage and update costs that are not thoroughly discussed.
- Experiments are limited to models up to 8B parameters; applicability to very large models (>70B) is unverified.
- Adaptation to non-standard Transformer architectures (e.g., MoE, SSM) is not discussed.

## Related Work & Insights
- **vs. ShadowNet**: Both use permutation for protection, but ShadowNet requires TEE authorization per layer (448 times); CoreGuard requires only one authorization plus propagation, reducing communication overhead by two orders of magnitude.
- **vs. DarkneTZ/SOTER (PPTE)**: These methods place final or partial layers in the TEE, providing insufficient security (8.43×); CoreGuard protects the same number of parameters but uses permutation instead of TEE execution.
- **vs. model encryption**: Encryption protects static parameters but cannot prevent runtime reverse-engineering; CoreGuard's permutation renders runtime parameters unusable without authorization.
- CoreGuard has direct practical value for secure edge AI deployment, and the propagation protocol concept is generalizable to other secure computation scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐ Reducing TEE interactions to a single invocation via the propagation protocol is an elegant engineering contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation across 4 models × 4 tasks × 8 baselines, covering both security and efficiency.
- Writing Quality: ⭐⭐⭐⭐ Threat model is clearly defined; security analysis is rigorous.
- Value: ⭐⭐⭐⭐ Edge LLM protection addresses a genuine industrial need, and the plug-and-play scheme is deployable in practice.

<!-- RELATED:START -->

## Related Papers

- [\[NeurIPS 2025\] Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties](trans-env_a_framework_for_evaluating_the_linguistic_robustness_of_llms_against_e.md)
- [\[NeurIPS 2025\] Bits Leaked per Query: Information-Theoretic Bounds on Adversarial Attacks Against LLMs](bits_leaked_per_query_information-theoretic_bounds_on_adversarial_attacks_agains.md)
- [\[NeurIPS 2025\] MaskSQL: Safeguarding Privacy for LLM-Based Text-to-SQL via Abstraction](masksql_safeguarding_privacy_for_llm-based_text-to-sql_via_abstraction.md)
- [\[NeurIPS 2025\] Self-Refining Language Model Anonymizers via Adversarial Distillation](self-refining_language_model_anonymizers_via_adversarial_distillation.md)
- [\[NeurIPS 2025\] Enhancing Sample Selection Against Label Noise by Cutting Mislabeled Easy Examples](enhancing_sample_selection_against_label_noise_by_cutting_mislabeled_easy_exampl.md)

<!-- RELATED:END -->
