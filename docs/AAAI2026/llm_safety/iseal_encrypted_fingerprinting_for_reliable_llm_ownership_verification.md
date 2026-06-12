---
title: >-
  [Paper Note] iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification
description: >-
  [AAAI 2026][LLM Safety][LLM fingerprinting] This paper proposes iSeal — the first active fingerprinting method capable of reliably verifying LLM ownership in a black-box setting where the model thief has full control ove…
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "LLM fingerprinting"
  - "ownership verification"
  - "encrypted encoder"
  - "Reed-Solomon error correction"
  - "verification robustness"
date: 2026-05-08
content_hash: 37d2d155bcef58cb
---

# iSeal: Encrypted Fingerprinting for Reliable LLM Ownership Verification

**Conference**: AAAI 2026
**arXiv**: [2511.08905](https://arxiv.org/abs/2511.08905)  
**Code**: [kitaharasetusna/iSeal](https://github.com/kitaharasetusna/iSeal)  
**Area**: Robotics
**Keywords**: LLM fingerprinting, ownership verification, encrypted encoder, Reed-Solomon error correction, verification robustness

## TL;DR

This paper proposes iSeal — the first active fingerprinting method capable of reliably verifying LLM ownership in a black-box setting where the model thief has full control over the inference process. Through a triple mechanism of an external encrypted encoder, RSC error correction, and similarity-based matching, iSeal maintains a 100% Fingerprint Success Rate (FSR) across 12 LLMs and 10+ attack types, while existing methods drop to 0%.

## Background & Motivation

**High cost of LLM training**: Training large models from scratch requires enormous computational and financial investment, making the models themselves valuable intellectual property (IP) in need of reliable ownership verification.

**Real-world threat scenario**: Model thieves obtain model weights through internal leaks or security vulnerabilities and deploy them as public APIs for profit; in litigation and similar contexts, the thief has full control over the model inference pipeline.

**Passive fingerprinting lacks anti-forgery resistance**: Passive methods such as HuRef, REEF, ProFLingo, and TRAP do not modify the model itself, allowing anyone with API access to extract similar features and falsely claim ownership.

**Existing active fingerprinting lacks external keys**: Methods such as WLM and IF embed fingerprints solely within model weights, enabling adversaries with white-box access to reverse-engineer or remove them.

**Collusion-based unlearning attacks**: During legal disputes, verifiers must disclose at least one prompt-response pair; the thief can collude with a third party to perform targeted unlearning on that pair, invalidating subsequent verification.

**Response tampering attacks**: Thieves can manipulate outputs at inference time (word deletion, insertion, synonym substitution, paraphrasing, etc.); existing methods rely on exact matching and are easily circumvented.

## Method

### Overall Architecture

iSeal consists of three phases: **Model Registration → Fingerprint Injection → Ownership Verification**.

1. **Model Registration**: The model owner submits a request to a registration authority; the authority samples a hexadecimal key $K$ of length $k=32$ (key space $16^{32} \approx 10^{38}$), uses HMAC-SHA256 to generate per-layer seeds and initialize encoder $E$, and returns $E$ along with a selected plaintext set $D$ to the owner.
2. **Fingerprint Injection**: Encoder $E$ encrypts plaintext $x$ into ciphertext $y = E(x)$; the LLM serves as the decoder and is fine-tuned via an adapter (with the encoder frozen), with the training objective being conditional language learning: $\mathcal{M}^* = \arg\max_{\mathcal{M}} p_{\mathcal{M}}(E'(x) \mid \mathcal{M}(E(x)))$, where $E'(\cdot)$ denotes RSC encoding.
3. **Ownership Verification**: A judge uses the encoder to encrypt plaintext queries to the suspect API; the output undergoes RSC decoding $D'$ and is then compared against the plaintext via BLEU similarity. If the score exceeds threshold $\alpha$, the model is judged to be stolen.

### Key Designs

- **External encrypted encoder**: A key-driven two-layer linear network serves as the encoder, decoupled from the model — even if the thief possesses all model weights, the fingerprint cannot be reconstructed. The encoder is frozen during training to prevent it from learning optimal representations that could independently reconstruct the plaintext.
- **Diffusion and Confusion**: The encoder is proven to satisfy cryptographic diffusion and confusion properties — flipping any single plaintext bit changes approximately half of the ciphertext bits (Theorem 1), and flipping any single key bit changes more than half of the ciphertext bits (Theorem 2), ensuring that limited observations are insufficient for reverse engineering.
- **Reed-Solomon Error Correction (RSC)**: The training objective applies RSC encoding to the plaintext; during verification, RSC decoding is applied to the LLM output before matching, providing provable robustness against response tampering.
- **Similarity matching over exact matching**: BLEU score rather than exact string matching is used for verification, naturally tolerating minor edits and deletions.
- **Anti-collusion mechanism**: Each verification round uses a different plaintext $z \neq x$; unlearning a single query-response pair cannot erase the entire fingerprint mapping.
- **Rationale for not using AES**: AES's high nonlinearity and discontinuous operations cause gradient vanishing, semantic information destruction, slow convergence, and poor reconstruction quality.

## Experiments

### Experimental Setup

- **Models**: 12 LLMs, including OPT-125M, LLaMA2-7B/13B, LLaMA3-7B, Mistral-7B, Amber-7B, Vicuna-v1.5-7B, RedPajama, Pythia-6.9B, GPT-J-6B, and mT5-11B.
- **Datasets**: AG's News (main experiments), DailyDialog, arXiv Abstracts; Alpaca-52K for persistence evaluation.
- **Baselines**: WLM, IF (two representative active fingerprinting methods).
- **Metrics**: BLEU score, Fingerprint Success Rate (FSR), SuperGLUE zero-shot performance (harmlessness).

### Table 1: Harmlessness Evaluation (0-shot SuperGLUE Accuracy)

| Method | LLaMA2-7B | LLaMA2-13B | Mistral-7B | Amber-7B |
|--------|-----------|------------|------------|----------|
| Vanilla | 59% | 60% | 64% | 54% |
| WLM | 49% | 49% | 50% | 48% |
| IF | 50% | 49% | 49% | 50% |
| **iSeal** | **56%** | **59%** | **55%** | **53%** |

iSeal has the smallest impact on model performance (1–9 percentage point drop), significantly outperforming WLM and IF (10–14 percentage point drop), as iSeal uses non-natural-language inputs that interfere less with normal task performance.

### Table 2: Persistence Evaluation (FSR after Alpaca Fine-tuning)

| Method | LLaMA2-7B | LLaMA2-13B | Mistral-7B | Amber-7B |
|--------|-----------|------------|------------|----------|
| WLM | 74.7% | 76% | 73.4% | 75% |
| IF | 100% | 100% | 100% | 100% |
| **iSeal** | **100%** | **100%** | **100%** | **100%** |

iSeal maintains 100% FSR after fine-tuning across all models, matching IF and far exceeding WLM.

### Table 3: Ablation Study (FSR)

| Variant | LLaMA2-7B | LLaMA2-13B | Mistral-7B | Amber-7B |
|---------|-----------|------------|------------|----------|
| iSeal (full) | 100% | 100% | 100% | 100% |
| w/o frozen encoder | 0% | 0% | 0% | 0% |
| w/o encoder (AES substitute) | 0% | 0% | 2% | 1% |

Removing encoder freezing or replacing the encoder with AES drops FSR to 0%, validating the necessity of both key design choices.

### Robustness Evaluation

- **Fingerprint guessing attacks**: Three guessing strategies (random hexadecimal F1, random-key encoder F2, single-logit difference key F3) achieve 0% FSR across 11 models.
- **Unlearning attacks**: iSeal maintains 100% FSR under three state-of-the-art unlearning methods, while WLM/IF degrade significantly within the first few rounds.
- **Response tampering attacks**: iSeal maintains high FSR under 6 attack types — word deletion, word insertion, synonym substitution, paraphrasing, copy-paste, and homoglyph attacks — while baseline methods degrade substantially. The RSC module further enhances tampering robustness.

### Efficiency

On LLaMA2-13B (A100 GPU): WLM requires 233.4 minutes to converge, IF requires 5 minutes, and iSeal likewise requires only 5 minutes. Encoder initialization takes only 1 millisecond (i7-9700K CPU).

## Key Findings

1. iSeal is the first fingerprinting method to achieve reliable verification in an end-to-end black-box setting where the thief has full control over the inference pipeline.
2. Cryptographic guarantees from the diffusion and confusion properties ensure that fingerprints cannot be reverse-engineered from limited observations — unlearning a single query-response pair does not affect the overall fingerprint.
3. The key space of order $10^{38}$, combined with verification failure upon a single-logit key change, fundamentally prevents overclaiming of ownership.
4. The non-natural-language input design causes far less interference with the model's normal capabilities compared to existing methods.
5. Both the threshold $\alpha$ and the encoder layer count $N$ have wide effective ranges; in practice, the optimal threshold can be selected automatically via Bayesian decision theory.

## Highlights & Insights

- **Dual theoretical and empirical guarantees**: Diffusion/confusion theorems provide cryptographic security proofs; 100% FSR is empirically validated across 12 models and 10+ attack types.
- **Three-layer defense architecture**: External encoder (anti-reverse-engineering) + RSC error correction (anti-tampering) + similarity matching (fault tolerance), with each layer complementing the others.
- **Strong practicality**: Training efficiency is on par with IF (5 minutes), with no additional inference overhead; encoder initialization takes only 1 ms; code is open-sourced.
- **Realistic threat model**: The four-party setup (owner, thief, judge, registration authority) closely reflects real-world IP litigation scenarios.

## Limitations & Future Work

1. **Base models only**: Experiments focus on pre-trained base models rather than instruction-tuned variants; effectiveness on models after RLHF/DPO training has not been fully validated.
2. **Limited model scale**: The largest tested model is 13B; applicability to models with 70B+ parameters or closed-source large-scale models is unknown.
3. **Simple encoder architecture**: The default two-layer linear network is straightforward; stronger nonlinear encoders could enhance security but converge more slowly; the trade-off between security and efficiency warrants further investigation.
4. **Limitations of BLEU**: Similarity-based verification relies on BLEU scores, which may be insufficient against more sophisticated semantic-level tampering (e.g., full paraphrasing that preserves meaning while altering surface form), potentially requiring stronger semantic matching metrics.
5. **Key management assumptions**: The approach relies on a trusted registration authority and judge; scenarios involving key leakage or institutional misconduct are not discussed.

## Related Work & Insights

- **Passive fingerprinting**: HuRef (parameters → human-readable images, requires white-box access), REEF/EasyDetector (white-box features), TRAP/ProFLingo/RAP-SM (optimized suffix/prefix triggers eliciting specific outputs, lacking anti-forgery resistance).
- **Active fingerprinting**: WLM (trigger words + predefined answer fine-tuning), IF (instruction-style prompts + adapter injection), PLMark (contrastive learning on [CLS], ineffective for LLMs), UTF (simplified IF), MYL (multiple queries + statistical testing, susceptible to reverse engineering), FP-VEC (direct weight vector addition), EditMark (math problem accuracy sequences, broken by single-question unlearning), PlugAE (optimized trigger token embeddings, easily detectable).
- **Error-correcting codes**: Reed-Solomon codes for response tampering resistance; the Singleton bound proves their optimality.
- **Cryptographic foundations**: Shannon's diffusion and confusion principles, HMAC-SHA256 key derivation.

## Rating

- Novelty: ⭐⭐⭐⭐ First to introduce an encrypted encoder + RSC error correction into LLM fingerprinting, addressing verification-time attacks that were previously overlooked
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 12 models, 10+ attack types, multiple datasets, complete ablation and sensitivity analysis
- Writing Quality: ⭐⭐⭐⭐ Threat model and security proofs are clearly articulated; tables and figures are well organized
- Value: ⭐⭐⭐⭐ Strong practical value for LLM IP litigation scenarios; open-sourced code facilitates reproducibility

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Antidistillation Fingerprinting](../../ICML2026/llm_safety/antidistillation_fingerprinting.md)
- [\[ACL 2026\] XMark: Reliable Multi-Bit Watermarking for LLM-Generated Texts](../../ACL2026/llm_safety/xmark_reliable_multi-bit_watermarking_for_llm-generated_texts.md)
- [\[ICML 2026\] Internalizing Safety Understanding in Large Reasoning Models via Verification](../../ICML2026/llm_safety/internalizing_safety_understanding_in_large_reasoning_models_via_verification.md)
- [\[CVPR 2026\] Demographic Fairness in Multimodal LLMs: A Benchmark of Gender and Ethnicity Bias in Face Verification](../../CVPR2026/llm_safety/demographic_fairness_in_multimodal_llms_a_benchmark_of_gender_and_ethnicity_bias.md)
- [\[AAAI 2026\] Democratizing LLM Efficiency: From Hyperscale Optimizations to Universal Deployability](democratizing_llm_efficiency_from_hyperscale_optimizations_to_universal_deployab.md)

</div>

<!-- RELATED:END -->
