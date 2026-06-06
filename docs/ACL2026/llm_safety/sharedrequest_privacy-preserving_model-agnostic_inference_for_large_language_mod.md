---
title: >-
  [Paper Note] SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models
description: >-
  [ACL 2026][LLM Safety][Privacy-Preserving Inference] SharedRequest is proposed as a model-agnostic privacy-preserving LLM inference framework. By elevating privacy protection from the individual prompt level to the batch…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Privacy-Preserving Inference"
  - "Model-Agnostic"
  - "Batch-level Obfuscation"
  - "Differential Privacy"
date: 2026-05-08
content_hash: 540b97ad059c5fde
---

# SharedRequest: Privacy-Preserving Model-Agnostic Inference for Large Language Models

**Conference**: ACL 2026  
**arXiv**: [2606.05004](https://arxiv.org/abs/2606.05004)  
**Code**: [GitHub](https://github.com/NusIoraPrivacy/SharedRequest)  
**Area**: llm_safety  
**Keywords**: Privacy-Preserving Inference, Model-Agnostic, Batch-level Obfuscation, Differential Privacy, LLM Safety

## TL;DR
SharedRequest is proposed as a model-agnostic privacy-preserving LLM inference framework. By elevating privacy protection from the individual prompt level to the batch level—mixing real and noise prompts and sharing inference overhead for semantically equivalent requests—it achieves a >20% utility gain and up to 5.6× reduction in query costs.

## Background & Motivation
**Background**: Public LLMs (ChatGPT/Claude/Gemini) are deployed in the cloud, where user prompts often contain sensitive information. Existing privacy-preserving methods face a privacy-utility-efficiency trilemma.

**Limitations of Prior Work**: (1) SMPC methods (Iron/BOLT/NEXUS) incur massive computational and communication overhead, making them unsuitable for large-scale deployment; (2) Local Differential Privacy (LDP) methods (RanText/CusText/DP-Prompt) introduce severe semantic damage through prompt-wise perturbation, leading to significant utility degradation; (3) Existing model-agnostic methods perturb each query independently, causing large semantic distortion.

**Key Challenge**: The fundamental contradiction between privacy protection and utility maintenance—stronger perturbation yields better privacy but worse utility. Existing methods limit privacy protection to a single prompt granularity, failing to leverage batch-level statistical properties to amortize costs.

**Goal**: To design a privacy-preserving inference framework that requires no modifications to LLM architectures or access to model parameters, while maintaining high utility, providing strong privacy guarantees, and reducing query costs.

**Key Insight**: Two key observations: (1) Commercial LLMs handle large-scale batch queries (ChatGPT handles >11,500 per second), allowing costs to be shared across users; (2) Sensitive information in prompts is often sparse (e.g., only the word "cybersecurity" is sensitive), and not all tokens require protection.

**Core Idea**: Batch-level privacy protection—grouping semantically equivalent requests to share inference overhead + mixing real and noise prompts to obfuscate sensitive attributes + using a three-party cryptographic protocol to ensure secure communication.

## Method

### Overall Architecture
SharedRequest involves three parties: the user (holding queries with sensitive attributes), the noise sampler (clustering requests by semantic equivalence and injecting noise prompts), and the service provider (receiving a shuffled set of mixed prompts and generating responses). The user encrypts sensitive attributes → the noise sampler performs clustering + samples noise combinations → after shuffling and mixing, the prompts are sent to the server → responses are securely returned via a masking scheme.

### Key Designs
1. **Batch-level Privacy-Preserving Paradigm**:

    - **Function**: Elevates privacy protection from the single prompt level to the batch level, utilizing the user population to amortize noise query costs.
    - **Mechanism**: Segregates user prompts into generic instructions $T_q$ and private attributes $A_q$. It groups semantically equivalent generic instructions for sharing and samples noise attribute substitutes for each group to generate noise prompts. These are mixed and shuffled with real prompts before being sent to the server. The server perceives an anonymous set of mixed real and fake prompts.
    - **Design Motivation**: Existing methods perturb prompts independently, which is costly and causes high semantic loss; batch-level sharing amortizes the extra cost of noise queries across a large number of users.

2. **Lightweight Three-party Cryptographic Protocol**:

    - **Function**: Ensures user anonymity and attribute privacy while allowing secure multi-party communication.
    - **Mechanism**: Forward transmission—users encrypt private attributes with the server's public key $pk_s$ (the noise sampler cannot see the plaintext); Backward transmission—users send a random seed $s$, the server generates a mask $e = PRG(s)$ via PRG to obfuscate the response $r_s = r + e$, and users recover $r = r_s - e$ locally (the noise sampler cannot see the response content).
    - **Design Motivation**: It is necessary to hide sensitive data from the noise sampler and user identity from the server simultaneously. The double-layer encryption and masking scheme achieve privacy protection in both directions.

3. **Noise Query Sampling and Composition Filtering**:

    - **Function**: Efficiently generates high-quality noise prompts to ensure indistinguishability from the server's perspective.
    - **Mechanism**: Users independently specify candidate substitutes $\{\mathcal{A}_1', ..., \mathcal{A}_{|A(q)|}'\}$ for each attribute. The noise sampler randomly samples candidate combinations → evaluates the "authenticity" of the combination using a pre-trained discriminator → retains only qualified combinations $\mathcal{A}^n$ whose scores exceed a threshold $\delta$. Coverage guarantee: it is required to sample $m \geq (\log(1-p) - \log(\mu k))/\log(1-1/k)$ combinations.
    - **Design Motivation**: The combination space for multi-attribute prompts grows exponentially ($k^\mu$), making direct enumeration infeasible. Discriminator filtering ensures that noise prompts are indistinguishable from real prompts to the server.

### Loss & Training
- No LLM training required: The framework is entirely model-agnostic.
- Privacy Formalization: $(A_n, \epsilon)$-indistinguishability, a user-defined variant of Differential Privacy.
- Theoretical Guarantee: Theorem 4 proves that the protocol satisfies $(A_n, \epsilon)$-indistinguishability.

## Key Experimental Results

### Main Results (Utility Comparison, 3 Datasets × 3 GPT Models)

| Setting | MMLU-Biz (F1) | Medical-QA (Rating) | Legal-QA (Rating) |
|------|-------------|-----------------|---------------|
| GPT-4o Non-private | 0.899 | 8.81 | 8.81 |
| GPT-4o + Ours (Original) | **0.900** | 8.74 | 8.79 |
| GPT-4o + Ours (Simplified) | 0.848 | 8.40 | 8.46 |
| GPT-4o-mini Non-private | 0.853 | 8.60 | 8.69 |
| GPT-4o-mini + Ours (Original) | 0.851 | 8.58 | 8.63 |

### Comparison with DP Baselines (MMLU-Biz F1, ε=1)

| Method | GPT-4o-mini | GPT-4o |
|------|-----------|--------|
| RanText (Standard DP) | 0.381 | 0.390 |
| CusText (Standard DP) | 0.511 | 0.473 |
| DP-Prompt (Standard DP) | 0.497 | 0.496 |
| CusText+ (Relaxed DP) | 0.686 | 0.694 |
| InferDPT (Relaxed DP) | 0.700 | 0.712 |
| **Ours (Simplified)** | **0.817** | **0.848** |

### Key Findings
- The Original version suffers almost no utility loss (difference from non-private setting <1%); the Simplified version has an average loss of approximately 4.9%.
- At ε=1, it achieves 2.2×/1.7×/1.7×/1.2×/1.2× higher average utility than RanText/CusText/DP-Prompt/CusText+/InferDPT, respectively.
- Query Cost: Reduced by up to 5.6× under concentrated distribution (β=0.05); Simplified version further improves batching efficiency.
- Attack Experiments: Composition filtering reduces attack success rates from ~80% to 58-63%, a decrease of about 32.7%.
- Attribute inference attack ASR is comparable to DP-Prompt/CusText+/InferDPT, but with significantly higher utility.

## Highlights & Insights
- Batch-level privacy protection represents an elegant paradigm shift—from "protecting each prompt" to "protecting the prompt's membership in a batch."
- Completely Model-Agnostic: Requires no access to model parameters or architecture modifications, making it directly applicable to any commercial LLM API.
- The Original version provides privacy protection with near-zero utility loss because real prompts are sent intact (merely mixed among noise prompts).
- Solid integration of theory and experimentation: $(A_n, \epsilon)$-indistinguishability is clearly defined, and its relationship with standard DP is explicitly articulated.

## Limitations & Future Work
- Assumes the noise sampler and service provider do not collude (curious-but-honest), although the paper discusses multi-server extensions in the appendix to address stronger threat models.
- Users must identify private attributes and generate substitutes themselves, which increases the burden on the user side.
- Request grouping depends on the quality of semantic clustering for generic instructions; long-tail or rare instructions may be difficult to group (though these are the scenarios where cost reduction is most needed).
- Prompt simplification in the Simplified version introduces additional utility loss; the choice of simplification strategy affects the final performance.

## Related Work & Insights
- SMPC methods like Iron / BOLT / NEXUS provide strong guarantees but high overhead; SharedRequest achieves practical privacy protection in a more lightweight manner.
- LDP methods like CusText / DP-Prompt directly perturb tokens; SharedRequest avoids semantic loss through batch obfuscation.
- The relaxed DP variant $(A_n, \epsilon)$-indistinguishability is a meaningful theoretical contribution that could inspire privacy definitions in other fields.
- The idea of leveraging the massive concurrency of commercial LLMs for privacy protection can be extended to other cloud service scenarios.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The batch-level privacy protection paradigm is a fundamental innovation; the three-party protocol design is comprehensive.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluation is extensive across utility, attacks, and costs, validated on multiple models and datasets.
- Writing Quality: ⭐⭐⭐⭐ Rigorous problem formalization with well-coordinated theoretical analysis and experimental validation.
- Value: ⭐⭐⭐⭐⭐ Addresses the practical need for LLM privacy protection with high utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] SafeMERGE: Preserving Safety Alignment in Fine-Tuned Large Language Models via Selective Layer-Wise Model Merging](safemerge_preserving_safety_alignment_in_fine-tuned_large_language_models_via_se.md)
- [\[ICLR 2026\] SecP-Tuning: Efficient Privacy-Preserving Prompt Tuning for Large Language Models via MPC](../../ICLR2026/llm_safety/secp-tuning_efficient_privacy-preserving_prompt_tuning_for_large_language_mode.md)
- [\[ACL 2026\] Privacy Collapse: Benign Fine-Tuning Can Break Contextual Privacy in Language Models](privacy_collapse_benign_fine-tuning_can_break_contextual_privacy_in_language_mod.md)
- [\[NeurIPS 2025\] CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing](../../NeurIPS2025/llm_safety/cryptomoe_privacy-preserving_and_scalable_mixture_of_experts_inference_via_balan.md)
- [\[ACL 2026\] Topic-Based Watermarks for Large Language Models](topic-based_watermarks_for_large_language_models.md)

</div>

<!-- RELATED:END -->
