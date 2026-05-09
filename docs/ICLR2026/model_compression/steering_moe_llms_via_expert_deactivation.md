---
title: >-
  [Paper Note] Steering MoE LLMs via Expert (De)Activation
description: >-
  [ICLR 2026][Model Compression][MoE] This paper proposes SteerMoE, which detects behavior-correlated experts via contrastive paired inputs and steers MoE LLM behavior at inference time by activating or deactivating specific experts (safety +20%, faithfulness +27%), while also exposing the fragility of safety alignment in MoE models (safety collapse −100%).
tags:
  - ICLR 2026
  - Model Compression
  - MoE
  - expert routing
  - behavior steering
  - safety
  - faithfulness
  - inference-time control
date: 2026-05-08
content_hash: 767a1b703a39324a
---

# Steering MoE LLMs via Expert (De)Activation

**Conference**: ICLR 2026
**arXiv**: [2509.09660](https://arxiv.org/abs/2509.09660)
**Code**: [github.com/adobe-research/SteerMoE](https://github.com/adobe-research/SteerMoE)
**Area**: Model Compression / Interpretability & Safety
**Keywords**: MoE, expert routing, behavior steering, safety, faithfulness, inference-time control

## TL;DR

This paper proposes SteerMoE, which detects behavior-correlated experts via contrastive paired inputs and steers MoE LLM behavior at inference time by activating or deactivating specific experts (safety +20%, faithfulness +27%), while also exposing the fragility of safety alignment in MoE models (safety collapse −100%).

## Background & Motivation

- MoE architectures achieve efficient inference via sparse routing, yet the controllability and interpretability of routing mechanisms remain insufficient.
- **Core Insight**: The MoE router not only allocates computation but also serves as a **signal-rich, controllable interface**.
- The hypothesis is that specific experts are entangled with specific behaviors (e.g., safety, faithfulness), and detecting and controlling these experts enables test-time behavioral steering.
- The approach is double-edged: it serves as a tool for alignment while simultaneously exposing unique safety vulnerabilities in MoE models.

## Method

### Paired-Sample Routing Discrepancy Detection

Given paired inputs $(x^{(1)}, x^{(2)})$ exhibiting opposing behaviors, the activation rate difference for each expert is computed as:

$$p^{(1)}_{\ell,i} = \frac{A^{(1)}_{\ell,i}}{N^{(1)}}, \quad p^{(2)}_{\ell,i} = \frac{A^{(2)}_{\ell,i}}{N^{(2)}}$$

$$\Delta_{\ell,i} = p^{(1)}_{\ell,i} - p^{(2)}_{\ell,i}$$

$\Delta_{\ell,i} > 0$ indicates that expert $i$ is associated with behavior 1, while $\Delta_{\ell,i} < 0$ indicates association with behavior 2. Experts to be manipulated are selected by ranking $|\Delta_{\ell,i}|$.

### Steering Configuration

1. Routing logits are mapped to log-softmax scores $\mathbf{s} = \log \text{softmax}(\mathbf{z})$ for scale normalization.
2. **Activation rule**: $s_e \leftarrow s_{\max} + \varepsilon$ for $e \in \mathcal{A}^+$
3. **Deactivation rule**: $s_e \leftarrow s_{\min} - \varepsilon$ for $e \in \mathcal{A}^-$
4. Re-apply softmax normalization → top-$k$ selection → weighted aggregation.

Key design: $\varepsilon$ is kept small to ensure that steered experts receive the highest or lowest priority without overwhelming other experts, thereby preserving the mixture-of-experts structure.

### Contrastive Pair Construction

- **Faithfulness**: $x^{(1)}$ = "Document: {Context} Question: {Q}" (with document); $x^{(2)}$ = "Question: {Q}" (without document).
- **Safety**: $x^{(1)}$ = safe refusal response; $x^{(2)}$ = unsafe compliant response (using the BeaverTails dataset).

## Key Experimental Results

### Safety Steering (AdvBench, evaluated with Llama-Guard-3-8B)

| Model | Direct Instruction | SteerMoE Unsafe | SteerMoE + AIM |
|---|---|---|---|
| GPT-OSS-120B | 100% safe | 90% safe | **0% safe** |
| Qwen3-30B | 98% safe | 60% safe | 2% safe |
| Phi-3.5-MoE | 100% safe | 94% safe | **0% safe** |

### Faithfulness Steering

| Steering Direction | FaithEval-CF | FaithEval-Unans | CF-TriviaQA | Avg. Improvement |
|---|---|---|---|---|
| Toward faithful | +10%~+27% | Significant gain | Significant gain | **Up to +27%** |
| Control set MCTest | No degradation | — | — | No impact on general QA |

### Key Safety Findings

| Attack Combination | GPT-OSS-120B | Qwen3 | Phi-3.5 | OLMoE |
|---|---|---|---|---|
| AIM alone | 100% | 2% | 96% | 100% |
| FFA alone | 100% | 48% | 100% | 92% |
| **SteerMoE + AIM** | **0%** | **2%** | **0%** | **36%** |

### Key Findings

1. Safety- and faithfulness-correlated experts concentrate in the **middle layers** of the model.
2. Safety experts activate predominantly on safe tokens and unsafe experts on unsafe tokens, yielding a natural token-level attribution signal.
3. SteerMoE is **orthogonal** to existing jailbreak methods; their combination can completely bypass all safety guardrails.
4. The findings expose "alignment illusion" in MoE models: safety alignment concentrates along a small number of expert pathways and collapses with minor routing perturbations.

## Highlights & Insights

- **Dual-edged analysis**: The same method can both enhance safety/faithfulness (+20%/+27%) and completely destroy safety (−100%).
- **Lightweight and efficient**: No modification to model weights, no additional training required; leverages existing routing computation.
- **Exposes fundamental vulnerability**: Safety guardrails of GPT-OSS-120B collapse from 100% → 0% under SteerMoE + AIM.
- **New dimension of "alignment illusion"**: Safety alignment must cover all routing paths, not merely a few expert pathways.
- **Interpretability byproduct**: Expert activation patterns serve as token-level attribution and hallucination detection signals.

## Limitations & Future Work

- Applicable only to MoE architectures; not directly transferable to dense models.
- Constructing contrastive paired inputs requires behavioral contrast, which is difficult for subtle behaviors.
- The optimal number of steered experts depends on model architecture hyperparameters and requires per-model tuning.
- The ethical risks associated with safety attacks warrant attention.

## Related Work & Insights

- MoE analysis: Mixtral lexical specialization, OLMoE routing saturation, domain specialization, etc.
- LLM steering: LM-Steers, representation engineering, RICE, etc.
- Safety attacks: GCG, ArtPrompt, AIM jailbreak methods.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — Reframes MoE routing as a controllable behavioral interface.
- **Technical Depth**: ⭐⭐⭐⭐ — Method is concise yet analysis is comprehensive.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — 11 benchmarks × 6 models, across safety and faithfulness dimensions.
- **Practicality**: ⭐⭐⭐⭐ — Zero-cost inference-time steering, though the safety attack surface warrants concern.

<!-- RELATED:START -->

## Related Papers

- [\[ICLR 2026\] SERE: Similarity-based Expert Re-routing for Efficient Batch Decoding in MoE Models](sere_similarity-based_expert_re-routing_for_efficient_batch_decoding_in_moe_mode.md)
- [\[ACL 2026\] Analytical FFN-to-MoE Restructuring via Activation Pattern Analysis](../../ACL2026/model_compression/analytical_ffn-to-moe_restructuring_via_activation_pattern_analysis.md)
- [\[AAAI 2026\] CAMERA: Multi-Matrix Joint Compression for MoE Models via Micro-Expert Redundancy Analysis](../../AAAI2026/model_compression/camera_multi-matrix_joint_compression_for_moe_models_via_mic.md)
- [\[ICLR 2026\] KBVQ-MoE: KLT-guided SVD with Bias-Corrected Vector Quantization for MoE Large Language Models](kbvq-moe_klt-guided_svd_with_bias-corrected_vector_quantization_for_moe_large_la.md)
- [\[ACL 2026\] Compositional Steering of Large Language Models with Steering Tokens](../../ACL2026/model_compression/compositional_steering_of_large_language_models_with_steering_tokens.md)

<!-- RELATED:END -->
