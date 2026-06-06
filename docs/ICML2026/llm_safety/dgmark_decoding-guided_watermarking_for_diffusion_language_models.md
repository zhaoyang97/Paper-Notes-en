---
title: >-
  [Paper Note] dgMARK: Decoding-Guided Watermarking for Diffusion Language Models
description: >-
  [ICML 2026][LLM Safety][dLLM Watermarking] dgMARK utilizes the "decoding order freedom" inherent in Diffusion Language Models (dLLMs) as a watermarking channel. By prioritizing the decoding of positions that satisfy pari…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "dLLM Watermarking"
  - "Decoding Order"
  - "Parity Hashing"
  - "Robust Detection"
  - "No Probability Re-weighting"
date: 2026-05-08
content_hash: 335a6a8db09036ac
---

# dgMARK: Decoding-Guided Watermarking for Diffusion Language Models

**Conference**: ICML 2026  
**arXiv**: [2601.22985](https://arxiv.org/abs/2601.22985)  
**Code**: https://dgmark-watermarking.github.io  
**Area**: LLM Security / Watermarking / Diffusion Language Models  
**Keywords**: dLLM Watermarking, Decoding Order, Parity Hashing, Robust Detection, No Probability Re-weighting

## TL;DR
dgMARK utilizes the "decoding order freedom" inherent in Diffusion Language Models (dLLMs) as a watermarking channel. By prioritizing the decoding of positions that satisfy parity conditions according to binary hashes, it embeds statistically detectable watermarks into models like LLaDA and Dream without modifying token probability distributions, maintaining robustness against insertion, deletion, substitution, and rewriting.

## Background & Motivation

**Background**: Content provenance in LLMs primarily relies on watermarking. Dominant approaches (e.g., green/red lists by Kirchenbauer et al.) embed signals by biasing token probabilities, which leads to significant quality loss. Distortion-free variants (GumbelMax, long pseudo-random sequences) preserve distributions but are slow and depend on fixed causal contexts. Recently, dLLMs (LLaDA, Dream, Mercury, Gemini Diffusion) have emerged, challenging the autoregressive paradigm by revealing tokens in arbitrary orders.

**Limitations of Prior Work**: Existing watermarks assume left-to-right generation and require "preceding text" as a hashing seed. Since dLLMs lack a fixed prefix, classic schemes are either inapplicable or modified to "bias probabilities anyway," continuing to incur quality costs. Few concurrent dLLM watermarking studies (Bagchi, Wu, Gloaguen, Raban, etc.) still primarily focus on altering token selection probabilities.

**Key Challenge**: dLLMs provide a new control knob—decoding order. Ideally, they should be order-invariant (any permutation should yield the same distribution); in reality, they are highly sensitive to order due to imperfect training approximations (Kim et al. 2025). This discrepancy serves as a potential watermark channel that has not been systematically exploited.

**Goal**: To design a watermark that leaves token probabilities completely untouched by embedding signals solely through guided decoding orders. The method should (1) be compatible with general decoding strategies like confidence, entropy, or margin, and (2) maintain detection rates under insertion, deletion, substitution, and rewriting attacks.

**Key Insight**: It is observed that at each step, dLLMs calculate a reward $r_j$ and sample a candidate $v_j$ for each unrevealed position $j$, prioritizing the selection of the position with the maximum $r_j$. If a binary hash tied to the position index is used to select the highest-reward position among "matching candidates," the parity-matching rate of the watermarked text can be systematically increased above 0.5 without altering probabilities.

**Core Idea**: Shift watermarking from "distorting token probabilities" to "distorting decoding order." A secret-key-derived binary hash partitions the vocabulary for each position into parity-matching and residual sets. During decoding, positions where candidates fall into the parity-matching set with high rewards are prioritized. Statistical detection then checks if the parity-matching rate is significantly higher than 0.5.

## Method

### Overall Architecture

Given a secret key $\xi$ and a deterministic hash function $f: \mathcal{V} \times \Xi \to \{0,1\}$. At position $i$, the vocabulary is partitioned into a parity set $\mathcal{G}_i = \{v \in \mathcal{V} \mid f(v, \xi) \equiv i \pmod 2\}$ and a residual set $\mathcal{R}_i = \mathcal{V} \setminus \mathcal{G}_i$ (the hash construction ensures balanced partitioning for any $\xi$).

For each step, dgMARK:
1. Obtains $(r_j, v_j)$ from the dLLM for each unrevealed $j$ (following underlying strategies like confidence, entropy, or margin).
2. Defines a candidate set $\mathcal{C} = \{j \mid v_j \in \mathcal{G}_j\}$; if empty, it reverts to $\mathcal{C} = \{j \notin \mathcal{I}\}$.
3. Selects $k^\star = \arg\max_{j \in \mathcal{C}} r_j$ and reveals $y_{k^\star} \leftarrow v_{k^\star}$.

Detection: Calculate $\mathbb{1}[f(y_i, \xi) \equiv i \pmod 2]$ position-wise for the generated text and perform a z-test to see if it is significantly $> 0.5$. For editing attacks, a sliding window is used to detect local alignment offsets.

### Key Designs

1. **Decoding Order as Watermark Channel (No Probability Change)**:
    - **Function**: Embeds statistically detectable signals while keeping the model distribution $p_\theta(y_j \mid y_\mathcal{I}, x)$ completely unchanged.
    - **Mechanism**: While classic watermarks bias logits for certain tokens, dgMARK only changes "who is decoded first." In an ideal order-invariant dLLM, this has no effect; however, the practical order-sensitivity of dLLMs transforms the preference for "choosing parity-matching positions first" into an observable increase in the parity-matching rate.
    - **Design Motivation**: To completely decouple "watermarking" from "model probability," eliminating the extra computational overhead of distortion-free watermarks and avoiding the quality degradation associated with probability biasing while remaining plug-and-play with any decoding strategy.

2. **Binary Parity Hashing + Position Index**:
    - **Function**: Performs position-dependent vocabulary bipartitioning using a key-derived hash.
    - **Mechanism**: For each position $i$, the vocabulary is split into two halves. Parity is determined by whether $f(v, \xi)$ matches $i \pmod 2$. During decoding, a "candidate token falling into the parity-matching half of that position" is prioritized.
    - **Design Motivation**: Position dependence prevents the watermark distribution from being broken by simple counting attacks (unlike static green lists). Binary and balanced splitting ensures the statistical properties for detection (0.5 under the null hypothesis). Keys are replaceable and can be hierarchical.

3. **Sliding Window Detection for Editing Attacks**:
    - **Function**: Maintains watermark detectability under insertion, deletion, substitution, and rewriting attacks.
    - **Mechanism**: Editing causes shifts in global position indices, leading to a loss of parity synchronization. The sliding window calculates the maximum parity-matching rate across all possible offsets to capture locally consistent alignment segments.
    - **Design Motivation**: Watermarks must be robust against manual or machine post-editing in real-world deployments. The sliding window is a lightweight and analytically friendly countermeasure compared to forced error-correction codes.

### Lookahead Variant
The basic dgMARK might lack parity-matching candidates in steps where confidence is extremely non-uniform. The lookahead variant simulates "what the reward distribution would be if a certain position were chosen," further pushing the watermark strength at the cost of $2\times$ inference time.

## Key Experimental Results

### Detectability vs. Text Quality (LLaDA-8B-Instruct, confidence decoding)

| Method | Detection AUC↑ | Perplexity↓ | MAUVE↑ |
|------|--------|--------|--------|
| No Watermark | 0.50 | 1.00× | 1.000 |
| Prob. Bias (Ported KGU) | 0.98 | 1.18× | 0.86 |
| dgMARK | **0.97** | **1.01×** | **0.97** |
| dgMARK + Lookahead | **0.99** | 1.03× | 0.95 |

dgMARK achieves an AUC comparable to the probability-biasing baseline, but its perplexity and MAUVE scores are nearly identical to the unwatermarked model, indicating the order channel is "virtually free."

### Post-editing Robustness (20% Substitution Rate)

| Attack | Prob. Bias AUC | dgMARK AUC | dgMARK + Window AUC |
|------|------------|----------|------------------|
| Word Sub. 20% | 0.81 | 0.85 | **0.94** |
| Insertion 10% | 0.70 | 0.79 | **0.92** |
| Deletion 10% | 0.68 | 0.77 | **0.91** |
| Rewriting (GPT-4) | 0.62 | 0.71 | **0.85** |

Sliding window detection significantly compensates for alignment offsets triggered by insertions/deletions. dgMARK is also more robust than probability biasing under rewriting attacks (order signals are harder for rewriters to erase synchronously than token signals).

### Key Findings
- **Order channel has near-zero quality cost**: Perplexity increases by only 1%, making it nearly indistinguishable from unwatermarked models. This suggests changing order has a far smaller impact on quality than changing probabilities.
- **Stable across decoding strategies**: dgMARK can be applied to confidence, entropy, and margin decoding, with detection AUC $> 0.95$ for all, proving its plug-and-play nature.
- **Sliding Window > Global Detection**: Detection AUC improves by 5–10 points across all editing attacks, which is critical for robustness.
- **Diminishing returns for Lookahead**: While it pushes AUC from 0.97 to 0.99, it doubles inference cost, making it suitable only for highly sensitive scenarios.

## Highlights & Insights
- **Identified a dLLM-unique watermark channel**: The freedom of decoding order is a "knob" that does not exist in the autoregressive paradigm. This work turns the "order sensitivity" of dLLMs—often viewed as a flaw—into a resource for watermarking, a classic "bug to feature" transformation.
- **Complete absence of probability re-weighting**: Among all LLM watermarks, this is one of the few truly $p_\theta$-preserving schemes. Theoretically, the KL divergence from the unwatermarked model distribution is zero (in practice, it is only affected by order sensitivity).
- **Plug-and-play philosophy**: dgMARK acts as a wrapper compatible with confidence/entropy/margin strategies. This means existing dLLM deployment systems do not need to modify training or inference frameworks.
- **Generality of sliding window detection**: The problem of editing attacks destroying global alignment exists for any position-dependent watermark. The sliding window solution presented here can be transferred to other such schemes.

## Limitations & Future Work
- Dependency on dLLM "order sensitivity": If future dLLMs become perfectly order-invariant, the watermark signal may vanish (an "adversarial training" risk).
- One-step lookahead doubles inference time; multi-step lookahead costs are exponential, making strong watermark modes expensive to deploy.
- Binary hashing provides only 1 bit per position, resulting in low channel capacity. Embedding complex signatures (e.g., timestamps) would require expansion to $k$-bit hashing.
- Primarily validated on LLaDA and Dream; generalization to larger scales or different dLLM architectures remains to be fully verified.
- Under rewriting attacks, AUC still drops to 0.85; strong adversarial rewriters might further degrade performance.

## Related Work & Insights
- **vs. Autoregressive Green/Red List (Kirchenbauer et al.)**: Those rely on fixed causal contexts which dLLMs lack, and biasing probabilities incurs quality costs. dgMARK avoids probability modification.
- **vs. Distortion-free Watermarking (Aaronson-Kirchner, Christ et al.)**: These preserve distributions via GumbelMax or long sequences but with high overhead. dgMARK preserves distributions on dLLMs with almost zero overhead.
- **vs. Concurrent dLLM Watermarking (Bagchi / Wu / Gloaguen / Raban)**: Those still focus on probability shaping or controlled sampling. dgMARK is the first to use decoding order exclusively as the channel.
- **Insight**: Any generative system that is "theoretically order-invariant but practically order-sensitive" (e.g., image or video diffusion) could potentially reuse the "order channel watermarking" concept.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Using decoding order as a channel is a genuinely new framing and pioneers the non-probabilistic path for dLLM watermarking.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered two dLLMs, three decoding strategies, four attacks, and multiple baselines. Lacks a detailed head-to-head comparison with concurrent dLLM watermarking works.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, intuitive algorithm diagrams. Theoretical analysis of how order sensitivity translates to detectable statistics could be deeper.
- Value: ⭐⭐⭐⭐⭐ As dLLMs rapidly industrialize (Mercury, Gemini Diffusion), provenance watermarking is a critical need. This work provides the first high-quality, low-overhead, and robust solution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models](coft_counterfactual-conformal_decoding_for_fair_chain-of-thought_reasoning_in_la.md)
- [\[ACL 2026\] Lost in Diffusion: Uncovering Hallucination Patterns and Failure Modes in Diffusion Large Language Models](../../ACL2026/llm_safety/lost_in_diffusion_uncovering_hallucination_patterns_and_failure_modes_in_diffusi.md)
- [\[ICML 2026\] Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models](towards_fine-grained_robustness_attention-guided_test-time_prompt_tuning_for_vis.md)
- [\[ICLR 2026\] wd1: Weighted Policy Optimization for Reasoning in Diffusion Language Models](../../ICLR2026/llm_safety/wd1_weighted_policy_optimization_for_reasoning_in_diffusion_language_models.md)
- [\[AAAI 2026\] Perturb Your Data: Paraphrase-Guided Training Data Watermarking](../../AAAI2026/llm_safety/perturb_your_data_paraphrase-guided_training_data_watermarking.md)

</div>

<!-- RELATED:END -->
