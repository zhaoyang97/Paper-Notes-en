---
title: >-
  [Paper Note] Anchored Decoding: Provably Reducing Copyright Risk for Any Language Model
description: >-
  [ICML2026][LLM Safety][Copyright memorization] This paper proposes Anchored Decoding: anchoring a high-performance but potentially risky LM to a safe LM trained only on open-licensed data during inference. It provides a…
tags:
  - "ICML2026"
  - "LLM Safety"
  - "Copyright memorization"
  - "inference-time decoding"
  - "safe reference model"
  - "KL constraint"
  - "ByteSampler"
date: 2026-05-08
content_hash: 661ecea4d8ef9f7c
---

# Anchored Decoding: Provably Reducing Copyright Risk for Any Language Model

**Conference**: ICML2026  
**arXiv**: [2602.07120](https://arxiv.org/abs/2602.07120)  
**Code**: No public code link (local cache does not contain a repository)  
**Area**: LLM Security / Copyright Risk Mitigation  
**Keywords**: Copyright memorization, inference-time decoding, safe reference model, KL constraint, ByteSampler  

## TL;DR
This paper proposes Anchored Decoding: anchoring a high-performance but potentially risky LM to a safe LM trained only on open-licensed data during inference. It provides a formal guarantee for the trade-off between copyright reproduction risk and generation quality using an adjustable information budget.

## Background & Motivation
**Background**: The capabilities of large language models largely stem from pre-training on massive web-scale corpora, which often contain a mix of open-licensed text, copyrighted material, and content of unclear origin. Research has shown that models do not just learn abstract patterns but can also memorize segments from the training set, outputting books, news, or other protected text verbatim under specific prompts.

**Limitations of Prior Work**: The most direct way to eliminate copyright material at the source is to retrain models after data cleaning. However, this is nearly unbearable for frontier models, and copyrighted text is often of high quality; simple removal sacrifices downstream capabilities. Common deployment strategies like system prompts, n-gram blocking, or retrieval-based rejection are fragile: system prompts fail to significantly reduce reproduction, and hard blocking depends on external databases and often conflates surface-level repetition with actual infringement risk.

**Key Challenge**: High-capability risky LMs possess stronger fluency, factuality, and long-tail knowledge but are more likely to reproduce protected training samples; safe LMs have cleaner training sources but are typically smaller and weaker. The problem is not choosing between a "safe-only" or "strong-only" model, but controlling the extent to which the risky LM deviates from the safe LM at each generation step, ensuring the output retains the utility of the strong model without following distribution peaks that likely stem from memorization.

**Goal**: The authors aim for an inference-time method that requires no retraining, no access to original training data, and can be applied to any language model that outputs logits. This method needs a user-adjustable risk knob to satisfy global $K$-NAF information budget constraints relative to a safe LM at the sequence level, while maintaining better utility than existing mitigation methods in realistic long-text reproduction evaluations.

**Key Insight**: The paper treats the safe LM as a "trusted anchor distribution" and the risky LM as a "high-utility candidate distribution." If the risky LM is exceptionally confident about a continuation that the safe LM does not support, this distributional discrepancy often corresponds to training data memorization or copyright-sensitive states. Therefore, the KL divergence between the two can serve as a risk signal, and the decoding distribution at each step can be projected near the safe LM.

**Core Idea**: Use distribution projection under a KL budget constraint to fuse the next-token distributions of the risky LM and safe LM into a new decoding distribution that is close to the risky LM yet strictly anchored by the safe LM.

## Method
The core of Anchored Decoding is an inference-time two-model fuser. It does not modify model parameters nor does it require knowledge of the specific texts in the risky LM's training set. As long as the logits of both the risky LM and safe LM under the current prefix are accessible, a new sampling distribution can be calculated at each decoding step. This new distribution is not a simple linear interpolation but is derived from a local optimization problem: "stay as close to the risky LM as possible while keeping the KL divergence relative to the safe LM within the current budget."

### Overall Architecture
The input consists of a user prompt $x$, a risky LM $p_r$ (possibly containing copyright memory), a safe LM $p_s$ (trained only on open-licensed data), a maximum generation length $T_{max}$, and a user-selected global information budget $K$. The system first calculates a "prefix debt" based on the prompt to determine if the prompt has already strongly triggered the risky LM's memorization patterns. Then, at each decoding step, both $p_r$ and $p_s$ are forwarded to obtain two next-token distributions.

If the remaining budget is generous, the fused distribution stays closer to $p_r$ to preserve the high-performance model's quality and factuality. If the current prefix appears risky or previous steps have consumed significant budget, the distribution moves closer to $p_s$. The local KL cost of each step is recorded in an accumulated ledger. The paper proves that as long as the sum of per-step budgets does not exceed $K$, the entire sequence distribution satisfies a global $K$-NAF guarantee relative to the safe LM.

To overcome the limitations of shared tokenizers, the paper also proposes AnchoredByte Decoding. It utilizes ByteSampler to convert token-level LMs into precise next-byte distributions, performing the same KL-constrained fusion over 256 bytes. This allows safe and risky LMs to be combined at the byte level even if they use different BPE tokenizers.

### Key Designs
1. **Geometric Mean Fusion under KL Constraints**:
	- **Function**: Constructs a new distribution $p_t^*$ at each decoding step that stays close to the risky LM while remaining within the local KL budget of the safe LM.
	- **Mechanism**: The local problem can be written as $\min_p D_{KL}(p \| p_r)$ subject to $D_{KL}(p \| p_s) \le k_t$. The closed-form solution is a weighted geometric mean of the two distributions: $p_t^* \propto p_s^{\lambda/(1+\lambda)} p_r^{1/(1+\lambda)}$. The Lagrange multiplier $\lambda$ is solved via 1D root finding; $\lambda$ increases (moving towards $p_s$) when budget is tight and decreases (moving towards $p_r$) when the budget is loose.
	- **Design Motivation**: Linear interpolation only offers empirical trade-offs and lacks a direct link to sequence-level risk guarantees; this projection form is derived from an optimization objective, making "utility maximization" and "risk boundaries" two sides of the same mathematical coin.

2. **Prefix Debt and Adaptive Budget Bank**:
	- **Function**: Incorporates the prompt's inherent risk into the decoding budget and allows low-risk steps to save budget for subsequent high-risk steps.
	- **Mechanism**: The paper computes the log-likelihood ratio at each position of the prompt: $\ell_i(x)=\log p_r(x_i|x_{<i})/p_s(x_i|x_{<i})$. The average of the largest positive LLRs is taken as $\delta_{init}(x)$. If a prompt (like the start of a famous novel) is much more strongly supported by $p_r$, budget is deducted upfront, making early decoding more dependent on $p_s$. The budget for each step is then set as $k_t=\max(0,(t+1)k-\sum_{i<t}a_i-\delta_{init})$, where $a_i$ is the actual KL cost.
	- **Design Motivation**: Reproduction events are often front-loaded; the first few steps are most likely to slide into memorized text. Fixed per-step budgets waste safety margin on naturally consistent steps. Prefix debt ensures "conservative starts," while the budget bank ensures "save where possible, spend where needed."

3. **AnchoredByte Cross-Tokenizer Fusion**:
	- **Function**: Allows the same anchoring principle to apply even when the safe LM and risky LM do not share a tokenizer.
	- **Mechanism**: ByteSampler marginalizes the token distribution based on the current byte prefix to get a next-byte distribution; AnchoredByte solves the same KL constraint problem in byte space $\mathcal{B}=\{0x00,\ldots,0xFF\}$. Since one token roughly corresponds to 4 bytes in English, the byte generation length is set to $B_{max}\approx 4T_{max}$, with a total budget $K=kB_{max}$.
	- **Design Motivation**: In copyright safety scenarios, the most trusted safe model might not use the same tokenizer as popular risky models (e.g., Comma 7B uses a custom tokenizer). Byte-level fusion sacrifices some inference efficiency but significantly expands the set of composable model pairs.

### Loss & Training
Anchored Decoding is a purely inference-time algorithm and does not require training the risky LM or fine-tuning the safe LM. The only new model is TinyComma 1.8B: the authors trained this decoder-only safe LM using 169.5B open-licensed tokens from the Common Pile to align with the Llama 3.1 tokenizer. Training occurred in two stages: first on 156B tokens from the full Common Pile, followed by a "cooldown" phase with 13.5B tokens of high-quality mixed data (70% Wikimedia, 15% DOAB, and 15% Data Provenance Initiative data).

TinyComma 1.8B has approximately 1.76 billion parameters, a hidden size of 2048, 24 layers, 32 attention heads, and uses the Llama 3 series' 128K vocabulary. The authors emphasize that the goal of TinyComma is not to top small-model leaderboards but to provide a safe anchor with clear sourcing and tokenizer compatibility.

## Key Experimental Results

### Main Results
The paper evaluates copyright reproduction risk in the "Books" domain, using fragments from 16 novels in CopyBench that are still under US copyright and known to be memorized by LLMs. Risk is measured by Normalized Copyright Reduction (NCR), the normalized average of ROUGE-1, ROUGE-L, MinHash, ACS, word-level LCS, and character-level LCS. NCR represents the gap closed between the risky baseline and the safe reference; an NCR of at least 75% is defined as the "high-protection operating point."

Utility is evaluated in two ways: fluency of book continuations (scored 1-5 by Prometheus-v2) and factuality of biographies (Bios prompts) measured by FActScore's supported claim precision. The main experiments cover six safe/risky model pairs.

| Method | Model Pair / Granularity | Factuality (High Protection) | Fluency (High Protection) | Key Info |
| :--- | :--- | :--- | :--- | :--- |
| Safe reference | TinyComma 1.8B / Llama 3.1 70B, token | 0.09 | 3.00 | Low risk, weak quality |
| MemFree | Same, token | 0.37 | 3.18 | Reaches threshold, high utility loss |
| RCAD | Same, token | 0.37 | 3.38 | More fluent than MemFree, still lower than two-model methods |
| CP-Fuse | Same, token | 0.20 | 3.21 | Stronger assumptions, subpar utility |
| TokenSwap | Same, token | 0.44 | 3.77 | One of the strongest baselines, relies on token seed list |
| **Anchored Decoding** | Same, token | **0.53** | **4.02** | **Highest utility at high protection** |
| **AnchoredByte** | Comma 7B / Llama 3.1 70B, byte | **0.52** | **4.23** | **Superior in cross-tokenizer scenarios** |
| **AnchoredByte** | Comma 7B / Llama 4 Scout, byte | **0.56** | **4.46** | **Strongest Pareto performance across pairs** |

Results show that Anchored/AnchoredByte defines a new Pareto frontier in the high-protection region. For the TinyComma + Llama 3.1 70B combination, it improves factuality to 0.53 and fluency to 4.02, whereas the strong baseline TokenSwap only achieves 0.44/3.77. Byte-level results are similar: AnchoredByte reaches 0.52/4.23 with Comma 7B + Llama 3.1 70B, significantly higher than CP-Fuse (0.23/3.75) and RCAD (0.46/3.46).

### Ablation Study
Ablations were conducted on the TinyComma 1.8B + Llama 3.1 70B token-level setup across three categories: optimization objective, prefix debt, and budget allocation.

| Configuration | Replaced Design | Observed Impact | Description |
| :--- | :--- | :--- | :--- |
| NoOpt | No KL projection; sample risky if budget allows, else safe | Pareto curve degrades significantly | Closed-form projection is key to preserving local utility |
| ColdStart | Sample safe for first $N$ steps, then risky | Worse than full method | Fixed cold starts fail to adapt to varying prompt risks |
| AD $\infty$ | Swap KL for $\infty$-Rényi divergence | Stronger fluency, weaker factuality | Provides an alternative worst-case guarantee |
| NoDebt | Remove prefix debt | Risk-utility curve degrades | Prompt-level signals are vital for front-loaded reproduction |
| AvgDebt | Average all prefix LLRs instead of top-n | Consistently worse than top-n | Risk comes from tail anomalies, not the mean |
| Fixed | Constant per-step budget; no carry-over | More conservative than adaptive | Wastes budget on low-risk steps; insufficient for high-risk ones |
| Global | Single global budget; switch to safe when exhausted | Performance inferior to budget bank | Leads to early overspending and quality collapses |

### Key Findings
- **Superiority at high protection**: Anchored Decoding does more than reduce risk; it preserves more factuality and fluency than baselines. This suggests "anchoring" is more granular than simple blocking or instructing.
- **Temporal structure of reproduction**: The top-n LLR design for prefix debt aligns with the observation that copyright prompts have heavy right-tailed LLRs and reproduction occurs early.
- **Efficiency**: Token-level wall-clock TPS slowdown is approx. 1.1× with a TTFT of 195.9 ms, more acceptable for deployment than RCAD's 2.0× slowdown.
- **Sanity Checks**: At a high-protection point ($k=1.5$), performance remains close to Llama 3.1 70B on standard tasks (e.g., HumanEval pass@1 is 0.488 vs. 0.506).

## Highlights & Insights
- **Drafting risk as a distribution constraint**: Rather than relying on prompt engineering or post-processing, this paper frames the problem as "keeping the generation distribution within an information distance from a safe reference."
- **Safe LM as an anchor, not a replacement**: The safe LM doesn't need to be as strong as the primary model; it only needs to define the boundaries of acceptable deviation. Trusted models can be smaller as long as they provide compliant distributions.
- **Prefix debt captures timing**: By using prompt LLR tail statistics to deduct budget upfront, the method targets the risk peaks typical of book openings without being overly restrictive throughout the sequence.
- **Byte-level fusion is practical**: Addressing the tokenizer mismatch problem allows the framework to be used with a much wider array of model pairs in real-world settings.

## Limitations & Future Work
- **Not a legal "non-infringement" proof**: $K$-NAF guarantees bounded divergence relative to a safe LM, which is a technical metric rather than a definitive legal certificate of compliance.
- **Cannot zero out reproduction probability**: As a sampling strategy, it cannot provide an absolute zero-repetition guarantee if the safe LM itself has a non-zero probability for certain segments.
- **KL divergence isn't purely risk**: Discrepancies between models might also stem from valuable, non-copyrighted long-tail facts. Anchoring can "collaterally damage" legitimate rare knowledge if the safe LM is too small.
- **Dependency on source provenance**: If the "safe" corpus itself contains fragments of protected text (e.g., quotes on forums), the anchor becomes biased.
- **Focus on parametric memory**: The method does not address in-prompt injection where a user provides copyrighted text and asks for a transformation; this remains an input governance issue.

## Related Work & Insights
- **vs. System prompt**: System prompts fail to consistently reach high-protection thresholds; Anchored Decoding acts directly on logits, making it more robust for both base and instruct models.
- **vs. MemFree**: MemFree relies on explicit n-gram blocklists. Anchored Decoding doesn't need to know the specific protected texts and handles broader memory risks via distributional differences.
- **vs. RCAD**: RCAD requires more compute (extra risky-model forward) and shows significant utility drops at high protection. Anchored Decoding is more efficient and preserves higher quality.
- **vs. TokenSwap**: TokenSwap uses a predefined list of function words and assumes small models lack memory. Anchored Decoding is more principled, utilizing a constrained safe LM and a formal KL budget.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Unifies copyright mitigation, safe references, and dual-model decoding under a KL budget with formal guarantees.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers six model pairs and diverse metrics, though legal risk and non-textual reproduction involve technical approximations.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clearly explained from theoretical guarantees to deployment constraints; well-structured figures and appendices.
- **Value**: ⭐⭐⭐⭐⭐ Highly valuable for practitioners needing to balance model capability with copyright compliance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] dgMARK: Decoding-Guided Watermarking for Diffusion Language Models](dgmark_decoding-guided_watermarking_for_diffusion_language_models.md)
- [\[ACL 2026\] RISK: A Framework for GUI Agents in E-commerce Risk Management](../../ACL2026/llm_safety/risk_a_framework_for_gui_agents_in_e-commerce_risk_management.md)
- [\[ICML 2026\] COFT: Counterfactual-Conformal Decoding for Fair Chain-of-Thought Reasoning in Large Language Models](coft_counterfactual-conformal_decoding_for_fair_chain-of-thought_reasoning_in_la.md)
- [\[ICLR 2026\] Self-Destructive Language Model](../../ICLR2026/llm_safety/self-destructive_language_model.md)
- [\[ICML 2026\] Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models](less_diverse_less_safe_the_indirect_but_pervasive_risk_of_test-time_scaling_in_l.md)

</div>

<!-- RELATED:END -->
