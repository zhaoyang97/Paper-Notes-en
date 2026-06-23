---
title: >-
  [Paper Note] Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives
description: >-
  [ICML 2026][Dialogue Systems][Paper Note] This paper models LLM-as-a-Service as a "principal-agent" problem, proving that current mainstream "pay-per-token" mechanisms naturally incentivize service providers to re-segment the same string into longer token sequences for overcharging. Furthermore, even if providers are forced to disclose next-token distributions
tags:
  - ICML 2026
  - Dialogue Systems
date: 2026-05-08
content_hash: 6cb69c95d27cdec0
---
# Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives

**Conference**: ICML 2026 Oral  
**arXiv**: [2505.21627](https://arxiv.org/abs/2505.21627)  
**Code**: https://github.com/Human-Centric-Machine-Learning/token-pricing (Available)  
**Area**: AI Safety / Mechanism Design / LLM-as-a-Service Pricing  
**Keywords**: Pay-per-token, Incentive Compatibility, Tokenization Multiplicity, Pay-per-character, Principal-Agent

## TL;DR
This paper models LLM-as-a-Service as a "principal-agent" problem, proving that current mainstream "pay-per-token" mechanisms naturally incentivize service providers to re-segment the same string into longer token sequences for overcharging. Furthermore, even if providers are forced to disclose next-token distributions, overcharging without detection remains NP-Hard rather than impossible—the authors provide a simple heuristic algorithm that increases reported tokens by up to 11.2% while maintaining plausibility. Finally, it is proven that the only additive pricing mechanism that eliminates this incentive is "linear pay-per-character."

## Background & Motivation
**Background**: Cloud LLM services (e.g., OpenAI, Gemini, Anthropic) almost exclusively use pay-per-token billing: users submit prompts, providers run models on their hardware, and charge based on the number of returned tokens multiplied by a unit price. Users only observe the returned string and the claimed token count; the internal vocabulary, actual segmentation, and next-token distributions remain proprietary to the provider.

**Limitations of Prior Work**: Tokenization is non-unique. The same string "Damascus" can be segmented as `|Dam|ascus|` (2 tokens) or `|Da|ma|s|cus|` (4 tokens), without user knowledge. Providers can "re-report" a generated 2-token sequence as 4 tokens to double the revenue while the string remains identical, leaving users with no technical means of detection.

**Key Challenge**: Moral hazard caused by information asymmetry—the provider fully observes the generation process, while the user only observes and pays for the final reported token sequence. As long as "pay-per-token" billing exists and the vocabulary contains multi-character tokens, replacing short segmentations with longer ones strictly increases revenue mathematically.

**Goal**: Decomposition into three sub-problems: (1) Does a structural incentive to lie exist under pay-per-token? (2) Does forcing providers to disclose next-token distributions (allowing users to verify plausibility) stop the cheating? (3) Is there a pricing mechanism that eliminates this incentive in principle?

**Key Insight**: The authors use the "principal-agent" framework from contract theory, treating the user as the principal, the provider as the agent, and the billing rule as the contract. They systematically characterize "incentive-compatibility" (IC)—a property where honest reporting is always at least as good as lying for the provider. This paradigm, common in auctions and insurance, is applied here to LLM pricing for the first time.

**Core Idea**: Billing by token length is inherently not incentive-compatible; the only additive and IC method is linear billing by character count. During transition, one simply sets $r_c = r_o \cdot \mathrm{tpc}$ (where $\mathrm{tpc}$ is the average tokens per character), allowing the provider's average profit margin to remain constant.

## Method

### Overall Architecture
Rather than proposing a new model, the paper constructs a complete chain of argumentation regarding the security of token-based LLM billing. It formalizes the service process using the principal-agent framework, proves that pay-per-token incentivizes re-segmenting strings into longer sequences, provides a heuristic algorithm to demonstrate the effectiveness of cheating even under transparency, and finally derives the only IC pricing method with a seamless migration formula.

The formal setup is: a user submits a prompt, the provider generates a true token sequence $\mathbf{t}$ (string $s = \mathrm{str}(\mathbf{t})$), and applies a reporting strategy $\pi$ to produce $\tilde{\mathbf{t}} \sim \pi(\mathbf{t})$, subject to the hard constraint $\mathrm{str}(\tilde{\mathbf{t}}) = s$. The provider's utility is $U_\pi(\tilde{\mathbf{t}}, \mathbf{t}) = r(\tilde{\mathbf{t}}) - c_\text{gen}(\mathbf{t}) - c_\pi(\mathbf{t})$, where revenue $r$ depends on billing rules, generation cost $c_\text{gen}(\mathbf{t}) \approx c_o \cdot \mathrm{len}(\mathbf{t})$ is proportional to true tokens, and $c_\pi$ is the overhead of the reporting strategy. The conclusions are based on "Incentive Compatibility"—per Definition 4, the honest strategy $\pi_0$ is weakly dominant if $U_{\pi_0}(\mathbf{t}, \mathbf{t}) \geq U_\pi(\tilde{\mathbf{t}}, \mathbf{t})$ for all strategies.

### Key Designs

**1. Formalization of Lying Incentives + Zero-cost Heuristic (Algorithm 1): Shattering the Illusion of Secure Pay-per-token**

Basic "pay-per-token" is an additive mechanism $r(\tilde{\mathbf{t}}) = \sum_i r(\tilde{t}_i)$, simplified as $r(\tilde{\mathbf{t}}) = r_o \cdot \mathrm{len}(\tilde{\mathbf{t}})$. This immediately reveals the problem: for any two reporting strategies with equal cost, if $\mathrm{len}(\tilde{\mathbf{t}}) > \mathrm{len}(\tilde{\mathbf{t}}')$, then $U_\pi > U_{\pi'}$. Longer sequences yield higher profits, making lying a structural optimum. Algorithm 1 illustrates this: it iteratively picks tokens that can be split into two non-empty sub-words and executes the split for $m$ steps. Since this requires no GPU verification, it is zero-cost. The authors argue that in a competitive market, a cheating provider can offer a lower per-token price while maintaining the same revenue by stretching sequences, making lying a "market weapon."

**2. Plausible Heuristic Lying (Algorithm 2) + NP-Hard Barrier: Transparency Cannot Stop Cheating**

If providers are forced to disclose next-token distributions (e.g., the set of candidates $\mathcal{V}_p$ in top-$p$ sampling), can users stop cheating by verifying the plausibility of segmentations? The authors prove that finding the "longest plausible segmentation" is NP-Hard (Theorem 3, via reduction from Hamiltonian Path). However, high complexity does not imply economic security. Algorithm 2 exploits the BPE empirical rule where higher IDs often correlate with longer tokens. It greedily splits tokens to keep the resulting sub-tokens common (high IDs) and then runs a single forward pass to verify if the sequence $\hat{t}_i \in \mathcal{V}_p(\hat{\mathbf{t}}_{\leq i-1})$. The criterion for profitability is $\mathbb{E}[\mathrm{plausible}(\hat{\mathbf{t}})] \cdot m \cdot r_o > c_v$. Since verification cost $c_v$ is independent of the sequence length added, this inequality holds easily under mainstream profit margins. Transparency merely forces providers to "cheat along the boundary of plausibility."

**3. Characterization of IC Pricing + Smooth Migration Formula: Pay-per-character is the Only Solution**

Proposition 5 proves that IC requires revenue $r(\tilde{\mathbf{t}})$ to depend only on the string $\mathrm{str}(\tilde{\mathbf{t}})$ and not the specific segmentation. Theorem 6 further proves that under the additive assumption, IC holds if and only if $r(\mathbf{t}) = \sum_{\sigma \in \Sigma} \mathrm{count}_\sigma(\mathbf{t}) \cdot r(\sigma)$ (linear character billing). Corollary 7 asserts that as long as the vocabulary contains multi-character tokens, pay-per-token is never IC. To facilitate industry adoption, the authors provide the migration formula $r_c = r_o \cdot \mathrm{tpc}$, where $\mathrm{tpc}$ is the mean tokens-per-character on a dataset. This keeps average profit margins stable while shifting the incentive: providers profit by building better tokenizers that compress strings more efficiently, rather than by re-segmenting strings.

### Loss & Training
This work focuses on mechanism design and theory rather than training. The "hyperparameters" are experimental knobs: top-$p$ sampling $p \in \{0.90, 0.95, 0.99\}$, temperature $T=1.3$, iteration count $m$ for Algorithm 2, and baseline profit margins $\rho_o \in \{0.2, 0.4, 0.6\}$. Profitability is determined by the criterion $\rho(\mathbf{t}) > 1 - \mathbb{E}[\mathrm{plausible}(\hat{\mathbf{t}})] \cdot m \cdot c_o / c_v$.

## Key Experimental Results

### Main Results
Evaluated on 600 prompts from LMSYS Chatbot Arena using Llama / Gemma / Ministral series ($T=1.3$).

| Model / Config | Overcharging ($p=0.99$) | Overcharging ($p=0.95$) | Overcharging ($p=0.90$) | Note |
|--------|------|------|------|------|
| Llama-3.2-1B + Alg. 2 | 11.2% | 1.8% | 0.28% | Higher $p$ allows more plausibility space for cheating |
| Llama-3.2-1B Utility Gain (any $\rho_o$) | >10.5% | 1.7% | 0.4% | Net profit remains positive after verification costs |
| Profit > 0 after pay-per-char ($\rho_o = 0.2/0.4/0.6$) | — | 92.4% / 93.3% / 94.1% | — | Most samples remain profitable after migration |

### Ablation Study

| Config | Key Metric | Note |
|------|---------|------|
| Algorithm 1 (No verification) | Linear increase in overcharge with $m$ | Easily detected via plausibility checks |
| Algorithm 2 vs. $m$ | Unimodal utility vs. $m$ | Excessive $m$ drops plausibility too fast, reducing expected gain |
| Profit margin $\rho_o$ changes | Higher relative gain for smaller $\rho_o$ | Incentive is stronger in low-margin price wars |

### Key Findings
- Key Design 2 contribution: Algorithm 2 + Theorem 3 show that while optimal cheating is NP-Hard, heuristic cheating is practically effective (10%+ revenue increase).
- Higher $p$ values increase the cheating space; thus, "creative writing" (high temperature/top-$p$) is the most vulnerable scenario.
- Lower profit margins increase the relative incentive to lie, suggesting that smaller providers in price wars are at higher systemic risk.

## Highlights & Insights
- The reduction to Hamiltonian Path for NP-Hardness, followed by the max-min ID heuristic, demonstrates that algorithmic complexity is not a substitute for economic security.
- Theorem 6 provides a necessary and sufficient characterization: pay-per-character is not just a suggestion but a mathematical necessity for additive IC pricing.
- The migration formula $r_c = r_o \cdot \mathrm{tpc}$ is engineering-friendly, requiring only a dataset statistic without changes to models or tokenizers.

## Limitations & Future Work
- Pay-per-character does not stop "verbosity attacks" (making models output more characters); this requires quality-based billing (e.g., pay-for-performance).
- Assumes providers cannot forge next-token distributions or the tokenizer itself; the authors suggest trusted execution environments (TEEs) or zero-knowledge proofs (ZKPs) for closed-source models.
- Experiments are on open-weights models and LMSYS data; performance on closed-source models and production-grade traffic remains to be seen.
- Analysis is micro-level; macro-market dynamics under multi-provider competition are left for future work.

## Related Work & Insights
- **vs. Saig et al. (2025)**: They also use principal-agent modeling but focus on model-substitution attacks; this work is complementary, focusing on segmentation-reporting attacks.
- **vs. Sun et al. (2025) / Cai et al. (2025)**: These are audit-oriented (detecting reasoning-step inflation or model swapping); this work is mechanism-design oriented (changing rules to remove incentives).
- **vs. Ahia et al. (2023)**: They noted that non-English users are overcharged due to BPE inefficiency; pay-per-character naturally resolves this fairness issue by equalizing the price per character across languages.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to rigorously characterize LLM pricing as a mechanism design problem with necessary/sufficient theorems.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered major open-weight families and multilingual prompts, though lacks closed-source production data.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical flow: modeling → revealing incentive → empirical heuristic → necessity theorem → migration recipe.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the core billing mechanism of LLM commercialization; conclusions are applicable to regulatory and contractual discussions.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Your Students Don't Use LLMs Like You Wish They Did](../../ACL2026/dialogue/your_students_dont_use_llms_like_you_wish_they_did.md)
- [\[ACL 2025\] Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles](../../ACL2025/dialogue/know_you_first_and_be_you_better_modeling_human-like_user_simulators_via_implici.md)
- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)
- [\[ACL 2026\] Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](../../ACL2026/dialogue/cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)
- [\[ICML 2025\] Investigating Non-Transitivity in LLM-as-a-Judge](../../ICML2025/dialogue/investigating_non-transitivity_in_llm-as-a-judge.md)

</div>

<!-- RELATED:END -->
