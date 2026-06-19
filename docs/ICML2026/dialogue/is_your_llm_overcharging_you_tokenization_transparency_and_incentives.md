---
title: >-
  [Paper Note] Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives
description: >-
  [ICML 2026][Dialogue Systems][Paper Note] This paper models LLM-as-a-Service as a "Principal-Agent" problem, proving that current "pay-per-token" billing naturally incentivizes providers to overcharge by re-segmenting the same string into longer token sequences. Even if providers are forced to disclose next-token distributions, overcharging without being detec
tags:
  - ICML 2026
  - Dialogue Systems
date: 2026-05-08
content_hash: 5e559dabb9673f8a
---
# Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives

**Conference**: ICML 2026 Oral  
**arXiv**: [2505.21627](https://arxiv.org/abs/2505.21627)  
**Code**: https://github.com/Human-Centric-Machine-Learning/token-pricing (Available)  
**Area**: AI Safety / Mechanism Design / LLM-as-a-Service Pricing  
**Keywords**: Pay-per-token, Incentive Compatibility, Tokenization Multiplicity, Pay-per-character, Principal-Agent

## TL;DR
This paper models LLM-as-a-Service as a "Principal-Agent" problem, proving that current "pay-per-token" billing naturally incentivizes providers to overcharge by re-segmenting the same string into longer token sequences. Even if providers are forced to disclose next-token distributions, overcharging without being detected remains feasible (NP-Hard but practically achievable). The authors provide a heuristic algorithm that increases token counts by up to 11.2% while maintaining plausibility, concluding that the only additive mechanism to eliminate this incentive is "linear billing per character length."

## Background & Motivation
**Background**: Cloud-based LLM services (e.g., OpenAI, Gemini, Anthropic) predominantly use pay-per-token billing: users submit prompts, providers run models on their hardware, and billing is based on the number of tokens generated multiplied by a unit price. Users only observe the output string and the reported token count; the specific vocabulary, actual segmentation, and next-token distributions remain proprietary to the provider.

**Limitations of Prior Work**: Tokenization is non-unique. The string "Damascus" can be segmented as `|Dam|ascus|` (2 tokens) or `|Da|ma|s|cus|` (4 tokens). Users have no technical means to detect if a provider "re-reports" a legitimate 2-token sequence as a 4-token sequence for double the revenue when the resulting string is identical.

**Key Challenge**: Moral hazard caused by asymmetry of information—the provider observes the full generation process, while the user only observes and pays for the final reported token sequence. As long as "pay-per-token" is used and the vocabulary contains multi-character tokens, replacing short segmentations with longer ones mathematically guarantees increased revenue.

**Goal**: To decompose the problem into three sub-questions: (1) Does structural incentive to lie exist under pay-per-token? (2) Does mandating the disclosure of next-token distributions (allowing users to verify plausibility) prevent cheating? (3) Does a pricing mechanism exist that eliminates this incentive in principle?

**Key Insight**: The authors use the Principal-Agent framework from contract theory—treating the user as the principal, the provider as the agent, and the billing rule as the contract—to systematically characterize "incentive compatibility": a property where reporting truthfully is never worse for the provider than lying. This paradigm, common in auctions and insurance, is applied here to LLM pricing for the first time.

**Core Idea**: Billing by token length is fundamentally not incentive-compatible. The only additive, incentive-compatible method is linear billing per character count. Transitioning can be achieved by setting $r_c = r_o \cdot \mathrm{tpc}$ (where $\mathrm{tpc}$ is the average tokens per character), allowing providers to maintain average profit margins.

## Method

### Overall Architecture
The paper does not propose a new model but constructs a complete logical chain regarding the security of pay-per-token LLM services: formalizing the service process via the principal-agent framework, proving the structural incentive to overcharge, demonstrating the failure of transparency to stop sophisticated cheating, and deriving a final incentive-compatible pricing formula.

The formal setup defines the user submission of a prompt, the provider generating a true sequence $\mathbf{t}$ (string $s = \mathrm{str}(\mathbf{t})$), and applying a reporting strategy $\pi$ to yield $\tilde{\mathbf{t}} \sim \pi(\mathbf{t})$ under the hard constraint $\mathrm{str}(\tilde{\mathbf{t}}) = s$. Provider utility is defined as $U_\pi(\tilde{\mathbf{t}}, \mathbf{t}) = r(\tilde{\mathbf{t}}) - c_\text{gen}(\mathbf{t}) - c_\pi(\mathbf{t})$. Revenue $r$ is determined by billing rules, generation cost $c_\text{gen}(\mathbf{t}) \approx c_o \cdot \mathrm{len}(\mathbf{t})$ is proportional to the true token count, and $c_\pi$ is the cost of the reporting strategy. "Incentive compatibility" (Definition 4) requires that the truthful strategy $\pi_0$ satisfies $U_{\pi_0}(\mathbf{t}, \mathbf{t}) \geq U_\pi(\tilde{\mathbf{t}}, \mathbf{t})$ for all strategies.

### Key Designs

**1. Formalizing the Incentive to Lie + Zero-Cost Heuristic (Algorithm 1): Exposing the Pay-per-Token Illusion**

Simple pay-per-token is an additive mechanism $r(\tilde{\mathbf{t}}) = \sum_i r(\tilde{t}_i)$, simplified to $r(\tilde{\mathbf{t}}) = r_o \cdot \mathrm{len}(\tilde{\mathbf{t}})$. This exposes a flaw: for any strategies $\pi, \pi'$ with equal costs, if $\mathrm{len}(\tilde{\mathbf{t}}) > \mathrm{len}(\tilde{\mathbf{t}}')$, then $U_\pi > U_{\pi'}$. Longer sequences yield higher profits. Algorithm 1 implements this by iteratively splitting tokens into non-empty sub-words until a limit is reached or only single characters remain, without GPU verification. This demonstrates how cheating allows a provider to offer lower unit prices while maintaining the same revenue, turning lying into a market-competitive weapon.

**2. Plausible Heuristic Cheating (Algorithm 2) + NP-Hard Barrier: Transparency Cannot Stop Cheating**

The paper investigates whether mandating the disclosure of the next-token distribution (specifically the top-$p$ set $\mathcal{V}_p$) prevents cheating. Theorem 3 proves that finding the "longest plausible segmentation" is NP-Hard (reduction from Hamiltonian Path), implying optimal cheating is computationally difficult. However, Algorithm 2 uses a max-min heuristic based on BPE properties (higher IDs often correspond to longer tokens) to split tokens into halves that remain likely in the model's distribution. After $m$ splits, a single forward pass validates the sequence. If plausible, it is reported. The strategy is profitable if $\mathbb{E}[\mathrm{plausible}(\hat{\mathbf{t}})] \cdot m \cdot r_o > c_v$ (where $c_v$ is verification cost). In practice, this achieves over 10% excess revenue for $p=0.99$, showing transparency only shifts cheating to the "boundaries of plausibility" rather than stopping it.

**3. Characterizing Incentive-Compatible Pricing + Transition Formula: Pay-per-Character as the Only Solution**

Proposition 5 proves that incentive compatibility requires revenue $r(\tilde{\mathbf{t}})$ to depend only on the string $\mathrm{str}(\tilde{\mathbf{t}})$ and not the segmentation. Theorem 6 further proves that, under additivity, incentive compatibility holds if and only if $r(\mathbf{t}) = \sum_{\sigma \in \Sigma} \mathrm{count}_\sigma(\mathbf{t}) \cdot r(\sigma)$ (linear character billing). Corollary 7 states that as long as a vocabulary contains multi-character tokens, pay-per-token is never incentive-compatible. To facilitate adoption, the author provides a migration formula $r_c = r_o \cdot \mathrm{tpc}$. While per-sample profit margins may fluctuate, this incentivizes providers to build better tokenizers to compress strings rather than exploiting segmentation.

### Loss & Training
This work utilizes mechanism design and theory; it does not involve training losses. Key experimental parameters include top-$p$ ($p \in \{0.90, 0.95, 0.99\}$), temperature ($T = 1.3$), iteration count $m$, and baseline profit margins $\rho_o \in \{0.2, 0.4, 0.6\}$. Cheating profitability is evaluated via the criterion $\rho(\mathbf{t}) > 1 - \mathbb{E}[\mathrm{plausible}(\hat{\mathbf{t}})] \cdot m \cdot c_o / c_v$.

## Key Experimental Results

### Main Results
Evaluated on 600 prompts from LMSYS Chatbot Arena using Llama / Gemma / Ministral models ($T=1.3$, top-$p$).

| Model / Config | $p=0.99$ Overcharge | $p=0.95$ Overcharge | $p=0.90$ Overcharge | Description |
|--------|------|------|------|------|
| Llama-3.2-1B + Alg. 2 | 11.2% | 1.8% | 0.28% | Higher $p$ allows more plausibility space for cheating |
| Llama-3.2-1B Utility Gain (any $\rho_o$) | >10.5% | 1.7% | 0.4% | Net profit remains positive after accounting for $c_v$ |
| Profitability after pay-per-char ($\rho_o = 0.2 / 0.4 / 0.6$) | — | 92.4% / 93.3% / 94.1% | — | Most samples remain profitable after migration |

### Ablation Study

| Config | Key Metric | Description |
|------|---------|------|
| Algorithm 1 (No verification) | Overcharge increases linearly with $m$ | Plausibility drops to ~0, making it easy to detect |
| Algorithm 2 vs. $m$ | Overcharge vs $m$ is unimodal | Optimal $m$ decreases as $p$ decreases due to plausibility constraints |
| Change in profit margin $\rho_o$ | Relative utility gain increases as $\rho_o$ decreases | Lower margins intensify the incentive to cheat |

### Key Findings
- Algorithm 2 and Theorem 3 confirm that transparency (disclosing next-token distributions) does not stop cheating; it merely forces it to be more sophisticated, still yielding ~10% gains.
- High $p$ values (often used in creative writing) are the most vulnerable to overcharging.
- Smaller providers with lower profit margins have stronger structural incentives to cheat, creating systemic risks in competitive markets.

## Highlights & Insights
- Proving the "longest plausible segmentation" is NP-Hard while simultaneously demonstrating that simple heuristics can still exploit the system significantly highlights that computational complexity does not guarantee economic security.
- Theorem 6 provides a necessary and sufficient characterization, establishing pay-per-character not just as a recommendation but as a mathematical necessity for incentive compatibility.
- The migration formula $r_c = r_o \cdot \mathrm{tpc}$ allows existing APIs to switch billing methods without changing models, tokenizers, or architectures, requiring only a simple dataset statistic.

## Limitations & Future Work
- Pay-per-character does not prevent providers from making models artificially verbose; this requires quality-of-service metrics (e.g., pay-for-performance).
- The assumption that providers do not falsify the next-token distribution is strong for closed-source models; TEEs or ZKPs are suggested as potential solutions.
- Experiments focused on open-source weights and the LMSYS Arena dataset; performance on proprietary models and production-scale traffic remains to be seen.
- The analysis is limited to a single user-provider pairing; future work should address multi-provider market dynamics.

## Related Work & Insights
- **vs. Saig et al. (2025)**: Both use principal-agent models, but Saig et al. address model-substitution attacks ("cheap model, expensive price") via pay-for-performance, while this work targets intra-model re-segmentation.
- **vs. Sun et al. (2025) / Cai et al. (2025)**: These works focus on auditing/detection (verifying reasoning steps or model identity). This paper focuses on mechanism design to eliminate incentives at the source.
- **vs. Ahia et al. (2023)**: They noted that BPE tokenization unfairly charges non-English users more; pay-per-character naturally resolves this cross-lingual fairness issue.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First work to rigorously characterize LLM pricing through mechanism design and prove a necessary/sufficient theorem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covered major open-source families and multi-lingual prompts, though lacks closed-source validation.
- Writing Quality: ⭐⭐⭐⭐⭐ Logical flow from modeling to exploiting incentives to the final mathematical solution is seamless.
- Value: ⭐⭐⭐⭐⭐ Highly relevant to the core commercialization of LLMs; conclusions can directly inform policy and contract terms.

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
