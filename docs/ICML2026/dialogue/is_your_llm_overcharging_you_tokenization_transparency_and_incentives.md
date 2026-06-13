---
title: >-
  [Paper Note] Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives
description: >-
  [ICML 2026][Dialogue Systems][pay-per-token] This paper models LLM-as-a-Service as a "principal-agent" problem, proving that current mainstream "pay-per-token" mechanisms naturally incentivize service providers to re-tok…
tags:
  - "ICML 2026"
  - "Dialogue Systems"
  - "pay-per-token"
  - "incentive compatibility"
  - "tokenization multiplicity"
  - "pay-per-character"
  - "principal-agent"
date: 2026-05-08
content_hash: 18aca63112e16e42
---

# Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives

**Conference**: ICML 2026 Oral  
**arXiv**: [2505.21627](https://arxiv.org/abs/2505.21627)  
**Code**: https://github.com/Human-Centric-Machine-Learning/token-pricing (Available)  
**Area**: AI Safety / Mechanism Design / LLM-as-a-Service Pricing  
**Keywords**: pay-per-token, incentive compatibility, tokenization multiplicity, pay-per-character, principal-agent

## TL;DR
This paper models LLM-as-a-Service as a "principal-agent" problem, proving that current mainstream "pay-per-token" mechanisms naturally incentivize service providers to re-tokenize the same string into longer token sequences to overcharge users. Furthermore, even if providers are forced to disclose the next-token distribution, overcharging without being detected remains NP-Hard rather than impossible—the authors provide a simple heuristic algorithm that increases billed tokens by up to 11.2% while maintaining plausibility. Finally, the authors prove that the only additive pricing mechanism that eliminates this incentive is "linear pay-per-character billing."

## Background & Motivation
**Background**: Cloud LLM services (OpenAI / Gemini / Anthropic, etc.) almost exclusively adopt pay-per-token billing: users submit a prompt, providers run the model on their hardware to generate output, and charge based on the number of tokens returned multiplied by a unit price. Users only see the returned string and the claimed token count; the internal vocabulary, actual segmentation, and next-token distribution remain on the provider's side.

**Limitations of Prior Work**: Tokenization is not unique. The same string "Damascus" can be split into `|Dam|ascus|` (2 tokens) or `|Da|ma|s|cus|` (4 tokens), and users are completely unaware of this. Providers can easily "re-report" a truthfully generated 2-token sequence as a 4-token sequence to double the charge without changing the string itself, and users have no technical means to detect it.

**Key Challenge**: Moral hazard caused by information asymmetry—the provider fully observes the generation process, while the user only observes and pays for the reported token sequence. As long as "pay-per-token" exists and multi-character tokens are present in the vocabulary, mathematically replacing shorter tokenizations with longer ones strictly increases revenue.

**Goal**: Decomposed into three sub-problems: (1) Do providers have structural incentives to lie under pay-per-token? (2) If providers are forced to disclose next-token distributions, can users block lying by verifying whether the tokenization is "plausible" under the model? (3) Does an incentive-compatible pricing mechanism exist that eliminates this incentive in principle?

**Key Insight**: The authors model the problem using the principal-agent framework from contract theory—treating the user as the principal, the provider as the agent, and the billing rule as the contract. They then systematically characterize "incentive-compatibility" (IC): a property where truthful reporting is never worse for the provider than lying. This is a paradigm repeatedly used in auctions, mechanism design, and insurance, here applied to LLM pricing for the first time.

**Core Idea**: Token length billing is fundamentally not incentive-compatible. The only additive and incentive-compatible method is linear pay-per-character billing. For a smooth transition, providers only need to set $r_c = r_o \cdot \mathrm{tpc}$ (where tpc is the average tokens per character), allowing the average profit margin to remain unchanged.

## Method

### Overall Architecture
The "method" of the paper constitutes a complete argument chain: (i) formalizing LLM-as-a-Service using a principal-agent model; (ii) analyzing lying incentives under pay-per-token and constructing two specific reporting policies (one with no GPU cost but easily detected, and one verified via GPU with higher plausibility) to prove that lying is feasible in reality; (iii) deriving the necessary and sufficient characterization of incentive-compatible pricing mechanisms, proving pay-per-character is the only feasible solution, and providing the tpc formula for painless transition.

The input is a user prompt $q$; the provider feeds it into the LLM to obtain the true token sequence $\mathbf{t}$ (string denoted as $s = \mathrm{str}(\mathbf{t})$), and then uses a reporting policy $\pi$ to report $\tilde{\mathbf{t}} \sim \pi(\mathbf{t})$, constrained by $\mathrm{str}(\tilde{\mathbf{t}}) = s$ (i.e., the text seen by the user remains unchanged). Provider utility $U_\pi(\tilde{\mathbf{t}}, \mathbf{t}) = r(\tilde{\mathbf{t}}) - c_\text{gen}(\mathbf{t}) - c_\pi(\mathbf{t})$, where $c_\text{gen}(\mathbf{t}) \approx c_o \cdot \mathrm{len}(\mathbf{t})$ is the GPU generation cost, and $c_\pi$ is the cost of executing the reporting policy itself ($c_\pi = 0$ for unverified policies, while policies requiring forward pass verification have $c_\pi = c_v$ as a constant).

### Key Designs

1.  **Formalization of Lying Incentives and Unverified Heuristic (Algorithm 1)**:
    - **Function**: Prove that "longer reporting leads to higher utility" under pay-per-token and provide a simple baseline lying strategy $\pi_m^R$.
    - **Mechanism**: Under pay-per-token, $r(\tilde{\mathbf{t}}) = r_o \cdot \mathrm{len}(\tilde{\mathbf{t}})$. Thus, for any two policies $\pi, \pi'$ with the same cost, as long as $\mathrm{len}(\tilde{\mathbf{t}}) > \mathrm{len}(\tilde{\mathbf{t}}')$, then $U_\pi > U_{\pi'}$—longer sequences yield higher profits. Algorithm 1 provides a zero-computation implementation: maintain the current sequence $\tilde{\mathbf{t}}$, and in each round, randomly pick a token that can be further split into two non-empty tokens and split it. This is repeated $m$ times or until all tokens are single characters. The process requires no GPU as it does not verify plausibility.
    - **Design Motivation**: Use a simple algorithm to puncture the illusion that "pay-per-token is fine"—experiments show that in competitive scenarios, a cheating provider can拉长 the sequence by $1/\alpha$ times to achieve the same revenue even if their per-token price is $\alpha$ times lower than competitors, using "lying" as a weapon to capture market share.

2.  **Plausible Heuristic Lying Algorithm (Algorithm 2) and the NP-Hard Barrier**:
    - **Function**: Find token sequences that look reasonable (plausible) yet are longer than the truth in "transparent" scenarios where providers are required to disclose next-token distributions (top-$p$ / top-$k$ sampling).
    - **Mechanism**: The paper first proves that "finding the longest plausible tokenization" is NP-Hard (Theorem 3, reduced from Hamiltonian Path), so providers cannot cheat optimally in polynomial time. However, the authors observe an empirical pattern—alternative tokenizations that are "close" to the most frequent BPE tokenizations are almost always plausible under actual models. Algorithm 2 exploits the property that "high ID $\approx$ longer token" in BPE. In each round, it picks the token $t_i$ with the largest ID and splits it into $(t_1', t_2')$ such that $\min(\mathrm{id}(t_1'), \mathrm{id}(t_2'))$ is maximized (max-min heuristic). After $m$ splits, a forward pass is run to verify if $\hat{t}_i \in \mathcal{V}_p(\hat{\mathbf{t}}_{\leq i-1})$ holds for all $i$. If plausible, it reports the sequence; otherwise, it reverts to the truth. The profit criterion is $\mathbb{E}[\mathrm{plausible}(\hat{\mathbf{t}})] \cdot m \cdot r_o > c_v$, i.e., "extra revenue when plausible > cost of one verification."
    - **Design Motivation**: To answer whether "NP-Hardness protects the user." The authors answer: No. The energy cost of one forward pass $c_v$ is a constant independent of the sequence, while every extra token split earns an additional $r_o$. Under mainstream profit margins, Algorithm 2 on Llama-3.2-1B with $p=0.99$ can achieve 10.5%+ excess revenue with positive total utility.

3.  **Characterization of Incentive-Compatible Pricing + Smooth Transition Formula**:
    - **Function**: Characterize all additive and incentive-compatible pricing mechanisms and provide a prescription for maintaining average profit margins when switching from pay-per-token to pay-per-character.
    - **Mechanism**: Proposition 5 is proven first—incentive compatibility implies $r(\tilde{\mathbf{t}})$ depends only on the string $\mathrm{str}(\tilde{\mathbf{t}})$ and not the specific tokenization (otherwise, the provider always picks the most expensive tokenization). Theorem 6 proves that a mechanism is additive + incentive-compatible if and only if $r(\mathbf{t}) = \sum_{\sigma \in \Sigma} \mathrm{count}_\sigma(\mathbf{t}) \cdot r(\sigma)$, i.e., linear billing by character; if every character has the same price $r_c$, then $r(\mathbf{t}) = |\mathrm{str}(\mathbf{t})| \cdot r_c$ is the only choice. Corollary 7 asserts that pay-per-token is never incentive-compatible when the vocabulary contains multi-character tokens. Transition formula: $r_c = r_o \cdot \mathrm{tpc}$, where tpc is the sample mean of the "token-to-character ratio" on a dataset, which keeps the average profit margin constant.
    - **Design Motivation**: Compress the engineering question of "what billing to use" into a mathematical theorem—not as a suggestion, but as a necessary and sufficient condition. Meanwhile, it acknowledges that pay-per-character will cause profit margin fluctuations for individual samples (because cost is linear to tokens while revenue is linear to characters), but this provides a healthy incentive: inventing better tokenizers or models to compress characters more efficiently.

### Loss & Training
This is a mechanism design / theory + empirical paper; there is no training loss. Key hyperparameters: top-$p$ sampling $p \in \{0.90, 0.95, 0.99\}$, temperature $T = 1.3$, iteration count $m$ for Algorithm 2 (experiments show a unimodal optimal $m$ for each $p$), and profit margin $\rho_o \in \{0.2, 0.4, 0.6\}$. The profitability criterion $\rho(\mathbf{t}) > 1 - \mathbb{E}[\mathrm{plausible}(\hat{\mathbf{t}})] \cdot m \cdot c_o / c_v$ formalizes "whether lying pays off" as a ratio of profit margin to verification cost, which can be directly calculated against real service energy bills.

### Formal Definitions
- **Additivity**: $r(\tilde{\mathbf{t}}) = \sum_i r(\tilde{t}_i)$, where pay-per-token is the simplest additive mechanism.
- **Incentive Compatibility (Definition 4)**: For any $\mathbf{t}, \pi, \tilde{\mathbf{t}} \sim \pi(\mathbf{t})$, $U_{\pi_0}(\mathbf{t}, \mathbf{t}) \geq U_\pi(\tilde{\mathbf{t}}, \mathbf{t})$, meaning the truthful reporting policy $\pi_0$ is weakly dominant.
- **Plausibility under top-$p$**: At each step $\hat{t}_i \in \mathcal{V}_p(\hat{\mathbf{t}}_{\leq i-1})$, where $\mathcal{V}_p$ is the smallest candidate set with cumulative probability $\geq p$.

## Key Experimental Results

### Main Results
Evaluated on 600 prompts from LMSYS Chatbot Arena using Llama / Gemma / Ministral series models with $T=1.3$ and top-$p$ sampling.

| Model / Config | $p=0.99$ Overcharging | $p=0.95$ Overcharging | $p=0.90$ Overcharging | Description |
|--------|------|------|------|------|
| Llama-3.2-1B + Alg. 2 | 11.2% | 1.8% | 0.28% | Larger $p$ and wider sampling space relax plausibility constraints, increasing cheating potential. |
| Llama-3.2-1B Utility Gain (any $\rho_o$) | >10.5% | 1.7% | 0.4% | Net profit remains even after subtracting verification forward pass costs. |
| Margin > 0 ratio after pay-per-char ($\rho_o = 0.2 / 0.4 / 0.6$) | — | 92.4% / 93.3% / 94.1% | — | Most samples remain profitable after transition. |

### Ablation Study
| Configuration | Key Metrics | Description |
|------|---------|------|
| Algorithm 1 (Unverified) | Overcharging increases linearly with $m$, but plausibility collapses to ~0 | Proves that unverified lying is easily detected via back-checking. |
| Algorithm 2 vs. $m$ | Overcharging vs. $m$ is unimodal; optimal $m$ decreases as $p$ decreases | If $m$ is too large, plausibility drops too fast, lowering expected revenue. |
| Margin $\rho_o$ variations | Smaller $\rho_o$ leads to larger relative utility gain | Cheating rewards are relatively amplified in price wars, making incentives even stronger. |

### Key Findings
- Key Design 2 contributes most: Algorithm 2 + Theorem 3 together answer "whether transparency blocks cheating," concluding "theoretically half-blocked (NP-Hard), practically not (heuristics still earn 10%)."
- Larger $p$ values increase cheating potential, but $p$ is also the diversity knob for real generation—this identifies "high-temperature/high-$p$" creative writing as the most vulnerable scenarios.
- Lower profit margins create higher relative returns for lying, implying small providers with low margins have the strongest incentives to cheat; this risk compounds with the reality that small providers have lower reputation costs.

## Highlights & Insights
- Using Hamiltonian Path reduction to categorize "longest plausible tokenization" as NP-Hard, then immediately using the max-min ID heuristic to debunk the idea that "NP-Hardness implies safety"—the combination of theory and empirics is very clear, proving that algorithmic complexity does not equal economic security.
- Theorem 6 compresses the question of "what billing eliminates lying incentives" into a necessary and sufficient characterization. Pay-per-character is not just one of many candidates but the unique solution (under equal-priced characters). This "inevitability" narrative is very powerful for policy-making.
- The transition formula $r_c = r_o \cdot \mathrm{tpc}$ can be directly applied to any existing API without changing the model, tokenizer, or architecture. The transition cost is merely a dataset statistical analysis, which is highly engineering-friendly.

## Limitations & Future Work
- Ours acknowledges: pay-per-character cannot prevent providers from making models artificially verbose to output more characters; this attack requires quality-metric-based mechanisms (e.g., pay-for-performance in Saig et al. 2025).
- It assumes providers cannot fake the next-token distribution itself or swap tokenizers—this assumption is strong for closed-source models. The authors suggest using trusted execution environments / zero-knowledge proofs.
- Experiments were only conducted on open-source models (Llama / Gemma / Ministral) and prompts from LMSYS Chatbot Arena, which has been criticized for lack of representativeness; performance on closed-source models and real production traffic remains open.
- The analysis is limited to the micro level of single user-single provider; macro market dynamics like multi-provider competition and user feedback are left for future work.

## Related Work & Insights
- **vs. Saig et al. (2025)**: They also use principal-agent models but target "model swapping" (using a cheap model but billing for a premium one), proposing pay-for-performance. Ours targets "re-tokenization within the same model"; the two are orthogonal and complementary.
- **vs. Sun et al. (2025) / Cai et al. (2025)**: These are audit/detection-oriented (verifying if reasoning steps are padded or if the model was swapped). Ours is mechanism design-oriented (directly changing billing rules to eliminate incentives).
- **vs. Ahia et al. (2023)**: They pointed out that different languages have vastly different token counts under BPE, causing non-English users to be overcharged under pay-per-token. The pay-per-character mechanism in Ours naturally fixes this fairness issue.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The first work to seriously characterize LLM pricing as a mechanism design problem and provide a necessary and sufficient theorem.
- Experimental Thoroughness: ⭐⭐⭐⭐ Three families of open-source models + 600 multilingual prompts cover major cases, but lacks validation on closed-source models and production-grade traffic.
- Writing Quality: ⭐⭐⭐⭐⭐ A logical argument chain: modeling → exposing incentives → heuristic empirics → inevitability theorem → smooth transition recipe.
- Value: ⭐⭐⭐⭐⭐ Directly addresses the core pricing rules of LLM commercialization. The conclusions can directly influence regulations and contract terms, carrying significant impact.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Your Students Don't Use LLMs Like You Wish They Did](../../ACL2026/dialogue/your_students_dont_use_llms_like_you_wish_they_did.md)
- [\[ICML 2026\] Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving](not_all_prefills_are_equal_ppd_disaggregation_for_multi-turn_llm_serving.md)
- [\[ACL 2026\] Cognitive Policy-Driven LLM for Diagnosis and Intervention of Cognitive Distortions in Emotional Support Conversation](../../ACL2026/dialogue/cognitive_policy-driven_llm_for_diagnosis_and_intervention_of_cognitive_distorti.md)
- [\[ACL 2026\] Codebook-Injected Dialogue Segmentation for Multi-Utterance Constructs Annotation: LLM-Assisted and Gold-Label-Free Evaluation](../../ACL2026/dialogue/codebook-injected_dialogue_segmentation_for_multi-utterance_constructs_annotatio.md)
- [\[ICML 2026\] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents](from_self-evolving_synthetic_data_to_verifiable-reward_rl_post-training_multi-tu.md)

</div>

<!-- RELATED:END -->
