---
title: >-
  [Paper Note] DPO Meets PPO: Reinforced Token Optimization for RLHF
description: >-
  [ICML 2025][LLM Alignment][RLHF] This paper proposes Reinforced Token Optimization (RTO), which models RLHF as a token-level MDP (rather than a sentence-level bandit). It leverages DPO to implicitly extract token-wise reward signals and then performs policy optimization using PPO. RTO outperforms PPO by 7.5 points on AlpacaEval 2 and by 4.1 points on Arena-Hard, achieving PPO-level performance with only 1/8 of the data.
tags:
  - "ICML 2025"
  - "LLM Alignment"
  - "RLHF"
  - "DPO"
  - "PPO"
  - "token-level reward"
  - "MDP modeling"
date: 2026-05-08
content_hash: f96f8071ab179dd7
---

# DPO Meets PPO: Reinforced Token Optimization for RLHF

**Conference**: ICML 2025  
**arXiv**: [2404.18922](https://arxiv.org/abs/2404.18922)  
**Code**: [https://github.com/zkshan2002/RTO](https://github.com/zkshan2002/RTO)  
**Area**: LLM Alignment/RLHF  
**Keywords**: RLHF, DPO, PPO, token-level reward, MDP modeling

## TL;DR
This paper proposes Reinforced Token Optimization (RTO), which models RLHF as a token-level MDP (rather than a sentence-level bandit). It leverages DPO to implicitly extract token-wise reward signals and then performs policy optimization using PPO. RTO outperforms PPO by 7.5 points on AlpacaEval 2 and by 4.1 points on Arena-Hard, achieving PPO-level performance with only 1/8 of the data.

## Background & Motivation
**Background**: RLHF is a core technology for aligning LLMs. The classic workflow is "first train a sentence-level reward model, then optimize with PPO." Closed-source models like ChatGPT and Claude rely on this paradigm.

**Limitations of Prior Work**: PPO performs poorly in open-source implementations. The underlying reason is that RLHF is modeled as a bandit problem—wherein the entire sentence serves as a single action and the reward is sentence-level. However, PPO is designed for multi-step MDPs, which require a step-by-step reward. In existing implementations, the sentence-level reward is only assigned to the final token, while the remaining tokens receive zero reward (the sparse reward problem).

**Key Challenge**: PPO requires dense, step-by-step reward signals for efficient learning, yet human preference annotations are naturally sentence-level, making it difficult to directly obtain token-level feedback.

**Goal**: (1) Establish a theoretical framework for token-level MDP in RLHF; (2) Find a practical method to extract token-wise rewards from preference data; (3) Improve the performance of PPO in RLHF.

**Key Insight**: It is discovered that models trained with DPO inherently imply token-level reward information ($r^*(s_h, a_h) = \beta \log \frac{\pi_{dpo}(y_h|x, y_{1:h-1})}{\pi_{ref}(y_h|x, y_{1:h-1})}$), which can be extracted to guide PPO training.

**Core Idea**: DPO "secretly" provides token-wise rewards. By integrating these rewards into PPO training as dense reward signals (utilizing DPO for reward learning + PPO for policy optimization = RTO).

## Method

### Overall Architecture
RTO is a two-stage framework:
1. **Token-wise Reward Learning**: Train a policy $\pi_{dpo}$ on preference data using DPO to extract implicit rewards for each token.
2. **Token-wise Reward Optimization**: Combine the extracted token-level rewards from DPO with a KL regularization term and an optional sentence-level reward to perform online policy optimization using PPO.

### Key Designs

1. **MDP Modeling of RLHF (From Bandit to MDP)**:

    - Function: Remodels RLHF as an MDP $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, r, \rho, H)$.
    - State $s_h = (x, y_{1:h-1})$: Prompt + previously generated tokens.
    - Action $a_h = y_h$: Currently generated token.
    - Transition $\mathcal{P}$: Deterministic (since the LLM generation process is concatenation).
    - Reward $r(s_h, a_h)$: Token-level reward.
    - Design Motivation: Precisely capture the autoregressive nature of LLMs to achieve finer-grained reward allocation.
    - **Theoretical Advantage (Proposition 3.2)**: In a deterministic MDP, finding the optimal response with token-level rewards requires $A^{\min\{\xi+1,H\}}$ samples, whereas sentence-level rewards require $A^H$ samples (representing an exponential gap).

2. **Implicit Token-wise Reward Extraction with DPO**:

    - Function: Extracts reward signals for each token from the DPO-trained policy.
    - Key Derivation: Under an MDP with deterministic transitions, the Bradley-Terry preference model is equivalent to:
    $$\mathbb{P}(\tau^1 \succ \tau^2) = \sigma\left(\sum_{h=1}^{H} \beta \log \frac{\pi_\beta^*(a_h^1|s_h^1)}{\pi_{ref}(a_h^1|s_h^1)} - \sum_{h=1}^{H} \beta \log \frac{\pi_\beta^*(a_h^2|s_h^2)}{\pi_{ref}(a_h^2|s_h^2)}\right)$$
    - This aligns perfectly with the DPO optimization objective, making the learned DPO policy $\pi_{dpo}$ an approximation of $\pi_\beta^*$.
    - The token-wise reward is defined as: $r^*(s_h, a_h) = \beta \log \frac{\pi_{dpo}(y_h|x, y_{1:h-1})}{\pi_{ref}(y_h|x, y_{1:h-1})}$
    - This is the core insight of the paper: **"Your DPO model is secretly a token-wise reward model"**

3. **RTO Reward Function Design**:

    - Function: Combines token-level DPO reward, KL regularization, and an optional sentence-level reward into the final reward.
    - Key Formula (RTO reward, Eq 4.7):
        - For $h \leq H-1$: $r_{rto} = \beta_1 \log \frac{\pi_{dpo}(y_h|...)}{\pi_{ref}(y_h|...)} - \beta_2 \log \frac{\pi(y_h|...)}{\pi_{ref}(y_h|...)}$
        - For $h = H$: plus an additional $\beta_3 \cdot r_{MLE}(x, y_{1:H})$
    - $\beta_1$ controls the strength of the DPO reward, $\beta_2$ controls the KL regularization, and $\beta_3$ controls the sentence-level reward.
    - Design Motivation: The sentence-level reward $r_{MLE}$ prevents responses from being overly long or short, while the DPO reward acts as reward shaping to provide dense signals.

### Loss & Training
- Stage 1: Train $\pi_{dpo}$ with the standard DPO loss.
- Stage 2: Use $r_{rto}$ computed via Eq 4.7 as the reward signal for PPO, executing standard PPO updates.
- The sentence-level reward model $r_{MLE}$ can be much smaller (e.g., 1B) than the policy or DPO models, keeping the total computational cost close to standard RLHF.
- In practice, $\beta_3 = 1$, $\beta_2$ is selected following standard PPO configurations, and $\beta_1$ is set to a small value to prevent the DPO reward from dominating.

## Key Experimental Results

### Main Results

| Method | AE LC↑ | AE WR↑ | AH SC↑ | AH WR↑ |
|------|--------|--------|--------|--------|
| SFT | 13.22 | 8.58 | 9.2 | 8.9 |
| DPO | 17.40 | 12.23 | 13.2 | 13.8 |
| R-DPO | 18.34 | 12.03 | 14.2 | 14.1 |
| SimPO | 25.46 | 20.20 | 14.5 | 15.2 |
| TDPO | 20.13 | 11.97 | 13.2 | 12.3 |
| PPO | 19.47 | 12.89 | 16.2 | 15.6 |
| **RTO** | **27.00** | **22.45** | **20.3** | **21.4** |

### Ablation Study

| Configuration | AE LC↑ | AE WR↑ | AH SC↑ | Description |
|------|--------|--------|--------|------|
| RTO (token-wise) | 27.00 | 22.45 | 20.3 | Dense reward, optimal |
| Semi-RTO (sentence-wise) | 23.77 | 19.17 | 19.0 | Sentence-delimiter-level reward |
| DDPO (EoS token) | 21.09 | 13.06 | 13.1 | Sparse reward, worst |
| RS-PPO (reward shaping) | 27.52 | 21.69 | 19.2 | Validating DPO reward as shaping |

### Key Findings
- **RTO >> PPO**: AlpacaEval 2 LC (+7.5 points), Arena-Hard SC (+4.1 points).
- **Dense reward >> Sparse reward**: Token-wise is significantly better than sentence-wise, which in turn outperforms EoS-only.
- **The role of DPO reward is reward shaping**: RS-PPO (where total reward equals $r_{MLE}$ but DPO is used for shaping) achieves similar performance to RTO, indicating that the core value of the DPO reward lies in redistributing the reward signal rather than changing the total reward.
- **Exceptional data efficiency**: RTO achieves PPO-level performance using only 1/8 of the data and continues to scale—where PPO saturates early, RTO continues to improve.

## Highlights & Insights
- **Elegant Design**: Organically combines the strengths of DPO and PPO—DPO provides token-level reward signals, while PPO provides online policy optimization capabilities.
- **Solid Theory**: From the sample complexity analysis of MDP vs. Bandit to the suboptimality bounds of RTO, the theoretical framework is comprehensive and of practical significance.
- **Brilliant Core Insight**: The observation that "the DPO model implicitly contains token-wise rewards" is intuitive in hindsight, but systematizing it for PPO training is a major contribution.
- The data efficiency advantage implies that RTO can function effectively in data-scarce scenarios, broadening its applicability.

## Limitations & Future Work
- The DPO model as an approximation of $\pi_\beta^*$ may be inaccurate, particularly in out-of-distribution regions.
- Training an additional DPO model is still required, which increases the total computational overhead (although the paper claims the overall cost is comparable).
- The experiments are primarily validated on Llama-3-8B; performance on larger models remains to be confirmed.
- Future exploration directions: Replace DPO with other direct preference learning algorithms (such as IPO, KTO) to extract token-level rewards.

## Related Work & Insights
- A concurrent work with Rafailov et al. (2024) shares the core insight of "DPO implicitly containing token-level rewards," but differs in application (this work applies it to PPO, while they apply it to search).
- Token-level DPO (TDPO) and SimPO fall under the line of direct preference learning improvements, whereas RTO belongs to the PPO improvement trajectory.
- Lays the foundation for a new hybrid "DPO + PPO" paradigm in RLHF.
- Subsequent works (Cui et al., 2025; Yin et al., 2025) apply RTO to reasoning and chat tasks, demonstrating the broad applicability of this method.

## Rating
- Novelty: ⭐⭐⭐⭐⭐
- Experimental Thoroughness: ⭐⭐⭐⭐⭐
- Writing Quality: ⭐⭐⭐⭐⭐
- Value: ⭐⭐⭐⭐⭐

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] TGDPO: Harnessing Token-Level Reward Guidance for Enhancing Direct Preference Optimization](tgdpo_harnessing_token-level_reward_guidance_for_enhancing_direct_preference_opt.md)
- [\[ICLR 2026\] Token-Importance Guided Direct Preference Optimization (TI-DPO)](../../ICLR2026/llm_alignment/token-importance_guided_direct_preference_optimization.md)
- [\[ACL 2025\] T-REG: Preference Optimization with Token-Level Reward Regularization](../../ACL2025/llm_alignment/t-reg_preference_optimization_with_token-level_reward_regularization.md)
- [\[ACL 2025\] Optimal Transport-Based Token Weighting for Enhanced Preference Optimization](../../ACL2025/llm_alignment/otpo_token_weighting.md)
- [\[ACL 2025\] Aligning to What? Limits to RLHF Based Alignment](../../ACL2025/llm_alignment/aligning_to_what_limits_to_rlhf_based_alignment.md)

</div>

<!-- RELATED:END -->
