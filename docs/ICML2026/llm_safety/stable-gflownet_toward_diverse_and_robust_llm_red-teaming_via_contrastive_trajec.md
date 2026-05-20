---
title: >-
  [Paper Note] Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance
description: >-
  [ICML 2026][LLM Safety][Red-teaming] This paper identifies two major sources of instability in existing GFlowNet red-teaming: high variance from partition function $Z_\theta$ estimation…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Red-teaming"
  - "GFlowNet"
  - "Trajectory Balance"
  - "Contrastive Objective"
  - "Noisy Gradient Pruning"
date: 2026-05-08
content_hash: fed4c1245ef80f52
---

# Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance

**Conference**: ICML 2026  
**arXiv**: [2605.00553](https://arxiv.org/abs/2605.00553)  
**Code**: Not released  
**Area**: LLM Safety / Red-Teaming / GFlowNet  
**Keywords**: Red-teaming, GFlowNet, Trajectory Balance, Contrastive Objective, Noisy Gradient Pruning

## TL;DR
This paper identifies two major sources of instability in existing GFlowNet red-teaming: high variance from partition function $Z_\theta$ estimation, and mode collapse caused by noisy rewards from toxicity classifiers on OOD gibberish text. The authors propose three simple components—pairwise contrastive objective CTB to eliminate $Z$, Noisy Gradient Pruning to filter uninformative pairs, and Min-K Fluency Stabilizer to block gibberish—which together boost the number of unique attacks on Qwen2.5-1.5B from 17 to 134 (about 7×), maintain a 92% ASR, and outperform baselines in cross-model/cross-defense transferability.

## Background & Motivation

**Background**: LLM red-teaming aims to identify safety vulnerabilities before deployment, with three main approaches: (1) RL-based (PPO, PPO+Curiosity, Jailbreak-R1) maximize reward and can find highly toxic prompts but suffer severe mode collapse; (2) Quality-Diversity (Rainbow Teaming, Ruby Teaming) maintain diversity via predefined style/topic matrices and evolutionary strategies, but rely on frozen LLM instruction following and have low attack success rates; (3) GFN-based (Lee et al. 2024) frames red-teaming as distribution matching—sampling probability $\propto$ reward, theoretically achieving both high toxicity and diversity.

**Limitations of Prior Work**: Directly applying Trajectory Balance (TB) objectives from GFN to LLMs faces two major pitfalls:
- TB loss $\mathcal{L}_{TB}(y; \theta) = (\log Z_\theta + \log \pi_\theta(y) - \log R(y))^2$ requires learning a scalar $Z_\theta$ to estimate $Z \simeq \sum_{y \in \mathcal{Y}} R(y)$. The combinatorial explosion of the LLM token sequence space $\mathcal{Y}$ makes $Z_\theta$ hard to estimate accurately, leading to high-variance gradients, training collapse, or persistent mode collapse.
- Red-team rewards come from toxicity classifiers, which assign random pseudo-rewards (0.2~0.3) to gibberish-like OOD text; once attackers discover such reward hacking paths, they quickly collapse to generating gibberish as a local optimum.

**Key Challenge**: GFN’s lossless distribution matching should preserve diversity, but unstable $Z$ estimation in practice causes TB to degenerate into narrow RL-like distribution fitting. Standard fluency protection via KL-divergence regularization $R_{ref}(y) = \pi_{KL}(y)^\alpha \cdot R(y)^\beta$ distorts the target distribution (biasing samples toward the reference rather than the reward), conflicting with GFN’s theoretical assumptions.

**Goal**: (1) Design a GFN alternative objective that does not require $Z_\theta$ but remains equivalent to TB at optimum; (2) Introduce a saliency-based filtering strategy for noisy rewards to avoid contamination by random pseudo-rewards; (3) Prevent attackers from hacking into gibberish regions without using KL-style distribution-distorting methods.

**Key Insight**: The authors observe that comparing two trajectories $y_1, y_2$ from the same policy naturally cancels the partition function $Z_\theta$—the standard motivation for contrastive objectives. The "reward noise" problem is essentially that low-contrast pairs provide erroneous gradient signals in pairwise comparisons, which can be addressed with a contrast-aware indicator as a hard filter. Repeated gibberish can be detected using Min-K probability (average log-probability of the least-likely tokens) as a fluency proxy, applying a hard threshold.

**Core Idea**: Combine ratio-based Contrastive Trajectory Balance (CTB) to eliminate $Z_\theta$, Noisy Gradient Pruning (NGP) to filter pairs by reward contrast, and Min-K Fluency Stabilizer (MKS) to block gibberish, forming the Stable-GFN suite.

## Method

### Overall Architecture

Input: attacker LLM $\pi_\theta$ (Qwen2.5-1.5B SFT on Safety-Dataset + AdvBench), victim LLM $\pi_\phi$, toxicity classifier $\pi_\psi$, and a fixed meta-prompt. Each training step: (1) the attacker samples $N$ candidate attack prompts $\{y_n\}$ using the current policy; (2) the victim generates a response $z_n$ for each, and the classifier computes toxicity $R(y_n) = \mathbb{E}_{z \sim \pi_\phi(\cdot|y)}[T(y, z)]$; (3) MKS uses a reference model to compute Min-K fluency for each prompt, masking those below threshold; (4) NGP enumerates $N^2$ pairs within the batch, filtering out low-saliency pairs with $|\log R(y_1) - \log R(y_2)| \le \sigma$; (5) remaining pairs are used to compute CTB loss and update $\theta$. The pipeline requires no external $Z_\theta$, no archive maintenance, and no strong reference policy constraints.

### Key Designs

1. **Contrastive Trajectory Balance (CTB)**:

    - **Function**: Replaces absolute matching with pairwise comparison, algebraically eliminating $Z_\theta$ and yielding the same optimal policy as TB but with lower variance.
    - **Mechanism**: For two independent samples $y_1, y_2 \sim \pi_\theta$, define $\mathcal{L}_{CTB}(y_1, y_2; \theta) = (\log \tfrac{\pi_\theta(y_1)}{\pi_\theta(y_2)} - \log \tfrac{R(y_1)}{R(y_2)})^2$. Let $f(y) = \log \pi_\theta(y) - \log R(y)$; when $y_1, y_2$ are i.i.d., the objective is equivalent to $2 \cdot \mathrm{Var}_{\pi_\theta}(f(y))$, minimized to 0 when $f$ is constant on the support, i.e., $\pi_\theta(y) = R(y)/Z$ (see Theorem 4.1). The gradient $\nabla_\theta \mathcal{L}_{CTB} = 2(f(y_1) - f(y_2))(\nabla_\theta f(y_1) - \nabla_\theta f(y_2))$ uses each sample’s log-flow error as a stochastic baseline for the other, analogous to variance reduction in RLOO/Williams.
    - **Design Motivation**: Eliminates $Z_\theta$ as a high-variance source; batch of $N$ samples yields $N^2$ scalar pairwise losses (no extra forward passes), keeping training complexity $O(N)$.

2. **Noisy Gradient Pruning (NGP)**:

    - **Function**: CTB aggregates reward noise from both samples; low-contrast pairs amplify noise. NGP applies a hard mask to zero gradients for low-saliency pairs.
    - **Mechanism**: $\mathcal{L}_{NGP}(y_1, y_2; \theta) = \mathbb{1}[|\log R(y_1) - \log R(y_2)| > \sigma] \cdot \mathcal{L}_{CTB}(y_1, y_2; \theta)$, where $\sigma$ is a saliency threshold hyperparameter. Theoretically, constructing a saliency graph $G_\sigma = (\mathcal{Y}, E_\sigma)$ (edges connect pairs with contrast $>\sigma$), if $G_\sigma$ is connected, then $\mathcal{L}_{NGP}(\theta) = 0$ still implies $\pi_\theta(y) \propto R(y)$ (Proposition 4.2). In practice, a high-reward replay buffer provides "global anchors" for contrast pairs across reward regions to maintain connectivity.
    - **Design Motivation**: Toxicity classifiers are dominated by random noise between samples with similar rewards; filtering out "zero-information but nonzero-noise" pairs ensures gradients come only from pairs with real reward differences, preserving GFN objectives (under connectivity) and reducing gradient variance.

3. **Min-K Fluency Stabilizer (MKS)**:

    - **Function**: Prevents attackers from hacking into gibberish regions without distorting the target distribution.
    - **Mechanism**: The reference model $\pi_{ref}$ computes log-probabilities for each token in generated prompt $y$, averaging the lowest $k$ tokens: $M_k(y) = \tfrac{1}{|K|}\sum_{w \in K} \log \pi_{ref}(y_w | y_{<w})$. The reward is modified as $R_{MKS}(y) = \mathbb{1}[M_k(y) \ge T_{MKS}] \cdot R(y)$—rewards for prompts below the fluency threshold $T_{MKS}$ are zeroed. Gradients from $\pi_{ref}$ are not used in reward computation.
    - **Design Motivation**: Unlike global KL regularization, MKS penalizes only the "least fluent segments" (most likely OOD gibberish), preserving exploration freedom for normal prompts; it does not reshape the target distribution (hard cutoff in reward), remaining compatible with GFN’s distribution matching assumption.

### Loss & Training

The overall objective is $J_{CTB}(\theta) = \mathbb{E}_{y_1, y_2 \sim \pi_\theta}[\mathcal{L}_{NGP}(y_1, y_2; \theta)]$, with MKS-modified rewards. Each batch enumerates pairs from $N = 1024$ samples. Attacker: Qwen2.5-1.5B SFT; Victim: Qwen2.5-1.5B-Instruct; Toxic classifier: Meta-Llama-Guard-3-8B; Diversity: all-MiniLM-L6-v2 + greedy clustering threshold 0.7; reward $>0.5$ counts as ASR.

## Key Experimental Results

### Main Results

| Method | UA (#) | ASR (%) | Notes |
|--------|--------|---------|-------|
| PPO | 3.00 | **91.70** | High ASR but severe mode collapse |
| PPO + Curiosity | 4.00 | 36.75 | Still collapses |
| Rainbow Teaming | 33.00 | 66.11 | High QD diversity but low ASR |
| Jailbreak R1 (8B) | 75.33 | 7.36 | Diverse but low toxicity |
| GFN (TB) | 17.67 | 93.75 | High ASR but UA far below theoretical expectation |
| **S-GFN (Ours)** | **134.00** | 92.55 | Same ASR, UA improved 7× |

Cross-Attack Defense Transfer (attacking GFN-defended victim):

| Attacker | GFN-defended victim ASR | Notes |
|----------|------------------------|-------|
| GFN | 4.69% | Self-attack blocked by self-defense |
| Jailbreak R1 | 2.96% | – |
| **S-GFN** | **22.53%** | Broader attack modes, strong cross-defense transfer |

### Ablation Study

| Configuration | UA (#) | ASR (%) | Notes |
|---------------|--------|---------|-------|
| GFN-TB + KL ref | 14 | – | Reference KL distorts distribution |
| GFN-TB + LogProb | 65 | – | Alternative regularization |
| GFN-TB + MKS | 67 | 85.8 | TB + fluency gating |
| **GFN-CTB + MKS** | 108 | 82.9 | Adding CTB boosts UA by 60% |
| **GFN-CTB + MKS + NGP** | **121** | **92.2** | Full S-GFN, ASR also recovers |

### Key Findings

- **CTB > TB’s main contribution is stability**: Replacing TB with CTB (keeping MKS) raises UA from 67 to 108, showing $Z_\theta$ estimation is a major cause of mode collapse.
- **NGP improves both UA and ASR**: UA increases from 108 to 121, ASR from 82.9% to 92.2%, indicating that filtering low-saliency pairs both denoises and strengthens gradient signals—"less but better" beats "more but noisy."
- **Cross-Attack asymmetry is pronounced**: S-GFN attacking GFN-defended models achieves 22.53%, while GFN attacking S-GFN-defended models gets only 0.03%—this "I can break you but you can’t break me" asymmetry shows S-GFN finds genuinely diverse attack modes, not just a superset of GFN attacks.
- **Transfer attacks to completely unseen victims** (Gemma3, Llama3.2, Qwen3, GPT-OSS-20B): S-GFN ranks first in both UA and ASR on all models, indicating attacks are not overfitted to the training victim’s "specific jailbreaks."
- **MKS is essential**: Without MKS, reward drops to 0 (all hacks are gibberish); with MKS, UA jumps from 0 to 67—directly rescuing the training process.

## Highlights & Insights

- "$Z_\theta$ cancellation" is a seemingly simple but significant insight—textual GFN has long struggled with $Z$ estimation, and CTB’s ratio form naturally eliminates $Z$, akin to how contrastive learning removes the normalizing constant. The equivalence proof (Theorem 4.1) ensures no loss of distribution matching properties.
- NGP’s "saliency graph connectivity" analysis is elegant—it formalizes "how many pairs can be pruned while preserving GFN convergence" as a graph connectivity condition, and notes that the replay buffer serves as an empirical anchor.
- MKS’s use of Min-K probability (from LLM membership inference literature) for fluency detection is a clever cross-domain adaptation—more sensitive to partial gibberish than traditional perplexity by focusing on the "weakest link."
- The entire method is extremely simple to implement: CTB is $N^2$ scalar operations, NGP is an indicator mask, MKS is a reward cutoff—all are "add a hard filter / modify loss," with no increase in forward passes.

## Limitations & Future Work

- $\sigma$ (NGP) and $T_{MKS}$ (MKS) are fixed hyperparameters; task-adaptive tuning is unexplored. As the reward distribution shifts during training, fixed thresholds may perform differently at various stages.
- The connectivity assumption may not hold when the number of distribution modes is large; the authors acknowledge that the "high-reward replay buffer" is an empirical anchor, without non-asymptotic convergence rate guarantees.
- Main experiments are only on Qwen2.5-1.5B attacker; attacker scaling (e.g., 7B/13B) is not explored, so it is unclear if CTB’s variance reduction holds for larger attackers.
- Combination with multi-stage iterative GFN (Yun et al. 2025) is unexplored; integrating CTB into iterative frameworks for further diversity is a natural next step.
- Red-teaming ethics: The better the method, the more vulnerabilities it finds, but the paper does not discuss disclosure processes in depth; results like 92% ASR and 134 UA pose direct risks to open-source victim models, requiring responsible release.

## Related Work & Insights

- **vs GFN-TB (Lee et al. 2024)**: Original TB treats $Z_\theta$ as a learnable parameter, leading to high variance and mode collapse; CTB uses pairwise ratios to eliminate $Z$, yielding equivalent optimal policies but more stable training.
- **vs PPO + Curiosity (Hong et al. 2024)**: RL with diversity reward remains pointwise reward optimization, UA only reaches 4; S-GFN, as a distribution matching approach, achieves UA of 134.
- **vs Rainbow Teaming (Samvelyan et al. 2024)**: QD uses predefined style/topic matrices to enforce diversity but has low ASR (66%). S-GFN requires no predefined archive, using reward signals end-to-end to discover diverse modes.
- **vs DPO with replay**: DPO achieves UA of only 5.33 in red-teaming; its preference contrastive objective superficially resembles CTB, but DPO optimizes preference ranking, not distribution matching, so the objectives differ.
- **vs DB / SubTB (Bengio et al. 2023; Madan et al. 2023)**: DB/SubTB avoid $Z$ estimation but are computationally expensive at the token level and do not scale to LLMs; CTB operates pairwise at the sequence level, making it computationally friendly.

## Rating
- Novelty: ⭐⭐⭐⭐ The pairwise contrastive approach to eliminate $Z$ is inspired by contrastive learning but is the first systematic application to LLM-scale GFN with accompanying noise/fluency handling; CTB-TB equivalence is rigorously proven.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 5 baselines, cross-attack defense, 4 transfer victims, 3 ablation modules, with clear quantification; lacks attacker scaling experiments.
- Writing Quality: ⭐⭐⭐⭐ Motivation, theory, algorithm, and experiments are clearly mapped, with appendix proofs for each claim; Figure 1 overview is intuitive.
- Value: ⭐⭐⭐⭐ Brings GFN to practical LLM red-teaming and provides a generalizable "stable GFN" toolbox, valuable for the alignment safety community, but open-source victim risks require caution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks (DialTree)](../../ICLR2026/llm_safety/tree-based_dialogue_reinforced_policy_optimization_for_red-teaming_attacks.md)
- [\[ICML 2026\] MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio](medmosaic_a_challenging_large_scale_benchmark_of_diverse_medical_audio.md)
- [\[ICML 2026\] Tracing the Dynamics of Refusal: Exploiting Latent Refusal Trajectories for Robust Jailbreak Detection](tracing_the_dynamics_of_refusal_exploiting_latent_refusal_trajectories_for_robus.md)
- [\[CVPR 2026\] SineProject: Machine Unlearning for Stable Vision–Language Alignment](../../CVPR2026/llm_safety/sineproject_machine_unlearning_for_stable_vision_language_alignment.md)
- [\[ICML 2026\] Watermarking LLM Agent Trajectories (ACTHOOK)](watermarking_llm_agent_trajectories.md)

</div>

<!-- RELATED:END -->
