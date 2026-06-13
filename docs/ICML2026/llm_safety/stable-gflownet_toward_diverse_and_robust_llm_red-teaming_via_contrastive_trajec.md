---
title: >-
  [Paper Note] Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance
description: >-
  [ICML 2026][LLM Safety][Red-Teaming] This paper identifies two major sources of instability in existing GFlowNet red-teaming: high variance introduced by the estimation of the partition function $Z_\theta$ and mode colla…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Red-Teaming"
  - "GFlowNet"
  - "Trajectory Balance"
  - "Contrastive Objective"
  - "Noisy Gradient Pruning"
date: 2026-05-08
content_hash: 0715778cbb153a7c
---

# Stable-GFlowNet: Toward Diverse and Robust LLM Red-Teaming via Contrastive Trajectory Balance

**Conference**: ICML 2026  
**arXiv**: [2605.00553](https://arxiv.org/abs/2605.00553)  
**Code**: Paper link not public  
**Area**: LLM Safety / Red-Teaming / GFlowNet  
**Keywords**: Red-Teaming, GFlowNet, Trajectory Balance, Contrastive Objective, Noisy Gradient Pruning

## TL;DR
This paper identifies two major sources of instability in existing GFlowNet red-teaming: high variance introduced by the estimation of the partition function $Z_\theta$ and mode collapse caused by noisy rewards assigned to OOD gibberish text by toxicity classifiers. By introducing three simple components—a pairwise contrastive objective (CTB) to eliminate $Z$, Noisy Gradient Pruning (NGP) to filter uninformative pairs, and a Min-K Fluency Stabilizer (MKS) to exclude gibberish—the proposed method increases the number of unique attacks on Qwen2.5-1.5B from 17 to 134 (approx. 7×) while maintaining a 92% ASR, significantly outperforming baselines in cross-model and cross-defense transferability.

## Background & Motivation

**Background**: LLM red-teaming aims to identify safety vulnerabilities before deployment, primarily following three paradigms: (1) RL-based (PPO, PPO+Curiosity, Jailbreak-R1), which pursues reward maximization and can find highly toxic prompts but suffers from severe mode collapse; (2) Quality-Diversity (Rainbow Teaming, Ruby Teaming), which relies on predefined style/topic matrices and evolutionary strategies to maintain diversity but suffers from low attack success rates due to reliance on the instruction-following of frozen LLMs; (3) GFN-based (Lee et al. 2024), which treats red-teaming as distribution matching—where sampling probability $\propto$ reward—theoretically allowing for both high toxicity and high diversity.

**Limitations of Prior Work**: Directly applying GFlowNet objectives like Trajectory Balance (TB) to LLMs faces two major pitfalls:
- The TB loss $\mathcal{L}_{TB}(y; \theta) = (\log Z_\theta + \log \pi_\theta(y) - \log R(y))^2$ requires learning a scalar $Z_\theta$ to estimate $Z \simeq \sum_{y \in \mathcal{Y}} R(y)$. Given the combinatorial explosion of the token sequence space $\mathcal{Y}$, $Z_\theta$ is difficult to estimate accurately, leading to high-variance gradients, training instability, or continued mode collapse.
- Red-teaming rewards originate from toxicity classifiers, which often assign random pseudo-rewards of 0.2~0.3 to gibberish-like OOD text. Once an attacker discovers such reward-hacking paths, it rapidly collapses to local optima generating gibberish.

**Key Challenge**: While GFN’s lossless distribution matching property should ensure diversity, the practical instability of $Z$ estimation causes TB to degenerate into a narrow distribution fit similar to RL. Standard methods for ensuring fluency, such as KL-divergence regularization $R_{ref}(y) = \pi_{KL}(y)^\alpha \cdot R(y)^\beta$, distort the target distribution (biasing the sampled distribution toward the reference rather than the reward), conflicting with GFN theoretical assumptions.

**Goal**: (1) Design an alternative GFN objective that does not require $Z_\theta$ while maintaining equivalence to TB at the optimum; (2) Implement a saliency-based filtering strategy for noisy rewards to avoid contamination by random pseudo-rewards; (3) Prevent attackers from hacking into gibberish regions without using methods that distort the target distribution like KL.

**Key Insight**: The authors observe that when performing a ratio-based comparison between two trajectories $y_1, y_2$ from the same policy, the partition function $Z_\theta$ naturally cancels out—a standard motivation for contrastive objectives. Furthermore, the "reward noise" problem essentially arises from low-contrast pairs providing incorrect gradient signals during pairwise comparisons, which can be addressed using a contrast-aware indicator as a hard filter. "Gibberish recurrence" can be mitigated by using the Min-K probability (average log-prob of the least-likely tokens) as a fluency proxy with a hard threshold.

**Core Idea**: Synthesize a "Stable-GFN" framework using three components: Contrastive Trajectory Balance (CTB) to eliminate $Z_\theta$, Noisy Gradient Pruning (NGP) to filter pairs by reward contrast, and a Min-K Fluency Stabilizer (MKS) to block gibberish.

## Method

### Overall Architecture

Input: Attacker LLM $\pi_\theta$ (Qwen2.5-1.5B SFT on Safety-Dataset + AdvBench), Victim LLM $\pi_\phi$, Toxicity Classifier $\pi_\psi$, and a fixed meta-prompt. Each training step: (1) The attacker samples $N$ candidate attack prompts $\{y_n\}$ using the current policy; (2) The victim generates a response $z_n$ for each prompt, and the classifier calculates toxicity $R(y_n) = \mathbb{E}_{z \sim \pi_\phi(\cdot|y)}[T(y, z)]$; (3) MKS uses a reference model to calculate the Min-K fluency of each prompt, masking those below the threshold; (4) NGP enumerates $N^2$ pairs within the batch, filtering low-saliency pairs where $|\log R(y_1) - \log R(y_2)| \le \sigma$; (5) Remaining pairs are used to calculate the CTB loss to update $\theta$. The entire pipeline eliminates the external parameter $Z_\theta$, does not maintain an archive, and requires no strong constraints from a reference policy.

### Key Designs

1. **Contrastive Trajectory Balance (CTB)**:

    - **Function**: Replaces absolute matching with pairwise comparisons to mathematically eliminate $Z_\theta$, resulting in an objective with lower variance that shares the same optimal policy as TB.
    - **Mechanism**: For a pair of independent samples $y_1, y_2 \sim \pi_\theta$, define $\mathcal{L}_{CTB}(y_1, y_2; \theta) = (\log \tfrac{\pi_\theta(y_1)}{\pi_\theta(y_2)} - \log \tfrac{R(y_1)}{R(y_2)})^2$. Let $f(y) = \log \pi_\theta(y) - \log R(y)$; when $y_1, y_2$ are i.i.d. samples, the objective is equivalent to $2 \cdot \mathrm{Var}_{\pi_\theta}(f(y))$. Minimizing this to 0 is equivalent to $f$ being a constant $C$ across the support, which (combined with normalization) implies $\pi_\theta(y) = R(y)/Z$—recovering the TB optimal solution (Theorem 4.1). In the gradient $\nabla_\theta \mathcal{L}_{CTB} = 2(f(y_1) - f(y_2))(\nabla_\theta f(y_1) - \nabla_\theta f(y_2))$, each sample serves as a stochastic baseline for the other's log-flow error, isomorphic to variance reduction in RLOO/Williams.
    - **Design Motivation**: Remove the high-variance source $Z_\theta$ fundamentally. Additionally, $N$ samples per batch allow for $N^2$ scalar pairwise losses without extra forward passes, maintaining $O(N)$ training complexity.

2. **Noisy Gradient Pruning (NGP)**:

    - **Function**: Since CTB aggregates reward noise from two samples, low-contrast pairs amplify noise; NGP uses a hard mask to zero out gradients from low-saliency pairs.
    - **Mechanism**: $\mathcal{L}_{NGP}(y_1, y_2; \theta) = \mathbb{1}[|\log R(y_1) - \log R(y_2)| > \sigma] \cdot \mathcal{L}_{CTB}(y_1, y_2; \theta)$, where $\sigma$ is a saliency threshold hyperparameter. Theoretically, by constructing a saliency graph $G_\sigma = (\mathcal{Y}, E_\sigma)$ (where edges define sample pairs with contrast > $\sigma$), if $G_\sigma$ is connected, then $\mathcal{L}_{NGP}(\theta) = 0$ is still equivalent to $\pi_\theta(y) \propto R(y)$ (Proposition 4.2). In practice, a high-reward replay buffer is used as "global anchors" to provide contrastive pairs across high/low reward regions to maintain connectivity.
    - **Design Motivation**: The toxicity classifier is dominated by random noise for samples with similar rewards. Filtering these "zero information but non-zero noise" pairs ensures gradients only come from pairs with real reward differences, preserving GFN properties (under connectivity assumptions) while significantly reducing gradient variance.

3. **Min-K Fluency Stabilizer (MKS)**:

    - **Function**: Prevents attackers from hacking into gibberish regions without distorting the target distribution.
    - **Mechanism**: A reference model $\pi_{ref}$ calculates token log-probs for the generated prompt $y$, taking the average of the lowest $k$ tokens: $M_k(y) = \tfrac{1}{|K|}\sum_{w \in K} \log \pi_{ref}(y_w | y_{<w})$. The reward is modified to $R_{MKS}(y) = \mathbb{1}[M_k(y) \ge T_{MKS}] \cdot R(y)$—zeroing rewards for prompts below the fluency threshold $T_{MKS}$. Gradients from $\pi_{ref}$ are not used.
    - **Design Motivation**: Unlike global KL regularization, MKS only penalizes samples in the "least fluent segments" (most susceptible to OOD gibberish), allowing freedom for normal prompt exploration. It does not modify the shape of the target distribution (hard cutoff within the reward rather than reshaping), remaining compatible with GFN distribution matching.

### Loss & Training

The total objective is $J_{CTB}(\theta) = \mathbb{E}_{y_1, y_2 \sim \pi_\theta}[\mathcal{L}_{NGP}(y_1, y_2; \theta)]$, utilizing the modified $R_{MKS}$ reward. Within a batch of $N = 1024$ samples, all pairs are enumerated. Attacker: Qwen2.5-1.5B SFT; Victim: Qwen2.5-1.5B-Instruct; Toxic classifier: Meta-Llama-Guard-3-8B; Diversity: all-MiniLM-L6-v2 + greedy clustering threshold 0.7; ASR defined as reward >0.5.

## Key Experimental Results

### Main Results

| Method | UA (#) | ASR (%) | Note |
|------|--------|---------|------|
| PPO | 3.00 | **91.70** | High ASR but extreme mode collapse |
| PPO + Curiosity | 4.00 | 36.75 | Still suffers collapse |
| Rainbow Teaming | 33.00 | 66.11 | High QD diversity but low ASR |
| Jailbreak R1 (8B) | 75.33 | 7.36 | Diverse but low toxicity |
| GFN (TB) | 17.67 | 93.75 | High ASR but UA far below theory |
| **S-GFN (Ours)** | **134.00** | 92.55 | Comparable ASR, 7× UA Gain |

Cross-Attack Defense Transfer (attacking a GFN-defended victim):

| Attacker | GFN-defended victim ASR | Note |
|----------|---------|------|
| GFN | 4.69% | Own attack blocked by own defense |
| Jailbreak R1 | 2.96% | – |
| **S-GFN** | **22.53%** | Broader attack patterns, strong cross-defense transfer |

### Ablation Study

| Config | UA (#) | ASR (%) | Description |
|------|--------|---------|------|
| GFN-TB + KL ref | 14 | – | Reference KL distorts distribution |
| GFN-TB + LogProb | 65 | – | Alternative regularization |
| GFN-TB + MKS | 67 | 85.8 | TB + Fluency cutoff |
| **GFN-CTB + MKS** | 108 | 82.9 | UA +60% after adding CTB |
| **GFN-CTB + MKS + NGP** | **121** | **92.2** | Full S-GFN, recovery of ASR |

### Key Findings

- **CTB's core contribution is stability**: Replacing TB with CTB alone (keeping MKS) increases UA from 67 to 108, proving that $Z_\theta$ estimation is a primary driver of mode collapse.
- **NGP improves both UA and ASR**: Increasing from 108 to 121 UA and 82.9% to 92.2% ASR indicates that filtering low-saliency pairs reduces noise while strengthening gradient signals; "quality over quantity" yields better results.
- **Significant Cross-Attack asymmetry**: S-GFN attacks GFN-defended models at 22.53%, while GFN attacks S-GFN-defended models at only 0.03%. This asymmetry suggests S-GFN identifies truly diverse attack patterns rather than just a superset of GFN attacks.
- **Transfer attacks to unseen victims** (Gemma3, Llama3.2, Qwen3, GPT-OSS-20B): S-GFN ranks first in both UA and ASR across all models, indicating these attacks are not overfit to the training victim.
- **Necessity of MKS**: Without MKS, rewards drop to zero (all samples hack into gibberish); adding MKS immediately improves UA to 67, essentially rescuing the training process.

## Highlights & Insights

- The "cancellation of $Z_\theta$" is a simple but significant insight—text-based GFNs have long been plagued by $Z$ estimation. CTB uses a ratio form to make $Z$ naturally disappear, similar to how contrastive learning eliminates normalizing constants. The equivalence proof (Theorem 4.1) ensures theoretical distribution matching is preserved.
- The "saliency graph connectivity" analysis is elegant—it formalizes "how many pairs can be pruned while preserving GFN convergence" as a graph connectivity condition and identifies the empirical anchor role of the replay buffer.
- Using Min-K probability (from LLM membership inference literature) for fluency detection is a clever cross-domain application—it is more focused on the "weakest link" than traditional perplexity, making it more sensitive to partial gibberish.
- Implementation complexity is extremely low: CTB is an $N^2$ scalar operation, NGP is an indicator mask, and MKS is a reward cutoff. All three are "hard filters or loss modifications" that do not increase forward pass frequency.

## Limitations & Future Work

- $\sigma$ (NGP) and $T_{MKS}$ (MKS) are fixed hyperparameters; task-adaptive adjustment was not explored. Since reward distributions change during training, fixed thresholds may perform differently across stages.
- The connectivity assumption may not hold when the number of distribution modes is very large; the authors acknowledge that the "high-reward replay buffer" is an empirical anchor without non-asymptotic convergence bounds.
- Main experiments used a Qwen2.5-1.5B attacker; scaling to larger attackers (e.g., 7B/13B) was not explored, and whether CTB maintains variance reduction at scale remains to be seen.
- Integration with multi-stage iterative GFNs (Yun et al. 2025) was not explored. Incorporating CTB into iterative frameworks for further diversity gains is a natural next step.
- Ethical considerations: Improved attack methods identifying vulnerabilities pose risks. The paper lacks an in-depth discussion on disclosure processes; results like 92% ASR and 134 UA pose direct risks to open-source victims, requiring responsible release.

## Related Work & Insights

- **vs GFN-TB (Lee et al. 2024)**: Original TB treats $Z_\theta$ as a learnable parameter; high variance leads to mode collapse. CTB uses pairwise ratios to eliminate $Z$, maintaining the same optimal policy with stable training.
- **vs PPO + Curiosity (Hong et al. 2024)**: RL with diversity reward terms remains a single-point reward optimization (UA of 4). S-GFN follows distribution matching (UA of 134).
- **vs Rainbow Teaming (Samvelyan et al. 2024)**: QD uses predefined style/topic matrices for diversity but suffers from low ASR (66%). S-GFN requires no predefined archive and automatically identifies diverse patterns via reward signals.
- **vs DPO with replay**: DPO achieves a UA of only 5.33 in red-teaming; while its preference contrast objective is superficially similar to CTB, DPO optimizes preference ranking rather than distribution matching.
- **vs DB / SubTB (Bengio et al. 2023; Madan et al. 2023)**: DB/SubTB avoids $Z$ estimation but is computationally expensive at the token level. CTB performs pairwise operations at the sequence level, making it computationally friendly for LLMs.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of contrastive $Z$ elimination is borrowed from contrastive learning but systematically applied to LLM-scale GFNs with noise/fluency handling; CTB-TB equivalence proof is solid.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 5 baselines, cross-attack defense, 4 transfer victims, and 3 ablation modules with clear quantification; lacks attacker scaling experiments.
- Writing Quality: ⭐⭐⭐⭐ Clear mapping between motivation, theory, algorithm, and experiments; Figure 1 provides an intuitive overview.
- Value: ⭐⭐⭐⭐ Moves GFN to a practical level for LLM red-teaming and provides a generalizable "stable GFN" toolbox useful for the alignment and safety community, though risks to open-source victims necessitate caution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] STAR-Teaming: A Strategy-Response Multiplex Network Approach to Automated LLM Red Teaming](../../ACL2026/llm_safety/star-teaming_a_strategy-response_multiplex_network_approach_to_automated_llm_red.md)
- [\[ICML 2026\] FoeGlass: Simple In-Context Learning Is Enough for Red Teaming Audio Deepfake Detectors](foeglass_simple_in-context_learning_is_enough_for_red_teaming_audio_deepfake_det.md)
- [\[ACL 2026\] Red-Bandit: Test-Time Adaptation for LLM Red-Teaming via Bandit-Guided LoRA Experts](../../ACL2026/llm_safety/red-bandit_test-time_adaptation_for_llm_red-teaming_via_bandit-guided_lora_exper.md)
- [\[ICLR 2026\] Tree-based Dialogue Reinforced Policy Optimization for Red-Teaming Attacks (DialTree)](../../ICLR2026/llm_safety/tree-based_dialogue_reinforced_policy_optimization_for_red-teaming_attacks.md)
- [\[ICML 2026\] MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio](medmosaic_a_challenging_large_scale_benchmark_of_diverse_medical_audio.md)

</div>

<!-- RELATED:END -->
