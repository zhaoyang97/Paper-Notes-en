---
title: >-
  [Paper Note] Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models
description: >-
  [ICML 2026][LLM Reasoning][TTS] The paper reveals an overlooked failure mode of Test-Time Scaling (TTS): by suppressing the diversity of candidate responses…
tags:
  - "ICML 2026"
  - "LLM Reasoning"
  - "TTS"
  - "Diversity Attack"
  - "Shannon Entropy"
  - "MCTS"
  - "Best-of-N"
date: 2026-05-08
content_hash: d1bc7bac4a9e14af
---

# Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2510.08592](https://arxiv.org/abs/2510.08592)  
**Code**: None  
**Area**: LLM Security / Test-Time Scaling / Jailbreak Attacks  
**Keywords**: TTS, Diversity Attack, Shannon Entropy, MCTS, Best-of-N

## TL;DR
The paper reveals an overlooked failure mode of Test-Time Scaling (TTS): by suppressing the diversity of candidate responses, TTS becomes more likely to output unsafe content than directly feeding highly adversarial prompts. It proposes RefDiv, a genetic algorithm driven by dual signals of Shannon entropy and reference guidance, which achieves efficient jailbreaking across models, closed-source APIs, and guardrails in MCTS and Best-of-N frameworks.

## Background & Motivation

**Background**: TTS has become a standard practice for improving LLM inference reliability, exemplified by Best-of-N (BoN) and Monte Carlo Tree Search (MCTS). In these frameworks, the model generates multiple candidates during inference, and a reward model or search process selects the best one. The community generally assumes that "higher candidate diversity leads to more robust TTS" and treats TTS as a safety guardrail against hallucinations and for enhancing reasoning.

**Limitations of Prior Work**: Existing jailbreak research primarily focuses on single-forward-pass attacks (GCG / AutoDAN / AutoDAN-Turbo). No systematic study has explored whether the TTS framework itself possesses structural weaknesses. Directly applying these attacks to TTS yields limited success because reward models or search processes can filter out obviously harmful candidates.

**Key Challenge**: The "safety gain" of TTS implicitly assumes that the candidate pool is highly diverse. However, if an attacker can cause the candidate pool to collapse into nearly identical responses (mode collapse), the selection step of TTS loses its filtering capability. Consequently, highly consistent harmful responses are promoted as "high-quality" outputs.

**Goal**: (1) Demonstrate that candidate diversity is an unidentified vulnerability in TTS; (2) Design a stress test protocol that stably reduces diversity while guiding outputs toward harmful responses; (3) Verify the cross-strategy, cross-model, and cross-guardrail transferability of this attack.

**Key Insight**: The authors view TTS from an information-theoretic perspective. The lower the token-level Shannon entropy $H$ of the candidate set, the less likely TTS is to select a "good" answer. However, simply minimizing entropy might cause candidates to degenerate into meaningless text. To be both low-diversity and harmful, candidates must simultaneously converge toward an affirmative token set $\mathcal{C}^\ast = \{"\text{Sure, I can help}"\ldots\}$.

**Core Idea**: The RefDiv genetic algorithm utilizes a dynamic weight $\alpha_t$. In early stages, reference-guided entropy pulls the population into harmful regions; in later stages, pure entropy minimization forces the population to converge in low-diversity regions. This curriculum of "teaching direction first, then collapsing" is more stable than direct attacks.

## Method

### Overall Architecture
RefDiv is a genetic algorithm (GA) with a population size $m$. In each iteration $t$: (1) For each candidate prompt $x_i$, a candidate response set $C_{x_i}$ is sampled under the target LLM + TTS; (2) Two types of entropy are calculated—pure candidate entropy $\text{DFS}(x_i) = H(C_{x_i})$ and entropy mixed with a reference affirmative set $\text{DFS}^\ast(x_i) = H(C_{x_i} \cup \mathcal{C}^\ast)$; (3) The top-$q$ individuals are selected based on fitness for crossover and mutation; (4) A dynamic weight $\alpha_t = \exp(\frac{\ln 2}{T-1}(t-1)) - 1$ smoothly transitions between "approaching reference" and "pure entropy reduction." After $T$ iterations, the prompt $x'$ with the highest fitness is returned.

### Key Designs

1.  **Diversity-Guided Fitness**:
    - **Function**: Encodes the objectives of "low diversity" and "semantic harm" into a scalar fitness value, allowing the GA to evolve in both directions simultaneously.
    - **Mechanism**: Fitness is defined as $\mathcal{F}(x,t) = (\alpha_t - 1) \cdot \text{norm}(\Delta\text{DFS}(x)) - \alpha_t \cdot \text{norm}(\text{DFS}(x))$, where $\Delta\text{DFS}(x) = |\text{DFS}(x) - \text{DFS}^\ast(x)|$. A small $\Delta\text{DFS}$ indicates candidates have absorbed affirmative tokens; a small $\text{DFS}$ indicates high concentration. Both terms are z-score normalized and weighted by $\alpha_t$.
    - **Design Motivation**: Pure entropy reduction might lead the GA to converge on meaningless text, which reward models would score as 0. Purely targeting affirmative tokens might trigger guardrails. Coupling them as binary constraints forces the GA to find prompts that are "both harmful and consistent."

2.  **Curriculum via Exponential Dynamic Weight $\alpha_t$**:
    - **Function**: Uses the reference term for guidance in early GA stages to avoid "premature convergence" and applies pure entropy reduction in later stages.
    - **Mechanism**: $\alpha_t \approx 0$ at $t=1$ and approaches $1$ at $t=T$. Early fitness is dominated by $(\alpha_t - 1) \cdot \Delta\text{DFS}$ (with a negative coefficient, making $\Delta\text{DFS}$ minimization equivalent to fitness increase), pulling the population into "compliant" regions. Later stages transition to $-\alpha_t \cdot \text{DFS}$ dominance, forcing convergence to low-entropy clusters.
    - **Design Motivation**: This is equivalent to curriculum learning—teaching the direction before applying pressure. Experiments show Shannon entropy decreases monotonically with iterations, whereas other attacks maintain high entropy, proving that RefDiv's convergence is directly driven by the $\alpha_t$ schedule.

3.  **Crossover and Mutation in Genetic Algorithm**:
    - **Function**: Performs evolutionary search at the character level of the prompt, avoiding reliance on gradient optimization and enabling black-box execution.
    - **Mechanism**: Each generation selects top-$q$ parents to generate $m$ offspring via crossover and local token replacement. Fitness evaluation uses forward inference only. This naturally supports closed-source API targets through white-box surrogate transfer.
    - **Design Motivation**: The authors deliberately avoid assuming visibility into rewards; if the optimization target were the reward itself, the attack would become trivial. The GA allows evolution using only scalar signals from $\text{DFS}$ and $\text{DFS}^\ast$.

### Loss & Training
RefDiv is an inference-time attack with no training phase. Hyperparameters: population size $m$, parent count $q$, iterations $T$, and affirmative token set $\mathcal{C}^\ast$ (following GCG / AutoDAN styles). MCTS uses a default of 3 children × 3 iterations; the BoN main experiment uses $N=8$ with PairRM as the reward model.

## Key Experimental Results

### Main Results

| TTS | Model | GCG | AutoDAN | AutoDAN-Turbo | RefDiv (Ours) |
|-----|------|-----|---------|----------------|----------------|
| BoN ($N=8$) | Qwen3-8B | 0.335 | **0.996** | 0.414 | 0.995 |
| BoN ($N=8$) | Mistral-7B | 0.877 | 0.973 | 0.733 | **0.976** |
| BoN ($N=8$) | Llama3.1-8B | 0.176 | 0.368 | 0.397 | **0.465** |
| BoN ($N=8$) | Gemma3-27B | 0.054 | 0.749 | 0.171 | **0.926** |
| MCTS | Llama3.1-8B | 0.254 | 0.831 | 0.446 | **0.967** |
| MCTS | Gemma3-27B | 0.336 | 0.904 | 0.156 | **0.989** |

### Ablation Study

| Configuration | Key Observation | Description |
|------|---------|------|
| Increasing $N$ (BoN candidates) | ASR barely decreases | Increasing diversity does not counteract RefDiv |
| Switching Reward (deberta / ToxiGuardRail) | RefDiv still outperforms AutoDAN | Attack is not tied to a specific reward model |
| Perplexity filter (top-10%/20%) | RefDiv maintains 42.7% ASR | Generated prompts do not have excessively high perplexity |
| Guardrail (LlamaGuard-3/4, OpenAI Mod) | Average ASR $\approx 82\%$ | Major guardrails are almost entirely bypassed |

### Key Findings
- Attacks on BoN and MCTS are bi-directionally transferable: adversarial prompts generated for BoN are effective against MCTS and vice-versa, indicating this is a common vulnerability in the TTS paradigm rather than an artifact of a specific reward or search method.
- Adversarial prompts optimized on Llama3.1-8B transfer successfully to closed-source LLMs like GPT-4.1, o3-mini, Gemini-2.5-Pro, and Claude-3.5-Haiku (with Gemini-2.5-Flash showing the highest ASR), suggesting a model-agnostic failure mode.
- RefDiv’s Shannon entropy curve decreases monotonically—initially slightly higher than AutoDAN (due to the reference term) but eventually dropping to the lowest point, validating the two-stage behavior of the fitness design.

## Highlights & Insights
- The paper exposes the implicit assumption that "TTS safety depends on diversity." This is validated not just through theory but via entropy metrics and a reproducible attack algorithm. This paradigm of "hypothesizing a vulnerability, designing a stress test, and statistical validation" is highly reusable.
- Being training-free and black-box transferable makes RefDiv a more "realistic threat model" than methods like AutoDAN-Turbo that require pre-trained agents. It serves as a reminder that safety assessments for BoN/MCTS systems must consider diversity degradation.
- The failure of almost all guardrails suggests that deployment modes relying on "single-point classifiers at the LLM entry" are insufficient. Defenders should introduce diversity-aware monitors into the inference pipeline.

## Limitations & Future Work
- The attack only addresses "safety-related" failures; it does not analyze whether TTS has similar diversity traps in factuality or reasoning correctness.
- Only AdvBench was used to measure ASR, and reliance on standard judge models may introduce evaluation bias.
- The default $\alpha_t$ schedule is exponential; while the authors tried other schedules in the appendix, all were monotonically increasing. An "inverse curriculum" (compressing entropy then releasing) was not explored.
- On the defense side, only negative results are provided without specific designs for diversity-aware TTS.

## Related Work & Insights
- **vs AutoDAN / AutoDAN-Turbo**: All are GA-based attacks, but RefDiv introduces reward-model-aware entropy signals, making it more effective against "assembly-selection" pipelines like TTS. AutoDAN-Turbo requires a pre-trained skill library, whereas RefDiv is entirely inference-time.
- **vs GCG**: GCG is a gradient-to-token method requiring white-box access. RefDiv uses GA with scalar signals, making it black-box friendly and significantly stronger on TTS (GCG achieved only 0.054 ASR on Gemma3-27B BoN).
- **vs PackLLM / Self-Consistency**: These are "honest TTS" methods that treat diversity as a default positive signal; RefDiv proves this assumption can be turned into an attack surface.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to reveal and systematically exploit TTS diversity failure modes.
- Experimental Thoroughness: ⭐⭐⭐⭐ Tested across 8+ models, two TTS strategies, 5 closed-source models, and 4 guardrails.
- Writing Quality: ⭐⭐⭐⭐ Algorithms and analyses are clear, though some charts in the appendix affect readability.
- Value: ⭐⭐⭐⭐⭐ Proposes a destructive new threat to safe LLM deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Prism: Efficient Test-Time Scaling via Hierarchical Search and Self-Verification for Discrete Diffusion Language Models](prism_efficient_test-time_scaling_via_hierarchical_search_and_self-verification_.md)
- [\[ICML 2026\] Lookahead Sample Reward Guidance for Test-Time Scaling of Diffusion Models](lookahead_sample_reward_guidance_for_test-time_scaling_of_diffusion_models.md)
- [\[NeurIPS 2025\] Provable Scaling Laws for the Test-Time Compute of Large Language Models](../../NeurIPS2025/llm_reasoning/provable_scaling_laws_for_the_testtime_compute_of_large_lang.md)
- [\[ICLR 2026\] Efficient Test-Time Scaling for Small Vision-Language Models](../../ICLR2026/llm_reasoning/efficient_test-time_scaling_for_small_vision-language_models.md)
- [\[ICML 2026\] Stabilizing Recurrent Dynamics for Test-Time Scalable Latent Reasoning in Looped Language Models](stabilizing_recurrent_dynamics_for_test-time_scalable_latent_reasoning_in_looped.md)

</div>

<!-- RELATED:END -->
