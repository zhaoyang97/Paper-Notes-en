---
title: >-
  [Paper Note] Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models
description: >-
  [ICML 2026][LLM Safety][TTS] This paper reveals a previously overlooked failure mode of Test-Time Scaling (TTS): by simply reducing the diversity of candidate responses…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "TTS"
  - "Diversity Attack"
  - "Shannon Entropy"
  - "MCTS"
  - "Best-of-N"
date: 2026-05-08
content_hash: 87a7fc31de821407
---

# Less Diverse, Less Safe: The Indirect But Pervasive Risk of Test-Time Scaling in Large Language Models

**Conference**: ICML 2026  
**arXiv**: [2510.08592](https://arxiv.org/abs/2510.08592)  
**Code**: None  
**Area**: LLM Safety / Test-Time Scaling / Jailbreak Attacks  
**Keywords**: TTS, Diversity Attack, Shannon Entropy, MCTS, Best-of-N

## TL;DR
This paper reveals a previously overlooked failure mode of Test-Time Scaling (TTS): by simply reducing the diversity of candidate responses, TTS becomes even more prone to outputting unsafe content than directly feeding adversarial prompts. The authors propose RefDiv, a genetic algorithm driven by dual signals—Shannon entropy and reference guidance—which efficiently jailbreaks across models, closed-source systems, and guardrails on both MCTS and Best-of-N.

## Background & Motivation

**Background**: TTS has become a standard approach for improving the reliability of LLM inference—typified by Best-of-N (BoN) and Monte Carlo Tree Search (MCTS): the model generates multiple candidates during inference, then selects the best one via a reward model or search process. The community generally assumes that "the more diverse the candidates, the more robust TTS is," and regards TTS as a safety barrier against hallucination and for enhancing reasoning.

**Limitations of Prior Work**: Existing jailbreak research mainly focuses on single forward-pass attacks (GCG / AutoDAN / AutoDAN-Turbo), with no systematic study of whether TTS frameworks themselves have structural weaknesses. Directly applying these attacks to TTS is limited in effectiveness, as the reward model or search process can filter out obviously harmful candidates.

**Key Challenge**: The "safety gain" of TTS implicitly assumes a highly diverse candidate pool; however, if an attacker can collapse the candidate pool to a few nearly identical responses (mode collapse), the selection step in TTS loses its filtering power and instead promotes highly consistent harmful responses as "high-quality" outputs.

**Goal**: (1) Demonstrate that candidate diversity is an unrecognized vulnerability in TTS; (2) Design a stress test protocol that can reliably reduce diversity while steering towards harmful responses; (3) Test whether this attack generalizes across TTS strategies, closed-source LLMs, and guardrails.

**Key Insight**: The authors approach TTS from an information-theoretic perspective—lower token-level Shannon entropy $H$ in the candidate set means TTS is less able to select a "good" answer; but simply minimizing entropy leads to meaningless text, so candidates must also be steered toward the affirmative token set $\mathcal{C}^\ast = \{"\text{Sure, I can help}"\ldots\}$ to achieve both low diversity and harmfulness.

**Core Idea**: Employ a dynamically weighted genetic algorithm, RefDiv, which initially uses reference-guided entropy to steer the population into harmful regions, then switches to pure entropy minimization for convergence in low-diversity regions—this "teach direction first, then collapse" curriculum is more stable than direct attacks.

## Method

### Overall Architecture
RefDiv is a genetic algorithm (GA) with population size $m$. At each iteration $t$: (1) For each candidate prompt $x_i$ in the current population, sample a candidate response set $C_{x_i}$ under the target LLM + TTS; (2) Compute two entropies—pure candidate entropy $\text{DFS}(x_i) = H(C_{x_i})$ and entropy after mixing in the reference affirmative set $\text{DFS}^\ast(x_i) = H(C_{x_i} \cup \mathcal{C}^\ast)$; (3) Rank by fitness and select the top-$q$ for crossover + mutation; (4) Use a dynamic weight $\alpha_t = \exp(\frac{\ln 2}{T-1}(t-1)) - 1$ to smoothly switch between "reference-guided" and "pure entropy minimization." After $T$ iterations, return the $x'$ with the highest fitness.

### Key Designs

1. **Dual-Signal Diversity-Guided Fitness**:

    - **Function**: Encodes both "low diversity" and "semantic harmfulness" into a scalar fitness, enabling the GA to evolve in both directions.
    - **Mechanism**: Fitness is defined as $\mathcal{F}(x,t) = (\alpha_t - 1) \cdot \text{norm}(\Delta\text{DFS}(x)) - \alpha_t \cdot \text{norm}(\text{DFS}(x))$, where $\Delta\text{DFS}(x) = |\text{DFS}(x) - \text{DFS}^\ast(x)|$. A small $\Delta\text{DFS}$ means the candidates have absorbed affirmative tokens; a small $\text{DFS}$ means high concentration. Both terms are z-score normalized, with weights determined by $\alpha_t$.
    - **Design Motivation**: Pure entropy minimization leads the GA to converge on meaningless text, which is scored zero by the reward model; pure reference guidance easily triggers guardrails. The coupling of these two (in opposite directions) imposes a "harmful yet consistent" dual constraint on the GA.

2. **Exponentially Dynamic Weight $\alpha_t$ Curriculum**:

    - **Function**: Uses the reference term for guidance in early GA stages to avoid "premature convergence," then tightens with the pure entropy term in later stages.
    - **Mechanism**: $\alpha_t$ is near 0 at $t=1$ and approaches 1 at $t=T$. Early fitness is dominated by $(\alpha_t - 1) \cdot \Delta\text{DFS}$ (negative coefficient, so minimizing $\Delta\text{DFS}$ increases fitness), pulling the population into the "obedient" region; later, $-\alpha_t \cdot \text{DFS}$ dominates, forcing convergence to low-entropy clusters.
    - **Design Motivation**: This is equivalent to curriculum learning—teach direction first, then apply pressure. Experiments show Shannon entropy decreases monotonically with iterations, while other attacks show little change, indicating RefDiv's low-entropy convergence is directly driven by the $\alpha_t$ schedule.

3. **GA Crossover + Mutation Operations**:

    - **Function**: Evolves prompts at the character level, avoiding reliance on gradient optimization and enabling pure black-box operation.
    - **Mechanism**: Each generation selects the top-$q$ parents, performs crossover and local token replacement to generate $m$ offspring; fitness is evaluated entirely via forward inference, with no gradients required. This makes RefDiv naturally compatible with closed-source API targets (though white-box surrogates are needed for adversarial sample training).
    - **Design Motivation**: The authors explicitly avoid assuming reward visibility—if the optimization target were directly the reward, it would degenerate into a trivial attack. GA allows evolution using only the scalar signals $\text{DFS}$ and $\text{DFS}^\ast$.

### Loss & Training
RefDiv is an inference-time attack with no training phase. Hyperparameters: population size $m$, number of parents $q$, iterations $T$, and affirmative token set $\mathcal{C}^\ast$ (following GCG / AutoDAN style). MCTS uses the default 3 children × 3 iterations; BoN main experiments use $N=8$ + PairRM as reward.

## Key Experimental Results

### Main Results

| TTS | Model | GCG | AutoDAN | AutoDAN-Turbo | RefDiv (Ours) |
|-----|-------|-----|---------|---------------|---------------|
| BoN ($N=8$) | Qwen3-8B | 0.335 | **0.996** | 0.414 | 0.995 |
| BoN ($N=8$) | Mistral-7B | 0.877 | 0.973 | 0.733 | **0.976** |
| BoN ($N=8$) | Llama3.1-8B | 0.176 | 0.368 | 0.397 | **0.465** |
| BoN ($N=8$) | Gemma3-27B | 0.054 | 0.749 | 0.171 | **0.926** |
| MCTS | Llama3.1-8B | 0.254 | 0.831 | 0.446 | **0.967** |
| MCTS | Gemma3-27B | 0.336 | 0.904 | 0.156 | **0.989** |

### Ablation Study

| Configuration | Key Observation | Description |
|---------------|----------------|-------------|
| Increase $N$ (BoN candidate count) | ASR barely drops | Increasing diversity does not counter RefDiv |
| Switch reward (deberta / ToxiGuardRail) | RefDiv still outperforms AutoDAN | Attack is not tied to a specific reward |
| Perplexity filter (top-10%/20%) | RefDiv still 42.7% success rate | Generated prompts do not have high perplexity |
| Guardrail (LlamaGuard-3/4, OpenAI Mod) | Average ASR $\approx 82\%$ | Mainstream guardrails are almost entirely bypassed |

### Key Findings
- BoN and MCTS attacks are mutually transferable: adversarial prompts generated for BoN are effective on MCTS and vice versa—indicating this is a TTS paradigm-wide issue, not an artifact of a particular reward/search.
- Attack prompts crafted on Llama3.1-8B transfer in black-box fashion to GPT-4.1, o3-mini, Gemini-2.5-Pro, Claude-3.5-Haiku, and other closed-source LLMs (with Gemini-2.5-Flash achieving the highest ASR), indicating this is a model-agnostic failure mode.
- RefDiv's Shannon entropy curve decreases monotonically, initially slightly higher than AutoDAN (due to the reference term), then drops to the lowest—confirming the two-stage behavior of the fitness design.

## Highlights & Insights
- Exposes the implicit assumption that "TTS safety depends on diversity," and validates it not just by reasoning but with entropy metrics and a reproducible attack algorithm—this "hypothesize vulnerability, design stress test, then statistically validate" paradigm is highly reusable.
- Training-free and black-box transferability make RefDiv closer to a "realistic threat model" than methods like AutoDAN-Turbo that require pre-trained agents, highlighting the need for diversity degradation to be considered in BoN/MCTS system security evaluations.
- The finding that almost all guardrails fail generalizes to any deployment mode using "single-point classifiers guarding LLM entry," suggesting defenders should introduce diversity-aware monitors into inference pipelines.

## Limitations & Future Work
- The attack only targets "safety-related" failures and does not analyze whether TTS also has similar diversity traps in factuality or reasoning correctness.
- ASR is measured only on the AdvBench dataset and relies on standard judge models, introducing evaluation bias.
- The default $\alpha_t$ schedule is exponential; although other schedules are explored in the appendix, all are monotonically increasing, with no exploration of "collapse then release" reverse curricula.
- Defense discussion is limited to negative results, with no concrete design for diversity-aware TTS, leaving this for future work.

## Related Work & Insights
- **vs AutoDAN / AutoDAN-Turbo**: All are GA-based attacks, but RefDiv introduces reward-model-aware entropy signals, making it more targeted for TTS "ensemble-selection" pipelines; AutoDAN-Turbo requires a pre-trained skill library, while RefDiv is entirely inference-time.
- **vs GCG**: GCG is a gradient-based method requiring white-box access; RefDiv uses GA + scalar signals, is black-box friendly, and is significantly stronger on TTS (GCG achieves only 0.054 ASR on Gemma3-27B BoN).
- **vs PackLLM / Self-Consistency**: These are "honest TTS" methods, all assuming diversity is inherently positive; RefDiv demonstrates that this assumption is itself an attack surface.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to reveal and systematically exploit TTS diversity failure mode
- Experimental Thoroughness: ⭐⭐⭐⭐ Across 8+ models / two TTS methods / 5 closed-source models + 4 guardrails
- Writing Quality: ⭐⭐⭐⭐ Algorithm and analysis are clear, but some figures relegated to the appendix affect readability
- Value: ⭐⭐⭐⭐⭐ Presents a genuinely disruptive new threat to LLM security deployment

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Towards Fine-Grained Robustness: Attention-Guided Test-Time Prompt Tuning for Vision-Language Models](towards_fine-grained_robustness_attention-guided_test-time_prompt_tuning_for_vis.md)
- [\[CVPR 2026\] Test-Time Attention Purification for Backdoored Large Vision Language Models](../../CVPR2026/llm_safety/test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[AAAI 2026\] SproutBench: A Benchmark for Safe and Ethical Large Language Models for Youth](../../AAAI2026/llm_safety/sproutbench_a_benchmark_for_safe_and_ethical_large_language_models_for_youth.md)
- [\[ICML 2026\] MedMosaic: A Challenging Large Scale Benchmark of Diverse Medical Audio](medmosaic_a_challenging_large_scale_benchmark_of_diverse_medical_audio.md)
- [\[ICLR 2026\] Converge Faster, Talk Less: Hessian-Informed Federated Zeroth-Order Optimization](../../ICLR2026/llm_safety/converge_faster_talk_less_hessian-informed_federated_zeroth-order_optimization.md)

</div>

<!-- RELATED:END -->
