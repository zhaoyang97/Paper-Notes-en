---
title: >-
  [Paper Note] New Wide-Net-Casting Jailbreak Attacks Risk Large Models
description: >-
  [ICML 2026][LLM Alignment][wide-net-casting] This paper defines and systematically analyzes the "wide-net-casting" jailbreak scenario (where an attacker queries a group of LLMs simultaneously and succeeds if any single m…
tags:
  - "ICML 2026"
  - "LLM Alignment"
  - "wide-net-casting"
  - "jailbreak"
  - "model-family-specific vulnerabilities"
  - "exploration-to-exploitation"
  - "adversarial sample generator"
date: 2026-05-08
content_hash: 925c8db3aa8a54ce
---

# New Wide-Net-Casting Jailbreak Attacks Risk Large Models

**Conference**: ICML 2026  
**arXiv**: [2605.17128](https://arxiv.org/abs/2605.17128)  
**Code**: The paper notes "Code is available here," but the repository link is not explicitly given in the main text.  
**Area**: LLM Security / Adversarial Attacks / Multi-model Joint Jailbreak  
**Keywords**: wide-net-casting, jailbreak, model-family-specific vulnerabilities, exploration-to-exploitation, adversarial sample generator

## TL;DR
This paper defines and systematically analyzes the "wide-net-casting" jailbreak scenario (where an attacker queries a group of LLMs simultaneously and succeeds if any single model is breached). Based on this, it designs a joint training method for "specialized" adversarial sample generators using exploration-to-exploitation scheduling, pushing the Attack Success Rate (ASR) to 100% on multiple LLMs/MLLMs without additional defenses. This reveals that current single-model jailbreak evaluations significantly underestimate real-world risks.

## Background & Motivation

**Background**: Current research on LLM and MLLM jailbreaking is almost entirely built upon the "single-model threat model." Given a target model $f_m$, attackers use optimization methods like GCG, AutoDAN, ReMiss, or MLAI to find an adversarial suffix or image to trigger harmful content. Evaluation metrics primarily rely on single-model ASR.

**Limitations of Prior Work**: In the real world, users do not rely on a single model; if Llama fails a math problem, they might switch to Gemma, Mistral, or Vicuna. If this habit is exploited maliciously, an attacker "wins" as long as **any one** of $M$ models provides technical details. Existing benchmarks fail to characterize this "joint vulnerability," leading to a systematic underestimation of actual deployment risks.

**Key Challenge**: Different model families (Llama / Gemma / Mistral / Vicuna; LLaVA / MiniGPT / InstructBLIP / Qwen-VL) possess distinct training data, alignment recipes, and KV cache structures, making their vulnerabilities naturally heterogeneous. Samples that fail on one model might easily breach another. Vulnerabilities are amplified at the group level via a logical "OR" operation, which single-model ASR cannot capture.

**Goal**: (1) Formalize the wide-net-casting scenario and provide evaluation metrics reflecting "breach-any-to-succeed"; (2) Quantify the risk amplification when single-model attacks are migrated to this scenario; (3) Design a customized jailbreak method for this scenario to expose the upper bound of risk.

**Key Insight**: Training $M$ generators independently on all harmful intents (an "all-cover" strategy) is naturally ill-suited for wide-net-casting. Since breaching a corresponding model on a single intent is sufficient, forcing every generator to cover all weaknesses dilutes their expertise. A superior strategy is **specialization**: each generator should focus on the unique vulnerabilities of its target model.

**Core Idea**: Formalize "specialization" as a dual-objective constrained optimization: increasing update weights where generator loss is minimal (exploitation) while maintaining a monotonically decreasing non-zero exploration term (exploration). Using Lagrangian multipliers and KKT conditions, a unique Boltzmann update weight is derived as $\eta_t^{m,*} = \exp(-\ell_t^m/\beta_t) / \sum_j \exp(-\ell_t^j/\beta_t)$, allowing dynamic allocation of the joint training budget across $M$ generators.

## Method

### Overall Architecture
The method consists of two phases. **Phase A (Independent Pre-training)**: Each target model $f_m$ ($m = 1, \dots, M$) is paired with an adversarial generator $g_m$ (an LLM uses a ReMiss-style text suffix generator; an MLLM uses an adversarial image generator fine-tuned from PixArt-α). These are trained independently using existing single-model methods to acquire general jailbreak knowledge. **Phase B (Specialized Joint Training)**: The $M$ generators are trained together for $T = 3000$ steps. At each step, a loss vector $\bm{\ell_t} = (\ell_t^1, \dots, \ell_t^M)$ is calculated for a single harmful intent $x_t$. A constrained minimization problem with entropy is solved to obtain update weights $\bm{\eta_t^*}$, and each generator is updated using the weighted loss $\eta_t^{m,*} \cdot \ell_t^m$. **Testing Phase**: For a given harmful intent, $M$ generators produce $M$ adversarial samples in parallel. The "most harmful" response is selected as the final output using Beaver-Dam-7B or template-based scoring.

The primary evaluation metric is W-ASR (Wide-net-casting ASR), defined as $\text{WASR} = \frac{1}{N} \sum_{n=1}^N \bigvee_{m=1}^M s_m^n$, where $s_m^n \in \{0, 1\}$ indicates if the $n$-th intent breached the $m$-th model. The logical OR ($\bigvee$) captures the "any-single-breach" essence. A corresponding W-Toxicity Score simulates an attacker selecting the most harmful response among $M$ candidates.

### Key Designs

1. **Formalization of Wide-net-casting and Baseline Adaptation**:
    - **Function**: Transforms "group jailbreaking" from an intuition into a quantifiable threat model and provides a baseline protocol for migrating single-model methods.
    - **Mechanism**: Instance-based methods run independent optimizations for each $f_m$; model-based methods instantiate $M$ generators trained independently on the full dataset. WASR=1 if any sample breaches its target. Beaver-Dam-7B serves as a unified judge.
    - **Design Motivation**: Existing papers report only single-model ASR, masking the threat of model switching. Establishing a baseline allows for quantitative assessment of risk amplification. Experiments on AdvBench show GCG ASR rising from 46.2% to a WASR of 75.0% (+28.8 pt), and ReMiss from 86.5% to 92.3%.

2. **Specialized Joint Training Objective: Exploitation + Sustainable Exploration**:
    - **Function**: Encourages $M$ generators to evolve into experts focusing on unique vulnerabilities of their target models rather than generalists.
    - **Mechanism**: Update weights $\bm{\eta_t}$ are constrained to the simplex $\Delta_M$. Objective ❶ minimizes $\sum_m \eta_t^m \ell_t^m$ (assigning higher weights to lower losses to maximize exploitation). Objective ❷ uses Shannon entropy $H(\bm{\eta_t}) = -\sum_m \eta_t^m \log \eta_t^m$ to ensure spread-out, constrained by $H(\bm{\eta_t}) \geq \bar{H}_t$, where $\bar{H}_t = \log M \cdot (T-t)/T$ decreases linearly (sustaining exploration while gradually tightening).
    - **Design Motivation**: Intermediate losses are noisy estimates of true expertise. Pure greed (updating only the minimum loss generator) is misled by noise, resulting in missed vulnerabilities; uniform updates degrade into "all-cover" strategies. The exploration-to-exploitation scheduling from simulated annealing and PSO is adapted here to handle noisy objective observations.

3. **Boltzmann Closed-form Solution and Four-step KKT Derivation**:
    - **Function**: Derives a unique closed-form solution for the constrained optimization, requiring only one $\beta_t$ calculation per step with near-zero extra overhead.
    - **Mechanism**: Introducing Lagrangian multipliers $\beta_t \geq 0$ for $H(\bm{\eta_t}) \geq \bar{H}_t$, and $\nu_t, \alpha_t^m$ for the simplex constraint. The Lagrangian $\mathcal{L}$ yields the KKT first-order condition $\ell_t^i + \nu_t - \alpha_t^i + \beta_t(1 + \log \eta_t^i) = 0$. Solving with complementary slackness gives $\eta_t^{i,*} = \exp(-\ell_t^i / \beta_t) / \sum_m \exp(-\ell_t^m / \beta_t)$, a Boltzmann distribution controlled by temperature $\beta_t$. $\beta_t$ is uniquely determined by $\bar{H}_t$ and $\bm{\ell_t}$ via a one-dimensional search.
    - **Design Motivation**: The closed-form solution ensures training overhead remains comparable to single-model methods. $\beta_t$ naturally couples with temperature annealing: as $\bar{H}_t$ is large, $\beta_t$ is large and the distribution is flat (exploration); as $\bar{H}_t \to 0$, $\beta_t \to 0$ and the distribution becomes one-hot (exploitation).

### Loss & Training
Joint training steps $T = 3000$. Phase A independent pre-training uses ReMiss for LLMs and "MLAI + PixArt-α" for MLLMs. At each step, jailbreak loss $\ell_t^m$ is calculated, $\beta_t$ is solved, and backpropagation is performed with $\eta_t^{m,*} \cdot \ell_t^m$. $\bar{H}_t$ follows linear annealing via $\log M \cdot (T-t)/T$. Hardware: 4×A100.

## Key Experimental Results

### Main Results: Single-model Methods Migrated to Wide-net-casting (AdvBench / 4 LLM Families)

| Defense | Attack | Best Single ASR | WASR | Gain |
|------|------|----------------|------|----------|
| Native Alignment | GCG | 46.2% (Mistral) | 75.0% | +28.8 pt |
| Native Alignment | ReMiss | 86.5% (Mistral) | 92.3% | +5.8 pt |
| + SmoothLLM | GCG | 26.9% | 46.1% | +19.2 pt |
| + SmoothLLM | ReMiss | 38.5% | 61.5% | +23.0 pt |
| + RobustKV | GCG | 24.6% | 37.3% | +12.7 pt |
| + RobustKV | ReMiss | 34.4% | 56.1% | +21.7 pt |

MLLM WASR on MM-SafetyBench is also significantly higher than best single-model ASR, even when targeting models within the same family (e.g., LLaVA-1.5/1.6 variants).

### Ours vs. Baselines vs. Naive Strategies (W-ASR)

| Dataset + Defense | Baseline (ReMiss/MLAI+PixArt) | Naive 1 | Naive 2 | Ours |
|---|---|---|---|---|
| AdvBench LLM, Alignment | 92.3% | 95.1% | 95.8% | **100%** |
| AdvBench LLM, +SmoothLLM | 61.5% | 64.1% | 64.9% | **76.7%** |
| AdvBench LLM, +RobustKV | 56.1% | 59.2% | 60.3% | **72.8%** |
| MM-SafetyBench MLLM, Alignment | 93.7% | 94.9% | 95.1% | **100%** |
| MM-SafetyBench MLLM, +VLGuard | 40.2% | 43.4% | 44.1% | **53.5%** |
| MM-SafetyBench MLLM, +ASTRA | 32.9% | 35.2% | 35.6% | **43.6%** |

### Ablation Study: Scheduling Strategies and $\bar{H}_t$ Forms (MLLM / AdvBench / Alignment Only)

| Configuration | W-ASR | Description |
|------|-------|------|
| Baseline (MLAI+PixArt) | 93.3% | Single-model OR aggregation |
| Naive 1 | 95.5% | Intent split using independent training loss |
| Naive 2 | 95.8% | Pure greedy update (min loss) |
| Variant I (Inverse-prop to loss) | 96.2% | Heuristic, no theory |
| Variant II (Fixed $\lambda_0=0.8$) | 96.1% | Heuristic, no theory |
| Variant III (Dynamic $\lambda_0$) | 96.7% | Heuristic with exploration-exploitation shape |
| Variant IV (Random $\bar{H}_t$) | 97.2% | Closed-form but no monotonic tightening |
| Variant V (Fixed $\bar{H}_t = \log M / 2$) | 98.0% | Closed-form but no annealing |
| Variant VI (Exponential decay) | **100%** | Monotonic tightening |
| Variant VII (Cosine decay) | **100%** | Monotonic tightening |
| **Ours (Linear decay)** | **100%** | Main proposal; simplest |

### Key Findings
- The logical OR aggregation itself is alarming: GCG's WASR jumps +28.8 pt under wide-net-casting. This suggests current benchmarks severely underestimate deployment risks.
- Intra-family target groups (e.g., LLaVA-1.5-13b and LLaVA-1.6-7b) still see WASR increases from 64.1% to 87.6%. Vulnerability heterogeneity exists even without switching families.
- The closed-form solution with monotonic tightening (Ours/VI/VII) significantly outperforms heuristic scheduling. Theoretical optimality and annealing are both essential.
- Even with MLLM-specific external defenses like VLGuard or ASTRA, WASR remains high (42-53%), indicating that single-model-focused defense paradigms fail against group threats.

## Highlights & Insights
- **The redefinition of the threat model is a major contribution**: Framing the common habit of "trying another model" as an attack vector formalized as $\bigvee_m s_m$ renders many "safe" models insecure. This is a high-leverage framing contribution.
- **Adapting simulated annealing to generator scheduling is clever**: Treating training loss as noise leads to the entropy-constrained Lagrangian solution, yielding Boltzmann weights with temperature annealing. The rule simplifies to a softmax with almost zero implementation cost.
- **The "group amplification" and "specialization" framework is reversible for defense**: Identifying "easy-to-breach" intents can help train lightweight filters or enable adversarial alignment. The method provides a dual-use tool for red-teaming.
- **"Model-specific vulnerabilities" can be extended to broader topics**: This dimension is applicable to ensemble robustness, model routing, and MoE security, where group threats should be re-evaluated.

## Limitations & Future Work
- **Limitations acknowledged by authors**: The paper contains potentially harmful examples; evaluation depends on Beaver-Dam-7B and Toxicity Scores, which may have inherent biases.
- **Methodological Limitations**: Joint training requires simultaneous backpropagation of $M$ generators, leading to linear memory growth (4x A100 is a cap for 4 MLLMs). Larger $M$ may require sparse sampling or asynchronous updates.
- **Threat Model Limitations**: The assumption that an attacker can query all $M$ models simultaneously ignores constraints like rate limits, IP association, or cross-provider collaborative detection.
- **Coverage Limitations**: Tests are limited to AdvBench and MM-SafetyBench. Performance on tool-calling or long-context scenarios remains unknown.
- **Future Work**: Research on cross-provider collaborative defenses (shared high-risk intent signatures), group-aware safety filters for routing, and using $\eta_t^*$ as an explanation tool to reverse-engineer specific model vulnerability patterns.

## Related Work & Insights
- **vs GCG (Zou et al. 2023)**: GCG is the representative single-model instance-based attack used here as an OR-aggregated baseline. This paper contributes to scheduling rather than the search operator.
- **vs ReMiss (Xie et al. 2024)**: ReMiss is a SOTA model-based generator. This paper uses it for Phase A pre-training before applying joint specialization.
- **vs MLAI + PixArt-α (Hao et al. 2025; Chen et al. 2024)**: To obtain an MLLM model-based generator, this work uses MLAI outputs as pseudo-labels to fine-tune PixArt-α, creating an independent MLLM generator as a byproduct.
- **vs Guzman-Rivera et al. 2012 "Multiple Choice Learning"**: MCL also assigns samples to one of $M$ predictors to maximize oracle performance but uses a discrete "winner-takes-all" allocation. This paper uses soft allocation with entropy and annealing for better stability.
- **vs Perera et al. 2024**: The Lagrangian + KKT framework utilized here is homologous to certain optimization paradigms, representing a clever migration of pure optimization concepts to adversarial generator joint training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] Heuristic-Induced Multimodal Risk Distribution Jailbreak Attack for Multimodal Large Language Models](../../ICCV2025/llm_alignment/heuristic-induced_multimodal_risk_distribution_jailbreak_attack_for_multimodal_l.md)
- [\[ICLR 2026\] Toward Universal and Transferable Jailbreak Attacks on Vision-Language Models (UltraBreak)](../../ICLR2026/llm_alignment/toward_universal_and_transferable_jailbreak_attacks_on_vision-language_models.md)
- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)
- [\[AAAI 2026\] BiasJailbreak: Analyzing Ethical Biases and Jailbreak Vulnerabilities in Large Language Models](../../AAAI2026/llm_alignment/biasjailbreakanalyzing_ethical_biases_and_jailbreak_vulnerabilities_in_large_lan.md)
- [\[ICLR 2026\] SEMA: Simple yet Effective Learning for Multi-Turn Jailbreak Attacks](../../ICLR2026/llm_alignment/sema_simple_yet_effective_learning_for_multi-turn_jailbreak_attacks.md)

</div>

<!-- RELATED:END -->
