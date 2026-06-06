---
title: >-
  [Paper Note] Debiasing Reward Models via Causally Motivated Inference-Time Intervention
description: >-
  [ACL 2026][LLM Alignment][reward model] The authors interpret the Bradley-Terry reward model as a causal graph for estimating the total effect. They identify bias-specific neurons (accounting for <2% of total neurons) hi…
tags:
  - "ACL 2026"
  - "LLM Alignment"
  - "reward model"
  - "causal intervention"
  - "neuron editing"
  - "length bias"
  - "format bias"
date: 2026-05-08
content_hash: 3b0cf8f5e46759a2
---

# Debiasing Reward Models via Causally Motivated Inference-Time Intervention

**Conference**: ACL 2026  
**arXiv**: [2604.27495](https://arxiv.org/abs/2604.27495)  
**Code**: Paper repository link not public (not in cache)  
**Area**: LLM Alignment / RLHF / Causal Intervention / Interpretability  
**Keywords**: reward model, causal intervention, neuron editing, length bias, format bias

## TL;DR
The authors interpret the Bradley-Terry reward model as a causal graph for estimating the total effect. They identify bias-specific neurons (accounting for <2% of total neurons) highly correlated with activations of five stylistic biases (length / paragraph / word overlap / exclamations / bolding). During inference, these activations are replaced with the validation set median to estimate the controlled direct effect. On RewardBench and RM-Bench, this method eliminates bias without performance degradation. When used for downstream DPO, an 8B model's alignment score matches that of a 70B SOTA reward model.

## Background & Motivation
**Background**: The reward model (RM) in RLHF is core to preference scoring for LLMs, typically implemented using the BT model $p(y_1\succ y_2\mid q)=\sigma(r_\theta(x_1)-r_\theta(x_2))$. However, research increasingly finds that RMs have systematic preferences for response length (length bias) and formats such as lists, bolding, paragraphs, and emojis (Singhal et al. 2024, Zhang et al. 2025).

**Limitations of Prior Work**: (i) Training-time debiasing (ensemble, weight averaging, infoBN, ODIN with extra heads, data augmentation) requires retraining. Costs are high as each new bias requires re-processing data or architectures. (ii) Inference-time debiasing relies on two methods: length penalty (subtracting reward based on character count) and LWR (Locally Weighted Regression to estimate length-only bias terms). Both handle only length and introduce a performance trade-off between biased and unbiased data subsets (unbiased improves, while biased drops significantly). (iii) Internal "encoding" of these biases in RMs remains a black box.

**Key Challenge**: Training-time debiasing is expensive and does not generalize to new bias types; existing inference-time methods perform coarse-grained reward subtractions without addressing internal RM representations, leading to inherent trade-offs between biased and unbiased performance.

**Goal**: (i) Provide a **training-free** inference-time debiasing method handling multiple stylistic biases simultaneously. (ii) Reveal which neurons and layers in the RM encode these biases to providing evidence for interpretability. (iii) Apply the method to DPO preference labeling to evaluate improvements in downstream LLM alignment.

**Key Insight**: View the RM as a causal graph: input $x$ → mediator $m$ (bias neuron activation) → output $r$. The BT model implicitly estimates the total effect $\hat{\mathrm{TE}}$, failing to separate "content quality" from "bias signals." The goal is shifted to estimating the controlled direct effect $\hat{\mathrm{CDE}}$—fixing $m$ to $m^*$ (the validation set median) before calculating the difference, effectively comparing content quality as if both responses had the same bias degree.

**Core Idea**: First, use Spearman correlation to identify the top/bottom-$k$ neurons most relevant to 5 stylistic biases. Then, replace their activations with the validation set median during inference. This is equivalent to CDE estimation, achieving debiasing without retraining, across bias types, and without trade-offs.

## Method

### Overall Architecture
Two stages (Figure 2). **Offline Phase**: On a 500-sample RewardBench validation subset, collect paired samples of "last-token activations" and five bias metrics $f_b(x)$ for every RM neuron. Calculate Spearman $\rho$ and select top-$k$ and bottom-$k$ as bias-specific neurons; record the median activation $m^*$ for each neuron on the validation set. Jointly search $k\in\{50,100,200,500,1000,2000,5000\}$ for each bias type using Optuna + TPE ($7^5 \approx 16,807$ combinations, 100 trials). **Online Phase**: During inference, perform a forward pass for each prompt-response, force-replace activations of hit bias-specific neurons with their respective $m^*$, and output the reward. The BT comparison then equates to estimating $\hat{\mathrm{CDE}} = r_\theta(x_1, m^*) - r_\theta(x_2, m^*)$.

### Key Designs

1. **Multi-bias metrics + Spearman ranking to identify bias-specific neurons**:

    - Function: Locate internal RM neuron sets representing specific stylistic biases without training.
    - Mechanism: Define quantifiable surface features for each bias $b\in\{\text{len, para, over, excl, bold}\}$: length = characters; paragraph = `\n\n` count; overlap = word overlap ratio between response and query; exclamations/bold = `!` / `**` counts. Calculate $\rho(a_n, f_b)$ for neuron $n$ on the validation set, taking top/bottom-$k$ as bias-specific neurons. Merging all biases results in editing only 1.7% of neurons in GRM and 0.085% in FsfairX.
    - Design Motivation: Neuron-level causal localization provides finer granularity than prompt-level or reward-level correction. Spearman correlation captures monotonic relationships and is robust to outliers, while taking both top and bottom $k$ accounts for both positive and negative encoding of biases.

2. **Causal intervention via CDE instead of TE**:

    - Function: Systematically eliminate the reward contribution from stylistic biases by fixing the mediator during BT scoring.
    - Mechanism: Formulate the RM causal graph (Figure 3) as a direct path $x \to r$ and an indirect path $x \to m \to r$. Original BT estimates $\hat{\mathrm{TE}} = r_\theta(x_1, m(x_1)) - r_\theta(x_2, m(x_2))$, covering both paths. This work uses $\hat{\mathrm{CDE}} = r_\theta(x_1, m^*) - r_\theta(x_2, m^*)$, where $m^*$ is the validation median activation scores (the authors also tested 0 and swapping, but median was most stable).
    - Design Motivation: CDE conceptually identifies "who has better content assuming $x_1, x_2$ are identical in bias dimensions," which is the "unbiased comparison" desired for RMs. The median is used over the mean for robustness to outliers, and a single fixed value avoids content differences from $x_1, x_2$ infecting the mediator.

3. **Optuna joint search for multi-bias $k$**:

    - Function: Balance the number of edited neurons and performance automatically to avoid coupling issues where fixing one bias breaks another.
    - Mechanism: Treat $k$ values for 5 biases as 5-dimensional hyperparameters for joint search, targeting overall reward accuracy on 500 validation samples using TPE sampling for 100 iterations. Final selections: GRM (len=5000, para=5000, over=500, excl=200, bold=50); FsfairX (len=500, para=100, over=100, excl=50, bold=200).
    - Design Motivation: Independent searches for $k$ ignore overlaps and interference between neuron sets. TPE joint search detects coupling, such as how editing too many paragraph neurons might degrade length accuracy.

### Loss & Training
**Completely training-free**. All edits are performed via forward hooks during inference to replace activations with $m^*$. Downstream DPO training uses standard hyperparameters ($\beta=0.1$, lr 5e-7, batch 64, 1 epoch).

## Key Experimental Results

### Main Results
RewardBench bias subsets and overall (excerpt from Table 2, $B_b$ is biased subset / $\overline{B_b}$ is unbiased subset, accuracy %):

| RM / Method | $B_{\text{len}}$ | $\overline{B_{\text{len}}}$ | $\overline{B_{\text{para}}}$ | $\overline{B_{\text{over}}}$ | $\overline{B_{\text{excl}}}$ | ALL |
|-----------|-----:|-----:|-----:|-----:|-----:|----:|
| FsfairX (7B base) | 95.14 | 77.93 | 75.63 | 82.49 | 71.13 | 86.68 |
| FsfairX + LP | 93.45 | 85.12 | 86.57 | 85.99 | 77.32 | 89.67 |
| FsfairX + LWR | 93.45 | 85.95 | 87.04 | 86.38 | 78.35 | 90.08 |
| **FsfairX + CIRM** | **95.25** | 78.02 | 74.91 | 83.27 | 72.16 | 86.80 |
| INF-ORM-70B (SOTA) | 96.72 | 95.70 | 93.91 | 95.72 | 90.72 | 96.60 |

While LP / LWR appear to gain more on $\overline{B_{\text{len}}}$, they sacrifice points on $B_{\text{len}}$ (FsfairX 95.14→93.45). CIRM maintains or slightly improves both subsets, and on GRM, $\overline{B_{\text{para}}}$ increases while $B_{\text{para}}$ also rises to 93.69. Similar behavior is observed on RM-Bench (Table 3), where LP/LWR trade off between Easy and Hard subsets, while CIRM keeps Easy stable and maintains Hard performance.

Downstream DPO + AlpacaEval 2.0 / MT-Bench (excerpt from Table 4, Llama-3-8B-Instruct):

| Reward model | LCWR | WR | length | MT-Bench |
|---|---:|---:|---:|---:|
| GRM (2B) | 37.53 | 47.47 | 2193 | 7.45 |
| GRM + LP | 44.49 | 40.18 | 1571 | 7.29 |
| GRM + LWR | 39.77 | 47.59 | 2119 | 7.58 |
| **GRM + CIRM** | 41.89 | 50.13 | 2201 | 7.53 |
| FsfairX (7B) | 37.78 | 49.74 | 2368 | 7.64 |
| FsfairX + LP | 44.03 | 46.88 | 1881 | 7.60 |
| FsfairX + LWR | 43.11 | 47.07 | 1929 | 7.44 |
| **FsfairX + CIRM** | 39.49 | **51.19** | 2345 | 7.62 |
| INF (70B SOTA) | 40.63 | 49.61 | 2201 | 7.42 |

The WR 51.19 for 7B + CIRM exceeds the 70B INF score of 49.61, with comparable MT-Bench results.

### Ablation Study
Table 5: Removing intervention for specific bias types sequentially (FsfairX + Llama3-8B):

| Configuration | LCWR | WR | MT-Bench |
|------|-----:|---:|---------:|
| CIRM (All 5) | 39.49 | **51.19** | 7.62 |
| −len | 37.10 | 49.83 | 7.81 |
| −para | 38.20 | 50.89 | 7.53 |
| −over | 40.02 | 50.24 | 7.54 |
| −excl | 37.06 | 50.21 | 7.56 |
| −bold | 38.38 | 50.06 | 7.29 |

Removing any bias intervention disrupts the overall balance across LCWR/WR/MT-Bench, proving the necessity of joint handling. Table 6 verifies that CIRM slightly reduces the biased proportion in GRM labeling (len 54.85→51.71, para 55.88→52.14) compared to the aggressive reduction by LP/LWR (len 25.65); this "moderate suppression" is why MT-Bench remains stable.

### Key Findings
- **Bias-specific neurons are concentrated in shallow layers** (Figures 4-5): Neurons for the 5 bias types are primarily located in early transformer layers of the RM, mostly in query/up/gate projections. Consistent with Meng et al. 2022's hypothesis on up-projections for knowledge retrieval, this suggests RMs place "surface style" in early fast-paths, facilitating local editing.
- **CIRM avoids the biased / unbiased trade-off**: LP drops Easy accuracy on RM-Bench from 88.43 to 82.39 while increasing Hard accuracy to 59.55; CIRM maintains Easy at 88.95 and Hard at 49.04. It only disables bias dependency without affecting semantic signals for correct comparisons.
- **2B / 7B + CIRM ≈ 70B INF**: 8B RMs with CIRM perform on par with the 70B SOTA RM on AlpacaEval LCWR and MT-Bench, implying that "debiasing" may contribute more to downstream DPO value than simply increasing RM scale.
- **Downstream LLM bias is also suppressed** (Table 7): After DPO, Gemma-2 using vanilla GRM labels amplifies length (1323→1538) and bolding (14.82→21.22). Using CIRM labels results in moderate changes (1323→1511) without pushing exclamation marks significantly higher, unlike LP/LWR.
- **TruthfulQA Improvement** (Table 8): Three out of four groups show increased truthfulness after CIRM, indicating that eliminating stylistic bias forces the RM to focus more on content veracity, benefiting downstream factuality.

## Highlights & Insights
- **Interpreting BT reward as TE estimation and substituting with CDE is a clean causal narrative**: This is the first formal connection of causal mediation analysis to RLHF reward modeling, providing a formal language for the "style vs. content" decomposition in rewards.
- **"Median activation replacement" outperforms zeroing or swapping**: Experimental results for three $m^*$ choices show the median is most effective, likely acting as a controlled value robust to abnormal activations. This insight is valuable for all neuron intervention work (Vig 2020, Kojima 2024).
- **Joint $k$ search reveals interactions between biases**: Paragraph bias in $\overline{B_{\text{para}}}$ was slightly harmed by CIRM, yet removing it caused downstream DPO to drop, suggesting that RM benchmark subset accuracy is not always a reliable proxy for downstream alignment.

## Limitations & Future Work
- Only 5 predefined bias types are covered (length / paragraph / overlap / exclamation / bold); other stylistic biases like lists, links, or emojis, or task-specific biases, require manual addition of surface features.
- Bias-specific neuron identification relies on a 500-sample validation set and hyperparameter search, risking overfitting; the optimal $k$ varies significantly across RMs (GRM 21k vs. FsfairX 1.9k).
- The causal graph is simplified to $x\to m\to r$; intervention efficiency might decrease if bias signals are widely distributed across multiple hidden states.
- Downstream alignment relies on LLM-as-judge (GPT-4o / GPT-4-turbo), which may inherit similar biases.
- The method depends on the RM's "last token activation" for correlation analysis, necessitating redesigns for encoder-only or non-autoregressive RMs.

## Related Work & Insights
- **vs LP (Dong et al. 2024) / LWR (Huang et al. 2024)**: Also training-free, but only handle length via reward-level subtraction. CIRM is a neuron-level causal intervention handling 5 bias types without trade-offs.
- **vs ODIN / InfoRM / Park et al. 2024a**: These require retraining RMs (adding heads, info-bottlenecks, or data augmentation); CIRM is performed entirely at inference time.
- **vs Vig et al. 2020 / Meng et al. 2022 / Kojima et al. 2024**: Those used causal mediation for single-class attributes (gender, knowledge, language); this work extends it to RLHF reward modeling and handles multiple biases jointly.
- **vs RM Ensemble (Eisenstein 2024) / WARM (Rame 2024)**: These use averaging for robustness, requiring multiple RMs; CIRM works with a single RM and is easier to deploy.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Interpreting BT as TE and using CDE to intervene in RM neurons is highly novel and lightweight in implementation.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multidimensional verification across RM benchmarks, downstream DPO, residual bias, and TruthfulQA; however, more RM architectures could be included.
- Writing Quality: ⭐⭐⭐⭐ Causal graphs and formulas are clear; case studies are intuitive.
- Value: ⭐⭐⭐⭐⭐ Enabling 7B RMs to reach 70B SOTA alignment quality directly benefits industrial RLHF pipelines.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Inference-time Alignment in Continuous Space](../../NeurIPS2025/llm_alignment/inference-time_alignment_in_continuous_space.md)
- [\[AAAI 2026\] W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search](../../AAAI2026/llm_alignment/w2s-aligntree_weak-to-strong_inference-time_alignment_for_large_language_models_.md)
- [\[ACL 2026\] Student Guides Teacher: Weak-to-Strong Inference via Spectral Orthogonal Exploration](student_guides_teacher_weak-to-strong_inference_via_spectral_orthogonal_explorat.md)
- [\[ACL 2026\] AgentV-RL: Scaling Reward Modeling with Agentic Verifier](agentv-rl_scaling_reward_modeling_with_agentic_verifier.md)
- [\[ACL 2026\] ConsistRM: Improving Generative Reward Models via Consistency-Aware Self-Training](consistrm_improving_generative_reward_models_via_consistency-aware_self-training.md)

</div>

<!-- RELATED:END -->
