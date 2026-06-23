---
title: >-
  [Paper Note] Mode-conditioning unlocks superior test-time compute scaling
description: >-
  [ICLR 2026][LLM Reasoning][Pass@k] Addressing the "diversity collapse" problem in parallel sampling—where models collapse into a single reasoning strategy and repeatedly commit the same errors—this paper proposes the **Mode-conditioning (ModC)** framework. By using expert models or mode prefixes to explicitly distribute test-time compute across differen
tags:
  - ICLR 2026
  - LLM Reasoning
  - Pass@k
date: 2026-05-08
content_hash: aebedd66c2255275
---
# Mode-conditioning unlocks superior test-time compute scaling

**Conference**: ICLR 2026  
**OpenReview**: [https://openreview.net/forum?id=JzkdJQzPw1](https://openreview.net/forum?id=JzkdJQzPw1)  
**Code**: To be confirmed  
**Area**: LLM Reasoning / Test-time Compute Scaling  
**Keywords**: Parallel Sampling, Diversity Collapse, Pass@k, Mode-conditioning, Test-time Scaling, Distillation  

## TL;DR
Addressing the "diversity collapse" problem in parallel sampling—where models collapse into a single reasoning strategy and repeatedly commit the same errors—this paper proposes the **Mode-conditioning (ModC)** framework. By using expert models or mode prefixes to explicitly distribute test-time compute across different reasoning modes, the framework lifts the Pass@k scaling curves in mathematical reasoning and graph search tasks, achieving approximately a 4× improvement in inference efficiency.

## Background & Motivation
- **Background**: Parallel sampling (sampling $k$ independent paths for the same problem and taking the best) is a pillar of test-time scaling and RL, particularly effective in automatically verifiable domains like mathematics, code, and scientific discovery. Its performance is characterized by $\text{Pass@}k_{\text{std}}(x)=1-(1-p_x)^k$.
- **Limitations of Prior Work**: Both SFT and RL induce **diversity collapse**, where models concentrate on a few modes. This leads to redundant samples that repeat the same mistake, resulting in diminishing marginal returns for additional compute. For some problems, the success probability $p_x$ of a viable strategy is suppressed to near-zero, requiring impractical sampling volumes.
- **Key Challenge**: Previous attempts to mitigate collapse (Pass@k training, weight regularization) mostly rely on **token-level temperature sampling** to generate diversity. However, temperature sampling only alters local token distributions and cannot guarantee coverage at the level of "high-level reasoning modes." The authors illustrate this with the Countdown task: a problem might be solvable via DFS but not BFS (or vice-versa), yet standard models often stabilize on one mode per problem, leading to failure if the wrong mode is chosen.
- **Goal**: Instead of modifying fine-tuning objectives to "prevent collapse," this paper takes a complementary route—**explicitly distributing the sampling budget across multiple modes** to force coverage of both dominant strategies and alternatives that might succeed where the dominant one fails.
- **Core Idea**: **[Mode-conditioning]** organizes test-time scaling around "multiple reasoning modes." It provides two practical training methods (Expert Models / Mode Prefixes) to reliably sample specific modes. When mode labels are unavailable, **gradient clustering** is used for automatic discovery.

## Method

### Overall Architecture
The starting point for ModC is a simple inequality: rather than repeatedly sampling from the model's own "uncertain mixture" distribution, it is better to partition the $k$ budget evenly across several known modes. During training, data is partitioned by mode, and the model learns to "switch modes via instruction" (implemented via experts or prefixes). During inference, the sampling budget is split equally among modes. If labels are absent, training samples are first grouped into $C$ "modes" via gradient clustering before following the same ModC pipeline.

```mermaid
flowchart TD
    A[Training Data] --> B{Mode labels available?}
    B -- Yes (Search algorithm/Teacher ID) --> C[Partition data by mode]
    B -- No --> D[Gradient Clustering<br/>∇θ log pθ y x → Random Projection → C clusters]
    D --> C
    C --> E1[Implementation 1: Train independent expert models]
    C --> E2[Implementation 2: Single model + Mode prefix Mode k]
    E1 --> F[Inference: Split k budget across modes]
    E2 --> F
    F --> G[Pass@k Lift / ~4× Efficiency]
```

### Key Designs

**1. Uniform mode budget is strictly superior to free mixed sampling: Theoretical guarantee.** The validity of ModC stems from a provable advantage. Suppose two modes have success probabilities $p_{1,x}, p_{2,x}$ on input $x$. After splitting the budget, the overall success rate is $\text{Pass@}k_{\text{ModC}}(x)=1-(1-p_{1,x})^{k/2}(1-p_{2,x})^{k/2}$. Conversely, if the model does not know which mode is better and mixes them with random weights $w_x$ (where $\mathbb{E}[w_x]=1/2$), the single-sample success rate is $w_x p_{1,x}+(1-w_x)p_{2,x}$. Since this function is concave with respect to $w$, Jensen’s inequality gives $\mathbb{E}_{w_{x}}[\text{Pass@}k_{\text{std}}(x;w_x)]\le \text{Pass@}k_{\text{std}}(x;1/2)$. Furthermore, as long as $p_{1,x}\ne p_{2,x}$, then $(1-p_{1,x})^{k/2}(1-p_{2,x})^{k/2}<(1-p_x)^k$, thus $\text{Pass@}k_{\text{ModC}}(x)>\text{Pass@}k_{\text{std}}(x;1/2)$. Conclusion: Even if the model's preference follows any distribution centered at 0.5, explicit uniform distribution is strictly better—this is the root cause for ModC gaining Pass@k gains solely through "compute allocation."

**2. Expert Model implementation: Strong specialization, low correlation in errors.** The most direct implementation is to partition training data by strategy and train an independent model on each subset (keeping total data and compute constant). At test time, the budget is split (e.g., $k/2$ per mode). This partitioning naturally makes each expert highly specialized and their errors less correlated, magnifying the gains of parallel sampling. The trade-off is the lack of knowledge sharing. On tasks like Countdown, where modes differ significantly and sharing requirements are low, expert models outperform prefixes (widening the gap by up to 20% Pass@1024 on adversarial sets).

**3. Mode Prefix implementation: Intra-model specialization, cross-mode knowledge sharing.** To compensate for the lack of knowledge sharing in expert models, the authors adopt the conditional token approach from controllable text generation. Discrete condition tokens (e.g., `[Mode 1]`, `[Mode 2]`, or teacher identity tokens) are prepended to the input. The model is trained to bind each prefix to a reasoning strategy. Inference involves uniform sampling across prefixes. This allows the model to specialize while sharing linguistic/mathematical foundations. In mathematical reasoning (Short/Long CoT), prefixes generally outperform expert models, confirming that "knowledge sharing is more critical in math tasks."

**4. Automatic mode discovery via gradient clustering: Eliminating label dependency.** Real-world data rarely comes with clean labels like DFS/BFS or teacher identities. The authors compute gradients $g_\theta(x,y)=\nabla_\theta \log p_\theta(y|x)$ for each training sample $(x,y)$, reduce dimensions using Rademacher random projection, and perform K-means clustering into $C$ clusters. Samples in the same cluster are treated as the same "mode." On NuminaMath, this automatic discovery without extra information consistently improves Pass@k (up to ~10%), indicating that standard training underutilizes the implicit diversity in data.

## Key Experimental Results

### Main Results

| Configuration | Task/Benchmark | Baseline Comparison | ModC Gain |
|------|-----------|----------|-----------|
| Countdown Graph Search (Natural) | Pass@1024 | Standard Training | Experts up to +8% |
| Countdown Graph Search (Adversarial, single algorithm solvable) | Pass@1024 | Standard Training | Up to +20% |
| Short CoT Multi-teacher Distillation (NuminaMath→MATH500) | Pass@k | Mixture / Best Single Teacher | Qwen2.5-0.5B +10%, OLMo2-7B +15% (Prefix preferred) |
| Long CoT Multi-teacher Distillation (OpenThoughts→AIME2025) | Pass@k | QwQ-32B/DeepSeek-R1 Single Teacher | Exceeds best single teacher; k=256 matches standard k=1024 (~4× efficiency) |
| Automatic Mode Discovery (NuminaMath→MATH500) | Pass@k | Standard Training | Constant gain across 0.5B–7B, up to ~10% |

### Ablation Study

| Ablation Dimension | Key Finding |
|----------|----------|
| Training Data Balance (Rejection sampling bias DFS vs 50-50) | Even with balanced data, standard training makes highly unbalanced mode allocations; ModC centers the BFS ratio of each problem to ~0.5. |
| Model Scale (0.5B → 7B, Qwen2.5 / OLMo2) | ModC gains appear consistently across all scales. |
| Expert Models vs. Prefixes | Experts are better for Countdown (large mode variance); Prefixes are better for Math (requires knowledge sharing). |
| Random Partitioning Control | Randomly splitting data into groups sometimes helps, but is inferior to ModC using real modes. |
| ModC + RL / Pass@k RL | RL brings Pass@1 to the same level, but ModC leads from k=2; it provides additional gains on top of explicit Pass@k RL for anti-collapse. |

### Key Findings
- **Diverse training data is only useful when paired with a "mode-preserving" mechanism**: Standard mixtures of teachers often fail to beat the best single teacher (counter-intuitive), whereas ModC truly converts teacher diversity into stronger test-time scaling.
- **ModC enriches the solution space without sacrificing top output**: Unlike standard SFT, Pass@1 does not drop after RL, and ModC leads immediately at $k \ge 2$.
- **Efficiency gains are directly quantifiable**: In Long CoT, ModC matches the Pass@1024 of standard training using only 1/4 of the samples.

## Highlights & Insights
- **Shifting "Diversity" from token-scale to mode-scale**: While temperature sampling adjusts local distributions, ModC addresses high-level strategy coverage. This simple approach identifies the true bottleneck in parallel sampling failure.
- **Balance of Theory and Engineering**: Jensen's inequality provides a clean proof of why "uniform distribution is strictly better." Both implementations are lightweight and easy to integrate into existing distillation/SFT pipelines.
- **Gradient clustering decouples the method from labels**: This extends the applicability to "general data" and proves that standard training wastes implicit diversity in datasets.
- **Orthogonal and additive with RL**: ModC is an SFT-side intervention that can be stacked on top of standard RL or specialized anti-collapse Pass@k RL.

## Limitations & Future Work
- **RL for Prefix Variants is unexplored**: The authors suspect RL might break the "prefix ↔ mode" binding (requiring the prefix to follow the reward). RL experiments were only conducted for the expert variant; Prefix+RL is left for future work.
- **Number of modes and clustering granularity**: Hyperparameters like the number of clusters $C$ and projection dimensions significantly affect the quality of discovered "modes." A systematic selection criterion is lacking.
- **Predefined modes still carry human priors**: BFS/DFS in Countdown and teacher identities are clear scenarios. In more open tasks, defining "what constitutes a mode" remains blurry.
- **Optimality of uniform budget**: Theory proves "uniform is better than free mixing" but does not prove it is the optimal allocation when mode success rates are unknown. Adaptive allocation might yield further gains.

## Related Work & Insights
- **Improving Parallel Test-time Scaling**: Compared to Pass@k training, weight ensembling (Dang et al. 2025), diverse beam search, and diverse prompting, ModC follows a data-centric conditioning route by explicitly encoding expert modes into the model.
- **Specialized Training**: Similar to Mixture-of-Experts (MoE), but while MoE routes data to sub-components to save parameters, ModC uses the full model for all data and conditions only on the "output mode." The goal is strategy coverage, not compute savings.
- **Creativity and Diversity Research**: Echoes findings that "temperature sampling is only weakly correlated with creativity and introduces incoherence" and that "global planning/seed conditioning is critical for creative generation." ModC is a concrete realization of "seed/conditioning-enhanced diversity" for reasoning tasks.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Reframing diversity from tokens to "reasoning modes" with clean theoretical guarantees is a fresh and insightful perspective. The implementations themselves have appeared elsewhere, hence not a full score.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers graph search, short/long CoT, two model families, 0.5B–7B, labeled/unlabeled, and SFT/RL settings. Missing prefix+RL and optimal budget allocation experiments.
- **Writing Quality**: ⭐⭐⭐⭐ Clear progression from motivation to theory, implementation, and experiments. Visuals are intuitive. Some minor typos in sections.
- **Value**: ⭐⭐⭐⭐ Low implementation cost, additive benefits, and quantifiable efficiency gains (~4×) make it directly valuable for teams working on test-time scaling or distillation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Strategic Scaling of Test-Time Compute: A Bandit Learning Approach](strategic_scaling_of_test-time_compute_a_bandit_learning_approach.md)
- [\[ICLR 2026\] T1: Tool-Integrated Verification for Test-Time Compute Scaling in Small Language Models](t1_tool-integrated_verification_for_test-time_compute_scaling_in_small_language_.md)
- [\[ICLR 2026\] Zero-Overhead Introspection for Adaptive Test-Time Compute](zero-overhead_introspection_for_adaptive_test-time_compute.md)
- [\[ICLR 2026\] CaTS: Calibrated Test-Time Scaling for Efficient LLM Reasoning](cats_calibrated_test-time_scaling_for_efficient_llm_reasoning.md)
- [\[ICLR 2026\] Test-Time Scaling in Diffusion LLMs via Hidden Semi-Autoregressive Experts](test-time_scaling_in_diffusion_llms_via_hidden_semi-autoregressive_experts.md)

</div>

<!-- RELATED:END -->
