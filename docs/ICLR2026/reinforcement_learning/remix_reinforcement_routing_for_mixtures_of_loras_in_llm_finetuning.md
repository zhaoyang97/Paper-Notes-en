---
title: >-
  [Paper Note] Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models
description: >-
  [ICLR 2026][Reinforcement Learning][Layer Pruning] Layer pruning in LLMs is formulated as a cooperative game (each layer = player, model performance = utility) → exact Shapley value computation is infeasible ($2^L$ combinations) → a two-stage approximation is proposed: (1) stratified Monte Carlo sampling generates masks + evaluates PPL as supervision signals → (2) a lightweight surrogate network is trained to predict the performance of arbitrary masks → efficient per-layer Shapley value estimation → captures inter-layer dependencies → substantially outperforms static heuristic pruning baselines.
tags:
  - ICLR 2026
  - Reinforcement Learning
  - Layer Pruning
  - Cooperative Game
  - Shapley Value
  - Surrogate Network
  - Monte Carlo Sampling
  - Depth Pruning
date: 2026-05-08
content_hash: fb0575484342c087
---

# Pruning as a Cooperative Game: Surrogate-Assisted Layer Contribution Estimation for Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2602.07804](https://arxiv.org/abs/2602.07804)
**Code**: [GitHub](https://github.com/920927/Pruning_As_A_Cooperative_Game)
**Area**: Reinforcement Learning
**Keywords**: Layer Pruning, Cooperative Game, Shapley Value, Surrogate Network, Monte Carlo Sampling, Depth Pruning

## TL;DR
Layer pruning in LLMs is formulated as a cooperative game (each layer = player, model performance = utility) → exact Shapley value computation is infeasible ($2^L$ combinations) → a two-stage approximation is proposed: (1) stratified Monte Carlo sampling generates masks + evaluates PPL as supervision signals → (2) a lightweight surrogate network is trained to predict the performance of arbitrary masks → efficient per-layer Shapley value estimation → captures inter-layer dependencies → substantially outperforms static heuristic pruning baselines.

## Background & Motivation

**Background**: LLM inference is costly → model compression is critical → depth pruning (layer pruning) directly removes entire Transformer layers → simpler to implement and yields more direct inference speedup than width pruning.

**Limitations of Prior Work**:
   - (1) **Static heuristic rules**: Existing methods score each layer using weight magnitude, activation norms, sensitivity analysis, etc. → assume layer importance is fixed and independent → in practice, layer importance is context-dependent.
   - (2) **Inter-layer dependencies are ignored**: Removing one layer alters the relative importance of others → importance rankings obtained from single-layer evaluation fluctuate drastically under multi-layer pruning (Fig. 1) → intermediate layer rankings are especially unstable.
   - (3) **Greedy strategies are not globally optimal**: Pruning layers one by one according to individual importance scores cannot find the optimal combination → e.g., the two individually least important layers (Layer 27+10) yield PPL = 15.4535, whereas the combination (Layer 10+11) yields PPL = 15.4279, which is superior (Tab. 1).
   - (4) **Re-evaluation is still insufficient**: Recomputing importance after each pruning step may still miss the globally optimal layer combination, as layer interactions are not explicitly modeled.

**Key Insight**: Reconsidering the problem from a game-theoretic perspective → Shapley values in cooperative games naturally capture interaction contributions among players → but direct computation is infeasible for LLMs → a scalable approximation is needed.

**Core Problem**: How to accurately estimate each layer's marginal contribution to model performance within tractable computation, while accounting for inter-layer dependencies?

**Mechanism**: Replace expensive full-model evaluations with a surrogate network → training data is collected from stratified mask–performance pairs → the surrogate generalizes to unseen masks → enabling large-scale Shapley value estimation.

**Key Insight**: Layer importance is not a fixed scalar → it depends on which other layers are retained → only a game-theoretic framework can systematically model such "coalition-dependent" contributions.

## Method

### 1. Cooperative Game Formulation

Layer pruning of an $L$-layer LLM is formalized as a cooperative game:
- **Player set**: $\mathcal{L} = \{1, 2, \dots, L\}$ (each Transformer layer is a player)
- **Utility function**: $u(S)$ is the PPL of the model retaining layer subset $S$ on calibration data (lower is better)
- **Marginal contribution**: Layer $i$'s contribution to subset $S$ is $\Delta_i(S) = u(S \cup \{i\}) - u(S)$
- **Shapley value**: The weighted average of marginal contributions over all possible coalitions → fairly quantifies each layer's contribution → but requires enumerating $2^L$ subsets → infeasible in practice

### 2. Stage 1: Mask Generation and Performance Evaluation

**Stratified Monte Carlo Sampling**:
- Binary mask $\mathbf{m} \in \{0, 1\}^L$ → $\mathbf{m}_i = 1$ denotes retaining layer $i$
- Masks are stratified by Hamming weight $k(\mathbf{m}) = \sum_{i=1}^L m_i$ → $N_{k_j}$ masks are uniformly sampled for each weight $k_j$
- Ensures adequate coverage across different pruning ratios → avoids bias toward any particular compression level

**Performance Scoring**:
$$s(\mathbf{m}) = \frac{\text{PPL}_{\text{orig}}}{\text{PPL}(M(\mathbf{m}))}$$
- $s(\mathbf{m})$ closer to 1 → smaller performance loss after pruning
- Each sampled mask is applied to the pruned model → PPL on calibration data is computed → forming training dataset $\{(\mathbf{m}_n, s(\mathbf{m}_n))\}$

### 3. Stage 2: Surrogate Training and Shapley Value Estimation

**Surrogate Network $f_\theta$**:
- Architecture: two-layer feedforward network (extremely lightweight)
- Input: binary mask $\mathbf{m}$ → Output: predicted performance score $f_\theta(\mathbf{m})$
- Training objective: MSE loss
$$\mathcal{L}(\theta) = \frac{1}{N} \sum_{n=1}^N \left(f_\theta(\mathbf{m}_n) - s(\mathbf{m}_n)\right)^2$$
- After training → predicts the performance of any mask at negligible cost → no further full LLM inference required

**Shapley Value Approximation**:
- Computed via the surrogate network → marginal contributions of each layer are estimated from a large number of candidate masks
$$\hat{\phi}_i = \frac{1}{Q} \sum_{q=1}^Q \left(f_\theta(\mathbf{m}^{(k_j,q)} \cup \{i\}) - f_\theta(\mathbf{m}^{(k_j,q)})\right)$$
- $Q$ can be set large → since surrogate inference is extremely fast → more samples yield more accurate estimates

**Layer Pruning Decision**: Layers are ranked by Shapley values $\{\hat{\phi}_i\}_{i=1}^L$ → layers with the lowest contributions are removed until the target compression ratio is reached.

### Key Designs

- **Stratified vs. uniform sampling**: Uniform sampling concentrates masks near the median Hamming weight → insufficient coverage of extreme pruning ratios → stratified sampling ensures representation at every pruning ratio.
- **Surrogate generalization**: Although trained on only $N$ masks, the surrogate accurately predicts the performance of arbitrary combinations from $2^L$ → layer interactions follow low-order patterns that a shallow MLP can effectively capture.
- **Two-stage decoupling**: Expensive LLM evaluations are performed only once (Stage 1) → large-scale Shapley estimation uses the cheap surrogate (Stage 2) → computationally highly efficient.

## Key Experimental Results

### Language Modeling (PPL Comparison)

| Method | LLaMA-2-7B Remove 3L WikiText2 | Remove 6L WikiText2 | Remove 9L WikiText2 | Remove 12L WikiText2 |
|------|:---:|:---:|:---:|:---:|
| SliceGPT | 108.10 | 212.89 | 291.85 | 393.89 |
| SLEB | 14.24 | 19.47 | 27.45 | 58.12 |
| Shortened-LLaMA | 16.65 | 36.37 | 81.96 | 304.52 |
| ShortGPT | 16.65 | 36.37 | 81.96 | 157.99 |
| **Ours** | **14.69** | **18.87** | **24.61** | **38.12** |

→ The advantage is most pronounced when removing 12 layers: Ours (38.12) vs. SLEB (58.12) vs. ShortGPT (157.99) → the value of modeling inter-layer dependencies becomes increasingly evident at high compression ratios.

### Meta-LLaMA-3-8B (High Compression Ratio Comparison)

| Method | Remove 3L WikiText2 | Remove 6L WikiText2 | Remove 9L WikiText2 | Remove 12L WikiText2 |
|------|:---:|:---:|:---:|:---:|
| SLEB | 20.40 | 33.64 | 63.83 | 126.94 |
| Shortened-LLaMA | 20.72 | 79.44 | 5928.34 | 15138.55 |
| ShortGPT | 23.85 | 84.56 | 2549.75 | 15138.55 |
| **Ours** | **18.58** | **25.39** | **45.26** | **304.52** |

→ Baselines collapse on LLaMA-3 when removing 9+ layers (PPL > 2000) → the proposed method maintains 45.26 → an order-of-magnitude gap.

### Zero-shot Performance (LLaMA-2-7B, Average over 8 Tasks)

| Params | SliceGPT | SLEB | Shortened-LLaMA | ShortGPT | **Ours** |
|--------|:---:|:---:|:---:|:---:|:---:|
| 6.1B | 0.4430 | 0.5635 | 0.5816 | 0.5709 | **0.5782** |
| 5.5B | 0.3865 | 0.5138 | 0.5050 | 0.5050 | **0.5227** |
| 4.9B | 0.3645 | 0.4543 | 0.4506 | 0.4506 | **0.4689** |
| 4.3B | 0.3441 | 0.3812 | 0.3640 | 0.3911 | **0.3951** |

### Non-Transformer Architectures (RWKV-7B / Mamba-2.8B)

| Model | Params | ShortGPT PPL_Wiki | **Ours PPL_Wiki** |
|------|--------|:---:|:---:|
| RWKV-7B | 6.2B | 38.72 | **34.17** |
| RWKV-7B | 5.6B | 90.02 | **56.33** |
| Mamba-2.8B | 2.5B | 378.99 | **24.23** |
| Mamba-2.8B | 2.3B | 4074.49 | **31.11** |

→ The advantage on Mamba is striking: ShortGPT collapses at 2.3B (PPL > 4000) → the proposed method achieves only 31.11.

## Key Findings

1. **Layer importance is context-dependent**: Single-layer pruning rankings fluctuate drastically under multi-layer pruning → intermediate layers are especially unstable → demonstrating that static heuristics are fundamentally inadequate for multi-layer pruning.

2. **Greedy pruning is not globally optimal**: Even with importance recomputation after each pruning step, the optimal combination may be missed → Tab. 1 clearly shows that (Layer 10+11) outperforms the greedily selected (Layer 27+10).

3. **Strong surrogate generalization**: Trained on a limited set of mask–performance pairs, the surrogate accurately predicts the performance of unseen mask combinations → indicating that inter-layer interaction patterns can be effectively captured by a low-order model.

4. **Advantage amplifies at high compression ratios**: As the number of pruned layers increases, baseline performance degrades rapidly while the proposed method remains stable → highlighting the core value of modeling inter-layer dependencies.

5. **Cross-architecture generalization**: The method is effective on non-Transformer architectures RWKV and Mamba → suggesting that inter-layer dependency is not specific to Transformers → the cooperative game framework is broadly applicable.

6. **Compatibility with quantization**: Applying quantization before pruning yields better results than the reverse order → because pruning decisions are based on the quantized model representation, which is closer to the final inference state.

## Highlights & Insights

- **Novelty of the game-theoretic perspective**: Layer pruning is elevated from "scoring layers individually" to "coalition contribution analysis" → Shapley values inherently account for all possible layer combinations → the theoretical foundation is substantially more rigorous than heuristic approaches.
- **Elegance of the surrogate network design**: A minimal two-layer MLP replaces expensive LLM evaluations → the training set requires only hundreds of masks → inference cost is nearly zero → enabling large-scale Shapley estimation.
- **Necessity of stratified sampling**: Uniform sampling concentrates most masks near $L/2$ → extreme pruning ratios are underrepresented → stratified sampling ensures adequate coverage at every pruning ratio → improving surrogate prediction accuracy across all compression levels.
- **Experimental comprehensiveness**: The evaluation spans multiple models (LLaMA-2/3, Vicuna, RWKV, Mamba), multiple benchmarks (WikiText2/PTB/C4 + 8 zero-shot tasks + ANLI), multiple compression ratios, and quantization compatibility → highly convincing.

## Limitations & Future Work

- **Stage 1 computational overhead**: Although surrogate inference is extremely fast, Stage 1 still requires full LLM inference over hundreds of masks → the calibration cost is non-negligible.
- **Surrogate generalization assumption**: The approach assumes that inter-layer interactions can be captured by a shallow MLP → this may not hold for very deep models or specialized architectures → theoretical guarantees for surrogate generalization are lacking.
- **Layer-level pruning only**: The combination with finer-grained head/channel pruning is not explored → a hybrid layer + width pruning strategy may yield further improvements.
- **Limited fine-tuning recovery in main experiments**: Main results primarily report post-pruning direct evaluation → LoRA-based recovery is relegated to the appendix → fine-tuning is typically required in practical deployment → the main experimental setup is therefore incomplete.

## Related Work & Insights

### vs. ShortGPT (Men et al., 2024)
ShortGPT uses Block Influence (BI) as a static measure of layer importance → evaluated layer by layer → inter-layer interactions are ignored. The proposed method models inter-layer dependencies via cooperative game theory → outperforms ShortGPT across all settings → ShortGPT collapses entirely at high compression ratios while the proposed method maintains reasonable PPL.

### vs. SLEB (Song et al., 2024)
SLEB performs iterative pruning → removing the least important block at each step → incorporating some degree of inter-layer consideration. However, it remains fundamentally a greedy strategy → making locally optimal decisions at each step → unable to guarantee global optimality. The proposed method globally evaluates all layer contributions via Shapley values → outperforms SLEB in most settings.

### vs. GTAP (Diaz-Ortiz Jr et al., 2023)
GTAP also employs cooperative game theory → but operates at the neuron granularity → using power indices to evaluate importance → computational complexity limits its scalability to large models. The proposed method operates at the layer granularity → drastically reduces computational cost via a surrogate network → making game-theoretic methods applicable to LLM-scale models for the first time.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The cooperative game + surrogate network framework for layer pruning is novel → Shapley values are not new to ML, but the contribution lies in the scalable design.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Six model families, 12+ benchmarks, multiple compression ratios, non-Transformer architectures, quantization compatibility → extremely comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Motivation is clear and method description is complete → some sections are formula-dense → overall readability is above average.
- **Value**: ⭐⭐⭐⭐ High practical value → open-source code → directly applicable as a drop-in replacement for existing layer pruning methods → particularly suited for high compression ratio scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Efficient Estimation of Kernel Surrogate Models for Task Attribution](efficient_estimation_of_kernel_surrogate_models_for_task_attribution.md)
- [\[ICLR 2026\] Robust Multi-Objective Controlled Decoding of Large Language Models](robust_multi-objective_controlled_decoding_of_large_language_models.md)
- [\[ICLR 2026\] VerifyBench: Benchmarking Reference-based Reward Systems for Large Language Models](verifybench_benchmarking_reference-based_reward_systems_for_large_language_model.md)
- [\[ICLR 2026\] AWM: Accurate Weight-Matrix Fingerprint for Large Language Models](awm_accurate_weight-matrix_fingerprint_for_large_language_models.md)
- [\[ICLR 2026\] GraphOmni: A Comprehensive and Extensible Benchmark Framework for Large Language Models on Graph-theoretic Tasks](graphomni_a_comprehensive_and_extensible_benchmark_framework_for_large_language_.md)

</div>

<!-- RELATED:END -->
