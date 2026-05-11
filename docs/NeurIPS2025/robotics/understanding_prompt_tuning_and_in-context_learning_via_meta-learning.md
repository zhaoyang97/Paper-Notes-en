---
title: >-
  [Paper Note] Understanding Prompt Tuning and In-Context Learning via Meta-Learning
description: >-
  [NeurIPS 2025][Robotics][prompt tuning] This paper systematically analyzes the theoretical foundations and limitations of prompt tuning from a Bayesian meta-learning perspective. It proves that soft prompts can achieve o…
tags:
  - "NeurIPS 2025"
  - "Robotics"
  - "prompt tuning"
  - "in-context learning"
  - "meta-learning"
  - "Bayesian inference"
  - "soft prompts"
date: 2026-05-08
content_hash: 25bc88e6566ae543
---

# Understanding Prompt Tuning and In-Context Learning via Meta-Learning

**Conference**: NeurIPS 2025  
**arXiv**: [2505.17010](https://arxiv.org/abs/2505.17010)  
**Code**: [GitHub](https://github.com/google-deepmind/thunnini)  
**Area**: Robotics  
**Keywords**: prompt tuning, in-context learning, meta-learning, Bayesian inference, soft prompts

## TL;DR

This paper systematically analyzes the theoretical foundations and limitations of prompt tuning from a Bayesian meta-learning perspective. It proves that soft prompts can achieve optimal adaptation on a single target task within the pretraining distribution, yet face fundamental limitations under multi-task mixture target distributions. Furthermore, soft prefixes can surpass the optimal hard-token sequence by manipulating activations outside the token space.

## Background & Motivation

One of the most impressive characteristics of large pretrained models is their capacity for rapid in-context adaptation—given a small number of tokens, a model can infer the current task and generate appropriate continuations without any weight update. This capability is referred to as In-Context Learning (ICL). Prompt Tuning is an important paradigm for adapting pretrained models to target tasks; however, existing methods are predominantly empirically driven, lacking a conceptual understanding of the underlying prompt mechanisms.

Specifically, the following key questions motivate this work:

**Theoretical conditions for optimal prompts**: Under what conditions does there exist a prompt that brings the prompted pretrained predictor to (near) Bayes-optimal performance on the target task?

**Soft prompts vs. hard prompts**: Why are soft prefixes (sequences of real-valued vectors) more effective than hard token sequences, and what is the underlying mechanism?

**Fundamental limitations of prompt tuning**: Under what circumstances is prompt tuning fundamentally insufficient, necessitating weight tuning?

Most existing work on prompt optimization remains at the empirical level. This paper aims to establish a unified theoretical framework. From the meta-learning perspective, memory-based meta-learning trains a parameterized sequence predictor by minimizing the log loss, yielding a Bayesian predictor over the pretraining distribution. The hallmark of this Bayesian predictor is maximally fast in-context adaptation. Accordingly, a prompt is essentially a conditioning mechanism applied to the Bayesian predictor for efficient adaptation to a target task.

## Method

### Overall Architecture

This paper constructs a unified theoretical framework spanning meta-learning, Bayesian inference, and prompt tuning, and validates the theory through controlled experiments on LSTMs and Transformers.

### Key Designs

1. **Bayesian Sequence Predictors and Meta-Learning**

   The core idea is that a neural network trained via the meta-learning loop (sample task → generate data → minimize log loss) converges to a Bayesian predictor. Given a task distribution $P(\tau)$ and conditional distribution $P(x_{1:N}|\tau)$, the marginal distribution (the Bayesian mixture) is:

    $\xi(x_n|x_{<n}) = \int P(x_n|x_{<n},\tau) P(\tau|x_{<n}) d\tau$

   The meta-learning objective minimizes $D_{KL}(\xi||\pi_\theta)$; when the network is sufficiently expressive and well-converged, $\pi_{\hat\theta}(x_n|x_{<n}) \approx \xi(x_n|x_{<n})$, meaning the network achieves Bayes-optimal prediction purely through its activations.

   Design Motivation: Treating the pretrained model as the product of implicit meta-learning, and leveraging properties of Bayesian inference to analyze prompt tuning.

2. **Theoretical Analysis of Prefix Tuning**

   A prefix $s_{1:L} \in \mathcal{S}^L$ is prepended to the observation sequence, and the optimization objective is:

    $\min_{s_{1:L} \in \mathcal{S}^L} \frac{1}{K} \sum_{k=1}^K \sum_{n=1}^N -\log P_\theta(x_n^k | x_{<n}^k, s_{1:L})$

   The paper examines four prefix approaches: hard token search (HardPT, $\mathcal{S}=\mathcal{A}$), probability simplex (SimplexPT), real-valued prefixes (RealPT), and soft prompts (SoftPT, $\mathcal{S}=\mathbb{R}^{d_{emb}}$).

   **Positive theoretical result**: When the target is a single task within the support of the pretraining distribution ($P^{Target}(\tau)=\delta(\tau=\tau^{Target})$ and $P^{Pre}(\tau^{Target})>0$), there exists a sufficiently long hard-token prefix that renders the predictor Bayes-optimal on the target task.

   **Negative theoretical result (Limitation I)**: When the target distribution is multimodal (e.g., a mixture of two tasks), the posterior collapses to a Dirac delta under infinite observations (for log-concave priors such as Beta-Bernoulli), making it impossible to achieve optimal prompting via a prefix.

   **Negative theoretical result (Limitation II)**: When the target includes "substantially novel" atomic tasks ($P^{Pre}(\tau^{Target})=0$), prefix tuning cannot approximate Bayes-optimal behavior.

3. **Mechanistic Analysis of Soft Prefixes**

   The key advantage of soft prefixes lies in their being off-distribution inputs, capable of manipulating network activations in ways that hard tokens cannot. Optimal prompting requires that the model's internal state after consuming the prefix constitute a sufficient statistic for the target distribution, without disrupting subsequent internal dynamics. Pretraining determines the state-update function and imposes strong constraints on hard-token inputs. By circumventing these constraints, soft prefixes can more effectively guide pretrained—and even untrained—neural predictors after careful tuning.

   Experimentally, the superiority of soft prompts is primarily attributable to the embedding dimensionality (128) being much larger than the input dimensionality (2), providing substantially more degrees of freedom. When the embedding dimension is reduced to 4, the performance gap between SoftPT and RealPT largely disappears. This suggests that in frontier LLMs—where input dimensionality typically exceeds embedding dimensionality—soft input tuning may be more effective than embedding tuning.

### Loss & Training

- **Pretraining**: 1,000 gradient steps, batch size 256, sequence length 100, learning rate 0.001
- **Tuning**: 1,000 steps, batch size 256 (K=256,000 sequences total), sequence length 50, learning rate 5e-3
- **Evaluation metric**: Expected cumulative regret, i.e., the excess log loss relative to the known data-generating probability:

$$\mathscr{R}_{\tilde\theta}^{P^{Target}}(N) = \mathbb{E}_{\tau^*} \mathbb{E}_{P(x_{1:N}|\tau^*)} [-\log\pi_{\tilde\theta}(x_{1:N}|s_{1:L}) + \log P(x_{1:N}|\tau^*)]$$

## Key Experimental Results

### Main Results

Experiments use coin-flip sequences. The pretraining distribution is a uniformly random biased coin (Beta(1,1) prior). Target tasks are divided into a single coin (bias=0.2) and a two-coin mixture.

| Category | Method | Single-Coin Target (Transformer) | Two-Coin Mixture Target (Transformer) |
|---|---|---|---|
| Prefix Tuning | HardPT (L=6) | Does not reach optimum | Does not reach optimum |
| Prefix Tuning | SoftPT (L=6) | **Achieves Bayes-optimum** | Improves but does not reach optimum |
| Weight Tuning | FullWT | Achieves Bayes-optimum | Achieves Bayes-optimum |
| Weight Tuning | LoRAWT | Achieves Bayes-optimum | Achieves Bayes-optimum |
| Baseline | TargetBayes | Optimal upper bound | Optimal upper bound |
| Baseline | NoTuning | Pretraining performance | Pretraining performance |

### Ablation Study

| Configuration | Key Metric | Remarks |
|---|---|---|
| SoftPT L=6 vs. L=25 (two-coin) | Only marginal improvement | Prefix length is not the bottleneck; theoretical limitation is the root cause |
| Embedding dim 128→4 | SoftPT advantage substantially reduced | High-dimensional embeddings are the primary source of soft prompt superiority |
| Untrained Transformer + SoftPT | Near Bayes-optimal | Soft prefixes can effectively "program" untrained networks |
| Untrained LSTM + SoftPT | Minimal effect | Fundamental difference between Transformers and LSTMs in this regard |
| Larger network (256-dim, 2-layer) | Consistent conclusions | Qualitative results are robust to model scale |

### Key Findings

1. Soft prompts (SoftPT) enable a pretrained network to achieve Bayes-optimal performance on a single target task, surpassing the optimal hard-token sequence of the same length with a prefix of only length 6.
2. Prompt tuning to a two-coin mixture distribution is theoretically and empirically infeasible, even when the prefix length is extended to 25.
3. Soft prefixes can induce **untrained** Transformers to exhibit approximately well-behaved sequence prediction, whereas LSTMs do not exhibit this property.
4. Weight tuning methods (FullWT, LoRA) are not subject to the theoretical limitations of prompt tuning and can successfully adapt to mixture distributions.

## Highlights & Insights

- Establishes a unified theoretical framework spanning meta-learning → Bayesian inference → prompt tuning.
- Provides the first formal proof of the fundamental limitation of prompt tuning under multimodal target distributions.
- The success of soft prefixes is supported not only by Bayesian theory but also by a mechanistic explanation (manipulation of off-distribution activations).
- PCA visualizations of internal states clearly illustrate how different tuning methods affect network dynamics.
- Despite the simplicity of the experimental setting (coin flipping), the analysis captures the essential problems of prompt tuning.

## Limitations & Future Work

- Experiments rely on simple Bernoulli tasks and small networks; extrapolation to frontier model scales should be approached with caution.
- Theoretical guarantees hold strictly for data within the pretraining distribution; out-of-distribution generalization requires further investigation.
- What constitutes a "task" for LLMs remains unclear, making it difficult to assess the practical impact of the multimodal limitation.
- The transferability of soft prompts across different models is not explored.
- The Bayesian perspective does not exhaustively characterize all in-context learning phenomena at every scale.

## Related Work & Insights

- Consistent with the findings of Petrov et al. (2024): prompt tuning methods can "elicit skills already present in the pretrained model" but cannot acquire new skills.
- Extends the generalized understanding of in-context learning from Lampinen et al. (2024).
- Complementary to the compression perspective of Deletang et al. (2024)—log loss minimization is equivalent to maximizing lossless compression.
- Motivates a future direction: distilling large numbers of in-context demonstrations into more effective tuned soft prefixes.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The theoretical framework is unified and elegant, though the experimental setup is relatively simple.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Controlled experiments are comprehensive, but scale is limited.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Logic is clear; theory and experiments are tightly integrated.
- **Value**: ⭐⭐⭐⭐ Provides important foundational theory for understanding prompt tuning, though a gap to practical application remains.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Policy Compatible Skill Incremental Learning via Lazy Learning Interface](policy_compatible_skill_incremental_learning_via_lazy_learning_interface.md)
- [\[ACL 2026\] DeCoVec: Building Decoding Space based Task Vector for Large Language Models via In-Context Learning](../../ACL2026/robotics/decovec_building_decoding_space_based_task_vector_for_large_language_models_via_.md)
- [\[NeurIPS 2025\] Learning Spatial-Aware Manipulation Ordering](learning_spatial-aware_manipulation_ordering.md)
- [\[NeurIPS 2025\] Zero-Shot Embedding Drift Detection: A Lightweight Defense Against Prompt Injections in LLMs](zero-shot_embedding_drift_detection_a_lightweight_defense_against_prompt_injecti.md)
- [\[NeurIPS 2025\] MindForge: Empowering Embodied Agents with Theory of Mind for Lifelong Cultural Learning](mindforge_empowering_embodied_agents_with_theory_of_mind_for_lifelong_cultural_l.md)

</div>

<!-- RELATED:END -->
