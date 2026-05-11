---
title: >-
  [Paper Note] Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment
description: >-
  [ICLR 2026][LLM Alignment][Wasserstein distance] This paper identifies a fundamental limitation of standard KL divergence regularization in RLHF: it compares token probabilities only at identical index positions…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "Wasserstein distance"
  - "RLHF regularization"
  - "semantic awareness"
  - "optimal transport"
  - "Sinkhorn algorithm"
date: 2026-05-08
content_hash: 6ef62e5337e36a72
---

# Semantic-aware Wasserstein Policy Regularization for Large Language Model Alignment

**Conference**: ICLR 2026
**arXiv**: [2602.01685](https://arxiv.org/abs/2602.01685)
**Code**: [https://github.com/aailab-kaist/WPR](https://github.com/aailab-kaist/WPR)
**Area**: Alignment / RLHF
**Keywords**: Wasserstein distance, RLHF regularization, semantic awareness, optimal transport, Sinkhorn algorithm

## TL;DR
This paper identifies a fundamental limitation of standard KL divergence regularization in RLHF: it compares token probabilities only at identical index positions, completely ignoring semantic similarity. The authors propose Wasserstein Policy Regularization (WPR), a semantic-aware policy regularization based on entropy-regularized Wasserstein distance. Through a dual formulation, WPR converts the regularization into token-level penalty terms compatible with standard RL algorithms such as PPO, and consistently outperforms KL divergence and various f-divergence baselines on dialogue generation and summarization tasks.

## Background & Motivation

RLHF is the dominant paradigm for LLM alignment. Its standard pipeline scores responses with a reward model while applying KL divergence regularization to prevent the policy from deviating too far from the reference model. KL divergence is widely adopted in practice because it can be computed directly from the token probabilities of the two policies and integrated seamlessly into PPO training.

However, KL divergence and other f-divergences (e.g., JS, $\chi^2$, TV) share a fundamental limitation: **they compare token probabilities only at identical index positions, entirely ignoring semantic relationships between tokens**.

The paper illustrates this with an intuitive example. Suppose the vocabulary is {cat, kitten, dog, table}, and the reference policy concentrates its probability mass on "cat." Policy 1 concentrates on "kitten" and Policy 2 concentrates on "table." Semantically, "cat" and "kitten" are closely related, whereas "cat" and "table" share no meaningful connection. Yet KL divergence assigns an extremely large value to Policy 1 due to support mismatch (penalizing it unfairly), while JS divergence assigns identical distances to both Policy 1 and Policy 2—neither captures the semantic distinction.

**Key Challenge**: KL/f-divergences perform index-wise comparisons and are entirely unable to exploit the geometric structure of the token space. In language generation, shifting probability mass from "cat" to "kitten" versus to "table" should incur fundamentally different penalties.

**Key Insight**: Replace KL divergence with Wasserstein distance as the policy regularizer, since Wasserstein distance naturally accounts for the metric structure of the underlying space and can encode semantic distances between tokens.

## Method

### Overall Architecture
WPR replaces the KL regularization term in the standard RLHF objective with an entropy-regularized Wasserstein distance (Sinkhorn distance), then transforms it into token-level reward penalty terms via a dual formulation, making it compatible with standard RL algorithms such as PPO.

### Key Designs

1. **Wasserstein Policy Regularization Objective**: The token-level KL regularization in the RLHF objective is replaced by Wasserstein regularization:
$$\max_{\pi_\theta} \mathbb{E}\left[\sum_n R(\mathbf{x}, \mathbf{y}_{1:n}) - \beta \sum_n D_{\tilde{W}}(\pi_\theta(y_n|\cdot) \| \pi_{ref}(y_n|\cdot))\right]$$
where $D_{\tilde{W}}$ is the entropy-regularized Wasserstein distance. The cost matrix $C$ is defined as the Euclidean distance in the token embedding space of the reference policy, encoding semantic similarity between tokens.

2. **Dual Formulation and Tractable Optimization**: Direct computation of the Wasserstein distance requires solving a linear program ($O(d^3)$), which is infeasible for large vocabularies. The paper applies entropy regularization to obtain the dual form of the Sinkhorn distance and proves (Theorem 2) that the optimal dual variable $\phi^*$ can serve directly as a token-level reward penalty:
$$\mathcal{J}_{\tilde{W}}(\pi_\theta) = \mathbb{E}\left[\sum_n \mathbb{E}_{y_n}[R(\mathbf{x}, \mathbf{y}_{1:n}) - \beta \phi^*_{y_n}]\right] + \mathcal{C}$$
This makes WPR fully compatible with standard PPO—the KL penalty is simply replaced by the Wasserstein dual variable.

3. **Efficient Computation Strategy**: To avoid $O(d^2)$ computation over the full vocabulary ($d \sim 256\text{K}$):
   - **Nearest-$k_1$ truncation**: The cost kernel $K = \exp(-\lambda C)$ retains only the $k_1 = 512$ nearest neighbors for each token and is stored in sparse form.
   - **Top-$k_2$ truncation**: The policy distribution is truncated to the top-$k_2 = 128$ tokens, reducing the effective support size from $d$ to $2k_2 + 2$.
   - Together, these truncations add only 2.5% computational overhead relative to KL regularization.

4. **Sinkhorn–Knopp Algorithm**: The dual variable $\phi^*$ is solved efficiently via Sinkhorn iterations. Ten iterations are generally sufficient for convergence, with a tolerance of $10^{-4}$.

### Loss & Training
The standard three-stage RLHF pipeline based on PPO is adopted: SFT → Reward Model → PPO. The base model is Gemma-2B; datasets are TL;DR (summarization) and HH-RLHF (dialogue). Hyperparameters: $\lambda = 100$, $k_1 = 512$, $k_2 = 128$.

## Key Experimental Results

### Main Results (GPT-4 Win Rate)

| Divergence / Method | TL;DR vs SFT | TL;DR vs RKL | HH-RLHF vs SFT | HH-RLHF vs RKL |
|---|---|---|---|---|
| RKL | 0.848 | — | 0.828 | — |
| FKL | 0.316 | 0.040 | 0.808 | 0.564 |
| JS | 0.540 | 0.204 | 0.744 | 0.424 |
| $\alpha$(0.5) | 0.724 | 0.304 | 0.792 | 0.524 |
| TV | 0.364 | 0.052 | 0.748 | 0.376 |
| $\chi^2$ | 0.904 | — | — | — |
| **Wasserstein** | **0.924** | **0.608** | **0.852** | **0.596** |

### Ablation Study

| Configuration | vs SFT | vs RKL | Note |
|---|---|---|---|
| Default (L2, k1=512, k2=128, λ=100) | 0.924 | 0.608 | Best |
| Cost: cosine | 0.932 | 0.644 | Cosine distance slightly better than L2 |
| k1=256 | 0.920 | 0.572 | Fewer neighbors, slight degradation |
| k2=64 | 0.864 | 0.528 | Excessive distribution truncation, notable drop |
| λ=10 | 0.868 | 0.552 | Entropy regularization too strong |
| Sinkhorn iter=5 | 0.708 | 0.328 | Insufficient convergence, severe degradation |
| Sinkhorn iter=30 | 0.880 | 0.536 | Additional iterations yield no benefit |

### Key Findings
- WPR consistently outperforms KL divergence and all f-divergence baselines across all tasks; it is the only method that achieves best performance on both TL;DR and HH-RLHF.
- FKL and TV exhibit training instability on TL;DR (probability ratio explosion); WPR remains well-defined even under support mismatch.
- WPR achieves the highest MT-Bench score (4.272 vs. RKL 4.000).
- WPR is also effective on code generation (APPS + CodeGemma-7B).
- The Wasserstein penalty is strongly correlated with the KL penalty ($r = 0.917$), but with a slope below 1, indicating that WPR is more lenient.
- Models trained with WPR exhibit significantly higher semantic coherence among their top-10 candidate tokens.
- Computational overhead is only 2.5% per thousand steps, with approximately 15 GB additional memory usage (A100).

## Highlights & Insights
- The work revisits RLHF regularization from the perspective of optimal transport theory, achieving both theoretical elegance and practical utility.
- The cat/kitten/table example in Figure 2 is highly compelling and intuitively exposes the blind spot of KL divergence.
- The dual formulation converts Wasserstein regularization into token-level reward penalties, enabling seamless integration with PPO.
- The truncation strategy is carefully designed, reducing complexity from $O(d^2)$ to $O(k_2^2)$ with negligible computational cost.
- The case study in Figure 6 visually demonstrates that WPR assigns small penalties for semantically similar tokens and large penalties when semantic drift occurs.

## Limitations & Future Work
- The cost matrix depends on the reference model's embeddings and cannot be directly transferred across models with different tokenizers.
- Validation is limited to the 2B–7B scale; scaling behavior on larger models remains unknown.
- $\beta$ still requires manual tuning (though WPR is more robust than f-divergences); automatic adjustment is a natural future direction.
- Extending WPR to the DPO paradigm (which does not require an explicit reward model) is a straightforward next step.

## Related Work & Insights
- **vs. KL-DPO/PPO**: Standard methods that compare probabilities index-wise and ignore semantic structure.
- **vs. f-DPO/χPO**: Extensions to other f-divergences, but comparisons remain index-wise.
- **vs. Wasserstein GAN**: Also leverages Wasserstein distance, but applies it to the discriminator of a generative model, whereas this work uses it for policy regularization.
- **vs. MA-RLHF**: A concurrent work that also improves RLHF regularization, but approaches the problem from the perspective of action granularity.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Introduces optimal transport into RLHF regularization with outstanding theoretical contribution.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Two tasks × seven divergence variants, multiple model scales, code generation, complete ablation and analysis.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivations, well-motivated intuitions, and insightful case studies.
- Value: ⭐⭐⭐⭐⭐ A fundamental improvement to RLHF regularization with the potential to become the new standard.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Towards Understanding Valuable Preference Data for Large Language Model Alignment](towards_understanding_valuable_preference_data_for_large_language_model_alignmen.md)
- [\[ICLR 2026\] GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models](guardalign_test-time_safety_alignment_in_multimodal_large_language_models.md)
- [\[ICLR 2026\] Chasing the Tail: Effective Rubric-based Reward Modeling for Large Language Model Post-Training](chasing_the_tail_effective_rubric-based_reward_modeling_for_large_language_model.md)
- [\[NeurIPS 2025\] GVPO: Group Variance Policy Optimization for Large Language Model Post-Training](../../NeurIPS2025/llm_alignment/gvpo_group_variance_policy_optimization_for_large_language_model_post-training.md)
- [\[ICLR 2026\] Unifying Stable Optimization and Reference Regularization in RLHF (DAR)](unifying_stable_optimization_and_reference_regularization_in_rlhf.md)

</div>

<!-- RELATED:END -->
