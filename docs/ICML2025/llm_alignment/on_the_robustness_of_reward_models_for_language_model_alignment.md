---
title: >-
  [Paper Note] On the Robustness of Reward Models for Language Model Alignment
description: >-
  [ICML 2025][LLM Alignment][Reward Model Robustness] This paper proposes Batch-wise Sum-to-Zero Regularization (BSR), which represses the excessive dispersion of hidden-state norms by constraining the sum of reward scores within each batch to zero, fundamentally addressing the over-optimization problem of reward models. This mechanism enables an 8B-scale RM to outperform the state-of-the-art (SOTA) by more than 5% on complex preference prediction tasks…
tags:
  - "ICML 2025"
  - "LLM Alignment"
  - "Reward Model Robustness"
  - "Over-optimization"
  - "Bradley-Terry Model"
  - "Regularization"
  - "RLHF"
date: 2026-05-08
content_hash: 32bff8832674042c
---

# On the Robustness of Reward Models for Language Model Alignment

**Conference**: ICML 2025  
**arXiv**: [2505.07271](https://arxiv.org/abs/2505.07271)  
**Code**: [LinkedIn-XFACT/RM-Robustness](https://github.com/LinkedIn-XFACT/RM-Robustness)  
**Area**: LLM Alignment/RLHF  
**Keywords**: Reward Model Robustness, Over-optimization, Bradley-Terry Model, Regularization, RLHF

## TL;DR

This paper proposes Batch-wise Sum-to-Zero Regularization (BSR), which represses the excessive dispersion of hidden-state norms by constraining the sum of reward scores within each batch to zero, fundamentally addressing the over-optimization problem of reward models. This mechanism enables an 8B-scale RM to outperform the state-of-the-art (SOTA) by more than 5% on complex preference prediction tasks, reduces generation length by 40% during downstream RLHF training, and improves win rate by 7%.

## Background & Motivation

### The Prevalence of Reward Model Over-optimization

Reward models (RMs) are core components in the RLHF pipeline, serving as proxies for human preferences to align large language models. The mainstream approach trains RMs using the Bradley-Terry (BT) model loss, which maximizes the reward difference between chosen and rejected responses. However, extensive literature has noted that RMs trained with BT loss suffer from **over-optimization**:

- RM accuracy consistently improves on the training set and in-distribution (ID) validation set.
- However, performance on out-of-distribution (OOD) data stagnates or even degrades.
- This degradation propagates to downstream RLHF training, causing the policy to fail in truly aligning with actual human preferences.

### Limitations of Prior Work

Previous works primarily mitigate over-optimization from the perspective of data quality (e.g., high-quality preference datasets, better annotation strategies) or by adding auxiliary components to the BT model (e.g., margin loss, label smoothing). However, these methods:

1. Do not fundamentally resolve the inherent flaws of the BT training objective.
2. Suffer from inherent biases such as verbosity bias and self-enhancement bias when utilizing LLMs as data generators or annotators.
3. Lack a systematic analysis of how over-optimization manifests across different generalization scenarios.

### Design Motivation

The authors observe a key phenomenon: the reward model's score can be decomposed as $r_\theta(x,y) = \|W_p\| \cdot \|h(x,y)\| \cdot \cos\psi$, where $W_p$ is the projection head and $h(x,y)$ is the last-layer hidden state. During training, $\|W_p\|$ remains virtually unchanged around its initial value of 1, whereas the variance of the hidden-state norm $\|h(x,y)\|$ **explodes dramatically**, which constitutes the root cause of over-optimization.

## Method

### Overall Architecture

The proposed methodology is divided into three levels:

1. **Problem Diagnosis**: Four generalization scenarios are presented to systematically analyze the behaviors of RM over-optimization.
2. **Causal Analysis**: It is mathematically shown, from the gradient dynamics perspective, that the dispersion of hidden-state norms is the primary cause of over-optimization.
3. **Solution**: Batch-wise Sum-to-Zero Regularization (BSR) is proposed to control the dispersion by constraining the batch-wise reward sum to zero.

#### Four Generalization Scenarios

The authors categorize the generalization capability of RMs based on the distribution shifts across the two dimensions of prompt and response:

| Scenario | Prompt Distribution | Response Distribution | Explanation |
|------|:---------:|:-----------:|------|
| In-domain ($\mathcal{D}_{ID}$) | In-training-set | In-training-set | Standard evaluation without distribution shift |
| Prompt-disjoint ($\mathcal{D}_{\sim\text{Prompt}}$) | **Unseen** | In-training-set | New prompt + same-source response model |
| Response-disjoint ($\mathcal{D}_{\sim\text{Response}}$) | In-training-set | **Unseen** | Same prompt + new response model |
| Mutual-disjoint ($\mathcal{D}_{\sim\text{Mutual}}$) | **Unseen** | **Unseen** | Most challenging scenario, double distribution shifts |

Definition of over-optimization: The accuracy on $\mathcal{D}_{ID}$ continues to improve, while performance on the latter three scenarios stagnates or degrades.

### Key Designs

#### 1. Root Cause Analysis of Over-optimization

The RM score is determined by the product of three elements:

$$r_\theta(x,y) = \|W_p\| \cdot \|h(x,y)\| \cdot \cos\psi$$

**The projection head $W_p$ is not the cause of over-optimization:**

Since the chosen and rejected responses share the same projection head, the gradient of $\mathcal{L}_{BT}$ with respect to $W_p$ is:

$$\frac{\partial \mathcal{L}_{BT}}{\partial W_p} = -\sigma(-\Delta r) \cdot (h(x,y_w) - h(x,y_l))$$

Its gradient norm is $\sigma(-\Delta r) \cdot \|h(x,y_w) - h(x,y_l)\|$. In the early stages of training, $\Delta r \approx 0$ and the difference in hidden states is small (as the effective rank of LLM hidden states is low), limiting the update magnitude of $W_p$. Empirical validation shows that after training, $\|W_p\| \approx 1$, remaining almost identical to its initial value.

**Dispersion of the hidden-state norm is the true cause:**

The BT loss is minimized by maximizing $\Delta r$, which drives the model to increase $\|h(x,y_w) - h(x,y_l)\|$. Specifically:

$$\Delta r = \|W_p\| \cdot \|h(x,y_w) - h(x,y_l)\| \cdot \cos\psi$$

Since $\|W_p\|$ remains essentially constant, the model can only increase $\Delta r$ by scaling up the norm of the hidden-state difference. This leads to:

- The variance of the hidden-state norm, $\text{Var}(\|h(x,y)\|)$, continuously exploding during training.
- A right-skewed distribution, resulting in extremely large outlier norms.
- These outliers generating uncontrollable, extreme reward scores on OOD data.

#### 2. Batch-wise Sum-to-Zero Regularization (BSR)

Core idea of BSR: Enforcing the sum of reward scores of all samples within each batch to tend to zero, thereby penalizing extreme reward values and controlling the dispersion of hidden-state norms.

The regularization term is defined as:

$$\mathcal{L}_{BSR} = \left(\frac{1}{2B} \sum_{i=1}^{B} \left[r(x_i, y_{w,i}) + r(x_i, y_{l,i})\right]\right)^2$$

where $B$ is the number of sample pairs in the batch, and the factor $2B$ represents the total number of samples in the batch (each pair consists of one chosen and one rejected sample).

**Mechanism of BSR:**

- When the mean of individual reward scores in a batch deviates from zero, $\mathcal{L}_{BSR}$ exerts a quadratic penalty.
- This forces the model to avoid scaling chosen rewards up or dragging rejected rewards down limitlessly.
- It indirectly constraints the hidden-state norms from excessive dispersion.
- The zero-centered constraint guarantees that reward scores fluctuate within a reasonable range.

**Advantages of BSR:**

- It does not alter the ability of the BT loss to learn preference ranking (since it only constrains the mean, not the difference).
- The computational overhead is practically negligible (requiring only one extra calculation of the mean and square per batch).
- It avoids introducing additional hyperparameter tuning dimensions (requiring only a single $\lambda$).

### Loss & Training

The final training objective is the standard BT loss combined with the BSR regularization:

$$\mathcal{L}_{BT\text{-}BSR} = \mathcal{L}_{BT} + \lambda \cdot \mathcal{L}_{BSR}$$

$$= -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[\log \sigma(\Delta r)\right] + \lambda \left(\frac{1}{2B} \sum_{i=1}^{B} \left[r(x_i, y_{w,i}) + r(x_i, y_{l,i})\right]\right)^2$$

**Training Setup:**

- Backbone Models: Llama-3 series (1B/3B/8B) and Qwen2.5 series (1.5B/3B/7B).
- All models undergo SFT on UltraChat first, followed by RM training.
- Training Data: UltraFeedback (51,200 samples) containing responses generated by 17 different models.
- Validation Model Set $\mathcal{M}_{valid}$: Gemma-2-2B-It, Olmo2-7B-Instruct, SmolLM2-1.7B-Instruct, Mistral-Instruct-v0.2 (excluding Llama and Qwen families to avoid contamination).
- Gold RM: ArmoRM (serving as the true preference model $r^*$).

**RLHF Experimental Setup:**

- Initial Policy: Qwen2.5-1.5B + UltraChat SFT.
- RM: Qwen2.5-3B (Part 2) / Llama-3.1-8B (Part 3).
- RL Algorithm: RLOO.
- High-quality Data Scaling: Skywork-Reward-Preference-80K-v0.2 + TULU3 SFT mixture.

## Key Experimental Results

### Main Results

**Comparison of RM Accuracy Under Four Generalization Scenarios:**

| Model | Method | In-domain | Prompt-disjoint | Response-disjoint | Mutual-disjoint |
|------|------|:---------:|:---------------:|:-----------------:|:---------------:|
| Qwen2.5-3B | BT | High (Baseline) | Obvious decline | Obvious decline | Worst |
| Qwen2.5-3B | BT-BSR | Slightly lower | **Significant improvement** | **Significant improvement** | **Significant improvement** |
| Llama-3.2-3B | BT | High (Baseline) | Obvious decline | Obvious decline | Worst |
| Llama-3.2-3B | BT-BSR | Slightly lower | **Significant improvement** | **Significant improvement** | **Significant improvement** |

BSR exhibits better robustness across all four generalization scenarios, showing the largest gain in Mutual-disjoint (the most rigorous scenario).

**8B Scale RM-Bench Evaluation (Complex Preference Prediction):**

| Configuration | RM-Bench Accuracy | Gain over BT | Explanation |
|------|:--------------:|:----------:|------|
| Llama-3.1-8B + BT | Baseline | — | Standard BT loss training |
| Llama-3.1-8B + BT-BSR | **+5%+** | +5%+ | BSR regularization |
| Prev. SOTA (8B) | Below BT-BSR | — | Surpassed |

### Ablation Study

| Configuration | Hidden-state Norm Variance | OOD Accuracy | Explanation |
|------|:----------:|:--------:|------|
| Standard BT | Continuous explosion | Degradation | Baseline, severe over-optimization |
| BT + BSR | **Stable & controlled** | **Improved** | BSR effectively represses dispersion |
| $\|W_p\|$ Analysis | ≈1 (unchanged) | — | Proves the projection head is not the cause of over-optimization |
| Hidden-state Difference Norm | Continuous growth + Right-skewed | — | Proves hidden-state norm dispersion is the root cause |

**RLHF Downstream Propagation Experiment (AlpacaEval 2.0):**

| RM Type | Generation Length Change | Win Rate Change | Explanation |
|---------|:--------:|:------:|------|
| BT-trained RM | Baseline | Baseline | Exhibits verbosity bias |
| BSR-trained RM | **↓40%** | **↑7%** | Robustness propagates to RLHF |

### Key Findings

1. **Hidden-State Norm Dispersion is the Root Cause of Over-optimization**: The BT loss drives the model to increase $\|h(x,y_w) - h(x,y_l)\|$, which causes the variance of the hidden-state norm to inflate continuously and display a right-skewed distribution, giving rise to extreme reward scores.
2. **The Projection Head $W_p$ is Not the Cause of Over-optimization**: $\|W_p\|$ remains $\approx 1$ before and after training, and its gradient is scaled down by the derivative of the sigmoid, resulting in a limited learning magnitude.
3. **Consistency of BSR**: Performance is consistent across two model families (Llama-3 and Qwen2.5) and three different model scales.
4. **Propagatable Robustness**: The robustness of the RM propagates to RLHF training; the policy trained under BSR-RM generates more concise and higher-quality responses.
5. **Scalability**: BSR remains highly effective on 8B-scale models trained on high-quality data, surpassing the state of the art (SOTA).

## Highlights & Insights

- **"Diagnosis $\rightarrow$ Treatment" Paradigm**: Pinpointing the root cause of over-optimization (hidden-state norm dispersion) via gradient analysis, then designing targeted regularization. This pipeline is clear and highly convincing.
- **Four Generalization Scenarios**: Structuring RM evaluations into a more grain-fine framework that distinguishes between prompt distribution shift, response distribution shift, and their combination.
- **Extremely Simple BSR Design**: Introducing only one additional regularization term with negligible computational overhead, yet delivering significant impact. This "minimal intervention" methodology is highly reference-worthy.
- **Impressive End-to-End RLHF Results**: A 40% reduction in generation length implies that BSR-RM successfully mitigates verbosity bias, which has been a long-standing challenge in RLHF.
- **Cross-Domain Insight Transfer**: Drawing parallels between hidden-state norm dispersion in RMs and the logit norm explosion in classification tasks (referencing Wei et al., 2022).

## Limitations & Future Work

1. **BSR Only Constrains the Batch Mean**: The actual reward distribution within a batch may still pose potential issues; more fine-grained formulations (such as variance constraints) could further improve results.
2. **Selection of $\lambda$**: Although there is only one hyperparameter, optimal values of $\lambda$ may vary across tasks and model scales, and the framework currently lacks an adaptive adjustment mechanism.
3. **Gold RM Assumption**: Modeling the ground-truth $r^*$ with ArmoRM is an approximation; direct evaluation against real human preferences is still absent.
4. **Decoder-Only Limitations**: The framework has only been validated on decoder-only backbones, leaving encoder-only or encoder-decoder backbones unexamined.
5. **Integration with Implicit Methods**: BSR is currently restricted to explicit RM training. Exploring how it might extend to implicit reward optimization (e.g., DPO) is a valuable direction.
6. **Dynamic Regularization**: The severity of dispersion shifts across training stages; whether $\lambda$ should dynamically adapt as training progresses remains to be studied.

## Related Work & Insights

- **Gao et al. (2023)**: First to systematically study RM over-optimization using gold RMs for evaluation. This paper builds on their work by proposing a more granular classification of four generalization scenarios.
- **Wei et al. (2022)**: Observed that norm growth in LLMs prompts softmax overconfidence. This work successfully transfers this insight to the RM scenario.
- **RLOO (Ahmadian et al., 2024)**: Serves as the RL algorithm for RLHF. Integrating BSR with RLOO demonstrates the positive propagation of a robust RM to downstream learning.
- **Skywork/TULU3 Data**: Demonstrates the complementary effect of BSR alongside high-quality data engineering without conflict.
- **Takeaway**: One could analyze the norm dispersion of implicit rewards in methods like DPO/ORPO, potentially revealing similar over-optimization behaviors and inspiring corresponding regularization techniques.

## Rating

- Novelty: ★★★★☆ — The categorization of the four generalization scenarios and the analysis of hidden-state norm dispersion are refreshing, though the BSR design is simple rather than revolutionary.
- Theoretical Depth: ★★★★☆ — The gradient analysis is rigorous, locating the root cause by ruling out the projection head $W_p$ and pointing to the hidden state $h$.
- Experimental Thoroughness: ★★★★★ — Validated across two model families, three scales, four scenarios, end-to-end RLHF pipelines, and RM-Bench evaluations.
- Practical Value: ★★★★★ — BSR is minimally intrusive and plug-and-play, providing direct assistance to practical RLHF pipelines.
- Writing Quality: ★★★★☆ — Clear structure, though notation-heavy; some mathematical derivations could be more concise.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HAF-RM: A Hybrid Alignment Framework for Reward Model Training](../../ACL2025/llm_alignment/haf-rm_a_hybrid_alignment_framework_for_reward_model_training.md)
- [\[ICLR 2026\] Reward Model Routing in Alignment](../../ICLR2026/llm_alignment/reward_model_routing_in_alignment.md)
- [\[ICML 2025\] Layer-wise Alignment: Examining Safety Alignment Across Image Encoder Layers in Vision Language Models](layer-wise_alignment_examining_safety_alignment_across_image_encoder_layers_in_v.md)
- [\[ICML 2025\] PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning](poisonbench_assessing_large_language_model_vulnerability_to_data_poisoning.md)
- [\[ACL 2025\] Rethinking Reward Model Evaluation Through the Lens of Reward Overoptimization](../../ACL2025/llm_alignment/rethinking_reward_model_evaluation_through_the_lens_of_reward_overoptimization.md)

</div>

<!-- RELATED:END -->
