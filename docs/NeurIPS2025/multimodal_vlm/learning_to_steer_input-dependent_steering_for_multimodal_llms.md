---
title: >-
  [Paper Note] Learning to Steer: Input-dependent Steering for Multimodal LLMs
description: >-
  [NeurIPS 2025][Multimodal VLM][steering] Addressing the limitation of existing steering methods that rely on fixed direction vectors incapable of adapting to diverse inputs, this paper proposes L2S (Learn-to-Steer): it first generates ideal input-specific steering vectors via contrastive prompting (P2S), then trains a lightweight 2-layer MLP to predict these vectors from the input context. This achieves input-dependent behavioral steering at negligible overhead, significantly outperforming static steering baselines on both safety enforcement and hallucination mitigation.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - steering
  - input-dependent
  - hallucination mitigation
  - safety enforcement
  - contrastive prompting
date: 2026-05-08
content_hash: 367fd9032d46804f
---

# Learning to Steer: Input-dependent Steering for Multimodal LLMs

**Conference**: NeurIPS 2025
**arXiv**: [2508.12815](https://arxiv.org/abs/2508.12815)
**Code**: [https://github.com/jayneelparekh/learn-to-steer](https://github.com/jayneelparekh/learn-to-steer)
**Area**: Multimodal VLM / Model Safety / Hallucination Mitigation / Representation Steering
**Keywords**: steering, input-dependent, hallucination mitigation, safety enforcement, contrastive prompting

## TL;DR
Addressing the limitation of existing steering methods that rely on fixed direction vectors incapable of adapting to diverse inputs, this paper proposes L2S (Learn-to-Steer): it first generates ideal input-specific steering vectors via contrastive prompting (P2S), then trains a lightweight 2-layer MLP to predict these vectors from the input context. This achieves input-dependent behavioral steering at negligible overhead, significantly outperforming static steering baselines on both safety enforcement and hallucination mitigation.

## Background & Motivation

**Background**: Steering guides model behavior by applying linear offsets to the latent representations of LLMs/MLLMs, serving as a lightweight post-hoc control mechanism. Mainstream approaches (e.g., CAA/mean-steering) compute the mean difference between positive and negative behavior representations as a fixed steering vector applied uniformly to all inputs.

**Limitations of Prior Work**: The fatal flaw of fixed steering vectors is that **the instantiation of desired behavior is input-dependent**. For instance, a safe response to an illegal activity query should be a refusal, whereas a safe response to a medical consultation should recommend seeking expert advice. These two "safe" behaviors are fundamentally different and cannot be captured by a single fixed vector.

**Key Challenge**: The ideal input-specific steering vector (P2S) requires knowledge of the desired output content to be computed — yet at inference time, the very reason steering is needed is that the answer is unknown, creating a chicken-and-egg problem.

**Key Insight**: Although the desired answer is unavailable at inference time, contrastive prompts from training data can be used to construct P2S vectors as "teacher signals," after which a minimal network is trained to predict these vectors from the input context.

**Core Idea**: A 2-layer MLP predicts input-specific steering vectors from intermediate-layer representations of the input, translating the theoretical advantages of P2S into the practically deployable L2S method.

## Method

### Overall Architecture
The framework consists of two phases:
- **Training phase**: For each sample $X=(I,T)$, input-specific positive/negative contrastive prompts $(T_X^+, T_X^-)$ are constructed. Under teacher forcing, the representation difference of the last token at layer $L^*$ is extracted as the P2S vector $z_{X,L^*}$. Simultaneously, the input context representation $h_{X,L'}$ at layer $L'$ is extracted. An MLP $g_{\Theta}$ is trained such that $g_\Theta(h_{X,L'}) \approx z_{X,L^*}$.
- **Inference phase**: For a new input, $h_{X,L'}$ is extracted, and the trained $g_{\Theta^*}$ predicts the steering vector, which is applied to the representations of all generated tokens at layer $L^*$.

### Key Designs

1. **Input-Specific Contrastive Prompting (P2S)**:

    - Function: Generates prompt completions reflecting desired/undesired behavior for each input.
    - Mechanism: Constructs $X^+ = (I, T||T_X^+)$ and $X^- = (I, T||T_X^-)$, and extracts the difference of the last-token representations at layer $L^*$ under teacher forcing: $z_{X,L^*} = h_{L^*}^{q^+}(X^+) - h_{L^*}^{q^-}(X^-)$.
    - Design Motivation: Unlike the fixed prompt pairs in CAA, P2S allows different inputs to use different behavioral descriptions. For example, in safety scenarios, illegal activity queries use a "refusal" template while medical consultations use a "recommend expert" template.

2. **Learn-to-Steer (L2S) Auxiliary Network**:

    - Function: Predicts P2S steering vectors from the input context, eliminating the need for contrastive prompts at inference time.
    - Mechanism: The input context is defined as the representation of the last input token at layer $L'$: $h_{X,L'} = h_{L'}^{N_V+N_T}(X)$. The training objective is mean squared error: $\Theta^* = \arg\min_\Theta \mathbb{E}_X[\|z_{X,L^*} - g_\Theta(h_{X,L'})\|_2^2]$. At inference, the steering is applied to generated token $p$ as $h_{L^*}^p \leftarrow h_{L^*}^p + \alpha g_{\Theta^*}(h_{X,L'})$.
    - Design Motivation: The 2-layer MLP (hidden size 100) is extremely lightweight. Training requires only representation-space operations without loading main model gradients, making memory overhead negligible.

3. **Multi-Behavior Scenario Handling**:

    - Function: Handles multiple distinct desired behaviors within the same steering framework.
    - Key Example (safety scenario): The first 9 categories of harmful content use "refusal/avoidance" templates for $(T_X^+, T_X^-)$; the remaining 3 sensitive consultation categories use "recommend expert" templates. L2S naturally supports multi-behavior by learning mappings from different inputs to different vectors.
    - Key Contrast: Mean-steering suffers interference when mixing vectors from different templates (Mean-S performs worse than Mean-S(BA)), whereas L2S handles this gracefully.

### Loss & Training
- Auxiliary network: 2-layer MLP, hidden size 100
- Trained for 100 epochs with the Adam optimizer, learning rate $10^{-4}$ or $5\times10^{-5}$
- Cosine learning rate schedule with plateau-based adaptation
- Steering strength $\alpha \in [1, 3.0)$ (LLaVA), ensuring response quality degradation < 10%
- Safety task: $L^*=15$ (steering layer), $L'=30$ (context extraction layer)
- Hallucination task: $L^*=14, L'=14$
- Evaluated on LLaVA-v1.5-7B and Qwen2-VL-7B; runnable on a single RTX5000 (24GB) GPU

## Key Experimental Results

### Safety Enforcement — MMSafetyBench (LLaVA-v1.5)

| Metric | No-steering | Prompt | Mean-S | Mean-S(BA) | L2S | P2S* |
|--------|------------|--------|--------|-----------|-----|------|
| $\mathbb{E}_{p\geq0.5}$[Unsafe]↓ | 0.276 | 0.248 | 0.161 | 0.089 | **0.082** | 0.094 |
| $\mathbb{E}_{p\geq0.7}$[Unsafe]↓ | 0.234 | 0.207 | 0.129 | 0.066 | **0.057** | 0.064 |
| $\mathbb{E}_{p\geq0.9}$[Unsafe]↓ | 0.204 | 0.183 | 0.102 | 0.041 | **0.034** | 0.042 |
| ED-score↑ | 0.250 | 0.197 | 0.329 | 0.276 | **0.395** | 0.382 |
| Response quality↑ | 6.92 | 7.34 | 6.61 | 6.42 | 6.56 | 6.49 |

### Hallucination Mitigation — POPE (LLaVA-v1.5)

| Subset | Metric | No-steering | Prompt | Norm-Rnd | Mean-S | L2S | P2S* |
|--------|--------|------------|--------|---------|--------|-----|------|
| Random | Accuracy↑ | 82.73 | 84.91 | 82.38 | 84.29 | **86.46** | 89.26 |
| Random | F1↑ | 90.55 | 91.84 | 90.34 | 91.47 | **92.74** | 94.33 |
| Popular | Accuracy↑ | 80.40 | 83.35 | 80.36 | 82.11 | **82.58** | 88.64 |
| Adversarial | Accuracy↑ | 76.82 | 76.36 | 75.77 | 76.36 | **77.76** | 82.58 |

### CHAIR Evaluation (LLaVA-v1.5, 500 COCO images)

| Method | CHAIR_s↓ | CHAIR_i↓ | Recall↑ | Gemini Win Rate↑ |
|--------|---------|---------|---------|-----------------|
| No-steering | 17.31 | 52.80 | 71.23 | 35.80% |
| **L2S** | **16.10** | **51.80** | **73.50** | **64.20%** |

### Key Findings
- **L2S surpasses the P2S oracle** on the safety task (Unsafe-score 0.082 vs. 0.094), indicating that the learned mapping generalizes better than per-sample ideal vector computation.
- Mean-S degrades when mixing multiple behavior templates (Mean-S: 0.161 vs. Mean-S(BA): 0.089), whereas L2S handles multi-behavior settings simultaneously (ED-score 0.395, far exceeding all baselines).
- Random-direction steering (Norm-Rnd) reduces harmful content but fails to elicit expert recommendation behavior, confirming the critical importance of steering direction precision.
- On the hallucination task, Mean-S and Prompt fail to consistently improve across all subsets, while L2S uniformly outperforms all available baselines.
- A Gemini Win Rate of 64.20% indicates that L2S not only reduces hallucinations but also improves description quality.

## Highlights & Insights
- **The core insight of input-dependent steering is precise**: desired behavior is not a fixed direction but a manifold conditioned on the input context — this is especially evident in safety scenarios (refusal vs. recommending experts vs. non-intervention).
- **The elegance of replacing teacher forcing with a 2-layer MLP** lies in converting a theoretically impractical method (requiring the answer to perform steering) into a lightweight, deployable solution.
- **Training cost is minimal**: only a small network in representation space needs to be trained without main model gradients, completable on a single 24GB GPU.
- **L2S outperforming the oracle P2S** on the safety task suggests that the learned mapping provides a regularization effect with strong generalization.

## Limitations & Future Work
- Contrastive prompt design still requires manual effort; different application scenarios necessitate customizing distinct $(T_X^+, T_X^-)$ templates.
- Steering is currently applied as a linear offset at a single layer $L^*$; multi-layer or nonlinear steering may prove more effective.
- The auxiliary network capacity (hidden size 100) may limit the modeling of complex behavioral patterns.
- Validation is primarily conducted on LLaVA-v1.5 and Qwen2-VL; broader evaluation across more models and tasks is needed.
- Performance is highly sensitive to the choice of $\alpha$ (notable degradation at $\alpha \geq 3$); automated $\alpha$ selection remains an open problem.
- Misuse risk: the same methodology could be exploited to steer models toward harmful behaviors.

## Related Work & Insights
- **vs. CAA (Contrastive Activation Addition)**: CAA uses a fixed mean-difference vector, suited for scenarios with a single behavioral instantiation; L2S extends this to input-dependent settings covering multi-behavior scenarios.
- **vs. CAST**: CAST scales a fixed steering vector based on similarity to a condition vector, but the direction remains unchanged; in L2S, both direction and magnitude are input-dependent.
- **vs. PAI / AD-HH (attention head intervention)**: These methods directly manipulate attention weights, while L2S operates on residual stream representations; the two are complementary and can be combined.
- **vs. fine-tuning (SFT/RLHF)**: Fine-tuning is costly and may cause forgetting; L2S is a post-hoc method that leaves model weights unchanged.

## Rating
- Novelty: ⭐⭐⭐⭐ The idea of input-dependent steering is natural yet previously underexplored; the two-stage P2S→L2S design is elegant.
- Experimental Thoroughness: ⭐⭐⭐⭐ Two applications (safety + hallucination), two models, multiple evaluation dimensions, and comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Motivation is clearly articulated, examples are intuitive, and the method description is concise.
- Value: ⭐⭐⭐⭐ Highly practical — a post-hoc behavioral control method with minimal cost, directly deployable in production environments.

<!-- RELATED:START -->

## Related Papers

- [\[ICCV 2025\] Analyzing Finetuning Representation Shift for Multimodal LLMs Steering](../../ICCV2025/multimodal_vlm/analyzing_finetuning_representation_shift_for_multimodal_llms_steering.md)
- [\[NeurIPS 2025\] Vision Function Layer in Multimodal LLMs](vision_function_layer_in_multimodal_llms.md)
- [\[NeurIPS 2025\] ADMN: A Layer-Wise Adaptive Multimodal Network for Dynamic Input Noise and Compute Resources](admn_a_layerwise_adaptive_multimodal_network_for_dynamic_inp.md)
- [\[NeurIPS 2025\] To See or To Read: User Behavior Reasoning in Multimodal LLMs](to_see_or_to_read_user_behavior_reasoning_in_multimodal_llms.md)
- [\[NeurIPS 2025\] VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning](vt-fsl_bridging_vision_and_text_with_llms_for_few-shot_learning.md)

<!-- RELATED:END -->
