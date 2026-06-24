---
title: >-
  [Paper Note] Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting
description: >-
  [AAAI 2026][LLM Safety][Supervised Fine-Tuning] This paper proposes Learning-from-the-Undesirable (LfU), a regularization method for SFT that simulates "undesirable behavior" by applying gradient ascent to an auxiliary model, then enforces representation-level consistency between the original and auxiliary models via an MSE loss. This effectively mitigates overfitting, catastrophic forgetting, and adversarial fragility in limited-data fine-tuning.
tags:
  - "AAAI 2026"
  - "LLM Safety"
  - "Supervised Fine-Tuning"
  - "Overfitting Mitigation"
  - "Consistency Regularization"
  - "Knowledge Retention"
  - "Adversarial Robustness"
date: 2026-05-08
content_hash: 5de9c53d3ba029ec
---

# Learning from the Undesirable: Robust Adaptation of Language Models without Forgetting

**Conference**: AAAI 2026
**arXiv**: [2511.13052](https://arxiv.org/abs/2511.13052)  
**Code**: [yunpal/LfU](https://github.com/yunpal/LfU)  
**Area**: LLM Safety
**Keywords**: Supervised Fine-Tuning, Overfitting Mitigation, Consistency Regularization, Knowledge Retention, Adversarial Robustness

## TL;DR

This paper proposes Learning-from-the-Undesirable (LfU), a regularization method for SFT that simulates "undesirable behavior" by applying gradient ascent to an auxiliary model, then enforces representation-level consistency between the original and auxiliary models via an MSE loss. This effectively mitigates overfitting, catastrophic forgetting, and adversarial fragility in limited-data fine-tuning.

---

## Background & Motivation

**Severe SFT Overfitting**: When fine-tuning language models on limited data, models tend to rely on spurious patterns, resulting in poor in-domain generalization and significant degradation of out-of-domain capabilities.

**Catastrophic Forgetting**: After fine-tuning on a downstream task, models often lose general knowledge acquired during pretraining—for example, mathematical ability can degrade after fine-tuning.

**Vulnerability to Prompt Variations**: SFT models are highly sensitive to semantically equivalent prompt variants at inference time, exhibiting high variance in output performance.

**Adversarial Fine-Tuning Attacks**: Safety alignment achieved through SFT can be easily undone by a small number of adversarial fine-tuning steps, suggesting that SFT learns shallow refusal patterns rather than robust safety behaviors.

**Limitations of Prior Work**: NEFTune only injects Gaussian noise into input embeddings with limited gains; Instruction Modelling extends the loss to instruction tokens with marginal improvement; GEM maintains output diversity via entropy maximization but yields limited improvement both in- and out-of-domain; SDFT requires an already instruction-tuned model to rewrite training outputs and is incompatible with unadapted base models.

**Core Insight**: Overfitting is fundamentally a form of memorization over limited data—data augmentation can alleviate this. The authors propose performing data augmentation at the **representation level**: by simulating one-step parameter updates in an "undesirable direction" to generate diverse internal representations, then applying consistency regularization to enforce model stability against such perturbations.

---

## Method

### Overall Architecture

LfU introduces a consistency regularization term in addition to the standard SFT loss. During training, an auxiliary model $\theta_{\text{aux}}$ is constructed and perturbed by one step of gradient ascent toward an undesirable direction. The internal representations of both the original and auxiliary models are then extracted at each layer, and an MSE loss enforces their alignment. The final training objective is:

$$\ell_{\text{LfU}}(\theta, \theta_{\text{aux}}) = \ell_{\text{SFT}}(\theta) + \lambda \cdot \ell_{\text{cons.}}(\theta, \theta_{\text{aux}})$$

### Key Design 1: Simulating Undesirable Updates (One Step towards the Undesirable)

- **Function**: An auxiliary model $\theta_{\text{aux}}$ is defined by adding extra trainable components on top of the original parameters $\theta$. The SFT loss gradient with respect to $\theta_{\text{aux}}$ is computed, and one gradient ascent step is taken to push the auxiliary model toward undesirable behavior.
- **Update Rule**: $\theta_{\text{aux}} \leftarrow \theta_{\text{aux}} + \alpha \cdot \frac{\nabla_{\theta_{\text{aux}}} \ell_{\text{SFT}}(\theta_{\text{aux}})}{\|\nabla_{\theta_{\text{aux}}} \ell_{\text{SFT}}(\theta_{\text{aux}})\|_2}$
- **Key Details**: Step size $\alpha$ controls the magnitude of perturbation; gradient normalization ensures a stable perturbation direction invariant to gradient norm fluctuations.
- **Distinction from SAM**: SAM applies gradient ascent over all parameters to find flat minima; LfU applies ascent only to the additional auxiliary components while keeping the original parameters $\theta$ frozen.

### Key Design 2: Two Auxiliary Model Construction Strategies

**Strategy 1: LoRA-Based Parameter Perturbation**

- A trainable low-rank matrix $\text{LoRA}_l$ is added to each layer $\theta_l$, forming $\theta_{\text{aux}} = [\theta_1 + \text{LoRA}_1, \dots, \theta_K + \text{LoRA}_K]$.
- Only LoRA parameters are updated; $\theta$ remains frozen. LoRA parameters are re-initialized after each consistency computation.
- **Advantage**: Introduces structured perturbation directly in parameter space and is naturally compatible with LoRA fine-tuning pipelines.

**Strategy 2: Representation Steering**

- A learnable steering vector $d_l$ is added directly to the internal representation at each layer $l$: $h'_{l,t} = h_{l,t} + d_l$.
- The original parameters $\theta$ are not modified; representations are altered purely via vector addition.
- **Advantage**: Computationally more efficient, requiring no additional matrix multiplications.

### Key Design 3: Representation-Level Consistency Regularization

- **Loss Function**: $\ell_{\text{cons.}}(\theta, \theta_{\text{aux}}) = \mathbb{E}_{(x,y)\sim\mathcal{D}} \left[\frac{1}{TK} \sum_{t=1}^{T} \sum_{l=1}^{K} \|\text{detach}(h_{l,t}) - h'_{l,t}\|^2\right]$
- **Detach Operation**: Gradient flow through the original model's representation $h_{l,t}$ is stopped, making it a fixed reference target and preventing the consistency loss from pulling the original model toward the undesirable direction.
- **Core Intuition**: The consistency loss encourages the auxiliary model's undesirable representations $h'_{l,t}$ to align with the original model's representations $h_{l,t}$, indirectly training the original model to adopt parameter configurations that are intrinsically resistant to undesirable perturbations.
- **Full-Layer Coverage**: Averaging over all $K$ layers and all $T$ token positions ensures representation stability at every layer.

### Training Procedure

1. Forward pass to compute SFT loss $\ell_{\text{SFT}}(\theta)$
2. Construct the auxiliary model and forward pass to compute $\ell_{\text{SFT}}(\theta_{\text{aux}})$
3. Apply one normalized gradient ascent step to the auxiliary components
4. Forward pass the auxiliary model again to extract undesirable representations $h'_{l,t}$
5. Compute the MSE consistency loss between the original and auxiliary models
6. Combine $\ell_{\text{SFT}} + \lambda \cdot \ell_{\text{cons.}}$ and backpropagate to update $\theta$

---

## Key Experimental Results

### Experimental Setup

- **Models**: Llama-3.1-8B, Llama-3.1-8B-Instruct, Llama-2-7B, Mistral-7B-v0.3
- **Fine-Tuning Data**: GSM8k (mathematics), Alpagasus Dolly 3k (multi-task)
- **Evaluation**: 11 tasks spanning four categories—mathematics (3), knowledge (4), reasoning (2), and helpfulness (2); additional experiments on prompt robustness and adversarial fine-tuning

### Main Results

| Metric | SFT Baseline | LfU |
|--------|-------------|-----|
| Math Average (Llama-3.1-8B) | 51.5 | **54.2** (+16.8% vs. vanilla) |
| Prompt Variant Std. Dev. | High | **Reduced by 92.1%** |
| Adversarial Fine-Tuning ASR (HEx-PHI) | High | **Reduced by 45.0%** |
| Overall Ranking | 3–5 | **1.0–1.5** (consistently first) |

### Key Findings

1. **Strong In-Domain Improvement**: On the GSM8k fine-tuning + math evaluation setting, LfU improves in-domain math scores by 16.8% over SFT on Llama-3.1-8B, outperforming all baselines.
2. **Best Out-of-Domain Retention**: Knowledge, reasoning, and helpfulness categories show no degradation; LfU ranks first overall across all models.
3. **Exceptional Prompt Robustness**: Under paraphrased prompts with equivalent semantics, the standard deviation of LfU outputs is reduced by 92.1%.
4. **Resistance to Adversarial Fine-Tuning**: After safety alignment via BeaverTails followed by adversarial fine-tuning, LfU exhibits a substantially lower attack success rate than SFT, with a 45% reduction in ASR on HEx-PHI.
5. **Strong Compatibility**: Both auxiliary model construction strategies (LoRA-based and representation steering) are effective, with the LoRA-based approach performing slightly better.

---

## Rating

### Strengths

- Elegant method design: The use of "undesirable updates" as representation-level data augmentation is novel, and its integration with consistency regularization is natural.
- Comprehensive experiments: Three orthogonal evaluation dimensions—in/out-of-domain generalization, prompt robustness, and adversarial safety—lend strong empirical credibility.
- Plug-and-play: The method does not alter the basic SFT pipeline; it only adds a regularization term, making it straightforward to integrate in practice.

### Limitations & Future Work

- Increased computational cost: Additional forward passes and gradient computations for the auxiliary model increase training time by approximately 30–50%.
- Hyperparameter sensitivity: The choice of $\lambda$ and $\alpha$ affects performance, and the paper provides insufficient guidance for automated tuning.
- Experiments are limited to 7–8B scale models; the effectiveness on larger models (e.g., 70B) remains unverified.

---

## Related Work & Insights

- **NEFTune**: Injects noise into input embeddings; a special case of LfU (perturbation at the input only, whereas LfU perturbs at the parameter/representation level) with weaker performance.
- **SAM (Sharpness-Aware Minimization)**: Applies gradient ascent over all parameters to find flat loss landscapes; LfU applies ascent only to auxiliary components and operates at the representation level.
- **Consistency Regularization**: Originates from semi-supervised learning (FixMatch/MixMatch); LfU generalizes it from "prediction consistency under input perturbation" to "representation consistency under parameter perturbation."
- **DPO/RLHF**: Alignment-stage methods; LfU enhances generalization at the SFT stage and can be combined with these approaches.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] TOFA: Training-Free One-Shot Federated Adaptation for Vision-Language Models](tofa_training-free_one-shot_federated_adaptation_for_vision-language_models.md)
- [\[CVPR 2026\] The Blind Spot of Adaptation: Quantifying and Mitigating Forgetting in Fine-tuned Driving Models](../../CVPR2026/llm_safety/blind_spot_of_adaptation_quantifying_and_mitigating_forgetting_in_fine_tuned_driving_models.md)
- [\[AAAI 2026\] Anti-adversarial Learning: Desensitizing Prompts for Large Language Models](anti-adversarial_learning_desensitizing_prompts_for_large_la.md)
- [\[ICCV 2025\] LATTE: Collaborative Test-Time Adaptation of Vision-Language Models in Federated Learning](../../ICCV2025/llm_safety/latte_collaborative_test-time_adaptation_of_vision-language_models_in_federated_.md)
- [\[ICLR 2026\] Bi-directional Bias Attribution: Debiasing Large Language Models without Modifying Prompts](../../ICLR2026/llm_safety/bi-directional_bias_attribution_debiasing_large_language_models_without_modifyin.md)

</div>

<!-- RELATED:END -->
