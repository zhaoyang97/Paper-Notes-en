---
title: >-
  [Paper Note] Safety Alignment Can Be Not Superficial With Explicit Safety Signals
description: >-
  [ICML 2025][LLM Safety][Safety Alignment] By introducing an explicit binary safety classification task (via a `[CLS]` token) into LLMs, and designing a strategic attention mechanism alongside strategic decoding strategies to dynamically evaluate safety during inference, this work reduces the attack success rate of adversarial attacks from over 90% to nearly 0% with less than $0.2\times$ extra overhead.
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "Safety Alignment"
  - "Jailbreak Defense"
  - "Binary Classification"
  - "Strategic Decoding"
  - "Adversarial Robustness"
date: 2026-05-08
content_hash: d4fdfdd84ba9d196
---

# Safety Alignment Can Be Not Superficial With Explicit Safety Signals

**Conference**: ICML 2025  
**arXiv**: [2505.17072](https://arxiv.org/abs/2505.17072)  
**Code**: [https://sa-ess.github.io/](https://sa-ess.github.io/)  
**Area**: LLM Alignment / Safety  
**Keywords**: Safety Alignment, Jailbreak Defense, Binary Classification, Strategic Decoding, Adversarial Robustness

## TL;DR

By introducing an explicit binary safety classification task (via a `[CLS]` token) into LLMs, and designing a strategic attention mechanism alongside strategic decoding strategies to dynamically evaluate safety during inference, this work reduces the attack success rate of adversarial attacks from over 90% to nearly 0% with less than $0.2\times$ extra overhead.

## Background & Motivation

Existing safety alignment methods (SFT/DPO/RLHF) for LLMs have been found to perform only "superficial alignment" — the safety mechanisms can be easily bypassed when facing well-crafted adversarial attacks (such as jailbreaks, prefill attacks, and decoding parameter attacks). Prior work (Li & Kim 2024) points out that safety alignment can essentially be reduced to a binary classification task (refuse/execute). However, existing methods force the model to learn this task **implicitly**, causing the safety signal to be diluted by other optimization objectives (such as tone, style, and preferences).

Specifically, this manifests as:
- Under adversarial attacks, the top-$K$ logits of the model exhibit obvious hesitation and confusion (high entropy, low sharpness).
- Existing data augmentation schemes (Qi et al. 2024, Yuan et al. 2024) can only handle simple safety-to-unsafety transitions, failing to address nested or late-stage harmful content within responses.
- Implicit safety signals are unreliable in adversarial scenarios, resulting in blurry decision boundaries.

**Key Insight**: Shifting safety judgment from implicit reasoning to an explicit binary classification task can fundamentally resolve the issue of superficial safety alignment.

## Method

### Overall Architecture

This approach introduces a special `[CLS]` token during the pre-training and SFT stages to act as a safety classifier. In the inference stage, two components—a strategic attention mechanism and sequential decoding strategies—are leveraged to implicitly and explicitly guide the generation process using safety signals. The overall design consists of three steps:
1. Training Phase: Prepend the `[CLS]` token to the input sequence and use a classification head to determine whether the output is benign or malicious.
2. Inference Phase (Implicit): Allow the hidden states of the `[CLS]` token to influence the generated tokens via the attention mechanism.
3. Inference Phase (Explicit): Direct the decoding process using predictions from the `[CLS]` classifier.

### Key Designs

#### 1. **Explicit Safety Binary Classification Task (`[CLS]` Token)**

Inspired by BERT, a `[CLS]` token is prepended to each input sequence, and its output is passed through a classification head to determine the safety of the input and the generated content. To balance classification and generation capabilities, a fine-grained attention routing mechanism is designed:

- **Pre-training Stage**: The `[CLS]` token can attend to all tokens, but other tokens cannot attend to `[CLS]`, maintaining the original causal attention mechanism.
- **SFT Stage**: Query tokens cannot attend to `[CLS]`, while response tokens can attend to `[CLS]`. The `[CLS]` token can only attend to query tokens and cannot attend to response tokens.
- **Small coefficients** are used in both stages to control the weight of the classification loss, preventing the classification objective from dominating optimization.

Regarding dataset construction, LLaMA3-Guard is used to automatically label Wikipedia data for pre-training. For SFT, a balanced dataset of 29,600 samples is constructed by sampling from LIMA (benign), ALERT (malicious), and Alpaca.

#### 2. **Strategic Attention Mechanism**

During inference, the `[CLS]` token is dynamically re-evaluated, and its attention span is adjusted based on the current generation state. Four strategic rules are designed:

- **Rule 1** (Initially Classified as Malicious): The `[CLS]` token only attends to the input tokens and the first $r_1$ generated tokens, without needing to attend to subsequent tokens.
- **Rule 2** (Initially Classified as Benign): The `[CLS]` token only attends to the most recent $r_2$ generated tokens, focusing on new content while reducing computational cost.
- **Rule 3** (Transition from Benign to Malicious): The transition point $S_t$ is recorded, and attention is focused on the range $[S_t - r_2, S_t + r_3]$. This provides a fault-tolerance mechanism to prevent key-word triggered misclassifications; if a false trigger occurs, the model automatically rolls back to Rule 2.
- **Rule 4**: Skip auxiliary tokens such as PAD, BOS, and instruction tokens.

Hyperparameters are set as $r_1 = r_2 = r_3 = 10$, which can be flexibly adjusted based on application scenarios.

#### 3. **Strategic Decoding Strategy**

Explicitly leveraging the output of the `[CLS]` classifier to guide decoding, a three-level dependency scheme is proposed:

- **Low Dependency**: Fully relies on the attention mechanism and ignores the classification prediction (unreliable).
- **High Dependency**: Immediately terminates generation and outputs a fixed refusal response once classified as malicious (leads to a high false-positive rate).
- **Medium Dependency (Ours)**:
    - If initially classified as malicious $\rightarrow$ immediately insert prompt tokens ("Sorry, I cannot fulfill your request because...") and explain the reason.
    - If initially classified as benign but subsequently classified as malicious for $\tau$ consecutive steps $\rightarrow$ insert prompt tokens at the transition point.

Design Motivation:
- Resolves blurry decision boundaries: In adversarial queries, classification probabilities often hover around 0.5. The decoding strategy enforces a clear decision.
- Adopts a Chain-of-Thought-style refusal: Besides refusing, it also explains the reason, which enhances the model's comprehension, reduces the possibility of reversals, and decreases false positives.

### Loss & Training

Pre-training Loss: 
$$\mathcal{L}_{\text{pretraining}} = \mathcal{L}_{\text{lm}} + \lambda_1 \cdot \mathcal{L}_{\text{cls}}$$

Alignment Loss: 
$$\mathcal{L}_{\text{alignment}} = \mathcal{L}_{\text{sft}} + \lambda_2 \cdot \mathcal{L}_{\text{cls}}$$

Where $\mathcal{L}_{\text{cls}}$ is the cross-entropy loss between the output of the `[CLS]` token and the ground truth, $\lambda_1 = 0.01$, $\lambda_2 = 0.1 / 0.01$, and $\tau \leq 3$.

During inference, an **Annealing strategy** is adopted to reduce the re-classification frequency: evaluations are frequent early on and gradually decrease in frequency until re-evaluation ceases. This maintains a safety performance comparable to step-by-step evaluation while incurring less than $0.2\times$ extra overhead.

## Key Experimental Results

### Main Results

Base model: Llama2-7B; aligned model: Mistral-7B-Instruct-v0.2.

| Dataset / Attack | Metric (ASR↓) | Llama2-7B-CLS (Ours) | Llama2-7B-Chat (RLHF) | Gain |
|-------------|-----------|---------------------|----------------------|------|
| AdvBench / Prefill | ASR | **0.4%** | 39.62% | ~100x |
| HEx-PHI / Prefill | ASR | **1.2%** | 60.91% | ~50x |
| HarmBench / GCG | ASR | **0.0%** | 28.0% | Complete Defense |
| AdvBench / Decoding | ASR | **0.0%** | 87.0% | Complete Defense |
| MaliciousInstruct / Decoding | ASR | **0.0%** | 83.0% | Complete Defense |
| AdvBench / AutoDAN-T | ASR | **0.77%** | 61.3% | ~80x |
| AdvBench / PAP | ASR | **0.0%** | 28.26% | Complete Defense |

Comparison with SOTA data augmentation methods (Qi et al. 2024):

| Method | Prefill 5T | Prefill 40T | Decoding (HEx-PHI) | Decoding (MalInst) |
|------|-----------|------------|-------------------|-------------------|
| Llama2-7B-Chat | 42.1% | 57.0% | 54.9% | 84.3% |
| Llama2-7B-Chat-Aug | 2.8% | 4.5% | 11.3% | 1.0% |
| **Llama2-7B-CLS** | **0.9%** | **2.1%** | **0.0%** | **0.0%** |

### Ablation Study

| Configuration | Key Metric (ASR) | Description |
|------|-------------|------|
| Without Pre-training Stage | Slightly increased | Pre-training is helpful but offers limited gain, potentially due to labeling noise in LLaMA3-Guard |
| Without Strategic Attention | Significant decline | The attention mechanism ensures the sensitivity of safety signals to changes in inference directions |
| Without Strategic Decoding | Significant decline | The decoding strategy ensures that the model responds to safety changes timely and effectively |
| FirstOnly Re-classification | Worst | Only initial classification is insufficient to counter subsequent attacks |
| Periodic (every 10 steps) | Close to optimal | Periodic evaluations already yield promising results |
| Annealing | ≈Every | Achieves comparable performance to step-by-step evaluation with <0.2x overhead |
| Every (step-by-step) | Optimal | Highest computational overhead |

### Key Findings

1. **Probing Experiments**: As adversarial complexity increases (Direct $\rightarrow$ Prefill $\rightarrow$ Nested), model output entropy increases while sharpness decreases, indicating a lack of confidence and unstable safety reasoning in the model under adversarial attacks.
2. **GCG Attacks Rendered Completely Ineffective**: The dynamic re-classification mechanism disrupts the static adversarial signals that GCG relies on, rendering the optimized adversarial suffixes ineffective.
3. **Applicable Across Model Families**: The augmented Mistral-7B-Instruct-v0.2-CLS surpasses Llama2-7B-Chat in safety for the first time, while preserving the Mistral family's advantages on MT-Bench (7.38) and GSM8K (41.77).
4. **Insensitive to Sampling**: The proposed method achieves a standard deviation close to zero across multiple rounds of generation, rendering it naturally immune to decoding attacks.

## Highlights & Insights

1. **Paradigm Shift from Implicit to Explicit**: Shifting safety judgment from implicit reasoning embedded within the generation process to an explicit binary classification task is a clear and highly effective paradigm.
2. **Dynamic Re-evaluation**: Instead of a one-off safety judgment, continuous monitoring throughout the entire generation process enables the model to handle nested attacks and late-occurring harmful content.
3. **Excellent Computational Efficiency**: The Annealing strategy keeps the additional overhead under $0.2\times$. Requiring only one extra token during the training phase makes it highly feasible for actual distribution.
4. **Deep Insight into GCG Defense**: Dynamic re-classification fundamentally breaks the prerequisite of gradient-optimized attacks, which is the reliance on static adversarial signals.
5. **Compatibility with Existing Methods**: The proposed approach can be applied as a post-enhancement stage on top of SFT/DPO/RLHF, rather than acting as a replacement.

## Limitations & Future Work

1. **Limited to Text Modality**: The effectiveness in multimodal scenarios (e.g., image-text, speech) has not yet been validated.
2. **Limited Pre-training Gain**: Constrained by the annotation quality of LLaMA3-Guard and computing resources, the additional gains from the pre-training stage are not significant.
3. **Overly Simplified Binary Classification**: The degree of harm is a continuous spectrum; binary classification may lack nuance for borderline cases.
4. **Sensitivity to Hyperparameters**: Hyperparameters such as $r_1$, $r_2$, $r_3$, and $\tau$ need to be adjusted for different scenarios, leaving generalizability to be fully verified.
5. **Risk of Over-refusal**: Although mitigated by CoT explanations and the $\tau$-consecutive-steps threshold, the false-positive rate in real-world deployment requires more extensive evaluation.

## Related Work & Insights

- **Li & Kim (2024)**: Proposed the superficial safety alignment hypothesis. This work provides a direct solution to that hypothesis.
- **Qi et al. (2024)**: Data augmentation methods increase training diversity, but this work demonstrates that explicit signals are more effective than simply using more data.
- **BERT [CLS] token**: Cleverly adapts the design from discriminative models to safety scenarios in generative models.
- **Insights**: Safety alignment should not rely solely on data or training strategies; explicit architectural designs at the model level may offer a more fundamental path to solutions. Similar ideas can be extended to other reliability requirements (such as truthfulness and fairness).

## Rating

- Novelty: ⭐⭐⭐⭐ — The combination of explicit safety signals and dynamic re-classification is novel, though the overall framework is relatively intuitive.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ — Covers a wide range of attack types, compares against multiple baselines, and includes rich ablation studies.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured and effectively organized in a challenge-solution layout, although the LaTeX equations are slightly redundant.
- Value: ⭐⭐⭐⭐ — Safety alignment is a critical issue; the proposed method is highly practical and can be stacked on top of existing solutions, though its extensibility to multimodality remains unverified.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2025\] Improving LLM Safety Alignment with Dual-Objective Optimization](improving_llm_safety_alignment_with_dual-objective_optimization.md)
- [\[ICLR 2026\] SOSBench: Benchmarking Safety Alignment on Six Scientific Domains](../../ICLR2026/llm_safety/sosbench_benchmarking_safety_alignment_on_six_scientific_domains.md)
- [\[ICLR 2026\] The Alignment Waltz: Jointly Training Agents to Collaborate for Safety](../../ICLR2026/llm_safety/the_alignment_waltz_jointly_training_agents_to_collaborate_for_safety.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](../../ACL2026/llm_safety/reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)
- [\[ICML 2025\] Learning Safety Constraints for Large Language Models](learning_safety_constraints_for_large_language_models.md)

</div>

<!-- RELATED:END -->
