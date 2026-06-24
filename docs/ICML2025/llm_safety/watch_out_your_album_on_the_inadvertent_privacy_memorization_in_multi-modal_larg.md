---
title: >-
  [Paper Note] Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models
description: >-
  [ICML 2025][LLM Safety][MLLM Privacy] Reveals that Multi-Modal Large Language Models (MLLMs) inadvertently memorize sensitive private content (e.g., random watermarks) completely unrelated to the training task during fine-tuning. This memorization stems from spurious correlations within mini-batches. A layer-wise linear probing framework is proposed to demonstrate that such information is encoded within the model's internal representations even when not directly manifest in t…
tags:
  - "ICML 2025"
  - "LLM Safety"
  - "MLLM Privacy"
  - "Inadvertent Memorization"
  - "Task-Irrelevant Content"
  - "Watermark Probing"
  - "Mini-batch Spurious Correlation"
  - "Layer-wise Probing"
date: 2026-05-08
content_hash: 37afc3623bc59c93
---

# Watch Out Your Album! On the Inadvertent Privacy Memorization in Multi-Modal Large Language Models

**Conference**: ICML 2025  
**arXiv**: [2503.01208](https://arxiv.org/abs/2503.01208)  
**Code**: [https://github.com/illusionhi/ProbingPrivacy](https://github.com/illusionhi/ProbingPrivacy)  
**Area**: AI Safety  
**Keywords**: MLLM Privacy, Inadvertent Memorization, Task-Irrelevant Content, Watermark Probing, Mini-batch Spurious Correlation, Layer-wise Probing

## TL;DR
Reveals that Multi-Modal Large Language Models (MLLMs) inadvertently memorize sensitive private content (e.g., random watermarks) completely unrelated to the training task during fine-tuning. This memorization stems from spurious correlations within mini-batches. A layer-wise linear probing framework is proposed to demonstrate that such information is encoded within the model's internal representations even when not directly manifest in the generated outputs.

## Background & Motivation

**Background**: Multi-Modal Large Language Models (MLLMs) such as LLaVA and InternVL perform exceptionally well in vision-language tasks like VQA, but their training data inevitably contains privacy-sensitive information. Existing studies on privacy leakage (such as data extraction attacks and Membership Inference Attacks - MIA) predominantly focus on sensitive content that naturally aligns with the training objectives.

**Limitations of Prior Work**: Prior work focuses on "task-relevant" private content—where personal information in textual modality is naturally memorized through next-token prediction, and private image features in visual modality are also tightly bound to the main task. However, an overlooked question is: Will private content completely irrelevant to the training target also be memorized by the model?

**Key Challenge**: From the perspective of global training objectives, task-irrelevant content should not affect model learning. However, within the dynamics of partial mini-batch training, such content can develop spurious correlations with VQA outputs, causing the model to encode this information into its parameters through normal gradient updates, representing a novel and unforeseen privacy risk.

**Goal**: To answer three core questions—(RQ1) Does random task-irrelevant content affect fine-tuning dynamics? (RQ2) Does the model encode this irrelevant content within its internal representations? (RQ3) How is this encoding distributed across different layers?

**Key Insight**: Systematically study the memorization mechanism and detectability by injecting fully random task-irrelevant watermarks in a controlled manner to simulate "private content."

**Core Idea**: Design watermark injection experiments coupled with a layer-wise linear probing framework to demonstrate that MLLMs encode task-irrelevant information within their hidden states.

## Method

### Overall Architecture

The paper proposes a two-stage experimental and probing framework:

- **Stage 1: Controlled Watermark Injection**: Embed randomly generated task-irrelevant watermark content onto images of the VQA fine-tuning dataset at varying probabilities (e.g., 10%, 30%, 50%, 100%). The watermark text is a randomly generated string, entirely unrelated to the VQA QA-pairs. Standard VQA fine-tuning is then executed.
- **Stage 2: Privacy Probing**: After training, a linear probe is trained on the hidden states of each layer to detect whether the model encodes the watermark information in its internal representations. In parallel, the impact of various watermark injection probabilities on training loss, convergence behavior, and downstream task performance is analyzed.

Key Insight: Within a mini-batch, if some images contain the same watermark while others do not, the presence of the watermark might form a spurious correlation with specific VQA answer targets within that batch, driving the model to unconsciously incorporate watermark information into the parameters during gradient descent.

### Key Designs

1. **Watermark Injection Strategy**:

    - Function: Overlay random text watermarks on VQA images to simulate task-irrelevant private content.
    - Mechanism: Use randomly generated text strings as watermarks, embedded into the fine-tuning images with different probabilities $p \in \{0.1, 0.3, 0.5, 1.0\}$. The watermark content is entirely random, ensuring no semantic association with the Q-A pairs.
    - Design Motivation: In reality, private photos in users' albums may contain various background details (e.g., road signs, ID numbers, document fragments). Although this content is noise for VQA tasks, it might still be encoded by the model.

2. **Mini-batch Spurious Correlation Analysis**:

    - Function: Analyze why globally irrelevant content becomes "relevant" at the mini-batch scale.
    - Mechanism: Within a single mini-batch, the subset of samples containing the same watermark shares an extra visual feature (the watermark). The SGD gradient updates tend to associate this feature with the answers of those samples. Although this association is statistically spurious across different batches, the model cannot automatically eliminate this memorization within finite training epochs.
    - Design Motivation: Reveal the inherent limitations of mini-batch SGD—showing that an unbiased estimate of the global gradient does not guarantee that the model will not memorize local information.

3. **Layer-wise Linear Probing**:

    - Function: Train linear classifiers on the hidden states of every layer of the trained MLLM to probe whether the watermark can be recovered from internal representations.
    - Mechanism: Let $h_l$ be the hidden state representation of the $l$-th layer. Train a linear classifier $f_l(h_l) = W_l h_l + b_l$ to predict whether the input image contains the watermark (or the watermark content). If the classification accuracy of a specific layer is significantly higher than chance, it proves that the layer encodes the watermark information.
    - Design Motivation: Direct prompting may fail to elicit watermark information (as the model does not show the watermark in its output), but linear probing can reveal the information encoded within the representation space.

4. **Training Behavior Difference Analysis**:

    - Function: Compare the training loss curves and convergence characteristics of the models with and without watermarks.
    - Mechanism: Observe the trends in training loss, differences in convergence speeds, and performance fluctuations on the validation set after injecting watermarks of varying ratios.
    - Design Motivation: If task-irrelevant content indeed affects training dynamics, training loss and performance metrics should display identifiable distinct patterns.

## Key Experimental Results

### Training Dynamics Analysis (RQ1)

Differences in fine-tuning behavior of MLLM under different watermark injection ratios:

| Watermark Injection Probability | Training Loss Trend | Downstream VQA Performance Impact | Convergence Behavior |
|------------|------------------|-----------------|---------|
| 0% (Baseline) | Normal decrease, stable convergence | Baseline level | Standard convergence |
| 10% | Slight fluctuation | Almost no impact | Near normal |
| 30% | Distinct training pattern | Slight degradation | Differences emerge |
| 50% | Significantly different training curve | Observable degradation | Obvious delay |
| 100% | Drastic change in training dynamics | Significant performance loss | Difficulty in convergence |

Key Finding: Even when only 10%–30% of the images contain watermarks, the model's training dynamics exhibit measurable shifts, indicating that the intuition "task-irrelevant content does not affect training" is incorrect.

### Layer-wise Probing Detection Results (RQ2 & RQ3)

Performance of layer-wise linear probing in detecting watermark encoding under different conditions:

| Model Layer | Accuracy on Watermarked Images | Accuracy on Clean Images | Interpretation of Difference |
|---------|-----------------|-----------------|---------|
| Shallow Layers (First 1/4) | Medium | ~Random guessing | Signs of encoding begin to emerge in shallow layers |
| Middle Layers (1/4 - 1/2) | High | ~Random guessing | Watermark information encoding is strengthened |
| Deep Layers (1/2 - 3/4) | Highest | ~Random guessing | Peak encoding region |
| Near Output Layer | Decreased | ~Random guessing | Information is partially filtered by the output projection |

Key Conclusions:
- Even if watermark content does not affect the model's VQA outputs, linear probes can still recover watermark info from middle layers with accuracy significantly higher than random guessing.
- When encountering previously "seen" task-irrelevant knowledge, MLLMs trigger activation patterns distinctly different from unseen content.
- This demonstrates "no leakage $\neq$ no memorization"—the model may avoid displaying private information at the output layer while still having encoded it within internal representations.

## Highlights & Insights

- **Counter-intuitive Finding**: Task-irrelevant content is also memorizing, which is more insidious and harder to defend against than task-relevant privacy leaks. Conventional privacy protection strategies (e.g., output filtering) cannot resolve memorization within internal representations.
- **Deep Explanation of the Mini-batch Spurious Correlation Mechanism**: Linking privacy risks directly to the intrinsic nature of stochastic gradient descent reveals a systematic vulnerability deeply bound to the training methodology itself.
- **Layer-wise Probing Methodology provides a new tool for privacy auditing**: It can serve as a privacy diagnostics tool prior to model deployment, quantitatively evaluating whether the model has encoded unauthorized information.
- **Novelty of Problem Definition**: The study systematically investigates "task-irrelevant privacy memorization" for the first time, expanding privacy research from "whether models can be attacked to extract data" to "whether models inadvertently encode completely unrelated data."

## Limitations & Future Work

- **Ecological Validity of Watermarks as Privacy Proxies**: The feature distribution of random text watermarks differs from real-world private information (e.g., faces, ID numbers). The degree of inadvertent memorization in practical scenarios may vary.
- **Lack of Mitigation Strategies**: The paper focuses on identifying and analyzing the issue but does not propose effective defenses or mitigation schemes (such as differential privacy fine-tuning or representation regularization).
- **Limited Model Scope**: The generalizability needs verification across more model architectures (like Qwen-VL or closed-source models such as GPT-4o) and larger parameter scales.
- **Single Watermark Injection Approach**: The study only considers text watermarks superimposed on images, without exploring other types of task-irrelevant content (e.g., background objects, noise patterns, or metadata embedding).
- **Impact of Training Epochs**: There is no in-depth analysis of how the number of training epochs affects memorization, nor is there an exploration of forgetting dynamics (e.g., whether continuing fine-tuning post-training can eliminate memorization).
- **Limitation of Linear Probes**: Linear probes can only detect linearly separable encoding patterns; private information encoded non-linearly might be missed.

## Related Work & Insights

- **vs. Membership Inference Attacks (MIA)**: Traditional MIA focuses on "whether data was in the training set," whereas this work focuses on "whether task-irrelevant information during training was encoded"—a more upstream and fundamental issue.
- **vs. Carlini et al. Data Extraction Attacks**: Carlini et al.'s work demonstrates that LLMs can regurgitate training data verbatim, but that data directly aligns with the training objective (next-token prediction). In contrast, the watermark content here is completely unaligned, pointing to a different memorization mechanism.
- **vs. Differential Privacy Training**: DP-SGD prevents privacy leakage by clipping gradients and adding noise. However, it remains unclear whether the protection of DP-SGD is sufficient against memorization caused by spurious correlations within mini-batches.
- **vs. Model Watermarking/Backdoor Attacks**: Backdoor attacks involve adversaries intentionally injecting triggers to tie to specific outputs. The setting here is "inadvertent" encoding of irrelevant content—mechanisms are similar, but the intent and scenarios differ.
- **Insights**: Based on these findings, privacy auditing tools for MLLM fine-tuning can be designed—scanning for improper memorization using layer-wise probes before model deployment.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of task-irrelevant privacy memorization in MLLMs; the problem definition is novel.
- Experimental Thoroughness: ⭐⭐⭐ Though cached content is limited, the experimental design (watermark injection + layer-wise probing) is logically clear and highly controllable.
- Writing Quality: ⭐⭐⭐⭐ Motivations are clearly articulated, and Figure 1 effectively contrasts existing work with ours.
- Value: ⭐⭐⭐⭐ Exposes an overlooked source of privacy risk, offering significant warning value for the secure deployment of MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Private Memorization Editing: Turning Memorization into a Defense to Strengthen Data Privacy in Large Language Models](../../ACL2025/llm_safety/private_memorization_editing_turning_memorization_into_a_defense_to_strengthen_d.md)
- [\[ICML 2025\] Cut out and Replay: A Simple yet Versatile Strategy for Multi-Label Online Continual Learning](cut_out_and_replay_a_simple_yet_versatile_strategy_for_multi-label_online_contin.md)
- [\[ICLR 2026\] Doxing via the Lens: Revealing Location-related Privacy Leakage on Multi-modal Large Reasoning Models](../../ICLR2026/llm_safety/doxing_via_the_lens_revealing_location-related_privacy_leakage_in_vlms.md)
- [\[AAAI 2026\] AUVIC: Adversarial Unlearning of Visual Concepts for Multi-modal Large Language Models](../../AAAI2026/llm_safety/auvic_adversarial_unlearning_of_visual_concepts_for_multi-mo.md)
- [\[ACL 2025\] Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport](../../ACL2025/llm_safety/opt-out_investigating_entity-level_unlearning_for_large_language_models_via_opti.md)

</div>

<!-- RELATED:END -->
