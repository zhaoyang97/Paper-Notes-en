---
title: >-
  [Paper Note] DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding
description: >-
  [ICCV 2025][Multimodal VLM][rule-based RL] This paper proposes DocThinker, the first framework to apply GRPO (Group Relative Policy Optimization) reinforcement learning to document understanding. By training MLLMs with a…
tags:
  - "ICCV 2025"
  - "Multimodal VLM"
  - "rule-based RL"
  - "GRPO"
  - "document understanding"
  - "interpretability"
  - "chain-of-thought"
date: 2026-05-08
content_hash: ca50bb0a1d590807
---

# DocThinker: Explainable Multimodal Large Language Models with Rule-based Reinforcement Learning for Document Understanding

**Conference**: ICCV 2025
**arXiv**: [2508.08589](https://arxiv.org/abs/2508.08589)
**Code**: [https://github.com/wenwenyu/DocThinker](https://github.com/wenwenyu/DocThinker)
**Area**: Multimodal VLM / Document Understanding / Reinforcement Learning Reasoning
**Keywords**: rule-based RL, GRPO, document understanding, interpretability, chain-of-thought

## TL;DR
This paper proposes DocThinker, the first framework to apply GRPO (Group Relative Policy Optimization) reinforcement learning to document understanding. By training MLLMs with a four-objective rule-based reward (format, answer accuracy, RoI IoU, and question rephrasing quality), DocThinker enables models to autonomously generate interpretable reasoning processes. Using only 4K training samples, it improves Qwen2.5-VL-7B on DocVQA from 0.355 to 0.579 (RL vs. SFT: 0.579 vs. 0.355) and achieves 82.4% precision on visual grounding tasks.

## Background & Motivation
**Background**: MLLMs have demonstrated strong performance in document understanding, yet their reasoning processes remain opaque, limiting trustworthiness in high-stakes domains such as law, finance, and medicine. Existing methods primarily employ fixed CoT templates for reasoning—e.g., ReFocus uses external tools for image editing, Visual CoT performs multi-round processing, and MVoT generates interleaved visual-textual reasoning chains.

**Limitations of Prior Work**: (1) Fixed CoT templates lack flexibility and generalize poorly across tasks; (2) SFT training is prone to catastrophic forgetting, leading to performance degradation on new document types; (3) existing methods output only final answers, lacking interpretability of intermediate reasoning steps.

**Key Challenge**: SFT causes models to memorize reasoning patterns from training data, preventing autonomous exploration of better reasoning strategies. Works such as DeepSeek-R1 have demonstrated that pure RL can elicit emergent reasoning capabilities, yet RL for document understanding remains largely unexplored.

**Goal**: To replace SFT with RL for training document understanding MLLMs, enabling models to autonomously learn flexible reasoning strategies while generating interpretable intermediate steps (reasoning process, rephrased question, region of interest, and final answer).

**Key Insight**: Inspired by DeepSeek-R1 and MedVLM-R1, this work applies GRPO to multimodal document understanding and designs four verifiable rule-based reward functions to guide model learning.

**Core Idea**: Replace SFT with GRPO reinforcement learning and four-objective rule-based rewards to train document understanding MLLMs, achieving adaptive reasoning and interpretable outputs.

## Method

### Overall Architecture
Built upon Qwen2.5-VL (3B/7B), the model takes a document image and a question as input and produces structured output in the form: `<think>reasoning process</think><answer>{"rephrase_question": ..., "bbox_2d": ..., "final_answer": ...}</answer>`. The GRPO algorithm samples $G=6$ candidate outputs; the four rule-based rewards evaluate each candidate, compute within-group relative advantages, and optimize the policy under KL-divergence constraints.

### Key Designs

1. **Four-Objective Rule-Based Reward Functions**:

    - **Format Reward $R_{\text{format}}$**: Checks whether the output adheres to the XML-style schema (`<think>...</think>` and `<answer>...</answer>`), and whether the JSON is valid and contains the required key-value pairs. Binary reward.
    - **Accuracy Reward $R_{\text{accuracy}}$**: Checks whether the final answer matches the ground truth. Binary reward.
    - **RoI IoU Reward $R_{\text{RoI}}$**: Checks whether the IoU between the predicted bounding box and the ground truth is $\geq 0.5$. Encourages the model to precisely localize document regions relevant to the answer, enhancing visual interpretability.
    - **Rephrase Reward $R_{\text{rephrase}}$**: Evaluates semantic similarity and lexical diversity of the rephrased question. Computed only when the answer is correct, to avoid rewarding well-rephrased questions paired with incorrect answers.
    - Total reward: $R_{\text{total}} = \lambda_1 R_{\text{format}} + \lambda_2 R_{\text{accuracy}} + \lambda_3 R_{\text{RoI}} + \lambda_4 R_{\text{rephrase}}$, with all $\lambda_i = 1$.
    - **Design Motivation**: Each reward signal is automatically verifiable (requiring no human preference annotation) and targets a distinct aspect of model behavior (format compliance, answer correctness, visual grounding, and question comprehension).

2. **GRPO Training Strategy (replacing PPO)**:

    - **Function**: For each question, $G=6$ candidate outputs are sampled, evaluated by the rule-based rewards, and normalized into within-group relative advantages $A_i = (r_i - \text{mean}) / \text{std}$. The policy is optimized to increase the probability of high-advantage responses.
    - **Mechanism**: No additional critic network is required (unlike PPO); advantage estimation relies solely on within-group relative comparison. KL divergence regularization ($\beta = 0.04$) prevents the policy from deviating too far from the reference model, mitigating catastrophic forgetting.
    - **Design Motivation**: GRPO is computationally efficient, eliminates the need to train a value network, and has been validated in DeepSeek-R1 for eliciting emergent reasoning capabilities.

3. **Structured Interpretable Output**:

    - **Function**: The model output comprises four components—free-form reasoning within `<think>`, `rephrase_question` (a clarified restatement of the input question), `bbox_2d` (coordinates of the document region supporting the answer), and `final_answer`.
    - **Mechanism**: The reasoning process reveals *how* the model thinks; the rephrased question reveals *how* the model understands the question; the bounding box reveals *where* the model looks in the document—providing three complementary dimensions of interpretability.
    - **Design Motivation**: This yields richer interpretable information than outputting only a final answer. Moreover, the reasoning process is continuously refined during RL training through self-reflection and self-correction, which methods such as VoT cannot achieve.

### Training Configuration
- Base models: Qwen2.5-VL 3B/7B; hardware: 8× A100 80G
- Training data: 4K or 8K samples from the Visual CoT dataset (far fewer than VisCoT's 438K)
- 2 training epochs; lr = 1e-6; AdamW optimizer
- Input resolution: $336^2$ or $1536^2$

## Key Experimental Results

### Main Results (Visual CoT Benchmark)

| Model | Strategy | Data | DocVQA | TextVQA | InfoQA | GQA | VSR |
|-------|----------|------|--------|---------|--------|-----|-----|
| Qwen2.5VL-7B (original) | — | — | 0.350 | 0.735 | 0.325 | 0.455 | 0.616 |
| Qwen2.5VL-7B* | SFT | 4K | 0.355 | 0.740 | 0.334 | 0.467 | 0.619 |
| **DocThinker-7B** | **RL** | **4K** | **0.579** | **0.802** | **0.347** | **0.546** | **0.656** |
| VisCoT-7B | SFT | 438K | 0.476 | 0.775 | 0.324 | 0.631 | 0.614 |
| DocThinker-7B (high-res) | RL | 4K | 0.795 | 0.827 | 0.689 | 0.694 | 0.721 |

Visual Grounding (TextREC):

| Model | Precision@1 |
|-------|------------|
| TAMN | 80.8% |
| MDETR | 63.3% |
| **DocThinker-7B** | **82.4%** |

### Ablation Study

| Configuration | DocVQA | TextVQA | InfoQA | Note |
|---------------|--------|---------|--------|------|
| DocThinker-7B (Full) | 0.795 | 0.827 | 0.689 | Complete model |
| w/o RoI IoU | 0.775 | 0.803 | 0.637 | Removing visual grounding reward; notable drop on InfoQA |
| w/o Rephrase | 0.763 | 0.772 | 0.658 | Removing rephrase reward; notable drop on TextVQA |
| w/o both | 0.741 | 0.758 | 0.602 | Removing both; largest overall drop |
| w/o KL ($\beta=0$) | 0.780 | 0.803 | 0.676 | Removing KL constraint; performance degrades |

### Key Findings
- **RL substantially outperforms SFT**: With the same 4K data, RL (DocThinker) achieves 0.579 vs. SFT's 0.355 on DocVQA—a 63% improvement—demonstrating that RL enables the model to learn more flexible reasoning strategies rather than memorizing training samples.
- **Exceptional data efficiency**: DocThinker-7B trained with RL on only 4K samples surpasses VisCoT-7B trained with SFT on 438K samples on multiple metrics (e.g., TextVQA: 0.802 vs. 0.775).
- **Complementary roles of RoI IoU and Rephrase rewards**: The RoI IoU reward more strongly benefits tasks requiring precise visual grounding (InfoQA), while the Rephrase reward more strongly benefits tasks requiring question comprehension (TextVQA/DocVQA).
- **KL constraint is critical for training stability**: Performance degrades at $\beta=0$; $\beta=0.04$ is optimal.
- **Strong zero-shot generalization**: DocThinker outperforms baselines on unseen data (DUDE: 0.568, SROIE: 0.814).

## Highlights & Insights
- **First application of GRPO-RL to document understanding**: This work demonstrates the effectiveness of RL for multimodal document understanding, with data efficiency far exceeding SFT (4K vs. 438K). The success of DeepSeek-R1 transfers effectively to specialized vertical domains.
- **Four-dimensional interpretability**: Reasoning process + rephrased question + visual region + final answer provides more comprehensive interpretable information than any prior method. In particular, the bounding box output allows users to intuitively see which part of the document the model attends to.
- **Design philosophy of rule-based rewards**: Rather than relying on human preference annotations, the framework designs automatically verifiable rule-based rewards, substantially reducing the cost of RL training and avoiding the subjectivity of preference labeling.

## Limitations & Future Work
- Validation is limited to Qwen2.5-VL; generalizability to other MLLMs has not been tested.
- All four reward weights are set to 1; finer-grained weight tuning may yield further improvements.
- Training data originates from Visual CoT (which includes bounding box annotations); the RoI IoU reward cannot be applied when such annotations are unavailable.
- **Inference efficiency**: The `<think>` process increases output length and thus inference latency.
- Evaluation is limited to document understanding tasks; whether the approach generalizes to broader MLLM reasoning scenarios warrants further investigation.

## Related Work & Insights
- **vs. DeepSeek-R1**: This work successfully transfers the GRPO framework from DeepSeek-R1 to multimodal document understanding, demonstrating the cross-domain feasibility of RL-based reasoning enhancement.
- **vs. Visual CoT**: Visual CoT relies on SFT with 438K samples, whereas DocThinker trained with RL on only 4K samples surpasses it, highlighting the superior data efficiency of RL over SFT.
- **vs. MedVLM-R1**: MedVLM-R1 applies RL to medical image understanding, while DocThinker applies it to document understanding; both validate the effectiveness of RL in specialized MLLM domains.

## Rating
- **Novelty**: ⭐⭐⭐⭐ First application of GRPO to document understanding; the four-objective rule-based reward design is creative.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Comprehensive evaluation on Visual CoT benchmark + TextREC + ablation studies; RL vs. SFT comparison is clear.
- **Writing Quality**: ⭐⭐⭐⭐ Well-structured with detailed method descriptions.
- **Value**: ⭐⭐⭐⭐ Demonstrates the potential of RL for document understanding; the result of 4K RL training surpassing 438K SFT training is impressive.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] AIGI-Holmes: Towards Explainable and Generalizable AI-Generated Image Detection via Multimodal Large Language Models](aigi-holmes_towards_explainable_and_generalizable_ai-generated_image_detection_v.md)
- [\[NeurIPS 2025\] To Think or Not To Think: A Study of Explicit Thinking in Rule-Based Visual Reinforcement Fine-Tuning](../../NeurIPS2025/multimodal_vlm/think_or_not_think_a_study_of_explicit_thinking_in_rule-based_visual_reinforceme.md)
- [\[NeurIPS 2025\] Unified Reinforcement and Imitation Learning for Vision-Language Models](../../NeurIPS2025/multimodal_vlm/unified_reinforcement_and_imitation_learning_for_vision-language_models.md)
- [\[CVPR 2026\] EMO-R3: Reflective Reinforcement Learning for Emotional Reasoning in Multimodal Large Language Models](../../CVPR2026/multimodal_vlm/emo-r3_reflective_reinforcement_learning_for_emotional_reasoning_in_multimodal_l.md)
- [\[ICCV 2025\] DOGR: Towards Versatile Visual Document Grounding and Referring](dogr_towards_versatile_visual_document_grounding_and_referring.md)

</div>

<!-- RELATED:END -->
