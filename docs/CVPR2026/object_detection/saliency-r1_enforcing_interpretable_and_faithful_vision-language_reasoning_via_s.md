---
title: >-
  [Paper Note] Saliency-R1: Enforcing Interpretable and Faithful Vision-language Reasoning via Saliency-map Alignment Reward
description: >-
  [CVPR 2026][Object Detection][Vision-language models] This paper proposes Saliency-R1, which uses a logit-decomposition-based efficient saliency map technique and chain-of-thought bottleneck attention rollout to compute…
tags:
  - "CVPR 2026"
  - "Object Detection"
  - "Vision-language models"
  - "saliency maps"
  - "GRPO reinforcement learning"
  - "interpretable reasoning"
  - "attention alignment"
date: 2026-05-08
content_hash: 4b225c1ffa436c4c
---

# Saliency-R1: Enforcing Interpretable and Faithful Vision-language Reasoning via Saliency-map Alignment Reward

**Conference**: CVPR 2026
**arXiv**: [2604.04500](https://arxiv.org/abs/2604.04500)  
**Code**: [https://github.com/peterant330/Saliency_R1](https://github.com/peterant330/Saliency_R1)  
**Area**: Object Detection
**Keywords**: Vision-language models, saliency maps, GRPO reinforcement learning, interpretable reasoning, attention alignment

## TL;DR

This paper proposes Saliency-R1, which uses a logit-decomposition-based efficient saliency map technique and chain-of-thought bottleneck attention rollout to compute alignment between saliency maps and human-annotated bounding boxes as a GRPO reward, training VLMs to focus on task-relevant image regions during reasoning and thereby improving the interpretability and faithfulness of the reasoning process.

## Background & Motivation

1. **Background**: VLMs have achieved remarkable progress on reasoning and question-answering tasks. To enhance trustworthiness, models are typically prompted to generate natural language explanations (e.g., Chain-of-Thought) to demonstrate their reasoning process. Reasoning models such as DeepSeek-R1 are also trained to produce detailed chains of thought.

2. **Limitations of Prior Work**: (1) VLMs tend to over-rely on textual cues, with visual signals playing a relatively minor role; (2) the generated reasoning traces are inconsistent with the final answers — models "think" one thing and "do" another; (3) the reasoning process itself may misuse visual cues or hallucinate details that do not exist.

3. **Key Challenge**: Different reasoning processes may attend to different image regions even when they arrive at the same correct answer. Unfaithful reasoning either focuses on irrelevant regions or ignores the image entirely, arriving at the answer through textual shortcuts.

4. **Goal**: (1) Design an efficient saliency map method to visualize how visual information influences generated tokens; (2) track how visual information flows through the chain of thought to the final answer; (3) use saliency alignment as a reward to train models via GRPO to attend to the correct regions.

5. **Key Insight**: Decompose token logits into the first-order direct contributions of each context token, extracting the contributions of visual tokens as saliency maps without any additional forward or backward passes.

6. **Core Idea**: Use zero-overhead logit-decomposition saliency maps to measure the visual focus region of VLM reasoning, and employ alignment with human annotations as a GRPO reward to train more faithful reasoning.

## Method

### Overall Architecture

The method proceeds in three steps: (1) generate per-token saliency maps via logit decomposition at zero additional computational cost; (2) propagate visual information through the chain-of-thought token bottleneck to answer tokens via attention rollout to produce an overall saliency map; (3) compute an alignment score against bounding box annotations and use it as a saliency reward in GRPO training.

### Key Designs

1. **Logit-Decomposition-Based Saliency Map Generation**:

    - **Function**: Efficiently localize the image regions on which each generated token depends.
    - **Mechanism**: The residual connections in Transformers allow the final output to be decomposed into a sum of direct contributions from tokens at each position. When predicting $t_{i+1}$, the direct contribution of token $t_p$ is $c_p = \sum_{l=1}^{L} \sum_{j=1}^{H} \alpha_{i,j,p}^l \mathbf{W}_{o,j}^l \mathbf{W}_{v,j}^l \mathbf{h}_p^{l-1} \mathbf{E}_u$, where $\alpha$ denotes attention weights, $\mathbf{W}_o$ and $\mathbf{W}_v$ are the output and value projection matrices, and $\mathbf{E}_u$ is the unembedding matrix. The contributions corresponding to visual tokens are extracted, rearranged by patch position, and filtered through ReLU to remove negative contributions, yielding the saliency map.
    - **Design Motivation**: Attention weights are naturally accessible in most attention implementations, and $\mathbf{W}_v^l \mathbf{h}_p^{l-1}$ is already computed in the KV cache. Consequently, this approach requires **zero additional forward or backward passes**, with negligible computational overhead, making it suitable for embedding directly into a training pipeline. Although only direct contributions are considered (indirect contributions are ignored), prior work shows that indirect contributions are small, and aligning on direct contributions is sufficient.

2. **Chain-of-Thought Bottleneck Attention Rollout**:

    - **Function**: Track how visual information flows through thinking tokens to answer tokens.
    - **Mechanism**: Define the visual-to-thinking-token attention matrix $\mathcal{A}_{vt}^{l,h}$ and the thinking-to-answer-token attention matrix $\mathcal{A}_{ta}^{l,h}$; multiplying them yields the transitive visual-to-answer attention $\tilde{\mathcal{A}}_{va}^{l,h} = \mathcal{A}_{vt}^{l,h} \mathcal{A}_{ta}^{l,h}$, with thinking tokens serving as the information-passing bottleneck. Column normalization of the attention matrices is deliberately omitted, since certain tokens (e.g., prepositions) inherently receive little contribution from thinking or visual tokens and should therefore exert minimal influence on the overall saliency map.
    - **Design Motivation**: A faithful reasoning process should follow the flow "visual information → thinking process → final answer." If answer tokens draw information directly from visual tokens while bypassing the thinking tokens, the CoT is unfaithful. Bottleneck rollout enables the detection and penalization of such shortcut behavior.

3. **Saliency Alignment Training via GRPO**:

    - **Function**: Encourage the model to attend to correct image regions during reasoning through reinforcement learning.
    - **Mechanism**: The alignment score is defined as $\frac{\sum_{i \in \text{BBox}} \text{Saliency}(i)}{\sum_{i \in \text{Image}} \text{Saliency}(i)}$, i.e., the proportion of total saliency mass that falls within the bounding box. The total reward is $\mathcal{R} = \mathcal{R}_{\text{accuracy}} + \mathcal{R}_{\text{format}} + \mathcal{R}_{\text{saliency}}$, where $\mathcal{R}_{\text{accuracy}}$ evaluates answer correctness via LLM-as-judge (GPT-4o-mini, binary 0/1), $\mathcal{R}_{\text{format}}$ checks for the `<think></think>` format (binary 0/1), and $\mathcal{R}_{\text{saliency}}$ is the alignment score. The GRPO algorithm samples 8 rollouts and uses normalized rewards as the advantage function.
    - **Design Motivation**: Accuracy reward alone cannot distinguish between "attended to the right region and answered correctly" and "attended to the wrong region but guessed correctly." The saliency reward directly encourages the model to focus on image regions relevant to the question, thereby producing more faithful and interpretable reasoning.

### Loss & Training

Training proceeds in two stages: (1) cold-start SFT using the filtered Vision-R1-cold dataset (272,881 samples) with llama-factory; (2) GRPO training using the saliency-r1-8k dataset (8,080 VQA samples with bounding box annotations) via the TRL framework, with batch size 64, KL coefficient 0.001, LoRA rank 16, learning rate $10^{-5}$, and 8× A6000 GPUs. The base models are Qwen2.5-VL (3B and 7B).

## Key Experimental Results

### Main Results (Saliency Map Faithfulness)

| Method | COCO Cap. Del. 5%↓ | Del. 15%↓ | Del. 30%↓ | Ins. 30%↑ |
|------|-------------------|-----------|-----------|-----------|
| CAM | 86.19 | 82.44 | 78.34 | 28.02 |
| ATTN-LRP | 76.42 | 64.22 | 52.92 | 45.67 |
| TAM | 83.91 | 79.33 | 73.29 | 45.24 |
| **Ours** | **70.96** | **59.45** | **50.34** | 45.22 |

On COCO Captions, the deletion metrics are 5.46%/4.77%/2.57% lower than the second-best method, demonstrating that the proposed approach faithfully captures the relative importance of visual patches.

### Ablation Study

| Reward Configuration | Description |
|---------|------|
| $\mathcal{R}_{\text{accuracy}}$ only | Accuracy reward only; baseline |
| $\mathcal{R}_{\text{accuracy}} + \mathcal{R}_{\text{format}}$ | Adding format reward |
| $\mathcal{R}_{\text{accuracy}} + \mathcal{R}_{\text{format}} + \mathcal{R}_{\text{saliency}}$ | Full Saliency-R1; improves reasoning faithfulness and accuracy |

### Key Findings

- **Zero-cost saliency maps outperform gradient-based methods**: By exploiting only attention weights and computations already present in the KV cache, the proposed method surpasses ATTN-LRP (which requires backpropagation) and TAM (which requires solving an optimization problem) on deletion tests.
- **Saliency reward improves both faithfulness and accuracy**: Adding the saliency reward not only leads the model to attend to correct regions (higher alignment scores) but also improves answer accuracy on downstream tasks, indicating that attending to the right regions inherently enhances reasoning quality.
- **CoT bottleneck rollout reveals differences in reasoning faithfulness**: Different reasoning trajectories may focus on entirely different visual regions even when reaching the same answer; unfaithful reasoning can be detected through this mechanism.
- **The method is effective for both 3B and 7B models**: Improvements are observed for both scales of Qwen2.5-VL, demonstrating the generalizability of the approach.

## Highlights & Insights

- **"Zero-overhead" saliency maps are a key innovation**: By cleverly leveraging already-available attention weights and KV cache computations in Transformers, the method achieves gradient-free efficient saliency maps. This allows the saliency signal to be embedded directly into the GRPO training loop without additional computational burden.
- **Training philosophy of "correct attention before correct answers"**: Traditional RL rewards only correct answers, whereas Saliency-R1 additionally requires the model to "look at the right place." This fundamentally addresses the problem of models that guess correct answers without faithful reasoning, and the idea can be generalized to any scenario requiring interpretable reasoning.
- **The chain-of-thought bottleneck concept**: Modeling thinking tokens as an information-passing bottleneck between visual inputs and final answers provides not only a visualization framework but also a quantitative tool for detecting CoT faithfulness.
- **Data efficiency**: Significant improvements are achieved with only 8,080 bounding-box-annotated VQA samples, indicating that the saliency reward constitutes a highly efficient training signal.

## Limitations & Future Work

- Only direct contributions (first-order decomposition) are considered; multi-layer indirect contributions (e.g., nonlinear transformations in FFN layers) are ignored, so the saliency maps are not fully precise.
- Training data with bounding box annotations are required, and annotation costs limit scalability.
- The current attention rollout is heuristic (matrix multiplication approximates information flow) and lacks theoretical guarantees.
- Validation is limited to Qwen2.5-VL; applicability to other architectures (e.g., LLaVA, InternVL) remains unknown.
- Complementary validation with methods such as GradCAM, or the use of finer-grained region annotations (segmentation masks vs. bounding boxes), could be explored.
- No detailed hyperparameter search is conducted for the weight of the saliency reward $\mathcal{R}_{\text{saliency}}$.

## Related Work & Insights

- **vs. DeepSeek-R1 / Claude**: These models are trained for CoT reasoning but do not monitor visual focus regions, potentially producing unfaithful reasoning that "looks at the wrong place but guesses correctly." Saliency-R1 directly addresses this issue through saliency alignment rewards.
- **vs. Grad-CAM / ATTN-LRP**: Conventional saliency methods require additional computation (backpropagation/perturbation). The logit-decomposition method in Saliency-R1 achieves comparable or superior faithfulness at zero computational cost.
- **vs. ADAPTVIS**: ADAPTVIS adaptively adjusts attention at inference time, whereas Saliency-R1 shapes attention patterns during training through reward design; the two approaches are complementary.
- This work is the first to align visual attention with human annotations during the post-training stage, opening a new direction for interpretable RL training of VLMs.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ Zero-overhead saliency maps + CoT bottleneck rollout + saliency-aligned GRPO reward — each of the three contributions is individually novel and they are organically integrated.
- Experimental Thoroughness: ⭐⭐⭐⭐ Faithfulness evaluation (deletion/insertion) is thorough; however, the complete downstream task performance table is truncated in the cache.
- Writing Quality: ⭐⭐⭐⭐⭐ The motivation figure (different reasoning trajectories attending to different regions) is highly intuitive, the method pipeline diagram is clear, and the mathematical derivations are rigorous.
- Value: ⭐⭐⭐⭐⭐ The work is highly significant for trustworthy VLM reasoning; the method is practical and reproducible, and it opens a new direction for saliency-guided RL training.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mining Instance-Centric Vision-Language Contexts for Human-Object Interaction Detection](mining_instance-centric_vision-language_contexts_for_human-object_interaction_de.md)
- [\[CVPR 2026\] VisualAD: Language-Free Zero-Shot Anomaly Detection via Vision Transformer](visualad_language-free_zero-shot_anomaly_detection_via_vision_transformer.md)
- [\[CVPR 2026\] Fourier Angle Alignment for Oriented Object Detection in Remote Sensing](fourier_angle_alignment_for_oriented_object_detection_in_remote_sensing.md)
- [\[ICCV 2025\] Visual Modality Prompt for Adapting Vision-Language Object Detectors](../../ICCV2025/object_detection/visual_modality_prompt_for_adapting_vision-language_object_detectors.md)
- [\[CVPR 2026\] HeROD: Heuristic-inspired Reasoning Priors Facilitate Data-Efficient Referring Object Detection](herod_heuristic_inspired_reasoning_data_efficient_rod.md)

</div>

<!-- RELATED:END -->
