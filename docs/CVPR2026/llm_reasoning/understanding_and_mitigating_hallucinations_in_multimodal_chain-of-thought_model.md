---
title: >-
  [Paper Note] Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models
description: >-
  [CVPR 2026][LLM Reasoning][Multimodal Hallucination] This paper systematically analyzes the causes of hallucinations in multimodal CoT models, identifies "divergent thinking" (associative reasoning) as the core trigger, and proposes a training-free detection and decoding intervention strategy based on visual entropy. The method reduces CHAIRS by over 30% on Object HalBench while maintaining or improving general reasoning capability.
tags:
  - CVPR 2026
  - LLM Reasoning
  - Multimodal Hallucination
  - Chain-of-Thought Reasoning
  - Divergent Thinking
  - Visual Entropy
  - Decoding Intervention
date: 2026-05-08
content_hash: e886895d16b3bd6a
---

# Understanding and Mitigating Hallucinations in Multimodal Chain-of-Thought Models

**Conference**: CVPR 2026
**arXiv**: [2603.27201](https://arxiv.org/abs/2603.27201)
**Code**: [https://github.com/ASGO-MM/MCoT-hallucination](https://github.com/ASGO-MM/MCoT-hallucination)
**Area**: LLM Reasoning
**Keywords**: Multimodal Hallucination, Chain-of-Thought Reasoning, Divergent Thinking, Visual Entropy, Decoding Intervention

## TL;DR

This paper systematically analyzes the causes of hallucinations in multimodal CoT models, identifies "divergent thinking" (associative reasoning) as the core trigger, and proposes a training-free detection and decoding intervention strategy based on visual entropy. The method reduces CHAIRS by over 30% on Object HalBench while maintaining or improving general reasoning capability.

## Background & Motivation

1. **State of the Field**: Multimodal chain-of-thought (MCoT) models (e.g., R1-Onevision, PixelReasoner, GRIT) have substantially improved complex visual reasoning through explicit reasoning chains and have become the dominant paradigm for multimodal reasoning.
2. **Limitations of Prior Work**: MCoT models produce severe hallucinations during reasoning chain generation—generating textual descriptions that contradict visual content. Prior work (Liu et al., Tian et al.) attributes this to visual attention decay caused by longer reasoning chains, yet visual attention decay is a longstanding issue in traditional LVLMs and is not unique to MCoT.
3. **Root Cause**: Traditional LVLMs employ implicit reasoning (direct answering), whereas MCoT models employ explicit reasoning (think before answering), representing a fundamentally different reasoning process. This raises the question of whether MCoT models have their own unique causes of hallucination.
4. **Paper Goals**: (1) Identify hallucination causes specific to MCoT models; (2) Design training-free methods to locate and mitigate these hallucinations.
5. **Starting Point**: Drawing on the cognitive science concept of "divergent vs. convergent thinking," the paper segments and annotates reasoning chains and finds that hallucinations are concentrated in divergent thinking steps (approximately 5× more frequent than in normal thinking steps).
6. **Core Idea**: Visual entropy is used to quantify the model's internal confidence in visual input; high-entropy steps correspond to divergent thinking, and high-entropy tokens are dynamically penalized during decoding.

## Method

### Overall Architecture

Input image and user query → MCoT model generates a reasoning chain → visual entropy is computed at each step → if visual entropy exceeds threshold $\gamma$ (identified as a divergent thinking step) → decoding intervention reduces the generation probability of high-entropy tokens → corrected answer is output. The entire process requires no additional training and operates directly at inference time.

### Key Designs

1. **Visual Entropy**

   - **Function**: Quantifies the model's internal certainty about visual input at each generation step.
   - **Mechanism**: Visual token hidden states are mapped to a vocabulary probability distribution via the language head; for the predicted token $y_t$, the visual activation probability $\mathbf{p}_v(y_t) \in \mathbb{R}^m$ is extracted, and normalized entropy is computed as $E(y_t, v) = -\frac{\sum_{i=1}^m p_{v,i}(y_t) \log(p_{v,i}(y_t))}{\log m}$, normalized to the range $[0,1]$.
   - **Design Motivation**: In divergent thinking steps, the model relies more on internal reasoning than on visual evidence, resulting in higher uncertainty about visual input. Experimental validation shows that logistic regression classification achieves a McFadden pseudo-$R^2$ exceeding 0.9, confirming that visual entropy reliably distinguishes divergent from normal thinking.

2. **Divergent Thinking Detection**

   - **Function**: Identifies in real time when the model enters a divergent thinking mode during reasoning.
   - **Mechanism**: Visual entropy is computed for each reasoning step; when $E(y_t, v) > \gamma$ (default $\gamma = 0.5$), the step is classified as divergent thinking.
   - **Design Motivation**: Traditional methods require external annotation or additional inference overhead. The proposed approach leverages the model's existing visual token representations without any additional forward pass.

3. **Decoding Intervention**

   - **Function**: Dynamically adjusts decoding probabilities upon detection of divergent thinking to suppress hallucinations.
   - **Mechanism**: The decoding probability for divergent thinking steps is corrected as $\hat{p}_t(\cdot|v,q,y_{<t}) = p_t(\cdot|v,q,y_{<t}) \cdot e^{-\alpha \cdot E(\cdot, v)}$, where $\alpha = 0.75$ controls intervention strength. Tokens with high visual entropy are penalized exponentially.
   - **Design Motivation**: This avoids the doubled inference overhead of contrastive decoding and requires no complex hidden-state editing. Computational cost is negligible, as visual token probabilities are precomputed during the prefill stage.

### Loss & Training

The proposed method is training-free and requires no additional training phase. The two hyperparameters ($\gamma = 0.5$, $\alpha = 0.75$) are experimentally shown to generalize across models without requiring per-model tuning.

## Key Experimental Results

### Main Results

| Model | Method | CHAIRS↓ | CHAIRI↓ | POPE Random Acc↑ | POPE Adv Acc↑ | POPE Pop Acc↑ |
|-------|--------|---------|---------|------------------|---------------|---------------|
| GRIT-3B | Baseline | 23.8 | 10.5 | 78.1 | 77.5 | 78.6 |
| GRIT-3B | +FlexAC | 19.2 | 7.4 | 79.8 | 79.0 | 79.6 |
| GRIT-3B | **+Ours** | **16.0** | **5.5** | **81.0** | **79.8** | **80.6** |
| PixelReasoner-7B | Baseline | 22.0 | 7.8 | 85.5 | 82.3 | 84.3 |
| PixelReasoner-7B | **+Ours** | **15.4** | **5.3** | **87.3** | **84.3** | **86.5** |
| R1-Onevision-7B | Baseline | 23.2 | 9.4 | 81.2 | 78.5 | 80.4 |
| R1-Onevision-7B | **+Ours** | **15.8** | **5.7** | **83.5** | **81.6** | **82.2** |

### Ablation Study

| Configuration | CHAIRS↓ | POPE Adv Acc↑ | Notes |
|---------------|---------|---------------|-------|
| Full ($\gamma$=0.5, $\alpha$=0.75) | 15.8 | 81.6 | Complete model |
| w/o detection (global intervention) | ~18.0 | ~80.0 | No distinction between divergent/normal steps |
| w/o intervention (detection only) | 23.2 | 78.5 | Detection without correction |
| + DoLa | ~14.5 | ~82.0 | Compatible with existing methods |
| + VCD | ~14.8 | ~82.5 | Compatible with contrastive decoding |

### Key Findings

- **Divergent thinking is the core problem**: The hallucination rate in divergent thinking steps is approximately 5× that of normal thinking steps, and hallucination ratios in the thinking and answering phases are highly correlated ($\rho > 0.96$, $R^2 > 0.92$).
- **Severe attention bias**: During answer generation, the model allocates less than 0.04 of its attention to image tokens, with the remainder biased toward the reasoning chain.
- **Plug-and-play compatibility**: The method integrates seamlessly with existing approaches such as DoLa, VCD, MemVR, and FlexAC to further improve performance.
- **Cross-model hyperparameter generalizability**: $\gamma = 0.5$ and $\alpha = 0.75$ are effective across GRIT-3B, PixelReasoner-7B, and R1-Onevision-7B without per-model tuning.

## Highlights & Insights

- **Cognitive science perspective**: Borrowing the "divergent vs. convergent thinking" conceptual framework to analyze MCoT model behavior provides a clear analytical lens and a theoretical anchor for future work.
- **Visual entropy as a three-in-one tool**: A single metric serves simultaneously for divergent thinking detection, guidance of decoding intervention, and composability with other methods—an elegantly minimal design.
- **Training-free and low-overhead**: Visual token probabilities are precomputed during the prefill stage, incurring near-zero additional latency at inference time, which is far more efficient than contrastive decoding methods that require doubled forward passes.
- **Transferable idea**: The visual entropy approach is transferable to other modalities (e.g., audio-language models); analogous "divergent detection and intervention" can be applied wherever modality-conditional uncertainty can be computed.

## Limitations & Future Work

- Divergent thinking annotation relies on GPT-5 and human verification, leaving annotation criteria partially subjective; re-annotation is required when extending to new models.
- Although $\gamma$ and $\alpha$ are shown experimentally to generalize across models, they may require re-tuning in extreme scenarios such as medical image reasoning.
- Validation is limited to three MCoT models; effectiveness on larger-scale models (e.g., 70B+) remains unknown.
- The visual entropy metric depends on the number of visual tokens $m$; different resolutions or tokenization strategies may affect metric stability.
- Future work could explore using visual entropy as a training signal (rather than solely an inference signal), directly optimizing models to reduce divergent thinking tendencies during SFT/RLHF.

## Related Work & Insights

- **vs. DoLa/VCD (contrastive decoding)**: These methods suppress hallucinations by contrasting logits across different layers or modalities, but require additional forward passes. The proposed method leverages precomputed visual probabilities with no extra overhead and can be stacked on top of these approaches.
- **vs. MemVR (memory augmentation)**: MemVR addresses attention decay through enhanced visual memory, which operates at a different level from the "divergent thinking" addressed in this paper; the two are complementary.
- **vs. FlexAC (attention control)**: FlexAC directly manipulates attention weights to increase visual focus, whereas this paper intervenes at the probability distribution level, achieving a lighter-weight and more effective solution.

## Rating

- **Novelty**: ⭐⭐⭐⭐ — The theoretical framework of divergent thinking combined with visual entropy is novel, though the decoding intervention mechanism itself is relatively straightforward.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ — Three models, multiple benchmarks, and comparison and stacking experiments with five existing methods constitute a comprehensive evaluation.
- **Writing Quality**: ⭐⭐⭐⭐ — The logical chain from problem identification to cause analysis to method proposal is clear and coherent.
- **Value**: ⭐⭐⭐⭐ — The paper offers a new theoretical perspective and a practical tool for MCoT hallucination research; the plug-and-play property facilitates broad adoption.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Harnessing Chain-of-Thought Reasoning in Multimodal Large Language Models for Face Anti-Spoofing](harnessing_chain-of-thought_reasoning_in_multimodal_large_language_models_for_fa.md)
- [\[CVPR 2026\] Understanding the Role of Hallucination in Reinforcement Post-Training of Multimodal Reasoning Models](understanding_the_role_of_hallucination_in_reinforcement_post-training_of_multim.md)
- [\[CVPR 2026\] Reinforcing Structured Chain-of-Thought for Video Understanding](reinforcing_structured_chain-of-thought_for_video_understanding.md)
- [\[NeurIPS 2025\] Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](../../NeurIPS2025/llm_reasoning/visual_thoughts_a_unified_perspective_of_understanding_multi.md)
- [\[CVPR 2026\] Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)

<!-- RELATED:END -->
