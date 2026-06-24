---
title: >-
  [Paper Note] Multimodal LLMs as Customized Reward Models for Text-to-Image Generation
description: >-
  [ICCV2025][Multimodal VLM][reward model] This paper proposes LLaVA-Reward, which leverages the hidden states (rather than text generation outputs) of a pretrained MLLM to directly predict reward scores. A Skip-connection Cross Attention (SkipCA) module is introduced to enhance bidirectional visual-text interaction, and LoRA adapters are employed to handle different evaluation dimensions. The method achieves state-of-the-art performance on text-image alignment, fidelity…
tags:
  - "ICCV2025"
  - "Multimodal VLM"
  - "reward model"
  - "text-to-image generation"
  - "multimodal evaluation"
  - "preference learning"
  - "inference-time scaling"
date: 2026-05-08
content_hash: dbc0f808dea9f898
---

# Multimodal LLMs as Customized Reward Models for Text-to-Image Generation

**Conference**: ICCV2025  
**arXiv**: [2507.21391](https://arxiv.org/abs/2507.21391)  
**Code**: [GitHub](https://github.com/sjz5202/LLaVAReward)  
**Area**: Multimodal VLM  
**Keywords**: reward model, text-to-image generation, multimodal evaluation, preference learning, inference-time scaling  

## TL;DR

This paper proposes LLaVA-Reward, which leverages the hidden states (rather than text generation outputs) of a pretrained MLLM to directly predict reward scores. A Skip-connection Cross Attention (SkipCA) module is introduced to enhance bidirectional visual-text interaction, and LoRA adapters are employed to handle different evaluation dimensions. The method achieves state-of-the-art performance on text-image alignment, fidelity, and safety evaluation, and can be applied to inference-time scaling for diffusion models.

## Background & Motivation

The rapid development of text-to-image generation models (e.g., Stable Diffusion) has created a strong demand for high-quality automatic evaluation and reward models. Existing approaches suffer from the following limitations:

**CLIP-based methods** (CLIPScore, PickScore, HPSv2, ImageReward): CLIP operates as a bag-of-words model, with limited capacity to model complex text-image relationships and poor generalization.

**VQA-based MLLM methods** (VIEScore, EvalAlign, LlavaGuard): These require verbose system prompts and instruction fine-tuning, resulting in low inference efficiency, with scoring precision constrained by discretization.

**Token probability methods** (Q-ALIGN, VQAScore, LLaVAScore): These rely on the probability of specific "golden tokens" (e.g., "Yes"/"No"), making it difficult to handle samples with subtle quality differences in preference data.

The core problem is: **how to build an efficient, flexible, and multi-dimensional T2I evaluation reward model?**

The authors observe that the **hidden states** of an MLLM can be directly used to predict rewards, without generating textual responses or relying on complex evaluation instructions, thereby simultaneously improving both efficiency and expressiveness.

## Method

### Overall Architecture

LLaVA-Reward is built on Phi-3.5-vision (4.2B) and takes text-image pairs as input, using the hidden states of the final MLLM layer to predict rewards. The architecture consists of three key components:

1. **Pretrained MLLM backbone**: Frozen Phi-3.5-vision, providing visual and linguistic representations.
2. **LoRA adapters**: Separate LoRA adapters for different evaluation dimensions (alignment, fidelity, safety).
3. **SkipCA reward head**: A bidirectional cross-attention module replacing the conventional linear layer.

### Key Designs: Skip-connection Cross Attention (SkipCA)

In decoder-only MLLMs, the causal attention mechanism prevents visual tokens from being influenced by subsequently injected text tokens, which impairs the model's ability to reason about image-text correspondence. SkipCA addresses this by establishing a skip connection between **early visual features** and **deep hidden representations**:

$$r_\theta(\bm{i}, \bm{t}) = f_r(\mathbf{e}_h, \mathbf{e}_v) = g(f_{\text{SCA}}(\mathbf{e}_h, \mathbf{e}_v))$$

Specifically, SkipCA is a standard cross-attention operation where:
- **Query**: the hidden state of the EOS token at the final layer $\mathbf{e}_h$ (encoding textual semantics);
- **Key/Value**: visual tokens output by the visual projector $\mathbf{e}_v$ (highly vision-specific);
- **Output**: a scalar reward (BT model) or vector reward (GPM model) obtained via linear projection.

**Design Motivation**: The influence of visual tokens diminishes in deeper MLLM layers; therefore, directly using the projected visual tokens (effectively "skipping" to the shallow layer) is more effective than performing cross-attention at the deep layer.

### Training Objectives

**Pairwise preference data**: Bradley-Terry ranking loss

$$\mathcal{L}_{\text{rank}} = -\mathbb{E}_{(\bm{i}_c, \bm{i}_r, \bm{t}) \sim \mathcal{D}_p}\left[-\log\sigma\left(\frac{s_{\theta_p}(\bm{i}_c, \bm{t}) - s_{\theta_p}(\bm{i}_r, \bm{t})}{T}\right)\right]$$

**Non-paired binary classification data** (e.g., safety labels): Cross-entropy loss

$$\mathcal{L}_{\text{CE}} = -\mathbb{E}\left[\log\sigma(s(\bm{i}_c, \bm{t})) + \log(1-\sigma(s(\bm{i}_r, \bm{t})))\right]$$

**General Preference Model (GPM)**: For complex preference relationships, a multi-dimensional reward vector and an antisymmetric preference operator are employed:

$$s(\bm{i}_c, \bm{t}) - s(\bm{i}_r, \bm{t}) = \langle \mathbf{R}^\succ r_\theta(\bm{i}_c, \bm{t}), r_\theta(\bm{i}_r, \bm{t})\rangle$$

where $\mathbf{R}^\succ$ is a skew-symmetric matrix. This enables the model to capture finer-grained preference relationships in latent space beyond scalar rewards.

### Multi-Dimensional Evaluation via LoRA Adaptation

Separate LoRA adapters are trained for each evaluation dimension (alignment, fidelity, safety, overall ranking). All dimensions share the MLLM backbone parameters; switching between evaluation dimensions is achieved by swapping LoRA adapters, balancing efficiency and flexibility.

### Training Details

- The visual encoder and internal LLM parameters are frozen.
- Only the visual projector, SkipCA reward head, and LoRA adapters are trained (~8% additional parameters).
- Training data: 158K pairs from the ImageReward alignment set, 84K pairs from the fidelity set, and 8.1K binary samples from UnsafeBench.

## Key Experimental Results

### MJ-Bench Multi-Dimensional Evaluation

| Method | Parameters | Alignment (Acc w/tie) | Safety (Acc w/tie) | Fidelity (Acc w/tie) |
|--------|-----------|----------------------|--------------------|----------------------|
| CLIPScore | 428M | 38.1 | 12.7 | 34.4 |
| ImageReward | 478M | 50.9 | 24.9 | 63.5 |
| HPS-v2.1 | 2B | 47.3 | 18.8 | 67.3 |
| VQAScore | 11B | 63.2 | - | - |
| GPT-4o | - | 61.5 | 35.3 | 97.6 |
| LlavaGuard | 7B | - | 5.6 | - |
| **LLaVA-Reward-Phi** | **4.2B** | **66.1** | **55.2** | **91.1** |
| **LLaVA-Reward-Qwen** | **8.2B** | **67.5** | **59.2** | **94.3** |

LLaVA-Reward-Qwen outperforms all open-source methods across all three dimensions, substantially surpassing all baselines on safety (59.2% vs. second-best 37.2%), while approaching GPT-4o-level performance with only 4.2B–8.2B parameters.

### Ablation Study

#### SkipCA Ablation

| Configuration | Alignment Acc | Safety Acc | Fidelity Acc |
|---------------|--------------|------------|--------------|
| w/o SkipCA (MLP) | 68.2 | 39.7 | 87.3 |
| **w/ SkipCA** | 66.1 | **55.2** | **91.1** |

SkipCA improves safety evaluation by 15.5 percentage points and fidelity by 3.8 percentage points, validating the importance of enhanced bidirectional visual-text interaction for T2I evaluation.

### Inference Efficiency Comparison

| Method | Type | Inference Time (s) |
|--------|------|--------------------|
| EvalAlign | VQA | 7.01 |
| LlavaGuard | VQA | 4.30 |
| VQAScore | Token | 2.81 |
| LLaVA-score | Token | 0.26 |
| **LLaVA-Reward** | **Hidden State** | **0.35** |

LLaVA-Reward ranks second in inference speed (0.35s), far faster than VQA-based methods and close to the most efficient token-based approach.

### Inference-Time Scaling for Diffusion Models

Applying LLaVA-Reward to SD v2.1 and SDXL via the FK steering method for inference-time scaling:

| Reward Model | GenEval Overall (SDXL) |
|-------------|------------------------|
| None | 0.563 |
| CLIPScore | 0.592 |
| ImageReward | 0.627 |
| **LLaVA-Reward** | **0.645** |

LLaVA-Reward achieves the best generation quality improvement in diffusion inference-time scaling.

## Highlights & Insights

1. **Hidden states replacing text generation**: By bypassing the MLLM text decoding process and directly leveraging hidden representations for reward prediction, the method simultaneously achieves high accuracy and high efficiency.
2. **Elegant design of SkipCA**: Skip connections across layers address the "forgetting" of visual tokens in decoder-only MLLMs.
3. **GPM preference embedding**: Multi-dimensional vector rewards and antisymmetric preference operators enable modeling of preference relationships more complex than those captured by scalar rewards.
4. **Unified multi-dimensional evaluation**: A single model covers alignment, fidelity, and safety dimensions via LoRA adapter switching.

## Limitations & Future Work

- On the MJ-Bench alignment dimension, the model with SkipCA slightly underperforms the variant without SkipCA (66.1 vs. 68.2), possibly because SkipCA over-emphasizes visual features for alignment tasks.
- The reliance on ImageReward as the training data source introduces its annotation biases.
- The effectiveness of hard negative sample construction is highly dataset-dependent.
- Validation is limited to Phi-3.5-vision and Qwen2.5-VL; performance on larger-scale models remains unknown.

## Related Work & Insights

- **CLIP-based evaluation**: CLIPScore, PickScore, HPSv2, ImageReward — constrained by CLIP's bag-of-words nature.
- **VQA-based MLLM evaluation**: VIEScore, EvalAlign, LlavaGuard, ImageGuard — rely on complex instructions and text generation, resulting in low efficiency.
- **Token probability evaluation**: Q-ALIGN, VQAScore, LLaVAScore — dependent on specific tokens, with precision limited by discretization.
- **Safety evaluation**: LlavaGuard, ImageGuard, PerspectiveVision — focused exclusively on the safety dimension.

## Rating

| Dimension | Score (1–5) |
|-----------|------------|
| Novelty | 4 |
| Technical Quality | 4 |
| Experimental Thoroughness | 5 |
| Writing Quality | 4 |
| Value | 5 |
| Overall | 4.4 |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Multimodal RewardBench 2: Evaluating Omni Reward Models for Interleaved Text and Image](../../CVPR2026/multimodal_vlm/multimodal_rewardbench_2_evaluating_omni_reward_models_for_interleaved_text_and_.md)
- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](controlling_multimodal_llms_via_rewardguided_decoding.md)
- [\[ACL 2025\] Enhance Multimodal Consistency and Coherence for Text-Image Plan Generation](../../ACL2025/multimodal_vlm/enhance_multimodal_consistency_and_coherence_for_text-image_plan_generation.md)
- [\[CVPR 2025\] CoMM: A Coherent Interleaved Image-Text Dataset for Multimodal Understanding and Generation](../../CVPR2025/multimodal_vlm/comm_a_coherent_interleaved_image-text_dataset_for_multimodal_understanding_and_.md)
- [\[ACL 2025\] Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation](../../ACL2025/multimodal_vlm/code_guided_text_rich_image.md)

</div>

<!-- RELATED:END -->
