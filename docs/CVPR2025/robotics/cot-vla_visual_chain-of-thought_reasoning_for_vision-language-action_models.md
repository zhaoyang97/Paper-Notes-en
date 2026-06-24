---
title: >-
  [Paper Note] CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models
description: >-
  [CVPR 2025][Robotics][VLA] This paper proposes CoT-VLA, which introduces visual Chain-of-Thought (CoT) reasoning into Vision-Language-Action (VLA) models. By utilizing a two-stage reasoning process—first predicting a subgoal image, then generating an action sequence—combined with hybrid attention and action chunking strategies, it achieves an 81.13% average success rate on the LIBERO benchmark, significantly outperforming existing methods.
tags:
  - "CVPR 2025"
  - "Robotics"
  - "VLA"
  - "Chain-of-Thought"
  - "Visual Reasoning"
  - "Robotic Manipulation"
  - "Subgoal Prediction"
date: 2026-05-08
content_hash: 01e52e45d4e39705
---

# CoT-VLA: Visual Chain-of-Thought Reasoning for Vision-Language-Action Models

**Conference**: CVPR 2025  
**arXiv**: [2503.22020](https://arxiv.org/abs/2503.22020)  
**Code**: [https://cot-vla.github.io/](https://cot-vla.github.io/)  
**Authors**: Qingqing Zhao, Yao Lu, Moo Jin Kim, Zipeng Fu, Zhuoyang Zhang, Yecheng Wu, Zhaoshuo Li, Qianli Ma, Song Han, Chelsea Finn, Ankur Handa, Ming-Yu Liu, Donglai Xiang, Gordon Wetzstein, Tsung-Yi Lin  
**Affiliations**: NVIDIA, Stanford University, MIT  
**Area**: LLM Reasoning  
**Keywords**: VLA, Chain-of-Thought, Visual Reasoning, Robotic Manipulation, Subgoal Prediction

## TL;DR
This paper proposes CoT-VLA, which introduces visual Chain-of-Thought (CoT) reasoning into Vision-Language-Action (VLA) models. By utilizing a two-stage reasoning process—first predicting a subgoal image, then generating an action sequence—combined with hybrid attention and action chunking strategies, it achieves an 81.13% average success rate on the LIBERO benchmark, significantly outperforming existing methods.

## Background & Motivation
**Background**: Vision-Language-Action models (VLAs) aim to leverage large-scale pre-trained vision-language models (VLMs) to directly generate robot actions from visual observations and language instructions. Existing VLAs like OpenVLA and RT-2 have demonstrated the potential to benefit from internet-scale pre-training, but their performance remains limited on long-horizon and complex reasoning tasks.

**Limitations of Prior Work**: Current VLA models employ a direct "observation $\rightarrow$ action" mapping, lacking intermediate reasoning steps. This end-to-end mapping is prone to failure when facing tasks that require multi-step planning, as the model cannot decompose complex tasks into manageable sub-steps. Furthermore, the training data for existing methods is limited to robot datasets with action annotations, failing to utilize the vast amount of unlabeled human activity videos.

**Key Challenge**: Humans naturally perform visual planning when executing complex manipulation tasks—imagining the target state of the next step before executing specific actions. However, existing VLAs lack this "plan-then-execute" capability. How to incorporate visual-level chain-of-thought reasoning into VLAs while maintaining the end-to-end nature of the model remains an unsolved problem.

**Key Insight**: Drawing on the success of CoT in LLMs, this work extends it to the vision-action domain. The key observation is that subgoal images can serve as "intermediate reasoning steps" in the visual domain, which not only provide clear goal guidance for action generation but can also be trained using massive unlabeled videos.

**Core Idea**: CoT-VLA decomposes robot action generation into two-stage reasoning: (1) predicting future subgoal images given current observations and language instructions; (2) generating action sequences conditioned on the subgoal images. This is realized via a unified end-to-end framework, VILA-U, where image generation and action prediction share weights.

## Method

### Overall Architecture
CoT-VLA is built upon the unified multimodal foundation model VILA-U and adopts a two-stage reasoning paradigm. Phase 1 receives the current image observation and language instruction, autoregressively generating subgoal image tokens. Phase 2 generates the action sequence conditioned on the current observation, language instruction, and the predicted subgoal image. The entire process is executed within a single model with shared weights.

### Key Designs

1. **Visual Chain-of-Thought Reasoning**:

    - **Function**: Decomposes the action generation process into two phases: "subgoal prediction + action generation".
    - **Mechanism**: In Phase 1, the current observation image ($256\times256$) is encoded into $16\times16\times4 = 1024$ discrete tokens through residual quantization, and the model autoregressively predicts the subgoal image tokens $0.5 \sim 1$ second into the future. In Phase 2, the predicted subgoal image tokens are concatenated with the current observation to generate a set of action chunks.
    - **Design Motivation**: Subgoal images provide a more precise target description than language instructions, particularly in tasks with complex spatial relationships. Ground-truth subgoal images can improve the success rate on OOD tasks by approximately 40%, verifying the crucial guidance of visual goals on action generation.

2. **Hybrid Attention**:

    - **Function**: Employs different attention patterns for different modalities.
    - **Mechanism**: Image tokens and text tokens use causal attention to maintain consistency in autoregressive generation, while action tokens use full attention to allow bidirectional information interaction within the action sequence.
    - **Design Motivation**: Action prediction is essentially a sequence-to-sequence regression task, differing from the autoregressive generation of text/images. Full attention allows each action token to reference other action tokens in the sequence, enhancing the temporal consistency of action sequences.

3. **Action Chunking**:

    - **Function**: The model predicts action sequences for 10 timesteps at once.
    - **Mechanism**: Each action is represented by 7 tokens (corresponding to the 7 robot DOFs), and each token is quantized through 256 discrete bins. A chunk consists of 70 tokens in total.
    - **Design Motivation**: Chunked prediction reduces the number of autoregressive loops, improving inference efficiency while maintaining temporal coherence within the chunk.

4. **Unlabeled Video Pre-training**:

    - **Function**: Benchmarks pre-training using human activity videos such as EPIC-KITCHENS and Something-Something V2.
    - **Mechanism**: Sampling frame pairs from videos as "current observation-subgoal image" pairs to train the model to predict future frames, without requiring action annotations.
    - **Design Motivation**: To acquire a large amount of training data without tedious annotations, learning the physical laws of the visual world and object interaction patterns.

### Loss & Training
- **Pre-training Phase**: Training subgoal prediction capabilities on unlabeled videos, using 12 A100 GPU nodes, totaling approximately 11K A100 GPU hours.
- **Fine-tuning Phase**: Fine-tuning the entire model on robot data with action annotations.
- **Action Discretization**: Each action dimension is uniformly quantized into 256 bins.
- **Image Encoding**: The $256\times256$ image is encoded into $16\times16\text{ spatial} \times 4\text{ codebook depth} = 1024$ tokens via residual quantization.

## Key Experimental Results

### Main Results: LIBERO Benchmark

| Method | Average | Spatial | Object | Goal | Long |
|------|------|---------|--------|------|------|
| Diffusion Policy | 72.4% | 78.3% | 92.5% | 68.3% | 50.5% |
| OpenVLA | 76.5% | 84.7% | 88.4% | 79.2% | 53.7% |
| $\pi_0$ (flow matching) | 79.2% | 86.0% | 90.3% | 82.5% | 58.0% |
| **CoT-VLA-7B** | **81.13%** | **87.5%** | **91.6%** | **87.6%** | **69.0%** |

CoT-VLA achieves optimal or near-optimal results across all four task suites, with the most significant improvement in Long-horizon tasks (+15.3% vs OpenVLA), validating the advantages of visual chain-of-thought in complex long-sequence tasks.

### Ablation Study

| Configuration | LIBERO Average | Relative Change |
|------|------------|---------|
| CoT-VLA (Full) | 81.13% | — |
| w/o Action Chunking (chunk=1) | 78.2% | -2.9% |
| w/o Hybrid Attention (All Causal) | 79.8% | -1.3% |
| w/o CoT (Direct Action Prediction) | 75.5% | -5.6% |
| w/o Video Pre-training | 53.7% | -27.4% |

### Pre-training Effects
- Baseline without pre-training: 53.7% average success rate
- With video pre-training: 78.8% average success rate
- **Relative improvement of 46.7%**, demonstrating the massive contribution of unlabeled video pre-training to VLAs.

### Subgoal Image Quality Analysis
- Replacing predicted subgoals with ground-truth subgoal images improves OOD task success rates by approximately 40%.
- This indicates that there is still room for improvement in current subgoal prediction quality, and better visual generation capability will directly translate into action performance gains.

### Key Findings
- **CoT reasoning is the most critical component**: Removing CoT (direct action prediction) leads to a 5.6% performance drop, which is much larger than other ablations.
- **Long-horizon tasks benefit the most**: CoT-VLA improves by 15.3 percentage points over OpenVLA on LIBERO-Long.
- **Pre-training shows striking outcomes**: Video pre-training brings a 46.7% relative improvement, revealing the massive potential of unlabeled video data.
- **Inference speed trade-off**: Generating 256 subgoal image tokens makes inference approximately 7 times slower.

## Highlights & Insights
- **Visual CoT is a natural extension**: Extending CoT from textual reasoning to visual planning is a very natural transition—robots "imagining" target states before action execution aligns with human spatial planning abilities.
- **Subgoals as interpretable bottlenecks**: Predicted subgoal images provide a visual explanation of the model's decision-making process, facilitating debugging and human-robot interaction.
- **Leveraging unlabeled video**: Through the visual CoT framework, vast amounts of unlabeled video can be directly used for training, breaking through the bottleneck of robotic data scarcity.

## Limitations & Future Work
- **Inference latency issue**: Generating 256 subgoal image tokens causes an approximate 7x slowdown in inference, which is unfavorable for real-time control tasks.
- **Subgoal prediction accuracy**: There is still a gap between predicted subgoals and true targets (GT targets can bring an additional 40% OOD improvement).
- **Single subgoal limitation**: Currently, only a single-step subgoal is predicted, which may be insufficient for very long-horizon tasks.
- **High computational cost**: The pre-training cost of 11K A100 GPU hours limits the accessibility of this method.

## Related Work & Insights
- **vs OpenVLA**: OpenVLA directly maps observations to actions, lacking planning capabilities. CoT-VLA introduces online planning through subgoal prediction.
- **vs RT-2**: RT-2 uses a VLM backbone but similarly lacks intermediate reasoning. CoT-VLA demonstrates the value of explicitly introducing reasoning steps in VLAs.
- **vs SuSIE**: SuSIE also uses subgoal image-guided policies but employs a separate diffusion model to generate subgoals. CoT-VLA achieves end-to-end training within a unified framework.

## Rating
- Novelty: ⭐⭐⭐⭐ Introduces visual CoT to VLAs for the first time; the idea is natural and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive evaluation on four LIBERO subsets with thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ The two-stage framework is clearly explained and visually intuitive.
- Value: ⭐⭐⭐⭐ Provides a new design paradigm for VLAs; the decomposition of subgoal prediction and action generation has broad application prospects.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models](../../CVPR2026/robotics/acot-vla_action_chain-of-thought_for_vision-language-action_models.md)
- [\[CVPR 2026\] TRM-VLA: Temporal-Aware Chain-of-Thought Reasoning and Memorization for Vision-Language-Action Models](../../CVPR2026/robotics/trm-vla_temporal-aware_chain-of-thought_reasoning_and_memorization_for_vision-la.md)
- [\[CVPR 2025\] Overcoming Visual Clutter in Vision Language Action Models via Concept-Gated Visual Distillation](overcoming_visual_clutter_in_vision_language_action_models_via_concept-gated_vis.md)
- [\[CVPR 2025\] SaPaVe: Towards Active Perception and Manipulation in Vision-Language-Action Models for Robotics](sapave_towards_active_perception_and_manipulation_in_vision-language-action_mode.md)
- [\[CVPR 2025\] ShowUI: One Vision-Language-Action Model for GUI Visual Agent](showui_one_vision-language-action_model_for_gui_visual_agent.md)

</div>

<!-- RELATED:END -->
