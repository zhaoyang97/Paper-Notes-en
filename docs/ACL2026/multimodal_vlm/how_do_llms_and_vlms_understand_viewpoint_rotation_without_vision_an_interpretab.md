---
title: >-
  [Paper Note] How Do LLMs and VLMs Understand Viewpoint Rotation Without Vision? An Interpretability Study
description: >-
  [ACL 2026][Multimodal VLM][Viewpoint Rotation Understanding] This paper introduces VRUBench, a textual Viewpoint Rotation Understanding (VRU) benchmark. Using layer-wise probing and head-wise path patching…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Viewpoint Rotation Understanding"
  - "Path Patching"
  - "Probing"
  - "Attention Heads"
  - "Selective Fine-tuning"
date: 2026-05-08
content_hash: 39b398fa7350f872
---

# How Do LLMs and VLMs Understand Viewpoint Rotation Without Vision? An Interpretability Study

**Conference**: ACL 2026  
**arXiv**: [2604.15294](https://arxiv.org/abs/2604.15294)  
**Code**: https://github.com/Young-Zhen/VRU_Interpret (Available)  
**Area**: Multimodal VLM / Interpretability / Spatial Intelligence  
**Keywords**: Viewpoint Rotation Understanding, Path Patching, Probing, Attention Heads, Selective Fine-tuning

## TL;DR
This paper introduces VRUBench, a textual Viewpoint Rotation Understanding (VRU) benchmark. Using layer-wise probing and head-wise path patching, it reveals that the near-random performance of LLMs/VLMs on this task stems from the failure of key heads in middle-to-late layers to bind "perceived orientation" with "corresponding observations." By fine-tuning only 32 key heads, the authors achieve results comparable to full fine-tuning in 50% of the GPU time without compromising general capabilities.

## Background & Motivation

**Background**: Spatial intelligence has become a focal point recently, but most research treats it as visual-spatial intelligence—assuming models must "see" to understand space. Purely textual evaluations of fundamental tasks like VRU (predicting final observations after multi-step rotations) remain systematically unexplored.

**Limitations of Prior Work**: VRUBench, a text-only benchmark, shows that almost all open-source LLMs/VLMs perform below 60%, while humans easily reach 100%. Even Qwen3-VL-32B-thinking only reaches 70%. Existing literature does not explain whether models "understand" orientation in their intermediate representations or why final outputs remain incorrect.

**Key Challenge**: Probing indicates that mid-early layers can already encode direction, angle, and even absolute orientation. However, orientation probing accuracy declines in layers 21-28, and final answers approach random guessing—suggesting that "information exists but is discarded."

**Goal**: (1) Quantify the textual spatial capabilities of LLMs/VLMs using a text-based benchmark; (2) Identify specific computational paths causing poor performance via layer-wise probing and head-wise path patching; (3) Translate interpretability findings into model improvement methods.

**Key Insight**: Drawing from Gardner's Theory of Multiple Intelligences, which suggests spatial perception is possible without vision, LLMs should possess partial spatial intelligence even without visual input. By serializing multi-step rotations and observations into text, the model is forced to maintain a "mental map" in the residual stream, which can then be analyzed via mechanistic interpretability.

**Core Idea**: Employs an "interpret-then-improve" paradigm—using path patching to locate sparse key heads for VRU and fine-tuning only these heads to achieve SOTA performance while preserving general capabilities.

## Method

### Overall Architecture
The methodology consists of three layers: (1) **Task Definition and Data**: Formalizing prompts as $P = I \oplus O_0 \oplus A_1 \oplus O_1 \oplus \cdots \oplus A_n$ with rotation angles $\theta \in \{0°, 90°, 180°, 270°, 360°\}$, creating 19,591 samples of 2-5 steps; (2) **Interpretability Analysis**: Utilizing layer-wise probes for direction/angle/orientation encoding and head-wise path patching to find attention heads with high causal effects; (3) **Selective SFT**: Making only the $W_{K/Q/V/O}^{i,j}$ of the top-32 key heads trainable while freezing the rest.

### Key Designs

1.  **Layer-wise Linear Probing**:
    - **Function**: Uses linear classifiers to detect if the hidden states of the last token in each layer encode direction (binary), angle (5-way), or absolute orientation (4-way).
    - **Mechanism**: Extracts $\boldsymbol{h}_{i\ell}$ from the last token of each action $A_i$ and trains $\mathcal{F}_\ell$ to map $\boldsymbol{h}_{i\ell}$ to labels. Linear probes are used to ensure the probe itself does not learn the knowledge.
    - **Key Findings**: Direction/angle probing exceeds 99% across almost all layers; however, absolute orientation accuracy rises slowly in layers 1-20 and drops in layers 21-28, suggesting a pattern shift in later layers.
    - **Design Motivation**: Control experiments (random embeddings) confirm the signal's reality. The 5-way/4-way design checks both the existence of local information and its aggregation into global orientation.

2.  **Head-wise Path Patching**:
    - **Function**: Locates attention heads in middle-to-late layers that have a causal effect on VRU output.
    - **Mechanism**: Constructs clean-corrupted pairs (flipping the last rotation direction), defines causal effect $\phi_i = (logit_{pt} - logit_{cl}) / (logit_{cor} - logit_{cl})$. Each head acts as a Sender, replacing its activations under corrupted input to observe the impact on the final logit difference $\mathcal{M}(t_{cl}|\cdot) - \mathcal{M}(t_{cor}|\cdot)$.
    - **Design Motivation**: Perturbing the last step maintains token length consistency. Filtering pairs where the answer remains unchanged ensures non-zero $logit_*$. Three types of heads were found: **Proposal heads (22.1)** attending to all candidates, **Answer Decision heads (26.14, 23.11)** focusing on the selected answer, and **Unknown heads (27.14)** showing a preference for "unknown" tokens (reflecting alignment-induced caution).

3.  **Selective Fine-tuning**:
    - **Function**: Improves VRU performance while preserving general capabilities with minimal parameters and GPU hours.
    - **Mechanism**: $W_{K/Q/V/O}^i$ is divided into $H$ head blocks. Only $W_{K/Q/V/O}^{i,j}$ for the top-32 key heads are trainable. Gradients are rescaled by $H/h$ to compensate for update biases.
    - **Design Motivation**: Path patching proves these heads are the "decision-making" locations. SFT on them acts as a targeted adjustment. On Qwen2.5-VL-3B, only 0.03B parameters are modified (vs. 3B), increasing throughput from 10 to 18 samples/sec and avoiding catastrophic forgetting.

### Loss & Training
Standard cross-entropy SFT + Adam, lr $2 \times 10^{-5}$, batch 32, warmup 0.02, weight decay 0.1, 1 epoch. The training set (19,641 items) is strictly separated from the VRUBench test set.

## Key Experimental Results

### Main Results

| Model | 2-step | 3-step | 4-step | 5-step | Avg |
|---|---|---|---|---|---|
| Human | 100 | 100 | 100 | 100 | **100** |
| LLaMA2-7B-chat | 5.44 | 17.22 | 26.24 | 25.64 | 18.90 |
| Qwen2.5-32B | 88.56 | 74.20 | 67.54 | 62.28 | 72.84 |
| Qwen3-VL-32B-thinking | 97.90 | 96.44 | 96.16 | 95.82 | **96.55** |
| Gemini3-Flash-thinking | 93.15 | 90.32 | 85.71 | 76.65 | 86.32 |

| Configuration (Qwen2.5-VL-7B) | Train Speed | Tuned Param | VRUBench Acc | SpinBench (OOD) | MMLU | BBH |
|---|---|---|---|---|---|---|
| baseline | - | - | 48.7 | 44.8 | 60.3 | 49.2 |
| Full SFT | 5 sam/sec | 7.0B | **96.3 (+47.6)** | 47.3 (+2.5) | 55.6 (**-4.7**) | 35.8 (**-13.4**) |
| Selective SFT (Ours) | 11 sam/sec | 0.06B | 78.7 (+30.0) | **48.4 (+3.6)** | 60.3 (+0.0) | 48.4 (-0.8) |

### Ablation Study

| Configuration | Key Observation | Description |
|---|---|---|
| All heads | VRU baseline | Qwen2.5-VL-7B |
| Remove top-K causal heads | Sharp performance drop | Proves identified heads perform VRU computation |
| Remove random K heads | Nearly no change | Control experiment excluding non-specific degradation |
| Remove Unknown head 27.14 | "Unknown" output % 65.78 → 40.73 | Proves head handles "caution toward the unknown" |
| LLM vs VLM (same size) | Qwen2.5-VL-7B > Qwen2.5-7B | Visual training strengthens spatial representation even without images |
| think vs no-think | Qwen3-VL-8B-think 85.57 vs no-think 64.33 | Explicit reasoning is effective for textual spatial tasks |

### Key Findings
- **Selective SFT outperforms full SFT by 1.1 points on OOD visual-spatial dataset SpinBench** with near-zero loss on MMLU/BBH. Full SFT shows catastrophic forgetting (dropping 13.4 points on BBH).
- **The 3B model exceeds the 7B model after selective SFT** (80.1 vs 78.7) because 32 heads represent a higher relative training proportion in the smaller model (5.6% vs 4.1%).
- **Reasoning is "effective" for textual spatial tasks but "ineffective" for visual-spatial tasks**, mirroring and contrasting results from Yang et al. 2025, suggesting fundamental differences between the two modalities.
- **Path patterns are consistent across models**: LLaMA2-7B, Qwen2.5-7B, and Qwen2.5-VL-3B all exhibit sparse key heads in later layers, indicating a universal Transformer phenomenon.

## Highlights & Insights
- **The "interpret-then-improve" paradigm is practical**: Finding heads via path patching followed by selective SFT creates a closed loop applicable to any "model understands but fails to express" task (e.g., math or factoid QA).
- **The "Unknown head" discovery is significant**: Alignment training implants a specific "I don't know" head. Ablating it makes the model more assertive, offering insights into RLHF side effects and honesty calibration.
- **VLMs outperform equivalent LLMs even without image input**: This supports dual-coding theory, suggesting visual training embeds "spatial-language" coupling into Transformer parameters.

## Limitations & Future Work
- **Scaled only to ≤7B models**: Computational costs of path patching prevented evaluation on 70B+ scales; whether key heads change with scale remains unknown.
- **Prompt sensitivity**: The authors acknowledge that phrasing changes may shift results, leaving robustness verification for future work.
- **Single spatial capability**: It is unclear if complex tasks like navigation or 3D rotation reuse the same head mechanism.
- **CoT interpretability excluded**: The study focuses on implicit reasoning ("direct answer"), while the "think-then-answer" mode is the strongest setting but not analyzed mechanistically.

## Related Work & Insights
- **vs. Yang et al. 2025 (VSI-Bench)**: Their finding that "CoT is useless" for visual-spatial tasks contrasts with this paper's finding that "CoT is useful" for textual ones, indicating modality-dependent reasoning gains.
- **vs. Guo et al. 2026 (Beyond Flatlands)**: Guo observed VLM > LLM across 17 textualized visual-spatial tasks, reinforcing this paper's findings.
- **vs. ITI (Li et al. 2023)**: While ITI adds steering vectors, this work fine-tunes head parameters directly. Both could be combined for inference-time steering after SFT.

## Rating
- Novelty: ⭐⭐⭐⭐ First systematic study of internal mechanisms for textual spatial intelligence.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated 27 models, dual-layer probing, OOD and general capability assessments.
- Writing Quality: ⭐⭐⭐⭐ Clear "interpret-then-improve" narrative with understandable visualization.
- Value: ⭐⭐⭐⭐ The "find then fine-tune heads" paradigm is highly relevant for resource-constrained scenarios and alignment research.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Do MLLMs Understand Pointing? Benchmarking and Enhancing Referential Reasoning in Egocentric Vision](do_mllms_understand_pointing_benchmarking_and_enhancing_referential_reasoning_in.md)
- [\[ICLR 2026\] How Do Medical MLLMs Fail? A Study on Visual Grounding in Medical Images](../../ICLR2026/multimodal_vlm/how_do_medical_mllms_fail_a_study_on_visual_grounding_in_medical_images.md)
- [\[ACL 2026\] Mitigating Hallucinations in Large Vision-Language Models without Performance Degradation](mitigating_hallucinations_in_large_vision-language_models_without_performance_de.md)
- [\[ICLR 2026\] SpinBench: Perspective and Rotation as a Lens on Spatial Reasoning in VLMs](../../ICLR2026/multimodal_vlm/spinbench_perspective_and_rotation_as_a_lens_on_spatial_reasoning_in_vlms.md)
- [\[ACL 2026\] What Do Vision-Language Models Encode for Personalized Image Aesthetics Assessment?](what_do_vision-language_models_encode_for_personalized_image_aesthetics_assessme.md)

</div>

<!-- RELATED:END -->
