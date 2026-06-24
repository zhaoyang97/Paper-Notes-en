---
title: >-
  [Paper Note] Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment
description: >-
  [CVPR 2025][Multimodal VLM][MLLM] This paper proposes Task Preference Optimization (TPO), which integrates specialized vision task heads (region grounding, temporal grounding, and segmentation) into MLLMs via learnable task tokens. By leveraging vision task annotations as "task preferences" to backpropagate and optimize the MLLM, this approach significantly enhances fine-grained visual understanding without compromising conversation capabilities…
tags:
  - "CVPR 2025"
  - "Multimodal VLM"
  - "MLLM"
  - "Multi-Task Learning"
  - "Task Alignment"
  - "Visual Perception Head"
  - "Video Understanding"
  - "Object Tracking"
  - "Temporal Grounding"
date: 2026-05-08
content_hash: 8f8b209721a5c80c
---

# Task Preference Optimization: Improving Multimodal Large Language Models with Vision Task Alignment

**Conference**: CVPR 2025  
**arXiv**: [2412.19326](https://arxiv.org/abs/2412.19326)  
**Code**: [https://github.com/OpenGVLab/TPO](https://github.com/OpenGVLab/TPO)  
**Area**: LLM Alignment  
**Keywords**: MLLM, Multi-Task Learning, Task Alignment, Visual Perception Head, Video Understanding, Object Tracking, Temporal Grounding

## TL;DR
This paper proposes Task Preference Optimization (TPO), which integrates specialized vision task heads (region grounding, temporal grounding, and segmentation) into MLLMs via learnable task tokens. By leveraging vision task annotations as "task preferences" to backpropagate and optimize the MLLM, this approach significantly enhances fine-grained visual understanding without compromising conversation capabilities, achieving an average improvement of 14.6% over the VideoChat baseline.

## Background & Motivation
**Background**: Current MLLMs (e.g., LLaVA, VideoChat) perform exceptionally well in general visual conversation but lack sufficient capability in fine-grained visual tasks (e.g., object tracking, temporal grounding, segmentation), suffering from a bottleneck in precise perception.

**Limitations of Prior Work**: Existing solutions either textualize vision tasks for autoregressive prediction (P2S approach, e.g., Shikra, TimeChat) or connect external tools (P2E approach, e.g., LISA). However, textualization leads to a loss of precision due to discretization, and joint multi-task training often degrades original conversational abilities—contradicting the "multi-task mutual benefit" experience in vision foundation models.

**Key Challenge**: The authors observe that the root cause of this conflict lies in the learning discrepancy between discrete text tokens and dense visual predictions—using autoregressive text loss to learn bounding box coordinates or timestamps is inherently mismatched. Decoupling the representations of the two can resolve this issue.

**Goal**: How to introduce precise supervision from multiple fine-grained vision tasks end-to-end without compromising the conversational capability of MLLMs, while enabling different tasks to mutually reinforce each other?

**Key Insight**: Inspired by DPO which uses preference signals to guide LLMs, visual task annotations can be viewed as human preferences for "precise perception". The MLLM can be optimized via the backpropagation of differentiable losses from specialized heads.

**Core Idea**: Learnable task tokens are introduced as a bridge between the MLLM and the vision task heads. The task heads receive dense visual supervision and propagate gradients back to the MLLMs, achieving the goal of "optimizing language models using vision task supervision."

## Method

### Overall Architecture
MLLM-TPO consists of two parts: a standard MLLM $M$ (visual encoder $E$ + connector $C$ + LLM $G$) and a Task Preference Model (TPM) $P$ (learnable task tokens $\{\mathbf{v}_i\}$ + task heads $\{H_i\}$).

Input: Image/video + user instruction. The MLLM first identifies the type of task involved in the instruction and activates the corresponding task token. The task token is processed by the LLM to obtain the embedding $\mathbf{e}_j = G(\mathbf{v}_j)$, which is then fed into the corresponding vision task head for dense prediction.

Total optimization objective: $\mathcal{L} = \mathcal{L}_{\text{mllm}} + \mathcal{L}_{\text{assign}}(G(\mathbf{T}_q), \mathbf{s}) + \sum_{i=1}^{n} \mathcal{L}_{\text{task}}(\mathbf{A}_i, H_i(G(\mathbf{v}_i)))$

Where $\mathcal{L}_{\text{mllm}}$ is the standard conversation loss, $\mathcal{L}_{\text{assign}}$ is the task-type assignment loss, and $\mathcal{L}_{\text{task}}$ represents the supervision losses (regression/classification) for each vision task head.

### Key Designs

1. **Three Vision Task Heads (Task Preference Model)**:

    - **Function**: Processes spatial grounding, temporal grounding, and pixel-level segmentation, which are three core visual perception tasks.
    - **Region Head**: Composed of a 2-layer MLP and ReLU, mapping LLM embeddings back to bounding box coordinates for spatial grounding (referring expression grounding).
    - **Temporal Head**: Based on the CG-DETR architecture, containing video and text encoders; it takes temporal task embeddings to predict moment start/end times and highlight scores for temporal grounding.
    - **Mask Head**: Reuses the image encoder and mask decoder of SAM2, replacing the prompt encoder with a single-layer MLP (mask adapter) to achieve referring segmentation and tracking.
    - **Design Motivation**: These three heads cover most discriminative vision tasks, enabling the reuse of mature, native expert model architectures.

2. **Learnable Task Tokens as a Bridge**:

    - **Function**: Decouples vision task representations from MLLM textual representations.
    - **Mechanism**: Each task is associated with a learnable token $\mathbf{v}_i \in \mathbb{R}^{1 \times C}$. After inputting into the LLM, it outputs a task embedding $\mathbf{e}_i = G(\mathbf{v}_i)$, which is then sent to the corresponding task head. This allows dense visual supervision of tasks to backpropagate gradients from head $\to$ embedding $\to$ LLM, indirectly strengthening the MLLM's visual understanding.
    - **Design Motivation**: Avoids hard-converting vision tasks into text (which leads to information loss) while allowing task gradients to flow back to the LLM.

3. **Three-stage Local-to-Global Training**:

    - **Stage 1 (Task Assignment)**: Fine-tune the LLM with LoRA to recognize task types from user instructions and generate corresponding special tokens (50k samples per task).
    - **Stage 2 (Vision Task Training)**: Train each task head and its corresponding task token individually to establish basic capabilities while freezing the visual encoder and connector.
    - **Stage 3 (Multi-task Training)**: Unfreeze all modules and jointly train them on a mixture of multi-task and conversational data, allowing task head gradients to flow back to the entire MLLM.
    - **Design Motivation**: The local-to-global strategy prevents the degradation of the MLLM’s conversational abilities caused by immediate joint training. The phased approach allows the model to adapt gradually.

### Loss & Training
- Region Head: MSE loss (coordinate regression)
- Temporal Head: Original loss of CG-DETR (regression + classification)
- Mask Head: Original mask loss of SAM2
- Training Configuration: 32-64 A100 GPUs, batch size 128-256, LR 2e-5, LoRA rank=16, alpha=32, DeepSpeed bf16
- Data Volume: Stage 1 ~150k, Stage 2 ~700k, Stage 3 ~3.5M (mostly conversational data)

## Key Experimental Results

### Main Results
Multimodal video understanding based on VideoChat2 (Mistral-7B, 16 frames):

| Model | MVBench | VideoMME | MLVU |
|------|---------|----------|------|
| VideoChat2 (baseline) | 60.4 | 39.5 | 44.5 |
| **VideoChat-TPO** | **66.8 (+6.4)** | **48.8 (+9.3)** | **54.7 (+10.2)** |
| ST-LLM (64 frames) | 54.9 | 37.9 | — |
| ShareGPT4Video | 51.2 | 39.9 | 46.4 |

Vision task performance (zero-shot + fine-tuned):

| Task | Dataset | VideoChat-TPO | Compared Methods |
|------|--------|---------------|---------|
| Temporal Grounding (Zero-shot) | Charades-STA R@0.5 | **40.2** | ChatVTG: 33.0 |
| Temporal Grounding (Fine-tuned) | Charades-STA R@0.5 | **65.0** | UniVTG: 60.2 |
| Spatial Grounding | RefCOCO val | **85.9** | NExT-Chat: 85.5 |
| Tracking | LaSOT Success | **69.4** | Merlin: 39.8 |
| Video Segmentation (Zero-shot) | MeViS J&F | **47.0** | VideoLISA: 44.4 |

### Ablation Study

| Configuration | Charades R@0.5 | RefCOCO Acc@0.5 | MeViS J&F | MVBench |
|------|---------------|----------------|-----------|---------|
| Temporal Head Only | 30.2 | — | — | — |
| Region Head Only | — | 77.3 | — | — |
| Mask Head Only | — | — | 55.1 | — |
| Region + Mask | — | 80.2 | 58.3 | — |
| T + R + M (w/o Conversational Data) | 36.7 | 81.6 | 61.4 | — |
| **T + R + M + Conv (Full)** | **40.2** | **82.0** | **63.9** | **66.8** |
| Alternative Textualized Task Data | 18.6 | — | — | 64.7 |
| Alternative Simple MLP instead of CG-DETR | 17.8 | — | — | 65.8 |

### Key Findings
- **Significant Multi-task Synergy**: Jointly training three task heads with conversational data far outperforms training them individually, with each additional head yielding incremental gains. This indicates that perception capabilities for different vision tasks can mutually reinforce one another.
- **TPO Outperforms Textualized Task Data**: Under identical data scales, TPO achieves a 2.1% performance gain on MVBench compared to textualized task data, proving that dense visual supervision is more effective than textualized autoregressive supervision.
- **Stronger Task Heads Yield Larger Gains**: CG-DETR outperforms a simple MLP by 22.4 points on temporal tasks and provides a greater boost to overall multimodal performance (MVBench +1.0%).
- **Cross-Model Generalization**: TPO also brings an 8.1% enhancement on MVBench when applied to LLaVA-OneVision, validating its generalizability.
- **Effective Data Scaling**: Adding temporal reasoning data increases Charades R@0.5 from 38.3 to 40.2, and scaling conversational data continuously improves all metrics.

## Highlights & Insights
- **The "Optimizing Language Models with Vision Supervision" Paradigm is Inspiring**: Enhancing the MLLM via the backpropagation of dense annotation gradients from task heads is more elegant than traditional textualization or tool-calling approaches, successfully avoiding information loss.
- **Simple and Effective Learnable Task Token Design**: A single token acts as a bridge between the MLLM and expert heads. Moreover, it is highly extensible for new vision tasks, requiring only new tokens and heads.
- **Practical Significance of Multi-task Synergy**: Jointly training tracking and segmentation mutually benefits both, and synergy is also observed between temporal and spatial grounding. This offers valuable insights for designing multi-functional visual Agents.

## Limitations & Future Work
- **Covers Only Discriminative Vision Tasks**: The authors acknowledge that generative tasks (such as image or video generation) are not included, which limits TPO's task diversity.
- **Reliance on Human-Annotated Data**: Unsupervised or self-supervised schemes (like contrastive learning) were not explored to provide task preference signals, which limits scalability.
- **High Training Cost**: The 3-stage training requires approximately 63.5 hours on 64 A100 GPUs, which is about 25% more than pure conversation fine-tuning.
- **Relatively Fixed Task Head Architectures**: Currently, the three heads are manually selected. Investigating how to automatically discover beneficial combinations of vision tasks is a worthy research direction.

## Related Work & Insights
- **vs. Shikra/TimeChat (P2S approach)**: These methods textualize vision tasks for autoregressive prediction, leading to information loss and degradation of conversation capability. TPO achieves better performance by using independent task heads to receive raw supervision signals.
- **vs. LISA/NExT-Chat (P2E approach)**: They also utilize external decoders but typically focus on a single task and do not emphasize the simultaneous enhancement of dialogue capability. TPO's three-stage training ensures the synchronized improvement of both multi-task and conversation capabilities.
- **vs. DPO/PPO (Preference Alignment)**: Traditional preference alignment focuses on textual response quality. TPO extends the concept of "preference" to visual perception accuracy, representing a novel application of preference alignment in the vision domain.

## Rating
- Novelty: ⭐⭐⭐⭐ The concept of "task preference" is novel, and optimizing MLLMs via differentiable task heads is a meaningful paradigm.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 6+ multimodal benchmarks and 7+ vision task benchmarks with highly comprehensive ablation studies.
- Writing Quality: ⭐⭐⭐⭐ Highly clear structure, rich charts, with the three-stage training process thoroughly explained.
- Value: ⭐⭐⭐⭐ Provides an effective training methodology for constructing multi-functional MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Debiasing Multimodal Large Language Models via Noise-Aware Preference Optimization](debiasing_multimodal_large_language_models_via_noise-aware_preference_optimizati.md)
- [\[CVPR 2025\] SymDPO: Boosting In-Context Learning of Large Multimodal Models with Symbol Demonstration Direct Preference Optimization](symdpo_boosting_in-context_learning_of_large_multimodal_models_with_symbol_demon.md)
- [\[CVPR 2025\] SPA-VL: A Comprehensive Safety Preference Alignment Dataset for Vision Language Models](spa-vl_a_comprehensive_safety_preference_alignment_dataset_for_vision_language_m.md)
- [\[CVPR 2026\] LVLM-Aided Alignment of Task-Specific Vision Models](../../CVPR2026/multimodal_vlm/lvlm-aided_alignment_of_task-specific_vision_models.md)
- [\[ICML 2025\] Vision-Language Models Create Cross-Modal Task Representations](../../ICML2025/multimodal_vlm/vision-language_models_create_cross-modal_task_representations.md)

</div>

<!-- RELATED:END -->
