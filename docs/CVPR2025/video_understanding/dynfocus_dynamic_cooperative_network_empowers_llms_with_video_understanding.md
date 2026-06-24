---
title: >-
  [Paper Note] DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding
description: >-
  [CVPR 2025][Video Understanding][Dynamic Encoding] This paper proposes DynFocus, an LLM-based dynamic cooperative video encoding network. It dynamically selects Q&A-related keyframes via the DPE module, and encodes keyframes with fine-grained tokens (analogous to visual Cones) and redundant frames with very few tokens for coarse-grained encoding (analogous to visual Rods) via the CCE module, balancing spatial details and temporal dynamics under a limited token budget.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Dynamic Encoding"
  - "Memory-Efficient"
  - "Large Language Models"
  - "Token Compression"
date: 2026-05-08
content_hash: 357c26f4dbbc8242
---

# DynFocus: Dynamic Cooperative Network Empowers LLMs with Video Understanding

**Conference**: CVPR 2025  
**arXiv**: [2411.12355](https://arxiv.org/abs/2411.12355)  
**Code**: [https://github.com/Simon98-AI/DynFocus](https://github.com/Simon98-AI/DynFocus)  
**Area**: Video Understanding  
**Keywords**: Video Understanding, Dynamic Encoding, Memory-Efficient, Large Language Models, Token Compression

## TL;DR

This paper proposes DynFocus, an LLM-based dynamic cooperative video encoding network. It dynamically selects Q&A-related keyframes via the DPE module, and encodes keyframes with fine-grained tokens (analogous to visual Cones) and redundant frames with very few tokens for coarse-grained encoding (analogous to visual Rods) via the CCE module, balancing spatial details and temporal dynamics under a limited token budget.

## Background & Motivation

The key challenge in LLM-based video understanding is: **long videos require a large number of tokens to retain visual semantic information, but LLMs have limited memory/context length**. Limitations of prior work include:

1. **Spatial compression methods** (e.g., average pooling, attention, dynamic masking): discard key visual details.
2. **Temporal sampling methods** (e.g., uniformly sampling a subset of frames): may miss keyframes.
3. **Memory bank methods** (e.g., MovieChat, MA-LMM): keyframes vary with the question, meaning static memory banks cannot adapt flexibly.

The authors make two key observations through statistical analysis:
- **Redundancy**: A large number of frames in a video are repetitive or unrelated to the answer (e.g., about 60-70% of frames in ActivityNet are redundant).
- **Correspondence**: Different questions require focusing on different frames in the video, meaning "keyframes" are question-dependent.

This inspires a dynamic encoding strategy: allocating different numbers of tokens based on the relevance of the frames to the given question.

## Method

### Overall Architecture

DynFocus consists of three parts: (1) visual + textual encoders to extract features; (2) a Dynamic Cooperative Network (DPE + CCE) acting as a connector to compress video tokens; and (3) an LLM that receives the compressed tokens to generate answers.

### Key Designs

1. **Dynamic Event Prototype Estimation (DPE)**:
    - **Function**: Dynamically select the $K$ keyframes most relevant to the Q&A from $T$ video frames.
    - **Mechanism**: Eliminates redundancy in two steps. **Step 1 (Temporal Redundancy Elimination)**: For each frame, spatial average pooling ($N \rightarrow P$ patches) is first applied, followed by DPC-KNN clustering along the temporal dimension to obtain $L$ event prototypes. Clustering uses the local density $\rho_t = \exp(-\frac{1}{C}\sum_{t' \in \mathcal{N}(t)} \frac{1}{P}\|\mathbf{f}_t - \mathbf{f}_{t'}\|_F^2)$ and a distance metric $\delta_t$, selecting the top-$L$ frames with the highest $\rho_t \times \delta_t$. **Step 2 (Answer-Irrelevant Redundancy Elimination)**: An MLP network $\mathcal{U}(\cdot)$ is used to regress a frame-level score $s_l = \mathcal{U}(\text{Max}(\mathbf{m}_l) || \text{Avg}(\mathbf{m}_l))$, selecting the Top-$K$ prototypes with the highest scores.
    - **Design Motivation**: DPC-KNN clustering is parameter-free and can adaptively discover "events" in the video. Since the Top-$K$ operation is non-differentiable, the perturbed maximum method is employed to convert it into a linear programming problem for end-to-end training: $\mathbf{P}_\sigma = \mathbb{E}_P[\text{argmax}_{\mathbf{P} \in \mathcal{C}} \langle \mathbf{P}, \mathbf{s}\mathbf{1}^\top + \sigma \mathbf{Z} \rangle]$, enabling the scoring network to be updated via the gradient of the LLM's autoregressive loss.

2. **Compact Cooperative Encoding (CCE)**:
    - **Function**: Encode keyframes and redundant frames separately, where the former retains details and the latter retains a summary.
    - **Mechanism**: Inspired by photoreceptors in the primate retina — **Cones Encoding** (for keyframes with $b_t=1$): concatenates event prototypes and multi-granularity spatial object prototypes (obtained via multi-layer spatial DPC-KNN clustering), and maps them using an MLP $\mathbf{U}_{t,b_t=1} = \mathcal{F}_{fine}(\mathbf{h}_t || \mathbf{G}_t)$, retaining all spatial tokens; **Rods Encoding** (for redundant frames with $b_t=0$): uses the text embedding $\mathbf{Q}$ for cross-attention modulation $\mathbf{E} = \text{Softmax}(\frac{f_q(\mathbf{G}_t) f_k(\mathbf{Q})^\top}{\sqrt{d}}) \mathbf{G}_t$, followed by average pooling to compress into only 2 tokens $\mathbf{U}_{t,b_t=0} = \mathcal{F}_{coarse}(\text{Avg}(\mathbf{E}) || \text{Avg}(\mathbf{G}_t))$.
    - **Design Motivation**: Keyframes require fine-grained spatial details (e.g., identifying object attributes), whereas redundant frames only need to provide temporal cues (e.g., sequence of events). 2 tokens are sufficient to maintain inter-frame temporal coherence while significantly compressing the token volume.

3. **Cooperative Encoding Fusion**:
    - **Function**: Unify Cones and Rods encoding.
    - **Mechanism**: $\mathbf{O}_t = b_t \cdot (\mathbf{U}_{t,b_t=1} || \mathbf{U}_{t,b_t=0}) + (1-b_t) \cdot \mathbf{U}_{t,b_t=0}$. Keyframes retain both fine-grained and coarse-grained information, while redundant frames only retain coarse-grained information.
    - **Design Motivation**: The Rods encoding of keyframes is also retained to provide temporal context for those frames; this cooperative design ensures that information is complementary.

### Loss & Training

**Two-Stage Training**:
- **Stage 1 (Vision-Language Alignment)**: Freezes the visual encoder and LLM, training only the projection layers of the dynamic cooperative network. It utilizes LLaVA-filter-CC3M image captions and WebVid-2.5M video captions.
- **Stage 2 (Instruction Fine-Tuning)**: Fully fine-tunes all parameters of the LLM, DPE, and CCE. It uses a subset of LLaVA-665K image QA, ScienceQA, and VideoChat2 datasets (VideoChatGPT-100K + NExT-QA + CLEVRER, etc.).

ViT-G/14 (EVA-CLIP) is used as the visual encoder, InstructBLIP's Qformer as the text encoder, and Vicuna-7B-1.5 as the LLM. Trained on 8×A100 80G GPUs.

## Key Experimental Results

### Main Results

| Dataset | Metric | DynFocus | Prev. SOTA (7B) | Gain |
|--------|------|------|----------|------|
| MSVD-QA | Acc/Score | 74.8/4.0 | ST-LLM 74.6/3.9 | Comparable |
| MSRVTT-QA | Acc/Score | 62.8/3.6 | ST-LLM 63.2/3.4 | Higher Score |
| ANet-QA | Acc/Score | 50.3/3.4 | ST-LLM 50.9/3.3 | Fewer tokens used |
| MLVU M-Avg | Multi-choice | 49.6% | MiniGPT4-V 44.5% | +5.1% |
| MLVU G-Avg | Generation | 4.38 | MiniGPT4-V 3.36 | +1.02 |
| LV-Bench | Overall | 32.9% | LLaVA-NeXT-34B 32.2% | 7B beats 34B |
| VideoMME Overall (w/o subs) | Accuracy | 44.1% | ST-LLM 37.9% | +6.2% |

### Ablation Study

| Configuration ($|\mathbf{U}_{b_t=0}|$ / $|\mathbf{U}_{b_t=1}|$) | MSVD-QA Acc | ANet-QA Acc | Description |
|------|---------|------|------|
| 0 / 40 (No Rods) | 63.7% | 41.4% | Loses temporal information |
| 2 / 0 (No Cones) | 58.2% | 38.6% | Loses spatial details, severe drop |
| 2 / 2 (Both coarse-encoded) | 62.0% | 40.5% | Compressing keyframes harms performance |
| 2 / 256 (No Cones compression) | 68.4% | 44.3% | Most tokens, best performance but large memory |
| **2 / 40 (Ours default)** | **67.9%** | **43.1%** | Close to uncompressed performance with significantly fewer tokens |

### Key Findings

- The initial number of event prototypes $L=25$ is optimal: too few ($<10$) cannot cover key events, while too many ($>25$) disrupt the temporal structure.
- A filtering ratio of $K/L=0.8$ is optimal for short videos; long videos (e.g., LV-Bench) require a larger $L$ and a smaller $K/L$ to handle more content.
- Removing Cones encoding (using only Rods) leads to a 9.7% drop on MSVD-QA, indicating that spatial details are critical.
- Removing Rods encoding (using only Cones) also results in a 4.2% drop, showing that temporal information from redundant frames cannot be ignored.
- DynFocus without subtitles (44.1%) outperforms the subtitled version of ST-LLM (42.3%), proving that dynamic encoding can effectively compensate for missing information.

## Highlights & Insights

- **Biologically inspired, unique and reasonable**: The analogy to Cones/Rods is not just rhetorical but practically maps to the functional division of fine-grained and coarse-grained encoding, making the design philosophy highly convincing.
- **End-to-end differentiable dynamic selection**: Resolves the non-differentiability of the Top-$K$ operation using the perturbed maximum method, allowing the DPE scoring network to be implicitly supervised by the LLM loss.
- **Highly token-efficient**: DynFocus uses far fewer tokens than competing methods on MLVU while achieving optimal performance.
- The framework is general, and the visual encoder can be replaced with other CLIP-based encoders.

## Limitations & Future Work

- Shows weaker performance on ego-centric videos (e.g., ER tasks), which may require targeted ego-centric video data.
- Hyperparameters of DPC-KNN clustering (such as the number of neighbors $C$) require manual tuning.
- Validated only on 7B models; performance on larger-scale LLMs remains unknown.
- The direct integration of text guidance into the DPE frame selection process (currently DPE only uses visual features for frame selection) can be explored.

## Related Work & Insights

- **Relation to LLaMA-VID**: LLaMA-VID uses a dual-token (context + content) representation for each frame, whereas DynFocus dynamically allocates different granularities.
- **Relation to Chat-UniVi**: Chat-UniVi compresses tokens via token merging, whereas DynFocus relies on a more flexible clustering and importance-scoring mechanism.
- **Insight**: The concept of dynamic encoding can be generalized to other long-sequence modalities (e.g., chunk-level dynamic representations for long audio or documents).

## Rating
- Novelty: ⭐⭐⭐⭐ End-to-end differentiable frame selection with DPE + cooperative Cones/Rods encoding with CCE, offering a clean and novel concept.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 5 benchmarks covering short/long videos and hallucination detection, with comprehensive ablations.
- Writing Quality: ⭐⭐⭐⭐ Appropriate biological analogy with rigorous mathematical formulations.
- Value: ⭐⭐⭐⭐ Provides an efficient and effective solution to the token-efficiency problem in LLM-based video understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] STOP: Integrated Spatial-Temporal Dynamic Prompting for Video Understanding](stop_integrated_spatial-temporal_dynamic_prompting_for_video_understanding.md)
- [\[CVPR 2025\] Object-Shot Enhanced Grounding Network for Egocentric Video](object-shot_enhanced_grounding_network_for_egocentric_video.md)
- [\[CVPR 2025\] OVO-Bench: How Far is Your Video-LLMs from Real-World Online Video Understanding?](ovo-bench_how_far_is_your_video-llms_from_real-world_online_video_understanding.md)
- [\[CVPR 2025\] DPU: Dynamic Prototype Updating for Multimodal Out-of-Distribution Detection](dpu_dynamic_prototype_updating_for_multimodal_out-of-distribution_detection.md)
- [\[CVPR 2026\] UFVideo: Towards Unified Fine-Grained Video Cooperative Understanding with Large Language Models](../../CVPR2026/video_understanding/ufvideo_towards_unified_fine-grained_video_cooperative_understanding_with_large_.md)

</div>

<!-- RELATED:END -->
