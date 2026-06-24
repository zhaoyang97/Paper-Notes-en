---
title: >-
  [Paper Note] Number it: Temporal Grounding Videos like Flipping Manga
description: >-
  [CVPR 2025][Video Understanding][Video Temporal Grounding] This paper introduces NumPro, which overlays frame indices directly onto the bottom-right corner of video frames. This binds the tasks of "observing an event" and "reporting the corresponding frame index" into a single OCR task for Vid-LLMs, significantly improving the mIoU and mAP of Video Temporal Grounding (VTG) under zero-shot settings or lightweight LoRA fine-tuning.
tags:
  - "CVPR 2025"
  - "Video Understanding"
  - "Video Temporal Grounding"
  - "Frame Number Overlay"
  - "Vid-LLM"
  - "Moment Retrieval"
  - "Highlight Detection"
date: 2026-05-08
content_hash: 7fec58c9853194e6
---

# Number it: Temporal Grounding Videos like Flipping Manga

**Conference**: CVPR 2025  
**arXiv**: [2411.10332](https://arxiv.org/abs/2411.10332)  
**Code**: [https://github.com/yongliang-wu/NumPro](https://github.com/yongliang-wu/NumPro)  
**Area**: Video Understanding / Vid-LLM / Visual Prompting  
**Keywords**: Video Temporal Grounding, Frame Number Overlay, Vid-LLM, Moment Retrieval, Highlight Detection

## TL;DR
This paper introduces NumPro, which overlays frame indices directly onto the bottom-right corner of video frames. This binds the tasks of "observing an event" and "reporting the corresponding frame index" into a single OCR task for Vid-LLMs, significantly improving the mIoU and mAP of Video Temporal Grounding (VTG) under zero-shot settings or lightweight LoRA fine-tuning.

## Background & Motivation
1. **Background**: Vid-LLMs (such as Qwen2-VL, LongVA, and GPT-4o) are highly capable in video QA, but perform poorly in Video Temporal Grounding (VTG)—which requires predicting start and end frames/timestamps given an event description.
2. **Limitations of Prior Work**: Existing VTG methods either require retraining dedicated tokens or temporal embeddings (e.g., TimeChat, VTG-LLM), or hard-coding timestamps into textual inputs (e.g., VTimeLLM). These methods suffer from poor scalability and frequent hallucinations, such as outputting "frame 200 to 599" for a 10-frame video.
3. **Key Challenge**: Through attention analysis, the authors find that Vid-LLMs actually attend to the correct frames. The problem is not "failing to see", but rather "seeing but failing to state the corresponding timestamp"—meaning there is a lack of a bridge to align visual recognition and temporal expression.
4. **Goal**: To transform the "temporal expression" problem into a "visual recognition" problem that Vid-LLMs already excel at, without modifying the architecture or introducing additional tokens.
5. **Key Insight**: Analogous to reading manga, where each frame is labeled with a page or panel number, readers can naturally ground where an event occurs.
6. **Core Idea**: Overlay frame numbers as visual prompts onto the frames, letting the model directly "read" time using its built-in OCR capability, thereby converting VTG into a visual alignment task.

## Method

### Overall Architecture
NumPro can be deployed in two modes, both sharing the same core mechanism of overlaying numbers on frames:
- **Training-free (NumPro)**: Renders corresponding frame indices onto each frame of the source video (red color, font size 40, pasted in the bottom-right corner) and prepends the prompt with "The red numbers on each frame represent the frame number.", then directly feeds it to an off-the-shelf Vid-LLM.
- **Fine-tuning (NumPro-FT)**: Applies NumPro to the VTG training set to construct augmented data and fine-tunes the Vid-LLM via LoRA (freezing the visual encoder while only fine-tuning the projection layer and the LoRA-adapted LLM).

The input consists of the video and a natural language event query. The output consists of the start/end frame numbers or time intervals (moment retrieval) or frame-wise highlight scores (highlight detection).

### Key Designs

1. **Visual Number Prompt (Number-Prompt Overlay)**

    - **Function**: Paints the global frame number at a fixed location of each frame, enabling the frame to simultaneously carry both content and temporal information.
    - **Mechanism**: Extracts the video frame sequence $\{f_1, ..., f_N\}$. For each frame $f_t$, PIL/OpenCV is used to render text $t$ in red, font size 40, at the bottom-right corner. When the model processes the frames, it retrieves $t$ as tokens via OCR and associates them with the event query.
    - **Design Motivation**: Vid-LLMs possess strong character recognition capabilities but lack the ability to "count frames." Explicitly visualizing frame numbers provides the model with a temporal ruler that attention can directly attend to, shifting the problem from "when did it happen" to "what is written on the image."

2. **NumPro Parameter Search (Font Size/Color/Position)**

    - **Function**: Striking a balance between the legibility of numbers and leaving visual content unobstructed.
    - **Mechanism**: Conducting proxy experiments on 1,000 MSCOCO images using CLIP ViT-B/32. Each image is overlaid with numbers 0-99 in different font sizes (20/40/60/80), colors (red/yellow/black/white, etc.), and positions (TL/TR/BL/BR/Center). The Number Accuracy (whether CLIP can select the correct number) and Caption Accuracy (whether CLIP can still select the original caption) are computed. A red color, font size 40, located at the bottom-right corner is finalized as it lies on the Pareto frontier of both metrics.
    - **Design Motivation**: Previous visual prompting works (e.g., red circle, SoM) have demonstrated that color and position strongly affect VLM attention. Using CLIP as a proxy enables low-cost hyperparameter scanning, avoiding large-scale parameter search on VTG datasets.

3. **NumPro-FT Fine-tuning Strategy**

    - **Function**: Internalize the new "read-frame-numbers" behavior into the model weights, enabling stable frame index responses even in boundary cases.
    - **Mechanism**: Renders training videos in Charades-STA / ActivityNet / QVHighlights using NumPro, formats answers into the form "from frame X to frame Y", and performs LoRA fine-tuning utilizing the autoregressive language modeling objective $P(\mathbf{A}\mid V, T_{\text{instruct}}) = \prod_{j} P_\theta(A_j\mid V, X_{\text{instruct}}, \mathbf{A}_{<j})$. The visual encoder is frozen, and only the visual projector and LoRA-injected LLM components are updated.
    - **Design Motivation**: Although the training-free mode provides stable performance gains, LoRA fine-tuning teaches the model more precise boundaries (e.g., for strict IoU metrics like R@0.7) without degrading general capabilities.

### Loss & Training
Only the standard next-token prediction loss is used, optimizing the average cross-entropy over answer tokens. During fine-tuning, only LoRA parameters and the Visual Projector are updated, maintaining a minimal parameter footprint. The training-free mode requires no parameter updates.

## Key Experimental Results

### Main Results
Comparison on three major VTG benchmarks against specialized VTG Vid-LLMs and general Vid-LLMs.

| Dataset | Metric | General Vid-LLM Baseline (Qwen2-VL-7B) | +NumPro (training-free) | NumPro-FT (Fine-tuned) | Prev. SOTA |
|--------|------|----------|----------|----------|----------|
| Charades-STA | R@0.5 | 5.4 | 36.8 | Higher | 33.8 (VTG-LLM) |
| Charades-STA | mIoU | 7.9 | 38.5 | — | 33.7 (HawkEye) |
| ActivityNet | R@0.5 | 9.4 | 26.4 | — | 27.8 (VTimeLLM) |
| QVHighlights | mAP | 21.5 | 23.6 | — | 16.5 (VTG-LLM) |
| GPT-4o | R@0.5 (Charades) | 32.0 | **35.5** | — | — |

NumPro-FT outperforms the previous state-of-the-art by **6.9%** in mIoU for moment retrieval and by **8.5%** in mAP for highlight detection, establishing a new SOTA.

### Ablation Study

| Configuration | Key Findings |
|------|---------|
| Font size 20 | Numbers are too small, OCR accuracy drops significantly |
| Font size 40 (Default) | Best overall trade-off for number and caption accuracy |
| Font size 80 | Highly recognizable numbers but severely blocks the scene, caption accuracy drops the most |
| Color: Red | Highest number recognition rate, consistent with "red circle" findings |
| Color: Black | Worst recognition rate against natural image backgrounds |
| Position: Bottom-right (Default) | Best balance between number and caption accuracy |
| Position: Center | Caption accuracy plummets (blocks main objects) |
| Without prompt hints | The model sometimes ignores numbers as noise |

### Key Findings
- General Vid-LLMs (such as Qwen2-VL-7B) are almost unusable for VTG (R@0.5 of only 5.4), but with NumPro, performance instantly jumps to 36.8. This indicates the primary bottleneck is not "failing to see", but "failing to name the timestamp."
- Closed-source models like GPT-4o also benefit from NumPro, proving the method is a true plug-and-play solution.
- Among the three hyperparameters—font size, color, and position—**position** exerts the most interference on the frame visual content, while **color** has the greatest impact on recognition accuracy.

## Highlights & Insights
- **Converting "linguistic temporal expression" into a "visual OCR" problem**: This represents an elegant paradigm to transfer existing VLM capabilities to new tasks at virtually zero cost.
- **CLIP-based Proxy Hyperparameter Search**: Best visual prompting parameters can be pre-determined using MSCOCO + CLIP without expensive VTG annotations. This strategy can be extended to any task involving drawing markers on images (e.g., referring expression, action counting).
- **Visual vs. Textual Prompting**: Previous works primarily injected timestamps into prompts. This paper demonstrates that rendering information onto images is much easier for VLMs to exploit, as the alignment between the visual encoder and the LLM is more robust on the visual side.
- This concept of "writing metadata at the pixel level" can be ported to downstream applications such as drawing orientation labels in 3D scenes, overlaying serial numbers for medical slice sequencing, or labeling timesteps in robotic videos.

## Limitations & Future Work
- It is only applicable to discrete frame representations where frames are already sampled and enumerable; real long-form streaming videos or high-FPS scenarios require pre-sampling frames, potentially losing temporal precision.
- Numbers are practically capped at around 99 due to font size and field of view limitations. Long videos would require segmented or hierarchical numbering.
- In the training-free mode, performance is still constrained by the base VLM's OCR capabilities at the given font size (e.g., the authors observed that LongVA underperforms Qwen2-VL).
- The comparison with non-numeric prompts (e.g., timestamp seconds, color bars) has not been explored, which remains a scalable direction for future ideas.
- Potential improvement: Render frame numbers, timestamps, and chapter IDs together to allow the model to learn hierarchical temporal structures.

## Related Work & Insights
- **vs. TimeChat / VTimeLLM**: These methods redesign models (adding temporal tokens or specialized heads), whereas ours leaves the model intact and only modifies the inputs. Ours stands out for its plug-and-play capability, though it is bottlenecked by the base OCR performance.
- **vs. SoM (Set-of-Mark)**: SoM uses numbers to mark image regions to assist referring; this work extends the same intuition from the spatial to the temporal dimension.
- **vs. ViP-LLaVA**: ViP-LLaVA uses visual prompts to hint at spatial regions; NumPro is the first to apply numeric prompts to the temporal dimension.
- **Insights**: For any task where the model "can see but cannot output", one can attempt "drawing markers on pixels"—such as marking camera IDs in multi-view 3D tasks or audio sources in audio-visual tasks.

## Rating
- Novelty: ⭐⭐⭐⭐ Simple concept but truly the first, highly transferable across tasks.
- Experimental Thoroughness: ⭐⭐⭐⭐ Evaluated across three major benchmarks, multiple base VLMs, and complete design parameter searches.
- Writing Quality: ⭐⭐⭐⭐ The manga analogy is well-explained, and the attention analysis is highly convincing.
- Value: ⭐⭐⭐⭐⭐ Training-free mode alone yields massive performance boosts. Exceptional engineering value, enabling direct integration into any off-the-shelf Vid-LLM.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] DeCafNet: Delegate and Conquer for Efficient Temporal Grounding in Long Videos](decafnet_delegate_and_conquer_for_efficient_temporal_grounding_in_long_videos.md)
- [\[CVPR 2025\] VideoGEM: Training-Free Action Grounding in Videos](videogem_training-free_action_grounding_in_videos.md)
- [\[ICCV 2025\] Moment Quantization for Video Temporal Grounding](../../ICCV2025/video_understanding/moment_quantization_for_video_temporal_grounding.md)
- [\[NeurIPS 2025\] DualGround: Structured Phrase and Sentence-Level Temporal Grounding](../../NeurIPS2025/video_understanding/dualground_phrase_temporal.md)
- [\[CVPR 2025\] Seq2Time: Sequential Knowledge Transfer for Video LLM Temporal Grounding](seq2time_sequential_knowledge_transfer_for_video_llm_temporal_grounding.md)

</div>

<!-- RELATED:END -->
