---
title: >-
  [Paper Note] VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking
description: >-
  [ICML 2026][Video Understanding][Long Video Understanding] VideoTemp-o3 is a unified Agentic video understanding framework that jointly models video temporal grounding and QA through a **unified masking strategy for cold…
tags:
  - "ICML 2026"
  - "Video Understanding"
  - "Long Video Understanding"
  - "Temporal Grounding"
  - "Agentic Thinking"
  - "Multi-turn Tool Invocation"
  - "Reward-aware RL"
date: 2026-05-08
content_hash: ef1cc42870513de1
---

# VideoTemp-o3: Harmonizing Temporal Grounding and Video Understanding in Agentic Thinking

**Conference**: ICML 2026  
**arXiv**: [2602.07801](https://arxiv.org/abs/2602.07801)  
**Code**: To be confirmed  
**Area**: Video Understanding / Agent / Multimodal VLM  
**Keywords**: Long Video Understanding, Temporal Grounding, Agentic Thinking, Multi-turn Tool Invocation, Reward-aware RL

## TL;DR
VideoTemp-o3 is a unified Agentic video understanding framework that jointly models video temporal grounding and QA through a **unified masking strategy for cold-start SFT** and **penalty-aware IoU rewards**. It achieves high-quality multi-turn iterative grounding and precise answering in long video understanding, surpassing Gemini-2.5-Pro's 14.8% mIoU with 15.6% on super-long videos (> 20 minutes).

## Background & Motivation

**Background**: In long video understanding, existing methods typically use a fixed frame sampling rate to control computational costs, which often leads to sparse sampling and missing key frames relevant to the question. Recently, the "thinking-with-videos" paradigm has emerged, borrowing ideas from thinking-with-images to adopt a "ground-crop-answer" workflow, allowing the model to actively locate relevant video segments.

**Limitations of Prior Work**: Although VideoExplorer, VITAL, and REVISOR have explored this paradigm, three key issues remain: (1) **High workflow complexity**, where multiple specialized models handle grounding and QA separately, increasing inference overhead; (2) **Low grounding accuracy**, making it difficult to locate precisely without evaluation or optimization mechanisms; (3) **Rigid processes**, as the fixed "crop once, answer immediately" pattern cannot support iterative grounding optimization in long videos.

**Key Challenge**: The primary obstacle is that current training strategies are insufficient for learning precise grounding and multi-turn iterative behaviors. Furthermore, the low quality of existing labeled data and the scarcity of long video samples mean models lack high-quality multi-turn trajectories to learn Agentic video understanding patterns.

**Goal**: To build a unified framework that optimizes temporal grounding and video QA within a single model, supporting on-demand cropping and iterative optimization, while designing specialized training strategies and data construction schemes.

**Key Insight**: This work approaches the problem from three dimensions: (1) a construction process for high-quality multi-turn data; (2) a unified masking strategy for cold-start SFT to encourage exploration while filtering noise; and (3) penalty-aware IoU rewards to prevent reward hacking.

**Core Idea**: By using a unified multi-turn dialogue framework, combined with carefully designed mask supervision and reward mechanisms, a single model learns to achieve precise grounding and accurate answering through iterative tool calls in long videos.

## Method

### Overall Architecture
The framework follows a multi-turn interaction "ground-crop-answer" process. Given a video-question pair $(V, Q)$, the model first skims the video at a low sampling rate $s_0$. It then iterates where each turn generates reasoning text $T$ and outputs either a time segment $P$ or a final answer $A$. If a time segment $P = [t_s, t_e]$ is predicted, an external cropping module extracts the corresponding segment $C = \text{Crop}(V, P, s_d)$ at a higher sampling rate $s_d > s_0$ and appends it to the context for the next turn. The interaction terminates when the model outputs an answer or reaches the maximum number of turns. Each training sample $i$ is represented as a multi-turn trajectory $\tau_i = \{(V, Q); ([T_{i,1}, P_{i,1}, C_{i,1}], \ldots, [T_{i,t}, A_i])\}$.

### Key Designs

1.  **Unified Masking Strategy**:
    - **Function**: Guides the model to learn multi-turn grounding and QA during the cold-start SFT phase while filtering noisy early grounding labels.
    - **Mechanism**: For multi-turn data, the second-to-last turn contains the correct temporal grounding, and the last turn outputs the final answer. Grounding in earlier turns is often imprecise. This method **only applies the training loss $L$ to the model outputs of the last two turns**, masking earlier generation and user inputs so they do not affect gradients.
    - **Design Motivation**: Supervising all turns traditionally leads to the model learning numerous incorrect groundings; selective supervision retains only reliable signals, improving training efficiency and robustness.

2.  **Penalty-aware IoU Reward**:
    - **Function**: Accurately measures grounding quality during the RL phase while preventing the model from "cheating" to obtain high rewards through arbitrary outputs.
    - **Mechanism**: Define IoU as $R_{\text{IoU}} = \frac{|[t_s, t_e]| \cap |[t_s', t_e']|}{|[t_s, t_e]| \cup |[t_s', t_e']|}$. A penalty $\lambda$ is applied when the IoU is below a threshold $\sigma$, such that $R_{\text{penalty-IoU}} = R_{\text{IoU}} - \lambda$ (if $R_{\text{IoU}} < \sigma$) or $R_{\text{IoU}}$ (if $\geq \sigma$). Hyperparameters are set to $\lambda = 0.1, \sigma = 0.1$.
    - **Design Motivation**: Pure IoU rewards are easily exploited, where models might output arbitrary time segments to trick the reward system. Introducing a threshold and penalty forces the model to ensure both grounding precision and rationality.

3.  **High-Quality Data Construction**:
    - **Function**: Constructs large-scale, high-quality long video grounding and QA (GQA) data, including single-turn data (no tool call) and multi-turn tool-calling data.
    - **Mechanism**: For single-turn data, Qwen3-VL-235B is used for chain-of-thought generation and answer prediction, keeping only samples where the prediction matches the ground truth. For multi-turn data, Gemini-2.5-Pro generates candidate groundings, which are verified through a two-stage process ensuring the grounding contains sufficient information to answer the question independently and maintains answer consistency across turns.
    - **Design Motivation**: Existing labels often suffer from bias and inconsistent quality, and long video samples are scarce. This process ensures high alignment between grounding and answers through rigorous model-assisted labeling and verification.

## Key Experimental Results

### Main Results

| Method | MLVU | VideoMMMU | VideoMME (w/o Sub) | LVBench | Average |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gemini-1.5-Pro | 49.3 | 53.3 | 59.0 | 33.1 | 48.7 |
| GPT-4o | 55.6 | 62.0 | 66.0 | 30.8 | 53.6 |
| Video-R1-7B | 48.0 | 46.0 | 67.3 | 40.1 | 50.4 |
| Qwen2.5-VL-7B | 45.2 | 36.1 | 57.6 | 39.2 | 44.5 |
| **Ours-7B-SFT** | 49.5 | 46.4 | 60.4 | 39.6 | 49.0 |
| **Ours-7B-RL** | **54.2** | **47.8** | **69.0** | **43.0** | **53.5** |

VideoTemp-o3-RL outperforms the best baselines on MLVU, VideoMME, and LVBench by 6.2%, 1.7%, and 2.9% respectively, with an average gain of 3.1%.

### Ablation Study

| ID | Variant | VideoMMMU | VideoMME | LVBench | ReXTime mIoU | ReXTime Acc |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| (a) | Full Model | 53.2 | 64.5 | 43.0 | 29.5 | 74.4 |
| (b) | w/o Grounding Data | 52.5 | 63.0 | 42.0 | **13.0** | 73.3 |
| (c) | w/o Unified Masking | 47.9 | 61.5 | 41.2 | 18.8 | 70.6 |
| (d) | w/o IoU Reward | 51.6 | 63.3 | 41.7 | 26.2 | 73.7 |
| (e) | w/o Penalty-aware | 44.2 | 63.7 | 40.7 | 23.8 | 73.6 |

### Performance Across Video Durations (VideoTemp-Bench)

| Method | 0-3 min | 3-10 min | 10-20 min | > 20 min | Average |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Gemini-2.5-Pro | 39.1 | 46.1 | 36.1 | 14.8 | 34.0 |
| VideoChat-R1-7B | 25.2 | 6.7 | 4.7 | 1.8 | 9.6 |
| **VideoTemp-o3-RL** | **35.3** | **32.0** | **24.8** | **15.6** | **27.0** |

The mIoU metric serves as the temporal grounding benchmark for long videos.

### Key Findings
- Removing grounding data causes mIoU to plunge from 29.5 to 13.0, proving that grounding supervision is critical for the model's internal localization capabilities.
- Removing unified masking leads to a 5.3% drop in VideoMMMU and a 10.7% drop in mIoU, validating the effectiveness of selective supervision.
- Performance collapses without the penalty-aware mechanism (mIoU 29.5 → 23.8, VideoMME 64.5 → 63.7), highlighting the necessity of preventing reward hacking.
- On super-long videos (> 20 minutes), it achieves the most stable performance with 15.6% mIoU (vs Gemini-2.5-Pro's 14.8%). Compared to baselines that collapse on > 20 min segments (mIoU < 2%), this framework demonstrates superior generalization in long videos.

## Highlights & Insights
- **Ingenuity of Unified Architecture**: By unifying temporal grounding and video QA within a single model through a shared representation space and consistent dialogue format, the model optimizes both tasks simultaneously. This reduces inference latency compared to multi-module serial designs and improves performance via task orthogonality.
- **Effectiveness of Selective Supervision**: The unified masking strategy applies losses only to the final turns, shielding the model from early noisy groundings. This balances efficacy and robustness in multi-turn trajectory learning; this technique is transferable to other multi-step reasoning tasks.
- **Reward Design Safeguards**: The penalty-aware IoU reward prevents blind guessing via explicit penalties, avoiding common reward hacking issues in RL. This constrained design is a valuable reference for other long video tasks like Temporal Action Localization or Event Detection.
- **Data-Quality-First Practice**: Strict multi-stage verification ensures high alignment between grounding and answers in GQA data. Compared to using low-quality annotations directly, this investment yields significant performance gains.

## Limitations & Future Work
- The framework is designed for specific multi-turn formats; generalization to ultra-long videos (> 60 min) with many interaction rounds or complex reasoning paths has not been fully explored.
- The upper limit of temporal precision is restricted by video frame rates and the discreteness of the cropping stage, making sub-second grounding difficult.
- The data construction process relies on high-quality VL models (Gemini-2.5) for labeling, which may limit feasibility in other domains or low-resource languages.
- **Future Work**: Explore continuous temporal grounding representations instead of discrete segments; introduce more flexible turn-limit strategies; design lightweight data labeling schemes to reduce dependence on high-end VL models.

## Related Work & Insights
- **vs VideoExplorer**: VideoExplorer uses multi-agent collaboration (separate planner/grounder/understander), whereas this work integrates them into a single model to reduce inference complexity and gain flexible iterative refinement via end-to-end training.
- **vs VITAL / REVISOR**: These also use a two-stage SFT-RL approach but lack explicit handling of multi-turn noise and anti-hacking reward designs. VideoTemp-o3's unified masking and penalty-aware rewards further stabilize multi-turn learning.
- **vs LongVT**: LongVT proposes a three-stage SFT-RL-RFT strategy. This work achieves similar or superior performance through a more compact SFT-RL framework and high-quality data construction, suggesting that data quality and training strategy design may be more localized than simply increasing training stages.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The combination of a unified framework, penalty-aware rewards, and a high-quality data pipeline is pioneering; the reward design is insightful for the RL community.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers long video understanding, temporal grounding, and video GQA, including a new benchmark (VideoTemp-Bench) with detailed ablations and duration-based analysis.
- Writing Quality: ⭐⭐⭐⭐ The methodology is clear and data processes are intuitive, though the theoretical motivation for reward designs could be further detailed.
- Value: ⭐⭐⭐⭐⭐ Establishes an efficient and reliable paradigm for Agentic long video understanding; reward and masking strategies have clear reuse value; the new benchmark provides a standard for future evaluations.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICCV 2025\] VTimeCoT: Thinking by Drawing for Video Temporal Grounding and Reasoning](../../ICCV2025/video_understanding/vtimecot_thinking_by_drawing_for_video_temporal_grounding_and_reasoning.md)
- [\[ICML 2026\] VideoSEAL: Mitigating Evidence Misalignment in Agentic Long Video Understanding by Decoupling Answer Authority](videoseal_mitigating_evidence_misalignment_in_agentic_long_video_understanding_b.md)
- [\[ICML 2026\] Video-MTR: Reinforced Multi-Turn Reasoning for Long Video Understanding](video-mtr_reinforced_multi-turn_reasoning_for_long_video_understanding.md)
- [\[CVPR 2026\] LensWalk: Agentic Video Understanding by Planning How You See in Videos](../../CVPR2026/video_understanding/lenswalk_agentic_video_understanding_by_planning_how_you_see_in_videos.md)
- [\[ICML 2026\] MetaphorVU: Towards Metaphorical Video Understanding](metaphorvu_towards_metaphorical_video_understanding.md)

</div>

<!-- RELATED:END -->
