---
title: >-
  [Paper Note] VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning
description: >-
  [NeurIPS 2025][Multimodal VLM][video reasoning] This paper proposes VideoRFT, which extends the reinforced fine-tuning (RFT) paradigm to video reasoning via a cognition-inspired multi-expert CoT data construction pipeline and a novel semantic consistency reward. Two datasets are constructed: VideoRFT-CoT-102K (for SFT) and VideoRFT-RL-310K (for RL), achieving state-of-the-art performance on 6 video reasoning benchmarks.
tags:
  - NeurIPS 2025
  - Multimodal VLM
  - video reasoning
  - reinforced fine-tuning
  - chain-of-thought
  - multimodal large language models
  - semantic consistency reward
date: 2026-05-08
content_hash: 805aa9db777ddd8e
---

# VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2505.12434](https://arxiv.org/abs/2505.12434)
**Code**: [GitHub](https://github.com/QiWang98/VideoRFT)
**Area**: Multimodal VLM
**Keywords**: video reasoning, reinforced fine-tuning, chain-of-thought, multimodal large language models, semantic consistency reward

## TL;DR

This paper proposes VideoRFT, which extends the reinforced fine-tuning (RFT) paradigm to video reasoning via a cognition-inspired multi-expert CoT data construction pipeline and a novel semantic consistency reward. Two datasets are constructed: VideoRFT-CoT-102K (for SFT) and VideoRFT-RL-310K (for RL), achieving state-of-the-art performance on 6 video reasoning benchmarks.

## Background & Motivation

Multimodal large language models (MLLMs) have achieved notable progress on video understanding tasks; however, existing models are primarily "answer-driven"—they produce answers directly without exposing the underlying reasoning process. With the success of "think-before-answer" models such as OpenAI-o1 and DeepSeek-R1, RFT has demonstrated promise in the image domain (Visual-RFT, Vision-R1, etc.), yet its application to video faces three core challenges:

**Scarcity of high-quality video CoT data**: The temporal complexity and causal structure of video make generating high-quality reasoning chains substantially harder than for images. Existing approaches such as Video-R1 merely trigger CoT by inserting surface-level phrases like "let me think" into responses, lacking genuine deep reasoning. VideoEspresso prompts GPT-4o with sparse keyframes to generate CoT, but sparse visual context is prone to hallucination.

**Lack of visual grounding in reasoning**: Reward functions in existing RFT methods (format reward + accuracy reward) cannot ensure that reasoning outputs are faithful to visual evidence—models may produce reasoning text that appears plausible yet is visually inconsistent.

**Absence of cognitive process**: Template-based reasoning (e.g., VoT's five-step fixed pipeline) conflicts with the flexible nature of human cognition, which adapts reasoning dynamically based on perceptual input.

## Method

### Overall Architecture

VideoRFT follows a standard two-stage RFT scheme:
- **Stage 1 (SFT warm-up)**: Supervised fine-tuning on VideoRFT-CoT-102K to teach the model structured reasoning.
- **Stage 2 (RL reinforcement)**: Reinforcement learning on VideoRFT-RL-310K using a triple reward function (format reward + accuracy reward + semantic consistency reward) via the GRPO algorithm.

### Key Designs

1. **Cognition-inspired CoT generation pipeline**: High-quality video CoT data is constructed in three steps.

   **Step 1: Structured video representation.** GPT-4o-mini is used to generate structured textual descriptions from uniformly sampled frames of each video, including a high-level summary and per-frame JSON-format metadata (timestamp annotations and key visual elements such as objects, actions, scenes, spatial relations, and potential interactions).

   **Step 2: Cognition-inspired blind-reasoning CoT generation.** The structured representation $S_v$ and question $q$ are fed into a reasoning LLM (e.g., DeepSeek-R1), which generates an initial CoT via a carefully designed composite prompt $P_{\text{cog}} = [p_s, p_t, p_a, p_v, p_r]$:
   $$\text{CoT}_v^{(0)} = \text{LLM}(q, S_v, P_{\text{cog}})$$
   The five sub-prompts simulate human cognitive processes: (i) simulated viewing—forming a holistic understanding; (ii) task comprehension—inferring the question type; (iii) selective focus—localizing relevant temporal segments; (iv) visual reasoning—analysis based on objects, actions, and spatiotemporal relations; (v) reflective answering—deriving and self-verifying the answer.

   **Step 3: Cross-modal CoT refinement.** Since the initial CoT is generated solely from textual descriptions, it may contain visual hallucinations. Qwen2.5-VL is used to process both the raw video and the initial CoT together, identifying and correcting visual-textual inconsistencies via a contrastive prompt $P_{\text{cross}}$:
   $$\text{CoT}_v = \text{MLLM}(v, \text{CoT}_v^{(0)}, P_{\text{cross}})$$
   Samples are then filtered by answer correctness (structured tasks) and semantic consistency (open-ended tasks, CLIP score), yielding 102K high-confidence samples from the original 310K.

2. **Semantic consistency reward**: The key observation is that MLLM reasoning traces typically consist of three components: question parsing, video describing, and abstract reasoning. The video describing segment should be aligned with actual video content.

   Reward computation: A regular expression is used to locate $M$ tokens following the first period in the reasoning text as the video description segment $t_{[i,i+M]}$, whose representation is obtained via SigLIP's text encoder; $F$ frames are uniformly sampled and encoded by SigLIP's image encoder to obtain the average video representation $\boldsymbol{v}$. The semantic consistency reward is:

   $$R_s = \min(1, w \times \max(\cos(\boldsymbol{t}_{[i,i+M]}, \boldsymbol{v}), 0))$$

   where $w=2$ is a scaling constant. $\max(\cdot, 0)$ ensures non-negativity and $\min(\cdot, 1)$ bounds the reward for training stability. This reward is activated only when $R_a > 0$, preventing reinforcement of semantically plausible but factually incorrect reasoning.

3. **GRPO training framework**: VideoRFT employs GRPO (Group Relative Policy Optimization) as the RL algorithm. For each query $q$, $K$ candidate responses are generated; after reward evaluation, group-relative advantages are computed as $A_i = \frac{r_i - \text{mean}(\{r_k\})}{\text{std}(\{r_k\})}$, and the policy is updated via a clipped objective with KL regularization. The total reward is $R = R_f + R_a + \mathbb{1}[R_a > 0] \cdot R_s$.

### Loss & Training

- Base model: Qwen2.5-VL-7B
- Hardware: 8× NVIDIA A800 (80 GB)
- Training input: 16 frames at resolution $128 \times 28 \times 28$
- Inference input: 32 frames at resolution $256 \times 28 \times 28$
- SFT stage: 1 epoch; RL stage: 1K steps
- Semantic consistency reward computed using lightweight SigLIP (400M)

## Key Experimental Results

### Main Results

**Accuracy (%) on 6 video reasoning/understanding benchmarks**

| Model | VSI-Bench | VideoMMMU | MMVU | MVBench | TempCompass | VideoMME |
|-------|-----------|-----------|------|---------|-------------|---------|
| GPT-4o | 34.0 | 61.2 | 75.4 | - | - | 71.9 |
| Qwen2.5-VL-7B (base) | 31.8 | 47.4 | 61.3 | 59.4 | 69.2 | 52.8 |
| Video-R1 | 35.8 | 52.3 | 63.8 | 63.9 | 73.2 | 59.3 |
| LLaVA-OneVision-7B | 32.4 | 33.8 | 49.2 | 56.7 | - | 58.2 |
| **VideoRFT** | **36.8** | 51.1 | **68.5** | 62.1 | **73.7** | **59.8** |

**Gain over base model Qwen2.5-VL-7B**

| Metric | VSI-Bench | VideoMMMU | MMVU | MVBench | TempCompass | VideoMME |
|--------|-----------|-----------|------|---------|-------------|---------|
| Gain | +5.0 | +3.7 | **+7.2** | +2.7 | +4.5 | **+7.0** |

### Ablation Study

| Configuration | VSI-Bench | VideoMMMU | MMVU | MVBench | TempCompass | VideoMME |
|---------------|-----------|-----------|------|---------|-------------|---------|
| w/o CoT refinement | 34.5 | 48.1 | 64.8 | 58.3 | 72.4 | 52.8 |
| SFT only | 31.7 | 48.5 | 60.5 | 57.0 | 68.4 | 54.1 |
| RL only | 32.1 | 47.4 | 63.5 | 59.2 | 70.8 | 51.9 |
| $R=R_f+R_a$ | 33.2 | 49.1 | 66.4 | 61.1 | 72.4 | 58.5 |
| $R=R_f+R_a+R_s$ (no gating) | 34.6 | 50.2 | 65.2 | 61.4 | 73.9 | 56.3 |
| **VideoRFT (full)** | **36.8** | **51.1** | **68.5** | **62.1** | **73.7** | **59.8** |

### Key Findings

- **Cross-modal CoT refinement is critical**: Removing refinement leads to −3.7% on MMVU and −7.0% on VideoMME, demonstrating that visual hallucinations in blind-reasoning CoT are a serious problem.
- **SFT and RL are complementary**: RL-only outperforms SFT-only on 4 out of 6 datasets (indicating stronger generalization from RL), while SFT+RL achieves the best results on all 6 (SFT provides a stable initialization).
- **Gated activation of the semantic consistency reward is effective**: Adding $R_s$ without gating actually degrades performance on MMVU and VideoMME, as it reinforces reasoning that is semantically plausible but factually incorrect. Gating via $\mathbb{1}[R_a > 0]$ restricts visual grounding incentives to correct answers only.
- **Aha Moment emerges**: VideoRFT exhibits human-like self-reflection behaviors such as "wait, let me check again," indicating that RL training genuinely induces an internal feedback loop rather than simple pattern matching.
- **Surpassing GPT-4o**: VideoRFT exceeds GPT-4o by 2.8% on VSI-Bench, demonstrating that a 7B model trained with RFT can approach or surpass closed-source large models on spatial reasoning tasks.

## Highlights & Insights

- The "blind reasoning first, then visual calibration" approach in the CoT construction pipeline is worth adopting: it leverages the strong reasoning capability of reasoning LLMs to generate initial CoT, then corrects hallucinations using the visual capability of MLLMs, achieving effective complementarity.
- The design philosophy of the semantic consistency reward—**selective alignment**—is particularly elegant: consistency is computed only for the portion of reasoning that directly corresponds to visual content, neither constraining abstract reasoning nor penalizing question parsing, requiring only that the video description segment be faithful.
- The comparative analysis of CoT data is compelling: the token distribution in VideoRFT-CoT-102K is longer and more dynamic than that in Video-R1, with high-frequency words including temporally indicative terms such as "video," "happen," and "first," reflecting deeper video understanding.

## Limitations & Future Work

- The semantic consistency reward depends on the quality of SigLIP's text-visual alignment and may be less effective in domains poorly covered by SigLIP (e.g., scientific diagrams).
- Locating the "video description segment" via regular expressions is overly heuristic—assuming the video description begins after the first period, which may fail for irregular reasoning outputs.
- Different frame counts and resolutions are used during training and inference (16 frames → 32 frames, 128 → 256 patches), introducing inconsistencies whose impact is not thoroughly analyzed.
- The filtering rate for the 102K CoT subset (310K → 102K) is approximately 67%, discarding a large amount of data; pipeline efficiency could be improved.
- No comparison with other RL algorithms (e.g., PPO) is provided; the choice of GRPO is justified only by reference to DeepSeek-R1.

## Related Work & Insights

- The core distinction from Video-R1 lies in data quality: Video-R1 triggers CoT with simple prompt words, whereas VideoRFT constructs high-quality CoT via a multi-expert cognitive pipeline. This data quality difference is directly reflected in performance.
- Compared to image-domain RFT works such as Vision-R1 and R1-OneVision, VideoRFT's unique contribution is the semantic consistency reward—explicit visual grounding rewards are generally unnecessary in the image domain.
- The cognition-inspired prompting strategy (simulated viewing → task comprehension → selective focus → visual reasoning → reflective answering) is transferable to other multimodal tasks requiring structured reasoning.

## Rating

- Novelty: ⭐⭐⭐⭐ Both the CoT construction pipeline and the semantic consistency reward represent substantive innovations.
- Experimental Thoroughness: ⭐⭐⭐⭐ Six benchmarks and complete ablations, though comparisons with additional RL algorithms are absent.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with intuitive pipeline diagrams.
- Value: ⭐⭐⭐⭐⭐ The high-quality CoT dataset and visual grounding reward have broad applicability to video reasoning MLLMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[NeurIPS 2025\] To Think or Not To Think: A Study of Explicit Thinking in Rule-Based Visual Reinforcement Fine-Tuning](think_or_not_think_a_study_of_explicit_thinking_in_rule-based_visual_reinforceme.md)
- [\[ICLR 2026\] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models](../../ICLR2026/multimodal_vlm/vision-r1_incentivizing_reasoning_capability_in_multimodal_large_language_models.md)
- [\[NeurIPS 2025\] Struct2D: A Perception-Guided Framework for Spatial Reasoning in MLLMs](struct2d_a_perception-guided_framework_for_spatial_reasoning_in_mllms.md)
- [\[NeurIPS 2025\] Uni-MuMER: Unified Multi-Task Fine-Tuning of Vision-Language Model for Handwritten Mathematical Expression Recognition](uni-mumer_unified_multi-task_fine-tuning_of_vision-language_model_for_handwritte.md)

</div>

<!-- RELATED:END -->
