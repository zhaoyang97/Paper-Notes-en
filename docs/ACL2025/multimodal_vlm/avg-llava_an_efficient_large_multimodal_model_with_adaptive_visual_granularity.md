---
title: >-
  [Paper Note] AVG-LLaVA: An Efficient Large Multimodal Model with Adaptive Visual Granularity
description: >-
  [ACL 2025][Multimodal VLM][Adaptive Visual Granularity] This work integrates a visual granularity scaler (obtaining multi-scale granularity tokens via spatial pyramid pooling) and a visual granularity router (adaptively selecting granularity based on the image + instruction) into LLaVA-NeXT. It also introduces the RGLF training paradigm, which utilizes the LMM's own generation probabilities as feedback to train the router, achieving the effect of "reducing tokens while improv…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Adaptive Visual Granularity"
  - "Visual Token Compression"
  - "MoE Routing"
  - "RGLF Training"
  - "LLaVA-NeXT"
date: 2026-05-08
content_hash: 875332850fb1aab8
---

# AVG-LLaVA: An Efficient Large Multimodal Model with Adaptive Visual Granularity

**Conference**: ACL 2025  
**arXiv**: [2410.02745](https://arxiv.org/abs/2410.02745)  
**Code**: [GitHub](https://github.com/DeepLearnXMU/AVG-LLaVA)  
**Area**: Multimodal VLM  
**Keywords**: Adaptive Visual Granularity, Visual Token Compression, MoE Routing, RGLF Training, LLaVA-NeXT

## TL;DR

This work integrates a visual granularity scaler (obtaining multi-scale granularity tokens via spatial pyramid pooling) and a visual granularity router (adaptively selecting granularity based on the image + instruction) into LLaVA-NeXT. It also introduces the RGLF training paradigm, which utilizes the LMM's own generation probabilities as feedback to train the router, achieving the effect of "reducing tokens while improving performance" across 11 benchmarks.

## Background & Motivation

**Background**: High-resolution LMMs (such as LLaVA-NeXT, Monkey, etc.) typically partition high-resolution images into multiple local image patches for separate encoding, which are then concatenated with global image tokens before being fed into the LLM. Although this approach enhances fine-grained perception capability, it comes at the cost of a catastrophic explosion in the number of visual tokens—for instance, a 672×672 image in LLaVA-NeXT generates 2,880 visual tokens.

**Limitations of Prior Work**: These visual tokens contain significant redundancy. Recognizing a jersey number requires fine-grained tokens, while asking about the jersey color only requires coarse-grained ones. Existing methods employ the same number of visual tokens for all image-question pairs, which not only wastes computational resources but also introduces redundant information that can distract LLM reasoning. Existing token compression methods (such as FastV pruning or LLaVA-PruMerge merging) often cause performance degradation when reducing token counts. Although LLaVA-M3 supports multi-granularity, it requires manual specification of the granularity level.

**Key Challenge**: The mismatch between the number of visual tokens and task requirements—fixed granularity strategies fail to simultaneously satisfy the twin requirements of "using fewer tokens for simple questions to accelerate inference" and "retaining fine-grained information for complex questions."

**Goal**: (1) How to automatically select the appropriate visual granularity based on the input image and user instructions? (2) How to effectively train this granularity selector—given that direct training via visual instruction tuning fails to acquire granularity differentiation capability?

**Key Insight**: Draw inspiration from human visual cognition—"gaze closely at difficult problems, glance at simple ones." Visual features of different granularities are analogized to different experts in a Mixture-of-Experts (MoE) system, and a router is used to dynamically select the most suitable "expert" (level of granularity) based on the image and instruction.

**Core Idea**: Employ an MoE-style router to adaptively select visual granularity based on the image + instruction, and train the router's granularity preference ranking using the generation probability feedback of the LMM itself.

## Method

### Overall Architecture

AVG-LLaVA introduces two new modules on top of LLaVA-NeXT: (1) a Visual Granularity Scaler that obtains multi-level granularity visual tokens through multi-stage pooling, and (2) a Visual Granularity Router that adaptively selects the most appropriate granularity based on the image and instructions. The overall pipeline is: Image $\rightarrow$ Visual Encoder (CLIP ViT-L/14) $\rightarrow$ Visual Granularity Scaler (generating 5 levels of granularity tokens) $\rightarrow$ Visual Granularity Router (selecting 1 granularity level) $\rightarrow$ the tokens of the selected granularity are fed into the LLM to generate the response. The training consists of two stages: first enabling the model to process multi-granularity visual information, followed by RGLF training for the router.

### Key Designs

1. **Visual Granularity Scaler**:

    - **Function**: Transforms the original visual tokens into token sequences of multiple granularity levels.
    - **Mechanism**: Employs a spatial pyramid pooling design that alternately stacks $1 \times 2$ and $2 \times 1$ average pooling layers. Taking CLIP ViT-L/14 (which outputs a $24 \times 24$ grid) as an example, it sequentially obtains four coarser granularity levels: $24 \times 12$ (288 tokens), $12 \times 12$ (144 tokens), $12 \times 6$ (72 tokens), and $6 \times 6$ (36 tokens). Together with the original $24 \times 24$ (576 tokens), this yields 5 granularity levels in total. This alternating pooling method halves the token count while preserving spatial layout information as much as possible.
    - **Design Motivation**: This is a training-free and purely operational module. It naturally leverages a pyramid structure to obtain coarse-to-fine multi-level visual representations, providing candidate granularities for the subsequent router.

2. **Visual Granularity Router**:

    - **Function**: Selects the most appropriate granularity level from the 5 candidates based on the input image and user instructions.
    - **Mechanism**: Inspired by MoE, different granularities of visual features are treated as different experts. The router consists of a three-layer architecture. First, all granularities of visual tokens are flattened and concatenated as $\bar{X}_v = [X_v^1; X_v^2; \ldots; X_v^N]$. Concurrently, the cosine similarity between instruction tokens and the original granularity visual tokens is computed to retain the top-$k$ ($k=32$) most relevant instruction tokens $\bar{X}_{\text{instruct}}$. Then, the visual tokens and the filtered instruction tokens are concatenated and fed into a Transformer layer for cross-modal fusion. Next, an MLP predicts the granularity logits $Z_{\text{out}} \in \mathbb{R}^{L \times N}$ for each token. Finally, a learnable weight matrix (Voter) $W \in \mathbb{R}^{1 \times L}$ performs a weighted aggregation over all token predictions to obtain the final logits $Z_{\text{final}} \in \mathbb{R}^{1 \times N}$, which are then passed through softmax to select the granularity with the highest probability.
    - **Design Motivation**: Unlike traditional MoE which uses a simple linear layer for routing, this module needs to consider both image content and instruction semantics simultaneously, prompting the use of a Transformer layer for cross-modal fusion. The top-$k$ filtering of instruction tokens aims to filter out noise (too few or too many instruction tokens both hamper performance; experiments show that $k=32$ is optimal). The Voter layer's design allows each token's prediction to be weighted by its learned importance, rather than performing a simple average.

3. **RGLF Training Paradigm** (Ranking Granularity based on LMM Feedback):

    - **Function**: Effectively trains the router to distinguish the pros and cons of different granularities and select the most appropriate one.
    - **Mechanism**: Freeze all modules except the router. For each training sample, feed the visual tokens of 5 different granularities into the LMM to generate answers, and calculate the log probability of the answers under each granularity as feedback signals. Sort the granularities in descending order based on the log probabilities, and then use the ranking loss $\mathcal{L}_{\text{rank}} = \sum_{i} \sum_{j>i} \max(0, s_j - s_i + \lambda_{ij})$ to align the router's probabilities with the LMM's preferences, where the margin $\lambda_{ij}$ is dynamically adjusted based on the log-probability differences between two granularities. Additionally, a cross-entropy loss $\mathcal{L}_{\text{ce}}$ is incorporated to guide the router to directly predict the LMM's most preferred granularity. The overall loss is $\mathcal{L}_2 = \mathcal{L}_{\text{rank}} + \alpha \mathcal{L}_{\text{ce}}$, with $\alpha = 0.1$.
    - **Design Motivation**: Directly training the router using visual instruction tuning (with backpropagation via Gumbel-Softmax) yields poor results because the router cannot learn the distinction between various granularities from end-to-end training. The key insight of RGLF is that the LMM itself already knows which granularity is more suitable for the current sample (revealed by its generation probability); hence, the LMM's feedback can serve as a "supervisory signal" to train the router. The ranking loss ensures the router learns the relative preference order of different granularities, while the cross-entropy loss ensures the optimal granularity is selected, making them complementary to each other.

### Loss & Training

**Stage 1 - Multi-Granularity Visual Instruction Tuning**: Optimize the visual encoder, connector, and LLM using 1M image-text pairs. For each sample, next-token prediction is performed separately using $N$ granularities of visual tokens. The loss function is defined as the mean cross-entropy over all granularities: $\mathcal{L}_1 = -\frac{1}{N} \sum_{i=1}^{N} \sum_{t=1}^{T} \log P(x_t | X_v^i, X_{\text{instruct}}, X_{a,<t})$. This stage enables the model to comprehend images under various granularities.

**Stage 2 - RGLF Router Training**: Freeze other modules and train only the router. Utilizing the same 1M dataset, the learning rate is set to 1e-3 (much higher than Stage 1's 1e-5). The total training time is approximately 14 hours (8×H800), which is only about 1/5 of Stage 1 (65 hours).

## Key Experimental Results

### Main Results

| Benchmark | Type | LLaVA-NeXT | AVG-LLaVA | Gain |
|------|------|-----------|-----------|------|
| GQA | General VQA | 64.2 | 63.0 | -1.2 |
| ScienceQA | General VQA | 70.1 | 71.1 | +1.0 |
| VizWiz | General VQA | 57.6 | 59.8 | +2.2 |
| TextVQA | Text VQA | 64.9 | 67.1 | +2.2 |
| ChartQA | Text VQA | 54.8 | 66.3 | **+11.5** |
| DocVQA | Text VQA | 74.4 | 74.6 | +0.2 |
| AI2D | Text VQA | 66.6 | 67.3 | +0.7 |
| MME | Multimodal | 1519.0 | 1557.4 | +38.4 |
| MMB | Multimodal | 67.4 | 69.9 | **+2.5** |
| POPE | Multimodal | 86.5 | 87.4 | +0.9 |
| MMMU | Multimodal | 35.8 | 37.4 | +1.6 |

Efficiency comparison (AVG-LLaVA vs LLaVA-NeXT):

| Benchmark | Token Reduction Ratio | Inference Speedup |
|------|-------------|---------|
| AI2D | **85.3%** | **2.53×** |
| MME | 69.3% | 1.19× |
| ScienceQA | 54.9% | 1.41× |
| GQA | 80.0% | 1.14× |
| VizWiz | 26.4% | 1.77× |
| MMMU | 30.0% | 1.87× |

### Ablation Study

| Configuration | ScienceQA | ChartQA | MME | MMB |
|------|-----------|---------|-----|-----|
| AVG-LLaVA (Full) | 71.1 | 66.3 | 1557.4 | 69.9 |
| Fixed granularity instead of adaptive | 70.0 | 66.4 | 1554.5 | 68.7 |
| Random selection instead of router | 69.7 | 56.8 | 1535.7 | 67.9 |
| Router using image only (w/o instruction) | 70.1 | 53.9 | 1525.2 | 69.0|
| Reducing granularity range {36,144,576} | 69.8 | 65.3 | 1547.7 | 66.3 |
| Instruction tuning to train router | 70.5 | 50.9 | 1514.8 | 68.6 |
| W/o ranking loss | 70.1 | 64.8 | 1534.6 | 68.6 |
| W/o cross-entropy loss | 70.2 | 66.3 | 1550.8 | 69.4 |

### Key Findings

- **Instruction information is crucial for routing**: Discarding instruction tokens causes ChartQA to plummet by 12.4 points, demonstrating that the same image indeed requires different granularities for different questions.
- **RGLF is vastly superior to end-to-end training**: Training the router via visual instruction tuning with Gumbel-Softmax achieves a ChartQA score of only 50.9 (15.4 points lower than RGLF), indicating that the router struggles to learn granularity discrimination through end-to-end gradients.
- **Ranking loss is more critical than cross-entropy loss**: Removing the ranking loss leads to a 22.8 drop on MME, whereas removing the cross-entropy loss only results in a 6.6 drop. The ranking loss provides relative preference signals among granularities, while relying solely on CE learns the "optimal granularity" but lacks global ranking capability.
- **Intermediate granularities are rarely selected but essential**: Although the 72 and 288 token granularities are extremely rarely selected, removing them degrades performance, indicating that they help the model progressively learn the gaps between different granularities.
- **Text-dense tasks prefer fine granularity**: In TextVQA/ChartQA/DocVQA, the router predominantly chooses the finest granularity (576 tokens), whereas conceptual tasks like AI2D/MMMU favor coarse granularity (36 tokens), showing intuitive behavior.

## Highlights & Insights

- **Reducing redundant tokens can surprisingly improve performance**: This is the most counter-intuitive finding. AVG-LLaVA outperforms LLaVA-NeXT on AI2D using only 14.7% of the tokens, indicating that too many visual tokens can actually introduce noise that interferes with LLM reasoning. This finding has significant guiding value for MLLM efficiency research.
- **RGLF training paradigm is the core contribution**: Utilizing the LMM's own generation probability as a ranking signal to train the router ingeniously bypasses the need for manual annotation of the ideal granularity for each sample. This concept of "using the model's own feedback to train auxiliary modules" can be transferred to other discrete selection scenarios (e.g., MoE expert routing, document selection in retrieval-augmented generation).
- **Dynamic margin design**: The margin $\lambda_{ij}$ in the ranking loss is not fixed but dynamically calculated based on the log-probability differences of the LMM across different granularities. This makes the penalty proportional to the performance gap, preventing excessive separation force on granularity pairs with similar capabilities.

## Limitations & Future Work

- **Excessive gaps between granularity steps**: Each pooling operation directly halves the token count, causing large gaps between adjacent granularities (576 $\rightarrow$ 288 $\rightarrow$ 144 $\rightarrow$ 72 $\rightarrow$ 36). For text-dense tasks, the router exclusively chooses 576 tokens, indicating a lack of sufficiently fine intermediate granularities. Designing a smoother granularity scaling network (e.g., variable-stride pooling) may prove beneficial.
- **Two-stage training overhead**: Stage 1 requires running forward passes with all 5 granularities for each sample, scaling the training cost to approximately 5 times that of single-granularity training. While Stage 2 is fast (14 hours), whether the two stages can be merged into alternating training warrants exploration.
- **Validation limited to LLaVA-NeXT architecture**: Can this be generalized to other architectures such as InternVL or Qwen-VL? Does the router design need modification for different architectures?
- **Small router parameter size but additional inference overhead**: Although the router only introduces 1.66% more parameters, all granularities of tokens must be generated before selecting one, meaning that the scaling of all granularities still needs to be executed during inference.

## Related Work & Insights

- **vs FastV/VTW (token pruning)**: FastV prunes 50% of tokens based on attention scores after the 2nd decoder layer, and VTW aggressively discards all visual tokens after specific layers. These methods prune after encoding, risking loss of crucial information. In contrast, AVG-LLaVA selects the granularity before encoding, thus retaining the complete spatial structure of the chosen granularity.
- **vs LLaVA-PruMerge (pruning + merging)**: PruMerge determines which tokens to discard or merge based on class token similarity, which is a static strategy that ignores instructions. The router in AVG-LLaVA evaluates both the image and instruction, dynamically adjusting based on the prompt.
- **vs LLaVA-M3 / MQT-LLaVA (multi-granularity)**: LLaVA-M3 supports multi-granularity but requires users to manually specify it; MQT-LLaVA employs nested dropout to train compatibility for multiple granularities. The advantage of AVG-LLaVA lies in its automated selection and alignment via RGLF training, requiring no human intervention.
- **Analogy between RGLF and RLHF**: RGLF essentially obtains preference signals from the LMM to train the router, akin to training reward models from human preference in RLHF. The difference is that RGLF's preference signals are objective log probabilities rather than subjective annotations, leading to higher stability.

## Rating

- Novelty: ⭐⭐⭐⭐ The adaptive granularity selection idea is intuitive yet effective, and the RGLF training paradigm is a highlight. However, the granularity scaler itself is relatively basic.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 11 benchmarks are comprehensively covered, ablation studies span 7 dimensions, and routing visualization as well as attention map analyses are in-depth.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation, well-structured method descriptions, and rich architectural diagrams and visualizations.
- Value: ⭐⭐⭐⭐ Directly practical for MLLM inference efficiency with a highly transferable RGLF paradigm; however, it has only been validated at the 7B scale and lacks validation on larger models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model](hidellava_hierarchical_decoupling_for_continual_instruction.md)
- [\[ACL 2025\] Error-driven Data-efficient Large Multimodal Model Tuning](error-driven_data-efficient_large_multimodal_model_tuning.md)
- [\[ACL 2025\] Adaptive Linguistic Prompting (ALP) Enhances Phishing Webpage Detection in Multimodal Large Language Models](adaptive_linguistic_prompting_alp_enhances_phishing_webpage_detection_in_multimo.md)
- [\[CVPR 2026\] Granulon: Awakening Pixel-Level Visual Encoders with Adaptive Multi-Granularity Semantics for MLLM](../../CVPR2026/multimodal_vlm/granulon_awakening_pixel-level_visual_encoders_with_adaptive_multi-granularity_s.md)
- [\[ICCV 2025\] LLaVA-KD: A Framework of Distilling Multimodal Large Language Models](../../ICCV2025/multimodal_vlm/llava-kd_a_framework_of_distilling_multimodal_large_language_models.md)

</div>

<!-- RELATED:END -->
