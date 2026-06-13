---
title: >-
  [Paper Note] ReGATE: Learning Faster and Better with Fewer Tokens in MLLMs
description: >-
  [ACL 2026][Multimodal VLM][Multimodal Large Language Models] ReGATE uses a frozen text-only teacher to estimate which output tokens require visual information…
tags:
  - "ACL 2026"
  - "Multimodal VLM"
  - "Multimodal Large Language Models"
  - "Training Acceleration"
  - "Token Elision"
  - "Teacher-Student"
  - "Sparse Computing"
date: 2026-05-08
content_hash: bf809b31d33cf299
---

# ReGATE: Learning Faster and Better with Fewer Tokens in MLLMs

**Conference**: ACL 2026  
**arXiv**: [2507.21420](https://arxiv.org/abs/2507.21420)  
**Code**: https://people-robots.github.io/regate (Project Page)  
**Area**: Multimodal VLM / Training Acceleration / Token Pruning  
**Keywords**: Multimodal Large Language Models, Training Acceleration, Token Elision, Teacher-Student, Sparse Computing

## TL;DR
ReGATE uses a frozen text-only teacher to estimate which output tokens require visual information, combined with the student's historical learning difficulty to dynamically select training tokens. This allows MLLMs to train faster with fewer tokens without changing the architecture or adding parameters, matching or exceeding standard fine-tuning on multiple image and video benchmarks.

## Background & Motivation
**Background**: The training cost of MLLMs is primarily driven by long-sequence self-attention and large-scale visual inputs. This is particularly evident in video tasks, where multiple frames are expanded into a massive number of visual tokens, which then enter the LLM backbone along with instruction and answer tokens, making each forward/backward pass expensive.

**Limitations of Prior Work**: Most token reduction, merging, and compression methods target the inference stage. While they speed up generation in trained models, they do not reduce the tokens that must be processed during training. Existing training-time acceleration methods either originate from text-only LMs or rely on extra modules, specific visual token scorers, or heuristic pruning, making them difficult to transfer seamlessly across different MLLMs.

**Key Challenge**: Not all tokens are equally worth calculating during training. Functional words, template words, or tokens directly predictable from textual context provide limited contribution to multimodal grounding. However, simply pruning tokens based on naive rules might accidentally remove color, action, object attributes, and temporal cues that require visual evidence.

**Goal**: To construct a training-time token elision framework that identifies critical tokens requiring visual grounding and dynamically adjusts the computation budget based on the student's current learning state, without changing model architecture or adding trainable parameters.

**Key Insight**: The authors use "whether the text context can predict the token" as a proxy signal for visual dependency. If a frozen text-only teacher can easily predict a token even after visual input is masked, the marginal value of that token for multimodal training may be low; conversely, a high reference loss often indicates a need for visual information.

**Core Idea**: Use the reference loss of a text-only teacher to represent visual necessity and the EMA loss of the student to represent current learning difficulty. By summing these, high-scoring tokens are selected for sparse computation, concentrating training capacity on tokens that are "both visual-dependent and still difficult to learn."

## Method
ReGATE stands for Reference-Guided Adaptive Token Elision. Instead of crudely deleting visual tokens, it calculates an importance mask for output tokens during training and allows the transformer decoder to execute primary calculations only for selected tokens. Inactive tokens do not participate in attention and MLP re-computation but retain their original representations through residual connections. This maintains the functional form and compatibility with pre-trained weights while processing fewer low-value tokens per step.

### Overall Architecture
The overall process is divided into two phases. The first phase is reference loss generation: a frozen text-only teacher is constructed from the student MLLM's LLM backbone by removing the visual encoder and projector and replacing visual tokens with `<pad>` placeholders to maintain sequence length. The teacher calculates the per-token negative log-likelihood for answer tokens, resulting in $\ell^{ref}_{b,i}$. A high reference loss indicates the teacher struggles to predict the token using text alone, suggesting a dependency on visual evidence.

The second phase is student training: during training, an EMA difficulty $m_{s,i}$ for each token in every sample is maintained and updated with the current student loss. ReGATE combines these into a token score $d_{b,i}=m_{s,i}+\lambda\ell^{ref}_{b,i}$, and then selects top-$k$ tokens as active tokens according to a sparsity schedule. Active tokens undergo normal attention, MLP, and backpropagation; inactive tokens skip major computations, reducing token usage, training time, and activation memory.

### Key Designs
1. **Reference loss of text-only teacher**:
    - **Function**: Estimates the degree of visual information dependency for each output token.
    - **Mechanism**: The teacher is derived from the student's LLM backbone but is frozen with visual components removed. Visual tokens are replaced by `<pad>` tokens. The teacher predicts answer tokens based on text context; if the negative log-likelihood of a token is high, it suggests the token cannot be inferred from text priors and requires visual grounding.
    - **Design Motivation**: In multimodal tasks, the real cost lies in learning cross-modal evidence rather than repeatedly training easy template language. Reference loss provides a pre-computable visual importance signal without manual labeling.

2. **Student EMA difficulty and dual-cycle sparse scheduling**:
    - **Function**: Allows token selection to change dynamically with training progress rather than fixing a static set of tokens to prune.
    - **Mechanism**: An EMA of historical student loss is maintained: $m_{s,i}\leftarrow\beta m_{s,i}+(1-\beta)\ell^{stu}_{b,i}$. Training steps run in cycles of $C$; the first $F$ steps of each cycle retain all tokens, while subsequent steps retain only a proportion $p_{sparse}$ of high-scoring tokens.
    - **Design Motivation**: The teacher only reflects static visual dependency, while student EMA reflects whether the current model has already learned the token. Combining them prevents wasting computation on tokens that have become easy while ensuring signals from the student state are not ignored.

3. **Parameter-free decoder sparse computation**:
    - **Function**: Converts the token mask into actual training acceleration instead of just changing loss weights.
    - **Mechanism**: In decoder layers, query/key/value transformations and attention are performed only for active tokens; similarly, the MLP only gathers active token hidden states for computation before scattering them back. Inactive tokens pass through via residuals and do not receive gradients from the corresponding computation path.
    - **Design Motivation**: Masking tokens only at the loss layer still incurs most forward/backward costs. ReGATE embeds sparsity into attention and MLP calculations, thereby reducing training time and VRAM usage.

### Loss & Training
ReGATE keeps the original MLLM fine-tuning objective unchanged; it only alters which tokens participate in major computations. In experiments, teacher reference loss is pre-computed and cached for the entire fine-tuning dataset before training to avoid running the teacher at every step. Default hyperparameters are cycle $C=128$, full-token steps $F=16$, sparsity ratio $p_{sparse}=0.5$, global warm-up of 100 iterations, EMA decay $\beta=0.9$, and teacher loss weight $\lambda=0.5$. VideoLLaMA2 and VideoChat2 experiments used 4 H100s, and InternVL3.5 used 16 H100s; the method covers both full fine-tuning and LoRA fine-tuning.

## Key Experimental Results

### Main Results
Image understanding experiments show that while reducing tokens by 41% to 44%, ReGATE often improves performance on tasks like ScienceQA, MME, and VizWiz. The two values for MME correspond to perception / cognition.

| Model | Tokens | ScienceQA | MME | VizWiz | POPE | SEED-I |
|------|--------|-----------|-----|--------|------|--------|
| VideoChat2 | 3.93B | 40.8 | 314.6 / 1244.0 | 28.5 | 86.2 | 45.9 |
| VideoChat2-ReGATE | 2.22B (↓43.51%) | 46.6 | 360.7 / 1287.8 | 32.5 | 85.1 | 47.2 |
| VideoLLaMA2 | 83.82M | 61.4 | 376.4 / 1474.0 | 46.8 | 86.7 | 70.4 |
| VideoLLaMA2-ReGATE | 49.27M (↓41.22%) | 80.5 | 391.1 / 1507.1 | 48.0 | 87.5 | 70.0 |
| InternVL3.5 | 3.96B | 93.3 | 681.6 / 1694.3 | 60.6 | 91.6 | 76.8 |
| InternVL3.5-ReGATE | 2.32B (↓41.41%) | 94.4 | 689.3 / 1698.8 | 61.5 | 93.1 | 76.6 |

Long and short video experiments further verify that ReGATE is not limited to static images. It shows improvements on most Video-MME, MLVU, MVBench, and Perception tasks, though slight decreases occur on EgoSchema, LongVideoBench, or NExT-QA, indicating limits to fixed sparsity rates.

| Model | Tokens | Video-MME | LongVideoBench | MLVU | EgoSchema | MVBench | Perception |
|------|--------|-----------|----------------|------|-----------|---------|------------|
| VideoChat2 | 3.93B | 26.0 | 21.8 | 36.0 | 55.6 | 55.7 | 48.4 |
| VideoChat2-ReGATE | 2.22B | 32.7 | 24.3 | 40.5 | 54.8 | 56.6 | 50.0 |
| VideoLLaMA2 | 83.82M | 53.7 | 47.7 | 53.2 | 58.2 | 52.0 | 53.0 |
| VideoLLaMA2-ReGATE | 49.27M | 54.5 | 47.6 | 54.5 | 56.4 | 53.6 | 54.1 |
| InternVL3.5 | 3.96B | 62.4 | 57.9 | 63.7 | 64.7 | 68.3 | 65.3 |
| InternVL3.5-ReGATE | 2.32B | 63.0 | 58.0 | 64.2 | 63.9 | 69.6 | 66.7 |

Efficiency tables directly demonstrate the "learning faster" conclusion. "Aggressive ReGATE" approaches the baseline in significantly fewer GPU-hours, while "extended ReGATE" exceeds baseline average accuracy while still requiring less training time than the baseline.

| Model | Setup | Tokens ↓ | Teacher Cost | Train Time | Avg. Mem/GPU | Avg. Acc. ↑ |
|------|------|----------|--------------|------------|--------------|-------------|
| VideoLLaMA2 | Baseline | 83.82M | - | 129.6 | 69.1 GB | 48.2 |
| VideoLLaMA2 | ReGATE extended | 49.27M | 2.1 | 107.6 | 61.3 GB | 48.9 |
| VideoLLaMA2 | ReGATE fast | 29.32M | 2.1 | 64.0 | - | 48.0 |
| VideoChat2 | Baseline | 3.93B | - | 148.8 | 70.8 GB | 46.1 |
| VideoChat2 | ReGATE extended | 2.22B | 10.0 | 130.0 | 63.7 GB | 47.8 |
| VideoChat2 | ReGATE fast | 1.51B | 10.0 | 86.4 | - | 46.0 |
| InternVL3.5 | Baseline | 3.96B | - | 435.2 | 58.3 GB | 61.8 |
| InternVL3.5 | ReGATE extended | 2.32B | 11.3 | 374.4 | 51.9 GB | 62.2 |
| InternVL3.5 | ReGATE fast | 1.63B | 11.3 | 262.4 | - | 61.6 |

### Ablation Study
Ablations in the appendix indicate that both signals are important. Using only student EMA or only reference loss performs worse than the combined signal. A capacity-aligned teacher is also more suitable than one that is too small or too large.

| Target | Setup | Avg. Acc. | Explanation |
|----------|------|-----------|------|
| $\lambda$ weight | $\lambda=0.0$, Student EMA only | 47.7 | Considers learning difficulty but lacks visual dependency signal |
| $\lambda$ weight | $\lambda=1.0$, Reference Loss only | 46.4 | Considers teacher only, ignoring current student state |
| $\lambda$ weight | $\lambda=0.5$, Combined signal | 48.9 | Complementary signals yield best results |
| Teacher Capacity | Qwen2-1.5B | 45.4 | Weak teacher; misinterprets linguistic difficulties as visual focus |
| Teacher Capacity | Qwen2-57B | 46.8 | Strong teacher; predicts via world knowledge, underestimating visual dependency |
| Teacher Capacity | Qwen2-7B | 48.9 | Aligned with student capacity and tokenizer; most reliable signal |

### Key Findings
- ReGATE reduces tokens by approximately 41% to 44% across 3 different MLLMs and improves most zero-shot image and video understanding metrics.
- Acceleration effectiveness is influenced by the fine-tuning method: Full fine-tuning for VideoLLaMA2 benefits the most, while VideoChat2 with LoRA sees smaller time gains because its backward pass is already light.
- Reference teacher capacity is not "the larger the better." An overly strong teacher might "guess" visual words using world knowledge, resulting in the pruning of tokens the student needs for visual grounding.
- A fixed $p_{sparse}=0.5$ is simple and stable, but failed cases in the appendix show some useful tokens are still skipped due to the fixed ratio.
- Attention visualization shows that models trained with ReGATE focus more on task-relevant regions like hands and manipulated objects, suggesting token elision shifts the learning focus rather than just saving computation.

## Highlights & Insights
- The most clever aspect is using a text-only teacher as a visual dependency probe. It does not require labeling "which word needs the image"; it simply observes prediction difficulty when visuals are masked to obtain fine-grained token signals.
- The paper avoids complex architecture routes. ReGATE does not add trainable parameters or require rewriting model structures, making it easier to transfer to existing MLLMs compared to many training-time compression methods.
- Combining student EMA and teacher reference loss is natural. The former tells the system "what the model doesn't know yet," and the latter tells the system "what likely needs visuals." Their combination is more stable than either single signal.
- Results suggest that training acceleration does not necessarily sacrifice accuracy. Removing low-value tokens from training computation may reduce background noise, allowing the model to focus more on cross-modal evidence.

## Limitations & Future Work
- The current sparse schedule is a fixed design and cannot adaptively adjust the retention ratio based on sample complexity, task type, or training stability.
- Reference supervision comes from a frozen text-only teacher, which has limited coverage for fine-grained spatial/temporal reasoning; future work could explore stronger but capacity-aligned multimodal-aware teachers.
- Per-token reference loss depends on tokenizer alignment, making it difficult to use cross-architecture teachers and limiting selection.
- The method was primarily validated on publicly trainable 7B/14B MLLMs; engineering benefits for larger-scale closed-source or web-scale training remain unclear.
- Fixed sparsity misses a few critical tokens; in failure cases, useful words like "static" or "NASA" were skipped, suggesting adaptive sparsity is a natural next step.

## Related Work & Insights
- **vs RHO-1**: RHO-1 uses a reference model to select high-value tokens in text-only LMs; ReGATE extends this to MLLMs and interprets reference loss as a visual dependency signal.
- **vs LaVi**: LaVi skips visual tokens via additional visual modulation modules, requiring architectural changes and new parameters; ReGATE is parameter-free and gates the training computation of output text tokens.
- **vs LLaVA-Meteor**: Meteor uses visual token scorers and heuristic pruning, primarily for image instruction tuning; ReGATE combines teacher loss with student EMA, covers both image and video, and adapts to LoRA/full fine-tuning.
- **vs inference-time token pruning**: Inference pruning does not reduce training costs; ReGATE acts directly on forward/backward passes, making it more suitable for the fine-tuning stage of large-scale MLLMs.

## Rating
- Novelty: ⭐⭐⭐⭐ Using text-only reference loss as a visual dependency signal for MLLM training is simple and effective.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Covers 3 models, image/long video/short video, efficiency, comparisons with similar methods, and appendix ablations; the evidence is complete.
- Writing Quality: ⭐⭐⭐⭐ The method logic is clear and experiments are rich, though there are many tables and the layout is somewhat dense.
- Value: ⭐⭐⭐⭐⭐ Highly practical for MLLM fine-tuning costs, particularly for video models and resource-constrained training scenarios.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Better Tokens for Better 3D: Advancing Vision-Language Modeling in 3D Medical Imaging](../../NeurIPS2025/multimodal_vlm/better_tokens_for_better_3d_advancing_vision-language_modeling_in_3d_medical_ima.md)
- [\[ACL 2026\] MathFlow: Enhancing the Perceptual Flow of MLLMs for Visual Mathematical Problems](mathflow_enhancing_the_perceptual_flow_of_mllms_for_visual_mathematical_problems.md)
- [\[CVPR 2026\] LFPC: Learning to Focus and Precise Cropping for MLLMs](../../CVPR2026/multimodal_vlm/lfpc_learning_to_focus_and_precise_cropping_for_mllms.md)
- [\[CVPR 2026\] SPARROW: Learning Spatial Precision and Temporal Referential Consistency in Pixel-Grounded Video MLLMs](../../CVPR2026/multimodal_vlm/sparrow_learning_spatial_precision_and_temporal_referential_consistency_in_pixel.md)
- [\[ICLR 2026\] Why Reinforcement Fine-Tuning Preserves Prior Knowledge Better: A Data Perspective](../../ICLR2026/multimodal_vlm/why_reinforcement_fine-tuning_enables_mllms_preserve_prior_knowledge_better_a_da.md)

</div>

<!-- RELATED:END -->
