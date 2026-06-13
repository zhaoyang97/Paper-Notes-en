---
title: >-
  [Paper Note] Reinforcing Structured Chain-of-Thought for Video Understanding
description: >-
  [CVPR 2026][LLM Reasoning][VideoQA] This paper proposes SDRL (Summary-Driven Reinforcement Learning), a single-stage RL framework that requires no SFT. By introducing a structured CoT (Summarize→Think→Answer) and two sel…
tags:
  - "CVPR 2026"
  - "LLM Reasoning"
  - "VideoQA"
  - "Reinforcement Learning"
  - "Structured CoT"
  - "GRPO"
  - "Temporal Reasoning"
date: 2026-05-08
content_hash: 8d48b239a78bdbc9
---

# Reinforcing Structured Chain-of-Thought for Video Understanding

**Conference**: CVPR 2026
**arXiv**: [2603.25942](https://arxiv.org/abs/2603.25942)  
**Code**: None  
**Area**: Video Understanding / Video Reasoning
**Keywords**: VideoQA, Reinforcement Learning, Structured CoT, GRPO, Temporal Reasoning

## TL;DR

This paper proposes SDRL (Summary-Driven Reinforcement Learning), a single-stage RL framework that requires no SFT. By introducing a structured CoT (Summarize→Think→Answer) and two self-supervised mechanisms (CVK and DVR), SDRL enhances temporal reasoning in video understanding and achieves state-of-the-art results on 7 VideoQA benchmarks.

## Background & Motivation

Multimodal large language models (MLLMs) have shown promise in video understanding, yet two core challenges remain:

**Thinking Drift**: Existing RL methods (e.g., GRPO) optimize solely based on reward signals from final answers, leaving intermediate reasoning steps unconstrained. This causes models to generate verbose or visually irrelevant reasoning chains, severely undermining result stability.

**Weak Temporal Understanding**: MLLMs typically represent videos as stacked or averaged frame embeddings, neglecting fine-grained temporal dependencies, which leads to poor performance on temporally sensitive VideoQA tasks.

Limitations of existing solutions:
- **Pure RL methods**: Unconstrained reasoning, unstable
- **SFT+RL methods**: Require costly CoT annotations, involve complex multi-stage training, and token-level imitation in SFT may limit generalization and risk overfitting

The core innovation of SDRL lies in **directly integrating structured CoT into the RL objective**, constraining the reasoning process in a self-supervised manner without requiring an additional SFT stage or CoT annotation data.

## Method

### Overall Architecture

SDRL adopts Qwen2.5-VL-7B as the backbone. Given a (video, question) input, the model is prompted to produce structured output consisting of:
- **Summary segment** (`<summary>`): Extracts key actions and their temporal order
- **Think segment** (`<think>`): Conducts logical reasoning grounded in the summary
- **Answer segment**: Produces the final answer

For each input, $G$ groups of outputs are sampled. Group advantage values are computed via token-level weights (CVK+DVR) and standard rewards (accuracy + format) to optimize the policy.

### Key Designs

1. **Structured CoT (Summarize→Think→Answer)**:

    - Empirical analysis confirms that CoT chains associated with correct predictions exhibit higher BLEU and sBERT similarity to ground-truth CoT chains.
    - Effective CoT must capture two core elements: (1) key actions/events and (2) their temporal order.
    - The Summary segment serves as a top-down reasoning anchor, providing a factual grounding for the subsequent Think segment.
    - **Design Motivation**: Using the summary as a structured anchor fundamentally addresses the thinking drift problem.

2. **Consistency of Vision Knowledge (CVK)**:

    - **Core assumption**: The visual content of a video is fixed and factual; therefore, summaries sampled multiple times from the same input should exhibit high semantic consistency.
    - **GT-supervised mode**: When ground-truth summaries are available, alignment is measured using a combined sBERT+BLEU similarity metric as an additional reward.
    - **Self-supervised mode**: A consistency anchor $S^C$ (position-wise centroid) is dynamically derived from correct predictions; KL divergence measures inconsistency and is converted into a Summary Token Weight $\omega_t^S$.
    - Higher KL divergence → lower consistency → smaller weight, encouraging the model to produce stable and consistent summary segments.
    - **Design Motivation**: Rather than directly supervising summary content, factual faithfulness is indirectly constrained through intra-group consistency.

3. **Dynamic Variety of Reasoning (DVR)**:

    - Encourages diversity in reasoning paths within the Think segment, measured by the entropy of token distributions.
    - High-entropy tokens receive higher diversity weights $\omega_{g,t}^d$.
    - **Dynamic modulation**: Diversity incentives are adjusted according to group accuracy $\mathcal{A}$:
        - Low-accuracy groups: $(1-\mathcal{A})$ is large, reinforcing exploration.
        - High-accuracy groups: $(1-\mathcal{A})$ is small, preserving stable reasoning paths.
    - **Design Motivation**: Prevents forced diversity from introducing noise when the model is already confident, while encouraging exploration under uncertainty.

4. **EventFlowQA Dataset Construction**:

    - A VideoQA dataset focused on complex action sequences and temporal causality.
    - 53K high-quality QA pairs (50K training + 3K validation), covering 15 temporal dimensions.
    - Serves as the core benchmark for all ablation experiments.

### Loss & Training

Structured policy objective: $\mathcal{J}_{total}(\theta) = \mathcal{J}_{grpo}^{SCoT}(\theta) - \mathcal{J}_{reg}(\theta)$

Token-level weights:
$$W_{g,t} = \begin{cases} \omega_t^S & \text{(Summary segment, consistency weight)} \\ \omega_{g,t}^{d'} & \text{(Think segment, dynamic diversity weight)} \end{cases}$$

Training configuration:
- Single-stage RL (no SFT), 32 A100 GPUs
- GRPO group size $G=8$, 1000 RL iterations in total
- 16 frames uniformly sampled, resolution $128\times28\times28$
- Hyperparameters: $\alpha=0.7$, $\beta=0.3$, $\gamma_1=1$, $\gamma_2=1$, $\lambda=0.5$, $\lambda'=0.7$

## Key Experimental Results

### Main Results

Performance on 7 public VideoQA benchmarks (Accuracy %):

| Dataset | SDRL (Ours) | Video-R1 (SFT+RL) | VideoRFT (SFT+RL) | TW-GRPO (RL) | Gain (vs best RL) |
|--------|-------------|--------------------|--------------------|--------------|-------------------|
| NExT-GQA | 79.3 | 74.3 | 75.1 | 76.1 | +3.2 |
| MMVU | 68.6 | 64.2 | 67.3 | 65.8 | +1.3 |
| VideoMMMU | 51.3 | 52.4 | 50.6 | - | +0.7 |
| VSIBench | 32.9/36.1† | 34.6 | 35.7 | - | +0.4† |
| MVBench | 64.2 | 62.7 | 61.4 | 63.3 | +0.9 |
| TempCompass | 74.4† | 72.6 | 73.1 | 73.3 | +1.1† |
| VideoMME | 54.7 | 57.4 | 58.1 | 55.1 | - |

Note: † denotes variants trained on EventFlowQA using only 20% of the data volume of Video-R1.

### Ablation Study

Ablation of CVK and DVR modules on EventFlowQA:

| Configuration | Accuracy | Notes |
|------|----------|------|
| Vanilla GRPO | 42.37 | Baseline |
| +sBERT (GT) | 43.85 | Semantic consistency helps |
| +BLEU (GT) | 46.32 | Lexical consistency helps more |
| +sBERT+BLEU (GT) | 48.56 | Combined is optimal |
| +GT CVK + Static Entropy DVR | 50.09 | Diversity brings further gains |
| +GT CVK + Dynamic DVR (Full) | 52.22 | Dynamic adjustment is optimal |
| Self-supervised CVK | 54.28 | **Self-supervision surpasses GT supervision** |
| Self-supervised CVK + Dynamic DVR | **56.10** | Best configuration |

Effect of model scale on supervision mode:

| Configuration | 3B Model | 7B Model |
|------|---------|---------|
| GT supervision gain | +3.01 | +6.19 |
| Self-supervised gain | +2.40 | **+11.91** |

### Key Findings

1. **Self-supervision outperforms GT supervision (7B)**: Larger models benefit more from self-supervised consistency (+11.91 vs +6.19), possibly because strict GT alignment suppresses pretrained semantic priors and causes catastrophic forgetting.
2. **Smaller models rely more on GT guidance**: The 3B model performs slightly better under GT supervision (+3.01 vs +2.40).
3. **Entropy outperforms KL divergence as a diversity measure**: Entropy as a global uncertainty control better preserves semantic diversity, whereas position-dependent KL alignment suppresses global variability.
4. **Dynamic diversity modulation significantly outperforms static**: Avoids introducing noise through excessive exploration in high-accuracy groups.
5. **Competitive performance with only 20% of the data**: SDRL trained on EventFlowQA surpasses all baselines on TempCompass, demonstrating high data efficiency.

## Highlights & Insights

- **Single-stage RL replaces the SFT+RL pipeline**: By leveraging structured CoT and self-supervised constraints, the need for costly CoT annotations and multi-stage training is eliminated, representing an elegant simplification.
- **Summary as a factual anchor**: Positioning the summary at the front of the reasoning chain ensures factual extraction precedes logical inference, fundamentally resolving thinking drift.
- **Balancing alignment and exploration**: CVK is responsible for consistency/alignment, while DVR handles diversity/exploration; both are unified within the same objective function through token-level weights.
- **Unexpected finding on self-supervised consistency**: Self-supervised performance surpassing GT supervision in larger models suggests that overly strong supervision signals may constrain expressive capacity.

## Limitations & Future Work

1. Experiments are conducted only under the 16-frame setting; scalability to longer videos (e.g., 64 frames or minute-level) remains unexplored.
2. Summary segment generation may introduce additional computational overhead; its impact on real-time applications requires evaluation.
3. Construction details of the EventFlowQA dataset are underreported in the main text, and quality control mechanisms lack transparency.
4. SDRL does not reach the best performance of SFT+RL methods on VideoMME (54.7 vs 58.1), indicating room for improvement in generalization.
5. The self-supervised consistency anchor relies on the existence of correct predictions and may fail in extremely low-accuracy scenarios.

## Related Work & Insights

- **GRPO/DAPO**: Provide the foundational RL optimization framework; SDRL introduces structured constraints on top of this.
- **Video-R1**: The first to apply GRPO to video understanding, but relies on a two-stage SFT+RL pipeline.
- **GRPO-CARE**: The idea of group-level consistency is related to CVK, but does not distinguish between different segments of the reasoning chain.
- **Process Reward Models**: The notion of process-level supervision shares conceptual similarities with the token-level weight design of CVK and DVR.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ (Innovative combination of structured CoT and self-supervised RL; single-stage pipeline is clean and effective)
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (7 benchmarks, detailed ablations, multi-scale analysis, visual comparisons)
- Writing Quality: ⭐⭐⭐⭐ (Method description is clear but formula-heavy; dataset details are insufficient)
- Value: ⭐⭐⭐⭐⭐ (Provides a more concise and efficient training paradigm for video reasoning)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] EagleVision: A Dual-Stage Framework with BEV-grounding-based Chain-of-Thought for Spatial Intelligence](eaglevision_a_dual-stage_framework_with_bev-grounding-based_chain-of-thought_for.md)
- [\[NeurIPS 2025\] Visual Thoughts: A Unified Perspective of Understanding Multimodal Chain-of-Thought](../../NeurIPS2025/llm_reasoning/visual_thoughts_a_unified_perspective_of_understanding_multi.md)
- [\[ICCV 2025\] Video-T1: Test-Time Scaling for Video Generation](../../ICCV2025/llm_reasoning/video-t1_test-time_scaling_for_video_generation.md)
- [\[ACL 2026\] TemplateRL: Structured Template-Guided Reinforcement Learning for LLM Reasoning](../../ACL2026/llm_reasoning/templaterl_structured_template-guided_reinforcement_learning_for_llm_reasoning.md)
- [\[CVPR 2026\] Rationale-Enhanced Decoding for Multi-modal Chain-of-Thought](rationale-enhanced_decoding_for_multi-modal_chain-of-thought.md)

</div>

<!-- RELATED:END -->
