---
title: >-
  [Paper Note] Critic-V: VLM Critics Help Catch VLM Errors in Multimodal Reasoning
description: >-
  [CVPR 2025][VLM Reasoning][VLM Reasoning Correction] This paper proposes the Critic-V framework, which decouples the VLM reasoning process into a Reasoner and a Critic. By utilizing a DPO-trained Critic model to provide natural language feedback for iteratively optimizing the reasoning path, this approach outperforms GPT-4V on 5 out of 8 benchmarks, showing particularly significant improvements on mathematical reasoning tasks (MathVista +11.8%).
tags:
  - "CVPR 2025"
  - "VLM Reasoning"
  - "VLM Reasoning Correction"
  - "Actor-Critic"
  - "DPO Preference Optimization"
  - "Natural Language Feedback"
  - "Multimodal Reasoning"
date: 2026-05-08
content_hash: 81ac50aa4d892638
---

# Critic-V: VLM Critics Help Catch VLM Errors in Multimodal Reasoning

**Conference**: CVPR 2025  
**arXiv**: [2411.18203](https://arxiv.org/abs/2411.18203)  
**Code**: [GitHub](https://github.com/kyrieLei/Critic-V)  
**Area**: Multimodal VLMs  
**Keywords**: VLM Reasoning Correction, Actor-Critic, DPO Preference Optimization, Natural Language Feedback, Multimodal Reasoning

## TL;DR
This paper proposes the Critic-V framework, which decouples the VLM reasoning process into a Reasoner and a Critic. By utilizing a DPO-trained Critic model to provide natural language feedback for iteratively optimizing the reasoning path, this approach outperforms GPT-4V on 5 out of 8 benchmarks, showing particularly significant improvements on mathematical reasoning tasks (MathVista +11.8%).

## Background & Motivation
Vision-Language Models have made significant progress in multimodal reasoning tasks but continue to generate inaccurate or irrelevant responses. Key issues include: hallucinations in image understanding (where models "see" content not present in the image), coarse reasoning paths (where minor errors in intermediate steps cascade and amplify during chain-of-thought steps), and over-reliance on internal knowledge (ignoring the actual visual context).

Existing mitigation strategies mainly rely on the model's intrinsic capabilities: Self-Refine allows the model to self-reflect and correct, Self-Consistency aggregates votes from multiple sampled outputs, and DPO/RLHF aligns models with human preferences during training. However, these methods share a common weakness: **they do not introduce an external, independent quality evaluation**. Methods like Self-Refine depend entirely on the model's self-judgment, whereas studies show that LLMs can often correct known errors but struggle to identify errors autonomously.

The core argument of this paper is that an independently trained, error-detection-specialized Critic model is required to provide specific natural language feedback (rather than scalar rewards) to the Reasoner during inference, thereby enabling more precise error localization and reasoning path optimization. This is inspired by the division of labor in the Actor-Critic paradigm of reinforcement learning.

## Method

### Overall Architecture
Critic-V comprises three core phases:
1. Offline Critic model training (via VEST training data generation + DPO optimization)
2. Inference-time Reasoner-Critic interaction loop
3. Iterative optimization based on natural language feedback

The entire workflow: Given an image and a question → Reasoner generates an initial reasoning response → Critic evaluates the response and points out errors → Reasoner corrects the response based on the feedback → iterate until the Critic is satisfied or the maximum number of rounds is reached.

### Key Designs
1. **Reasoner**:

    - Directly uses existing VLMs (e.g., Qwen2-VL-7B, DeepSeek-VL-7B, etc.) as the Reasoner without modifying their parameters.
    - Core Innovation: Replaces parametric policies in traditional RL with dynamic text prompts. The reasoning strategy is adjusted through prompt changes rather than gradient updates.
    - Critic's feedback $\delta P^{reasoner}$ is directly concatenated to the next-round prompt to guide the reasoning direction.
    - Leverages the TextGrad framework for prompt updates: treats the Critic's natural language feedback as "textual gradients" to guide reasoning path adjustments.
    - Represents a "plug-and-play" design — requiring no training or fine-tuning of the Reasoner.

2. **Critic Model Training**:

    - **VEST (Vision Error inSertion Technique) Data Construction**:
        - Problem-image pairs are collected from VQA datasets, and GPT-4o is used to insert 1-5 false details into correct answers (generating incorrect answers).
        - Three VLMs (GLM-4V-9B, GPT-4o mini, and MiniCPM-V) are employed to generate critiques for these incorrect answers.
        - Under normal conditions, a total of 29,012 multimodal QA pairs and corresponding critique data are constructed.
    - **Rule-based Reward (RBR) Scoring**:
        - Combines the Jaccard index and GPT scores to evaluate critique quality: $Score(i) = Jaccard(i) + \alpha \times GPT(i)$
        - Jaccard Index: $J(G,C) = |G \cap C| / |G \cup C|$, where $G$ is the set of inserted errors and $C$ is the set of errors detected by the critique.
        - The Jaccard index is crucially introduced to prevent "long critique bias" — excessively long critiques might contain more false positives (nitpicks).
        - Pairwise preference pairs (preferred vs. disfavored critique) are constructed based on RBR scores.
    - **DPO Training**:
        - The Critic model is trained based on Qwen2-VL-7B.
        - Optimized using the standard DPO loss function to encourage the model to assign higher probabilities to high-quality critiques.
        - Preference dataset $\mathcal{D}_{cri} = \{(Q, I, C_w, C_l)\}$, where $C_w$ is the preferred critique and $C_l$ is the disfavored critique.

3. **Reasoner-Critic Interaction Framework**:

    - Reasoner generates initial answer → Critic evaluates within the full context of the question, image, and response → gives feedback in natural language (identifying what is wrong, why it is wrong, and how it should be corrected).
    - Reasoner incorporates the feedback into a new prompt and continues reasoning → loops until the Critic is satisfied or the maximum number of iterations is reached.
    - Natural language feedback is more informative than scalar rewards: it precisely points out error locations and types rather than just giving a binary "good/bad" signal.
    - Each Critic evaluation consumes only a few dozen extra tokens, presenting a negligible computational overhead.

### Loss & Training
- The Critic is trained using the DPO loss: $\mathcal{L}_{DPO} = -\mathbb{E}[\log \sigma f(\pi_\theta; \pi_{ref})]$
- Temperature is set to 0 or close to 0 to ensure output stability.
- A two-round dialogue is adopted during inference: the first round consists of the Reasoner's response, and the second round consists of the Reasoner's correction after the Critic's evaluation.
- Plug-and-play characteristic of the Critic: the same Critic can be paired with different Reasoner VLMs.

## Key Experimental Results

### Main Results

| Model | RealWorldQA | MMBench | MathVista | MathVerse | MMT-Bench |
|------|-------------|---------|-----------|-----------|-----------|
| Qwen2-VL-7B baseline | 70.1 | 80.7 | 61.4 | 25.8 | 60.4 |
| **+Critic-V** | **74.9(+4.8)** | **82.8(+2.1)** | **73.2(+11.8)** | **32.9(+7.1)** | **62.0(+1.6)** |
| DeepSeek-VL-7B baseline | 58.1 | 73.5 | 35.3 | 18.4 | 46.5 |
| **+Critic-V** | **62.1(+4.0)** | **79.0(+5.5)** | **53.1(+17.8)** | **28.9(+10.5)** | **53.6(+7.1)** |
| LLaVA-v1.5-7B baseline | 50.7 | 68.4 | 37.8 | 26.0 | 36.0 |
| **+Critic-V** | **63.5(+12.8)** | **73.8(+5.4)** | **53.1(+15.3)** | **30.5(+4.5)** | **47.4(+11.4)** |
| GPT-4V | 61.4 | 74.3 | 49.9 | 54.4 | 55.5 |

### Comparison with Other Methods (Based on LLaVA-v1.5-7B)

| Method | RealWorldQA | MMStar | MMBench | SEEDBench | MMT-Bench |
|------|-------------|--------|---------|-----------|-----------|
| +POVID | 51.8 | 33.6 | 71.6 | 65.4 | 33.4 |
| +SCL (Strongest Competitor) | 53.2 | 35.8 | 70.8 | 68.6 | 39.6 |
| **+Critic-V** | **63.5** | **38.4** | **73.8** | **70.1** | **49.7** |

### Ablation Study

| Configuration | MathVista | MMT-Bench | MMBench | Notes |
|------|-----------|-----------|---------|------|
| Qwen2-VL-7B baseline | 61.4 | 60.4 | 80.7 | - |
| +Self-Refine (Without DPO) | 63.4 | 57.8 | 82.1 | Degrades performance on MMT-Bench instead |
| +Critic-V (With DPO) | **73.2** | **62.0** | **82.8** | DPO is crucial |
| +Special Prompt Only (Without Critic) | 61.8 | 59.0 | 81.0 | Rules out the impact of prompt design |

### Key Findings
- Critic-V achieves improvements in 23 out of 24 comparative experiment groups, showing highly strong generalization.
- The performance gain on mathematical reasoning tasks is the most significant (MathVista +11.8% to +17.8%), suggesting that "logic-intensive tasks benefit the most from external error correction."
- Qwen2-VL-7B + Critic-V surpasses GPT-4V on 5 out of 8 benchmarks.
- Self-Refine (self-correction) underperforms compared to using an external Critic, which indicates that indeed models struggle to find errors autonomously.
- DPO training is key to Critic-V's effectiveness; models without DPO training acting as Critics can even be detrimental.
- Each Critic evaluation consumes only a few dozen additional tokens, keeping the computational overhead extremely low.

## Highlights & Insights
1. Conceptually transfers the RL Actor-Critic paradigm to VLM reasoning scenarios, where using natural language feedback instead of scalar rewards is a key innovation.
2. The VEST data construction method is highly practical: it automatically generates Critic training data by inserting known errors, bypassing manual annotation.
3. The integration of the Jaccard index in RBR addresses the bias where "longer critiques receive higher scores."
4. The plug-and-play design means one Critic can serve various different VLMs, demonstrating high practical value.
5. Validates an important hypothesis: VLMs can effectively correct their output when errors are pointed out (an extension of Tyen et al.'s finding into the VLM domain).
6. LLaVA-v1.5-7B improves from 50.7 to 63.5 (+12.8) on RealWorldQA, showing that even weaker models can benefit significantly from a strong Critic.

## Limitations & Future Work
- The Critic model itself is trained on top of Qwen2-VL-7B, indicating its capability upper bound is limited by the backbone model — making it potentially unable to recognize errors beyond its capacity.
- Error insertion in VEST relies on GPT-4o, posing issues related to cost and reproducibility.
- The choice of iteration counts lacks an adaptive mechanism — currently fixed at 2 rounds, though different difficulty levels might require varying numbers of rounds.
- A decrease of -4.5% is observed for Qwen2-VL-7B+Critic-V on the MMStar benchmark, indicating that Critic comments can sometimes mislead rather than help.
- Only 7B-scale models are evaluated as Reasoners; whether the approach remains effective on larger models (e.g., 70B+) has not yet been verified.
- The combined effects of using different VLMs for the Critic and the Reasoner have not been systematically explored.

## Related Work & Insights
- Relation to CriticGPT: While CriticGPT focuses on code review for textual LLMs, Critic-V extends this idea to multimodal visual reasoning.
- Comparison with Self-Refine/Self-Consistency: The external Critic outperforms self-refinement, demonstrating that the "self-checking capability" is a distinct capability independent of the reasoning capability.
- Application of DPO in Critic-V differs from standard usage: while conventional DPO aligns model generation preferences, here it is used to train the evaluation preferences of the Critic.
- Direct application value for embodied AI and autonomous driving — scenarios that demand extremely high reasoning reliability.
- Inspiration: Could the Critic be trained as a more fine-grained "bug detector" that not only points out flaws in reasoning steps but also localizes them to specific visual regions?

## Rating
- Novelty: ⭐⭐⭐⭐ The transfer of Actor-Critic to VLMs is novel, and the VEST+RBR data construction pipeline is well-designed.
- Experimental Thoroughness: ⭐⭐⭐⭐ 8 benchmarks, 3 Reasoner VLMs, and thorough ablation studies, though lacking experiments on larger models.
- Writing Quality: ⭐⭐⭐⭐ The framework description is clear and mathematical derivations are complete, though parts are slightly redundant.
- Value: ⭐⭐⭐⭐⭐ Highly practical, with a plug-and-play design, substantial performance gains, and low computing overhead.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] VAGEN: Reinforcing World Model Reasoning for Multi-Turn VLM Agents](../../NeurIPS2025/vlm_reasoning/vagen_reinforcing_world_model_reasoning_for_multi-turn_vlm_agents.md)
- [\[NeurIPS 2025\] SpatialTraceGen: High-Fidelity Traces for Efficient VLM Spatial Reasoning Distillation](../../NeurIPS2025/vlm_reasoning/spatialtracegen_high-fidelity_traces_for_efficient_vlm_spatial_reasoning_distill.md)
- [\[NeurIPS 2025\] iFinder: Structured Zero-Shot VLM Grounding for Dash-Cam Video Reasoning](../../NeurIPS2025/vlm_reasoning/ifinder_structured_zero-shot_vision-based_llm_grounding_for_dash-cam_video_reaso.md)
- [\[CVPR 2026\] A Causal Marriage between VLM and IRM from Understanding to Reasoning](../../CVPR2026/vlm_reasoning/a_causal_marriage_between_vlm_and_irm_from_understanding_to_reasoning.md)
- [\[ICLR 2026\] VLM-SubtleBench: How Far Are VLMs from Human-Level Subtle Comparative Reasoning?](../../ICLR2026/vlm_reasoning/vlm-subtlebench_how_far_are_vlms_from_human-level_subtle_comparative_reasoning.md)

</div>

<!-- RELATED:END -->
