---
title: >-
  [Paper Note] Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning
description: >-
  [NeurIPS 2025][Multimodal VLM][VLM decision-making] This paper discovers that the decision-making reasoning capability of VLMs can be decoupled from visual perception—replacing image inputs with textual descriptions yiel…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "VLM decision-making"
  - "text-driven RL"
  - "GRPO"
  - "cross-modal transfer"
  - "embodied reasoning"
date: 2026-05-08
content_hash: ae698bb995223bdc
---

# Praxis-VLM: Vision-Grounded Decision Making via Text-Driven Reinforcement Learning

**Conference**: NeurIPS 2025
**arXiv**: [2503.16965](https://arxiv.org/abs/2503.16965)
**Code**: [https://github.com/Derekkk/Praxis-VLM](https://github.com/Derekkk/Praxis-VLM)
**Area**: Multimodal VLM / Agent Decision-Making / Reinforcement Learning
**Keywords**: VLM decision-making, text-driven RL, GRPO, cross-modal transfer, embodied reasoning

## TL;DR

This paper discovers that the decision-making reasoning capability of VLMs can be decoupled from visual perception—replacing image inputs with textual descriptions yields equal or higher decision accuracy. Building on this insight, Praxis-VLM trains decision-making reasoning on purely textual scenarios via multi-stage GRPO with adaptive rewards, then transfers zero-shot to visual inputs at inference time, achieving comprehensive improvements over SFT baselines on three decision-making benchmarks, with especially notable gains in OOD generalization.

## Background & Motivation

- **Background**: VLMs excel at visual understanding but lack explicit multi-step reasoning for complex situational decision-making tasks (e.g., "what to do upon witnessing a traffic accident," "which action should a robot take"). Recent works such as DeepSeek-R1 and OpenAI o1 demonstrate that RL can substantially enhance LLM reasoning, with subsequent efforts (R1-OneVision, Vision-R1, OpenVLThinker) attempting to transfer this enhancement to VLMs.
- **Limitations of Prior Work**: Existing VLM reasoning enhancement methods rely heavily on large-scale image–text paired data—triplets of image + question + reasoning chain for SFT or RL training. Such paired data is extremely scarce in decision-making scenarios and prohibitively costly to annotate, particularly when covering diverse real-world decision contexts.
- **Key Challenge**: Enhancing VLM decision-making reasoning requires abundant image–text paired data, yet such data is extremely scarce in decision-making domains—a fundamental conflict between data demand and data availability.
- **Goal**: To endow VLMs with strong situational decision-making reasoning capabilities without relying on image–text paired data, and to enable generalization across diverse visual decision-making scenarios.
- **Key Insight**: The authors conduct a critical preliminary experiment on VIVA and PCA-Bench: replacing image inputs with textual descriptions (GPT-4o captions or dataset-annotated text) reveals that decision accuracy of Qwen2.5-VL matches or even exceeds that of image-based inputs. This yields a key insight—the core capability for decision-making reasoning resides in the language domain and can be learned independently of visual perception. This aligns with the *mental model theory* in cognitive science: humans also construct internal linguistic representations for reasoning and decision-making, then apply these internal models to perceptual experiences.
- **Core Idea**: Train VLM decision-making reasoning capabilities on purely textual data (via multi-stage GRPO with adaptive rewards); the acquired reasoning capability automatically transfers to visual inputs at inference time, enabling data-efficient VLM decision-making enhancement.

## Method

### Overall Architecture

Praxis-VLM comprises three key stages:

1. **Text Decision Data Construction**: GPT-4o is used to synthesize purely textual decision-making scenarios at scale (10K training + 1K validation). Each sample consists of a textual situation description, a multiple-choice decision question, and the correct answer—no image data is required.
2. **Multi-Stage GRPO Training**: Stage 1 uses geometry3k mathematics data as a cold start to establish foundational reasoning capabilities; Stage 2 applies RL training on textual decision data to elicit complex decision-making reasoning, with different adaptive rewards at each stage.
3. **Visual Reasoning Transfer**: At inference time, textual inputs are replaced with real visual inputs (images or video frames); the complete VLM architecture (including the vision encoder) processes multimodal inputs, and reasoning capabilities acquired through text training transfer automatically.

A key design principle: only LLM parameters are updated during training; the vision encoder is frozen. At inference, the complete VLM architecture processes image inputs, enabling zero-shot transfer from text to vision.

### Key Designs

1. **Empirical Validation and Exploitation of the Decoupling Between Decision Reasoning and Visual Perception**
   - **Function**: Serves as the empirical foundation for the entire methodology, demonstrating that the bottleneck for VLM decision-making lies in reasoning rather than visual perception.
   - **Mechanism**: On VIVA and PCA-Bench, two settings are compared: (1) original images as situational input, and (2) textual descriptions (GPT-4o captions or dataset annotations) replacing images. The textual-situation setting matches or outperforms image-based inputs, indicating that decision-making reasoning can be fully learned from textual representations.
   - **Design Motivation**: If core decision-making capabilities are not bound to visual perception, expensive image–text paired data can be bypassed and purely textual data used to train reasoning—this is the theoretical foundation of the entire Praxis-VLM framework.

2. **Multi-Stage GRPO Training with Adaptive R1 Reward**
   - **Function**: Incrementally builds a capability chain from format compliance → logical reasoning → complex decision-making across stages.
   - **Mechanism**:
     - **Stage 1 (Cold Start)**: GRPO training on geometry3k geometric data. Reward = $R_{\text{accuracy}} + R_{\text{format}} + 0.5 \cdot R_{\text{tag}}$. $R_{\text{tag}}$ ensures the model learns the `<think></think><answer></answer>` format (checking that each special token appears exactly once); once the format stabilizes, $R_{\text{tag}}$ is removed and $R_{\text{accuracy}}$ becomes dominant.
     - **Stage 2 (Decision RL)**: Training on synthetic textual decision data. Reward = $R_{\text{accuracy}} + 0.8 \cdot R_{\text{format}} + 0.5 \cdot R_{\text{len}}$. $R_{\text{len}} = \min(\text{word\_count}/250,\; 1.0)$ encourages longer and more thorough reasoning chains.
     - **Key Finding**: SFT cold start before RL is unnecessary—with a well-designed adaptive reward strategy, GRPO can be applied directly to instruction-tuned VLMs, simplifying the training pipeline.
   - **Design Motivation**: Directly applying RL on decision data without preparation yields poor results (the model has not yet learned to produce formatted reasoning outputs). The logical reasoning nature of mathematics data makes it naturally suited for cold-start initialization; stage-wise adaptive rewards allow the model to focus on distinct skills at each stage, avoiding multi-objective conflicts.

3. **GPT-4o-Driven Textual Decision Data Synthesis**
   - **Function**: Provides high-quality, diverse, purely textual decision-making training data to support RL training.
   - **Mechanism**: Ten seed questions are manually constructed as in-context examples; GPT-4o is then prompted to generate samples in batches of 10 with deduplication, yielding 10K training + 1K validation samples. Each sample is a triplet of textual situation description, multiple-choice decision question, and answer, designed to require multi-step reasoning and evaluable by simple rules (multiple-choice format).
   - **Design Motivation**: Training data must be sufficiently challenging (forcing the model to learn reasoning rather than pattern matching) and rule-evaluable (avoiding complex reward modeling and reward hacking risks). The batch generation + deduplication strategy ensures data diversity; no image data or manual filtering is required, enabling rapid domain-agnostic data construction.

### Loss & Training

- **Optimization Algorithm**: GRPO (Group Relative Policy Optimization)—samples $G=5$ responses per query, computes group-normalized advantage $\hat{A}$, and updates the policy with a clipped-PPO objective and KL divergence penalty.
- **KL Coefficient**: $\beta = 0.01$, balancing policy update magnitude against deviation from the reference policy.
- **Learning Rate**: $1 \times 10^{-6}$
- **Training Scope**: Full fine-tuning of the LLM component; vision encoder is frozen during training.
- **Inference Configuration**: vLLM + greedy decoding, maximum sequence length 1024 tokens.
- **Hardware**: 4× A100/H100 GPUs.

## Key Experimental Results

### Main Results

| Model | VIVA (%) | PCA-Bench (%) | EgoNormia (OOD, %) |
|---|---|---|---|
| Qwen2.5-VL-3B | 76.61 | 48.58 | 51.92 |
| + SFT | 77.42 | 46.37 | 35.06 |
| + Reason SFT | 75.81 | 49.53 | 28.34 |
| **Praxis-VLM-3B** | **79.03** | **50.79** | **54.27** |
| Qwen2.5-VL-7B | 80.97 | 46.37 | 46.19 |
| + SFT | 81.13 | 45.74 | 34.83 |
| + Reason SFT | 78.79 | 53.00 | 34.08 |
| **Praxis-VLM-7B** | **84.03** | **60.25** | **54.33** |

Key finding: SFT and Reason SFT suffer severe degradation on the OOD EgoNormia benchmark (dropping from 46.19 to 34.83/34.08), whereas Praxis-VLM improves over the base model (54.33 > 46.19), demonstrating that RL-acquired reasoning capabilities are genuinely transferable.

### Ablation Study

| Ablation | VIVA (%) | PCA-Bench (%) | EgoNormia (%) |
|---|---|---|---|
| Praxis-VLM-7B (full) | 84.03 | 60.25 | 54.33 |
| w/o math cold start (one-stage) | 83.87 | 58.99 | 49.57 |
| Praxis-VLM-3B (full) | 79.03 | 50.79 | 54.27 |
| w/o math cold start (one-stage) | 79.52 | 50.79 | 53.13 |

**Diverse Sampling (7B, 8 samples, T=0.2)**:

| Method | VIVA Orig→Major→Pass@1 | PCA-Bench Orig→Major→Pass@1 | EgoNormia Orig→Major→Pass@1 |
|---|---|---|---|
| Qwen2.5-VL-7B | 80.97→80.73→80.81 | 46.37→48.27→56.47 | 46.19→46.36→54.50 |
| Reason SFT | 78.79→80.64→89.03 | 53.00→58.36→82.33 | 34.08→35.69→66.04 |
| Praxis-VLM-7B | 83.87→84.36→89.27 | 58.99→61.83→77.92 | 49.57→55.08→72.23 |

### Key Findings

- **Math cold start primarily benefits OOD generalization**: Differences on in-distribution tasks (VIVA/PCA-Bench) are modest, but on OOD EgoNormia the 7B model improves from 49.57 to 54.33 (+4.76), indicating that mathematical cold start reinforces foundational logical reasoning architecture.
- **Reasoning length correlates with sample difficulty**: Samples are grouped into five quintile bins by the length of Praxis-VLM's generated reasoning chains; longer reasoning corresponds to harder samples. However, Praxis-VLM consistently outperforms the no-reasoning baseline at equivalent difficulty levels.
- **Majority voting advantage**: Praxis-VLM outperforms Reason SFT across all majority voting settings, indicating that the reasoning distribution learned by GRPO has a more reliable center—not only capable of finding correct paths (high Pass@1) but also more stably converging to correct answers.
- **Overthinking risk with excessively long reasoning**: The longest 20% of samples exhibit accuracy drops, partly due to truncation beyond 1024 tokens and partly due to noise introduced by overly long reasoning chains interfering with final decisions.

## Highlights & Insights

- **"Decision-making reasoning can be decoupled from visual perception"** is a finding with profound cognitive science implications—echoing mental model theory, wherein humans construct internal linguistic representations for reasoning and decision-making and then apply them to perceptual experience. This discovery opens a new data-efficient pathway for VLM training.
- The paradigm of **purely textual training → visual reasoning transfer** is highly elegant: no image–text paired data is needed during training, yet the model directly handles visual inputs at inference time, achieving a clean decoupling between training data requirements and inference-time capability.
- **Bypassing SFT cold start and applying GRPO directly** simplifies the training pipeline—provided adaptive rewards are carefully designed (format first, then accuracy), instruction-tuned models can learn to reason directly through RL.
- **Effectiveness of the $R_{\text{len}}$ reward** challenges the stereotype that "longer reasoning is not necessarily better"—in decision-making tasks, more thorough situational analysis demonstrably yields better decision quality.
- **Four-dimensional reasoning analysis** (situational analysis, action–outcome evaluation, safety–risk management, rule–norm compliance) reveals that the model learns structured decision-making thought patterns rather than end-to-end black-box mappings.
- **Error analysis** identifies three major failure modes (situational misunderstanding, safety priority errors, and norm alignment failures), pointing to concrete directions for future improvement.

## Limitations & Future Work

- Validation is limited to 3B and 7B models; performance on larger models (e.g., 72B) remains unknown, particularly whether text-to-vision transfer scales linearly with model size.
- Textual decision data is synthesized by GPT-4o and may carry domain bias—whether the situational distribution of synthetic data covers long-tail real-world scenarios remains an open question.
- EgoNormia is evaluated by stitching video frames into a single image, which does not natively assess VLM video understanding; validation on true video input is needed.
- The 1024-token reasoning chain limit causes truncation for some complex scenarios—performance under longer context settings warrants further exploration.
- No direct comparison is made with other decision-making enhancement methods (e.g., VLA models), nor is the combination with other RL methods (DPO, PPO) explored.
- Stage 1 cold start relies on a fixed geometry3k dataset; whether alternative cold-start data choices yield better outcomes deserves investigation.

## Related Work & Insights

- **vs. R1-OneVision / Vision-R1**: These methods apply RL on image–text paired data to enhance VLM reasoning. Praxis-VLM demonstrates that for decision-making tasks, purely textual training suffices—the two approaches are complementary, with text training handling high-level reasoning and visual training handling perceptual details.
- **vs. NoisyRollout**: NoisyRollout enhances RL exploration through visual perturbations; Praxis-VLM entirely bypasses the visual domain by training on pure text—two data-efficiency strategies addressing the training data bottleneck from different dimensions.
- **vs. DeepSeek-R1**: Praxis-VLM extends the R1 RL-based reasoning enhancement paradigm from purely textual LLMs to cross-modal VLM decision-making, validating the cross-modal transferability of RL-acquired reasoning capabilities.
- **AI instantiation of mental model theory**: This work provides computational validation for a cognitive science hypothesis (humans reason and make decisions via linguistic representations), which may in turn inspire cognitive science research.
- **Data efficiency implications**: For VLM tasks requiring both understanding and reasoning, the pipeline can be decomposed into visual perception (leveraging existing VLM capabilities) and high-level reasoning (trained on textual data) as two independent modules, substantially reducing training data requirements.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ — The finding that "decision-making reasoning and visual perception can be decoupled" is empirically grounded and resonates with cognitive theory; the paradigm of training VLM decision-making reasoning on purely textual data is pioneering.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Three benchmarks, multiple baselines, diverse sampling, reasoning dimension analysis, and error analysis constitute a highly comprehensive evaluation; points deducted for model scale limited to 7B and absence of comparisons with other decision-making methods (e.g., VLA models).
- **Writing Quality**: ⭐⭐⭐⭐⭐ — The narrative arc from preliminary finding to method design to experimental validation is exceptionally coherent; the epigraph "Language is the dress of thought" is aptly chosen, and every design decision is motivated clearly.
- **Value**: ⭐⭐⭐⭐⭐ — The text-training → visual-transfer paradigm is broadly applicable to other visual reasoning tasks; the adaptive reward design strategy is directly reusable; the four-dimensional decision-making reasoning framework can guide the design of finer-grained rewards.

## Related Papers

- [\[NeurIPS 2025\] AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making](antigrounding_lifting_robotic_actions_into_vlm_representatio.md)
- [\[NeurIPS 2025\] Unified Reinforcement and Imitation Learning for Vision-Language Models](unified_reinforcement_and_imitation_learning_for_vision-language_models.md)
- [\[NeurIPS 2025\] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)
- [\[NeurIPS 2025\] To Think or Not To Think: A Study of Explicit Thinking in Rule-Based Visual Reinforcement Fine-Tuning](think_or_not_think_a_study_of_explicit_thinking_in_rule-based_visual_reinforceme.md)
- [\[NeurIPS 2025\] VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning](vt-fsl_bridging_vision_and_text_with_llms_for_few-shot_learning.md)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] AntiGrounding: Lifting Robotic Actions into VLM Representation Space for Decision Making](antigrounding_lifting_robotic_actions_into_vlm_representatio.md)
- [\[NeurIPS 2025\] Unified Reinforcement and Imitation Learning for Vision-Language Models](unified_reinforcement_and_imitation_learning_for_vision-language_models.md)
- [\[NeurIPS 2025\] SRPO: Enhancing Multimodal LLM Reasoning via Reflection-Aware Reinforcement Learning](srpo_enhancing_multimodal_llm_reasoning_via_reflection-aware_reinforcement_learn.md)
- [\[NeurIPS 2025\] VT-FSL: Bridging Vision and Text with LLMs for Few-Shot Learning](vt-fsl_bridging_vision_and_text_with_llms_for_few-shot_learning.md)
- [\[NeurIPS 2025\] To Think or Not To Think: A Study of Explicit Thinking in Rule-Based Visual Reinforcement Fine-Tuning](think_or_not_think_a_study_of_explicit_thinking_in_rule-based_visual_reinforceme.md)

</div>

<!-- RELATED:END -->
