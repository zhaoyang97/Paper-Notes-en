---
title: >-
  [Paper Note] BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards
description: >-
  [AAAI 2026][Reasoning][Ad moderation] This paper proposes BLM-Guard, an explainable multimodal moderation framework for short-video commercial advertisements. It first establishes structured reasoning capability via rule-driven ICoT data synthesis and SFT cold-start, then applies Self-Adaptive GRPO reinforcement learning (combining rule correctness rewards and a self-adaptive consistency reward SCA-R) to optimize policy alignment, achieving 91.4% strict accuracy and 0.845 rea…
tags:
  - "AAAI 2026"
  - "Reasoning"
  - "Ad moderation"
  - "multimodal CoT reasoning"
  - "policy-aligned RL"
  - "GRPO"
  - "cross-modal inconsistency detection"
date: 2026-05-08
content_hash: 599e10a75a124348
---

# BLM-Guard: Explainable Multimodal Ad Moderation with Chain-of-Thought and Policy-Aligned Rewards

**Conference**: AAAI 2026
**arXiv**: [2602.18193](https://arxiv.org/abs/2602.18193)  
**Code**: None  
**Area**: LLM Reasoning
**Keywords**: Ad moderation, multimodal CoT reasoning, policy-aligned RL, GRPO, cross-modal inconsistency detection

## TL;DR
This paper proposes BLM-Guard, an explainable multimodal moderation framework for short-video commercial advertisements. It first establishes structured reasoning capability via rule-driven ICoT data synthesis and SFT cold-start, then applies Self-Adaptive GRPO reinforcement learning (combining rule correctness rewards and a self-adaptive consistency reward SCA-R) to optimize policy alignment, achieving 91.4% strict accuracy and 0.845 reasoning consistency score on a real-world ad benchmark.

## Background & Motivation

**Background**: Commercial ad content moderation on short-video platforms (TikTok, Kuaishou, etc.) is in high demand, yet existing content safety models (e.g., LlamaGuard, LlavaGuard) primarily target coarse-grained risks such as violence and pornography, and do not support the fine-grained, policy-driven moderation required for advertising compliance.

**Limitations of Prior Work**: Ad violations are often subtle—visually legitimate content may contain exaggerated claims in audio, captions and speech may be inconsistent, or seemingly normal visuals may carry misleading implications. Existing methods suffer from three problems: (a) lack of cross-modal causal reasoning; (b) inability to adapt to policy drift as rules are frequently updated; (c) insufficient domain-specific reasoning for commercial risk scenarios.

**Key Challenge**: Ad moderation requires not only accurate violation detection, but also interpretable reasoning chains (why a violation occurred and which rule was violated), while simultaneously detecting intra-modal manipulation (e.g., exaggerated visuals) and inter-modal mismatches (e.g., caption–speech inconsistency).

**Goal**: To build a multimodal ad moderation system that is both accurate and explainable, requiring structured reasoning, policy alignment, and cross-modal consistency detection.

**Key Insight**: Ad moderation is formulated as a policy-aligned multi-step reasoning task—observation (modality content description) → risk screening → causal analysis → final verdict—with CoT structuring this process and RL ensuring dynamic policy alignment.

**Core Idea**: Interleaved CoT for structured reasoning cold-start combined with self-adaptive GRPO for policy-aligned reinforcement learning, handling both intra-modal and inter-modal violations simultaneously.

## Method

### Overall Architecture
A two-stage training pipeline:
- **Stage 1 (Rule-driven SFT Cold-Start)**: Structured reasoning data is synthesized using InternVL-3-78B (keyframe extraction + ICoT generation), followed by rule-anchored SFT of Qwen2.5-VL-7B.
- **Stage 2 (Self-Adaptive GRPO RL)**: Reinforcement learning fine-tuning with a composite reward (rule correctness + format + self-adaptive consistency) and an improved GRPO formulation.

### Key Designs

1. **Keyframe & Region Extraction**:

    - *Function*: Selects keyframes and visual regions from short videos that are most indicative of risk.
    - *Mechanism*: 16 frames are uniformly sampled; CLIP-ViT-L/14 computes the semantic similarity of each frame against 7 risk prompts (e.g., "false marketing," "illegal content") as $s_i = \max_k \frac{\mathbf{v}_i^\top \mathbf{t}_k}{\|\mathbf{v}_i\|\cdot\|\mathbf{t}_k\|}$. A BIN+TOP strategy selects 3 frames (temporal uniform coverage + globally highest-scoring complement), and InternViT-6B extracts patch-level salient regions.
    - *Design Motivation*: The majority of short-video frames are irrelevant; precise localization of risk-relevant visual cues is necessary. Temporal uniform sampling prevents omissions.

2. **Interleaved Multi-stage CoT Generation**:

    - *Function*: A frozen large model (InternVL-3-78B) synthesizes structured reasoning data.
    - *Mechanism*: A 4-step reasoning process—Step 1 (Observation) describes visual content and ASR transcription and assesses cross-modal consistency → Steps 2–3 (Risk Screening + Causal Analysis) identify violations and analyze their causes → Step 4 (Final Verdict) integrates reasoning for a compliance decision. Output format: `<think>reasoning process</think><answer>violation scenarios and types</answer>`.
    - *Design Motivation*: Synthetic data replaces manual annotation, substantially reducing cost; structured reasoning ensures interpretability.

3. **Rule-Anchored SFT**:

    - *Function*: Supervised fine-tuning of the base VLM using synthesized ICoT data.
    - *Mechanism*: The loss function is $\mathcal{L} = \mathcal{L}_{CE}(\langle\text{answer}\rangle) + \lambda \cdot \text{KL}(p_{\text{think}} \| p_{\text{rule}})$, where the primary term applies cross-entropy to the answer, and the auxiliary KL divergence term aligns the `<think>` reasoning distribution to a rule prior (violation scenario/type keywords normalized into a soft target distribution).
    - *Design Motivation*: The KL regularization ensures that the reasoning process does not deviate from policy rules—not only must the answer be correct, but the reasoning chain must also be consistent with the rules.

4. **SCA-R: Self-Consistency and Adaptive Reward**:

    - *Function*: Provides dynamic, policy-aware reasoning quality rewards during the RL stage.
    - *Mechanism*: A guide model serves as a scoring expert, receiving the `<think>` reasoning chain, ground truth, and moderation rules, and dynamically constructing scoring criteria $\mathcal{P} = \{p_k\}$ (e.g., causal clarity, risk attribution), each with weight $w_k$. The final reward is $r_{\text{scaR}} = \sum_k w_k \cdot \text{score}_{p_k}(\hat{y})$.
    - *Design Motivation*: Fixed rewards cannot adapt to policy drift. SCA-R adaptively adjusts scoring dimensions per scenario, ensuring the pre-screening model remains continuously aligned with the latest policies.

### Loss & Training
Total reward: $r = r_{\text{rule}} + r_{\text{format}} + r_{\text{scaR}}$. GRPO improvements include: token-level normalization (to avoid reward bias from output length variation), dynamic sampling (skipping batches where all group rewards are identical to prevent gradient collapse), and clip factor annealing.

## Key Experimental Results

### Main Results

| Model | Strict Acc. | Wide Acc. | Risky Precision | Risky F1 | Consistency |
|------|------------|-----------|----------------|----------|-------------|
| Qwen2.5-VL-7B | 0.701 | 0.712 | 0.831 | 0.680 | 0.642 |
| Qwen2.5-VL-32B | 0.682 | 0.703 | 0.769 | 0.801 | 0.667 |
| Kimi-VL-A3B-Thinking | 0.511 | 0.529 | 0.588 | 0.711 | 0.701 |
| InternVL3-14B | 0.502 | 0.521 | — | — | — |
| **BLM-Guard (7B)** | **0.914** | **0.976** | **0.962** | **0.969** | **0.845** |

### Ablation Study

| Configuration | Strict Acc. | Risky Precision | Risky F1 | Consistency |
|------|------------|----------------|----------|-------------|
| Ans-SFT | 0.648 | 0.765 | 0.732 | 0.412 |
| Think-SFT | 0.612 | 0.720 | 0.699 | 0.585 |
| Rule-SFT (5k) | 0.783 | 0.882 | 0.867 | 0.776 |
| + Rule-RL | 0.801 | 0.915 | 0.894 | 0.781 |
| **+ SCA-R (Full)** | **0.914** | **0.976** | **0.969** | **0.845** |

### Key Findings
- **The leap from 0.783 to 0.914**: The SCA-R adaptive reward is the largest single source of performance gain (+11.3% Strict Acc.), far exceeding the incremental improvement from Rule-RL alone.
- **Reasoning consistency correlates with accuracy**: BLM-Guard is the only model to substantially surpass all baselines on both accuracy and consistency simultaneously, demonstrating that structured reasoning is essential for moderation tasks.
- **A 7B model outperforms a 32B generalist**: BLM-Guard (7B) at 91.4% far exceeds Qwen2.5-VL-32B at 68.2%, reaffirming the power of domain specialization.
- **Answer-only vs. reasoning-only training**: Ans-SFT (answer supervision only) achieves slightly higher accuracy but extremely poor consistency (0.412); Think-SFT (reasoning supervision only) yields better consistency but lower accuracy—combining both is critical.
- **Generalization to 5 public datasets**: Performance is particularly strong on FakeSV/FVC (misinformation detection), indicating that cross-modal inconsistency detection capability transfers broadly.

## Highlights & Insights
- The **ICoT data synthesis pipeline** is a pragmatic innovation—using a large model to synthesize structured reasoning data for training a smaller model substantially reduces annotation cost while maintaining reasoning quality.
- The **scene-adaptive design of SCA-R** addresses a practical pain point: platform moderation rules change frequently, and fixed reward functions cannot track policy drift; adaptive scoring dimensions and weights enable the system to remain continuously aligned.
- The idea of **KL regularization to align reasoning with rule priors** is elegant—it requires not only correct answers, but also the presence of correct violation keywords in the reasoning chain, ensuring the model understands both the conclusion and its justification.
- **Industrial practice from Kuaishou**: The dataset is drawn from real short-video advertisements spanning e-commerce, healthcare, and education, with a practical three-tier violation taxonomy (severity–scenario–type).

## Limitations & Future Work
- The BLM-Guard Benchmark is a private dataset that is not publicly released, making reproduction and comparison difficult.
- SCA-R depends on a guide model (likely GPT-4o), introducing additional inference cost and potential bias.
- Only ASR is used, not OCR (due to high noise in short-video OCR), yet some ad violations appear precisely in text overlay layers.
- The paper lacks concrete dataset scale descriptions (number of training/test samples).
- Latency is not discussed—it remains unclear whether a 7B model performing multi-frame video inference meets real-world moderation throughput requirements.

## Related Work & Insights
- **vs. LlamaGuard/LlavaGuard**: These models perform coarse-grained safety detection (violence/pornography), while BLM-Guard performs fine-grained policy compliance detection (false marketing, income exaggeration, etc.)—the problem dimensions are entirely different.
- **vs. SafeWatch**: SafeWatch also addresses video safety moderation but lacks structured reasoning and policy-aligned RL; BLM-Guard offers substantially stronger interpretability and adaptability.
- **Implications for Agent research**: The scene-adaptive reward design of SCA-R is transferable to any RL task requiring dynamic alignment—such as agent behavior alignment or value alignment in dialogue systems.

## Rating
- Novelty: ⭐⭐⭐⭐ ICoT data synthesis and SCA-R adaptive reward are notable contributions, though the overall framework (two-stage SFT+RL) follows a relatively standard pipeline.
- Experimental Thoroughness: ⭐⭐⭐⭐ Ablations are comprehensive and generalization to 5 public datasets is demonstrated, but the private benchmark limits reproducibility.
- Writing Quality: ⭐⭐⭐⭐ Method descriptions are clear and figures are well-designed, though notation is occasionally inconsistent.
- Value: ⭐⭐⭐⭐ Ad compliance moderation is a genuine and important industrial need; the proposed approach has practical deployment value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] TumorChain: Interleaved Multimodal Chain-of-Thought Reasoning for Traceable Clinical Tumor Analysis](../../ICLR2026/llm_reasoning/tumorchain_interleaved_multimodal_chain-of-thought_reasoning_for_traceable_clini.md)
- [\[ACL 2025\] MM-Verify: Enhancing Multimodal Reasoning with Chain-of-Thought Verification](../../ACL2025/llm_reasoning/mm-verify_enhancing_multimodal_reasoning_with_chain-of-thought_verification.md)
- [\[ICLR 2026\] Random Policy Valuation is Enough for LLM Reasoning with Verifiable Rewards](../../ICLR2026/llm_reasoning/random_policy_valuation_is_enough_for_llm_reasoning_with_verifiable_rewards.md)
- [\[CVPR 2026\] E-comIQ-ZH: A Human-Aligned Dataset and Benchmark for Fine-Grained Evaluation of E-commerce Posters with Chain-of-Thought](../../CVPR2026/llm_reasoning/e-comiq-zh_a_human-aligned_dataset_and_benchmark_for_fine-grained_evaluation_of_.md)
- [\[ACL 2026\] AIM-CoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](../../ACL2026/llm_reasoning/aim-cot_active_information-driven_multimodal_chain-of-thought_for_vision-languag.md)

</div>

<!-- RELATED:END -->
