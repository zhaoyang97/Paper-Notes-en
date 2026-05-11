---
title: >-
  [Paper Note] Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models
description: >-
  [ICLR 2026][Multimodal VLM][Multimodal Reasoning] This paper proposes Vision-R1, which constructs 200K high-quality multimodal CoT data via Modality Bridging for cold-start initialization…
tags:
  - "ICLR 2026"
  - "Multimodal VLM"
  - "Multimodal Reasoning"
  - "Reinforcement Learning"
  - "Chain-of-Thought"
  - "GRPO"
  - "Cold-Start Initialization"
date: 2026-05-08
content_hash: 88f8443df381248e
---

# Vision-R1: Incentivizing Reasoning Capability in Multimodal Large Language Models

**Conference**: ICLR 2026
**arXiv**: [2503.06749](https://arxiv.org/abs/2503.06749)
**Code**: [GitHub](https://github.com/Osilly/Vision-R1)
**Area**: Multimodal VLM
**Keywords**: Multimodal Reasoning, Reinforcement Learning, Chain-of-Thought, GRPO, Cold-Start Initialization

## TL;DR

This paper proposes Vision-R1, which constructs 200K high-quality multimodal CoT data via Modality Bridging for cold-start initialization, followed by Progressive Thinking Suppression Training (PTST) combined with GRPO reinforcement learning. At the 7B parameter scale, Vision-R1 achieves multimodal mathematical reasoning performance approaching OpenAI O1.

## Background & Motivation

DeepSeek-R1 successfully demonstrated that pure RL can elicit complex reasoning capabilities in LLMs (e.g., self-reflection, self-questioning). However, whether this success can be transferred to Multimodal LLMs (MLLMs) remains an open question.

The authors first attempted to train MLLMs directly with RL (termed Vision-R1-Zero), identifying the following key challenges:

**Direct RL training fails to elicit complex reasoning**: Due to the lack of large-scale, high-quality multimodal reasoning data, the model cannot generate complex CoT.

**Insufficient quality of existing multimodal CoT data**: Existing data lacks human cognitive processes such as self-reflection and self-questioning, amounting to formatted "pseudo-CoT."

**Overthinking after cold-start initialization**: After SFT on CoT data, the model generates excessively long reasoning chains, while correct reasoning is concentrated in shorter chains, making RL optimization difficult.

## Method

### Overall Architecture

A two-stage pipeline: (1) construct the Vision-R1-cold dataset → cold-start SFT to obtain Vision-R1-CI; (2) Progressive Thinking Suppression Training (PTST) + GRPO reinforcement learning → final Vision-R1.

### Key Designs

1. **Modality Bridging for Multimodal CoT Data Construction**: DeepSeek-R1 can generate human-like complex reasoning but cannot process images directly as a text-only model. The solution proceeds in three steps:

   - Input image-text pairs into an MLLM to generate "pseudo-CoT" (containing image descriptions and reasoning processes), exposing richer visual details.
   - Feed the "pseudo-CoT" along with the original image-text pair back into the MLLM to obtain detailed descriptions — achieving **modality bridging**.
   - Pass the pure-text descriptions into DeepSeek-R1 to obtain high-quality complex CoT.

   After rule-based filtering, this yields the 200K Vision-R1-cold dataset, in which "Wait" appears 585K times (vs. 2.3K in LLaVA-CoT), demonstrating significantly richer self-reflection characteristics.

2. **Overthinking Optimization Problem**: After cold-start initialization, the model tends to generate extremely long reasoning chains for all problems, while correct answers are predominantly found in shorter chains. Directly applying RL with a 16K context length causes the model to generate longer but incorrect reasoning, degrading performance.

3. **Progressive Thinking Suppression Training (PTST)**: Training is divided into stages. In early stages, reasoning length is strictly constrained (e.g., 4K tokens), forcing the model to learn correct reasoning within a limited budget. As training progresses, the constraint is gradually relaxed (e.g., 8K), allowing the model to autonomously learn to apply more complex reasoning for harder problems. Specifically: Stage 1 uses 4K×16 (length × number of samples), Stage 2 uses 8K×8, keeping the total sampling volume × length constant across stages. A **Hard Format Result Reward Function (HFRRF)** is employed: a reward of 1 is given only when both the format is correct **and** the answer is correct; otherwise the reward is 0.

### Loss & Training

GRPO objective with PTST:

$$J_{\text{GRPO}}^{(s)}(\theta) = \mathbb{E}\left[\frac{1}{G_s}\sum_{i=1}^{G_s}\min\left(\frac{\pi_\theta(o_i^{(s)}|q)}{\pi_{\theta_{\text{old}}}(o_i^{(s)}|q)}A_i^{(s)}, \text{clip}(\cdot, 1-\varepsilon, 1+\varepsilon)A_i^{(s)}\right) - \beta D_{\text{KL}}\right]$$

where $\varepsilon=0.2$, $\beta=10^{-2}$, and the advantage estimate is $A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$.

The cold-start stage applies standard SFT on Vision-R1-cold to fine-tune the base model (Qwen2.5-VL).

## Key Experimental Results

### Main Results

| Model | Params | MathVista | MathVerse | MM-Math | DynaMath | Avg. |
|-------|--------|-----------|-----------|---------|----------|------|
| OpenAI O1 | - | 73.9 | - | - | - | - |
| GPT-4o | - | 63.8 | 37.6 | 31.8 | 64.9 | - |
| Qwen2.5-VL-7B | 7B | 68.1 | 46.7 | 34.1 | 50.7 | 49.9 |
| Qwen2.5-VL-72B | 72B | 73.5 | 51.3 | 45.6 | 61.2 | 57.9 |
| **Vision-R1-7B** | **7B** | **73.5** | **52.4** | **40.2** | **56.3** | **55.6** |
| **Vision-R1-32B** | **32B** | **76.4** | **62.1** | **55.3** | **65.6** | **64.9** |
| **Vision-R1-72B** | **72B** | **78.2** | **63.2** | **59.3** | **66.4** | **66.8** |

Vision-R1-7B vs. base Qwen2.5-VL-7B: GEO +13.4, ALG +10.3, GPS +16.4, MathVista overall +5.4.

### Ablation Study

| Method | Cold Start | GRPO | PTST | Avg. Reasoning Length | Avg. Score (MathVista/MathVerse/MM-Math) |
|--------|-----------|------|------|-----------------------|------------------------------------------|
| Vision-R1-Zero | ✗ | ✓ | ✗ | 1285 | 50.7 |
| Vision-R1-CI | ✓ | ✗ | ✗ | 3566 | 44.5 |
| Vision-R1-Long | ✓ | ✓ | ✗ | 3107 | 47.7 |
| **Vision-R1** | **✓** | **✓** | **✓** | **2057** | **55.4** |

| PTST Config | Stage 1 | Stage 2 | MathVista | Avg. | Note |
|-------------|---------|---------|-----------|------|------|
| Fixed 16K | 16K×4 | 16K×4 | 70.3 | 47.7 | Severe overthinking with no early constraint |
| Fixed 4K | 4K×16 | 4K×16 | 72.6 | 54.3 | Effective but limits complex reasoning |
| **PTST 2-stage** | **4K×16** | **8K×8** | **73.5** | **55.4** | Optimal; progressive relaxation |
| PTST 3-stage | 4K×16 | 6K×12 → 8K×8 | 73.0 | 55.1 | Additional stage yields no significant gain |

### Key Findings

- **7B surpasses 70B**: Vision-R1-7B achieves 73.5% on MathVista, only 0.4% below OpenAI O1, outperforming Qwen2.5-VL-72B.
- **Direct RL training is insufficient**: Vision-R1-Zero achieves only 50.7 average score and fails to elicit effective reasoning.
- **Cold-start is necessary but not sufficient**: The CI model scores 44.5 (severe overthinking) and must be combined with PTST.
- **PTST is simple yet effective**: The two-stage setup (4K→8K) reaches optimum; additional stages provide no benefit, demonstrating robustness.
- **Data quality is critical**: "Wait" appears 586K times in Vision-R1-cold vs. only 2.3K in LLaVA-CoT — self-reflection token frequency is two orders of magnitude higher.
- Cross-model generalization is verified on Llama-3.2-11B-V: Vision-R1-cold SFT outperforms LLaVA-CoT and Mulberry across all benchmarks.

## Highlights & Insights

- This work is the first systematic exploration of R1-style RL in MLLMs, clearly delineating the individual contributions of direct RL, cold-start initialization, and PTST.
- Modality Bridging elegantly resolves the limitation that DeepSeek-R1 cannot process images.
- PTST offers a deep insight: the model should first learn to "reason correctly" before learning to "reason complexly," analogous to human learning progressions.
- Using only 10K data for RL yields approximately 6% average improvement, demonstrating exceptional data efficiency.
- The "Aha moment" is observed in MLLMs for the first time (e.g., self-correction and self-reflection).

## Limitations & Future Work

- RL training relies solely on mathematical data; generalization to general-purpose reasoning tasks remains to be validated.
- The number of stages and length settings in PTST are currently determined empirically, lacking theoretical grounding.
- Modality Bridging risks information loss during the visual-to-text conversion process.
- The 32B and 72B variants use additional data, making direct comparison with the 7B model not fully controlled.
- The scale of cold-start data (200K) may be a bottleneck; the benefits of larger-scale data remain to be explored.

## Related Work & Insights

- Vision-R1 serves as the multimodal counterpart to DeepSeek-R1, pointing toward a viable path for enhancing reasoning in MLLMs.
- The PTST methodology is applicable to other RL scenarios requiring control over generation length.
- The Modality Bridging approach can be generalized to other settings where text-only LLMs need to process multimodal data.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ First successful transfer of the R1-style reasoning paradigm to MLLMs; PTST strategy is original.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Multiple benchmarks (MathVista/MathVerse/MM-Math/DynaMath), multiple scales (7B/32B/72B), and extensive ablations.
- **Writing Quality**: ⭐⭐⭐⭐ Fluent exposition with a well-structured problem-driven narrative; notation is occasionally dense.
- **Value**: ⭐⭐⭐⭐⭐ Achieves O1-level multimodal reasoning at 7B parameters; significant inspiration for the research community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Shuffle-R1: Efficient RL Framework for Multimodal Large Language Models via Data-centric Dynamic Shuffle](shuffle-r1_efficient_rl_framework_for_multimodal_large_language_models_via_data-.md)
- [\[ICLR 2026\] DIVA-GRPO: Enhancing Multimodal Reasoning through Difficulty-Adaptive Variant Advantage](diva-grpo_enhancing_multimodal_reasoning_through_difficulty-adaptive_variant_adv.md)
- [\[NeurIPS 2025\] VideoRFT: Incentivizing Video Reasoning Capability in MLLMs via Reinforced Fine-Tuning](../../NeurIPS2025/multimodal_vlm/videorft_incentivizing_video_reasoning_capability_in_mllms_via_reinforced_fine-t.md)
- [\[NeurIPS 2025\] Video-R1: Reinforcing Video Reasoning in MLLMs](../../NeurIPS2025/multimodal_vlm/video-r1_reinforcing_video_reasoning_in_mllms.md)
- [\[CVPR 2026\] MUPO: All Roads Lead to Rome - Incentivizing Divergent Thinking in Vision-Language Models](../../CVPR2026/multimodal_vlm/mupo_all_roads_lead_to_rome_incentivizing_divergent_thinking_in_vlms.md)

</div>

<!-- RELATED:END -->
