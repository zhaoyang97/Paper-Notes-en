---
title: >-
  [Paper Note] InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model
description: >-
  [ACL 2025][Multimodal VLM][Multi-modal reward model] Based on InternLM-XComposer2.5, a discriminative multi-modal reward model named IXC-2.5-Reward is constructed. By training on a meticulously curated preference dataset spanning multiple domains across text, images, and videos, it surpasses GPT-4o (62.4%) with a 70.0% Macro Acc on the multi-modal reward benchmark VL-RewardBench. Furthermore, three downstream applications—RL training, Best-of-N test-time scaling…
tags:
  - "ACL 2025"
  - "Multimodal VLM"
  - "Multi-modal reward model"
  - "RLHF"
  - "preference alignment"
  - "PPO"
  - "test-time scaling"
  - "data cleaning"
date: 2026-05-08
content_hash: 3f44ffe2df125e3a
---

# InternLM-XComposer2.5-Reward: A Simple Yet Effective Multi-Modal Reward Model

**Conference**: ACL 2025  
**arXiv**: [2501.12368](https://arxiv.org/abs/2501.12368)  
**Authors**: Yuhang Zang, Xiaoyi Dong, Pan Zhang, Yuhang Cao et al. (Shanghai AI Laboratory, CUHK, etc.)
**Code**: [GitHub](https://github.com/InternLM/InternLM-XComposer)  
**Area**: Multi-Modal VLM  
**Keywords**: Multi-modal reward model, RLHF, preference alignment, PPO, test-time scaling, data cleaning  

## TL;DR

Based on InternLM-XComposer2.5, a discriminative multi-modal reward model named IXC-2.5-Reward is constructed. By training on a meticulously curated preference dataset spanning multiple domains across text, images, and videos, it surpasses GPT-4o (62.4%) with a 70.0% Macro Acc on the multi-modal reward benchmark VL-RewardBench. Furthermore, three downstream applications—RL training, Best-of-N test-time scaling, and data cleaning—are successfully demonstrated.

## Background & Motivation

### Background
Reward models (RMs) have been extensively studied in the LLM domain as crucial components for RLHF training and test-time scaling. However, in the realm of large vision-language models (LVLMs), publicly available multi-modal reward models are extremely scarce, and the implementation details of closed-source models remain opaque.

### Limitations of Prior Work
- **Domain Restrictions**: Existing multi-modal RMs (such as RLAIF-V, LLaVA-Critic) are mostly restricted to specific areas like reducing hallucinations, lacking coverage over diverse domains such as instruction following, safety, and reasoning.
- **Weak Base Models**: Some prior works employ weaker base models, resulting in multi-modal RMs that lag significantly behind text-only RMs.
- **Scarcity of Preference Data**: Existing preference data is predominantly text-centric, with limited image preference data and almost non-existent video preference data.
- **Limitations of Generative RMs**: Generative RMs that evaluate via prompting LVLMs exhibit weaker discriminative capabilities than discriminative RMs.

### Goal
To build a cross-modal (text + image + video) and multi-domain (instruction following, general understanding, document understanding, mathematical reasoning, video understanding) general discriminative multi-modal reward model, and to validate its practical value in RL training, test-time scaling, and data cleaning.

## Method

### Overall Architecture
IXC-2.5-Reward is built upon the SFT model of InternLM-XComposer2.5 (IXC-2.5). The pre-trained vision encoder and MLP projection layer are kept frozen, while the final linear layer is replaced with a **score head** that maps the average hidden state features of all tokens to a scalar reward score $r(x, y)$. During training, the vision encoder and projection layer remain frozen, and only the LLM (InternLM2) and the score head are trained.

### Key Designs

#### Key Design 1: Multi-Domain Multi-Modal Preference Data Construction

Data sources consist of two parts:

**Open-source data** (primarily textual): includes Tulu-3 instruction-following data, UltraFeedback general feedback, HH-RLHF/PKU-Safe safety data, WildVision-Battle multi-modal conversations, and image preference data such as LLaVA-Critic/VL-Feedback/RLAIF-V.

**Newly collected data** (focusing on filling the gap in images and videos):
- Images: covers instruction following (MM-IFDPO-23k), knowledge VQA (KVQA, A-OKVQA, PMC-VQA), document understanding (AI2D, ChartQA, DVQA, etc.), and mathematical reasoning (GeoQA, CLEVR-Math, TabMWP, etc.).
- Videos: TrafficQA, FunQA, MiraData.

Preference annotation pipeline for the new data: The IXC-2.5 SFT model is used to generate multiple outputs for each prompt. Subsequently, pairwise evaluation is conducted using **GPT-4o** (for general/document categories) or **validator functions against ground-truth** (for math/instruction-following categories) to determine the "chosen" and "rejected" responses.

#### Key Design 2: Length Constraint Debiasing

During training on preference data, pairs where the chosen response is significantly longer than the rejected response are removed. This prevents the reward model from equating "length" with "quality". This is essential because LLM-as-a-Judge evaluations exhibit a strong length bias—GPT-4o tends to award higher scores to longer responses.

Ablation studies show that removing length constraints increases the WildVision score from 74.6 to 76.2 (as the Judge prefers longer responses), but the average token length surges from 274 to 361, which degrades the actual user experience. The authors retain length constraints to optimize real user experience rather than blindly chasing benchmark scores.

#### Key Design 3: Three Downstream Applications

**Application 1 — PPO Reinforcement Learning**: IXC-2.5-Reward is utilized as the reward signal to train the policy model IXC-2.5-Chat via the PPO algorithm. The critic model is initialized from IXC-2.5-Reward, advantages are estimated using GAE, and the policy is updated with the PPO clipped objective. The training data primarily covers instruction following and open-ended dialogue.

**Application 2 — Best-of-N Test-Time Scaling**: For each prompt, $N$ candidates are generated using IXC-2.5-Chat, scored by IXC-2.5-Reward, and the response with the highest score is selected. Significant improvements are observed even at $N=4$.

**Application 3 — Data Cleaning**: Samples with low reward scores are strongly correlated with problematic instances, such as hallucinations, empty responses, and text-image mismatches. Consequently, the RM can be used to filter out noise in pre-training and post-training datasets.

### Loss & Training
- Reward model learning rate: 1e-5, batch size 256
- PPO policy model learning rate: 5e-5, batch size 256
- PPO hyperparameters: $\gamma=0.99$, $\beta=0.95$, $\epsilon=0.2$
- Loss function: Bradley-Terry preference loss:
$$\mathcal{L}_{\text{RM}} = -\mathbb{E}[\log\sigma(r(x, y_w) - r(x, y_l))]$$

## Key Experimental Results

### Table 1: Multi-Modal Reward Benchmark Results on VL-RewardBench

| Model | Params | General | Hallucination | Reasoning | Overall Acc | Macro Acc |
|------|--------|---------|---------------|-----------|-------------|-----------|
| GPT-4o | - | 49.1 | 67.6 | 70.5 | 65.8 | 62.4 |
| Gemini-1.5-Pro | - | 50.8 | 72.5 | 64.2 | 67.2 | 62.5 |
| LLaVA-Critic-8B | 8B | 54.6 | 38.3 | 59.1 | 41.2 | 44.0 |
| InternVL2-8B | 8B | 35.6 | 41.1 | 59.0 | 44.5 | 45.2 |
| Llama-3.2-90B | 90B | 42.6 | 57.3 | 61.7 | 56.2 | 53.9 |
| **IXC-2.5-Reward** | **7B** | **84.7** | **62.5** | **62.9** | **65.8** | **70.0** |

IXC-2.5-Reward surpasses all models (including closed-source ones) in Macro Acc with only 7B parameters, leading significantly in the General category with 84.7%.

### Table 2: Instruction Following and Dialogue Evaluation of IXC-2.5-Chat (≤10B Open-Source Models)

| Benchmark | Type | Closed-source SOTA | Open-source SOTA | IXC-2.5 (SFT) | IXC-2.5-Chat (PPO) |
|------|------|---------|---------|---------------|-------------------|
| WildVision | Open | 89.2 (GPT-4o) | 67.3 | 37.5 | **74.6** |
| MIA-bench | Open | 88.6 (GPT-4o) | 80.7 | 80.4 | **84.0** |
| MM-MT | Open | 7.72 (GPT-4o) | 5.45 | 3.85 | **5.70** |
| MM-Vet v2 | Open | 71.8 | 58.1 | 45.8 | **54.8** |

Substantial improvements are achieved in instruction following and open dialogue after PPO training (WildVision increases from 37.5 to 74.6), while keeping capabilities like knowledge, reasoning, and document understanding intact.

### Table 3: Best-of-N Test-Time Scaling Results

| Setting | Avg Tokens | WildVision | MIA | MM-MT | MM-Vet v2 |
|------|-----------|------------|-----|-------|-----------|
| IXC-2.5-Chat | 274 | 74.6 | 84.0 | 5.70 | 54.8 |
| IXC-2.5-Chat + BoN (N=4) | 283 | **77.7** | **87.3** | **6.03** | **56.3** |

BoN sampling further enhances performance on top of PPO, with average token length only increasing from 274 to 283, demonstrating that the gain stems from response quality rather than length exploitation.

### Table 4: Ablation Study on Length Constraints

| Setting | Avg Tokens | WildVision | MIA | MM-MT | MM-Vet v2 |
|------|-----------|------------|-----|-------|-----------|
| W/o Length Constraints | 361 | 76.2 | 87.0 | 5.86 | 56.6 |
| W/ Length Constraints (Final) | 274 | 74.6 | 84.0 | 5.70 | 54.8 |

## Key Findings

1. **Discriminative RMs significantly outperform generative RMs**: On VL-RewardBench, the 7B discriminative IXC-2.5-Reward (70.0%) substantially outperforms the 90B Llama-3.2 generative RM (53.9%), indicating that a specifically trained score head is much more effective than prompt-based evaluation.
2. **Multi-modal RMs can preserve language capabilities**: IXC-2.5-Reward achieves 88.6% on the text-only RewardBench and 68.8% on RM-Bench, which is close to specialized text-only RMs (such as InternLM2-7B-Reward at 87.6%).
3. **Length preference is a systemic issue in multi-modal evaluation**: Length bias exists not only in LLM-as-a-Judge evaluations but also in multi-modal VQA evaluations. Removing length constraints leads to higher benchmark scores but degrades the actual user experience.
4. **PPO training does not compromise other capabilities**: On metrics testing knowledge (MMBench, MMMU), reasoning (MathVista), and document understanding (ChartQA), the PPO-trained Chat model performs on par with the SFT baseline.

## Highlights & Insights

- **Simple yet effective architecture**: Instead of introducing complex multi-modal alignment modules, the model simply adds a score head to an already-aligned LVLM and trains on preference data. This approach is straightforward yet highly effective.
- **Data-centric design philosophy**: Broad generalizability across domains is achieved by systematically supplementing preference data across image, video, document, and reasoning domains, rather than modifying the model architecture.
- **Pragmatic length-debiasing strategy**: It is clearly pointed out that "higher benchmark scores do not equate to a better user experience." The authors actively sacrifice higher benchmark scores to ensure superior conversation quality in practice.
- **Threefold application validation**: Beyond demonstrating the accuracy of the RM itself, its practical utility is validated through RL training, BoN sampling, and data cleaning.
- **Overwhelming advantage in the General category**: Achieving 84.7% compared to the second place (54.6%) indicates that discriminative RMs hold a unique advantage in making tie-breaking decisions.

## Limitations & Future Work

- **Monolingual English limitation**: The training data is predominantly English, resulting in limited multilingual capabilities and potentially poorer performance in non-English scenarios.
- **Outcome RMs only (No PRM)**: The model performs outcome-level scoring (Outcome RM) without incorporating process-level rewards (Process RM), lacking robust support for scenarios like math reasoning that require step-by-step verification.
- **Simplistic score head design**: It aggregates scores using the average hidden states of all tokens, leaving alternative methods such as attention pooling or using only the last token unexplored.
- **PPO efficiency issues**: Running PPO requires maintaining four models simultaneously (policy, reference, reward, and critic), incurring high computational overhead. More efficient RL algorithms like GRPO are not compared.
- **Lack of quantitative validation for data cleaning**: The work only provides qualitative visualizations of low-score samples without offering quantitative ablation studies on retraining the LVLM on cleaned data.
- **Limited video preference data**: Video data is sourced from only three datasets, representing a small domain coverage and scale.

## Related Work & Insights

- **LLaVA-Critic**: Uses LVLMs for generative evaluation but achieves only 44.0% on VL-RewardBench, indicating a relatively low ceiling for prompt-based approaches.
- **RLAIF-V**: Employs AI feedback instead of human feedback for visual alignment. This work directly incorporates its output as part of the training data.
- **Tulu-3**: Demonstrates the effectiveness of the PPO+RM paradigm in the LLM domain, which this work fully migrates to the multi-modal domain.
- **Insights**: The core bottleneck for multi-modal RMs lies in the availability of preference data rather than model architecture; constructing high-quality, multi-domain preference datasets is a crucial area of investment. Additionally, length bias remains a severe issue in multi-modal evaluation, calling for improved community-wide evaluation protocols.

## Rating

- Novelty: ⭐⭐⭐ — The methodology is a combination and migration of matured techniques; the core innovation lies in data curation and engineering practices.
- Experimental Thoroughness: ⭐⭐⭐⭐ — Comprehensive coverage across 3 RM benchmarks and over 12 downstream tasks, complete with ablation and application validation.
- Writing Quality: ⭐⭐⭐⭐ — Well-structured, fully motivated, with abundant tables and visualizations.
- Value: ⭐⭐⭐⭐ — Fills a gap in open-source multi-modal RMs, providing a practical utility tool and a paradigm for data curation to the community.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] BaseReward: A Strong Baseline for Multimodal Reward Model](../../ICLR2026/multimodal_vlm/basereward_a_strong_baseline_for_multimodal_reward_model.md)
- [\[ACL 2025\] Agent-RewardBench: Towards a Unified Benchmark for Reward Modeling across Perception, Planning, and Safety in Real-World Multimodal Agents](agent_rewardbench.md)
- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](../../ICCV2025/multimodal_vlm/controlling_multimodal_llms_via_rewardguided_decoding.md)
- [\[ICML 2025\] The Devil Is in the Details: Tackling Unimodal Spurious Correlations for Generalizable Multimodal Reward Models](../../ICML2025/multimodal_vlm/the_devil_is_in_the_details_tackling_unimodal_spurious_correlations_for_generali.md)
- [\[NeurIPS 2025\] A Frustratingly Simple Yet Highly Effective Attack Baseline: Over 90% Success Rate Against the Strong Black-box Models of GPT-4.5/4o/o1](../../NeurIPS2025/multimodal_vlm/a_frustratingly_simple_yet_highly_effective_attack_baseline.md)

</div>

<!-- RELATED:END -->
