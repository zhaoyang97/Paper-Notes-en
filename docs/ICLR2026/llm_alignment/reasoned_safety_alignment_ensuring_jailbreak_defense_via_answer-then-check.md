---
title: >-
  [Paper Note] Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check
description: >-
  [ICLR 2026][LLM Alignment][Jailbreak Defense] This paper proposes the "Answer-Then-Check" strategy: the model first generates an intended answer summary within its chain-of-thought, performs a safety analysis based on safety policies, and finally decides whether to output or refuse. After training on the constructed 80K ReSA dataset, the defense rate reaches 99.3% (RL version) against 7 types of jailbreak attacks, with 500 samples being sufficient to achieve performance compa…
tags:
  - "ICLR 2026"
  - "LLM Alignment"
  - "Jailbreak Defense"
  - "Answer-Then-Check"
  - "Safety Reasoning"
  - "Long Chain-of-Thought"
  - "Data-Efficient Alignment"
date: 2026-05-08
content_hash: e974ab882a7263d2
---

# Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check

**Conference**: ICLR 2026  
**arXiv**: [2509.11629](https://arxiv.org/abs/2509.11629)  
**Code**: [https://huggingface.co/datasets/ByteDance-Seed/ReSA](https://huggingface.co/datasets/ByteDance-Seed/ReSA)  
**Area**: AI Safety / LLM Alignment  
**Keywords**: Jailbreak Defense, Answer-Then-Check, Safety Reasoning, Long Chain-of-Thought, Data-Efficient Alignment

## TL;DR
This paper proposes the "Answer-Then-Check" strategy: the model first generates an intended answer summary within its chain-of-thought, performs a safety analysis based on safety policies, and finally decides whether to output or refuse. After training on the constructed 80K ReSA dataset, the defense rate reaches 99.3% (RL version) against 7 types of jailbreak attacks, with 500 samples being sufficient to achieve performance comparable to the full dataset.

## Background & Motivation

**Background**: LLM safety alignment employs methods like SFT/RLHF to ensure models refuse harmful requests. However, jailbreak attacks continue to evolve, bypassing safety mechanisms through role-playing (PAP), template variations (GPTFuzzer), and iterative optimization (PAIR/TAP).

**Limitations of Prior Work**:
   - Current alignment follows a "judge-then-answer" approach—the model decides whether to refuse or answer upon seeing the query, but malicious intent can be deeply disguised.
   - Post-hoc detection methods (e.g., LlamaGuard) can only perform direct refusal and cannot provide supportive safety responses for sensitive queries (e.g., self-harm).
   - Inference-time defense strategies (e.g., prompt engineering) have limited effectiveness as models are not sufficiently familiar with safety policies.

**Key Challenge**: Malicious intent may be deeply disguised at the query level, making it difficult to identify. However, once the model attempts to generate an answer, the harmful content is exposed—a critical asymmetry.

**Goal**: Utilize this asymmetry to design a defense where the model answers first (exposing intent) and then checks (identifying risk).

**Key Insight**: Combine with LongCoT (Chain-of-Thought), generating an answer summary and performing safety analysis during the `<think>` phase; only content that passes the check is presented to the user.

**Core Idea**: First attempt to answer to make malicious intent manifest, then review against safety policies = logical immunity to jailbreak attacks.

## Method

### Overall Architecture

This paper addresses the challenge of jailbreak attacks disguising malicious intent at the query level to bypass safety alignment. It leverages an attack-defense asymmetry: while intent can be hidden in a carefully constructed prompt, harmful content becomes unavoidable once the model begins drafting the actual response. The core mechanism of ReSA is to postpone safety judgment from "deciding whether to refuse upon seeing the question" to "reviewing the drafted answer."

The pipeline consists of training and inference phases. For training, the ReSA dataset is constructed—a general LLM generates approximately 80K samples in the format of "intended answer summary → safety analysis → final response" (including a small subset for self-harm safety completion). After two-stage filtering, SFT is performed, followed by optional GRPO reinforcement to obtain the ReSA model. During inference, the model enters a `<safety_check>` phase upon receiving a query: it writes an intended answer summary in `<intended_answer_summary>`, performs a safety analysis against policies, and concludes with `</safety_check>`. Only content generated after this tag is shown to the user. If the review passes, it replies normally; if harmful, it refuses; for sensitive topics like self-harm, it provides a supportive safety completion. An adaptive variant allows obviously benign queries to skip the entire check, maintaining latency comparable to the base model.

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}%%
flowchart TD
    subgraph TRAIN["ReSA Dataset Construction + SFT/RL"]
        direction TB
        D1["Four categories ~80K samples<br/>(inc. 524 self-harm safe completions)"] --> D2["Two-stage Filtering:<br/>LlamaGuard + Consistency Check"]
        D2 --> D3["SFT, optional GRPO<br/>(Intent summary + answer dual safety reward)"]
    end
    TRAIN --> M["ReSA Model"]
    M --> Q["User Query"]
    Q --> ADP{"Adaptive Decision:<br/>Is it a common request?"}
    ADP -->|Common Request| DIRECT["Direct Answer<br/>(Skip Inspection)"]
    ADP -->|Potentially Harmful| IAS["Intended Answer Summary<br/>(intended_answer_summary)"]
    IAS --> ANA["Safety analysis against policy<br/>(safety_check)"]
    ANA --> DEC{"Does the answer violate policy?"}
    DEC -->|Safe| ANS["Normal Reply"]
    DEC -->|Harmful/General| REF["Refusal"]
    DEC -->|Sensitive (e.g., self-harm)| SAFE["Safe Completion<br/>Supportive Reply"]
```

### Key Designs

**1. Answer-Then-Check Inference Template: Postponing Safety Judgment from Question to Answer**

Traditional alignment uses "judge-then-answer," where the model decides to refuse or answer based on the query, which jailbreak attacks exploit by disguising the query. This template reverses the order—intended answer summary → safety analysis → final response—forcing the model to summarize the answer first. This makes malicious intent manifest in the answer summary, which is then evaluated against safety policies. At this point, the content is explicit and impossible to hide. Structurally, the intent summary and safety analysis are enclosed in `<safety_check>...</safety_check>`, with the summary further wrapped in `<intended_answer_summary>`, and the final response following `</safety_check>`. Only content after `</safety_check>` is visible to the user. This contrasts with OpenAI's Deliberative Alignment; while the latter analyzes before answering, ReSA answers before analyzing, exposing malicious content more thoroughly and providing stronger resistance to disguised queries.

**2. ReSA Dataset Construction: Bulk Generation of ~80K Samples via General Models**

The template requires data to train the model, covering vanilla harmful/benign and adversarial harmful/benign categories. This ensures the model learns to answer-then-check for both clean and jailbreak queries. The process involves: using the uncensored Dolphin-2.9.2-Qwen2-72B to generate intended answers for harmful queries, Qwen2.5-72B for benign queries, and then compressing them into summaries; safety analysis is written by Llama3.3-70B against safety policies; adversarial samples are generated using PAIR, GPTFuzzer, and PAP. The final distribution includes 12K vanilla harmful, 16K vanilla benign, 23K adversarial harmful, and 29K adversarial benign samples (~80K total). Quality is ensured through two-stage filtering: first, LlamaGuard classification (retaining only "benign labeled safe" and "harmful labeled unsafe"), followed by a consistency check to remove samples where the safety analysis contradicts the answer content. Notably, the dataset relies solely on general LLMs (Dolphin/Qwen2.5/Llama3.3) rather than o1-level reasoning models, reducing the barrier to entry.

**3. Safe Completion: Beyond the Binary "Refusal" for Sensitive Topics**

Post-hoc detection defenses often result in a blunt refusal. However, for high-risk sensitive queries like self-harm, an abrupt refusal can be damaging. Safe Completion addresses this by extracting 524 self-harm samples (167 vanilla harmful + 357 adversarial harmful) using LlamaGuard. For vanilla self-harm queries, the final answer in the template is replaced with a caring response from a general LLM. For adversarial self-harm queries, the model is paired with the corresponding vanilla query to recognize the malicious intent and generate a supportive reply. This provides an alternative to simple refusal or compliance, ensuring the model is neither unsafe nor indifferent toward sensitive topics. Experiments show that with even a small amount of carefully constructed data, the model can identify intent under adversarial prompts and provide safety-aligned responses.

**4. Two Variants: Adaptive to Reduce Overhead + RL for Internal CoT Management**

The basic template adds the overhead of generating an intent summary and safety analysis for every query. Two variants address this. **Adaptive Answer-Then-Check** improves efficiency by mixing in instruction-tuning samples that do not use the full template, allowing the model to learn to skip the review for obviously benign queries. This brings latency back to base model levels for normal requests without sacrificing safety performance. **ReSA-RL** enhances robustness using GRPO reinforcement on top of SFT. The reward consists of three parts: a safety reward using LlamaGuard to **simultaneously** evaluate if the intended answer summary $o_{\text{intended}}$ and final answer $o_{\text{ans}}$ are safe, a refusal reward (determined by Qwen2.5-7B) to suppress over-refusal of benign queries, and a format reward to maintain the Answer-Then-Check structure. Crucially, scoring the intent summary within the CoT ensures the entire generation process remains safe, closing the "CoT harmful content leakage" loophole.

### Loss & Training
- SFT: Standard cross-entropy, AdamW + cosine schedule, lr=5e-6, 2 epochs.
- RL: GRPO, Reward = $\lambda_{\text{safety}} \cdot (R_{\text{safety}}(o_{\text{intended}}) + R_{\text{safety}}(o_{\text{ans}})) + \lambda_{\text{format}} \cdot R_{\text{format}}(o) + \lambda_{\text{refusal}} \cdot R_{\text{refusal}}(o_{\text{ans}})$.

## Key Experimental Results

### Main Results: Jailbreak Defense Rate (LlamaGuard evaluation, Llama3.1-8B-Instruct)

| Method | None | PAIR-GPT | PAIR | PAP | GPTFuzzer | ReNeLLM | TAP | DeepInception | Avg |
|------|------|----------|------|-----|-----------|---------|-----|--------------|-----|
| Base | 99.7 | 35.1 | 26.2 | 64.9 | 13.7 | 66.1 | 42.5 | 52.4 | 50.1 |
| Post-hoc LlamaGuard | 100 | 46.3 | 50.8 | 71.6 | 99.7 | 93.0 | 65.8 | 97.8 | 78.1 |
| STAIR-DPO | 100 | 68.4 | 42.2 | 94.3 | 100 | 83.4 | 69.3 | 98.7 | 82.0 |
| WJ-SFT | 99.4 | 44.7 | 32.9 | 76.0 | 94.3 | 67.7 | 60.4 | 98.4 | 71.7 |
| **ReSA-SFT** | 99.4 | 89.8 | 69.7 | 96.8 | 95.5 | 88.2 | 85.0 | 99.4 | **90.5** |
| **ReSA-RL** | 100 | 98.7 | 96.8 | 99.7 | 100 | 99.7 | 99.7 | 100 | **99.3** |

### Ablation Study: Impact of Data Volume

| Training Samples | 500 | 1K | 5K | 80K |
|-----------|-----|----|----|-----|
| Avg Defense Rate (LlamaGuard) | ~89% | ~89% | ~90% | 90.5% |
| Note | Close to full dataset performance | Diminishing marginal returns | Near saturation | Full dataset |

### Key Findings
- **ReSA-RL is nearly perfect**: Achieves an average defense rate of 99.3%, approaching 100% on most attacks, significantly outperforming all baselines.
- **500 samples are sufficient**: Just 500 samples achieve performance close to the full dataset, verifying the data efficiency of safety alignment.
- **Safety rewards on intent summaries are vital**: Ensures the internal chain-of-thought is also secure.
- **SFT significantly outperforms methods like STAIR/WJ**: Improving from 82% to 90.5% demonstrates the inherent effectiveness of the Answer-Then-Check template.
- **Adaptive version is efficient**: Normal queries do not trigger extra reasoning, matching the latency of the base model.

## Highlights & Insights
- **Leveraging Attack-Defense Asymmetry**: Jailbreak attacks disguise intent at the query level, but malicious content cannot be hidden once answering begins. Answer-Then-Check utilizes this sophisticated insight.
- **Safe Completion vs. Blanket Refusal**: Providing supportive responses for sensitive topics like self-harm is a rare but critical capability in defense methods.
- **Independence from Reasoning Models for Data Generation**: Unlike OpenAI’s Deliberative Alignment, ReSA uses general LLMs (Qwen2.5/Llama3.3) to build training data, lowering the barrier.
- **Dual Safety Rewards in RL**: Applying safety rewards to both the final answer and the intent summary ensures safety within the CoT, preventing "CoT leakage" of harmful content.

## Limitations & Future Work
- **Efficiency Overhead**: Despite the adaptive version, Answer-Then-Check still requires generating summaries and analyses, increasing computation.
- **LlamaGuard Dependency**: Both data construction and RL rewards rely on LlamaGuard's classification accuracy.
- **Safety Policy Coverage**: The safety policies in the training data must be pre-defined and may not cover emerging risk types.
- **Future Directions**: Potentially integrating SSAH’s safety unit freezing strategy to protect safety-critical neurons from being compromised by downstream tasks during ReSA fine-tuning.

## Related Work & Insights
- **vs. STAIR-DPO**: STAIR uses DPO for safety reasoning alignment but achieves much lower performance (82%) than ReSA (90.5%/99.3%) due to the lack of an explicit Answer-Then-Check structure.
- **vs. OpenAI Deliberative Alignment**: OpenAI reviews before answering, while ReSA answers before reviewing. The latter is more effective against disguised queries and does not require o1-level models.
- **vs. Post-hoc detection (LlamaGuard)**: Post-hoc detection (78.1%) is significantly less effective than ReSA (90.5%) and cannot perform Safe Completion.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ The core insight of Answer-Then-Check (utilizing asymmetry) is highly sophisticated.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 7 attacks × 3 evaluators × 2 models, 13 baseline comparisons; very comprehensive.
- Writing Quality: ⭐⭐⭐⭐ Clear structure with detailed methodology.
- Value: ⭐⭐⭐⭐⭐ Practical and efficient; 500-sample threshold is very low, and the RL version offers near-perfect defense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] SafePTR: Token-Level Jailbreak Defense in Multimodal LLMs via Prune-then-Restore Mechanism](../../NeurIPS2025/llm_alignment/safeptr_token-level_jailbreak_defense_in_multimodal_llms_via_prune-then-restore_.md)
- [\[ICLR 2026\] Superficial Safety Alignment Hypothesis](superficial_safety_alignment_hypothesis.md)
- [\[CVPR 2026\] Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](../../CVPR2026/llm_alignment/principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)
- [\[AAAI 2026\] AlignTree: Efficient Defense Against LLM Jailbreak Attacks](../../AAAI2026/llm_alignment/aligntree_efficient_defense_against_llm_jailbreak_attacks.md)
- [\[ICLR 2026\] Alignment-Weighted DPO: A Principled Reasoning Approach to Improve Safety Alignment](alignment-weighted_dpo_a_principled_reasoning_approach_to_improve_safety_alignme.md)

</div>

<!-- RELATED:END -->
