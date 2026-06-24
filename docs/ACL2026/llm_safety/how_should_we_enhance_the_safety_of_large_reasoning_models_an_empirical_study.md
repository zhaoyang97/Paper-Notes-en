---
title: >-
  [Paper Note] How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study
description: >-
  [ACL 2026][LLM Safety][Reasoning model safety] This paper systematically investigates how to enhance the safety of Large Reasoning Models (LRMs) through SFT. It identifies that the root cause of the limited effectiveness of direct safety response distillation is five risk reasoning patterns (especially "weak vacillation"). The authors propose targeted distillation strategies that reduce the PAIR attack success rate from 63% to 13%, and find that short reasoning chains and tem…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Reasoning model safety"
  - "risk reasoning patterns"
  - "weak vacillation"
  - "safety distillation"
  - "short reasoning chains."
date: 2026-05-08
content_hash: 29e15d9f75b38b13
---

# How Should We Enhance the Safety of Large Reasoning Models: An Empirical Study

**Conference**: ACL 2026  
**arXiv**: [2505.15404](https://arxiv.org/abs/2505.15404)  
**Code**: [GitHub](https://github.com/thu-coai/LRM-Safety-Study)  
**Area**: LLM Safety / Reasoning Models  
**Keywords**: Reasoning model safety, risk reasoning patterns, weak vacillation, safety distillation, short reasoning chains.

## TL;DR

This paper systematically investigates how to enhance the safety of Large Reasoning Models (LRMs) through SFT. It identifies that the root cause of the limited effectiveness of direct safety response distillation is five risk reasoning patterns (especially "weak vacillation"). The authors propose targeted distillation strategies that reduce the PAIR attack success rate from 63% to 13%, and find that short reasoning chains and template reasoning perform comparably to long reasoning chains in terms of safety.

## Background & Motivation

**Background**: Large reasoning models such as DeepSeek-R1 have achieved significant success in reasoning-intensive tasks like mathematics and programming. However, their enhanced reasoning capabilities do not automatically translate into improved safety performance—and in some cases, safety even degrades. For example, an LRM might generate detailed criminal plans within intermediate reasoning steps or final outputs.

**Limitations of Prior Work**: (1) The effectiveness of direct safety response distillation from DeepSeek-R1 is far below expectations—after filtering with ReasoningShield and fine-tuning, the ASR of PAIR only dropped from 66% to 54%; (2) Existing safety filters (e.g., ReasoningShield) cannot effectively identify all risk patterns, especially "weak vacillation"—where the reasoning correctly rejects the core harmful request but vacillates over superficially benign elements in the jailbreak prompt (such as role-playing requirements); (3) There is a lack of systematic comparative studies on safety fine-tuning options for LRMs.

**Key Challenge**: "Weak vacillation" itself is not directly harmful and thus cannot be detected by safety classifiers. However, including samples containing weak vacillation in training teaches the model to partially comply with harmful instructions, significantly reducing safety.

**Goal**: To systematically study the best practices for enhancing LRM safety through SFT, including data construction, reasoning chain length, and training configurations.

**Key Insight**: Starting from the analysis of failure cases—why do models remain unsafe after training on distilled "safe" data? The authors identify five risk reasoning patterns and eliminate them one by one.

**Core Idea**: (1) Risk reasoning patterns (especially weak vacillation) are key reasons for the failure of safety distillation—they need to be actively eliminated during the distillation phase; (2) Long, complex reasoning chains are not necessary for safety—short reasoning chains or even template reasoning can achieve comparable safety performance.

## Method

### Overall Architecture

The study is divided into three stages: (1) **Analysis stage**—identifying five risk reasoning patterns and verifying that weak vacillation is the core issue; (2) **Improved distillation stage**—designing targeted prompting strategies to eliminate risk patterns (RealSafe CoT and Improved CoT); (3) **Reasoning simplification stage**—verifying the safety equivalence of short reasoning chains (Short CoT) and template reasoning (Template CoT).

```mermaid
%%{init: {'flowchart': {'rankSpacing': 24, 'nodeSpacing': 28, 'padding': 6, 'wrappingWidth': 400, 'subGraphTitleMargin': {'top': 8, 'bottom': 16}}}}%%
flowchart TD
    A["Harmful Queries × Jailbreak Templates<br/>Distill safety responses from DeepSeek-R1"] --> B
    subgraph S1["Identification of 5 Risk Reasoning Patterns"]
        direction TB
        B["Classify risk reasoning in distilled data<br/>Lack of safety awareness / Strong vacillation / Weak vacillation / Harmful supplements / Reasoning-answer inconsistency"] --> C["Locate weak vacillation as the culprit<br/>Classifiers fail to capture it; ASR 21% vs 45% after removal"]
    end
    C --> D["Improve distillation prompting strategies<br/>RealSafe / Improved CoT constrain teacher vacillation at source"]
    D --> E["SFT Fine-tuning"]
    E --> F["Reasoning simplification experiments<br/>Short / Template / No CoT comparison"]
    F -->|Short chains & templates ≈ full chains, No CoT fails| G["Safe LRM<br/>PAIR ASR 63%→13%"]
```

### Key Designs

**1. Identification of 5 Risk Reasoning Patterns: Explaining why safety distillation fails**

Directly distilling "safe responses" from DeepSeek-R1 for fine-tuning yielded results far below expectations—after filtering with ReasoningShield, the PAIR ASR only moved from 66% to 54%. This paper analyzes failure cases and categorizes risk reasoning in distilled data into five types: (1) **Lack of safety awareness**—reasoning fails to identify the harmful nature of the query; (2) **Strong vacillation**—hesitation regarding the core harmful intent itself during overthinking; (3) **Weak vacillation**—hesitation only regarding superficially benign elements in jailbreak prompts (e.g., "is it okay to play this role?"), while the core harmful request is actually rejected; (4) **Harmful supplements**—unintentionally completing harmful details during the reasoning process; (5) **Reasoning-answer inconsistency**—reasoning decides to reject, but the final answer still outputs harmful content.

Among these five, "weak vacillation" is the most problematic because it is not directly harmful and eludes safety classifiers. However, once mixed into the training set, it teaches the model that "partial compliance is acceptable." Hard evidence shows that after ReasoningShield filtering, the proportion of weak vacillation doubled (0.33 → 0.66) because it became the primary residue after other more obvious risk patterns were filtered; simply removing weak vacillation samples dropped the ASR from 45% to 21%. This identifies the true culprit behind the failure of safety distillation.

**2. Improved Distillation Prompting Strategies: Actively eliminating risk patterns during distillation rather than post-filtering**

Posterior filtering can only remove known risk patterns, and data utilization is extremely low—only 6.6% of samples can be retained. This paper instead operates at the source of distillation, using prompts to directly constrain the teacher's reasoning style: **RealSafe CoT** modifies the distillation prompt to require the model to "directly identify and reject harmful queries without being distracted by role-play or contextual framing"; **Improved CoT** further instructs the model "not to show vacillation toward any part of the jailbreak prompt," specifically blocking weak vacillation. This proactive elimination is both more efficient and cleaner than post-filtering, reducing the average PAIR ASR from 63.0% to 13.0% across four 7B–32B models.

**3. Reasoning Simplification Experiments: Verifying if long reasoning chains are necessary for safety**

The phenomenon of weak vacillation suggests a counter-intuitive possibility—that "thinking one step further" in a long reasoning chain might shake an already-made safety decision. This paper evaluates safety by gradually shortening the reasoning chain: **Short CoT** distills brief reasoning processes from GPT-4o; **Template CoT** uses a fixed template ("This is a harmful request → Reject"); **No CoT** rejects directly without reasoning. The results are clear: Short CoT and Template CoT achieve safety performance comparable to full reasoning chains, while No CoT fails completely. This indicates that safety requires "some form of reasoning" rather than "extensive reasoning"—long chains are not only unnecessary but may be counter-productive by leaving room for the model to vacillate.

### Loss & Training

Standard SFT cross-entropy loss. Training set: 200 harmful queries × 20 jailbreak templates = 4000 queries, with 1000 sampled after filtering + 100 XSTest benign queries to prevent over-refusal.

## Key Experimental Results

### Main Results

**DeepSeek-R1-Distill-Qwen-7B**

| Method | MATH500 | AIME24 | LiveCodeBench | None ASR↓ | PAP ASR↓ | PAIR ASR↓ | Over-refusal↓ |
|------|---------|--------|-------------|----------|---------|----------|---------|
| Original | 93.0 | 49.2 | 33.1 | 60.0 | 64.0 | 66.0 | 0.8 |
| Default CoT | 90.4 | 50.0 | 36.1 | 20.0 | 40.0 | 54.0 | 2.7 |
| Improved CoT | 90.2 | 51.7 | 34.9 | 4.0 | 4.0 | **12.0** | 6.7 |
| Short CoT | 89.6 | 53.3 | 36.1 | 4.0 | 2.0 | 16.0 | 12.0 |
| Template CoT | 89.4 | 49.2 | 33.7 | 12.0 | 0.0 | **0.0** | 10.7 |
| No CoT | 90.8 | 52.5 | 31.9 | 62.0 | 64.0 | 64.0 | 10.0 |

### Ablation Study

| Risk Patterns Eliminated | PAIR ASR |
|-------------|---------|
| ReasoningShield Filtering (includes weak vacillation) | 45.0% |
| Remove all risk patterns **except** weak vacillation | 45.0% (No improvement) |
| Remove **all** risk patterns (including weak vacillation) | **21.0%** |

### Key Findings

- **Weak vacillation is the core reason for the failure of safety distillation**—removing other risk patterns is ineffective (ASR unchanged), while removing only weak vacillation reduces ASR by 24 percentage points.
- **No CoT is completely ineffective**—the reasoning process is necessary for safety (some form of reasoning), but it does not need to be long.
- **Template CoT reaches 0% PAIR ASR**—the simplest reasoning pattern is the safest because template reasoning leaves no room for the model to "vacillate."
- **Safety fine-tuning does not impair reasoning performance**—MATH500, AIME24, and LiveCodeBench remain largely unchanged.
- Short CoT has a higher over-refusal rate (12%), requiring a trade-off between safety and helpfulness.

## Highlights & Insights

- The proposal of the "weak vacillation" concept is significant—it appears harmless but signals that "partial compliance" is possible, acting as a hidden safety hazard.
- The discovery that "longer reasoning is not necessarily safer" subverts intuition—in the safety domain, more thinking may lead to more vacillation.
- The finding that safety classifiers cannot detect weak vacillation exposes a blind spot in existing safety evaluation tools.

## Limitations & Future Work

- The study primarily focuses on SFT and does not cover the impact of alignment methods like RLHF/DPO on LRM safety.
- The training scale of 1000 samples is relatively small; effects might differ at a larger scale.
- While Template CoT has the lowest ASR, it has a high over-refusal rate, requiring a finer balance.
- Evaluations were conducted only in English; the safety performance of multilingual LRMs may vary.

## Related Work & Insights

- **vs Zhang et al. (2025a)**: Proposed safety-aware response distillation but did not systematically analyze failure causes; this paper provides an in-depth identification of five risk patterns.
- **vs Jiang et al. (2025)**: Found that safety tuning impairs reasoning performance; this paper demonstrates that safety can be significantly improved without compromising reasoning.
- **vs Wang et al. (2025)**: Proposed low-cost mitigation strategies; this paper provides more comprehensive ablation studies.

## Rating

- Novelty: ⭐⭐⭐⭐⭐ The concept of "weak vacillation" and the discovery that "short reasoning is safer" are highly insightful.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ 4 models × 6 methods × multi-dimensional evaluation of reasoning + safety + over-refusal + detailed ablations.
- Writing Quality: ⭐⭐⭐⭐⭐ The logic chain from problem analysis to solution is exceptionally clear.
- Value: ⭐⭐⭐⭐⭐ Provides a systematic practical guide for the safety alignment of LRMs.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] AutoRAN: Automated Hijacking of Safety Reasoning in Large Reasoning Models](autoran_automated_hijacking_of_safety_reasoning_in_large_reasoning_models.md)
- [\[ACL 2026\] Reasoning Structure Matters for Safety Alignment of Reasoning Models](reasoning_structure_matters_for_safety_alignment_of_reasoning_models.md)
- [\[ICLR 2026\] Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models](../../ICLR2026/llm_safety/reasoning_or_retrieval_a_study_of_answer_attribution_on_large_reasoning_models.md)
- [\[ACL 2026\] Reasoning Hijacking: The Fragility of Reasoning Alignment in Large Language Models](reasoning_hijacking_the_fragility_of_reasoning_alignment_in_large_language_model.md)
- [\[ACL 2026\] When Models Outthink Their Safety: Unveiling and Mitigating Self-Jailbreak in Large Reasoning Models](when_models_outthink_their_safety_unveiling_and_mitigating_self-jailbreak_in_lar.md)

</div>

<!-- RELATED:END -->
