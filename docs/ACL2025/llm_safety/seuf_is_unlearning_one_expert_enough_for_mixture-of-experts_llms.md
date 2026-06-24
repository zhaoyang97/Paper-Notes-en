---
title: >-
  [Paper Note] SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?
description: >-
  [ACL 2025][LLM Safety][Machine Unlearning] SEUF reveals for the first time that existing LLM unlearning methods fail severely on MoE models (causing over 35% utility drop). The root cause is that the unlearning process leads to expert selection drift in the router, creating a "shortcut" where target experts to be forgotten are bypassed while innocent experts are damaged. To address this, SEUF proposes a framework that locates target experts through expert attribution and stab…
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Machine Unlearning"
  - "MoE"
  - "Expert Routing"
  - "Parameter-Efficient"
  - "Knowledge Deletion"
date: 2026-05-08
content_hash: d660691a217600ac
---

# SEUF: Is Unlearning One Expert Enough for Mixture-of-Experts LLMs?

**Conference**: ACL 2025  
**arXiv**: [2411.18797](https://arxiv.org/abs/2411.18797)  
**Code**: None  
**Area**: LLM Safety  
**Keywords**: Machine Unlearning, MoE, Expert Routing, Parameter-Efficient, Knowledge Deletion

## TL;DR

SEUF reveals for the first time that existing LLM unlearning methods fail severely on MoE models (causing over 35% utility drop). The root cause is that the unlearning process leads to expert selection drift in the router, creating a "shortcut" where target experts to be forgotten are bypassed while innocent experts are damaged. To address this, SEUF proposes a framework that locates target experts through expert attribution and stabilizes routing selection with a router anchor loss, updating only 0.06% of parameters to simultaneously improve unlearning quality and model utility.

## Background & Motivation

**Background**: Machine unlearning in LLMs aims to remove the influence of specific knowledge (such as harmful information or copyrighted data) from pretrained models while maintaining model utility on other tasks. Existing methods, including Gradient Ascent (GA), Gradient Difference (GDiff), Negative Preference Optimization (NPO), and Representation Misdirection Unlearning (RMU), have demonstrated effectiveness on dense LLMs.

**Limitations of Prior Work**: MoE LLMs (e.g., Mixtral, Qwen-MoE, DeepSeek-V2) are key members of the LLM family, yet their unlearning remains entirely unstudied. The authors discover for the first time that directly applying existing unlearning methods to MoE LLMs causes a catastrophic utility drop (over 20% decline on MMLU) and yields poor unlearning performance. The root cause lies in the dynamic routing mechanism of MoE: the unlearning process alters expert parameters, which indirectly affects the router's selection, leading to key experts being bypassed and non-critical experts mistakenly unlearned.

**Key Challenge**: The dynamic routing of MoE is its core advantage for inference efficiency, but becomes a fatal weakness in unlearning scenarios. The router creates "shortcuts" by directing inputs to irrelevant experts to minimize unlearning loss, destroying these innocent experts while shielding the target experts that actually need to be unlearned.

**Goal**: To design an MoE-specific unlearning framework that achieves controllable and efficient knowledge deletion by locating target experts, stabilizing routing selection, and performing targeted unlearning.

**Key Insight**: The authors find that for specific thematic unlearning targets, only a small number of experts (about 6-9 out of 64) in the MoE model are activated with high frequency (a long-tail distribution). SEUF identifies these target experts through expert attribution, uses a router anchor loss to prevent selection drift, and then applies unlearning algorithms solely to the target experts.

**Core Idea**: Locating target experts via expert attribution + preventing selection drift via router anchor loss + unlearning only the top-1 expert to achieve controllable and parameter-efficient knowledge deletion in MoE LLMs.

## Method

### Overall Architecture

SEUF consists of three steps: (1) Expert Attribution: collecting router affinity scores on the forget set to identify the top-$M$ target experts activated most frequently in each layer; (2) Parameter Selection: activating gradients solely for target experts and their corresponding routers; (3) Unlearning and Anchoring: adding a router anchor loss to the standard unlearning loss to ensure that target experts maintain high activation states throughout the unlearning process. SEUF is a plug-and-play framework compatible with any existing unlearning algorithm (such as GA, GDiff, NPO, or RMU).

### Key Designs

1. **Expert Attribution**:

    - Function: Identifies the most relevant experts for the unlearning target in each layer.
    - Mechanism: Samples a subset from the forget set, records the routing affinity score $s_{i,t}^{(l)}$ of each token in each layer, averages these scores across all tokens and samples to obtain an attribution score for each expert, and selects the top-$M$ highest-scoring experts as target experts.
    - Design Motivation: Observing that routing selections for the forget set follow a long-tail distribution, where a few experts contain most of the target knowledge. Accurately locating these experts prevents collateral damage to unrelated experts.

2. **Router Anchor Loss**:

    - Function: Prevents router drift toward non-target experts during unlearning.
    - Mechanism: For each token in the forget set, the Kullback-Leibler (KL) divergence between the current routing distribution and the original routing distribution of the pretrained model is calculated. This is incorporated into the total optimization objective as an anchor loss: $L_{anchor} = \text{KL}(s^{(l)}_{orig} \| s^{(l)}_{current})$. This ensures that even when expert parameters are modified, the router still directs relevant tokens to the target experts.
    - Design Motivation: Addresses the core problem of routing selection drift. Experiments show that without anchoring, the overlap rate of selected experts continuously declines across unlearning iterations, whereas it remains stable with the anchor loss.

3. **Targeted Parameter Unlearning**:

    - Function: Restricts unlearning to a minimal subset of parameters to preserve overall model utility.
    - Mechanism: Gradient updates are enabled only for the target experts (top-1 in each layer) and their routers, while the remaining 63/64 expert parameters in FPN are completely frozen. Ablations show that selecting top-1 experts outperforms top-2, top-3, or random selection.
    - Design Motivation: Updating only 0.06% of parameters (compared to 0.87% for LoRA or 14% for ESFT) significantly reduces the utility degradation caused by unlearning.

### Loss & Training

Total Loss = Unlearning Loss (e.g., Gradient Ascent for GA, Gradient Difference for GDiff) + Retention Loss (maintaining performance on the retain set) + Anchor Loss (stabilizing routing selection).

## Key Experimental Results

### Main Results

Qwen1.5-MoE unlearning results on the WMDP benchmark:

| Method | Forget Efficacy ↓ | Utility (MMLU) ↑ | Description |
|------|------------------|------------------|------|
| Original Model | 0.4192 | 0.5979 | No unlearning |
| GA | 0.2953 | 0.3393 | Utility collapse -43% |
| GA + SEUF | ~0.29 | **0.5012** | Utility recovery +47.7% |
| GDiff + SEUF | 0.2445 | **0.5295** | Best configuration |
| NPO + SEUF | ~0.32 | 0.5468 | |
| RMU + SEUF | 0.2536 | 0.5351 | |

### Ablation Study

| Ablation Configuration | Utility ↑ | Description |
|---------|----------|------|
| Full-parameter Unlearning | 0.3393 | Baseline (catastrophic drop) |
| Router Only | 0.2977 | Worse |
| Expert Only | 0.3242 | Slightly better |
| SEUF top-1 | **0.5012** | Optimal |
| SEUF top-2 | ~0.48 | Slight drop |
| SEUF Random Expert | ~0.42 | Validates attribution necessity |
| SEUF w/o Anchor Loss | ~0.44 | Validates anchoring necessity |

### Key Findings

- **MoE unlearning is a unique challenge**: All four unlearning algorithms lead to over 20% utility collapse across two MoE LLMs, whereas the utility loss on dense models is controllable.
- **Router drift is the root cause**: During the unlearning process, the overlap rate of expert selections continuously decreases (from >90% to <60%), demonstrating that the router creates "shortcuts" to bypass target experts.
- **SEUF is highly parameter-efficient**: Updating only 0.06% of the parameters recovers over 35% of utility while even yielding a 5% improvement in unlearning quality.
- **Top-1 expert is sufficient**: Unlearning a single top-1 expert outperforms unlearning top-2/top-3, indicating that target knowledge is highly concentrated.

## Highlights & Insights

- **Discovery of the "shortcut" issue in MoE unlearning**: This is a counter-intuitive yet highly insightful finding. Standard unlearning methods in MoE do not unlearn too much, but rather target the wrong objects, where the adaptability of the router becomes an obstacle instead.
- **Intricate design of the anchor loss**: Instead of freezing the router parameters (experiments show that freezing the router cannot prevent indirect drift), it maintains distribution consistency via KL divergence.
- **Plug-and-play versatility**: SEUF can be applied alongside any existing unlearning algorithm without modifying the algorithm itself.

## Limitations & Future Work

- **Evaluation restricted to 2 standard benchmarks (WMDP, RWKU)**: MoE unlearning benchmarks are highly scarce.
- **Lack of validation on larger MoE models**: Full-parameter experiments on models like DeepSeek-R1 and Mixtral 8x7B are constrained by computational resources.
- **Insufficient discussion**: The paper lacks analysis on which types of knowledge are more prone to escaping via routing drift.

## Related Work & Insights

- **vs. Dense LLM Unlearning (GA/GDiff/NPO/RMU)**: These methods are effective on dense models but collapse on MoE; SEUF acts as a wrapper layer that enables them to work on MoE models.
- **vs. Parameter-efficient methods like LoRA/ESFT**: SEUF only updates 0.06% of parameters, which is significantly lower than LoRA (0.87%) and ESFT (14%).

## Rating
- Novelty: ⭐⭐⭐⭐ Discovers and analyzes the unique challenges of MoE unlearning for the first time, providing deep insights into router drift.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models/algorithms/benchmarks with comprehensive ablations, though limited by the availability of standard benchmarks.
- Writing Quality: ⭐⭐⭐⭐ The narrative logic from problem discovery to root cause analysis and proposed solution is highly clear.
- Value: ⭐⭐⭐⭐ Fills a research gap in MoE unlearning, carrying practical significance for the safety governance of MoE models.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] CryptoMoE: Privacy-Preserving and Scalable Mixture of Experts Inference via Balanced Expert Routing](../../NeurIPS2025/llm_safety/cryptomoe_privacy-preserving_and_scalable_mixture_of_experts_inference_via_balan.md)
- [\[ACL 2025\] Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs](lacuna_inc_at_semeval-2025_task_4_lora-enhanced_influence-based_unlearning_for_l.md)
- [\[NeurIPS 2025\] One Token Embedding Is Enough to Deadlock Your Large Reasoning Model](../../NeurIPS2025/llm_safety/one_token_embedding_is_enough_to_deadlock_your_large_reasoning_model.md)
- [\[ACL 2025\] ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging](zjuklab_at_semeval-2025_task_4_unlearning_via_model_merging.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
