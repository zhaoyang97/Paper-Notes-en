---
title: >-
  [Paper Note] AIM-CoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning
description: >-
  [ACL 2026][LLM Reasoning][interleaved multimodal chain-of-thought] This paper proposes AIM-CoT, a framework driven by Information Foraging Theory that addresses two core problems in Interleaved Multimodal Chain-of-Though…
tags:
  - "ACL 2026"
  - "LLM Reasoning"
  - "interleaved multimodal chain-of-thought"
  - "information foraging theory"
  - "active visual probing"
  - "dynamic triggering"
  - "visual question answering"
date: 2026-05-08
content_hash: c6b9eaa2e863ab56
---

# AIM-CoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning

**Conference**: ACL 2026
**arXiv**: [2509.25699](https://arxiv.org/abs/2509.25699)
**Code**: [GitHub](https://anonymous.4open.science/r/AIMCoT)
**Area**: Vision-Language Reasoning / Multimodal CoT
**Keywords**: interleaved multimodal chain-of-thought, information foraging theory, active visual probing, dynamic triggering, visual question answering

## TL;DR

This paper proposes AIM-CoT, a framework driven by Information Foraging Theory that addresses two core problems in Interleaved Multimodal Chain-of-Thought (I-MCoT)—*what to see* and *when to see*—through Active Visual Probing (AVP) based on information gain and a Dynamic Attention-shift Trigger (DAT) mechanism.

## Background & Motivation

**State of the Field**: Interleaved Multimodal Chain-of-Thought (I-MCoT) represents a significant paradigm advancement in vision-language reasoning (e.g., VQA). This paradigm selects fine-grained visual evidence from input images and inserts it as visual tokens into the reasoning chain context, enabling models to reference specific visual details during inference.

**Limitations of Prior Work**: Existing I-MCoT methods (e.g., ICoT) exhibit deficiencies on two core problems: (1) **What to see**: reliance on attention maps for visual region selection, where attention signals are unreliable—when a severe granularity mismatch exists between a short textual query and an information-rich image, attention peaks frequently fail to align with genuinely critical visual regions (over 75% of samples exhibit IoU below 50%); (2) **When to see**: adoption of static triggering strategies (e.g., inserting evidence upon encountering a newline character), which fail to capture the model's dynamic demand for visual evidence.

**Root Cause**: Attention maps capture semantic correlations between tokens, whereas I-MCoT fundamentally requires visual evidence that provides maximal information for subsequent reasoning—semantic relevance does not equate to informativeness.

**Paper Goals**: To transform VLM reasoning from "passive, static perception" into "active, dynamic exploration," enabling models to proactively seek the most valuable visual cues in the manner of information foragers.

**Core Idea**: Drawing on Information Foraging Theory (IFT), the paper replaces attention scores with information gain (entropy reduction) as the criterion for visual evidence selection, and replaces fixed triggering conditions with attention shift as the signal for determining evidence insertion timing.

## Method

### Overall Architecture

AIM-CoT is a training-free framework comprising three collaborative components: (1) **CAG** (Context-enhanced Attention-map Generation): mitigates text–vision granularity imbalance by generating query-conditioned image descriptions; (2) **AVP** (Active Visual Probing): actively selects the most informative visual evidence based on information gain; (3) **DAT** (Dynamic Attention-shift Trigger): monitors attention shifts from text to vision to dynamically trigger visual evidence insertion. The overall pipeline follows a *trigger–select–insert* paradigm.

### Key Designs

1. **Context-enhanced Attention-map Generation (CAG)**

    - **Function**: Mitigates text–vision granularity imbalance, providing a more reliable foundation for subsequent attention maps and candidate regions.
    - **Mechanism**: Prior to VQA, the VLM generates a query-conditioned explanatory description of the image, $\mathcal{D}_{\mathrm{CAG}} = \mathrm{VLM}(I, x, \mathcal{P}_{\mathrm{CAG}})$, which is concatenated with the original query to form an enhanced query $x' = \mathrm{concat}(x, \mathcal{D}_{\mathrm{CAG}})$. The description provides semantic anchors that render the cross-attention distribution more accurate. The prompt incorporates negative constraints to suppress hallucinations.
    - **Design Motivation**: Raw queries are too brief to effectively guide attention distributions. CAG is not simple caption generation; rather, it provides additional textual semantic anchors on the text side to improve the fidelity of attention distributions.

2. **Active Visual Probing (AVP)**

    - **Function**: Addresses the *what to see* problem by actively selecting the most informative visual regions based on information gain.
    - **Mechanism**: A three-step pipeline—(a) **Candidate set construction**: a candidate pool is built from the attention-driven set $C_{\mathrm{attn}}$ (top-$N$ high-attention regions) and an exploration set $C_{\mathrm{exp}}$ ($M$ uniformly sampled regions); (b) **Information gain quantification**: for each candidate region $R_i$, the entropy reduction upon adding it to the context is computed as $\mathrm{IG}(\{R_i\}) = U_B - U_{C,i}$, where $U_B = H(Y|I,x,y_{<t})$ is the baseline uncertainty and $U_{C,i} = H(Y|I,x,y_{<t},R_i)$ is the conditional uncertainty; (c) **Sequential greedy selection**: the region with the highest information gain is iteratively selected, with the context updated at each step before re-evaluating the remaining candidates.
    - **Design Motivation**: Per IFT, information is only valuable when it reduces an agent's uncertainty. Greedy algorithms provide near-optimal guarantees for subset selection problems, and the sequential process emulates the dynamic trajectory of foraging behavior.

3. **Dynamic Attention-shift Trigger (DAT)**

    - **Function**: Addresses the *when to see* problem by precisely triggering evidence insertion when the model's cognitive demand shifts toward visual information.
    - **Mechanism**: The text–vision attention shift at each step of autoregressive generation is monitored as $\Delta A_{\mathrm{vision}}(t) = A_{\mathrm{vision}}(t) - A_{\mathrm{vision}}(t-1)$. When the shift exceeds threshold $\delta$, AVP is triggered to insert visual evidence. A "safety instruction" is also employed, directing the model to treat inserted evidence as "supplementary reference" to reduce noise introduction.
    - **Design Motivation**: Although attention is unreliable as a selection criterion, the *shift* in attention is a reliable diagnostic signal indicating the model's need for visual information.

### Loss & Training

AIM-CoT is a fully training-free framework that operates directly on frozen VLMs. All components are realized through carefully designed prompt templates and internal attention signals, requiring no parameter updates. Inference-time overhead is kept within $1.36\times$ that of the baseline.

## Key Experimental Results

### Main Results

| Backbone | Benchmark | AIM-CoT | Prev. SOTA (ICoT) | Gain |
|----------|-----------|---------|-------------------|------|
| Chameleon-7B | M3CoT (0-shot) | 31.4 | 29.8 | +5.4% |
| Chameleon-7B | LLaVA-W (0-shot) | 29.8 | 25.2 | +18.3% |
| Janus-Pro-7B | M3CoT (1-shot) | 41.5 | 39.4 | +5.3% |
| Qwen2-VL-7B | ScienceQA (1-shot) | 66.3 | 65.4 | +1.4% |
| Qwen2.5-VL-32B | M3CoT (1-shot) | 61.2 | 59.1 | +3.6% |
| Qwen2.5-VL-32B | LLaVA-W (1-shot) | 49.1 | 44.7 | +9.8% |

### Ablation Study

| Configuration | Key Metric | Note |
|---------------|-----------|------|
| Attention coverage (IoU) | >75% of samples below 50% | Severe misalignment between attention peaks and critical regions |
| Masking high-attention regions | Only marginal performance drop | High attention ≠ critical region |
| CAG negative constraints | Effectively suppresses hallucinations | Validates the necessity of the cautious description strategy |
| Safety instruction | Effectively filters noise | Prevents visual evidence from introducing interference |
| Inference time | $\leq 1.36\times$ baseline | Deployment-friendly |

### Key Findings
- Regions selected via information gain differ substantially from those selected via attention peaks; the former effectively filters high-attention but non-informative regions.
- Dynamic triggering outperforms static triggering (newline-based) across all benchmarks, with the largest gains on LLaVA-W (open-ended QA).
- The exploration set (uniform sampling), despite its simplicity, contributes critical regions overlooked by the attention-driven set.
- Consistent improvements are observed on stronger backbone models (Qwen2.5-VL-32B), demonstrating the generality of the approach.

## Highlights & Insights
- **Elegant introduction of Information Foraging Theory**: IFT provides a unified theoretical account of both the *what to see* and *when to see* problems, establishing a solid theoretical foundation.
- **Dialectical treatment of attention**: Attention is unreliable as a selection criterion, yet attention *shift* is reliable as a triggering signal—this distinction is particularly insightful.
- **Training-free design**: Operating entirely on inference-time signals, the framework is plug-and-play for any frozen VLM, offering strong practical utility.
- **Information gain vs. attention analysis**: The paper rigorously demonstrates that semantic relevance ≠ informativeness, offering a new perspective on visual evidence selection.
- **Safety instruction mechanism**: By directing the model to treat inserted visual evidence as "reference rather than reliance," the approach effectively reduces noise risk.

## Limitations & Future Work
- Information gain quantification requires additional forward passes; while kept within $1.36\times$, there remains optimization potential for latency-sensitive scenarios.
- Candidate regions are based on fixed partitioning, with adaptive region segmentation strategies left unexplored.
- Validation is primarily conducted on VQA tasks; generalization to visual reasoning, chart understanding, and other tasks remains to be confirmed.
- The quality of CAG-generated descriptions is bounded by the VLM's own capabilities; weaker models may produce low-quality descriptions.
- Although an adaptive strategy exists for threshold $\delta$, tuning may still be required across different datasets.

## Related Work & Insights
- **vs. ICoT**: ICoT employs attention-based selection with static triggering; AIM-CoT uses information-gain-based selection with dynamic triggering, achieving comprehensive improvements across all settings.
- **vs. DDCoT/CCoT**: These methods generate textual descriptions to assist reasoning but do not directly insert visual evidence; AIM-CoT simultaneously leverages description-enhanced attention and direct visual evidence insertion.
- **vs. SCAFFOLD**: SCAFFOLD employs structured reasoning but handles visual evidence less precisely; AIM-CoT's information gain quantification provides a more principled selection criterion.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ — Information Foraging Theory-driven visual evidence selection represents an entirely new perspective; the insight of using attention shift as a triggering signal is particularly profound.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ — Four backbone models, three benchmarks, comprehensive ablation studies, and reliability analysis.
- **Writing Quality**: ⭐⭐⭐⭐⭐ — Motivation analysis is thorough, with a complete logical chain from problem identification to theoretical grounding to method design.
- **Value**: ⭐⭐⭐⭐ — Provides a novel theoretical framework and a practical training-free solution for multimodal CoT.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] AIMCoT: Active Information-driven Multimodal Chain-of-Thought for Vision-Language Reasoning](../../ICLR2026/llm_reasoning/aimcot_active_information-driven_multimodal_chain-of-thought_for_vision-language.md)
- [\[ACL 2026\] Self-Consistency from Only Two Samples: CoT-PoT Ensembling for Efficient LLM Reasoning](self-consistency_from_only_two_samples_cot-pot_ensembling_for_efficient_llm_reas.md)
- [\[ACL 2026\] Learning to Edit Knowledge via Instruction-based Chain-of-Thought Prompting](learning_to_edit_knowledge_via_instruction-based_chain-of-thought_prompting.md)
- [\[ACL 2026\] CRISP: Compressing Redundancy in Chain-of-Thought via Intrinsic Saliency Pruning](crisp_compressing_redundancy_in_chain-of-thought_via_intrinsic_saliency_pruning.md)
- [\[ACL 2026\] Efficient Process Reward Modeling via Contrastive Mutual Information](efficient_process_reward_modeling_via_contrastive_mutual_information.md)

</div>

<!-- RELATED:END -->
