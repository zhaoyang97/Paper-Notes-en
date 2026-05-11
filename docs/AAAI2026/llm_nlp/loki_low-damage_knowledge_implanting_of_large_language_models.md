---
title: >-
  [Paper Note] LoKI: Low-damage Knowledge Implanting of Large Language Models
description: >-
  [AAAI 2026][LLM/NLP][Parameter-efficient fine-tuning] This paper proposes LoKI, a parameter-efficient fine-tuning method grounded in the mechanistic understanding of knowledge storage in Transformers. It introduces Knowl…
tags:
  - "AAAI 2026"
  - "LLM/NLP"
  - "Parameter-efficient fine-tuning"
  - "catastrophic forgetting"
  - "knowledge vector attribution"
  - "layer-balanced strategy"
  - "FFN knowledge storage"
date: 2026-05-08
content_hash: ed083791bc1135c8
---

# LoKI: Low-damage Knowledge Implanting of Large Language Models

**Conference**: AAAI 2026
**arXiv**: [2505.22120](https://arxiv.org/abs/2505.22120)
**Code**: [https://github.com/Nexround/LoKI](https://github.com/Nexround/LoKI)
**Area**: LLM Fine-tuning / Catastrophic Forgetting
**Keywords**: Parameter-efficient fine-tuning, catastrophic forgetting, knowledge vector attribution, layer-balanced strategy, FFN knowledge storage

## TL;DR

This paper proposes LoKI, a parameter-efficient fine-tuning method grounded in the mechanistic understanding of knowledge storage in Transformers. It introduces Knowledge Vector Attribution (KVA) to quantify the contribution of each knowledge vector in FFN layers, and applies a layer-balanced strategy to select low-contribution vectors for targeted knowledge implanting. The approach achieves strong task performance while substantially mitigating catastrophic forgetting.

## Background & Motivation

**Background**: LLMs accumulate rich world knowledge during pre-training, and fine-tuning adapts them to downstream tasks. Parameter-efficient fine-tuning (PEFT) methods such as LoRA have significantly reduced the cost of fine-tuning.

**Limitations of Prior Work**: Fine-tuning is accompanied by catastrophic forgetting (CF). Conventional PEFT methods update all Transformer modules indiscriminately, ignoring the locations of critical knowledge-storing weights and potentially causing irreversible damage to existing knowledge.

**Key Challenge**: A fundamental tension exists between downstream task performance and the preservation of pre-trained general capabilities. Existing methods either sacrifice task performance to retain knowledge (e.g., orthogonal subspace approaches) or sacrifice general capabilities for task performance. Prior work on knowledge localization and editing (ROME, KN, etc.) has not been effectively integrated into the PEFT pipeline.

**Goal**: To leverage mechanistic understanding of knowledge storage in LLMs to identify low-contribution parameters that can be safely repurposed for new knowledge implanting, thereby achieving "low-damage" knowledge injection.

**Key Insight**: Building on interpretability research that treats FFN layers as "key-value memories"—each row of $W_{down}$ constitutes a "knowledge vector," and different vectors vary greatly in their contribution to the model's general capabilities, with low-contribution vectors being amenable to reuse.

**Core Idea**: A three-stage pipeline: analyze (KVA quantifies the contribution of each knowledge vector), select (a layer-balanced strategy identifies low-contribution vectors), and implant (only selected vectors are updated while the rest are frozen). This process translates knowledge localization research into a practical fine-tuning method.

## Method

### Overall Architecture

LoKI consists of three stages: Analyzing → Selecting → Implanting.

1. KVA is applied on the MMLU dataset to quantify the contribution of all knowledge vectors across all layers.
2. A layer-balanced strategy selects the top q% low-contribution vectors as trainable parameters.
3. All other parameters are frozen; only the selected knowledge vectors are updated during downstream fine-tuning.

### Key Designs

1. **Knowledge Vector Attribution (KVA)**:

    - An attribution method based on Integrated Gradients.
    - For each knowledge output node in every FFN layer, the path-integrated contribution to the target logit is computed.
    - Formula: $Attr_{l,j}(\mathbf{x}) = \int_0^1 \frac{\partial \mathcal{L}(\alpha \mathbf{z}_{l,j})}{\partial \mathbf{z}_{l,j}} d\alpha$
    - Approximated via Riemann summation ($m=7$ steps); per-sample runtime is approximately 9.69 seconds on an RTX 4090.
    - Computed once on MMLU; the result is task-agnostic and reusable.

2. **Layer-Balanced Strategy**:

    - **Key finding**: High-contribution and low-contribution knowledge vectors are both densely co-located in the same layers (non-uniform distribution).
    - **Design Motivation**: Without layer balancing, naive selection concentrates updates in a small number of layers, disrupting the hierarchical knowledge structure of the Transformer.
    - **Method**: An equal trainable quota $k_l = \lfloor T/L \rfloor$ is assigned to each layer, and the $k_l$ lowest-contribution vectors within each layer are selected.
    - **Frequency aggregation**: Selection frequency is aggregated across multiple samples to identify the most consistently low-contribution vectors.

3. **Knowledge Implanting**:

    - Each layer's $W_{down}$ is decomposed into $W_{\mathcal{S}}$ (trainable) and $W_{\setminus\mathcal{S}}$ (frozen).
    - Optionally combined with LoRA: $\Delta W_{\mathcal{S}} = A_l B_l$, further reducing the number of trainable parameters.
    - Implemented as a module-level operation that can be readily integrated into existing training pipelines.

### Loss & Training

- Only $W_{\mathcal{S}}$ is updated during training, using the standard downstream task loss.
- The proportion of trainable parameters is controlled by the hyperparameter $q$ (e.g., $q=5\%$ means only 5% of $W_{down}$ parameters are updated).
- KVA analysis is performed only once and can be reused across multiple downstream tasks.

## Key Experimental Results

### Main Results

**Experiment 1: ToolACE Function-Calling (Llama3.1-8B-Instruct)**

| Method | Overall Acc | Single Turn Acc | Multi Turn | Hallucination Rate |
|--------|-------------|-----------------|------------|--------------------|
| ToolACE Full Fine-tuning | 58.32 | 87.56 | 76.10 | — |
| LoKI | Comparable or superior to full fine-tuning | — | — | Lower |
| LoRA/DoRA and other PEFT | Lower | — | — | Higher |

**Experiment 2: LB Reranker (Qwen2.5-0.5B-Instruct)**

- LoKI maintains task performance while achieving notably better preservation of general capabilities compared to other PEFT methods.

### Ablation Study

- **Layer-balanced vs. global ranking**: Removing layer balancing significantly increases CF, validating the critical role of the layer-balanced strategy.
- **High-contribution vs. low-contribution vectors**: Updating high-contribution vectors leads to more severe CF.
- **KVA sample size**: The overlap in selected nodes between full MMLU and a sampled subset reaches 97.57%.
- **Combination with LoRA**: LoKI+LoRA further reduces the parameter count while maintaining performance.

### Key Findings

- High-contribution and low-contribution knowledge vectors exhibit a striking co-localization pattern across Transformer layers—both cluster densely in the same layers.
- The layer-balanced strategy is critical for mitigating CF, confirming the practical significance of hierarchical knowledge organization in Transformers.
- LoKI modifies only a small fraction of vectors in $W_{down}$ yet achieves task performance comparable to full fine-tuning.
- Modifications to earlier layers may "overwrite" knowledge flows in intermediate layers, consistent with findings reported by Hase et al.

## Highlights & Insights

- The work successfully bridges interpretability research on knowledge localization/editing and practical PEFT methods, connecting two previously separate subfields.
- The KVA method is elegant and efficient: computed once, task-agnostic, and reusable.
- The layer-balanced strategy is well-supported by both theoretical reasoning and empirical evidence, offering a principled answer to the question of how to select parameters for updating.
- The discovery of co-localization of knowledge vector contributions across layers provides meaningful insight into the internal knowledge organization of Transformers.

## Limitations & Future Work

- Experiments are conducted on only two model architectures (Llama3.1-8B and Qwen2.5-0.5B); generalizability remains to be verified.
- KVA relies on MMLU as a proxy for general capability, which may not be appropriate for non-English or domain-specific models.
- The study focuses exclusively on $W_{down}$; the knowledge vector properties of attention layers and $W_{up}$ are not explored.
- The layer-balanced strategy assumes equal quota per layer, whereas adaptive quota allocation accounting for varying knowledge density across layers may yield better results.
- There is methodological overlap with knowledge editing approaches (ROME, etc.) in their operations on FFN weights; the theoretical distinction is not sufficiently articulated.

## Related Work & Insights

- Knowledge editing methods such as ROME, KN, and AlphaEdit focus on targeted factual corrections; LoKI generalizes this perspective to general-purpose fine-tuning.
- CorDA freezes dominant singular directions and O-LoRA employs orthogonal subspaces; LoKI's KVA provides a more direct quantification of knowledge contribution.
- The theoretical foundation of LoKI rests on Geva et al.'s "FFN as key-value memory" framework.
- The analyze–select–implant paradigm is broadly applicable to other scenarios requiring fine-grained control over parameter updates.

## Rating

⭐⭐⭐⭐ (4/5)

The method is well-motivated and elegantly designed, successfully translating interpretability findings into a practical PEFT approach with strong novelty. The combination of KVA and the layer-balanced strategy is thoroughly validated both theoretically and empirically. The main limitations are the relatively small scale of the experimental models and the fact that deeper analysis of the knowledge vector co-localization phenomenon is deferred to future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] ProFuser: Progressive Fusion of Large Language Models](profuser_progressive_fusion_of_large_language_models.md)
- [\[NeurIPS 2025\] The Rise of Parameter Specialization for Knowledge Storage in Large Language Models](../../NeurIPS2025/llm_nlp/the_rise_of_parameter_specialization_for_knowledge_storage_in_large_language_mod.md)
- [\[AAAI 2026\] CoEvo: Continual Evolution of Symbolic Solutions Using Large Language Models](coevo_continual_evolution_of_symbolic_solutions_using_large_language_models.md)
- [\[AAAI 2026\] Control Illusion: The Failure of Instruction Hierarchies in Large Language Models](control_illusion_the_failure_of_instruction_hierarchies_in_large_language_models.md)
- [\[AAAI 2026\] Identifying and Analyzing Performance-Critical Tokens in Large Language Models](identifying_and_analyzing_performance-critical_tokens_in_large_language_models.md)

</div>

<!-- RELATED:END -->
