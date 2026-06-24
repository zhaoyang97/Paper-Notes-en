---
title: >-
  [Paper Note] Safety Alignment via Constrained Knowledge Unlearning
description: >-
  [ACL 2025][LLM Alignment][Safety Alignment] This paper proposes Constrained Knowledge Unlearning (CKU), which removes harmful knowledge by locating useful knowledge neurons in MLP layers and protecting their gradients during the unlearning process, significantly enhancing the safety of LLMs without compromising their general capabilities.
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "Safety Alignment"
  - "Knowledge Unlearning"
  - "Jailbreak Defense"
  - "Neuron Localization"
  - "Model Editing"
date: 2026-05-08
content_hash: ad7870c1cb0ca567
---

# Safety Alignment via Constrained Knowledge Unlearning

**Conference**: ACL 2025  
**arXiv**: [2505.18588](https://arxiv.org/abs/2505.18588)  
**Code**: None  
**Area**: Alignment RLHF  
**Keywords**: Safety Alignment, Knowledge Unlearning, Jailbreak Defense, Neuron Localization, Model Editing

## TL;DR

This paper proposes Constrained Knowledge Unlearning (CKU), which removes harmful knowledge by locating useful knowledge neurons in MLP layers and protecting their gradients during the unlearning process, significantly enhancing the safety of LLMs without compromising their general capabilities.

## Background & Motivation

**Background**: The safety alignment of large language models (LLMs) has always been a key challenge in deployment. Mainstream methods, such as RLHF and instruction tuning, implement safety barriers by teaching models to refuse harmful requests, and they have been widely deployed on major commercial models.

**Limitations of Prior Work**: Existing safety alignment methods are essentially "behavior-level" patches—they teach the model "not to output" harmful content, but harmful knowledge remains stored within the model weights. This leaves opportunities for jailbreak attacks: attackers can bypass behavioral constraints using carefully crafted prompts (such as role-playing, code rewriting, gradient guidance, etc.) to induce the model to output pre-existing harmful knowledge. Recent studies have demonstrated that almost all mainstream safety alignment methods are, to varying degrees, vulnerable to being bypassed by jailbreak attacks.

**Key Challenge**: The fundamental contradiction between behavioral constraints (refusing to answer) and the existence of knowledge (still remembering harmful content). As long as harmful knowledge remains in the model weights, there is always a risk of it being bypassed. However, direct "unlearning" of harmful knowledge faces a challenge: knowledge in models is stored in an interleaved manner, and naive erasure may simultaneously damage useful general knowledge.

**Goal**: To design a method that can truly "erase" harmful knowledge (rather than merely suppressing its output) while ensuring that useful knowledge is not affected.

**Key Insight**: Knowledge in LLMs is mainly encoded in the neurons of multilayer perceptron (MLP) layers. Different types of knowledge are likely encoded by different subsets of neurons. If "useful knowledge neurons" can be accurately located and protected during the unlearning process, selective knowledge deletion can be achieved.

**Core Idea**: First locate the subset of neurons U in the MLP layers that encode useful knowledge. Then, during knowledge unlearning, prune the gradients of neurons in U to keep their weights unchanged, thereby retaining useful capabilities while deleting harmful knowledge.

## Method

### Overall Architecture

CKU is executed in two steps: the first step (knowledge localization and preservation) identifies the key neuron subset U by analyzing the contribution of neurons in each MLP layer to useful knowledge; the second step (constrained unlearning) clips the gradients of neurons in U to zero during gradient descent (knowledge unlearning) on harmful data, ensuring that only harmful knowledge is forgotten without affecting useful knowledge. The inputs are the model to be aligned and the harmful/retained datasets, and the output is a model with enhanced safety and preserved general capabilities.

### Key Designs

1. **Neuron Knowledge Sensitivity Scoring**:

    - **Function**: Formulates a score for each neuron in each MLP layer to measure its importance to useful knowledge.
    - **Mechanism**: Conducts forward propagation on useful knowledge data (such as general dialogues, Q&A data) and records the activation values of each neuron. By analyzing the statistical features of activation patterns (such as activation frequency, activation magnitude, or correlation with output quality), a sensitivity score is allocated to each neuron. A higher score indicates greater importance of the neuron to useful knowledge. A threshold is then set to classify high-scoring neurons into the protected set U.
    - **Design Motivation**: MLP layers are the primary locations for storing factual knowledge in Transformers (as widely verified by model editing research). The neuron-level localization granularity is sufficiently fine to distinguish the encoding locations of useful versus harmful knowledge.

2. **Gradient-Constrained Unlearning**:

    - **Function**: Protects useful knowledge from being destroyed during the unlearning process.
    - **Mechanism**: Performs gradient ascent on harmful knowledge data—i.e., training the model to maximize the loss of harmful outputs so that the model gradually "forgets" how to generate harmful content. However, after calculating the gradients through backpropagation, the gradients of the neurons belonging to the protected set U are forcibly set to zero, maintaining their weights unchanged throughout the unlearning process. Mathematically, this is equivalent to imposing a constraint in the parameter space: $\nabla_{\theta_U} \mathcal{L} = 0$, allowing only non-U neurons to update their weights.
    - **Design Motivation**: Unconstrained knowledge unlearning (naive unlearning) indiscriminately modifies all parameters, leading to the impairment of useful knowledge. Gradient constraint offers an elegant mechanism to achieve selective unlearning—the computational overhead is merely a masking operation on the gradients.

3. **Cross-Layer Knowledge Sensitivity Analysis**:

    - **Function**: Analyzes the distribution of knowledge sensitivity across different MLP layers to guide the layer-wise selection of the unlearning strategy.
    - **Mechanism**: Different MLP layers contribute differently to knowledge encoding—shallow layers encode more general linguistic patterns, while deep layers encode more specific factual knowledge. By conducting a statistical analysis of neuron sensitivity across all layers, CKU can identify which layers contain more safety-related knowledge, thereby executing unlearning in these layers with more targeting. This analysis also provides visual insights for understanding the safety mechanisms of LLMs.
    - **Design Motivation**: The knowledge distribution characteristics differ across layers. A layer-aware unlearning strategy is more precise than a globally unified strategy, reducing unnecessary damage to useful knowledge.

### Loss & Training

CKU utilizes a two-part loss function: (1) Unlearning loss—the negative of the negative log-likelihood (gradient ascent) on harmful data, which drives the model away from harmful outputs; (2) Preservation loss—the standard cross-entropy loss (gradient descent) on useful data, which restores the model's performance on general tasks. The two losses are balanced via hyperparameter weighting. During the unlearning process, the gradients of neurons in U are clipped to zero, while other neurons are updated normally.

## Key Experimental Results

### Main Results

| Method | GCG Attack Success Rate ↓ | AutoDAN Attack Success Rate ↓ | General Capability (MMLU, etc.) Retention ↑ | Safety (Refusal Rate) ↑ |
|------|--------------|------------------|---------------------|--------------|
| No Alignment | High | High | 100% (Baseline) | Low |
| RLHF/DPO | Medium | Medium | ~98% | Medium-High |
| Naive Unlearning | Low | Low | ~85-90% | High |
| **CKU** | **Lowest** | **Lowest** | **~96-98%** | **Highest** |

CKU outperforms all baseline methods in defense effectiveness under various jailbreak attacks (GCG, AutoDAN, etc.), while exhibiting minimal performance degradation on general benchmarks.

### Ablation Study

| Configuration | Attack Defense | General Capability | Description |
|------|---------|---------|------|
| Full CKU | Optimal | ~97% | Complete method |
| No Gradient Constraint (Naive Unlearning) | Good | ~87% | Good safety but general capabilities drop significantly |
| No Knowledge Localization (Random Protection) | Moderate | ~93% | Imprecise protection effect |
| Shallow-layer Unlearning Only | Poor | ~98% | Shallow layers have limited contribution to safety knowledge |
| Deep-layer Unlearning Only | Good | ~95% | Deep layers contain more safety-related knowledge |

### Key Findings

- Gradient constraint is the key to protecting general capabilities—although unconstrained unlearning methods show significant safety improvements, they lead to a roughly 10-15% drop in general capabilities.
- The accuracy of knowledge localization directly affects the preservation effect—randomly selecting the protected set is far inferior to selection based on sensitivity scoring.
- Deep MLP neurons encode safety-related knowledge more densely, and executing unlearning in deep layers yields the best performance.
- CKU is effective in defending against different types of jailbreak attacks (gradient-based, semantic-based), indicating that it indeed deletes harmful knowledge rather than merely altering superficial behavioral patterns.
- The distribution of neuron knowledge sensitivity varies significantly across layers, a finding that itself offers valuable reference for understanding the knowledge storage mechanisms in LLMs.

## Highlights & Insights

- **Paradigm Shift from "Behavioral Constraint" to "Knowledge Deletion"**: CKU represents an important paradigm shift in safety alignment—instead of teaching the model "not to say," it is better to make the model "no longer know." This fundamentally eliminates the attack surface of jailbreak attacks, offering a more thorough solution than superficial refusal.
- **Elegance of Gradient Clipping**: The method of protecting useful knowledge is extremely succinct—just applying a mask on the gradients. This implementation incurs almost zero computational overhead and is compatible with any gradient-based training method.
- **Safety Alignment from a Knowledge Editing Perspective**: Reformatting the safety alignment problem as a "selective knowledge editing" problem bridges the gap between safety alignment and model editing, facilitating cross-disciplinary references.

## Limitations & Future Work

- The boundary between harmful and useful knowledge is blurry in practice—for instance, chemistry knowledge can be used for both beneficial and harmful purposes; how does CKU handle such dual-use knowledge?
- The accuracy of neuron localization depends on the representativeness of the reference dataset used for scoring; inappropriate data selection may lead to misjudgment.
- The irreversibility of unlearning is a double-edged sword—once certain knowledge is forgotten, it cannot be recovered even for legitimate needs (by contrast, behavioral constraints can be lifted through privilege granting).
- MLPs are not the sole storage location of knowledge—attention layers also participate in knowledge encoding, and CKU's current focus strictly on MLP layers might be insufficient.
- Future work could integrate knowledge graphs to define the boundaries of "harmful knowledge" more precisely, avoiding accidental deletion.

## Related Work & Insights

- **vs RLHF/DPO Safety Alignment**: RLHF/DPO teaches models to refuse harmful requests through preference learning without deleting harmful knowledge. CKU directly erases harmful knowledge from the weights, resulting in more robust defense but irreversibility.
- **vs Machine Unlearning (SISA, etc.)**: Traditional machine unlearning focuses on data privacy (removing the influence of specific training data). CKU adapts the unlearning framework to safety scenarios, aiming to erase harmful capabilities rather than specific data.
- **vs Knowledge Editing (ROME/MEMIT)**: Model editing methods precisely modify specific facts. CKU approaches from a similar perspective but with a different goal—batch-deleting a class of harmful knowledge rather than modifying individual facts.

## Rating

- Novelty: ⭐⭐⭐⭐ Combining knowledge unlearning with neuron localization for safety alignment is a novel combination, and the safety paradigm of "knowledge deletion" is highly inspiring.
- Experimental Thoroughness: ⭐⭐⭐⭐ Comprehensive validation across multiple jailbreak attack scenarios, with ablation designs revealing the contributions of each component.
- Writing Quality: ⭐⭐⭐⭐ The motivation is clearly articulated, the method is concisely described, and the experimental analysis is in-depth.
- Value: ⭐⭐⭐⭐ Provides a new technical path for LLM safety alignment and offers practical reference value for jailbreak attack defense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] LSSF: Safety Alignment via Low-Rank Safety Subspace Fusion](lssf_safety_subspace.md)
- [\[ICLR 2026\] Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](../../ICLR2026/llm_alignment/mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)
- [\[ACL 2025\] PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference](pku-saferlhf_towards_multi-level_safety_alignment_for_llms_with_human_preference.md)
- [\[ACL 2025\] SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings](sea_lowresource_safety_alignment_for_multimodal.md)
- [\[NeurIPS 2025\] Alignment of Large Language Models with Constrained Learning](../../NeurIPS2025/llm_alignment/alignment_of_large_language_models_with_constrained_learning.md)

</div>

<!-- RELATED:END -->
