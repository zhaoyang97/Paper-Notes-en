---
title: >-
  [Paper Note] SAKE: Steering Activations for Knowledge Editing
description: >-
  [ACL 2025][Knowledge Editing][Activation Steering] SAKE proposes modeling knowledge editing as a distribution mapping problem in the activation space. By constructing source and target activation distributions through generating a set of paraphrased and logically entailed prompts for the edited facts, and then replacing the activation vectors with a linear mapping from optimal transport, SAKE achieves more robust fact editing than methods like ROME/MEMIT…
tags:
  - "ACL 2025"
  - "Knowledge Editing"
  - "Activation Steering"
  - "Optimal Transport"
  - "Distribution Mapping"
  - "Robustness"
date: 2026-05-08
content_hash: 7df7148325ca9a3a
---

# SAKE: Steering Activations for Knowledge Editing

**Conference**: ACL 2025  
**arXiv**: [2503.01751](https://arxiv.org/abs/2503.01751)  
**Code**: [axa-rev-research/knowledge-editing](https://github.com/axa-rev-research/knowledge-editing)  
**Area**: Knowledge Editing / LLM  
**Keywords**: Knowledge Editing, Activation Steering, Optimal Transport, Distribution Mapping, Robustness

## TL;DR

SAKE proposes modeling knowledge editing as a distribution mapping problem in the activation space. By constructing source and target activation distributions through generating a set of paraphrased and logically entailed prompts for the edited facts, and then replacing the activation vectors with a linear mapping from optimal transport, SAKE achieves more robust fact editing than methods like ROME/MEMIT, significantly leading in logical entailment generalization and context robustness.

## Background & Motivation

**Background**: Knowledge Editing (KE) aims to precisely modify factual knowledge stored in LLMs without full fine-tuning. Existing methods are mainly divided into three categories: weight editing (ROME, MEMIT; directly modifying model parameters), external memory (GRACE, SERAC; training auxiliary networks to store edits), and in-context editing (IKE; injecting new knowledge into prompts). Evaluation dimensions mainly include edit Accuracy, Generality (generalization to paraphrases), and Specificity (retaining unrelated knowledge).

**Limitations of Prior Work**: Existing KE methods suffer from three systemic drawbacks. (1) Poor logical entailment generalization: after modifying "The US president is X", the model often fails to correctly answer compositional reasoning questions such as "Who is the son of the US president?". Experiments by Cohen et al. (2024) show that ROME achieves only 16.7% accuracy on compositional reasoning (Compositionality II). (2) Poor context robustness: in conversational scenarios, a single query of doubt (e.g., "Are you sure?") can cause the edited model to revert to its original answer. (3) Poor flexibility: weight editing and external memory methods cannot easily undo a specific edit, and reverse editing can even severely harm the overall model performance.

**Key Challenge**: The root cause of the problem is that existing methods equate "knowledge" with a "single prompt." A fact in language corresponds to a distribution—including all paraphrases, logical entailments, and contextual variants—but methods like ROME only locate and modify parameters based on a single prompt $(s, r)$, leading to overfitting on this prompt and failing to generalize to other samples in the distribution.

**Goal**: To redefine knowledge editing from "single-prompt mapping" to "distribution-to-distribution mapping", and design an activation steering-based method that covers both paraphrasing and logical entailment, thereby achieving more robust knowledge editing.

**Key Insight**: The authors observe that all prompts related to a fact form a distribution in the final hidden layer activation space of LLMs. Mapping this distribution from the "old fact" to the "new fact" can accomplish the edit. This mapping can be efficiently computed using the closed-form solution of optimal transport theory without gradient optimization or modifying model weights.

**Core Idea**: Generate paraphrases and logical entailments of the edited fact using GPT-4 to construct source/target distributions in the LLM activation space, and then replace activation vectors during inference using a linear mapping of optimal transport to achieve robust knowledge editing.

## Method

### Overall Architecture

The pipeline of SAKE consists of two phases: training and inference. Training phase: Given an edit $(s, r, o \to o^*)$, (1) GPT-4 is first used to generate a set $P_e$ of $n$ paraphrased and logically entailed prompts; (2) These prompts are run through the original model and the model with a steering context, respectively, to collect activation vectors of the last token at the last layer, forming the source distribution $\mathcal{S}_e$ and target distribution $\mathcal{T}_e$; (3) The affine transformation $m: h \to \mathbf{A}h + \mathbf{b}$ is learned from $\mathcal{S}_e$ to $\mathcal{T}_e$ via the linear mapping of optimal transport (the Gaussian closed-form solution of the Monge map). Inference phase: For a new input, a distance threshold is first used to judge whether it falls within the editing scope $\mathcal{X}_e$. If so, the activations are collected, replaced using the mapping $m$, and then generation continues.

### Key Designs

1. **Distribution Modeling of Fact Editing**:

    - **Function**: Expands a single edited fact into a set of prompts covering paraphrasing and logical entailments, forming a meaningful distribution in the activation space.
    - **Mechanism**: Supports two generation strategies: "agent generation" (using GPT-4 to generate paraphrases, subject aliases, multi-hop reasoning questions, etc., based on instructions) and "expert generation" (manually written). The activations of the source distribution are collected by directly feeding the prompts into the original model. The activations of the target distribution are collected by prefixing the prompts with a steering context (e.g., "Do not mention $o$. Repeat: $p_i + o^*$"), allowing the original model to output the new answer without modifying parameters, and then collecting the corresponding activations.
    - **Design Motivation**: This is the core to solving the "single-prompt overfitting" problem—by explicitly modeling the scope of the edit's influence, subsequent mapping can cover paraphrasing and logical entailment. Ablation experiments show that 50 prompts are sufficient to achieve Accuracy of 0.92 and Generality of 0.84.

2. **Optimal Transport Linear Mapping**:

    - **Function**: Learns an affine transformation from the source activation distribution to the target activation distribution, and replaces activations during inference to change the model's output.
    - **Mechanism**: Assuming the source and target activations are approximately Gaussian, the closed-form solution of the Monge map is used: $\mathbf{A} = \Sigma_s^{-1/2}(\Sigma_s^{1/2}\Sigma_t\Sigma_s^{1/2})^{1/2}\Sigma_s^{-1/2}$, $\mathbf{b} = \mu_t - \mathbf{A}\mu_s$, where $\mu_s, \mu_t, \Sigma_s, \Sigma_t$ are the empirical means and covariance matrices of the source/target distributions. Compared to simple mean shift (ActAdd's $h + (\mu_t - \mu_s)$), the OT mapping matches both the mean and the covariance, avoiding the "bias-by-neighbors" problem.
    - **Design Motivation**: Ablation experiments show that Uniform Steering only achieves 35-41% in Generality, whereas the OT mapping reaches 82-85%, proving that covariance matching is crucial for generalization. The closed-form solution requires no gradient optimization and is computationally highly efficient.

3. **Scope Detection**:

    - **Function**: Determines whether a new input falls within the influence scope of a certain edit during inference to decide whether to apply the activation mapping.
    - **Mechanism**: Computes the distance between the activation vector $h$ of the new input and the center of the source distribution $\bar{h}^s$. If $\|h - \bar{h}^s\| < \epsilon$, it is determined to be within the edit scope and the mapping is applied; otherwise, the original behavior is preserved. The threshold $\epsilon$ controls the trade-off between generalization and specificity.
    - **Design Motivation**: Since the mapping of each edit is independent, adding/deleting edits only requires adding/removing the corresponding mappings and distributions without affecting other edits, solving the flexibility problem. Although a simple distance threshold is not optimal, it is computationally efficient and sufficient to demonstrate the effectiveness of the method.

### Loss & Training

SAKE does not involve gradient training in the traditional sense. The mapping $m$ is directly computed via the closed-form solution of OT, requiring only the mean and covariance matrices of the source/target activations. The model weights are completely frozen, and all modifications are implemented via activation replacement at inference time.

## Key Experimental Results

### Main Results

Traditional KE metrics on the Counterfact dataset (2000 edits):

| Method | Model | Accuracy | Generality | Specificity |
|------|------|----------|------------|-------------|
| ROME | GPT2-XL | 99.55 | 73.70 | 82.67 |
| MEMIT | GPT2-XL | 60.00 | 36.60 | 67.21 |
| ActAdd | GPT2-XL | 85.00 | 29.78 | 82.75 |
| **SAKE** | GPT2-XL | **97.00** | **84.85** | **84.52** |
| ROME | LLaMA 2-7B | 99.95 | 68.20 | 93.48 |
| MEMIT | LLaMA 2-7B | 74.40 | 55.13 | 74.37 |
| **SAKE** | LLaMA 2-7B | **97.70** | **82.03** | **85.59** |

Logical entailment generalization metrics on the Popular dataset (GPT2-XL):

| Method | CI (Multi-hop I) | CII (Multi-hop II) | SA (Subject Alias) | RS (Relation Specificity) |
|------|-----------|-------------|-------------|-------------|
| ROME | 38.62 | 16.67 | 51.96 | 39.43 |
| MEMIT | 2.47 | 1.95 | 7.17 | 3.75 |
| ActAdd | 26.63 | 29.17 | 42.12 | 50.68 |
| **SAKE** | **50.00** | **33.33** | **54.59** | **58.39** |

### Ablation Study

| Ablation Configuration | Accuracy | Generality | Description |
|---------|----------|------------|------|
| SAKE (OT Mapping) | 97.00 | 84.85 | Full model |
| Uniform Steering (Mean Shift) | 85.05 | 35.45 | Without OT, Generality drops by 49% |
| 50 training prompts | 92.0 | 84.0 | 50 prompts are sufficient |
| 10 training prompts | ~85 | ~70 | Performance decreases significantly |

Context robustness (GPT2-XL, query of doubt prompts):

| Method | DI (Indirect Doubt) | DII (Direct Doubt) |
|------|-------------|--------------|
| ROME | 33.33 | 14.00 |
| ICL | 4.00 | 3.33 |
| ActAdd | 82.00 | 80.67 |
| **SAKE** | **98.67** | **98.67** |

### Key Findings

- **OT mapping is the key to improving Generality**: From mean shift to OT mapping, Generality increases from 35.45% to 84.85%, indicating that matching the covariance structure is crucial for covering paraphrasing variants.
- **Distribution modeling is effective but does not apply to MEMIT**: Attempting to feed the generated logical entailment prompts as multiple edits into MEMIT (CompMEMIT) performs worse than the original ROME, showing that weight editing methods cannot easily utilize data augmentation.
- **Context robustness outperforms all competitors**: SAKE achieves a 98.67% retention rate in doubt scenarios, compared to only 14-33% for ROME and 3-4% for ICL, proving that distribution mapping in the activation space is more stable than parameter modification or in-context injection.
- **50 prompts are sufficient**: Ablation on the number of training prompts shows that 50 paraphrased prompts are enough to achieve close-to-saturated performance, keeping the cost controllable.

## Highlights & Insights

- **Redefining knowledge editing as a distribution mapping problem**: This shift in perspective is the most important contribution of this paper—the defects of existing methods do not lie in the editing technique itself, but in oversimplifying "facts" as "single prompts." From the perspective of distribution, the OT mapping emerges naturally as an elegant solution.
- **Engineering advantages of closed-form solutions**: The linear OT mapping has a closed-form solution under the Gaussian assumption, requiring no iterative optimization, having extremely low computational costs, and yielding reproducible results. Adding/deleting edits is also an O(1) operation, facilitating continuous updates.
- **Generality of activation steering**: Since model weights are not modified, multiple independent edit mappings can be maintained simultaneously without interfering with each other. This workflow could be extended to other scenarios requiring precise control over LLM behavior (e.g., safety alignment, style transfer).

## Limitations & Future Work

- **Completeness of distribution modeling**: Can the current prompt set generated by GPT-4 cover all logical entailments? Reverse relationships (e.g., "Which country is X the president of?") are not covered and are difficult to integrate.
- **Overly simplistic scope detection**: The detection mechanism based on the Euclidean distance threshold may suffer from overlapping distributions when there are many edits, leading to false positives or false negatives. A more sophisticated classifier is needed.
- **Accuracy slightly lower than ROME**: SAKE's Accuracy (97%) on Counterfact is slightly lower than ROME's (99.55%), indicating that scope detection occasionally misses precisely matched prompts.
- **Dependency on external LLMs to generate prompts**: Distribution modeling relies on GPT-4, which increases cost and dependency on external APIs.

## Related Work & Insights

- **vs ROME/MEMIT**: These methods directly modify model weights, performing excellently on precise matching prompts but poorly on generalization; SAKE does not modify weights and achieves better generalization through activation mapping.
- **vs ActAdd**: ActAdd also uses activation steering but only employs mean shift vectors, failing to match the covariance structure of the distribution; SAKE's OT mapping leads by about 50 percentage points in Generality.
- **vs IKE (In-Context Editing)**: In-context methods are extremely fragile in doubt scenarios (retaining only 3-4%), whereas SAKE operates in the activation space and is unaffected by prompt content.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Redefining knowledge editing as a distribution mapping problem solved via OT offers a novel perspective and elegant theory.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers traditional metrics, logical entailments, context robustness, and ablations, though only verified on GPT2-XL and LLaMA 2-7B.
- **Writing Quality**: ⭐⭐⭐⭐ Clear elaboration of the problem motivation, natural derivation of the methodology, and highly informative figures and tables.
- **Value**: ⭐⭐⭐⭐ The knowledge editing paradigm based on OT mapping possesses high versatility and scalability potential, offering strong inspiration for future work.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Model Editing as a Double-Edged Sword: Steering Agent Ethical Behavior](../../AAAI2026/knowledge_editing/model_editing_as_a_double-edged_sword_steering_agent_ethical_behavior_toward_ben.md)
- [\[ACL 2025\] Efficient Knowledge Editing via Minimal Precomputation](efficient_knowledge_editing.md)
- [\[ACL 2025\] ScEdit: Script-based Assessment of Knowledge Editing](scedit_script-based_assessment_of_knowledge_editing.md)
- [\[ACL 2025\] Revealing the Deceptiveness of Knowledge Editing: A Mechanistic Analysis of Superficial Editing](revealing_the_deceptiveness_of_knowledge_editing_a_mechanistic_analysis_of_super.md)
- [\[ACL 2025\] Context-Robust Knowledge Editing for Language Models](context-robust_knowledge_editing_for_language_models.md)

</div>

<!-- RELATED:END -->
