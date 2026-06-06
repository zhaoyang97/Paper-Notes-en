---
title: >-
  [Paper Note] One Mask to Rule Them All: On Hidden Facts after Editing and How to Find Them
description: >-
  [ACL2026][Knowledge Editing][ROME] This paper discovers that ROME / MEMIT does not truly overwrite old knowledge but instead suppresses it through a shared overattention mechanism. A sparse binary mask can reverse the ma…
tags:
  - "ACL2026"
  - "Knowledge Editing"
  - "ROME"
  - "MEMIT"
  - "binary mask"
  - "overattention"
date: 2026-05-08
content_hash: e683ff49692e7063
---

# One Mask to Rule Them All: On Hidden Facts after Editing and How to Find Them

**Conference**: ACL2026  
**arXiv**: [2605.28839](https://arxiv.org/abs/2605.28839)  
**Code**: https://github.com/holmov1/one-mask-ke  
**Area**: Knowledge Editing / Mechanistic Interpretability  
**Keywords**: ROME, MEMIT, knowledge editing, binary mask, overattention

## TL;DR
This paper discovers that ROME / MEMIT does not truly overwrite old knowledge but instead suppresses it through a shared overattention mechanism. A sparse binary mask can reverse the majority of edits and reduce the success rate of new edits from 98% to 38%.

## Background & Motivation

**Background**: Knowledge editing aims to update specific facts without retraining LLMs. Locate-and-edit methods like ROME and MEMIT identify MLP weights related to target facts and modify them directly, an approach widely interpreted as "overwriting old facts with new ones." Common assessments primarily focus on output behavior: whether the model outputs the new object for the edited prompt.

**Limitations of Prior Work**: Focusing only on output behavior does not confirm that internal knowledge has been truly rewritten. Factual knowledge in Transformers typically exists across redundant paths and self-repair mechanisms. If knowledge is distributed across multiple layers and paths, why does modifying a single layer or a few consecutive layers allow the model to stably output new facts? This creates tension with the "true overwriting" claim.

**Key Challenge**: If ROME / MEMIT truly introduces fact-specific updates for every fact, different edits should rely on different weight locations. However, if multiple types of edits can be reversed by the same mask, it indicates that these methods share a functional mechanism rather than just writing individual new facts.

**Goal**: The authors aim to answer three questions: whether a single mask can reverse edits for a large number of different facts; whether this mask can generalize to unseen edits and relations; and what internal mechanism it disrupts when reversing edits.

**Key Insight**: A compact binary mask is trained on edited MLP weight matrices. The mask does not retrain the original model or modify unedited layers; it only learns "which edited weights are necessary to maintain the edit." If setting a small number of weights to zero restores the original fact, it indicates the old knowledge remains in the model.

**Core Idea**: Knowledge editing success relies on an overattention subspace shared across facts. The learned mask restores old knowledge by eliminating overattention in subsequent layers rather than simply rolling back the largest weight changes.

## Method

The paper first applies ROME / MEMIT to edit factual knowledge in the model, then fixes all edited weights and trains only a binary mask. This mask is applied to the edited weight matrix to create a pruned model. The training objective is to make the pruned model prefer the original object over the edited object while minimizing the number of pruned weights and keeping the overall language modeling distribution close to the original model.

### Overall Architecture

Given an original model $M$, a fact prompt $x$ contains a subject and relation (e.g., "Marie Curie discovered"), and the original object is "radium." Knowledge editing changes the object to $o^*$, resulting in the edited model $M_e$. The paper obtains a set of edited weight matrices $\{\hat{W}_1, \ldots, \hat{W}_N\}$ from a batch of different facts and learns a shared mask $K$.

During training, each sample is routed to its corresponding edited matrix $\hat{W}_i$, which is then multiplied element-wise by the shared mask $\hat{W}_i \odot K$. Only the mask parameters $\Theta$ receive gradients; the original model and edited weights remain frozen. Consequently, the learned mask cannot memorize any single fact and must capture the weight structure that multiple edits commonly depend on.

Experiments utilize CounterFact. ROME experiments involve training the mask on single edits; for MEMIT, which supports batch editing, the authors edit 1,000 facts simultaneously and train the mask only on specific editing layers. Models include GPT-2 XL (1.5B), LLaMA-3.2 (3B), and Qwen2.5 (7B). The training set consists of 3,000 samples covering 10 relations, while the test set includes 1,700 held-out samples. Some OOD experiments also use 1,000 additional ROME edits and 10 unseen relations.

### Key Designs

1. **Cross-Edit Shared Binary Mask**:
    - **Function**: To test whether different factual edits rely on the same functional weight locations.
    - **Mechanism**: A mask value of 1 indicates the edited weight is preserved, while 0 indicates its contribution is removed. The same $K$ is applied to editing matrices of various facts during training, with only $K$ allowed to update.
    - **Design Motivation**: If edits were fact-specific, a shared mask should not generalize. If the mask can reverse unseen edits, it proves ROME / MEMIT utilize a common mechanism.

2. **Triple-Constraint Training Objective**:
    - **Function**: To ensure the mask restores old facts without crudely damaging the model.
    - **Mechanism**: The restoration loss requires the pruned model to assign higher probability to the original object than the edited one; the sparsity loss limits the proportion of pruned weights; the KL preservation loss keeps the output distribution of the pruned model close to the original. The combined objective includes $\mathcal{L}_{KL}$, $\max(0,\mathcal{L}_{sparsity}-S_{max})$, and $\max(0,\mathcal{L}_{restoration}+\delta)$.
    - **Design Motivation**: Simply pursuing edit reversal might be achieved by breaking the model. Adding sparsity and KL constraints encourages the mask to locate critical pathways that specifically maintain the edit.

3. **Residual Stream Decomposition and Edit Blocking Experiments**:
    - **Function**: To explain what mechanism the mask eliminates when reversing edits and verify the necessity of that mechanism.
    - **Mechanism**: The authors use Logit Lens to decompose the contribution of MLPs and attention to the target token logit in the residual stream, comparing original, edited, and pruned models. Subsequently, the mask is injected prior to editing to see if ROME can still complete new edits by bypassing that subspace.
    - **Design Motivation**: A high RSR only shows the mask is effective, not why. Residual decomposition and blocking experiments together prove that overattention is sufficient to explain reversal and necessary for edit success.

### Loss & Training

Since binary masks are non-differentiable, the authors initialize trainable parameters $\Theta$ and use a sigmoid to obtain a soft mask $K \in (0,1)$, which is binarized using a threshold $\gamma$ during inference. The restoration loss is $\mathcal{L}_{restoration}=-[\log P_{M_p}(o|x)-\log P_{M_p}(o^*|x)]$, with a margin $\delta$ used to encourage the original object to significantly outperform the edited object. Sparsity loss calculates the ratio of pruned weights $\frac{1}{|K|}\sum(1-k_{a,b})$, and KL loss constrains behavioral consistency. Evaluation metrics include Reversal Success Rate (RSR), Top-1 Overlap, and WikiText-2 perplexity.

## Key Experimental Results

### Main Results

| Method | Model | Pruning Ratio | Train RSR / Top-1 | Test RSR / Top-1 | PPL (Orig / Edit / Mask) | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| ROME | GPT-2 XL | 10.0% | 83% / 78% | 82% / 77% | 17.80 / 44.51 / 25.68 | Minor pruning restores most edits and fixes PPL degradation |
| ROME | LLaMA-3.2 | 10.0% | 90% / 75% | 79% / 72% | 9.46 / 9.59 / 10.14 | Edits are more stable, yet the mask still reverses them |
| ROME | Qwen-2.5 | 15.3% | 87% / 77% | 68% / 66% | 7.43 / 8.07 / 7.76 | Larger models require higher pruning ratios |
| MEMIT | GPT-2 XL | 4.5% | 82% / 81% | 74% / 78% | 17.80 / 17.90 / 19.00 | Distributed edits also rely on a few critical weights |
| MEMIT | LLaMA-3.2 | 8.8% | 87% / 67% | 78% / 65% | 9.46 / 10.78 / 12.53 | Test RSR maintained at 78% |
| MEMIT | Qwen-2.5 | 7.9% | 88% / 67% | 70% / 60% | 7.43 / 7.66 / 7.74 | PPL barely changes but edits are still reversible |

### Ablation Study

| Analysis Item | Key Metric | Conclusion |
| :--- | :--- | :--- |
| Residual Old Fact | Edited model RSR / Top-1 nearly 0 | Edits successfully cover old answers in terms of output behavior |
| ROME GPT-2 XL Prob Amp | Orig Prob 0.045, Edit Prob 0.866, Cohen's d=3.78 | Edited facts are pushed abnormally high |
| MEMIT GPT-2 XL Prob Amp| 0.046 → 0.614, Cohen's d=1.54 | MEMIT is gentler but still significantly amplifies |
| LLaMA-3.2 Prob Amp | ROME: $2.46e{-5}$ → $1.96e{-4}$; MEMIT: $2.48e{-5}$ → $1.17e{-4}$ | Absolute prob is low but direction remains consistent |
| Edit Blocking | ROME editing success: 98% → 38% | The subspace identified by the mask is necessary for new edit success |
| OOD relations | GPT-2 XL RSR 77%, LLaMA-3.2 RSR 59% | The mask does more than just memorize training relations |
| Mask Structure | GPT-2 XL dim 214 pruned at 74.6%; LLaMA dim 1659 at 62.0% | Pruning concentrates on few output dimensions, but not full columns |
| $\Delta W$ Magnitude | GPT-2 XL: edited 18.47 vs masked 1.85; LLaMA: 3.91 vs 0.57 | Mask does not just prune largest updates; it finds functional paths |

### Key Findings

- A single mask can reverse approximately 80% of edits in the training set and maintain high RSR on test sets and OOD relations, strongly supporting the "shared edit mechanism" hypothesis.
- Residual stream decomposition reveals that after editing, the MLP path largely preserves the old knowledge trajectory; what is truly amplified is subsequent attention. The mask works by eliminating these attention spikes.
- For ROME, editing can significantly damage PPL; the mask can restore some language modeling capabilities (e.g., GPT-2 XL PPL drops from 44.51 to 25.68). For MEMIT, PPL changes are small, but the mask still reverses edits, indicating that the edit maintenance mechanism and overall perplexity damage are distinct phenomena.

## Highlights & Insights

- The most impactful conclusion is that "editing does not erase knowledge." Old facts remain in the MLP/residual paths but are suppressed by edit-induced overattention, allowing a small mask to resurface the old facts.
- Generalizing a single mask to different facts and unseen relations provides strong causal evidence. This goes further than mere observations of attention drift by showing that blocking this structure directly reverses or prevents edits.
- This work reinterprets overattention—previously seen as a side effect—as the core mechanism for edit success. This perspective explains why it is difficult for ROME / MEMIT to propagate changes to related facts: they suppress retrieval rather than updating the knowledge graph.
- The mask does not simply prune the largest $\Delta W$ but targets positions with smaller update magnitudes that are functionally critical, suggesting that knowledge editing defense cannot rely solely on the magnitude of weight changes.

## Limitations & Future Work

- **Methodological Scope Concentration**: The paper focuses on ROME and MEMIT, excluding other parameter-modifying or meta-learning editing methods like AlphaEdit, MEND, or MALMEN.
- **Exclusion of Parameter-Preserving Edits**: Methods such as retrieval-based memory, external memory, or in-context editing might not rely on the same MLP weight subspace; the mask conclusions cannot be directly extrapolated.
- **Dataset Focus**: While CounterFact is a standard benchmark for counterfactual editing, real-world knowledge updates, long contexts, and multi-hop associations may involve different internal mechanisms.
- **Mask as a Clue, Not a System**: In practice, detecting when a mask is needed, avoiding accidental damage to legitimate updates, and handling adaptive bypasses by attackers requires additional design.
- **Computational Cost Constraints**: The authors did not study meta-learning KE due to the high cost of training editing hypernetworks. Future work should compare internal mechanisms across different editing paradigms if resources permit.

## Related Work & Insights

- **Vs. ROME / MEMIT Hypotheses**: ROME / MEMIT are built on the intuition that MLP layers act as associative memory; this paper suggests output changes are more like attention hijacking than true knowledge overwriting.
- **Vs. Attention Drift / Superficial Editing**: Existing work views overattention to edited entities as a specificity failure; this paper demonstrates that overattention is the structural mechanism of the edit success itself.
- **Vs. Edit Detection Methods**: Existing research detects edited facts from internal representations; this paper provides a possible explanation: different edits leave a shared attention signature, allowing classifiers to learn universal patterns.
- **Inspiration**: Knowledge editing evaluation should look beyond rewrite success to include propagation to related facts, internal retrieval paths, attention/MLP decomposition, and reversibility. For security, small shared masks could serve as a tool for defending open-weight models against malicious edits.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ Using a shared mask to reverse and block edits directly challenges the default "overwritten knowledge" explanation with strong mechanistic insights.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers ROME/MEMIT, three models, train/test/OOD, PPL, and residual flow analysis; lacks broader editing methods and real-world update scenarios.
- **Writing Quality**: ⭐⭐⭐⭐☆ The chain of reasoning is clear and tabular data is persuasive; some mechanism diagrams require context from the text.
- **Value**: ⭐⭐⭐⭐⭐ Highly valuable for knowledge editing, security defense, and mechanistic interpretability, particularly in prompting the community to re-examine the boundaries of locate-and-edit methods.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Spectral Characterization and Mitigation of Sequential Knowledge Editing Collapse](spectral_characterization_and_mitigation_of_sequential_knowledge_editing_collaps.md)
- [\[ACL 2026\] HiEdit: Lifelong Model Editing with Hierarchical Reinforcement Learning](hiedit_lifelong_model_editing_with_hierarchical_reinforcement_learning.md)
- [\[ACL 2026\] Aligning Language Models with Real-time Knowledge Editing](aligning_language_models_with_real-time_knowledge_editing.md)
- [\[ICLR 2026\] EAMET: Robust Massive Model Editing via Embedding Alignment Optimization](../../ICLR2026/knowledge_editing/eamet_robust_massive_model_editing_via_embedding_alignment_optimization.md)
- [\[ACL 2026\] FABLE: Fine-grained Fact Anchoring for Unstructured Model Editing](fable_fine-grained_fact_anchoring_for_unstructured_model_editing.md)

</div>

<!-- RELATED:END -->
