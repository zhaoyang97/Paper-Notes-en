---
title: >-
  [Paper Note] ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging
description: >-
  [ACL 2025][LLM Safety][Machine Unlearning] This work achieved second place in SemEval-2025 Task 4 (LLM Sensitive Content Unlearning). The core mechanism is to train two complementary models (one over-forgetting, one under-forgetting) and merge them via TIES-Merging to obtain a balanced unlearning model, achieving a near-perfect MIA score of 0.501 in local experiments.
tags:
  - "ACL 2025"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Model Merging"
  - "TIES-Merging"
  - "Privacy Protection"
date: 2026-05-08
content_hash: 3bc23e0e139890b5
---

# ZJUKLAB at SemEval-2025 Task 4: Unlearning via Model Merging

**Conference**: ACL 2025  
**arXiv**: [2503.21088](https://arxiv.org/abs/2503.21088)  
**Code**: Yes ([https://github.com/zjunlp/unlearn/tree/main/semeval25](https://github.com/zjunlp/unlearn/tree/main/semeval25))  
**Area**: NLP / Machine Unlearning  
**Keywords**: Machine Unlearning, Model Merging, TIES-Merging, Privacy Protection, LLM Safety

## TL;DR

This work achieved second place in SemEval-2025 Task 4 (LLM Sensitive Content Unlearning). The core mechanism is to train two complementary models (one over-forgetting, one under-forgetting) and merge them via TIES-Merging to obtain a balanced unlearning model, achieving a near-perfect MIA score of 0.501 in local experiments.

## Background & Motivation

Machine unlearning is a critical technology in AI safety, aiming to selectively remove sensitive data (such as copyrighted materials and personal information) from trained models. However, existing unlearning methods face fundamental difficulties:

**Over-forgetting**: Removing excessive non-sensitive information, leading to an overall degradation of model performance.

**Under-forgetting**: Failing to completely remove the targeted sensitive data.

**Hyperparameter Sensitivity**: Finding optimal hyperparameters to balance performance across multiple evaluation dimensions is extremely difficult, and sometimes even impossible.

The authors' key insight is that instead of trying to find a perfect set of hyperparameters, it is better to train two biased models and combine their strengths through model merging.

## Method

### Overall Architecture

A two-stage system: Training Stage $\rightarrow$ Merging Stage

### Key Designs

1. **Training Stage—Two Complementary Models**:
   OLMo-7B-0724-Instruct is fine-tuned using LoRA (Low-Rank Adaptation) with identical training objectives but different hyperparameters, yielding:
    - **model₁ (Over-forgetting)**: High Task Aggregate (0.968) but low MIA Score (0.022)—forgetting too much, even what should not be forgotten.
    - **model₂ (Under-forgetting)**: Low Task Aggregate (0.659) but high MIA Score (0.818)—retaining too much information that should be forgotten.

   The training loss function consists of three components:
    - **NPO (Negative Preference Optimization)**: Minimizes the probability of target tokens on the forget set.
    - **GDR (Gradient Descent on Retain set)**: Maintains the model's original capabilities on the retain set.
    - **KLR (KL Divergence Minimization on Retain set)**: Ensures that the output distribution of the unlearned model does not shift on the retain set.

   Total loss: `$L_{\text{total}} = \alpha \cdot L_{\text{npo}} + \beta \cdot L_{\text{gdr}} + \gamma \cdot L_{\text{klr}}$`

2. **Merging Stage—TIES-Merging**:
   Merging the LoRA adapters of the two models in three steps:
    - **Trimming**: Retaining the most important parameters according to a density threshold and setting the rest to zero. A density of 0.8 achieves the best performance—lower densities over-prune, while higher densities introduce redundancy.
    - **Electing**: Creating a unified sign vector based on parameter absolute values to resolve parameter direction conflicts.
    - **Disjoint Merging**: Averaging only the non-zero parameters with consistent signs, discarding conflicting parameters.

### Loss & Training

- NPO Loss: Similar to DPO but only targets the forget set, negatively optimizing to reduce the model's generation probability of the forgotten data.
- GDR Loss: Standard cross-entropy, maintaining model capability on the retain set.
- KLR Loss: Minimizes the KL divergence between the unlearned model and the original model on the retain set.
- The two models achieve complementary biases through different combinations of $\alpha$, $\beta$, and $\gamma$.

## Key Experimental Results

### Online and Local Experimental Results

| Environment | Method | Aggregate | Task Aggregate | MIA Score/AUC | MMLU Avg. |
|------|------|-----------|---------------|---------------|-----------|
| Online | AILS-NTUA (First) | 0.706 | 0.827 | 0.847/– | 0.443 |
| Online | **ZJUKLAB (Ours, Second)** | **0.487** | **0.944** | 0.048/– | 0.471 |
| Local | model₁ (Over-forgetting) | 0.481 | 0.968 | 0.045/0.022♣ | 0.431 |
| Local | model₂ (Under-forgetting) | 0.504 | 0.659 | 0.364/0.818♠ | 0.491 |
| Local | **Merged (Ours)** | **0.806** | **0.939** | **0.997/0.501♡** | **0.480** |

### Comparison of Merging Methods

| Merging Method | Aggregate |
|---------|-----------|
| Linear | 0.244 |
| DARE-Linear | 0.440 |
| DARE-TIES | 0.561 |
| Magnitude Prune | 0.558 |
| **TIES** | **0.806** |

### Density Parameter Ablation

| Density | Effect |
|------|------|
| 0.6 | Over-pruning, MIA too low |
| **0.8** | **Optimal balance** |
| 1.0 | Introduces redundancy, MIA slightly high |

### Key Findings

1. **Model merging paradigm is highly effective**: In local experiments, the MIA AUC of the merged model reaches 0.501 (near-perfect), and the Aggregate score jumps from ~0.49 of the two sub-models to 0.806.
2. **TIES-Merging substantially outperforms other merging methods**: Outperforming Linear by 0.562 and DARE-TIES by 0.245.
3. **Training process analysis**: The Regurgitation Score and Knowledge Score decrease simultaneously in the early stages of training (epochs 0-0.8). After that, the Knowledge Score steadily rises while the Regurgitation Score fluctuates upward, reflecting a gradual divergence in the optimization directions of unlearning and retention.
4. **Parameter shift direction analysis**: The angle between the parameter change vectors before and after the training inflection point is approximately 70-85 degrees, indicating a significant shift in the optimization direction.

## Highlights & Insights

- The idea of **"merging two flawed models to get a good one"** is simple yet profound, bypassing the difficulty of finding a perfect balance within a single model.
- Detailed analyses of the training process (performance trajectory, loss dynamics, and parameter angle analysis) provide a deep understanding of the unlearning process.
- Critical analysis of unlearning evaluation metrics is valuable: ROUGE measures textual overlap rather than knowledge leakage, and MIA also suffers from reliability issues.
- The observation that over-forgetting causes model collapse (generating repetitive "6 6 6" outputs) serves as an important warning for real-world deployment.

## Limitations & Future Work

1. **Over-forgetting is not completely resolved**: The merged model may still produce degenerate outputs.
2. **Unrelated knowledge side-effects**: The unlearning operation degrades accuracy on unrelated knowledge (e.g., "What is the capital of France") from 0.88 to 0.35.
3. **Reliability of evaluation metrics**: ROUGE-L fails to detect semantic-level information leakage, and the reliability of MIA scores is questionable.
4. **Online vs. Local discrepancy**: The large gap between the online MIA Score (0.048) and the local one (0.997) is likely due to dataset mismatches.
5. The current paradigm lacks positive guidance (such as reinforcement learning), leaving the model to degenerate under negative optimization pressure.

## Related Work & Insights

- NPO (Zhang et al., 2024) is the core unlearning algorithm; this work overcomes its issue of over/under-forgetting when used in isolation by using model merging.
- TIES-Merging (Yadav et al., 2023), originally designed for multi-task model merging, is cleverly transferred to the unlearning scenario.
- The International Scientific Report on the Safety of Advanced AI (Bengio et al., 2025) points out the insufficiency of current unlearning methods, which aligns with findings in this paper.
- Future directions: Incorporating positive signals by combining data augmentation and reinforcement learning, and exploring unlearning on demand.

## Rating

- **Novelty**: ⭐⭐⭐⭐ The "over-forgetting + under-forgetting $\rightarrow$ merged balance" idea is simple and elegant.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Online competition validation + detailed local analysis + multi-dimensional ablation + comparison of merging methods.
- **Writing Quality**: ⭐⭐⭐⭐ In-depth and thorough analysis, especially the discussions and reflections in Sec. 5-6.
- **Value**: ⭐⭐⭐⭐ Inspiring for both methodology and evaluation in the machine unlearning field.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Lacuna Inc. at SemEval-2025 Task 4: LoRA-Enhanced Influence-Based Unlearning for LLMs](lacuna_inc_at_semeval-2025_task_4_lora-enhanced_influence-based_unlearning_for_l.md)
- [\[ACL 2025\] CLIPErase: Efficient Unlearning of Visual-Textual Associations in CLIP](cliperase_efficient_unlearning_of_visual-textual_associations_in_clip.md)
- [\[ICML 2025\] NegMerge: Sign-Consensual Weight Merging for Machine Unlearning](../../ICML2025/llm_safety/negmerge_sign-consensual_weight_merging_for_machine_unlearning.md)
- [\[ACL 2025\] Merge Hijacking: Backdoor Attacks to Model Merging of Large Language Models](merge_hijacking_backdoor_attacks_to_model_merging_of_large_language_models.md)
- [\[ACL 2025\] ReLearn: Unlearning via Learning for Large Language Models](relearn_unlearning_via_learning_for_large_language_models.md)

</div>

<!-- RELATED:END -->
