---
title: >-
  [Paper Note] LSSF: Safety Alignment via Low-Rank Safety Subspace Fusion
description: >-
  [ACL 2025][LLM Alignment][safety alignment] LSSF proposes the hypothesis that the safety information of LLMs resides in a low-rank subspace. It extracts the principal components of the safety-aligned model via SVD, adaptively determines the kept rank for each layer using safety singular value entropy, and finally linearly fuses the extracted safety principal components into the fine-tuned model. This restores the safety alignment degraded by fine-tuning without any additional…
tags:
  - "ACL 2025"
  - "LLM Alignment"
  - "safety alignment"
  - "low-rank subspace"
  - "SVD"
  - "fine-tuning robustness"
  - "post-hoc safety"
  - "singular value entropy"
date: 2026-05-08
content_hash: eb0180682f81a2c6
---

# LSSF: Safety Alignment via Low-Rank Safety Subspace Fusion

**Conference**: ACL 2025  
**arXiv**: [2602.00038](https://arxiv.org/abs/2602.00038)  
**Code**: None  
**Area**: LLM Alignment / Safety  
**Keywords**: safety alignment, low-rank subspace, SVD, fine-tuning robustness, post-hoc safety, singular value entropy

## TL;DR

LSSF proposes the hypothesis that the safety information of LLMs resides in a low-rank subspace. It extracts the principal components of the safety-aligned model via SVD, adaptively determines the kept rank for each layer using safety singular value entropy, and finally linearly fuses the extracted safety principal components into the fine-tuned model. This restores the safety alignment degraded by fine-tuning without any additional training, while maintaining downstream task performance.

## Background & Motivation

**Pervasiveness of fine-tuning breaking safety alignment**: Safety-aligned LLMs lose their safety alignment capability significantly after being fine-tuned on downstream tasks, even when completely benign data is used. Just a few dozen harmful samples can "jailbreak" an aligned model. This phenomenon has been widely observed in mainstream models like Llama and Qwen.

**Safety-utility entanglement**: During fine-tuning, task-related parameter updates interfere with the parameters encoding safety knowledge—both are entangled in the full parameter space. Simple freezing of certain layers or regularization methods show limited effectiveness since it is unknown which specific parameters contain safety information.

**Limitations of prior work**: Safety fine-tuning (mixing safety data into fine-tuning data) requires modifying the training pipeline and is not always effective; DPO safety alignment requires additional preference data and training; inference-time defenses (e.g., safety prompts) are easily bypassed.

**Need for post-hoc methods**: An ideal solution should be applied after fine-tuning is completed without interfering with the fine-tuning process itself—especially for API providers, who need to restore safety after users fine-tune models on their own.

**Intuition behind the low-rank hypothesis**: Safety alignment training primarily teaches the model to "refuse harmful requests," which is a relatively simple behavioral pattern. Compared to complex language and reasoning capabilities, safety knowledge likely resides in a lower-dimensional parameter subspace.

**Ours Novelty**: (a) Proposes and experimentally validates the low-rank safety subspace hypothesis; (b) proposes safety singular value entropy to automatically determine the kept rank of each layer; (c) designs a training-free safety principal component fusion operation to restore safety alignment.

## Method

### Overall Architecture

The full pipeline of LSSF: (1) Obtain safety vectors: calculate the difference between weight matrices of the aligned model and the unaligned base model for each layer; (2) SVD factorization: perform singular value decomposition on the difference matrices; (3) Safety singular value entropy calculation: adaptively determine the kept rank r for each layer using the entropy of the singular value distribution; (4) Low-rank safety principal component extraction: retain the components corresponding to the top r singular values; (5) Linear fusion: add the safety principal components to the corresponding layers of the fine-tuned model.

### Key Designs

**1. Safety Vector Extraction**

- **Function**: Defines the "safety vector" by contrasting the parameter differences between the aligned model and the base model.
- **Mechanism**: Calculates weight difference matrices layer by layer, which encode all parameter changes introduced by safety training.
- **Design Motivation**: Most parameter changes introduced by safety training (such as RLHF/DPO) are safety-related; the "safety increment" can be isolated via the difference.

**2. SVD Low-Rank Decomposition and Safety Principal Components**

- **Function**: Performs SVD on the safety vectors to extract low-rank components carrying primary safety information.
- **Mechanism**: Observes that singular values decay rapidly—the first few singular values contain most of the safety information. Retaining the first r components is sufficient to reconstruct the key characteristics of safety behavior.
- **Design Motivation**: Low-rank hypothesis—safety behaviors (refusing, apologizing, warning) are much simpler than general language capabilities, so the information is naturally concentrated in a small number of principal components.

**3. Safety Singular Value Entropy**

- **Function**: Adaptively determines the optimal kept rank for each layer instead of setting it manually or uniformly.
- **Mechanism**: Computes the normalized entropy of the singular value distribution for each layer. Low entropy means safety information is concentrated in a few components (requiring smaller rank), while high entropy means information is sparse (requiring larger rank).
- **Design Motivation**: Different layers encode safety information with different densities—the safety information distribution characteristics of attention layers and FFN layers differ. A uniform rank would cause information deficiency in some layers and introduce noise in others.

**4. Linear Fusion Operation**

- **Function**: Adds the extracted safety principal components to the corresponding layers of the fine-tuned model.
- **Mechanism**: Fine-tuned weights + scaling coefficient x low-rank safety principal components = safety-restored weights.
- **Design Motivation**: Linear addition guarantees retrieval of downstream task capabilities (fine-tuned parameters are not overwritten) while injecting safety information. This is analogous to a LoRA addition operation, but oriented towards "safety recovery."

### Loss & Training

LSSF is a completely training-free post-hoc method. It involves no loss functions or gradient optimization. The only hyperparameter is the fusion coefficient (typically between 0.5-1.0), which is determined via a grid search on a small safety validation set.

## Key Experimental Results

### Main Results

| Model | Fine-tuning Task | Method | Downstream ACC | AdvBench Refusal Rate | HarmfulQA | CATQA |
|------|---------|------|---------|--------------|-----------|-------|
| Qwen2.5-7B | AG's News LoRA | Fine-tuned | 0.94 | 0.12 | 0.15 | 0.18 |
| Qwen2.5-7B | AG's News LoRA | SafeLoRA | 0.91 | 0.85 | 0.82 | 0.79 |
| Qwen2.5-7B | AG's News LoRA | **LSSF** | **0.92** | **1.00** | **0.98** | **0.93** |
| Llama3.1-8B | AG's News LoRA | Fine-tuned | 0.93 | 0.08 | 0.11 | 0.14 |
| Llama3.1-8B | AG's News LoRA | SafeLoRA | 0.90 | 0.89 | 0.87 | 0.83 |
| Llama3.1-8B | AG's News LoRA | **LSSF** | **0.92** | **0.99** | **0.99** | **0.99** |

### Ablation Study

| Ablation | AdvBench | Downstream ACC | Analysis |
|--------|----------|---------|------|
| Full LSSF | 1.00 | 0.92 | Baseline |
| Fixed Rank (r=10) | 0.91 | 0.91 | Uniform rank is inferior to adaptive rank; some layers lack information |
| Fixed Rank (r=50) | 0.97 | 0.88 | Excessive rank introduces noise and degrades downstream task performance |
| No SVD (Directly Adding Safety Difference) | 0.95 | 0.83 | Full rank introduces too much noise, severely damaging task performance |
| Fusion Coefficient = 0.3 | 0.82 | 0.93 | Insufficient fusion, safety recovery is incomplete |
| Fusion Coefficient = 0.7 | 0.98 | 0.91 | Good balance |
| Fusion Coefficient = 1.0 | 1.00 | 0.89 | Optimal safety but slightly decreased task performance |

### Key Findings

- Safety information indeed exhibits low-rank characteristics: In most layers, the top 5-15 singular values already capture 90%+ of the safety information.
- Safety singular value entropy varies significantly across different layers: The entropy of attention layers is typically lower than that of FFN layers, suggesting that safety information in attention layers is more concentrated.
- LSSF approaches or even matches the raw safety alignment level on almost all safety metrics (AdvBench 1.00) while losing less than 2% of downstream task performance.
- The effect is particularly prominent on Llama3.1-8B: All safety metrics >= 0.99, indicating a stronger low-rank structure of safety information in this model.
- Compared to methods like SafeLoRA, LSSF completely leads in safety recovery without requiring any training.

## Highlights & Insights

- **Proposal and validation of the low-rank safety subspace hypothesis**: This is a theoretically significant finding—although safety behaviors manifest diversely (refusals, explanations, warnings), their information in the parameter space is low-rank and can be captured by a few principal components.
- **Novelty of safety singular value entropy**: Determining the kept rank adaptively layer-by-layer avoids info deficiency or excessive noise caused by blindly picking a uniform rank—this metric itself holds independent research value.
- **Completely training-free**: A post-hoc linear operation that requires no extra safety data, loss functions, or GPU training, offering high practicality—suitable for API providers to offer "safety-recovery-as-a-service."
- **Elegant analogy to LoRA**: While LoRA performs task adaptation in a low-rank space, LSSF performs safety recovery in a low-rank space—both are mathematically dual.

## Limitations & Future Work

- Reliance on the availability of base models (unaligned versions)—some aligned models do not release their base versions.
- Linear fusion assumption: The interaction between safety information and task information in the parameter space might not be entirely linear.
- The fusion coefficient requires a small-scale grid search, which is simple but still requires a safety validation set.
- Not fully validated under full-parameter fine-tuning (non-LoRA) scenarios—it is unknown whether the safety subspace remains stable when fine-tuning scales up.
- Only covers English safety evaluations; validation of cross-lingual safety recovery remains to be performed.
- For continuous fine-tuning (multi-round) scenarios, whether the safety subspace can be reused is worth exploring.

## Related Work & Insights

- **vs SafeLoRA**: SafeLoRA projects out components aligned with safety vectors in LoRA updates, taking the approach of "pruning harmful updates"; LSSF takes the approach of "adding back safety components"—opposite but complementary.
- **vs Safety Tuning**: Mixing safety samples into fine-tuning data requires modifying the training pipeline and yields unstable results; LSSF is completely post-hoc and does not hook into fine-tuning.
- **vs Representation Engineering**: RepE injects safety direction vectors during inference, incurring additional overhead each run; LSSF modifies model parameters one-off with no inference-time cost.
- **vs Model Merging**: The linear fusion operation in LSSF shares a similar formulation with model merging methods (like TIES, DARE), but focuses on selective fusion along the safety dimension.
- **Insight**: The concept of a low-rank subspace can be extended to other attributes—e.g., does creativity or multi-lingual capability exhibit a similar low-rank structure?

## Rating

- Novelty: ⭐⭐⭐⭐ The low-rank safety subspace hypothesis is novel and backed by experiments.
- Experimental Thoroughness: ⭐⭐⭐⭐ Multiple models + multiple safety evaluations + thorough ablations.
- Writing Quality: ⭐⭐⭐⭐ Clear logical chain of hypothesis-validation-application.
- Value: ⭐⭐⭐⭐⭐ Training-free safety recovery has high practical utility.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] SEA: Low-Resource Safety Alignment for Multimodal Large Language Models via Synthetic Embeddings](sea_lowresource_safety_alignment_for_multimodal.md)
- [\[ACL 2025\] Safety Alignment via Constrained Knowledge Unlearning](safety_alignment_via_constrained_knowledge_unlearning.md)
- [\[ACL 2025\] Atyaephyra at SemEval-2025 Task 4: Low-Rank Negative Preference Optimization](atyaephyra_at_semeval-2025_task_4_low-rank_negative_preference_optimization.md)
- [\[NeurIPS 2025\] Towards Understanding Safety Alignment: A Mechanistic Perspective from Safety Neurons](../../NeurIPS2025/llm_alignment/towards_understanding_safety_alignment_a_mechanistic_perspective_from_safety_neu.md)
- [\[ACL 2025\] PKU-SafeRLHF: Towards Multi-Level Safety Alignment for LLMs with Human Preference](pku-saferlhf_towards_multi-level_safety_alignment_for_llms_with_human_preference.md)

</div>

<!-- RELATED:END -->
