---
title: >-
  [Paper Note] CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge
description: >-
  [ACL 2026][LLM Safety][Continual Unlearning] CURaTE proposes a behavior unlearning framework based on sentence embedding matching: a general unlearning embedder is trained pre-deployment (without using any unlearning set…
tags:
  - "ACL 2026"
  - "LLM Safety"
  - "Continual Unlearning"
  - "Real-time Unlearning"
  - "Behavior Unlearning"
  - "Sentence Embedding"
  - "Knowledge Preservation"
date: 2026-05-08
content_hash: 0b51c5303587aa7a
---

# CURaTE: Continual Unlearning in Real Time with Ensured Preservation of LLM Knowledge

**Conference**: ACL 2026 Findings  
**arXiv**: [2604.14644](https://arxiv.org/abs/2604.14644)  
**Code**: [GitHub](https://github.com/bsu1313/CURaTE)  
**Area**: Information Retrieval  
**Keywords**: Continual Unlearning, Real-time Unlearning, Behavior Unlearning, Sentence Embedding, Knowledge Preservation

## TL;DR
CURaTE proposes a behavior unlearning framework based on sentence embedding matching: a general unlearning embedder is trained pre-deployment (without using any unlearning sets); post-deployment, new unlearning requests are embedded and stored in a database in real-time. During inference, cosine similarity determines whether to answer or refuse, achieving near-perfect knowledge preservation by strictly avoiding any modifications to LLM weights.

## Background & Motivation

**Background**: Current LLM unlearning methods primarily include parameter-modifying approaches such as Gradient Ascent (GA), Gradient Difference (GradDiff), and Preference Optimization (PO/NPO), alongside continual unlearning methods like GUARD, O3, and UniErase.

**Limitations of Prior Work**: All methods that modify LLM weights suffer from catastrophic forgetting—the model's performance on the retain set drops sharply as unlearning requests accumulate. Furthermore, existing methods require training or optimization processes for each unlearning request, leaving sensitive information exposed during processing.

**Key Challenge**: Unlearning requires "changing model behavior," but modifying weights inevitably leads to "losing other knowledge"—these two objectives are fundamentally in conflict within the parameter space.

**Goal**: Achieve real-time continual unlearning without modifying LLM weights, supporting an arbitrary number of sequential unlearning requests without compromising model utility.

**Key Insight**: The unlearning objective is redefined—shifting from "parameter unlearning" (erasing knowledge) to "behavior unlearning" (preventing the output of flagged information). This opens a solution space that does not require weight modifications.

**Core Idea**: A task-agnostic sentence embedder is trained for semantic similarity judgment—if a query is similar to an unlearning request, the model refuses to answer; otherwise, it generates normally.

## Method

### Overall Architecture
CURaTE consists of two phases: (1) Pre-deployment training: Training data containing paraphrase positive pairs and contrastive negative pairs is generated from a seed QA dataset to fine-tune a sentence embedder $U$ using contrastive loss; (2) Post-deployment inference: When an unlearning request arrives, it is immediately embedded and stored in database $F$. During a user query, the maximum cosine similarity with all embeddings in $F$ is calculated. If it exceeds a threshold $\delta$, the response is refused.

### Key Designs

1. **Task-Agnostic Unlearning Embedder Training**:
    - **Function**: Learns a general capability for semantic similarity judgment, requiring no retraining post-deployment.
    - **Mechanism**: Three types of training data are generated from a seed QA dataset (e.g., Natural Questions): Type-1 (original question + paraphrase, positive), Type-2 (original question + contrastive question, hard negative—lexically similar but semantically different), and Type-3 (paraphrase + its contrastive question, hard negative). The embedder is trained with contrastive loss $\mathcal{L} = y \cdot d_U^2 + (1-y) \cdot \max(0, m-d_U)^2$.
    - **Design Motivation**: Hard negative pairs ensure the embedder distinguishes between "asking the same thing in different words" and "looking similar but asking different things." This is core to unlearning—blocking paraphrase variants without mis-intercepting unrelated queries.

2. **Real-Time Unlearning via Embedding Database**:
    - **Function**: Enables unlearning requests to take effect immediately without any optimization process.
    - **Mechanism**: When an unlearning request $f_m$ arrives, its embedding $f_m^{emb} = U(f_m)$ is computed and appended to set $F$, which is an $O(1)$ operation. For a user query, $s_{max} = \max_{i} \text{cos}(p^{emb}, f_i^{emb})$ is calculated. If $s_{max} \geq \delta$, a response is sampled from a predefined refusal set $R$.
    - **Design Motivation**: Parameter unlearning requires gradient computation taking minutes to hours, during which sensitive information remains accessible. Embedding storage achieves true "instant unlearning."

3. **Knowledge Preservation via Zero Weight Modification**:
    - **Function**: Maintains perfect knowledge retention after any number of unlearning requests.
    - **Mechanism**: Since LLM parameters are never modified, all knowledge unrelated to unlearning is fully preserved—catastrophic forgetting is impossible. The only risk is false refusals (misidentifying unrelated queries as unlearning requests), which is minimized through hard negative training.
    - **Design Motivation**: Catastrophic forgetting is the fundamental bottleneck of parameter unlearning; bypassing parameter modification entirely is the most thorough solution.

### Loss & Training
Contrastive loss: $\mathcal{L} = \frac{1}{2|T|}\sum [y \cdot d_U^2 + (1-y) \cdot \max(0, m-d_U)^2]$, using cosine distance as the metric. Training is completed once on the seed dataset and requires no additional training after deployment.

## Key Experimental Results

### Main Results

| Method | Forget Effectiveness (10 stages) | Knowledge Preservation (10 stages) | Real-time Capability |
|------|-----------------|-----------------|---------|
| GA | Effective but over-forgets | Severe drop (~0) | No |
| GradDiff | Over-forgets | Severe drop | No |
| NPO | Moderate | Moderate drop | No |
| O3 | Insufficient unlearning | Partial preservation | No |
| UniErase | Insufficient unlearning | Partial preservation | No |
| **Ours** | Effective unlearning | Near-perfect preservation | Yes |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| W/O Hard Negatives | High False Refusal Rate | Hard negative pairs are critical for decision boundary precision |
| Fixed Threshold $\delta$ | Stable Performance | Threshold shows some sensitivity to different tasks |
| Paraphrase Evaluation | **Ours** remains effective | Embedder is robust against paraphrasing |

### Key Findings
- **Ours** is the only method that maintains near-perfect knowledge preservation after 10 stages of continual unlearning.
- Parameter unlearning methods (GA, GradDiff) experience severe utility collapse after 3-5 stages.
- An embedder trained on a single seed dataset can transfer across domains to entirely different unlearning tasks.
- The system is robust to paraphrase attacks due to the design of positive pairs during training.

## Highlights & Insights
- The **redefinition of "Behavior Unlearning"** is a key contribution—shifting the goal from "erasing knowledge" to "blocking output" fundamentally changes the solution space.
- An extremely simple method (embedding similarity + thresholding) achieves the best results, revealing the over-complexity of parameter unlearning methods.
- The approach generalizes to any scenario requiring "selective refusal," such as copyright protection, privacy preservation, and information filtering.

## Limitations & Future Work
- Behavior unlearning is not true knowledge erasure; knowledge still resides in LLM weights and might be bypassed via indirect prompting.
- The selection of threshold $\delta$ is a performance bottleneck; too loose leads to incomplete unlearning, while too tight increases false refusals.
- The unlearning database $F$ grows with requests; large-scale scenarios will require approximate nearest neighbor search.
- Not applicable to legal requirements that mandate "true erasure" of knowledge (e.g., the right to be forgotten under GDPR).

## Related Work & Insights
- **vs GUARD**: GUARD also trains a classifier, but each unlearning set requires retraining; **Ours** is trained once and is cross-domain universal.
- **vs O3**: O3 trains orthogonal LoRA adapters and OOD detectors, still modifying parameters; **Ours** does not touch weights at all.
- **vs UniErase**: UniErase uses model editing to inject unlearning tokens; as a parameter-modifying approach, catastrophic forgetting remains inevitable.

## Rating
- Novelty: ⭐⭐⭐⭐ "Behavior unlearning" concept and minimalist design are novel.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Four benchmarks, 10-stage continual unlearning, and comparison with multiple baselines.
- Writing Quality: ⭐⭐⭐⭐ Clear motivation and straightforward methodology.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Representation-Guided Parameter-Efficient LLM Unlearning](representation-guided_parameter-efficient_llm_unlearning.md)
- [\[ICLR 2026\] LLM Unlearning with LLM Beliefs](../../ICLR2026/llm_safety/llm_unlearning_with_llm_beliefs.md)
- [\[ICLR 2026\] Inference-Time Backdoors via Hidden Instructions in LLM Chat Templates](../../ICLR2026/llm_safety/inference-time_backdoors_via_hidden_instructions_in_llm_chat_templates.md)
- [\[CVPR 2026\] Which Concepts to Forget and How to Refuse? Decomposing Concepts for Continual Unlearning in Large Vision-Language Models](../../CVPR2026/llm_safety/which_concepts_to_forget_and_how_to_refuse_decomposing_concepts_for_continual_un.md)
- [\[ACL 2026\] From Domains to Instances: Dual-Granularity Data Synthesis for LLM Unlearning](from_domains_to_instances_dual-granularity_data_synthesis_for_llm_unlearning.md)

</div>

<!-- RELATED:END -->
