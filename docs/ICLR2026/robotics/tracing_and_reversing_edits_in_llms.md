---
title: >-
  [Paper Note] Tracing and Reversing Edits in LLMs
description: >-
  [ICLR 2026][Robotics][knowledge editing] Addressing the dual-use risks of Knowledge Editing (KE), this paper proposes EditScope, a method that infers edited target entities from post-edit weights with up to 99% accuracy…
tags:
  - "ICLR 2026"
  - "Robotics"
  - "knowledge editing"
  - "model security"
  - "SVD"
  - "edit tracing"
  - "edit reversal"
date: 2026-05-08
content_hash: 36fabcbe81b101ea
---

# Tracing and Reversing Edits in LLMs

**Conference**: ICLR 2026
**arXiv**: [2505.20819](https://arxiv.org/abs/2505.20819)  
**Code**: [https://github.com/paulyoussef/trace-and-reverse/](https://github.com/paulyoussef/trace-and-reverse/)  
**Area**: Robotics
**Keywords**: knowledge editing, model security, SVD, edit tracing, edit reversal

## TL;DR
Addressing the dual-use risks of Knowledge Editing (KE), this paper proposes EditScope, a method that infers edited target entities from post-edit weights with up to 99% accuracy, alongside a training-free edit reversal approach based on SVD bottom-rank approximation achieving up to 94% reversal rate—requiring only the post-edit weights, without access to the editing prompt or original weights.

## Background & Motivation

**Background**: KE methods such as ROME and MEMIT enable low-cost updates of factual knowledge in LLMs, formalized as $(s, r, o \to o')$, e.g., changing "the German Chancellor is Scholz" to "Merz."

**Limitations of Prior Work**: KE carries dual-use risks—it can be exploited maliciously to inject misinformation, bias, or backdoors. Existing defenses assume access to a candidate set of potentially edited facts for verification, which is impractical in real-world settings.

**Key Challenge**: How can malicious edits be detected and reversed using only the post-edit weights, with no knowledge of the editing prompt, original weights, or the edited facts?

**Goal**: Two tasks are formally defined: (1) *edit tracing*—inferring the edited target entity $o'$ from post-edit weights; (2) *edit reversal*—restoring the model's original output without any additional information.

**Key Insight**: The structural properties of KE methods are exploited—methods such as ROME produce rank-1 updates $W'_V = W_V + W_N$, concentrating edit information in the component corresponding to the largest singular value.

**Core Idea**: In post-edit weight matrices, edit information is concentrated in the top singular value components; this property enables high-accuracy tracing and training-free reversal of malicious knowledge edits.

## Method

### Overall Architecture
Given an edited model, two core tasks are addressed: (1) EditScope decodes the edited target entity $o'$ from the edit matrix $W'_V$ by training the non-edited layers of the model; (2) the reversal method removes edit information via SVD bottom-rank approximation to restore original outputs. Both methods rely solely on post-edit weights.

### Key Designs

1. **EditScope (Edit Tracing)**:

    - **Function**: Infer the edited target entity $o'$ from the post-edit weight matrix $W'_V$.
    - **Mechanism**: A fixed random input $x_{fixed} = (t_1, ..., t_m)$ (with $m=5$ newly added tokens) is used. The edit matrix $W'_{V_i}$ replaces the original matrix in the model as the "input," and the remaining layer parameters are trained so that the model outputs the corresponding edit target $o'_i$. The training loss is cross-entropy: $\mathcal{L} = -\sum_{j=1}^{|\mathcal{V}|} \mathbb{1}_{i=j} \cdot \log(Q_j)$
    - **Design Motivation**: KE causes the edited object to be over-represented in the weights (overfitting to edited objects); training the model to adapt to this representation effectively "decodes" the edited target. Using a fixed random input eliminates the need for knowledge of the editing prompt.

2. **Bottom-Rank Approximation (Edit Reversal)**:

    - **Function**: Remove edit information to restore the model's original output.
    - **Mechanism**: SVD is applied to the edit matrix: $W'_V = U\Sigma V^T$. The top $k$ largest singular values and their corresponding components are removed, yielding the bottom-rank approximation $\tilde{W'}_V^{(r,k)} = \sum_{i=k+1}^{r} \Sigma_{ii} u_i v_i^T$.
    - **Design Motivation**: The update matrix $W_N$ produced by ROME and similar methods is rank-1, so edit information is primarily concentrated in the largest singular value component of the edit matrix (empirically verified: on GPT2-XL, the cosine similarity between $W_N$ and its rank-1 approximation at $k=1$ reaches 0.98). Removing these components is equivalent to removing the edit information.

3. **Post-Edit Detection**:

    - **Function**: Distinguish edited from unedited weight matrices.
    - **Mechanism**: The bottom-rank approximation method is reused—if removing the top $k$ components causes a large change in output, those components contain anomalous (edited) information.
    - **Design Motivation**: In deployment, identifying which layers have been edited is a prerequisite for targeted reversal.

### Loss & Training
- **EditScope**: Trained on 600 edit matrices (from 20 relations), with 100 for validation, 300 for in-distribution (ID) testing, and 330 for out-of-distribution (OOD) testing (from 11 unseen relations). Up to 100 epochs with patience=3 early stopping.
- **Reversal method**: Completely training-free; applies SVD decomposition and approximation directly.

## Key Experimental Results

### Main Results — EditScope Tracing Accuracy

| Model | ROME ID | ROME OOD | r-ROME ID | r-ROME OOD |
|------|---------|----------|-----------|------------|
| GPT2-XL | 99.40% | 99.70% | 99.73% | 99.70% |
| GPT-J-6B | 97.60% | 94.42% | 96.50% | 95.91% |
| LLAMA3-8B | 96.47% | 91.21% | 94.87% | 88.18% |
| QWEN2.5-7B | 91.20% | 87.45% | 88.53% | 85.45% |

### Ablation Study — Reversal Method (ROME, Optimal $k$)

| Model | Optimal $k$ | Reversal Rate | Edit Accuracy↓ | KL Divergence ($k=0 \to k_{opt}$) |
|------|--------|-----------|------------|----------------------|
| GPT2-XL | 11 | 94.52% | 1.29% | 6.038 → 0.010 |
| GPT-J-6B | 14 | 80.00% | 2.26% | 11.567 → 0.218 |
| LLAMA3-8B | 15 | 80.00% | 6.45% | 10.068 → 0.604 |
| QWEN2.5-7B | 13 | 62.90% | 26.13% | 8.988 → 1.615 |

### Key Findings
- **GPT-series models are easiest to trace and reverse**: GPT2-XL achieves 99% tracing and 94% reversal accuracy, as edit information is highly concentrated in the $k=1$ component.
- **Larger and newer models are harder to reverse**: QWEN2.5-7B achieves only 62.9% reversal, as edit information is spread across more singular value components.
- **Strong OOD generalization**: EditScope maintains >85% accuracy on unseen relations.
- **Reversal does not degrade model capability**: Performance on benchmarks such as CoLA and MMLU is nearly identical before and after reversal.
- **KL divergence decreases substantially**: The output distribution after reversal closely approximates the original (GPT2-XL: 6.038 → 0.010).

## Highlights & Insights
- **Minimal-assumption defense design**: No knowledge of the editing prompt, original weights, or edited content is required; tracing and reversal are performed solely from post-edit weights—reflecting a genuinely practical defense scenario.
- **Elegant use of SVD**: The natural mapping of rank-1 edits to the largest singular value component is a concise and theoretically elegant insight. Bottom-rank approximation as a denoising technique is transferable to other scenarios requiring the removal of specific information from model weights.
- **Fixed random input in EditScope**: Using fixed random tokens as input ensures the training process focuses exclusively on edit information encoded in the weight matrix, removing any dependency on the editing prompt—a particularly clever design choice.

## Limitations & Future Work
- **Focus on rank-1 edits only**: ROME/r-ROME produce rank-1 updates; batch editing methods such as MEMIT do not yield strictly rank-1 update matrices, and reversal performance may degrade accordingly.
- **Lower reversal rate for QWEN (62.9%)**: Newer model architectures may require more refined SVD strategies.
- **Primarily single-edit setting**: Although batch and sequential edits are discussed in the appendix, the main experiments address only single edits.
- **Requires knowledge of the edited layer**: While the appendix provides a layer detection method, identifying the edit location in practice remains a prerequisite challenge.
- **Computational cost**: SVD decomposition of large matrices may become a bottleneck for very large models.

## Related Work & Insights
- **vs. Youssef et al. (2025c)**: That work detects edits by analyzing hidden states and output probabilities, requiring a candidate set of edited facts; the present work requires no prior knowledge.
- **vs. Li et al. (2025)**: That work identifies the type of edit (misinformation/bias); this paper traces the specific content of edits.
- **vs. AlphaEdit (Fang et al. 2025)**: AlphaEdit is an editing method; the appendix verifies that the proposed approach is also effective against it.
- This work has broader implications for model watermarking and intellectual property protection—if edits can be traced, the provenance of specific knowledge within a model may also be recoverable.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First formal treatment of edit tracing and reversal tasks, with concise and effective methods.
- Experimental Thoroughness: ⭐⭐⭐⭐ Four models × two KE methods × two datasets, with extensive generalization experiments in the appendix.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem formulation and rigorous methodological derivation.
- Value: ⭐⭐⭐⭐⭐ Significant practical implications for LLM security and defense.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] When Agents Persuade: Propaganda Generation and Mitigation in LLMs](when_agents_persuade_propaganda_generation_and_mitigation_in_llms.md)
- [\[ICLR 2026\] Enhancing Instruction Following of LLMs via Activation Steering with Dynamic Rejection](enhancing_instruction_following_of_llms_via_activation_steering_with_dynamic_rej.md)
- [\[ICLR 2026\] String Seed of Thought: Prompting LLMs for Distribution-Faithful and Diverse Generation](string_seed_of_thought_prompting_llms_for_distribution-faithful_and_diverse_gene.md)
- [\[ICLR 2026\] One Demo Is All It Takes: Planning Domain Derivation with LLMs from A Single Demonstration](one_demo_is_all_it_takes_planning_domain_derivation_with_llms_from_a_single_demo.md)
- [\[ICLR 2026\] What's the Plan? Metrics for Implicit Planning in LLMs and Their Application to Rhyme Generation and Question Answering](whats_the_plan_metrics_for_implicit_planning_in_llms_and_their_application_to_rh.md)

</div>

<!-- RELATED:END -->
