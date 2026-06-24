---
title: >-
  [Paper Note] Towards Universal Offline Black-Box Optimization via Learning Language Model Embeddings
description: >-
  [ICML 2025][LLM (Other)][Universal Offline Optimization] This work proposes the UniSO framework, which encodes optimization variables of different types and dimensions into unified JSON strings before feeding them into language models. It trains universal regressors using two modeling paradigms: token prediction (UniSO-T) and numerical regression (UniSO-N). The embedding space quality is improved via metadata-guided contrastive learning and Lipschitz smoothness regularization…
tags:
  - "ICML 2025"
  - "LLM (Other)"
  - "Universal Offline Optimization"
  - "Language Model Embeddings"
  - "Cross-Domain Generalization"
  - "Metadata Alignment"
  - "Embedding Space Regularization"
date: 2026-05-08
content_hash: dfe69627c66fef3a
---

# Towards Universal Offline Black-Box Optimization via Learning Language Model Embeddings

**Conference**: ICML 2025  
**arXiv**: [2506.07109](https://arxiv.org/abs/2506.07109)  
**Code**: [https://github.com/lamda-bbo/universal-offline-bbo](https://github.com/lamda-bbo/universal-offline-bbo)  
**Area**: LLM / Black-Box Optimization  
**Keywords**: Universal Offline Optimization, Language Model Embeddings, Cross-Domain Generalization, Metadata Alignment, Embedding Space Regularization

## TL;DR

This work proposes the UniSO framework, which encodes optimization variables of different types and dimensions into unified JSON strings before feeding them into language models. It trains universal regressors using two modeling paradigms: token prediction (UniSO-T) and numerical regression (UniSO-N). The embedding space quality is improved via metadata-guided contrastive learning and Lipschitz smoothness regularization, achieving universal offline black-box optimization across domains and dimensions.

## Background & Motivation

**Background**: Offline Black-Box Optimization (Offline BBO) aims to find optimal designs for unknown target functions using only pre-collected static datasets, demonstrating value in scenarios such as protein design, molecular optimization, and engineering design. However, existing methods are limited to single-task, fixed-dimension settings, failing to exploit correlated knowledge across tasks.

**Limitations of Prior Work**: (1) Heterogeneous search spaces pose a core obstacle—different tasks have varying variable types (continuous, discrete, combinatorial permutation) and dimensions, preventing unified representation; (2) existing methods require collecting large amounts of independent data for each new task, which is unsustainable in data-scarce real-world scenarios; (3) the No Free Lunch (NFL) theorem states that a universal optimizer without task priors cannot perform well across all problems.

**Key Challenge**: How can one break the limitation of search space heterogeneity on cross-domain generalization while maintaining optimization performance? The recent success of LMs in string spaces provides a viable path, but direct application to offline BBO still faces issues such as indistinguishable and non-smooth embedding spaces.

**Goal**: Establish a universal framework capable of handling offline BBO tasks of multiple types and dimensions simultaneously, enabling cross-task knowledge transfer to address data scarcity.

**Key Insight**: Serialize all design variables into JSON strings (e.g., `{"x0":0.5, "x1":1.2}`), leverage the natural sequential processing ability of language models to uniformly represent heterogeneous spaces, and then employ metadata guidance and embedding regularization to ensure multi-task discriminability and optimization friendliness.

**Core Idea**: Unified string representation + language model embeddings + metadata-guided embedding space regularization = universal offline black-box optimization.

## Method

### Overall Architecture

The UniSO pipeline contains four components: (1) converting design-score data pairs into JSON string representations; (2) constructing textual metadata containing the task name, description, and optimization objective; (3) training two universal multi-task regressors (UniSO-T or UniSO-N); (4) searching for the final design internally within the trained model using Bayesian optimization. The workflow follows the forward modeling paradigm: first learn a scorer, then maximize the output of the scorer.

### Key Designs

1. **Two Modeling Paradigms (UniSO-T and UniSO-N)**:

    - **Function**: UniSO-T encodes target values as token sequences (P10 encoding) and trains an end-to-end sequence model using next-token prediction; UniSO-N uses a pre-trained T5 encoder to embed strings into a unified latent space, and then trains an MLP regressor to map them to numerical scores.
    - **Mechanism**: These align with two design philosophies: "treating optimization purely as language modeling" and "using LMs as feature extractors + numerical regressors," respectively. UniSO-T shows stronger cross-task generalization (Avg Rank 2.0) as end-to-end training better coordinates the encoder and decoder, whereas UniSO-N struggles with the mapping from LM embeddings to numerical space.
    - **Design Motivation**: OmniPred has demonstrated the effectiveness of token-targeted regression in multi-task online BBO, but its data is not public; Nguyen et al. demonstrated the efficacy of numeric-targeted regression for Bayesian optimization but did not explore offline BBO scenarios. The two paradigms are complementary, and a systematic comparison helps understand their respective trade-offs.

2. **Metadata-Guided Embedding Distribution Alignment**:

    - **Function**: Align the similarity distribution of input embeddings with that of metadata embeddings using contrastive learning—task embeddings with similar metadata should be closer, and dissimilar ones should be separated.
    - **Mechanism**: Encode metadata using a pre-trained T5-Small to obtain reference embeddings, calculate cosine similarity matrices for input and metadata embeddings respectively, and align the two distributions using a KL-divergence-formed contrastive loss. This resolves the overlapping and lack of clear boundaries among different tasks in the naive UniSO embedding space.
    - **Design Motivation**: Visualization shows that the embeddings of naive UniSO-T exhibit a chaotic, overlapping ring structure where tasks are indistinguishable. After alignment, clear clustering structures are formed, and similar tasks (such as Ant and D'Kitty) remain close to support knowledge sharing.

3. **Local Lipschitz Smoothness Regularization**:

    - **Function**: Constrain local smoothness of the embedding space within the same task, ensuring that the ratio of target value differences to embedding distances does not exceed a certain Lipschitz constant.
    - **Mechanism**: Calculate the Lipschitz ratios $|y_i - y_j| / \|\mathbf{z}_i - \mathbf{z}_j\|_2$ between embedding pairs for each task, take the median as a threshold $L$, and penalize ratios exceeding $L$. Contribution weighting across tasks is inversely proportional to dataset sizes to balance the tasks.
    - **Design Motivation**: Although contrastive learning addresses inter-task discriminability, embeddings within the same task may distribute arbitrarily in local regions as long as contrast is maintained with other tasks. Lipschitz constraints ensure that "designs with similar target values are also close in the embedding space," which is a prerequisite for effective downstream Bayesian optimization search.

### Loss & Training

- **Main Loss**: Cross-entropy for UniSO-T, MSE for UniSO-N.
- **Regularization**: Contrastive loss $\mathcal{L}_{con}$ + Lipschitz loss $\mathcal{L}_{lip}$.
- **Gradient Balancing**: Automatically scale the gradient contribution of auxiliary losses based on the main loss value (similar to MetaBalance) to prevent auxiliary losses from dominating the optimization.
- **Training Setup**: T5 architecture, AdamW optimizer, lr=1e-4, 200 epochs, batch size 128.

## Key Experimental Results

### Main Results (Multi-Task Training vs. Single-Task Expert)

| Method | Ant | D'Kitty | Superconductor | TF Bind 8 | TF Bind 10 | Avg Rank |
|------|-----|---------|----------------|-----------|------------|----------|
| BN+BO (Single-task expert) | 241.4 | 103.0 | 83.9 | 0.898 | 0.454 | 3.11 |
| BN+Grad (Single-task expert) | 229.5 | 183.3 | 97.1 | 0.959 | 0.888 | 4.11 |
| Improved UniSO-T (Multi-task) | **455.7** | **222.0** | 82.6 | 0.857 | **0.944** | **2.22** |
| Improved UniSO-N (Multi-task) | 381.8 | 42.2 | 82.0 | 0.856 | 0.528 | 4.11 |

### Zero-Shot / Few-Shot Generalization (Unseen Tasks)

| Task | Best in Dataset | UniSO-T Zero-Shot | UniSO-T Few-Shot |
|------|----------|-------------|-------------|
| RobotPush | 0.102 | >>0.102 | **7.067** |
| Rover | -16.148 | >>-16.148 | **-8.239** |
| LunarLander | 7.038 | >>7.038 | **248.6** |

### Ablation Study

| Configuration | Key Metrics | Description |
|------|---------|------|
| Remove metadata | Avg Rank 3.53→ | Decreased generalization capability |
| Remove Lipschitz loss | Avg Rank 2.89→ | Reduced local smoothness of embeddings |
| Remove contrastive loss | Avg Rank 2.33→ | Degraded task discriminability |
| Remove gradient balancing | Avg Rank 1.89→ | Auxiliary losses might dominate the main loss |
| Pre-trained T5 initialization (UniSO-N) | Worse than training from scratch | LM priors are harmful for numerical optimization |

### Key Findings

- UniSO-T trained under a multi-task setting outperforms single-task numerical experts (Avg Rank 2.22 vs 3.11), demonstrating the viability of cross-domain knowledge transfer.
- Pre-trained LMs are harmful for numerical optimization—attention concentrates on syntactic tokens (such as EOS) instead of numerical tokens; UniSO-N trained from scratch yields better performance.
- Metadata quality is critical for cross-domain generalization—removing any component (name, description, or objective) leads to a performance drop.
- BO outperforms EA as an in-model searcher, showing clear advantages in discrete string spaces.
- DeepSeek-R1 assigns more attention to numerical tokens compared to the base Qwen; LMs with stronger mathematical abilities might be better suited for numerical optimization.

## Highlights & Insights

- "Unified heterogeneous search space via strings" is a simple yet powerful insight—the JSON format naturally handles variable-length and mixed-type inputs, avoiding traditional challenges like alignment or padding.
- Metadata-guided contrastive learning elegantly utilizes the text semantics of task descriptions—metadata embeddings of similar tasks are naturally closer, acting as anchors to guide the organization of input embeddings.
- "Pre-trained LMs are harmful for numerical optimization" is a counter-intuitive yet important discovery—the linguistic prior of LMs becomes an obstacle in numerical optimization scenes, with attention allocation bias being the root cause.
- The zero-shot generalization capability is impressive—zero-shot results on unseen tasks already exceed the best dataset records, showing that the model indeed captures cross-domain optimization patterns.

## Limitations & Future Work

- There remains a gap between UniSO-T and SOTA single-task methods (Avg Rank 9.8/22 on Design-Bench), representing a trade-off between universality and specialized extreme performance.
- The performance of UniSO-N is significantly weaker than UniSO-T—mapping LM embeddings to the numerical space remains a bottleneck, requiring better architectural designs.
- The precision of numerical tokens is limited by tokenizer granularity—the upper precision bound from P10 encoding might impact high-precision optimization scenarios.
- Scalability on extremely high-dimensional problems (e.g., >1000 dimensions) has not been verified—JSON strings will become very long, potentially challenging sequence modeling.
- Trained on only 9 tasks—whether more diverse training tasks can further improve generalization remains to be validated.

## Related Work & Insights

- **OmniPred**: Direct inspiration source for UniSO-T, but OmniPred's dataset is not public; this work serves as an independent implementation.
- **Nguyen et al. (2024)**: Inspiration source for UniSO-N, which uses LM embeddings for Bayesian optimization but is limited to online settings.
- **Relationship with BO**: BO is highly efficient on fixed dimensions but cannot generalize across domains; with LM embeddings unifying the space, Bayesian search becomes available cross-domain.
- **Latent Space Optimization (e.g., COLA/LIRE)**: Emphasizes the importance of embedding space smoothness for optimization, from which the Lipschitz regularization concept is derived.

## Rating

- **Novelty**: ⭐⭐⭐⭐ Unified strings + universal LM-based offline BBO is a new paradigm.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ 9 tasks for training, 3 tasks for generalization, detailed ablation, and comparisons with multiple baselines.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definitions are clear, with systematic comparisons between the two paradigms.
- **Value**: ⭐⭐⭐⭐ Directional contribution to the BBO field, although gaps to SOTA remain.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] PRESTO: Preimage-Informed Instruction Optimization for Prompting Black-Box LLMs](../../NeurIPS2025/llm_nlp/presto_preimage-informed_instruction_optimization_for_prompting_black-box_llms.md)
- [\[ACL 2025\] Length Controlled Generation for Black-box LLMs](../../ACL2025/llm_nlp/length_controlled_generation_for_black-box_llms.md)
- [\[ACL 2025\] Self-Instructed Derived Prompt Generation Meets In-Context Learning: Unlocking New Potential of Black-Box LLMs](../../ACL2025/llm_nlp/self-instructed_derived_prompt_generation_meets_in-context_learning_unlocking_ne.md)
- [\[ICML 2025\] Interchangeable Token Embeddings for Extendable Vocabulary and Alpha-Equivalence](interchangeable_token_embeddings_for_extendable_vocabulary_and_alpha-equivalence.md)
- [\[ACL 2025\] MergePrint: Merge-Resistant Fingerprints for Robust Black-box Ownership Verification of Large Language Models](../../ACL2025/llm_nlp/mergeprint_fingerprint_ownership.md)

</div>

<!-- RELATED:END -->
