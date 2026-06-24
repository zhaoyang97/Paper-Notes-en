---
title: >-
  [Paper Note] Better Embeddings with Coupled Adam
description: >-
  [ACL 2025][LLM (Other)][Word Embedding Anisotropy] Theoretically proves that the token-wise second moment of the Adam optimizer is the root cause of word embedding anisotropy (mean shift) in LLMs, and proposes Coupled Adam—which averages the second moment of the embedding layer across the vocabulary—eliminating the anisotropy issue and improving embedding quality and downstream performance in large-scale experiments.
tags:
  - "ACL 2025"
  - "LLM (Other)"
  - "Word Embedding Anisotropy"
  - "Adam Optimizer"
  - "Coupled Adam"
  - "Representation Degeneration"
  - "LLM Training"
date: 2026-05-08
content_hash: 5fd1194c0bab83a0
---

# Better Embeddings with Coupled Adam

**Conference**: ACL 2025  
**arXiv**: [2502.08441](https://arxiv.org/abs/2502.08441)  
**Code**: None  
**Area**: Others  
**Keywords**: Word Embedding Anisotropy, Adam Optimizer, Coupled Adam, Representation Degeneration, LLM Training

## TL;DR
Theoretically proves that the token-wise second moment of the Adam optimizer is the root cause of word embedding anisotropy (mean shift) in LLMs, and proposes Coupled Adam—which averages the second moment of the embedding layer across the vocabulary—eliminating the anisotropy issue and improving embedding quality and downstream performance in large-scale experiments.

## Background & Motivation
**Background**: Word embeddings trained in LLMs exhibit anisotropy, where all embeddings are clustered in a small subspace far from the origin, limiting semantic expressiveness. This is a widely observed phenomenon whose root cause remains unknown.

**Limitations of Prior Work**: Biś et al. (2021) discovered that anisotropy is primarily caused by embedding mean shift, but attributed it to the "common enemy effect" (gradients of non-target tokens all point in the $-h$ direction). This explanation is imprecise as it overlooks the scaling of gradients by prediction probabilities.

**Key Challenge**: If the sum of embedding gradients is always zero (as proven in this paper), why does the mean still shift? Is it an issue with the gradients themselves or the optimization algorithm?

**Goal**: Find the true root cause of embedding anisotropy and provide a simple and effective solution.

**Key Insight**: Mathematically analyze the summation properties of embedding update vectors under SGD and Adam—the sum of updates under SGD is always zero (keeping the mean constant), whereas under Adam, the sum is non-zero due to varying token-wise second moments.

**Core Idea**: The second moment of Adam introduces different effective learning rates for each token, violating the property of zero-sum gradients and leading to mean shift. Averaging the second moments across the vocabulary dimension (Coupled Adam) solves this problem.

## Method

### Overall Architecture
- Theoretical analysis: Prove that embedding mean remains constant under SGD vs. shifts under Adam
- Method proposal: Coupled Adam—modifies only the embedding layer by replacing the second moment $\hat{v}_i$ with the vocabulary-level average $\hat{\nu} = \frac{1}{V}\sum_i \hat{v}_i$
- Experimental validation: Small-scale (125M-760M) + Large-scale (1.3B-2.6B)

### Key Designs

1. **Theoretical Derivation—Zero-Sum Gradients**:

    - Function: Prove that for standard language model heads, the sum of all embedding gradients is always zero $\sum_i g_i = 0$
    - Mechanism: Directly derived from the sum of softmax probabilities $\sum_i p_i = 1$. This implies that if the optimizer's update vectors are proportional to the gradients, the mean remains unchanged
    - Design Motivation: Establish the theoretical foundation—the issue resides not in the gradients, but in how the optimizer processes them

2. **Key Differences Between SGD and Adam**:

    - SGD update: $u_i = -\eta \cdot g_i$, sum of updates $= -\eta \sum g_i = 0$ → constant mean
    - Adam update: $u_i = -\eta_i \cdot \hat{m}_i$, where $\eta_i = \eta / (\sqrt{\hat{v}_i} + \epsilon)$ is the token-dependent effective learning rate. Since $\eta_i$ differs across tokens, the weighted sum $\sum \eta_i \hat{m}_i \neq 0$ → mean shift
    - Key Findings: $\mathbb{E}[\hat{v}_i] \propto \tilde{p}_i$ (the expectation of the second moment is proportional to the unigram probability), validated experimentally with $R^2=0.85$

3. **Coupled Adam (Core Innovation)**:

    - Function: Replaces Adam's second moment with the vocabulary-level average, applied exclusively to embedding parameters
    - Mechanism: $\hat{\nu}^{(\tau)} = \frac{1}{V}\sum_{i=1}^V \hat{v}_i^{(\tau)}$, then $\hat{v}_i^{(\tau)} \leftarrow \hat{\nu}^{(\tau)}$ for all $i$
    - Effect: The effective learning rate becomes token-independent, the sum of updates restores to zero, and the mean remains constant
    - Implementation: Requires only a 2-line code modification, with no changes to non-embedding parameters

## Key Experimental Results

### Small-Scale Experiments (OpenWebText, GPT-2 Architecture, 3 seeds)

| Data/Model | Method | Test Loss | Iso↑ | $\|\mu\|^r$↓ | Word Sim. $\bar{r}$↑ |
|----------|------|-----------|------|-------------|-----------------|
| 20B/125M | Standard | 3.03 | 0.10 | 0.82 | 5 |
| 20B/125M | **Coupled** | **2.97** | **0.83** | **0.03** | **57** |
| 20B/355M | Standard | 2.79 | 0.25 | 0.82 | 5 |
| 20B/355M | **Coupled** | **2.75** | **0.95** | **0.02** | **57** |
| 20B/760M | Standard | 2.68 | 0.28 | 0.73 | 3 |
| 20B/760M | **Coupled** | **2.65** | **0.94** | **0.01** | **58** |

### Large-Scale Experiments (SlimPajama, LLaMA-like, 1.3B-2.6B)

| Model/Data | Method | Loss↓ | Acc↑ | Iso↑ |
|----------|------|-------|------|------|
| 1.3B/26B | Standard | 2.46 | 0.397 | 0.22 |
| 1.3B/26B | **Coupled** | **2.44** | **0.399** | **0.94** |
| 2.6B/210B | Standard | 2.17 | 0.425 | 0.22 |
| 2.6B/210B | **Coupled** | **2.16** | **0.428** | **0.97** |

### Key Findings
- Coupled Adam improves isotropy from 0.1–0.28 to 0.83–0.97, and reduces the mean shift ratio from 0.63–0.82 to 0.01–0.03—nearly a perfect fix.
- Word semantic similarity evaluation (e.g., SimLex999) improves from single digits to 55–58, marking a qualitative leap in embedding quality.
- On sufficiently large datasets (20B+ tokens), Coupled Adam simultaneously improves both perplexity and downstream accuracy.
- Performance is neutral or slightly worse on smaller datasets, indicating that the benefits of isotropy require sufficient training data to materialize.
- The 2.6B model on 210B tokens continues to benefit, demonstrating the effectiveness of the method in large-scale scenarios.

## Highlights & Insights
- **Elegant Theoretical Analysis**: Starting from softmax probability normalization, the paper derives that the sum of gradients is zero, and then proves that Adam disrupts this property—making the entire chain of reasoning clear and complete.
- **Extremely Simple Solution**: Simply averaging the second moments of the embeddings requires only 2 lines of code and introduces zero extra computational overhead—highly practical.
- **Reveals Structural Limitations of Adam**: Adam's adaptive learning rate for sparse parameters (embeddings) is detrimental in this scenario—a finding that provides valuable insights for optimizer design.
- **Second Moment $\approx$ Unigram Probability**: High-frequency words have larger second moments $\rightarrow$ smaller effective learning rates $\rightarrow$ smaller updates; the opposite holds for low-frequency words. Coupled Adam unifies this asymmetry.

## Limitations & Future Work
- Only analyzes gradients of the language modeling head, neglecting the gradient contributions flowing from internal layers (non-output layers) to the embeddings.
- Conclusions are drawn under the weight-tying assumption; scenarios without weight tying are not discussed.
- Performance is neutral or slightly worse under small data volumes, requiring sufficient training to yield benefits.
- Has not been tested on non-autoregressive models (e.g., BERT).
- Future work could consider extending Coupled Adam to other sparse parameters (e.g., MoE routing parameters).

## Related Work & Insights
- **vs. Biś et al. (2021)**: They proposed the "common enemy effect" as the root cause of anisotropy. This paper corrects this view, identifying Adam's token-wise second moments as the actual cause.
- **vs. Post-Processing Methods (Mean Subtraction, etc.)**: Post-processing fixes the issue during inference, whereas Coupled Adam resolves it during training, addressing the problem at its root.
- **vs. AdamW**: Weight decay can partially alleviate anisotropy, but Coupled Adam addresses it directly from within the optimizer, offering a more thorough solution.
- Inspiration for optimizer design: For parameters with specific structures (such as shared embedding matrices), general-purpose optimizers might not be optimal.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ Theoretical analysis precisely identifies the root cause of anisotropy within the optimizer, offering an elegant and simple fix.
- Experimental Thoroughness: ⭐⭐⭐⭐ Fully covers scale from small to large (2.6B/210B) with various embedding quality metrics.
- Writing Quality: ⭐⭐⭐⭐⭐ Rigorous theoretical derivation, clear presentation of experiments, and a complete logical argument.
- Value: ⭐⭐⭐⭐⭐ Has a direct impact on LLM training practice—improving embedding quality with just 2 lines of code is a outstanding contribution.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Token Prepending: A Training-Free Approach for Eliciting Better Sentence Embeddings from LLMs](token_prepending_training_free.md)
- [\[ACL 2025\] Hierarchical Attention Generates Better Proofs](hierarchical_attention_generates_better_proofs.md)
- [\[ACL 2025\] Training Language Model to Critique for Better Refinement](training_language_model_to_critique_for_better_refinement.md)
- [\[ACL 2025\] Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering](contrastive_prompting_embeddings.md)
- [\[ACL 2025\] Cheaper and Better Diffusion Language Model via Task-Specific Training](cheaper_and_better_diffusion_language_model_via_task-specific_training.md)

</div>

<!-- RELATED:END -->
