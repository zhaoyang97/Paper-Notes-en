---
title: >-
  [Paper Note] A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks
description: >-
  [CVPR 2025][LLM Safety][VLM debiasing] Proposed a training-free, data-free debiasing method for VLMs. By deriving closed-form solutions in a cross-modal space, it achieves Pareto-optimal trade-offs between fairness and utility retention, consistently outperforming existing approaches across three downstream tasks: zero-shot classification, text-to-image retrieval, and text-to-image generation.
tags:
  - "CVPR 2025"
  - "LLM Safety"
  - "VLM debiasing"
  - "fairness"
  - "closed-form solution"
  - "Pareto-optimal"
  - "cross-modal space"
date: 2026-05-08
content_hash: 7008f8dbdb92ca76
---

# A Closed-Form Solution for Debiasing Vision-Language Models with Utility Guarantees Across Modalities and Tasks

**Conference**: CVPR 2025  
**arXiv**: [2603.12998](https://arxiv.org/abs/2603.12998)  
**Code**: [https://github.com/Supltz/Debias_VLM](https://github.com/Supltz/Debias_VLM)  
**Area**: Multimodal VLM  
**Keywords**: VLM debiasing, fairness, closed-form solution, Pareto-optimal, cross-modal space

## TL;DR
Proposed a training-free, data-free debiasing method for VLMs. By deriving closed-form solutions in a cross-modal space, it achieves Pareto-optimal trade-offs between fairness and utility retention, consistently outperforming existing approaches across three downstream tasks: zero-shot classification, text-to-image retrieval, and text-to-image generation.

## Background & Motivation
**Background**: VLMs like CLIP achieve remarkable performance in many downstream tasks but inherit societal biases (e.g., gender, race) from large-scale web-crawled data, causing abnormally high similarity between "nurse" and "female".

**Limitations of Prior Work**: Existing debiasing methods have specific drawbacks: some require training additional networks (DeAR, FairerCLIP); some need data annotated with sensitive attributes (SFID, CLIP-clip); some only address a single modality (BiasedPrompt, SANER); some target only a single task (PRISM works on classification only); and nearly none provide theoretical guarantees for utility preservation.

**Key Challenge**: Fairness and utility preservation are inherently contradictory—removing bias information inevitably degrades semantic information. Prior methods either sacrifice utility or require extensive hyperparameter tuning to balance them, while failing to provide a theoretical upper bound on performance loss.

**Goal**: How to simultaneously debias both visual and textual modalities under zero-training and zero-data conditions, while providing a provable upper bound on utility loss? How to handle multiple downstream tasks in a unified manner?

**Key Insight**: Formulate the debiasing problem as an optimization problem on a cross-modal unit hypersphere, decompose embeddings orthogonally into "attribute leakage" and "neutral content" components, and find the optimal point on the Pareto frontier.

**Core Idea**: Model VLM debiasing as a Chebyshev scalarisation problem on a hypersphere to derive the closed-form optimal solution $\alpha^\star$, achieving a Pareto-optimal fairness-utility trade-off.

## Method

### Overall Architecture
Input: original VLM image/text embeddings $\vec{e}_I, \vec{e}_T$; Output: debiased embeddings $\vec{u}_I, \vec{u}_T$. Two phases: (1) Use an LLM to build group prototypes to define the attribute subspace $\mathcal{A}$; (2) Search for the optimal debiased embeddings in the cross-modal space.

### Key Designs

1. **LLM-Guided Group Prototype Construction**:

    - **Function**: Build representative embeddings for each sensitive attribute group (e.g., male/female).
    - **Mechanism**: Use an LLM (GPT-5) to inject attribute words into input prompts and generate multiple phrasing variants (e.g., "male doctor" $\rightarrow$ "man doctor", "masculine doctor"), then compute the spherical average $\vec{p}_g$ as the group prototype. The attribute subspace $\mathcal{A}$ is spanned by the prototype difference vectors $\vec{a}_i = \vec{p}_{g_i} - \vec{p}_{g_1}$.
    - **Design Motivation**: Prior methods directly define attribute directions using a single prompt, ignoring diverse linguistic expressions of the same attribute ("man", "gentleman", and "boy" all refer to male), which leads to an inaccurate attribute subspace.

2. **Closed-Form Debiasing Solution**:

    - **Function**: Locate the debiased embedding $\vec{u}^\star$ on the hypersphere $\mathbb{S}^{d-1}$.
    - **Mechanism**: Orthogonally decompose the embedding as $\vec{e} = \vec{e}_{\mathcal{A}_\parallel} + \vec{e}_{\mathcal{A}_\perp}$ (attribute leakage + neutral content). Define dual objectives: minimize attribute leakage $L(\alpha) = \alpha$ and minimize utility loss $V(\alpha) = 1 - \alpha\|\vec{e}_{\mathcal{A}_\parallel}\| - \sqrt{1-\alpha^2}\|\vec{e}_{\mathcal{A}_\perp}\|$. Using Lemma 1-2, reduce the high-dimensional search to 1D, then solve the minimax problem via Chebyshev scalarisation to obtain the closed-form solution: $\alpha^\star = \frac{E - \|\vec{e}_{\mathcal{A}_\perp}\|\sqrt{E^2 - \|\vec{e}_{\mathcal{A}_\parallel}\|^2}}{E^2 + \|\vec{e}_{\mathcal{A}_\perp}\|^2}$.
    - **Design Motivation**: Prior methods (e.g., Orth-Proj) perform direct orthogonal projection which is equivalent to $\alpha=0$ (perfect fairness but worst utility); this work identifies the optimal point on the Pareto frontier to achieve the best trade-off.

3. **Utility Upper Bound Guarantees**:

    - **Function**: Provide a provable upper bound on cross-utility loss.
    - **Mechanism**: Proposition 1 proves that $\ell_{cross} \leq \sqrt{2\ell_{self}^{(I)}} + \sqrt{2\ell_{self}^{(T)}}$, so bounding the self-utility loss is sufficient to guarantee cross-modal alignment. Theorem 1 further provides the precise upper-bound expression.
    - **Design Motivation**: Prior methods claim to preserve utility without theoretical proof, relying solely on empirical tuning.

### Loss & Training
This method is training-free and requires no training process. The core optimization is directly computed via the closed-form solution without gradient descent or iteration. During inference, the attribute subspace projection matrix is computed only once (can be pre-computed and cached), and subsequently, each embedding requires only an $O(d)$ vector operation.
The computational bottleneck of the overall pipeline lies in the one-time cost of LLM variant generation. Once constructed, the attribute subspace can be reused across all samples.

## Key Experimental Results

### Main Results

| Dataset/Task | Metric | Ours | Prev. SOTA | Description |
|------------|------|------|----------|------|
| CelebA Classification | F1↑ | **56.5** | 53.1 (FairerCLIP) | Significant utility boost +3.4 |
| CelebA Classification | $\Delta_{EO}^{Max}$ (G×A) ↓ | **40.1** | 40.0 (RoboShot) | Comparable fairness |
| Flickr30K Retrieval | R@5↑ | **90.4** | 87.9 (FairerCLIP) | Leading retrieval utility |
| Flickr30K Retrieval | MS@1000↓ | **11.8** | 11.7 (CLIP-clip) | Comparable fairness |
| T2I Generation | $\overline{SP}_5$↓ | **28.8** | 28.4 (Orth-Proj) | Comparable fairness |
| T2I Generation | Acc^G↑ | **74.6** | 67.2 (SFID) | Significant utility lead +7.4 |

### Ablation Study

| Configuration | MS@1000 (Flickr30K) | $\Delta_{EO}^{Max}$ (CelebA) | Description |
|------|-------|---------|------|
| Full model | 11.8 | 40.1 | Full model |
| Anchor embedding only | 13.4 | 41.1 | No LLM variants, performance drops |
| Mean embedding only | 14.1 | 41.8 | No anchor, worse performance |
| Image-only debiasing | 13.4 | 41.7 | Single modality is insufficient |
| Text-only debiasing | 13.3 | 41.1 | Single modality is insufficient |
| Swap with DeepSeek v3.2 | 12.0 | 40.1 | Insensitive to LLM choice |
| Swap with Gemini 2.5 Pro | 11.8 | 40.4 | Insensitive to LLM choice |

### Key Findings
- Debiasing both modalities (I&T) simultaneously consistently outperforms single-modality debiasing, validating the hypothesis that bias is encoded within the cross-modal alignment.
- The LLM prototype construction is stable across different LLM choices (GPT-5/DeepSeek/Gemini yield similar results), as the task is relatively simple (generating synonymous variants) and does not require complex reasoning.
- Methods requiring labeled data do not necessarily perform better — they are sensitive to the data domain (e.g., a debiasing network trained on face data generalizes poorly to full-body images).
- The inference-time computational overhead is extremely low (requiring only one matrix projection and closed-form formula evaluation), showing a massive GPU time advantage over training-based methods.

## Highlights & Insights
- **Closed-form solution instead of iterative optimization**: Through delicate mathematical derivations (Lemma 1 for dimensionality reduction + Lemma 2 for search space bounding + Chebyshev scalarisation for multi-objective optimization), the seemingly complex hyperspherical optimization is boiled down to a single-line formula. The elegance lies in proving that the optimal solution is guaranteed to lie within the 2D plane spanned by the two components of the original embedding.
- **Both theoretical guarantee and practical utility**: Unlike most fairness methods that only offer empirical results or provide theories that lack practicality, this work delivers both a rigorous upper-bound proof and multi-task experimental validation.
- **LLM-assisted attribute subspace construction** can be transferred to other tasks requiring the definition of semantic subspaces: e.g., using LLMs to generate synonymous variants to construct more robust concept directions.
- **Modular design of the methodology** is exemplary: the attribute subspace construction and the closed-form solving phases are completely decoupled, allowing independent replacement or improvement of each component.

## Limitations & Future Work
- The utility guarantee is at the embedding space (cosine similarity) level, which does not directly guarantee downstream task-specific metrics (F1, Recall); end-to-end performance verification still requires task-level evaluations in practice.
- The attribute subspace assumes that bias can be modeled as a linear subspace, which might be inadequate for non-linear biases. Real-world biases are often intertwined with multiple factors (e.g., intersectional bias of race $\times$ gender), which linear subspaces find hard to capture fully.
- Validation was conducted only on CLIP-based models, leaving newer VLMs (such as LLaVA, InternVL) untested. Whether the closed-form solution remains applicable to their different embedding structures warrants further research.
- Future work could extend this approach to generative model debiasing on the decoder side.
- The LLM prototype construction relies on manually defined sensitive attribute categories; defining group coordinates for emerging or fine-grained bias dimensions (such as age or disability) requires additional domain expertise.

## Related Work & Insights
- **vs PRISM/Orth-Proj**: They process orth-projection (the extreme case where $\alpha=0$), while this work locates the optimal point on the Pareto frontier, rendering it strictly superior in theory. Orth-Proj can be viewed as a boundary case of this method.
- **vs FairerCLIP/DeAR**: They require training + labeled data, whereas this work is training-free and yields better performance, indicating that a data-driven path is not necessarily the optimal route. FairerCLIP also generalizes poorly under distribution shifts.
- **vs SANER**: Both employ the concept of attribute directions, but SANER uses static word lists while this work dynamically generates more accurate descriptors via LLMs.
- **vs BiasedPrompt**: It only handles bias in the textual modality, ignoring bias propagation within the visual encoder. This work handles both modalities, making it more comprehensive.
- **Inspiration**: The paradigm of using Chebyshev scalarisation for multi-objective optimization can be transferred to other scenarios needing fairness-utility tradeoffs (such as recommendation systems or NLP classifier debiasing).
- **Potential connection to optimal transport**: The debiasing process can be viewed as conveying a biased embedding distribution towards a fair distribution. Future works could explore extensions from an optimal transport perspective.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First to apply a closed-form solution to VLM debiasing, with rigorous and elegant mathematical derivations.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Extensive validations on three tasks, multiple datasets, and comprehensive ablation and sensitivity analyses.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem formulation and logical, coherent theorem proofs.
- Value: ⭐⭐⭐⭐ Significant contribution to the fairness field, though deployment scenarios still require further validation.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2025\] Hyperbolic Safety-Aware Vision-Language Models](hyperbolic_safety-aware_vision-language_models.md)
- [\[CVPR 2025\] CleanSight: Test-Time Attention Purification for Backdoored Large Vision Language Models](test-time_attention_purification_for_backdoored_large_vision_language_models.md)
- [\[CVPR 2025\] TAPT: Test-Time Adversarial Prompt Tuning for Robust Inference in Vision-Language Models](tapt_test-time_adversarial_prompt_tuning_for_robust_inference_in_vision-language.md)
- [\[CVPR 2025\] ForensicZip: More Tokens are Better but Not Necessary in Forensic Vision-Language Models](forensiczip_more_tokens_are_better_but_not_necessary_in_forensic_vision-language.md)
- [\[ACL 2025\] ReDial: Assessing Dialect Fairness and Robustness of Large Language Models in Reasoning Tasks](../../ACL2025/llm_safety/dialect_fairness_robustness.md)

</div>

<!-- RELATED:END -->
