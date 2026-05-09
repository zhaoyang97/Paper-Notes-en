---
title: >-
  [Paper Note] Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning
description: >-
  [NeurIPS 2025][Recommender Systems] This work identifies two categories of sparsity artifacts introduced by L1 loss in Crosscoders—Complete Shrinkage (which erroneously zeros out weakly shared concepts) and Latent Decoupling (which splits shared concepts into spurious model-specific latents)—and proposes Latent Scaling as a diagnostic tool and BatchTopK Crosscoder as an alternative training scheme, substantially improving the reliability of chat-tuning concept discovery.
tags:
  - NeurIPS 2025
  - Recommender Systems
date: 2026-05-08
content_hash: 6edbfd9009b41336
---

# Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning

**Conference**: NeurIPS 2025
**arXiv**: [2504.02922](https://arxiv.org/abs/2504.02922)
**Authors**: Julian Minder, Clément Dumas, Caden Juang, Bilal Chughtai, Neel Nanda
**Affiliations**: EPFL, ETHZ, ENS Paris-Saclay, Northeastern University
**Code**: [GitHub](https://github.com/jkminder/dictionary_learning)
**Area**: Interpretability / Fine-tuning Analysis

## TL;DR

This work identifies two categories of sparsity artifacts introduced by L1 loss in Crosscoders—Complete Shrinkage (which erroneously zeros out weakly shared concepts) and Latent Decoupling (which splits shared concepts into spurious model-specific latents)—and proposes Latent Scaling as a diagnostic tool and BatchTopK Crosscoder as an alternative training scheme, substantially improving the reliability of chat-tuning concept discovery.

## Background & Motivation

**Background**: Model diffing is an emerging direction in interpretability research, aiming to understand how fine-tuning modifies a model's internal representations and algorithms. Crosscoder (Lindsey et al., 2024) is a model diffing tool that learns a shared dictionary of interpretable concepts between a base model and a fine-tuned model, representing each concept as a pair of latent directions (one for the base model, one for the chat model), thereby tracking how concepts change or emerge during fine-tuning.

**Limitations of Prior Work**: Prior work identifies chat-only concepts (i.e., those present only in the fine-tuned model) by checking whether the base model decoder vector norm is zero. While seemingly reasonable, this criterion can be severely confounded by systematic biases introduced by the L1 sparsity training loss.

**Key Challenge**: L1 regularization, while encouraging sparse representations, produces two categories of artifacts: (1) **Complete Shrinkage**—when a concept contributes weakly in the base model but strongly in the chat model, L1 directly shrinks the base decoder norm to zero, spuriously labeling the concept as chat-only; (2) **Latent Decoupling**—a concept that should be shared is equivalently decomposed by the L1 loss into one chat-only and one base-only latent pair, since both representations incur identical L1 cost.

**Goal**: To diagnose and eliminate spurious chat-only latent attributions caused by L1 loss in Crosscoders, and to identify genuinely interpretable concepts introduced by chat-tuning.

**Key Insight**: The paper begins with a theoretical analysis of the inherent deficiencies of L1 loss, then designs a diagnostic tool (Latent Scaling) and an alternative training scheme (BatchTopK).

**Core Idea**: Replace L1 loss with BatchTopK loss for Crosscoder training, eliminating shrinkage bias at the source and making decoder norm differences a reliable measure of chat-specificity.

## Method

### Overall Architecture

The proposed methodology proceeds in three progressive steps: (1) **Diagnosis**—theoretical analysis of the Complete Shrinkage and Latent Decoupling artifact mechanisms in L1 Crosscoders; (2) **Quantification**—Latent Scaling, which learns a scaling factor β for each chat-only latent to precisely measure its true presence across models; (3) **Replacement**—training a BatchTopK Crosscoder that enforces sparsity by directly constraining the number of active latents per batch, fundamentally avoiding L1 shrinkage bias. All models are trained on the layer-13 residual stream of Gemma 2 2B base/chat pairs with expansion factor 32 and dictionary size 73,728.

### Key Designs

1. **Latent Scaling Diagnostic Tool**
    - Function: Computes two pairs of scaling factors β per chat-only latent, precisely measuring each latent's capacity to explain reconstruction error and activation in the base and chat models.
    - Mechanism: For each chat-only latent $j$, the chat decoder direction $\mathbf{d}_j^{\text{chat}}$ is used to fit the base model's error and reconstruction via least squares, yielding scaling factors β. The base/chat ratios $\nu_\varepsilon = \beta_\varepsilon^{\text{base}} / \beta_\varepsilon^{\text{chat}}$ and $\nu_r = \beta_r^{\text{base}} / \beta_r^{\text{chat}}$ are then computed. High $\nu_\varepsilon$ indicates Complete Shrinkage (the latent explains base error); high $\nu_r$ indicates Latent Decoupling (the latent's information already exists in the base reconstruction).
    - Design Motivation: The standard Δnorm metric is unreliable in L1 Crosscoders due to systematic bias, necessitating an independent diagnostic to distinguish genuinely chat-specific latents from training artifacts.

2. **BatchTopK Crosscoder Training Scheme**
    - Function: Replaces L1 loss with BatchTopK loss, directly controlling the number of active latents per batch ($k=100$) rather than indirectly encouraging sparsity via L1.
    - Mechanism: BatchTopK selects the top-$k$ most strongly activated latents per sample without applying explicit regularization penalties on decoder norms, thereby avoiding Complete Shrinkage. The top-$k$ selection also creates competition among latents, making Latent Decoupling—where two latents represent the same concept—inefficient (requiring L0=2 rather than L0=1 of the sparsity budget).
    - Design Motivation: BatchTopK is borrowed from the SAE literature (Bussmann et al., 2024) and introduced to the Crosscoder setting for the first time. Its L0-based sparsity optimization naturally favors a three-latent solution (shared + chat-only + base-only) over a two-latent solution (chat-only + base-only) for shared concepts.

3. **Causal Intervention Validation Framework**
    - Function: Validates the causal effectiveness of identified chat-specific latents by substituting their representations into base model activations.
    - Mechanism: A mixed activation is constructed as $\mathbf{h}_S(x) = \mathbf{h}_{\text{base}}(x) + \sum_{j \in S} f_j(x)(\mathbf{d}_j^{\text{chat}} - \mathbf{d}_j^{\text{base}})$, fed into the chat model's subsequent layers, and the KL divergence from the original chat model output is computed. Lower KL indicates that the substituted latent set $S$ better recovers chat behavior.
    - Design Motivation: Interpretability alone is insufficient; causal evidence is needed to demonstrate that the identified chat-specific latents actually drive behavioral differences in the chat model.

### Loss & Training

**L1 Crosscoder Loss**:

$$\mathcal{L}_{L1}(x) = \frac{1}{2}\|\varepsilon_{\text{base}}\|^2 + \frac{1}{2}\|\varepsilon_{\text{chat}}\|^2 + \mu \sum_j f_j(x)(\|\mathbf{d}_j^{\text{base}}\|_2 + \|\mathbf{d}_j^{\text{chat}}\|_2)$$

**BatchTopK Crosscoder Loss**: Contains only reconstruction error plus an auxiliary dead-latent recovery loss, with no explicit sparsity regularization:

$$\mathcal{L}_{\text{BatchTopK}}(\mathcal{X}) = \frac{1}{n}\sum_i \left[\frac{1}{2}\|\varepsilon_{\text{base}}\|^2 + \frac{1}{2}\|\varepsilon_{\text{chat}}\|^2\right] + \alpha \cdot \mathcal{L}_{\text{aux}}$$

Training details: base/chat models are Gemma 2 2B / Gemma 2 2B-it, layer-13 residual stream, 100M tokens (Fineweb + LMSYS-CHAT), L0 ≈ 100, approximately 60 GPU·h (H100) total.

## Key Experimental Results

### Main Results

| Metric | L1 Crosscoder | BatchTopK Crosscoder |
|--------|--------------|---------------------|
| Chat-only latents (Δnorm > 0.9) | 3,176 | 134 |
| Base-only latents (Δnorm < 0.1) | 1,437 | 5 |
| Shared latents (Δnorm 0.4–0.6) | 53,569 | 62,373 |
| Truly chat-specific proportion (ν_ε < 0.2 and ν_r < 0.5) | Very low (mostly artifact-affected) | Vast majority |
| Full replacement KL reduction (all tokens) | ~59% | ~59% |
| Full replacement KL reduction (first 9 tokens) | ~78% | ~78% |
| Top-50% vs Bottom-50% Δnorm KL (all tokens) | 0.241 vs 0.242 (failed) | 0.230 vs 0.267 (successful distinction) |
| Top-50% vs Bottom-50% Δnorm KL (first 9 tokens) | 0.619 vs 0.740 (reversed!) | 50% reduction vs 6% reduction |
| Validation FVE | 84.6% | 87.6% |

### Ablation Study

| Ablation | Result |
|----------|--------|
| Latent Scaling diagnosis on L1 → top-50% ν vs bottom-50% ν | Successfully identifies causally important latents; effect on first 9 tokens approaches BatchTopK |
| Replacing activations at template token positions only | KL 0.239 / 0.507, ≈ top-50% chat-specific latents |
| Validation on independently trained L1 Crosscoder (Kissane et al.) | 17.7% of chat-only latents fall within the 95% interval of the shared distribution; results are consistent |
| Latent Decoupling cosine similarity | 109 (chat-only, base-only) latent pairs with cosim > 0.9; 60% activate in different contexts |
| High cosine similarity coupling pairs in BatchTopK | 0 pairs (no cosim > 0.9 pairings in the Δnorm < 0.6 region) |

### Key Findings

- **Δnorm is a training artifact in L1 Crosscoders**: 18% of L1 chat-only latents fall within the 95% central interval of the shared distribution (ν_r); ν_ε values reach ~0.5, indicating a large proportion of false positives.
- **BatchTopK exhibits virtually no artifacts**: ν_r does not overlap with the shared distribution at all; ν_ε values are uniformly near zero; Pearson correlation between Δnorm and ν metrics reaches 0.73 / 0.87.
- **Chat behavior is concentrated at template tokens**: 40% of BatchTopK chat-only latents activate primarily on template tokens; 67% have at least one-third of activations on template tokens.
- **Interpretable chat concepts discovered by BatchTopK**: harmful instruction request detection, sensitive content detection, racial/gender discrimination content detection, post-refusal behavior, personal question recognition, misinformation detection, missing information detection, rephrasing requests, joke detection, reply length measurement, summarization requests, knowledge boundary recognition, and others.
- **Behavioral differences are largest in the first 9 tokens**: The base–chat KL divergence is 1.69 for the first 9 tokens versus only 0.482 across the full response, with differences concentrated at the start of the reply.

## Highlights & Insights

- **From tool deficiency to methodological contribution**: The paper not only identifies a fundamental flaw in L1 Crosscoders but simultaneously provides both a diagnostic tool (Latent Scaling) and a root-cause solution (BatchTopK), forming a complete methodological loop.
- **Strong alignment between theory and empirics**: The two artifact mechanisms arising from L1 loss have clear mathematical explanations (the gradient difference between L1's $\sqrt{x^2+y^2}$ and SAE's treatment), and experimental results perfectly validate the theoretical predictions.
- **Causal validation enhances credibility**: Rather than stopping at finding interpretable features, the paper uses activation patching experiments to demonstrate that the identified latents causally drive chat behavior.
- **The critical role of template tokens**: The paper reveals that much of the chat model's distinctive behavior is encoded through template tokens, consistent with concurrent work (Leong et al., 2025).
- **Both Crosscoders capture equivalent information but organize it differently**: L1 and BatchTopK achieve nearly identical KL reduction under full replacement, yet BatchTopK cleanly organizes chat-specific information within high-Δnorm latents, while L1 distributes it across all latents.

## Limitations & Future Work

- **Single model scale**: Experiments are conducted only on Gemma 2 2B; whether larger models (e.g., 7B, 70B) exhibit the same issues remains to be verified.
- **Focus limited to chat-only latents**: Base-only and shared latents—particularly those with low cosine similarity between base and chat decoders—are not deeply analyzed; these may encode more subtle fine-tuning effects.
- **Residual information in reconstruction error**: The error term of BatchTopK still accounts for approximately 45% of the KL divergence on the first 9 tokens, indicating that the dictionary does not fully capture chat behavioral differences.
- **Cannot distinguish new concepts from activation shifts**: The Crosscoder architecture cannot differentiate between concepts genuinely learned during chat-tuning and existing concepts that have merely shifted their activation patterns.
- **Lack of systematic comparison with other sparse methods**: No comparison is made against recent sparsification approaches such as JumpReLU SAE or Gated SAE.

## Related Work & Insights

- **Sparse Autoencoders (SAE)**: The foundational architecture for Crosscoders; BatchTopK SAE (Bussmann et al., 2024) is directly adapted to the Crosscoder setting in this work.
- **Original Crosscoder work**: Lindsey et al. (2024) proposed Crosscoders and first identified chat-only latents; the present paper reveals that some of those findings may be artifacts.
- **Representation stability under fine-tuning**: Multiple studies suggest fine-tuning primarily modulates rather than creates new capabilities (Jain et al., 2024; Wu et al., 2024; Merchant et al., 2020); the chat-specific latents identified here provide a new, precise tool for this line of inquiry.
- **The safety role of template tokens**: Leong et al. (2025) concurrently find that safety mechanisms rely heavily on aggregated information at template tokens, consistent with the findings of this paper.

## Rating

| Dimension | Score | Rationale |
|-----------|-------|-----------|
| Novelty | ⭐⭐⭐⭐ | First systematic identification of training artifacts in Crosscoders, with two novel tools: Latent Scaling and BatchTopK Crosscoder |
| Technical Depth | ⭐⭐⭐⭐⭐ | Combines gradient-level analysis of L1 loss, closed-form derivation of Latent Scaling, and causal intervention experiments in a tightly integrated framework |
| Experimental Thoroughness | ⭐⭐⭐⭐ | Multi-dimensional experiments covering artifact diagnosis, causal validation, interpretability assessment, and independent replication, though limited to a single model scale |
| Practical Value | ⭐⭐⭐⭐⭐ | Directly improves best practices for the Crosscoder methodology; open-source code; provides immediate value to the interpretability and AI safety communities |

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Transformer Copilot: Learning from The Mistake Log in LLM Fine-tuning](transformer_copilot_learning_from_the_mistake_log_in_llm_fine-tuning.md)
- [\[NeurIPS 2025\] PAC-Bayes Bounds for Multivariate Linear Regression and Linear Autoencoders](pac-bayes_bounds_for_multivariate_linear_regression_and_linear_autoencoders.md)
- [\[NeurIPS 2025\] R²ec: Towards Large Recommender Models with Reasoning](r2ec_towards_large_recommender_models_with_reasoning.md)
- [\[NeurIPS 2025\] Who You Are Matters: Bridging Topics and Social Roles via LLM-Enhanced Logical Recommendation](who_you_are_matters_bridging_topics_and_social_roles_via_llm-enhanced_logical_re.md)
- [\[AAAI 2026\] CroPS: Improving Dense Retrieval with Cross-Perspective Positive Samples in Short-Video Search](../../AAAI2026/recommender/crops_improving_dense_retrieval_with_cross-perspective_positive_samples_in_short.md)

</div>

<!-- RELATED:END -->
