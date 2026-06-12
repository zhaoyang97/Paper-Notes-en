---
title: >-
  [Paper Note] Enhancing Vision-Language Model Reliability with Uncertainty-Guided Dropout Decoding
description: >-
  [NeurIPS 2025][Multimodal VLM][VLM hallucination] This paper proposes Dropout Decoding — a training-free inference-time method that projects visual tokens into the text space to quantify their epistemic uncertainty…
tags:
  - "NeurIPS 2025"
  - "Multimodal VLM"
  - "VLM hallucination"
  - "uncertainty quantification"
  - "visual token dropout"
  - "epistemic uncertainty"
  - "ensemble decoding"
date: 2026-05-08
content_hash: c6e7f42f80215aff
---

# Enhancing Vision-Language Model Reliability with Uncertainty-Guided Dropout Decoding

**Conference**: NeurIPS 2025
**arXiv**: [2412.06474](https://arxiv.org/abs/2412.06474)  
**Code**: [https://github.com/kigb/DropoutDecoding](https://github.com/kigb/DropoutDecoding)  
**Area**: Multimodal VLM
**Keywords**: VLM hallucination, uncertainty quantification, visual token dropout, epistemic uncertainty, ensemble decoding

## TL;DR
This paper proposes Dropout Decoding — a training-free inference-time method that projects visual tokens into the text space to quantify their epistemic uncertainty, selectively masks high-uncertainty visual tokens, and aggregates multiple masked decoding results via majority voting to substantially reduce object hallucinations in LVLMs.

## Background & Motivation

**Severity of LVLM Hallucinations**: Large vision-language models (LVLMs) have demonstrated strong capabilities in image captioning, visual question answering, and related tasks, yet they frequently produce hallucinations — generating descriptions inconsistent with image content. Object hallucination in particular poses a serious credibility barrier for real-world deployment. Such hallucinations often stem from the model misinterpreting certain visual tokens, incorrectly mapping specific visual patch information to non-existent objects or erroneous attributes.

**Limitations of Prior Work**: Training-time approaches (e.g., task-specific fine-tuning, RLHF) require substantial computational resources and generalize poorly to new tasks. Inference-time approaches (e.g., OPERA modifying beam search, VCD using contrastive decoding) are largely heuristic and lack a principled measure of which visual tokens are unreliable. More fundamentally, these methods do not directly answer a core question: among hundreds to thousands of visual tokens, which carry reliable information and which are uncertain or prone to misinterpretation?

**Transferring the Dropout Idea**: Traditional dropout applies random masking to model parameters during training to prevent overfitting, but applying parameter-level dropout to pretrained LVLMs is infeasible. The authors propose transferring the core idea of dropout from parameter space to input token space — selectively masking visual input tokens at inference time to reduce over-reliance on noisy visual tokens by introducing stochasticity into the decoding context.

**Key Insight**: The approach exploits a latent capability of the LVLM text decoder — the hidden representations of visual tokens at the top decoder layer inherently encode textual semantic information. By projecting visual tokens into the text vocabulary space, a "textualized" interpretation of each visual token can be obtained, enabling uncertainty quantification. Epistemic uncertainty (reflecting insufficient model knowledge) is particularly well-suited for identifying critical visual tokens that are informative yet prone to misinterpretation.

## Method

### Overall Architecture
Dropout Decoding consists of two stages: (1) **Pre-decoding** — quantifying and decomposing the uncertainty of all visual tokens; (2) **During decoding** — epistemic-uncertainty-guided token dropout, multi-mask ensemble, and majority voting. The entire process operates at inference time without modifying any model parameters or requiring additional training.

### Key Designs

1. **Textual Interpretation of Visual Tokens**:

    - **Function**: Maps each visual token into the text vocabulary space to obtain a "textualized" probability distribution, revealing the model's semantic interpretation of the corresponding visual patch.
    - **Mechanism**: Applies the logit lens method to the LVLM decoder. For the $i$-th visual token $x_i^v$, its top-layer hidden representation is $h_i^v = f_\theta(x_{\leq i}^v)$, and the textualized distribution is obtained via the vocabulary projection matrix: $q_i^{\text{proj}} = \text{softmax}(W_\mathcal{V} h_i^v)$. Informative patches project to specific words (e.g., "Berlin", "computer"), while uninformative background patches project to high-frequency function words (e.g., "a", "the").
    - **Design Motivation**: The top-layer hidden representations of the LVLM decoder naturally approximate text vocabulary projections — even at visual token positions where the model was not explicitly trained to generate text — and can effectively capture semantic information. This provides an unsupervised, model-intrinsic means of assessing the informativeness of visual tokens.

2. **Uncertainty Decomposition and Epistemic Uncertainty Estimation**:

    - **Function**: Decomposes the total uncertainty of each visual token into aleatoric uncertainty (inherent in the data) and epistemic uncertainty (due to insufficient model knowledge), finding that epistemic uncertainty is the best indicator for identifying critical but easily misinterpreted visual tokens.
    - **Mechanism**: Defines the mean textualized distribution over all visual tokens $q^{\text{proj}} = \frac{1}{N}\sum_{i}^{N} q_i^{\text{proj}}$ as a baseline. Aleatoric uncertainty is $U_{\text{ale}}(i) = \mathbb{H}[q_i^{\text{proj}}]$, the entropy of each individual token's distribution; epistemic uncertainty is $U_{\text{epi}}(i) = D_{\text{KL}}(q_i^{\text{proj}} \| q^{\text{proj}})$, the KL divergence of each token's distribution from the global mean. Total uncertainty decomposes as $U_{\text{total}} = \mathbb{E}_i[U_{\text{ale}}(i) + U_{\text{epi}}(i)]$.
    - **Design Motivation**: Intuitively, high epistemic uncertainty indicates that the textualized interpretation of a visual token differs substantially from the average interpretation of the image as a whole — it carries unique, "surprising" information. These are precisely the regions most critical yet prone to misinterpretation. Experiments confirm that epistemic uncertainty correlates positively with visual token informativeness, whereas aleatoric and total uncertainty lack such correlation.

3. **Uncertainty-Guided Token Dropout and Ensemble Voting**:

    - **Function**: Generates multiple dropout masks based on epistemic uncertainty, applies selective masking to visual tokens, and aggregates the decoding results across all masked versions via majority voting to produce the final output.
    - **Mechanism**: Constructs a dropout probability distribution from normalized epistemic uncertainty: $P_{\text{dropout}}^{(k)}(x_i^v) = \gamma^{(k)} \frac{U_{\text{epi}}(i) - U_{\text{epi}}^{\min}}{U_{\text{epi}}^{\max} - U_{\text{epi}}^{\min}} + \delta^{(k)}$, where $\gamma^{(k)}$ and $\delta^{(k)}$ control dropout strength. $K$ binary masks $M^{(k)}$ are independently sampled; each masked visual context is decoded to produce a candidate token $y_j^{(k)}$, and the final output token is selected by majority voting. Optionally, a preliminary forward pass is performed before each decoding step to produce an initial prediction $y_j^{\text{init}}$, retaining visual tokens relevant to this initial prediction from being dropped.
    - **Design Motivation**: A single decoding pass may generate hallucinations due to over-reliance on misinterpreted visual tokens. Ensembling across multiple masking configurations diversifies the model's perspective on the visual content, reducing the impact of individual misinterpretations — analogous to the variance-reduction effect of model ensembles.

### Loss & Training
No training is required, which is one of the greatest practical advantages of Dropout Decoding. All operations are performed entirely at inference time without modifying any model parameters. Key hyperparameters include: the number of dropout masks $K$ (recommended 5–10 for accuracy-efficiency trade-off); the dropout probability range parameters $\gamma^{(k)}$ and $\delta^{(k)}$ (modulating ensemble members with varying dropout intensities to increase mask diversity); and the optional top-$k$ threshold for the relevant token retention step. In practice, the $K$ forward passes can be batched and executed in parallel, partially mitigating latency overhead.

## Key Experimental Results

### Main Results

| Model | Method | CHAIR_S↓ | CHAIR_I↓ | THRONE $F^1_{\text{all}}$↑ | THRONE $P_{\text{all}}$↑ |
|------|------|---------|---------|-----------|-----------|
| LLaVA-1.5 | Greedy | 42.20 | 12.83 | 0.795 | 0.772 |
| LLaVA-1.5 | Beam Search | 46.33 | 13.90 | 0.790 | 0.759 |
| LLaVA-1.5 | OPERA | 41.47 | 12.37 | 0.802 | 0.782 |
| LLaVA-1.5 | VCD | 49.20 | 14.87 | 0.786 | 0.759 |
| LLaVA-1.5 | **Dropout Decoding** | **39.80** | **11.73** | **0.804** | **0.784** |
| InstructBLIP | Greedy | 27.87 | 7.90 | 0.809 | - |
| InstructBLIP | **Dropout Decoding** | **24.53** | **6.63** | **0.814** | - |
| LLaVA-NEXT | Greedy | 28.80 | 8.10 | 0.815 | - |
| LLaVA-NEXT | **Dropout Decoding** | **26.26** | **7.39** | **0.821** | - |

### Ablation Study

| Configuration | CHAIR_S↓ | CHAIR_I↓ | Note |
|------|---------|---------|------|
| Aleatoric uncertainty guidance | 43.10 | 13.20 | Poor; fails to identify critical tokens effectively |
| Total uncertainty guidance | 41.80 | 12.50 | Slight improvement but unstable |
| **Epistemic uncertainty guidance** | **39.80** | **11.73** | Best; precisely locates critical yet easily misinterpreted tokens |
| K=1 (single dropout) | 41.30 | 12.40 | Insufficient ensemble |
| K=5 | 40.10 | 11.90 | Near optimal |
| **K=10** | **39.80** | **11.73** | Optimal ensemble size |
| With relevant token retention | 39.80 | 11.73 | Best CHAIR |
| Without relevant token retention | 40.20 | 11.85 | Slightly worse CHAIR, possibly better THRONE |

### Key Findings
- VCD substantially degrades performance on InstructBLIP (CHAIR_S: 27.87 → 39.33), whereas Dropout Decoding consistently improves all models.
- Epistemic uncertainty substantially outperforms aleatoric and total uncertainty as a dropout guidance signal.
- InstructBLIP uses only 32 visual tokens (high information density) while LLaVA variants use hundreds to thousands; the method is effective across different token count scales.
- During majority voting ties, the forward pass retaining the most tokens (most complete information) is selected — a detail that contributes to output stability.

## Highlights & Insights
- **Elegant Transfer of the Dropout Concept**: Transferring dropout from training-time parameter regularization to inference-time input token space is conceptually natural, yet had not been previously explored in the LVLM context. The key innovation lies in uncertainty-guided rather than random masking, grounding the approach in information-theoretic principles.
- **Intuitive Interpretation of Epistemic Uncertainty**: High-epistemic-uncertainty visual tokens correspond to informative but potentially misinterpreted critical patches — a finding that offers a new perspective on how LVLMs perceive visual content, and explains why random dropout underperforms targeted dropout.
- **Training-Free Plug-and-Play Design**: The entire method relies solely on the LVLM's own forward-pass capabilities (logit lens + text projection), introduces no external models, and is compatible with arbitrary LVLM architectures.

## Limitations & Future Work
- Multiple forward passes ($K$ dropout passes plus an optional preliminary prediction) increase inference latency by approximately 5–10×, making the method unsuitable for real-time interactive scenarios.
- The approach relies on the quality of logit lens projections — if the model's visual-text alignment is poor, the textualized distributions may be inaccurate, undermining the uncertainty estimates.
- Applicability to open-ended generation tasks (e.g., creative writing, complex reasoning) has not been validated; current evaluation is primarily on descriptive tasks (image captioning, VQA).
- Majority voting may be inappropriate in settings requiring high output diversity — the ensemble tends toward consensus answers, potentially suppressing creative responses.
- When the number of visual tokens is small (e.g., 32 in InstructBLIP), dropout risks removing critical information; when the token count is very large (e.g., 2880+ in LLaVA-NEXT), the computational cost of uncertainty estimation grows accordingly.

## Related Work & Insights
- **vs. OPERA**: OPERA reduces hallucinations by penalizing over-attention in beam search, a heuristic operating at the decoding strategy level; Dropout Decoding quantifies token-level uncertainty from an information-theoretic perspective, offering a more principled foundation.
- **vs. VCD (Visual Contrastive Decoding)**: VCD contrasts output distributions with and without visual input to reduce hallucinations, but degrades performance on certain models (e.g., InstructBLIP); Dropout Decoding operates directly on visual token subsets and achieves consistent gains.
- **vs. HALC**: HALC relies on an external visual grounding model to localize relevant regions; Dropout Decoding performs uncertainty estimation using only the LVLM's own capabilities.
- **vs. GAN-DIME / MI Estimation**: From an information-theoretic standpoint, the epistemic uncertainty measure in Dropout Decoding essentially quantifies the unique mutual information carried by each individual visual token.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Transferring dropout from parameter space to input token space is a clever conceptual innovation; uncertainty guidance provides a sound theoretical basis.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 3 models and dual benchmarks (CHAIR + THRONE) with detailed ablations, but lacks efficiency analysis and evaluation on broader task types.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The projection visualization in Figure 1 and the uncertainty decomposition are highly intuitive; mathematical derivations are well balanced with intuitive explanations.
- **Value**: ⭐⭐⭐⭐ A practical inference-time method for enhancing VLM reliability, though inference overhead remains a bottleneck for real-world deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] ViSpec: Accelerating Vision-Language Models with Vision-Aware Speculative Decoding](vispec_accelerating_vision-language_models_with_vision-aware_speculative_decodin.md)
- [\[NeurIPS 2025\] SSR: Enhancing Depth Perception in VLMs via Rationale-Guided Spatial Reasoning](ssr_enhancing_depth_perception_in_vision-language_models_via_rationale-guided_sp.md)
- [\[ICCV 2025\] Controlling Multimodal LLMs via Reward-guided Decoding](../../ICCV2025/multimodal_vlm/controlling_multimodal_llms_via_rewardguided_decoding.md)
- [\[NeurIPS 2025\] AdaLRS: Loss-Guided Adaptive Learning Rate Search for Efficient Foundation Model Pretraining](adalrs_lossguided_adaptive_learning_rate_search_for_efficien.md)
- [\[ICCV 2025\] Enhancing Few-Shot Vision-Language Classification with Large Multimodal Model Features](../../ICCV2025/multimodal_vlm/enhancing_few-shot_vision-language_classification_with_large_multimodal_model_fe.md)

</div>

<!-- RELATED:END -->
