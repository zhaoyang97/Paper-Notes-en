---
title: >-
  [Paper Note] Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation
description: >-
  [ACL 2025][Text Generation][Decoding Strategies] Proposed Dynamic Focus Decoding (DFD), which identifies knowledge-intensive decoding steps by tracking inter-layer distribution discrepancies (KL divergence) in LLMs and adaptively adjusts temperature—lowering temperature on knowledge-intensive steps to preserve factuality, and raising temperature on non-knowledge-intensive steps to promote diversity—simultaneously improving factuality and diversity across seven datasets.
tags:
  - "ACL 2025"
  - "Text Generation"
  - "Decoding Strategies"
  - "Factuality"
  - "Diversity"
  - "Dynamic Temperature"
  - "Knowledge-Awareness"
date: 2026-05-08
content_hash: 18c5423f276f8750
---

# Odysseus Navigates the Sirens' Song: Dynamic Focus Decoding for Factual and Diverse Open-Ended Text Generation

**Conference**: ACL 2025  
**arXiv**: [2503.08057](https://arxiv.org/abs/2503.08057)  
**Code**: [https://github.com/lllllw-222/Siren-DFD](https://github.com/lllllw-222/Siren-DFD)  
**Area**: Text Generation  
**Keywords**: Decoding Strategies, Factuality, Diversity, Dynamic Temperature, Knowledge-Awareness  

## TL;DR
Proposed Dynamic Focus Decoding (DFD), which identifies knowledge-intensive decoding steps by tracking inter-layer distribution discrepancies (KL divergence) in LLMs and adaptively adjusts temperature—lowering temperature on knowledge-intensive steps to preserve factuality, and raising temperature on non-knowledge-intensive steps to promote diversity—simultaneously improving factuality and diversity across seven datasets.

## Background & Motivation
**Background**: Open-ended text generation requires both factual accuracy and text diversity. Existing stochastic decoding methods (e.g., nucleus sampling, top-k) perform sampling using a fixed temperature.

**Limitations of Prior Work**: A fixed high temperature increases diversity but harms factuality (e.g., replacing "Isaac Newton" with "Galileo"); a fixed low temperature preserves factuality but yields repetitive and uninteresting outputs (e.g., outputting "Newton" every time). This "decoding focus distortion" is a fundamental limitation of fixed temperature.

**Key Challenge**: Different steps during the generation process have varying dependencies on factual knowledge—producing entity names requires precision, whereas generating function words/connectives can be highly flexible—but fixed temperature fails to distinguish between them.

**Goal**: How to automatically identify "when factual precision is needed, and when diversity is allowed" during decoding, and dynamically adjust the sampling strategy.

**Key Insight**: The hierarchical knowledge encoding property of Transformers—lower layers capture syntactic features, whereas higher layers store factual knowledge. When a token heavily relies on deep knowledge, the distribution discrepancy across layers becomes more significant.

**Core Idea**: Using the KL divergence of the output distributions between layers as a knowledge-aware signal to dynamically adjust the decoding temperature at each step.

## Method

### Overall Architecture
A plug-and-play module is inserted into the standard autoregressive decoding pipeline: at each decoding step, the output distributions of the intermediate layers are first obtained via the LM head. The KL divergence between these intermediate distributions and the final output distribution is calculated and aggregated into a knowledge-awareness intensity signal. Then, this signal is converted into a dynamic temperature for the step through a transformation function, and finally, sampling is performed using this temperature.

### Key Designs

1. **Knowledge-Awareness Positioning**:

    - Function: Assessing the degree of dependence on factual knowledge at the current decoding step.
    - Mechanism: Apply the LM head to the hidden state $h_t^{(i)}$ of each intermediate layer to obtain the layer-wise distributions $p^{(i)}(\cdot|x_{\leq t})$, and compute the KL divergence with respect to the final output distribution $\text{KL}_t^{(i)} = \text{KL}(p^{(N)} \| p^{(i)})$. The knowledge-awareness intensity is defined as the average KL divergence across all layers: $\text{KA}_t = \frac{1}{N-1}\sum_{i=1}^{N-1} \text{KL}_t^{(i)}$
    - Design Motivation: Knowledge-intensive steps (e.g., generating "Isaac Newton") introduce substantial factual knowledge in the higher layers, causing significant distribution shifts and resulting in large, consistently high KL divergence in intermediate layers. Non-knowledge-intensive steps (e.g., generating "was") are already determined in lower layers, leading to small and rapidly decaying KL divergence. Mathematically, the KL divergence is the expectation of Pointwise Mutual Information (PMI), which quantifies the dependency strength between the token and deep knowledge.

2. **Focus Transformation**:

    - Function: Converting the knowledge-aware signal into a decoding temperature.
    - Mechanism: Three transformation functions are provided—
        - **Linear**: $T_t = \sigma \cdot \text{KA}_t + T_0$
        - **Sigmoid Scaling**: $T_t = \frac{\sigma}{\sigma + e^{\text{KA}_t/\sigma}} + T_0$
        - **Exponential Decay**: $T_t = T_0 \cdot e^{\ln(1/2) \cdot \text{KA}_t / \sigma}$ (Best performing)
    - Design Motivation: High knowledge-awareness intensity $\rightarrow$ low temperature $\rightarrow$ sharper distribution $\rightarrow$ more deterministic sampling $\rightarrow$ preserving factuality; conversely, low intensity $\rightarrow$ high temperature $\rightarrow$ smoother distribution $\rightarrow$ more randomized sampling $\rightarrow$ promoting diversity.

3. **Dynamic Focus Training (DFT)**:

    - Function: Integrating the dynamic temperature mechanism into the training stage.
    - Mechanism: During training, the cross-entropy loss for each token also incorporates the dynamic temperature $T_i$, i.e., $\mathcal{L}_{FT} = -\frac{1}{k}\sum_{i=1}^k \log P'_{DFD}(x^*_{i+1}|x^*_{\leq i})$.
    - Design Motivation: Enhanced model attention on knowledge-intensive tokens during training while relaxing constraints on non-knowledge-intensive tokens, promoting flexible generation.

### Loss & Training
- No training is required at inference time; it functions directly as a plug-and-play module.
- Optional DFT training can be applied to further enhance performance.
- By default, the exponential decay transformation is used, with the half-life $\sigma$ selected via grid search in $[0.5, 10]$.
- The KL divergence is calculated only over a subset of high-probability tokens $\mathcal{V}_{\text{head}}$ (plausibility constraint $\alpha=0.1$).

## Key Experimental Results

### Main Results

| Dataset | Decoding Method | Factuality Metric | +DFD | Diversity Dist-1 | +DFD |
|--------|---------|----------|------|-------------|------|
| TruthfulQA | Top-k | 41.04 | **44.55** (+3.5) | 71.63 | **75.71** |
| TruthfulQA | Nucleus | 40.31 | **44.19** (+3.9) | 72.23 | **77.57** |
| StrategyQA | Nucleus | 65.40 | **68.60** (+3.2) | 51.67 | **52.76** |
| WikiText-103 | Top-k (MAUVE) | 12.74 | **13.96** | 49.04 | **49.73** |
| Wikinews | Top-k (FactScore) | 54.62 | **57.05** | 49.92 | **50.65** |

### Ablation Study

| Configuration | Accuracy(↑) | Dist-1(↑) | P-BLEU(↓) | Description |
|------|------------|----------|----------|------|
| Top-k baseline | 63.53 | 51.96 | 20.85 | No DFD |
| +DFD low (Only Lower Layers) | 66.40 | 51.26 | 21.52 | High accuracy but slightly decreased diversity |
| +DFD high (Only Higher Layers) | 63.80 | 52.48 | 19.31 | High diversity but low accuracy |
| **+DFD (All Layers)** | **67.20** | **54.52** | **17.54** | Both balanced |

### Key Findings
- DFD consistently improves both factuality and diversity across all 4 stochastic decoding methods (temperature/top-k/nucleus/typical).
- The exponential decay transformation performs the best, outperforming linear and sigmoid.
- Generalizable across model scales—effective from Llama-3.2-1B to Llama-3.1-70B and MPT-7B.
- Complementary to DoLa (a factuality enhancement method)—DoLa improves factuality but severely harms diversity, whereas adding DFD can partially restore diversity.
- Minimal computational overhead—only requiring extra calculations of the LM head projections at intermediate layers, with less than 1% increase in FLOPs.
- Wins in all three dimensions—fluency, accuracy, and coherence—in the Vicuna QA general conversation scenario.

## Highlights & Insights
- **Using inter-layer distribution discrepancies inside the model as a knowledge-aware signal** is the core innovation—requiring no external knowledge or additional models, purely mining the hierarchical information of the LLM itself. This signal has a solid information-theoretic explanation (PMI expectation).
- The **plug-and-play, algorithm-agnostic** design is highly practical—it can be directly layered on top of any stochastic decoding method with zero migration cost.
- **Simultaneously improving factuality and diversity** breaks the traditional consensus about their trade-off—the key lies in the insight that "different tokens require different randomness."
- This "step-by-step adaptive" concept can be transferred to other decoding scenarios requiring dynamic control (e.g., distinguishing identifiers vs. syntax tokens in code generation).

## Limitations & Future Work
- Computing the KL divergence requires LM head projections at each layer; although the increase in FLOPs is minor, it may impact the efficiency of KV cache and batch inference.
- Limited advantages in short-output scenarios (such as single-word answers).
- The half-life parameter $\sigma$ needs to be searched on the validation set, lacking an automatic selection mechanism.
- The granularity of the knowledge-awareness signal is at the token-level; a coarser chunk-level might be more stable.
- Only validated on Llama and MPT series, and applicability to other architectures (such as Mixtral/MoE) remains unknown.

## Related Work & Insights
- **vs DoLa (Contrastive Decoding)**: DoLa contrasts mature and premature layers to improve factuality but uses fixed layers and does not dynamically adjust temperature; DFD uses full-layer KL divergence and dynamically changes temperature, which is more flexible.
- **vs Nucleus Sampling**: Nucleus uses a fixed $p$-value for truncation, whereas DFD dynamically adjusts temperature after truncation; the two are orthogonal and complimentary.
- **vs Adaptive Temperature (e.g., EntropySampling)**: Prior adaptive temperature methods are based on the entropy of the output distribution, while DFD is based on the discrepancy between intermediate layer distributions, which represents a different signal source and better reflects knowledge dependence.

## Rating
- Novelty: ⭐⭐⭐⭐ Layer-wise KL divergence as a knowledge-aware signal is novel, with an elegant theoretical explanation using PMI.
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Seven datasets, four decoding methods, five model scales, and rich ablation analyses.
- Writing Quality: ⭐⭐⭐⭐ Safe and smooth writing, well-fitting Odysseus metaphor, and clear explanations of formulas.
- Value: ⭐⭐⭐⭐⭐ Plug-and-play, zero-additional-cost decoding improvement with extreme practical value.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Towards Better Open-Ended Text Generation: A Multicriteria Evaluation Framework](towards_better_open-ended_text_generation_a_multicriteria_evaluation_framework.md)
- [\[ACL 2025\] Balancing Diversity and Risk in LLM Sampling: How to Select Your Method and Parameter for Open-Ended Text Generation](balancing_diversity_and_risk_in_llm_sampling_how_to_select_your_method_and_param.md)
- [\[ACL 2025\] CoCoLex: Confidence-guided Copy-based Decoding for Grounded Legal Text Generation](cocolex_legal_text_gen.md)
- [\[ACL 2025\] TagRouter: Learning Route to LLMs through Tags for Open-Domain Text Generation Tasks](tagrouter_learning_route_to_llms_through_tags_for_open-domain_text_generation_ta.md)
- [\[ICLR 2026\] Diverse Text Decoding via Iterative Reweighting](../../ICLR2026/nlp_generation/diverse_text_decoding_via_iterative_reweighting.md)

</div>

<!-- RELATED:END -->
