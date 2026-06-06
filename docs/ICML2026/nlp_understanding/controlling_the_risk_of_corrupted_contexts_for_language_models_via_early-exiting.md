---
title: >-
  [Paper Note] Controlling the Risk of Corrupted Contexts for Language Models via Early-Exiting
description: >-
  [ICML 2026][NLP Understanding][Corrupted Context] This paper formalizes the issue of "performance degradation in LLMs caused by user-provided corrupted contexts" as a risk control problem. By using zero-shot performance…
tags:
  - "ICML 2026"
  - "NLP Understanding"
  - "Corrupted Context"
  - "Early-Exiting"
  - "Distribution-Free Risk Control"
  - "Overthinking"
  - "Learn-then-Test"
date: 2026-05-08
content_hash: d0c8b98b89d8f993
---

# Controlling the Risk of Corrupted Contexts for Language Models via Early-Exiting

**Conference**: ICML 2026  
**arXiv**: [2510.02480](https://arxiv.org/abs/2510.02480)  
**Code**: https://github.com/andreawynn/controlling-corrupted-context-risk  
**Area**: NLP Understanding / LLM Safety / Adaptive Inference  
**Keywords**: Corrupted Context, Early-Exiting, Distribution-Free Risk Control, Overthinking, Learn-then-Test

## TL;DR
This paper formalizes the issue of "performance degradation in LLMs caused by user-provided corrupted contexts" as a risk control problem. By using zero-shot performance as a "safety baseline" and combining dynamic early-exiting (exiting at intermediate layers to avoid "overthinking" harmful contexts) with a context-aware loss and an improved Learn-then-Test framework (preserving negative loss values via risk transformation rather than clipping), this method ensures risk $\leq$ user-specified $\epsilon$ across 9 tasks while achieving over 50% computational speedup.

## Background & Motivation

**Background**: LLMs adapt to tasks through prompt tuning or in-context learning using user-provided contexts. However, corrupted contexts (unintentional mislabeling, malicious injection, task misspecification) can significantly degrade output quality. This risk is particularly severe in medical or clinical deployment scenarios (e.g., mislabeling by busy nurses or biases from clinicians).

**Limitations of Prior Work**: (1) Existing hallucination mitigation mostly involves output post-processing or full fine-tuning, lacking a principled safeguard against arbitrary user contexts; (2) Halawi et al. (2024) noted that LLMs "overthink" harmful inputs—achieving correctness in intermediate layers but being misled by harmful contexts in deeper layers; they proposed pruning attention heads, but fixed pruning cannot guarantee risk bounds; (3) Existing early-exit works (like CALM) focus on acceleration rather than safety and do not utilize zero-shot performance as a safe baseline.

**Key Challenge**: The quality of user context is unknown—it could be helpful (exploitable for performance and speed) or harmful (requiring fallback to zero-shot). A framework adaptive to both ends is necessary.

**Goal**: (1) Formalize context-aided robustness as a risk control problem; (2) Use zero-shot performance as a safe baseline anchor; (3) Simultaneously control risk and gain efficiency through dynamic early-exit and specialized loss/risk design; (4) Validate with theoretical guarantees and empirical results across 9 tasks and 5 models.

**Key Insight**: Observations show that (a) overthinking primarily occurs in deeper layers, which early-exiting naturally bypasses; (b) falling back to zero-shot when confidence is consistently low provides a robust safety net; (c) the context-aware loss $\ell_c = \ell(\bar y_\lambda(x,c), y) - \ell(\hat y(x), y)$ quantifies the "context-driven gain/loss"—positive values indicate overthinking, while negative values indicate helpfulness; (d) the Learn-then-Test (LTT) framework handles non-monotonic risks but requires loss $\in [0, 1]$. This paper extends LTT to negative losses using a risk-preserving transformation.

**Core Idea**: A safe context-aware predictor $\bar p_\lambda$ uses early-exiting for predictions. If confidence across all layers is insufficient, it falls back to zero-shot. LTT is used to select $\hat\lambda$ to guarantee that the overthinking risk $\leq \epsilon$.

## Method

### Overall Architecture

Given a base LLM $p$ and $(x, c) \rightarrow p(\cdot|x, c)$:
1.  **Safe context-aware predictor** $\bar p_\lambda$: Iteratively computes confidence $\mathfrak{C}_l$ from shallow to deep layers. If $\mathfrak{C}_l \geq \lambda$, it exits early; if no layer reaches the threshold, it falls back to zero-shot $p(\cdot | x)$.
2.  **Context-aware loss** $\ell_c(\lambda; x, y, c) = \ell(\bar y_\lambda(x,c), y) - \ell(\hat y(x), y)$: Positive values represent overthinking, negative values represent helpfulness.
3.  **Risk control**: Uses LTT on a calibration set to select $\hat\lambda$ such that $R_c(\hat\lambda) \leq \epsilon$.
4.  **Risk-preserving transformation**: Linearly maps $\ell_c \in [-1, 1]$ to $[0, 1]$ (without clipping) to preserve negative loss information.

### Key Designs

1.  **Safe Context-Aware Predictor (Early-exit + Zero-shot Fallback)**:
    *   **Function**: Exits early when confidence is sufficient (to avoid deep-layer overthinking) and falls back to zero-shot when confidence is consistently low (reliable baseline).
    *   **Mechanism**: Confidence $\mathfrak{C}_l = \max_k p_l(k|x,c)$. Traverses starting from layer 1; the first layer with $\mathfrak{C}_l \geq \lambda$ outputs the prediction. If no layer satisfies this, use $p_L(\cdot | x)$ (zero-shot). So $\bar y_\lambda = \hat y_\lambda$ if any layer is confident, else $\arg\max p_L(k|x)$.
    *   **Design Motivation**: While traditional early-exiting focuses on efficiency, adding zero-shot fallback is key to "safety." Zero-shot is a "known reliable" behavior tested before deployment. If the context is harmful, the model can "ignore" it.

2.  **Context-Aware Loss (Measuring Overthinking)**:
    *   **Function**: Explicitly quantifies the "gain/loss" brought by the context as a loss term.
    *   **Mechanism**: $\ell_c(\lambda; x, y, c) = \ell(\bar y_\lambda(x,c), y) - \ell(\hat y(x), y)$—the difference in loss between the "context-aware prediction" and the "zero-shot prediction" for the same model. Positive values show the context makes it worse; negative values show it helps.
    *   **Design Motivation**: Previous early-exit losses compared "early exit vs. full forward," but the full forward pass itself is already contaminated by harmful context—the reference frame was wrong. Using zero-shot as the reference correctly measures context utility.

3.  **Domain-Preserving Risk Transformation**:
    *   **Function**: Enables the LTT framework to handle losses containing negative values (which cannot be simply clipped to 0).
    *   **Mechanism**: LTT requires $\ell \in [0,1]$. Clipping negative values leads to information loss and over-conservatism. This work uses a linear transformation $\ell' = (\ell_c - a)/(b-a)$ and $\epsilon' = (\epsilon - a)/(b-a)$, ensuring $\ell' \in [0,1]$. LTT selects $\hat\lambda$ in the transformed space, proving $\hat\lambda$ also controls the original risk $R(\ell) \leq \epsilon$.
    *   **Design Motivation**: Clipping fails to distinguish between "exactly equal to zero-shot" and "much better than zero-shot," making LTT too conservative. This transformation preserves signals from helpful contexts, allowing LTT to select more aggressive early-exit thresholds, gaining efficiency while maintaining safety.

## Key Experimental Results

### Main Results across 9 Tasks and 5 Models (Partial)

| Task | Llama-3-8B baseline | + Early-Exit (Loss-Clip) | **+ Ours** | Risk Control ✓ |
| :--- | :--- | :--- | :--- | :--- |
| AG News | 84.2 (Slow) | 78.5 (Fast but conservative) | **84.0** (Fast & $\epsilon \leq 0.05$) | ✓ |
| Financial | 79.3 | 73.8 | **78.9** | ✓ |
| TE-Hate | 65.4 | 58.7 | **64.8** | ✓ |
| SQuAD-v2.0 (open-ended) | 71.2 | 64.5 | **70.7** | ✓ |

Risk transformation prevents over-conservatism; accuracy drops are minimal while latency is significantly reduced.

### Efficiency Gain (vs. CALM clip baseline)

| Task | Our Acceleration | Clip Acceleration |
| :--- | :--- | :--- |
| AG News | **2.1×** | 1.3× |
| Financial | **1.8×** | 1.2× |
| TE-Atheism | **1.9×** | 1.2× |
| SQuAD-v2 | **1.7×** | 1.1× |

Ours consistently achieves > 50% additional speedup compared to clipping; all tasks exceed 1.7×.

### Risk Control Validation
Across all task × model × $\epsilon \in \{0.01, 0.05, 0.1\}$ combinations: empirical risk remains consistently $\leq \epsilon$, proving that DFRC is valid regardless of the ratio of helpful to harmful context (verifying distribution-free properties).

### Selective Fallback
On unsafe (unanswerable) questions, the fallback frequency is **3.3×** higher than on safe questions (3.0% vs 0.9%). When fallback occurs, zero-shot outperforms early-exit in 71% of cases—proving the model falls back when it should.

### Key Findings
- **Zero-shot as a safe baseline is critical**: Formalizing the "to use or not to use context" decision relative to zero-shot makes the risk control framework viable.
- **Risk-preserving transformation vs. clipping**: Clipping makes LTT too conservative, sacrificing speed. Transformation preserves signals, boosting speed by +50% or more.
- **Consistency**: The method is data- and model-agnostic, benefiting models from LLaMA-2/3 to LayerSkip variants.
- **Selective fallback**: Fallback frequency correlates strongly with unsafe-ness (3.3×), showing the discrimination mechanism works.

## Highlights & Insights
- **First to frame "user context risk" as a principled DFRC problem**: Unlike heuristic post-processing for hallucinations, this provides statistical $\epsilon-control$ guarantees, which is vital for high-stakes deployments (medical, legal, financial).
- **Smart framing with zero-shot**: Utilizing zero-shot to anchor "safe behavior" bridges the gap between "knowing little" and "being deceived."
- **Methodological contribution via risk-preserving transformation**: Extending LTT to losses with negative values is a small but important methodological improvement applicable to any "safety-efficiency" risk control scenario.
- **Efficiency as a free byproduct**: Safe and 2× faster—challenging the typical safety-efficiency trade-off by proving that correct design can optimize both.

## Limitations & Future Work
- Reliance on confidence-based exiting; confidence can be unstable for certain tasks (like open generation)—consider ensemble or learned exit strategies.
- If zero-shot itself performs poorly on a task, fallback cannot save the outcome—requires a stronger "safe baseline" concept.
- LTT requires a calibration set $\mathcal{D}_{\text{cal}}$; calibration might fail under significant distribution shifts.
- Validated only on classification and open QA; performance in multi-turn dialogues or agent tasks is unknown.
- Robustness against adversarial contexts specifically targeting the early-exit mechanism has not been analyzed.

## Related Work & Insights
- **vs. CALM (Schuster 2022)**: CALM uses early-exit for speed but its loss clipping is too conservative; this work uses risk-preserving transformation and zero-shot fallback.
- **vs. Halawi 2024 (Attention head pruning)**: That work uses fixed pruning; this work uses dynamic early-exit with DFRC guarantees.
- **vs. Conformal Prediction**: CP provides coverage guarantees for prediction sets; this work provides risk control guarantees, which is more general.
- **Inspiration**: The "zero-shot as safe baseline" idea can be generalized to all scenarios where user input might be harmful (e.g., using baseline LLMs as fallbacks in RAG or no-tool versions as fallbacks for tool-use agents).

## Rating
- **Novelty**: ⭐⭐⭐⭐ The combination of DFRC, early-exit, and zero-shot fallback is novel, though individual components have precedents.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ 9 tasks × 5 models + risk control theoretical validation + clip vs. transformation comparison.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Clear formalization; intuitive figures; solid logic from motivation to experiment.
- **Value**: ⭐⭐⭐⭐⭐ Directly addresses high-stakes LLM deployment with a win-win for risk control and efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] The Imperfective Paradox in Large Language Models](../../ACL2026/nlp_understanding/the_imperfective_paradox_in_large_language_models.md)
- [\[AAAI 2026\] Language Models and Logic Programs for Trustworthy Tax Reasoning](../../AAAI2026/nlp_understanding/language_models_and_logic_programs_for_trustworthy_tax_reasoning.md)
- [\[ACL 2026\] AdapTime: Enabling Adaptive Temporal Reasoning in Large Language Models](../../ACL2026/nlp_understanding/adaptime_enabling_adaptive_temporal_reasoning_in_large_language_models.md)
- [\[ACL 2026\] Lost in the Prompt Order: Revealing the Limitations of Causal Attention in Language Models](../../ACL2026/nlp_understanding/lost_in_the_prompt_order_revealing_the_limitations_of_causal_attention_in_langua.md)
- [\[ACL 2026\] Table Question Answering in the Era of Large Language Models: A Comprehensive Survey](../../ACL2026/nlp_understanding/table_question_answering_in_the_era_of_large_language_models_a_comprehensive_sur.md)

</div>

<!-- RELATED:END -->
