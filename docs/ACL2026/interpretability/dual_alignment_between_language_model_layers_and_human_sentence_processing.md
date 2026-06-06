---
title: >-
  [Paper Note] Dual Alignment Between Language Model Layers and Human Sentence Processing
description: >-
  [ACL 2026][Interpretability][surprisal] The authors used logit-lens to extract "internal surprisal" from each layer of 19 LMs (GPT-2/Pythia/OPT) and discovered a counter-intuitive "dual alignment": while surprisal from *…
tags:
  - "ACL 2026"
  - "Interpretability"
  - "surprisal"
  - "Logit-Lens"
  - "syntactic ambiguity"
  - "reading time"
  - "dual alignment"
date: 2026-05-08
content_hash: c21090c5b662a93a
---

# Dual Alignment Between Language Model Layers and Human Sentence Processing

**Conference**: ACL 2026  
**arXiv**: [2604.18563](https://arxiv.org/abs/2604.18563)  
**Code**: https://github.com/kuribayashi4/internal_surprisal_targeted_assessment (Available)  
**Area**: Interpretability / Cognition / Psycholinguistics  
**Keywords**: surprisal, Logit-Lens, syntactic ambiguity, reading time, dual alignment

## TL;DR
The authors used logit-lens to extract "internal surprisal" from each layer of 19 LMs (GPT-2/Pythia/OPT) and discovered a counter-intuitive "dual alignment": while surprisal from **shallow** layers aligns best with humans on naturalistic corpora, it is the **deep** layers that align better on **syntactic challenge sentences** (e.g., garden-path, NPS, NPZ, RC, Attachment). This corresponds to human dual-mechanism reading models—"shallow by default + switch to deep reanalysis when difficult"—leading to the proposal of using inter-layer surprisal differences (KL/JS) as "inter-layer prediction updates" to serve as supplementary reading-time features.

## Background & Motivation

**Background**: Computational psycholinguistics has long utilized LM surprisal $S_t = -\log P(w_t \mid w_{<t})$ as a predictor for reading-time (RT), as experiments show RT is approximately linearly correlated with surprisal (Smith & Levy 2013). Recently, Kuribayashi et al. (2025) extended logit-lens to hierarchical levels, finding that surprisal from **early layers** is more human-like than the final layer on naturalistic corpora, partially addressing the "holistic misalignment" issue.

**Limitations of Prior Work**: However, the "**targeted misalignment**" remains unresolved—on **syntactic challenge sentences** such as garden-path (e.g., MVRR "The girl fed the lamb remained..."), NPS, and NPZ, humans slow down significantly at the disambiguating point, but surprisal from the final layer of all LMs **severely underestimates** the magnitude of this slowdown. A natural question arises: Are early layers also better on syntactic challenge sentences?

**Key Challenge**: The authors' experiments provide a negative answer—**early layers show almost no difference on syntactic challenge sentences** (surprisal for D+ and D− is nearly identical because they only capture local co-occurrence and are insensitive to long-range dependencies). This implies that the "best layer" for alignment is not a global constant but depends on task difficulty.

**Goal**: (i) Clarify where the "best layer" resides under different syntactic difficulties; (ii) Provide a unified perspective explaining why this dual alignment exists; (iii) Explicitly formulate this "inter-layer difference" into a new reading-time predictor.

**Key Insight**: The authors analogize the hierarchical forward computation of LMs to the two-stage processing in human reading—**shallow layers ≈ default shallow processing** (fast, surface-level, local) / **deep layers ≈ reanalysis / deep integration** (slow, requiring full context). If this metaphor holds, garden-path sentences should necessitate a "shift to deep layers."

**Core Idea**: Layer-wise surprisal **is not a monotonic curve**—shallow layers align best for naturalistic text, while deep layers align best for syntactically challenging text. Furthermore, the **prediction update from shallow to deep** (surprisal update / KL / JS) can itself serve as a proxy for processing cost.

## Method

### Overall Architecture
The method consists of three steps, all centered around "hierarchical surprisal" in psycholinguistic regressions:

1. **Extract Internal Surprisal**: For each LM, token, and layer $l$, the internal distribution is calculated via logit-lens $P^{(l)}(W = w_t \mid w_{<t}) = \text{softmax}(W_U \text{LN}(h_{t-1}^{(l)}))_{\text{id}(w_t)}$, yielding $S^{(l)}_t = -\log P^{(l)}(w_t \mid w_{<t})$. To address potential unreliability of logit-lens in early layers, the authors supplement the analysis with tuned-lens to verify the stability of the conclusions.
2. **Layer-by-layer Slowdown Regression on Syntactic Ambiguity Data**: Using five syntactic challenge constructions (MVRR / NPS / NPZ / RC / Attachment) from Huang et al. (2024), 120 pairs of D+ / D− sentences are used alongside 1.2M token-level RTs from 2K human self-paced reading experiments. Linear regressions $\hat{y} = \beta_0 + \beta_1 \cdot \text{Surprisal} + \beta_2 \cdot \text{Length} + \beta_3 \cdot \text{LogFreq} + \ldots$ (including spillover) are fitted independently for each layer. Slowdowns estimated by the model at the disambiguating points $t^*$ and $t^*+1$ are compared against human data.
3. **Inter-layer Prediction Update as RT Feature**: Surprisal update is defined as $\text{SU}(w_t \mid w_{<t}) = S^{\text{shallow}}_t - S^{\text{deep}}_t = \log \frac{Q_t(w_t)}{P_t(w_t)}$, and extended to full-distribution $\text{KL}(Q_t \| P_t)$ and the symmetric $\text{JS}(Q_t \| P_t)$. These are integrated with surprisal to observe gains in Predictive Power (PPP).

### Key Designs

1. **Layer-wise Surprisal Extraction via Logit-Lens + Tuned-Lens Robustness**:
    - **Function**: Explicitly extracts "which layer is predicting the next word," transforming surprisal from a single scalar into a layer-wise curve.
    - **Mechanism**: The model's own unembedding matrix $W_U$ (with LayerNorm) is applied to the hidden state $h^{(l)}_{i}$ of the $i$-th token at each layer to obtain a prediction distribution. Subwords are handled via joint probability. Tuned-Lens (Belrose 2023) is used to ensure conclusions are not artifacts of logit-lens bias in early layers (Appendix B.1).
    - **Design Motivation**: Unfolding "model prediction" into a hierarchical sequence allows the investigation of which LM layers correspond to human fast vs. slow processing.

2. **D+/D− × ROI/¬ROI Four-Quadrant PPP Analysis**:
    - **Function**: Precisely isolates the conditions under which the "layer depth → PPP improvement" trend occurs.
    - **Mechanism**: Tokens are categorized into 2×2 quadrants: ambiguous vs. unambiguous sentences (D+ vs D−) and inside vs. outside the disambiguating window (ROI: $t^*-2$ to $t^*+2$ vs ¬ROI). Pearson correlation between layer depth and PPP ($\Delta\text{LL}$) is calculated for each quadrant.
    - **Design Motivation**: This validates the dual alignment hypothesis—if humans switch to deep processing only during ambiguity resolution, the deep-layer advantage should only appear in the D+ ∩ ROI quadrant.

3. **Probability-Update Metrics (SU / KL / JS) as New RT Features**:
    - **Function**: Uses the "prediction difference between shallow and deep layers" as an independent proxy for cognitive cost.
    - **Mechanism**: Three metrics are defined: (i) **SU** looks at $\log Q_t(w_t)/P_t(w_t)$ only at the target word; (ii) **KL** calculates $\mathbb{E}_{w \sim Q_t}[\text{SU}(w)]$ over the entire vocabulary; (iii) **JS** is the symmetric version. Respective per-layer surprisals are z-score normalized. These are then added to regressions to evaluate PPP gains.
    - **Design Motivation**: Based on the "shallow predicts first → deep revises" model, the magnitude of revision should represent effort. JS performs best due to its symmetry and inclusion of full-distribution information.

### Loss & Training
- **Probe-only, No Training**: The authors do not fine-tune LMs; they extract layer outputs via logit-lens/tuned-lens and run linear regressions on reading-time data.
- **Regression Model**: $\text{RT}(w_t) = \beta_0 + \beta_1 \text{Surprisal}(w_t) + \beta_2 \text{Length}(w_t) + \beta_3 \text{LogFreq}(w_t) + \text{spillover}(w_{t-1}, w_{t-2}) + \epsilon$.
- **PPP Metric**: $\Delta\text{LL} = \text{LL}_{\text{full}} - \text{LL}_{\text{baseline}}$, where the full model includes surprisal.
- **Filler Training / Target Testing**: Regressions are trained on filler sentences and tested on target (D+/D−) sentences to avoid overfitting to garden-path structures.

## Key Experimental Results

### Main Results
**Exp.1** (Fig.2): Estimated slowdown vs. human baseline (red line) for 19 LMs:

| Construction | Human Slowdown (ms) | LM Estimated (Across Layers) | Best Layer Position |
|--------------|---------------------|-----------------------------|---------------------|
| MVRR         | ~100                | Max ~50 (Late GPT2-xl)      | Late Layers         |
| NPS          | ~45                 | Max ~25                     | Late Layers         |
| NPZ          | ~100                | Max ~50                     | Late Layers         |
| RC           | ~25                 | Max ~15                     | Late Layers         |
| Attachment   | ~10                 | ~5-10                       | Late Layers         |

Conclusion: **All layers underestimate** human slowdown, but **later layers are significantly closer** than early ones, contradicting the "early layer best" finding in naturalistic reading (Kuribayashi 2025).

**Exp.2** (Tab.2): Pearson correlation (depth vs. PPP) for D+ ∩ ROI:

| Model      | MVRR D+∩RoI | NPS D+∩RoI | NPZ D+∩RoI | RC D+∩RoI | Attachment D+∩RoI |
|------------|-------------|------------|------------|-----------|-------------------|
| GPT2-xl    | +0.88       | -0.07      | +0.88      | +0.96     | -0.32             |
| OPT-13b    | +0.09       | +0.71      | +0.81      | +0.88     | +0.26             |
| Pythia-12b | **+0.88**   | **+0.93**  | **+0.79**  | **+0.97** | **+0.80**         |

Pythia-12B shows strong positive correlations (+0.79 to +0.97) across all constructions in D+ ∩ ROI, whereas D− ∩ ROI shows negative correlations (-0.41 to -0.89).

### Ablation Study
**Exp.3** (Fig.4): Replacing or adding SU / KL / JS to surprisal regressions (average PPP across 19 LMs):

| Feature            | Full Avg PPP        | RoI Avg PPP         | Note                      |
|--------------------|---------------------|---------------------|---------------------------|
| Surprisal (last)   | Baseline            | Baseline            | Standard Approach         |
| Surprisal Up (SU)  | > Baseline          | Marginal            | Target word only          |
| KL(Q‖P)            | Sig. for most       | Partially Sig.      | Asymmetric                |
| JS                 | **Best among three**| Partially Sig.      | Symmetric                 |
| Surprisal + JS     | **Best overall**    | **Best overall**    | Complementary             |

LR tests confirm that Surprisal+JS is significantly superior to Surprisal alone for MVR, RC, and Attachment.

### Key Findings
- **Early Layers Fail on Syntactic Challenges**: In MVRR ("fed the lamb remained"), early layers only perceive local co-occurrence, resulting in nearly identical surprisal for D+ and D−, indicating a lack of long-range syntactic sensitivity.
- **Dual Alignment is Scale-Dependent**: In Pythia models (70M to 12B), the depth-PPP correlation in D+ ∩ ROI increases from 0 to +0.97. Scaling allows models to diverge into "shallow vs. deep" mechanisms, mirroring human dual-processing.
- **Metric Ranking**: JS > KL > SU. JS provides extra explanatory power in ROI regions beyond simple surprisal.
- **Persistent Underestimation**: Even with deep layers and JS, LMs only capture ~50% of human slowdown, suggesting that LM "reanalysis" efforts do not perfectly match human cognitive effort.

## Highlights & Insights
- **Context-Dependent Alignment**: The finding that the "best layer" depends on task difficulty shifts hierarchical probing from a static search to a dynamic perspective.
- **Precision via 2×2 Design**: Using D+/D− and ROI/non-ROI allows for clean causal isolation of the deep-layer advantage.
- **Universal Framework for Cost**: JS/KL/SU metrics provide a way to quantify "reanalysis effort" in any cognitive modeling task.
- **Honest Limitations**: The authors report the persistent underestimation of slowdown rather than over-engineering features to fit the data.

## Limitations & Future Work
- **Residual Underestimation**: LM-based metrics still lack about 50% of the effort required for human garden-path processing.
- **Scope**: Limited to English and written reading data.
- **Instruction-Tuning**: Excludes SFT/RLHF models, which are known to distort cognitive alignment.
- **Theoretical Gap**: The mapping between LM layers (discrete) and human brain dynamics (temporal) still requires a more robust theoretical bridge.

## Related Work & Insights
- **vs. Kuribayashi et al. (2025)**: Extends the "early layer best" theory to a "dual alignment" theory by introducing syntactic challenges.
- **vs. Huang et al. (2024)**: While prior work noted slowdown underestimation in final layers, this paper shows that while final layers are better than early ones for these cases, they are still insufficient.
- **vs. Tenney et al. (2019)**: aligns with the "BERT rediscovers the NLP pipeline" idea, showing a progression from shallow/POS to deep/syntactic-semantic processing.
- **vs. Li & Futrell (2024)**: Quantifies their theoretical shallow/deep processing models using hierarchical surprisal metrics.

## Rating
- Novelty: ⭐⭐⭐⭐ The discovery of "deep layer advantage specifically in garden-paths" is a counter-intuitive finding for the field.
- Experimental Thoroughness: ⭐⭐⭐⭐ Extensive coverage across LMs and phenomena; robust validation with Tuned-Lens.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear framing and excellent visual representation of the core "dual alignment" concept.
- Value: ⭐⭐⭐⭐ Provides new features (JS update) for the psycholinguistics community and dynamic interpretability insights for NLP.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] A Systematic Comparison between Extractive Self-Explanations and Human Rationales in Text Classification](a_systematic_comparison_between_extractive_self-explanations_and_human_rationale.md)
- [\[ICML 2026\] Discovering Implicit Large Language Model Alignment Objectives](../../ICML2026/interpretability/discovering_implicit_large_language_model_alignment_objectives.md)
- [\[AAAI 2026\] Can LLMs Truly Embody Human Personality? Analyzing AI and Human Behavior Alignment in Dispute Resolution](../../AAAI2026/interpretability/can_llms_truly_embody_human_personality_analyzing_ai_and_human_behavior_alignmen.md)
- [\[NeurIPS 2025\] Probabilistic Token Alignment for Large Language Model Fusion](../../NeurIPS2025/interpretability/probabilistic_token_alignment_for_large_language_model_fusion.md)
- [\[ACL 2026\] DPN-LE: Dual Personality Neuron Localization and Editing for Large Language Models](dpn-le_dual_personality_neuron_localization_and_editing_for_large_language_model.md)

</div>

<!-- RELATED:END -->
