---
title: >-
  [Paper Note] Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators
description: >-
  [ICML 2026][AIGC Detection][AI Text Detection] Under a "single fixed threshold protocol," this paper systematically exposes the fragility of AI text detectors under cross-dataset/cross-generator shifts. It proposes fusin…
tags:
  - "ICML 2026"
  - "AIGC Detection"
  - "AI Text Detection"
  - "DeBERTa-v3"
  - "Feature Attention"
  - "Distribution Shift"
  - "Fixed Threshold Protocol"
date: 2026-05-08
content_hash: a9d992c59db13fe7
---

# Feature-Augmented Transformers for Robust AI-Text Detection Across Domains and Generators

**Conference**: ICML 2026  
**arXiv**: [2605.03969](https://arxiv.org/abs/2605.03969)  
**Code**: No public link  
**Area**: AI-Generated Content Detection / NLP / Robustness Evaluation  
**Keywords**: AI Text Detection, DeBERTa-v3, Feature Attention, Distribution Shift, Fixed Threshold Protocol

## TL;DR
Under a "single fixed threshold protocol," this paper systematically exposes the fragility of AI text detectors under cross-dataset/cross-generator shifts. It proposes fusing handcrafted linguistic features with learnable attention weighting into transformer [CLS] representations. Combined with a DeBERTa-v3 backbone, it achieves 85.9% balanced accuracy on the M4 multi-domain multi-generator benchmark, outperforming strong zero-shot baselines (Fast-DetectGPT, RADAR, Log-Rank) by up to +7.22.

## Background & Motivation
**Background**: As cutting-edge LLM outputs become indistinguishable from human text, there is a strong societal demand for AI text detection in academic integrity, content moderation, and data filtering. Mainstream approaches fall into three categories: (i) supervised classifiers (fine-tuned BERT/RoBERTa); (ii) zero-shot methods (DetectGPT/Fast-DetectGPT/RADAR/Binoculars) leveraging LM probability structures; and (iii) provenance schemes like watermarking.

**Limitations of Prior Work**: Research commonly reports near-perfect in-domain metrics (>99% BA), but performance collapses when deployed against unseen domains, generators, or post-hoc rewriting. Crucially, many papers report inflated figures by "tuning thresholds for each test set separately," masking the reality that target domains in real-world deployment lack labels and cannot be recalibrated. Such evaluations are misleading despite their high scores.

**Key Challenge**: Models exhibit high in-domain performance on source distributions $\leftrightarrow$ performance drops sharply on cross-domain, cross-generator, and rewritten semantic-invariant distributions. Furthermore, different backbones show complementary failure modes (BERT tends to be "pro-human," while RoBERTa is "pro-AI"), which overall accuracy alone fails to reveal.

**Goal**: (1) Design a deployment-oriented "single fixed threshold" evaluation protocol; (2) Propose a detector that maintains robustness across domains and generators; (3) Conduct a rigorous comparison between supervised methods and zero-shot baselines under the same protocol.

**Key Insight**: While transformer [CLS] embeddings are powerful, they are easily contaminated by surface cues (topic, formatting). Handcrafted linguistic features (lexical diversity, POS patterns, readability, burstiness, etc.) serve as stable "form-independent, style-related" signals. Fusing the two via dynamic attention allows the model to automatically weight features for each sample, filling the gaps caused by distribution shifts. Additionally, switching to a DeBERTa-v3 backbone is beneficial, as its ELECTRA-style RTD pre-training objective is less sensitive to surface cues.

**Core Idea**: Integrate 30-dimensional handcrafted linguistic features into the transformer [CLS] via "dynamic feature attention" for MLP-based binary classification. A global threshold $\tau^*$ is selected on the HC3 PLUS validation set and **never recalibrated on target domains**, thereby exposing and addressing robustness issues in real deployments.

## Method

### Overall Architecture
Two parallel branches: (i) **Text Branch** feeds input $x$ into BERT/RoBERTa/DeBERTa-v3 to obtain the [CLS] representation $h_{[CLS]}(x)\in\mathbb{R}^d$; (ii) **Feature Branch** extracts 62-dimensional handcrafted linguistic features $f(x)\in\mathbb{R}^{62}$ (lexical diversity, POS, readability, punctuation, LM-perplexity, burstiness, etc.). A top-30 selection $f_k(x)$ is determined on the source domain, then passed through a feature-attention module to obtain a 128-dimensional feature embedding $z_f(x)$. The concatenated vector $h(x)=[h_{[CLS]}(x);z_f(x)]$ is passed to an MLP to output the AI probability $p_\theta(x)$. After training on HC3 PLUS, a one-time $\tau^*$ is selected by maximizing balanced accuracy on the validation set and remains fixed for all target distributions such as M4 and AI-Text-Detection-Pile.

### Key Designs

1.  **Dynamic Feature Attention Module**:
    *   **Function**: Dynamically learns which handcrafted features are more important for each sample, preventing the loss of critical signals that occurs with static top-k selection.
    *   **Mechanism**: A small importance network $u(x)=W_2 \phi(\text{LN}(W_1 f_k(x)))$ maps the 30-dimensional features to importance logits. A softmax yields attention $a(x)=\text{softmax}(u(x))$, followed by element-wise weighting $\bar{f}_k(x)=a(x)\odot f_k(x)$ and projection to 128 dimensions: $z_f(x)=W_3 \bar{f}_k(x)$. While the static top-30 selection (pool of candidates) is fixed in the source domain, the dynamic attention fine-tunes weights at the sample level.
    *   **Design Motivation**: Different samples have different feature sensitivities—long academic texts may rely on burstiness/perplexity, while social media posts may rely on punctuation/POS. Letting the model automatically select weights is more robust than manual tuning.

2.  **DeBERTa-v3 Backbone (RTD Pre-training)**:
    *   **Function**: Replaces BERT/RoBERTa with an encoder trained using ELECTRA-style replaced-token detection to improve cross-distribution robustness.
    *   **Mechanism**: DeBERTa-v3 is trained to distinguish whether each token is a "replaced token" from another model. This objective is inherently isomorphic to AI-text detection, effectively providing "pre-training" for the downstream task. Its disentangled attention also decouples content and position, enhancing robustness to syntactic transformations.
    *   **Design Motivation**: Observations show BERT/RoBERTa drift when encountering rewriting or new generators. DeBERTa-v3, being more sensitive to "real vs. replaced" tokens, reduces reliance on surface lexical/syntactic cues.

3.  **Validation-Calibrated Single Threshold $\tau^*$**:
    *   **Function**: Simulates real-world deployment where target domains are unlabeled and non-recalibratable by locking a single threshold for robustness evaluation.
    *   **Mechanism**: After training, grid search is performed only on the combined HC3 PLUS validation set $\mathcal{D}_\text{val}=\texttt{val\_qa}\cup\texttt{val\_si}$ to find $\tau^*=\arg\max_\tau \text{BA}_{\mathcal{D}_\text{val}}(\tau)$. This $\tau^*$ is used across all M4 domains, generators, and the AI-Text-Detection-Pile without further adjustment.
    *   **Design Motivation**: Exposes the inflated performance in prior research caused by "per-test-set threshold tuning" and aligns benchmarks with practical requirements.

### Loss & Training
Standard binary cross-entropy on HC3 PLUS; feature attention and text encoding are trained end-to-end. Top-30 feature selection is performed once on the source domain using a combination of mutual information and |point-biserial correlation| to ensure zero target-domain leakage. Stability analysis is conducted with 5 random seeds. Zero-shot baselines (Fast-DetectGPT/RADAR/Log-Rank) are re-run under the same fixed threshold protocol.

## Key Experimental Results

### Main Results

| Model | $\tau^*$ | BA test_qa | BA test_si | M4 macro BA | AI-Text-Pile |
| :--- | :--- | :--- | :--- | :--- | :--- |
| BERT | 0.72 | 98.26 | 85.97 | 79.6 | Weak |
| RoBERTa | 0.76 | 99.54 | 86.58 | 77.9 | Weak |
| **Ours** (DeBERTa-v3-base + FeatAttn) | — | — | — | **85.9** (H-R 81.3, AI-R 90.5) | Best |
| Fast-DetectGPT / RADAR / Log-Rank | — | — | — | $\ge$ 7.22 below Ours | — |

(5-seed macro average: 83.15 ± 1.04, demonstrating high stability)

### Ablation Study

| Configuration | Key Observation | Description |
| :--- | :--- | :--- |
| BERT vs RoBERTa (no FeatAttn) | Complementary across M4 domains: BERT high H-R, RoBERTa high AI-R | Single backbone bias |
| + FeatAttn | Consistent transfer BA improvement | Feature attention is a key robustness contributor |
| Feature Category Ablation | Readability and Vocabulary features contribute most | These serve as anchors even after LLM rewriting |
| DeBERTa-v3-base (Standalone) | More balanced than BERT/RoBERTa | RTD pre-training advantage manifests in cross-domain |
| Static top-k vs Dynamic FeatAttn | Dynamic is more robust | Dynamic weighting handles sample heterogeneity |

### Key Findings
*   **Misleading In-domain Scores**: While BERT/RoBERTa achieve >98% on in-domain tests, macro BA drops to 77-80% on M4, proving traditional protocols significantly overstate reliability.
*   **Complementary Failure Modes**: BERT shows high H-R (pro-human) but low AI-R on Reddit/Wikipedia, while RoBERTa shows the opposite on arXiv. This suggests single backbones struggle to balance both ends, making fusion or ensembles promising.
*   **Readability/Vocabulary Stability**: These features are more stable than perplexity-like signals under LLM rewriting, indicating "low-level stylistic cues that LLMs cannot easily strip" are critical for detection.
*   **Beating Zero-Shot Under Equal Protocol**: Under the strict fixed-threshold protocol, supervised fusion (DeBERTa-v3+FeatAttn) outperforms Fast-DetectGPT, RADAR, and Log-Rank by up to +7.22 BA, challenging the narrative that zero-shot methods are inherently "fairer."

## Highlights & Insights
*   Treating the "evaluation protocol" as a first-class methodological contribution. Proving through extensive experiments that traditional reporting is inflated provides significant value to the community.
*   Dynamic feature attention is a lightweight yet effective component that reintegrates "old-school stylometric features" into the transformer pipeline, proving handcrafted features still contribute irreplaceable robustness in the LLM era.
*   The motivation for DeBERTa-v3 is clear—RTD pre-training acts as an ideal pretext task for AI-text detection. This "pre-training-to-downstream alignment" is a useful paradigm for other detection tasks.
*   The use of 5 seeds and re-running zero-shot baselines under the same protocol provides a solid benchmark practice for the field.

## Limitations & Future Work
*   Handcrafted feature selection (62 $\rightarrow$ 30) still relies on prior design; portability to non-English, code, or specialized domains (legal/medical) is unverified.
*   Comparison with the latest instruction-tuned/aligned open-source LLMs (e.g., Qwen, LLaMA-3-Instruct) is missing, though these are primary threats in deployment.
*   While the single threshold protocol mimics deployment, it ignores potential calibration-free adaptation using small amounts of unlabeled target data.
*   The robustness of DeBERTa-v3-base (~184M parameters) against watermarked text or adversarial paraphrasing has not been tested.
*   Inference latency and memory usage are not reported, leaving deployment costs unquantified from an engineering perspective.

## Related Work & Insights
*   **vs. BERT/RoBERTa Supervised Baselines (Devlin 2019, Liu 2019)**: Ours shows these backbones collapse under stricter evaluation protocols, highlighting that the issue lies in evaluation and feature robustness rather than model scale.
*   **vs. Fast-DetectGPT/RADAR/Log-Rank (Zero-Shot)**: While these claim unsupervised fairness, they are outperformed by supervised + FeatAttn by 7+ BA under fixed thresholds, showing that zero-shot "fairness" is conditional.
*   **vs. Binoculars (Hans 2024)**: While Binoculars uses model comparison, Ours takes a complementary route via supervised learning and stylistic features.
*   **vs. HC3 / HC3 PLUS (Guo 2023, Su 2023)**: Ours utilizes HC3 PLUS semantic-invariant rewrites to reveal performance pitfalls hidden by high in-domain scores.
*   **vs. Watermarking (Wouters 2024, Zhang 2024)**: Watermarking provides provenance but has quality trade-offs. Non-provenance detectors remain the primary choice when watermarks are unavailable, making robustness research vital.

## Rating
*   Novelty: ⭐⭐⭐ Simple idea, but evaluation protocol + dynamic feature attention are effective contributions.
*   Experimental Thoroughness: ⭐⭐⭐⭐⭐ Three evaluation suites × multiple backbones × 5 seeds × category ablation + zero-shot re-runs.
*   Writing Quality: ⭐⭐⭐⭐ Clear logic across protocol, method, and experiments; failure-mode tables are intuitive.
*   Value: ⭐⭐⭐⭐ Provides a feasible baseline and rigorous evaluation template for AI text detection deployment.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] On the Salience of Low-Probability Tokens for AI-Generated Text Detection: A Multiscale Uncertainty Perspective](on_the_salience_of_low-probability_tokens_for_ai-generated_text_detection_a_mult.md)
- [\[ACL 2026\] When Personalization Tricks Detectors: The Feature-Inversion Trap in Machine-Generated Text Detection](../../ACL2026/aigc_detection/when_personalization_tricks_detectors_the_feature-inversion_trap_in_machine-gene.md)
- [\[ICLR 2026\] Is Your Paper Being Reviewed by an LLM? Benchmarking AI Text Detection in Peer Review](../../ICLR2026/aigc_detection/is_your_paper_being_reviewed_by_an_llm_benchmarking_ai_text_detection_in_peer_re.md)
- [\[NeurIPS 2025\] DuoLens: A Framework for Robust Detection of Machine-Generated Multilingual Text and Code](../../NeurIPS2025/aigc_detection/duolens_a_framework_for_robust_detection_of_machine-generated_multilingual_text_.md)
- [\[ICML 2026\] Generating Robust Portfolios of Optimization Models using Large Language Models](generating_robust_portfolios_of_optimization_models_using_large_language_models.md)

</div>

<!-- RELATED:END -->
