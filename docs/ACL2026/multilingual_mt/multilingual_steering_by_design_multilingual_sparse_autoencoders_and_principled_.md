---
title: >-
  [Paper Note] Multilingual Steering by Design: Multilingual Sparse Autoencoders and Principled Layer Selection
description: >-
  [ACL2026][Multilingual & Machine Translation][Multilingual SAE] This paper demonstrates that multilingual sparse autoencoders (SAEs) and layer selection at the intersection of "multilingual alignment and language separab…
tags:
  - "ACL2026"
  - "Multilingual & Machine Translation"
  - "Multilingual SAE"
  - "activation steering"
  - "layer selection"
  - "language vectors"
  - "CrossSumm"
date: 2026-05-08
content_hash: 2ffc78abe45ca02e
---

# Multilingual Steering by Design: Multilingual Sparse Autoencoders and Principled Layer Selection

**Conference**: ACL2026  
**arXiv**: [2605.23036](https://arxiv.org/abs/2605.23036)  
**Code**: https://github.com/Yusser96/Multilingual-Steering-by-Design/  
**Area**: Multilingual Control / Mechanistic Interpretability  
**Keywords**: Multilingual SAE, activation steering, layer selection, language vectors, CrossSumm

## TL;DR
This paper demonstrates that multilingual sparse autoencoders (SAEs) and layer selection at the intersection of "multilingual alignment and language separability" make SAE language steering more stable. This shifts multilingual control from an empirical layer-sweeping problem to a predictable representation diagnostics problem.

## Background & Motivation

**Background**: Sparse autoencoders (SAEs) have become essential tools for interpreting and intervening in the internal activations of LLMs. Prior work shows that activation steering along specific sparse features or language directions can alter the output language. However, common practices still rely on English-only SAEs, manual layer sweeps, or empirical rules like "middle-to-late layers are more effective."

**Limitations of Prior Work**: Multilingual control is not simply about finding and amplifying a "language feature." If the intervention layer is too early, the model might only access shared cross-lingual semantics, making the language switch imprecise. If the intervention is too late, language identity may be stronger, but generation quality and semantic preservation often decline. Furthermore, the optimal layer varies across models and SAE variants, making experiments expensive to replicate and lacking a mechanistic explanation.

**Key Challenge**: Reliable language steering must satisfy two conditions simultaneously: preserving shared cross-lingual semantic structures to maintain readability, while exposing enough language-specific information for the intervention to push the output toward the target language. Pursuing only separability or only alignment leads to imbalance.

**Goal**: The authors aim to answer three questions: whether SAEs trained on multilingual corpora are better for language control than English-only SAEs; whether effective steering layers can be predicted a priori without full downstream sweeps; and whether this prediction holds across LLaMA-3.1-8B, Gemma-2-9B, machine translation, and cross-lingual summarization tasks.

**Key Insight**: The paper models language steering as a search for a balance point in the representation space. Instead of looking at downstream generation metrics first, the authors analyze the correlation matrix of language vectors at each layer. A high explanation rate by the first principal component indicates strong shared cross-lingual alignment, while the complementary volume indicates strong language separability. The intersection of these two is considered the optimal intervention candidate.

**Core Idea**: Training a MULTI21-SAE covering 21 languages and selecting layers based on the crossover of multilinguality and separability to replace manual layer sweeps.

## Method

The method consists of three parts: constructing language vectors in the dense residual stream or SAE sparse codes; comparing English-only and MULTI21-SAE language structure preservation; and calculating multilinguality/separability from the correlation matrices to perform SAE steering at the crossover layers.

### Overall Architecture

The input is a set of multilingual text samples. For each model layer $\ell$, the method collects SAE codes $\mathcal{Z}^+$ for target language samples and $\mathcal{Z}^-$ for other languages, then constructs a language vector $w_{\mathrm{DiffMean}}(\ell)=\bar{z}_{\ell}^{+}-\bar{z}_{\ell}^{-}$ via DiffMean. This vector serves as both a probe for analysis and a steering direction added to the SAE space during inference.

Two sets of JumpReLU SAEs are trained for LLaMA-3.1-8B and Gemma-2-9B: one using only English Wikipedia, and the other using a balanced Wikipedia corpus of 21 FLORES-200 languages. Both sets share the same total token count, architecture, and optimization hyperparameters to isolate the "language coverage" variable.

Layer selection is independent of downstream metrics. An eigenvalue decomposition is performed on the pairwise Pearson correlation matrix of language vectors at each layer. The explanation rate of the first principal component $f_\ell$ represents multilinguality (strength of shared directions), while $s_\ell=1-f_\ell$ represents separability (degree to which languages remain distinct). The intersection area where $f_\ell$ and $s_\ell$ are balanced is chosen for intervention and validated on machine translation and CrossSumm.

### Key Designs

1.  **DiffMean Language Vectors**:
    - **Function**: Constructs analyzable and intervenable directions for each target language.
    - **Mechanism**: Averages SAE sparse codes for target language tokens and other language tokens in a given layer, taking the difference as the language direction. Additive steering is applied along this direction during inference.
    - **Design Motivation**: Operating on sparse codes is more interpretable than adding vectors to the dense residual stream and allows observation of structures like language family clustering.

2.  **Multilingual SAE Training**:
    - **Function**: Ensures the sparse feature space preserves both shared cross-lingual and language-specific structures.
    - **Mechanism**: MULTI21-SAE uses a balanced Wikipedia corpus across 21 languages (2.1B tokens); EN-SAE uses an equal volume of English data. Both share the JumpReLU architecture and training settings.
    - **Design Motivation**: English-only SAEs tend to encode English high-frequency structures better while weakening low-frequency or cross-lingual features. Multilingual training improves the availability of features required for steering.

3.  **Alignment-Separability Crossover Layer Selection**:
    - **Function**: Predicts effective intervention layers without running downstream layer sweeps.
    - **Mechanism**: Calculates the first principal component explanation rate $f_\ell$. High $f_\ell$ indicates strong multilingual alignment; high $1-f_\ell$ indicates strong language separability. Layers where these are balanced are selected (e.g., L14/L23 for Gemma-2-9B, L13-L15 for LLaMA-3.1-8B).
    - **Design Motivation**: Early layers lack controllable language identity, while late layers may sacrifice semantic quality. The crossover point provides a falsifiable a priori hypothesis.

### Loss & Training

SAE training uses the JumpReLU architecture acting on the residual stream at the `blocks.{layer}.hook_resid_post` hook site. Key hyperparameters include an expansion factor of 8, $L_1$ coefficient of 5.0, JumpReLU bandwidth of $10^{-3}$, 30,000 training steps, batch size of 4,096 tokens, context size of 512, Adam optimizer, learning rate of $5 \times 10^{-5}$, 1,500 warmup steps, and 3,000 decay steps. Each SAE is trained on approximately 123M tokens (approx. 3 H100 GPU hours).

Downstream evaluation uses greedy decoding (temperature 0). Machine translation uses FLORES-200 dev to construct steering vectors and devtest for evaluation. CrossSumm uses 108 English-document/target-language-summary pairs overlapping with the language set.

## Key Experimental Results

### Main Results

| Model / Task | Layer | SAE | LangID | Quality Metric | Semantic Metric | Notes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Gemma-2-9B / FLORES | L14 | MULTI21-SAE | 54.38 | SpBLEU 24.80 | COMET 73.55 | More balanced than Gemma-Scope (45.04 / 15.65 / 61.79) |
| Gemma-2-9B / FLORES | L14 | EN-SAE | 52.19 | SpBLEU 24.90 | COMET 73.17 | Close to MULTI21, but lower LangID / COMET |
| LLaMA-3.1-8B / FLORES | L15 | MULTI21-SAE | 56.97 | SpBLEU 22.53 | COMET 73.25 | Highest semantic quality near crossover layer |
| LLaMA-3.1-8B / FLORES | L15 | EN-SAE | 60.92 | SpBLEU 21.02 | COMET 71.57 | Higher LangID but lower semantic metrics |
| LLaMA-3.1-8B / FLORES | L15 | LLaMA-Scope | 0.10 | SpBLEU 0.00 | COMET 2.72 | Open-source SAE barely supports this steering |

The authors emphasize that no-steering prompt baseline scores are calculated based on the prompt language, while steering results are based on the steering-vector language; they should not be treated as a direct fair comparison. Prompt baselines recorded: Gemma FLORES 75.51 / 31.31 / 85.12, LLaMA FLORES 91.06 / 31.22 / 83.58.

### CrossSumm Analysis

| Model / Task | Layer | SAE | LangID | ROUGE-L | LaSE | Observation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Gemma-2-9B / CrossSumm | L14 | MULTI21-SAE | 48.33 | 4.17 | 16.55 | Higher across all three metrics vs EN-SAE |
| Gemma-2-9B / CrossSumm | L14 | EN-SAE | 42.92 | 4.02 | 15.75 | Weaker control and semantic preservation |
| Gemma-2-9B / CrossSumm | L23 | MULTI21-SAE | 11.81 | 1.25 | 12.38 | Performance degrades significantly in late layers |
| LLaMA-3.1-8B / CrossSumm | L13 | MULTI21-SAE | 66.25 | 3.90 | 24.89 | Strong LangID within crossover region |
| LLaMA-3.1-8B / CrossSumm | L15 | MULTI21-SAE | 30.46 | 2.12 | 30.47 | High LaSE but LangID drops, showing trade-off |
| LLaMA-3.1-8B / CrossSumm | L13 | LLaMA-Scope | 0.00 | 0.29 | 0.00 | Sparse space lacks effective language separability |

### Key Findings

- The value of MULTI21-SAE is not just improving all metrics but providing a more stable trade-off between LangID, SpBLEU/ROUGE-L, and COMET/LaSE, particularly on FLORES compared to open-source SAEs.
- Crossover layer selection is falsifiable: predictions for Gemma-2-9B (L14/L23) and LLaMA-3.1-8B (L13-L15) align with downstream performance peaks.
- LLaMA-Scope exhibits weak separability across all layers, translating to near-zero steering effects, indicating that SAE training corpora and architecture directly impact multilingual controllability.

## Highlights & Insights

- The cleverest aspect is shifting "which layer to steer" from an empirical hyperparameter to a representational statistical problem. This criterion is more interpretable than blind layer sweeps.
- The comparison of multilingual SAEs is controlled: both MULTI21 and EN use 2.1B tokens with identical training configurations, making the conclusions attributable to language coverage rather than training scale.
- Results suggest that language control is not "stronger is better." Excessively high LangID accompanied by dropping COMET/LaSE indicates the model is being pushed toward a target language surface without retaining task semantics.
- Implications for multilingual safety and alignment: if target behaviors require both shared semantics and language-specific features, intervention layers should seek this balance rather than defaulting to the final layer.

## Limitations & Future Work

- **Limited Model Range**: Experiments only cover LLaMA-3.1-8B and Gemma-2-9B; it remains unclear if larger models, encoder-decoder models, or strongly instruction-tuned models follow the same crossover patterns.
- **Insufficient Automated Metrics**: LangID, SpBLEU, COMET, ROUGE-L, and LaSE cannot fully capture stylistic fidelity, code-switching, cultural context, or robustness under ambiguous prompts.
- **Narrow SAE Scope**: The paper focuses on JumpReLU SAEs on the residual stream. Attention/MLP activations, other sparse architectures, or more complex steering constructions remain open questions.
- **Operational Definition of Thresholds**: The 0.5 intersection represents equal alignment and separability, but the authors acknowledge this may not be the only optimal cutoff. Adaptive or model-specific thresholds are needed.
- **Gap with Strong Multilingual Systems**: The paper focuses on mechanistic explanations rather than replacing specialized translation systems; future work requires human error analysis and stronger baselines.

## Related Work & Insights

- **vs. Sparse Activation Steering / FGAA / SAE-TS**: While these prove SAE features can intervene in behavior, they rely on manual selection or local features. This work turns layer selection into a representation-level criterion.
- **vs. Language Neurons (Tang et al. / Deng et al.)**: Prior work focuses on linear encoding of language identity or switchable neurons. This paper argues that separability alone is insufficient; shared structures must also be preserved.
- **vs. LLaMA-Scope / Gemma-Scope**: Open-source SAEs are important baselines, but if training data is English-biased or the sparse space loses multilingual separability, they fail at language steering.
- **Insight**: For multilingual alignment, cross-lingual safety classifiers, or low-resource control, one should perform representation-level balance diagnosis before deciding on intervention layers and training data.

## Rating
- **Novelty**: ⭐⭐⭐⭐☆ Crossover layer selection advances multilingual steering from empirical tuning to mechanistic prediction.
- **Experimental Thoroughness**: ⭐⭐⭐⭐☆ Covers two models, two tasks, and multiple SAE comparisons; lacks human evaluation and broader model families.
- **Writing Quality**: ⭐⭐⭐⭐☆ Mechanistic narrative is coherent; raw numbers are in the appendix, though more figures in the main text could have included direct values.
- **Value**: ⭐⭐⭐⭐☆ Highly relevant for interpretable steering and multilingual control, especially in guiding future SAE training data design.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] DFKI-MLT at SemEval-2026 TASK 7: Steering Multilingual Models Towards Cultural Knowledge](dfki-mlt_at_semeval-2026_task_7_steering_multilingual_models_towards_cultural_kn.md)
- [\[NeurIPS 2025\] Enhancing Multilingual LLM Pretraining with Model-Based Data Selection](../../NeurIPS2025/multilingual_mt/enhancing_multilingual_llm_pretraining_with_model-based_data_selection.md)
- [\[ACL 2026\] SteerEval: Inference-time Interventions Strengthen Multilingual Generalization in Neural Summarization Metrics](steereval_inference-time_interventions_strengthen_multilingual_generalization_in.md)
- [\[ACL 2026\] EMCEE: Improving Multilingual Capability of LLMs via Bridging Knowledge and Reasoning with Extracted Synthetic Multilingual Context](emcee_improving_multilingual_capability_of_llms_via_bridging_knowledge_and_reaso.md)
- [\[ACL 2026\] MultiHaluDet: Multilingual Hallucination Detection via LLM Hidden State Probing](multihaludet_multilingual_hallucination_detection_via_llm_hidden_state_probing.md)

</div>

<!-- RELATED:END -->
