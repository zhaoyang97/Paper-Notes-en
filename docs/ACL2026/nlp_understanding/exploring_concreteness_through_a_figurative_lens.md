---
title: >-
  [Paper Note] Exploring Concreteness Through a Figurative Lens
description: >-
  [ACL 2026][NLP Understanding][Concreteness] The authors use prompt-based probing + DiffMean + SVD to decompose the internal representation of "concreteness" across four LLMs (Llama-3.1-8B / Qwen3-8B / Gemma2-9B / GPT-OSS…
tags:
  - "ACL 2026"
  - "NLP Understanding"
  - "Concreteness"
  - "Figurative Language"
  - "Internal Representations"
  - "Geometric Subspaces"
  - "Representation Intervention"
date: 2026-05-08
content_hash: 573a6a9d2d01e87c
---

# Exploring Concreteness Through a Figurative Lens

**Conference**: ACL 2026  
**arXiv**: [2604.18296](https://arxiv.org/abs/2604.18296)  
**Code**: https://github.com/cincynlp/concreteness-interpretability  
**Area**: NLP Understanding / Linguistics / LLM Interpretability  
**Keywords**: Concreteness, Figurative Language, Internal Representations, Geometric Subspaces, Representation Intervention

## TL;DR
The authors use prompt-based probing + DiffMean + SVD to decompose the internal representation of "concreteness" across four LLMs (Llama-3.1-8B / Qwen3-8B / Gemma2-9B / GPT-OSS-20B). It is discovered that: early layers already distinguish between literal (high concrete) vs. figurative (low concrete) noun usage; mid-to-late layers compress the entire concreteness information into **a single one-dimensional direction**. This axis is shown to perform zero-shot figurative text classification nearly on par with a supervised 4096-dimensional classifier and can be directly added to hidden states for "literal $\leftrightarrow$ figurative" controllable generation.

## Background & Motivation
**Background**: Psycholinguistics and NLP have long regarded "concreteness" as a core semantic dimension—high-concrete words refer to tangible objects perceptible by senses (apple, chair), while low-concrete words refer to abstract concepts (justice, idea). Brysbaert et al. (2014) provided static concreteness scores (1-5) for 40k English words via 4000+ annotators, serving as the gold standard. Subsequent works (Charbonnier & Wartena 2019, Tater 2022, Wartena 2024) predicted these scores using contextual embeddings, proving that models like BERT encode "concreteness shifts in context."

**Limitations of Prior Work**: However, these works are limited to external evaluations like "predicting concreteness scores" or "embedding correlation." No systematic research has addressed: (i) **Which layers** of modern decoder LLMs truly encode concreteness? (ii) Does concreteness occupy a **dedicated geometric direction** in the representation space? (iii) Can this direction be used to **intervene** in the model's literal/figurative generative tendencies?

**Key Challenge**: Concreteness is "context-sensitive"—for the same word "window," "window was broken" is a high-concrete literal use, while "window of opportunity" is a low-concrete figurative use (metaphors/metonyms/idioms trigger this shift, per Lakoff & Johnson 1980). However, existing probing methods face an engineering bottleneck in decoder-only LLMs: extracting contextual embeddings for a noun is limited by "left-to-right" causal masking, meaning the embedding might not "see" the figurative cues appearing later in the sentence (e.g., in "chain of events led to his downfall," the embedding for "chain" is computed before reading "events/downfall").

**Goal**: (1) Design a probing scheme for decoder LLMs that correctly captures **contextual concreteness**; (2) Characterize internal representations across layer-wise and geometric dimensions; (3) Verify that these representations are interpretable, usable for downstream tasks, and causal via intervention.

**Key Insight**: Ours draws on the DiffMean method from the "geometry of truth" work by Marks & Tegmark (2024)—a simple linear direction formed by "high-class mean minus low-class mean" can capture many semantic dimensions. SVD is then used to synthesize multi-layer DiffMeans into a global axis.

**Core Idea**: By using the prompt "On a scale of 1 to 5, what is the concreteness of [word] in this sentence?", the LLM aggregates full sentence information into the last token. The hidden state of that token carries "contextual concreteness," which is used as input for probes to quantitatively analyze layer-wise encoding, geometric structure, and causal intervention.

## Method

### Overall Architecture
The method consists of three steps: (a) **Layer-wise probing**: Using 25,000 Wikipedia sentences and 600 literal/figurative synthetic sentence pairs with prompts to extract hidden states at each layer. An MLP regressor is trained to predict Brysbaert concreteness scores, generating a "layer × Pearson r" curve to identify encoding layers. (b) **Geometric axis**: Nouns in Wikipedia sentences are categorized as high (score > 4) or low (score < 2). For each layer, a DiffMean vector is calculated as $w^{(l)} = \mu^{(l)}_{high} - \mu^{(l)}_{low}$. All layer DiffMeans are stacked into a matrix $W$, and SVD is applied to extract the top-$k$ right singular vectors $B_k = V^\top_{1:k}$ as the "global concreteness subspace." $k=1$ is tested to check for single-direction compression. (c) **Causal steering**: A unit direction $\mathbf{u}$ is added to hidden states at a mid-to-late layer: $h^{(\ell)}_{\text{steer}} = h^{(\ell)} + \alpha \mathbf{u}$ ($\alpha > 0$ for literal, $\alpha < 0$ for figurative). The model then continues decoding to perform rewrites.

### Key Designs

1. **Prompt-based probing + Last token representation**:
    - **Function**: Overcomes the causal masking limitation where decoder-only LLMs "cannot see the future context" at the noun's position, obtaining a hidden state that reflects the full contextual concreteness.
    - **Mechanism**: Placing the full sentence and target word in the prompt: "Sentence: [sentence] On a scale of 1 to 5... what is the concreteness of the word [target_word]?". **The hidden state of the prompt's last token** is taken as the representation—this token has "seen" the whole sentence via causal attention. Two paths are used: (Gen) model generates a number; (Tok) hidden state is fed to an MLP.
    - **Design Motivation**: Prompt sensitivity analysis showed that if the target word is not at the end, Pearson r drops from 0.98 to 0.80±0.10, confirming strong recency bias in decoders. Additionally, the Tok path yielded much higher r (0.82-0.92) than the Gen path (0.58-0.70), indicating "models know concreteness but cannot verbalize the exact number."

2. **DiffMean + Multi-layer SVD for global 1-D axis**:
    - **Function**: Identifies a "shared concreteness primary direction" across all layers to verify if concreteness is compressed into a single geometric dimension.
    - **Mechanism**: 2,256 high and 2,116 low concrete instances are balanced from Wikipedia. Each layer's DiffMean $w^{(l)}$ is computed. These are stacked into matrix $W$. SVD provides orthogonal directions $V^\top$ ranked by discriminative power. Top-$k$ vectors form the layer-agnostic subspace $B_k$. Projecting hidden states onto $B_k$ yields scores for ROC AUC evaluation.
    - **Design Motivation**: DiffMean is a lightweight linear method more interpretable than logistic regression—it represents a specific direction vector. SVD aggregation provides a stable global axis; results show AUROC stabilizes around 0.90 for $k=1$ in mid-to-late layers, while increasing $k$ to 2/3/4 decreases AUROC, providing clear evidence of inverse scaling.

3. **Causal steering: Adding axis to hidden states**:
    - **Function**: Upgrades the geometric axis from a "correlation discovery" to a "causal control knob," allowing LLMs to generate literal or figurative sentences without parameter updates or prompt engineering.
    - **Mechanism**: Selecting a layer with clear signals (e.g., L20 for Llama-3.1-8B), hidden states are modified by $\alpha \cdot \mathbf{u}$ ($\alpha = \pm 40$) during decoding. The model is then asked to "Rewrite the following sentence clearly and naturally:". No mention of figurative/literal is made in the prompt.
    - **Design Motivation**: Proves the axis is a control signal rather than a bystander feature. Human evaluation of 100 sentences shows Lit $\rightarrow$ Fig increases from 0 to 15% and Fig $\rightarrow$ Lit from 39-52% to 67-75%, providing clean causal evidence.

### Loss & Training
No traditional training is involved in the core method: (a) Probing MLP uses hyperparameters from Wartena (2024) (512 $\rightarrow$ 256 $\rightarrow$ 128 layers + ReLU + 0.2 dropout, AdamW lr=1e-5, 50 epochs, batch=15, 10-fold CV); (b) DiffMean is a closed-form calculation; (c) SVD is a closed-form decomposition; (d) Steering is an inference-time addition with zero parameter updates.

## Key Experimental Results

### Main Results
**Probing Correlation** (Pearson r between predicted concreteness and Brysbaert human ratings):

| Model | With Context (Gen) | With Context (Tok) | W/o Context (Gen) | W/o Context (Tok) |
|-------|---------------------|---------------------|--------------------|--------------------|
| Llama-3.1-8B | 0.66 | 0.88 | 0.70 | **0.98** |
| Qwen3-8B | 0.60 | 0.87 | 0.65 | **0.98** |
| Gemma2-9B | 0.64 | 0.92 | 0.68 | **0.98** |
| GPT-OSS-20B | 0.58 | 0.82 | 0.63 | **0.98** |

**Zero-shot Figurative Classification** (Llama-3.1-8B, AUROC, 1-D axis vs. 4096-D trained classifier):

| Task | Dataset | 1-D Subspace (zero-shot) | Full Rep. (trained) | Retention |
|------|---------|--------------------------|---------------------|-----------|
| Idioms | MAGPIE | 95.2 | 98.5 | 96.6% |
| Idioms | EPIE | 95.3 | 99.2 | 96.1% |
| Metaphor | VUA | 95.7 | 97.6 | 98.1% |
| Metaphor | MUNCH | 93.2 | 95.1 | 98.0% |
| Metonymy | ConMeC | 60.2 | 62.6 | 96.2% |
| Metonymy | MetFuse | 85.7 | 96.3 | 89.0% |

### Ablation Study
**Impact of Subspace Dimension $k$ + Causal Steering Human Eval**:

| Configuration | Key Metric | Description |
|---------------|------------|-------------|
| $k=1$ subspace (Mid-late layers) | AUROC $\approx$ 0.90 | Single direction is sufficient; verifies 1D compression |
| $k=2$ | AUROC $\downarrow$ | Additional directions dilute the signal |
| $k=3$ | AUROC $\downarrow\downarrow$ | Further decline |
| $k=4$ | AUROC $\downarrow\downarrow\downarrow$ | Introduction of noise directions |
| Lit $\rightarrow$ Fig (Llama-3.1, no steering) | 0/100 | Strong literal bias in models |
| Lit $\rightarrow$ Fig (Llama-3.1, $\alpha=-40$) | 12/100 | 12% converted to figurative |
| Fig $\rightarrow$ Lit (Llama-3.1, no steering) | 42/100 | Natural bias toward literal |
| Fig $\rightarrow$ Lit (Llama-3.1, $\alpha=+40$) | 71/100 | Literal proportion nearly doubles |

### Key Findings
- **Early layer literal vs. figurative classification**: On synthetic data, the predicted score offset $\delta^{(l)}_{\text{mean}} = C^{(l)}_{\text{pred}} - C_{\text{static}}$ diverges as early as layer 2, consistent with mechanistic interpretability findings that early layers handle semantic type judgment.
- **Mid-to-late layer 1D compression**: All 4 models compress concreteness into $k=1$ dimension in middle-to-late layers. GPT-OSS-20B (MoE) starts this compression even earlier.
- **1-D axis matches 4096-D supervised classifiers**: Zero-shot 1-D AUROC retains 95-98% of the information available to a fully supervised classifier for idioms and metaphors.
- **Metonymy as an exception**: 1-D AUROC on metonymy is lower (60-86%), aligning with linguistic theory that metonymy (e.g., "the church joined the movement") involves smaller concreteness shifts compared to metaphors.
- **Steering asymmetry**: Fig $\rightarrow$ Lit is significantly easier than Lit $\rightarrow$ Fig, reinforcing observations that LLMs have a strong inherent literal bias.
- **Weak verb involvement**: Concreteness shifts in verbs are far smaller than in nouns, matching linguistic consensus.

## Highlights & Insights
- **Engineering details dictate conclusions**: The requirement for the target word to be at the prompt's end reveals recency bias as a hidden trap in decoder probing.
- **Representation-as-control paradigm**: Using geometric axes as knobs without re-prompting or fine-tuning represents an emerging paradigm for controlling attributes like figurativity or truthfulness.
- **DiffMean + SVD synthesis**: Using SVD to find a "consensus direction" across layers provides a more robust and layer-agnostic axis than single-layer probes.
- **"Models know but can't verbalize"**: The gap between Tok path (0.92) and Gen path (0.70) suggests hidden states are more accurate than text output for assessment.
- **Downstream Value**: Training-free inference-time control offers a low-cost alternative for style transfer, literary creation, and figurative translation.

## Limitations & Future Work
- **Lack of human contextual labels**: Correlation is measured against static scores; direct contextual concreteness ground truth is missing.
- **Axis purity**: The direction might encode entropic signals like frequency or imageability alongside concreteness.
- **Concreteness $\neq$ Figurativity**: Word sense disambiguation also causes shifts; distinguishing these from figurativity remains a challenge.
- **Domain specificity**: Relies on Wikipedia and synthetic data; verification in poetry or dialogue is needed.
- **Future Work**: Use sparse autoencoders (SAE) to decouple concreteness from other latent features and explore multi-axis joint intervention.

## Related Work & Insights
- **vs. Wartena (2024)**: Upgrades concreteness probing from BERT-era correlation to decoder LLM geometry and causal control.
- **vs. Marks & Tegmark (2024)**: Adapts the "Linear Representation Hypothesis" framework from truthfulness to figurative language.
- **vs. Chakrabarty (2021) MERMAID**: Offers a training-free alternative to fine-tuned figurative generation models.

## Rating
- Novelty: ⭐⭐⭐⭐ (Solid application of mi-framework to a new linguistic dimension).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ (Extensive multi-model, multi-dataset, and human-in-the-loop verification).
- Writing Quality: ⭐⭐⭐⭐ (Clear visualization and honest discussion of limitations).
- Value: ⭐⭐⭐⭐ (Practical zero-shot control technique and strong case study for interpretability).

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Agree, Disagree, Explain: Decomposing Human Label Variation in NLI through the Lens of Explanations](agree_disagree_explain_decomposing_human_label_variation_in_nli_through_the_lens.md)
- [\[ACL 2026\] MetFuse: Figurative Fusion between Metonymy and Metaphor](metfuse_figurative_fusion_between_metonymy_and_metaphor.md)
- [\[NeurIPS 2025\] Generalization Error Analysis for Selective State-Space Models Through the Lens of Attention](../../NeurIPS2025/nlp_understanding/generalization_error_analysis_for_selective_state-space_models_through_the_lens_.md)
- [\[ACL 2026\] BoundRL: Efficient Structured Text Segmentation through Reinforced Boundary Generation](boundrl_efficient_structured_text_segmentation_through_reinforced_boundary_gener.md)
- [\[ACL 2026\] LLM-Guided Semantic Bootstrapping for Interpretable Text Classification with Tsetlin Machines](llm-guided_semantic_bootstrapping_for_interpretable_text_classification_with_tse.md)

</div>

<!-- RELATED:END -->
