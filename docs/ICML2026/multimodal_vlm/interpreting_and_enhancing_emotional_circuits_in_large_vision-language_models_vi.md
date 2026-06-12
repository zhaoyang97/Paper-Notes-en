---
title: >-
  [Paper Note] VEENA: Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow
description: >-
  [ICML 2026][Multimodal VLM][Emotional Circuits] VEENA utilizes a steering-vector causal attribution framework to locate emotional circuits in LVLMs—discovering a three-stage mechanism: "Adapt (shallow modal alignment) →…
tags:
  - "ICML 2026"
  - "Multimodal VLM"
  - "Emotional Circuits"
  - "steering vector"
  - "causal intervention"
  - "attention head localization"
  - "training-free inference-time intervention"
date: 2026-05-08
content_hash: 6b46ee733b0c486a
---

# VEENA: Interpreting and Enhancing Emotional Circuits in Large Vision-Language Models via Cross-Modal Information Flow

**Conference**: ICML 2026  
**arXiv**: [2605.21980](https://arxiv.org/abs/2605.21980)  
**Code**: To be confirmed  
**Area**: Multimodal VLM / Mechanistic Interpretability / Emotional Understanding  
**Keywords**: Emotional Circuits, steering vector, causal intervention, attention head localization, training-free inference-time intervention

## TL;DR
VEENA utilizes a steering-vector causal attribution framework to locate emotional circuits in LVLMs—discovering a three-stage mechanism: "Adapt (shallow modal alignment) → Aggregate (middle emotion-specific heads aggregation) → Execute (deep emotion-general heads + neurons generation)." It further implements training-free inference-time interventions via "visual emotion enhancement + emotional neuron amplification," significantly mitigating emotional hallucinations.

## Background & Motivation

**Background**: LVLMs are evolving from static perception models to "empathic agents," but suffer from severe emotional hallucinations (e.g., describing a crying face as happy). Unlike object hallucinations, emotional misalignment violates social norms and ethical boundaries. Existing methods follow black-box data-driven routes like visual instruction tuning + RLHF, which do not guarantee internal alignment.

**Limitations of Prior Work**: Emotional circuits in LVLMs remain entirely unexplored. While mechanistic interpretability in LLMs can locate emotional processing components (Tak 2025, Lee 2025), LLM methodologies cannot be directly transferred: (1) **Lack of counterfactuals**: LLMs use word replacement (happy ↔ sad); how can LVLMs "change emotion without changing the narrative"? (2) **Discrete metrics failure**: Emotion is diffusive (overall tone of long text); Next-Token-Prediction (NTP) logits fail to capture it.

**Key Challenge**: Causal analysis of LVLM emotion requires (a) controllable visual counterfactuals and (b) continuous latent space metrics, rather than LLM-style word replacement + logit differences.

**Goal**: (1) Establish a methodology for causal analysis of LVLM emotional circuits; (2) Locate key layers/heads/neurons; (3) Propose training-free inference-time interventions to mitigate emotional hallucinations based on discoveries.

**Key Insight**: (1) Use steering vectors instead of logit differences—extracting emotional directions $S_l$ from hidden states via paired emotional vs. neutral inputs, turning "emotion" into an intervenable continuous vector. (2) Use hit rate (proportion of tokens matching the normalized emotional wheel) instead of NTP accuracy as the latent restoration metric. (3) Coarse-to-fine hierarchical localization—identifying key layers first, then tracing back to heads/neurons.

**Core Idea**: Steering vector probes + latent restoration metric + hierarchical causal attribution → revealing the "Adapt-Aggregate-Execute" mechanism → designing VEENA (VEE reinforces attention flow + ENA amplifies semantic activation) for inference-time intervention.

## Method

### Overall Architecture

Two stages:
- **Stage I**: Extract emotional direction $S_l$ from paired emotional/neutral inputs (filtering valid samples by hit rate threshold).
- **Stage II**: Use $S_l$ as a probe—first identifying key emotional layers (injecting $S_l$ and monitoring hit rate changes), then finding key attention heads (backward activation patching), and finally tracing back to MLP neurons.

VEENA Inference-time Intervention:
- **VEE (Visual Emotion Enhancement)**: Adjusts emotional information routing by strengthening the attention flow of key attention heads.
- **ENA (Emotional Neuron Augmentation)**: Amplifies the semantic activation of explicit state neurons.

### Key Designs

1. **Steering Vector + Latent Restoration Metric**:

    - **Function**: Replaces LLM-style logit difference, making causal analysis of LVLM emotion feasible in continuous latent space.
    - **Mechanism**: Construct paired inputs $X^+ = \text{Concat}(I_{emo}, T_{neu})$ (emotional image + neutral query) vs. $X^- = \text{Concat}(I_{neu}, T_{neu})$ (neutral image + neutral query); compute the residual difference $s_{i,l} = h^+_{i,l,N} - h^-_{i,l,N}$ at the last token for each layer. Filter valid samples where hit rate $\mathcal{H}(X_i^+, y_i) > \tau$, and calculate the global steering vector $S_l = \tfrac{1}{|\mathcal{U}|}\sum_{i \in \mathcal{U}} s_{i,l}$. Evalution is switched to hit rate, which is more robust than logit.
    - **Design Motivation**: Descriptive emotional reasoning cannot rely on single-token logits; steering vectors transform "emotion" into addition-subtraction compatible continuous vectors that can be injected to measure causal effects.

2. **Hierarchical Causal Localization (Layer → Head → Neuron)**:

    - **Function**: Identifies key components of the emotional circuit from coarse to fine.
    - **Mechanism**:
        - Key Layers: Inject $\tilde h^-_{j,l,t} = h^-_{j,l,t} + \alpha S_l$ and observe the relative change in hit rate $\mathcal{C}$.
        - Key Heads: Emotional intention $\mathcal{I}(A_c) = \text{sim}(A_c, S_l)$ + backward activation patching.
        - Key Neurons: Trace the alignment of MLP neuron activations with $S_l$.
    - **Design Motivation**: Direct observation of heads or neurons is noisy; a coarse-to-fine approach makes searching efficient and ensures results at each level are independently interpretable.

3. **"Adapt-Aggregate-Execute" Mechanism Discovery + VEENA Intervention**:

    - **Function**: Reveals the three-stage mechanism of LVLM emotional processing and designs inference-time interventions accordingly.
    - **Mechanism**:
        - **Shallow Layers (Adapt)**: Multimodal alignment of visual features.
        - **Middle Layers (Aggregate)**: Contextual Trigger Neurons encode situational cues → emotion-specific heads aggregate signals to the Query token (visual summarizer), where different emotions activate different heads.
        - **Deep Layers (Execute)**: Query token activates Explicit State Neurons (encoding the emotion itself) → emotion-general heads drive narrative generation.
        - VEENA = VEE (strengthening emotion-specific head attention) + ENA (amplifying Explicit State Neuron activation).
    - **Design Motivation**: Functional decoupling (middle-layer emotion-specific routing vs. deep-layer emotion-general execution) is a key discovery—it implies that "emotion recognition" and "emotion expression" can be intervened in separately; VEENA is training-free and plug-and-play.

## Key Experimental Results

### Main Results on MER-UniBench (hit rate $\mathcal{H}$)

| Method | LLaVA-1.5-7B | LLaVA-1.6-13B | Qwen2-VL-7B |
|------|------|------|------|
| Baseline | 38.2 | 42.7 | 45.6 |
| + Training Data Augmentation | 41.5 | 44.8 | 47.2 |
| + RLHF | 43.7 | 46.1 | 48.5 |
| **+ VEENA (Training-free)** | **48.9** | **51.6** | **53.4** |

VEENA is training-free yet outperforms training-based methods like RLHF by 4-5 points, proving that mechanistic intervention is more precise than black-box optimization.

### Quantitative Evidence for the Three-Stage Mechanism

| Layer Range | Hit Rate Change $\mathcal{C}$ after $S_l$ injection | Explanation |
|------|--------------------|------|
| 1-8 (Shallow) | +3% | Modal adaptation, minimal impact |
| 9-20 (Middle) | **+24%** | Primary field for emotion aggregation |
| 21-32 (Deep) | **+19%** | Emotional execution and narrative generation |

Both middle and deep layers are critical but have different effects—validating functional decoupling.

### emotion-specific vs. emotion-general heads

| Head Type | Avg. Specificity (Selective Activation) | Intervention Effect |
|--------|----------|--------|
| Middle layer emotion-specific | 0.78 | Regulates specific emotions (e.g., fear vs. joy) |
| Deep layer emotion-general | 0.21 | Regulates narrative intensity regardless of emotion |

Clear dichotomy—middle-layer heads are sensitive to emotional categories, while deep-layer heads only manage expression intensity.

### Key Findings
- **Decoupling of middle-layer emotion-specific aggregation and deep-layer emotion-general execution**: Routing (determining the emotion) and execution (expressing it) occur in different layers using different mechanisms, which is the key differentiator for LVLMs compared to LLMs.
- **Training-free SOTA**: VEENA achieves superior results without any training data or parameter updates, outperforming RLHF-class methods.
- **Causal Fidelity**: Intervention experiments confirm that the identified circuit is indeed the true emotional processing path (rather than accidental correlation).
- **Cross-Architecture Generalization**: Consistent results across LLaVA and Qwen2-VL indicate the universality of the mechanism.

## Highlights & Insights
- **First systematic revelation of LVLM emotional circuits**: Fills the gap in emotional mechanistic interpretability for LVLMs; previous work focused primarily on object hallucinations and modal alignment.
- **Steering vector + hit rate as a methodological template for descriptive reasoning**: Can be generalized to examine any LVLM behavior where "output is diffusive long-form text" (e.g., style, stance, abstract reasoning).
- **Correspondence between the Adapt-Aggregate-Execute stages and cognitive science**: Evokes Marr’s tri-level hypothesis (computation-representation-implementation) and Working Memory (encoding-storage-retrieval); LVLMs appear to spontaneously emerge similar functional stratification.
- **Engineering value of training-free interventions**: VEENA does not modify weights or require data, allowing direct deployment on already SFT-ed models—this "post-hoc surgical patch" approach is highly valuable for production LVLMs.

## Limitations & Future Work
- Counterfactual construction relies on paired emotional/neutral images, which is costly and may not cover the full emotional spectrum.
- The intervention coefficient $\alpha$ is manually tuned; adaptive tuning (e.g., based on current emotion confidence) would be preferable.
- Evaluated only on MER-UniBench; generalization across other benchmarks (especially fine-grained emotions like nuance/mixed emotion) is not fully tested.
- VEE + ENA each address specific parts; their effectiveness in higher-level expressive tasks (e.g., irony, empathetic dialogue) remains unknown.
- The number of "emotion-specific" middle-layer heads increases with emotion categories; whether this scales to dozens of fine-grained emotions is uncertain.

## Related Work & Insights
- **vs. LLM Emotion Mechanisms (Tak 2025, Lee 2025)**: Those use word replacement + logit diff, suitable only for short outputs; this work extends to the diffusive outputs of LVLMs.
- **vs. LVLM Interpretability (Jiang 2025, Neo 2025)**: Those focus on object hallucinations; this work specializes in emotional hallucinations.
- **vs. RLHF/DPO for Mitigating Hallucination**: Those are black-box optimizations; VEENA is a surgical mechanistic intervention offering higher controllability.
- **Insights**: The "recognition → expression" functional decoupling framing can be generalized to other LVLM capabilities (reasoning, persona, creativity); whether the "middle-specific + deep-general" pattern is universal remains an open question.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic mechanistic parsing of LVLM emotional circuits with a unique methodology (steering + latent restoration).
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ Multi-model × full MER-UniBench benchmark + hierarchical ablation + head-level causal verification.
- Writing Quality: ⭐⭐⭐⭐⭐ Figures 1/2 intuitively explain the three-stage mechanism, achieving a complete loop of theory and experiment.
- Value: ⭐⭐⭐⭐ Training-free intervention offers direct engineering value for LVLM deployment; the methodology is generalizable to other diffusive behavior analyses.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] Focusing Where Vision Matters: Selective Training for Large Vision Language Models via Visual Information Gain](focusing_where_vision_matters_selective_training_for_large_vision_language_model.md)
- [\[CVPR 2026\] Aligning What Vision-Language Models See and Perceive with Adaptive Information Flow](../../CVPR2026/multimodal_vlm/aif_adaptive_information_flow_vlm.md)
- [\[ACL 2026\] Cross-Modal Taxonomic Generalization in (Vision-) Language Models](../../ACL2026/multimodal_vlm/cross-modal_taxonomic_generalization_in_vision-_language_models.md)
- [\[ICML 2026\] Vision-aligned Latent Reasoning for Multi-modal Large Language Model](vision-aligned_latent_reasoning_for_multi-modal_large_language_model.md)
- [\[NeurIPS 2025\] FlowCut: Rethinking Redundancy via Information Flow for Efficient Vision-Language Models](../../NeurIPS2025/multimodal_vlm/flowcut_rethinking_redundancy_via_information_flow_for_effic.md)

</div>

<!-- RELATED:END -->
