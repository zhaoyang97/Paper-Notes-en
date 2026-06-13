---
title: >-
  [Paper Note] Evolution and compression in LLMs: On the emergence of human-aligned categorization
description: >-
  [ICLR2026][Model Compression][information bottleneck] Through the Information Bottleneck (IB) framework and the Iterated In-Context Language Learning (IICLL) paradigm…
tags:
  - "ICLR2026"
  - "Model Compression"
  - "information bottleneck"
  - "color naming"
  - "iterated learning"
  - "semantic categories"
  - "LLM alignment"
date: 2026-05-08
content_hash: bec3533cc3aef582
---

# Evolution and compression in LLMs: On the emergence of human-aligned categorization

**Conference**: ICLR2026
**arXiv**: [2509.08093](https://arxiv.org/abs/2509.08093)  
**Code**: [infocoglab/evolution-compression-llms](https://infocoglab.github.io/evolution-compression-llms)  
**Area**: Model Compression
**Keywords**: information bottleneck, color naming, iterated learning, semantic categories, LLM alignment

## TL;DR

Through the Information Bottleneck (IB) framework and the Iterated In-Context Language Learning (IICLL) paradigm, this paper demonstrates that LLMs can spontaneously develop category structures that are highly aligned with human semantic categorization systems and achieve near-optimal compression efficiency, without having been trained on any IB objective.

## Background & Motivation

### Root Cause

**Background**: Extensive evidence indicates that human semantic categorization systems (e.g., color naming) adhere to the Information Bottleneck (IB) principle—achieving a near-optimal trade-off between lexical information complexity and communicative accuracy. This theoretical framework was proposed by Zaslavsky et al. (2018) and has been broadly validated across 110 languages in the World Color Survey (WCS).

However, LLMs are trained on a language modeling objective (next-token prediction), not on an IB objective. This raises a fundamental question: are LLMs merely imitating categorization patterns present in training data, or do they possess an intrinsic, human-like inductive bias that spontaneously drives efficient semantic compression?

Color naming is a canonical domain for studying categorization in cognitive science, with uniquely rich cross-linguistic human data (the WCS dataset) and cultural evolution experiments (Xu et al., 2013), making it an ideal testbed for assessing whether LLMs are aligned with human cognition.

**Goal**:
1. How do LLMs' English color naming systems perform in terms of IB efficiency and human alignment?
2. Are LLMs merely imitating patterns in training data, or do they possess a genuine inductive bias toward IB efficiency?
3. Does such a bias generalize beyond the color domain to other semantic domains?

## Method

### Experiment 1: English Color Naming

- The WCS standard color grid (330 color chips) is used as stimuli.
- 39 models spanning 6 model families (Gemini, Gemma 3, Llama 3, Qwen 2.5, OLMo 2, GPT-2) are evaluated.
- Input modalities: text (sRGB coordinates) and image (for multimodal models).
- Constrained generation: Gemini uses controlled generation; open-source models are evaluated via log-probability scoring.
- Evaluation metrics: IB efficiency loss $\varepsilon = \min_\beta \frac{1}{\beta}(\mathcal{F}_\beta[q] - \mathcal{F}_\beta^*)$ and Normalized Information Distance (NID) for alignment.

### Experiment 2: Iterated In-Context Language Learning (IICLL)

This is the paper's core methodological contribution, combining the iterated learning paradigm from cognitive science with the in-context learning capabilities of LLMs:

1. **Initialization**: A pseudo color naming system is initialized with a random partition, with category count $k \in \{2, 3, 4, 5, 6, 14\}$.
2. **Each generation**: A small number of "color–pseudo-label" pairs are sampled from the previous generation's language system $L_{t-1}$ as in-context examples $d_{t-1}$.
3. **Inference**: After in-context learning, the LLM assigns labels to all 330 colors to produce the new system $L_t$.
4. **Iteration**: This process is repeated over multiple generations, and the evolutionary trajectory of the system is observed.

Key design: pseudo label words (non-English color terms) are used, and the model is not informed that the inputs are colors—stimuli are described only as objects with "features." This ensures the model cannot directly draw on color knowledge from training data.

### IB Theoretical Framework

The IB objective is:

$$\mathcal{F}_\beta[q] = I_q(M;W) - \beta \cdot I_q(W;U)$$

where $I_q(M;W)$ is the complexity (mutual information between speaker meanings and words), $I_q(W;U)$ is the accuracy (information about world states retained by words), and $\beta \geq 0$ controls the trade-off. Optimal systems lie on the IB theoretical bound.

## Key Experimental Results

### English Color Naming Results

- LLMs vary substantially in complexity and alignment with English color naming.
- **Model scale and instruction fine-tuning** are two key factors: larger instruction-tuned models achieve better alignment and IB efficiency.
- Gemini 2.0 and Gemma 3 27B (inst.) most closely approximate human English color naming.
- Notably, many state-of-the-art models fail to reproduce English color naming (e.g., Llama 3.3 70B inst.).
- Some models (OLMo 2 32B inst., Qwen 2.5 VL 7B inst.) produce systems more similar to low-resource languages in the WCS than to English.
- Image input does not consistently outperform text input; CIELAB coordinates generally underperform sRGB.

### IICLL Cultural Evolution Results

- **Gemini 2.0**: The only model capable of spanning the full range of complexity observed in human languages; IICLL chains converge to near-optimal IB solutions resembling WCS languages and human iterated learning data.
- **Gemma 3 27B, Qwen 2.5 32B, Llama 3.3 70B**: Also converge to IB-efficient solutions, but only within the low-complexity regime.
- All models converge to near the IB bound within approximately 4 generations, paralleling the dynamics of human iterated learning.
- Rotation analysis confirms that Gemini's efficiency and alignment are non-trivial—randomly rotating the color mapping leads to significant degradation.

### Extension to the Shepard Circular Domain

- Gemini is tested on a two-dimensional Shepard circular space (64 stimuli) defined by radius and spoke angle.
- After IICLL transmission, categories progressively become spatially compact and distinguish regions along both dimensions.
- This provides preliminary evidence that the LLM's IB bias may generalize across domains.

## Highlights & Insights

1. **Deep integration of theory and experiment**: The IB framework and iterated learning paradigm from cognitive science are seamlessly transferred to LLM research, yielding a highly persuasive methodology.
2. **IICLL paradigm innovation**: The use of pseudo-labels eliminates the confound of training data imitation, directly probing LLMs' intrinsic inductive biases.
3. **Large-scale model comparison**: A systematic comparison of 39 models across 6 families reveals clear relationships between model scale, instruction fine-tuning, and IB efficiency.
4. **Cross-domain generalization**: The Shepard circle experiment provides preliminary evidence of generalization beyond the color domain.
5. **A new perspective on human–AI alignment**: IB efficiency may be an emergent property of intelligent behavior, arising in both humans and LLMs despite neither being explicitly trained to optimize this objective.

## Limitations & Future Work

1. **Only Gemini 2.0 achieves the full complexity range**: Other state-of-the-art models are limited to low-complexity solutions, indicating that IICLL places very high demands on in-context learning ability, and the generality of the conclusions remains to be verified.
2. **Source of the bias is unclear**: Whether the IB efficiency bias originates from the training data distribution, instruction fine-tuning, or model architecture remains unresolved, as the paper does not disentangle these factors.
3. **Specificity of the color domain**: Colors are richly represented numerically in internet text (hex, RGB), potentially giving LLMs a natural advantage in this domain; cross-domain generalization is supported only by preliminary results on the Shepard circle.
4. **Absence of communicative pressure**: IICLL simulates cultural transmission only and does not incorporate the functional pressure of actual communication, leaving a gap relative to real language evolution.
5. **Evaluation limited to English**: Although WCS data are used, direct evaluation of LLMs is conducted only on English color terms.

## Related Work & Insights

| Work | Focus | Distinction from This Paper |
|------|--------|-----------------------------|
| Marjieh et al. (2024) | Color naming in a small set of models (GPT-3/4, etc.) | Systematic comparison of 39 models + IB analysis + IICLL |
| Abdou et al. (2021) | Internal color representations in LLMs | Focuses on naming behavior under prompt interaction |
| Zhu & Griffiths (2024) I-ICL | In-context priors of LLMs | Extended to IICLL, directly replicating human iterated language learning experiments |
| Carlsson et al. (2024) | IB-efficient color naming in neural network agents | Uses LLMs rather than agents trained from scratch |
| Ren et al. (2020) NIL | Compositional language in neural iterated learning | Focuses on semantic compression efficiency rather than compositionality |

## Related Work & Insights

- **Implications for LLM alignment research**: IB efficiency as a quantifiable dimension of human–AI alignment captures semantic-level correspondence more deeply than conventional benchmark evaluations.
- **Implications for model compression**: Although categorized under model compression, the paper addresses "semantic compression" rather than parameter compression; nonetheless, its information-theoretic perspective (IB principle) provides important reference for understanding how models encode semantics within limited representations.
- **Cognitive effects of instruction fine-tuning**: Instruction fine-tuning not only improves task performance but may also reshape the model's semantic organization, bringing it closer to human cognitive structures.
- **Cultural evolution × AI**: IICLL provides a scalable experimental paradigm for studying cultural evolution dynamics within LLMs.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ (Integrates the IB framework from cognitive science with LLMs; IICLL is a significant methodological contribution)
- Experimental Thoroughness: ⭐⭐⭐⭐ (Large-scale comparison of 39 models, though cross-domain generalization evidence remains preliminary)
- Writing Quality: ⭐⭐⭐⭐⭐ (Clear structure, well-motivated theory, and polished figures)
- Value: ⭐⭐⭐⭐ (Provides a fundamentally new theoretical perspective on LLMs' semantic organization capabilities)

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICML 2026\] xKV: Cross-Layer KV-Cache Compression via Aligned Singular Vector Extraction](../../ICML2026/model_compression/xkv_cross-layer_kv-cache_compression_via_aligned_singular_vector_extraction.md)
- [\[ICLR 2026\] LLM DNA: Tracing Model Evolution via Functional Representations](llm_dna_tracing_model_evolution_via_functional_representations.md)
- [\[ICLR 2026\] Paper Copilot: Tracking the Evolution of Peer Review in AI Conferences](paper_copilot_tracking_the_evolution_of_peer_review_in_ai_conferences.md)
- [\[CVPR 2026\] MEMO: Human-like Crisp Edge Detection Using Masked Edge Prediction](../../CVPR2026/model_compression/memo_human-like_crisp_edge_detection_using_masked_edge_prediction.md)
- [\[NeurIPS 2025\] TokenSqueeze: Performance-Preserving Compression for Reasoning LLMs](../../NeurIPS2025/model_compression/tokensqueeze_performance-preserving_compression_for_reasoning_llms.md)

</div>

<!-- RELATED:END -->
