---
title: >-
  [Paper Note] Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning
description: >-
  [ICLR 2026][LLM Reasoning][Compositional Generalization] This paper leverages information-theoretic generalization bounds and mechanistic interpretability to demonstrate that the core mechanism of CoT training is **compo…
tags:
  - "ICLR 2026"
  - "LLM Reasoning"
  - "Compositional Generalization"
  - "Chain-of-Thought Training"
  - "Information-Theoretic Generalization Bounds"
  - "Reasoning Circuits"
  - "OOD Generalization"
date: 2026-05-08
content_hash: b1877680afa737f3
---

# Compositional Generalization from Learned Skills via CoT Training: A Theoretical and Structural Analysis for Reasoning

**Conference**: ICLR 2026
**arXiv**: [2502.04667](https://arxiv.org/abs/2502.04667)  
**Code**: [https://github.com/chen123CtrlS/T-CotMechanism](https://github.com/chen123CtrlS/T-CotMechanism)  
**Area**: AI Safety / LLM Reasoning
**Keywords**: Compositional Generalization, Chain-of-Thought Training, Information-Theoretic Generalization Bounds, Reasoning Circuits, OOD Generalization

## TL;DR
This paper leverages information-theoretic generalization bounds and mechanistic interpretability to demonstrate that the core mechanism of CoT training is **compositional generalization**—the model learns to systematically compose previously acquired simple skills to solve novel complex problems, internalizing this ability as a two-stage compositional reasoning circuit that extracts intermediate results at shallower layers, freeing deeper layers to focus on subsequent reasoning steps.

## Background & Motivation

**Background**: CoT training (e.g., DeepSeek-R1's long CoT cold-start, OpenAI O1's RFT) has become a central paradigm for enhancing LLM reasoning, yet the **mechanism** by which it improves generalization remains poorly understood.

**Limitations of Prior Work**:
   - Prior theoretical analyses focused on how CoT enhances Transformer expressivity and computational complexity classes, without explaining how capabilities emerge during training.
   - Models trained without CoT exhibit a "compositionality gap" in ID generalization—they may know all basic facts yet fail to compose them.
   - More critically, models without CoT training entirely fail at OOD generalization (novel compositional patterns).

**Key Challenge**: Why does CoT enable generalization to extend from ID to OOD settings? Does the model learn only "what to think" (correct answers) or "how to think" (reasoning procedures)?

**Goal**:
   - (Q1) Does CoT training improve ID and OOD generalization, and what is the theoretical principle?
   - (Q2) How is this generalization capability realized internally within the model?

**Key Insight**: CoT is decomposed as $P(Y|X) = \sum_C P(Y|X,C) \cdot P(C|X)$, where $C$ denotes the reasoning chain. CoT training explicitly learns $P(C|X)$ and $P(Y|X,C)$, whereas training without CoT learns only $P(Y|X)$.

**Core Idea**: CoT training teaches the model "how to think"—by decomposing complex problems into compositions of already-learned simple skills ($P(C|X)$ + $P(Y|X,C)$), it brings OOD problems closer to the ID distribution, enabling systematic generalization.

## Method

### Overall Architecture
The analysis proceeds along two parallel tracks: (1) information-theoretic generalization bounds that decompose the error into ID and OOD components, proving that CoT training reduces the OOD component; and (2) mechanistic analysis via Logit Lens and causal tracing, revealing that CoT training is internalized as a two-stage compositional circuit.

### Key Designs

1. **Information-Theoretic Generalization Bounds (Theorem 1 & 2)**:

    - **Function**: Quantify the effect of CoT training on generalization error.
    - **Mechanism**: The generalization error upper bound is proportional to $\sqrt{\frac{1}{N}[(1-\alpha)D_{KL}(P_{test}^{ID} \| P_{train}) + \alpha D_{KL}(P_{test}^{OOD} \| P_{train})]}$. For ID, $D_{KL} \to 0$ since compositional patterns are shared. For OOD, without CoT the $D_{KL}$ is large ($P(C|X)$ reduces to a uniform prior); with CoT it decomposes into $D_{KL}(P_{test}^{OOD}(C|X) \| P_{train}(C|X)) + \mathbb{E}[D_{KL}(P_{test}^{OOD}(Y|X,C) \| P_{train}(Y|X,C))]$, both of which can be small because simple skills ($P(C|X)$ and $P(Y|X,C)$) are covered during training.
    - **Design Motivation**: Extend the advantage of CoT training from "good in-distribution performance" to "good performance on unseen compositional patterns."

2. **Two-Stage Compositional Reasoning Circuit (Structural Analysis)**:

    - **Function**: Reveal the internal reasoning pathway after CoT training.
    - **Mechanism**: Logit Lens and causal tracing are applied to an 8-layer GPT-2 on two-hop reasoning tasks $(e_1, r_1, r_2) \to e_3$. After CoT training, the model forms a **two-stage circuit**:
        - Stage 1 (shallow layers $0 \to l$): extracts the bridge entity $e_2$ from inputs $e_1, r_1, r_2$, stored in state $E[l, r_2]$.
        - Stage 2 (deep layers $l \to 8$): uses $e_2$ to perform the second-hop inference yielding $e_3$.
    - **Key Finding**: CoT training causes $e_2$ to be extracted at **shallower layers** ($l=3$ for ID), compared to $l=5$ without CoT—earlier extraction leaves more layers available for the second reasoning step.
    - **Design Motivation**: Demonstrate that CoT does not merely prompt the model to "verbalize" intermediate results, but fundamentally restructures internal computation.

3. **Noise Robustness Analysis**:

    - **Function**: Investigate the effect of erroneous reasoning steps in CoT training data.
    - **Mechanism**: Noise at varying rates $\xi$ is injected into intermediate steps of the training data. When $\xi < 0.2$, both ID and OOD generalization are largely unaffected. As noise increases, the generalization error bound grows (consistent with Theorem 3), yet the model continues to function until $\xi \approx 0.4$.
    - **Design Motivation**: Explain why CoT training on DeepSeek-R1's 600K long CoT examples remains effective despite errors—compositional generalization is robust under moderate noise.

### Loss & Training
- **Without CoT**: $\mathcal{L} = \mathbb{E}[\ell(e_3, \mathcal{M}(e_1, r_1, r_2))]$
- **With CoT**: $\mathcal{L} = \mathbb{E}[\ell(e_3, \mathcal{M}(e_1, r_1, r_2, \hat{e}_2)) + \ell(e_2, \mathcal{M}(e_1, r_1, r_2))]$—jointly predicting the bridge entity and the final answer.
- Autoregressive next-token prediction is used (non-teacher-forcing).

## Key Experimental Results

### Main Results (Controlled Setting, 2000 entities × 200 relations)

| Method | ID Accuracy | OOD Accuracy | Steps to Converge |
|--------|-------------|--------------|-------------------|
| Without CoT (grokking) | ~100% (delayed) | ~0% | >1M steps |
| **With CoT** | **~100%** | **~90%+** | **~4000 steps** |

### Ablation Study ($\lambda$ = two-hop/one-hop data ratio, CoT training)

| $\lambda$ | OOD Generalization Speed | Final OOD Accuracy |
|-----------|--------------------------|-------------------|
| 0.001 (very few two-hop samples) | Fastest | ~85% |
| 0.9 | Fast | ~90% |
| 7.2 | Moderate | ~95% |
| 12.6 | Slow | ~95% |

### Key Findings
- **CoT training accelerates convergence by 250×**: ~4,000 steps vs. >1M steps without CoT.
- **CoT enables OOD compositional generalization**: models without CoT remain at 0% even after one million steps, while CoT models reach 90%+ in ~4,000 steps.
- **Less two-hop data paradoxically accelerates OOD generalization**: at $\lambda=0.001$ (minimal two-hop data), OOD generalization is actually faster—analogous to OpenAI O1's RFT, where a small amount of fine-tuning data suffices to elicit reasoning ability.
- **Shallow extraction of intermediate results**: CoT training causes $e_2$ to be extracted by layer 3 (vs. layer 5 without CoT), freeing deeper layers for subsequent reasoning.
- **Two-layer Transformers suffice to learn compositional circuits**: CoT-trained compositional circuits can fully emerge in a 2-layer model.
- **Noise robustness**: generalization is nearly unaffected at $\xi < 0.2$, explaining the tolerance of errors in practical CoT data.

## Highlights & Insights
- **Formalization of "how to think vs. what to think"**: The advantage of CoT is precisely characterized as decomposing $P(Y|X)$ into $P(C|X) \cdot P(Y|X,C)$—CoT teaches the model the reasoning process ($P(C|X)$) rather than the direct answer. This information-theoretic framing is particularly elegant.
- **The finding of shallow intermediate-result extraction** is an important contribution to understanding Transformer reasoning—CoT training fundamentally teaches the model to utilize depth more efficiently, distributing distinct reasoning steps across different layers. This explains why CoT training increases the model's effective depth.
- **The counterintuitive finding that less data accelerates generalization** carries strong practical implications: RFT/SFT does not require large volumes of CoT data; a small set of high-quality CoT examples may elicit generalization more rapidly than a large corpus.

## Limitations & Future Work
- Validation is primarily conducted on synthetic data (entity-relation); experiments on real NLP tasks are relegated to the appendix.
- Only two-hop reasoning is analyzed; the compositional circuit structure for multi-hop ($>3$ steps) settings remains to be studied.
- The information-theoretic bounds are upper bounds and may not be tight—actual generalization may differ from theoretical predictions.
- The behavior of compositional generalization under RL fine-tuning (GRPO/PPO) is not analyzed.
- How "simple learned skills" are defined and identified in real-world tasks remains an open question.

## Related Work & Insights
- **vs. Wang et al. [102]**: Both study compositional reasoning circuits in Transformers, but that work finds systematic circuits emerge only in the ID setting; this paper demonstrates that CoT training extends them to OOD.
- **vs. Feng et al. [17]**: That work proves from an expressivity perspective that CoT increases effective depth; this paper complements it from the perspectives of generalization and internal structure by explaining "how this is realized."
- **vs. COCONUT/CoT2**: These works explore continuous reasoning spaces; the compositional generalization theory presented here can serve as a foundation for understanding such approaches.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ First to formally characterize the generalization mechanism of CoT training from both information-theoretic and internal circuit perspectives.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Synthetic experiments are highly systematic, but validation on real-world tasks is insufficient.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theory and experiments are seamlessly interwoven, with a clear and compelling argumentative chain.
- **Value**: ⭐⭐⭐⭐⭐ Represents a milestone in the theoretical understanding of CoT training and provides direct guidance for RFT/SFT practice.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Training Large Reasoning Models Efficiently via Progressive Thought Encoding](training_large_reasoning_models_efficiently_via_progressive_thought_encoding.md)
- [\[ICLR 2026\] TumorChain: Interleaved Multimodal Chain-of-Thought Reasoning for Traceable Clinical Tumor Analysis](tumorchain_interleaved_multimodal_chain-of-thought_reasoning_for_traceable_clini.md)
- [\[ICML 2026\] On the Generalization Gap in Self-Evolving Language Model Reasoning](../../ICML2026/llm_reasoning/on_the_generalization_gap_in_self-evolving_language_model_reasoning.md)
- [\[ICLR 2026\] CoT-RVS: Zero-Shot Chain-of-Thought Reasoning Segmentation for Videos](cot-rvs_zero-shot_chain-of-thought_reasoning_segmentation_for_videos.md)
- [\[ICLR 2026\] Native Reasoning Models: Training Language Models to Reason on Unverifiable Data](native_reasoning_models_training_language_models_to_reason_on_unverifiable_data.md)

</div>

<!-- RELATED:END -->
