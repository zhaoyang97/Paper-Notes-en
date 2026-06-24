---
title: >-
  [Paper Note] Why LVLMs Are More Prone to Hallucinations in Longer Responses: The Role of Context
description: >-
  [ICCV 2025][Hallucination Detection][Hallucination Mitigation] This work deeply investigates the root cause of frequent hallucinations in LVLM long-form generation—demonstrating that it is not the length itself but the demands of contextual coherence and completeness that drive the model to extrapolate and hallucinate. Based on this insight, the authors propose HalTrapper, an "induce-detect-suppress" three-stage framework.
tags:
  - "ICCV 2025"
  - "Hallucination Detection"
  - "Hallucination Mitigation"
  - "Long-form Generation"
  - "Contextural Analysis"
  - "Contrastive Decoding"
  - "Large Vision-Language Models"
date: 2026-05-08
content_hash: 3a37eddac1c1cbb4
---

# Why LVLMs Are More Prone to Hallucinations in Longer Responses: The Role of Context

**Conference**: ICCV 2025  
**arXiv**: [2510.20229](https://arxiv.org/abs/2510.20229)  
**Code**: [GitHub](https://github.com/SooLab/HalTrapper)  
**Area**: Hallucination Detection  
**Keywords**: Hallucination Mitigation, Long-form Generation, Contextural Analysis, Contrastive Decoding, Large Vision-Language Models

## TL;DR

This work deeply investigates the root cause of frequent hallucinations in LVLM long-form generation—demonstrating that it is not the length itself but the demands of contextual coherence and completeness that drive the model to extrapolate and hallucinate. Based on this insight, the authors propose HalTrapper, an "induce-detect-suppress" three-stage framework.

## Background & Motivation

LVLMs exhibit a significant increase in hallucinations when generating long, free-form responses, a well-known phenomenon that remains poorly understood:

- **Superficial Phenomenon**: The frequency of hallucinated objects is positively correlated with their position in the output sequence (appearing later in the sequence increases the hallucination likelihood).
- **Traditional Attribution**: Prior works simply attribute this to cumulative errors and increasing uncertainty caused by sequence length in autoregressive generation.
- **Key Challenge**: Is the increase in hallucinations truly caused by length-induced cumulative errors, or is there a deeper underlying mechanism?

Through innovative context manipulation experiments (cropping images, enriching textual prompts), the authors find that modifying the context significantly shifts the positions of hallucinations (making them appear earlier). This challenges the simplistic attribution of "length = hallucination," revealing that **context is the deep-seated driver beneath the surface**.

## Method

### Overall Architecture

HalTrapper is an "induce-detect-suppress" three-stage framework comprising two complementary pathways:
- **Internal Grounding (IG)**: Attention similarity detection based on contextual coherence.
- **External Expansion (EE)**: Cross-prompt consistency detection based on contextual completeness.

### Key Designs

1. **Contextual Coherence Analysis**:

    - **Hypothesis**: Contextual coherence creates a contradiction in image attention—the model needs to attend to already-described regions to maintain consistency, while simultaneously shifting towards new regions to avoid repetition. This tension leads to scattered attention and hallucinations.
    - **Validation**: The image attention similarity between object pairs in the same response is calculated. The attention similarities for the set of hallucinated objects $\mathcal{H}$ and non-hallucinated objects $\mathcal{N}$ are defined as:
    $S_{\mathcal{H}} = \{\text{sim}(A_{s,i}, A_{s,j}) \mid o_{s,i}, o_{s,j} \in \mathcal{H}\}$
    $S_{\mathcal{N}} = \{\text{sim}(A_{s,i}, A_{s,j}) \mid o_{s,i}, o_{s,j} \in \mathcal{N}\}$
   The results show that **the attention similarity among hallucinated objects is significantly higher than that among real objects**, indicating that hallucinated objects share a similar scattered attention pattern.

2. **Contextual Completeness Analysis**:

    - **Hypothesis (a) Generation Mechanism**: When a response already contains correctly identified objects but remains incomplete in information or structure, the model compensates by imagining/fabricating details—leading to hallucinations.
    - **Hypothesis (b) Inherent Tendency**: Extrapolated hallucinations depend on multimodal context, especially visual inputs.
    - **Validation**: Image descriptions are incrementally added to the prompt, and the PoScore (relative position score) of hallucinations is observed. The results find that **the more complete the context, the earlier hallucinations appear**. Furthermore, using five different prompts for the same image, approximately 70% of the hallucinated objects reappear, demonstrating that hallucinations follow inherent patterns.

3. **Induction Phase**:

    - **IG Induction**: After the model completes its response, the EOS token is replaced with "There is also" to force the model to continue generating. Since completeness has already been satisfied, the newly generated objects are highly likely to be hallucinations, which serve as the reference anchor $o_s^{ref}$.
    - **EE Induction**: Prompts such as "Please imagine what object might be outside the frame" are utilized to induce the model to imagine content beyond the image frame. Existent objects are filtered out via a "reason first, imagine later" prompt design.

4. **Detection Phase**:

    - **IG Detection**: The attention similarity scores between the induced hallucination reference object and the previous objects are calculated:
    $\text{IGScore}_{s,i} = \text{sim}(A_s^{ref}, A_{s,i})$
    $S_{IG} = \{o_{s,i} \mid \text{IGScore}_{s,i} > \theta_{IG}\}$
    - **EE Detection**: Multidirectional prompt consistency scoring is used:
    $\text{EEScore}_{s,i} = \sum_{d \in \mathcal{D}} [\mathbb{1}(o_{s,i} \in S_{I,d}) - \mathbb{1}(o_{s,i} \in S_{R,d})]$
    - Merged detection results: $S_{induction} = S_{IG} \cup S_{EE}$

5. **Suppression Phase - Contrastive Contextual Decoding (CCD)**:
   The detected latent hallucinated objects are encoded into **Contrastive Contextual Tokens (CCT)** $x_{cct}$, serving as an additional input for the contrastive branch:
    $p_{ccd}(y_i|v,x,y_{<i}) = \text{softmax}[(1+\alpha)\text{logit}_\theta(y_i|v,x,y_{<i}) - \alpha \cdot \text{logit}_\theta(y_i|v,x,x_{cct},y_{<i})]$
   CCT naturally increases the probability of hallucinated objects in the contrastive branch, research has shown that this effectively reduces their generation probability in the original branch.

### Loss & Training

HalTrapper is a **fully training-free** method that does not modify model parameters, intervening solely during the decoding phase. Hyperparameters used are $\alpha=1.0$ and $\beta=0.1$.

## Key Experimental Results

### Main Results (CHAIR Metrics, LLaVA v1.5 7B)

| Decoding Strategy | Method | CHAIR_S ↓ | CHAIR_I ↓ | Precision ↑ | F1 ↑ |
|---------|------|-----------|-----------|-------------|------|
| Greedy | Vanilla | 52.2 | 14.6 | 73.7 | 76.9 |
| Greedy | ICD | 51.4 | 14.7 | 73.4 | 77.0 |
| Greedy | CODE | 50.0 | 13.7 | 75.8 | 76.4 |
| Greedy | **HalTrapper** | **41.6** | **11.9** | **78.7** | **79.4** |
| Nucleus | Vanilla | 58.6 | 18.8 | 68.1 | 72.0 |
| Nucleus | VCD | 58.2 | 16.9 | 70.8 | 74.6 |
| Nucleus | **HalTrapper** | **48.6** | **14.5** | **74.6** | **76.1** |

### Ablation Study (Hallucination Detection Performance)

| Model | Detection Method | AUROC | TPR@5%FPR | F1_max | Accuracy |
|------|---------|-------|-----------|--------|----------|
| LLaVA v1.5 | PoScore | 70.7 | 4.3 | 38.3 | 70.7 |
| LLaVA v1.5 | Top Logit | 64.0 | 13.0 | 32.2 | 61.9 |
| LLaVA v1.5 | Logits' Entropy | 67.7 | 16.6 | 36.6 | 71.4 |
| LLaVA v1.5 | Image Attn. Ratio | 44.9 | 6.0 | 27.3 | 32.0 |
| LLaVA v1.5 | **IG Score** | **82.3** | **43.3** | **54.8** | **86.3** |
| LLaVA v1.5 | EE Score | 77.5 | - | 46.1 | 72.9 |

### Key Findings

- The AUROC of IG Score reaches 82.3, which is 11.6 percentage points higher than the best baseline, PoScore.
- HalTrapper under Greedy decoding reduces CHAIR_S from 52.2 to 41.6 (↓10.6), and CHAIR_I from 14.6 to 11.9 (↓2.7).
- The distribution of detected hallucination positions is highly consistent with the distribution of actual hallucinations, validating the context hypothesis.
- On the AMBER benchmark, HalTrapper yields significant improvements across three models: LLaVA v1.5, Qwen2 VL, and Janus Pro.
- **Key Insight**: Hallucinations are not directly caused by length, but rather driven by context—their occurrence position can be controlled through context manipulation.

## Highlights & Insights

- **Deep Causal Analysis**: Rather than being satisfied with the surface-level correlation of "length -> hallucination," this work delves deeply into the causal mechanism of "context -> hallucination."
- **Elegant Experimental Design**: Through two complementary context manipulation operations—cropping images and enriching prompts—the role of context is precisely uncovered.
- **Discovery of Attention Similarity**: Hallucinated objects share highly similar scattered attention patterns, a finding that possesses independent research value.
- **Hallucination Predictability**: The recurrence rate of hallucinations across different prompts is up to 70%, indicating that hallucinations are not random but follow predictable patterns.
- **Hypothesis-Driven Method Design**: A complete closed loop is constructed starting from hypotheses -> statistical validation -> methodological design -> back-validation of hypotheses via empirical efficacy.

## Limitations & Future Work

- The IG method relies on access to attention maps, making it inapplicable to certain closed-source models or optimized attention implementations.
- The EE method exhibits limited effectiveness on models with weaker instruction-following capabilities (e.g., MiniGPT-4).
- The construction of CCT depends on the quality of the detection phase; false positives might suppress correct content.
- It focuses solely on object-level hallucinations; attribute-level and relationship-level hallucinations are not addressed in depth.
- It increases computational overhead during inference due to the required induction and detection processes.

## Related Work & Insights

- Unlike contrastive decoding methods such as VCD, ICD, or OPERA, HalTrapper constructs the contrastive branch using the detected hallucinated objects (instead of perturbed inputs).
- A new analytical perspective is provided for LVLM hallucination research: research attention should focus on the impact of context rather than solely on sequence length.
- Attention similarity analysis may inspire novel hallucination detection metrics.

## Rating

- **Novelty**: ⭐⭐⭐⭐⭐ Explaining hallucinations from the perspective of contextual coherence and completeness is a brand-new viewpoint, and the closed-loop design of hypothesis validation is excellent.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Solid experiments are conducted on both detection and suppression, but the evaluation is primarily on COCO; dataset diversity could be further enhanced.
- **Writing Quality**: ⭐⭐⭐⭐⭐ The narrative structure is exceptionally clear (phenomenon -> hypothesis -> validation -> application -> re-validation), with elegant and highly informative figures and tables.
- **Value**: ⭐⭐⭐⭐⭐ The insights into hallucination mechanisms offer long-term research value, providing not only a method but also a deeper understanding.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity](../../NeurIPS2025/hallucination/glsim_detecting_object_hallucinations_in_lvlms_via_globalloc.md)
- [\[ACL 2025\] Stochastic Chameleons: Irrelevant Context Hallucinations Reveal Class-Based (Mis)Generalization in LLMs](../../ACL2025/hallucination/stochastic_chameleons_irrelevant_context_hallucinations_reveal_class-based_misge.md)
- [\[ICCV 2025\] ONLY: One-Layer Intervention Sufficiently Mitigates Hallucinations in Large Vision-Language Models](only_onelayer_intervention_sufficiently_mitigates_hallucinat.md)
- [\[ACL 2025\] Cracking the Code of Hallucination in LVLMs with Vision-aware Head Divergence](../../ACL2025/hallucination/cracking_hallucination_vhd.md)
- [\[ICLR 2026\] LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals](../../ICLR2026/hallucination/lumina_detecting_hallucinations_in_rag_system_with_context-knowledge_signals.md)

</div>

<!-- RELATED:END -->
