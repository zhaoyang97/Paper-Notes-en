---
title: >-
  [Paper Note] Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs
description: >-
  [CVPR 2026][Multimodal VLM][Multimodal safety] This paper proposes DACO, a framework that constructs a multimodal concept dictionary of 15,000 concepts from WordNet and CC-3M…
tags:
  - "CVPR 2026"
  - "Multimodal VLM"
  - "Multimodal safety"
  - "activation steering"
  - "sparse autoencoders"
  - "concept dictionary"
  - "jailbreak defense"
date: 2026-05-08
content_hash: 3120e258662ee1b3
---

# Dictionary-Aligned Concept Control for Safeguarding Multimodal LLMs

**Conference**: CVPR 2026
**arXiv**: [2604.08846](https://arxiv.org/abs/2604.08846)
**Code**: None
**Area**: Multimodal VLM
**Keywords**: Multimodal safety, activation steering, sparse autoencoders, concept dictionary, jailbreak defense

## TL;DR
This paper proposes DACO, a framework that constructs a multimodal concept dictionary of 15,000 concepts from WordNet and CC-3M, and combines it with sparse autoencoders (SAE) to achieve fine-grained concept control over frozen MLLM activation spaces, significantly improving safety across multiple benchmarks while preserving general capability.

## Background & Motivation

1. **Background**: Multimodal large language models (MLLMs) face safety risks from malicious queries, including textual jailbreaks, visual adversarial attacks, and typographic triggers. Existing safety control strategies include prompt engineering, response filtering, fine-tuning, and the emerging approach of activation steering.
2. **Limitations of Prior Work**: Prompt engineering is fragile under distribution shift; response filtering incurs additional computational overhead; fine-tuning is costly. Although activation steering is flexible, it faces three challenges: (1) non-sparse methods typically handle fewer than 20 concept vectors, resulting in limited coverage; (2) steering intensity is difficult to calibrate — insufficient suppression fails to achieve safety goals while excessive suppression degrades general capability; (3) SAE-based methods lack semantic grounding, as their learned features require expensive probing or manual interpretation.
3. **Key Challenge**: Manually constructed concept vectors offer limited coverage and are often redundant or entangled; SAE-learned dictionaries are expressive but lack semantic annotation. Both approaches have complementary strengths that have not been effectively unified.
4. **Goal**: To construct a framework that jointly leverages a large-scale concept dictionary and SAE for effective and interpretable safety steering in MLLM activation spaces.
5. **Key Insight**: Over 15,000 concepts are extracted from WordNet, and 400K+ image-text stimulus pairs are retrieved from CC-3M and aggregated into a concept vector dictionary; this dictionary is used to initialize SAE training and to automatically annotate the semantics of SAE atoms.
6. **Core Idea**: The concept dictionary provides semantic grounding while the SAE provides expressiveness; their combination enables fine-grained activation control that is simultaneously concept-aware and compositionally decomposable.

## Method

### Overall Architecture
DACO proceeds in four steps: (1) concept names are extracted from WordNet, and CLIP is used to retrieve positive and negative image-text stimulus pairs from CC-3M, forming the DACO-400K dataset; (2) stimulus pairs are passed through the MLLM to extract activations, which are aggregated into a concept vector dictionary $\mathbf{D}_\ell$ via contrastive representation reading; (3) the concept dictionary is used to initialize the SAE decoder before training on CC-3M activations, after which each SAE atom is automatically annotated as "desirable" or "undesirable" via cosine distance; (4) at inference time, the SAE decomposes activations into sparse coefficients, the contributions of undesirable atoms are zeroed out, and desirable atoms are amplified.

### Key Designs

1. **Large-Scale Multimodal Concept Dictionary Construction (DACO-400K)**:
    - **Function**: Provides a semantically grounded set of anchor vectors with broad coverage.
    - **Mechanism**: Approximately 15,000 deduplicated concepts are extracted from WordNet. For each concept, cross-modal relevance with CC-3M image-text pairs is computed using CLIP: $\text{dist}_M(c, \mathbf{x}) = \sqrt{-(\ln s(c, \mathbf{x}_{\text{image}}) + \ln s(c, \mathbf{x}_{\text{text}}))}$, employing geometric aggregation rather than arithmetic averaging to emphasize cross-modal consistency. High-scoring pairs serve as positive stimuli $\mathcal{X}_c^+$ and low-scoring pairs as negative stimuli $\mathcal{X}_c^-$. The concept vector at each layer is obtained via contrastive reading: $\mathbf{d}_{\ell,c} = \mathbb{E}_{\mathbf{x} \in \mathcal{X}_c^+}[\mathbf{z}_\ell] - \mathbb{E}_{\mathbf{x} \in \mathcal{X}_c^-}[\mathbf{z}_\ell]$. An expert MLLM is used to annotate concepts as desirable or undesirable.
    - **Design Motivation**: The 15,000 concepts far exceed the ~20 concepts used in prior methods, providing comprehensive coverage of the activation space geometry. Geometric aggregation ensures that positive stimuli match the concept in both visual and textual modalities, avoiding single-modality bias.

2. **Dictionary-Initialized SAE Training and Automatic Annotation**:
    - **Function**: Obtains more expressive steering vectors and automatically assigns semantic labels.
    - **Mechanism**: Normalized concept vectors are used to initialize the SAE decoder: $\mathbf{W}_{\ell,i}^{\text{dec},(0)} \leftarrow \mathbf{D}_{\ell,i}/\|\mathbf{D}_{\ell,i}\|_2$, after which an L1-SAE or TopK-SAE is trained on CC-3M activations. After training, centroids of the undesirable concept set $\mathcal{K}^-$ and desirable concept set $\mathcal{K}^+$ are computed, and each SAE atom is annotated by thresholding its cosine distance to these centroids (Eq. 9). Single-centroid annotation balances efficiency and effectiveness.
    - **Design Motivation**: SAE learns a superior representational basis from data compared to a handcrafted dictionary, while dictionary initialization resolves the cold-start problem and semantic annotation problem of SAE — yielding semantically interpretable sparse representations.

3. **Compositional Inference-Time Steering**:
    - **Function**: Applies safety steering to frozen MLLM activations at inference time.
    - **Mechanism**: For a target activation $\mathbf{z}_\ell$, sparse coefficients are computed via the SAE encoder: $\mathbf{c}_\ell^* = \sigma(\mathbf{W}_\ell^{\text{enc}} \mathbf{z}_\ell + \mathbf{b}_\ell^{\text{enc}})$. Control coefficients are constructed by negating undesirable atoms (zeroing their contribution), adding a positive constant $\gamma$ to desirable atoms (amplifying safe responses), and setting all others to zero. The modified activation $\hat{\mathbf{z}}_\ell = \mathbf{z}_\ell + \mathbf{W}_\ell^{\text{dec}} \hat{\mathbf{c}}_\ell$ replaces the original activation for continued autoregressive generation.
    - **Design Motivation**: Compared to directly adding or subtracting concept vectors (ActAdd), SAE decomposition more precisely localizes the components requiring modification; compositional steering allows simultaneous removal of harmful concepts and reinforcement of benign ones.

### Loss & Training
SAE training uses standard reconstruction plus sparsity loss (Eq. 3), supporting both L1 regularization and TopK constraint variants. Dictionary initialization substantially improves training convergence (FVE improved by 2–5%). Hyperparameters $\eta$ (annotation threshold) and $\gamma$ (desirable concept amplification strength) are tuned on a validation set.

## Key Experimental Results

### Main Results

**Safety evaluation (Qwen2.5-VL-7B)**:

| Method | MS-R↑ | MS-QG↑ | JBV-R↑ | JBV-QG↑ | Fluency↑ | MMMU↑ |
|--------|-------|--------|--------|--------|---------|-------|
| No Steering | 0.442 | 0.652 | 0.564 | 0.543 | 0.917 | 0.546 |
| Prompting | 0.607 | 0.711 | 0.659 | 0.622 | 0.923 | 0.516 |
| ActAdd | 0.653 | 0.735 | 0.691 | 0.675 | 0.691 | 0.441 |
| MOP | 0.771 | 0.840 | 0.835 | 0.752 | 0.816 | 0.496 |
| **DACO** | **0.990** | **0.984** | **0.903** | **0.841** | **0.905** | **0.521** |

**Inference-time overhead (per token)**:

| Method | Additional Time | Overhead |
|--------|----------------|---------|
| ActAdd | +0.023s | +10.8% |
| MOP | +0.107s | +49.4% |
| **DACO** | **+0.031s** | **+14.6%** |

### Ablation Study

| Configuration | JBV-QG | MMMU | Note |
|---------------|--------|------|------|
| DACO (TopK-SAE, dict. init.) | 0.841 | 0.521 | Full method |
| MOP (sparse coding, no SAE) | 0.752 | 0.496 | SAE more effective than handcrafted dict. |
| SAE random init. | ~0.80 | ~0.51 | Dict. init. improves safety by ~4% |
| η too small (too many atoms labeled harmful) | High | Low | Excessive refusal, hurts general capability |
| η too large (too few atoms labeled harmful) | Low | High | Insufficient safety steering |

### Key Findings
- **DACO consistently and substantially outperforms all baselines across three MLLMs**: on Qwen2.5-VL, safety improves from 0.442 to 0.990 (MS-R) while MMMU drops by only 2.5%.
- **Inference overhead is minimal (+14.6%)**: far below MOP (+49.4%), as SAE encoding requires only a single matrix multiplication rather than iterative sparse solving.
- **SAE atom semantics are interpretable** (Figure 7): SAE decompositions of jailbreak queries show that the nearest concept vectors to highly activated atoms are semantically consistent with query content (e.g., "drugs," "violence"), validating the effectiveness of automatic annotation.

## Highlights & Insights
- The **complementary design of concept dictionary + SAE** is the central innovation: the dictionary provides semantic grounding to resolve SAE's "black-box" problem, while the SAE provides data-driven expressiveness to overcome the limitations of handcrafted dictionaries. This paradigm of "prior knowledge + data learning" is broadly transferable.
- The **DACO-400K dataset** is itself a valuable contribution: activation direction vectors for 15,000 multimodal concepts can support diverse downstream tasks including mechanistic interpretability and concept editing.
- **Geometric cross-modal stimulus retrieval** (Eq. 4) is more principled than arithmetic averaging: requiring simultaneous match in both image and text modalities for a high score, thereby avoiding noisy stimuli where only one modality matches the concept.

## Limitations & Future Work
- The concept dictionary depends on WordNet and CC-3M, and may miss emerging or culturally specific harmful concepts.
- SAE atom annotation uses a single centroid, which may be insufficiently precise for concepts with complex semantic distributions.
- Safety steering may cause over-refusal on MOSSBench (sensitive but legitimate queries requiring an answer).
- Tuning hyperparameters $\eta$ and $\gamma$ requires validation data, and no automated strategy is provided.
- Future work could explore dynamic concept dictionary updates and multi-layer joint steering strategies.

## Related Work & Insights
- **vs. PaCE (Parsimonious Concept Engineering)**: PaCE constructs concepts from synthetic text only; DACO uses real multimodal stimuli, yielding superior performance on MLLMs (MOP is an extension of PaCE to the MLLM setting).
- **vs. ActAdd**: ActAdd directly adds and subtracts a small number of contrastive vectors, lacking fine-grained control. DACO's SAE decomposition precisely localizes the activation components requiring modification.
- **vs. Constitutional AI**: Constitutional AI achieves safety alignment through RLHF during training. DACO operates at inference time, offering greater flexibility and applicability to any frozen model.

## Rating
- **Novelty**: ⭐⭐⭐⭐⭐ The synergistic framework of concept dictionary + SAE is novel; DACO-400K is a significant contribution.
- **Experimental Thoroughness**: ⭐⭐⭐⭐⭐ Three MLLMs, two safety benchmarks, two judges, and general capability evaluation — very comprehensive.
- **Writing Quality**: ⭐⭐⭐⭐ Framework description is clear, though the dense notation requires careful reading.
- **Value**: ⭐⭐⭐⭐⭐ A highly practical MLLM safety solution with low overhead and strong performance.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] LLaVAShield: Safeguarding Multimodal Multi-Turn Dialogues in Vision-Language Models](llavashield_multimodal_multiturn_safety.md)
- [\[CVPR 2026\] Concept-wise Attention for Fine-grained Concept Bottleneck Models](coat_cbm_concept_wise_attention.md)
- [\[CVPR 2026\] Joint-Aligned Latent Action: Towards Scalable VLA Pretraining in the Wild](joint-aligned_latent_action_towards_scalable_vla_pretraining_in_the_wild.md)
- [\[CVPR 2026\] Customized Visual Storytelling with Unified Multimodal LLMs](customized_visual_storytelling_with_unified_multimodal_llms.md)
- [\[CVPR 2026\] PersonaVLM: Long-Term Personalized Multimodal LLMs](personavlm_long_term_personalized_multimodal_llms.md)

</div>

<!-- RELATED:END -->
