---
title: >-
  [Paper Note] Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization
description: >-
  [NeurIPS 2025][LLM Alignment][DPO] This paper proposes SymMPO (Symmetric Multimodal Preference Optimization), which addresses two key limitations of existing vision-augmented DPO methods—namely…
tags:
  - "NeurIPS 2025"
  - "LLM Alignment"
  - "DPO"
  - "multimodal hallucination"
  - "preference optimization"
  - "visual understanding"
  - "symmetric preference learning"
date: 2026-05-08
content_hash: 70d1d41105247800
---

# Mitigating Hallucination Through Theory-Consistent Symmetric Multimodal Preference Optimization

**Conference**: NeurIPS 2025
**arXiv**: [2506.11712](https://arxiv.org/abs/2506.11712)
**Code**: [https://github.com/Liuwq-bit/SymMPO](https://github.com/Liuwq-bit/SymMPO)
**Area**: LLM Alignment
**Keywords**: DPO, multimodal hallucination, preference optimization, visual understanding, symmetric preference learning

## TL;DR

This paper proposes SymMPO (Symmetric Multimodal Preference Optimization), which addresses two key limitations of existing vision-augmented DPO methods—namely, theoretically unsound objective functions and indirect preference supervision—through symmetric paired preference learning over contrastive images and preference margin consistency regularization. Consistent performance gains are achieved across five hallucination benchmarks.

## Background & Motivation

Multimodal large language models (MLLMs) perform well on tasks such as visual question answering and image captioning, yet suffer severely from **hallucination**—generating content inconsistent with the input image. DPO has been widely adopted to mitigate MLLM hallucination. Existing methods typically comprise two components:

**Response-oriented preference learning**: comparing a preferred response $y_w$ and a dispreferred response $y_l$ under the same input.

**Vision-oriented preference learning** (optional): pairing contrastive images $(m_w, m_l)$ with the same response $y_w$ to enhance the model's attention to visual inputs.

Through careful theoretical analysis, the authors identify two critical flaws in existing approaches:

**Flaw 1: Unsound objective function.** In vision-oriented preference learning, contrastive images $m_w \neq m_l$ mean that the partition functions $Z(m_w, x)$ and $Z(m_l, x)$ **cannot** be directly cancelled. Existing methods assume this cancellation is valid, which is inconsistent with the theoretical derivation of standard DPO. A detailed mathematical proof is provided in Appendix B.

**Flaw 2: Indirect preference supervision.** Existing vision-oriented methods use triplets of contrastive images paired with the same response, which fundamentally relies on image contrast rather than response contrast. This deviates from the core principle of DPO—establishing preference relations through paired responses—and yields limited improvement in visual understanding.

## Method

### Overall Architecture

The complete loss function of SymMPO is:

$$\mathcal{L}_{SymMPO} = \mathcal{L}_{DPO_m} + \lambda \mathcal{L}_{Pair} + \gamma \mathcal{L}_{Margin} + \eta \mathcal{L}_{AncPO}$$

The four components are responsible for response quality assurance, symmetric paired preference learning, preference margin consistency regularization, and anchor preference regularization, respectively.

### Key Designs

**1. Symmetric Paired Preference Learning ($\mathcal{L}_{Pair}$)**

The core innovation lies in generating an optimal response $y_{w}'$ for the contrastive image $m'$. Since $m'$ and $m$ are highly similar (selected via CLIP similarity), their optimal responses are also similar yet differ subtly, naturally forming hard negatives.

Symmetric preference modeling:

$$P_{BT}(y_w \succ y_{w}'|m,x) \wedge P_{BT}(y_{w}' \succ y_w|m',x)$$

That is, $y_w$ is preferred over $y_{w}'$ given image $m$, while $y_{w}'$ is preferred over $y_w$ given image $m'$. Since response pairs rather than image pairs serve as preference supervision, the partition functions cancel naturally, and the objective strictly conforms to the standard DPO derivation.

**2. Preference Margin Consistency Regularization ($\mathcal{L}_{Margin}$)**

The Bradley-Terry model only constrains the ordinal preference relation. However, for highly similar inputs $(m,x)$ and $(m',x)$, the preference margins should also be consistent. This motivates:

$$\mathcal{L}_{Margin} = \mathbb{E}\left(\Delta(m,x,y_w,y_{w}') - \Delta(m',x,y_{w}',y_w)\right)^2$$

where $\Delta(m,x,y_w,y_{w}') = r(m,x,y_w) - r(m,x,y_{w}')$, ensuring that the magnitude of preference gaps in both directions remains consistent.

**3. Anchor Preference Regularization ($\mathcal{L}_{AncPO}$)**

This component prevents the model from widening the preference gap by reducing the probability of preferred responses:

$$\mathcal{L}_{AncPO} = -\mathbb{E}\left[\log\sigma\left(\beta\log\frac{\pi_\theta(y_w|m,x)}{\pi_{ref}(y_w|m,x)} - \delta\right) + \log\sigma\left(\beta\log\frac{\pi_\theta(y_{w}'|m',x)}{\pi_{ref}(y_{w}'|m',x)} - \delta\right)\right]$$

**4. Low-Cost Preference Data Construction Pipeline**

A four-stage Caption-Anchored Claim Extraction-and-Rewriting pipeline is proposed:
1. An open-source MLLM (Qwen2.5-VL-32B) generates image captions.
2. The reference model samples multiple responses.
3. An LLM (DeepSeek-V3) compares responses against captions and extracts consistent/inconsistent claims.
4. The LLM rewrites claims into positive/negative responses.

Contrastive images are retrieved via CLIP nearest-neighbor search rather than conventional image transformations, preserving semantic similarity.

### Loss & Training

Hyperparameter settings: $\beta=0.1$, $\delta=0$, $\lambda=0.5$, $\gamma=1e-4$, $\eta=1.0$. Training runs for 2 epochs with a learning rate of 5e-6, batch size of 64, on 4 × NVIDIA A100-40GB GPUs. Training data comes from 21.4k image–prompt pairs from TPO.

## Key Experimental Results

### Main Results

Controlled comparison on LLaVA-1.5-7B under identical data and training conditions (DPO vs. mDPO vs. SymMPO):

| Method | HallusionBench aAcc↑ | MMHal Score↑ | AMBER Acc↑ | AMBER F1↑ | MMStar↑ |
|--------|---------------------|--------------|-----------|----------|---------|
| DPO | 40.21 | 2.44 | 71.3 | 82.6 | 33.4 |
| mDPO | 42.78 | 2.71 | 80.6 | 86.3 | 34.2 |
| **SymMPO** | **44.28** | **2.89** | **82.6** | **87.7** | **34.8** |

SymMPO maintains its advantage on LLaVA-1.5-13B:

| Method | HallusionBench aAcc↑ | MMHal Score↑ | AMBER Acc↑ | AMBER F1↑ | MMStar↑ |
|--------|---------------------|--------------|-----------|----------|---------|
| DPO | 39.50 | 2.65 | 69.2 | 84.6 | 33.0 |
| mDPO | 39.85 | 2.93 | 83.8 | 88.8 | 35.0 |
| **SymMPO** | **44.55** | **3.01** | **84.9** | **89.1** | **35.2** |

### Ablation Study

Component ablation on LLaVA-1.5-7B:

| Variant | HallusionBench aAcc↑ | MMHal Score↑ | AMBER Acc↑ | MMStar↑ |
|---------|---------------------|--------------|-----------|---------|
| SymMPO (full) | 44.28 | 2.89 | 82.6 | 34.8 |
| w/o $\mathcal{L}_{Pair}$ | 43.22 | 2.53 | 81.7 | 33.8 |
| w/o $\mathcal{L}_{Margin}$ | 44.46 | 2.40 | 82.0 | 34.5 |
| w/o $\mathcal{L}_{AncPO}$ | 40.83 | 2.39 | 79.5 | 36.2 |

**Contrastive image type experiments** evaluate five strategies (Similar / Black / Cropped / Noisy / Synthetic), with the following findings:
- SymMPO outperforms mDPO under nearly all image types (except Black).
- Similar, Noisy, and Synthetic images yield better results than Black and Cropped, as the former better preserve semantic similarity.

### Key Findings

1. Paired preference learning ($\mathcal{L}_{Pair}$) contributes most to overall performance; removing it reduces MMHal Score by 0.36.
2. Anchor regularization is critical for HallusionBench (removing it drops aAcc by 3.45 points).
3. Both SymMPO and mDPO underperform DPO on Object-HalBench, likely because the data construction pipeline favors holistic scene descriptions over fine-grained visual detail.
4. CLIP nearest-neighbor contrastive images are more effective than conventional noise injection or cropping strategies.

## Highlights & Insights

1. **Theoretical rigor**: The paper provides a thorough analysis of the role of partition functions in multimodal DPO, identifying the overlooked flaw of directly cancelling $Z(m_w,x)$ and $Z(m_l,x)$ in prior work and proposing a theoretically sound alternative.
2. **Symmetric design naturally resolves the partition function issue**: By ensuring each direction uses the same multimodal input paired with different responses, partition functions cancel naturally.
3. **Quantifying preference margins**: Beyond conventional ordinal preference ($y_w \succ y_l$), a consistency constraint on the magnitude of the preference gap is introduced.
4. **Practical data construction pipeline**: The approach avoids costly GPT-4V API calls by leveraging a combination of open-source models and DeepSeek-V3.

## Limitations & Future Work

1. **Limited fine-grained visual understanding**: Suboptimal performance on Object-HalBench indicates that the data construction pipeline should incorporate greater emphasis on detailed visual descriptions.
2. **Additional computational overhead**: Constructing optimal responses for contrastive images increases data preparation costs.
3. Evaluation is conducted solely on the LLaVA-1.5 architecture; stronger base models (e.g., Qwen-VL, InternVL) remain untested.
4. CLIP nearest-neighbor retrieval for contrastive images may introduce selection bias and sensitivity to dataset distribution.

## Related Work & Insights

- **mDPO** (Wang et al., 2024): Introduces vision-oriented contrastive learning but with an unsound objective. SymMPO directly corrects its theoretical flaw.
- **RLAIF-V** (Yu et al., 2024): Employs disentangled candidate response generation; SymMPO offers a more efficient data construction approach.
- **TPO** (He et al., 2024): A subject-level self-correction paradigm targeting a different aspect from SymMPO.
- **OPA-DPO** (Yang et al., 2025): Proposes adaptive exploration–exploitation balancing but still relies on indirect preference supervision.
- The symmetric design of SymMPO is generalizable to other preference learning scenarios requiring multi-perspective contrastive learning.

## Rating

- Novelty: ⭐⭐⭐⭐ Symmetric preference learning and margin consistency regularization represent valuable contributions; theoretical analysis reveals previously overlooked issues.
- Experimental Thoroughness: ⭐⭐⭐⭐ Five benchmarks, two model scales, complete ablation study, and contrastive image type analysis.
- Writing Quality: ⭐⭐⭐⭐ Problem formulation is clear and mathematical derivations are rigorous, though notation is dense in places.
- Value: ⭐⭐⭐⭐ Makes substantive contributions to the theoretical foundations of multimodal preference optimization with strong methodological generalizability.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Mitigating Mismatch within Reference-based Preference Optimization](../../ICLR2026/llm_alignment/mitigating_mismatch_within_reference-based_preference_optimization.md)
- [\[NeurIPS 2025\] Preference Optimization by Estimating the Ratio of the Data Distribution](preference_optimization_by_estimating_the_ratio_of_the_data_distribution.md)
- [\[ICCV 2025\] MagicID: Hybrid Preference Optimization for ID-Consistent and Dynamic-Preserved Video Customization](../../ICCV2025/llm_alignment/magicid_hybrid_preference_optimization_for_id-consistent_and_dynamic-preserved_v.md)
- [\[NeurIPS 2025\] g-DPO: Scalable Preference Optimization for Protein Language Models](g-dpo_scalable_preference_optimization_for_protein_language_models.md)
- [\[NeurIPS 2025\] On Extending Direct Preference Optimization to Accommodate Ties](on_extending_direct_preference_optimization_to_accommodate_ties.md)

</div>

<!-- RELATED:END -->
