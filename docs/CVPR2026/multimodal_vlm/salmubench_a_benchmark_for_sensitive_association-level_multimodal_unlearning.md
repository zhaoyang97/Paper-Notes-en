---
title: >-
  [Paper Note] SALMUBench: A Benchmark for Sensitive Association-Level Multimodal Unlearning
description: >-
  [CVPR2026][Multimodal VLM][machine unlearning] This paper proposes SALMUBench—the first association-level machine unlearning benchmark for CLIP-style models—comprising a 60K synthetic person–sensitive-attribute paired dataset, from-scratch-trained Compromised/Clean model pairs, and a structured holdout evaluation protocol. It is the first work to systematically reveal three failure modes of existing unlearning methods: catastrophic destruction, over-generalized unlearning, and ineffective unlearning.
tags:
  - CVPR2026
  - Multimodal VLM
  - machine unlearning
  - CLIP
  - privacy protection
  - association-level unlearning
  - benchmark
date: 2026-05-08
content_hash: 03168ef1f85a4dfa
---

# SALMUBench: A Benchmark for Sensitive Association-Level Multimodal Unlearning

**Conference**: CVPR2026
**arXiv**: [2603.26316](https://arxiv.org/abs/2603.26316)
**Code**: [cvc-mmu.github.io/salmubench](http://cvc-mmu.github.io/salmubench)
**Area**: Multimodal VLM
**Keywords**: machine unlearning, CLIP, privacy protection, association-level unlearning, benchmark

## TL;DR
This paper proposes SALMUBench—the first association-level machine unlearning benchmark for CLIP-style models—comprising a 60K synthetic person–sensitive-attribute paired dataset, from-scratch-trained Compromised/Clean model pairs, and a structured holdout evaluation protocol. It is the first work to systematically reveal three failure modes of existing unlearning methods: catastrophic destruction, over-generalized unlearning, and ineffective unlearning.

## Background & Motivation
Vision-language models such as CLIP are trained on massive web-crawled data and may inadvertently memorize sensitive personal information (e.g., associating a face with a phone number). The "right to be forgotten" under GDPR requires models to selectively erase learned sensitive associations.

**Limitations of Prior Work**: (1) Unimodal unlearning methods do not transfer well to contrastive learning-based embedding models; (2) existing multimodal unlearning benchmarks (MLLMU-Bench, FIUBench) primarily target VQA evaluation for generative MLLMs and are ill-suited to CLIP's embedding space; (3) existing evaluations inject sensitive knowledge via fine-tuning, making it impossible to isolate unlearning effects from pre-training artifacts; (4) most critically, the prevailing simple forget–retain evaluation framework **cannot detect over-generalized unlearning**—a method may successfully erase the target information while inadvertently erasing related knowledge that should be retained.

## Method

### Overall Architecture
SALMUBench provides a complete unlearning evaluation ecosystem consisting of three core components: (1) the SALMU synthetic dataset; (2) from-scratch-trained Compromised/Clean model pairs; and (3) an evaluation protocol based on structured holdout sets.

### Key Designs

1. **Synthetic Dataset Construction (5-Stage Pipeline)**:

    - Anchor seeding: 1,000 synthetic faces are selected from SFHQ as identity anchors.
    - Identity-preserving image generation: ~100 diverse images per identity are generated using IP-Adapter-FaceID Plus.
    - CLIP filtering and demographic curation: zero-shot annotation and consistency checking yield 774 coherent identities.
    - PII attribute assignment: each identity is assigned culturally consistent yet entirely fictitious and unique names, cities, phone numbers, emails, IBANs, etc.
    - LLM-based annotation diversification: Gemma3-12B rewrites template captions in 5 linguistic variants to ensure diversity.
   The final dataset contains ~60K image–text pairs covering 774 fictitious identities across 65 countries.

2. **From-Scratch-Trained Controlled Experimental Setup**: Two ViT-B/16 CLIP models are trained:

    - **Clean model**: trained exclusively on the retain set (~400M real image–text pairs).
    - **Compromised model**: trained on the retain set plus the sensitive set (60K pairs).
   Both models share the same random seed, architecture, and training configuration (32 epochs, 128 H100s), ensuring that differences arise solely from the presence or absence of sensitive data.

3. **Structured Holdout Evaluation Protocol**: The 774 sensitive identities are split into two groups:

    - **Forget set**: all data visible to the unlearning algorithm (containing forget_identity and forget_association).
    - **holdout_identity**: identities not present in the forget set—used to detect **cross-identity collateral damage**.
    - **holdout_association**: other associations of the same identity (e.g., whether occupation is erased after phone number unlearning)—used to detect **intra-identity collateral damage**.

   This design enables quantitative diagnosis of **over-generalized unlearning** for the first time.

### Evaluation Metrics
**Unlearning Efficacy** (5 metrics):
- RetFail (Retrieval Failure Rate, primary metric): lower MRR is better.
- AssocStr (Association Strength): mean cosine similarity on the forget set.
- ACS (Association Consistency Score): logistic regression accuracy on correctly vs. shuffled paired associations.
- IdZSC (Identity Zero-Shot Classification Accuracy).
- CoreAssoc (Core Association Robustness).

**Utility Preservation** (5 metrics):
- GenKnow (primary metric): ImageNet-1K zero-shot Top-1 accuracy.
- InterIdSim / IntraIdSim: cosine similarity on holdout sets.
- VisIdInt (Visual Identity Integrity), FragSim (Fragile Knowledge Retention).

## Key Experimental Results

### Main Results (5× Budget)

| Method | RetFail ↓ | GenKnow ↑ | InterIdSim | IntraIdSim |
|--------|-----------|-----------|------------|------------|
| Clean (target) | 0.001 | 0.633 | 0.143 | 0.143 |
| Compromised | 0.236 | 0.638 | 0.321 | 0.321 |
| CLIPErase | 0.001 | 0.634 | 0.024 | 0.024 |
| DELETE | 0.001 | 0.632 | 0.023 | 0.023 |
| VLUnlearn | 0.001 | 0.638 | 0.210 | 0.210 |
| Finetuning | 0.003 | 0.638 | 0.209 | 0.209 |
| Neg. Gradient | 0.009 | 0.630 | 0.063 | 0.061 |
| Shuffled Captions | 0.004 | 0.548 | 0.212 | 0.212 |
| Direct Sim. Min. | 0.001 | 0.615 | -0.420 | -0.425 |

### Analysis of Three Failure Modes

| Failure Type | Representative Methods | Characteristics |
|---|---|---|
| Catastrophic Destruction | Shuffled Captions, Direct Sim. Min. | Effective unlearning but significant drop in GenKnow |
| Over-Generalized Unlearning | DELETE, CLIPErase | Precise unlearning and preserved GenKnow, but severe damage to holdout sets |
| Ineffective Unlearning | Generic Captions | Minimal collateral damage but failure to effectively unlearn |

### Key Findings
- **No method simultaneously avoids all three failure modes**—this is the central open problem in this field.
- Utility-efficient unlearning (>99% leakage reduction + <1% GenKnow drop) is achievable, but existing methods that attain this (DELETE, CLIPErase) do so through over-generalization.
- When AssocStr is pushed below the Clean model baseline (0.142), over-generalization is triggered—methods over-correct and erase related, unseen associations.
- Simple forget–retain evaluation is completely blind to over-generalized unlearning.

## Highlights & Insights
- The **structured holdout evaluation design** is the paper's greatest contribution—the holdout_identity and holdout_association splits are elegantly designed and enable quantification of over-generalized unlearning for the first time.
- Training two full CLIP models from scratch (400M data, 128 H100s) as controlled experiments is costly but provides the cleanest possible evaluation baseline.
- The synthetic data pipeline is well-engineered: IP-Adapter identity preservation + CLIP filtering + LLM paraphrasing + Faker-generated PII, offering strong reusability.
- The **taxonomy of three failure modes** provides clear research objectives for future methods: unlearning efficacy, utility preservation, and avoidance of over-generalization must be jointly addressed.

## Limitations & Future Work
- The benchmark targets only CLIP dual-encoders; evaluating how sensitive information propagates through diffusion models that use CLIP as a backbone is a natural extension.
- Coverage is limited to structured PII (names, phone numbers, etc.); generalization to implicit sensitive concepts (artistic styles, political affiliations, etc.) remains unknown.
- The benchmark lacks **recoverability diagnostics**—whether a post-unlearning model can rapidly relearn erased information via fine-tuning is not assessed.
- Although the synthetic faces pass KS tests for domain consistency, the validation sample of 100 real portraits is limited in size.

## Related Work & Insights
- **vs. MultiDelete / CLIPErase**: existing methods do not target personal privacy information specifically, and inject sensitive data via fine-tuning, resulting in insufficiently rigorous evaluation.
- **vs. TOFU / FIUBench**: VQA-based evaluations for generative MLLMs are not applicable to embedding-space evaluation of CLIP.
- **Insight**: The phenomenon of over-generalized unlearning may also manifest in knowledge editing/unlearning for LLMs, warranting cross-domain investigation.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First CLIP association-level unlearning benchmark; structured holdout evaluation design is highly innovative.
- Experimental Thoroughness: ⭐⭐⭐⭐ Nine baseline methods, multi-budget comparisons, and comprehensive three-failure-mode analysis; however, main experiments are limited to a single ViT-B/16 architecture.
- Writing Quality: ⭐⭐⭐⭐⭐ Dataset construction, evaluation protocol, and failure mode taxonomy are all presented with clarity and rigor.
- Value: ⭐⭐⭐⭐⭐ Establishes a new standard for multimodal machine unlearning; all data, models, and evaluation code are publicly released.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[CVPR 2026\] Mixture of States (MoS): Routing Token-Level Dynamics for Multimodal Generation](mos_mixture_of_states_multimodal_generation.md)
- [\[AAAI 2026\] Cross-Modal Unlearning via Influential Neuron Path Editing in Multimodal Large Language Models](../../AAAI2026/multimodal_vlm/cross-modal_unlearning_via_influential_neuron_path_editing_i.md)
- [\[CVPR 2026\] Empowering Semantic-Sensitive Underwater Image Enhancement with VLM](empowering_semanticsensitive_underwater_image_enha.md)
- [\[CVPR 2026\] Training High-Level Schedulers with Execution-Feedback Reinforcement Learning for Long-Horizon GUI Automation](training_high-level_schedulers_with_execution-feedback_reinforcement_learning_fo.md)
- [\[ACL 2026\] VLA-Forget: Vision-Language-Action Unlearning for Embodied Foundation Models](../../ACL2026/multimodal_vlm/vla-forget_vision-language-action_unlearning_for_embodied_foundation_models.md)

</div>

<!-- RELATED:END -->
