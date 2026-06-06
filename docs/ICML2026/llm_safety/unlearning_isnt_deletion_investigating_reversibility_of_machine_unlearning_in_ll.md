---
title: >-
  [Paper Note] Forgetting is Not Erasing: A Survey of Reversibility in Large Language Model Machine Unlearning
description: >-
  [ICML 2026][LLM Safety][Machine Unlearning] This paper systematically analyzes the reversibility of LLM unlearning using representation-level diagnostic tools—finding that many unlearning methods merely suppress rather t…
tags:
  - "ICML 2026"
  - "LLM Safety"
  - "Machine Unlearning"
  - "Reversibility"
  - "Representational Analysis"
  - "LLM Security"
  - "Privacy"
date: 2026-05-08
content_hash: 72ac80caba11fe58
---

# Forgetting is Not Erasing: A Survey of Reversibility in Large Language Model Machine Unlearning

**Conference**: ICML 2026  
**arXiv**: [2505.16831](https://arxiv.org/abs/2505.16831)  
**Code**: https://github.com/XiaoyuXU1/Representational_Analysis_Tools  
**Area**: LLM Security / Privacy Protection  
**Keywords**: Machine Unlearning, Reversibility, Representational Analysis, LLM Security, Privacy

## TL;DR
This paper systematically analyzes the reversibility of LLM unlearning using representation-level diagnostic tools—finding that many unlearning methods merely suppress rather than truly delete information, and proposes a four-tier unlearning taxonomy to distinguish true information erasure from surface-level performance degradation.

## Background & Motivation

**Limitations of Prior Work**: Current LLM unlearning methods primarily use task-level metrics (accuracy, perplexity) for evaluation, but these metrics are deceptive—even when a model appears to "forget," its original behavior can be rapidly recovered via minimal fine-tuning, suggesting information is only suppressed rather than truly deleted.

**Key Challenge**: The evaluation flaw lies in the inability to distinguish true information erasure from reversible surface performance collapse. Current evaluation frameworks overlook representation-level changes, leading to false unlearning claims.

**Goal**: Establish a representation-level unlearning evaluation framework to uncover the intrinsic mechanisms of unlearning methods and distinguish true information deletion from suppression.

**Key Insight**: Starting from two dimensions—reversibility (whether forgotten information can be recovered) and catastrophic nature (collateral damage to retained knowledge)—this work introduces tools like PCA similarity, CKA, and Fisher information to systematically analyze representation dynamics.

## Method

### Overall Architecture
This paper proposes a unified diagnostic toolkit for unlearning evaluation, containing four complementary representation analysis tools—(1) **PCA Similarity and Shift**: measures directional alignment and translational drift of feature subspaces; (2) **Centered Kernel Alignment (CKA)**: evaluates the preservation of activation subspaces; (3) **Fisher Information Matrix (FIM)**: tracks parameter sensitivity changes in the loss landscape; (4) **Mean PCA Distance**: quantifies the degree of representation drift.

### Key Designs

1. **Four-tier Unlearning Taxonomy**:

    - Function: Categorizes unlearning methods into four types based on reversibility and catastrophic nature—Reversible-Non-catastrophic (Target), Reversible-Catastrophic, Irreversible-Catastrophic, and Irreversible-Non-catastrophic (Ideal but difficult to achieve).
    - Mechanism: Defines performance degradation $\Delta_u(\mathcal{T}) = E(\theta_0, \mathcal{T}) - E(\theta_u, \mathcal{T})$ and performance change after recovery $\Delta_r(\mathcal{T})$, validating the true nature of forgetting through relearning probes.
    - Design Motivation: Task-level metrics cannot reveal the essential mechanism of forgetting; controlled relearning experiments are necessary to detect whether information is truly deleted.

2. **Representational Diagnostic Toolkit**:

    - Function: Jointly captures feature geometry, activation subspace preservation, and parameter sensitivity.
    - Mechanism: Integrates geometric perspective (PCA), subspace perspective (CKA), and optimization perspective (FIM) to verify whether fundamental changes occur in representations from multiple angles.
    - Design Motivation: A single metric might be misleading; multiple tools used jointly can more accurately determine the true state of representations.

3. **Restricted Relearning Protocol**:

    - Function: Probes whether forgotten knowledge still exists latently through limited-budget fine-tuning, where the budget equals the size of the forget set.
    - Mechanism: Uses three types of data sources (forget set, domain-related retain set, unrelated data) for relearning, comparing sample efficiency to judge recoverability.
    - Design Motivation: Full retraining is too costly; restricted relearning is a low-cost, efficient means of probing reversibility.

## Key Experimental Results

### Main Results

| Unlearning Method | Forget Accuracy ↓ | Retain Accuracy ↓ | Reversibility | Catastrophic | Classification |
|---------|------------|-----------|------|------|------|
| GA | 13.5-20.7% | 11.5-16.0% | ✓ | ✓ | Reversible-Catastrophic |
| GA+GD | 3.8-15.7% | 0.9-4.3% | ✓ | ✗ | Reversible-Non-catastrophic |
| GA+KL | 7.9-12.7% | 7.0-12.8% | ✓ | ✓ | Reversible-Catastrophic |
| NPO | 2.7-4.3% | 0.8-2.9% | ✓ | ✗ | Reversible-Non-catastrophic |
| NPO+KL | 2.5-4.1% | 0.7-6.3% | ✓ | ✗ | Reversible-Non-catastrophic |
| RLabel | 1.2-4.6% | 0.8-3.4% | ✓ | ✗ | Reversible-Non-catastrophic |

### Relearning Recovery Efficiency

| Data Source Type | Sample Requirement | Recovery Speed | Final Performance | Remarks |
|---------|---------|--------|--------|------|
| Forget set itself | 100% | Fastest | Near original | Worst-case scenario |
| Domain-related retain set | 150-200% | Medium | Partial recovery | Realistic scenario |
| Unrelated data | 300%+ | Slowest | Limited recovery | Robustness test |

### Key Findings
- All six standard methods exhibit reversibility under single-session unlearning, but only GA+GD, NPO variants, and RLabel achieve non-catastrophicity.
- Recovery strategies without parameter updates, such as prompt attacks, jailbreaking, and quantization, fail completely, indicating that representations are truly altered after unlearning.
- Sample efficiency analysis reveals heterogeneous recovery characteristics across different data sources.
- In sequential unlearning scenarios, Reversible-Catastrophic methods lead to irreversible collapse of retained knowledge.

## Highlights & Insights
- **Innovative combination of representation tools**: First to jointly use PCA, CKA, and FIM to diagnose unlearning.
- **Relearning as a universal probe**: Standardizes relearning as a reversibility testing method, formalizing a new paradigm for unlearning evaluation.
- **Clarity of the four-tier taxonomy**: Clearly characterizes essential differences in unlearning through the orthogonal decomposition of reversibility and catastrophicity.

## Limitations & Future Work
- Computational cost—representation analysis requires large-scale computation with limited scalability on ultra-large models.
- Ambiguity of reversibility thresholds—no explicit threshold is provided to determine when "basic recovery" is achieved.
- Irreversible-Non-catastrophic unlearning remains hard to achieve—a case was identified, but no systematic algorithm was proposed.

## Related Work & Insights
- **vs mechanistic interpretability work**: This paper diagnoses existing unlearning methods via representation analysis without modifying model architectures.
- **vs privacy protection work**: Focuses on the reversibility of information deletion rather than the mathematical boundaries of privacy leakage.

## Rating
- Novelty: ⭐⭐⭐⭐⭐ First systematic representation-level reversibility analysis.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers 6 unlearning methods, 2 models, and multiple data domains.
- Writing Quality: ⭐⭐⭐⭐⭐ Clear problem statement and intuitive four-tier taxonomy.
- Value: ⭐⭐⭐⭐⭐ Reveals fundamental flaws in unlearning evaluation and sets new standards for LLM security assessment and privacy protection.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Model Collapse Is Not a Bug but a Feature in Machine Unlearning for LLMs](../../ICLR2026/llm_safety/model_collapse_is_not_a_bug_but_a_feature_in_machine_unlearning_for_llms.md)
- [\[ICML 2026\] DualOptim+: Bridging Shared and Decoupled Optimizer States for Better Machine Unlearning in Large Language Models](dualoptim_bridging_shared_and_decoupled_optimizer_states_for_better_machine_unle.md)
- [\[ICML 2026\] Multilingual Unlearning in LLMs: Transfer, Dynamics, and Reversibility](multilingual_unlearning_in_llms_transfer_dynamics_and_reversibility.md)
- [\[CVPR 2026\] SineProject: Machine Unlearning for Stable Vision–Language Alignment](../../CVPR2026/llm_safety/sineproject_machine_unlearning_for_stable_vision_language_alignment.md)
- [\[ICML 2026\] Differentially Private Preference Data Synthesis for Large Language Model Alignment](differentially_private_preference_data_synthesis_for_large_language_model_alignm.md)

</div>

<!-- RELATED:END -->
