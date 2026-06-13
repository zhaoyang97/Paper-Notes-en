---
title: >-
  [Paper Note] On Emergent Social World Models -- Evidence for Functional Integration of Theory of Mind and Pragmatic Reasoning in Language Models
description: >-
  [ACL2026][Interpretability][Theory of Mind] This paper uses large-scale behavioral evaluation alongside functional localization and ablation experiments inspired by cognitive neuroscience to provide evidence that Theory…
tags:
  - "ACL2026"
  - "Interpretability"
  - "Theory of Mind"
  - "Pragmatic Reasoning"
  - "Functional Localization"
  - "Causal Ablation"
  - "Social World Models"
date: 2026-05-08
content_hash: 092bb501acacec29
---

# On Emergent Social World Models -- Evidence for Functional Integration of Theory of Mind and Pragmatic Reasoning in Language Models

**Conference**: ACL2026  
**arXiv**: [2602.10298](https://arxiv.org/abs/2602.10298)  
**Code**: https://github.com/polina-tsvilodub/lm-emergent-social-world-models  
**Area**: Interpretability / Social Cognition Evaluation  
**Keywords**: Theory of Mind, Pragmatic Reasoning, Functional Localization, Causal Ablation, Social World Models

## TL;DR
This paper uses large-scale behavioral evaluation alongside functional localization and ablation experiments inspired by cognitive neuroscience to provide evidence that Theory of Mind (ToM) and pragmatic reasoning in language models may share internal computational mechanisms. This shifts the concept of "Social World Models" from a simple performance score to a testable functional integration hypothesis.

## Background & Motivation
**Background**: Discussions regarding whether Large Language Models (LLMs) possess "world models" often focus on environments with clear boundaries like board games, navigation, or text adventures. However, functional capabilities in natural language understanding are more complex, as models must process not only syntax and semantics but also infer the speakers' intentions, knowledge, beliefs, and social commitments. Theory of Mind and pragmatic reasoning lie at this intersection: the former concerns the representation of others' mental states, while the latter concerns understanding the true intent of utterances within specific communicative contexts.

**Limitations of Prior Work**: Existing works often evaluate ToM and pragmatic abilities separately, reporting only whether a model succeeds on specific benchmarks. Such behavioral-level scores fail to clarify whether the model utilizes the same internal mechanism; correlated performance across two tasks might simply stem from larger model scale, stronger instruction tuning, or broader world knowledge. High scores could also result from data contamination, heuristic matching, or biases in answer formatting.

**Key Challenge**: If ToM and pragmatic reasoning are merely two isolated skills, models could learn two specialized mechanisms independently. However, if social meaning in language necessitates recurring calls to representations of others' beliefs, intentions, and knowledge, then pressure for compression and reuse might drive the model to form a "social world model" invokable across tasks. The core question is not whether the model can solve a problem, but whether these capabilities are functionally integrated internally.

**Goal**: The authors break this broad question into two measurable levels: first, whether behavioral performance on ToM and pragmatic tasks is systematically correlated across models beyond what general language ability can explain; second, whether ToM-supporting sub-networks can be localized within the model and if ablating these sub-networks also impairs pragmatic reasoning.

**Key Insight**: Leveraging the "functional localizer" approach from human cognitive neuroscience, this study identifies ToM-related units by observing activation differences between target and control conditions. This approach moves beyond asking "is the model correct?" to "which internal units are selectively invoked under specific conditions, and do these units serve a causal role in downstream tasks?"

**Core Idea**: Use a three-stage evidentiary approach—behavioral correlation, functional localization, and causal ablation—to examine whether ToM and pragmatic reasoning share reusable internal mechanisms rather than remaining at the level of surface-level benchmark scores.

## Method
The methodology proceeds along two parallel tracks: cross-model statistical analysis at the behavioral level and functional localization and ablation at the mechanistic level. The former addresses whether the two abilities fluctuate together, while the latter addresses whether the same internal units truly support both functions.

### Overall Architecture
The input consists of multiple sets of multiple-choice ToM, pragmatic, and general language ability datasets, along with a suite of open-source/open-weight language models. The authors first score each model using conditional log-probability: calculating normalized conditional log-probabilities for all candidate answers and selecting the highest-probability option as the prediction to calculate accuracy.

The behavioral experiments cover 48 models from seven families: Llama, Qwen, Falcon, Mistral, OLMo, Pythia, and Gemma, with parameters ranging from 0.5B to 72B, distinguishing between base and fine-tuned versions. The tasks include 16 pragmatic datasets, 22 ToM datasets, and two general language ability controls: BLiMP and SNLI.

Mechanistic experiments select 20 models, primarily from Qwen-2.5, Llama, Falcon-3, and Gemma-2. The authors construct four ToM localizer suites: LatentBeliefs, CommunicativeIntent, GameBeliefs, and MoralIntent. Each localizer includes a target condition requiring mental state reasoning and a control condition matched for surface form that does not require ToM. As models process these stimuli, unit activations in transformer blocks are recorded at the final token before the answer. Functional sub-networks are localized via target-control activation differences, and their causal roles are verified through zeroing-out ablation.

### Key Designs
1. **Functional Integration Testing at the Behavioral Level**:
    - **Function**: Use cross-model statistical relationships to determine if ToM and pragmatic reasoning are merely co-dependent on model scale or if they exhibit specialized coupling.
    - **Mechanism**: The authors propose three predictions. P1 requires that ToM accuracy and pragmatics accuracy be positively correlated across models. P2 requires that after controlling for variables like model family, size, training type, and dataset type, the task domain (ToM vs. pragmatics) no longer significantly predicts accuracy. P3 requires that ToM accuracy predicts pragmatics accuracy better than general language ability (represented by BLiMP/SNLI).
    - **Design Motivation**: Simple correlation can lead to "big models are good at everything" being misinterpreted as capability integration. P2 and P3 incorporate scale, fine-tuning, and general language ability into the statistical model to ensure the specific link between social cognitive abilities cannot be easily explained away.

2. **ToM Sub-network Localization Based on Cognitive Neuroscience Localizers**:
    - **Function**: Localize units within the model that are selectively active for ToM target conditions relative to control conditions, forming ablatable functional sub-networks.
    - **Mechanism**: For each unit $(l,i)$, the authors compare activation differences between target and control stimulus sets, obtaining statistics using Welch's $t$-test. A "simple localizer" aggregates conditions, while a "conjunctive localizer" takes the minimum statistic across all target-control pairs—akin to the minimum statistic in neuroscience—to identify units consistently active across conditions. If significant units exceed 1% of the total, only the top 1% by absolute statistic are kept.
    - **Design Motivation**: ToM is not a monolith; it involves beliefs, intentions, desires, emotions, knowledge, etc. The authors utilize the ATOMS framework to cover these facets, synthesizing 1400 localizer stimuli based on validated fMRI materials to reduce the risk of localizing only specific "false-belief" templates.

3. **Causal Ablation and Cross-domain Transfer Validation of ToM Sub-networks**:
    - **Function**: Verify whether localized ToM sub-networks contribute causally to ToM and pragmatic tasks rather than just manifesting correlated activations.
    - **Mechanism**: For each model and localizer, two types of ablation are performed: zeroing out key units selected by the ToM target, and zeroing out an equal number of the least active control units. ToM, pragmatics, BLiMP, and SNLI tasks are then re-evaluated to compare the accuracy drop relative to the intact model.
    - **Design Motivation**: If functional integration holds, ToM sub-network ablation should impair both ToM and pragmatic tasks. Furthermore, this impairment should be stronger than control ablation and should not equally disrupt general language ability. This design transforms "shared mechanisms" into observable causal predictions.

### Loss & Training
No new language models were trained; the core focuses on evaluation, localization, and ablation. Model predictions were completed via conditional log-probability scoring of candidate answers, with average normalization for token length. Statistical analysis primarily utilized Bayesian beta regression and leave-one-out cross-validation. In mechanistic analysis, units were filtered by target-control activation differences during the localization phase and zeroed out during the ablation phase.

## Key Experimental Results

### Main Results
Behavioral experiments support a close relationship between ToM and pragmatic reasoning, which is not merely a byproduct of "larger models are better." Crucially, ToM accuracy explains pragmatics accuracy better than general language ability, grounding the argument more firmly than standard correlation analysis.

| Question | Setting | Key Findings | Interpretation |
|------|------|----------|----------|
| Correlation between ToM and Pragmatics | Avg ToM accuracy vs. pragmatics accuracy across 48 models | $r=0.68$, $p=1.24 \times 10^{-7}$ | Significant moderately-strong positive correlation, supporting P1 |
| Importance of task domain after controlling for model factors | Bayesian beta regression controlling for family, size, type, dataset type | Domain coefficient $\beta=-0.03[-0.74,0.67]$ | ToM/pragmatics labels provide no significant extra explanation, supporting P2 |
| ToM as a better predictor of pragmatics than general language | Comparing M1 (with ToM accuracy) vs M0 (with BLiMP/SNLI accuracy) | M1 significantly superior to M0, reported difference $ELPD=-16.1$, $p=0$ | ToM performance is a stronger predictor of pragmatics, supporting P3 |
| Explanatory power of internal ToM facets | Comparison of 128 regression models across 7 ATOMS categories | Best model includes intentions, desires, emotions, percepts, non-literal communication; vs baseline $ELPD=-58.80$, $p=0$ | ToM is not homogeneous; facets like percepts are critical for variance |

Functional localization experiments show non-random unit distributions, with different localizers showing different layer biases. CommunicativeIntent units are concentrated in later layers, while LatentBeliefs are in mid-to-late layers. The LB + CI conjunctive localizer was more conservative, with many models having <1% significant units. Cross-validation showed high generalization for MoralIntent and GameBeliefs; simple CommunicativeIntent and LatentBeliefs were also stable.

### Ablation Study
Ablation results provide the most significant mechanistic evidence: ToM sub-network ablation impairs both ToM and pragmatic tasks, while general language benchmarks show no credible decline. This aligns with the "shared social reasoning mechanism" explanation rather than a general disruption of language processing.

| Prediction | Content | Global Analysis Results | Supported? |
|------|----------|--------------|----------|
| P1.1 | Ablation of ToM sub-network reduces ToM performance | $\beta=0.25[0.14,0.35]$ | Yes |
| P1.2 | ToM ablation effect on ToM > Control ablation | $\beta=0.06[0.02,0.11]$ | Yes |
| P2.1 | Ablation of ToM sub-network reduces pragmatics performance | $\beta=0.30[0.20,0.39]$ | Yes |
| P2.2 | ToM ablation effect on pragmatics > Control ablation | $\beta=0.07[0.02,0.12]$ | Yes |
| P3.1 | ToM ablation does not credibly reduce general language ability | $\beta=0.13[-0.10,0.35]$ | Yes (No credible drop) |
| P3.2 | Effect on pragmatics > General language ability | $\beta=0.17[-0.07,0.41]$ | Weak evidence |

Granular analysis shows not all localizers contribute equally. P1.1, P2.1, and P3.1 generally hold, but P1.2 and P2.2 are primarily driven by LatentBeliefs (simple and conjunctive) and GameBeliefs. This suggests that sub-networks related to beliefs, percepts, desires, and emotions most stably support both ToM and pragmatic tasks.

| Analysis Object | Observed Phenomenon | Implication |
|----------|--------------|------|
| LatentBeliefs / GameBeliefs | ToM and pragmatics decline more noticeably after ablation | Units related to beliefs, percepts, desires, and emotions are primary sources of the shared mechanism |
| CommunicativeIntent | Clear layer distribution, but causal results less consistent | "Non-literal communication" units are not necessarily the core pragmatic mechanism |
| Entity tracking | Some key ablations impact entity tracking similarly to ToM/pragmatics | Shared mechanisms may involve entity state tracking, not just pure social-mental representation |
| Model Scale | No credible difference in ablation effects across sizes | Functional integration is not just a phenomenon of model capacity, though further models are needed |

### Key Findings
- Behaviorally, ToM and pragmatic accuracy are significantly correlated; ToM accuracy predicts pragmatics better than BLiMP/SNLI, indicating the link is not purely via general language ability.
- Mechanistically, ToM localizers identify sub-networks with causal effects on both ToM and pragmatics, especially those related to LatentBeliefs and GameBeliefs.
- Detailed ATOMS analysis is valuable: percepts, intentions, desires, emotions, and non-literal communication explain variances in ToM benchmarks, suggesting ToM should not be treated as a single aggregate score.
- The paper acknowledges complexities: CommunicativeIntent results are occasionally counter-intuitive, entity tracking may be an underlying factor, and P3.2 lacked strong credible support.

## Highlights & Insights
- The primary highlight is transforming the question of "whether LLMs have social world models" into an actionable investigation of functional integration. The study avoids claiming models "understand" human minds, instead asking if ToM and pragmatic reasoning share localizable and ablatable mechanisms.
- The localizer design is highly instructive: using diverse suites (LatentBeliefs, CommunicativeIntent, etc.) and conjunctive analysis to find stable units across conditions provides a framework transferable to moral reasoning or multi-agent planning.
- Behavioral statistics and mechanistic ablation complement each other. Correlation suggests abilities move together; ablation shows that damaging specific units hurts both, moving beyond surface-level benchmarking toward mechanistic interpretability.
- The paper suggests social cognition is layered: some localizers look promising in activation but lack strong causal roles; some shared effects might stem from foundational mechanisms like entity tracking.

## Limitations & Future Work
- The authors admit conclusions depend on whether benchmarks truly measure ToM and pragmatics. If benchmarks contain heuristic shortcuts, the study localizes "problem-solving skills" rather than mental state representations.
- Ablation is relatively coarse: zeroing units proves causal relevance but does not detail information flow or distinguish between encoding beliefs versus task formats. Future work could use activation patching or circuit discovery.
- Localizer stimuli are synthetic. While inspired by human neuroscience and checked with PCA, they lack human fMRI validation and should be viewed as neuroscience-inspired probes rather than proofs of human-like organization.
- The evaluation focuses on English and traditional false-belief tasks. Future research should extend to true-belief, cultural norms, and multi-turn interactions.
- Potential data contamination remains an open issue for behavioral benchmarks, though synthetic localizers mitigate this risk for localization.

## Related Work & Insights
- **vs AlKhamissi et al. 2025a**: AlKhamissi introduced functional localization for language networks; this paper extends the approach to ToM with larger datasets and finer categories, providing stronger causal evidence.
- **vs Hu et al. / Ma et al. / Sap et al.**: These focused on performance scores; this paper asks if the underlying mechanisms are shared by connecting the domains via statistical control and ablation.
- **vs Saxe & Kanwisher**: Human neuroscience found overlaps between ToM and non-literal language nets; this paper migrates that hypothesis to LLMs while cautiously avoiding direct model-to-human equivalence.
- **vs Circuit Analysis**: Mechanistic interpretability often looks at specific circuits for single variables; this paper’s coarser granularity focuses on cross-task functional sub-networks, suggesting a workflow of localization before fine-grained tracing.

## Rating
- Novelty: ⭐⭐⭐⭐☆ Connects ToM, pragmatics, localizers, and ablation creatively, though tools are expansions of existing methods.
- Experimental Thoroughness: ⭐⭐⭐⭐☆ Extensive behavioral and mechanistic coverage across 48/20 models; synthetic localizer validity remains a point for replication.
- Writing Quality: ⭐⭐⭐⭐☆ Clear organization of hypotheses and results; Bayesian terminology may have a slight learning curve.
- Value: ⭐⭐⭐⭐☆ Provides a testable path for "social world models" and a reusable paradigm for mechanistic interpretability in social cognition.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] LLM World Models Are Mental: Output Layer Evidence of Brittle World Model Use in LLM Mechanical Reasoning](../../NeurIPS2025/interpretability/llm_world_models_are_mental_output_layer_evidence_of_brittle_world_model_use_in_.md)
- [\[ACL 2026\] Knowledge Vector of Logical Reasoning in Large Language Models](knowledge_vector_of_logical_reasoning_in_large_language_models.md)
- [\[ACL 2026\] METER: Evaluating Multi-Level Contextual Causal Reasoning in Large Language Models](meter_evaluating_multi-level_contextual_causal_reasoning_in_large_language_model.md)
- [\[ACL 2026\] Preference Heads in Large Language Models: A Mechanistic Framework for Interpretable Personalization](preference_heads_in_large_language_models_a_mechanistic_framework_for_interpreta.md)
- [\[ACL 2026\] Sparse Feature Coactivation Reveals Causal Semantic Modules in Large Language Models](sparse_feature_coactivation_reveals_causal_semantic_modules_in_large_language_mo.md)

</div>

<!-- RELATED:END -->
