---
title: >-
  [Paper Note] LLM Circuit Analyses Are Consistent Across Training and Scale
description: >-
  [AAAI 2026][Interpretability][Mechanistic Interpretability] This paper presents the first systematic tracking of internal circuits in decoder-only LLMs across 300 billion tokens of training and model scales ranging from…
tags:
  - "AAAI 2026"
  - "Interpretability"
  - "Mechanistic Interpretability"
  - "Circuit Analysis"
  - "Training Dynamics"
  - "Model Scale"
  - "Attention Heads"
date: 2026-05-08
content_hash: 74342cfa2daa0b5b
---

# LLM Circuit Analyses Are Consistent Across Training and Scale

**Conference**: AAAI 2026
**arXiv**: [2407.10827](https://arxiv.org/abs/2407.10827)
**Code**: None
**Area**: Interpretability
**Keywords**: Mechanistic Interpretability, Circuit Analysis, Training Dynamics, Model Scale, Attention Heads

## TL;DR
This paper presents the first systematic tracking of internal circuits in decoder-only LLMs across 300 billion tokens of training and model scales ranging from 70M to 2.8B parameters. It finds that while specific attention heads may be replaced over the course of training, the underlying algorithms remain stable and consistent across scales, suggesting that circuit analyses conducted on smaller models generalize to larger models and longer training runs.

## Background & Motivation
1. **Background**: Mechanistic interpretability has advanced rapidly in recent years. Researchers reverse-engineer neural networks by identifying internal "circuits"—computational subgraphs responsible for specific tasks. Prior work has uncovered concrete circuit structures and key components (e.g., name-mover heads, induction heads) on tasks such as IOI (Indirect Object Identification) and Greater-Than (year comparison).
2. **Limitations of Prior Work**: The vast majority of circuit analyses study only a single snapshot of a model at the end of pretraining. However, deployed LLMs commonly undergo continued training or fine-tuning. Existing interpretability work on training dynamics focuses on encoder-based or toy models, which differ substantially from the mainstream decoder-only architecture, calling into question the transferability of prior findings.
3. **Key Challenge**: If the conclusions of circuit analyses apply only to a specific model at a specific training step, the broader utility of the field is severely undermined. It is therefore critical to determine whether such analyses exhibit temporal stability and cross-scale transferability.
4. **Goal**: (1) When do functional components within circuits emerge during training, and is this consistent across scales? (2) When specific attention heads are replaced, does the underlying algorithm change? (3) How do graph-level circuit properties (size, composition) evolve with training and scale?
5. **Key Insight**: The authors leverage the Pythia model suite—a unique resource offering models ranging from 70M to 12B parameters, each with 154 training checkpoints spanning 300 billion tokens of training—enabling systematic longitudinal tracking.
6. **Core Idea**: By leveraging the Pythia model suite across 300 billion tokens of training and 70M–2.8B parameter scales, this work systematically tracks circuit evolution and reveals a stability principle: components change, but algorithms do not.

## Method

### Overall Architecture
The research framework consists of three levels of analysis: (1) **Behavioral evaluation and component emergence**: tracking when model performance on four tasks emerges and when the corresponding functional components appear; (2) **Algorithmic stability analysis**: verifying whether the underlying algorithm changes when components are replaced; (3) **Graph-level circuit analysis**: studying how the circuit subgraph itself (node sets, size) evolves with training. The inputs are all checkpoints from the Pythia model suite; the outputs are systematic conclusions about circuit stability across time and scale.

### Key Designs

1. **Efficient Circuit Discovery (EAP-IG)**:
    - Function: Automatically discovers circuits at each checkpoint using Edge Attribution Patching with Integrated Gradients.
    - Mechanism: EAP-IG approximates the impact on loss when each edge is ablated via gradient estimation. After scoring all edges, a greedy search identifies the minimal circuit that achieves at least 80% of full-model performance. Binary search is used to determine the optimal circuit size, with the search range spanning from 1 edge to 5% of the total number of edges in the model.
    - Design Motivation: Traditional patching methods (e.g., edge-by-edge activation patching) require a number of forward passes that grows with model size, making them entirely infeasible for a setting involving 154 checkpoints across multiple model scales. EAP-IG completes attribution in a fixed number of forward and backward passes, enabling large-scale longitudinal study.

2. **Functional Component Emergence Tracking**:
    - Function: Quantitatively tracks the emergence and evolution of four types of key attention head components—induction heads, successor heads, copy suppression heads, and name-mover heads—over the course of training.
    - Mechanism: At each checkpoint, attention heads within the discovered circuit are scored using established functional metrics (e.g., copy score, CSPA score, induction score, succession score). Scores are summed across all heads in the circuit and normalized across checkpoints, yielding a time series of component behavior intensity.
    - Design Motivation: Understanding when functional components emerge is essential for explaining when task capabilities arise and for verifying cross-scale consistency.

3. **Algorithmic Stability Verification (Path Patching)**:
    - Function: Conducts an in-depth three-stage analysis of the IOI circuit—reverse-engineering the final circuit algorithm, developing quantitative metrics, and verifying algorithmic stability across checkpoints.
    - Mechanism: The IOI algorithm is decomposed into three logical steps: (Step 1) name-mover heads and copy suppression heads directly influence the logit difference; (Step 2) S-inhibition heads guide name-mover heads to attend to the correct name using token and positional information; (Step 3) induction heads and duplicate-token heads supply information to S-inhibition heads. Path patching metrics (contribution ratios of target components) are constructed for each step and tracked across checkpoints to assess stability.
    - Design Motivation: Component replacement does not necessarily imply algorithmic change. Distinguishing "fluctuations in implementation details" from "changes in the underlying algorithm" is critical for the credibility of circuit analyses.

4. **Graph-Level Circuit Analysis**:
    - Function: Computes the Jaccard similarity (smoothed with EWMA) of circuit node sets between adjacent checkpoints, and analyzes the relationship between circuit size and model scale.
    - Mechanism: EWMA-Jaccard similarity is computed as $\hat{x}_t = 0.5 \hat{x}_{t-1} + 0.5 x_t$, measuring the temporal stability of circuit composition.
    - Key Findings: Larger models tend to form more stable circuits; circuit size is positively correlated with model scale (Pearson $r = 0.72$–$0.9$).

### Four Tasks Studied
- **IOI (Indirect Object Identification)**: Given "When John and Mary went to the store, John gave a drink to", the model should predict Mary rather than John. Measured by the logit difference between the two names.
- **Gendered-Pronoun**: Given "So Paul is such a good cook, isn't", the model should prefer "he" over "she". Measured by logit difference.
- **Greater-Than**: Given "The war lasted from the year 1732 to the year 17", the model should output a year suffix ≥ 32. Measured by probability difference.
- **SVA (Subject-Verb Agreement)**: Given "The keys on the cabinet", the model should predict "are" rather than "is". Measured by probability difference.

These tasks are sufficiently simple to be tractable for small models and have been the subject of thorough prior circuit analyses, providing a basis for validation.

## Key Experimental Results

### Main Results: Consistency of Component Emergence Timing

| Component Type | Task | Emergence (tokens) | Cross-Scale Consistency |
|---|---|---|---|
| Induction Heads | IOI, Greater-Than | ~2×10⁹ | Emerge at similar points across all scales |
| Successor Heads | Greater-Than | ~2–5×10⁹ | Consistent across scales; strength declines later |
| Name-Mover Heads | IOI | ~2–8×10⁹ | Consistent across scales; high intensity |
| Copy Suppression Heads | IOI | ~2–8×10⁹ | Emergence speed and intensity vary by scale |

### Ablation Study: Algorithmic Stability Verification

| Metric | Pythia-160M | Pythia-410M | Pythia-1B | Pythia-2.8B |
|---|---|---|---|---|
| Name-Mover + Copy Suppression contribution ratio | >70% | >70% | >70% | >70% |
| S-Inhibition → Name-Mover path importance | >50% | >50% | >50% | >50% |
| Induction/Dup-Token → S-Inhibition path importance | >50% | Varies | >50% | >50% |

### Key Findings
- **High consistency in component emergence**: Models at all scales (except 70M) acquire task capabilities at similar token counts, and the emergence timing of functional components closely tracks the task learning curves, confirming that these components drive capability emergence.
- **Stable algorithms despite component turnover**: Taking Pythia-160M as an example, name-mover head (4,6) suddenly loses functionality at approximately 3×10¹⁰ tokens, but other heads assume its role, while overall algorithmic metrics remain stable. This "load balancing" mechanism ensures continuity of model behavior.
- **Upper bound on learning rate**: A surprising finding is that larger models do not always learn faster—beyond a certain scale, learning speed on some tasks plateaus or even slightly decreases (e.g., on the IOI task, the learning curves of the 6.9B and 12B models more closely resemble that of the 160M model).
- **Circuit size scales positively with model size**: Larger models require more components to perform the same task (Pearson $r$ up to 0.9), indicating that functional roles are distributed across more heads rather than concentrated.
- **Larger models have more stable circuits**: EWMA-Jaccard similarity analysis shows that circuits in Pythia-70M/160M exhibit greater fluctuation, while circuits in larger models change more gradually over training, indicating a stability advantage conferred by scale.
- **Circuits progressively converge toward their final state**: Although intermediate-checkpoint circuits differ noticeably from the final circuit (with components continuously being replaced), the overall trajectory is a directional convergence toward the final circuit structure, indicating that training is not a random walk.

## Highlights & Insights
- **Core finding: "Components change, algorithms do not"**: This is the paper's most critical insight—even when the specific attention heads executing a function are replaced during training, the overall algorithm the model uses to solve the task remains unchanged. This provides strong empirical grounding for the reliability and transferability of circuit analyses. The finding can be analogized to "the company's employees change, but the business process stays the same."
- **Large-scale longitudinal empirical design**: The systematic experimental design—spanning 154 checkpoints × multiple model scales × 4 tasks—is unprecedented in the mechanistic interpretability literature and serves as a methodological template for future work.
- **Validation of small-model research value**: If circuit analyses on small models genuinely generalize to large models, interpretability research can substantially reduce computational costs—a finding of significant practical importance for the field.

## Limitations & Future Work
- **Tasks are overly simple**: The four tasks studied (IOI, gendered pronouns, year comparison, subject-verb agreement) can all be solved by small models. For more complex tasks (e.g., multi-step reasoning, code generation), a greater diversity of algorithmic solutions may exist, and the stability conclusions may not hold.
- **Limited to the Pythia model suite**: All models share the same architecture and training setup, making it impossible to disentangle whether conclusions are architecture-general or Pythia-specific. Validation on architectures such as Llama and GPT is necessary.
- **No SAE feature-level analysis**: The authors themselves note that the current analysis operates at the attention-head level, but recent feature-level analyses based on Sparse Autoencoders (SAEs) may reveal finer-grained patterns.
- **Circuit completeness is difficult to guarantee**: Although an 80% faithfulness threshold is used, there is no guarantee that the circuit captures all relevant mechanisms; in particular, MLP contributions may be underestimated.

## Related Work & Insights
- **vs. Wang et al. (IOI Circuit)**: Wang et al. manually discovered the complete IOI circuit algorithm in GPT-2 Small via path patching. This paper verifies that a similar but not identical IOI algorithm exists in Pythia models (e.g., copy suppression heads make positive rather than negative contributions in Pythia) and further demonstrates that this algorithm remains stable across training time. The automated method (EAP-IG) enables large-scale analysis and addresses the scalability limitations of manual analysis.
- **vs. Olsson et al. (Induction Heads)**: Olsson et al. found that induction heads emerge consistently across scales at approximately 2B–5B tokens. This paper replicates that finding and extends the analysis to additional component types (successor heads, name-mover heads, copy suppression heads), demonstrating that "cross-scale consistent emergence" is a more general phenomenon rather than a property unique to induction heads.
- **vs. Prakash et al. (Fine-tuning & Circuits)**: Prakash et al. studied circuit changes after fine-tuning, but only compared single before-and-after checkpoints. This paper extends the analysis to a continuous pretraining trajectory of 300 billion tokens, providing a far more comprehensive longitudinal perspective.

## Rating
- Novelty: ⭐⭐⭐⭐ First large-scale longitudinal circuit tracking of decoder-only LLMs, uncovering important stability regularities
- Experimental Thoroughness: ⭐⭐⭐⭐⭐ The systematic experimental design covering multiple scales, checkpoints, and tasks is exemplary
- Writing Quality: ⭐⭐⭐⭐ Clear structure and coherent logic, though some metric definitions and experimental details require consulting the appendix
- Value: ⭐⭐⭐⭐ Provides important empirical evidence on the reproducibility and transferability of mechanistic interpretability findings, though the simplicity of the tasks limits the generality of the conclusions

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Attention Gathers, MLPs Compose: A Causal Analysis of an Action-Outcome Circuit in VideoViT](attention_gathers_mlps_compose_a_causal_analysis_of_an_action-outcome_circuit_in.md)
- [\[CVPR 2026\] Reallocating Attention Across Layers to Reduce Multimodal Hallucination](../../CVPR2026/interpretability/reallocating_attention_across_layers_to_reduce_multimodal_hallucination.md)
- [\[NeurIPS 2025\] Sloth: Scaling Laws for LLM Skills to Predict Multi-Benchmark Performance Across Families](../../NeurIPS2025/interpretability/sloth_scaling_laws_for_llm_skills_to_predict_multi-benchmark_performance_across_.md)
- [\[ICLR 2026\] Evolution of Concepts in Language Model Pre-Training](../../ICLR2026/interpretability/evolution_of_concepts_in_language_model_pre-training.md)
- [\[ICLR 2026\] Formal Mechanistic Interpretability: Automated Circuit Discovery with Provable Guarantees](../../ICLR2026/interpretability/formal_mechanistic_interpretability_automated_circuit_discovery_with_provable_gu.md)

</div>

<!-- RELATED:END -->
