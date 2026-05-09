---
title: >-
  [Paper Note] AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint
description: >-
  [ICLR 2026][LLM Alignment][activation steering] This paper proposes AlphaSteer, which learns a null-space-constrained transformation matrix to dynamically construct steering vectors that produce near-zero vectors for benign inputs (preserving utility) while reconstructing the refusal direction vector for malicious inputs (enhancing safety), providing theoretical guarantees for the decoupling of safety and utility.
tags:
  - ICLR 2026
  - LLM Alignment
  - activation steering
  - refusal direction
  - null-space projection
  - jailbreak defense
  - safety-utility trade-off
date: 2026-05-08
content_hash: baadd38968c4a223
---

# AlphaSteer: Learning Refusal Steering with Principled Null-Space Constraint

**Conference**: ICLR 2026
**arXiv**: [2506.07022](https://arxiv.org/abs/2506.07022)
**Code**: [https://github.com/AlphaLab-USTC/AlphaSteer](https://github.com/AlphaLab-USTC/AlphaSteer)
**Area**: LLM Alignment
**Keywords**: activation steering, refusal direction, null-space projection, jailbreak defense, safety-utility trade-off

## TL;DR
This paper proposes AlphaSteer, which learns a null-space-constrained transformation matrix to dynamically construct steering vectors that produce near-zero vectors for benign inputs (preserving utility) while reconstructing the refusal direction vector for malicious inputs (enhancing safety), providing theoretical guarantees for the decoupling of safety and utility.

## Background & Motivation
**State of the Field**: Activation steering is an emerging approach to LLM safety enhancement. The core mechanism injects a "refusal direction vector" $\mathbf{r}$ into the model's internal activations at inference time, causing the model to refuse malicious prompts.

**Limitations of Prior Work**: Injecting the same $\mathbf{r}$ into all inputs causes excessive refusal of benign prompts, resulting in a safety–utility trade-off. Existing methods either perform vector calibration (e.g., Surgical uses PCA decomposition to subtract spurious refusal components) or apply conditional steering (e.g., CAST sets a threshold to steer only "malicious" activations), but both rely on heuristic designs without theoretical guarantees.

**Root Cause**: Safety enhancement and utility preservation are fundamentally conflicting demands on the same steering operation—malicious activations need to be substantially altered while benign activations must remain unchanged, yet existing methods cannot guarantee this mathematically.

**Paper Goals**: (1) How can steering be made strictly non-interfering with benign activations? (2) How can steering reliably reconstruct the refusal direction for malicious activations?

**Starting Point**: The authors observe the mathematical properties of null spaces—if the row vectors of a transformation matrix lie in the null space of the benign activation matrix, the transformation necessarily produces a zero vector when applied to benign activations.

**Core Idea**: Replace the fixed refusal direction vector with a learnable transformation matrix constrained to the null space, enabling adaptive steering for benign versus malicious inputs.

## Method

### Overall Architecture
Given the activation $\mathbf{h}^{(l)}$ at a particular layer of the LLM, AlphaSteer dynamically constructs a steering vector $\mathbf{s}^{(l)} = \Delta^{(l)} \mathbf{h}^{(l)}$ via a learned transformation matrix $\Delta^{(l)}$, which is then added to the activation: $\mathbf{h}'^{(l)} = \mathbf{h}^{(l)} + \lambda \Delta^{(l)} \mathbf{h}^{(l)}$. The key is that $\Delta = \tilde{\Delta} \hat{\mathbf{P}}$, where $\hat{\mathbf{P}}$ is the null-space projection matrix of the benign activations and $\tilde{\Delta}$ is a matrix learned via regularized least squares.

### Key Designs

1. **Null-Space Projection for Utility Preservation**:

    - **Function**: Guarantees that for any benign activation $\mathbf{h}_b$, the steering vector satisfies $\Delta \mathbf{h}_b \approx \mathbf{0}$.
    - **Mechanism**: Benign prompt activations from $N_b$ samples are collected to form matrix $\mathbf{H}_b$. The SVD of the non-centered covariance matrix $\mathbf{H}_b \mathbf{H}_b^\top$ is computed, and the eigenvectors corresponding to zero eigenvalues are used to construct the projection matrix $\hat{\mathbf{P}} = \hat{\mathbf{U}} \hat{\mathbf{U}}^\top$, such that $\tilde{\Delta} \hat{\mathbf{P}} \mathbf{H}_b = \mathbf{0}$ holds exactly.
    - **Design Motivation**: The mathematical properties of null spaces provide a theoretical guarantee that benign activations are unaffected, rather than relying on heuristic thresholds. Lemma 1 reduces the null-space computation from $N_b$-dimensional space to $d$-dimensional space ($d \ll N_b$), improving computational efficiency.

2. **Regularized Linear Regression for Safety Enhancement**:

    - **Function**: Learns $\tilde{\Delta}$ such that for malicious activations $\mathbf{H}_m$, the steering vector reconstructs the refusal direction $\mathbf{R}$.
    - **Mechanism**: Solves the regularized least-squares problem $\min_{\tilde{\Delta}} \|\tilde{\Delta} \hat{\mathbf{P}} \mathbf{H}_m - \mathbf{R}\| + \alpha \|\tilde{\Delta} \hat{\mathbf{P}}\|$, which admits the closed-form solution $\tilde{\Delta}^\star = \mathbf{R} \mathbf{H}_m^\top \hat{\mathbf{P}}^\top (\hat{\mathbf{P}} \mathbf{H}_m \mathbf{H}_m^\top \hat{\mathbf{P}}^\top + \alpha \hat{\mathbf{P}} \hat{\mathbf{P}}^\top)^+$.
    - **Design Motivation**: The closed-form solution eliminates the need for iterative optimization, making deployment extremely simple. The regularization term $\alpha$ prevents overfitting.

3. **Refusal Direction Vector Extraction**:

    - **Function**: Extracts a direction $\mathbf{r}$ representative of "refusal behavior."
    - **Mechanism**: Follows the difference-in-means approach by computing the mean difference between activations of refusal responses and compliance responses.
    - **Distinction from Prior Work**: Although the extraction of $\mathbf{r}$ follows Arditi et al., AlphaSteer does not inject $\mathbf{r}$ directly; instead, it uses the learned $\Delta$ to reconstruct $\mathbf{r}$ exclusively for malicious inputs.

### Loss & Training
No gradient-based optimization is required. The null-space projection matrix is computed analytically via SVD, and the transformation matrix is obtained via the closed-form solution to regularized least squares. The entire method requires only forward-pass activation collection and matrix operations—no backpropagation—resulting in minimal deployment overhead.

## Key Experimental Results

### Main Results: Defense Success Rate (DSR)

| Model | Method | AIM | AutoDAN | Cipher | GCG | Jailbroken | PAIR | ReNeLLM | Avg DSR |
|-------|--------|-----|---------|--------|-----|------------|------|---------|---------|
| Llama-3.1-8B | Vanilla | 92 | 48 | 0 | 58 | 75 | 45 | 28 | 48.0 |
| | Surgical | 100 | 76 | 61 | 98 | 88 | 90 | 67 | 82.8 |
| | Circuit Breaker | 100 | 100 | 34 | 100 | 80 | 96 | 81 | 84.4 |
| | **AlphaSteer** | **100** | **99** | **63** | **97** | **92** | **98** | **100** | **91.9** |
| Qwen2.5-7B | Vanilla | 25 | 2 | 1 | 22 | 71 | 19 | 4 | 20.6 |
| | **AlphaSteer** | **100** | **100** | **100** | **100** | **95** | **88** | **98** | **97.3** |
| Gemma-2-9b | Vanilla | 0 | 5 | 0 | 75 | 68 | 17 | 8 | 24.7 |
| | **AlphaSteer** | **100** | **98** | **100** | **100** | **99** | **91** | **99** | **98.2** |

### Utility Preservation Comparison

| Model | Method | XSTest CR↑ | AlpacaEval WR↑ | MATH Acc↑ | GSM8K Acc↑ | Utility Score |
|-------|--------|------------|----------------|-----------|------------|---------------|
| Llama-3.1-8B | Vanilla | 92.4 | 50.0 | 45.0 | 81.0 | 67.1 |
| | CAST | 90.0 | 31.1 | 0.0 | 0.0 | 30.2 |
| | RV (direct injection) | 4.0 | 10.4 | 37.0 | 65.0 | 29.1 |
| | **AlphaSteer** | **91.2** | **48.1** | **46.0** | **84.0** | **67.3** |
| Qwen2.5-7B | Vanilla | 97.2 | 50.0 | 67.0 | 96.0 | 77.6 |
| | **AlphaSteer** | **95.6** | **48.1** | **65.0** | **95.0** | **75.9** |

### Key Findings
- AlphaSteer achieves average DSR exceeding 90% across all three models while maintaining utility scores nearly identical to vanilla baselines (gap < 2%).
- The contrast is stark: CAST reduces accuracy on math tasks to 0% (misclassifying math problems as malicious), while direct RV injection reduces utility scores to 2.4–29.1%.
- PCA visualization confirms that the L2 norm of AlphaSteer's steering vectors is substantially smaller for benign activations than for malicious ones, validating the effectiveness of the null-space constraint.
- As the steering strength $\lambda$ increases, AlphaSteer's safety continuously improves while utility remains stable, whereas baselines exhibit sharp utility degradation.

## Highlights & Insights
- **Seamless Translation from Theory to Practice**: Applying null-space projection—a classical linear algebra tool—to activation steering provides a mathematical guarantee that benign activations are unaffected. The method admits a closed-form solution, requires no training loop, and adds only a single matrix multiplication at inference time, making it remarkably elegant.
- **Complete Safety–Utility Decoupling**: Unlike prior trade-off-based designs, AlphaSteer structurally decouples the two objectives—null-space projection handles utility preservation and linear regression handles safety enhancement, without mutual interference.
- **Transferable Design Paradigm**: The null-space-constrained paradigm of "zero effect on one class of data, maximum effect on another" can be transferred to other scenarios requiring selective intervention, such as preventing forgetting in continual learning or protecting non-target knowledge in model editing.

## Limitations & Future Work
- Validation is limited to 7–9B models; large-scale reasoning models (e.g., o1/DeepSeek-R1) are not evaluated, and whether the null-space dimensionality remains effective at larger scales is uncertain.
- The method requires pre-collecting benign and malicious activations to compute the null space and learn $\tilde{\Delta}$, making it sensitive to data distribution—if malicious patterns at test time differ substantially from those seen during training, reconstruction quality may degrade.
- In practice, the null space is approximated by treating the smallest $p\%$ of eigenvalues as zero; this threshold is a hyperparameter that affects the safety–utility balance.
- Since the method only modifies activations at inference time without altering model weights, it does not defend against weight-level attacks.

## Related Work & Insights
- **vs. Surgical (Wang et al., 2024)**: Surgical calibrates the refusal direction via PCA but still applies it uniformly to all inputs; PCA visualization shows that benign activations are noticeably distorted. AlphaSteer's null-space constraint avoids this problem mathematically.
- **vs. CAST (Lee et al., 2024)**: CAST uses a threshold to decide whether to apply steering, but the threshold design is heuristic, causing math problems to be misclassified as malicious and driving utility to zero. AlphaSteer requires no threshold.
- **vs. Circuit Breaker (Zou et al., 2024)**: Circuit Breaker requires additional training, whereas AlphaSteer can be deployed with only matrix operations, incurring lower computational cost.
- The null-space projection approach shares conceptual similarities with OWM/PackNet in continual learning, both of which project into orthogonal spaces to preserve existing functionality.

## Rating
- **Novelty**: ⭐⭐⭐⭐ Applying null-space constraints to activation steering is a novel theoretical contribution, though the refusal direction extraction still follows existing methods.
- **Experimental Thoroughness**: ⭐⭐⭐⭐ Covers 3 models × 7 attack types × 4 utility benchmarks comprehensively, but lacks experiments on larger models.
- **Writing Quality**: ⭐⭐⭐⭐⭐ Theoretical derivations are clear; equations, figures, and tables are well-integrated; the logical chain from motivation to method to experiments is complete.
- **Value**: ⭐⭐⭐⭐ Provides a zero-cost inference-time safety enhancement solution with high practical value, though scalability remains to be verified.

<!-- RELATED:START -->

## Related Papers

- [\[CVPR 2026\] Principled Steering via Null-space Projection for Jailbreak Defense in Vision-Language Models](../../CVPR2026/llm_alignment/principled_steering_via_null-space_projection_for_jailbreak_defense_in_vision-la.md)
- [\[ICLR 2026\] Mitigating the Safety Alignment Tax with Null-Space Constrained Policy Optimization](mitigating_the_safety_alignment_tax_with_null-space_constrained_policy_optimizat.md)
- [\[ICLR 2026\] Learning Ordinal Probabilistic Reward from Preferences (OPRM)](learning_ordinal_probabilistic_reward_from_preferences.md)
- [\[ICLR 2026\] Reasoned Safety Alignment: Ensuring Jailbreak Defense via Answer-Then-Check](reasoned_safety_alignment_ensuring_jailbreak_defense_via_answer-then-check.md)
- [\[ICLR 2026\] Swap-guided Preference Learning for Personalized RLHF (SPL)](swap-guided_preference_learning_for_personalized_reinforcement_learning_from_hum.md)

<!-- RELATED:END -->
