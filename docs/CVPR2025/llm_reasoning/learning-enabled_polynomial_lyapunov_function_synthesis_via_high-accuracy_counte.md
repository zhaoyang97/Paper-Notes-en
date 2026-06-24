---
title: >-
  [Paper Note] Learning-enabled Polynomial Lyapunov Function Synthesis via High-Accuracy Counterexample-Guided Framework
description: >-
  [CVPR 2025][Reasoning][Lyapunov function] This paper proposes a learning-enabled polynomial Lyapunov function synthesis method which combines learning and verification. It uses data-driven machine learning to guide the selection of polynomial forms and iteratively optimizes them through a high-accuracy counterexample-guided framework, striking a balance between flexibility and mathematical rigor.
tags:
  - "CVPR 2025"
  - "Reasoning"
  - "Lyapunov function"
  - "polynomial network"
  - "formal verification"
  - "counterexample-guided"
  - "stability analysis"
date: 2026-05-08
content_hash: 643fa8ec70215a25
---

# Learning-enabled Polynomial Lyapunov Function Synthesis via High-Accuracy Counterexample-Guided Framework

**Conference**: CVPR 2025  
**Code**: None  
**Area**: Audio & Speech  
**Keywords**: Lyapunov function, polynomial network, formal verification, counterexample-guided, stability analysis

## TL;DR
This paper proposes a learning-enabled polynomial Lyapunov function synthesis method which combines learning and verification. It uses data-driven machine learning to guide the selection of polynomial forms and iteratively optimizes them through a high-accuracy counterexample-guided framework, striking a balance between flexibility and mathematical rigor.

## Background & Motivation

### Background

**Background**: Lyapunov functions are the core tools for verifying the stability of dynamical systems. Polynomial-form Lyapunov functions can transform stability analysis into efficiently solvable optimization problems (such as SOS optimization).

**Limitations of Prior Work**:

### Limitations of Prior Work

**Limitations of Prior Work**: Traditional numerical methods (such as SOS programming) rely on user-predefined polynomial templates, and improper template selection can lead to search failures.

### Key Challenge

**Key Challenge**: Although emerging neural Lyapunov methods offer flexibility, those based on naive squared polynomial networks exhibit poor generalization and lack formal correctness guarantees.

### Key Insight

**Key Insight**: Counterexample-guided verification methods suffer from slow convergence or even divergence when the precision of the counterexamples is insufficient.

**Key Challenge**: The trade-off between flexibility and verifiability—neural network methods are flexible but unverifiable, while traditional SOS methods are verifiable but lack flexibility.

**Goal**: How to utilize machine learning to automatically discover suitable polynomial forms while maintaining the mathematical rigor of polynomial Lyapunov functions.

**Key Insight**: Decompose the problem into two complementary stages: "learning-guided search direction" and "formal verification to ensure correctness," with high-accuracy counterexamples serving as a bridge to connect them.

**Core Idea**: A data-driven ML model is responsible for proposing candidate Lyapunov functions, while a high-accuracy counterexample-guided framework handles verification and iterative refinement, achieving a unification of learning-enabled and formally-verified approaches.

## Method

### Overall Architecture
The framework consists of two alternating iterative stages: (1) **Learning Stage**: Train an ML model based on sampled data and counterexamples to automatically select the polynomial form and estimate coefficients; (2) **Verification Stage**: Formally verify the candidate Lyapunov function to find high-accuracy counterexamples and feed them back to the learning stage.

### Key Designs

1. **Data-driven Polynomial Form Selection**:
    - **Function**: Leverage ML models to learn the polynomial structure suitable for the target system from data.
    - **Mechanism**: Instead of requiring users to manually specify polynomial templates, a neural network is used to search within the polynomial basis function space to learn both coefficients and structure.
    - **Design Motivation**: The selection of polynomial templates heavily relies on expert experience; automated selection can expand the applicability of the method.

2. **High-Accuracy Counterexample-Guided**:
    - **Function**: When the verification stage detects that candidate functions do not satisfy Lyapunov conditions at certain state points, precisely locate these violation points and feed them back as high-accuracy counterexamples.
    - **Mechanism**: Compared to random sampling or low-precision approximate counterexamples, high-accuracy counterexamples point out the flaws of candidate functions more precisely, making the subsequent learning iteration more efficient.
    - **Design Motivation**: Low-precision counterexamples result in noisy learning signals and slow convergence, whereas high-accuracy counterexamples can accelerate convergence and reduce the number of iterations.

3. **Formal Verification to Ensure Correctness**:
    - **Function**: Conduct rigorous mathematical verification of candidate Lyapunov functions using Sum-of-Squares (SOS) decomposition or Satisfiability Modulo Theories (SMT) solvers.
    - **Mechanism**: The final output Lyapunov function is formally proven rather than being a mere numerical approximation.
    - **Design Motivation**: For safety-critical systems, numerical approximations are not reliable enough, thereby requiring formal proofs.

## Key Experimental Results

### Main Comparison


### Main Results

| Method | Synthesis Success Rate | Number of Iterations | Applicable System Dimensions |
|------|-----------|---------|-------------|
| Traditional SOS (Fixed Template) | Limited by template selection | N/A | Low-to-Medium |
| Neural Lyapunov | Flexible but no guarantees | Many | Medium-to-High |
| Ours | High success rate + formal verification | Fewer | Medium-to-High |

### Key Findings
- High-accuracy counterexample guidance significantly reduces the number of iterations compared to randomly sampled counterexamples.
- Learning-guided polynomial search outperforms fixed-template methods on high-dimensional systems.
- The final synthesized Lyapunov functions possess rigorous mathematical guarantees and can be directly applied to controller design.

## Highlights & Insights
- **Learning + Verification Paradigm**: Chemically combining the flexibility of ML with the rigor of formal verification provides a valuable paradigm in the AI for Science field.
- **Crucial Role of High-Accuracy Counterexamples**: The quality of counterexamples directly determines the iteration efficiency, an insight that also inspires other counterexample-guided methods.
- **Automated Discovery of Polynomial Forms**: Eliminates the step of manual template selection, lowering the barrier to entry.

## Limitations & Future Work
- The expressiveness of polynomial Lyapunov functions themselves is limited; polynomial-form Lyapunov functions may not exist for certain non-linear systems.
- The computational complexity of the formal verification stage (SOS/SMT) increases drastically as the system dimension grows.
- Current work mainly targets stability analysis of autonomous systems, leaving extensions to controlled systems and switched systems to be explored.
- Developing a unified framework with safety certificates such as Barrier Functions is a promising future direction.

## Related Work & Insights

- **vs. Traditional SOS Methods**: Traditional methods rely on user-defined manual polynomial templates, where template selection largely determines the success or failure of the search. This work automatically learns the polynomial structure using ML, eliminating human template design and offering wider applicability.
- **vs. Neural Lyapunov Methods (e.g., SMT-based)**: Neural network approaches are flexible but lack formal correctness guarantees; the Learner-Verifier iterative framework in this paper balances both flexibility and verifiability.
- **vs. CEGIS Framework**: Traditional CEGIS (Counterexample-Guided Inductive Synthesis) uses counterexamples of lower precision. This work highlights high-accuracy counterexample guidance as the key to enhancing convergence efficiency.
- The Learning + Verification paradigm of this framework has transfer value for other formal verification problems (such as Barrier Certificates and invariant synthesis).

## Rating

- **Novelty**: ⭐⭐⭐⭐ The idea of combining learning-enabled and formal verification is valuable, and high-accuracy counterexample guidance is an effective engineering contribution.
- **Experimental Thoroughness**: ⭐⭐⭐ The comparison with SMT-based and traditional SOS is sufficient, but the scale and diversity of test systems remain to be expanded.
- **Writing Quality**: ⭐⭐⭐⭐ Problem definitions are clear, and the Learner-Verifier framework is systematically described.
- **Value**: ⭐⭐⭐ Makes a contribution to the intersection of AI4Science and formal verification, though the application scenarios are relatively niche.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[NeurIPS 2025\] Beyond Accuracy: Dissecting Mathematical Reasoning for LLMs Under Reinforcement Learning](../../NeurIPS2025/llm_reasoning/beyond_accuracy_dissecting_mathematical_reasoning_for_llms_u.md)
- [\[ICLR 2026\] DESIGNER: Design-Logic-Guided Multidisciplinary Data Synthesis for LLM Reasoning](../../ICLR2026/llm_reasoning/designer_design-logic-guided_multidisciplinary_data_synthesis_for_llm_reasoning.md)
- [\[NeurIPS 2025\] Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective Reinforcement Learning for LLM Reasoning](../../NeurIPS2025/llm_reasoning/beyond_the_8020_rule_highentropy_minority_tokens_drive_effec.md)
- [\[ACL 2025\] LogicPro: Improving Complex Logical Reasoning via Program-Guided Learning](../../ACL2025/llm_reasoning/logicpro_program_guided_reasoning.md)
- [\[NeurIPS 2025\] ExPO: Unlocking Hard Reasoning with Self-Explanation-Guided Reinforcement Learning](../../NeurIPS2025/llm_reasoning/expo_unlocking_hard_reasoning_with_self-explanation-guided_reinforcement_learnin.md)

</div>

<!-- RELATED:END -->
