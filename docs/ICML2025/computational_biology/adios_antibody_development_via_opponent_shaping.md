---
title: >-
  [Paper Note] ADIOS: Antibody Development via Opponent Shaping
description: >-
  [ICML 2025][Computational Biology][Opponent Shaping] Introducing opponent shaping from multi-agent reinforcement learning into antibody design, this paper proposes the ADIOS meta-learning framework: the outer loop optimises the antibody, and the inner loop simulates adaptive viral escape, ensuring that the designed "shaper" antibodies (shapers) not only counter current viral variants but also actively steer viral evolution toward weaker, more easily targeted directions.
tags:
  - "ICML 2025"
  - "Computational Biology"
  - "Opponent Shaping"
  - "Antibody Design"
  - "Viral Escape"
  - "Meta-Learning"
  - "Game Theory"
date: 2026-05-08
content_hash: 8174f68a152961f1
---

# ADIOS: Antibody Development via Opponent Shaping

**Conference**: ICML 2025  
**arXiv**: [2409.10588](https://arxiv.org/abs/2409.10588)  
**Code**: [github.com/olakalisz/adios](https://github.com/olakalisz/adios)  
**Area**: Computational Biology / Antibody Design  
**Keywords**: Opponent Shaping, Antibody Design, Viral Escape, Meta-Learning, Game Theory

## TL;DR

Introducing opponent shaping from multi-agent reinforcement learning into antibody design, this paper proposes the ADIOS meta-learning framework: the outer loop optimises the antibody, and the inner loop simulates adaptive viral escape, ensuring that the designed "shaper" antibodies (shapers) not only counter current viral variants but also actively steer viral evolution toward weaker, more easily targeted directions.

## Background & Motivation

Traditional antiviral therapies (including vaccines and monoclonal antibodies) are designed only for **current** viral strains—a **myopic** strategy. Although initial efficacy might be high, the selective pressure exerted by the treatment itself drives the emergence of new escape mutants, rendering the therapy ineffective. The COVID-19 pandemic is a classic example: the B.1.351 variant substantially reduced the protective efficacy of vaccines.

Key Insight: **Our therapies inevitably influence viral evolution; rather than passively responding, we should actively exploit this influence**. Specifically:
- Myopic antibodies: Strong initial binding, but the virus quickly evolves escape mutations (variant B), rendering the therapy ineffective
- Shaper antibodies (shapers): Consider the long-term game; while maintaining binding affinity, they guide viral evolution toward weaker variants (variant C)

Existing antibody design methods (energy optimisation, sequence language models, diffusion models, etc.) all fail to consider the feedback effects of therapy on viral evolution. While methods like EVEscape can **predict** viral escape, they cannot **influence** the direction of escape. ADIOS fills this gap.

## Method

### Overall Architecture

ADIOS models the interaction between antibodies and viruses as a **two-player zero-sum game** and employs a nested meta-learning architecture:

- **Antibody Optimisation Loop (Outer Loop)**: Optimises antibody sequences using genetic algorithms, aiming to maximise long-term antibody fitness (considering post-escape viral states)
- **Simulated Viral Escape via Evolution (Inner Loop)**: Simulates the process of the virus gradually escaping through mutations given the current antibody

Three core components:
1. **Virus-antibody game**: Defines the action spaces and reward functions for both parties
2. **Simulated viral escape**: Simulates how the virus adapts to a given antibody based on evolution
3. **Antibody optimisation**: Optimises the antibody using Monte Carlo sampling + genetic algorithms

### Key Designs

#### 1. Virus-Antibody Game

The actions of both parties are amino acid sequences: virus sequence $v \in \mathbb{A}^{N_v}$ and antibody sequence $a \in \mathbb{A}^{N_a}$ (where $\mathbb{A}$ represents the set of 20 amino acids).

The antibody reward function is elegantly designed with three terms:

$$R_a(v, a) = B(v, a) - B(t_a^-, a) - B(v, t_v^+)$$

- $B(v, a)$: Binding affinity between the antibody and the virus (higher is better)
- $B(t_a^-, a)$: Binding of the antibody to human proteins (anti-target) (penalty term to prevent the antibody from being "too sticky" and attacking the self)
- $B(v, t_v^+)$: Binding of the virus to host cell receptors (encourages the antibody to indirectly block the infection capability of the virus)

The viral reward is $R_v = -R_a$ (zero-sum game). This design ensures that:
- The virus cannot escape by becoming completely inert (otherwise losing its capability to infect)
- The antibody cannot win by becoming a universal binder (otherwise attacking human proteins)

#### 2. Simulated Viral Escape (Inner Loop)

Given an initial virus $\hat{v}$ and a fixed antibody $a$, simulate $H$ steps of evolutionary escape:

**Algorithm 1 — Inner Loop:**
1. Initialize $\hat{v}_0 = \hat{v}$
2. For each generation $i = 0, ..., H-1$:
    - Generate population: Clone $\hat{v}_i$ for $P=15$ copies, applying approximately 1 random amino acid mutation to each
    - Calculate the fitness of each mutant: $R_v(v_k^i, a)$
    - Sample the next generation according to a Boltzmann distribution: $\mathbb{P}(\hat{v}_{i+1} = v_k^i) \propto \exp(\beta \cdot R_v(v_k^i, a))$
3. Output the escape trajectory $\hat{\mathbf{v}} = [\hat{v}_0, \hat{v}_1, ..., \hat{v}_H]$

The temperature parameter $\beta$ controls the selection randomness: as $\beta \to \infty$, selection approaches deterministic optimal fitness selection.

#### 3. Antibody Optimisation (Outer Loop)

Define the **true** objective function of the antibody—the average fitness under escape:

$$F_{\hat{v}}^H(a) = \mathbb{E}_{\hat{\mathbf{v}} \sim \text{Ev}(\hat{v}, a)} \left[ \frac{1}{H+1} \sum_{i=0}^{H} R_a(\hat{v}_i, a) \right]$$

When $H=0$, this degenerates to the myopic objective $F_{\hat{v}}^0(a) = R_a(\hat{v}, a)$.

**Algorithm 2 — Outer Loop:**
1. Start from a random antibody $\hat{a}_0$
2. For each step $i = 0, ..., N-1$ ($N=30$):
    - Generate population: $P_a = 40$ antibodies (the original + 39 single-point mutants)
    - Estimate $F_{\hat{v}}^H(a_k^i)$ for each candidate antibody using $\eta = 5$ Monte Carlo escape trajectories
    - Greedily select: $\hat{a}_{i+1} = \arg\max_k \mathbb{E}[F_{\hat{v}}^H(a_k^i)]$
3. Output the optimal antibody $\hat{a}_N$

#### 4. GPU-Accelerated Binding Simulator

The original Absolut! framework runs on CPUs in C++, preventing large-scale game simulations. The author's key engineering contribution:
- Re-implemented the core binding calculations using JAX, supporting GPU acceleration
- Leveraged the Miyazawa-Jernigan energy matrix to compute binding energy
- **Pose Pruning**: The Dengue virus has about 1.5 million possible poses, but only 1027 have ever served as the minimum energy pose. Retaining poses with $\ge 18$ residue pairs (approx. 37,000) achieves practically zero loss in accuracy
- Provides both high-resolution and low-resolution simulators: training uses low-resolution, while validation uses high-resolution

### Loss & Training

- **Objective Function**: Maximise the average escape fitness $F_{\hat{v}}^H(a)$; a larger horizon $H$ is closer to the true long-term goal
- **Optimisation Method**: Genetic algorithm (single-point mutation + greedy selection), which is a non-gradient method
- **Computational Budget Trade-off**: Each step of the inner loop requires $O(H \cdot P)$ binding queries; a longer horizon is more accurate but more expensive
- **Validation Strategy**: Use the low-resolution simulator for training, and report results using the high-resolution simulator (simulating transfer from simulation to reality)

## Key Experimental Results

### Main Results

Experiments were conducted on the Dengue virus (PDB: 2R29), with antibody sequence length $N_a = 11$ (CDRH3 region) and virus sequence length $N_v = 97$.

| Metric | Shaper (H=100) | Myopic (H=0) | Key Findings |
|------|---------------|--------------|-------------|
| Escape Average Fitness $F_v^{100}$ | Significantly higher | Lower | Top 10% shapers outperform all myopic |
| Myopic Fitness $R_a(v,a)$ | Slightly lower | Higher | Shapers sacrifice short-term for long-term |
| 10 steps post-escape | Slightly inferior to myopic | Initially better | Myopic holds advantage in the first 10 steps |
| 100 steps post-escape | Significantly better | Drastic decline | Shapers exhibit clear long-term advantages |

**Cross-pathogen generalisability** (4 additional pathogens):

| Pathogen | PDB | Shaping Effect | Specific Discovery |
|--------|-----|-------------|---------|
| West Nile Virus | 1ZTX | ✓ Effective | H=20 performs better under a limited budget |
| Influenza Neuraminidase | 4QNP | ✓ Effective | Trend is consistent with Dengue |
| MERS-CoV | 5DO2 | ✓ Effective | H=100 requires more optimisation steps to converge |
| Clostridioides difficile (Bacteria) | 4NP4 | ✓✓ Particularly strong | H=100 significantly dominates all other configurations |

### Ablation Study

| Configuration | Key Metric $F_v^{100}$ | Description |
|------|---------------------|------|
| H=0 (myopic) | Baseline | Ignores escape, initial binding is good but long-term is poor |
| H=5 | Outperforms myopic | Short horizon already yields improvement |
| H=10 | Further improvement | Medium horizon |
| H=20 | Close to H=100 | **Best value**—nearly identical when computations are normalised |
| H=100 | Optimal | Best when steps are normalised, but computationally expensive |

JAX acceleration effect:

| Implementation | Hardware | Speedup Ratio |
|------|------|--------|
| Original Absolut! (C++) | Apple M2 Max (CPU) | 1× |
| JAX Re-implementation | Nvidia A40 (GPU) | ~10,000× |

### Key Findings

1. **"The best defense is a good offense"**: Cross-evaluation experiments demonstrate that the escape virus $v_{100}$ induced by H=100 shapers is more easily targeted by **all** antibodies (not just the shaping antibody that induced it). This proves that shapers indeed **shape** viral evolution rather than merely becoming more robust.

2. **Amino Acid Distribution Differences**: Shapers have a more uniform amino acid distribution, whereas myopic antibodies tend to cluster around amino acids with extreme binding energies. The uniform distribution makes shapers more robust to mutations—making it harder for the virus to escape by avoiding specific high-binding amino acids.

3. **Pose Matrix Analysis**: H=100 shapers constrain viral escape through two mechanisms: (a) preventing the virus from incorporating the antibody's lowest-binding amino acids (e.g., Lysine) into the pose; (b) discouraging the virus from removing its own high-binding amino acids (e.g., Isoleucine, Methionine) from the pose.

4. **Robustness Under External Pressure**: Even when applying external pressure from additional myopic antibodies (simulating multi-vaccine coexistence scenarios), the shaping effect, though weakened, remains statistically significant.

## Highlights & Insights

- **Cross-Domain Innovation**: Applying opponent shaping concepts from multi-agent RL (LOLA, M-FOS) to computational biology for the first time is an excellent paradigm of AI for Science
- **Outstanding Engineering Contribution**: The JAX re-implementation achieves a 10,000× speedup, rendering large-scale game simulations feasible
- **Practical Guidance**: H=20 is the most cost-effective horizon choice, offering practical recommendations for computationally constrained scenarios
- **Interpretability Analysis**: Not only shows that shapers work, but also deeply analyses **why** they work (amino acid distribution, pose matrix shifts), making the findings biologically interpretable
- **Inspiration for Hybrid Strategies**: Cocktail therapies combining shaping and myopic antibodies might balance short-term efficacy and long-term evolutionary control

## Limitations & Future Work

1. **Simplified Binding Model**: Absolut! is a simplified discrete simulator with a substantial gap from real protein-protein interactions; future work can integrate more precise models like AlphaFold3
2. **Fixed Structure Assumption**: Assumes the viral antigen structure remains static during escape, but actual mutations may lead to conformational changes
3. **Single Antibody Optimisation**: Currently optimises only a single antibody, without considering the synergistic effects of antibody combinations
4. **Sequence Space Constraints**: Only optimises the CDRH3 region (11 amino acids), leaving out other variable regions of the antibody
5. **Distanced from Wet Lab Validation**: More precise simulators and experimental validation are required to move toward clinical applications
6. **Simplified Evolutionary Model**: Assumes an average of 1 mutation per generation and only considers point mutations, lacking recombination, insertions, or deletions

## Related Work & Insights

- **Opponent Shaping Lineage**: LOLA → M-FOS → ADIOS, extending from game-theoretic agents to biomolecular games
- **Viral Escape Prediction**: EVEscape, Han et al. predict variants but do not influence them; ADIOS predicts and guides them simultaneously
- **Antibody Design Methods**: Methods like RAbD and AbDiffuser focus on designing better antibodies, while ADIOS focuses on **how to design under evolutionary feedback**
- **Potential Expansion Areas**: Antimicrobial resistance, monoclonal antibody treatments in cancer (optimising mAbs while shaping tumor cell evolution)

## Rating

- Novelty: ⭐⭐⭐⭐⭐ — First to introduce opponent shaping to antibody design, highly innovative cross-domain contribution
- Experimental Thoroughness: ⭐⭐⭐⭐ — Verified on 5 pathogens with extensive ablations, but lacks wet lab validation
- Writing Quality: ⭐⭐⭐⭐⭐ — Clear framework, elegant illustrations, and solid interpretability analysis
- Value: ⭐⭐⭐⭐ — Although in a proof-of-concept phase, the methodology provides profound inspiration for future antiviral/anticancer therapeutics

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ICLR 2026\] Fine-Tuning Diffusion Models via Intermediate Distribution Shaping](../../ICLR2026/computational_biology/fine-tuning_diffusion_models_via_intermediate_distribution_shaping.md)
- [\[ICML 2026\] CoSiNE: Conditional Site-Independent Neural Evolution Model for Antibody Sequences](../../ICML2026/computational_biology/conditionally_site-independent_neural_evolution_of_antibody_sequences.md)
- [\[ICLR 2026\] Antibody: Strengthening Defense Against Harmful Fine-Tuning for Large Language Models via Attenuating Harmful Gradient Influence](../../ICLR2026/computational_biology/antibody_strengthening_defense_against_harmful_fine-tuning_for_large_language_mo.md)
- [\[ICML 2025\] Supercharging Graph Transformers with Advective Diffusion](supercharging_graph_transformers_with_advective_diffusion.md)
- [\[ICML 2025\] Improving Flow Matching by Aligning Flow Divergence](improving_flow_matching_by_aligning_flow_divergence.md)

</div>

<!-- RELATED:END -->
