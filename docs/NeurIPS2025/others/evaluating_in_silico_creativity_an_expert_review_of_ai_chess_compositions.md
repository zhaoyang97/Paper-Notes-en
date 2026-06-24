---
title: >-
  [Paper Note] Evaluating In Silico Creativity: An Expert Review of AI Chess Compositions
description: >-
  [NeurIPS 2025][Chess puzzles] Google DeepMind trained three generative neural networks (autoregressive Transformer, discrete diffusion, MaskGit) to learn the distribution of chess puzzles, optimized for uniqueness and counter-intuitiveness using reinforcement learning. This process generated approximately 4 million board positions. After filtering via reward functions and aesthetic theme detectors, three world-class chess experts reviewed the compositions…
tags:
  - "NeurIPS 2025"
  - "Chess puzzles"
  - "AI creativity"
  - "Generative models"
  - "Expert review"
  - "Counter-intuitiveness"
date: 2026-05-08
content_hash: 70e8384b2d435696
---

# Evaluating In Silico Creativity: An Expert Review of AI Chess Compositions

**Conference**: NeurIPS 2025  
**arXiv**: [2510.23772](https://arxiv.org/abs/2510.23772)  
**Code**: None  
**Area**: AI Creativity / Generative AI  
**Keywords**: Chess puzzles, AI creativity, Generative models, Expert review, Counter-intuitiveness

## TL;DR

Google DeepMind trained three generative neural networks (autoregressive Transformer, discrete diffusion, MaskGit) to learn the distribution of chess puzzles, optimized for uniqueness and counter-intuitiveness using reinforcement learning. This process generated approximately 4 million board positions. After filtering via reward functions and aesthetic theme detectors, three world-class chess experts reviewed the compositions, providing positive but constructively critical feedback.

## Background & Motivation

**Background**: Generative AI has demonstrated impressive generative capabilities in domains such as text and images, but its capability in structured domains requiring creativity remains controversial. Chess composition is a centuries-old creative human endeavor requiring originality, counter-intuitiveness, and aesthetic elegance.

**Limitations of Prior Work**: Prior AI systems in chess have been primarily used for two tasks: (1) verifying puzzle solutions (using engine validation) and (2) mining interesting positions from existing databases. However, systematically *generating* novel, aesthetically valuable puzzles from scratch remains largely unexplored. Evaluating the "creativity" of AI-generated content also remains a fundamentally open question.

**Key Challenge**: Creativity is highly subjective—even top experts frequently disagree on the quality of a single puzzle. Quantitative metrics (such as engine evaluation) can determine correctness but fail to capture "surprise," "beauty," and "depth." A methodology combining AI generation with expert human review is required to systematically evaluate AI creativity.

**Goal**: Can AI systems generate chess puzzles with aesthetic appeal, counter-intuitive solutions, and creative thematic combinations? How can the creativity of these puzzles be systematically evaluated through expert review?

**Key Insight**: Chess puzzles serve as an ideal vehicle for studying creativity: solutions have objective correctness (verifiable), while aesthetics involve subjectivity (requiring expert judgment). The rules are fully formalized, yet the creative space is vast.

**Core Idea**: Learn puzzle distributions using generative models, optimize for counter-intuitiveness using RL, and evaluate creativity via expert review—establishing a complete closed loop from generation to evaluation.

## Method

### Overall Architecture

The entire system is divided into three stages: training generative models $\rightarrow$ reinforcement learning optimization $\rightarrow$ filtering and expert review. The input is the Lichess dataset containing 4 million chess puzzles, and the output is a curated collection of AI-generated chess puzzles.

### Key Designs

1. **Multi-Model Generation Architecture**:

    - **Function**: Learn the distribution of "what board configurations might constitute good puzzles" from data.
    - **Mechanism**: Three different generative neural networks are trained in parallel. Board positions are encoded as FEN (Forsyth-Edwards Notation) character sequences. The autoregressive Transformer predicts character-by-character $p(c_t | c_1, ..., c_{t-1})$; the discrete diffusion model generates the complete FEN via a denoising process; and MaskGit employs a mask-and-predict strategy. The outputs of all three models are aggregated and fed into the filtering pipeline.
    - **Design Motivation**: Different generative architectures may offer distinct advantages for various puzzle types. Autoregressive models excel at sequential coherence, diffusion models capture global structures, and MaskGit provides advantages in parallel generation.

2. **RL Optimization and Dual Reward Design**:

    - **Function**: Guide the generative models from "mimicking training data" to "generating high-quality puzzles."
    - **Mechanism**: A two-part reward function is designed. (1) **Uniqueness check** (similar to the Lichess approach): ensures that the board position has one and only one winning move, verified by a chess engine; (2) **Counter-intuitiveness check**: ensures that the position is solvable by a strong engine but not by a weak engine—if a weak engine can solve it, the solution is deemed too obvious. The best samples are then selected to iteratively train the networks.
    - **Design Motivation**: Puzzles generated purely through imitation learning tend to represent the "average" of the training data and lack surprise. RL guides the model to explore high-reward regions, generating more challenging and counter-intuitive puzzles.

3. **Hybrid Filtering Pipeline (Reward + Theme Detectors)**:

    - **Function**: Filter out candidates worthy of human review from approximately 4 million generated positions.
    - **Mechanism**: Candidates are first ranked using the reward function and then classified using aesthetic theme detectors (e.g., detecting tactical motifs like sacrifices, pins, and forks). While theme detectors alone are not precise enough, combining them with reward ranking significantly improves performance. The top 50 samples for each theme are manually audited by FIDE players (rated 2200-2300 ELO), and a final curated puzzle set is presented to three experts.
    - **Design Motivation**: Purely automated metrics cannot fully capture aesthetic quality, but they can substantially narrow down the scope of manual evaluation. Hierarchical filtering (automated $\rightarrow$ semi-automated $\rightarrow$ expert) balances efficiency and quality.

### Evaluation Methodology

Three world-class experts were invited: International Master of Chess Composition Amatzia Avni, Grandmaster Jonathan Levitt, and Grandmaster Matthew Sadler. Each independently selected their favorite puzzles and explained their reasoning.

## Key Experimental Results

### Expert Selection Results

| Puzzle | Avni (IM) | Levitt (GM) | Sadler (GM) | Key Features |
|------|-----------|-------------|-------------|----------|
| Puzzle 1 | ✓ | ✓ | ✓ | **The only unanimous choice**: double rook sacrifice + backfield redeployment, with a geometric theme crossing both wings |
| Puzzle 2 | ✓ | | | Long calculation chain, black king actively advances into the danger zone |
| Puzzle 3 | ✓ | | | Counter-intuitive rook sacrifice followed by a quiet move to finish |
| Puzzle 4 | ✓ | | | Underpromotion to a knight (instead of a queen), counter-intuitive |
| Puzzle 5 | | ✓ | | Approaching the level of endgame studies, with natural positions and precise moves |
| Puzzle 6 | | ✓ | | Elegant endgame with precise king moves from Black |
| Puzzle 7 | | | ✓ | Combination of underpromotion to knight + smothered mate themes, never seen before |
| Puzzle 8 | | | ✓ | Surprising stalemate theme where all "winning" moves fail |
| Puzzle 9 | | | ✓ | A new interpretation of classical decoy + smothered mate themes |

### Ablation/Analysis

| Aspect | Positive Feedback | Constructive Criticism |
|------|---------|-----------|
| Aesthetic Motifs | Innovative thematic integration, "game-like" perspective | Some positions are too simple |
| Depth of Solution | Certain counter-intuitive moves are jaw-dropping | Lacks the overall depth and complexity of traditional endgame studies |
| Naturalness of Positions | Sadler highly values natural positions | Unrealistic piece placement in certain positions |
| Creativity | Novel combinations of themes (e.g., knight promotion + smothered mate) | Needs more complex variations and stronger counter-play lines |

### Key Findings

- **Experts rarely reach consensus**: Among the 9 curated puzzles, only Puzzle 1 received unanimous votes. The three experts selected different puzzles, highlighting the high subjectivity of creativity and aesthetics.
- **Counter-intuitiveness is the most favored trait**: Almost all selected puzzles contain counter-intuitive key moves (e.g., double rook sacrifice, underpromotion to a knight instead of a queen, or seemingly failing moves).
- **"Game-likeness" is crucial**: Sadler particularly emphasized that positions should resemble those arising in actual play; unnatural piece placements detract from the puzzle's appeal.
- **AI demonstrates capability in combining themes**: The combination of underpromotion to a knight and a smothered mate in Puzzle 7 was described by Sadler as "never seen before," showing that the AI can discover thematic intersections that human composers might overlook.

## Highlights & Insights

- **Innovative Evaluation Methodology**: A complete closed loop from AI generation to expert review was established to evaluate AI creativity. This framework can be mapped onto other creative domains (e.g., music, mathematical conjectures, drug design)—with the key requirement being an objective correctness criterion paired with subjective aesthetic evaluation.
- **The Ingenuity of "Weak Engine Filtering"**: Counter-intuitiveness is operationally defined as "solvable by a strong engine but not by a weak engine," which is simple and highly effective. Analogously in other domains, a good mathematical problem might be one "solvable by experts but not by amateurs"—providing a generalizable path for automated filtering of creative outputs.
- **Emergence of Thematic Combinations**: The AI does not merely replicate patterns from the training data but combines distinct themes (e.g., underpromotion + smothered mate) to yield results that human experts consider novel. This suggests the potential of generative models in making creative combinations.

## Limitations & Future Work

- **Insufficient Depth**: The experts consistently noted that the AI-generated puzzles lack the depth found in traditional endgame studies—variations are not complex enough, and alternative defensive lines are scarce. Future work needs to optimize the reward function to encourage deeper search trees.
- **Naturalness of Positions**: Unrealistic piece placements in some generated positions (e.g., highly misplaced pieces) diminish their aesthetic value. Incorporating "positional naturalness" as an additional reward signal, or sampling initial states from real-world games, may resolve this.
- **Limited Scale of Review**: The evaluation relied on only three experts reviewing a highly filtered selection (~50 candidates down to 9 curated puzzles from 4 million initial positions). A larger-scale evaluation with a more diverse pool of reviewers might reveal different patterns.
- **Lack of Quantitative Creativity Metrics**: The study relies primarily on qualitative expert feedback. While reasonable (since creativity is inherently hard to quantify), future work could explore more structured scoring frameworks.
- **No Direct Comparison with Human Composers**: There is a lack of blind test experiments comparing human-authored and AI-generated compositions.

## Related Work & Insights

- **vs AlphaZero/Stockfish**: Traditional chess AIs focus on the optimal moves in a game. This work focuses on *generating* interesting positions—where the goal is not winning games, but creating beauty. This marks a paradigm shift for AI from "solving problems" to "posing problems."
- **vs Image/Text Generative AI**: Evaluation of creativity in models like DALL-E or ChatGPT generally lacks an objective correctness criterion. The advantage of chess puzzles lies in the existence of a unique correct solution (verifiable), paired with subjective aesthetics (requiring expert judgment). Combining both makes it an ideal testing ground for creativity research.
- **vs Mathematical Conjecture Generation (AlphaProof direction)**: A similar "generation + verification + filtering" framework. The commonality lies in the vast generation space but highly sparse high-quality outputs, which necessitates automated filtering to focus human attention. The work on chess compositions may provide methodological insights for AI creativity evaluation in mathematical domains.

## Rating

- Novelty: ⭐⭐⭐⭐ First systematic study of AI chess puzzle composition and expert evaluation, providing a valuable methodological reference.
- Experimental Thoroughness: ⭐⭐⭐ Expert review is of high quality but limited in scale, lacking quantitative metrics and direct comparison with humans.
- Writing Quality: ⭐⭐⭐⭐ Excellent presentation of puzzles, vivid and engaging expert commentaries, though technical details are deferred to an auxiliary technical paper.
- Value: ⭐⭐⭐⭐ Establishes a reference methodological framework for evaluating AI creativity, offering strong insights beyond the domain of chess.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2025\] Completing A Systematic Review in Hours instead of Months with Interactive AI Agents](../../ACL2025/others/completing_a_systematic_review_in_hours.md)
- [\[NeurIPS 2025\] Military AI Needs Technically-Informed Regulation to Safeguard AI Research and its Applications](military_ai_needs_technically-informed_regulation_to_safeguard_ai_research_and_i.md)
- [\[NeurIPS 2025\] Emergency Response Measures for Catastrophic AI Risk](emergency_response_measures_for_catastrophic_ai_risk.md)
- [\[ACL 2025\] Evaluating the Evaluation of Diversity in Commonsense Generation](../../ACL2025/others/evaluating_the_evaluation_of_diversity_in_commonsense_generation.md)
- [\[NeurIPS 2025\] Fostering the Ecosystem of AI for Social Impact Requires Expanding and Strengthening Evaluation Standards](fostering_the_ecosystem_of_ai_for_social_impact_requires_expanding_and_strengthe.md)

</div>

<!-- RELATED:END -->
