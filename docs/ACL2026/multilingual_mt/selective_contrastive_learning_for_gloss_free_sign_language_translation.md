---
title: >-
  [Paper Note] Selective Contrastive Learning For Gloss Free Sign Language Translation
description: >-
  [ACL2026][Multilingual & Machine Translation][Gloss-free Sign Language Translation] This paper discovers that random in-batch negatives in sign language translation (SLT) are often unreliable or semantically conflicting…
tags:
  - "ACL2026"
  - "Multilingual & Machine Translation"
  - "Gloss-free Sign Language Translation"
  - "Selective Contrastive Learning"
  - "Negative Selection"
  - "Curriculum Learning"
  - "Cross-modal Alignment"
date: 2026-05-08
content_hash: 4ab7f80f78534772
---

# Selective Contrastive Learning For Gloss Free Sign Language Translation

**Conference**: ACL2026  
**arXiv**: [2604.22374](https://arxiv.org/abs/2604.22374)  
**Code**: Not yet released  
**Area**: Multimodal VLM / Sign Language Translation  
**Keywords**: Gloss-free Sign Language Translation, Selective Contrastive Learning, Negative Selection, Curriculum Learning, Cross-modal Alignment

## TL;DR
This paper discovers that random in-batch negatives in sign language translation (SLT) are often unreliable or semantically conflicting supervisory signals. Consequently, it utilizes similarity trajectories from a reference model to filter informative negative samples and improves gloss-free SLT quality through an easy-to-hard curriculum contrastive learning approach.

## Background & Motivation
**Background**: Gloss-free SLT directly maps continuous sign language videos to natural language sentences without relying on word-level gloss annotations. Recent strong methods typically incorporate CLIP-style vision-language pre-training between the video and text encoders to pull matching video-text pairs closer and push non-matching pairs apart before feeding the aligned visual representation into a translation decoder.

**Limitations of Prior Work**: Standard contrastive learning treats all texts in a mini-batch other than the positive instance as negative samples. However, due to the high computational cost of sign language video modeling, batch sizes are often small, leading to low coverage of negative samples per update. More critically, in narrow-domain datasets like PHOENIX14T and datasets with redundant target sentences like CSL-Daily, many "negatives" are semantically similar or even textually identical. Forcing them apart introduces conflicting supervision for cross-modal alignment.

**Key Challenge**: SLT requires stronger video-text alignment, yet random negative samples suffer from both insufficient coverage and false negatives. Simply increasing the batch size or adding external annotations increases costs, while continuing to rely on random contrastive learning pushes away samples that should ideally be close.

**Goal**: The authors aim to answer two questions: first, whether in-batch negative samples are actually effectively pushed away during training; and second, whether training dynamics themselves can be used to select more valuable negatives to improve alignment without additional annotations or LLM assistance.

**Key Insight**: The authors first train a standard contrastive model and track the similarity trajectories of all video-text negative pairs every 5 epochs. They observe that only 35.9% of negative samples follow the ideal trend of "high similarity to low similarity," while 31.9% remain highly similar long-term, and 17.8% even become more similar as training progresses, indicating highly non-uniform negative sample contributions.

**Core Idea**: Use the similarity variation from a reference contrastive model to score negatives, then employ curriculum learning to transition from stably separable negatives to harder-to-distinguish negatives, replacing completely random in-batch contrast.

## Method

### Overall Architecture
The SCL-SLT workflow consists of three steps. First, standard CLIP-style training is performed on sign videos and target texts to train a reference contrastive model, with multiple checkpoints saved. Second, these checkpoints are used to calculate the similarity changes of arbitrary video-text negative pairs, using the "how it changes during training" as a proxy for negative difficulty and informativeness. Third, when training the target SLT model, mini-batches are no longer composed randomly but constructed according to a curriculum ratio to include specific negative structures. Selective contrastive training is performed first, followed by translation fine-tuning.

The model itself includes a Sign Embedding, Visual Encoder, Text Encoder, and Decoder. Sign Embedding extracts spatial features using an ImageNet-pre-trained ResNet-18, followed by two Conv1D/BN/ReLU layers for temporal modeling. The Visual Encoder, Text Encoder, and Decoder are initialized from mBART-large-50; the text encoder is frozen during the contrastive phase, while the visual encoder and decoder are adapted via LoRA. The alignment stage uses CiCo-style fine-grained video-text similarity to compute bidirectional contrastive loss while retaining translation loss. The translation fine-tuning stage disconnects the text branch and optimizes only for the auto-regressive translation objective.

### Key Designs
1.  **Trajectory-based Negative Scoring**:
    - **Function**: Transitions negative sampling from "learning whatever is encountered randomly" to "active selection based on training dynamics."
    - **Mechanism**: For video $V_i$ and non-matching text $T_j$, the trajectory change of the negative pair is represented by the similarity difference between early and late checkpoints of the reference model: $\delta_{i,j}=\hat{s}^{K}(V_i,T_j)-\hat{s}^{0}(V_i,T_j)$. A batch score is the sum of variations of all off-diagonal video-text negative pairs. Intuitively, H→L negatives indicate the model can successfully push them apart, L→H negatives indicate increasing difficulty in distinguishing them, and H→H may represent semantic neighbors or false negatives.
    - **Design Motivation**: Standard contrastive learning only considers instantaneous similarity in the current batch, failing to distinguish "truly hard" from "should not be pushed away." Introducing training history via trajectory signals reduces updates dominated by noisy negatives.

2.  **Curriculum Pair Selection**:
    - **Function**: Constructs mini-batches that match the current training stage during each epoch.
    - **Mechanism**: The algorithm first randomly selects a positive example as a batch seed, then calculates the incremental score $\Delta(s_u;\mathcal{C})$ for candidate samples. Candidates are sorted by score, and samples are selected according to the curriculum ratio $\alpha=e/E$ based on quantiles. Early training favors negatives that are easier to separate, while later training shifts toward higher similarity or harder-to-distinguish negatives.
    - **Design Motivation**: If only the hardest negatives are fed initially, the model may be confused by false negatives and semantic neighbors; if only easy samples are fed throughout, alignment remains superficial. Curriculum scheduling allows the model to learn stable boundaries before handling fine-grained differences.

3.  **Decoupling Selective Contrastive Training and Translation Fine-tuning**:
    - **Function**: Prioritizes cross-modal alignment learning before focusing on translation generation.
    - **Mechanism**: The selective contrastive stage computes CLIP-style losses in both video-to-text and text-to-video directions, retaining a translation loss with a weight of 1.0 to ensure representations do not exclusively serve retrieval. Subsequently, the alignment module and auxiliary text encoder are removed in the SLT fine-tuning stage, where the decoder generates target sentences purely based on video representations.
    - **Design Motivation**: Appendix results show that incorporating standard contrastive objectives directly into end-to-end translation can significantly degrade performance; treating contrastive learning as a pre-training alignment stage followed by fine-tuning better aligns with the distinct goals of representation learning and sentence generation.

### Loss & Training
Training is divided into two stages: Stage 1 involves 80 epochs of selective contrastive training, and Stage 2 involves 200 epochs of translation fine-tuning. The optimizer is AdamW with a learning rate of $1\times10^{-4}$, cosine decay, and label smoothing of 0.2. Batch sizes for Stage 1 and Stage 2 are 16 and 8, respectively. Inference uses a beam size of 8. LoRA is applied to the visual encoder and decoder with a rank of 16 and scale of 32; the text encoder is frozen to provide a stable language prior.

## Key Experimental Results

### Main Results
The paper reports ROUGE and BLEU on two gloss-free SLT benchmarks: PHOENIX14T and CSL-Daily. PHOENIX14T contains 7,096/519/642 training/validation/test samples, and CSL-Daily contains 18,401/1,077/1,076 samples.

| Setting | Method | PHOENIX14T R | PHOENIX14T B4 | CSL-Daily R | CSL-Daily B4 |
|------|------|--------------|---------------|-------------|--------------|
| w/o VLP | SignLLM | 44.49 | 23.40 | 39.91 | 15.75 |
| w/o VLP | Ours (SCL-SLT) | 46.33 | 25.30 | 48.53 | 21.41 |
| w/ VLP | LLAVA-SLT | 50.44 | 23.43 | 51.26 | 20.42 |
| w/ VLP | C2RL | 50.96 | 26.75 | 48.21 | 21.61 |
| w/ VLP | MMSLT | 47.97 | 25.73 | 48.92 | 21.11 |
| w/ VLP | Ours (SCL-SLT) | 47.02 | 26.00 | 51.08 | 23.25 |

Ours (SCL-SLT) shows the most significant improvement on CSL-Daily, achieving a BLEU-4 of 23.25 under the w/ VLP setting, which is 1.64 higher than C2RL (21.61) and 2.83 higher than LLAVA-SLT (20.42). Although it does not outperform C2RL's 26.75 on PHOENIX14T, achieving 26.00 without complex auxiliary tasks demonstrates that negative selection alone provides strong alignment gains.

### Ablation Study

| Configuration | PHOENIX14T R | PHOENIX14T B4 | CSL-Daily R | CSL-Daily B4 | Description |
|------|--------------|---------------|-------------|--------------|------|
| BaseLine End-to-End | 41.81 | 21.97 | 41.04 | 16.31 | Direct translation training |
| w/ CL | 43.55 | 22.03 | 47.77 | 20.59 | Standard random contrastive learning |
| w/ SCL | 46.33 | 25.30 | 48.53 | 21.41 | Added selective negatives |
| CL-SLT | 46.13 | 25.01 | 48.34 | 20.70 | Standard CL pre-training then fine-tuning |
| SCL-SLT | 47.02 | 26.00 | 51.08 | 23.25 | Selective CL pre-training then fine-tuning |

| Analysis Item | Setting | PHOENIX14T B4 | Key Conclusion |
|--------|------|---------------|----------|
| Curriculum Schedule | Hard-Only | 12.41 | Only using hard negatives severely disrupts training |
| Curriculum Schedule | Easy-Only | 24.00 | Stable but lacks late-stage fine-grained challenges |
| Curriculum Schedule | Linear | 24.59 | Dynamic curriculum outperforms static strategies |
| Curriculum Schedule | Sqrt | 24.54 | Close to Linear |
| Curriculum Schedule | Log | 25.30 | Optimal, indicates smoother difficulty transition is more stable |
| Trajectory Interval | 1 epoch | 24.68 | Over-sampling introduces training noise |
| Trajectory Interval | 5 epochs | 25.30 | Best balance between trend and noise |
| Trajectory Interval | 10 epochs | 6.26 | Under-sampling misjudges negative dynamics |

### Key Findings
- Standard CL significantly improves results on CSL-Daily but shows almost no gain on PHOENIX14T because weather-domain texts are highly homogeneous, making random negatives more likely to include semantic neighbors. SCL's more pronounced improvement on PHOENIX14T indicates it effectively addresses the false negative problem.
- In CSL-Daily, many target sentences correspond to multiple videos; textual redundancy makes false negatives very prominent. Pair Selection explicitly avoids some of these conflicts.
- CiCo aggregation far outperforms CLS Pooling and Mean Pooling. PHOENIX14T BLEU-4 is 25.30 compared to 14.85 for CLS and 12.44 for Mean Pooling, indicating that fine-grained cross-modal aggregation is a necessary foundation for selective contrastive learning.

## Highlights & Insights
- The strength of this paper lies in reframing "whether a negative is useful" from a static semantic similarity problem to a training dynamics problem. Similarity trajectories naturally contain information about whether the model can learn to distinguish a negative pair, making them better suited for curriculum learning than single similarity snapshots.
- The method does not rely on external glosses, action descriptions, or LLMs, but instead mines cleaner contrastive signals from within existing training data. This is valuable for sign language data where annotations are scarce.
- Contrastive learning is not always "the more the better" in generation tasks. The results suggest that cross-modal alignment and translation goals are best handled in stages; otherwise, alignment loss may suppress generation capacity.
- The Pair Selection concept can be transferred to image-text retrieval, video captioning, and medical report generation—essentially any scenario where semantic neighbors or duplicate texts exist and random in-batch negatives cause issues.

## Limitations & Future Work
- The method requires training a reference contrastive model beforehand, increasing computational overhead during data preparation. The authors suggest considering off-the-shelf pre-trained models to approximate semantic similarity to simplify the process.
- Negative selection still depends on internal training set dynamics and cannot fundamentally identify all semantically equivalent samples. If the data annotations are noisy, trajectory signals might be affected by incorrect target sentences.
- Experiments are focused on PHOENIX14T and CSL-Daily, both relative controlled datasets. Real-world sign language scenarios involving dialects, signer variations, occlusions, and non-standard expressions may present more complex cross-domain issues.
- The current curriculum strategy uses hand-crafted quantile scheduling; future work could explore adaptive scheduling based on validation alignment or translation quality.

## Related Work & Insights
- **vs GFSLT-VLP**: While GFSLT-VLP introduced CLIP-style pre-training to gloss-free SLT, this work points out the unstable quality of standard in-batch negatives and improves negative construction within the same framework.
- **vs CiCo**: CiCo improves video-text similarity aggregation. SCL-SLT adopts CiCo for computing the similarity matrix, but its core contribution is the selection of samples participating in contrast.
- **vs LLAVA-SLT / SignLLM**: These methods emphasize leveraging LLM language priors. SCL-SLT demonstrates that strong results can be achieved strictly by optimizing internal data signals without external LLM assistance.
- **vs C2RL / MMSLT**: C2RL and MMSLT rely on auxiliary objectives or descriptive supervision. The advantage of this work is its simplicity and plug-and-play nature, while the disadvantage is the requirement for extra reference model training.

## Rating
- Novelty: ⭐⭐⭐⭐ Trajectory-driven pair-level negative curriculum learning is highly targeted and the problem definition is clear.
- Experimental Thoroughness: ⭐⭐⭐⭐ Covers curriculum scheduling, aggregation methods, and sampling intervals, though cross-domain and real-world validation is limited.
- Writing Quality: ⭐⭐⭐⭐ Motivation and trajectory analysis are clear, though some tables appear crowded in HTML format.
- Value: ⭐⭐⭐⭐⭐ Directly instructive for SLT and other cross-modal tasks suffering from false negatives.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[ACL 2026\] Think in Latent Thoughts: A New Paradigm for Gloss-Free Sign Language Translation](think_in_latent_thoughts_a_new_paradigm_for_gloss-free_sign_language_translation.md)
- [\[ACL 2026\] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning](clewr_curriculum_learning_with_restarts_for_machine_translation_preference_learn.md)
- [\[ACL 2026\] NeoAMT: Neologism-Aware Agentic Machine Translation with Reinforcement Learning](neoamt_neologism-aware_agentic_machine_translation_with_reinforcement_learning.md)
- [\[ACL 2026\] Reinforcement Learning with Semantic Rewards Enables Low-Resource Language Expansion without Alignment Tax](reinforcement_learning_with_semantic_rewards_enables_low-resource_language_expan.md)
- [\[ACL 2026\] From Fragments to Facts: A Curriculum-Driven DPO Approach for Generating Hindi News Veracity Explanations](from_fragments_to_facts_a_curriculum-driven_dpo_approach_for_generating_hindi_ne.md)

</div>

<!-- RELATED:END -->
