---
title: >-
  [Paper Note] CleverBirds: A Multiple-Choice Benchmark for Fine-grained Human Knowledge Tracing
description: >-
  [NeurIPS 2025][Self-Supervised Learning][Knowledge Tracing] Introduces CleverBirds—the largest visual knowledge tracing benchmark to date, collecting 17M+ multiple-choice bird species identification questions answered by 40,000+ participants via the eBird citizen-science platform (covering 10,000+ species), systematically evaluating diverse knowledge tracing and classification methods, and revealing core challenges in fine-grained visual knowledge modeling, particularly in predicting learners' incorrect answer choices.
tags:
  - "NeurIPS 2025"
  - "Self-Supervised Learning"
  - "Knowledge Tracing"
  - "Fine-grained Recognition"
  - "Benchmark Dataset"
  - "Visual Learning"
  - "Human Cognitive Modeling"
  - "Bird Identification"
date: 2026-05-08
content_hash: e0e8a222e9bbce98
---

# CleverBirds: A Multiple-Choice Benchmark for Fine-grained Human Knowledge Tracing

**Conference**: NeurIPS 2025  
**arXiv**: [2511.08512](https://arxiv.org/abs/2511.08512)  
**Code**: [https://cleverbirds-benchmark.github.io](https://cleverbirds-benchmark.github.io)  
**Area**: Video Understanding  
**Keywords**: Knowledge Tracing, Fine-grained Recognition, Benchmark Dataset, Visual Learning, Human Cognitive Modeling, Bird Identification

## TL;DR

Introduces CleverBirds—the largest visual knowledge tracing benchmark to date, collecting 17M+ multiple-choice bird species identification questions answered by 40,000+ participants via the eBird citizen-science platform (covering 10,000+ species), systematically evaluating diverse knowledge tracing and classification methods, and revealing core challenges in fine-grained visual knowledge modeling, particularly in predicting learners' incorrect answer choices.

This paper proposes CleverBirds, the largest visual knowledge tracing benchmark to date. By collecting over 17 million multiple-choice bird species identification questions and answers completed by more than 40,000 participants (covering over 10,000 species) via the eBird citizen science platform, it systematically evaluates various knowledge tracing and classification methods, revealing the core challenges in fine-grained visual knowledge modeling, particularly in predicting learners' incorrect choices.

## Background & Motivation

### Key Challenges in Knowledge Tracing

Knowledge Tracing (KT) aims to model learners' knowledge states and predict their future performance by observing their interaction processes with instructional materials, serving as a core component for building effective automated tutoring systems. Early methods modeled learners' degree of knowledge mastery based on latent variable probabilistic models. Although deep learning methods in recent years can capture more complex relationships, they require a massive amount of training data. However, existing knowledge tracing datasets are primarily concentrated in a few domains such as mathematics, programming, and language learning, leaving a severe deficiency in the visual recognition field.

### The Gap in Visual Knowledge Tracing

Many professional domains (e.g., medicine, art, biology) require learners to master visual recognition skills, where the tasks are inherently classification problems—learners must learn the decision boundaries between different visual concepts. Taking bird identification as an example, there are over 11,000 different bird species globally, and distinguishing between certain species requires extremely fine-grained visual judgment. Existing visual knowledge tracing datasets are very small in scale: the gravitational wave classification data from [Crowston et al., 2019] contains only 21 categories, and each of the three datasets from [Lee et al.] has only 6,750 interactions and fewer than 1,000 images. These datasets fall far short of research needs in terms of scale, number of concepts, and depth of interaction.

### Positioning of CleverBirds

CleverBirds fills this gap. Compared with existing image-based knowledge tracing datasets, CleverBirds possesses the following unique advantages:

1. **Extremely large concept space**: 10,779 bird species, far exceeding the number of concepts in any existing KT dataset.
2. **A large number of participants with diverse skill levels**: Over 40,000 real volunteers, rather than short-term participants from crowdsourcing platforms.
3. **Long interaction sequences**: Participants answered an average of 400 questions, with over 50% of users answering more than 100 questions.
4. **Realistic learning dynamics**: Significant progress of participants over time can be observed.

## Method

### Dataset Construction

#### Quiz Design

The CleverBirds dataset is derived from the online bird species identification quiz of eBird (a citizen science project created by the Cornell Lab of Ornithology). The quiz was first launched in March 2018, with the primary intention of encouraging users to provide quality ratings for newly uploaded images. The specific process is as follows:

1. **Parameter selection**: Users select parameters such as geographical location, time of year, and species frequency of occurrence to generate quiz questions. For example, selecting "Edinburgh, Scotland, May 15, common species" will only present questions regarding common bird species in that area during that period.
2. **Each quiz round**: Consists of 20 multiple-choice questions, each displaying a bird image and 5 options (4 species names + a "none of the above" [NOTA] option).
3. **Answer option design**: To increase difficulty, candidate answers are drawn from a sliding window over a taxonomic list centered on the correct species, ensuring that distractors are taxonomically similar to the correct answer.
4. **Feedback mechanism**: After submitting the answer, the system reveals the correct species, and users are asked to rate the image quality on a scale of 1 to 5.
5. **Image source**: Images are sourced from photos uploaded by citizen scientists to the Macaulay Library. Their species labels are cross-validated against expected geographical distributions, and inconsistencies are reviewed by expert auditors.

#### Image Quality Control

Quiz images are sampled from those uploaded within the last 5 to 365 days, containing a certain proportion of unrated images and images with quality scores $\ge 2.4$ (out of 5). This design obtains quality annotations for unrated images through the quiz while retaining enough high-quality images for user learning. Criteria for image quality include sharpness, bird visibility, photo size, and watermarks.

#### Data Filtering and Splitting

The dataset is based on all online quizzes between March 14, 2018, and October 8, 2024. To prevent overfitting to specific users, the dataset is split by user ID:

| Dataset | Users | Share of Interactions |
|--------|--------|-----------|
| Train | 28,100 | 70.6% |
| Val | 6,021 | 14.6% |
| Test | 6,023 | 14.8% |

Due to image copyright restrictions, the dataset does not provide raw images; instead, feature embeddings are provided for each image:
- **DINOv2** (ViT-B/14 backbone): Average pooling is applied to the patch tokens after the last LayerNorm layer (excluding special tokens) to obtain 14,747,840 feature vectors.
- **ResNet-50** (ImageNet pre-trained): Using the output of the global average pooling layer prior to the classifier to obtain 14,753,114 feature vectors.

The difference of 5,274 between the two is because DINOv2 feature extraction was conducted later, and some images had been deleted by their owners.

#### Privacy Protection

The dataset anonymizes all user-related identifiers, as well as quiz, question, and image asset identifiers. Quiz locations are aggregated at resolution 3 using the H3 geospatial index, with an average area of approximately 12,393 km² per cell. The project has been approved by the Ethics Committee of the School of Informatics, University of Edinburgh (Project ID 954242).

### Dataset Feature Analysis

The scale and complexity of CleverBirds are unprecedented in the knowledge tracing field. Below is a detailed analysis of its core statistical characteristics:

**Data Scale and Composition**:

| Statistical Metric | Value | Description |
|---------|------|------|
| Total interactions | 17,859,392 | Far exceeding any existing visual KT dataset |
| Unique image-species-options combinations | 98% | Rarely repeated question combinations |
| Unique species-options pairs | 26% | Diverse distractor variations |
| Image uniqueness | 83% | The vast majority of images are never shown repeatedly |
| Number of species | 10,779 | Covering a broad scope of global bird taxonomy |
| Number of geographic locations | 4,000+ | Locations chosen for quizzes across the globe |
| Temporal coverage | 52 weeks a year | Complete temporal distribution across all seasons |
| Number of DINOv2 feature vectors | 14,747,840 | Extracted by the ViT-B/14 backbone |
| Number of ResNet-50 feature vectors | 14,753,114 | Extracted by ImageNet pre-training |

The unique feature of this dataset lies in the highly non-repetitive nature of its interaction data—98% of interactions involve unique image-species-options combinations, which means models cannot simply memorize the answer patterns of specific combinations but must learn to model deeper cognitive processes.

#### User Engagement Analysis

The distribution of user engagement exhibits a typical long-tail characteristic, but the proportion of "heavy users" in the tail is quite considerable:

- Over 50% of the 40,000+ users answered more than 100 questions, providing long enough interaction sequences to analyze learning dynamics.
- Over 10% of users answered more than 1,000 questions, constituting extremely valuable deep learning trajectory data.
- Users encountered an average of 138 different species, with 10% of users encountering more than 300, covering a highly diverse concept space.
- User accuracy is widely distributed, centering around 60-70%, ranging from novices with accuracy close to 30% to veteran birders with accuracy exceeding 90%.
- Average accuracy at the species level is also widely distributed, reflecting large differences in visual distinguishability among various species.

This diverse engagement pattern provides ideal conditions for studying cognitive differences among learners of different skill levels—ranging from novices who have just started bird identification to quasi-expert participants who have accumulated thousands of interactions.

#### Learning Dynamics

The data clearly demonstrates learning progress, which is a key feature distinguishing CleverBirds from static classification benchmarks:

- During the first 20 exposures to a specific species, user accuracy increased by an average of 20%, an improvement that is highly statistically significant.
- Over half of the users demonstrated measurable progress within a sliding window of 20 questions (one complete quiz).
- Even for low-quality images (blurry, partially occluded, etc.), the average accuracy of users remained above 50%.
- The shape of the learning curve varies by species—common and visually distinct species are learned faster, while improving the ability to distinguish between pairs of highly similar-looking species takes longer.
- Validation of feedback effects: Studies show that providing only the correct species label as feedback is sufficient for learners to achieve considerable progress in fine-grained classification tasks, which aligns with previous findings that "humans can acquire visual expertise solely through label supervision."

These learning dynamics characteristics make CleverBirds much more than a static prediction benchmark; it serves as a "laboratory" for studying the acquisition process of human visual skills.

#### Task Difficulty

The dataset reflects task difficulty through the most commonly confused species pairs. For example:

- **Swinhoe's Snipe vs. Common Snipe**: The distinction lies in the characteristic white trailing edge on the wings of the Common Snipe.
- **Sharp-shinned Hawk vs. Cooper's Hawk**: The Sharp-shinned Hawk has a smaller head, a more squared tail, and smaller feet.
- **Pine Grosbeak (pale morph) vs. Pine Grosbeak (normal morph)**, **Ross's Goose vs. Snow Goose**, etc.

The differences between these species pairs are extremely subtle and require trained observation skills to distinguish. More importantly, images in the quizzes may contain partial occlusions, be shot from a distance, or features unusual angles, further increasing the difficulty.

### Problem Formulation

#### Core Definition

At time step $t \in \{1, \dots, T\}$, a learner is presented with an image $I_t$ and an ordered candidate answer list $\mathbf{c}_t = (c_{t,1}, \dots, c_{t,K-1}, \text{NOTA})$, which contains $K-1$ randomly ordered candidate species and a "none of the above" (NOTA) option.

The image is represented as $\mathbf{x}_t = f(I_t) \in \mathbb{R}^d$ through a fixed visual encoder. After observing the image, the learner selects a response $r_t \in \mathbf{c}_t$ based on their internal state, and then receives the true species label $y_t$ as feedback. A single interaction is denoted as $h_t = (\mathbf{x}_t, \mathbf{c}_t, y_t, r_t)$.

#### Learner Model

The learner's response is determined by their unobservable internal state $\theta_t$ (which summarizes their accumulated knowledge and memory), along with the input image and candidate options. Assuming the learner's state updates after each interaction, the learner's response process is modeled as:

$$r_t = \arg\max_{c \in \mathbf{c}_t} P(c \mid \mathbf{x}_t, \mathbf{c}_t, \theta_t)$$

where $r_t^{\text{bin}} = \mathbb{I}[y_t = r_t] \in \{0, 1\}$ represents whether the answer is correct.

#### Approximation Model

Since $P(\cdot)$ is unobservable, it is approximated by a shared parameter model $\phi$ trained across learners. $\phi$ does not contain learner-specific parameters; learner-specific behavior is realized by conditioning on the recent interaction history of each individual $\mathcal{H}_t = (h_\tau)_{\tau=\max(1, t-W)}^{t-1}$ and the current question. The model predicts the classification outcome:

$$\hat{r}_t = \arg\max_{c \in \mathbf{c}_t} \phi(c \mid \mathbf{x}_t, \mathbf{c}_t, y_t, \mathcal{H}_t)$$

The context can be divided into three categories:
- **User Context (U)**: Individual interaction history, historical performance, and preferences.
- **Species Context (S)**: Species features aggregated from the training set (e.g., average difficulty).
- **Image Context (Img)**: Extracted image features, implicitly encoding information such as quality and ambiguity.

### Evaluation Metrics

**Binary Classification Task** (predicting correct/incorrect):
- **Binary Macro Accuracy**: The macro-average accuracy of the correct and incorrect classes.
- **Binary AP**: Average precision with the minority class (incorrect) as the positive class.

**Multiple Choice Task** (predicting specific choices):
- **Multiple Choice Accuracy**: Accuracy on labels 1-5 (where 5 denotes "none of the above").
- **Multiple Choice Incorrect Set Accuracy**: Accuracy calculated exclusively on the subset of questions where participants answered incorrectly.

### Baseline Methods

#### Multiple Choice Classifier

1. **Heuristics**:
    - **Always Correct**: Assumes an omniscient learner, always choosing the correct answer.
    - **Random**: Random guessing.

2. **Confusion Prior Classifier (Conf Prior)**: Estimates the probability of each option based on the confusion probability between correct species and distractors in the training set, which is then re-normalized to form a valid distribution. There is also a regulated version (Conf Prior Inc), which restricts predictions to always be incorrect choices.

3. **MLP**: A single-layer MLP taking the 250-dimensional embedding of the correct species and optional contexts (User/Species/Image) as input. It outputs a probability distribution over all species, which is then masked to contain only the 5 presented options. Image features are mapped to the MLP input dimension via an embedding layer.

4. **Knowledge Tracing Models**:

   | Model | Architecture | Key Characteristics |
   |------|------|---------|
   | simpleKT | Lightweight baseline | Simple and efficient |
   | KQN (Knowledge Query Networks) | Lightweight baseline | Query-based |
   | DKT (Deep Knowledge Tracing) | LSTM | Hidden states model history |
   | DKT+ | LSTM + regularization | Improved version of DKT |
   | ATKT | LSTM + adversarial training | Adversarial enhancement |
   | SAKT | Self-attention | Selectively attends to relevant history |
   | AKT | Attention + Rasch regularization | Regularization inspired by the Rasch model |
   | DKVMN | Key-Value Memory Networks | Externalizes knowledge into key-value memory |

5. **Language Modeling Paradigm**:
    - **LM-Seq2seq**: Fine-tunes a T5-style encoder-decoder transformer to generate the correct species token based on interaction history and the current question. Token-level cross-entropy loss is used during training, and NOTA is handled by aggregating all probabilities outside the visible options.
    - **LM-MCC**: Fine-tunes a BERT-style encoder transformer to score each question-option pair, supervised using binary labels (exactly one correct answer per question).

   Both models use a custom tokenizer (containing only 11,142 species tokens plus special tokens such as padding, segment separators, and type markers), with each question occupying exactly 8 tokens. The history window length is set to $W=50$, while other models use the full history. This compact tokenization design avoids potential ambiguity issues that could arise when using natural language descriptions for species names.

#### Binary Classifier

- **Logistic Regression (LR)**
- **XGBoost (XG)**
- **Random Forest (RF)**
- **Average Species** (average species accuracy baseline from training data)

Each model uses different context combinations (U, S, U+S).

## Key Experimental Results

### Main Results — Multiple Choice Task

| Model | Context | Full Dataset Accuracy | Incorrect Subset Accuracy |
|------|--------|--------------|--------------|
| Always Correct | - | ~Upper Bound | 0% |
| Random | - | ~20% | ~20% |
| Conf Prior | S | <9% | ~23% |
| Conf Prior Inc | S | ~10% | ~24% |
| MLP | U+S | ~70% | <25% |
| MLP | U+S+Img (DINOv2) | **~76%** | **~25%** |
| MLP | Img only | ~76% | ~11% |
| LM-MCC | U+S | ~70% | <25% |
| LM-Seq2seq | U+S | ~70% | <25% |
| AKT | U | ~70% | <25% |
| ATKT | U | ~70% | <25% |
| simpleKT | U | ~70% | <25% |

### Main Results — Binary Classification Task  

| Model | Context | Macro Accuracy | Average Precision (AP) |
|------|--------|------------|--------------|
| Avg Species | S | <70% | <60% |
| RF | S | ~72% | ~70% |
| RF | U | ~78% | ~78% |
| RF | **U+S** | **~80%** | **>80%** |
| KT methods (AKT, etc.) | U | ~54% | ~35% |

### Key Findings

#### Finding 1: Context Feature Engineering Yields Strong Baselines

In the binary classification task, dedicated binary models (especially Random Forest + a combination of user and species contexts) perform best, with both Macro Accuracy and AP exceeding 80%. In the multiple-choice task, a simple single-layer MLP combined with user and species contexts can match the performance of larger-capacity Transformer models. The significance of this result is that carefully designed feature engineering can largely compensate for the lack of model complexity. In many practical application scenarios, simple, interpretable models paired with strong features are far more practical than complex end-to-end models.

#### Finding 2: User Context is More Valuable Than Species Context

Comparing different context types, models perform best when receiving both user and species contexts (U+S), followed closely by models using only user context (U). Among binary classifiers, models containing user context consistently lead in accuracy and average precision. Although models relying solely on species context still outperform simple heuristics, user-specific context is necessary for strong predictions. This suggests that individual differences among learners (including historical accuracy, learning trajectories, the set of species encountered, and strong or weak species categories) contain richer and more predictive signals than global species difficulty statistics. This finding provides direct guidance for the design of intelligent tutoring systems—systems should prioritize tracking and utilizing individual learners' interaction histories.

#### Finding 3: Multi-Choice Trained Classifiers Perform Worse on Binary Tasks Than Directly Trained Binary Classifiers

Although simple probabilistic models have smaller capacities, they can achieve 80%+ AP and average accuracy when directly trained on the binary objective. The reason is that multi-choice models need to allocate capacity to learn the structure of incorrect responses rather than strictly deciding whether an answer is correct. Future work could explore using the outputs of binary classifiers as auxiliary signals or gating mechanisms to enhance multi-choice models.

#### Finding 4: Image Features Complement User and Species Contexts, Especially for Predicting Incorrect Choices

The MLP with DINOv2 image features (U+S+Img) achieves slightly higher accuracy on the full dataset (~76%), but performs significantly better than other parameterized models on the subset of incorrect answers (~25%). Conversely, the image-only MLP without user and species contexts shows similar performance on the full dataset but drops to only ~11% on the incorrect subset. This demonstrates that image features are beneficial for knowledge tracing in CleverBirds, especially when combined with the correct context.

The deep implication of this finding is that images themselves carry key information that influences learners' error patterns—certain angles, lighting conditions, or bird postures can systematically lead to specific types of errors. For instance, an image of a Common Snipe shot from the side might be more easily confused with a Swinhoe's Snipe, whereas a front-facing shot is easier to identify correctly. Image features can encode these "error-prone conditions," thereby providing more precise error predictions when combined with user history and species statistics.

#### Finding 5: Predicting Incorrect Choices is Highly Challenging

This is one of the most important findings of this paper. While the multi-choice accuracy of trained models (~70%) on the full dataset seems decent, on the subset of questions where participants made incorrect choices, the accuracy of all trained models is below 25%—only slightly above the random baseline of 20%. The best model shows only a slight improvement over the confusion prior based purely on error assumptions. This indicates that:

- Existing methods are far from approximating learners' internal knowledge states.
- Understanding "why a person makes a specific mistake" is much more difficult than predicting "whether they will answer correctly."
- High performance on this subset would imply that the model truly understands learners' cognitive biases.

#### Finding 6: Minimal Performance Variance Across Knowledge Tracing Baselines

All tested KT methods (AKT, ATKT, simpleKT, DKT, DKT+, SAKT, DKVMN, KQN, etc.) perform similarly, with an average precision of about 0.35 and an average accuracy of about 54%. Despite their diverse architectures (ranging from simple RNNs to attention mechanisms and external memory), their performance on CleverBirds is far inferior to simple feature-engineered classifiers. These models might exploit shortcuts on positive classes—achieving high accuracy on positive classes but poor performance on negative classes.

This finding profoundly exposes the mismatch between the design assumptions of existing KT methods and large-scale fine-grained visual scenarios: most KT methods are designed for datasets with dozens to hundreds of concepts and emphasize concept-level explainability, which limits their flexibility when facing a concept space of over 10,000 concepts. They typically maintain independent knowledge state representations for each concept; when the number of concepts expands drastically, the parameter space explodes while the training samples for each concept become relatively sparse, leading to insufficient learning. Furthermore, these models usually assume that concepts are relatively independent, failing to utilize the rich taxonomic hierarchical information (such as family, genus, and species relationships) in bird taxonomy.

### Comparison with Existing KT Datasets

| Dataset | Domain | Number of Concepts | Number of Participants | Total Interactions | Visual? |
|--------|------|--------|---------|---------|---------|
| ASSISTments | Math | ~100 | ~4,000 | ~350K | No |
| EdNet | General Education | ~300 | ~780K | ~130M | No |
| Gravity Spy | Gravitational Waves | 21 | ~8,000 | ~1M | Partially |
| Butterflies/Retina/Greebles | Visual | 3-5 | ~150 | ~6.75K | Yes |
| **CleverBirds** | **Bird Identification** | **10,779** | **40,000+** | **17.8M** | **Yes** |

CleverBirds is three orders of magnitude larger in concept number and nearly four orders of magnitude larger in interaction count compared to the largest existing visual KT datasets, representing a massive leap in the field of visual knowledge tracing.

### Predictive Capability of Image Features

The experiment also evaluates the performance of species classification using only image features (see Appendix Table A9) to verify the effectiveness of pre-trained visual features for fine-grained bird identification. DINOv2 features outperform ResNet-50 in this task, confirming the rationality of utilizing these features to approximate learners' visual representations.

## Highlights & Insights

### Academic Value

- **Filling a Critical Gap**: Built the first large-scale visual knowledge tracing benchmark. Its 10,779 concepts far exceed any existing KT dataset (where the largest previously had only a few hundred concepts).
- **Exposing Limitations of Existing Methods**: Systematically revealed the limitations of current knowledge tracing methods in large-scale fine-grained visual scenarios, particularly their poor performance in predicting learner mistakes.
- **Elegant Dataset Design**: Real learning data collected from a citizen science project is of higher quality, larger scale, and richer dynamics than data collected in laboratory environments. The voluntary participation of learners ensures high-quality interactions.

### Methodological Insights

- **The Vitality of Simple Models**: Feature engineering + simple classifiers (Random Forest) significantly outperform all deep learning KT methods on binary classification tasks, suggesting that model complexity is not critical when data features are sufficiently strong.
- **Importance of User Context**: The predictive value of individual interaction history is far greater than global species statistics, implying that knowledge tracing models should focus more on personalized modeling.
- **Bridge Between Visual Features and Cognitive Modeling**: The paper assumes that human learners and neural networks extract similar image concepts for the same task, and uses pre-trained CNN/ViT features to approximate human visual representations, a hypothesis partially validated by the experiments.

### Application Prospects

- Personalized recommendation of appropriately difficult practice questions in intelligent tutoring systems, adaptively adjusting instructional progress based on the learner's knowledge blind spots.
- Adaptive identification and remediation of learner blind spots in citizen science training, improving the quality of citizen science data.
- Generalization to other domains requiring visual expertise, such as medical imaging (e.g., pathology slides, X-ray identification) and art appreciation (e.g., determining painting style, era).
- Providing a large-scale experimental platform for developing adaptive pedagogical strategies based on reinforcement learning (e.g., optimal question sequence design).
- Studying the cognitive science of human visual expertise acquisition—such as exploring changes in attention patterns during the transition from novice to expert, and the patterns of transfer learning across concepts.

## Related Work & Insights

The core differences between this work and prior visual knowledge tracing research manifest in three dimensions:

**Scale Dimension**: CleverBirds' 10,779 concepts and 17.8M interactions are 3 to 4 orders of magnitude larger than the largest existing visual KT datasets, making research on knowledge tracing in truly large-scale concept spaces possible for the first time.

**Task Dimension**: Unlike previous visual KT datasets that use laboratory-controlled subjects and limited image sets (e.g., only a few hundred images for 3 categories of retinopathy), CleverBirds utilizes natural data from spontaneous real-user participation. The images come from field shots by citizen scientists and contain natural variations in quality, angle, and lighting, making them closer to real-world visual recognition challenges.

**Evaluation Dimension**: This paper introduces "incorrect subset accuracy" as a key evaluation metric, shifting the evaluation focus from "predicting correctness" to "predicting specific mistakes." This represents a deeper test of the model's ability to model learners' cognitive states.

## Key Designs

1. **Large-scale Real Learning Data**: Over 17M multiple-choice interaction data points covering 10,779 bird species and 40,000+ participants were collected from the eBird citizen science platform. The data naturally embeds complete learning dynamics (with accuracy increasing by 20% with exposure frequency), making it the largest visual knowledge tuning benchmark to date.

2. **Multi-level Evaluation Framework**: The prediction task is decomposed into two levels: binary classification (correct/incorrect) and multiple-choice (specific option), using complementary metrics such as Macro Accuracy, AP, overall dataset accuracy, and incorrect subset accuracy to systematically quantify the prediction capability of different models and context combinations. In particular, the unique metric of "incorrect subset accuracy" reveals the true capability of models to approximate learners' internal knowledge states.

3. **Three-tiered Contextual Signal Design**: The available information is systematically categorized into user context (individual history), species context (global statistics), and image context (visual features). Ablation experiments reveal differences in the contribution of each signal type ($U > S$, and $Img$ is particularly crucial in predicting errors), pointing the way for future research.

## Limitations & Future Work

- **Domain Specificity**: The dataset focuses solely on bird species identification, and its direct transferability to other domains such as medical imaging and general object recognition remains to be verified.
- **Geographical and Selection Biases**: Participants are sourced from the eBird platform, which is known to bias toward users in developed countries of the Northern Hemisphere (such as North America and Europe). This leads to underrepresentation of tropical and Southern Hemisphere species, potentially affecting model generalization on species from these regions.
- **Limitations of the Multiple-Choice Format**: The multiple-choice format differs from open-ended identification scenarios, restricting knowledge tracing to a finite set of options and thus impacting ecological validity, although this is partially mitigated by the dynamic sampling of taxonomically similar distractors.
- **Monolithic Feedback Format**: Learners only receive the correct species label as feedback, lacking richer differential descriptions (such as "note the white trailing edge on the wing"), which warrants exploring more fine-grained feedback mechanisms on knowledge tracing models in the future.
- **Label Noise**: Although the error rate of bird identification on iNaturalist is only 3.3%, labeling errors in citizen science data might still have a minor impact on model training. However, since the core task is to predict user responses rather than the ground-truth species labels, the impact of label noise is relatively limited.
- **Limitations of Existing Methods**: All tested models perform extremely poorly in predicting incorrect choices ($<25\%$), indicating that approximating learners' internal knowledge states remains highly challenging. Future work should explore longer temporal contexts, cross-species knowledge transfer, and methods for integrating binary signals into multi-choice models to enhance error prediction.
- **Limitations of Image Representations**: While using fixed DINOv2/ResNet-50 features to approximate human visual representations is theoretically supported, human learners may focus on different visual cues (e.g., novices focus on color while experts focus on morphostructural details). Fine-grained human attention modeling could yield performance improvements.
- **Temporal Window Constraints**: Transformer models only use a history window of $W=50$, whereas some heavy users have over 1,000 interactions. Exploring how to efficiently utilize full interaction histories (e.g., through retrieval augmentation or memory compression mechanisms) is an important direction.
- **Under-explored Cross-Species Knowledge Transfer**: Existing methods do not utilize the taxonomic hierarchy of birds—the mastery of other species within the same family or genus should provide transfer information. For instance, users who can distinguish between two warbler species may more easily distinguish other visually similar species pairs. Building knowledge tracing models that exploit taxonomic hierarchies is a promising research direction.
- **Lack of Active Learning/Teaching Components**: CleverBirds currently acts only as a passive evaluation benchmark—recording user performance on randomly presented questions. Future work can combine active learning strategies to study optimal question presentation sequences (such as using spaced repetition or curriculum learning strategies) to maximize user learning efficiency.

<!-- RELATED:START -->

<div class="related-papers" markdown="1">

## Related Papers

- [\[AAAI 2026\] Towards LLM-Empowered Knowledge Tracing via LLM-Student Hierarchical Behavior Alignment in Hyperbolic Space](../../AAAI2026/self_supervised/towards_llm-empowered_knowledge_tracing_via_llm-student_hierarchical_behavior_al.md)
- [\[ICCV 2025\] Generate, Refine, and Encode: Leveraging Synthesized Novel Samples for On-the-Fly Fine-Grained Category Discovery](../../ICCV2025/self_supervised/generate_refine_and_encode_leveraging_synthesized_novel_samples_for_on-the-fly_f.md)
- [\[NeurIPS 2025\] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark](mmtu_a_massive_multi-task_table_understanding_and_reasoning_benchmark.md)
- [\[CVPR 2026\] TAR: Token-Aware Refinement for Fine-grained Generalized Category Discovery](../../CVPR2026/self_supervised/tar_token-aware_refinement_for_fine-grained_generalized_category_discovery.md)
- [\[AAAI 2026\] FineXtrol: Controllable Motion Generation via Fine-Grained Text](../../AAAI2026/self_supervised/finextrol_controllable_motion_generation_via_fine-grained_text.md)

</div>

<!-- RELATED:END -->
