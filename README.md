# Emotional Triggers and Social Network Structures in Misinformation Propagation

### A Behavioral Data Science Approach


## Overview

Misinformation spreads across social media through a combination of emotional content, user interactions, and network structures. Understanding why some misleading claims reach large audiences while others receive limited attention is an important problem in behavioral data science.

This research investigates the association between emotional characteristics and social-network structures and the propagation of misinformation. Using the FakeNewsNet dataset as a starting point, the project combines natural language processing (NLP), network analysis, statistical modeling, and machine learning to study factors associated with misinformation cascade size.

The project aims to develop an evidence-based understanding of misinformation propagation and evaluate whether combining emotional and network features improves predictive performance.

> **Research focus:** Understanding why some misinformation cascades become larger than others.

## Research Question

**RQ1:** How do emotional characteristics and social-network structures influence the propagation of misinformation?

The study focuses on the following hypotheses:

- **H1 — Emotional Intensity:** Emotionally intense misinformation is associated with larger propagation cascades.
- **H2 — Network Position:** Users occupying important positions in social networks are associated with larger misinformation cascades.
- **H3 — Network Structure:** Certain network structures are associated with greater misinformation propagation and faster diffusion.
- **H4 — Combined Features:** Combining emotional characteristics and network-structure features improves cascade-size prediction compared with using either feature group alone.

These hypotheses will be evaluated empirically. Because the study is observational, statistical associations will not automatically be interpreted as causal effects.

## Research Objectives

1. Analyze emotional characteristics in misinformation-related text.
2. Examine the relationship between emotional features and cascade size.
3. Investigate how network centrality and structural characteristics relate to misinformation propagation.
4. Identify network structures associated with larger or faster diffusion cascades.
5. Compare emotion-based, network-based, and combined predictive models.
6. Develop a reproducible research pipeline for misinformation analysis.

## Dataset

### FakeNewsNet

The project uses [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet), a research dataset developed for studying fake news through news content and social context.

The repository provides article-level metadata for the PolitiFact and GossipCop sources, including:

- News article identifiers
- Article URLs and headlines
- Fake/real labels encoded in the dataset filenames
- Associated tweet IDs

**Important:** The starter CSV files do not contain a complete, ready-to-use social propagation graph. Additional social-media records may be required to reconstruct cascades, obtain timestamps, and measure user-network structure.

### Data Requirements

| Data component | Purpose |
|---             |---      |
| Article headlines and metadata | Text analysis and article identification |
| Emotion scores | Quantify predicted emotional expression |
| Cascade size | Measure propagation extent |
| Social-media timestamps | Reconstruct diffusion over time |
| User interaction edges | Construct propagation and social graphs |
| Network centrality | Measure structural positions of users |
| Network clustering and communities | Characterize network structure |

Data availability will determine which hypotheses can be tested. Missing network measurements will not be treated as zero.

### Dataset Access

- Official repository: https://github.com/KaiDMML/FakeNewsNet
- Research paper: https://arxiv.org/abs/1809.01286

The complete social-context dataset may require additional collection and is subject to platform access restrictions, copyright, privacy requirements, and applicable research-ethics policies.

## Methodology

### 1. Data Collection and Preprocessing

- Load the available FakeNewsNet article-level CSV files.
- Select the fake-labelled PolitiFact and GossipCop records for the primary analysis.
- Clean headlines and remove unusable records.
- Maintain unique article identifiers and source metadata.
- Inspect missing values, duplicates, and dataset coverage.
- Collect or reconstruct social-media data where permitted and available.

### 2. Emotion Analysis

Natural language processing will be used to estimate emotional characteristics in article headlines and, where available, associated social-media posts.

Potential emotion categories include:

- Anger
- Fear
- Sadness
- Joy
- Disgust
- Surprise
- Neutrality

An emotion-classification model may be used to produce probability scores. A predefined measure of emotional intensity will be evaluated, and a manually reviewed sample will be used to assess prediction quality.

Model-generated emotion scores are estimates rather than verified psychological states. Headline emotion will be distinguished from the emotional responses of social-media users.

### 3. Cascade Reconstruction

Where suitable social data are available, each article or claim will be treated as a cascade-level observation.

The primary outcome is **cascade size**, operationalized as the number of unique observed users sharing a claim within a predefined observation window.

The analysis will distinguish cascade size from the total number of reshare events and diffusion speed. Observation windows, missing records, and collection coverage will be documented.

### 4. Social Network Analysis

A graph will be constructed from observed social interactions.

Potential network features include:

- Degree centrality
- Betweenness centrality
- PageRank
- Clustering coefficient
- Community structure
- Network modularity

Centrality and structural measurements will be defined using an explicit graph construction procedure. Where possible, network position will be measured before the cascade being studied to reduce information leakage.

### 5. Statistical Analysis

| Hypothesis | Proposed analysis |
|---         |---                |
| H1 | Negative binomial regression relating emotion features to cascade size |
| H2 | Negative binomial regression relating user centrality to cascade size |
| H3 | Regression of cascade size on network structure; survival analysis for diffusion speed where timestamps support it |
| H4 | Held-out predictive comparison of emotion-only, network-only, and combined models |

The analysis will account for relevant confounders and collection limitations where data permit. Effect sizes, confidence intervals, sample sizes, and model diagnostics will be reported.

### 6. Predictive Model Evaluation

The following model configurations will be compared:

- **Model A:** Emotion features only
- **Model B:** Network features only
- **Model C:** Emotion and network features combined

Potential evaluation metrics include:

- Mean Absolute Error (MAE)
- Poisson deviance
- Appropriate confidence intervals for differences in predictive performance

All models will use comparable observations and consistent data splits. Related articles about the same underlying claim will be grouped when appropriate to reduce information leakage.

## Technology Stack

- **Programming:** Python
- **Environment:** Google Colab, Jupyter Notebook
- **Data processing:** Pandas, NumPy
- **Natural language processing:** Hugging Face Transformers
- **Machine learning:** Scikit-learn
- **Statistical modeling:** Statsmodels
- **Network analysis:** NetworkX
- **Visualization:** Matplotlib, Seaborn
- **Version control:** Git and GitHub

## Project Structure

```text
misinformation-propagation-research/
│
├── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_emotion_analysis.ipynb
│   ├── 03_cascade_reconstruction.ipynb
│   ├── 04_network_analysis.ipynb
│   ├── 05_hypothesis_testing.ipynb
│   └── 06_model_comparison.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── emotion_analysis.py
│   ├── cascade_analysis.py
│   ├── network_features.py
│   └── statistical_models.py
│
├── configs/
│   └── experiment_config.yaml
│
├── reports/
│   └── figures/
│
├── requirements.txt
└── .gitignore
```

This is the proposed repository structure. Files should be added as the corresponding implementation becomes available.

## Getting Started

### Prerequisites

- A Google account
- Access to Google Colab
- Basic knowledge of Python and data analysis
- Additional social-data access if conducting cascade and network analyses

### Step 1: Open Google Colab

Visit https://colab.research.google.com/ and create a new notebook.

### Step 2: Clone the official dataset repository

```python
!git clone --depth 1 \
  https://github.com/KaiDMML/FakeNewsNet.git
```

### Step 3: Inspect the available dataset files

```python
from pathlib import Path

data_dir = Path("/content/FakeNewsNet/dataset")

for path in sorted(data_dir.glob("*")):
    print(path.name)
```

Verify which files are available before running preprocessing.

### Step 4: Install core libraries

```python
!pip install pandas numpy scikit-learn statsmodels \
    networkx transformers torch matplotlib seaborn
```

### Step 5: Continue through the analysis pipeline

Run the notebooks in sequence, beginning with data exploration and preprocessing. Conduct cascade-level hypothesis tests only after validating the required propagation and network data.

## Expected Outcomes

The project aims to:

- Characterize emotional patterns in misinformation-related content.
- Quantify associations between emotional characteristics and cascade size.
- Evaluate how network position and structure relate to propagation.
- Compare the predictive value of emotion and network features.
- Identify data-quality and measurement limitations in misinformation research.
- Produce reproducible analytical notebooks and documented experimental results.

These are research objectives, not claims of results already obtained.

## Limitations

- The starter FakeNewsNet CSV files do not provide complete propagation histories or user-network edges.
- Emotion classification may be affected by sarcasm, context, ambiguity, and model bias.
- Deleted or inaccessible social-media content can create collection bias.
- The available sample may not represent all platforms, communities, or types of misinformation.
- Observational relationships do not establish causality.
- Network features may be unavailable for some articles, reducing the usable sample.

The study will report these limitations alongside the empirical findings.

## Ethical Considerations

The project will follow applicable platform policies, dataset terms, institutional research-ethics requirements, and privacy protections. Restricted raw social-media data and personally identifying information will not be published without appropriate authorization.

## References

1. Shu, K., Mahudeswaran, D., Wang, S., Lee, D., & Liu, H. (2019). *FakeNewsNet: A Data Repository with News Content, Social Context and Spatiotemporal Information for Studying Fake News on Social Media.* https://arxiv.org/abs/1809.01286
2. FakeNewsNet official repository. https://github.com/KaiDMML/FakeNewsNet
3. Hartmann, J., Heitmann, M., Siebert, C., & Schamp, C. (2023). *More than a Feeling: Accuracy and Application of Sentiment Analysis.* The emotion model and its documentation should be cited separately if used in the final implementation.

## Project Status

**Current stage:** Research design and dataset preparation.

Planned milestones:

- [ ] Download and validate the available FakeNewsNet starter files.
- [ ] Complete exploratory data analysis.
- [ ] Implement and validate emotion scoring.
- [ ] Obtain suitable social-media propagation records.
- [ ] Reconstruct and validate article-level cascades.
- [ ] Calculate network features.
- [ ] Test hypotheses H1–H3.
- [ ] Compare predictive models for H4.
- [ ] Document results, limitations, and reproducibility.

---

**Author:** Aiyaz Khan & Vishal Yadav 
