# EibAIS-MLOps-excercise

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This is a toy project used in the tutorial [Software Engineering for ML Systems](https://conf.researchr.org/track/cibse-2025/cibse-2025-eibais#Tutorial-3) at the [EibAIS 2025](https://conf.researchr.org/track/cibse-2025/cibse-2025-eibais#About) school. The goal of this project is to provide a starting point for building a machine learning project using MLOps practices.

## Project Organization

```
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         src and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── src   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes src a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    └── main.py                 <- Main script to run the project                
```

--------

## Requirements
To complete this excercise, you will need access to the following:

### External Services
- A [GitHub](https://github.com) account
- A [DagsHub](https://dagshub.com/) account linked to your GitHub account
- A [Hugging Face](https://huggingface.co/) account and an access token. You can create a token by going to your profile settings and selecting "Access Tokens". Make sure to select the `write` scope for the token. You can find more information on how to create a token [here](https://huggingface.co/docs/huggingface_hub/how-to-use-huggingface-hub#creating-a-token).

### Software
- Git
- The [uv](https://docs.astral.sh/uv/) dependency manager
- Python 3.11 or later ([can be installed using uv](https://docs.astral.sh/uv/guides/install-python/))

If it is the first time using GitHub, follow [this](https://docs.github.com/en/authentication/connecting-to-github-with-ssh) guidelines to connect to GitHub with SSH with your local Git installation.

> Note: For attendees using Windows 10/11 it is highly recommended to use the Windows Subsystem for Linux (WSL). For instructions on how to set it up see [here](https://learn.microsoft.com/en-us/windows/wsl/install).

## Setup
1. Fork this repository to your GitHub account. See [this](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) for instructions.
2. Connect your forked repository to your DagsHub account. See [this](https://dagshub.com/docs/integration_guide/github/) for instructions.
3. Clone the forked repository to your local machine.
4. Copy the `.env-template` file to `.env` and fill in the required variables. The `.env` file is used to store environment variables that are used in the project.
5. Install the required dependencies using `uv`:
   ```bash
   uv sync
   ```


## Dataset
For this excercise, we will be using the [Emotion Dataset](https://huggingface.co/datasets/dair-ai/emotion) from the following paper:
```bibtex
@inproceedings{saravia-etal-2018-carer,
    title = "{CARER}: Contextualized Affect Representations for Emotion Recognition",
    author = "Saravia, Elvis  and
      Liu, Hsien-Chi Toby  and
      Huang, Yen-Hao  and
      Wu, Junlin  and
      Chen, Yi-Shin",
    booktitle = "Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing",
    month = oct # "-" # nov,
    year = "2018",
    address = "Brussels, Belgium",
    publisher = "Association for Computational Linguistics",
    url = "https://www.aclweb.org/anthology/D18-1404",
    doi = "10.18653/v1/D18-1404",
    pages = "3687--3697",
    abstract = "Emotions are expressed in nuanced ways, which varies by collective or individual experiences, knowledge, and beliefs. Therefore, to understand emotion, as conveyed through text, a robust mechanism capable of capturing and modeling different linguistic nuances and phenomena is needed. We propose a semi-supervised, graph-based algorithm to produce rich structural descriptors which serve as the building blocks for constructing contextualized affect representations from text. The pattern-based representations are further enriched with word embeddings and evaluated through several emotion recognition tasks. Our experimental results demonstrate that the proposed method outperforms state-of-the-art techniques on emotion recognition tasks.",
}
```