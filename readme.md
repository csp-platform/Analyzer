<p align="center">
  <img src="extra/project_logo.png" width="200" alt="project-logo">
</p>
<p align="center">
    <h1 align="center">CSP Analyzer</h1>
</p>
<p align="center">
    <em><code>► CSP Analyze</code></em>
</p>
<p align="center">
	<!-- local repository, no metadata badges. -->
<p>

<br><!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary><br>

- [ Overview](#overview)
- [ Features](#features)
- [ Repository Structure](#repository-structure)
- [ Modules](#modules)
- [ Getting Started](#getting-started)
  - [ Installation](#installation)
  - [ Usage](#usage)
- [ Project Roadmap](#project-roadmap)
- [ Contributing](#contributing)
</details>
<hr>

##  Overview

<p>The project aims to analyze data generated by ranking competitions (CSP Platform). It evaluates several classes of metrics, including mimicking the winner, competition diversity, competition convergence, and the properties of winners.</p>

---

##  Features

1. **Comprehensive Data Analysis Framework**  
   Provides an in-depth analysis of data generated by ranking competitions like CSP Platform, focusing on extracting meaningful insights from complex datasets.

2. **Winner Mimicking Evaluation**  
   Assesses how closely participants can replicate the strategies and outcomes of competition winners. This feature helps identify effective tactics and benchmarks for success.

3. **Competition Diversity Metrics**  
   Measures the variety of strategies, behaviors, and document rankings among participants to evaluate the overall diversity within the competition.

4. **Convergence Analysis Tools**  
   Analyzes the degree to which participant strategies converge or diverge over time, identifying patterns that lead to similar outcomes or highlight unique approaches.

5. **Winner Property Characterization**  
   Investigates the attributes and strategies of winners, uncovering key factors that contribute to their success and distinguishing them from other participants.

6. **Advanced Embedding Analysis with BERT, E5, and SBERT**  
    Utilizes state-of-the-art embedding models like BERT, E5, and SBERT to perform deep semantic analysis of documents and rankings. This enables a nuanced understanding of textual data, participant strategies, and their impact on competition outcomes.

7. **Advanced Statistical Modeling**  
   Integrates sophisticated statistical methods to evaluate metrics such as variance, standard deviation, and correlation within the competition data.

8. **Visualization and Reporting**  
   Generates intuitive graphs, charts, and reports that illustrate findings on mimicking, diversity, convergence, and winner properties for easier interpretation.

9. **Customizable Metric Classes**  
   Allows for the addition of new metric classes or modification of existing ones, providing flexibility to adapt the analysis to specific research needs.

10. **Automated Data Processing Pipeline**  
   Streamlines the ingestion, cleaning, and preprocessing of competition data, ensuring efficient and accurate analysis workflows.

11. **Modular and Extensible Architecture**  
    Features a modular design that enables easy integration with other tools or datasets, facilitating ongoing development and scalability of the project.

---

##  Repository Structure

```sh
└── Analyzer/
    ├── analysis
    │   ├── __init__.py
    │   ├── analyzer.py
    │   ├── data_cleaner.py
    │   ├── embedding_analyzer.py
    │   ├── feature_history.py
    │   ├── graphing.py
    │   ├── rank_analysis.py
    │   └── report_table.py
    ├── constants/
    │   ├── __init__.py
    │   └── constants.py
    ├── data/
    │   ├── web_track/
    │   │   ├── full-topics.xml
    │   │   ├── trec2013-topics.xml
    │   │   ├── trec2014-topics.xml
    │   │   ├── wt09.topics.full.xml
    │   │   ├── wt2010-topics.xml
    │   │   └── wt2012.xml
    ├── data_processing/
    │   ├── __init__.py
    │   ├── data_processor.py
    │   ├── feature_extractor.py
    │   └── index_manager.py
    ├── experiments/
    │   └── <experiment_hash>/
    │       ├── embeddings_graphs/
    │       │   ├── average_and_diameter_of_player_documents–mean/
    │       │   ├── average_and_diameter_of_player_documents–min/
    │       │   ├── average_of_player_documents_consecutive_rounds/
    │       │   ├── average_unique_documents_over_time/
    │       │   ├── plot_diameter_and_average_over_time–mean/
    │       │   ├── plot_diameter_and_average_over_time–min/
    │       │   ├── plot_first_second_similarity_over_time/
    │       │   ├── plot_rank_diameter_and_average_over_time–mean/
    │       │   ├── plot_rank_diameter_and_average_over_time–min/
    │       │   ├── rank_diameter_and_average_last_round–mean/
    │       │   ├── rank_diameter_and_average_last_round–min/
    │       │   ├── winner_similarity_over_time/
    │       │   ├── average_and_diameter_of_player_documents.csv
    │       │   ├── average_of_player_documents_consecutive_rounds.csv
    │       │   └── rank_diameter_and_average_last_round.csv
    │       ├── faiss/
    │       │   ├── e5_index.index
    │       │   └── sbert_index.index
    │       ├── graphs/
    │       │   ├── query_dependent/
    │       │   └── query_independent/
    │       │       └── number_non_and_consecutive_matches_won.png
    │       ├── bert_similarities_dict.pkl
    │       ├── competition_history.csv
    │       ├── competition_history_pivot.csv
    │       ├── config.json
    │       ├── e5_cosine_similarities_dict.pkl
    │       ├── experiment.log
    │       ├── feature_matrix.csv
    │       ├── output.trectext
    │       ├── report_table.csv
    │       ├── tfidf-krovetz-jaccard_dict.pkl
    │       ├── tfidf-krovetz-jaccard_similarity.pkl
    │       ├── tfidf-krovetz_dict.pkl
    │       └── tfidf-krovetz_similarity.pkl
    ├── extra/
    │   ├── project_logo.png
    ├── faiss_index/
    │   ├── __init__.py
    │   └── faiss_index.py
    ├── feature_engineering/
    │   ├── __init__.py
    │   ├── bert_scorer.py
    │   ├── custom_features.py
    │   ├── e5.py
    │   ├── feature_extractor_wrapper.py
    │   ├── sbert.py
    │   └── tf_idf.py
    ├── input/
    │   ├── competition_history.csv
    │   ├── config.json
    │   └── output.trectext
    ├── parsers/
    │   ├── __init__.py
    │   ├── query_parser.py
    │   └── trec_parser.py
    ├── utils/
    │   ├── __init__.py
    │   ├── config_loader.py
    │   ├── file_operations.py
    │   ├── logging_setup.py
    │   └── utils.py
    ├── .gitignore
    ├── main.py
    ├── readme.md
    └── requirements.txt
```

---

## Modules

<details closed><summary>analysis</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [analyzer.py](analysis/analyzer.py)         | Provides a comprehensive analysis of competition data, including metrics for mimicking winners, diversity, convergence, and winner properties. |
| [data_cleaner.py](analysis/data_cleaner.py) | Cleans and preprocesses competition data, preparing it for analysis and visualization. |
| [embedding_analyzer.py](analysis/embedding_analyzer.py) | Utilizes advanced embedding models like BERT, E5, and SBERT to perform deep semantic analysis of documents and rankings. |
| [feature_history.py](analysis/feature_history.py) | Tracks the history of competition features and metrics over time, enabling longitudinal analysis and trend identification. |
| [graphing.py](analysis/graphing.py)         | Generates visualizations and reports to illustrate competition data, metrics, and trends for easier interpretation. |
| [rank_analysis.py](analysis/rank_analysis.py) | Analyzes the rankings and strategies of competition participants, evaluating their performance and success factors. |
| [report_table.py](analysis/report_table.py) | Creates tables and reports summarizing competition data, metrics, and insights for detailed analysis and comparison. |

</details>

<details closed><summary>constants</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [constants.py](constants/constants.py)      | Contains constant values and configurations used throughout the project, ensuring consistency and ease of maintenance. |

</details>

<details closed><summary>data</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [full-topics.xml](data/web_track/full-topics.xml) | Contains full topics for the Web Track competition, providing detailed information on search queries and topics. |
| [trec2013-topics.xml](data/web_track/trec2013-topics.xml) | Includes topics for the TREC 2013 competition, specifying search queries and topics for participants. |
| [trec2014-topics.xml](data/web_track/trec2014-topics.xml) | Contains topics for the TREC 2014 competition, outlining search queries and topics for participants to address. |
| [wt09.topics.full.xml](data/web_track/wt09.topics.full.xml) | Includes full topics for the Web Track 2009 competition, detailing search queries and topics for participants to analyze. |
| [wt2010-topics.xml](data/web_track/wt2010-topics.xml) | Contains topics for the Web Track 2010 competition, specifying search queries and topics for participants to explore. |
| [wt2012.xml](data/web_track/wt2012.xml)    | Includes topics for the Web Track 2012 competition, outlining search queries and topics for participants to address. |

</details>

<details closed><summary>data_processing</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [data_processor.py](data_processing/data_processor.py) | Processes competition data, including cleaning, indexing, and feature extraction, to prepare it for analysis and visualization. |
| [feature_extractor.py](data_processing/feature_extractor.py) | Extracts features and metrics from competition data, enabling detailed analysis and evaluation of participant strategies and outcomes. |
| [index_manager.py](data_processing/index_manager.py) | Manages the indexing of competition data, ensuring efficient retrieval and processing of documents and rankings for analysis. |

</details>

<details closed><summary>experiments</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [experiment_hash](experiments/experiment_hash) | Contains experiment-specific data, including embeddings, graphs, and analysis results, for detailed evaluation and comparison. |

</details>

<details closed><summary>extra</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [project_logo.png](extra/project_logo.png) | Contains the project logo image, providing a visual representation of the project branding and identity. |
    
</details>

<details closed><summary>faiss_index</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [faiss_index.py](faiss_index/faiss_index.py) | Implements the FAISS indexing algorithm for competition data, enabling efficient retrieval and processing of documents and rankings. |

</details>

<details closed><summary>feature_engineering</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [bert_scorer.py](feature_engineering/bert_scorer.py) | Utilizes the BERT embedding model to score documents on semantic similarity and relevance. |
| [custom_features.py](feature_engineering/custom_features.py) | Defines custom features and metrics for competition data analysis, enabling tailored evaluation and insights. |
| [e5.py](feature_engineering/e5.py)          | Implements the E5 embedding model for deep semantic analysis of documents, providing detailed insights and evaluation. |
| [feature_extractor_wrapper.py](feature_engineering/feature_extractor_wrapper.py) | Wraps the feature extraction process, enabling seamless integration of multiple feature engineering methods and models. |
| [sbert.py](feature_engineering/sbert.py)    | Utilizes the SBERT embedding model for deep semantic analysis of documents, providing nuanced insights and evaluation. |
| [tf_idf.py](feature_engineering/tf_idf.py)  | Implements the TF-IDF feature extraction method for competition data analysis, enabling detailed evaluation and comparison of documents. |

</details>
    
<details closed><summary>input</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [competition_history.csv](input/competition_history.csv) | Contains the competition history data, including rankings, participants, and outcomes, for analysis and evaluation. |
| [config.json](input/config.json)            | Specifies the configuration settings for the competition analysis, including metrics, features, and visualization options. |
| [output.trectext](input/output.trectext)    | Contains the output data in TREC text format, enabling further processing and analysis of competition results. |

</details>

<details closed><summary>parsers</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [query_parser.py](parsers/query_parser.py) | Parses search queries and topics from competition data, enabling detailed analysis and evaluation of participant strategies. |
| [trec_parser.py](parsers/trec_parser.py)   | Parses TREC-formatted data, including topics, documents, and rankings, for analysis and visualization. |

</details>

<details closed><summary>utils</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [config_loader.py](utils/config_loader.py) | Loads and manages configuration settings for the competition analysis, ensuring consistency and ease of use. |
| [file_operations.py](utils/file_operations.py) | Provides file operations and utilities for reading, writing, and processing competition data, ensuring efficient data management. |
| [logging_setup.py](utils/logging_setup.py)  | Sets up logging and error handling for the project, enabling detailed tracking and monitoring of analysis processes. |
| [utils.py](utils/utils.py)                  | Contains utility functions and helpers for various tasks, including data processing, visualization, and analysis. |

</details>
    
<details closed><summary>main.py</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [main.py](main.py)                          | Contains the main entry point for the project, executing the competition analysis and generating insights and reports. |
    
</details>

<details closed><summary>requirements.txt</summary>

| File                                        | Summary                         |
|---------------------------------------------| ---                             |
| [requirements.txt](requirements.txt)        | Lists the required dependencies and packages for the project, ensuring compatibility and ease of installation. |

</details>


---

##  Getting Started

**System Requirements:**

* **Python**: `version 3.10+`

###  Installation

<h4>From <code>source</code></h4>

> 1. Clone the CSP Analyzer repository:
>
> ```console
> $ git clone ../CSP-Analyzer
> ```
>
> 2. Change to the project directory:
> ```console
> $ cd CSP-Analyzer
> ```
>
> 3. Install the dependencies:
> ```console
> $ pip install -r requirements.txt
> ```
> 
> 4. Update all dependencies:
> ```console
> $ pip install --upgrade -r requirements.txt
> ```

###  Usage
> 1. Analyzing Results: Use the output folder generated by your CSP- latform competition run as the input for analysis. Specify this folder with the --input_folder option.\
>
> 2. Using or Creating an Index: If you already have an existing index(pyserini), specify its location using the --index_folder option. If not, a new index will be created automatically.
>
> 3. Run CSP Analyzer using the command below:
> ```console
> $ python main.py --input_folder <path_to_output_folder> --index_folder <path_to_index_folder>
> ```


---

##  Project Roadmap

- [ ] `► Clean the code.`

---

##  Contributing

<details closed>
<summary>Contributing Guidelines</summary>

1. **Fork the Repository**: Start by forking the project repository to your local account.
2. **Clone Locally**: Clone the forked repository to your local machine using a git client.
   ```sh
   git clone ../CSP-Analyzer
   ```
3. **Create a New Branch**: Always work on a new branch, giving it a descriptive name.
   ```sh
   git checkout -b new-feature-x
   ```
4. **Make Your Changes**: Develop and test your changes locally.
5. **Commit Your Changes**: Commit with a clear message describing your updates.
   ```sh
   git commit -m 'Implemented new feature x.'
   ```
6. **Push to local**: Push the changes to your forked repository.
   ```sh
   git push origin new-feature-x
   ```
7. **Submit a Pull Request**: Create a PR against the original project repository. Clearly describe the changes and their motivations.
8. **Review**: Once your PR is reviewed and approved, it will be merged into the main branch. Congratulations on your contribution!
</details>
