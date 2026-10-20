Eco-Sort Research Benchmark

1. Project Overview

Eco-Sort is an AI-assisted research prototype designed to explore waste classification and recycling guidance.

2. Research Question

How consistently can AI models classify common waste items and identify items that require special handling?

3. Objectives

- Compare AI model predictions on the same set of waste items.
- Measure classification accuracy and macro F1 score.
- Measure response latency.
- Examine errors involving hazardous or special-disposal items.
- Support reproducible AI evaluation.

4. Methodology

The prototype evaluates selected AI models using a fixed dataset of waste items and preliminary reference labels. Each model receives the same classification task, and its predictions are compared with the reference labels.

Reference labels should be verified using reliable waste-management sources before final conclusions are drawn.

5. Evaluation Metrics

- Accuracy
- Macro F1 score
- Hazardous-item classification performance
- Response latency
- Failed or invalid responses

6. Technology Stack

- Python
- Streamlit
- Pandas
- Scikit-learn
- Google Gen AI SDK

7. Limitations and Ethics

This project is a research prototype and does not replace official local waste-disposal guidance. Disposal rules vary by location, and AI predictions may be incorrect. The dataset is limited, and no performance results are claimed before experiments are conducted.

API keys must be kept private and must never be committed to the public repository.

8. Reproducibility

The repository will contain the source code and dependency list required to run the experiment. Evaluation results should be reported only after the experiment has been executed.

9. Future Work

Future improvements may include expanding the dataset, verifying reference labels, testing additional AI models, and evaluating performance across different regions.
