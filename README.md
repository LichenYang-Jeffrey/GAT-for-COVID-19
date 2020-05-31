# GAT-for-COVID-19
This implementation of GAT ([Graph Attention Network, Veličković et. al](https://arxiv.org/abs/1710.10903)) is revised from [pyGAT](https://github.com/Diego999/pyGAT) to predict antiviral activity against SARS-CoV-2 (COVID-19). The task is specified in (https://www.aicures.mit.edu/tasks)[https://www.aicures.mit.edu/tasks].

Data is available in folder /data_covid, which is described in [https://github.com/yangkevin2/coronavirus_data](https://github.com/yangkevin2/coronavirus_data). Since the dataset is highly imbabanced, [imbalanced sampling technique](https://github.com/ufoym/imbalanced-dataset-sampler) is applied to generate training data.

The training ans validation process is specified in main.py.

Results:
| Model | Positive Ratio | Accuracy (Negative) | Accuracy (Positive) | ROC-AUC |
|------|------|------|------|------|
| GAT | 12.5% | 92.92% | 41.67% | 0.8094 |
| GAT | 25.0% | 95.52% | 47.22% | 0.8338 |
| GAT | 50.0% | 96.24% | 91.67% | 0.8070 |
