
# Runtime Prediction

Adapted from Kaggle's competition [Google - Fast or Slow? Predict AI Model Runtime](kaggle.com/competitions/predict-ai-model-runtime), this project aims to predict a machine learning models' runtime given its graph representation. The graphs are generated from TensorFlow's XLA compiler. Runtime prediction from `tile:xla` is only implemented in this project. The tile_gnn.ipynb shows the code used for loading data, preprocessing it, building the model, training it, and the evaluation on test data. Different Graph Convolutional Network types from pytorch were used, results of each can be observed in the gcn_tile_xla.txt file.

## Data Description

The data is provided by the Kaggle's competition mentioned above.
the dataset, called TpuGraphs, is the performance prediction dataset on XLA HLO graphs running on Tensor Processing Units (TPUs) v3. There are 5 data collections in total: `layout:xla:random`, `layout:xla:default`, `layout:nlp:random`, `layout:nlp:default`, and `tile:xla`.
Visualizing an AI model involves depicting it as a graph, with nodes denoting tensor operations (like matrix multiplication, convolution, etc.) and edges symbolizing tensors. The compiler's functioning relies on a compilation configuration that dictates how the graph undergoes transformation for specific optimization passes. Notably, we have control over two types of configurations/optimizations:

1. Layout configuration: manages the arrangement of tensors within the graph in physical memory. It achieves this by specifying the dimension order for each input and output of an operation node.

1. Tile configuration: dictates the size of tiles for each fused subgraph.

The data can be accessed from the competition's website on [Kaggle](kaggle.com/competitions/predict-ai-model-runtime).

## Evaluation

For fair and comprehensive comparison on the empirical studies, we use two evaluation metrics in our experiments. 

**Kaggle evaluation metrics:** The metric used in the Kaggle competition for scoring is based on top- $K$ predictions, which is defined as follows:

For the collection `tile:xla`, the metric (1-slowdown) incurred of the top- $K$ predictions to measure how much slower the top- $K$ predictions are from the actual fastest configurations. This can be calculated as follows:

$$
    1- \left( \frac{\text{The best runtime of top-K predictions}}{\text{The best runtime of all configurations}}-1 \right) = 2 - \frac{min_{i\in K}y_i}{min_{i\in A}y_i} 
$$

where $K$ is the top- $K$ predictions, A is all configurations of the given graph from the dataset collection, and $y$ is the measured execution time.

**Model aspect evaluation:** We involve additional evaluation metric for measuring the baselines from the perspective of AI models. One common metric to evaluate our model's runtime prediction is Mean Squared Error (MSE) which is 

$$
    \mathrm{MSE} = \frac{1}{n} \sum_{i=1}^{n}{\left( y_i - \hat{y}_i \right)^2}
$$

where $n$ is the number of samples, $y_i$ is the true value of the $i$ th sample and $\hat{y}_i$ is its prediction by our model. This is a loss function that we would like to minimize (i.e. find the best model that predicts $\hat{y}_i$ for all $i$ such that MSE is as small as possible).


## Deployment

To deploy the jupyter notebook, install the needed following libraries:

```bash
    pip install numpy
    pip install pandas
    pip install tqdm
    pip install scikit-learn
    pip install torch
    pip install torch-geometric
    pip install timm
    pip install matplotlib
```

## Usage 
For more convenient usage with our code, we provide details about the key values and functions in our code as follows:

### Code details
---
**Values**:
- The string value `conv_layer` determines which graph learning method will be used to construct the graph layer of the model. **16** different choices (**15** graph neural networks + **1** pure MLP) are constructed for the graph layer, where more details can be found in `Model` class. 
- The bool value `retrain_mark` controls retraining or direct testing. When it is set `True`, the model will conduct new training process and save the new trained model parameters into `./save_model/`; when it is set `False`, that will be the testing process only with using the trained model in `./save_model/`.

**Functions**:
- The `Model` class is used for initialization of model instance, where the only initialization parameter is the `conv_layer`.
- `score_tile_mean` and `score_tile_max` are used to compute the Top-K score for the model, where a 0.1 smoothness factor is used to avoid 0 in denominator. 

### Example usage
---
**Train your own model**: If you want to train your own model with, such as, *GCN* graph layer, you should:
- Set `conv_layer` to be `gcnconv`;
- Set `retrain_mark` to be `True`.

**Test our trained models**: If you want to use the trained models we provide, for example, *ARMA*, you should:
- Set `conv_layer` to be `armaconv`;
- Set `retrain_mark` to be `False`.


## Authors

- Guoming Li
- Omar SayedElAhl
- Yasser Ashraf

