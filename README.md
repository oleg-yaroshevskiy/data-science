# Data science learning projects

A personal collection of notebooks, small implementations, and experiments built
while learning data science. Started years ago and occasionally extended with
new examples, from statistical tests and classical machine learning to language
models and image generation.

Browse by topic:

- [Statistics and probability](#statistics-and-probability)
- [Classification and regression](#classification-and-regression)
- [Clustering and dimensionality reduction](#clustering-and-dimensionality-reduction)
- [Text and language](#text-and-language)
- [Images and generative models](#images-and-generative-models)
- [Time series](#time-series)
- [Applied projects and competitions](#applied-projects-and-competitions)
- [Evaluation and library reference](#evaluation-and-library-reference)

## Statistics and probability

- [Statistical inference notebooks](statistics/) — confidence intervals,
  bootstrap, hypothesis tests, correlation, regression, and multiple testing.
- [Statistical testing in Python](statistics_testing/) — standalone examples
  of t-tests, proportion tests, bootstrap intervals, and A/B testing.
- [SciPy statistics: part 1](scipy.stats.ipynb) and
  [part 2](scipy.stats2.ipynb) — explorations of probability distributions and
  statistical functions.

## Classification and regression

- [Linear regression and SGD from scratch](LR%20and%20SGD%20implementation.ipynb)
  — learning the algorithms through their implementation.
- [Height–weight regression](LR%20height%20woight%20data%20set.ipynb)
  — a small regression experiment.
- [Classifier comparison on Gaussian blobs](classifier%20comparison%20on%20gaussian%20blobs.ipynb)
  — comparing decision boundaries on synthetic data.
- [Gradient boosting and XGBoost](Gradient%20boosting%20implementation,%20XGBoost%20%28boston%20house%20prices%29.ipynb)
  — boosting experiments using the Boston house prices dataset.
- [Wine quality with a neural network](Neural%20network%20for%20wine%20quality%20prediction.ipynb)
  — a neural network applied to tabular data.

## Clustering and dimensionality reduction

- [Principal component analysis](Principal%20component%20analysis.ipynb)
  — exploring dimensionality reduction.
- [Clustering and retrieval on Amazon reviews](clustering%20on%20amazon%20data/)
  — text embeddings, clustering comparisons, dimensionality reduction,
  retrieval adaptation, and custom metrics.
- [MeanShift on Foursquare data](clustering%20on%20foursquare%20data/)
  — a clustering application using location data.
- [Visualizing and clustering handwritten digits](handwritten%20digits/Data%20visualization%20and%20clustering%20on%20the%20handwritten%20digits.ipynb)
  — exploring the structure of digit data.
- [Document clustering](text%20analytics/Documents%20clustering%20examples.ipynb)
  — grouping text documents by similarity.

## Text and language

### Classification, embeddings, and topic modelling

- [Text analytics](text%20analytics/) — sentiment analysis on movie reviews,
  SMS spam classification, and word2vec experiments.
- [Enron email classification](enron/) — email classification experiments,
  including a word2vec approach.
- [Topic modelling](topic%20modelling/) — LDA, gensim, and BigARTM examples
  using recipes, lectures, and other text collections.

### Recurrent networks and language model basics

- [Elman recurrent network](elman%20rnn/) — an early recurrent network example.
- [Character RNNs and LSTMs](rnn/) — sequence modelling experiments,
  including Shakespeare and Shevchenko text generation.
- [Tokenization and sampling](lm/) — byte-pair encoding and language model
  sampling implementations.

## Images and generative models

### Digit recognition

These examples compare classical classifiers and neural networks on image data.

- [Bayesian digit classification](handwritten%20digits/Bayesian%20for%20handwritten%20digits.ipynb)
- [Digit classification with decision trees](handwritten%20digits/Classification%20of%20handwritten%20digits%20via%20decision%20trees.ipynb)
- [MNIST with TensorFlow](handwritten%20digits/MNIST%20Tensorflow%20beginner.ipynb)
- [VGG16 experiment](handwritten%20digits/Tensorflow%20vgg16.ipynb)
- [notMNIST with TensorFlow](notmnist%20tensorflow/) — data exploration,
  linear models, multilayer networks, and convolutional networks.

### Generative models

- [Variational autoencoder](vae/) — a VAE learning notebook.
- [MNIST generation with DDIM (2020)](ddim_mnist/) — a small U-Net and
  diffusion formulas in plain PyTorch, CPU training, and deterministic sampling.
  Includes a [generated digit grid](ddim_mnist/examples/) from a 20-epoch run.

## Time series

- [Australian wine sales](autocorrelation/australian%20wine%20sales%20prediction.ipynb)
- [Russian wages](autocorrelation/russian%20wages%20prediction.ipynb)

## Applied projects and competitions

Projects organized around a dataset or prediction task, often combining
exploration, preprocessing, model fitting, and comparison.

- [Bike sharing demand](kaggle%20bike%20sharing%20demand/) — rental demand
  exploration and prediction with several regression methods.
- [Titanic survival](kaggle%20titanic%20problem/) — random forests, XGBoost,
  SVMs, and other survival prediction experiments.
- [Prudential life insurance](kaggle%20prudention%20life%20insurance/)
  — data visualization and exploration.
- [University of Melbourne grant applications](kaggle%20university%20of%20melbourne%20grant%20applications/)
  — preprocessing and logistic regression.
- [House price prediction assignment](test_assingment%20%28house%20pricing%29.ipynb)
  — a separate house pricing exercise.

## Evaluation and library reference

### Metrics and model evaluation

- [Binary classifier evaluation](Evaluation%20of%20binary%20classifiers.ipynb)
- [Confusion matrix utilities](classification/) — binary and multiclass examples.
- [Recommendation metrics](recommendations%20metrics/) — metric implementations
  and an example of their use.
- [scikit-learn metrics](sklearn.metrics.ipynb)

### Library explorations and supporting data

- [scikit-learn datasets](sklearn.datasets.ipynb)
- [scikit-learn linear models: part 1](sklearn.linear_model1.ipynb) and
  [part 2](sklearn.linear_model2.ipynb)
- [Supporting datasets](resources/)

## Running the examples

Most older projects are Jupyter notebooks; newer examples also include plain
Python scripts. Open the notebook or folder you want to explore and check its
imports, local data paths, and any project-specific README or requirements file.
Dependencies vary across projects. Older notebooks may need adjustments for
current library versions. Examples with their own setup instructions include
[DDIM MNIST](ddim_mnist/README.md),
[statistical testing](statistics_testing/README.md), and
[Amazon review clustering](clustering%20on%20amazon%20data/README.md).
