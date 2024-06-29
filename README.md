# A Twitter Dataset Experiment on ie-HGCN

Discover the original model source [here](https://github.com/kepsail/ie-HGCN/).

## Requirements

### For the Embedder Pipeline:
- [Gensim](https://radimrehurek.com/gensim/)
- [Twitter4SSE](https://huggingface.co/digio/Twitter4SSE)
- [Pandas](https://pandas.pydata.org/)

### For ie-HGCN:
- [PyTorch](https://pytorch.org)
- [NumPy](https://numpy.org)
- [scikit-learn](https://scikit-learn.org)

## How to Run

**Step 1: Install Dependencies**

Ensure all the necessary dependencies are installed.

**Step 2: Execute the Dummy Run on the Sample IMDb Dataset**

Run the following scripts in order:
1. `imdb10197_clean.py`
2. `imdb10197_preprocessing.py`
3. `imdb10197_train.py`

This dummy run serves as a small, runnable example and a reference for newcomers wanting to adapt the model to new datasets.

**Step 3: Run the Experiments on the X Dataset**

1. Extract the entire content of the file `twitter_dataset/twitter_daset.7z` into the folder `twitter_dataset`.
2. Execute the file `twitter_pipeline.py`. For the first run, ensure all boolean settings are set to `True` to compute all preliminary steps.
3. The result will be stored into the folder `output_twitter`
4. For subsequent runs, set only the `trainModel` variable to `True` since all previous steps are already computed.
