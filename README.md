# beeFormer

This is the official implementation provided with our paper "beeFormer: Bridging the Gap Between Semantic and Interaction Similarity in Recommender Systems". 

## Steps for reproducing the results from our paper:

1. create virtual environment `python3.10 -m venv beef` and activate it `source beef/bin/activate`
2. clone this repository and navigate to it `cd beeFormer`
3. install packages `pip install -r requirements.txt`
4. download the data: navigate to the datasets folder and run `source download_data`
5. in the root folder of the project run the `train.py`:

```bash
python train.py --seed 42 --scheduler None --lr 1e-5 --warmup_epochs 0 --decay_epochs 5 --devices "[0,1,2,3]" --dataset goodbooks --sbert "sentence-transformers/all-mpnet-base-v2" --max_seq_length 384 --top_k 0 --batch_size 1024 --max_output 10000 --sbert_batch_size 250 --preproces_html false --evaluate_epoch false 
```

## Datasets and preprocessing

### Preprocessing information

We consider ratings of 4.0 and higher as an interaction. We only keep the users with at least 5 interactions.

### Statistics of datasets used for evalauation after preprocessing

|                        | GoodBooks-10k | MovieLens-20M |
|------------------------|---------------|---------------|
| # of items in X        | 10000         | 20720         |
| # of users in X        | 53366         | 136677        |
| # of interactions in X | 4122007       | 9990682       |
| density of X [%]       | 0.7724        | 0.3528        |
| density of X^TX [%]    | 41.08         | 22.82         |

## Hyperparameters

We used hyperparameters for training our models as follows.

| hyperparameter   | description                                                                                                          | goodbooks-mpnet-base-v2                 | movielens-mpnet-base-v2                 | goodlens-mpnet-base-v2                  |
|------------------|----------------------------------------------------------------------------------------------------------------------|-----------------------------------------|-----------------------------------------|-----------------------------------------|
| seed             | random seed used during training                                                                                     | 42                                      | 42                                      | 42                                      |
| scheduler        | learning rate scheduling strategy                                                                                    | constant learning rate                  | constant learning rate                  | constant learning rate                  |
| lr               | learning rate                                                                                                        | 1e-5                                    | 1e-5                                    | 1e-5                                    |
| epochs           | number of trained epochs                                                                                             | 5                                       | 5                                       | 10                                      |
| devices          | training script allow to train on multiple gpus in parralel - we used 4xV100                                         | [0,1,2,3]                               | [0,1,2,3]                               | [0,1,2,3]                               |
| dataset          | dataset used for training                                                                                            | goodbooks                               | ml20m                                   | ml20m,goodbooks                         |
| sbert            | original sentence transformer model used as an initial model for training                                            | sentence-transformers/all-mpnet-base-v2 | sentence-transformers/all-mpnet-base-v2 | sentence-transformers/all-mpnet-base-v2 |
| max_seq_length   | limitation of sequence length; shorter sequences trains faster original mpnet model uses max 512 tokens in. sequence | 384                                     | 256                                     | 384                                     |
| batch_size       | number of users sampled in random batch from interaction matrix                                                      | 2048                                    | 1024                                    | 1024                                    |
| max_output       | negative sampling hyperparameter (_m_ in the paper)                                                                  | 7500                                    | 10000                                   | 10000                                   |
| sbert_batch_size | number of items processed together during training step (gradient accumulation step size)                            | 200                                     | 250                                     | 200                                     |


