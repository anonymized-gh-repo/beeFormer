# beeFormer

This is the official implementation provided with our paper "beeFormer: Bridging the Gap Between Semantic and Interaction Similarity in Recommender Systems". 

## main idea of beeFormer

Collaborative filtering methods can capture patterns from interaction data that are not obvious at first sight. For example, when buying a printer, users can also buy toners, papers, or cables to connect the printer, and collaborative filtering can take such patterns into account. However, in the cold-start recommendation setup, where new items do not have any interaction at all, collaborative filtering methods cannot be used, and recommender systems are forced to use other approaches, like content-based filtering. The problem with content-based filtering is that it relies on item attributes, such as text descriptions. In our printer example, semantic similarity-trained language models will put other printers closer than accessories that users might be searching for. Our method is training language models to learn these user behavior patterns from interaction data to transfer that knowledge to previously unseen items. Our experiments show that performance benefits from this approach are enormous.

## updates

27.6.2024:
  - added LLM-generated data to the dataset folders
  - added scripts to download movielens and goodbooks datasets
  - added the description of the LLM-generating procedure
  - added tables with results

## Steps to start training the models:

1. create virtual environment `python3.10 -m venv beef` and activate it `source beef/bin/activate`
2. clone this repository and navigate to it `cd beeFormer`
3. install packages `pip install -r requirements.txt`
4. download the data for movielens: navigate to the `dataset/ml20m` folder and run `source download_data`
5. download the data for goodbooks: navigate to the `dataset/goodbooks` folder and run `source download_data`
6. in the root folder of the project run the `train.py`:

```bash
python train.py --seed 42 --scheduler None --lr 1e-5 --warmup_epochs 1 --decay_epochs 4 --devices "[0,1,2,3]" --dataset goodbooks --sbert "sentence-transformers/all-mpnet-base-v2" --max_seq_length 384 --top_k 0 --batch_size 1024 --max_output 10000 --sbert_batch_size 250 --preproces_html false --evaluate_epoch false --use_cold_start true --save_every_epoch true --model_name my_model
```

7. Evaluate the results

```bash
python evaluate.py --seed 42 --dataset goodbooks --sbert my-model
```

## Datasets and preprocessing

### Preprocessing information

We consider ratings of 4.0 and higher as an interaction. We only keep the users with at least 5 interactions.

### Statistics of datasets used for evaluation after basic preprocessing

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

## LLM Data augmentations

Since there are no text descriptions in the original data, originally we manually connect several datasets with the original data and train our models on it. However, this approach has several limitations: texts from different sources have different styles and different lengths, and this might influence the results. Therefore, we use the Llama-3-8b-instruct model to generate item descriptions for us. We use the following conversation template:

```python
import pandas as pd

from tqdm import tqdm
from vllm import LLM, SamplingParams

items = pd.read_feather("items_with_gathered_side_info.feather")

llm = LLM(model="meta-llama/Meta-Llama-3-8B-Instruct",dtype="float16")

tokenizer = llm.get_tokenizer()
conversation = [ tokenizer.apply_chat_template(
        [
            {'role': 'system','content':"You are ecomerce shop designer. Given a item description create one paragraph long summarization of the product."},
            {'role': 'user', 'content': "Item description: "+x},
            {'role': 'assistant', 'content': "Sure, here is your one paragraph summary of your product:"},
        ],
        tokenize=False,
    ) for x in tqdm(items.gathered_features.to_list())]

output = llm.generate(
    conversation,
    SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=512,
        stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")],  
    )
)

items_descriptions = [o.outputs[0].text for o in output]
```

However, LLM refused to generate descriptions for some items (For example, because it refuses to generate explicit content). We removed such items from the dataset. We also removed items for which we were not able to connect meaningful descriptions from other datasets, which led to LLM completely hallucinating item descriptions. The final statistics of the data changed accordingly:

|                        | GoodBooks-10k | MovieLens-20M |
|------------------------|---------------|---------------|
| # of items in X        | 9992          | 16919         |
| # of users in X        | 53366         | 136590        |
| # of interactions in X | 4121217       | 9698239       |
| density of X [%]       | 0.7729        | 0.4197        |
| density of X^TX [%]    | 41.11         | 26.90         |

We share the resulting LLM-generated item descriptions in `datasets/ml20m` and `dataset/goodbooks` folders.

## Results (item-based splitting)

### Results with Llama-3-8b-instruct generated items description as a side information

Training/evaluation in progress, we are adding the results as we get them.

| Dataset  Scenario | Sentence Transformer                | R@20 | R@50 | N@100 |
|-------------------|-------------------------------------|:----:|:----:|:-----:|
| Books  CBF        | all-mpnet-base-v2                   |0.1124|0.1948| 0.1822|
| Books  CBF        | nomic-embed-text-v1.5               |0.1298|0.2225| 0.2164|
| Books  CBF        | bge-m3                              |0.1237|0.2050| 0.1963|
| Books  CBF        | movielens-mpnet-base-v2 (zero-shot) |0.1708|0.2644| 0.2622|
| Books  CBF        | goodbooks-mpnet-base-v2             |0.2512|0.3871| 0.3804|
| Books  CBF        | goodlens-mpnet-base-v2              |      |      |       |
| Books  Heater     | all-mpnet-base-v2                   |      |      |       |
| Books  Heater     | nomic-embed-text-v1.5               |      |      |       |
| Books  Heater     | bge-m3                              |      |      |       |
| Books  Heater     | movielens-mpnet-base-v2             |      |      |       |
| Books  Heater     | goodbooks-mpnet-base-v2             |      |      |       |
| Books  Heater     | goodlens-mpnet-base-v2              |      |      |       |
| Movies  CBF       | all-mpnet-base-v2                   |0.1664|0.2697| 0.1676|
| Movies  CBF       | nomic-embed-text-v1.5               |0.1167|0.2181| 0.1411|
| Movies  CBF       | bge-m3                              |0.1311|0.2339| 0.1506|
| Movies  CBF       | movielens-mpnet-base-v2             |0.4120|0.5626| 0.4010|
| Movies  CBF       | goodbooks-mpnet-base-v2 (zero-shot) |0.3052|0.4281| 0.2882|
| Movies  CBF       | goodlens-mpnet-base-v2              |      |      |       |
| Movies  Heater    | all-mpnet-base-v2                   |      |      |       |
| Movies  Heater    | nomic-embed-text-v1.5               |      |      |       |
| Movies  Heater    | bge-m3                              |      |      |       |
| Movies  Heater    | movielens-mpnet-base-v2             |      |      |       |
| Movies  Heater    | goodbooks-mpnet-base-v2             |      |      |       |
| Movies  Heater    | goodlens-mpnet-base-v2              |      |      |       |

### Results with raw text data collected from various datasets as a side information 

These are original results currently present in the paper.

| Dataset  Scenario | Sentence Transformer                |  R@20  |  R@50  |  N@100 |
|-------------------|-------------------------------------|:------:|:------:|:------:|
| Books  CBF        | all-mpnet-base-v2                   | 0.0911 | 0.1647 | 0.1512 |
| Books  CBF        | nomic-embed-text-v1.5               | 0.0935 | 0.1645 | 0.1475 |
| Books  CBF        | bge-m3                              | 0.0776 | 0.1396 | 0.1333 |
| Books  CBF        | movielens-mpnet-base-v2 (zero-shot) | 0.1515 | 0.2471 | 0.2317 |
| Books  CBF        | goodbooks-mpnet-base-v2             | 0.1400 | 0.2280 | 0.2158 |
| Books  CBF        | goodlens-mpnet-base-v2              | **0.2249** | **0.3592** | **0.3302** |
| Books  Heater     | all-mpnet-base-v2                   | 0.1659 | 0.2683 | 0.2565 |
| Books  Heater     | nomic-embed-text-v1.5               | 0.1724 | 0.2718 | 0.2612 |
| Books  Heater     | bge-m3                              | 0.1585 | 0.2462 | 0.2428 |
| Books  Heater     | movielens-mpnet-base-v2             | 0.1785 | 0.2776 | 0.2656 |
| Books  Heater     | goodbooks-mpnet-base-v2             | 0.1751 | 0.2745 | 0.2618 |
| Books  Heater     | goodlens-mpnet-base-v2              | **0.2096** | **0.3188** | **0.3050** |
| Movies  CBF       | all-mpnet-base-v2                   | 0.0607 | 0.1160 | 0.0807 |
| Movies  CBF       | nomic-embed-text-v1.5               | 0.0462 | 0.1013 | 0.0666 |
| Movies  CBF       | bge-m3                              | 0.0549 | 0.1003 | 0.0714 |
| Movies  CBF       | movielens-mpnet-base-v2             | **0.3179** | 0.4460 | **0.3201** |
| Movies  CBF       | goodbooks-mpnet-base-v2 (zero-shot) | 0.2838 | 0.4416 | 0.3101 |
| Movies  CBF       | goodlens-mpnet-base-v2              | 0.3087 | **0.4587** | 0.3159 |
| Movies  Heater    | all-mpnet-base-v2                   | 0.2060 | 0.3477 | 0.2355 |
| Movies  Heater    | nomic-embed-text-v1.5               | 0.1810 | 0.2883 | 0.2120 |
| Movies  Heater    | bge-m3                              | 0.1395 | 0.2298 | 0.1691 |
| Movies  Heater    | movielens-mpnet-base-v2             | **0.3245** | 0.4711 | **0.3355** |
| Movies  Heater    | goodbooks-mpnet-base-v2             | 0.2997 | 0.4533 | 0.3090 |
| Movies  Heater    | goodlens-mpnet-base-v2              | 0.3133 | **0.4592** | 0.3219 |

## Results (user-based splitting)

TBA

## Results (timestamp-based splitting)

TBA
