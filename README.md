# beeFormer

This is the official implementation provided with our paper "beeFormer: Bridging the Gap Between Semantic and Interaction Similarity in Recommender Systems". 

Steps for reproducing the results from our paper:

1. create virtual environment `python3.10 -m venv beef` and activate it `source beef/bin/activate`
2. clone this repository and navigate to it `cd beeFormer`
3. install packages `pip install -r requirements.txt`
4. download the data: navigate to the datasets folder and run `source download_data`
5. in the root folder of the project run the `train.py`:

```bash
python train.py --seed 42 --scheduler None --lr 1e-5 --warmup_epochs 0 --decay_epochs 5 --devices "[0,1,2,3]" --dataset goodbooks --sbert "sentence-transformers/all-mpnet-base-v2" --max_seq_length 384 --top_k 0 --batch_size 1024 --max_output 10000 --sbert_batch_size 250 --preproces_html false --evaluate_epoch false 
```