import os

os.environ["KERAS_BACKEND"] = "torch"

import keras
import math 
import numpy as np
import torch

from dataloaders import beeformerDataset
from datasets.pydatasets import SparseRecSysDatasetWithNegatives
from models import NMSEbeeformer, SparseKerasELSA, simpleBee
from schedules import LinearWarmup

import sentence_transformers
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

#from read_data import *
from utils import *
from datasets.utils import preproces_html
from config import config
from datasets.utils import *

import argparse
from callbacks import evaluateWriter

import time
import subprocess

parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--device", default=None, type=str, help="Limit device to run on")
parser.add_argument("--devices", default=None, type=str, help="Devices for multi-device training ex. [0,1,2,3]")
parser.add_argument("--flag", default="none", type=str, help="flag for distinction of experiments, default none")
parser.add_argument("--validation", default="false", type=str, help="Use validation split: true/false")

# lr
parser.add_argument("--lr", default=.001, type=float, help="Learning rate for model training, only if scheduler is none")
parser.add_argument("--scheduler", default="none", type=str, help="Scheduler: LinearWarmup, CosineDecay or none")
parser.add_argument("--init_lr", default=.0, type=float, help="starting lr, only if scheduler is not none")
parser.add_argument("--warmup_lr", default=.002, type=float, help="max warmup lr, only if scheduler is not none")
parser.add_argument("--target_lr", default=.001, type=float, help="final lr, only if scheduler is LinearWarmup")
parser.add_argument("--warmup_epochs", default=0, type=int, help="Warmup epochs")
parser.add_argument("--decay_epochs", default=0, type=int, help="Decay epochs")
parser.add_argument("--tuning_epochs", default=1, type=int, help="Final lr epochs, only if scheduler is LinearWarmup or none")
# dataset
parser.add_argument("--dataset", default="-", type=str, help="Dataset to run on")
parser.add_argument("--use_partial", default="false", type=str, help="Use only half of the data for training.")
parser.add_argument("--partial", default=.0, type=float, help="portion of items to return from excluded items")
parser.add_argument("--use_cold_start", default="false", type=str, help="Use cold start evaluation.")

# sentence transformer details
parser.add_argument("--sbert", default="-", type=str, help="Input sentence transformer model to train")
parser.add_argument("--max_seq_length", default=128, type=int, help="Maximum sequence length")
parser.add_argument("--preproces_html", default="false", type=str, help="whether to get rid of html inside descriptions")
parser.add_argument("--add_dense", default="false", type=str, help="add dense layer on the output of the sentence transformer [true/false]")
parser.add_argument("--dense_hidden_size", default=2000, type=int, help="Dense hidden size (int). Only if add_dense is true")
parser.add_argument("--dense_output_size", default=768, type=int, help="Dense output size (int). Only if add_dense is true")
parser.add_argument("--train_only_dense", default="true", type=str, help="If set to true, transformer weights will be frozen. Only if add_dense is true. Supports only true rn.[true/false]")

# model hyperparams
parser.add_argument("--max_output", default=10000, type=int, help="Max number of items on output for super sparse method.")
parser.add_argument("--batch_size", default=1024, type=int, help="Batch size of sampled users per training step")
parser.add_argument("--top_k", default=2500, type=int, help="Optimize only for top-k predictions on the output of the model")
parser.add_argument("--sbert_batch_size", default=128, type=int, help="Batch size for computing embeddings with sentence transformer")
# output model name
parser.add_argument("--model_name", default="my_model", type=str, help="Otput sentence transformer model name to train")
# tensorboard
parser.add_argument("--log_dir", default="logs", type=str, help="Otput sentence transformer model name to train")
# evaluate
parser.add_argument("--evaluate", default="true", type=str, help="final evaluation after training [true/false]")
parser.add_argument("--evaluate_epoch", default="true", type=str, help="evaluation after every epoch [true/false]")
parser.add_argument("--save_every_epoch", default="false", type=str, help="save after every epoch [true/false]")



args = parser.parse_args([] if "__file__" not in globals() else None)
print(args)

if args.device is not None:
    print(f"Limiting devices to {args.device}")
    os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.device}"

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device {DEVICE}")

def NMSE(x,y):
    x=torch.nn.functional.normalize(x, dim=-1)
    y=torch.nn.functional.normalize(y, dim=-1)
    return keras.losses.mean_squared_error(x,y)

def get_sbert_dense(module, activation):
    print(module.weight.shape)
    use_bias = False if module.bias is None else True
    print(use_bias)
    return sentence_transformers.models.Dense(module.in_features, module.out_features, 
                                              init_weight=module.weight, 
                                              bias=use_bias,
                                              init_bias=module.bias, 
                                              activation_function=activation,
    )

def main(args):
    folder = f"results/{str(pd.Timestamp('today'))} {9*int(1e6)+np.random.randint(999999)}".replace(" ", "_")
    if not os.path.exists(folder):
        os.makedirs(folder)
    vargs = vars(args)
    vargs["cuda_or_cpu"]=DEVICE
    pd.Series(vargs).to_csv(f"{folder}/setup.csv")
    print(f"Saving results to {folder}")
    torch.manual_seed(args.seed)
    keras.utils.set_random_seed(args.seed)
    np.random.seed(args.seed)
    print(f"seeds setted to {args.seed}")

    #print("Reading evaluators")
    #evaluators = get_eval()
    #print("Reading datasets")
    #datasets = get_data()
    #print("Reading items")
    #items = get_items()
    #items["description"]=items["description"].fillna("[]")
    #items["title"]=items["title"].fillna("Unknown")

    #if args.dataset in datasets.keys():
    #    print(f"Selecting dataset {args.dataset}")
    #    dataset,evaluator = datasets[args.dataset], evaluators[args.dataset]
    #    dataset = dataset[0]
    #    items_d = items[items.asin.isin(dataset.full_train_interactions.item_id)]
    if args.dataset in config.keys():
        dataset, params = config[args.dataset]
        dataset.load_interactions(**params)
        if args.use_partial=="true":
            dataset.partial = args.partial
            evaluator = PartialEvaluation(dataset)
        elif args.use_cold_start=="true":
            evaluator = ColdStartEvaluation(dataset)
        else:
            evaluator = Evaluation(dataset)
        items_d = dataset.items_texts
        items_d["asin"]=items_d.item_id
        print(dataset)
    else:
        print("Unknown dataset. List of available datsets: \n")
        for x in datasets.keys():
            print(x)
        return 

    print("Preprocessing texts.")
    #am_itemids = items_d.asin.to_numpy()
    #am_locator = np.array([np.argwhere(am_itemids==q).item() for q in tqdm(dataset.all_interactions.item_id.cat.categories)])
    am_itemids = items_d.asin.to_numpy()
    cc = np.array(dataset.full_train_interactions.item_id.cat.categories)
    ccdf = pd.Series(cc).to_frame()
    ccdf.columns=["item_id"]
    amdf = pd.Series(am_itemids).to_frame().reset_index()
    amdf.columns=["idx", "item_id"]
    am_locator = pd.merge(how="inner", left=ccdf, right=amdf).idx.to_numpy()

    if args.dataset in config.keys():
        am_texts = items_d._text_attributes
    elif args.preproces_html=="true":
        am_texts=items_d.fillna(0).apply(lambda row: f"{row.title}: {preproces_html('. '.join(eval(row.description)))}", axis=1)
    else:
        print("using html preprocessing")
        am_texts=items_d.fillna(0).apply(lambda row: f"{row.title}: {'. '.join(eval(row.description))}", axis=1)

    am_texts=am_texts.to_numpy()[am_locator]

    print("Creating sbert")

    sbert = SentenceTransformer(args.sbert, device=DEVICE)

    sbert.max_seq_length = args.max_seq_length
    am_tokenized = sbert.tokenize(am_texts)
    
    if args.devices is not None:
        print(f"Will run sbert on devices {args.devices}")
        module_sbert = torch.nn.DataParallel(sbert, device_ids=eval(args.devices), output_device=0)
    else:
        module_sbert = sbert

    print("Creating interaction matrix for training")
    X = get_sparse_matrix_from_dataframe(dataset.full_train_interactions)

    if args.add_dense=="false":
        print("Creating dataloader")
        datal = beeformerDataset(X, am_tokenized, DEVICE, shuffle=True, max_output=args.max_output, batch_size=args.batch_size)
        steps_per_epoch = len(datal)
        print(sbert)
        model = NMSEbeeformer(
            tokenized_sentences=am_tokenized, 
            items_idx=dataset.full_train_interactions.item_id.cat.categories, 
            sbert=keras.layers.TorchModuleWrapper(module_sbert), 
            device=DEVICE,
            top_k=args.top_k,
            sbert_batch_size=args.sbert_batch_size,
            #sentences = am_texts
        )


        if args.scheduler == "CosineDecay":
            schedule = keras.optimizers.schedules.CosineDecay(
                0.0,
                steps_per_epoch*(args.decay_epochs+args.warmup_epochs),
                alpha=0.0,
                name="CosineDecay",
                warmup_target=args.warmup_lr,
                warmup_steps=steps_per_epoch*args.warmup_epochs,
            )
            print("Using schedule with config", schedule.get_config())
        elif args.scheduler == "LinearWarmup":
            schedule = LinearWarmup(
                warmup_steps=steps_per_epoch*args.warmup_epochs,
                decay_steps=steps_per_epoch*args.decay_epochs,
                starting_lr=args.init_lr, 
                warmup_lr=args.warmup_lr, 
                final_lr=args.target_lr, 
            )
            print("Using schedule with config", schedule.get_config())
        else:
            schedule = args.lr
            print("Using constant learning rate of", schedule)

        # this cause tensorflow import and crashes everything during train step in fit method :(
        #tb_callback = keras.callbacks.TensorBoard(
        #    log_dir=args.log_dir,
        #    histogram_freq=0,
        #    write_graph=False,
        #    write_images=False,
        #    write_steps_per_second=False,
        #    update_freq="epoch",
        #    profile_batch=0,
        #    embeddings_freq=0,
        #    embeddings_metadata=None,
        #)


        model.to(DEVICE)
        cbs = []
        #if args.evaluate_epoch=="true":
        eval_cb = evaluateWriter(items_idx=dataset.full_train_interactions.item_id.cat.categories,sbert=sbert,evaluator=evaluator,logdir=folder, DEVICE=DEVICE, texts=am_texts, sbert_name=args.model_name, evaluate_epoch=args.evaluate_epoch, save_every_epoch=args.save_every_epoch)
        cbs.append(eval_cb)

        # normalization is inside the model
        model.compile(optimizer=keras.optimizers.Nadam(learning_rate=schedule), loss=NMSE, metrics=[keras.metrics.CosineSimilarity()])

        print("Building the model")  
        model.train_step(datal[0])
        model.built = True
        print(model.summary())
        print("Starting training loop")
        train_time = 0
        fits=[]
    
        print(f"Training for {args.warmup_epochs+args.decay_epochs+args.tuning_epochs} epochs.")
        f=model.fit(
            datal, 
            epochs=args.warmup_epochs+args.decay_epochs+args.tuning_epochs, 
            callbacks=cbs,    # wont work without tensorflow
        )
        fits.append(f)
        train_time = time.time()-train_time
        sbert.save(args.model_name)
    else:
        #add dense layers on top of the transformer and train them
        embs = sbert.encode(am_texts, show_progress_bar=True)
        model = simpleBee(embs, args.dense_output_size, args.dense_hidden_size , dataset.full_train_interactions.item_id.cat.categories, DEVICE)
        data_loader = SparseRecSysDatasetWithNegatives(X, device=DEVICE, batch_size=1024, max_output=7500, shuffle=True)  

        if args.scheduler == "CosineDecay":
            schedule = keras.optimizers.schedules.CosineDecay(
                0.0,
                steps_per_epoch*(args.decay_epochs+args.warmup_epochs),
                alpha=0.0,
                name="CosineDecay",
                warmup_target=args.warmup_lr,
                warmup_steps=steps_per_epoch*args.warmup_epochs,
            )
            print("Using schedule with config", schedule.get_config())
        elif args.scheduler == "LinearWarmup":
            schedule = LinearWarmup(
                warmup_steps=steps_per_epoch*args.warmup_epochs,
                decay_steps=steps_per_epoch*args.decay_epochs,
                starting_lr=args.init_lr, 
                warmup_lr=args.warmup_lr, 
                final_lr=args.target_lr, 
            )
            print("Using schedule with config", schedule.get_config())
        else:
            schedule = args.lr
            print("Using constant learning rate of", schedule)
        
        model.compile(optimizer=keras.optimizers.Nadam(learning_rate=schedule), loss=NMSE, metrics=[keras.metrics.CosineSimilarity()])
        model.train_step(data_loader[0])
        model.built = True
        print(model.summary())
        print("Starting training loop")
        train_time = 0
        fits=[]
        print(f"Training for {args.warmup_epochs+args.decay_epochs+args.tuning_epochs} epochs.")
        f=model.fit(
            data_loader, 
            epochs=args.warmup_epochs+args.decay_epochs+args.tuning_epochs, 
        )
        fits.append(f)
        train_time = time.time()-train_time

        sbert_modules=[sbert.get_submodule(key) for key in sbert._modules.keys()]

        sbert_modules+=[
            get_sbert_dense(model.hidden.module, model.relu.module),
            get_sbert_dense(model.transform.module, model.tanh.module),
            sentence_transformers.models.Normalize()
        ]

        sbert = SentenceTransformer(
                    modules=sbert_modules
        )

        print(sbert)
        sbert.save(args.model_name)

    am_itemids = items_d.asin.to_numpy()
    cc = np.array(dataset.all_interactions.item_id.cat.categories)
    ccdf = pd.Series(cc).to_frame()
    ccdf.columns=["item_id"]
    amdf = pd.Series(am_itemids).to_frame().reset_index()
    amdf.columns=["idx", "item_id"]
    am_locator = pd.merge(how="inner", left=ccdf, right=amdf).idx.to_numpy()

    if args.dataset in config.keys():
        am_texts = items_d._text_attributes
    elif args.preproces_html=="true":
        am_texts=items_d.fillna(0).apply(lambda row: f"{row.title}: {preproces_html('. '.join(eval(row.description)))}", axis=1)
    else:
        print("using html preprocessing")
        am_texts=items_d.fillna(0).apply(lambda row: f"{row.title}: {'. '.join(eval(row.description))}", axis=1)

    am_texts=am_texts.to_numpy()[am_locator]

    if args.evaluate == "true":
        embs = sbert.encode(am_texts, show_progress_bar=True)
        model = SparseKerasELSA(len(dataset.all_interactions.item_id.cat.categories), embs.shape[1], dataset.all_interactions.item_id.cat.categories, device=DEVICE)
        model.to(DEVICE)
        model.set_weights([embs])
        if args.use_cold_start:
            df_preds = model.predict_df(evaluator.test_src, candidates_df=evaluator.cold_start_candidates_df if hasattr(evaluator, "cold_start_candidates_df") else None, k=1000)
            df_preds = df_preds[~df_preds.set_index(["item_id", "user_id"]).index.isin(evaluator.test_src.set_index(["item_id", "user_id"]).index)]
        else:
            df_preds = model.predict_df(evaluator.test_src)
        

        results=evaluator(df_preds)
        
        # this should be logged to tensorboard after every epoch
        print(results)
        pd.Series(results).to_csv(f"{folder}/result.csv")
        print("results file written")

    ks = list(f.history.keys())    
    dc = {k:np.array([(f.history[k]) for f in fits]).flatten() for k in ks}
    dc["epoch"] = np.arange(len(dc[list(dc.keys())[0]]))+1
    df = pd.DataFrame(dc)
    df[list(df.columns[-1:])+list(df.columns[:-1])]
    
    df.to_csv(f"{folder}/history.csv")
    print("history file written")
    
    try:
        pd.concat([pd.Series(x).to_frame().T for x in eval_cb.results_list]).to_csv(f"{folder}/results-history.csv")
    except:
        print("eval_cb not exist")

    pd.Series(train_time).to_csv(f"{folder}/timer.csv")
    print("timer written")

    out = subprocess.check_output(
        [
            "nvidia-smi"
        ]
    )
    
    with open(os.path.join(folder, f'{args.dataset}_{args.flag}.log'), 'w') as f:
        f.write(out.decode("utf-8"))

if __name__ == "__main__":
    main(args)