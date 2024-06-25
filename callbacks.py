from time import time
from typing import Callable
import keras
from models import SparseKerasELSA
import pandas as pd
from datasets.utils import *

class evaluateWriter(keras.callbacks.Callback):
    def __init__(self, items_idx, sbert, texts, evaluator, logdir, DEVICE, sbert_name="sbert_temp_model", evaluate_epoch="false", save_every_epoch="false"):
        super().__init__()
        self.evaluator = evaluator
        self.logdir = logdir
        self.sbert = sbert
        self.texts = texts
        self.items_idx=items_idx
        self.DEVICE=DEVICE
        self.results_list = []
        self.sbert_name = sbert_name
        self.evaluate_epoch = evaluate_epoch
        self.save_every_epoch = save_every_epoch
    
    def on_epoch_end(self, epoch, logs=None):
        print()
        if self.save_every_epoch=="true":
            print("saving sbert model")
            self.sbert.save(f"{self.sbert_name}-epoch-{epoch}")
        if self.evaluate_epoch == "true":
            embs = self.sbert.encode(self.texts, show_progress_bar=True)
            model = SparseKerasELSA(len(self.items_idx), embs.shape[1], self.items_idx, device=self.DEVICE)
            model.to(self.DEVICE)
            model.set_weights([embs])
            if isinstance(self.evaluator, ColdStartEvaluation):
                df_preds = model.predict_df(self.evaluator.test_src, candidates_df=self.evaluator.cold_start_candidates_df if hasattr(self.evaluator, "cold_start_candidates_df") else None, k=1000)
                df_preds = df_preds[~df_preds.set_index(["item_id", "user_id"]).index.isin(self.evaluator.test_src.set_index(["item_id", "user_id"]).index)]
            else:
                df_preds = model.predict_df(self.evaluator.test_src)
            results=self.evaluator(df_preds)
            
            # this should be logged to tensorboard after every epoch
            print(results)
            pd.Series(results).to_csv(f"{self.logdir}/result-epoch-{epoch}.csv")
            print("results file written")
            self.results_list.append(results)


class loggerCallback(keras.callbacks.Callback):
    def __init__(self, log_every_n_step: int=100, prefix: str="ELSA trainer", logger=print):
        super().__init__()
        self.log_every_n = log_every_n_step
        self.train_start = time()
        self.train_end = time()
        self.train_step_end = time()
        self.train_epoch_start = time()
        self.train_epoch_end = time()
        self.train_step_start = time()
        self.prefix = prefix
        if logger==print:
            self.log_fn=print
        else:
            self.log_fn=logger.info

    def on_train_begin(self, logs=None):
        self.train_start = time()
        self._log(f"Training started with params: {self.params}")
        stringlist = []
        self._model.summary(print_fn=lambda x, line_break: stringlist.append(x), expand_nested=True,show_trainable=True)
        for line in stringlist[0].split("\n"):
            self._log(line)
        

    def on_train_end(self, logs=None):
        self.train_end = time()
        time_taken = self.train_end - self.train_start
        self._log(f"Training finished in{self._format_time(time_taken, None)}.")
    
    def on_epoch_begin(self, epoch, logs=None):
        self.train_epoch_start = time()
        self.train_step_start = time()
        self._log(f"Training of epoch {epoch+1}/{self.params['epochs']} started.")
        
    def on_epoch_end(self, epoch, logs=None):
        self.train_epoch_end = time()
        time_taken = self.train_epoch_end - self.train_epoch_start
        log_ = self.format_log(self.params['steps'], time_taken/self.params['steps'], logs)
        self._log(log_)
        self._log(f"Training of epoch {epoch+1}/{self.params['epochs']} finished in{self._format_time(time_taken, None)}.")
                
    def on_train_batch_end(self, batch, logs=None):
        if batch//self.log_every_n==batch/self.log_every_n and logs is not None:
            self.train_step_end = time()
            time_taken = (self.train_step_end - self.train_step_start)
            step = batch
            log = self.format_log(step, time_taken, logs)
            self._log(log)
        if (batch+1)//self.log_every_n==(batch+1)/self.log_every_n and logs is not None and batch!=self.params['steps']-1:
            self.train_step_start = time()
        
    def format_log(self, step, time_taken, logs):
        now = time()
        log = []
        log.append(f"step {step}/{self.params['steps']}:")
        log.append(f"{self._format_time(time_taken)}")
        time_epoch_till_now = now-self.train_epoch_start
        log.append(f"{self._format_time(time_epoch_till_now, None)} elapsed")
        if step>0:
            time_per_step = time_epoch_till_now/step
            time_remaining = time_per_step*(self.params['steps']-step)
            log.append(f"{self._format_time(time_remaining, None)} remaining")
        else:
            log.append(f" ??? remaining")
        
        for k,v in logs.items():
            if abs(v) > 1e-3:
                info = f"{v:.4f}"
            else:
                info = f"{v:.4e}"
            log.append(f" {k}: {info}")
            
        return log[0]+" |".join(log[1:])

    def _log(self, *args):
        log=f"[{self.prefix}] {' '.join([str(x) for x in args])}"
        self.log_fn(log)
        
    
    def _format_time(self, time_per_unit, unit_name="step"):
        """format a given duration to display to the user.

        Given the duration, this function formats it in either milliseconds
        or seconds and displays the unit (i.e. ms/step or s/epoch).

        Args:
            time_per_unit: the duration to display
            unit_name: the name of the unit to display

        Returns:
            A string with the correctly formatted duration and units
        """
        formatted = ""
        if time_per_unit >= 1 or time_per_unit == 0:
            formatted += f' {time_per_unit:.0f}s{"/"+unit_name if unit_name is not None else ""}'
        elif time_per_unit >= 1e-3:
            formatted += f' {time_per_unit * 1000.0:.0f}ms{"/"+unit_name if unit_name is not None else ""}'
        else:
            formatted += f' {time_per_unit * 1000000.0:.0f}us{"/"+unit_name if unit_name is not None else ""}'
        return formatted


