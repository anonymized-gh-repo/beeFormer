
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import time
import warnings
from pandas.core.generic import SettingWithCopyWarning
#from pandas.core.common import SettingWithCopyWarning
import recpack.metrics
from ipywidgets import IntProgress
from IPython.display import display
from math import ceil,floor
import scipy.sparse
import torch
import scipy
from tqdm import tqdm
from bs4 import BeautifulSoup

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

import re
def striptags(data):
    try: 
        p = re.compile(r'\[.*?\]')
        return p.sub('', data)
    except:
        return ""

def preproces_html(html):
    soup = BeautifulSoup(html, "lxml")
    return soup.text

def get_random_indices(row, frac=0.2, part=0):
    a = row.indices
    pick = ceil(len(a)*0.2)
    if part==0:
        return np.random.choice(a, pick)
    q=[]
    for i in range(int(1/0.2)):
        q.append(a[i*pick:i*pick+pick])
    return q[part]

def get_src_target_rand(X_val):
    X_val_src = X_val.copy()
    for i in range(X_val_src.shape[0]):
        ind = get_random_indices(X_val_src[i])
        X_val_src[i,ind]=0

    X_val_src.eliminate_zeros()
    X_val_targets=X_val-X_val_src



    bl = torch.from_numpy(1-X_val_src.toarray()).to("cpu")
    target = torch.from_numpy(X_val_targets.toarray().astype(bool))
    return X_val_src, X_val_targets

def get_src_target_fold(X_val, fold=0):
    X = []
    XV = []
    X_val_src = X_val.copy()
    for i in tqdm(range(X_val_src.shape[0])):
        ind = get_random_indices(X_val_src[i], 1)
        X_val_src[i,ind]=0
    X.append(X_val_src)
    XV.append(X_val)
    if fold!=1:
        X_val_src = X_val.copy()
        for i in tqdm(range(X_val_src.shape[0])):
            ind = get_random_indices(X_val_src[i], 2)
            X_val_src[i,ind]=0
        X.append(X_val_src)
        XV.append(X_val)

        X_val_src = X_val.copy()
        for i in tqdm(range(X_val_src.shape[0])):
            ind = get_random_indices(X_val_src[i], 3)
            X_val_src[i,ind]=0
        X.append(X_val_src)
        XV.append(X_val)

        X_val_src = X_val.copy()
        for i in tqdm(range(X_val_src.shape[0])):
            ind = get_random_indices(X_val_src[i], 4)
            X_val_src[i,ind]=0
        X.append(X_val_src)
        XV.append(X_val)

        X_val_src = X_val.copy()
        for i in tqdm(range(X_val_src.shape[0])):
            ind = get_random_indices(X_val_src[i], 5)
            X_val_src[i,ind]=0
        X.append(X_val_src)
        XV.append(X_val)
        
    X_val_src = scipy.sparse.vstack(X)
    X_val = scipy.sparse.vstack(XV)
    
    X_val_src.eliminate_zeros()
    X_val_targets=X_val-X_val_src



    #bl = torch.from_numpy(1-X_val_src.toarray())
    #target = torch.from_numpy(X_val_targets.toarray().astype(bool))
    
    return X_val_src, X_val_targets

def get_get_src_target_rand_df(test_interactions):

    X_test = get_sparse_matrix_from_dataframe(test_interactions)
    X_test_src, X_test_target = get_src_target_rand(X_test)
    df_src = sparse_matrix_to_df(X_test_src, test_interactions.item_id.cat.categories, test_interactions.user_id.cat.categories)
    df_target = sparse_matrix_to_df(X_test_target, test_interactions.item_id.cat.categories, test_interactions.user_id.cat.categories)
    
    return df_src, df_target, X_test_src, X_test_target

def get_get_src_target_rand_df_fold(test_interactions, fold=0):

    X_test = get_sparse_matrix_from_dataframe(test_interactions)
    X_test_src, X_test_target = get_src_target_fold(X_test, fold)
    if X_test_src.shape[0]!=len(test_interactions.user_id.cat.categories):
        uids = pd.Index(np.arange(X_test_src.shape[0]).astype(str))
    else:
        uids = test_interactions.user_id.cat.categories
    df_src = sparse_matrix_to_df(X_test_src, test_interactions.item_id.cat.categories, uids)
    df_target = sparse_matrix_to_df(X_test_target, test_interactions.item_id.cat.categories, uids)
    
    return df_src, df_target, X_test_src, X_test_target


def sparse_matrix_to_df(X, item_ids, user_ids, verbose=10000):
    #if verbose>0:
    #    f = IntProgress(min=0, max=X.shape[0]//verbose+2) # instantiate the bar
    #    display(f) # display the bar

    split = np.split(X.indices, X.indptr)[1:-1]
    split2 = np.split(X.data, X.indptr)[1:-1]

    #if verbose>0:    
    #    f.value += 1
    
    dfs = []

    for i in tqdm(range(len(split))):
        dfs.append(pd.DataFrame({"user_id":user_ids[i], "item_id":item_ids[split[i]], "value":split2[i]}))
    
    ret = pd.concat(dfs)
    ret["user_id"]=ret["user_id"].astype(str).astype('category').cat.remove_unused_categories()
    ret["item_id"]=ret["item_id"].astype(str).astype('category').cat.remove_unused_categories()

    return ret



class logger:
    @staticmethod
    def info(*args):
        print(*args)
    @staticmethod
    def debug(*args):
        print(*args)

def convert_user_item_pairs_into_sparse_matrix(interactions: pd.DataFrame, sparse_type):
        """
        Create sparse matrix from the interaction DataFrame.
        Parameters
        ----------
        interactions : pandas.DataFrame
            DataFrame containing interactions with columns 'user_id' (.select_dtypes(['object'])), 'item_id' (category) and 'value' (float)
                where can be maximal one value for each user-item pair.
        sparse_type : str
            Type of the sparse matrix. Allowed values are 'csc' and 'csr'.
        Returns
        -------
        tuple
            First element is a list of item IDs that can served as row indexes to created matrix.
            Second element is a list of user IDs that can served as column indexes to created matrix.
            Third element is created sparse matrix.
        """
        if len(interactions) == 0:
            return [], [], InteractionPreparator.SPARSE_MATRIXES[sparse_type](([], ([], [])), shape=(0, 0), dtype=np.float64)

        return (
            interactions["item_id"].cat.categories,
            interactions["user_id"].cat.categories,
            csr_matrix(
                (
                    interactions["value"].values,
                    (interactions["item_id"].cat.codes, interactions["user_id"].cat.codes),
                ),
                shape=(len(interactions["item_id"].cat.categories), len(interactions["user_id"].cat.categories)),
                dtype=np.float64,
            ),
        )

def get_sparse_matrix_from_dataframe(df, item_indices=None, user_indices=None):
    
    if item_indices is None:
        item_indices = df.item_id.cat.categories

    if user_indices is None:
        user_indices = df.user_id.cat.categories
    
    df = df.copy()
    df = df[df.item_id.isin(item_indices)]
    df = df[df.user_id.isin(user_indices)]
    df["user_id"]=df.user_id.astype("category")
    
    row_ind = [item_indices.get_loc(x) for x in df.item_id]
    col_ind = [user_indices.get_loc(x) for x in df.user_id]
        
    
    mat = csr_matrix(
        (
            df.value.values,
            (row_ind, col_ind),
        ),
        shape=(len(item_indices), len(user_indices)),
        dtype=np.float64,
    )
    
    return mat.T.tocsr()
    
def fast_pruning(
    interactions: pd.DataFrame, 
    pruning_user: int, 
    pruning_item: int, 
    logger=logger, 
    item_users_are_unique: bool = False,
    max_user_support: int = 0,
    max_item_support: int = 0,
    max_steps: int = 0,
) -> pd.DataFrame:
    stable = False
    step = 1
    item_map, user_map, X = convert_user_item_pairs_into_sparse_matrix(interactions, "csr")
    X=X.astype(bool).T
    users_cnt_old = len(interactions["user_id"].cat.categories)
    items_cnt_old = len(interactions["item_id"].cat.categories)
    logger.info("Starting reduction: {} interactions, {} pruning_user, {} pruning_item".format(X.getnnz(), pruning_user, pruning_item))
    while not stable:
        logger.debug("Number of interactions at the start of {} step: {}".format(step, X.getnnz()))
        stable = True
        
        
        number_of_items = len(item_map)
        matching_items = np.where(X.sum(0)>=pruning_item)[1]
        X = X[:, matching_items]
        if max_item_support>0:
            matching_items = np.where(X.sum(0)<=max_item_support)[0]
            X = X[:, matching_items]
            
        item_map = item_map[matching_items]
        number_of_items_with_support = len(item_map)
        logger.info(
                "Total number of items in {} step: {}. Number of items with minimal support of {} users: {} => removing {} items".format(
                    step, number_of_items, pruning_item, number_of_items_with_support, number_of_items - number_of_items_with_support
                )
            )
        logger.debug("Number of interactions after removing items in {} step: {}".format(step, X.getnnz()))
        if number_of_items > number_of_items_with_support:
            stable = False
        
        number_of_users = len(user_map)
        matching_users = np.where(X.sum(1)>=pruning_user)[0]
        X = X[matching_users, :]
        
        if max_user_support>0:
            matching_users = np.where(X.sum(1)<=max_user_support)[0]
            X = X[matching_users, :]
            
        user_map = user_map[matching_users]
        number_of_users_with_support = len(user_map)
        logger.info(
                "Total number of users in {} step: {}. Number of users with minimal support of {} items: {} => removing {} users".format(
                    step, number_of_users, pruning_user, number_of_users_with_support, number_of_users - number_of_users_with_support
                )
            )
        logger.debug("Number of interactions after removing users in {} step: {}".format(step, X.getnnz()))
        if number_of_users > number_of_users_with_support:
            stable = False

        
        
        if max_steps>0 and step>=max_steps:
            stable=True

        if stable:
            logger.info("Data stable after {} reduction steps ({} users, {} items)".format(step, number_of_users, number_of_items))
        step += 1
        
    now = time.time()    
    interactions = interactions[(interactions.user_id.isin(user_map))&(interactions.item_id.isin(item_map))]
    print()
    interactions["user_id"] = interactions["user_id"].cat.remove_unused_categories()
    interactions["item_id"] = interactions["item_id"].cat.remove_unused_categories()
    #interactions["user_id"].cat.remove_unused_categories(inplace=True)
    #interactions["item_id"].cat.remove_unused_categories(inplace=True)
    logger.info(
        """Due to a pruning, the number of unique users and items could changed:
            Users: {} => {}
            Items: {} => {}""".format(
            users_cnt_old, len(interactions["user_id"].cat.categories), items_cnt_old, len(interactions["item_id"].cat.categories)
        )
    )
    return interactions

def df_recall(self, df, targets, k):
        df=df[df.value<=k]
        mat = get_sparse_matrix_from_dataframe(
            df, 
            targets.item_id.cat.categories, 
            targets.user_id.cat.categories
        )
        X_test_target = get_sparse_matrix_from_dataframe(
            targets, 
            targets.item_id.cat.categories, 
            targets.user_id.cat.categories
        )
        denominator = X_test_target.sum(1)
        denominator[denominator.sum(1)>k]=k
        return (X_test_target.multiply(mat).astype(bool).sum(1)/denominator).mean()

def df_ndcg(self, df, targets, k):
    df=df[df.value<=k]
    tdf=targets
    bl=df.copy()
    bl["value"]=1/np.log2(bl.value+1)
    ndcgdf = pd.merge(how="inner", left=tdf, right=bl, right_on=["user_id","item_id"], left_on=["user_id","item_id"])
    dcg = ndcgdf.groupby(["user_id"]).sum("value_y").reset_index().value_y.mean()
    idcg = tdf.groupby(["user_id"]).item_id.size().apply(lambda x: sum((1/np.log2(i+1) for i in range(1,x+2)))).mean()

    return dcg/idcg

class Dataset:
    def __init__(self, name: str = "dummy"):
        self.name = name
    
    def load_interactions(
        self,
        filename: str = None, 
        item_id_name: str = "item_id", 
        user_id_name: str = "user_id",
        value_name: str = "value",
        timestamp_name: str = None,
        min_value_to_keep: float = None,
        user_min_support: int = 1,
        item_min_support: int = 1,
        set_all_values_to: float = None,
        raw_data = None,
        num_test_users = 10000,
        random_state = 42,
        duplicates_map: dict = None,
        max_steps=1,
        load_previous_splits=False,
        items_raw_data = None,
        items_item_id_name = None,
        items_preprocess = None,
        coldstart_fraction = None,
        num_coldstart_items = None,
        image_embeddings = None,
        items_features = None,
        partial=0
    ):
        self.filename = filename
        mapping = {item_id_name: "item_id", user_id_name: "user_id", value_name: "value"}
        if timestamp_name is not None:
            mapping[timestamp_name]="timestamp"
        
        if raw_data is None:
            raw_data = pd.read_csv(filename)
        
        if isinstance(raw_data, str):
            self.filename = raw_data.split('"')[1]
            raw_data=eval(raw_data)
        self.max_steps=max_steps    
        self.random_state=random_state    
        cols = [mapping[x] if x in mapping else x  for x in raw_data.columns]
        raw_data.columns = cols
        if min_value_to_keep is not None:
            raw_data = raw_data[raw_data["value"]>=min_value_to_keep]
        
        if duplicates_map is not None:
            raw_data["item_id"]=raw_data.item_id.apply(lambda x: duplicates_map.get(x, x))
        
        if set_all_values_to is not None:
            raw_data["value"] = set_all_values_to
        
        if not isinstance(raw_data.item_id.dtype, pd.core.dtypes.dtypes.CategoricalDtype):
            raw_data["item_id"] = raw_data["item_id"].astype(str)
        
        if not isinstance(raw_data.user_id.dtype, pd.core.dtypes.dtypes.CategoricalDtype):
            raw_data["user_id"] = raw_data["user_id"].astype(str)
        
        raw_data["item_id"] = raw_data.item_id.astype('category')
        raw_data["user_id"] = raw_data.user_id.astype('category')
        
        self.all_interactions = raw_data
        
        if items_raw_data is not None:
            self.items = eval(items_raw_data)
            self.items["item_id"]=self.items[items_item_id_name].astype(str)
            self.items["_text_attributes"]=self.items.apply(lambda row: eval(items_preprocess), axis=1)
            #self.items=self.items[~self.items._text_attributes.str.endswith("Error")]
            self.all_interactions=self.all_interactions[self.all_interactions.item_id.isin(self.items.item_id)]
            self.all_interactions["item_id"] = self.all_interactions.item_id.cat.remove_unused_categories()
            self.all_interactions["user_id"] = self.all_interactions.user_id.cat.remove_unused_categories()
            # text processing
            it=self.items[["item_id","_text_attributes"]]
            am_itemids = it.item_id.to_numpy()
            am_locator = np.array([np.argwhere(am_itemids==q).item() for q in tqdm(self.all_interactions.item_id.cat.categories)])
            self.items_texts=it.iloc[am_locator].reset_index(drop=True)
            self.texts=self.items_texts._text_attributes.to_numpy()

            if image_embeddings is not None:
                image_embeddings = eval(image_embeddings)
                am_itemids = image_embeddings.item_id.to_numpy()
                am_locator = np.array([np.argwhere(am_itemids==q).item() for q in tqdm(self.all_interactions.item_id.cat.categories)])
                self.items_images=image_embeddings.iloc[am_locator].reset_index(drop=True)
                self.image_embeddings=np.vstack(self.items_images.image_embeddings)

            if items_features is not None:
                items_features = eval(items_features)
                am_itemids = items_features.item_id.to_numpy()
                am_locator = np.array([np.argwhere(am_itemids==q).item() for q in tqdm(self.all_interactions.item_id.cat.categories)])
                self.items_features=items_features.iloc[am_locator].reset_index(drop=True)
                self.features_embeddings=np.vstack(self.items_features.items_features).astype("float32")
        
        self.all_interactions = fast_pruning(self.all_interactions, user_min_support, item_min_support, max_steps=self.max_steps)
        
        self.all_interactions["item_id"] = self.all_interactions.item_id.cat.remove_unused_categories()
        self.all_interactions["user_id"] = self.all_interactions.user_id.cat.remove_unused_categories()

        if items_raw_data is not None:
            self.items=self.items[self.items.item_id.isin(self.all_interactions["item_id"])]
            it=self.items[["item_id","_text_attributes"]]
            am_itemids = it.item_id.to_numpy()
            am_locator = np.array([np.argwhere(am_itemids==q).item() for q in tqdm(self.all_interactions.item_id.cat.categories)])
            self.items_texts=it.iloc[am_locator].reset_index(drop=True)
            self.texts=self.items_texts._text_attributes.to_numpy()

        self.item_ids = self.all_interactions.item_id.cat.categories
        self.make_test_split(num_test_users, random_state, load_previous_splits)

        self.coldstart_fraction = coldstart_fraction
        self.num_coldstart_items = num_coldstart_items
        self.partial = partial

    def make_test_split(self, n_test_users=10000, random_state=42, load_previous_splits=False):
        #self.test_users = pd.read_json("/".join(self.filename.split("/")[:-1])+"test_users.json")["userid"]
        if isinstance(load_previous_splits, str):
            self.test_users = eval(load_previous_splits).iloc[:,0].astype(str)
            print(f"Loaded previous test users form {load_previous_splits}")
        elif load_previous_splits:
            try:
                self.test_users = pd.read_json("/".join(self.filename.split("/")[:-1])+"/test_users.json").iloc[:,0].astype(str)
                print(f'test users loaded from {"/".join(self.filename.split("/")[:-1])+"/test_users.json"}')
            except: 
                print(f'{"/".join(self.filename.split("/")[:-1])+"/test_users.json"} not found')
                self.test_users = pd.Series(self.all_interactions.user_id.cat.categories.to_list()).sample(n_test_users, random_state=random_state)
        else:
            print(f"Creating test splits for {n_test_users} with seed {random_state}.")
            self.test_users = pd.Series(self.all_interactions.user_id.cat.categories.to_list()).sample(n_test_users, random_state=random_state)
        self.test_interactions = self.all_interactions[self.all_interactions.user_id.isin(self.test_users)]
        self.test_interactions["user_id"]=self.test_interactions.user_id.cat.remove_unused_categories()
        self.test_interactions["item_id"]=self.test_interactions.item_id.cat.remove_unused_categories()
        
        self.full_train_interactions = self.all_interactions[~self.all_interactions.user_id.isin(self.test_users)]
        self.full_train_interactions["user_id"]=self.full_train_interactions.user_id.cat.remove_unused_categories()
        self.full_train_interactions["item_id"]=self.full_train_interactions.item_id.cat.remove_unused_categories()
        if load_previous_splits:
            try:
                self.val_users = pd.read_json("/".join(self.filename.split("/")[:-1])+"/val_users.json").iloc[:,0].astype(str)
                print(f'val users loaded from {"/".join(self.filename.split("/")[:-1])+"/val_users.json"}')
            except:
                print(f'{"/".join(self.filename.split("/")[:-1])+"/val_users.json"} not found')
                self.val_users = pd.Series(self.full_train_interactions.user_id.cat.categories.to_list()).sample(n_test_users, random_state=random_state)
        else:
            print(f"Creating validation splits for {n_test_users} with seed {random_state}.")
            self.val_users = pd.Series(self.full_train_interactions.user_id.cat.categories.to_list()).sample(n_test_users, random_state=random_state)
            
       
        self.val_interactions = self.all_interactions[self.all_interactions.user_id.isin(self.val_users)]
        self.val_interactions["user_id"]=self.val_interactions.user_id.cat.remove_unused_categories()
        self.val_interactions["item_id"]=self.val_interactions.item_id.cat.remove_unused_categories()
        
        self.train_interactions = self.full_train_interactions[~self.full_train_interactions.user_id.isin(self.val_users)]
        self.train_interactions["user_id"]=self.train_interactions.user_id.cat.remove_unused_categories()
        self.train_interactions["item_id"]=self.train_interactions.item_id.cat.remove_unused_categories()
    
    def update_test_texts(self):
        it=self.items[["item_id","_text_attributes"]]
        am_itemids = it.item_id.to_numpy()
        am_locator = np.array([np.argwhere(am_itemids==q).item() for q in tqdm(self.full_train_interactions.item_id.cat.categories)])
        self.full_train_items_texts=it.iloc[am_locator].reset_index(drop=True)
        self.full_train_texts=self.full_train_items_texts._text_attributes.to_numpy()

        am_locator = np.array([np.argwhere(am_itemids==q).item() for q in tqdm(self.train_interactions.item_id.cat.categories)])
        self.train_items_texts=it.iloc[am_locator].reset_index(drop=True)
        self.train_texts=self.train_items_texts._text_attributes.to_numpy()

        if hasattr(self, "items_images"):
                image_embeddings = self.items_images
                am_itemids = image_embeddings.item_id.to_numpy()
                am_locator = np.array([np.argwhere(am_itemids==q).item() for q in tqdm(self.full_train_interactions.item_id.cat.categories)])
                self.full_train_items_images=image_embeddings.iloc[am_locator].reset_index(drop=True)
                self.full_train_image_embeddings=np.vstack(self.full_train_items_images.image_embeddings)
                
                am_locator = np.array([np.argwhere(am_itemids==q).item() for q in tqdm(self.train_interactions.item_id.cat.categories)])
                self.train_items_images=image_embeddings.iloc[am_locator].reset_index(drop=True)
                self.train_image_embeddings=np.vstack(self.train_items_images.image_embeddings)

        if hasattr(self, "items_features"):
                items_features = self.items_features
                am_itemids = items_features.item_id.to_numpy()
                am_locator = np.array([np.argwhere(am_itemids==q).item() for q in tqdm(self.full_train_interactions.item_id.cat.categories)])
                self.train_items_features=items_features.iloc[am_locator].reset_index(drop=True)
                self.train_features_embeddings=np.vstack(self.train_items_features.items_features).astype("float32")
                am_locator = np.array([np.argwhere(am_itemids==q).item() for q in tqdm(self.train_interactions.item_id.cat.categories)])
                self.full_train_items_features=items_features.iloc[am_locator].reset_index(drop=True)
                self.full_train_features_embeddings=np.vstack(self.full_train_items_features.items_features).astype("float32")
                

    def _test_interactions(self):
        if hasattr(self, 'test_interactions'):
            return self.test_interactions
    
    def _full_train_interactions(self):
        if hasattr(self, 'full_train_interactions'):
            return self.full_train_interactions

    def _train_interactions(self):
        if hasattr(self, 'train_interactions'):
            return self.train_interactions
        
    def _val_interactions(self):
        if hasattr(self, 'val_interactions'):
            return self.val_interactions
        
    def __repr__(self):
        s = f"""\nDataset for recsys experimenting
        
          name: {self.name}"""
        
        if hasattr(self, 'all_interactions'):
            s+=f"""
          total stats:
            # of interactions {len(self.all_interactions)}
            # of users {self.all_interactions.user_id.cat.categories.size}
            # of items {self.all_interactions.item_id.cat.categories.size}"""
        else:
            s+="""
          interactions not loaded yet"""
        if hasattr(self, 'test_interactions'):
            s+=f"""    
          test set:
            # of interactions {len(self.test_interactions)}
            # of users {self.test_interactions.user_id.cat.categories.size}
            # of items {self.test_interactions.item_id.cat.categories.size}"""
            s+=f"""    
          validation set:
            # of interactions {len(self.val_interactions)}
            # of users {self.val_interactions.user_id.cat.categories.size}
            # of items {self.val_interactions.item_id.cat.categories.size}"""
            s+=f"""    
          train set:
            # of interactions {len(self.train_interactions)}
            # of users {self.train_interactions.user_id.cat.categories.size}
            # of items {self.train_interactions.item_id.cat.categories.size}"""
            s+=f"""    
          full train set:
            # of interactions {len(self.full_train_interactions)}
            # of users {self.full_train_interactions.user_id.cat.categories.size}
            # of items {self.full_train_interactions.item_id.cat.categories.size}"""
        else:
            s+="""
          splits has not been done yet"""
        s+="\n\n"
        return s
    
    
class Evaluation:
    
    RECPACK_METRICS = {
        "recall": recpack.metrics.CalibratedRecallK,
        "ndcg": recpack.metrics.NDCGK,
        "coverage": recpack.metrics.CoverageK,
    }
    
    def __init__(self, dataset, what="test", how="5-folds", metrics=["recall@20", "recall@50", "ndcg@100"]):
        self.dataset = dataset
        self.what = what
        self.how = how
        self.metrics = {}
        for metric in metrics:
            metric_name, k = metric.split("@")
            self.metrics[metric] = self.RECPACK_METRICS[metric_name](int(k))
        
        print(self.metrics)
        
        if what=="test":
            self.test_src, self.test_target, self.X_test_src, self.X_test_target =  get_get_src_target_rand_df_fold(self.dataset.test_interactions)
            
        else:
            self.test_src, self.test_target, self.X_test_src, self.X_test_target =  get_get_src_target_rand_df_fold(self.dataset.val_interactions)
        
        self.candidates_df=pd.DataFrame({"item_id":self.dataset.test_interactions.item_id.unique().to_numpy(), "user_id":"0", "value": 1.})
        self.candidates_df["item_id"]=self.candidates_df["item_id"].astype("category")
        self.candidates_df["user_id"]=self.candidates_df["user_id"].astype("category")
        self.candidates_df["item_id"]=self.candidates_df["item_id"].cat.remove_unused_categories()
        
    def __call__(self, df):
        preds = get_sparse_matrix_from_dataframe(
            df, 
            item_indices=self.test_target.item_id.cat.categories,
            user_indices=self.test_target.user_id.cat.categories,
        )
        #print(preds)
        trues = get_sparse_matrix_from_dataframe(
            self.test_target,
            item_indices=self.test_target.item_id.cat.categories,
            user_indices=self.test_target.user_id.cat.categories,
        )
        #print(trues)
        results = {}
        for name, metric in self.metrics.items():
            metric.calculate(trues, preds)
            results[name]=metric.value
        return results
    
    def __repr__(self):
        s = f"""\nEvaluation for recsys experimenting
        
          on dataset: {self.dataset.name}"""
    
        s+="\n\n"
        return s

class _ColdStartEvaluation(Evaluation):
    def __init__(self, dataset, what="test", coldstart_fraction=0.1, metrics=["recall@20", "recall@50", "ndcg@100", "coverage@20"]):
        self.dataset = dataset
        self.metrics = {}
        
        for metric in metrics:
            metric_name, k = metric.split("@")
            self.metrics[metric] = self.RECPACK_METRICS[metric_name](int(k))

        if self.dataset.coldstart_fraction is not None:
            coldstart_fraction = self.dataset.coldstart_fraction
        
        self.coldstart_fraction=coldstart_fraction
        cold_start_items = np.random.choice(dataset.all_interactions.item_id.unique(), int(coldstart_fraction*len(dataset.all_interactions.item_id.cat.categories)))
        cold_start_interactions = dataset.all_interactions[dataset.all_interactions.item_id.isin(cold_start_items)]
        interactions = dataset.all_interactions[~dataset.all_interactions.item_id.isin(cold_start_items)]
        cold_start_interactions = cold_start_interactions[cold_start_interactions.user_id.isin(interactions.user_id)]
        test_users = np.random.choice(cold_start_interactions.user_id.unique(), len(dataset.test_users))
        val_users = np.random.choice(cold_start_interactions[~cold_start_interactions.user_id.isin(test_users)].user_id.unique(), len(dataset.val_users))

        test_interactions_source = interactions[interactions.user_id.isin(test_users)]
        test_interactions_target = cold_start_interactions[cold_start_interactions.user_id.isin(test_users)]
        val_interactions_source = interactions[interactions.user_id.isin(val_users)]
        val_interactions_target = cold_start_interactions[cold_start_interactions.user_id.isin(val_users)]
        
        if what=="test":
            self.test_src = test_interactions_source
            self.test_target = test_interactions_target
            self.X_test_src = get_sparse_matrix_from_dataframe(test_interactions_source, item_indices=dataset.all_interactions.item_id.cat.categories, user_indices=pd.Categorical(test_interactions_source.user_id.unique()).remove_unused_categories().categories)
            self.X_test_target = get_sparse_matrix_from_dataframe(test_interactions_target, item_indices=dataset.all_interactions.item_id.cat.categories, user_indices=pd.Categorical(test_interactions_target.user_id.unique()).remove_unused_categories().categories)
            
        else:
            self.val_src = test_interactions_source
            self.val_target = test_interactions_target
            self.X_val_src = get_sparse_matrix_from_dataframe(test_interactions_source, item_indices=dataset.all_interactions.item_id.cat.categories, user_indices=pd.Categorical(test_interactions_source.user_id.unique()).remove_unused_categories().categories)
            self.X_val_target = get_sparse_matrix_from_dataframe(test_interactions_target, item_indices=dataset.all_interactions.item_id.cat.categories, user_indices=pd.Categorical(test_interactions_target.user_id.unique()).remove_unused_categories().categories)

        test_interactions_target["item_id"]=test_interactions_target.item_id.cat.remove_unused_categories()
        test_interactions_target["user_id"]=test_interactions_target.user_id.cat.remove_unused_categories()
        val_interactions_target["item_id"]=val_interactions_target.item_id.cat.remove_unused_categories()
        val_interactions_target["user_id"]=val_interactions_target.user_id.cat.remove_unused_categories()
        interactions["item_id"]=interactions.item_id.cat.remove_unused_categories()
        interactions["user_id"]=interactions.user_id.cat.remove_unused_categories()


        candidates_df=pd.DataFrame({"item_id":self.test_target.item_id.unique().to_numpy(), "user_id":"0", "value": 1.})
        candidates_df["item_id"]=candidates_df["item_id"].astype("category")
        candidates_df["user_id"]=candidates_df["user_id"].astype("category")
        self.cold_start_candidates_df = candidates_df
        
        dataset.test_users = test_users
        dataset.val_users = val_users
        dataset.full_train_interactions = interactions
        dataset.train_interactions = interactions
        dataset.cold_start_items = cold_start_items
        dataset.test_interactions = test_interactions_target
        dataset.val_interactions = val_interactions_target

        dataset.update_test_texts()
        

    
    def __repr__(self):
        s = f"""\nCold start evaluation for recsys experimenting
        
          on dataset: {self.dataset.name}"""
    
        s+="\n\n"
        return s
    
class PartialEvaluation(Evaluation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        dataset = self.dataset
        excluded_items = pd.Series(dataset.all_interactions.item_id.cat.categories).sample(frac=0.5, random_state=42)
        if dataset.partial>0:
            included_items = excluded_items.sample(frac=dataset.partial, random_state=dataset.random_state)
            excluded_items = excluded_items[~excluded_items.isin(included_items)]
        dataset.full_train_interactions=dataset.full_train_interactions[~dataset.full_train_interactions.item_id.isin(excluded_items)]
        dataset.train_interactions=dataset.train_interactions[~dataset.train_interactions.item_id.isin(excluded_items)]

        dataset.full_train_interactions["item_id"]= dataset.full_train_interactions["item_id"].cat.remove_unused_categories()
        dataset.full_train_interactions["user_id"]= dataset.full_train_interactions["user_id"].cat.remove_unused_categories()
        dataset.train_interactions["item_id"]= dataset.train_interactions["item_id"].cat.remove_unused_categories()
        dataset.train_interactions["user_id"]= dataset.train_interactions["user_id"].cat.remove_unused_categories()

class LongTailEvaluation(Evaluation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        evaluator = self
        dataset = self.dataset
        long_tail_items = dataset.full_train_interactions.groupby("item_id").value.sum().sort_values().iloc[:int(len(dataset.full_train_interactions.item_id.unique())*0.8)]
        trgt = evaluator.test_target[evaluator.test_target.item_id.isin(long_tail_items.index)]
        src = evaluator.test_src[evaluator.test_src.user_id.isin(trgt.user_id)]
        trgt["item_id"]=trgt["item_id"].cat.remove_unused_categories()
        trgt["user_id"]=trgt["user_id"].cat.remove_unused_categories()
        src["item_id"]=src["item_id"].cat.remove_unused_categories()
        src["user_id"]=src["user_id"].cat.remove_unused_categories()
        self.test_target = trgt
        self.test_src = src
        self.candidates_df=pd.DataFrame({"item_id":self.test_target.item_id.unique().to_numpy(), "user_id":"0", "value": 1.})
        self.candidates_df["item_id"]=self.candidates_df["item_id"].astype("category")
        self.candidates_df["user_id"]=self.candidates_df["user_id"].astype("category")

class ColdStartEvaluation(Evaluation):
    def __init__(self, dataset, what="test", coldstart_fraction=0.1, metrics=["recall@20", "recall@50", "ndcg@100", "coverage@20"]):
        self.dataset = dataset
        self.metrics = {}
        
        for metric in metrics:
            metric_name, k = metric.split("@")
            self.metrics[metric] = self.RECPACK_METRICS[metric_name](int(k))

        if self.dataset.coldstart_fraction is not None:
            coldstart_fraction = self.dataset.coldstart_fraction
        
        self.coldstart_fraction=coldstart_fraction
        uniqe_items = dataset.all_interactions.item_id.unique()
        num_split_items = int(coldstart_fraction*len(uniqe_items))
                              
        
        uniqe_items = dataset.all_interactions.item_id.unique()
        num_split_items = int(coldstart_fraction*len(uniqe_items)/2)

        if dataset.num_coldstart_items is not None:
            num_split_items = dataset.num_coldstart_items
        
        
        np.random.seed(dataset.random_state)
        val_cold_start_items = np.random.choice(uniqe_items, num_split_items)
        np.random.seed(dataset.random_state)
        left_items = list(set(uniqe_items).difference(set(val_cold_start_items)))
        left_items.sort()
        test_cold_start_items = np.random.choice(np.array(left_items),num_split_items)
        
        #known_items = np.array(list(set(uniqe_items).difference(set(val_cold_start_items)).difference(set(test_cold_start_items))))
        
        
        
        
        
        val_cold_start_interactions = dataset.all_interactions[dataset.all_interactions.item_id.isin(val_cold_start_items)].copy()
        test_cold_start_interactions = dataset.all_interactions[dataset.all_interactions.item_id.isin(test_cold_start_items)].copy()
        val_cold_start_interactions["item_id"]=val_cold_start_interactions.item_id.cat.remove_unused_categories()
        val_cold_start_interactions["user_id"]=val_cold_start_interactions.user_id.cat.remove_unused_categories()
        test_cold_start_interactions["item_id"]=test_cold_start_interactions.item_id.cat.remove_unused_categories()
        test_cold_start_interactions["user_id"]=test_cold_start_interactions.user_id.cat.remove_unused_categories()
        cold_start_items = np.hstack([val_cold_start_items, test_cold_start_items])
        interactions = dataset.all_interactions[~dataset.all_interactions.item_id.isin(cold_start_items)].copy()
        #cold_start_interactions = dataset.all_interactions[dataset.all_interactions.item_id.isin(cold_start_items)]
        #val_cold_start_interactions = dataset.all_interactions[dataset.all_interactions.item_id.isin(val_cold_start_items)]
        #test_cold_start_interactions = dataset.all_interactions[dataset.all_interactions.item_id.isin(test_cold_start_items)]
        
        test_users = test_cold_start_interactions.user_id.unique()
        if len(test_users)>200000:
            test_users = np.random.choice(test_users, 100000)

        val_users = val_cold_start_interactions.user_id.unique() #[~cold_start_interactions.user_id.isin(test_users)].user_id.unique()
        test_interactions_source = interactions[interactions.user_id.isin(test_users)]
        test_interactions_target = test_cold_start_interactions[test_cold_start_interactions.user_id.isin(test_users)]
        val_interactions_source = interactions[interactions.user_id.isin(val_users)]
        val_interactions_target = val_cold_start_interactions[val_cold_start_interactions.user_id.isin(val_users)]
        
        if what=="test":
            self.test_src = test_interactions_source
            self.test_target = test_interactions_target
            self.X_test_src = get_sparse_matrix_from_dataframe(test_interactions_source, item_indices=dataset.all_interactions.item_id.cat.categories, user_indices=pd.Categorical(test_interactions_source.user_id.unique()).remove_unused_categories().categories)
            self.X_test_target = get_sparse_matrix_from_dataframe(test_interactions_target, item_indices=dataset.all_interactions.item_id.cat.categories, user_indices=pd.Categorical(test_interactions_target.user_id.unique()).remove_unused_categories().categories)
            candidates_df=pd.DataFrame({"item_id":self.test_target.item_id.unique().to_numpy(), "user_id":"0", "value": 1.})
        
        else:
            self.val_src = val_interactions_source
            self.val_target = val_interactions_target
            self.X_val_src = get_sparse_matrix_from_dataframe(val_interactions_source, item_indices=dataset.all_interactions.item_id.cat.categories, user_indices=pd.Categorical(val_interactions_source.user_id.unique()).remove_unused_categories().categories)
            self.X_val_target = get_sparse_matrix_from_dataframe(val_interactions_target, item_indices=dataset.all_interactions.item_id.cat.categories, user_indices=pd.Categorical(val_interactions_target.user_id.unique()).remove_unused_categories().categories)
            candidates_df=pd.DataFrame({"item_id":self.val_target.item_id.unique().to_numpy(), "user_id":"0", "value": 1.})
        

        #test_interactions_target["item_id"]=test_interactions_target.item_id.cat.remove_unused_categories()
        #test_interactions_target["user_id"]=test_interactions_target.user_id.cat.remove_unused_categories()
        #val_interactions_target["item_id"]=val_interactions_target.item_id.cat.remove_unused_categories()
        #val_interactions_target["user_id"]=val_interactions_target.user_id.cat.remove_unused_categories()
        interactions["item_id"]=interactions.item_id.cat.remove_unused_categories()
        interactions["user_id"]=interactions.user_id.cat.remove_unused_categories()
        #("concat")
        full_train_interactions = pd.concat([interactions, val_interactions_source, val_interactions_target])
        full_train_interactions = full_train_interactions[~full_train_interactions.duplicated()]
        full_train_interactions["item_id"]=full_train_interactions.item_id.astype("category")
        full_train_interactions["item_id"]=full_train_interactions.item_id.cat.remove_unused_categories()
        full_train_interactions["user_id"]=full_train_interactions.user_id.astype("category")
        full_train_interactions["user_id"]=full_train_interactions.user_id.cat.remove_unused_categories()
    
        candidates_df["item_id"]=candidates_df["item_id"].astype("category")
        candidates_df["user_id"]=candidates_df["user_id"].astype("category")
        self.cold_start_candidates_df = candidates_df
        
        dataset.test_users = test_users
        dataset.val_users = val_users
        dataset.full_train_interactions = full_train_interactions
        dataset.train_interactions = interactions
        dataset.cold_start_items = cold_start_items
        dataset.test_interactions = test_interactions_target
        dataset.val_interactions = val_interactions_target
        dataset.val_cold_start_items= val_cold_start_items
        dataset.test_cold_start_items=test_cold_start_items
        dataset.update_test_texts()
        

    
    def __repr__(self):
        s = f"""\nCold start evaluation for recsys experimenting
        
          on dataset: {self.dataset.name}"""
    
        s+="\n\n"
        return s