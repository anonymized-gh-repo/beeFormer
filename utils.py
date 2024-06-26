
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
import keras

warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

def NMSE(x,y):
    x=torch.nn.functional.normalize(x, dim=-1)
    y=torch.nn.functional.normalize(y, dim=-1)
    return keras.losses.mean_squared_error(x,y)

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
    for i in range(X_val_src.shape[0]):
        ind = get_random_indices(X_val_src[i], 1)
        X_val_src[i,ind]=0
    X.append(X_val_src)
    XV.append(X_val)
    if fold!=1:
        X_val_src = X_val.copy()
        for i in range(X_val_src.shape[0]):
            ind = get_random_indices(X_val_src[i], 2)
            X_val_src[i,ind]=0
        X.append(X_val_src)
        XV.append(X_val)

        X_val_src = X_val.copy()
        for i in range(X_val_src.shape[0]):
            ind = get_random_indices(X_val_src[i], 3)
            X_val_src[i,ind]=0
        X.append(X_val_src)
        XV.append(X_val)

        X_val_src = X_val.copy()
        for i in range(X_val_src.shape[0]):
            ind = get_random_indices(X_val_src[i], 4)
            X_val_src[i,ind]=0
        X.append(X_val_src)
        XV.append(X_val)

        X_val_src = X_val.copy()
        for i in range(X_val_src.shape[0]):
            ind = get_random_indices(X_val_src[i], 5)
            X_val_src[i,ind]=0
        X.append(X_val_src)
        XV.append(X_val)
        
    X_val_src = scipy.sparse.vstack(X)
    X_val = scipy.sparse.vstack(XV)
    
    X_val_src.eliminate_zeros()
    X_val_targets=X_val-X_val_src



    bl = torch.from_numpy(1-X_val_src.toarray())
    target = torch.from_numpy(X_val_targets.toarray().astype(bool))
    
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
    if verbose>0:
        f = IntProgress(min=0, max=X.shape[0]//verbose+2) # instantiate the bar
        display(f) # display the bar

    split = np.split(X.indices, X.indptr)[1:-1]
    split2 = np.split(X.data, X.indptr)[1:-1]

    if verbose>0:    
        f.value += 1
    
    dfs = []

    for i in range(len(split)):
        if verbose>0:
            if i%verbose == 0:
                f.value += 1
        dfs.append(pd.DataFrame({"user_id":user_ids[i], "item_id":item_ids[split[i]], "value":split2[i]}))
    
    ret = pd.concat(dfs)
    ret["user_id"]=ret["user_id"].astype(str).astype('category').cat.remove_unused_categories()
    ret["item_id"]=ret["item_id"].astype(str).astype('category').cat.remove_unused_categories()
    if verbose>0:    
        f.value += 1
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
        
        if stable:
            logger.info("Data stable after {} reduction steps ({} users, {} items)".format(step, number_of_users, number_of_items))
        step += 1
        
        if max_steps>0 and step>max_steps:
            stable=True
            
        print(step, max_steps, stable)
        
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
