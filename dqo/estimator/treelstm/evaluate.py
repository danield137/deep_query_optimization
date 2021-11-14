from typing import Union, List, NamedTuple

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn import metrics as skm

from dqo.datasets import QueriesDataset
from dqo.db.models import Database
from dqo.estimator import metrics as gm


def display_summary(true, pred, confusion_only=False):
    true = np.round(np.log2(np.array(true))).astype(int)
    pred = np.round(np.log2(np.array(pred))).astype(int)

    labels = [str(l) for l in sorted(list(set(list(pred)) | set(list(true))))]

    conf = skm.confusion_matrix(true, pred)
    conf_df = pd.DataFrame(conf)
    conf_df.columns.name = 'pred'
    conf_df.index.name = 'true'
    sns.heatmap(conf_df, fmt="g", annot=True, xticklabels=labels, yticklabels=labels)

    m = gm.mcc_metrics(true, pred)
    plt.title('\n'.join([f'{name} : {value}' for name, value in m.items()]))
    plt.show()

    if not confusion_only:
        print(skm.classification_report(true, pred, zero_division=0))


QueryEstimation = NamedTuple('QueryEstimation', query=str, true=float, pred=float)


def evaluate_results(results: List[QueryEstimation]):
    results_df = pd.DataFrame(results, columns=['query', 'true', 'pred'])
    results_df['bucket'] = results_df['true'].apply(np.log2).apply(np.round).apply(int)
    results_df['abs_err'] = np.abs(results_df['pred'] - results_df['true'])
    results_df['err_ratio'] = results_df['true'] / results_df['pred']

    return results_df


def evaluate_metrics(results_df: pd.DataFrame):
    true = results_df.true.apply(np.log2).apply(np.round).apply(int)
    pred = results_df.pred.apply(np.log2).apply(np.round).apply(int)

    return gm.mcc_metrics(true, pred)


def evaluate_binary_split(results_df: pd.DataFrame, boundary: int):
    p = len(results_df.query(f'pred <= {boundary}').query(f'true <= {boundary}'))
    t = len(results_df.query(f'true <= {boundary}'))
    total = len(results_df)
    acc = (p / t)
    zero_acc = (t / total)
    gain = acc / zero_acc
    return acc, zero_acc, gain


def maximize_binary_split(results_df: pd.DataFrame):
    ddf = pd.DataFrame(columns=['model', 'zerorule'])
    best_gain = 0
    best_idx = 1
    best_acc = 0
    best_zero_acc = 1
    for i in range(1, 30):
        acc, zero_acc, gain = evaluate_binary_split(results_df, boundary=i)
        if gain > best_gain:
            best_idx = i
            best_gain = gain
            best_acc = acc
            best_zero_acc = zero_acc
        ddf = ddf.append({'model': acc, 'zero rule': zero_acc, 'gain': gain}, ignore_index=True)
    return ddf, best_idx, best_acc, best_zero_acc


def display_best_binary(results_df: pd.DataFrame):
    ddf, best, best_p, zero = maximize_binary_split(results_df)
    ddf.plot()
    plt.title(f'boundary: {best}, prob: {best_p}, zero rule: {zero}')


def display_results(results_df: pd.DataFrame, confusion_only=False):
    display_summary(results_df['true'], results_df['pred'], confusion_only)
    print('2')
    if not confusion_only:
        error_df = results_df.groupby('bucket').agg({
            'abs_err': ['mean', 'median'],
            'err_ratio': ['mean', 'median']
        })
        print(error_df)
        display_best_binary(results_df)


def load_pretrained_model(checkpoint_path, model_cls):
    torch.manual_seed(0)
    model = model_cls()
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['state_dict'])
    model.eval()

    return model


class QueryEstimater:
    def __init__(
            self,
            model_checkpoint: str,
            model_cls=None,
            encoder=None,
            schema: Database = None,
            dataset: Union[QueriesDataset, str] = None
    ):
        """
        Provide either a `Database` schema or a `Dataset`
        :param model_checkpoint:
        :param schema:
        :param dataset:
        """
        self.model = load_pretrained_model(model_checkpoint, model_cls)
        self.model_cls = model_cls
        self.encoder = encoder
        if dataset is not None:
            self.ds = dataset if isinstance(dataset, QueriesDataset) else QueriesDataset(dataset)
        self._schema = schema

    @property
    def schema(self):
        if not self._schema:
            self._schema = self.ds.schema()

        return self._schema

    def predict_query(self, q):
        x = self.encoder.encode_query(self.schema, q)
        return self.predict_encoded(x)

    def predict_encoded(self, x):
        with torch.no_grad():
            return 2 ** self.model(x).item()

    def evaluate(
            self,
            dataset: Union[QueriesDataset, str] = None, sample_frac: float = None,
            sample_size: int = None, bucketed: bool = True, df: pd.DataFrame = None
    ) -> pd.DataFrame:
        if df is None:
            if dataset is None:
                dataset = self.ds
            if type(dataset) is str:
                dataset = QueriesDataset(dataset)

            train_df, test_df = dataset.load_splits()
            test_df['bucket'] = test_df.runtime.apply(np.log2).apply(np.round).apply(int).apply(lambda x: min(x, 8)).apply(lambda x: max(x, -3))
            df = test_df

            if bucketed:
                m = df.groupby('bucket').count().min()[0]
                df = df.groupby('bucket').head(m * 4)
            elif sample_size is not None and sample_size < len(test_df):
                df = df.groupby('bucket').apply(lambda x: x.sample(frac=sample_size / len(test_df)))

        results = []
        with torch.no_grad():
            from tqdm import tqdm
            for index, row in tqdm(df.iterrows(), total=len(df), position=0, leave=True):
                q, true = row['query'], row['runtime']
                pred = self.predict_query(q)
                results.append((q, true, pred))

        return evaluate_results(results)

    def predict_batch(self, batch, encoded=False, total=None, lowest=None):
        preds = []
        with torch.no_grad():
            from tqdm import tqdm
            for x in tqdm(batch, total=total or len(batch)):
                pred = self.predict_encoded(x) if encoded else self.predict_query(x)
                preds.append(max(pred, lowest) if lowest is not None else pred)

        return preds


def save_log_results(pred_log: List[float], true_log: List[float], path=None, prefix=None):
    import os
    path = path or os.getcwd()

    results_df = pd.DataFrame(np.column_stack((pred_log, true_log)), columns=['true', 'pred'])

    results_df['bucket'] = results_df['true'].apply(np.round).apply(int)
    results_df['true'] = results_df['true'].apply(lambda x: 2 ** x)
    results_df['pred'] = results_df['pred'].apply(lambda x: 2 ** x)
    results_df['abs_err'] = np.abs(results_df['pred'] - results_df['true'])
    results_df['err_ratio'] = results_df['true'] / results_df['pred']

    error_df = results_df.groupby('bucket').agg({
        'abs_err': ['mean', 'median'],
        'err_ratio': ['mean', 'median']
    })

    fig = plt.figure(figsize=(15, 20))

    gs = plt.GridSpec(20, 1, figure=fig)
    ax1 = fig.add_subplot(gs[0:10, :])

    pred_class = np.round(pred_log)
    true_class = np.round(true_log)
    labels = [str(l) for l in sorted(list(set(pred_class) | set(true_class)))]
    conf = skm.confusion_matrix(true_class, pred_class)
    conf_df = pd.DataFrame(conf)
    conf_df.columns.name = 'pred'
    conf_df.index.name = 'true'
    sns.heatmap(conf_df, ax=ax1, fmt="g", annot=True, xticklabels=labels, yticklabels=labels)
    #
    m = gm.mcc_metrics(true_class, pred_class)
    ax1.set_title('\n'.join([f'{name} : {value}' for name, value in m.items()]))
    # fig.print()

    ax2 = fig.add_subplot(gs[10:13, 0])
    ax2.axis('off')
    ax2.text(0.2, -1.2, skm.classification_report(true_class, pred_class, zero_division=0), ha="center", fontsize=12)

    ax3 = fig.add_subplot(gs[13:16, 0])
    ax3.axis('off')
    ax3.text(0.8, 0.5, str(error_df), ha="center", fontsize=12)

    ax4 = fig.add_subplot(gs[16:19, :])
    ddf, best, best_p, zero = maximize_binary_split(results_df)
    ddf.plot(ax=ax4)
    ax4.set_title(f'boundary: {best}, prob: {best_p}, zero rule: {zero}')

    file_name = f'_acc_{m["accuracy"]:.5f}_mae_{m["mae"]:.5f}_f1_{m["f1 macro"]:.5f}.png'
    if prefix:
        file_name = prefix + file_name

    file_path = os.path.join(path, file_name)
    fig.subplots_adjust(hspace=2)
    fig.savefig(file_path, bbox_inches='tight')


if __name__ == '__main__':
    preds = np.random.randint(-3, 9, 1000)
    trues = np.random.randint(-3, 9, 1000)

    save_log_results(preds, trues)
