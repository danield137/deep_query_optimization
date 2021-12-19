import os
import re
from typing import Union, List, NamedTuple, Tuple, Dict, Any, cast, Optional

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
    regression_metrics = gm.regression_metrics(results_df.true, results_df.pred)

    true = results_df.true.apply(np.log2)
    pred = results_df.pred.apply(np.log2)
    custom_metrics = gm.custom_metrics(true, pred)

    true = true.apply(np.round).apply(int)
    pred = pred.apply(np.round).apply(int)
    mcc_metrics = gm.mcc_metrics(true, pred)

    return {**mcc_metrics, **regression_metrics, **custom_metrics}


def evaluate_binary_split(results_df: pd.DataFrame, boundary: int):
    p = len(results_df.query(f'pred <= {boundary}').query(f'true <= {boundary}'))
    t = len(results_df.query(f'true <= {boundary}'))
    total = len(results_df)
    acc = (p / t) if t > 0 else 0
    zero_acc = (t / total)
    gain = acc / zero_acc if zero_acc > 0 else 0
    return acc, zero_acc, gain


def maximize_binary_split(results_df: pd.DataFrame):
    ddf = pd.DataFrame(columns=['model', 'zerorule'])
    best_gain = 0
    best_idx = 1
    best_acc = 0
    best_zero_acc = 1
    for i in range(0, 9):
        boundary = 2 ** i
        acc, zero_acc, gain = evaluate_binary_split(results_df, boundary=boundary)
        if gain > best_gain:
            best_idx = boundary
            best_gain = gain
            best_acc = acc
            best_zero_acc = zero_acc
        ddf = ddf.append({'x':boundary, 'model': acc, 'zerorule': zero_acc, 'gain': gain}, ignore_index=True)
    return ddf, best_idx, best_acc, best_zero_acc


def display_best_binary(results_df: pd.DataFrame):
    ddf, best, best_p, zero = maximize_binary_split(results_df)
    ddf.plot(x='x')
    plt.title(f'boundary: {best}, prob: {best_p}, zerorule: {zero}')


def display_results(results_df: pd.DataFrame, confusion_only=False):
    display_summary(results_df['true'], results_df['pred'], confusion_only)
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


epoch_num_regex = re.compile("epoch=(\d.*)-")


def list_runs(version) -> List[str]:
    base_path = os.path.dirname(str(version.__file__))
    runs_path = os.path.join(base_path, f'logs')
    return sorted([p for p in os.listdir(runs_path) if not p.startswith('.')])


def list_checkpoints(version, experiment_name) -> List[Tuple[int, str]]:
    base_path = os.path.dirname(str(version.__file__))

    experiment_versions_path = os.path.join(base_path, f'logs/{experiment_name}')
    try:
        experiment_versions = sorted([p for p in os.listdir(experiment_versions_path) if not p.startswith('.')])
        latest_experiment = experiment_versions[-1]

        checkpoints_path = os.path.join(experiment_versions_path, latest_experiment, 'checkpoints')
        checkpoints = [p for p in os.listdir(checkpoints_path) if not p.startswith('.')]
        checkpoints = [(int(epoch_num_regex.search(cp)[1]), os.path.join(checkpoints_path, cp)) for cp in checkpoints]
    except:
        return []

    return sorted(checkpoints)


class MissingCheckpointError(ValueError):
    ...


class QueryEstimater:
    def __init__(
            self,
            model_checkpoint,
            model_cls=None,
            encoder=None,
            schema: Database = None,
            dataset: Union[QueriesDataset, str] = None,
            version=None
    ):
        """
        Provide either a `Database` schema or a `Dataset`
        :param model_checkpoint:
        :param schema:
        :param dataset:
        """
        if dataset is not None:
            self.ds = dataset if isinstance(dataset, QueriesDataset) else QueriesDataset(dataset)

        if version is not None:
            self.encoder = getattr(version, 'encoder')
            self.model_cls = getattr(version, 'model_cls')
        else:
            self.model_cls = model_cls
            self.encoder = encoder

        # not a file, but an experiment name
        if '.' not in model_checkpoint:
            checkpoints = list_checkpoints(version, experiment_name=model_checkpoint)
            if not checkpoints:
                raise MissingCheckpointError()
            epoch, model_checkpoint = checkpoints[-1]

        self.model_checkpoint = model_checkpoint
        self.model = load_pretrained_model(model_checkpoint, self.model_cls)

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
            dataset: Union[QueriesDataset, str] = None,
            sample_size: int = None, bucketed: bool = True, df: pd.DataFrame = None
    ) -> pd.DataFrame:
        if df is None:
            if dataset is None:
                dataset = self.ds
            if type(dataset) is str:
                dataset = QueriesDataset(dataset)

            train_df, test_df = dataset.load_splits(include_bucket=True)

            df = test_df

            if bucketed:
                m = df.groupby('bucket').count().min()[0]
                # df = df.groupby('bucket').head(4)
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
    m = gm.regression_metrics(pred_log, true_log)

    pred_class = np.round(pred_log)
    true_class = np.round(true_log)
    labels = [str(l) for l in sorted(list(set(pred_class) | set(true_class)))]
    conf = skm.confusion_matrix(true_class, pred_class)
    conf_df = pd.DataFrame(conf)
    conf_df.columns.name = 'pred'
    conf_df.index.name = 'true'
    sns.heatmap(conf_df, ax=ax1, fmt="g", annot=True, xticklabels=labels, yticklabels=labels)
    #
    m.update(**gm.mcc_metrics(true_class, pred_class))
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


def zero_rule_test_accuracy(test_ds, bucketed=True):
    df = QueriesDataset(f'{test_ds}:optimized:test').load(include_bucket=True)
    if bucketed:
        df = df.groupby('bucket').head(df.groupby('bucket').count().min()[0] * 4)
    elif 500 < len(df):
        df = df.groupby('bucket', group_keys=False).apply(lambda x: x.sample(frac=500 / len(df)))

    return df.groupby('bucket').count().max()[0] / len(df)


def empty_version(train_ds, test_ds, version) -> Tuple[List[Any], Optional[pd.DataFrame]]:
    version_short, arch = str(version.__name__).split('.')[-2:]
    result = None
    info = [train_ds, test_ds, version_short, arch, '']
    metrics = [0 for mc in metric_columns]
    desc = ''

    return [*info, *metrics, desc], result


def evaluate_version(version, train_ds: str, test_ds: str, bucketed=True, cp: Tuple[int, str] = None) -> Tuple[List[Any], pd.DataFrame]:
    version_short, arch = str(version.__name__).split('.')[-2:]

    try:
        estimater = QueryEstimater(
            train_ds if not cp else cp[1],
            dataset=f'{test_ds}:optimized',
            version=version,
        )

        result = estimater.evaluate(bucketed=bucketed, sample_size=500 if not bucketed else None)
        result_metrics = evaluate_metrics(result)

        info = [train_ds, test_ds, version_short, arch, cp[0]]
        metrics = [result_metrics[mc] for mc in metric_columns]
        desc = getattr(version, 'description', version_short)

        print(f'df: {test_ds}, v: {str(version.__name__)}, cp: {estimater.model_checkpoint} ::'
              f' accuracy: {result_metrics["accuracy"]}, mae: {result_metrics["mae"]}, f1: {result_metrics["f1 macro"]}')
    except Exception as e:
        print(e)
        return empty_version(train_ds, test_ds, version)

    return [*info, *metrics, desc], result


info_columns = ['trained_on', 'tested_on', 'version', 'arch', 'epoch']
metric_columns = [
    'accuracy', 'balanced accuracy', 'recall', 'f1 weighted', 'mae', 'mean_rounded_two_sided_error', 'bucket_accuracy', 'bucket_accuracy', 'values'
]
summary_columns = [*info_columns, *metric_columns, 'description']


def compare_versions(versions, trained_on: Union[str, List[str]], test_on: Union[str, List[str]], bucketed=True, zero_rule=True, search_best_cp=False) -> Tuple[
    pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    :param versions:
    :param trained_on:
    :param test_on:
    :return: (summary_df, Dict[version, Dict[ds, results_df]])
    """

    trained_on = trained_on if type(trained_on) is list else [trained_on]
    test_on = test_on if type(test_on) is list else [test_on]

    results = {}
    result_summary = []

    for train_ds in trained_on:
        if zero_rule:
            for test_ds in test_on:
                accuracy = zero_rule_test_accuracy(test_ds, bucketed)
                print(f'test_ds: {test_ds} zero rule accuracy is: {accuracy}')
                info = [train_ds, test_ds, '', 'zero_rule', '']
                metrics = [accuracy] + ([0] * (len(metric_columns) - 1))
                desc = 'zero rule is always guessing the biggest group'

                result_summary.append([*info, *metrics, desc])

        for version in versions:
            cps = list_checkpoints(version, train_ds)
            if not cps:
                print(f'no checkpoints found for version {version.__name__}')
                continue
            if search_best_cp:
                # rank = mean(acc), sum(mae)
                epoch_ranks: List[float] = []
                epoch_results: List[Dict[str, Tuple[List, pd.DataFrame]]] = []
                for cp in cps:
                    accs, maes, rs = [], [], {}
                    for test_ds in test_on:
                        r = evaluate_version(version, train_ds, test_ds, bucketed, cp)
                        accs.append(r[0][5])
                        maes.append(np.log2(r[0][9]))
                        rs[test_ds] = r
                    epoch_ranks.append((np.sum(maes) * -1 + (np.mean(accs) * 2)))
                    epoch_results.append(rs)

                best_epoch_idx = cast(int, np.argmax(epoch_ranks))
                print(f'picked {cps[best_epoch_idx][1]} checkpoint, had rank of: {epoch_ranks[best_epoch_idx]}')

                for test_ds, r in epoch_results[best_epoch_idx].items():
                    result_summary.append(r[0])
                    results[f'{train_ds}_{str(version.__name__)}_{test_ds}'] = r[1]
            else:
                for test_ds in test_on:
                    r = evaluate_version(version, train_ds, test_ds, bucketed, cp=cps[-1])
                    result_summary.append(r[0])
                    results[f'{train_ds}_{str(version.__name__)}_{test_ds}'] = r[1]

    summary_df = pd.DataFrame(
        result_summary,
        columns=summary_columns)

    return summary_df, results


def compare_runs(version, test_on, bucketed=True, stratified=False) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    runs = list_runs(version)
    return inspect_version(version, trained_on=runs, test_on=test_on, bucketed=bucketed, stratified=stratified, latest=True)


def inspect_version(version, trained_on, test_on, bucketed=True, stratified=False, latest=False) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    trained_on = trained_on if type(trained_on) is list else [trained_on]
    test_on = test_on if type(test_on) is list else [test_on]

    results = {}
    result_summary = []

    for train_ds in trained_on:
        for test_ds in test_on:
            version_short, arch = str(version.__name__).split('.')[-2:]
            cps = list_checkpoints(version, train_ds)
            if latest:
                cps = cps[-1:]
            for cp in cps:
                r = evaluate_version(version, train_ds, test_ds, bucketed, cp=cp)
                result_summary.append(r[0])
                results[f'{train_ds}_{str(version.__name__)}_{test_ds}'] = r[1]

    summary_df = pd.DataFrame(result_summary, columns=summary_columns)

    return summary_df, results


if __name__ == '__main__':
    from dqo.estimator.gerelt import v2, v15

    compare_versions(versions=[v2, v15], trained_on='tpch', test_on='tpch', search_best_cp=True)
    inspect_version(v15, trained_on='tpch', test_on='tpch')
