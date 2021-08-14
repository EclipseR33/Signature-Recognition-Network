from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
from tqdm import tqdm


def f1_score(y_true, y_pred):
    y_true = y_true.apply(lambda x: set(x.split()))
    y_pred = y_pred.apply(lambda x: set(x.split()))
    intersection = np.array([len(x[0] & x[1]) for x in zip(y_true, y_pred)])
    len_y_pred = y_pred.apply(lambda x: len(x)).values
    len_y_true = y_true.apply(lambda x: len(x)).values
    f1 = 2 * intersection / (len_y_pred + len_y_true)
    return f1


def get_neighbors(df, embeddings, threshold, get_threshold=False, threshold_range=None, n_neighbors=10, notebook=None):
    pd.set_option('precision', 8)

    model = NearestNeighbors(n_neighbors=n_neighbors)
    model.fit(embeddings)

    distances, indices = model.kneighbors(embeddings)
    if get_threshold:
        thresholds = list(np.arange(*threshold_range))
        scores = []
        # thresholds = tqdm(thresholds)
        for threshold in thresholds:
            predictions = []
            for k in range(embeddings.shape[0]):
                idx = np.where(distances[k,] < threshold)[0]
                ids = indices[k, idx]
                posting_ids = ' '.join([str(int(i)) for i in df['index'].iloc[ids].values])
                predictions.append(posting_ids)
            df['pred_matches'] = predictions
            df['f1'] = f1_score(df['matches'], df['pred_matches'])
            score = df['f1'].mean()
            if notebook is not None:
                notebook.update_f1_scores(threshold, score)
            print(f'Threshold: {threshold} -> f1 score {score}')
            scores.append(score)
        thresholds_scores = pd.DataFrame({'thresholds': thresholds, 'scores': scores})
        max_score = thresholds_scores[thresholds_scores['scores'] == thresholds_scores['scores'].max()]
        best_threshold = max_score['thresholds'].values[0]
        best_score = max_score['scores'].values[0]
        print(f'*Best score: {best_score} <- Threshold {best_threshold}')

    predictions = []
    for k in range(embeddings.shape[0]):
        idx = np.where(distances[k, ] < threshold)[0]
        ids = indices[k, idx]
        posting_ids = df['index'].iloc[ids].values
        predictions.append(posting_ids)
    df['pred_matches'] = predictions
    return df
