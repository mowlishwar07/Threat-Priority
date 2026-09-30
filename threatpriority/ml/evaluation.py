"""Evaluate the same fixed investigation budget for every queue."""
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, average_precision_score, brier_score_loss

def evaluate(y, probabilities, timestamps, budget=20, threshold=.5):
    y = np.asarray(y, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    k = min(max(int(budget),1),len(y))
    ml_order = np.argsort(-p, kind='stable')
    complete_time = bool(timestamps.notna().all())
    arrival = np.argsort(timestamps.astype('int64').to_numpy(), kind='stable') if complete_time else np.arange(len(y))
    positives = int(y.sum())
    def queue(order):
        relevant = y[order[:k]]
        found = int(relevant.sum())
        dcg = float(np.sum(relevant/np.log2(np.arange(2,k+2))))
        ideal = float(np.sum(1/np.log2(np.arange(2,min(k,positives)+2))))
        return {'attacks_found':found, 'precision_at_k':found/k,
                'recall_at_k':found/positives if positives else None,
                'ndcg_at_k':dcg/ideal if ideal else None}
    tn,fp,fn,tp = confusion_matrix(y, p >= threshold, labels=[0,1]).ravel()
    return {'available':True, 'n_events':len(y), 'n_attacks':positives, 'requested_budget':int(budget), 'effective_budget':k,
            'threshold':threshold, 'precision':float(precision_score(y,p>=threshold,zero_division=0)),
            'recall':float(recall_score(y,p>=threshold,zero_division=0)) if positives else None,
            'f1':float(f1_score(y,p>=threshold,zero_division=0)) if positives else None,
            'fpr':float(fp/(fp+tn)) if fp+tn else None,
            'average_precision':float(average_precision_score(y,p)) if positives else None,
            'brier_score':float(brier_score_loss(y,p)),
            'confusion_matrix':{'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp)},
            'arrival_basis':'timestamp' if complete_time else 'file row order',
            'ml_queue':queue(ml_order), 'arrival_queue':queue(arrival),
            'additional_attacks_at_k':queue(ml_order)['attacks_found']-queue(arrival)['attacks_found'],
            'note':'Labels are evaluation-only; they never enter prediction. Ties preserve file order. Priority is attack likelihood, not measured impact.'}
