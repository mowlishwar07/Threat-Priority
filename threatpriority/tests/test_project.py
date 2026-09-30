import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from app import create_app
from ml.schema import prepare, read_csv, labels, FEATURES
from ml.scoring import Scorer, severity
from ml.evaluation import evaluate

ROOT=Path(__file__).resolve().parents[1]
HEADERS={'X-Requested-With':'ThreatPriority'}

@pytest.fixture(scope='module')
def scorer():
    return Scorer(ROOT/'models'/'risk_model.joblib')

@pytest.fixture
def client(tmp_path):
    return create_app(feedback_db=tmp_path/'feedback.sqlite').test_client()

def demo():
    return pd.read_csv(ROOT/'data'/'demo_labeled.csv').head(16)

def upload(client,frame,**fields):
    return client.post('/api/upload',data={'file':(io.BytesIO(frame.to_csv(index=False).encode()),'events.csv'),**fields},headers=HEADERS)

def test_missing_columns_and_imputation(scorer):
    frame=demo().drop(columns=['dst_bytes','dst_packets','failed_logins'])
    raw,x,ts,p,warnings,ms=scorer.score(frame)
    assert np.isfinite(p).all() and np.all((p>=0)&(p<=1))
    assert x.failed_logins.isna().all()
    assert any('Missing columns' in w for w in warnings)

@pytest.mark.parametrize('index',[0,1,2,3,4])
def test_shap_reconstructs_probability(scorer,index):
    raw,x,ts,p,warnings,ms=scorer.score(demo())
    values={c:None if pd.isna(x.iloc[index][c]) else x.iloc[index][c] for c in FEATURES}
    explanation=scorer.explain(values)
    assert abs(explanation['score']-p[index]*100)<1e-6
    assert abs(explanation['base_score']+sum(f['contribution_points'] for f in explanation['features'])-explanation['score'])<.001
    assert len(explanation['features'])==9

def test_shap_imputed_features_are_marked(scorer):
    record=demo().iloc[0].to_dict()
    record['failed_logins']=None
    explanation=scorer.explain(record)
    assert next(f for f in explanation['features'] if f['feature']=='failed_logins')['imputed']

def test_labels_and_scenario_do_not_change_scores(scorer):
    frame=demo()
    p1=scorer.score(frame)[3]
    frame['label']=1-frame.label
    frame['scenario']='secret attack label'
    frame['timestamp']='2030-01-01T00:00:00Z'
    np.testing.assert_allclose(p1,scorer.score(frame)[3])

def test_unknown_category_warns(scorer):
    frame=demo()
    frame['protocol']='unknown_protocol'
    assert any('unseen category' in w for w in scorer.score(frame)[4])

def test_aliases_and_explicit_mapping(scorer):
    frame=demo().rename(columns={'src_bytes':'sbytes','duration':'duration_seconds','protocol':'proto'})
    assert len(scorer.score(frame,{'duration_seconds':'duration'})[3])==16

def test_invalid_values_become_missing(scorer):
    frame=demo().astype({'duration':object})
    frame.loc[0,'duration']='not-number'
    frame.loc[1,'duration']=-8
    frame.loc[2,'duration']=float('inf')
    assert scorer.score(frame)[1].duration.iloc[:3].isna().all()

@pytest.mark.parametrize('frame',[pd.DataFrame(),pd.DataFrame({'junk':[1]}),pd.DataFrame({'protocol':['tcp'],'service':['ssh']})])
def test_insufficient_evidence_rejected(frame):
    with pytest.raises(ValueError):
        prepare(frame)

def test_duplicate_headers_and_alias_collision():
    with pytest.raises(ValueError): read_csv('src_bytes,src_bytes\n1,2')
    with pytest.raises(ValueError): prepare(pd.DataFrame({'duration':[1],'dur':[2]}))

def test_bad_mapping_rejected():
    with pytest.raises(ValueError): prepare(demo(),{'absent':'src_bytes'})
    with pytest.raises(ValueError): prepare(demo(),{'duration':'invented'})

def test_mixed_epoch_and_iso_timestamps():
    frame=demo().head(3).copy()
    frame['timestamp']=[1767225600,'2026-01-01T00:00:02Z',None]
    timestamps=prepare(frame)[2]
    assert timestamps.iloc[0].year==2026
    assert timestamps.iloc[1].year==2026
    assert pd.isna(timestamps.iloc[2])

def test_known_ranking_values_and_timestamp_baseline():
    y=[0,1,1,0]
    p=[.2,.9,.8,.1]
    ts=pd.Series(pd.to_datetime(['2026-01-01T00:00:03Z','2026-01-01T00:00:02Z','2026-01-01T00:00:01Z','2026-01-01T00:00:00Z']))
    result=evaluate(y,p,ts,2)
    assert result['ml_queue']['precision_at_k']==1
    assert result['ml_queue']['recall_at_k']==1
    assert result['ml_queue']['ndcg_at_k']==1
    assert result['arrival_queue']['attacks_found']==1
    assert result['additional_attacks_at_k']==1

def test_ties_stable_and_missing_time_fallback():
    ts=pd.Series(pd.to_datetime([None,None,None],utc=True))
    result=evaluate([0,1,1],[.5,.5,.5],ts,1)
    assert result['ml_queue']['attacks_found']==0
    assert result['arrival_basis']=='file row order'

@pytest.mark.parametrize('y',[[0,0,0],[1,1,1]])
def test_one_class_metrics_no_nan(y):
    result=evaluate(y,[.1,.5,.9],pd.Series(pd.to_datetime([None]*3,utc=True)),20)
    assert result['effective_budget']==3
    assert 'NaN' not in json.dumps(result,allow_nan=False)
    if not sum(y): assert result['ml_queue']['recall_at_k'] is None
    else: assert result['fpr'] is None

@pytest.mark.parametrize('p,expected',[(.349,('Low','P4')),(.35,('Medium','P3')),(.65,('High','P2')),(.85,('Critical','P1'))])
def test_severity_boundaries(p,expected):
    assert severity(p)==expected

def test_labeled_csv_evaluation_and_queue_sorted(client):
    result=upload(client,demo(),budget='5')
    assert result.status_code==200
    evaluation=result.json['evaluation']
    assert evaluation['available'] and evaluation['effective_budget']==5
    events=client.get('/api/queue').json['events']
    assert [e['score'] for e in events]==sorted((e['score'] for e in events),reverse=True)
    assert all('label' not in e for e in events)
    explanation=client.get('/api/explain/'+events[0]['id']).json
    assert abs(explanation['score']-events[0]['score'])<.0001

@pytest.mark.parametrize('mode',['absent','unknown','missing'])
def test_no_fake_evaluation(client,mode):
    frame=demo()
    if mode=='absent':frame=frame.drop(columns=['label'])
    elif mode=='unknown':frame['label']='maybe'
    else:frame.loc[0,'label']=np.nan
    response=upload(client,frame)
    assert response.status_code==200 and not response.json['evaluation']['available']

def test_invalid_upload_does_not_replace_queue(client):
    upload(client,demo())
    response=upload(client,pd.DataFrame({'junk':[1]}))
    assert response.status_code==400
    assert len(client.get('/api/queue').json['events'])==16

def test_feedback_does_not_change_labels_or_scores(client):
    upload(client,demo())
    before=client.get('/api/evaluate').json
    event=client.get('/api/queue').json['events'][0]
    result=client.post('/api/feedback/'+event['id'],json={'decision':'Benign'},headers=HEADERS)
    assert result.status_code==200
    assert client.get('/api/evaluate').json==before
    assert client.get('/api/queue').json['events'][0]['status']=='Benign'

def test_api_event_ingestion_and_unlabeled_queue(client):
    record=demo().drop(columns=['scenario','label']).iloc[0].to_dict()
    record={k:None if pd.isna(v) else v for k,v in record.items()}
    response=client.post('/api/events',json={'event':record},headers=HEADERS)
    assert response.status_code==200
    assert not client.get('/api/evaluate').json['available']

def test_replay_demo_export_and_clear(client):
    assert len(client.get('/api/replay-data').json)==300
    assert client.post('/api/demo',headers=HEADERS).status_code==200
    exported=client.get('/api/export')
    assert exported.status_code==200 and len(exported.data.decode().splitlines())==301
    assert client.delete('/api/queue',headers=HEADERS).status_code==200
    assert not client.get('/api/queue').json['events']

def test_csv_formula_escape(client):
    frame=demo()
    frame['service']='=HYPERLINK("bad")'
    upload(client,frame)
    assert "'=hyperlink" in client.get('/api/export').data.decode()

def test_bad_budget_upload_does_not_mutate_queue(client):
    assert upload(client,demo(),budget='bad').status_code==400
    assert not client.get('/api/queue').json['events']

def test_write_header_required(client):
    assert client.post('/api/demo').status_code==403

def test_request_and_row_limits(client):
    with pytest.raises(ValueError):
        prepare(pd.concat([demo()]*626,ignore_index=True))
    response=client.post('/api/events',data=b'x'*(8*1024*1024+1),content_type='application/json',headers=HEADERS)
    assert response.status_code==413

def test_home_static_security_headers(client):
    response=client.get('/')
    assert response.status_code==200 and 'Content-Security-Policy' in response.headers
    assert client.get('/static/app.js').status_code==200
