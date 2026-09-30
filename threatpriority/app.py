"""Single-user local analyst queue. Flask routes contain no ML training logic."""
import argparse
import csv
import io
import json
from pathlib import Path
import sqlite3
import threading
import uuid
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, Response
from ml.scoring import Scorer, severity
from ml.schema import FEATURES, labels, read_csv
from ml.evaluation import evaluate

ROOT=Path(__file__).resolve().parent

def create_app(model_path=None,feedback_db=None):
    app=Flask(__name__)
    app.config['MAX_CONTENT_LENGTH']=8*1024*1024
    scorer=Scorer(model_path or ROOT/'models'/'risk_model.joblib')
    queue=[]
    lock=threading.RLock()
    db_path=Path(feedback_db or ROOT/'runtime'/'feedback.sqlite')
    db_path.parent.mkdir(parents=True,exist_ok=True)
    with sqlite3.connect(db_path) as db:
        db.execute('CREATE TABLE IF NOT EXISTS feedback (event_id TEXT, decision TEXT, created_at TEXT DEFAULT CURRENT_TIMESTAMP, record TEXT)')
    app.extensions['scorer']=scorer

    def budget():
        try:
            value=int(request.values.get('budget',20))
        except (TypeError,ValueError) as exc:
            raise ValueError('Investigation budget must be an integer from 1 to 10,000.') from exc
        if not 1<=value<=10000:
            raise ValueError('Investigation budget must be an integer from 1 to 10,000.')
        return value

    def ingest(frame,mapping=None,replace=False,source='API event'):
        raw,x,ts,p,warnings,latency=scorer.score(frame,mapping)
        y,label_reason=labels(raw)
        added=[]
        for i in range(len(x)):
            internal_id=uuid.uuid4().hex
            reference=str(raw.iloc[i].get('event_id',''))
            if reference in ['', 'nan', 'None']:
                reference=internal_id[:12]
            values={c:None if pd.isna(x.iloc[i][c]) else x.iloc[i][c].item() if isinstance(x.iloc[i][c],np.generic) else x.iloc[i][c] for c in FEATURES}
            sev,priority=severity(float(p[i]))
            timestamp=None if pd.isna(ts.iloc[i]) else ts.iloc[i].isoformat()
            added.append({'id':internal_id,'event_ref':reference,'timestamp':timestamp,'category':str(values['service'] or 'network')+' / '+str(values['protocol'] or 'unknown'),
                          'probability':float(p[i]),'score':float(p[i]*100),'severity':sev,'priority':priority,
                          'status':'Unreviewed','features':values,'observed_features':sum(v is not None for v in values.values()),
                          'imputed_fields':[c for c,v in values.items() if v is None], 'label':None if y is None else int(y[i]),
                          'source':source,'batch_warning':label_reason,'latency_ms':latency/len(x)})
        with lock:
            if not replace and len(queue)+len(added)>10000:
                raise ValueError('Queue limit is 10,000 events. Clear it or upload a replacement CSV.')
            if replace:
                queue.clear()
            queue.extend(added)
        return {'accepted':len(added),'ids':[e['id'] for e in added],'warnings':warnings,
                'label_note':label_reason,'scoring_batch_ms':latency,'queue_size':len(queue)}

    @app.errorhandler(ValueError)
    def invalid(error):
        return jsonify(error=str(error)),400

    @app.errorhandler(413)
    def too_large(error):
        return jsonify(error='File/request exceeds the 8 MB limit.'),413

    @app.after_request
    def local_only(response):
        response.headers['X-Content-Type-Options']='nosniff'
        response.headers['Content-Security-Policy']="default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'"
        response.headers['Cache-Control']='no-store'
        return response

    @app.before_request
    def protect_local_writes():
        # Avoid cross-site browser requests mutating this local single-user queue.
        if request.method in ['POST','DELETE'] and request.headers.get('X-Requested-With')!='ThreatPriority':
            return jsonify(error='Local API writes require X-Requested-With: ThreatPriority.'),403

    @app.get('/')
    def home():
        return render_template('index.html')

    @app.get('/api/model')
    def model():
        return jsonify(metadata=scorer.bundle['metadata'],test_metrics=scorer.bundle['test_metrics'],features=FEATURES)

    @app.get('/api/queue')
    def get_queue():
        with lock:
            events=sorted(queue,key=lambda e:-e['probability'])
            # Ground truth is kept outside the normal analyst view.
            view=[{k:v for k,v in e.items() if k not in ['features','label','batch_warning']} for e in events]
        return jsonify(events=view)

    @app.post('/api/upload')
    def upload():
        if 'file' not in request.files:
            raise ValueError('Choose a CSV file first.')
        try:
            mapping=json.loads(request.form.get('mapping','{}'))
        except json.JSONDecodeError as exc:
            raise ValueError('Column mapping must be valid JSON.') from exc
        b=budget()
        result=ingest(read_csv(request.files['file'].read()),mapping,True,'Uploaded CSV')
        result['evaluation']=queue_evaluation(b)
        return jsonify(result)

    @app.post('/api/events')
    def events():
        body=request.get_json(silent=True)
        if not isinstance(body,dict):
            raise ValueError('Provide a JSON object containing an event or events list.')
        records=body.get('events',[body.get('event',body)])
        if not isinstance(records,list) or not records or not all(isinstance(v,dict) for v in records):
            raise ValueError('events must be a non-empty list of JSON objects.')
        return jsonify(ingest(pd.DataFrame(records),source='Incoming API event'))

    @app.post('/api/demo')
    def demo():
        return jsonify(ingest(read_csv((ROOT/'data'/'demo_labeled.csv').read_bytes()),replace=True,source='Synthetic demo'))

    @app.get('/api/replay-data')
    def replay_data():
        raw=read_csv((ROOT/'data'/'demo_labeled.csv').read_bytes()).drop(columns=['scenario'])
        return Response(raw.to_json(orient='records'),mimetype='application/json')

    @app.delete('/api/queue')
    def clear():
        with lock:
            queue.clear()
        return jsonify(cleared=True)

    def get_event(event_id):
        with lock:
            event=next((e for e in queue if e['id']==event_id),None)
        if event is None:
            raise ValueError('Event not found in the active queue.')
        return event

    @app.get('/api/explain/<event_id>')
    def explain(event_id):
        return jsonify(scorer.explain(get_event(event_id)['features']))

    @app.post('/api/feedback/<event_id>')
    def feedback(event_id):
        event=get_event(event_id)
        body=request.get_json(silent=True) or {}
        decision=body.get('decision')
        if decision not in ['Confirmed threat','Benign','Needs investigation']:
            raise ValueError('Choose Confirmed threat, Benign, or Needs investigation.')
        with lock:
            event['status']=decision
            with sqlite3.connect(db_path) as db:
                db.execute('INSERT INTO feedback(event_id,decision,record) VALUES (?,?,?)',
                           (event_id,decision,json.dumps(event)))
        return jsonify(status=decision,note='Stored locally. Feedback does not alter the trained model or evaluation labels.')

    def queue_evaluation(b):
        with lock:
            snapshot=list(queue)
        if not snapshot:
            return {'available':False,'reason':'Load events before evaluating.'}
        if any(e['label'] is None for e in snapshot):
            return {'available':False,'reason':'Every active event needs a valid ground-truth label. '+next((e['batch_warning'] for e in snapshot if e['batch_warning']),'An incoming event is unlabeled.')}
        ts=pd.to_datetime(pd.Series([e['timestamp'] for e in snapshot]),utc=True)
        return evaluate([e['label'] for e in snapshot],[e['probability'] for e in snapshot],ts,b)

    @app.get('/api/evaluate')
    def evaluation():
        return jsonify(queue_evaluation(budget()))

    @app.get('/api/export')
    def export():
        output=io.StringIO()
        fields=['rank','event_ref','timestamp','category','score','severity','priority','status','observed_features','imputed_fields','source']
        writer=csv.DictWriter(output,fieldnames=fields)
        writer.writeheader()
        with lock:
            for rank,e in enumerate(sorted(queue,key=lambda v:-v['probability']),1):
                row={c:e[c] for c in fields if c in e}
                row.update(rank=rank,score=round(e['score'],4),imputed_fields='; '.join(e['imputed_fields']))
                for c,value in row.items():
                    if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')):
                        row[c]="'"+value
                writer.writerow(row)
        return Response(output.getvalue(),mimetype='text/csv',headers={'Content-Disposition':'attachment; filename=ranked_alerts.csv'})

    return app

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--port',type=int,default=5050)
    parser.add_argument('--model',type=Path,default=ROOT/'models'/'risk_model.joblib')
    args=parser.parse_args()
    print('ThreatPriority local dashboard: http://127.0.0.1:'+str(args.port))
    create_app(args.model).run(host='127.0.0.1',port=args.port,debug=False,threaded=True)
