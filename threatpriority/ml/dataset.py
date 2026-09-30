"""Synthetic research scenarios, explicitly not captured network traffic."""
from pathlib import Path
import numpy as np
import pandas as pd
from ml.schema import FEATURES

SCENARIOS = ['normal_web', 'backup', 'password_mistakes', 'busy_service', 'brute_force', 'scan', 'exfiltration', 'flood']

def generate(n=6000, seed=42):
    r = np.random.default_rng(seed)
    scenarios = r.choice(SCENARIOS, n, p=[.35,.13,.12,.12,.08,.07,.06,.07])
    rows = []
    for i, s in enumerate(scenarios):
        attack = int(s in SCENARIOS[4:])
        duration = float(r.lognormal(1.6,1.0))
        src_bytes = float(r.lognormal(8.1,1.2))
        dst_bytes = float(r.lognormal(8.8,1.2))
        src_packets = max(1, int(src_bytes/r.uniform(200,1200)))
        dst_packets = max(1, int(dst_bytes/r.uniform(200,1200)))
        failed = int(r.poisson(.6))
        ports = int(r.poisson(1)+1)
        protocol = r.choice(['tcp','udp','icmp'],p=[.8,.17,.03])
        service = r.choice(['http','dns','ssh','other'],p=[.55,.2,.15,.1])
        if s in ['backup','exfiltration']:
            src_bytes = float(r.lognormal(15 if s=='exfiltration' else 14.5,1.25))
            src_packets = max(1,int(src_bytes/r.uniform(500,1400)))
            duration = float(r.lognormal(4.5,1))
            dst_bytes *= r.uniform(.01,.3) if attack else r.uniform(.1,1)
        if s in ['password_mistakes','brute_force']:
            failed = int(r.negative_binomial(4,.22 if attack else .55))
            service = r.choice(['ssh','http'],p=[.75,.25])
        if s=='scan':
            ports = int(r.lognormal(3.1,.75))
            src_bytes *= .12
            dst_bytes *= .08
            duration *= .25
        if s in ['busy_service','flood']:
            src_packets = int(r.lognormal(7.5 if attack else 6.8,1.0))
            duration = float(r.lognormal(1.2 if attack else 3.2,1.0))
            protocol = r.choice(['tcp','udp'],p=[.5,.5])
        # Label noise models imperfect scenario attribution; it is documented, not hidden.
        if r.random() < .025:
            attack = 1 - attack
        rows.append({'event_id':f'S{seed}-{i:05}', 'timestamp':(pd.Timestamp('2026-01-01',tz='UTC')+pd.Timedelta(seconds=i*2)).isoformat(),
                     'duration':round(duration,4),'src_bytes':round(src_bytes),'dst_bytes':round(dst_bytes),
                     'src_packets':src_packets,'dst_packets':dst_packets,'failed_logins':failed,
                     'unique_dst_ports':ports,'protocol':protocol,'service':service,'label':attack,'scenario':s})
    df = pd.DataFrame(rows)
    # Missingness is label-independent. Entire absent fields are also tested at inference.
    for c in FEATURES:
        df.loc[r.random(n) < .07, c] = np.nan
    return df

def write_data(root):
    root = Path(root)
    root.mkdir(exist_ok=True, parents=True)
    generate(6000,42).to_csv(root/'synthetic_events.csv',index=False)
    demo = generate(300,2026)
    demo.to_csv(root/'demo_labeled.csv',index=False)
    demo.drop(columns=['label','scenario']).to_csv(root/'demo_unlabeled.csv',index=False)
    demo.drop(columns=['dst_bytes','dst_packets','failed_logins','scenario']).to_csv(root/'demo_missing_columns.csv',index=False)
    demo.head(3).drop(columns=['label','scenario']).to_csv(root/'upload_template.csv',index=False)

if __name__ == '__main__':
    write_data(Path(__file__).resolve().parents[1]/'data')
