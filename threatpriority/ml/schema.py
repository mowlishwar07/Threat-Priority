"""Validate inputs without guessing absent security measurements."""
import io
import re
import numpy as np
import pandas as pd

NUMERIC = ['duration', 'src_bytes', 'dst_bytes', 'src_packets', 'dst_packets', 'failed_logins', 'unique_dst_ports']
CATEGORICAL = ['protocol', 'service']
FEATURES = NUMERIC + CATEGORICAL
ALIASES = {'dur':'duration', 'sbytes':'src_bytes', 'dbytes':'dst_bytes', 'spkts':'src_packets',
           'dpkts':'dst_packets', 'proto':'protocol', 'timestamp':'timestamp', 'ts':'timestamp',
           'time':'timestamp', 'label':'label', 'class':'label', 'is_attack':'label',
           'attack_cat':'attack_type', 'attack_category':'attack_type', 'id':'event_id'}
BENIGN = {'0', '0.0', 'normal', 'benign', 'normal.', 'false'}
ATTACK = {'1', '1.0', 'attack', 'malicious', 'true', 'dos', 'ddos', 'bruteforce', 'brute_force',
          'exploits', 'fuzzers', 'generic', 'reconnaissance', 'shellcode', 'worms', 'backdoor',
          'backdoors', 'analysis', 'injection', 'scanning', 'xss', 'password', 'mitm', 'ransomware'}

def name(value):
    return re.sub(r'[^a-z0-9]+', '_', str(value).strip().lower()).strip('_')

def read_csv(data):
    import csv
    if isinstance(data, bytes):
        try:
            data = data.decode('utf-8-sig')
        except UnicodeError as exc:
            raise ValueError('Save your CSV as UTF-8 before uploading.') from exc
    try:
        header = next(csv.reader(io.StringIO(data)))
        if len(set(header)) != len(header):
            raise ValueError('Duplicate CSV headers are not allowed.')
        frame = pd.read_csv(io.StringIO(data), low_memory=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError, StopIteration) as exc:
        raise ValueError('Upload a non-empty comma-separated CSV with a header row.') from exc
    return frame

def normalize(frame, mapping=None):
    mapping = mapping or {}
    if not isinstance(mapping, dict):
        raise ValueError('Column mapping must be a JSON object: source header to canonical field.')
    allowed=set(FEATURES)|{'timestamp','label','event_id','attack_type'}
    if any(not isinstance(v,str) or v not in allowed for v in mapping.values()):
        raise ValueError('Mapping targets must be documented canonical feature or metadata names.')
    missing_sources = set(mapping) - set(frame.columns)
    if missing_sources:
        raise ValueError('Mapped source headers do not exist: ' + ', '.join(sorted(missing_sources)))
    columns = [mapping.get(c, ALIASES.get(name(c), name(c))) for c in frame.columns]
    if len(set(columns)) != len(columns):
        raise ValueError('Two headers map to the same field. Remove the duplicate or change the mapping.')
    result = frame.copy()
    result.columns = columns
    return result

def prepare(frame, mapping=None):
    raw = normalize(frame, mapping)
    if not 1 <= len(raw) <= 10000:
        raise ValueError('Provide between 1 and 10,000 events per batch.')
    warnings = []
    missing_columns = [c for c in FEATURES if c not in raw]
    x = raw.reindex(columns=FEATURES).copy()
    for c in NUMERIC:
        original = x[c]
        x[c] = pd.to_numeric(original, errors='coerce').replace([np.inf, -np.inf], np.nan)
        invalid = original.notna() & x[c].isna()
        negative = x[c] < 0
        count = int((invalid | negative).sum())
        if count:
            warnings.append(f'{c}: {count} invalid/negative value(s) treated as missing.')
        x.loc[negative, c] = np.nan
    for c in CATEGORICAL:
        x[c] = x[c].map(lambda v: np.nan if pd.isna(v) or not str(v).strip() else str(v).strip().lower())
        x[c] = x[c].astype(object)
    coverage = x.notna().sum(axis=1)
    bad = np.flatnonzero(coverage.to_numpy() < 3)
    if len(bad):
        rows = ', '.join(str(int(i)+2) for i in bad[:8])
        raise ValueError(f'Too little usable evidence at CSV row(s) {rows}. Each event needs at least 3 of 9 recognized features. See DATA_DICTIONARY.md or map headers.')
    timestamps = pd.Series(pd.NaT, index=raw.index, dtype='datetime64[ns, UTC]')
    if 'timestamp' in raw:
        values = raw.timestamp
        numeric = pd.to_numeric(values, errors='coerce')
        numeric_mask=numeric.notna()
        timestamps.loc[numeric_mask]=pd.to_datetime(numeric.loc[numeric_mask],unit='s',errors='coerce',utc=True)
        timestamps.loc[~numeric_mask]=pd.to_datetime(values.loc[~numeric_mask],errors='coerce',utc=True,format='mixed')
    if timestamps.isna().any():
        warnings.append(f'{int(timestamps.isna().sum())} absent/invalid timestamp(s): arrival baseline uses file row order for the whole batch; no capture times invented.')
    if missing_columns:
        warnings.append('Missing columns imputed: ' + ', '.join(missing_columns))
    ignored = sorted(set(raw.columns) - set(FEATURES) - {'timestamp', 'label', 'event_id', 'attack_type'})
    if ignored:
        warnings.append('Ignored columns: ' + ', '.join(ignored[:20]))
    return raw, x, timestamps, warnings

def labels(raw):
    if 'label' not in raw:
        return None, 'No label column: scoring works, but ranking quality and detection metrics require ground truth.'
    values = raw.label.map(lambda v: str(v).strip().lower())
    unknown = sorted(set(values) - BENIGN - ATTACK)
    if unknown:
        return None, 'Unrecognized/missing labels: ' + ', '.join(unknown[:8]) + '. Map every label to 0 (benign) or 1 (attack) to evaluate.'
    return values.isin(ATTACK).astype(int).to_numpy(), None
