"""Generate the report from actual saved metrics, model outputs, and validation evidence."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'runtime'/'mpl'))
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from ml.schema import read_csv, FEATURES
from ml.scoring import Scorer
from ml.evaluation import evaluate

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'reports'
INK='#122a42'; TEAL='#087f76'; MUTED='#566b7b'; LIGHT='#eaf5f4'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})


def pct(v):
    return 'N/A' if v is None else f'{v*100:.1f}%'


def architecture():
    fig,ax=plt.subplots(figsize=(10,5.5))
    ax.set_xlim(0,10);ax.set_ylim(0,6);ax.axis('off')
    positions={'incoming':(.6,4.5),'validate':(3.3,4.5),'pre':(6.1,4.5),'forest':(6.1,2.7),'queue':(3.3,2.7),'shap':(8.1,.9),'evaluate':(3.3,.9),'ui':(.6,2.7),'feedback':(.6,.9)}
    labels={'incoming':'CSV / JSON events','validate':'Schema validation','pre':'Fitted preprocessing','forest':'Random Forest','queue':'Score + ranked queue','shap':'Tree SHAP','evaluate':'Label-based evaluation','ui':'Analyst dashboard','feedback':'Feedback / SQLite'}
    for key,(x,y) in positions.items():
        width=2.2 if key!='shap' else 1.7
        ax.add_patch(FancyBboxPatch((x,y),width,.7,boxstyle='round,pad=0.08',linewidth=1,edgecolor=TEAL,facecolor=LIGHT))
        ax.text(x+width/2,y+.35,labels[key],ha='center',va='center',fontsize=9,color=INK)
    def arrow(a,b,start_side,end_side):
        x,y=positions[a];u,v=positions[b];wa=1.7 if a=='shap' else 2.2;wb=1.7 if b=='shap' else 2.2
        sides=lambda x,y,w:{'left':(x-.1,y+.35),'right':(x+w+.1,y+.35),'top':(x+w/2,y+.82),'bottom':(x+w/2,y-.1)}
        ax.annotate('',xy=sides(u,v,wb)[end_side],xytext=sides(x,y,wa)[start_side],arrowprops={'arrowstyle':'->','color':MUTED,'lw':1.3})
    arrow('incoming','validate','right','left');arrow('validate','pre','right','left');arrow('pre','forest','bottom','top');arrow('forest','queue','left','right');arrow('queue','ui','left','right');arrow('queue','evaluate','bottom','top');arrow('ui','feedback','bottom','top');arrow('forest','shap','bottom','top')
    ax.plot([9.0,9.0,2.97,2.97],[1.76,4.03,4.03,3.05],color=TEAL,lw=1.1)
    ax.annotate('',xy=(2.82,3.05),xytext=(2.97,3.05),arrowprops={'arrowstyle':'->','color':TEAL,'lw':1.1})
    ax.text(5.0,4.12,'SHAP explanation to dashboard',ha='center',fontsize=8,color=TEAL)
    ax.text(4.4,.35,'Ground-truth labels enter evaluation only',ha='center',fontsize=9,color=MUTED)
    ax.text(8.9,.35,'Training background',ha='center',fontsize=8,color=MUTED)
    ax.text(5.1,5.75,'ONLINE INFERENCE AND ANALYST WORKFLOW',ha='center',fontsize=12,fontweight='bold',color=INK)
    fig.tight_layout()
    fig.savefig(OUT/'architecture.png',dpi=180,bbox_inches='tight');fig.savefig(OUT/'architecture.svg',bbox_inches='tight');plt.close(fig)


def figures(scorer,frame):
    raw,x,ts,p,warnings,ms=scorer.score(frame)
    y=raw.label.to_numpy(dtype=int)
    ks=[10,20,50,100,200]
    budget_results=[evaluate(y,p,ts,k) for k in ks]
    (OUT/'ranking_budgets.json').write_text(json.dumps(budget_results,indent=2))
    fig,ax=plt.subplots(figsize=(9,3.8))
    ax.plot(ks,[r['ml_queue']['precision_at_k']*100 for r in budget_results],marker='o',color=TEAL,label='ML priority')
    ax.plot(ks,[r['arrival_queue']['precision_at_k']*100 for r in budget_results],marker='o',color='#c36f2c',label='Arrival order')
    ax.set_xlabel('Investigation budget K');ax.set_ylabel('Precision@K (%)');ax.set_ylim(0,105);ax.grid(axis='y',alpha=.15);ax.legend();fig.tight_layout();fig.savefig(OUT/'ranking_quality.png',dpi=180);plt.close(fig)
    # Fixed-model missing-column stress test; not used for training/tuning.
    incomplete=frame.drop(columns=['dst_bytes','dst_packets','failed_logins'])
    raw2,x2,ts2,p2,w2,ms2=scorer.score(incomplete)
    ablation=evaluate(raw2.label.to_numpy(dtype=int),p2,ts2,20)
    (OUT/'missing_columns_evaluation.json').write_text(json.dumps(ablation,indent=2))
    ix=int(np.argmax(p))
    record={c:None if pd.isna(x.iloc[ix][c]) else x.iloc[ix][c] for c in FEATURES}
    explanation=scorer.explain(record)
    (OUT/'shap_example.json').write_text(json.dumps(explanation,indent=2))
    features=list(reversed(explanation['features']))
    fig,ax=plt.subplots(figsize=(8.5,4.4))
    vals=[f['contribution_points'] for f in features]
    ax.barh([f['feature']+(' (imputed)' if f['imputed'] else '') for f in features],vals,color=[TEAL if v<0 else '#cc5966' for v in vals])
    ax.axvline(0,color=MUTED,lw=.8);ax.set_xlabel('Contribution to the attack-likelihood score (points)');ax.grid(axis='x',alpha=.12);fig.tight_layout();fig.savefig(OUT/'shap_example.png',dpi=180);plt.close(fig)
    # Warm single-event timings are separate from bulk-throughput averages.
    latencies=[]
    for _ in range(20):
        began=time.perf_counter();scorer.score(frame.head(1));latencies.append((time.perf_counter()-began)*1000)
    performance={'environment':'Linux x86-64; Python 3.12; machine-specific measurements, not guaranteed production latency',
                 'warm_single_event_mean_ms':float(np.mean(latencies)),'warm_single_event_p95_ms':float(np.percentile(latencies,95)),
                 'sample_shap_ms':explanation['latency_ms'],'replay_rate':'1 event per second'}
    (OUT/'latency.json').write_text(json.dumps(performance,indent=2))
    return budget_results,ablation,explanation,performance


def build_report():
    OUT.mkdir(exist_ok=True)
    metrics=json.loads((OUT/'metrics.json').read_text())
    scorer=Scorer(ROOT/'models'/'risk_model.joblib')
    budgets,ablation,explanation,performance=figures(scorer,read_csv((ROOT/'data'/'held_out_test.csv').read_bytes()))
    architecture()
    font_root=Path(matplotlib.get_data_path())/'fonts'/'ttf'
    for family,file in [('DejaVuSans','DejaVuSans.ttf'),('DejaVuSans-Bold','DejaVuSans-Bold.ttf'),('DejaVuSansMono','DejaVuSansMono.ttf')]:
        pdfmetrics.registerFont(TTFont(family,str(font_root/file)))
    pdfmetrics.registerFontFamily('DejaVuSans',normal='DejaVuSans',bold='DejaVuSans-Bold',italic='DejaVuSans',boldItalic='DejaVuSans-Bold')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle('TitleX',fontName='DejaVuSans-Bold',fontSize=27,leading=31,textColor=colors.HexColor(INK),spaceAfter=17))
    styles.add(ParagraphStyle('SubtitleX',fontName='DejaVuSans',fontSize=12,leading=17,textColor=colors.HexColor(MUTED),spaceAfter=14))
    styles.add(ParagraphStyle('BodyX',fontName='DejaVuSans',fontSize=10,leading=14.8,textColor=colors.HexColor(INK),spaceAfter=10))
    styles.add(ParagraphStyle('HeadingX',fontName='DejaVuSans-Bold',fontSize=18,leading=23,textColor=colors.HexColor(TEAL),spaceAfter=14))
    styles.add(ParagraphStyle('SmallX',fontName='DejaVuSans',fontSize=8.4,leading=12,textColor=colors.HexColor(MUTED),spaceAfter=8))
    styles.add(ParagraphStyle('CellX',fontName='DejaVuSans',fontSize=8.5,leading=11.5,textColor=colors.HexColor(INK)))
    styles.add(ParagraphStyle('CodeX',fontName='DejaVuSansMono',fontSize=8.3,leading=13,textColor=colors.HexColor(INK),spaceAfter=10))
    story=[]
    def p(text,style='BodyX'):
        story.append(Paragraph(text,styles[style]))
    def h(text):p(text,'HeadingX')
    def table(rows,widths):
        data=[[Paragraph(str(v),styles['CellX']) for v in row] for row in rows]
        t=Table(data,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor(LIGHT)),('FONTNAME',(0,0),(-1,0),'DejaVuSans-Bold'),('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),.8,colors.HexColor(TEAL)),('LINEBELOW',(0,1),(-1,-1),.35,colors.HexColor('#d8e1e6')),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8)]))
        story.extend([t,Spacer(1,12)])
    def image(path,width,max_height):
        from PIL import Image as PILImage
        a,b=PILImage.open(path).size
        height=width*b/a
        if height>max_height:width*=max_height/height;height=max_height
        story.append(Image(str(path),width=width,height=height,hAlign='LEFT'))
        story.append(Spacer(1,10))
    def new():story.append(PageBreak())
    content_width=A4[0]-92
    p('MLCS-HACK-07 / II YEAR CYBERSECURITY','SmallX')
    p('ThreatPriority','TitleX')
    p('Real-Time Cyber Threat Risk Scoring<br/>Project report and validation evidence','SubtitleX')
    p('<b>Problem:</b> Security analysts receive more events than they can investigate at once. The system assigns incoming events a 0-100 attack-likelihood score, ranks the queue, and explains each prediction so the analyst can choose what to inspect first.')
    p('<b>Implementation:</b> Python 3.12, Flask, scikit-learn Random Forest, genuine Tree SHAP, local HTML/CSS/JavaScript dashboard, and SQLite feedback. Code, model, datasets, architecture, tests and jury guide are bundled.')
    if (OUT/'dashboard.png').exists():image(OUT/'dashboard.png',content_width,320)
    p('Dashboard excerpt: ingestion, investigation-budget comparison and an event-specific SHAP explanation. The full queue is available by scrolling.','SmallX')
    p('<b>Data disclosure:</b> This release uses 6,000 documented synthetic events. It contains no downloaded Kaggle or production logs. Results validate the simulated distribution; they are not evidence of real-world attack-detection performance. Compatible public datasets require mapping, provenance checks and retraining.')
    new();h('1. Architecture and end-to-end workflow')
    image(OUT/'architecture.png',content_width,300)
    p('CSV/JSON events are validated, transformed by training-fitted preprocessing, and scored by Random Forest. The queue sorts unrounded scores. SHAP uses the same transformed event and a training-only background. Labels enter the evaluation branch; feedback remains a separate local record.')
    table([['Stage','What happens'],['Offline training','Synthetic generator or compatible labeled CSV; duplicate removal; 60/20/20 split; train-only preprocessing and augmentation; baseline comparison; test evaluation; serialized model.'],['Online ingestion','CSV upload replaces queue; event API appends records. The dashboard polls every 2 seconds for external incoming events. Replay sends one recorded synthetic event per second.'],['Explanation','Tree SHAP explains class-1 probability. Encoded contributions are grouped by source feature and checked for additive reconstruction.'],['Analyst workflow','Inspect top-ranked events, examine missingness and explanations, record a decision, export scores or retrieve evaluation JSON.']],[110,content_width-110])
    p('Queue state resets on restart. Feedback history is stored in runtime/feedback.sqlite. The local app has no packet capture, external services, automatic blocking, production SOC integration or automatic online retraining.','SmallX')
    new();h('2. Dataset, labels and security features')
    p('The generator includes normal web traffic, backups, password mistakes, busy services, brute force, scans, exfiltration and floods. Benign and attack distributions overlap. Scenarios set binary labels before training; 2.5% label-independent flips model imperfect attribution. Random feature missingness is 7%.')
    table([['Feature group','Meaning'],['duration','Connection duration in seconds.'],['src_bytes / dst_bytes','Directional byte volumes. Large outgoing transfers can be backups or suspicious transfers.'],['src_packets / dst_packets','Directional packet counts; high counts can reflect busy services or flooding.'],['failed_logins','Synthetic preceding-five-minute authentication-failure count.'],['unique_dst_ports','Synthetic preceding-five-minute distinct destination-port count.'],['protocol / service','Observed network protocol and application service, one-hot encoded.']],[145,content_width-145])
    p('Window counts are supplied feature summaries; the app does not compute them from captured historical logs. All units and assumptions are documented in DATA_DICTIONARY.md. Missing public-dataset measurements should be omitted rather than fabricated.')
    p('<b>Leakage controls:</b> Only nine allowlisted features enter X. Label, scenario, attack family, ID and timestamp are excluded. Feature-identical rows are removed before splitting. Imputation, encoding and SHAP background use only training data. A training-only masked copy augments missingness without creating independent evidence.')
    table([['Split','Rows','Purpose'],['Training','3,600','Learn preprocessing and model patterns.'],['Validation','1,200','Compare Logistic Regression and Random Forest.'],['Test','1,200','Measure the fixed forest on held-out records.'],['Demo','300','New seed; same synthetic distribution.']],[105,75,content_width-180])
    p('A random split is appropriate for this independent synthetic generator; it does not prove chronological generalization. Public logs may require time, session, host or collection-day splits.','SmallX')
    new();h('3. Model, scores and missing inputs')
    p('Random Forest uses 120 trees, maximum depth 10 and minimum 5 samples per leaf. Logistic Regression is a validation baseline. The forest was chosen for nonlinear interactions and native Tree SHAP; the test set is not used to search for a favorable threshold.')
    val=metrics['validation_comparison']
    table([['Validation model','Precision','Recall','F1'],*[ [title,pct(r['precision']),pct(r['recall']),pct(r['f1'])] for title,r in val.items()]],[190,100,100,content_width-390])
    p('<b>Risk score = 100 x estimated attack probability.</b> The forest output is uncalibrated; it is a likelihood proxy rather than a guaranteed attack percentage or a measured business-impact score. The event category is observed service/protocol, not a predicted attack family.')
    table([['Score band','Severity / priority'],['0 to below 35','Low / P4'],['35 to below 65','Medium / P3'],['65 to below 85','High / P2'],['85 to 100','Critical / P1']],[220,content_width-220])
    p('Missing/invalid numeric values use training medians plus fixed missing flags. Missing categorical values receive a dedicated token; unknown categories warn. At least 3 of 9 features must be usable in every row. This is a schema-validity minimum, not a confidence guarantee. Invalid batches are rejected atomically; imputed fields remain visible in the analyst explanation.')
    p('No timestamps are invented. ISO strings and Unix seconds are supported. If any time is missing/invalid, the arrival comparison uses input row order for the whole batch. Missing/unknown labels disable evaluation without preventing compatible-event scoring.')
    p(f"<b>Missing-column stress test:</b> Dropping dst_bytes, dst_packets and failed_logins from the same held-out file produced F1 {pct(ablation['f1'])}, FPR {pct(ablation['fpr'])}, and Precision@20 {pct(ablation['ml_queue']['precision_at_k'])}. This is a fixed-model robustness check; it was not used to tune the model.")
    new();h('4. SHAP: why is this event suspicious?')
    p('Tree SHAP is computed in probability space using interventional feature perturbation and 64 transformed training-background events. Positive values raise the attack estimate relative to the background; negative values lower it. One-hot and missing-indicator values are grouped back to the original input field.')
    image(OUT/'shap_example.png',content_width,290)
    p(f"<b>Measured example:</b> baseline {explanation['base_score']:.2f} points + signed contributions = score {explanation['score']:.2f}. Additive reconstruction error: {explanation['additivity_error']:.8f} points. All displayed contributions come from SHAP; no security-rule weights are relabeled as SHAP.")
    p(explanation['summary'])
    p('A missing feature is labeled imputed and is excluded from the short observed-evidence summary. Its contribution can still reflect the imputed value or missing flag. SHAP explains model behavior, not proof of an attack or the causal effect of changing a measurement. Correlated features may divide attribution, and the background selection matters.')
    p(f"<b>Timing:</b> measured warm single-event mean {performance['warm_single_event_mean_ms']:.1f} ms and p95 {performance['warm_single_event_p95_ms']:.1f} ms over 20 runs. Example SHAP computation {performance['sample_shap_ms']:.1f} ms. These Linux-machine measurements are separate from browser refresh delay and are not guaranteed production throughput.")
    new();h('5. Detection and ranking evaluation')
    t=metrics['test'];cm=t['confusion_matrix']
    table([['Held-out test metric','Measured value'],['Precision / Recall',pct(t['precision'])+' / '+pct(t['recall'])],['F1 / False-positive rate',pct(t['f1'])+' / '+pct(t['fpr'])],['Average precision / Brier score',pct(t['average_precision'])+f" / {t['brier_score']:.4f}"],['Confusion counts',f"TP {cm['tp']}; FP {cm['fp']}; TN {cm['tn']}; FN {cm['fn']}"],['Top 20 attacks: ML / arrival',f"{t['ml_queue']['attacks_found']} / {t['arrival_queue']['attacks_found']}"],['Recall@20: ML / arrival',pct(t['ml_queue']['recall_at_k'])+' / '+pct(t['arrival_queue']['recall_at_k'])]],[260,content_width-260])
    image(OUT/'ranking_quality.png',content_width,215)
    p('Precision@K = attacks in the top K / K; Recall@K = attacks in the top K / all attacks. Binary NDCG@K discounts lower ranks and normalizes against the best possible placement. The same K and labels are used for ML and arrival queues; ties preserve input order.')
    p('Detection metrics use a fixed score threshold of 50. A high top-20 precision can coexist with modest overall recall: 20 investigations cannot recover all 349 test attacks. Undefined quantities, such as Recall@K with no attacks or FPR with no benign events, are shown as N/A. Uploaded-file results describe that file and are not automatically an independent test.')
    new();h('6. Run, validate and explain to the jury')
    p('<b>Windows Python 3.12.10:</b> Extract the folder, open it in VS Code, and use Command Prompt. The direct environment executable avoids activation-policy issues. After setup, only the app.py command is needed.')
    p('py -3.12 -m venv .venv<br/>.venv\\Scripts\\python.exe -m pip install -r requirements.txt<br/>.venv\\Scripts\\python.exe app.py --port 5050','CodeX')
    p('<b>Mac:</b> Use native Python 3.12 (preferably 3.12.10), create the environment and run:')
    p('python3.12 -m venv .venv<br/>.venv/bin/python -m pip install -r requirements.txt<br/>.venv/bin/python app.py --port 5050','CodeX')
    p('Open <b>http://127.0.0.1:5050</b>. Install dependencies once with internet; afterward the local app works offline. Setup/run scripts, troubleshooting, the full feature schema and a Kaggle adaptation guide are supplied.')
    p('<b>Demo:</b> load demo; compare top-K queues; inspect SHAP; upload missing-column CSV; upload unlabeled CSV; start recorded replay; record feedback; export ranked results. For code questions, open Scorer.score/explain, train.make_preprocessor/train, and evaluation.evaluate. JURY_GUIDE.md contains a presentation script and detailed answers.')
    validation=json.loads((OUT/'browser_checks.json').read_text()) if (OUT/'browser_checks.json').exists() else {'passed':[]}
    test_text=(OUT/'test_results_python31210.txt').read_text() if (OUT/'test_results_python31210.txt').exists() else ''
    import re
    match=re.search(r'(\d+) passed',test_text)
    count=match.group(1) if match else 'See validation file'
    p(f'<b>Verification:</b> {count} automated tests passed under clean Linux Python 3.12.10; {len(validation["passed"])} browser checks passed. Binary dependency resolution succeeded for Windows x64 and Apple Silicon macOS Python 3.12. Native Windows/Mac execution was not available here. See VALIDATION.md for exact scope.')
    p('<b>References:</b> SHAP TreeExplainer official documentation: shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html; scikit-learn SimpleImputer documentation: scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html; Lundberg and Lee, A Unified Approach to Interpreting Model Predictions (2017), arxiv.org/abs/1705.07874. Challenge requirements supplied by the user.','SmallX')
    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor('#d8e1e6'));canvas.line(46,38,A4[0]-46,38)
        canvas.setFont('DejaVuSans',8);canvas.setFillColor(colors.HexColor(MUTED));canvas.drawString(46,25,'ThreatPriority / Synthetic-data research prototype');canvas.drawRightString(A4[0]-46,25,str(doc.page))
    doc=SimpleDocTemplate(str(OUT/'Project_Report.pdf'),pagesize=A4,rightMargin=46,leftMargin=46,topMargin=44,bottomMargin=51,title='ThreatPriority - Real-Time Cyber Threat Risk Scoring',author='ThreatPriority Project')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('Created reports/Project_Report.pdf from measured artifacts.')

if __name__=='__main__':
    build_report()
