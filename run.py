import base64,csv,gzip,io,json,math,urllib.request,hashlib
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT=Path(__file__).resolve().parent
packed=''.join(p.read_text().strip() for p in sorted(ROOT.glob('oos.part*')))
assert hashlib.sha256(packed.encode()).hexdigest()=='762a8349237ec8ab2b44c38bfa8591505e5f7e306ef6acc181a64ee4c6002b72'
rows=json.loads(gzip.decompress(base64.b64decode(packed)))
# row: race_code,date,split,result,payout100,p_iwc,p_outer_follow,p_outer_head,p_outer_pair_top3, [[rank,combination,ordering_score],...]
by_date=defaultdict(list)
for r in rows: by_date[r[1]].append(r)
combos=[f'{a}-{b}-{c}' for a in range(1,7) for b in range(1,7) if b!=a for c in range(1,7) if c not in (a,b)]
coef=(-2.410753468858385,-0.018870052006360726,-0.07620231216222971,-0.35281962387142796,-0.5336285907516856,-0.5785323173669995,-0.8371690294337591)
def sig(z):
    if z>=0: return 1/(1+math.exp(-z))
    e=math.exp(z); return e/(1+e)
def prob(rank,score,r):
    return sig(coef[0]+coef[1]*score+coef[2]*rank+coef[3]*r[5]+coef[4]*r[6]+coef[5]*r[7]+coef[6]*r[8])

market={}; fetch_errors=[]
for date,races in sorted(by_date.items()):
    y,m,d=date.split('-'); url=f'https://raw.githubusercontent.com/BoatraceCSV/boatracecsv.github.io/main/data/previews/od3/{y}/{m}/{d}.csv'
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'pv4-oos-replay/1.0'})
        text=urllib.request.urlopen(req,timeout=30).read().decode('utf-8-sig')
        want={r[0] for r in races}
        for x in csv.DictReader(io.StringIO(text)):
            rc=x.get('レースコード','')
            if rc not in want: continue
            vals={}; complete=True
            for c in combos:
                v=x.get('3連単_'+c,'')
                try:
                    q=float(v)
                    if not math.isfinite(q) or q<0: complete=False
                    vals[c]=q
                except: complete=False; vals[c]=None
            try:
                acq=datetime.fromisoformat(x['取得日時']); ddl=datetime.fromisoformat(x['レース日']+'T'+x['締切時刻']+':00+09:00'); lead=(ddl-acq).total_seconds()/60
            except: lead=None
            market[rc]={'odds':vals,'complete':complete,'lead':lead,'preclose':lead is not None and lead>0,'acquired':x.get('取得日時'),'deadline':x.get('締切時刻')}
    except Exception as e:
        fetch_errors.append([date,repr(e)])

def select(rank_cap=10,odds_cap=None,min_odds=None):
    out=[]
    for r in rows:
        rc,date,split,result,payout=r[:5]; mk=market.get(rc)
        if not mk or not mk['complete'] or not mk['preclose']: continue
        cs=[]
        for rank,comb,score in r[9]:
            if rank>rank_cap: continue
            o=mk['odds'].get(comb)
            if o is None or o<=0: continue
            if odds_cap is not None and o>odds_cap: continue
            if min_odds is not None and o<min_odds: continue
            p=prob(rank,score,r); ev=p*o
            cs.append((ev,p,rank,comb,o))
        if not cs: continue
        ev,p,rank,comb,o=sorted(cs,key=lambda z:(-z[0],-z[1],z[2],z[3]))[0]
        hit=(comb==result)
        out.append({'race_code':rc,'date':date,'result':result,'selected':comb,'rank':rank,'odds':o,'p':p,'ev':ev,'hit':hit,'payout':payout if hit else 0,'lead_min':mk['lead']})
    return out

def metrics(sel,threshold=1.25):
    b=[x for x in sel if x['ev']>=threshold]
    stake=100*len(b); pay=sum(x['payout'] for x in b); hits=sum(x['hit'] for x in b)
    cum=peak=dd=0; lose=maxlose=0
    for x in sorted(b,key=lambda z:(z['date'],z['race_code'])):
        cum+=x['payout']-100; peak=max(peak,cum); dd=max(dd,peak-cum)
        if x['hit']: lose=0
        else: lose+=1; maxlose=max(maxlose,lose)
    def excl(k):
        q=sorted(b,key=lambda x:x['payout'],reverse=True)[k:]; st=100*len(q)
        return None if not st else 100*sum(x['payout'] for x in q)/st
    return {'bets':len(b),'hits':hits,'hit_rate_pct':None if not b else 100*hits/len(b),'stake':stake,'payout':pay,'profit':pay-stake,'roi_pct':None if not stake else 100*pay/stake,'max_dd':dd,'max_losing_streak':maxlose,'l1_roi_pct':excl(1),'l2_roi_pct':excl(2),'largest_payout':max([x['payout'] for x in b],default=0)}

base=select()
summary={'target_races':len(rows),'market_rows':len(market),'eligible_races':len(base),'fetch_errors':fetch_errors,'frozen_ev125':metrics(base,1.25),'august':metrics([x for x in base if x['date']<'2026-09-01'],1.25),'sep1_9':metrics([x for x in base if x['date']>='2026-09-01'],1.25)}
summary['threshold_scan']={str(t):metrics(base,t) for t in [1.0,1.25,1.5,1.75,2,2.5,3,4,5,6,8,10]}
strategies=[]
for rank_cap in [1,3,5,10]:
    for oc in [30,50,75,100,150,200,300,None]:
        m=metrics(select(rank_cap,oc),1.25); m.update({'rank_cap':rank_cap,'odds_cap':oc}); strategies.append(m)
summary['strategy_scan']=strategies
(ROOT/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
with open(ROOT/'selected.csv','w',newline='') as f:
    fields=['race_code','date','result','selected','rank','odds','p','ev','hit','payout','lead_min']; w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(base)
print(json.dumps(summary,ensure_ascii=False,indent=2))
