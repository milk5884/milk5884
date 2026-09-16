import base64,csv,gzip,io,json,math,urllib.request,itertools
from collections import defaultdict
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
b64=''.join((ROOT/f'oos_chunk_{i:02d}').read_text().strip() for i in range(8))
rows=json.loads(gzip.decompress(base64.b64decode(b64)))
by_date=defaultdict(list)
for r in rows: by_date[r[1]].append(r)
combos=[f'{a}-{b}-{c}' for a in range(1,7) for b in range(1,7) if b!=a for c in range(1,7) if c not in (a,b)]
coef=(-2.410753468858385,-0.018870052006360726,-0.07620231216222971,-0.35281962387142796,-0.5336285907516856,-0.5785323173669995,-0.8371690294337591)
def sig(z):
    if z>=0:return 1/(1+math.exp(-z))
    e=math.exp(z);return e/(1+e)
def prob(rank,score,r):
    return sig(coef[0]+coef[1]*score+coef[2]*rank+coef[3]*r[5]+coef[4]*r[6]+coef[5]*r[7]+coef[6]*r[8])
market={}; fetch_errors=[]
for date,races in sorted(by_date.items()):
    y,m,d=date.split('-'); url=f'https://raw.githubusercontent.com/BoatraceCSV/boatracecsv.github.io/main/data/previews/od3/{y}/{m}/{d}.csv'
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'pv4-buying-strategy-research/1.0'})
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
                acq=datetime.fromisoformat(x['取得日時']);ddl=datetime.fromisoformat(x['レース日']+'T'+x['締切時刻']+':00+09:00');lead=(ddl-acq).total_seconds()/60
            except:lead=None
            market[rc]={'odds':vals,'complete':complete,'preclose':lead is not None and lead>0,'lead':lead}
    except Exception as e:fetch_errors.append([date,repr(e)])
race_tickets_all={}; race_tickets_market={}
for r in rows:
    rc,date,split,result,payout=r[:5]
    aa=[]; mm=[]; mk=market.get(rc)
    for rank,comb,score in r[9]:
        base={'race_code':rc,'date':date,'rank':rank,'comb':comb,'hit':comb==result,'payout':payout if comb==result else 0}
        aa.append(base)
        if mk and mk['complete'] and mk['preclose']:
            o=mk['odds'].get(comb)
            if o is not None and o>0:
                p=prob(rank,score,r); mm.append({**base,'odds':o,'p':p,'ev':p*o})
    race_tickets_all[rc]=aa
    if mm: race_tickets_market[rc]=mm

def metrics(bets):
    n=len(bets); stake=100*n; pay=sum(b['payout'] for b in bets); hits=sum(1 for b in bets if b['hit'])
    race_net=defaultdict(float)
    for b in bets: race_net[(b['date'],b['race_code'])]+=b['payout']-100
    cum=peak=dd=0
    for k in sorted(race_net):
        cum+=race_net[k]; peak=max(peak,cum); dd=max(dd,peak-cum)
    pays=sorted((b['payout'] for b in bets if b['payout']>0),reverse=True)
    def excl(k):
        if n<=k:return None
        removed=sum(pays[:k]); st=100*(n-k)
        return 100*(pay-removed)/st if st else None
    return {'bets':n,'hits':hits,'hit_rate':100*hits/n if n else None,'roi':100*pay/stake if stake else None,'profit':pay-stake,'maxdd':dd,'l1':excl(1),'l2':excl(2),'largest':pays[0] if pays else 0}
def period(bets,p):
    if p=='aug': return [b for b in bets if b['date']<'2026-09-01']
    return [b for b in bets if b['date']>='2026-09-01']
def pack_metrics(bets):
    a=metrics(bets); am=metrics(period(bets,'aug')); sm=metrics(period(bets,'sep'))
    return {**a,'aug_roi':am['roi'],'aug_bets':am['bets'],'sep_roi':sm['roi'],'sep_bets':sm['bets'],'sep_l1':sm['l1']}

out=[]
for mask in range(1,1<<10):
    rs={i+1 for i in range(10) if mask&(1<<i)}
    bets=[b for aa in race_tickets_all.values() for b in aa if b['rank'] in rs]
    m=pack_metrics(bets); m.update({'family':'fixed_ranks','ranks':','.join(map(str,sorted(rs))),'min_odds':None,'max_odds':None,'ev_min':None,'ev_max':None})
    out.append(m)
ranksets=[]
ranksets += [{i} for i in range(1,11)]
ranksets += [{1,2,3},{1,2,3,4,5},{3,10},{3,9,10},{5,7,10},{4,7,10},{3,5,7,10},{7,10},{9,10}]
seen=set(); ranksets2=[]
for s in ranksets:
    t=tuple(sorted(s))
    if t not in seen: seen.add(t); ranksets2.append(s)
mins=[0,5,10,15,20,30,40,50,75,100]
maxs=[30,50,75,100,150,200,300,500,None]
evmaxs=[2,3,4,5,6,8,10,15,20,None]
for rs in ranksets2:
  for lo in mins:
    for hi in maxs:
      if hi is not None and lo>=hi: continue
      for evhi in evmaxs:
        bets=[]
        for aa in race_tickets_market.values():
            for b in aa:
                if b['rank'] not in rs: continue
                if b['odds']<lo: continue
                if hi is not None and b['odds']>hi: continue
                if evhi is not None and b['ev']>evhi: continue
                bets.append(b)
        if len(bets)<40: continue
        m=pack_metrics(bets);m.update({'family':'market_filter','ranks':','.join(map(str,sorted(rs))),'min_odds':lo,'max_odds':hi,'ev_min':None,'ev_max':evhi})
        out.append(m)
for rankcap in [1,3,5,10]:
  for hi in [30,50,75,100,150,200,300,None]:
    for evhi in [2,3,4,5,6,8,10,None]:
      for selector in ['rank','p','ev','low_odds']:
        bets=[]
        for aa in race_tickets_market.values():
            cc=[b for b in aa if b['rank']<=rankcap and (hi is None or b['odds']<=hi) and (evhi is None or b['ev']<=evhi)]
            if not cc: continue
            if selector=='rank': b=min(cc,key=lambda z:(z['rank'],-z['p'],z['comb']))
            elif selector=='p': b=max(cc,key=lambda z:(z['p'],-z['rank'],z['comb']))
            elif selector=='ev': b=max(cc,key=lambda z:(z['ev'],z['p'],-z['rank'],z['comb']))
            else: b=min(cc,key=lambda z:(z['odds'],z['rank'],z['comb']))
            bets.append(b)
        if len(bets)<40: continue
        m=pack_metrics(bets);m.update({'family':'one_ticket','ranks':f'1-{rankcap}','min_odds':None,'max_odds':hi,'ev_min':None,'ev_max':evhi,'selector':selector})
        out.append(m)
valid=[x for x in out if x['bets']>=50 and x['roi'] is not None]
def v(x,k,default=-1e9):
    y=x.get(k);return default if y is None else y
rob=[x for x in valid if v(x,'roi')>=120 and v(x,'aug_roi')>=100 and v(x,'sep_roi')>=100 and v(x,'l1')>=100]
rob=sorted(rob,key=lambda x:(min(v(x,'aug_roi'),v(x,'sep_roi'),v(x,'l1')),v(x,'roi'),x['bets']),reverse=True)
top=sorted([x for x in valid if x['bets']>=100],key=lambda x:(v(x,'roi'),v(x,'l1')),reverse=True)
names=[]
for ranks in ['10','3,10','3,9,10','5,7,10','4,7,10']:
    z=[x for x in out if x['family']=='fixed_ranks' and x['ranks']==ranks]
    if z:names.append(z[0])
summary={'target_races':len(rows),'market_rows':len(market),'market_usable_races':len(race_tickets_market),'fetch_errors':fetch_errors,'named_fixed_rank':names,'robust_exploratory_count':len(rob),'robust_exploratory_top20':rob[:20],'overall_top20_min100bets':top[:20]}
(ROOT/'explore_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
fields=sorted(set().union(*(x.keys() for x in out)))
with open(ROOT/'explore_all.csv','w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(out)
print(json.dumps(summary,ensure_ascii=False,indent=2))
