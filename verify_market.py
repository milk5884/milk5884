import base64,csv,gzip,io,json,math,urllib.request,hashlib,random,statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
packed=''.join(p.read_text().strip() for p in sorted(ROOT.glob('oos.part*')))
assert hashlib.sha256(packed.encode()).hexdigest()=='762a8349237ec8ab2b44c38bfa8591505e5f7e306ef6acc181a64ee4c6002b72'
rows=json.loads(gzip.decompress(base64.b64decode(packed)))
by_date=defaultdict(list)
for r in rows: by_date[r[1]].append(r)
combos=[f'{a}-{b}-{c}' for a in range(1,7) for b in range(1,7) if b!=a for c in range(1,7) if c not in (a,b)]
coef=(-2.410753468858385,-0.018870052006360726,-0.07620231216222971,-0.35281962387142796,-0.5336285907516856,-0.5785323173669995,-0.8371690294337591)
def sig(z):
    if z>=0:return 1/(1+math.exp(-z))
    e=math.exp(z);return e/(1+e)
def prob(rank,score,r): return sig(coef[0]+coef[1]*score+coef[2]*rank+coef[3]*r[5]+coef[4]*r[6]+coef[5]*r[7]+coef[6]*r[8])
market={}
for date,races in sorted(by_date.items()):
    y,m,d=date.split('-');url=f'https://raw.githubusercontent.com/BoatraceCSV/boatracecsv.github.io/main/data/previews/od3/{y}/{m}/{d}.csv'
    text=urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'pv4-verify/1.0'}),timeout=30).read().decode('utf-8-sig')
    want={r[0] for r in races}
    for x in csv.DictReader(io.StringIO(text)):
        rc=x.get('レースコード','')
        if rc not in want:continue
        vals={c:float(x['3連単_'+c]) for c in combos}
        acq=datetime.fromisoformat(x['取得日時']);ddl=datetime.fromisoformat(x['レース日']+'T'+x['締切時刻']+':00+09:00')
        market[rc]={'odds':vals,'lead':(ddl-acq).total_seconds()/60}
all_tickets=[]
for r in rows:
    rc,date,split,result,payout=r[:5];mk=market.get(rc)
    for rank,comb,score in r[9]:
        b={'race_code':rc,'date':date,'rank':rank,'comb':comb,'hit':comb==result,'payout':payout if comb==result else 0}
        if mk and mk['lead']>0:
            o=mk['odds'].get(comb)
            if o>0:
                p=prob(rank,score,r); b.update({'odds':o,'p':p,'ev':p*o})
        all_tickets.append(b)

def metrics(bets):
    n=len(bets);pay=sum(b['payout'] for b in bets);hits=sum(b['hit'] for b in bets)
    grouped=defaultdict(lambda:[0,0])
    for b in bets:
        k=(b['date'],b['race_code']);grouped[k][0]+=100;grouped[k][1]+=b['payout']
    cum=peak=dd=0
    for k in sorted(grouped):
        st,ret=grouped[k];cum+=ret-st;peak=max(peak,cum);dd=max(dd,peak-cum)
    pays=sorted([b['payout'] for b in bets if b['payout']>0],reverse=True)
    def lx(k):
        if n<=k:return None
        return 100*(pay-sum(pays[:k]))/(100*(n-k))
    return {'bets':n,'hits':hits,'hit_rate_pct':100*hits/n if n else None,'roi_pct':100*pay/(100*n) if n else None,'profit':pay-100*n,'maxdd':dd,'l1':lx(1),'l2':lx(2),'l3':lx(3),'largest_payouts':pays[:5]}
def boot(bets,mode='race',B=20000,seed=20260916):
    rng=random.Random(seed)
    blocks=defaultdict(lambda:[0,0])
    for b in bets:
        key=(b['date'],b['race_code']) if mode=='race' else b['date']
        blocks[key][0]+=100;blocks[key][1]+=b['payout']
    vals=list(blocks.values());n=len(vals);rs=[]
    if not vals:return None
    for _ in range(B):
        st=ret=0
        for __ in range(n):
            s,r=vals[rng.randrange(n)];st+=s;ret+=r
        rs.append(100*ret/st if st else 0)
    rs.sort()
    def q(p):return rs[min(len(rs)-1,max(0,int(p*(len(rs)-1))))]
    return {'blocks':n,'p_roi_gt_100':sum(v>100 for v in rs)/B,'p_roi_gt_120':sum(v>120 for v in rs)/B,'p05':q(.05),'median':q(.5),'p95':q(.95)}
def report(bets):
    return {'all':metrics(bets),'aug':metrics([b for b in bets if b['date']<'2026-09-01']),'sep':metrics([b for b in bets if b['date']>='2026-09-01']),'bootstrap_race':boot(bets,'race'),'bootstrap_day':boot(bets,'day')}
def fixed(rs):return [b for b in all_tickets if b['rank'] in rs]
def marketf(rs,lo=None,hi=None,evhi=None):
    z=[]
    for b in all_tickets:
        if b['rank'] not in rs or 'odds' not in b:continue
        if lo is not None and b['odds']<lo:continue
        if hi is not None and b['odds']>hi:continue
        if evhi is not None and b['ev']>evhi:continue
        z.append(b)
    return z
strategies={
 'F_rank10':fixed({10}),
 'F_rank3_10':fixed({3,10}),
 'F_rank5_8_10':fixed({5,8,10}),
 'F_rank4_5_8_10':fixed({4,5,8,10}),
 'M_rank10_odds50_200_evle5':marketf({10},50,200,5),
 'M_rank10_odds40_200_evle5':marketf({10},40,200,5),
 'M_rank10_odds30_200_evle5':marketf({10},30,200,5),
 'M_rank10_odds50_200_evle4':marketf({10},50,200,4),
 'M_rank4_7_10_odds50_200_evle5':marketf({4,7,10},50,200,5),
}
out={k:report(v) for k,v in strategies.items()}
(ROOT/'verify_summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
print(json.dumps(out,ensure_ascii=False,indent=2))
