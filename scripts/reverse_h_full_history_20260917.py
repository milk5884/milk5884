#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, math, os
from collections import Counter, defaultdict
from dataclasses import dataclass
import numpy as np

TICKETS=[f"{a}-{b}-{c}" for a in range(1,7) for b in range(1,7) if b!=a for c in range(1,7) if c not in (a,b)]
TIX={t:i for i,t in enumerate(TICKETS)}
PARTS=np.array([[int(x) for x in t.split('-')] for t in TICKETS],dtype=np.int8)

def sym(j:int)->str:
    a,b,c=[int(x) for x in PARTS[j]]
    h='A' if a==1 else ('B' if a in (2,3) else ('K' if a==4 else 'O'))
    p1='1' if a==1 else ('2' if b==1 else ('3' if c==1 else 'x'))
    p4='1' if a==4 else ('2' if b==4 else ('3' if c==4 else 'x'))
    return h+p1+p4
SYMS=np.array([sym(i) for i in range(120)],dtype=object)
HEAD=np.array([s[0] for s in SYMS],dtype=object)
HEAD1=(PARTS[:,0]==1)

def pattern_keys(order):
    top=order[:20]; sy=SYMS[top]; hd=HEAD[top]
    counts=''.join(f"{int(np.sum(hd==z)):02d}" for z in 'ABKO')
    p1=''.join(str(sum(s[1]==p for s in sy)).zfill(2) for p in '123x')
    p4=''.join(str(sum(s[2]==p for s in sy)).zfill(2) for p in '123x')
    return [
        '|'.join(sy[:8])+'#'+''.join(hd[8:12])+'#'+counts+'#'+p1+'#'+p4,
        '|'.join(sy[:5])+'#'+''.join(hd[5:10])+'#'+counts+'#'+p1+'#'+p4,
        ''.join(hd[:10])+'#'+counts+'#'+p1+'#'+p4,
        ''.join(hd[:6])+'#'+counts+'#'+p1+'#'+p4,
        ''.join(hd[:3])+'#'+counts+'#'+p1+'#'+p4,
        counts+'#'+p1+'#'+p4,
    ]

def flatten_tri(obj):
    out=np.full(120,np.nan,float)
    for a,d2 in (obj.get('trifecta_odds') or {}).items():
        for b,d3 in (d2 or {}).items():
            for c,v in (d3 or {}).items():
                t=f"{int(a)}-{int(b)}-{int(c)}"
                if t in TIX:
                    try: out[TIX[t]]=float(v)
                    except Exception: pass
    return out

def key(r): return (str(r.get('date')),int(r.get('stadium_number')),int(r.get('number')))

def payout_map(r):
    out={}
    for z in (((r.get('payouts') or {}).get('trifecta')) or []):
        try: out[str(z['combination'])]=float(z['amount'])
        except Exception: pass
    return out

@dataclass
class Race:
    dt:str; year:int; month:str; odds:np.ndarray; order:np.ndarray; rank:np.ndarray; q:np.ndarray; true_idx:int; payout:float; keys:list; official_match:bool

def load_races(odds_root,results_root,end_date):
    end8=end_date.replace('-',''); res={}
    for fp in sorted(glob.glob(os.path.join(results_root,'docs/v3/20*/*.json'))):
        b=os.path.basename(fp)[:8]
        if not (b.isdigit() and b<=end8): continue
        try: js=json.load(open(fp,encoding='utf-8'))
        except Exception: continue
        for r in js.get('results',[]):
            pm=payout_map(r)
            if len(pm)==1: res[key(r)]=pm
    races=[]; seen=set(); skipped=Counter()
    for fp in sorted(glob.glob(os.path.join(odds_root,'docs/v3/20*/*.json'))):
        b=os.path.basename(fp)[:8]
        if not (b.isdigit() and b<=end8): continue
        try: js=json.load(open(fp,encoding='utf-8'))
        except Exception: skipped['bad_odds_json']+=1; continue
        for r in js.get('odds',[]):
            k=key(r)
            if k in seen: continue
            seen.add(k); pm=res.get(k)
            if not pm: skipped['missing_result']+=1; continue
            wt,pay=next(iter(pm.items()))
            if wt not in TIX: skipped['bad_result']+=1; continue
            od=flatten_tri(r)
            if np.sum(np.isfinite(od))!=120 or np.any(od<=0): skipped['incomplete_odds']+=1; continue
            order=np.lexsort((np.arange(120),od)); rank=np.empty(120,dtype=np.int16); rank[order]=np.arange(1,121,dtype=np.int16)
            inv=1.0/od; q=inv/inv.sum(); ti=TIX[wt]; dt=str(r.get('date'))
            races.append(Race(dt,int(dt[:4]),dt[:7],od,order,rank,q,ti,float(pay),pattern_keys(order),abs(od[ti]*100.0-float(pay))<=1.01))
    races.sort(key=lambda x:x.dt)
    print('LOAD',json.dumps({'races':len(races),'first':races[0].dt if races else None,'last':races[-1].dt if races else None,'skipped':dict(skipped)},ensure_ascii=False),flush=True)
    return races

VARIANTS=('RH_A','RH_T20','RH_A20')

def event_mask(r,variant):
    if variant=='RH_A': return HEAD1.copy()
    if variant=='RH_T20': return r.rank<=20
    if variant=='RH_A20': return HEAD1 & (r.rank<=20)
    raise ValueError(variant)

def y(r,variant): return int(event_mask(r,variant)[r.true_idx])

class BinaryPattern:
    def __init__(self,min_group,alpha):
        self.min_group=min_group; self.alpha=alpha
        self.tables=[defaultdict(lambda:np.zeros(2,float)) for _ in range(6)]
        self.ns=[Counter() for _ in range(6)]; self.g=np.zeros(2,float)
    def fit(self,races,variant):
        for r in races:
            yy=y(r,variant); self.g[yy]+=1
            for lev,k in enumerate(r.keys): self.tables[lev][k][yy]+=1; self.ns[lev][k]+=1
        return self
    def predict(self,r):
        prior=(self.g+0.5)/(self.g.sum()+1.0); cnt=None; n=0; used=5
        for lev,k in enumerate(r.keys):
            nn=self.ns[lev][k]
            if nn>=self.min_group: cnt=self.tables[lev][k]; n=nn; used=lev; break
        if cnt is None:
            k=r.keys[-1]; cnt=self.tables[-1][k]; n=self.ns[-1][k]
        if n==0: return float(prior[1]),0,used
        post=(cnt+self.alpha*prior)/(n+self.alpha)
        return float(post[1]),int(n),used

def nll(model,races,variant):
    vals=[]
    for r in races:
        p,_,_=model.predict(r); yy=y(r,variant); vals.append(-(math.log(max(min(p,1-1e-12),1e-12)) if yy else math.log(max(min(1-p,1-1e-12),1e-12))))
    return float(np.mean(vals))

def choose(train,valid,variant):
    grid=[]
    for mg in (20,50,100,200):
        for a in (25.0,50.0,100.0,200.0):
            m=BinaryPattern(mg,a).fit(train,variant); grid.append({'min_group':mg,'alpha':a,'nll':nll(m,valid,variant)})
    best=min(grid,key=lambda z:z['nll'])
    return best,grid

def bet_rows(model,races,variant,ev_thr=1.20,topn=3):
    out=[]
    for r in races:
        p,gn,lev=model.predict(r); mask=event_mask(r,variant); idx=np.where(mask)[0]
        if len(idx)==0: continue
        qev=float(r.q[idx].sum())
        if qev<=0: continue
        lift=p/qev
        idx=idx[np.argsort(r.rank[idx])][:topn]
        probs=p*r.q[idx]/qev; evs=probs*r.odds[idx]; ev=float(np.max(evs))
        if lift<=1.0 or ev<ev_thr: continue
        hit=bool(np.any(idx==r.true_idx)); ret=r.payout if hit else 0.0
        out.append({'date':r.dt,'month':r.month,'variant':variant,'lift':lift,'ev':ev,'tickets':[TICKETS[j] for j in idx],'ranks':[int(r.rank[j]) for j in idx],'hit':hit,'return':ret,'payout':r.payout,'true':TICKETS[r.true_idx],'true_rank':int(r.rank[r.true_idx]),'q_event':qev,'p_event':p,'market_selected_prob':float(r.q[idx].sum()),'model_selected_prob':float(probs.sum()),'group_n':gn,'level':lev})
    return out

def metrics(bets):
    if not bets: return {'races':0,'tickets':0,'hits':0,'roi':None,'l1_roi':None,'l2_roi':None,'profit':0.0,'maxdd':0.0}
    stake=float(sum(100*len(b['tickets']) for b in bets)); returns=np.array([b['return'] for b in bets],float); rs=np.array([100*len(b['tickets']) for b in bets],float)
    pnl=returns-rs; eq=np.cumsum(pnl); peak=np.maximum.accumulate(np.r_[0.0,eq])[1:]; dd=peak-eq; wins=np.sort(returns[returns>0])[::-1]; total=float(returns.sum())
    market_exp=float(sum(b['market_selected_prob'] for b in bets)); model_exp=float(sum(b['model_selected_prob'] for b in bets)); hits=int(sum(b['hit'] for b in bets))
    return {'races':len(bets),'tickets':int(sum(len(b['tickets']) for b in bets)),'hits':hits,'stake':stake,'return':total,'profit':total-stake,'roi':total/stake*100.0,'l1_roi':(total-(wins[0] if len(wins) else 0))/stake*100.0,'l2_roi':(total-wins[:2].sum())/stake*100.0,'maxdd':float(dd.max()) if len(dd) else 0.0,'actual_over_market':hits/market_exp if market_exp>0 else None,'actual_over_model':hits/model_exp if model_exp>0 else None,'mean_lift':float(np.mean([b['lift'] for b in bets])),'mean_ev':float(np.mean([b['ev'] for b in bets])),'mean_rank':float(np.mean([x for b in bets for x in b['ranks']]))}

def split_metrics(bets):
    yy=defaultdict(list); mm=defaultdict(list)
    for b in bets: yy[b['date'][:4]].append(b); mm[b['month']].append(b)
    return {'by_year':{k:metrics(v) for k,v in sorted(yy.items())},'by_month':{k:metrics(v) for k,v in sorted(mm.items())}}

def eval_variant(train,valid,t25,t26,variant):
    best,grid=choose(train,valid,variant)
    m25=BinaryPattern(best['min_group'],best['alpha']).fit(train+valid,variant); b25=bet_rows(m25,t25,variant)
    m26=BinaryPattern(best['min_group'],best['alpha']).fit(train+valid+t25,variant); b26=bet_rows(m26,t26,variant)
    bets=b25+b26; sm=split_metrics(bets); met=metrics(bets)
    gate={'roi_ge_120':met['roi'] is not None and met['roi']>=120.0,'l1_gt_100':met['l1_roi'] is not None and met['l1_roi']>100.0,'both_years_profitable':all(sm['by_year'].get(z,{}).get('roi',0)>100.0 for z in ('2025','2026')),'min_100_races':met['races']>=100}
    gate['pass_all']=all(gate.values())
    return {'variant':variant,'selected_hyper_by_2024_nll':best,'validation_grid':grid,'oos_2025_2026':met,**sm,'gate':gate,'sample':bets[:20]}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--odds-root',required=True); ap.add_argument('--results-root',required=True); ap.add_argument('--out',default='REVERSE_H_FULL_HISTORY_RESULT.json'); ap.add_argument('--end-date',default='2026-09-15'); a=ap.parse_args()
    races=load_races(a.odds_root,a.results_root,a.end_date)
    tr=[r for r in races if r.year==2023]; va=[r for r in races if r.year==2024]; t25=[r for r in races if r.year==2025]; t26=[r for r in races if r.year==2026]
    audit={'n':len(races),'match':int(sum(r.official_match for r in races))}; audit['match_rate']=audit['match']/audit['n'] if audit['n'] else 0
    out={'source':{'odds':'lamrongol/BoatraceOdds gh-pages docs/v3','results':'lamrongol/BoatraceResults gh-pages docs/v3'},'window':{'start':races[0].dt,'end':races[-1].dt},'race_count':len(races),'final_odds_audit':audit,'splits':{'train_2023':len(tr),'valid_2024':len(va),'test_2025':len(t25),'test_2026':len(t26)},'definition':{'input':'top20 popularity symbolic pattern only','variants':['RH_A=1-head','RH_T20=winning ticket rank<=20','RH_A20=1-head and rank<=20'],'candidate_tickets':'market top3 within event','stake_per_ticket_yen':100,'buy':'predicted event lift>1 and ticket EV>=1.20','hyperparameter_selection':'2024 binary NLL only; no ROI tuning','oos':'2025 and 2026; expanding refit, frozen hyperparameters','caveat':'hypothesis was motivated by prior full-period H descriptive audit, so this is retrospective structural validation, not pristine untouched hypothesis OOS'},'variants':{}}
    for v in VARIANTS:
        print('EVAL',v,flush=True); out['variants'][v]=eval_variant(tr,va,t25,t26,v)
    out['decision']='REVERSE_H_RESEARCH_CANDIDATE' if any(z['gate']['pass_all'] for z in out['variants'].values()) else 'REVERSE_H_NO_GO_CURRENT_RULE'
    json.dump(out,open(a.out,'w',encoding='utf-8'),ensure_ascii=False,indent=2)
    lines=['# Reverse-H Full Historical Final-Odds Backtest','',f"Window: {out['window']['start']} to {out['window']['end']}  ",f"Usable races: {out['race_count']:,}  ",f"Final-odds audit match: {audit['match_rate']*100:.3f}%",'',f"Decision: **{out['decision']}**",'']
    for v,z in out['variants'].items():
        m=z['oos_2025_2026']; lines += [f"## {v}",f"OOS: {m['races']} races / {m['tickets']} tickets / {m['hits']} hits / ROI **{m['roi']:.2f}%** / L1 **{m['l1_roi']:.2f}%** / L2 {m['l2_roi']:.2f}% / Profit {m['profit']:.0f} / MaxDD {m['maxdd']:.0f}",f"Actual/Market: {m['actual_over_market']:.3f}x / Actual/Model: {m['actual_over_model']:.3f}x",f"Gate: {z['gate']}",'','### By year']
        for yy,ym in z['by_year'].items(): lines.append(f"- {yy}: {ym['races']} races / ROI {ym['roi']:.2f}% / L1 {ym['l1_roi']:.2f}% / Profit {ym['profit']:.0f}")
        lines.append('')
    open(os.path.splitext(a.out)[0]+'.md','w',encoding='utf-8').write('\n'.join(lines)+'\n')
    print(json.dumps({v:out['variants'][v]['oos_2025_2026'] for v in VARIANTS},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
