#!/usr/bin/env python3
from __future__ import annotations
import argparse, glob, json, math, os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

TICKETS = [f"{a}-{b}-{c}" for a in range(1,7) for b in range(1,7) if b!=a for c in range(1,7) if c not in (a,b)]
TIX = {t:i for i,t in enumerate(TICKETS)}
PARTS = np.array([[int(x) for x in t.split('-')] for t in TICKETS], dtype=np.int8)
def head_group(a:int)->int: return 0 if a==1 else (1 if a in (2,3) else (2 if a==4 else 3))
HEAD_GROUP = np.array([head_group(int(a)) for a in PARTS[:,0]], dtype=np.int8)
EVENT_NAMES_4 = ['A_1HEAD','B_23HEAD','K_4HEAD','O_56HEAD']
EVENT_NAMES_5 = ['NORMAL_TOP20','A_OUTSIDE','B_23_OUTSIDE','K_4_OUTSIDE','O_56_OUTSIDE']
def sym(j:int)->str:
    a,b,c = [int(x) for x in PARTS[j]]
    h = 'A' if a==1 else ('B' if a in (2,3) else ('K' if a==4 else 'O'))
    p1 = '1' if a==1 else ('2' if b==1 else ('3' if c==1 else 'x'))
    p4 = '1' if a==4 else ('2' if b==4 else ('3' if c==4 else 'x'))
    return h+p1+p4
SYMS=np.array([sym(i) for i in range(120)],dtype=object); HEAD_CHARS=np.array([s[0] for s in SYMS],dtype=object)
def flatten_trifecta(obj:dict)->np.ndarray:
    out=np.full(120,np.nan,dtype=np.float64); tri=obj.get('trifecta_odds') or {}
    for a,d2 in tri.items():
        for b,d3 in (d2 or {}).items():
            for c,v in (d3 or {}).items():
                t=f"{int(a)}-{int(b)}-{int(c)}"
                if t in TIX:
                    try: out[TIX[t]]=float(v)
                    except Exception: pass
    return out
def race_key(r:dict)->Tuple[str,int,int]: return (str(r.get('date')),int(r.get('stadium_number')),int(r.get('number')))
def payout_map(r:dict)->Dict[str,float]:
    out={}
    for z in (((r.get('payouts') or {}).get('trifecta')) or []):
        try: out[str(z['combination'])]=float(z['amount'])
        except Exception: pass
    return out
def pattern_keys(order:np.ndarray)->List[str]:
    top=order[:20]; sy=SYMS[top]; hd=HEAD_CHARS[top]
    counts=''.join(f"{int(np.sum(hd==z)):02d}" for z in 'ABKO')
    p1=''.join(str(sum(s[1]==p for s in sy)).zfill(2) for p in '123x')
    p4=''.join(str(sum(s[2]==p for s in sy)).zfill(2) for p in '123x')
    return ['|'.join(sy[:8])+'#'+''.join(hd[8:12])+'#'+counts+'#'+p1+'#'+p4,
            '|'.join(sy[:5])+'#'+''.join(hd[5:10])+'#'+counts+'#'+p1+'#'+p4,
            ''.join(hd[:10])+'#'+counts+'#'+p1+'#'+p4,
            ''.join(hd[:6])+'#'+counts+'#'+p1+'#'+p4,
            ''.join(hd[:3])+'#'+counts+'#'+p1+'#'+p4,
            counts+'#'+p1+'#'+p4]
@dataclass
class Race:
    dt:str; year:int; month:str; odds:np.ndarray; order:np.ndarray; rank:np.ndarray; q:np.ndarray; q4:np.ndarray; q5:np.ndarray; true_idx:int; payout:float; keys:List[str]; y4:int; y5:int; official_match:bool
def load_races(odds_root:str,results_root:str,end_date:str)->List[Race]:
    res_by_key={}
    for fp in sorted(glob.glob(os.path.join(results_root,'docs/v3/20*/*.json'))):
        base=os.path.basename(fp)[:8]
        if not (base.isdigit() and base<=end_date.replace('-','')): continue
        try:
            with open(fp,'r',encoding='utf-8') as f: js=json.load(f)
        except Exception: continue
        for r in js.get('results',[]):
            pm=payout_map(r)
            if len(pm)==1: res_by_key[race_key(r)]=pm
    races=[]; seen=set(); skipped=Counter()
    for fp in sorted(glob.glob(os.path.join(odds_root,'docs/v3/20*/*.json'))):
        base=os.path.basename(fp)[:8]
        if not (base.isdigit() and base<=end_date.replace('-','')): continue
        try:
            with open(fp,'r',encoding='utf-8') as f: js=json.load(f)
        except Exception: skipped['bad_odds_json']+=1; continue
        for r in js.get('odds',[]):
            k=race_key(r)
            if k in seen: continue
            seen.add(k); pm=res_by_key.get(k)
            if not pm: skipped['missing_result']+=1; continue
            win_t,payout=next(iter(pm.items()))
            if win_t not in TIX: skipped['bad_result_ticket']+=1; continue
            od=flatten_trifecta(r)
            if np.sum(np.isfinite(od))!=120 or np.any(od<=0): skipped['incomplete_odds']+=1; continue
            ti=TIX[win_t]; order=np.lexsort((np.arange(120),od)); rank=np.empty(120,dtype=np.int16); rank[order]=np.arange(1,121,dtype=np.int16)
            inv=1.0/od; q=inv/inv.sum(); q4=np.array([q[HEAD_GROUP==g].sum() for g in range(4)],float); q5=np.zeros(5,float); q5[0]=q[rank<=20].sum()
            for g in range(4): q5[g+1]=q[(rank>20)&(HEAD_GROUP==g)].sum()
            y4=int(HEAD_GROUP[ti]); y5=0 if rank[ti]<=20 else y4+1; dt=str(r.get('date')); om=abs(od[ti]*100.0-float(payout))<=1.01
            races.append(Race(dt,int(dt[:4]),dt[:7],od,order,rank,q,q4,q5,ti,float(payout),pattern_keys(order),y4,y5,om))
    races.sort(key=lambda x:x.dt)
    print('LOAD',json.dumps({'races':len(races),'first':races[0].dt if races else None,'last':races[-1].dt if races else None,'skipped':skipped},ensure_ascii=False),flush=True); return races
class PatternModel:
    def __init__(self,n_events:int,min_group:int,alpha:float):
        self.n_events=n_events; self.min_group=min_group; self.alpha=alpha; self.tables=[defaultdict(lambda:np.zeros(n_events,dtype=np.float64)) for _ in range(6)]; self.ns=[Counter() for _ in range(6)]; self.global_counts=np.zeros(n_events,dtype=np.float64)
    def fit(self,races:List[Race],target:str):
        for r in races:
            y=r.y4 if target=='y4' else r.y5; self.global_counts[y]+=1
            for lev,k in enumerate(r.keys): self.tables[lev][k][y]+=1; self.ns[lev][k]+=1
        return self
    def predict_one(self,r:Race)->Tuple[np.ndarray,int,int]:
        prior=(self.global_counts+0.5)/(self.global_counts.sum()+0.5*self.n_events); cnt=None; n=0; used=5
        for lev,k in enumerate(r.keys):
            nn=self.ns[lev][k]
            if nn>=self.min_group: cnt=self.tables[lev][k]; n=nn; used=lev; break
        if cnt is None: k=r.keys[-1]; cnt=self.tables[-1][k]; n=self.ns[-1][k]
        if n==0: return prior.copy(),0,used
        return (cnt+self.alpha*prior)/(n+self.alpha),int(n),used
def logloss(ps,ys): return float(-np.log(np.clip(ps[np.arange(len(ys)),ys],1e-12,1)).mean())
def brier(ps,ys):
    oh=np.zeros_like(ps); oh[np.arange(len(ys)),ys]=1; return float(np.mean(np.sum((ps-oh)**2,axis=1)))
def fit_eval(train,valid,target,min_group,alpha):
    ne=4 if target=='y4' else 5; m=PatternModel(ne,min_group,alpha).fit(train,target); ps=[];ys=[];ns=[];ls=[]
    for r in valid:
        p,n,l=m.predict_one(r); ps.append(p); ys.append(r.y4 if target=='y4' else r.y5); ns.append(n); ls.append(l)
    ps=np.array(ps);ys=np.array(ys); return m,{'min_group':min_group,'alpha':alpha,'nll':logloss(ps,ys),'brier':brier(ps,ys),'mean_group_n':float(np.mean(ns)),'level_counts':dict(Counter(ls))}
def choose_hyper(train,valid,target):
    rows=[]
    for mg in (20,50,100,200):
        for al in (25.0,50.0,100.0,200.0):
            _,met=fit_eval(train,valid,target,mg,al); rows.append(met)
    return min(rows,key=lambda z:(z['nll'],z['brier'])),rows
def bankroll_metrics(bets):
    if not bets: return {'races':0,'tickets':0,'hits':0,'roi':None,'profit':0.0,'l1_roi':None,'l2_roi':None,'maxdd':0.0}
    stake=sum(100*len(x['tickets']) for x in bets); returns=np.array([x['return'] for x in bets],float); race_stakes=np.array([100*len(x['tickets']) for x in bets],float); pnl=returns-race_stakes; eq=np.cumsum(pnl); peak=np.maximum.accumulate(np.r_[0.0,eq]); dd=peak[1:]-eq; wins=np.sort(returns[returns>0])[::-1]; total=float(returns.sum())
    return {'races':len(bets),'tickets':int(sum(len(x['tickets']) for x in bets)),'hits':int(sum(x['hit'] for x in bets)),'stake':float(stake),'return':total,'profit':float(total-stake),'roi':float(total/stake*100),'l1_roi':float((total-(wins[0] if len(wins)>0 else 0))/stake*100),'l2_roi':float((total-wins[:2].sum())/stake*100),'maxdd':float(dd.max()) if len(dd) else 0.0,'mean_pred_lift':float(np.mean([x['lift'] for x in bets])),'mean_ev':float(np.mean([x['ev'] for x in bets])),'mean_selected_rank':float(np.mean([rr for x in bets for rr in x['ranks']]))}
def make_bets(model,races,variant,ev_thr=1.20,topn=3,pattern_only=False):
    out=[]
    for r in races:
        p,gn,lev=model.predict_one(r); qev=r.q4 if variant=='head4' else r.q5
        if variant=='head4': events=range(1,4); cand_event=lambda e:np.where(HEAD_GROUP==e)[0]
        else: events=range(1,5); cand_event=lambda e:np.where((r.rank>20)&(HEAD_GROUP==(e-1)))[0]
        lift,e=max((float(p[e]/qev[e]) if qev[e]>0 else -1.0,e) for e in events); cand=cand_event(e); cand=cand[np.argsort(r.rank[cand])][:topn]
        if len(cand)==0 or qev[e]<=0: continue
        probs=np.array([p[e]*r.q[j]/qev[e] for j in cand],float); evs=probs*r.odds[cand]; ev=float(np.max(evs))
        if (not pattern_only) and ev<ev_thr: continue
        hit=bool(np.any(cand==r.true_idx)); ret=r.payout if hit else 0.0
        out.append({'date':r.dt,'month':r.month,'event':int(e),'event_name':(EVENT_NAMES_4 if variant=='head4' else EVENT_NAMES_5)[e],'lift':lift,'ev':ev,'tickets':[TICKETS[j] for j in cand],'ranks':[int(r.rank[j]) for j in cand],'hit':hit,'return':ret,'payout':r.payout,'true':TICKETS[r.true_idx],'true_rank':int(r.rank[r.true_idx]),'group_n':gn,'level':lev})
    return out
def slices(bets):
    years=defaultdict(list); months=defaultdict(list); events=defaultdict(list)
    for b in bets: years[b['date'][:4]].append(b); months[b['month']].append(b); events[b['event_name']].append(b)
    return {'by_year':{k:bankroll_metrics(v) for k,v in sorted(years.items())},'by_month':{k:bankroll_metrics(v) for k,v in sorted(months.items())},'by_event':{k:bankroll_metrics(v) for k,v in sorted(events.items())}}
def eval_variant(train,valid,test2025,test2026,variant):
    target='y4' if variant=='head4' else 'y5'; best,grid=choose_hyper(train,valid,target); ne=4 if target=='y4' else 5
    m25=PatternModel(ne,best['min_group'],best['alpha']).fit(train+valid,target); b25=make_bets(m25,test2025,variant); p25=make_bets(m25,test2025,variant,pattern_only=True)
    m26=PatternModel(ne,best['min_group'],best['alpha']).fit(train+valid+test2025,target); b26=make_bets(m26,test2026,variant); p26=make_bets(m26,test2026,variant,pattern_only=True)
    bets=b25+b26; po=p25+p26; sl=slices(bets); psl=slices(po)
    return {'variant':variant,'target':target,'selected_hyper_by_2024_nll':best,'validation_grid':grid,'ev120_oos_2025_2026':bankroll_metrics(bets),**sl,'pattern_only_oos_2025_2026':bankroll_metrics(po),'pattern_only_by_year':psl['by_year'],'bets_sample':bets[:20]}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--odds-root',required=True); ap.add_argument('--results-root',required=True); ap.add_argument('--out',default='H_FULL_HISTORY_RESULT.json'); ap.add_argument('--end-date',default='2026-09-15'); a=ap.parse_args(); races=load_races(a.odds_root,a.results_root,a.end_date)
    if not races: raise SystemExit('no races')
    audit={'n':len(races),'winning_odds_x100_exact_or_1yen':int(sum(r.official_match for r in races))}; audit['match_rate']=audit['winning_odds_x100_exact_or_1yen']/audit['n']
    split={'train_2023':[r for r in races if r.year==2023],'valid_2024':[r for r in races if r.year==2024],'test_2025':[r for r in races if r.year==2025],'test_2026':[r for r in races if r.year==2026]}
    summary={'source':{'odds':'lamrongol/BoatraceOdds gh-pages docs/v3','results':'lamrongol/BoatraceResults gh-pages docs/v3'},'window':{'start':races[0].dt,'end':races[-1].dt},'race_count':len(races),'final_odds_audit':audit,'splits':{k:len(v) for k,v in split.items()},'definition':{'pattern_input':'popularity order top20 only; symbolic ticket = head family A/B/K/O + position of boat1 + position of boat4; hierarchical pattern backoff','no_racer_features':True,'candidate_tickets':3,'stake_per_ticket_yen':100,'ev_threshold':1.20,'hyperparameter_selection':'2024 event NLL only; no ROI tuning','oos':'2025 and 2026 separately; expanding refit with frozen hyperparameters'},'variants':{}}
    for variant in ('head4','outside5'):
        print('EVAL',variant,flush=True); summary['variants'][variant]=eval_variant(split['train_2023'],split['valid_2024'],split['test_2025'],split['test_2026'],variant)
    desc={}
    for target,nm in [('y4','head4'),('y5','outside5')]:
        n=4 if target=='y4' else 5; obs=np.bincount(np.array([r.y4 if target=='y4' else r.y5 for r in races]),minlength=n)/len(races); mq=np.mean(np.stack([r.q4 if target=='y4' else r.q5 for r in races]),axis=0); names=EVENT_NAMES_4 if target=='y4' else EVENT_NAMES_5; desc[nm]={names[i]:{'actual':float(obs[i]),'market':float(mq[i]),'actual_over_market':float(obs[i]/mq[i])} for i in range(n)}
    summary['descriptive_event_rates']=desc
    with open(a.out,'w',encoding='utf-8') as f: json.dump(summary,f,ensure_ascii=False,indent=2)
    md=['# H Full Historical Final-Odds Backtest','',f"Window: {summary['window']['start']} to {summary['window']['end']}  ",f"Usable races: {len(races):,}  ",f"Final-odds audit match: {audit['match_rate']*100:.3f}%",'']
    for vn,v in summary['variants'].items():
        m=v['ev120_oos_2025_2026']; md += [f'## {vn}',f"OOS EV>=1.20: {m['races']} races / {m['tickets']} tickets / {m['hits']} hits / ROI {m['roi']:.2f}% / L1 {m['l1_roi']:.2f}% / L2 {m['l2_roi']:.2f}% / Profit {m['profit']:.0f} / MaxDD {m['maxdd']:.0f}",f"Pattern-only: ROI {v['pattern_only_oos_2025_2026']['roi']:.2f}%",'', '### By year']
        for y,z in v['by_year'].items(): md.append(f"- {y}: {z['races']} races / ROI {z['roi']:.2f}% / L1 {z['l1_roi']:.2f}% / Profit {z['profit']:.0f}")
        md += ['', '### By event']
        for e,z in v['by_event'].items(): md.append(f"- {e}: {z['races']} races / ROI {z['roi']:.2f}% / L1 {z['l1_roi']:.2f}% / Profit {z['profit']:.0f}")
        md.append('')
    with open(os.path.splitext(a.out)[0]+'.md','w',encoding='utf-8') as f: f.write('\n'.join(md)+'\n')
    print(json.dumps({'window':summary['window'],'race_count':len(races),'audit':audit,'headline':{k:v['ev120_oos_2025_2026'] for k,v in summary['variants'].items()}},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__': main()
