import csv, itertools, math, random
def load(f): return {r["endpoint"]:(r["family"],r["model"]) for r in csv.DictReader(open(f))}
def mcc(pred, truth):
    tp=fp=fn=tn=0
    for x,y in itertools.combinations(sorted(truth),2):
        if truth[x][0]!=truth[y][0]: continue
        t=truth[x][1]==truth[y][1]; p=pred[x][1]==pred[y][1] and pred[x][0]==pred[y][0]
        tp+=t and p; fp+=p and not t; fn+=t and not p; tn+=(not t) and (not p)
    d=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn)) or 1
    return (tp*tn-fp*fn)/d
FAM={"A":"05 09 10 11 13 15 16 19 27 29 31","B":"03 06 07 08 20 21 22 26 28 30 32","C":"01 02 04 12 14 17 18 23 24 25 33"}
DIS={"A":"29 16 31 10","B":"08 07 20 30 06","C":"02 23 33 04"}
CORE={f:[x for x in FAM[f].split() if x not in DIS[f].split()] for f in FAM}
v1,v2=load("submission_v1.csv"),load("submission_v2.csv")
def truth_H1(plain):  # ref-like: S = disguises+wm+one plain, others singletons
    t={}
    for f in FAM:
        for x in FAM[f].split():
            t["ep-"+x]=(f, f+"S" if (x in DIS[f].split() or x==plain[f]) else f+x)
    return t
def v3(plain): return truth_H1(plain)
res1,res2,res3=[],[],[]
for pa in CORE["A"]:
  for pb in CORE["B"]:
    for pc in CORE["C"]:
      T=truth_H1({"A":pa,"B":pb,"C":pc}); res1.append(mcc(v1,T)); res2.append(mcc(v2,T))
print("H1 (schema ref, plain ignoto): MCC_model v1 %.2f-%.2f (media %.2f) | v2 %.2f-%.2f (media %.2f)"%(min(res1),max(res1),sum(res1)/len(res1),min(res2),max(res2),sum(res2)/len(res2)))
print("  -> score atteso v2 = 0.1+0.9*MCC = %.2f-%.2f"%(0.1+0.9*min(res2),0.1+0.9*max(res2)))
# v3 when plain guess right/wrong
T=truth_H1({"A":"09","B":"21","C":"01"})
for g,lab in [({"A":"09","B":"21","C":"01"},"plain giusto in 3/3"),({"A":"11","B":"21","C":"01"},"2/3"),({"A":"11","B":"26","C":"01"},"1/3"),({"A":"11","B":"26","C":"12"},"0/3")]:
    print("  v3 sotto H1 con %s: MCC %.2f"%(lab,mcc(v3(g),T)))
# v3 if truth instead = v2 structure (big S)
print("v3 se invece la verità fosse v2: MCC %.2f"%mcc(v3({"A":"09","B":"21","C":"01"}),v2))
# Alternative truth H2: per family 11 = S(5) + 3 variants each with 2 endpoints (variant + its disguised copy)?? random: core split in pairs
random.seed(0); r=[]
for _ in range(300):
    t={}
    for f in FAM:
        c=CORE[f][:]; random.shuffle(c); pl=c[0]
        for x in FAM[f].split(): t["ep-"+x]=(f,f+"S" if x in DIS[f].split() or x==pl else None)
        rest=c[1:]
        for i,x in enumerate(rest): t["ep-"+x]=(f,f+"V%d"%(i//2))
    r.append((mcc(v2,t),mcc(v3({"A":"09","B":"21","C":"01"}),t)))
print("H2 (varianti a coppie, casuali): v2 media %.2f | v3 media %.2f"%(sum(a for a,b in r)/len(r),sum(b for a,b in r)/len(r)))
print("\n--- scelta ottimale sotto H1, plain uniforme tra i core indistinguibili dal watermark")
UND={"A":"09 11 13 19".split(),"B":"21 26 32".split(),"C":"01 12 17 18 24 25".split()}
def sub(choice):  # choice[f] = list of cores put in S
    t={}
    for f in FAM:
        for x in FAM[f].split(): t["ep-"+x]=(f,f+"S" if x in DIS[f].split() or x in choice[f] else f+x)
    return t
opts={"uno (il piu simile)":{f:[UND[f][0]] for f in FAM},"nessuno":{f:[] for f in FAM},"tutti gli indistinguibili":UND}
for name,ch in opts.items():
    vals=[mcc(sub(ch),truth_H1({"A":a,"B":b,"C":c})) for a in UND["A"] for b in UND["B"] for c in UND["C"]]
    print("  core in S = %-26s MCC atteso %.2f (min %.2f max %.2f)"%(name,sum(vals)/len(vals),min(vals),max(vals)))
