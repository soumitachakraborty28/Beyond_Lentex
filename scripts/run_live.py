import argparse,json
from pathlib import Path
from src.core import Experiment

def load(path):
    rows=[]
    with open(path,encoding="utf-8") as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows

def demo():
    return [
      {"query_id":"q1","query":"contract breach duty disclosure material information",
       "gold_entities":["contract breach","duty disclosure"],
       "documents":[
        {"doc_id":"d1","text":"Contract breach may arise when a party fails to disclose material information.","relevant":True},
        {"doc_id":"d2","text":"A duty of disclosure can apply to material information in contracts.","relevant":True},
        {"doc_id":"d3","text":"Unrelated procedural rules for court filing.","relevant":False}]},
      {"query_id":"q2","query":"drug treatment for chronic kidney disease biomarkers",
       "gold_entities":["drug treatment","chronic kidney disease","biomarkers"],
       "documents":[
        {"doc_id":"d4","text":"Drug treatment options for chronic kidney disease include several therapies.","relevant":True},
        {"doc_id":"d5","text":"Biomarkers are used to monitor chronic kidney disease progression.","relevant":True},
        {"doc_id":"d6","text":"Unrelated cancer screening information.","relevant":False}]},
      {"query_id":"q3","query":"legal jurisdiction and contractual obligation",
       "gold_entities":["legal jurisdiction","contractual obligation"],
       "documents":[
        {"doc_id":"d7","text":"Legal jurisdiction can determine how contractual obligations are interpreted.","relevant":True},
        {"doc_id":"d8","text":"Unrelated property registration rules.","relevant":False}]}
    ]

ap=argparse.ArgumentParser()
ap.add_argument("--dataset")
ap.add_argument("--model",default="sentence-transformers/all-MiniLM-L6-v2")
ap.add_argument("--outdir",default="outputs/live")
ap.add_argument("--demo",action="store_true")
args=ap.parse_args()

rows=demo() if args.demo else load(args.dataset)
summary,test=Experiment(rows,args.model).run(args.outdir)

print("\\n=== RETRIEVAL SUMMARY ===")
print(summary.to_string(index=False,float_format=lambda x:f"{x:.3f}"))
print("\\n=== ADAPTIVE vs TOP-1 BOOTSTRAP ===")
print(json.dumps(test,indent=2))
print(f"\\nFull tables saved to: {Path(args.outdir).resolve()}")
