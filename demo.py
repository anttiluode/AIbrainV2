"""Run preregistered AIbrainV2 operator versus fast-weight experiment."""
import argparse
import json
from pathlib import Path
import numpy as np
from operator_memory import benchmark


def figure(metrics,outfile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    loads=[8,16,32,64]
    modes=["operator","fast_weight","indexed"]
    names={"operator":"Graph operator","fast_weight":"Direct fast weights",
           "indexed":"Packed indexed lookup"}
    data=metrics["summary"]
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout="constrained")
    fig.suptitle("AIbrainV2 | same writer + same matrix, different readout",fontsize=16)

    for mode in modes:
        for query,style in (("exact","-"),("one_swap","--")):
            y=[data[str(load)][mode]["before_"+query]["mean"] for load in loads]
            axes[0,0].plot(loads,y,marker="o",ls=style,label=names[mode]+" "+query)
    axes[0,0].set(xlabel="stored random associations",ylabel="recall",
                  ylim=(-.04,1.04),title="A. Recall BEFORE rewriting")
    axes[0,0].legend(fontsize=7,ncol=2)

    for mode in modes:
        y=[data[str(load)][mode]["untouched_one_swap"]["mean"] for load in loads]
        axes[0,1].plot(loads,y,marker="o",label=names[mode])
    axes[0,1].set(xlabel="stored random associations",ylabel="noisy retention",
                  ylim=(-.04,1.04),title="B. Untouched associations AFTER 25% rewrites")
    axes[0,1].legend(fontsize=8)

    for mode in modes:
        y=[data[str(load)][mode]["rewritten_one_swap"]["mean"] for load in loads]
        axes[1,0].plot(loads,y,marker="o",label=names[mode])
    axes[1,0].set(xlabel="stored random associations",ylabel="new label recall",
                  ylim=(-.04,1.04),title="C. Modified associations AFTER rewriting")
    axes[1,0].legend(fontsize=8)

    for mode in modes:
        y=[data[str(load)][mode]["storage_bytes"] for load in loads]
        axes[1,1].plot(loads,y,marker="o",label=names[mode])
    axes[1,1].set(xlabel="stored random associations",ylabel="stored data (bytes)",
                  title="D. Memory footprint (no model program code)")
    axes[1,1].legend(fontsize=8)
    fig.savefig(outfile,dpi=160)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path("runs/operator_gate1"))
    p.add_argument("--quick",action="store_true",help="run two-seed smoke test")
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    result=benchmark(seeds=(0,1) if args.quick else tuple(range(12)))
    (args.output/"metrics.json").write_text(json.dumps(result,indent=2)+"\n")
    figure(result,args.output/"operator_gate1.png")
    print(json.dumps(result["decision"],indent=2))
    for load,ms in result["summary"].items():
        print(f"load {load:>2}: ",end="")
        print(" ".join(
            f"{k}: {v['untouched_one_swap']['mean']:.3f} retained, "
            f"{v['rewritten_one_swap']['mean']:.3f} rewritten"
            for k,v in ms.items()))
    print("Saved",args.output)


if __name__=="__main__":
    main()
