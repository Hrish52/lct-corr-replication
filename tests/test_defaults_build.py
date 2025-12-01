import pandas as pd
from scripts.make_defaults import choose_B_for_alpha

def test_choose_B_picks_smallest_B_under_alpha():
    fdp = pd.DataFrame({
        "B":[50,100,200,50,100,200],
        "fdp":[0.061,0.049,0.047,0.060,0.051,0.048],
    })
    rt  = pd.DataFrame({"B":[50,100,200],"sec":[1.0,1.5,2.0]})
    B_star, agg = choose_B_for_alpha(fdp, rt, alpha=0.05, fdr_tol=0.0)
    assert B_star == 100

def test_choose_B_runtime_tie_break():
    fdp = pd.DataFrame({
        "B":[50,100,50,100],
        "fdp":[0.048,0.048,0.049,0.049]
    })
    rt  = pd.DataFrame({"B":[50,100],"sec":[2.0,1.0]})
    B_star, _ = choose_B_for_alpha(fdp, rt, alpha=0.05, fdr_tol=0.0)
    # both qualify; picks lower runtime -> B=100
    assert B_star == 100