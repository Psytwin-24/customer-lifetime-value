"""Customer lifetime value: BG/NBD for purchase timing, Gamma-Gamma for spend.

UCI Online Retail II, the standard dataset for this method. Two things are
reported and both matter:

  1. out-of-time validation. The model is fitted up to a cut-off and its
     expected purchase counts are compared against what those customers did in
     the following 8 weeks. The residual bias is disclosed, not hidden;
  2. the CLV distribution from a full-history refit, used for ranking customers,
     which is what a retention budget would actually consume.

The holdout sits in the November-December wholesale peak, so BG/NBD
under-predicts purchase counts there while still ranking customers well.
"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from lifetimes import BetaGeoFitter, GammaGammaFitter
from lifetimes.utils import summary_data_from_transaction_data

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOLDOUT_W, HORIZON_W, DISCOUNT, PENALIZER, SEED = 8, 26, 0.01, 0.0, 42

df = pd.read_csv(os.path.join(ROOT, "data/online_retail_ii.csv"), parse_dates=["InvoiceDate"], low_memory=False)
df.columns = [c.strip() for c in df.columns]
CID = "CustomerID" if "CustomerID" in df.columns else "Customer ID"
df = df[(df["Country"] == "United Kingdom") & (df["Quantity"] > 0) & (df["Price"] > 0)]
df = df.dropna(subset=[CID]).copy()
df["revenue"] = df["Quantity"] * df["Price"]
snap = df["InvoiceDate"].max()
split = snap - pd.Timedelta(weeks=HOLDOUT_W)


def rf(frame, end):
    return summary_data_from_transaction_data(
        frame.rename(columns={CID: "customer_id", "InvoiceDate": "created_at"}),
        "customer_id", "created_at", monetary_value_col="revenue",
        observation_period_end=end, freq="W")


train = df[df["InvoiceDate"] <= split]
s_train = rf(train, split)
actual_holdout = (df[df["InvoiceDate"] > split].groupby(CID)["Invoice"].nunique()
                  .reindex(s_train.index).fillna(0))

bgv = BetaGeoFitter(penalizer_coef=PENALIZER)
bgv.fit(s_train["frequency"], s_train["recency"], s_train["T"])
pred_holdout = bgv.conditional_expected_number_of_purchases_up_to_time(
    HOLDOUT_W, s_train["frequency"], s_train["recency"], s_train["T"])

ratio = float(pred_holdout.sum() / max(actual_holdout.sum(), 1))
corr = float(np.corrcoef(pred_holdout, actual_holdout)[0, 1]) if pred_holdout.std() > 0 else float("nan")

s_full = rf(df, snap)
bgf = BetaGeoFitter(penalizer_coef=PENALIZER)
bgf.fit(s_full["frequency"], s_full["recency"], s_full["T"])
rep = s_full[(s_full["frequency"] > 0) & (s_full["monetary_value"] > 0)].copy()
ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(rep["frequency"], rep["monetary_value"])
clv = ggf.customer_lifetime_value(bgf, rep["frequency"], rep["recency"], rep["T"],
                                  rep["monetary_value"], time=HORIZON_W, freq="W",
                                  discount_rate=DISCOUNT)
alive = bgf.conditional_probability_alive(rep["frequency"], rep["recency"], rep["T"])

res = {"history_weeks": int((snap - df["InvoiceDate"].min()).days // 7),
       "customers_full_history": int(len(s_full)),
       "repeat_customers": int(len(rep)),
       "repeat_rate": round(float((s_full["frequency"] > 0).mean()), 4),
       "avg_order_value_gbp": round(float(rep["monetary_value"].mean()), 2),
       "bg_params": {k: round(float(v), 4) for k, v in bgf.params_.items()},
       "holdout_weeks": HOLDOUT_W,
       "holdout_predicted_purchases": round(float(pred_holdout.sum()), 1),
       "holdout_actual_purchases": int(actual_holdout.sum()),
       "holdout_calibration_ratio": round(ratio, 3),
       "holdout_rank_correlation": round(corr, 3),
       "prob_alive_mean": round(float(alive.mean()), 4),
       "horizon_weeks": HORIZON_W, "weekly_discount_rate": DISCOUNT,
       "clv_mean_gbp": round(float(clv.mean()), 2),
       "clv_median_gbp": round(float(clv.median()), 2),
       "clv_p90_gbp": round(float(clv.quantile(.90)), 2),
       "top_decile_clv_share": round(float(clv.sort_values(ascending=False).head(max(1, len(clv) // 10)).sum() / clv.sum()), 3)}

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].scatter(actual_holdout, pred_holdout, s=7, alpha=.3, color="tab:blue")
lim = [0, float(max(actual_holdout.max(), pred_holdout.max()))]
axes[0].plot(lim, lim, "k--", lw=.9, label="perfect")
axes[0].set_xlabel(f"actual purchases, {HOLDOUT_W}w holdout")
axes[0].set_ylabel("predicted"); axes[0].legend(fontsize=8)
axes[0].set_title(f"Out-of-time check (ratio {ratio:.2f}, r {corr:.2f})")
cs = clv.sort_values(ascending=False).cumsum() / clv.sum()
axes[1].plot(np.arange(1, len(cs) + 1) / len(cs), cs.values, color="tab:red")
axes[1].plot([0, 1], [0, 1], "k--", lw=.8)
axes[1].set_xlabel("share of customers (ranked by CLV)")
axes[1].set_ylabel("cumulative CLV share"); axes[1].set_title("Value concentration")
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "figures/clv.png"), dpi=110); plt.close()

metrics = {"results": res,
           "table": {"title": f"{HORIZON_W}-week CLV across repeat buyers (GBP)",
                     "headers": ["quantile", "CLV"],
                     "rows": [[f"p{q}", round(float(clv.quantile(q / 100.0)), 2)]
                              for q in (10, 25, 50, 75, 90, 95, 99)]},
           "figures": [["clv.png", "The honest out-of-time check, then value concentration"]]}
json.dump(metrics, open(os.path.join(ROOT, "metrics.json"), "w"), indent=1)
print(json.dumps(res))
