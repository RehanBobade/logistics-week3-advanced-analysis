"""
SCMS Delivery History - Logistics Performance Analysis
Cleans the Kaggle SCMS dataset, computes KPIs, runs statistical tests,
fits two explanatory models and writes the figures used in the report.
"""
import json, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
from scipy import stats


# ---------- data loading and cleaning ----------
import re
RAW = 'Raw_Data.csv'   # path to the Kaggle SCMS file
def load_clean():
    df=pd.read_csv(RAW)
    df.columns=[c.replace('Ã¯Â»Â¿','').strip() for c in df.columns]
    df=df.drop(columns=['Load_Date','Source_System'])
    df['Country']=df['Country'].str.replace(r"^C.{1,4}te d'Ivoire$","Côte d'Ivoire",regex=True)
    df['Manufacturing Site']=df['Manufacturing Site'].str.replace(r"Gr.{2,6}nland","Grønland",regex=True)
    df['Shipment Mode']=df['Shipment Mode'].fillna('Not recorded')
    for src,dst in [('Scheduled Delivery Date','sched'),('Delivered to Client Date','deliv'),('Delivery Recorded Date','recorded')]:
        df[dst]=pd.to_datetime(df[src],format='%d-%b-%y')
    df['po_date']=pd.to_datetime(df['PO Sent to Vendor Date'],format='%m/%d/%y',errors='coerce')
    df['pq_date']=pd.to_datetime(df['PQ First Sent to Client Date'],format='%m/%d/%y',errors='coerce')
    # numeric
    df['weight_kg']=pd.to_numeric(df['Weight (Kilograms)'],errors='coerce')
    df['freight_usd']=pd.to_numeric(df['Freight Cost (USD)'],errors='coerce')
    df['insurance_usd']=pd.to_numeric(df['Line Item Insurance (USD)'].astype(str).str.replace(',',''),errors='coerce')
    def fstat(v):
        v=str(v)
        if v.startswith('See '): return 'Referenced (shared shipment)'
        if v.startswith('Freight Included'): return 'Included in commodity cost'
        if v.startswith('Invoiced'): return 'Invoiced separately'
        return 'Direct'
    df['freight_status']=df['Freight Cost (USD)'].map(fstat)
    df=df.rename(columns={'Line Item Value':'line_value','Line Item Quantity':'quantity','Unit Price':'unit_price','Pack Price':'pack_price','Product Group':'product_group','Fulfill Via':'fulfill_via','Vendor INCO Term':'inco','Shipment Mode':'mode','Country':'country','Vendor':'vendor','Managed By':'managed_by','Sub Classification':'sub_class','Dosage Form':'dosage_form','First Line Designation':'first_line'})
    # derived
    df['delay_days']=(df['deliv']-df['sched']).dt.days
    df['outcome']=np.select([df.delay_days<0,df.delay_days==0],['Early','On schedule'],'Late')
    df['on_time']=(df.delay_days<=0).astype(int)
    df['late']=(df.delay_days>0).astype(int)
    df['late_gt7']=(df.delay_days>7).astype(int)
    df['late_gt14']=(df.delay_days>14).astype(int)
    df['late_gt30']=(df.delay_days>30).astype(int)
    df['year']=df['deliv'].dt.year
    df['record_lag']=(df['recorded']-df['deliv']).dt.days
    lt=(df['deliv']-df['po_date']).dt.days
    df['po_to_delivery_days']=lt.where(lt>0)
    df['po_date_invalid']=(lt<=0)
    ok=df.weight_kg.gt(0)&df.freight_usd.gt(0)
    df['freight_per_kg']=(df.freight_usd/df.weight_kg).where(ok)
    df['freight_pct_value']=(df.freight_usd/df.line_value*100).where(df.freight_usd.gt(0)&df.line_value.gt(0))
    return df

OUT = "figs"
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------
# 1. LOAD + CLEAN
# ---------------------------------------------------------------
df = load_clean()
S = {}
S["rows"], S["cols_raw"] = int(len(df)), 35
S["countries"] = int(df.country.nunique())
S["vendors"] = int(df.vendor.nunique())
S["date_min"] = str(df.deliv.min().date())
S["date_max"] = str(df.deliv.max().date())
S["total_value"] = float(df.line_value.sum())
S["total_qty"] = int(df.quantity.sum())
S["total_weight_direct"] = float(df.weight_kg.sum())
S["freight_direct_total"] = float(df.freight_usd.sum())
S["freight_status"] = df.freight_status.value_counts().to_dict()
S["mode_missing"] = int((df["mode"] == "Not recorded").sum())
S["po_valid"] = int(df.po_to_delivery_days.notna().sum())
S["po_invalid"] = int(df.po_date_invalid.sum()) - int(df.po_date.isna().sum() * 0)  # negative / zero lead times
S["po_invalid"] = int(((df.deliv - df.po_date).dt.days <= 0).sum())
S["po_missing"] = int(df.po_date.isna().sum())
S["zero_value"] = int((df.line_value == 0).sum())
S["zero_weight"] = int((df.weight_kg == 0).sum())
S["dup_excl_id"] = int(df.duplicated(subset=[c for c in df.columns if c != "ID"]).sum())
S["ins_rows"] = int(df.insurance_usd.notna().sum())
S["extreme_early"] = int((df.delay_days < -30).sum())
S["record_lag_gt0_pct"] = round(float((df.record_lag > 0).mean() * 100), 1)

# KPIs
S["on_time_pct"] = round(float(df.on_time.mean() * 100), 1)
S["exact_pct"] = round(float((df.delay_days == 0).mean() * 100), 1)
S["early_pct"] = round(float((df.delay_days < 0).mean() * 100), 1)
S["late_pct"] = round(float(df.late.mean() * 100), 1)
S["late7_pct"] = round(float(df.late_gt7.mean() * 100), 1)
S["late14_pct"] = round(float(df.late_gt14.mean() * 100), 1)
S["late30_pct"] = round(float(df.late_gt30.mean() * 100), 1)
lt = df[df.delay_days > 0].delay_days
S["late_median_days"] = float(lt.median())
S["late_mean_days"] = round(float(lt.mean()), 1)
S["late_p90_days"] = float(lt.quantile(.9))
S["late_value"] = float(df.loc[df.late == 1, "line_value"].sum())
S["late_value_pct"] = round(S["late_value"] / S["total_value"] * 100, 1)
S["late7_value"] = float(df.loc[df.late_gt7 == 1, "line_value"].sum())
S["late7_value_pct"] = round(S["late7_value"] / S["total_value"] * 100, 1)
S["po_lt_median"] = float(df.po_to_delivery_days.median())
S["po_lt_mean"] = round(float(df.po_to_delivery_days.mean()), 1)
S["po_lt_p90"] = float(df.po_to_delivery_days.quantile(.9))

# Freight analytic base
fa = df[df.freight_per_kg.notna()].copy()
S["freight_base_n"] = int(len(fa))
S["freight_base_total"] = float(fa.freight_usd.sum())
S["freight_per_kg_median"] = round(float(fa.freight_per_kg.median()), 2)
S["freight_per_kg_mean"] = round(float(fa.freight_per_kg.mean()), 2)
S["freight_pct_val_median"] = round(float(fa.freight_pct_value.median()), 1)
fa_w = fa.freight_usd.sum() / fa.line_value.sum() * 100
S["freight_pct_val_weighted"] = round(float(fa_w), 1)
S["freight_weighted_per_kg"] = round(float(fa.freight_usd.sum() / fa.weight_kg.sum()), 2)

# Descriptives
dcols = ["quantity", "line_value", "weight_kg", "freight_usd", "unit_price", "delay_days",
         "po_to_delivery_days", "freight_per_kg", "freight_pct_value"]
desc = pd.DataFrame({
    "n": df[dcols].count(), "mean": df[dcols].mean(), "median": df[dcols].median(),
    "std": df[dcols].std(), "p5": df[dcols].quantile(.05), "p95": df[dcols].quantile(.95),
    "skew": df[dcols].skew(),
})
S["desc"] = desc.round(2).to_dict("index")

# Segment tables
def seg(by, extra=None):
    g = df.groupby(by).agg(
        lines=("ID", "count"), value=("line_value", "sum"),
        late=("late", "mean"), late7=("late_gt7", "mean"), late30=("late_gt30", "mean"),
        med_delay=("delay_days", "median"), fpk=("freight_per_kg", "median"),
        fpv=("freight_pct_value", "median"), lt=("po_to_delivery_days", "median"),
    )
    g["late"] *= 100; g["late7"] *= 100; g["late30"] *= 100
    g["value_share"] = g.value / g.value.sum() * 100
    g["line_share"] = g.lines / g.lines.sum() * 100
    return g

for key, col in [("mode", "mode"), ("fulfill", "fulfill_via"), ("product", "product_group"),
                 ("inco", "inco"), ("year", "year"), ("country", "country"), ("vendor", "vendor")]:
    g = seg(col)
    S["by_" + key] = g.round(2).reset_index().to_dict("records")

# Pareto / concentration
v = df.groupby("vendor").line_value.sum().sort_values(ascending=False)
S["vendor_top3_share"] = round(float(v.head(3).sum() / v.sum() * 100), 1)
vd = v.drop("SCMS from RDC")
S["vendor_ex_rdc_top3_share"] = round(float(vd.head(3).sum() / vd.sum() * 100), 1)
S["vendor_ex_rdc_80"] = int((vd.cumsum() / vd.sum() <= .8).sum() + 1)
S["vendors_ex_rdc"] = int(len(vd))
c = df.groupby("country").line_value.sum().sort_values(ascending=False)
S["country_top5_share"] = round(float(c.head(5).sum() / c.sum() * 100), 1)
S["country_top10_share"] = round(float(c.head(10).sum() / c.sum() * 100), 1)
S["country_80"] = int((c.cumsum() / c.sum() <= .8).sum() + 1)

# ---------------------------------------------------------------
# 2. STATISTICAL TESTS
# ---------------------------------------------------------------
T = {}
core = df[df["mode"].isin(["Air", "Air Charter", "Truck", "Ocean"])]
H, p = stats.kruskal(*[g.delay_days.values for _, g in core.groupby("mode")])
T["kruskal_delay_mode"] = [round(float(H), 1), float(p)]
chi, p, dof, _ = stats.chi2_contingency(pd.crosstab(core["mode"], core.late))
T["chi2_late_mode"] = [round(float(chi), 1), float(p), int(dof)]
chi, p, dof, _ = stats.chi2_contingency(pd.crosstab(df.fulfill_via, df.late))
T["chi2_late_fulfill"] = [round(float(chi), 1), float(p), int(dof)]
cr = df.fulfill_via.eq("From RDC")
U, p = stats.mannwhitneyu(df.loc[cr, "delay_days"], df.loc[~cr, "delay_days"])
T["mwu_delay_fulfill"] = [float(U), float(p)]
top_c = df.country.value_counts().head(10).index
H, p = stats.kruskal(*[g.freight_per_kg.dropna().values for n, g in fa[fa.country.isin(top_c)].groupby("country")])
T["kruskal_fpk_country"] = [round(float(H), 1), float(p)]
H, p = stats.kruskal(*[g.freight_per_kg.values for _, g in fa[fa["mode"].isin(["Air", "Air Charter", "Truck", "Ocean"])].groupby("mode")])
T["kruskal_fpk_mode"] = [round(float(H), 1), float(p)]
r, p = stats.spearmanr(fa.weight_kg, fa.freight_usd)
T["spearman_weight_freight"] = [round(float(r), 3), float(p)]
r, p = stats.spearmanr(df.quantity, df.weight_kg, nan_policy="omit")
T["spearman_qty_weight"] = [round(float(r), 3), float(p)]
chi, p, dof, _ = stats.chi2_contingency(pd.crosstab(df.year, df.late))
T["chi2_late_year"] = [round(float(chi), 1), float(p), int(dof)]
r, p = stats.spearmanr(df.unit_price, df.delay_days)
T["spearman_price_delay"] = [round(float(r), 3), float(p)]
# effect size: Cramer's V for mode x late
n = len(core); k = min(pd.crosstab(core["mode"], core.late).shape) - 1
T["cramers_v_mode_late"] = round(float(np.sqrt(T["chi2_late_mode"][0] / (n * k))), 3)
S["tests"] = T

# ---------------------------------------------------------------
# 3. MODELS
# ---------------------------------------------------------------
# 3a. log-log OLS for freight cost
m = fa[fa["mode"].isin(["Air", "Air Charter", "Truck", "Ocean"]) & fa.weight_kg.gt(0)].copy()
X = pd.DataFrame({"ln_weight": np.log(m.weight_kg)})
for lvl in ["Air Charter", "Truck", "Ocean"]:
    X["mode_" + lvl.replace(" ", "_")] = (m["mode"] == lvl).astype(float)
X["from_RDC"] = (m.fulfill_via == "From RDC").astype(float)
X["ln_value"] = np.log(m.line_value.clip(lower=1))
X["year_trend"] = m.year - 2006
X.insert(0, "const", 1.0)
y = np.log(m.freight_usd)
beta, *_ = np.linalg.lstsq(X.values, y.values, rcond=None)
res = y.values - X.values @ beta
n_, k_ = X.shape
sigma2 = (res ** 2).sum() / (n_ - k_)
cov = sigma2 * np.linalg.inv(X.values.T @ X.values)
se = np.sqrt(np.diag(cov))
r2 = 1 - (res ** 2).sum() / ((y - y.mean()) ** 2).sum()
S["ols_freight"] = {"n": int(n_), "r2": round(float(r2), 3),
                    "coef": {c: {"b": round(float(b), 3), "se": round(float(s), 3), "t": round(float(b / s), 1)}
                             for c, b, s in zip(X.columns, beta, se)}}

# 3b. logistic regression for late delivery (sklearn)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
d = df[(df["mode"] != "Not recorded") & df.product_group.isin(["ARV", "HRDT"])].copy()
feat = pd.DataFrame({
    "mode_Air Charter": (d["mode"] == "Air Charter").astype(float),
    "mode_Truck": (d["mode"] == "Truck").astype(float),
    "mode_Ocean": (d["mode"] == "Ocean").astype(float),
    "fulfill_via_From RDC": (d.fulfill_via == "From RDC").astype(float),
    "product_group_HRDT": (d.product_group == "HRDT").astype(float),
})
cont = pd.DataFrame({"ln_value": np.log1p(d.line_value), "ln_qty": np.log1p(d.quantity), "year_trend": d.year - 2006})
featn = pd.concat([feat, (cont - cont.mean()) / cont.std()], axis=1)   # dummies kept raw, continuous standardised
Xtr, Xte, ytr, yte = train_test_split(featn, d.late, test_size=.25, random_state=42, stratify=d.late)
lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit(Xtr, ytr)
S["logit_auc"] = round(float(roc_auc_score(yte, lr.predict_proba(Xte)[:, 1])), 3)
S["logit_n"] = int(len(d))
odds = pd.Series(np.exp(lr.coef_[0]), index=featn.columns).sort_values(ascending=False)
S["logit_or"] = odds.round(2).to_dict()

# ---------------------------------------------------------------
# 4. FIGURES
# ---------------------------------------------------------------
NAVY, TEAL, ORANGE, RED, GREY, GOLD = "#1F3A5F", "#2A9D8F", "#E9873A", "#C8453F", "#8A94A0", "#E0B040"
sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#BFC9D4", "grid.color": "#E6EBF0",
                                      "axes.titleweight": "bold", "axes.titlesize": 12,
                                      "axes.labelsize": 10, "xtick.labelsize": 9, "ytick.labelsize": 9})
MODE_ORDER = ["Air", "Air Charter", "Truck", "Ocean"]
MODE_PAL = {"Air": NAVY, "Air Charter": TEAL, "Truck": ORANGE, "Ocean": RED, "Not recorded": GREY}

def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{OUT}/{name}.png", dpi=170, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# Fig 1 delivery outcome
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.3), gridspec_kw={"width_ratios": [1.6, 1]})
clipped = df.delay_days.clip(-60, 60)
bins = np.arange(-60, 62, 2)
ax[0].hist(clipped[clipped < 0], bins=bins, color=TEAL, label="Early")
ax[0].hist(clipped[clipped > 0], bins=bins, color=RED, label="Late")
ax[0].bar([0], [(df.delay_days == 0).sum()], width=2, color=NAVY, label="On schedule")
ax[0].set_yscale("log")
ax[0].set(title="Delivery vs. schedule (days, clipped at ±60)", xlabel="Days delivered after scheduled date (negative = early)", ylabel="Shipment lines (log scale)")
ax[0].legend(frameon=False)
shares = [S["early_pct"], S["exact_pct"], S["late_pct"]]
bars = ax[1].barh(["Early", "On schedule", "Late"][::-1], shares[::-1], color=[RED, NAVY, TEAL])
for b_, v_ in zip(bars, shares[::-1]):
    ax[1].text(v_ + 1, b_.get_y() + b_.get_height() / 2, f"{v_:.1f}%", va="center", fontsize=10, fontweight="bold")
ax[1].set(title="Delivery outcome share", xlim=(0, 78)); ax[1].xaxis.set_major_formatter(mtick.PercentFormatter())
save(fig, "fig1_delivery_outcome")

# Fig 2 annual trend: value + lines + late rate
yr = df.groupby("year").agg(lines=("ID", "count"), value=("line_value", "sum"),
                            late=("late", "mean"), late7=("late_gt7", "mean")).reset_index()
yr = yr[yr.year >= 2007]  # 2006 has only 65 lines
fig, ax1 = plt.subplots(figsize=(10.5, 4.4))
ax1.bar(yr.year, yr.value / 1e6, color=NAVY, alpha=.85, width=.65, label="Delivered value (USD m)")
ax1.set(xlabel="Delivery year", ylabel="Delivered value (USD millions)", title="Annual delivered value vs. late-delivery rate")
ax1.set_xticks(yr.year)
ax2 = ax1.twinx()
ax2.plot(yr.year, yr.late * 100, color=RED, marker="o", lw=2.2, label="Late (> 0 days) %")
ax2.plot(yr.year, yr.late7 * 100, color=ORANGE, marker="s", lw=2.2, label="Late > 7 days %")
ax2.set_ylabel("% of shipment lines"); ax2.grid(False); ax2.set_ylim(0, 30)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, loc="upper left", frameon=False)
save(fig, "fig2_annual_trend")

# Fig 3 mode scorecard (3 panels)
mm = df[df["mode"].isin(MODE_ORDER)].groupby("mode").agg(
    lines=("ID", "count"), value=("line_value", "sum"), late=("late", "mean"), late7=("late_gt7", "mean"),
    late30=("late_gt30", "mean")).loc[MODE_ORDER]
fig, ax = plt.subplots(1, 3, figsize=(12, 4.0))
ax[0].bar(MODE_ORDER, mm.value / mm.value.sum() * 100, color=[MODE_PAL[k] for k in MODE_ORDER])
for i, v_ in enumerate(mm.value / mm.value.sum() * 100): ax[0].text(i, v_ + .8, f"{v_:.0f}%", ha="center", fontsize=9, fontweight="bold")
ax[0].set(title="Share of delivered value", ylabel="% of value (modes recorded)"); ax[0].tick_params(axis="x", rotation=20)
w = .38; xs = np.arange(4)
ax[1].bar(xs - w / 2, mm.late * 100, w, color=ORANGE, label="Late > 0 days")
ax[1].bar(xs + w / 2, mm.late7 * 100, w, color=RED, label="Late > 7 days")
ax[1].set_xticks(xs); ax[1].set_xticklabels(MODE_ORDER, rotation=20)
for i, (a, b2) in enumerate(zip(mm.late * 100, mm.late7 * 100)):
    ax[1].text(i - w / 2, a + .4, f"{a:.0f}", ha="center", fontsize=8); ax[1].text(i + w / 2, b2 + .4, f"{b2:.0f}", ha="center", fontsize=8)
ax[1].set(title="Late-delivery rate by mode", ylabel="% of lines"); ax[1].legend(frameon=False, fontsize=8)
fpk_m = fa[fa["mode"].isin(MODE_ORDER)].groupby("mode").freight_per_kg.median().loc[MODE_ORDER]
ax[2].bar(MODE_ORDER, fpk_m, color=[MODE_PAL[k] for k in MODE_ORDER])
for i, v_ in enumerate(fpk_m): ax[2].text(i, v_ + .2, f"${v_:.1f}", ha="center", fontsize=9, fontweight="bold")
ax[2].set(title="Median freight cost per kg", ylabel="USD per kg"); ax[2].tick_params(axis="x", rotation=20)
save(fig, "fig3_mode_scorecard")

# Fig 4 freight per kg by mode box (log)
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
sns.boxplot(data=fa[fa["mode"].isin(MODE_ORDER)], x="mode", y="freight_per_kg", order=MODE_ORDER, palette=MODE_PAL,
            hue="mode", legend=False, fliersize=2, ax=ax[0])
ax[0].set_yscale("log"); ax[0].set_ylim(0.1, 200); ax[0].set(title="Freight cost per kg by mode (log scale, axis trimmed)", xlabel="", ylabel="USD per kg")
sns.boxplot(data=fa[fa["mode"].isin(MODE_ORDER)], x="mode", y="freight_pct_value", order=MODE_ORDER, palette=MODE_PAL,
            hue="mode", legend=False, fliersize=2, ax=ax[1])
ax[1].set_yscale("log"); ax[1].set_ylim(0.1, 1000); ax[1].set(title="Freight as % of commodity value (log scale, axis trimmed)", xlabel="", ylabel="% of line value")
save(fig, "fig4_freight_by_mode")

# Fig 5 weight vs freight scatter
fig, ax = plt.subplots(figsize=(9.2, 5.2))
samp = m.sample(min(2500, len(m)), random_state=4)
for md in MODE_ORDER:
    s_ = samp[samp["mode"] == md]
    ax.scatter(s_.weight_kg, s_.freight_usd, s=14, alpha=.45, color=MODE_PAL[md], label=md, edgecolor="none")
xx = np.logspace(np.log10(m.weight_kg.min()), np.log10(m.weight_kg.max()), 50)
b0 = S["ols_freight"]["coef"]["const"]["b"]; b1 = S["ols_freight"]["coef"]["ln_weight"]["b"]
ax.set_xscale("log"); ax.set_yscale("log")
ax.set(title="Shipment weight vs. freight cost (log-log, sample of 2,500 lines)", xlabel="Weight (kg, log)", ylabel="Freight cost (USD, log)")
ax.legend(frameon=False, title="Mode")
save(fig, "fig5_weight_vs_freight")
S["elasticity"] = round(b1, 3)

# Fig 6 country bubble: value, late rate
cc = df.groupby("country").agg(lines=("ID", "count"), value=("line_value", "sum"), late=("late", "mean"),
                               late7=("late_gt7", "mean"), fpk=("freight_per_kg", "median"),
                               fpv=("freight_pct_value", "median")).sort_values("value", ascending=False)
top = cc.head(15).copy()
fig, ax = plt.subplots(1, 2, figsize=(12.2, 5.0))
tt = top.sort_values("value")
ax[0].barh(tt.index, tt.value / 1e6, color=NAVY)
ax[0].set(title="Top 15 destination countries by delivered value", xlabel="USD millions")
ax2 = ax[1]
sc = ax2.scatter(top.fpk, top.late * 100, s=top.value / 1e6 * 4, c=top.late7 * 100, cmap="OrRd", edgecolor=NAVY, alpha=.85)
for n_c, r_ in top.iterrows():
    ax2.annotate(n_c.replace("Côte d'Ivoire", "Côte d'Iv."), (r_.fpk, r_.late * 100), fontsize=8, xytext=(5, 3), textcoords="offset points")
ax2.set(title="Cost vs. reliability (bubble = value)", xlabel="Median freight cost per kg (USD)", ylabel="Late-delivery rate (%)")
cb = fig.colorbar(sc, ax=ax2, shrink=.8); cb.set_label("% late > 7 days")
save(fig, "fig6_country_view")

# Fig 7 vendor Pareto + late
vv = df.groupby("vendor").agg(lines=("ID", "count"), value=("line_value", "sum"), late=("late", "mean"), late7=("late_gt7", "mean")).sort_values("value", ascending=False)
vv["cum"] = vv.value.cumsum() / vv.value.sum() * 100
topv = vv.head(10).copy()
short = lambda s: (s.replace("MYLAN LABORATORIES LTD (FORMERLY MATRIX LABORATORIES LIMITED)", "Mylan Laboratories")
                   .replace("MYLAN LABORATORIES LTD (FORMERLY MATRIX LABORAT...", "Mylan Laboratories")
                   .replace("ABBVIE LOGISTICS (FORMERLY ABBOTT LOGISTICS BV)", "AbbVie Logistics")
                   .replace("ABBVIE LOGISTICS (FORMERLY ABBOTT LOGISTICS B...", "AbbVie Logistics")
                   .replace("SCMS from RDC", "SCMS from RDC (stock)")
                   .replace("HETERO LABS LIMITED", "Hetero Labs").replace("CIPLA LIMITED", "Cipla")
                   .replace("Aurobindo Pharma Limited", "Aurobindo Pharma").replace("STRIDES ARCOLAB LIMITED", "Strides Arcolab")
                   .replace("Orgenics, Ltd", "Orgenics").replace("Trinity Biotech, Plc", "Trinity Biotech"))
def short2(n):
    for k, v_ in [("MYLAN", "Mylan Laboratories"), ("ABBVIE", "AbbVie Logistics"), ("SCMS from RDC", "SCMS from RDC (stock)"),
                  ("HETERO", "Hetero Labs"), ("CIPLA", "Cipla"), ("Aurobindo", "Aurobindo Pharma"), ("STRIDES", "Strides Arcolab"),
                  ("Orgenics", "Orgenics"), ("Trinity", "Trinity Biotech"), ("S. BUYS", "S. Buys Wholesaler")]:
        if n.startswith(k): return v_
    return n[:26]
labels = [short2(i) for i in topv.index]
fig, ax = plt.subplots(1, 2, figsize=(12.2, 4.8))
ax[0].bar(range(10), topv.value / 1e6, color=NAVY)
ax[0].set_xticks(range(10)); ax[0].set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
ax[0].set(title="Vendor Pareto: delivered value (top 10)", ylabel="USD millions")
axp = ax[0].twinx(); axp.plot(range(10), topv.cum, color=ORANGE, marker="o", lw=2); axp.set_ylim(0, 105); axp.grid(False)
axp.yaxis.set_major_formatter(mtick.PercentFormatter()); axp.set_ylabel("Cumulative share of value")
cols_ = [RED if r > 10 else (GOLD if r > 5 else TEAL) for r in topv.late7 * 100]
ax[1].barh(range(10)[::-1], topv.late * 100, color=cols_)
ax[1].set_yticks(range(10)[::-1]); ax[1].set_yticklabels(labels, fontsize=8)
for i, (a, l_) in enumerate(zip(topv.late * 100, topv.lines)): ax[1].text(a + .3, 9 - i, f"{a:.1f}%  (n={l_})", va="center", fontsize=8)
ax[1].set(title="Late-delivery rate, top 10 vendors by value", xlabel="% of lines delivered late", xlim=(0, 24))
save(fig, "fig7_vendor_view")

# Fig 8 fulfilment route x mode
fm = core.groupby(["mode", "fulfill_via"]).agg(lines=("ID", "count"), late=("late", "mean"), late7=("late_gt7", "mean")).reset_index()
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
pv = fm.pivot(index="mode", columns="fulfill_via", values="late").reindex(MODE_ORDER) * 100
ln = fm.pivot(index="mode", columns="fulfill_via", values="lines").reindex(MODE_ORDER)
pv = pv.where(ln >= 30)  # suppress cells with fewer than 30 lines
pv.plot(kind="bar", ax=ax[0], color=[ORANGE, NAVY], width=.7)
ax[0].set(title="Late rate by mode and route (cells with n ≥ 30)", xlabel="", ylabel="% of lines late"); ax[0].tick_params(axis="x", rotation=0)
ax[0].legend(title="", frameon=False)
for i, md in enumerate(MODE_ORDER):
    for j, col in enumerate(pv.columns):
        n_ = ln.loc[md, col]
        if pd.notna(n_) and pd.notna(pv.loc[md, col]):
            ax[0].text(i + (j - .5) * .36, pv.loc[md, col] + .3, f"n={int(n_)}", ha="center", fontsize=7, rotation=0)
rt = df.groupby("fulfill_via").agg(late=("late", "mean"), late7=("late_gt7", "mean"), late30=("late_gt30", "mean")) * 100
rt.T.plot(kind="bar", ax=ax[1], color=[ORANGE, NAVY], width=.65)
ax[1].set_xticklabels(["Late > 0 d", "Late > 7 d", "Late > 30 d"], rotation=0)
ax[1].set(title="Late severity: Direct Drop vs. RDC", ylabel="% of lines"); ax[1].legend(title="", frameon=False)
save(fig, "fig8_fulfilment_route")

# Fig 9 correlation heatmap (Spearman)
cc_cols = ["quantity", "line_value", "weight_kg", "freight_usd", "unit_price", "freight_per_kg", "delay_days", "po_to_delivery_days"]
nm = {"quantity": "Quantity", "line_value": "Line value", "weight_kg": "Weight (kg)", "freight_usd": "Freight cost",
      "unit_price": "Unit price", "freight_per_kg": "Freight / kg", "delay_days": "Delay (days)", "po_to_delivery_days": "PO-to-delivery"}
corr = df[cc_cols].corr(method="spearman").rename(index=nm, columns=nm)
S["corr"] = corr.round(2).to_dict()
fig, ax = plt.subplots(figsize=(8.2, 6.3))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, linewidths=.5, cbar_kws={"shrink": .8}, ax=ax)
ax.set_title("Spearman rank correlation of key shipment metrics")
save(fig, "fig9_correlation")

# Fig 10 lead time distribution for direct drop by mode
ld = df[df.po_to_delivery_days.notna() & df["mode"].isin(MODE_ORDER)]
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
sns.boxplot(data=ld, x="mode", y="po_to_delivery_days", order=[m_ for m_ in MODE_ORDER if m_ in ld["mode"].unique()],
            palette=MODE_PAL, hue="mode", legend=False, fliersize=2, ax=ax[0])
ax[0].set(title="PO-to-delivery lead time by mode (Direct Drop)", xlabel="", ylabel="Days")
ly = ld.groupby("year").po_to_delivery_days.median().reset_index()
ly = ly[ly.year >= 2007]
ax[1].plot(ly.year, ly.po_to_delivery_days, color=NAVY, marker="o", lw=2.2)
ax[1].fill_between(ly.year, ly.po_to_delivery_days, color=NAVY, alpha=.12)
ax[1].set(title="Median PO-to-delivery lead time by year", xlabel="Delivery year", ylabel="Days"); ax[1].set_xticks(ly.year)
save(fig, "fig10_leadtime")
S["leadtime_mode"] = ld.groupby("mode").po_to_delivery_days.agg(["count", "median", "mean"]).round(1).reset_index().to_dict("records")
S["leadtime_year"] = ly.round(1).to_dict("records")

# Fig 11 heatmap country x mode late rate (top 10 countries)
t10 = df.country.value_counts().head(10).index
hm = core[core.country.isin(t10)].pivot_table(index="country", columns="mode", values="late", aggfunc="mean") * 100
hm = hm.reindex(columns=MODE_ORDER)
cnt = core[core.country.isin(t10)].pivot_table(index="country", columns="mode", values="late", aggfunc="count").reindex(columns=MODE_ORDER)
ann = hm.round(0).astype("Int64").astype(str)
ann = ann.where(cnt >= 15, "n<15")
hm_plot = hm.where(cnt >= 15)
fig, ax = plt.subplots(figsize=(8.2, 5.0))
sns.heatmap(hm_plot, annot=ann, fmt="", cmap="OrRd", vmin=0, vmax=30, linewidths=.5, cbar_kws={"label": "% late"}, ax=ax)
ax.set(title="Late-delivery rate (%) by country and mode (cells with n ≥ 15)", xlabel="", ylabel="")
ax.tick_params(axis="y", rotation=0)
save(fig, "fig11_country_mode_heatmap")
S["hm"] = hm_plot.round(1).fillna(-1).to_dict("index")

# Fig 12 product group
pg = df[df.product_group.isin(["ARV", "HRDT"])].groupby("product_group").agg(
    value=("line_value", "sum"), late=("late", "mean"), fpk=("freight_per_kg", "median"), fpv=("freight_pct_value", "median"), up=("unit_price", "median"))
fig, ax = plt.subplots(1, 3, figsize=(12, 3.8))
for a, (col, ttl, fmt) in zip(ax, [("late", "Late-delivery rate (%)", "{:.1f}%"), ("fpk", "Median freight cost per kg (USD)", "${:.1f}"), ("fpv", "Median freight % of value", "{:.1f}%")]):
    vals = pg[col] * (100 if col == "late" else 1)
    a.bar(pg.index, vals, color=[NAVY, TEAL], width=.55)
    for i, v_ in enumerate(vals): a.text(i, v_ * 1.02, fmt.format(v_), ha="center", fontsize=10, fontweight="bold")
    a.set_title(ttl); a.set_ylim(0, vals.max() * 1.25)
save(fig, "fig12_product_group")

# Fig 13 drivers of late delivery (odds ratios)
orr = odds.sort_values()
labs = {"mode_Truck": "Mode: Truck (vs Air)", "mode_Ocean": "Mode: Ocean (vs Air)", "mode_Air Charter": "Mode: Air Charter (vs Air)",
        "fulfill_via_From RDC": "Route: From RDC (vs Direct Drop)", "ln_value": "Line value (log)", "ln_qty": "Quantity (log)",
        "year_trend": "Year trend", "product_group_ARV": "Product: ARV (vs ACT)", "product_group_HRDT": "Product: HRDT test kits (vs ARV)",
        "product_group_MRDT": "Product: MRDT (vs ACT)", "product_group_ANTM": "Product: ANTM (vs ACT)"}
fig, ax = plt.subplots(figsize=(9, 5.0))
ax.barh([labs.get(i, i) for i in orr.index], orr.values - 1, left=1, color=[RED if v_ > 1 else TEAL for v_ in orr.values])
ax.axvline(1, color="black", lw=.9)
ax.set(title="Drivers of late delivery (logistic regression odds ratios)", xlabel="Odds ratio (1.0 = no effect). Categories vs reference; continuous per 1 SD")
save(fig, "fig13_late_drivers")

# Fig 14 late severity by year/ share of value late
sv = df.groupby("year").apply(lambda g: pd.Series({
    "late_value_pct": g.loc[g.late == 1, "line_value"].sum() / g.line_value.sum() * 100,
    "late7_value_pct": g.loc[g.late_gt7 == 1, "line_value"].sum() / g.line_value.sum() * 100}), include_groups=False).reset_index()
sv = sv[sv.year >= 2007]
fig, ax = plt.subplots(figsize=(9.5, 4.2))
ax.bar(sv.year - .2, sv.late_value_pct, .4, color=ORANGE, label="Value delivered late (> 0 d)")
ax.bar(sv.year + .2, sv.late7_value_pct, .4, color=RED, label="Value delivered late (> 7 d)")
ax.set(title="Share of annual delivered value arriving late", xlabel="Delivery year", ylabel="% of annual value"); ax.set_xticks(sv.year)
ax.yaxis.set_major_formatter(mtick.PercentFormatter()); ax.legend(frameon=False)
save(fig, "fig14_value_at_risk")
S["value_late_by_year"] = sv.round(1).to_dict("records")

# ---------------------------------------------------------------
# 5. EXPORT
# ---------------------------------------------------------------
keep = ["ID", "Project Code", "country", "managed_by", "fulfill_via", "inco", "mode", "product_group", "sub_class", "vendor", "dosage_form",
        "quantity", "line_value", "unit_price", "weight_kg", "freight_usd", "freight_status", "insurance_usd",
        "sched", "deliv", "recorded", "po_date", "year", "delay_days", "outcome", "on_time", "late", "late_gt7", "late_gt14", "late_gt30",
        "record_lag", "po_to_delivery_days", "freight_per_kg", "freight_pct_value"]
df[keep].to_csv("scms_cleaned.csv", index=False)
with open("stats.json", "w") as f:
    f.write(json.dumps(S, indent=1, default=str).replace("NaN", "null"))   # strict JSON for downstream tools
print("done")
