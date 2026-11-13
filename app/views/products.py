"""Product catalogue: search and filter every product, see its funnel, and open one for its month-by-month view."""

from __future__ import annotations

import plotly.graph_objects as go
import polars as pl
import streamlit as st

import ui
from src.serving.catalog import products as collapse

ui.page_header("Product catalogue",
               "Search any product by ID, brand or category, filter the list, and open a product to see how it "
               "sells month by month.", rq="Business")
pm = ui.product_month()
if pm is None:
    st.info("The cleaned events for this dataset are not on this machine, so the catalogue is unavailable. "
            f"Run `python scripts/01_prepare_data.py --mode {ui.store().mode}` to create them.",
            icon=":material/hourglass_empty:")
    st.stop()

ink = ui.INK[ui.theme()]
pal = ui.SERIES[ui.theme()]
mode = ui.store().mode
ALL = "All months"
month_list = sorted(pm["month"].unique().to_list())
MIN_VIEWS_RATE = 100        # below this many views a conversion rate is too noisy to rank on


@st.cache_data(show_spinner="Filtering the catalogue…", max_entries=16)
def catalogue(mode: str, period: str) -> pl.DataFrame:
    rows = pm if period == ALL else pm.filter(pl.col("month") == period)
    return collapse(rows)


def nice(dept: str) -> str:
    return dept.replace("_", " ").capitalize()


# --------------------------------------------------------------------------- filters
f1, f2, f3, f4 = st.columns([2.2, 1, 1.4, 1])
query = f1.text_input("Search", placeholder="Product ID, brand or category (e.g. 1005115, samsung, headphone)",
                      icon=":material/search:", key="prod_query")
period = f2.selectbox("Period", [ALL] + month_list, key="prod_period",
                      format_func=lambda m: m if m == ALL else ui.month_name(m))
cat = catalogue(mode, period)
depts = (cat.group_by("department").agg(pl.col("revenue").sum()).sort("revenue", descending=True)["department"]
         .to_list())
dept = f3.multiselect("Department", depts, key="prod_dept", format_func=nice, placeholder="All departments")
SORTS = {"Revenue": "revenue", "Orders": "purchases", "Views": "views", "View → buy rate": "conversion",
         "Cart rate": "cart_rate", "Price": "price"}
sort_by = f4.selectbox("Sort by", list(SORTS), key="prod_sort")

with st.expander("More filters", icon=":material/tune:"):
    g1, g2, g3, g4 = st.columns([1.6, 1.4, 1, 1])
    pool = cat.filter(pl.col("department").is_in(dept)) if dept else cat
    brands = (pool.filter(pl.col("brand") != "unknown").group_by("brand").agg(pl.col("views").sum())
              .sort("views", descending=True).head(400)["brand"].to_list())
    brand = g1.multiselect("Brand", brands, key="prod_brand", placeholder="All brands (top 400 by views)")
    hi = float(cat["price"].quantile(0.995) or 1000)
    price = g2.slider("Price ($)", 0.0, round(hi, -1) + 10, (0.0, round(hi, -1) + 10), step=5.0, key="prod_price")
    min_views = g3.number_input("Min. views", 0, value=0, step=50, key="prod_minviews")
    only_sold = g4.toggle("Sold at least once", key="prod_sold")


# --------------------------------------------------------------------------- apply
d = cat
if query.strip():
    q = query.strip().lower()
    if q.isdigit():
        d = d.filter(pl.col("product_id").cast(pl.String).str.starts_with(q))
    else:
        d = d.filter(pl.col("brand").str.contains(q, literal=True)
                     | pl.col("category_code").str.contains(q, literal=True))
if dept:
    d = d.filter(pl.col("department").is_in(dept))
if brand:
    d = d.filter(pl.col("brand").is_in(brand))
if price != (0.0, round(hi, -1) + 10):
    d = d.filter(pl.col("price").is_between(*price))
if min_views:
    d = d.filter(pl.col("views") >= min_views)
if only_sold:
    d = d.filter(pl.col("purchases") > 0)
key = SORTS[sort_by]
if key in ("conversion", "cart_rate"):           # rates on a handful of views are noise: rank only solid ones
    d = d.filter(pl.col("views") >= max(min_views, MIN_VIEWS_RATE))
d = d.sort(key, descending=True, nulls_last=True)
if key in ("conversion", "cart_rate"):
    st.caption(f"Ranking by a rate uses only products with at least {MIN_VIEWS_RATE} views.")

# --------------------------------------------------------------------------- KPIs
k = st.columns(5)
k[0].metric("Products", ui.fmt_int(d.height), f"of {ui.fmt_int(cat.height)} in the period", delta_color="off",
            delta_arrow="off")
k[1].metric("Views", ui.fmt_int(d["views"].sum()))
k[2].metric("Orders", ui.fmt_int(d["purchases"].sum()))
k[3].metric("Revenue", ui.fmt_money(d["revenue"].sum()))
k[4].metric("View → buy rate", f"{d['purchases'].sum() / max(d['views'].sum(), 1):.2%}",
            help="Purchases per product view, across the products listed.")

# --------------------------------------------------------------------------- list
SHOW = 500
listing, brand_tab = st.tabs([":material/list: Products", ":material/sell: Brands"])
with listing:
    view = d.head(SHOW).select(
        "product_id", "category_code", "brand", "price", "views", "carts", "purchases", "conversion", "revenue",
        "months")
    rate_max = float(view["conversion"].max() or 0.1)
    event = st.dataframe(
        view, hide_index=True, width="stretch", height=min(420, 40 + 35 * max(view.height, 1)), on_select="rerun", selection_mode="single-row",
        key=f"prod_table_{period}",
        column_config={
            "product_id": st.column_config.NumberColumn("Product ID", format="%d"),
            "category_code": "Category", "brand": "Brand",
            "price": st.column_config.NumberColumn("Price", format="dollar"),
            "views": st.column_config.NumberColumn("Views", format="localized"),
            "carts": st.column_config.NumberColumn("Carts", format="localized"),
            "purchases": st.column_config.NumberColumn("Orders", format="localized"),
            "conversion": st.column_config.ProgressColumn("View → buy", format="percent", min_value=0,
                                                          max_value=max(rate_max, 0.01)),
            "revenue": st.column_config.NumberColumn("Revenue", format="dollar"),
            "months": st.column_config.NumberColumn("Months on sale", help="Months with at least one event."),
        },
    )
    l, r = st.columns([3, 1])
    l.caption(f"Showing {min(SHOW, d.height):,} of {d.height:,} products. Select a row to open the product."
              + (f" {ui.sample_note()}" if ui.sample_note() else ""))
    r.download_button("Download list (CSV)", d.head(50_000).write_csv(), file_name=f"products_{mode}.csv",
                      mime="text/csv", icon=":material/download:", width="stretch")

with brand_tab:
    br = (d.group_by("brand").agg(products=pl.len(), views=pl.col("views").sum(), orders=pl.col("purchases").sum(),
                                  revenue=pl.col("revenue").sum(), price=pl.col("price").median())
          .with_columns(conversion=pl.col("orders") / pl.col("views")).sort("revenue", descending=True))
    st.dataframe(br.head(SHOW), hide_index=True, width="stretch", height=420, column_config={
        "brand": "Brand", "products": st.column_config.NumberColumn("Products", format="localized"),
        "views": st.column_config.NumberColumn("Views", format="localized"),
        "orders": st.column_config.NumberColumn("Orders", format="localized"),
        "revenue": st.column_config.NumberColumn("Revenue", format="dollar"),
        "price": st.column_config.NumberColumn("Median price", format="dollar"),
        "conversion": st.column_config.NumberColumn("View → buy", format="percent"),
    })

# --------------------------------------------------------------------------- detail
rows = event.selection.rows if event is not None else []
pid = int(view["product_id"][rows[0]]) if rows else (int(view["product_id"][0]) if view.height == 1 else None)
st.divider()
if pid is None:
    st.markdown("<div class='card'><div class='step'>Product detail</div><p>Select a product in the list (or "
                "search for its ID) to see its month-by-month sales, price and how it compares with its "
                "category.</p></div>", unsafe_allow_html=True)
    st.stop()

p = cat.filter(pl.col("product_id") == pid).row(0, named=True)
hist = pm.filter(pl.col("product_id") == pid).sort("month")
st.markdown(f"<div class='step'>Product detail · {ui.month_name(period) if period != ALL else ALL}</div>"
            f"<h3 style='margin:.1rem 0 .2rem 0'>#{pid} · {p['brand'].capitalize()}</h3>"
            f"<p style='opacity:.7;margin:0 0 .8rem 0'>{p['category_code']} · median price ${p['price']:,.2f}</p>",
            unsafe_allow_html=True)

same_cat = cat.filter((pl.col("category_code") == p["category_code"]) & (pl.col("views") >= MIN_VIEWS_RATE))
cat_rate = same_cat["purchases"].sum() / max(same_cat["views"].sum(), 1)
c = st.columns(5)
c[0].metric("Views", ui.fmt_int(p["views"]))
c[1].metric("Add to cart", ui.fmt_int(p["carts"]), f"{(p['cart_rate'] or 0):.1%} of views", delta_color="off",
            delta_arrow="off")
c[2].metric("Orders", ui.fmt_int(p["purchases"]))
c[3].metric("Revenue", ui.fmt_money(p["revenue"]))
if p["views"] >= MIN_VIEWS_RATE and p["category_code"] != "unknown":
    c[4].metric("View → buy rate", f"{p['conversion']:.2%}",
                f"{(p['conversion'] - cat_rate) * 100:+.2f} pp vs category", delta_arrow="off",
                help=f"Category average: {cat_rate:.2%} (products with {MIN_VIEWS_RATE}+ views).")
else:
    c[4].metric("View → buy rate", f"{(p['conversion'] or 0):.2%}",
                "too few views to compare" if p["views"] < MIN_VIEWS_RATE else "no category", delta_color="off",
                delta_arrow="off")

left, right = st.columns(2, gap="large")
labels = [ui.month_name(m, short=True) for m in hist["month"]]
with left:
    st.subheader("Orders per month", anchor=False)
    rate = [f"{o / v:.2%}" if v else "–" for o, v in zip(hist["purchases"], hist["views"])]
    fig = go.Figure(go.Bar(
        x=labels, y=hist["purchases"], marker=dict(color=pal[0], cornerradius=4),
        text=[f"{v:,}" for v in hist["purchases"]], textposition="outside", cliponaxis=False,
        textfont=dict(color=ink["text"], size=11), customdata=list(zip(hist["views"], rate)),
        hovertemplate="%{x}: %{y:,} orders<br>%{customdata[0]:,} views · %{customdata[1]} view → buy<extra></extra>",
    ))
    ui.style(fig, height=300, legend=False, top=30)
    fig.update_yaxes(title="Orders", range=[0, max(float(hist["purchases"].max()), 1) * 1.2])
    fig.update_xaxes(showgrid=False)
    ui.show(fig, key="p_hist")
with right:
    st.subheader("Price per month", anchor=False)
    fig = go.Figure(go.Scatter(
        x=labels, y=hist["price"], mode="lines+markers+text", line=dict(color=pal[1], width=2),
        marker=dict(size=8, line=dict(color=ink["surface"], width=2)),
        text=[f"${v:,.0f}" for v in hist["price"]], textposition="top center",
        textfont=dict(color=ink["text"], size=11), hovertemplate="%{x}: $%{y:,.2f}<extra></extra>",
    ))
    ui.style(fig, height=300, legend=False, top=30)
    lo, hi_ = float(hist["price"].min()), float(hist["price"].max())
    pad = max((hi_ - lo) * 0.4, hi_ * 0.05, 1)
    fig.update_yaxes(title="Median price ($)", range=[max(lo - pad, 0), hi_ + pad])
    fig.update_xaxes(showgrid=False)
    ui.show(fig, key="p_price")

if p["category_code"] != "unknown":
    st.subheader(f"Best sellers in {p['category_code']}", anchor=False)
    peers = (cat.filter(pl.col("category_code") == p["category_code"]).sort("revenue", descending=True).head(8)
             .with_columns(this=pl.when(pl.col("product_id") == pid).then(pl.lit("◀ this product")).otherwise(pl.lit(""))))
    st.dataframe(peers.select("product_id", "brand", "price", "views", "purchases", "conversion", "revenue", "this"),
                 hide_index=True, width="stretch", column_config={
                     "product_id": st.column_config.NumberColumn("Product ID", format="%d"), "brand": "Brand",
                     "price": st.column_config.NumberColumn("Price", format="dollar"),
                     "views": st.column_config.NumberColumn("Views", format="localized"),
                     "purchases": st.column_config.NumberColumn("Orders", format="localized"),
                     "conversion": st.column_config.NumberColumn("View → buy", format="percent"),
                     "revenue": st.column_config.NumberColumn("Revenue", format="dollar"), "this": "",
                 })
