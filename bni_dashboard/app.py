import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from components.styling import inject_global_css, data_badge, NAVY, ORANGE, TEXT_MUTED, CATEGORICAL_SEQUENCE, GREEN, RED, AMBER
from components.kpi_card import kpi_row
from components.flow import provenance_chain_html, tax_normalization_flow_html
from utils import data_loader as dl
from utils import procurement_calc as pc
from utils import rag_metrics as rm
from utils import governance as gov
from utils import roi_calc as rc

st.set_page_config(page_title="BNI | Procurement, Legal-RAG & Governance Intelligence", page_icon="🏦", layout="wide")
inject_global_css()

# ---------------------------------------------------------------- header --
st.markdown(f"""
<div class="bni-header">
  <div style="font-size:0.75rem;letter-spacing:0.1em;color:#C9D9E8;text-transform:uppercase;">Credits to Hafizh Yasril and Revel</div>
  <h1 style="color:white;margin:6px 0 0 0;">Legal Document Creation, Citation with RAG <span class="accent">&amp;</span> Procurement Efficiency / Anti-Corruption</h1>
  <div class="subtitle">Evidence-based laptop procurement benchmarking, RAG-grounded legal document intelligence, and governance indicators — for BNI management review.</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------- nav -----
PAGES = [" Executive Summary", " Procurement Intelligence", " Legal & RAG Intelligence",
         " Governance & Anti-Corruption", " Data & Methodology"]
st.sidebar.markdown("### Navigasi")
page = st.sidebar.radio("", PAGES, label_visibility="collapsed")
st.sidebar.divider()
st.sidebar.caption("Legenda label data: **Observed/Real**, **Synthetic**, **Illustrative**, **Assumption**, **Calculated**. Lihat tab Data & Metodologi.")

# ---------------------------------------------------------------- load ----
listings = dl.load_marketplace_listings()
history = dl.load_procurement_history()
catalog = dl.load_procurement_catalog()
scenarios = dl.load_scenarios()
assumptions = dl.load_assumptions()
legal_docs = dl.load_legal_documents()
rag_answers = dl.load_rag_answers()
citations = dl.load_citations()
pilot_metrics = dl.load_pilot_metrics().set_index("metric_id")
bu_impact = dl.load_business_unit_impact()
process_time = dl.load_process_time()
doc_prep = dl.load_doc_prep_time()
cite_validity = dl.load_citation_validity()
pilot_ramp = dl.load_pilot_ramp()
roi_scenarios = dl.load_roi_scenarios()
citation_evals = dl.load_citation_evaluations()
investment_breakdown = dl.load_investment_breakdown()
process_annual_volume = dl.load_process_annual_volume()

valid_listings = pc.valid_listings(listings)

# =============================================================================================
# PAGE 1 — EXECUTIVE SUMMARY
# =============================================================================================
if page == PAGES[0]:
    total_opportunity = bu_impact["estimated_annual_impact_rp"].sum()
    citation_cov = rm.citation_coverage_pct(rag_answers)
    procurement_opp = bu_impact.loc[bu_impact.business_unit == "Procurement", "estimated_annual_impact_rp"].iloc[0]
    roi_headline = pilot_metrics.loc["ROI_HEADLINE", "value"]
    docs_processed = pilot_metrics.loc["TOTAL_DOCS_PROCESSED", "value"]
    records_searchable = pilot_metrics.loc["RECORDS_SEARCHABLE_TOTAL", "value"]
    split_flags = gov.detect_split_purchases(history)

    st.markdown('<div class="section-tag">Business Impact Snapshot</div>', unsafe_allow_html=True)
    kpi_row([
        dict(label="Total Annual Opportunity (computed)", value=total_opportunity/1e9, prefix="Rp", suffix="B",
             decimals=2, data_type="calculated", note="= Σ business_unit_impact.csv (Governance tab shows the allocation rule)"),
        dict(label="Citation Coverage (jawaban RAG bersitasi valid)", value=citation_cov, suffix="%", decimals=1,
             data_type="calculated", note="Dihitung live dari rag_answers.csv"),
        dict(label="Procurement Benchmarking Opportunity", value=procurement_opp/1e9, prefix="Rp", suffix="B",
             decimals=2, data_type="calculated", note="= procurement_savings_benefit_rp, Base Case (lihat rincian di bawah)"),
        dict(label="ROI on Pilot Investment (Base Case)", value=roi_headline, suffix="x", decimals=2,
             data_type="calculated", note="= (Benefit − Investment) / Investment — rincian penuh di bawah"),
    ])
    st.write("")
    kpi_row([
        dict(label="Documents in Prototype Corpus", value=docs_processed, decimals=0, data_type="calculated", note="= len(legal_documents.csv)"),
        dict(label="Records Searchable (this prototype dataset)", value=records_searchable/1000, suffix="k",
             decimals=2, data_type="calculated", note="= Σ baris di semua dataset yang di-ingest (lihat Data & Metodologi)"),
        dict(label="Potential Duplicate/Split-Purchase Indicators", value=len(split_flags), decimals=0,
             data_type="calculated", note="Dihitung live dari procurement_history.csv — lihat tab Governance"),
        dict(label="Marketplace Listings dalam Benchmark Dataset", value=len(valid_listings), decimals=0,
             data_type="calculated", note=f"dari {len(listings)} total listing mentah"),
    ], height=190)
    st.caption("Tidak ada angka pada halaman ini yang diketik manual ke CSV — setiap nilai adalah hasil fungsi di `utils/roi_calc.py`, `utils/rag_metrics.py`, atau `utils/governance.py` yang dijalankan atas dataset di `data/`. Lihat expander \"Tunjukkan perhitungannya\" di bawah.")

    st.write("")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### Impact by Business Unit")
        fig = px.bar(bu_impact.sort_values("estimated_annual_impact_rp"), x="estimated_annual_impact_rp", y="business_unit",
                     orientation="h", text=bu_impact.sort_values("estimated_annual_impact_rp")["estimated_annual_impact_rp"].apply(pc.fmt_rp),
                     color_discrete_sequence=[ORANGE])
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_title="Estimasi dampak tahunan (Rp) — skala log", yaxis_title="", showlegend=False,
                           plot_bgcolor="white", margin=dict(l=0, r=30), xaxis_type="log")
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Procurement = procurement_savings_benefit_rp langsung dari mesin benchmarking. Legal/Compliance/Corp. Affairs = time_savings_benefit_rp dialokasikan proporsional terhadap pangsa penggunaan RAG tiap unit (business_unit di rag_answers.csv). Skala sumbu log karena penghematan volume pengadaan jauh lebih besar dari penghematan waktu administratif pada tingkat adopsi saat ini — temuan ini sendiri adalah insight, bukan cacat model. " + data_badge("calculated"), unsafe_allow_html=True)
    with c2:
        st.markdown("#### ROI Scenario Modeling")
        for _, r in roi_scenarios.iterrows():
            is_base = r["scenario"] == "Base Case"
            bg = NAVY if is_base else "white"
            fg = "white" if is_base else NAVY
            sub = "#C9D9E8" if is_base else TEXT_MUTED
            roi_color = "#7CD9A8" if r['roi_x'] >= 0 else "#F2A8A0"
            st.markdown(f"""
            <div style="background:{bg};border:1px solid #E4E9F0;border-radius:12px;padding:12px 16px;margin-bottom:8px;">
              <div style="color:{fg};font-weight:700;">{r['scenario']} <span style="font-weight:400;font-size:0.72rem;">(adopsi {r['adoption_rate']*100:.0f}%, tier diskon {int(r['discount_qty_tier'])} unit)</span></div>
              <div style="color:{sub};font-size:0.78rem;display:flex;justify-content:space-between;margin-top:4px;flex-wrap:wrap;gap:4px;">
                <span>Investment: {pc.fmt_rp(r['pilot_investment_rp'])}</span>
                <span>Benefit: {pc.fmt_rp(r['est_annual_benefit_rp'])}</span>
                <span style="color:{roi_color if is_base else sub};font-weight:700;">ROI: {r['roi_x']:.2f}x</span>
                <span>Payback: {r['payback_months']:.1f} mo</span>
              </div>
            </div>""", unsafe_allow_html=True)
        st.caption("ROI = (Benefit − Investment) / Investment · Payback = Investment / Benefit × 12 — dihitung di `utils/roi_calc.py:roi_and_payback()`. " + data_badge("calculated"), unsafe_allow_html=True)

    with st.expander("🔎 Tunjukkan perhitungannya — investasi, penghematan waktu, penghematan pengadaan, semua input & rumus", expanded=False):
        st.markdown("##### 1) Investment breakdown (Σ = Pilot Investment)")
        inv_disp = investment_breakdown.copy()
        inv_disp["unit_cost_rp"] = inv_disp["unit_cost_rp"].apply(pc.fmt_rp)
        inv_disp["total_rp"] = inv_disp["total_rp"].apply(pc.fmt_rp)
        st.dataframe(inv_disp[["component", "unit_cost_rp", "quantity", "unit", "total_rp", "basis"]], use_container_width=True, hide_index=True)
        st.markdown(f"**Total Pilot Investment = {pc.fmt_rp(investment_breakdown.total_rp.sum())}**")

        st.markdown("##### 2) Time-savings benefit inputs")
        a1, a2, a3 = st.columns(3)
        salary_annual = 70_749_733
        hourly_rate = rc.hourly_rate_rp(salary_annual / 12, 173.0)
        a1.metric("Gaji tahunan benchmark (Compliance Officer, Payscale)", pc.fmt_rp(salary_annual))
        a2.metric("Jam kerja/bulan (konvensi 40 jam/minggu)", "173")
        a3.metric("Tarif per jam (= gaji/bulan ÷ 173)", pc.fmt_rp(hourly_rate))
        st.dataframe(process_annual_volume, use_container_width=True, hide_index=True)
        st.caption("Formula per proses: jam_dihemat/tahun = ((before−after) menit / 60) × estimated_annual_occurrences × adoption_rate; rupiah_dihemat = jam_dihemat × tarif_per_jam. Diimplementasikan di `utils/roi_calc.py:time_savings_benefit()`.")
        scen_pick = st.selectbox("Lihat rincian time-savings untuk skenario:", roi_scenarios["scenario"].tolist(), index=1)
        adoption = float(roi_scenarios.loc[roi_scenarios.scenario == scen_pick, "adoption_rate"].iloc[0])
        ts_detail = rc.time_savings_benefit(process_time, process_annual_volume, hourly_rate, adoption)
        ts_show = ts_detail[["process_name", "before_minutes", "after_minutes", "estimated_annual_occurrences", "hours_saved_per_year", "rp_saved_per_year"]].copy()
        ts_show["rp_saved_per_year"] = ts_show["rp_saved_per_year"].apply(pc.fmt_rp)
        ts_show["hours_saved_per_year"] = ts_show["hours_saved_per_year"].round(1)
        st.dataframe(ts_show, use_container_width=True, hide_index=True)
        st.markdown(f"**Total time-savings benefit ({scen_pick}, adopsi {adoption*100:.0f}%) = {pc.fmt_rp(ts_detail.rp_saved_per_year.sum())}**")

        st.markdown("##### 3) Procurement-savings benefit inputs")
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Karyawan BNI", "27.201", help="Sumber: Wikipedia infobox BNI, mengutip profil perusahaan/laporan BNI FY2025")
        b2.metric("% eligible laptop (asumsi)", "35%")
        b3.metric("Refresh rate/tahun (asumsi)", "25%")
        b4.metric("Volume pengadaan/tahun (dihitung)", f"{27201*0.35*0.25:,.0f} unit")
        _sub_live, _lvl_live = pc.find_comparable(valid_listings, "mid", 16, 512, False, min_n=5)
        _stats_live = pc.bucket_stats(_sub_live)
        current_quote_avg = float(history.loc[history.spec_id == "SPEC-MID", "current_quote_unit_price_rp"].mean())
        c1b, c2b = st.columns(2)
        c1b.metric("Market benchmark (median, dari marketplace_listings)", pc.fmt_rp(_stats_live["median"]), help=f"n={_stats_live['n']}, cakupan: {_lvl_live}")
        c2b.metric("Rata-rata current quote BNI (spec sama, synthetic)", pc.fmt_rp(current_quote_avg))
        disc_tier = int(roi_scenarios.loc[roi_scenarios.scenario == scen_pick, "discount_qty_tier"].iloc[0])
        disc_pct = float(scenarios.loc[scenarios.qty_tier == disc_tier, "assumed_discount_pct"].iloc[0]) / 100.0
        effective_benchmark = rc.effective_benchmark_with_discount(current_quote_avg, _stats_live["median"], disc_pct)
        st.caption(f"Formula: procurement_savings = max(current_quote_avg − effective_benchmark, 0) × volume_tahunan × adoption_rate, di mana effective_benchmark memperhitungkan asumsi diskon volume tier {disc_tier} unit ({disc_pct*100:.0f}%) tapi tidak pernah lebih murah dari benchmark pasar mentah. Diimplementasikan di `utils/roi_calc.py:effective_benchmark_with_discount()` + `procurement_savings_benefit()` — fungsi yang SAMA dipakai generator data dan halaman ini.")
        proc_saving_live = rc.procurement_savings_benefit(current_quote_avg, effective_benchmark, 27201*0.35*0.25, adoption)
        st.markdown(f"**Procurement-savings benefit ({scen_pick}, adopsi {adoption*100:.0f}%, effective benchmark {pc.fmt_rp(effective_benchmark)}) = {pc.fmt_rp(proc_saving_live)}**")

        st.markdown("##### 4) Roll-up")
        total_benefit_live = ts_detail.rp_saved_per_year.sum() + proc_saving_live
        rp_live = rc.roi_and_payback(investment_breakdown.total_rp.sum(), total_benefit_live)
        st.markdown(f"**Total Annual Benefit = {pc.fmt_rp(ts_detail.rp_saved_per_year.sum())} (time) + {pc.fmt_rp(proc_saving_live)} (procurement) = {pc.fmt_rp(total_benefit_live)}**")
        st.markdown(f"**ROI = ({pc.fmt_rp(total_benefit_live)} − {pc.fmt_rp(investment_breakdown.total_rp.sum())}) / {pc.fmt_rp(investment_breakdown.total_rp.sum())} = {rp_live['roi_x']:.2f}x · Payback = {rp_live['payback_months']:.1f} bulan**")
        st.caption("Angka di atas dihitung ulang secara live saat halaman ini dirender — bandingkan dengan roi_scenarios.csv untuk memverifikasi konsistensi.")

    st.divider()
    st.markdown("#### Business Story")
    bcol1, bcol2, bcol3 = st.columns(3)
    bcol1.info("**Legal + RAG**\n\nFaster → More Grounded → More Traceable → More Auditable")
    bcol2.warning("**Procurement**\n\nMarket Data → Statistical Benchmark → Evidence-Based Negotiation → Potential Savings")
    bcol3.success("**Anti-Corruption / Governance**\n\nTransparent Benchmark → Detect Anomalies → Require Evidence → Improve Controls")

# =============================================================================================
# PAGE 2 — PROCUREMENT INTELLIGENCE
# =============================================================================================
elif page == PAGES[1]:
    st.markdown('<div class="section-tag">Laptop Procurement Category · Specification-Based Benchmarking</div>', unsafe_allow_html=True)
    st.caption("Ruang lingkup versi ini **hanya laptop**. Sistem membandingkan berdasarkan spesifikasi (tier CPU/RAM/storage/GPU), bukan merek atau vendor tertentu.")

    with st.expander("🔍 Filter", expanded=True):
        f1, f2, f3, f4, f5 = st.columns(5)
        with f1:
            f_tier = st.multiselect("Tier CPU", pc.CPU_TIER_ORDER, default=pc.CPU_TIER_ORDER, format_func=lambda x: pc.CPU_TIER_LABEL[x])
        with f2:
            ram_values = sorted(listings.ram_gb.dropna().unique()) + ["Missing"]
            f_ram = st.multiselect("RAM (GB)", ram_values, default=ram_values)
        with f3:
            f_gpu = st.selectbox("GPU diskrit", ["Semua", "Hanya integrated", "Hanya diskrit"])
        with f4:
            f_marketplace = st.multiselect("Sumber marketplace", sorted(listings.marketplace.unique()), default=sorted(listings.marketplace.unique()))
        with f5:
            price_max = int(listings.price_rp.max())
            f_price = st.slider("Rentang harga (Rp juta)", 0, price_max // 1_000_000 + 1, (0, price_max // 1_000_000 + 1))

    include_ppn = st.toggle("Harga listing sudah termasuk PPN 12%", value=True,
                             help="UU HPP No.7/2021 — tarif PPN umum 12% berlaku sejak 1 Jan 2025.")

    ram_filter = listings.ram_gb.isin([value for value in f_ram if value != "Missing"])
    if "Missing" in f_ram:
        ram_filter |= listings.ram_gb.isna()
    view = listings[listings.cpu_tier.isin(f_tier) & ram_filter & listings.marketplace.isin(f_marketplace)]
    if f_gpu == "Hanya integrated":
        view = view[~view.dedicated_gpu]
    elif f_gpu == "Hanya diskrit":
        view = view[view.dedicated_gpu]
    view = view.copy()
    view["price_rp"] = pc.normalize_tax(view, "price_rp", include_ppn)
    view = view[(view.price_rp >= f_price[0] * 1_000_000) & (view.price_rp <= f_price[1] * 1_000_000)]

    tabs = st.tabs(["Market Benchmark", "Data Quality Pipeline", "Volume Scenario Calculator", "Spec-Price Relationships", "Tax Normalization"])

    with tabs[0]:
        if view.empty:
            st.warning("Tidak ada listing sesuai filter.")
        else:
            p = view.price_rp.astype(float)
            k1, k2, k3, k4, k5, k6 = st.columns(6)
            k1.metric("n listings", len(view)); k2.metric("Mean", pc.fmt_rp(p.mean())); k3.metric("Median", pc.fmt_rp(p.median()))
            k4.metric("Std Dev", pc.fmt_rp(p.std())); k5.metric("Min", pc.fmt_rp(p.min())); k6.metric("Max", pc.fmt_rp(p.max()))
            pr1, pr2, pr3 = st.columns(3)
            pr1.metric("P25", pc.fmt_rp(p.quantile(.25))); pr2.metric("P75", pc.fmt_rp(p.quantile(.75))); pr3.metric("IQR", pc.fmt_rp(p.quantile(.75)-p.quantile(.25)))

            st.markdown("#### Laptop yang masuk dalam rentang harga dan filter")
            listing_results = view.sort_values("price_rp").copy()
            listing_results["specification"] = listing_results.apply(
                lambda row: f"{row['cpu_model'] if pd.notna(row['cpu_model']) else 'CPU tidak tercantum'} · "
                f"{int(row['ram_gb']) if pd.notna(row['ram_gb']) else '-'}GB RAM · "
                f"{int(row['storage_gb']) if pd.notna(row['storage_gb']) else '-'}GB storage"
                + (f" · {row['gpu_model']}" if row["dedicated_gpu"] else " · Integrated GPU"),
                axis=1,
            )
            listing_results = listing_results.rename(columns={
                "title": "Laptop",
                "marketplace": "Marketplace",
                "seller": "Penjual",
                "price_rp": "Harga",
                "rating": "Rating",
                "sold_min": "Terjual (min.)",
            })
            page_size = 10
            total_pages = max(1, (len(listing_results) + page_size - 1) // page_size)
            current_page = min(st.session_state.get("procurement_listing_page", 1), total_pages)
            st.session_state["procurement_listing_page"] = current_page

            first_row = (current_page - 1) * page_size
            last_row = min(first_row + page_size, len(listing_results))
            st.caption(f"Menampilkan listing {first_row + 1}-{last_row} dari {len(listing_results)} yang sesuai filter aktif. Harga mengikuti pengaturan PPN di atas; urutan dari harga terendah.")
            st.dataframe(
                listing_results.iloc[first_row:last_row][["Laptop", "brand", "specification", "Marketplace", "Penjual",
                                 "Harga", "Rating", "Terjual (min.)"]].assign(
                    Harga=lambda data: data["Harga"].apply(pc.fmt_rp),
                ).rename(columns={"brand": "Merek"}),
                use_container_width=True,
                hide_index=True,
            )
            if total_pages <= 7:
                page_options = list(range(1, total_pages + 1))
            elif current_page <= 4:
                page_options = [1, 2, 3, 4, 5, "...", total_pages]
            elif current_page >= total_pages - 3:
                page_options = [1, "...", total_pages - 5, total_pages - 4, total_pages - 3,
                                total_pages - 2, total_pages - 1, total_pages]
            else:
                page_options = [1, "...", current_page - 1, current_page, current_page + 1, "...", total_pages]

            page_controls = st.columns([0.85] * (len(page_options) + 2) + [7])
            if page_controls[0].button("‹", disabled=current_page == 1, help="Halaman sebelumnya", key="procurement_listing_previous"):
                st.session_state["procurement_listing_page"] = current_page - 1
                st.rerun()
            for option_index, (control, page_option) in enumerate(zip(page_controls[1:1 + len(page_options)], page_options)):
                if page_option == "...":
                    control.button("...", disabled=True, key=f"procurement_listing_ellipsis_{current_page}_{option_index}")
                elif control.button(
                    str(page_option),
                    type="primary" if page_option == current_page else "secondary",
                    key=f"procurement_listing_page_{page_option}",
                ):
                    st.session_state["procurement_listing_page"] = page_option
                    st.rerun()
            if page_controls[-2].button("›", disabled=current_page == total_pages, help="Halaman berikutnya", key="procurement_listing_next"):
                st.session_state["procurement_listing_page"] = current_page + 1
                st.rerun()

            cc1, cc2 = st.columns(2)
            with cc1:
                fig = px.histogram(view, x="price_rp", nbins=20, color_discrete_sequence=[NAVY], title="Distribusi Harga (Benchmark Dataset)")
                fig.update_layout(xaxis_title="Harga (Rp)", yaxis_title="Jumlah listing", plot_bgcolor="white")
                st.plotly_chart(fig, use_container_width=True)
            with cc2:
                fig2 = px.box(view, x="cpu_tier", y="price_rp", points="all", color="cpu_tier",
                              category_orders={"cpu_tier": pc.CPU_TIER_ORDER}, color_discrete_sequence=CATEGORICAL_SEQUENCE,
                              title="Sebaran Harga & Outlier per Tier CPU")
                fig2.update_layout(xaxis_title="Tier CPU", yaxis_title="Harga (Rp)", showlegend=False, plot_bgcolor="white")
                st.plotly_chart(fig2, use_container_width=True)

            n_susp = int(listings.is_suspicious.sum())
            st.warning(f"⚠️ {n_susp} listing pada dataset mentah ditandai **berpotensi tidak wajar / perlu verifikasi** (outlier bawah + rating/terjual minim) — dikeluarkan dari benchmark dataset di atas.")

            with st.expander("Sebaran per merek (opsional — konteks pasar, bukan dasar rekomendasi vendor)"):
                fig3 = px.box(view, x="brand", y="price_rp", points="all", color_discrete_sequence=[ORANGE])
                fig3.update_layout(xaxis_title="Merek", yaxis_title="Harga (Rp)", plot_bgcolor="white")
                st.plotly_chart(fig3, use_container_width=True)
                st.caption("Sistem tidak merekomendasikan merek/vendor tertentu meskipun harganya terendah — lihat prinsip di tab Governance.")

    with tabs[1]:
        st.markdown("##### Raw Listings → Cleaned → Valid → Benchmark Dataset")
        funnel_df = pc.data_quality_funnel(listings)
        fig = go.Figure(go.Funnel(y=funnel_df["stage"], x=funnel_df["count"], marker=dict(color=[NAVY, "#4C82A6", ORANGE, "#F4A65A"])))
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)
        qc1, qc2, qc3, qc4 = st.columns(4)
        qc1.metric("Duplicate listing", int(listings.is_duplicate_listing.sum()))
        qc2.metric("Bundle/promo produk", int((listings.is_bundle | listings.is_promo).sum()))
        qc3.metric("Spesifikasi tidak lengkap", int(listings.missing_spec.sum()))
        qc4.metric("Outlier harga (IQR)", int(listings.is_price_outlier.sum()))
        st.dataframe(listings[listings.listing_status != "valid"][["listing_id", "title", "price_rp", "rating", "sold_min", "listing_status", "data_type"]],
                     use_container_width=True, hide_index=True)
        st.caption("Harga marketplace adalah **benchmark/referensi pasar**, bukan otomatis harga akhir pengadaan BNI — lihat kondisi vendor/garansi/kondisi barang sebelum dipakai sebagai acuan kontrak.")

    with tabs[2]:
        st.markdown("##### \"BNI butuh 100 laptop dengan spesifikasi tertentu — berapa benchmark & potensi penghematannya?\"")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            cpu_sel = st.selectbox("Tier CPU", pc.CPU_TIER_ORDER, index=1, format_func=lambda x: pc.CPU_TIER_LABEL[x])
        with c2:
            ram_sel = st.selectbox("RAM min. (GB)", [4, 8, 16, 32], index=2)
        with c3:
            sto_sel = st.selectbox("Storage min. (GB)", [128, 256, 512, 1024], index=2)
        with c4:
            gpu_sel = st.selectbox("GPU diskrit", [False, True], format_func=lambda x: "Ya" if x else "Tidak")
        custom_qty = st.number_input("Jumlah laptop yang dibutuhkan", min_value=1, value=100, step=1)

        table = pc.volume_scenario_table(valid_listings, cpu_sel, ram_sel, sto_sel, gpu_sel, scenarios)
        if table.empty:
            st.error("Tidak ada data pembanding.")
        else:
            st.info(f"Cakupan data: **{table.attrs['match_level']}** (n={table.attrs['n_sample']})")
            custom_qty = int(custom_qty)
            applicable_scenario = scenarios[scenarios.qty_tier <= custom_qty].iloc[-1] if (scenarios.qty_tier <= custom_qty).any() else scenarios.iloc[0]
            custom_discount = float(applicable_scenario.assumed_discount_pct) / 100.0
            custom_benchmark = float(table.benchmark_unit_price.iloc[0])
            custom_unit_price = custom_benchmark * (1 - custom_discount)
            custom_total = custom_unit_price * custom_qty
            custom_saving = (custom_benchmark * custom_qty) - custom_total
            result_col1, result_col2, result_col3, result_col4 = st.columns(4)
            result_col1.metric("Benchmark unit price", pc.fmt_rp(custom_benchmark))
            result_col2.metric("Harga unit setelah diskon", pc.fmt_rp(custom_unit_price))
            result_col3.metric("Estimasi total", pc.fmt_rp(custom_total))
            result_col4.metric("Potensi penghematan", pc.fmt_rp(custom_saving))
            st.caption(f"Jumlah {custom_qty:,} unit memakai tier diskon {int(applicable_scenario.qty_tier):,} unit ({applicable_scenario.assumed_discount_pct:.0f}%).")
            fig = go.Figure()
            fig.add_trace(go.Bar(x=table.qty, y=table.estimated_total, name="Estimasi total (dengan asumsi diskon)", marker_color=ORANGE))
            fig.add_trace(go.Scatter(x=table.qty, y=table.benchmark_unit_price * table.qty, name="Tanpa diskon (linear)", line=dict(color=NAVY, dash="dot")))
            fig.add_trace(go.Scatter(x=[custom_qty], y=[custom_total], mode="markers+text", name="Jumlah Anda",
                                     text=[f"{custom_qty:,} unit"], textposition="top center",
                                     marker=dict(size=12, color=GREEN, line=dict(width=2, color="white"))))
            fig.update_layout(xaxis_title="Jumlah unit", yaxis_title="Estimasi total (Rp)", plot_bgcolor="white",
                               title="Required Quantity → Market Benchmark → Expected Procurement Price → Potential Savings")
            st.plotly_chart(fig, use_container_width=True)
            disp = table.copy()
            custom_row = pd.DataFrame([dict(
                qty=custom_qty, benchmark_unit_price=custom_benchmark, assumed_discount_pct=applicable_scenario.assumed_discount_pct,
                unit_price_after_discount=custom_unit_price, estimated_total=custom_total,
                potential_saving_vs_no_discount=custom_saving, rationale="Jumlah manual Anda",
            )])
            disp = pd.concat([disp, custom_row], ignore_index=True)
            for c in ["benchmark_unit_price", "unit_price_after_discount", "estimated_total", "potential_saving_vs_no_discount"]:
                disp[c] = disp[c].apply(pc.fmt_rp)
            st.dataframe(disp[["qty", "benchmark_unit_price", "assumed_discount_pct", "unit_price_after_discount",
                                "estimated_total", "potential_saving_vs_no_discount", "rationale"]],
                         use_container_width=True, hide_index=True)
            st.caption("Kolom **assumed_discount_pct** adalah **asumsi skenario** (procurement_scenarios.csv), bukan angka yang dijamin vendor atau ditetapkan regulasi — lihat tab Governance & Data-Metodologi.")

    with tabs[3]:
        st.markdown("##### Bagaimana harga berubah seiring perubahan spesifikasi?")
        g1, g2 = st.columns(2)
        with g1:
            agg = valid_listings.groupby("ram_gb")["price_rp"].median().reset_index()
            fig = px.line(agg, x="ram_gb", y="price_rp", markers=True, color_discrete_sequence=[NAVY], title="Median Harga vs RAM")
            fig.update_layout(xaxis_title="RAM (GB)", yaxis_title="Median harga (Rp)", plot_bgcolor="white")
            st.plotly_chart(fig, use_container_width=True)
        with g2:
            agg2 = valid_listings.groupby("storage_gb")["price_rp"].median().reset_index()
            fig2 = px.line(agg2, x="storage_gb", y="price_rp", markers=True, color_discrete_sequence=[ORANGE], title="Median Harga vs Storage")
            fig2.update_layout(xaxis_title="Storage (GB)", yaxis_title="Median harga (Rp)", plot_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)
        fig3 = px.scatter(valid_listings, x="ram_gb", y="price_rp", color="cpu_tier", size="storage_gb",
                           category_orders={"cpu_tier": pc.CPU_TIER_ORDER}, color_discrete_sequence=CATEGORICAL_SEQUENCE,
                           title="Peta Spesifikasi vs Harga (ukuran bubble = storage)")
        fig3.update_layout(plot_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)

    with tabs[4]:
        st.markdown("##### Normalisasi pajak agar harga marketplace sebanding dengan struktur HPS")
        sample_price = float(valid_listings.price_rp.median())
        st.markdown(tax_normalization_flow_html(pc.fmt_rp(sample_price), "÷ 1.12 (PPN 12%)" if include_ppn else "sudah tidak termasuk PPN",
                                                 pc.fmt_rp(sample_price / 1.12 if include_ppn else sample_price)), unsafe_allow_html=True)
        st.write("")
        st.markdown(assumptions[assumptions.assumption_id == "TAX-PPN"][["description", "value", "status", "source", "notes"]].to_html(index=False), unsafe_allow_html=True)

# =============================================================================================
# PAGE 3 — LEGAL & RAG INTELLIGENCE
# =============================================================================================
elif page == PAGES[2]:
    st.markdown('<div class="section-tag">AI-generated outputs become trustworthy when grounded in traceable sources</div>', unsafe_allow_html=True)

    citation_cov = rm.citation_coverage_pct(rag_answers)
    docs_processed = pilot_metrics.loc["TOTAL_DOCS_PROCESSED", "value"]
    kpi_row([
        dict(label="Answers in sample (this prototype)", value=len(rag_answers), decimals=0, data_type="synthetic"),
        dict(label="Citations logged (this prototype)", value=len(citations), decimals=0, data_type="synthetic"),
    ])
    st.caption("source_type & confidence_score disampel dari probabilitas yang didokumentasikan di `assumptions.csv` (SOURCE-TYPE-PROB, CONFIDENCE-BAND-PROB), lalu persentase di bawah dihitung dari hasil sampling itu — bukan dipaksa sama persis dengan asumsinya.")

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Answer Confidence Distribution")
        conf = rm.confidence_distribution(rag_answers)
        fig = px.pie(conf, names="band", values="pct", hole=0.55, color="band",
                     color_discrete_map={rm.CONF_LABELS[2]: NAVY, rm.CONF_LABELS[1]: ORANGE, rm.CONF_LABELS[0]: AMBER})
        fig.update_traces(textinfo="percent+label")
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"Asumsi input (assumptions.csv, CONFIDENCE-BAND-PROB): High 75% / Medium 20% / Low 5%. Hasil realisasi sampling (n={len(rag_answers)}) ditampilkan di chart — berbeda dari asumsi karena ini hasil random sampling, bukan dipaksa cocok. " + data_badge("calculated"), unsafe_allow_html=True)
    with c2:
        st.markdown("#### Where Answers Are Grounded")
        gsrc = rm.grounding_source_breakdown(rag_answers)
        fig2 = px.bar(gsrc, x="pct", y="source", orientation="h", color_discrete_sequence=[NAVY], text=gsrc["pct"].apply(lambda v: f"{v}%"))
        fig2.update_traces(textposition="outside")
        fig2.update_layout(xaxis_title="% jawaban", yaxis_title="", plot_bgcolor="white", xaxis_range=[0, 100])
        st.plotly_chart(fig2, use_container_width=True)
        st.caption(f"Asumsi input (SOURCE-TYPE-PROB): Internal KB 70% / Dokumen pengguna 20% / Web eksternal 10%. Retrieval internal-first mengurangi ketergantungan pada web eksternal; web search hanya dipicu pada kasus low-confidence. " + data_badge("calculated"), unsafe_allow_html=True)

    st.write("")
    st.markdown("#### Provenance Chain — Contoh Kasus")
    st.caption("Ini bukan sekadar \"AI menjawab\" — setiap klaim faktual bisa ditelusuri balik ke klausul & halaman persis sumbernya.")
    chain = rm.provenance_example(rag_answers, citations, legal_docs)
    st.markdown(provenance_chain_html(chain), unsafe_allow_html=True)

    st.write("")
    c3, c4 = st.columns(2)
    with c3:
        st.markdown("#### Document Preparation Time by Complexity")
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(x=doc_prep.complexity, y=doc_prep.before_minutes, name="Before AI", marker_color=NAVY))
        fig3.add_trace(go.Bar(x=doc_prep.complexity, y=doc_prep.after_minutes, name="With AI-assisted RAG", marker_color=ORANGE))
        fig3.update_layout(barmode="group", yaxis_title="Menit", plot_bgcolor="white")
        st.plotly_chart(fig3, use_container_width=True)
        st.caption("Hipotesis desain pengukuran untuk divalidasi pada pilot riil — belum ada time-motion study aktual karena sistemnya belum dibangun. " + data_badge("illustrative"), unsafe_allow_html=True)
    with c4:
        st.markdown("#### Answers Supported by Valid Citation (%), per Kategori Dokumen")
        fig4 = px.bar(cite_validity, x="pct_valid_citation", y="category", orientation="h", color_discrete_sequence=[ORANGE],
                      text=cite_validity["pct_valid_citation"].apply(lambda v: f"{v}%"))
        fig4.update_traces(textposition="outside")
        fig4.update_layout(xaxis_title="% didukung sitasi valid", yaxis_title="", plot_bgcolor="white", xaxis_range=[0, 100])
        st.plotly_chart(fig4, use_container_width=True)
        st.caption(f"Dihitung dari {len(citation_evals)} evaluasi sitasi individual di citation_evaluations.csv (lihat expander di bawah), bukan angka yang diketik langsung. " + data_badge("calculated"), unsafe_allow_html=True)

    with st.expander("🔎 Bagaimana persentase validitas sitasi ini dihitung?"):
        st.markdown("""
Setiap baris `citations.csv` dievaluasi satu per satu (`utils/roi_calc.py:build_citation_evaluations`):

1. Kategorikan dokumen sumber (`doc_type` → SOP&Sirkular / Kontrak&Perjanjian / Referensi Regulasi).
2. Ambil **prior** probabilitas valid per kategori (asumsi, `assumptions.csv`: CITATION-PRIOR-*) — SOP=0.80, Kontrak=0.72, Regulasi=0.62 (SOP paling terstruktur & mudah dilacak persisnya, regulasi eksternal paling panjang & paling sulit).
3. Sesuaikan dengan confidence_score jawaban induk: `validity_prob = prior + (confidence_score − 0.7) × 0.35`, di-clip ke [0.05, 0.99].
4. Sampling Bernoulli `is_valid ~ Bernoulli(validity_prob)` dengan RNG seed=42 (reproducible).
5. `citation_validity_by_category.csv` = `groupby(category)["is_valid"].mean() × 100` — **hasil agregasi, bukan angka yang ditulis manual.**
        """)
        st.dataframe(citation_evals.head(15), use_container_width=True, hide_index=True)
        st.caption(f"Total {len(citation_evals)} baris evaluasi tersimpan penuh di `data/citation_evaluations.csv`.")

    st.write("")
    c5, c6 = st.columns(2)
    with c5:
        st.markdown("#### Historical/Legal/RAG Records Made Searchable (cumulative, pilot weeks)")
        fig5 = px.line(pilot_ramp, x="pilot_week", y="cumulative_records_searchable", markers=True, color_discrete_sequence=[NAVY])
        fig5.update_layout(xaxis_title="Pilot week", yaxis_title="Cumulative records", plot_bgcolor="white")
        st.plotly_chart(fig5, use_container_width=True)
        st.caption(f"Kurva logistik: total/(1+e^(−0.9×(week−3.5))), total = {int(pilot_metrics.loc['RECORDS_SEARCHABLE_TOTAL','value']):,} (RECORDS_SEARCHABLE_TOTAL). Parameter ramp adalah asumsi (assumptions.csv: RAMP-CURVE-FORMULA); totalnya dihitung, bukan diketik. " + data_badge("calculated"), unsafe_allow_html=True)
    with c6:
        st.markdown("#### Process Time — Before vs. AI-Assisted")
        pt = process_time.copy()
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(y=pt.process_name, x=pt.before_minutes, name="Manual baseline", orientation="h", marker_color="#C9D9E8"))
        fig6.add_trace(go.Bar(y=pt.process_name, x=pt.after_minutes, name="AI-assisted (pilot)", orientation="h", marker_color=ORANGE))
        fig6.update_layout(barmode="overlay", xaxis_title="Menit", plot_bgcolor="white")
        st.plotly_chart(fig6, use_container_width=True)
        st.dataframe(pt[["process_name", "before_minutes", "after_minutes", "pct_reduction"]].rename(
            columns={"pct_reduction": "% reduction"}), use_container_width=True, hide_index=True)

    with st.expander("Document registry (sample, synthetic)"):
        st.dataframe(legal_docs, use_container_width=True, hide_index=True)

# =============================================================================================
# PAGE 4 — GOVERNANCE & ANTI-CORRUPTION
# =============================================================================================
elif page == PAGES[3]:
    st.markdown('<div class="section-tag">Objective market evidence, not accusation — every flag requires human review</div>', unsafe_allow_html=True)
    st.caption("Terminologi yang digunakan: **potential anomaly**, **requires review**, **outside benchmark range**, **potential procurement risk indicator**. Tidak ada label \"korupsi\" hanya karena harga mahal.")

    split_flags = gov.detect_split_purchases(history)
    ev = gov.evidence_coverage(history)
    flagged_hist = pc.outside_benchmark_flags(history, listings)
    ob = gov.outside_benchmark_summary(flagged_hist)

    kpi_row([
        dict(label="Potential Duplicate/Split-Purchase Indicators", value=len(split_flags), decimals=0, data_type="calculated",
             note=f"Aturan: unit+spesifikasi+vendor sama, jarak ≤{gov.SPLIT_WINDOW_DAYS} hari, kedua order < Rp{gov.APPROVAL_THRESHOLD_RP/1e6:.0f}jt"),
        dict(label="Records with Benchmark Reference", value=ev["pct_with_benchmark"], suffix="%", decimals=1, data_type="calculated"),
        dict(label="Records with Supporting Documentation", value=ev["pct_with_doc"], suffix="%", decimals=1, data_type="calculated"),
        dict(label="Current Quotes Outside Benchmark Range", value=ob["pct_outside"], suffix="%", decimals=1, data_type="calculated"),
    ])

    st.write("")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.markdown("#### Illustrative Procurement Savings Opportunity by Specification Tier")
        st.caption("Catatan adaptasi ruang lingkup: materi asli membandingkan Server/Laptop/Printer/Monitor — versi ini **dibatasi laptop saja**, sehingga sumbu kategori diganti tier spesifikasi laptop (Entry/Mid/High/Premium), struktur analisis (Historical vs Current Quote vs Market Benchmark vs Negotiation Gap) tetap sama.")
        rows = []
        for tier in pc.CPU_TIER_ORDER:
            h = history[history.cpu_tier == tier]
            if h.empty:
                continue
            gpu_mode = bool(h.dedicated_gpu.mode()[0])
            sub, _ = pc.find_comparable(valid_listings, tier, int(h.ram_gb.mode()[0]), int(h.storage_gb.mode()[0]), gpu_mode, min_n=3)
            stats = pc.bucket_stats(sub)
            if stats is None:
                continue
            rows.append(dict(tier=pc.CPU_TIER_LABEL[tier].split(" (")[0], historical=h.historical_unit_price_rp.mean(),
                              current_quote=h.current_quote_unit_price_rp.mean(), market_benchmark=stats["median"],
                              negotiation_gap=h.current_quote_unit_price_rp.mean() - stats["median"]))
        gap_df = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_trace(go.Bar(y=gap_df.tier, x=gap_df.negotiation_gap, orientation="h", marker_color=ORANGE, name="Negotiation Gap"))
        fig.update_layout(xaxis_title="Negotiation gap = Current Quote − Market Benchmark (Rp)", yaxis_title="", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)
        disp = gap_df.copy()
        for c in ["historical", "current_quote", "market_benchmark", "negotiation_gap"]:
            disp[c] = disp[c].apply(pc.fmt_rp)
        st.dataframe(disp.rename(columns={"tier": "Tier", "historical": "Historical (BNI)", "current_quote": "Current Quote",
                                           "market_benchmark": "Market Benchmark", "negotiation_gap": "Negotiation Gap"}),
                     use_container_width=True, hide_index=True)
        st.caption("Values above are constructed pilot scenarios used to demonstrate the comparison capability — not confirmed BNI transaction data. " + data_badge("synthetic"), unsafe_allow_html=True)
    with c2:
        st.markdown("#### Potential Duplicate / Split-Purchase Pairs")
        if split_flags.empty:
            st.info("Tidak ada pasangan yang cocok dengan aturan saat ini.")
        else:
            st.dataframe(split_flags[["business_unit", "spec_id", "vendor", "date_1", "date_2", "gap_days", "combined_value"]]
                         .assign(combined_value=lambda d: d.combined_value.apply(pc.fmt_rp))
                         .rename(columns={"combined_value": "Combined Value"}), use_container_width=True, hide_index=True, height=380)

    st.write("")
    st.markdown("#### Current Quotes Outside Market Benchmark Range")
    outside = flagged_hist[flagged_hist.outside_benchmark].copy()
    if outside.empty:
        st.info("Tidak ada current quote di luar rentang benchmark pasar saat ini.")
    else:
        outside["risk_label"] = "⚠️ Outside benchmark range — requires additional evidence"
        show = outside[["po_id", "business_unit", "spec_id", "vendor", "current_quote_unit_price_rp", "benchmark_median",
                         "deviation_pct", "risk_label"]].sort_values("deviation_pct", ascending=False).head(20)
        show["current_quote_unit_price_rp"] = show["current_quote_unit_price_rp"].apply(pc.fmt_rp)
        show["benchmark_median"] = show["benchmark_median"].apply(pc.fmt_rp)
        show["deviation_pct"] = show["deviation_pct"].round(1).astype(str) + "%"
        st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption("Governance concepts visualized: price outside market benchmark · unusual price deviations · duplicate purchase opportunities · lack of benchmark evidence · missing documentation. Setiap baris memerlukan verifikasi manusia sebelum kesimpulan apa pun diambil.")

# =============================================================================================
# PAGE 5 — DATA & METHODOLOGY
# =============================================================================================
elif page == PAGES[4]:
    st.markdown('<div class="section-tag">Transparency in, transparency out</div>', unsafe_allow_html=True)

    st.markdown("#### Ringkasan Sumber Data")
    st.markdown(f"""
    | Dataset | Baris | Tipe Data |
    |---|---|---|
    | marketplace_listings.csv | {len(listings)} ({int((listings.data_type=='observed').sum())} observed, {int((listings.data_type=='synthetic').sum())} synthetic) | Campuran |
    | procurement_history.csv | {len(history)} | Synthetic |
    | procurement_laptops.csv | {len(catalog)} | Internal assumption |
    | procurement_scenarios.csv | {len(scenarios)} | Assumption |
    | legal_documents.csv | {len(legal_docs)} | Synthetic |
    | rag_answers.csv | {len(rag_answers)} | Synthetic (confidence/source disampel dari probabilitas, bukan diketik) |
    | citations.csv | {len(citations)} | Synthetic |
    | citation_evaluations.csv | {len(citation_evals)} | **Calculated** — evaluasi validitas per sitasi |
    | investment_breakdown.csv | {len(investment_breakdown)} | Assumption |
    | process_annual_volume.csv | {len(process_annual_volume)} | Assumption |
    | roi_scenarios.csv | {len(roi_scenarios)} | **Calculated** — lihat expander perhitungan di Executive Summary |
    | pilot_metrics.csv | {len(pilot_metrics)} | **Calculated** — setiap value = hasil hitung/count aktual |
    """)
    st.info("Dua input memakai sumber eksternal nyata (bukan BNI-spesifik, lihat assumptions.csv untuk detail): **jumlah karyawan BNI (27.201)** dari infobox Wikipedia yang mengutip profil perusahaan BNI, dan **benchmark gaji Compliance Officer Indonesia (Rp70,7 juta/tahun)** dari Payscale.com (crowdsourced, indikatif).")

    st.markdown("#### Assumptions Register (22 baris — baca ini untuk setiap sumber/rationale)")
    st.dataframe(assumptions, use_container_width=True, hide_index=True)

    st.markdown("#### Calculation Methodology")
    st.markdown("""
- **Market benchmark** = *median* harga listing berstatus `valid` (lolos data-quality pipeline) yang cocok dengan spesifikasi terpilih.
- **Outlier deteksi** = pagar 1.5×IQR per bucket spesifikasi (tier CPU × RAM × storage).
- **Potential savings (volume, tab Procurement)** = benchmark unit price × qty × asumsi diskon volume (procurement_scenarios.csv) — **bukan** angka yang dijamin vendor.
- **Citation validity per kategori** = simulasi Bernoulli per sitasi: `validity_prob = prior_kategori + (confidence_score − 0.7) × 0.35`, lalu `pct = groupby(kategori).mean()`. Lihat `citation_evaluations.csv` untuk tiap baris evaluasinya.
- **Citation coverage** = % baris rag_answers.csv dengan `has_citation=True`.
- **Confidence bands** = High >0.85, Medium 0.60–0.85, Low <0.60 dari `confidence_score` (disampel dari probabilitas di assumptions.csv, bukan dipaksa).
- **Time-savings benefit** = Σ proses [((menit_before − menit_after)/60) × estimated_annual_occurrences × adoption_rate] × tarif_per_jam. Tarif per jam = gaji tahunan benchmark ÷ 12 ÷ 173 jam kerja/bulan.
- **Procurement-savings benefit** = max(rata-rata current quote BNI − benchmark median pasar, 0) × volume_pengadaan_tahunan × adoption_rate. Volume tahunan = jumlah karyawan BNI × % eligible laptop × refresh rate/tahun.
- **ROI skenario** = (Time-savings + Procurement-savings − Pilot Investment) / Pilot Investment; **Payback (bulan)** = Pilot Investment / Total Benefit × 12. Rincian penuh & live recompute ada di expander "Tunjukkan perhitungannya" (tab Executive Summary).
- **Potential duplicate/split-purchase** = dua PO unit+spesifikasi+vendor sama, jarak tanggal ≤14 hari, kedua nilai order < Rp200 juta (ambang contoh approval Direktur).
- **Outside benchmark range** = current quote BNI > pagar atas IQR (Q3+1.5×IQR) dari benchmark pasar untuk spesifikasi yang sama.
    """)

    st.markdown("#### Dasar Regulasi & Sumber")
    st.markdown("""
- Perpres No.16/2018 jo. Perpres No.12/2021 jo. Perpres No.46/2025 tentang Pengadaan Barang/Jasa Pemerintah — dasar penyusunan HPS; untuk barang, HPS lazim disusun dari survei harga pasar.
- **Tidak ditemukan** angka margin/keuntungan vendor yang dibakukan secara persentase dalam regulasi pengadaan barang yang ditelusuri — setiap angka margin/diskon di dashboard ini adalah **asumsi**, bukan fakta regulasi.
- UU No.7/2021 tentang Harmonisasi Peraturan Perpajakan (UU HPP) — tarif PPN umum naik ke 12% berlaku sejak 1 Januari 2025.
- Data marketplace: hasil pencarian langsung Tokopedia (`tokopedia.com/find/laptop-ram-16gb-ssd-512gb`, `.../laptop-i5-ryzen-5`), diakses 27 Agustus 2026.
- Referensi kelembagaan pengadaan: LKPP (Lembaga Kebijakan Pengadaan Barang/Jasa Pemerintah); referensi pajak: DJP (Direktorat Jenderal Pajak).
    """)

    st.markdown("#### Data Replacement Guide (ringkas — detail penuh di README.md)")
    st.markdown("""
1. Ganti isi CSV di `data/` dengan data nyata **dengan skema kolom yang sama** — kode di `utils/` dan `app.py` tidak perlu diubah.
2. `marketplace_listings.csv`: tambahkan baris real dengan `data_type="observed"`; pipeline data-quality otomatis menghitung ulang.
3. `procurement_history.csv`: ganti dengan data PO aktual BNI (redaksi info sensitif bila perlu) — kolom vendor bisa tetap dianonimkan jadi "Vendor A/B/C".
4. `rag_answers.csv` / `citations.csv` / `legal_documents.csv`: ganti dengan log RAG produksi sungguhan begitu tersedia.
5. `assumptions.csv` / `procurement_scenarios.csv`: perbarui begitu ada data historis negosiasi vendor riil atau ketentuan internal resmi.
    """)

    with st.expander("Lihat semua tabel mentah"):
        for name, df in [("marketplace_listings", listings), ("procurement_history", history), ("procurement_laptops", catalog),
                          ("procurement_scenarios", scenarios), ("legal_documents", legal_docs), ("rag_answers", rag_answers),
                          ("citations", citations), ("citation_evaluations", citation_evals),
                          ("investment_breakdown", investment_breakdown), ("process_annual_volume", process_annual_volume),
                          ("pilot_metrics", dl.load_pilot_metrics()), ("business_unit_impact", bu_impact),
                          ("roi_scenarios", roi_scenarios)]:
            st.markdown(f"**{name}.csv** ({len(df)} baris)")
            st.dataframe(df, use_container_width=True, hide_index=True)

st.divider()
st.caption("Internal prototype — semua figur Illustrative/Synthetic/Assumption bukan data resmi BNI. Lihat tab Data & Metodologi untuk sumber dan formula lengkap.")
