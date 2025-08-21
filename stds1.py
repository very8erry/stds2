# -*- coding: utf-8 -*-
# Streamlit 대시보드 (PyCharm에서 실행 → 이후 Cloud 업로드 가능)
# 실행: streamlit run app.py

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="경영 대시보드", layout="wide")

# -----------------------------
# 색상 시스템
# -----------------------------
COLORS = {
    "primary600": "#2563EB",
    "primary700": "#1E40AF",
    "primary100": "#DBEAFE",
    "neutral900": "#111827",
    "neutral400": "#9CA3AF",
    "neutral50":  "#F9FAFB",
    "success600": "#16A34A",
    "warning600": "#CA8A04",
    "danger600":  "#DC2626",
}

# -----------------------------
# 유틸
# -----------------------------
def kpi_card(label, value, delta_text=None, positive=True):
    """심플 KPI 카드"""
    bg = "#FFFFFF"; border = "#E5E7EB"; title = COLORS["neutral900"]; num = COLORS["primary700"]
    up = "▲"; down = "▼"; sig = up if positive else down
    color = COLORS["success600"] if positive else COLORS["danger600"]
    st.markdown(f"""
    <div style="padding:16px;border:1px solid {border};border-radius:16px;background:{bg}">
      <div style="font-size:13px;color:{title};margin-bottom:6px">{label}</div>
      <div style="font-size:28px;font-weight:800;color:{num};line-height:1.2">{value}</div>
      {f'<div style="color:{color};font-size:12px;margin-top:6px">{sig} {delta_text}</div>' if delta_text else ''}
    </div>
    """, unsafe_allow_html=True)

def pct(a, b):
    if b == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return (a - b) / b

def safe_div(a, b):
    return np.where(b == 0, np.nan, a / b)

# 공통 레이아웃 옵션 (제목/여백 규격화로 겹침 방지)
def apply_layout(fig, title_text, h=420, tpad=80, bpad=60):
    fig.update_layout(
        height=h,
        margin=dict(l=20, r=20, t=tpad, b=bpad),
        paper_bgcolor="white",
        plot_bgcolor="white",
        title=dict(
            text=title_text,
            y=0.98,              # 상단에 제목 띄우기 (겹침 방지)
            x=0.02,              # 좌측 정렬 느낌
            xanchor="left",
            yanchor="top",
            font=dict(size=18, color=COLORS["neutral900"], family="Arial, sans-serif")
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0)
    )

# 스파크라인 (상단 카드 아래 미니 추세) — 겹침 방지 버전
def sparkline(x, y, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(width=2, color=COLORS["primary600"])))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(
        height=160,
        margin=dict(l=10, r=10, t=46, b=24),  # 위 여백 확대
        paper_bgcolor="white",
        plot_bgcolor="white",
        showlegend=False,
        title=dict(
            text=title,
            y=0.98, x=0.02, xanchor="left", yanchor="top",
            font=dict(size=14, color=COLORS["neutral900"])
        )
    )
    return fig

# -----------------------------
# 사이드바: 업로드 & 필터
# -----------------------------
st.sidebar.header("1) 데이터 업로드")
uploaded = st.sidebar.file_uploader("CSV 업로드 (예: KPI_Master_Small_12M_KR.csv)", type=["csv"])
if uploaded is None:
    st.info("좌측에서 CSV를 업로드하세요. (샘플 스키마: 월/매출액/매출총이익/영업이익/마케팅비용/전환율/CSAT/채널/제품카테고리/고객세그먼트/지역/정시배송율/SLA위반건수 등)")
    st.stop()

# CSV 로드
try:
    df = pd.read_csv(uploaded)
except UnicodeDecodeError:
    uploaded.seek(0)
    df = pd.read_csv(uploaded, encoding="cp949")

# 전처리
if "월" in df.columns:
    df["월"] = pd.to_datetime(df["월"], errors="coerce", format="%Y-%m").fillna(pd.to_datetime(df["월"], errors="coerce"))
    df = df.sort_values("월")
else:
    df["월"] = pd.to_datetime("today").normalize()

if {"매출총이익","매출액"}.issubset(df.columns):
    df["매출총이익률"] = safe_div(df["매출총이익"], df["매출액"])
if {"매출총이익","마케팅비용"}.issubset(df.columns):
    df["ROI"] = safe_div(df["매출총이익"], df["마케팅비용"])

# 필터
st.sidebar.header("2) 필터")
region = st.sidebar.multiselect("지역", sorted(df["지역"].dropna().unique()) if "지역" in df.columns else [])
segment = st.sidebar.multiselect("세그먼트", sorted(df["고객세그먼트"].dropna().unique()) if "고객세그먼트" in df.columns else [])
channel = st.sidebar.multiselect("채널", sorted(df["채널"].dropna().unique()) if "채널" in df.columns else [])
category = st.sidebar.multiselect("제품카테고리", sorted(df["제품카테고리"].dropna().unique()) if "제품카테고리" in df.columns else [])

filtered = df.copy()
if region:   filtered = filtered[filtered["지역"].isin(region)]
if segment:  filtered = filtered[filtered["고객세그먼트"].isin(segment)]
if channel:  filtered = filtered[filtered["채널"].isin(channel)]
if category: filtered = filtered[filtered["제품카테고리"].isin(category)]

# -----------------------------
# 상단
# -----------------------------
st.title("📊 경영 대시보드 (Executive · Deep Dive · Risk · What-if)")

with st.expander("데이터 미리보기", expanded=False):
    st.dataframe(filtered.head(50), use_container_width=True)

# -----------------------------
# 1) Executive Summary
# -----------------------------
st.subheader("1) Executive Summary")

agg = filtered.groupby("월", dropna=False).agg({
    "매출액":"sum",
    "매출총이익":"sum",
    "영업이익":"sum",
    "CSAT":"mean",
    "마케팅비용":"sum"
}).reset_index()
agg["ROI"] = safe_div(agg["매출총이익"], agg["마케팅비용"])

def last_and_delta(series):
    if len(series) == 0:
        return np.nan, None
    v_now = series.iloc[-1]
    if len(series) == 1:
        return v_now, None
    v_prev = series.iloc[-2]
    delta = pct(v_now, v_prev)
    return v_now, delta

m_now, m_delta = last_and_delta(agg["매출액"])
g_now, g_delta = last_and_delta(agg["매출총이익"])
o_now, o_delta = last_and_delta(agg["영업이익"])
c_now, c_delta = last_and_delta(agg["CSAT"])
r_now, r_delta = last_and_delta(agg["ROI"])

col1, col2, col3, col4, col5 = st.columns(5)
with col1: kpi_card("매출액", f"{m_now:,.0f}" if pd.notna(m_now) else "-", f"MoM {m_delta*100:.1f}%" if m_delta is not None else None, positive=(m_delta or 0) >= 0)
with col2: kpi_card("매출총이익", f"{g_now:,.0f}" if pd.notna(g_now) else "-", f"MoM {g_delta*100:.1f}%" if g_delta is not None else None, positive=(g_delta or 0) >= 0)
with col3: kpi_card("영업이익", f"{o_now:,.0f}" if pd.notna(o_now) else "-", f"MoM {o_delta*100:.1f}%" if o_delta is not None else None, positive=(o_delta or 0) >= 0)
with col4: kpi_card("CSAT", f"{c_now:.2f}" if pd.notna(c_now) else "-", f"MoM {c_delta*100:.1f}%" if c_delta is not None else None, positive=(c_delta or 0) >= 0)
with col5: kpi_card("ROI", f"{r_now:.2f}" if pd.notna(r_now) else "-", f"MoM {r_delta*100:.1f}%" if r_delta is not None else None, positive=(r_delta or 0) >= 0)

# 스파크라인 (겹침 방지)
spark1, spark2, spark3, spark4, spark5 = st.columns(5)
with spark1: st.plotly_chart(sparkline(agg["월"], agg["매출액"], "매출액 추세"), use_container_width=True)
with spark2: st.plotly_chart(sparkline(agg["월"], agg["매출총이익"], "매출총이익 추세"), use_container_width=True)
with spark3: st.plotly_chart(sparkline(agg["월"], agg["영업이익"], "영업이익 추세"), use_container_width=True)
with spark4: st.plotly_chart(sparkline(agg["월"], agg["CSAT"], "CSAT 추세"), use_container_width=True)
with spark5: st.plotly_chart(sparkline(agg["월"], agg["ROI"], "ROI 추세"), use_container_width=True)

st.markdown("---")

# -----------------------------
# 2) 현황분석 (탭)
# -----------------------------
st.subheader("2) 현황분석 (Deep Dive)")
tab1, tab2, tab3, tab4 = st.tabs(["매출총이익", "영업이익", "CSAT", "ROI·전환율"])

with tab1:
    if {"제품카테고리","매출총이익","매출액"}.issubset(filtered.columns):
        cat = filtered.groupby("제품카테고리", dropna=False).agg({"매출총이익":"sum","매출액":"sum"}).reset_index()
        cat["GM%"] = safe_div(cat["매출총이익"], cat["매출액"]) * 100
        fig = px.bar(cat.sort_values("GM%", ascending=False), x="제품카테고리", y="GM%",
                     color_discrete_sequence=[COLORS["primary600"]])
        apply_layout(fig, "제품카테고리별 Gross Margin (%)", h=420, tpad=90, bpad=70)
        fig.update_yaxes(title="GM %", tickformat=".1f")
        fig.update_xaxes(title="")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("필요 컬럼: 제품카테고리, 매출총이익, 매출액")

with tab2:
    if {"월","영업이익"}.issubset(agg.columns):
        fig = go.Figure()
        fig.add_hline(y=0, line=dict(color=COLORS["neutral400"], width=1))
        fig.add_trace(go.Scatter(x=agg["월"], y=agg["영업이익"], mode="lines+markers",
                                 line=dict(color=COLORS["primary600"], width=3), name="영업이익"))
        # 음수 영역 음영
        neg = agg["영업이익"].clip(upper=0)
        fig.add_trace(go.Bar(x=agg["월"], y=neg, marker_color=COLORS["danger600"], opacity=0.2, name="적자"))
        apply_layout(fig, "월별 영업이익 추이", h=420, tpad=90, bpad=70)
        fig.update_yaxes(title="영업이익")
        fig.update_layout(barmode="overlay", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("필요 컬럼: 월, 영업이익")

with tab3:
    if {"고객세그먼트","CSAT"}.issubset(filtered.columns):
        seg = filtered.groupby("고객세그먼트", dropna=False)["CSAT"].mean().reset_index()
        fig1 = px.bar(seg.sort_values("CSAT", ascending=False), x="고객세그먼트", y="CSAT",
                      color_discrete_sequence=[COLORS["primary600"]])
        apply_layout(fig1, "세그먼트별 평균 CSAT", h=420, tpad=90, bpad=70)
        st.plotly_chart(fig1, use_container_width=True)
    else:
        st.info("필요 컬럼: 고객세그먼트, CSAT")

    if {"지역","고객세그먼트","CSAT"}.issubset(filtered.columns):
        pivot = filtered.pivot_table(index="지역", columns="고객세그먼트", values="CSAT", aggfunc="mean")
        fig2 = px.imshow(pivot, text_auto=True, aspect="auto", color_continuous_scale="Blues")
        apply_layout(fig2, "지역 × 세그먼트 CSAT 히트맵", h=480, tpad=90, bpad=80)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("필요 컬럼: 지역, 고객세그먼트, CSAT")

with tab4:
    if {"채널","ROI","전환율","매출액"}.issubset(filtered.columns):
        ch = filtered.groupby("채널", dropna=False).agg({"ROI":"mean","전환율":"mean","매출액":"sum"}).reset_index()
        fig = px.scatter(ch, x="전환율", y="ROI", size="매출액", color="채널",
                         hover_data={"매출액":":,", "전환율":":.2%", "ROI":":.2f"})
        apply_layout(fig, "채널별 ROI vs 전환율 (버블 크기=매출액)", h=480, tpad=90, bpad=80)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("필요 컬럼: 채널, ROI, 전환율, 매출액")

st.markdown("---")

# -----------------------------
# 3) 문제도출 (Risk Indicator)
# -----------------------------
st.subheader("3) 문제도출 (Risk Indicator)")

CSAT_MIN = 4.0
GM_MIN = 0.20
SLA_MAX = 0

alerts = []
if "CSAT" in agg.columns and not agg["CSAT"].empty and pd.notna(agg["CSAT"].iloc[-1]) and agg["CSAT"].iloc[-1] < CSAT_MIN:
    alerts.append(f"CSAT 하락: 현재 {agg['CSAT'].iloc[-1]:.2f} (< {CSAT_MIN})")

if {"매출총이익","매출액"}.issubset(agg.columns) and agg["매출액"].iloc[-1] != 0:
    gm_now = agg["매출총이익"].iloc[-1] / agg["매출액"].iloc[-1]
    if pd.notna(gm_now) and gm_now < GM_MIN:
        alerts.append(f"저마진 위험: 현재 총이익률 {gm_now*100:.1f}% (< {GM_MIN*100:.0f}%)")

if "영업이익" in agg.columns and not agg["영업이익"].empty:
    last3 = agg["영업이익"].tail(3).mean()
    if pd.notna(last3) and last3 < 0:
        alerts.append("영업이익 적자 확대: 최근 3개월 평균 < 0")

if "SLA위반건수" in filtered.columns:
    sla = filtered.groupby("월")["SLA위반건수"].sum().reset_index()
    if not sla.empty and sla["SLA위반건수"].iloc[-1] > SLA_MAX:
        alerts.append(f"SLA 위반 증가: 최근 {int(sla['SLA위반건수'].iloc[-1])}건")

if alerts:
    for a in alerts:
        st.markdown(
            f"<div style='padding:10px;border-radius:10px;background:{COLORS['danger600']};color:white;margin-bottom:6px'>{a}</div>",
            unsafe_allow_html=True
        )
else:
    st.success("현재 설정된 임계값 기준에서 즉시 경고 사항이 없습니다.")

# 보조 차트
cols_r1, cols_r2 = st.columns(2)
with cols_r1:
    if "CSAT" in agg.columns:
        fig = px.line(agg, x="월", y="CSAT")
        apply_layout(fig, f"CSAT 추이 (임계값 {CSAT_MIN})", h=420, tpad=90, bpad=70)
        fig.add_hline(y=CSAT_MIN, line_color=COLORS["warning600"])
        st.plotly_chart(fig, use_container_width=True)
with cols_r2:
    if "SLA위반건수" in filtered.columns:
        sla = filtered.groupby("월")["SLA위반건수"].sum().reset_index()
        fig = px.bar(sla, x="월", y="SLA위반건수")
        apply_layout(fig, "SLA 위반건수 (월별)", h=420, tpad=90, bpad=70)
        st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# -----------------------------
# 4) 해결방안 (Action Simulation)
# -----------------------------
st.subheader("4) 해결방안 (Action Simulation)")

colSim1, colSim2, colSim3 = st.columns(3)
with colSim1:
    st.caption("마케팅비용 조정 → ROI 영향")
    if {"매출총이익","마케팅비용"}.issubset(agg.columns) and agg.shape[0] > 0:
        slider = st.slider("마케팅비용 조정(%)", -30, 30, 0, step=5)
        base_mkt = agg["마케팅비용"].iloc[-1]
        base_gp  = agg["매출총이익"].iloc[-1]
        if base_mkt:
            new_mkt  = base_mkt * (1 + slider/100)
            new_roi  = (base_gp / new_mkt) if new_mkt != 0 else np.nan
            st.metric("현재 ROI", f"{(base_gp/base_mkt):.2f}")
            st.metric("시나리오 ROI", f"{new_roi:.2f}" if pd.notna(new_roi) else "-")
        else:
            st.info("마케팅비용이 0으로 계산할 수 없습니다.")
    else:
        st.info("필요 컬럼: 매출총이익, 마케팅비용")

with colSim2:
    st.caption("고마진 제품 비중 ↑ → 총이익률 변화 (가정)")
    mix = st.slider("고마진 믹스 변화(%p)", -20, 20, 0, step=5)
    if {"매출총이익","매출액"}.issubset(agg.columns) and agg.shape[0] > 0 and agg["매출액"].iloc[-1]:
        base_gm = agg["매출총이익"].iloc[-1]/agg["매출액"].iloc[-1]
        uplift_pp = mix * 0.15 / 10  # +10 → +1.5pp (임시 가정)
        new_gm = (base_gm*100 + uplift_pp)
        st.metric("현재 총이익률", f"{base_gm*100:.1f}%")
        st.metric("시나리오 총이익률", f"{new_gm:.1f}%")
    else:
        st.info("필요 컬럼: 매출총이익, 매출액")

with colSim3:
    st.caption("정시배송율 목표 → CSAT 변화 (가정)")
    if "정시배송율" in filtered.columns and "CSAT" in filtered.columns:
        target = st.slider("정시배송율 목표(%)", 80, 99, 95)
        recent = filtered.groupby("월").agg({"정시배송율":"mean","CSAT":"mean"}).reset_index().tail(6)
        if not recent.empty:
            csat_now = recent["CSAT"].iloc[-1]
            ontime_now = recent["정시배송율"].iloc[-1]*100
            predicted = csat_now + (target - ontime_now) * 0.03
            st.metric("현재 CSAT", f"{csat_now:.2f}")
            st.metric("시나리오 CSAT", f"{predicted:.2f}")
        else:
            st.info("최근 데이터가 부족합니다.")
    else:
        st.info("필요 컬럼: 정시배송율, CSAT")

st.markdown("---")

# -----------------------------
# Drill-Down: 영업이익 → 마케팅비용 Breakdown
# -----------------------------
st.subheader("즉각 Drill-Down: 영업이익 → 마케팅비용 Breakdown")

by_candidates = [c for c in ["채널", "지역", "제품카테고리", "고객세그먼트"] if c in filtered.columns]
metric_candidates = [c for c in ["마케팅비용","매출총이익","영업이익","매출액"] if c in filtered.columns]

by = st.selectbox("Breakdown 기준", by_candidates)
metric = st.selectbox("지표 선택", metric_candidates)

if by and metric:
    dd = filtered.groupby(by, dropna=False).agg({metric:"sum"}).reset_index()
    fig = px.bar(dd.sort_values(metric, ascending=False), x=by, y=metric,
                 color_discrete_sequence=[COLORS["primary600"]])
    apply_layout(fig, f"{by}별 {metric} Breakdown", h=420, tpad=90, bpad=70)
    st.plotly_chart(fig, use_container_width=True)

st.caption("※ 모든 차트에 충분한 상단 여백과 제목 위치를 적용해, 상단 카드/박스와 겹치지 않도록 조정했습니다.")
