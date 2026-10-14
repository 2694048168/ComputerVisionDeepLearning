#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File: 29_stock.py
@Python Version: 3.12.13
@Platform: PyTorch 2.2.1 + cu121
@Author: Wei Li (Ithaca)
@Date: 2026-10-14
@Contact: weili_yzzcq@163.com
@Blog: https://2694048168.github.io/blog/#/
@Version: V0.1
@License: Apache License Version 2.0, January 2004
    Copyright 2026. All rights reserved.

@Description:
2026年 A股 & 港股 国庆前后行情分析与可视化
数据源：Baostock（A股主流） + akshare（科创50/港股） + quotekit（北证50）
多源兜底，无需代理
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "PingFang SC"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.bbox"] = "tight"

# ==================== 1. 指数配置 ====================
BAOSTOCK_INDICES = {
    "上证指数": "sh.000001",
    "深证成指": "sz.399001",
    "创业板指": "sz.399006",
    "沪深300": "sh.000300",
}

# 科创50 / 北证50 使用专用获取函数
SPECIAL_INDICES = ["科创50", "北证50"]

HK_INDICES = {
    "恒生指数": "HSI",
    "恒生科技指数": "HSTECH",
}

START_DATE = "2026-08-15"
END_DATE = "2026-10-13"


# ==================== 2. 数据获取函数 ====================


def fetch_baostock(code: str, start: str, end: str) -> pd.DataFrame:
    """Baostock 获取A股指数"""
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"Baostock 登录失败: {lg.error_msg}")

    rs = bs.query_history_k_data_plus(
        code,
        "date,close",
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag="3",
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()

    if not rows:
        raise ValueError(f"Baostock 无数据: {code}")

    df = pd.DataFrame(rows, columns=["date", "close"])
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().set_index("date").sort_index()


def fetch_baostock_with_amount(code: str, start: str, end: str) -> pd.DataFrame:
    """Baostock 获取指数含成交额数据（用于成交额图表）"""
    import baostock as bs

    lg = bs.login()
    if lg.error_code != "0":
        raise RuntimeError(f"Baostock 登录失败: {lg.error_msg}")

    rs = bs.query_history_k_data_plus(
        code,
        "date,close,volume,amount",
        start_date=start,
        end_date=end,
        frequency="d",
        adjustflag="3",
    )
    rows = []
    while rs.error_code == "0" and rs.next():
        rows.append(rs.get_row_data())
    bs.logout()

    if not rows:
        raise ValueError(f"Baostock 无数据: {code}")

    df = pd.DataFrame(rows, columns=["date", "close", "volume", "amount"])
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    return df.dropna().set_index("date").sort_index()


def fetch_star50_akshare(start: str, end: str) -> pd.DataFrame:
    """科创50：使用 akshare 新浪源（stock_zh_index_daily）"""
    import akshare as ak

    # 新浪源代码格式：sh000688
    df = ak.stock_zh_index_daily(symbol="sh000688")

    if df is None or df.empty:
        raise ValueError("akshare 科创50 无数据")

    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df[["date", "close"]].set_index("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().sort_index()


def fetch_star50_pipeworx(start: str, end: str) -> pd.DataFrame:
    """科创50备选：Pipeworx API"""
    import requests

    url = "https://gateway.pipeworx.io/v1/tools/ashares_index_bars"
    resp = requests.get(url, timeout=15)
    schema = resp.json()

    days = 60  # 获取最近60个交易日
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "ashares_index_bars",
            "arguments": {"index": "000688", "days": days},
        },
    }
    r = requests.post(
        "https://gateway.pipeworx.io/china-exchange-data/mcp", json=payload, timeout=20
    )
    data = r.json()

    # 解析返回结构
    content = data.get("result", {}).get("content", [])
    if not content:
        raise ValueError("Pipeworx 科创50 无数据")

    bars = content[0].get("text", "")
    import json

    bars_list = json.loads(bars) if isinstance(bars, str) else bars

    df = pd.DataFrame(bars_list)
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df[["date", "close"]].set_index("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().sort_index()


def fetch_bj50_quotekit(start: str, end: str) -> pd.DataFrame:
    """北证50：使用 quotekit（通达信协议）"""
    from quotekit import get_bars

    # 北证50代码 899050
    df = get_bars("899050", freq="day", sd=start, ed=end)

    if df is None or df.empty:
        raise ValueError("quotekit 北证50 无数据")

    # quotekit 返回列名标准化
    if "datetime" in df.columns:
        df["date"] = pd.to_datetime(df["datetime"])
    elif "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
    else:
        df["date"] = pd.to_datetime(df.index)

    df = df[["date", "close"]].set_index("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().sort_index()


def fetch_bj50_akshare(start: str, end: str) -> pd.DataFrame:
    """北证50备选：akshare"""
    import akshare as ak

    df = ak.stock_zh_index_daily(symbol="bj899050")
    if df is None or df.empty:
        raise ValueError("akshare 北证50 无数据")

    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df[["date", "close"]].set_index("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().sort_index()


def fetch_hk_akshare(symbol: str, start: str, end: str) -> pd.DataFrame:
    """港股指数：akshare（东方财富港股源）"""
    import akshare as ak

    df = ak.stock_hk_index_daily_em(symbol=symbol)

    if df is None or df.empty:
        raise ValueError(f"akshare 港股 {symbol} 无数据")

    # 动态识别日期列
    date_col = None
    for c in ["date", "trade_date", "日期"]:
        if c in df.columns:
            date_col = c
            break
    if date_col is None:
        raise ValueError(f"akshare 港股无日期列: {list(df.columns)}")

    df["date"] = pd.to_datetime(df[date_col])
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df[["date", "close"]].set_index("date")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().sort_index()


def fetch_hk_yfinance(symbol: str, start: str, end: str) -> pd.DataFrame:
    """港股备选：yfinance（需代理）"""
    import yfinance as yf

    yf_symbol = "^HSI" if symbol == "HSI" else "^HSTECH"
    df = yf.download(yf_symbol, start=start, end=end, progress=False)

    if df is None or df.empty:
        raise ValueError(f"yfinance {symbol} 无数据")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[["Close"]].rename(columns={"Close": "close"})
    df.index.name = "date"
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    return df.dropna().sort_index()


def load_all_data(start=START_DATE, end=END_DATE):
    """多源兜底加载所有指数"""
    data = {}

    # --- A股主流指数：Baostock ---
    print("=" * 55)
    print("加载 A股主流指数 (Baostock)")
    print("=" * 55)
    for name, code in BAOSTOCK_INDICES.items():
        try:
            data[name] = fetch_baostock(code, start, end)
            print(f"  ✓ {name} — {len(data[name])} 条")
        except Exception as e:
            print(f"  ✗ {name} 失败: {e}")

    # --- 科创50：akshare → Pipeworx → efinance ---
    print("\n" + "=" * 55)
    print("加载 科创50 (akshare → Pipeworx)")
    print("=" * 55)
    try:
        data["科创50"] = fetch_star50_akshare(start, end)
        print(f"  ✓ 科创50 (akshare) — {len(data['科创50'])} 条")
    except Exception as e:
        print(f"  ✗ 科创50 akshare 失败: {e}")
        try:
            data["科创50"] = fetch_star50_pipeworx(start, end)
            print(f"  ✓ 科创50 (Pipeworx) — {len(data['科创50'])} 条")
        except Exception as e2:
            print(f"  ✗ 科创50 Pipeworx 也失败: {e2}")

    # --- 北证50：quotekit → akshare ---
    print("\n" + "=" * 55)
    print("加载 北证50 (quotekit → akshare)")
    print("=" * 55)
    try:
        data["北证50"] = fetch_bj50_quotekit(start, end)
        print(f"  ✓ 北证50 (quotekit) — {len(data['北证50'])} 条")
    except Exception as e:
        print(f"  ✗ 北证50 quotekit 失败: {e}")
        try:
            data["北证50"] = fetch_bj50_akshare(start, end)
            print(f"  ✓ 北证50 (akshare) — {len(data['北证50'])} 条")
        except Exception as e2:
            print(f"  ✗ 北证50 akshare 也失败: {e2}")

    # --- 港股指数：akshare → yfinance ---
    print("\n" + "=" * 55)
    print("加载 港股指数 (akshare → yfinance)")
    print("=" * 55)
    for name, symbol in HK_INDICES.items():
        try:
            data[name] = fetch_hk_akshare(symbol, start, end)
            print(f"  ✓ {name} (akshare) — {len(data[name])} 条")
        except Exception as e:
            print(f"  ✗ {name} akshare 失败: {e}")
            try:
                data[name] = fetch_hk_yfinance(symbol, start, end)
                print(f"  ✓ {name} (yfinance) — {len(data[name])} 条")
            except Exception as e2:
                print(f"  ✗ {name} yfinance 也失败: {e2}")

    return data


# ==================== 3. 计算涨跌幅 ====================


def calc_returns(data: dict) -> pd.DataFrame:
    pre_date = "2026-09-30"
    post_date_1d = "2026-10-08"
    post_date_5d = "2026-10-13"

    results = []
    for name, df in data.items():
        if df is None or df.empty:
            continue
        try:
            pre_close = df.loc[:pre_date, "close"].iloc[-1]
            post_close_1d = df.loc[:post_date_1d, "close"].iloc[-1]
            post_close_5d = df.loc[:post_date_5d, "close"].iloc[-1]

            results.append(
                {
                    "指数名称": name,
                    "节前收盘(9/30)": round(pre_close, 2),
                    "节后首日收盘(10/8)": round(post_close_1d, 2),
                    "节后首日涨跌幅(%)": round(
                        (post_close_1d / pre_close - 1) * 100, 2
                    ),
                    "节后一周涨跌幅(%)": round(
                        (post_close_5d / pre_close - 1) * 100, 2
                    ),
                }
            )
        except Exception as e:
            print(f"  计算 {name} 失败: {e}")

    if not results:
        raise ValueError("所有指数均无有效数据")

    df_result = pd.DataFrame(results)
    df_result = df_result.sort_values("节后首日涨跌幅(%)", ascending=False)
    return df_result


# ==================== 4. 可视化函数 ====================


def plot_1d_comparison(df_result):
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = ["#e74c3c" if v > 0 else "#2ecc71" for v in df_result["节后首日涨跌幅(%)"]]
    bars = ax.barh(
        df_result["指数名称"],
        df_result["节后首日涨跌幅(%)"],
        color=colors,
        edgecolor="white",
    )

    for bar, val in zip(bars, df_result["节后首日涨跌幅(%)"]):
        ax.text(
            bar.get_width() + 0.08 if val > 0 else bar.get_width() - 0.08,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.2f}%",
            va="center",
            ha="left" if val > 0 else "right",
            fontsize=10,
            fontweight="bold",
        )

    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xlabel("涨跌幅 (%)", fontsize=12)
    ax.set_title(
        "2026年国庆节后首日各指数涨跌幅对比", fontsize=15, fontweight="bold", pad=15
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig("fig1_节后首日涨跌幅_2026.png", dpi=150)
    plt.show()


def plot_normalized_curves(data):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    ax = axes[0]
    all_a_names = list(BAOSTOCK_INDICES.keys()) + SPECIAL_INDICES
    for name in all_a_names:
        if name not in data or data[name].empty:
            continue
        df = data[name].loc["2026-09-01":"2026-10-13"]
        if len(df) == 0:
            continue
        norm = df["close"] / df["close"].iloc[0] * 100
        ax.plot(norm.index, norm.values, label=name, linewidth=1.8)

    ax.axvline(
        pd.Timestamp("2026-09-30"),
        color="red",
        linestyle="--",
        alpha=0.6,
        label="节前最后交易日",
    )
    ax.axvline(
        pd.Timestamp("2026-10-08"),
        color="green",
        linestyle="--",
        alpha=0.6,
        label="节后首日",
    )
    ax.axvspan(
        pd.Timestamp("2026-10-01"),
        pd.Timestamp("2026-10-07"),
        alpha=0.08,
        color="orange",
    )
    ax.set_title("A股主要指数归一化走势（9/1=100）", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_ylabel("归一化价格")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    ax2 = axes[1]
    for name in HK_INDICES:
        if name not in data or data[name].empty:
            continue
        df = data[name].loc["2026-09-01":"2026-10-13"]
        if len(df) == 0:
            continue
        norm = df["close"] / df["close"].iloc[0] * 100
        ax2.plot(norm.index, norm.values, label=name, linewidth=1.8)

    ax2.axvline(pd.Timestamp("2026-09-30"), color="red", linestyle="--", alpha=0.6)
    ax2.axvline(pd.Timestamp("2026-10-08"), color="green", linestyle="--", alpha=0.6)
    ax2.set_title("港股主要指数归一化走势（9/1=100）", fontsize=14, fontweight="bold")
    ax2.legend(fontsize=9, loc="upper left")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))

    plt.tight_layout()
    plt.savefig("fig2_归一化走势_2026.png", dpi=150)
    plt.show()


def plot_heatmap(data):
    periods = {
        "节前一周\n(9/23-9/30)": ("2026-09-23", "2026-09-30"),
        "节后首日\n(10/8)": ("2026-10-07", "2026-10-08"),
        "节后一周\n(10/8-10/13)": ("2026-10-07", "2026-10-13"),
    }

    heat_data = []
    for name, df in data.items():
        if df is None or df.empty:
            continue
        row = {"指数": name}
        for pname, (ps, pe) in periods.items():
            try:
                sub = df.loc[ps:pe]
                if len(sub) >= 2:
                    row[pname] = round(
                        (sub["close"].iloc[-1] / sub["close"].iloc[0] - 1) * 100, 2
                    )
                else:
                    row[pname] = np.nan
            except Exception:
                row[pname] = np.nan
        heat_data.append(row)

    if not heat_data:
        print("热力图：无有效数据")
        return

    df_heat = pd.DataFrame(heat_data).set_index("指数")
    fig, ax = plt.subplots(figsize=(9, 6))
    im = ax.imshow(df_heat.values, cmap="RdYlGn", aspect="auto", vmin=-10, vmax=10)

    ax.set_xticks(range(len(df_heat.columns)))
    ax.set_xticklabels(df_heat.columns, fontsize=10)
    ax.set_yticks(range(len(df_heat.index)))
    ax.set_yticklabels(df_heat.index, fontsize=11)

    for i in range(len(df_heat.index)):
        for j in range(len(df_heat.columns)):
            val = df_heat.values[i, j]
            if not np.isnan(val):
                ax.text(
                    j,
                    i,
                    f"{val:+.1f}%",
                    ha="center",
                    va="center",
                    fontsize=10,
                    fontweight="bold",
                    color="white" if abs(val) > 6 else "black",
                )

    plt.colorbar(im, ax=ax, label="涨跌幅 (%)", shrink=0.8)
    ax.set_title(
        "2026年各指数分阶段涨跌幅热力图", fontsize=15, fontweight="bold", pad=15
    )
    plt.tight_layout()
    plt.savefig("fig3_热力图_2026.png", dpi=150)
    plt.show()


def plot_radar(data):
    categories = ["节后首日\n涨跌幅", "节后一周\n涨跌幅", "9月涨幅"]
    key_indices = ["上证指数", "创业板指", "科创50", "恒生指数", "恒生科技指数"]

    values = {}
    for name in key_indices:
        if name not in data or data[name].empty:
            continue
        df = data[name]
        try:
            sep_close = df.loc[:"2026-09-30", "close"].iloc[-1]
            aug_close = df.loc[:"2026-08-31", "close"].iloc[-1]
            post1_close = df.loc[:"2026-10-08", "close"].iloc[-1]
            post5_close = df.loc[:"2026-10-13", "close"].iloc[-1]

            values[name] = [
                (post1_close / sep_close - 1) * 100,
                (post5_close / sep_close - 1) * 100,
                (sep_close / aug_close - 1) * 100,
            ]
        except Exception:
            continue

    if len(values) < 2:
        print("雷达图：有效数据不足")
        return

    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors_list = ["#e74c3c", "#e67e22", "#f1c40f", "#3498db", "#2ecc71"]

    for (name, vals), color in zip(values.items(), colors_list):
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, "o-", linewidth=2, label=name, color=color)
        ax.fill(angles, vals_plot, alpha=0.08, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_title(
        "2026年 A股 vs 港股核心指数多维度对比", fontsize=14, fontweight="bold", pad=25
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=10)
    plt.tight_layout()
    plt.savefig("fig5_雷达图_2026.png", dpi=150)
    plt.show()


def plot_volume_comparison():
    """图4：上证指数节前后成交额对比（Baostock数据源）"""
    try:
        # 数据区间覆盖节前一周 + 节后一周
        df_sh = fetch_baostock_with_amount("sh.000001", "2026-09-20", "2026-10-13")
    except Exception as e:
        print(f"成交额图表数据获取失败: {e}")
        return

    if df_sh.empty:
        print("成交额图表：日期范围内无数据")
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    # 节后交易日用红色，节前用蓝色
    colors = [
        "#e74c3c" if d >= pd.Timestamp("2026-10-08") else "#3498db" for d in df_sh.index
    ]

    # Baostock 的 amount 单位是元，除以 1e8 转为亿元
    ax.bar(df_sh.index, df_sh["amount"] / 1e8, color=colors, width=0.7)

    ax.axvline(
        pd.Timestamp("2026-09-30"),
        color="red",
        linestyle="--",
        alpha=0.7,
        label="节前最后交易日",
    )
    ax.axvspan(
        pd.Timestamp("2026-10-01"),
        pd.Timestamp("2026-10-07"),
        alpha=0.15,
        color="orange",
        label="国庆假期",
    )

    # 在柱子上标注数值
    for d, v in zip(df_sh.index, df_sh["amount"] / 1e8):
        ax.text(
            d, v + 30, f"{v:.0f}", ha="center", va="bottom", fontsize=8, color="#333333"
        )

    ax.set_ylabel("成交额（亿元）", fontsize=12)
    ax.set_title("上证指数节前节后成交额对比（2026年）", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10, loc="upper left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("fig4_成交额对比_2026.png", dpi=150)
    plt.show()


# ==================== 5. 主程序 ====================
if __name__ == "__main__":
    print("=" * 55)
    print("  2026年国庆前后行情分析")
    print("=" * 55)

    data = load_all_data()

    if not data:
        print("\n所有指数数据加载失败，请检查网络和依赖安装")
    else:
        print("\n" + "=" * 55)
        print("计算涨跌幅")
        print("=" * 55)
        try:
            df_result = calc_returns(data)
            print(df_result.to_string(index=False))

            print("\n" + "=" * 55)
            print("生成可视化图表")
            print("=" * 55)

            plot_1d_comparison(df_result)
            plot_normalized_curves(data)
            plot_heatmap(data)
            plot_volume_comparison()
            plot_radar(data)

            print("\n所有图表已保存至当前目录。分析完成！")
        except Exception as e:
            print(f"\n分析失败: {e}")
