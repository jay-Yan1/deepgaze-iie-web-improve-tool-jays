"""Streamlit entry point for the DeepGaze IIE website analyzer."""
from __future__ import annotations

import io
import os
import time
from typing import Optional

import numpy as np
import streamlit as st
from PIL import Image

from src import aoi as aoi_mod
from src.llm_advisor import suggest_improvements
from src.saliency import DeepGazePredictor
from src.screenshot import capture_url
from src.visualization import draw_hotspots, find_hotspots, overlay_heatmap

st.set_page_config(page_title="DeepGaze IIE Website Analyzer", layout="wide")


@st.cache_resource(show_spinner="載入 DeepGaze IIE 模型中…")
def get_predictor() -> DeepGazePredictor:
    return DeepGazePredictor()


def _bytes_from_image(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def load_input_image() -> Optional[Image.Image]:
    """Render the input selector and return the chosen image."""
    tab_upload, tab_url = st.tabs(["📤 上傳截圖", "🌐 輸入網址"])

    image: Optional[Image.Image] = None

    with tab_upload:
        upload = st.file_uploader(
            "選擇 PNG / JPG 截圖",
            type=["png", "jpg", "jpeg", "webp"],
            key="uploader",
            help=(
                "支援 PNG / JPG / JPEG / WEBP。建議使用 1024–2560 px 寬度的網頁截圖，"
                "解析度太低會讓 DeepGaze 預測不準，太大會吃光記憶體。"
            ),
        )
        if upload is not None:
            image = Image.open(upload).convert("RGB")

    with tab_url:
        url = st.text_input(
            "網址 (例如 https://example.com)",
            key="url",
            help=(
                "完整網址含 https://。截圖會在伺服器端用 Playwright 開無頭 Chromium 完成，"
                "你只要貼網址、按下方按鈕即可，不需要在自己電腦安裝任何東西。"
            ),
        )
        col_a, col_b = st.columns(2)
        with col_a:
            vp_w = st.number_input(
                "Viewport 寬",
                320,
                2560,
                1440,
                step=20,
                help="模擬瀏覽器視窗寬度 (px)。1440 接近一般筆電解析度，375 模擬 iPhone。",
            )
        with col_b:
            vp_h = st.number_input(
                "Viewport 高",
                320,
                2560,
                900,
                step=20,
                help="模擬瀏覽器視窗高度 (px)。整頁截圖時這只影響首屏載入；視窗截圖時就是最終圖片高度。",
            )

        capture_mode = st.radio(
            "截圖範圍",
            ["🌐 整個網頁 (含需捲動的部分)", "🖥️ 只擷取視窗範圍 (首屏)"],
            index=0,
            help=(
                "整個網頁：把整份 HTML 從頂到底拍下來，圖片可能很長，但能分析整體配置和下方內容。\n"
                "視窗範圍：只截 viewport 大小的首屏，更貼近使用者打開網頁第一眼看到的畫面。"
            ),
        )
        full_page = capture_mode.startswith("🌐")

        auto_scroll = st.checkbox(
            "整頁截圖前先自動捲動觸發 lazy-load",
            value=True,
            help=(
                "勾選後會在截圖前慢慢捲到頁底再回頂，讓延遲載入的圖片/區塊都出現。"
                "如果你發現截到的整頁有空白或圖片沒載入，就勾這個。只在『整個網頁』模式下生效。"
            ),
            disabled=not full_page,
        )

        if url and st.button("擷取網頁截圖", use_container_width=True):
            spinner_msg = "整頁截圖中（捲動載入 + 拍照可能要 10–30 秒）…" if full_page else "Playwright 截圖中…"
            with st.spinner(spinner_msg):
                try:
                    captured = capture_url(
                        url,
                        viewport=(int(vp_w), int(vp_h)),
                        full_page=full_page,
                        auto_scroll=auto_scroll,
                    )
                    st.session_state["captured_image"] = captured
                    st.success(f"截圖完成：{captured.size[0]} × {captured.size[1]} px")
                except Exception as exc:
                    st.error(f"截圖失敗：{exc}")
        if "captured_image" in st.session_state and not image:
            image = st.session_state["captured_image"]

    return image


def render_aoi_table(results) -> None:
    import pandas as pd

    df = pd.DataFrame(
        [
            {
                "Region": r.name,
                "Attention %": round(r.attention_share * 100, 2),
                "Area %": round(r.area_share * 100, 2),
                "Intensity (Attn/Area)": round(r.intensity_ratio, 2),
            }
            for r in results
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)


def main() -> None:
    st.title("👁️ DeepGaze IIE 網站視覺分析工具")
    st.caption(
        "上傳網站截圖或輸入網址，使用 DeepGaze IIE 預測使用者第一眼注意力分布，"
        "再用 Claude 產生改版建議。"
    )

    with st.sidebar:
        st.header("⚙️ 設定")
        api_key = st.text_input(
            "Anthropic API Key",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            type="password",
            help="僅在本機 session 使用，不會被儲存。",
        )
        st.divider()
        top_k = st.slider(
            "Top-K 熱點數量",
            1,
            10,
            5,
            help=(
                "要在圖上框出幾個最吸睛的區域。數字越大會包含次要熱點，"
                "通常 3–5 個最能聚焦核心問題。"
            ),
        )
        threshold = st.slider(
            "熱點門檻 (× max)",
            0.3,
            0.95,
            0.6,
            step=0.05,
            help=(
                "判定為熱點的最低顯著度，以圖中最高值的百分比為基準。"
                "提高 → 只抓最強的小塊熱區；降低 → 連較弱的關注區域也會被框出。"
            ),
        )
        alpha = st.slider(
            "熱力圖透明度",
            0.1,
            0.9,
            0.55,
            step=0.05,
            help=(
                "熱力圖疊加在原圖上的不透明度。"
                "調高熱力圖更明顯但會遮住原圖細節；調低能看清原始 UI 但熱區較淡。"
            ),
        )
        st.divider()
        layout_mode = st.radio(
            "AOI 分區方式",
            ["垂直分區 (Header/Hero/Body/Footer)", "3×3 網格"],
            help=(
                "如何把畫面切成多個分析區域 (Area of Interest)。\n"
                "垂直分區：依網頁結構切 Header / Hero / Body / Footer，適合典型 landing page。\n"
                "3×3 網格：均分成九宮格，適合非標準版面或想看左右/上下分布。"
            ),
        )
        user_goal = st.text_area(
            "網站目標 / 場景描述 (給 LLM)",
            placeholder="例如：這是 SaaS 註冊頁，主要目標是讓訪客點擊『開始試用』按鈕。",
            height=110,
            help=(
                "提供業務目標和情境給 Claude 參考。寫得越具體 (主要 CTA 是什麼、"
                "目標受眾、想驗證的假設)，建議就越有針對性。"
            ),
        )

    image = load_input_image()
    if image is None:
        st.info("從左側分頁上傳截圖或輸入網址開始分析。")
        return

    st.subheader("原始截圖")
    st.image(image, use_container_width=True)

    if not st.button("🚀 開始分析", type="primary", use_container_width=True):
        return

    predictor = get_predictor()

    t0 = time.time()
    with st.spinner("DeepGaze IIE 推論中…"):
        saliency = predictor.predict(image)
    infer_ms = (time.time() - t0) * 1000

    overlay = overlay_heatmap(image, saliency, alpha=alpha)
    hotspots = find_hotspots(saliency, top_k=top_k, threshold_pct=threshold)
    annotated = draw_hotspots(overlay, hotspots)

    w, h = image.size
    if layout_mode.startswith("垂直"):
        regions = aoi_mod.vertical_band_layout(w, h)
    else:
        regions = aoi_mod.grid_layout(w, h, rows=3, cols=3)
    aoi_results = aoi_mod.analyze(saliency, regions)

    st.success(f"完成 DeepGaze IIE 推論 (耗時 {infer_ms:.0f} ms，裝置 {predictor.device})")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔥 顯著性熱力圖")
        st.image(overlay, use_container_width=True)
        st.download_button(
            "下載熱力圖",
            data=_bytes_from_image(overlay),
            file_name="saliency_overlay.png",
            mime="image/png",
        )
    with col2:
        st.markdown("#### 📍 Top-K 注意力熱點")
        st.image(annotated, use_container_width=True)
        st.download_button(
            "下載標註圖",
            data=_bytes_from_image(annotated),
            file_name="hotspots.png",
            mime="image/png",
        )

    st.markdown("#### 🗺️ AOI 區域注意力分布")
    render_aoi_table(aoi_results)
    st.caption(
        "Intensity 表示 Attention% / Area%。>1 代表該區域被過度注視（吸睛效率高），"
        "<1 代表該區域相對被忽略。"
    )

    st.markdown("#### 🧠 Claude 改版建議")
    if not api_key:
        st.warning("請在左側填入 Anthropic API Key 以啟用 LLM 建議。")
    else:
        try:
            with st.spinner("Claude 分析中…"):
                advice = suggest_improvements(
                    original=image,
                    overlay=overlay,
                    hotspots=hotspots,
                    aoi=aoi_results,
                    user_goal=user_goal,
                    api_key=api_key,
                )
            st.markdown(advice)
        except Exception as exc:
            st.error(f"LLM 呼叫失敗：{exc}")


if __name__ == "__main__":
    main()
