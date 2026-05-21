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
        )
        if upload is not None:
            image = Image.open(upload).convert("RGB")

    with tab_url:
        url = st.text_input("網址 (例如 https://example.com)", key="url")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            vp_w = st.number_input("Viewport 寬", 320, 2560, 1440, step=20)
        with col_b:
            vp_h = st.number_input("Viewport 高", 320, 2560, 900, step=20)
        with col_c:
            full_page = st.checkbox("整頁截圖", value=True)
        if url and st.button("擷取網頁截圖", use_container_width=True):
            with st.spinner("Playwright 開啟瀏覽器中…"):
                try:
                    captured = capture_url(url, viewport=(int(vp_w), int(vp_h)), full_page=full_page)
                    st.session_state["captured_image"] = captured
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
        top_k = st.slider("Top-K 熱點數量", 1, 10, 5)
        threshold = st.slider("熱點門檻 (× max)", 0.3, 0.95, 0.6, step=0.05)
        alpha = st.slider("熱力圖透明度", 0.1, 0.9, 0.55, step=0.05)
        st.divider()
        layout_mode = st.radio("AOI 分區方式", ["垂直分區 (Header/Hero/Body/Footer)", "3×3 網格"])
        user_goal = st.text_area(
            "網站目標 / 場景描述 (給 LLM)",
            placeholder="例如：這是 SaaS 註冊頁，主要目標是讓訪客點擊『開始試用』按鈕。",
            height=110,
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
