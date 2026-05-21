"""Streamlit entry point for the DeepGaze IIE website analyzer."""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
from datetime import datetime
from typing import List, Optional

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

from src import aoi as aoi_mod
from src.aoi import AOIResult
from src.llm_advisor import suggest_improvements
from src.saliency import DeepGazePredictor
from src.screenshot import capture_url
from src.visualization import Hotspot, draw_hotspots, find_hotspots, overlay_heatmap

st.set_page_config(page_title="DeepGaze IIE Website Analyzer", layout="wide")


_CUSTOM_CSS = """
<style>
/* === Primary action button (Step 3 · 🚀 開始分析) === */
div[data-testid="stButton"] > button[kind="primary"] {
    font-size: 1.05rem !important;
    font-weight: 700 !important;
    padding: 0.75rem 1.25rem !important;
    background: linear-gradient(135deg, #ff8a4d 0%, #ff5d3a 100%) !important;
    color: #fff !important;
    border: 0 !important;
    box-shadow: 0 4px 14px rgba(255, 93, 58, 0.30) !important;
    transition: transform 0.05s ease, box-shadow 0.2s ease, filter 0.15s ease;
    letter-spacing: 0.02em;
}
div[data-testid="stButton"] > button[kind="primary"]:hover {
    filter: brightness(1.06);
    box-shadow: 0 6px 18px rgba(255, 93, 58, 0.40) !important;
}
div[data-testid="stButton"] > button[kind="primary"]:active {
    transform: translateY(1px);
}

/* === File uploader drop zone === */
section[data-testid="stFileUploaderDropzone"],
[data-testid="stFileUploaderDropzone"] {
    background: #eff6ff !important;
    border: 2px dashed #4c9aff !important;
    border-radius: 12px !important;
    padding: 1.5rem 1.25rem !important;
    min-height: 130px !important;
    transition: background 0.2s ease, border-color 0.2s ease;
}
section[data-testid="stFileUploaderDropzone"]:hover,
[data-testid="stFileUploaderDropzone"]:hover {
    background: #e0edff !important;
    border-color: #2680eb !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: #2680eb !important;
    color: #fff !important;
    border: 0 !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.1rem !important;
    border-radius: 6px !important;
}
[data-testid="stFileUploaderDropzone"] button:hover {
    background: #1c6ed0 !important;
}

/* === URL text input focus pop === */
[data-testid="stTextInput"] input:focus {
    border-color: #2680eb !important;
    box-shadow: 0 0 0 3px rgba(38, 128, 235, 0.15) !important;
}

/* === Tabs: bigger Step 1 label, blue indicator when active === */
div[data-testid="stTabs"] button[role="tab"] {
    font-size: 0.98rem !important;
    font-weight: 600 !important;
}
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #2680eb !important;
}

/* === Step 1 info banner: tighter, accent bar === */
div[data-testid="stAlert"] {
    border-left: 4px solid #2680eb !important;
}
</style>
"""


def _inject_custom_css() -> None:
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner="載入 DeepGaze IIE 模型中…")
def get_predictor() -> DeepGazePredictor:
    return DeepGazePredictor()


def _bytes_from_image(img: Image.Image, fmt: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def load_input_image() -> Optional[Image.Image]:
    """Render the input selector and return the chosen image."""
    tab_upload, tab_url = st.tabs(
        ["Step 1 · 📤 上傳截圖", "Step 1 · 🌐 輸入網址"]
    )

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


def build_report(
    *,
    image_size: tuple,
    layout_mode: str,
    user_goal: str,
    hotspots: List[Hotspot],
    aoi_results: List[AOIResult],
    advice: Optional[str],
    infer_ms: float,
    device: str,
) -> str:
    """Render a Markdown report containing the full diagnosis."""
    w, h = image_size
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines: List[str] = []
    lines.append("# DeepGaze IIE 網站視覺分析報告")
    lines.append("")
    lines.append(f"- **分析時間：** {ts}")
    lines.append(f"- **截圖尺寸：** {w} × {h} px")
    lines.append(f"- **AOI 分區方式：** {layout_mode}")
    lines.append(f"- **DeepGaze 推論：** {infer_ms:.0f} ms（{device}）")
    if user_goal.strip():
        lines.append(f"- **網站目標 / 場景：**")
        for ln in user_goal.strip().splitlines():
            lines.append(f"  > {ln}")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 🔥 Top-K 注意力熱點")
    lines.append("")
    if hotspots:
        lines.append("| 排名 | 中心 X% | 中心 Y% | 注意力佔比 | 區域大小 (px) |")
        lines.append("|------|---------|---------|------------|----------------|")
        for hs in hotspots:
            x_pct = hs.cx / w * 100 if w else 0
            y_pct = hs.cy / h * 100 if h else 0
            bw, bh = hs.bbox[2], hs.bbox[3]
            lines.append(
                f"| #{hs.rank} | {x_pct:.0f}% | {y_pct:.0f}% | "
                f"{hs.attention_share * 100:.1f}% | {bw}×{bh} |"
            )
    else:
        lines.append("_(沒有偵測到顯著熱點，可以試著調低門檻)_")
    lines.append("")
    lines.append("## 🗺️ AOI 區域注意力分布")
    lines.append("")
    if aoi_results:
        lines.append("| 區域 | 注意力 % | 面積 % | 強度比 (Attn/Area) |")
        lines.append("|------|---------|--------|---------------------|")
        for r in aoi_results:
            lines.append(
                f"| {r.name} | {r.attention_share * 100:.2f} | "
                f"{r.area_share * 100:.2f} | {r.intensity_ratio:.2f} |"
            )
    else:
        lines.append("_(無 AOI 結果)_")
    lines.append("")
    lines.append("> Intensity > 1 代表該區「被過度注視」（吸睛效率高），< 1 代表相對被忽略。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 🧠 Claude 改版建議")
    lines.append("")
    if advice:
        lines.append(advice.strip())
    else:
        lines.append("_(本次未產生 LLM 建議，可能是未提供 API Key)_")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*Generated by DeepGaze IIE Website Analyzer*")
    return "\n".join(lines)


def copy_to_clipboard_button(text: str, label: str = "📋 複製到剪貼簿") -> None:
    """Render a JS-powered copy button that works in HTTP (LAN) too.

    Streamlit's built-in `st.code()` copy uses `navigator.clipboard`, which
    silently no-ops outside Secure Contexts (HTTPS / localhost). We fall back
    to `document.execCommand('copy')` so LAN IP access still works.
    """
    payload = json.dumps(text)
    button_id = "copy-btn-" + hashlib.md5(text.encode("utf-8")).hexdigest()[:10]
    html = f"""
    <div style="margin:0.25rem 0;">
      <button id="{button_id}"
              style="padding:0.45rem 1rem;border:1px solid #ccc;border-radius:6px;
                     background:#f6f8fa;cursor:pointer;font-size:0.92rem;
                     font-family:inherit;">
        {label}
      </button>
      <span id="{button_id}-status"
            style="margin-left:0.6rem;color:#0a7d2c;font-size:0.85rem;"></span>
    </div>
    <script>
      (function() {{
        const btn = document.getElementById("{button_id}");
        const status = document.getElementById("{button_id}-status");
        if (!btn) return;
        btn.addEventListener("click", async () => {{
          const text = {payload};
          let ok = false;
          if (navigator.clipboard && window.isSecureContext) {{
            try {{
              await navigator.clipboard.writeText(text);
              ok = true;
            }} catch (e) {{ ok = false; }}
          }}
          if (!ok) {{
            const ta = document.createElement("textarea");
            ta.value = text;
            ta.style.position = "fixed";
            ta.style.left = "-9999px";
            ta.setAttribute("readonly", "");
            document.body.appendChild(ta);
            ta.select();
            try {{ ok = document.execCommand("copy"); }} catch (e) {{ ok = false; }}
            document.body.removeChild(ta);
          }}
          status.style.color = ok ? "#0a7d2c" : "#c92a2a";
          status.textContent = ok ? "✓ 已複製" : "✗ 複製失敗 (請手動選取)";
          setTimeout(() => {{ status.textContent = ""; }}, 2500);
        }});
      }})();
    </script>
    """
    components.html(html, height=60)


def render_export_controls(report_md: str, advice: Optional[str]) -> None:
    """Download buttons + copy expander for the diagnosis."""
    ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    st.markdown("#### 📤 匯出 / 複製診斷")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.download_button(
            "📄 下載完整報告 (.md)",
            data=report_md.encode("utf-8"),
            file_name=f"deepgaze_report_{ts_tag}.md",
            mime="text/markdown",
            use_container_width=True,
            help="包含 metadata、Top-K 表格、AOI 表格、Claude 建議的完整 Markdown 報告。",
        )
    with col2:
        st.download_button(
            "💬 只下載 Claude 建議 (.md)",
            data=(advice or "").encode("utf-8"),
            file_name=f"deepgaze_advice_{ts_tag}.md",
            mime="text/markdown",
            disabled=not advice,
            use_container_width=True,
            help="只匯出 Claude 產生的純文字建議（不含數據表格）。",
        )
    with col3:
        st.download_button(
            "📝 下載純文字 (.txt)",
            data=report_md.encode("utf-8"),
            file_name=f"deepgaze_report_{ts_tag}.txt",
            mime="text/plain",
            use_container_width=True,
            help="同樣的內容但副檔名為 .txt，方便貼到 Slack / Notion / Word。",
        )

    with st.expander("📋 複製到剪貼簿", expanded=False):
        st.caption(
            "點下方按鈕即可複製到剪貼簿。如果你透過區網 IP（http://192.168...）"
            "進來，瀏覽器會擋住 Secure Clipboard API，這裡用 fallback 方案讓 HTTP 也能複製。"
        )
        st.markdown("**完整報告（Markdown）**")
        copy_to_clipboard_button(report_md, "📋 複製完整報告")
        st.code(report_md, language="markdown")
        if advice:
            st.markdown("**只複製 Claude 建議**")
            copy_to_clipboard_button(advice, "📋 複製 Claude 建議")
            st.code(advice, language="markdown")


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
    _inject_custom_css()
    st.markdown(
        """
        <div style="margin-bottom:0.6rem;">
          <h2 style="margin:0;font-size:1.55rem;line-height:1.3;">
            👁️ DeepGaze IIE 網站視覺分析
          </h2>
          <p style="margin:0.1rem 0 0;color:#777;font-size:0.88rem;">
            預測使用者第一眼注意力分布 → Claude 給改版建議
          </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.info(
        "⬇️ **Step 1：在下方上傳截圖或貼上網址 → 按「🚀 開始分析」即可**",
        icon="👇",
    )

    with st.sidebar:
        st.markdown(
            "<p style='font-size:0.78rem;color:#888;margin:0;'>🔑 API 設定</p>",
            unsafe_allow_html=True,
        )
        api_key = st.text_input(
            "Anthropic API Key",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
            type="password",
            help="僅在本機 session 使用，不會被儲存。",
            label_visibility="collapsed",
            placeholder="sk-ant-... (沒有 key 就跳過，只看熱力圖)",
        )
        st.divider()
        st.header("Step 2 · ⚙️ 分析參數")
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

    if st.button("Step 3 · 🚀 開始分析", type="primary", use_container_width=True):
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

        st.session_state["analysis"] = {
            "image": image,
            "overlay": overlay,
            "annotated": annotated,
            "hotspots": hotspots,
            "aoi_results": aoi_results,
            "infer_ms": infer_ms,
            "device": predictor.device,
            "layout_mode": layout_mode,
            "user_goal": user_goal,
        }
        st.session_state.pop("advice", None)
        st.session_state.pop("advice_error", None)

    if "analysis" not in st.session_state:
        return

    a = st.session_state["analysis"]
    st.success(
        f"完成 DeepGaze IIE 推論 (耗時 {a['infer_ms']:.0f} ms，裝置 {a['device']})"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🔥 顯著性熱力圖")
        st.image(a["overlay"], use_container_width=True)
        st.download_button(
            "下載熱力圖",
            data=_bytes_from_image(a["overlay"]),
            file_name="saliency_overlay.png",
            mime="image/png",
        )
    with col2:
        st.markdown("#### 📍 Top-K 注意力熱點")
        st.image(a["annotated"], use_container_width=True)
        st.download_button(
            "下載標註圖",
            data=_bytes_from_image(a["annotated"]),
            file_name="hotspots.png",
            mime="image/png",
        )

    st.markdown("#### 🗺️ AOI 區域注意力分布")
    render_aoi_table(a["aoi_results"])
    st.caption(
        "Intensity 表示 Attention% / Area%。>1 代表該區域被過度注視（吸睛效率高），"
        "<1 代表該區域相對被忽略。"
    )

    st.markdown("#### 🧠 Claude 改版建議")
    if not api_key:
        st.warning("請在左側填入 Anthropic API Key 以啟用 LLM 建議。")
    else:
        if "advice" not in st.session_state and "advice_error" not in st.session_state:
            try:
                with st.spinner("Claude 分析中…"):
                    st.session_state["advice"] = suggest_improvements(
                        original=a["image"],
                        overlay=a["overlay"],
                        hotspots=a["hotspots"],
                        aoi=a["aoi_results"],
                        user_goal=a["user_goal"],
                        api_key=api_key,
                    )
            except Exception as exc:
                st.session_state["advice_error"] = str(exc)

        if "advice_error" in st.session_state:
            st.error(f"LLM 呼叫失敗：{st.session_state['advice_error']}")
            if st.button("🔁 重試 Claude 建議"):
                st.session_state.pop("advice_error", None)
                st.rerun()
        elif "advice" in st.session_state:
            st.markdown(st.session_state["advice"])
            if st.button("🔁 重新產生 Claude 建議"):
                st.session_state.pop("advice", None)
                st.rerun()

    advice = st.session_state.get("advice")
    report_md = build_report(
        image_size=a["image"].size,
        layout_mode=a["layout_mode"],
        user_goal=a["user_goal"],
        hotspots=a["hotspots"],
        aoi_results=a["aoi_results"],
        advice=advice,
        infer_ms=a["infer_ms"],
        device=a["device"],
    )
    st.divider()
    render_export_controls(report_md, advice)


if __name__ == "__main__":
    main()
