---
name: saliency-check
description: 用 DeepGaze IIE 預測使用者第一眼會看向網頁的哪裡，回傳熱力圖與各區注意力佔比。當你設計或修改網頁版面、調整 hero 區、CTA 位置、配色對比或視覺層級之後，用它驗證注意力有沒有落在該落的地方，再依據數據改版。也適用於分析既有網站或截圖的視覺動線。
---

# 網頁視覺動線檢查

不論用哪種呼叫方式，結果都是 JSON 數據 + 三張 PNG。**一定要用 Read 工具打開 `heatmap` 和 `hotspots` 兩張圖親眼看**，數字只說明「多少」，圖才說明「在哪裡、是哪個元件」。

## 先看有沒有 MCP 工具

如果工具列表裡有 `deepgaze_analyze_page` / `deepgaze_compare_pages`，**優先用它們**，不要跑 CLI：MCP server 常駐把模型留在記憶體，第二次之後的分析省掉 10–30 秒的載入。判讀準則（下面「判讀」一節）兩邊通用。

- 一次分析：`deepgaze_analyze_page`，參數同下方 CLI 選項（`aoi` 傳字串陣列如 `["CTA:820,340,300,90"]`）。
- 改版前後對照：`deepgaze_compare_pages`，兩邊用同一組 `aoi`，會直接給你 intensity 的變化量。
- 預期等一下才會用到：先呼叫 `deepgaze_warm_up` 把權重載起來，趁你還在寫 markup 的時候。
- `out_dir` 是相對 server 的工作目錄，要確定圖存在哪就傳絕對路徑。

沒有這些工具就用下面的 CLI。

## CLI

`analyze_cli.py` 把 JSON 印到 stdout、進度印到 stderr。

```bash
python analyze_cli.py <target> --out <輸出資料夾>
```

`<target>` 三種形式：

| 情境 | 寫法 |
|---|---|
| 本機還沒起 server 的 HTML | `./index.html`（會自動轉成 `file://`） |
| 本機 dev server | `http://localhost:3000` |
| 既有截圖 | `./shot.png` |

常用選項：

- `--first-screen`：只截首屏。**評估「打開網頁第一眼」時用這個**；預設是整頁截圖，會把頁尾也算進 AOI，稀釋首屏的佔比。
- `--aoi "CTA:820,340,300,90"`：指定你在意的元件範圍（可重複）。你知道 CTA、標題、表單在哪，就直接量它，比看四個粗略分區準得多。座標從 `hotspots.png` 或 DOM 量。
- `--layout grid`：改用 3×3 分區（預設 `bands` 是 Header/Hero/Body/Footer）。
- `--top-k 3` / `--threshold 0.7`：只看最強的幾個熱點。
- `--max-side 768`：CPU 上想快一點就調小（預設 1024）。

第一次執行會下載 DeepGaze 權重，CPU 推論一張約 5–15 秒，別以為是卡住了。若缺套件：`pip install -r requirements.txt && python -m playwright install chromium`。

## 判讀

JSON 裡每個 AOI 有三個數字，關鍵是 `intensity_ratio`（注意力佔比 ÷ 面積佔比）：

- `> 1.5`：這區吸走的注意力遠超過它的版面大小。是好是壞看它是不是重點——是 CTA 就對了，是裝飾圖或 cookie 橫幅就是在偷注意力。
- `≈ 1.0`：不特別吸睛也不特別被忽略。
- `< 0.5`：幾乎被跳過。如果重要內容落在這裡，就是問題。

再對照 `hotspots`（依吸睛程度排名，含 `attention_share` 與 `bbox`）：

- **主要 CTA 不在 top-3** → 提高對比、放大、往上移，或把周圍搶戲的元素弱化。
- **top-1 是裝飾性元素**（大圖、插畫、動畫）→ 降低它的飽和度／對比，或縮小。
- **熱點全擠在同一塊** → 頁面缺乏視覺節奏，下半部等於沒人看。
- **注意力集中在畫面正中** → 留意這可能是 DeepGaze 的 center bias，不見得是你的設計造成的，要搭配圖判斷。

## 改完要再跑一次

這個工具的價值在前後對照。改完 CSS／版面後用同一組參數重跑，比較目標區域的 `intensity_ratio` 和 CTA 的 hotspot 排名有沒有往上走。用不同的 `--out` 資料夾分開存，才不會覆蓋掉改版前的圖。

## 限制

- DeepGaze 預測的是**自由觀看下的眼動**，不是點擊率也不是轉換率。「被看到」是必要條件不是充分條件——結論要回到網站的商業目標上談。
- 它看的是視覺特徵（對比、邊緣、人臉、文字密度），**讀不懂語意**。文案好不好、資訊架構對不對，它一概不知道。
- 整頁截圖很長時，熱力圖會被壓縮得難以判讀，這種頁面建議分段用 `--first-screen` 搭配捲動位置各截一次。
- 這是模型預測，不能取代真實使用者測試。
