# DeepGaze IIE 網站視覺分析工具

用 [DeepGaze IIE](https://github.com/matthias-k/DeepGaze) 預測使用者第一眼會落在網頁的哪些位置，再用 Claude 產生具體的改版建議，幫助你判斷網站的視覺動線是否符合商業目標。

## 功能

- 📤 **雙模式輸入**：上傳截圖，或輸入網址自動用 Playwright 截整頁。
- 🔥 **顯著性熱力圖**：DeepGaze IIE 推論結果以 JET colormap 疊加在原圖上。
- 📍 **Top-K 熱點偵測**：以連通元件抓出最吸睛的 N 個區域，含座標、面積、注意力佔比。
- 🗺️ **AOI 區域分析**：垂直分區 (Header/Hero/Body/Footer) 或 3×3 網格，計算每區的「注意力 / 面積」效率。
- 🧠 **Claude 改版建議**：傳送原圖 + 熱力圖 + AOI 數據給 Claude，產出可行動的設計建議。
- 🤖 **CLI / MCP 模式**：`analyze_cli.py`（命令列）與 `mcp_server.py`（MCP server）跑同一套流程並輸出結構化結果，讓 Claude Code 等 agent 在設計網頁時直接呼叫。

## 安裝

```bash
pip install -r requirements.txt
python -m playwright install chromium
```

DeepGaze IIE 的預訓練權重和 MIT1003 centerbias 會在第一次推論時自動下載到 `assets/`。

## 啟動

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # 或在側邊欄輸入
streamlit run app.py
```

預設在 <http://localhost:8501> 開啟。

## 使用流程

1. 從側邊欄填入 Anthropic API Key（也可改用環境變數）。
2. 選擇「上傳截圖」或「輸入網址」，把目標網頁送進來。
3. 在側邊欄調整 Top-K 數量、熱點門檻、AOI 分區方式，並填寫網站目標。
4. 按下「🚀 開始分析」，會依序得到：
   - 熱力圖疊加
   - Top-K 框選圖
   - AOI 表格
   - Claude 文字建議

## 給 Claude Code 用（CLI / MCP）

Streamlit 是給人操作的，agent 沒辦法呼叫。以下兩種方式跑的是同一套 `src/pipeline.py`，結果一致，且都不呼叫 LLM（agent 自己就是那個 LLM，直接讀圖判讀即可）。

### CLI

`analyze_cli.py` 是無介面版本：JSON 走 stdout、進度走 stderr、圖片存到 `--out` 資料夾。

```bash
python analyze_cli.py https://example.com --out ./saliency_out
python analyze_cli.py ./index.html --first-screen          # 本機 HTML，只看首屏
python analyze_cli.py http://localhost:3000 --aoi "CTA:820,340,300,90"
python analyze_cli.py shot.png --layout grid --top-k 3
python analyze_cli.py --help                               # 完整選項
```

輸出的 JSON 包含 `hotspots`（排名、bbox、注意力佔比）與 `aoi`（每區的 `attention_share` / `area_share` / `intensity_ratio`），並附上 `original.png`、`heatmap.png`、`hotspots.png` 的路徑。

### MCP server（模型常駐，推薦）

CLI 每次執行都要重載一次 DeepGaze 權重（CPU 上 10–30 秒）。MCP server 是常駐行程，權重只載入一次，之後每次分析只花推論的時間。

```bash
claude mcp add deepgaze -- python3 /絕對路徑/mcp_server.py
```

本 repo 也附了 `.mcp.json`，從 repo 根目錄啟動 Claude Code 就會自動載入這個 server（Windows 或 venv 環境請把裡面的 `python3` 換成你的直譯器）。

提供三個工具：

| 工具 | 用途 |
|---|---|
| `deepgaze_analyze_page` | 分析單一頁面，回傳熱點、AOI 數據與三張 PNG 路徑 |
| `deepgaze_compare_pages` | 改版前後用同一組設定各跑一次，直接給出每區 intensity 的變化量 |
| `deepgaze_warm_up` | 預先載入權重，把 10–30 秒的成本挪到你還在寫程式的時候 |

回傳預設是精簡的 Markdown 摘要，傳 `response_format: "json"` 可拿完整結構化資料。工具只回圖片路徑不回圖片本身——整頁截圖很大，讓 agent 自己決定要不要讀。

`.claude/skills/saliency-check/` 是搭配的 Claude Code Skill：載入這個 repo 後，Claude Code 在改網頁版面時會自動想到用它來驗證視覺動線，也可以打 `/saliency-check` 手動觸發。Skill 內含執行方式、`intensity_ratio` 判讀準則與模型限制說明。

## 專案結構

```
.
├── app.py                 # Streamlit 介面
├── analyze_cli.py         # 無介面 CLI，輸出 JSON 給 agent 用
├── mcp_server.py          # MCP server（模型常駐），供 Claude Code 直接呼叫
├── .mcp.json              # 專案層級 MCP 設定
├── requirements.txt
├── .claude/skills/
│   └── saliency-check/    # Claude Code Skill：設計網頁時自動驗證視覺動線
├── src/
│   ├── saliency.py        # DeepGaze IIE 推論包裝
│   ├── screenshot.py      # Playwright 整頁截圖
│   ├── visualization.py   # 熱力圖疊加 & Top-K 偵測
│   ├── aoi.py             # 區域注意力分析
│   ├── pipeline.py        # CLI 與 MCP 共用的分析流程
│   └── llm_advisor.py     # Claude API 呼叫
├── assets/                # 自動下載的權重/centerbias
└── examples/
```

## 注意事項

- DeepGaze IIE 在 CPU 上推論一張 1024px 寬的截圖大約 5–15 秒；建議使用 GPU。
- `saliency.predict()` 內部會把長邊縮到 1024 以下做推論，再重採回原圖尺寸，避免 OOM。
- DeepGaze 預測的是「自由觀看」狀態下的眼動，不等於「點擊機率」，請對照網站業務目標一起解讀。
- 自動截圖功能需要 Chromium，請務必執行 `playwright install chromium`。

## 授權

本工具的程式碼以 MIT 釋出；DeepGaze IIE 模型權重請依照 [上游專案](https://github.com/matthias-k/DeepGaze) 的授權使用。
