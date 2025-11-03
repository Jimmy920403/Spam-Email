## 專案：Spam Email Classification (Phase1 baseline)

這個專案是為了示範使用傳統機器學習的垃圾郵件（短訊/Email）分類基線流程（Phase 1）。專案包含 OpenSpec 變更提案、資料處理、訓練腳本、模型匯出、視覺化、Streamlit 介面與 CI 範例。

> Repo: Spam-Email（HW3）

### 重點功能
- 用途：建立可重現的 SVM 基線模型（含文字向量化 pipeline）。
- 可執行腳本：下載資料、前處理、訓練、預測、視覺化。
- 互動介面：`apps/streamlit_app.py` 提供即時推論與視覺化檢視。
- CI：`.github/workflows/ml-baseline.yml` 提供快速 smoke 測試與 metric gate（F1 判斷）。
- 規範：OpenSpec 變更提案位於 `openspec/changes/`。

## 儲存位置（主要檔案/資料夾）
- `scripts/` — 各種可執行腳本（訓練、預處理、下載、預測、視覺化）。
- `apps/` — Streamlit 應用程式模組（`apps/streamlit_app.py`）。
- `models/` — 訓練輸出之模型檔（joblib）。
- `artifacts/` — 評估指標與中間輸出（JSON、圖檔等）。
- `reports/visualizations/` — 由視覺化腳本產生的圖表（ROC, PR, CM, token freq 等）。
- `data/` — 原始與 sample 測試資料（`data/sample.csv`, `data/sms_spam_no_header.csv`）。
- `openspec/` — 專案規格與變更提案（`project.md`、`changes/`）。
- `tests/` — pytest 測試。
- `requirements.txt` — Python 依賴清單。

## 快速啟動（PowerShell 範例）
1. 建議先建立虛擬環境並啟動：

```powershell
python -m venv .venv
.\ .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

2. 執行測試（pytest）：

```powershell
pytest -q
```

3. 在 sample 資料上做一次 smoke 訓練（快速）：

```powershell
python -m scripts.train_baseline --data data/sample.csv --model-out models/baseline-svm.joblib --metrics-out artifacts/metrics.json --cm-out artifacts/confusion_matrix.png
```

4. 在完整 Packt 資料集上訓練（本機/非 CI）：

```powershell
python -m scripts.train_baseline --data data/sms_spam_no_header.csv --model-out models/pipeline.joblib --metrics-out artifacts/metrics_full.json --cm-out artifacts/confusion_matrix_full.png
```

5. 使用預測 CLI（單句範例）：

```powershell
python scripts/predict_spam.py --model models/pipeline.joblib --text "Free entry in 2 a wkly comp to win cash"
```

6. 啟動 Streamlit 介面（本機開發/示範）：

```powershell
streamlit run apps/streamlit_app.py
```

## 可用資產與輸出
- 模型：`models/` 下的 `.joblib` 檔（建議使用 `pipeline.joblib`，其中包含向量器 + 分類器）。
- 指標：`artifacts/metrics*.json`。
- 視覺化：`reports/visualizations/` 下的 PNG 與 threshold sweep CSV。

## OpenSpec 與變更提案
- 專案的設計與變更規範在 `openspec/` 內，變更提案放在 `openspec/changes/`。請參閱 `openspec/project.md` 與 `openspec/AGENTS.md` 了解如何提出與審查變更。

## 開發者與貢獻流程（快速指南）
1. 在開發新功能前，先建立 OpenSpec 變更提案草案（`openspec/changes/<your-change>/proposal.md`）。
2. 實作時：新增必要的測試（`tests/`）與小型 sample 測資，以確保 CI 可快速檢查。
3. 保持模型匯出格式穩定：請優先使用 sklearn `Pipeline`（vectorizer + classifier），這樣可以簡化序列化/反序列化與推論流程。
4. CI 原則：維持 smoke tests 快速；完整訓練可在本機或專用 runner 上執行以產生完整指標。

## 常見問題與注意事項
- 跑腳本時可使用模組方式以避免 import 路徑問題：`python -m scripts.train_baseline`。若直接執行檔案，請確保專案根目錄在 `PYTHONPATH`。
- 若在 unpickle 時碰到錯誤，請確認模型檔是 `Pipeline`，且專案原始碼位置未變動；必要時可用 `joblib` 讀取並重構為 `Pipeline`。

## 下一步建議
- 整合 `reports/visualizations` 的圖片到 `apps/streamlit_app.py`（若需要，我可以直接修改 Streamlit 應用，將剛產生的圖顯示並提供下載連結）。
- 實作更完整的預處理流程並把每個步驟輸出存檔，以利可追蹤性與除錯。

## 聯絡與支援
- 專案規格與變更討論請在 `openspec/changes/` 新增變更提案。
- 若要我繼續：我可以幫你把視覺化整合進 Streamlit、完善 preproc 文檔，或加入 CI badge 等。請回覆你希望我下一步做什麼。

---
最後更新：2025-11-03
