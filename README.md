# 🚦 交通監控系統

即時車輛偵測系統，使用 YOLOv4-tiny + OpenCV，支援任意 RTSP / HTTP 串流，並自動將偵測結果輸出至 CSV。

---

## ✨ 功能特色

- 自動下載 YOLOv4-tiny 模型，無需手動準備
- 偵測四類車輛：**汽車 / 機車 / 巴士 / 卡車**
- 支援自訂串流網址（命令列 `--url` 或互動輸入）
- 定期將統計資料寫入 CSV
- 支援無頭模式（伺服器 / headless 環境）

---

## 🖥️ 環境需求

| 套件 | 版本 |
|------|------|
| Python |  3.10 |
| opencv-python |  4.8 |
| numpy |  1.24 |

> **不需要** PyTorch 或 CUDA。

---

## 📦 安裝

```bash
git clone https://github.com/lingh0487-ship-it/YOLO_live-vedio.git
cd YOLO_live-vedio

pip install -r requirements.txt
```

---

## 🚀 使用方式

### 方式 A：互動式輸入網址（最簡單）

```bash
python traffic_monitor.py
```

程式啟動後會顯示預設網址，直接按 Enter 使用，或輸入自訂網址。

---

### 方式 B：命令列指定網址

```bash
# 使用台南市交通攝影機（預設）
python traffic_monitor.py --url https://trafficvideo2.tainan.gov.tw/e0029830

# 使用 RTSP 串流
python traffic_monitor.py --url rtsp://192.168.1.100:554/live/stream

# 使用本機攝影機
python traffic_monitor.py --url 0
```

---

### 方式 C：無頭模式（適合雲端 / 伺服器）

```bash
python traffic_monitor.py --url <串流網址> --no-display
```

---

## ⚙️ 所有參數

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `--url` | 串流網址（RTSP / HTTP / `0` 為本機攝影機） | 互動詢問 |
| `--save-interval` | CSV 儲存間隔（秒） | `5` |
| `--confidence` | 偵測信心閾值（0~1） | `0.3` |
| `--no-display` | 無頭模式，不開啟視窗 | 關閉 |

---

## 📊 CSV 輸出格式

每隔 `--save-interval` 秒自動儲存一筆，檔名為 `traffic_data_YYYYMMDD_HHMMSS.csv`。

| 欄位 | 說明 |
|------|------|
| 時間戳記 | `YYYY-MM-DD HH:MM:SS` |
| 日期 | `YYYY-MM-DD` |
| 時刻 | `HH:MM:SS` |
| 總車輛數 | 當前畫面偵測總數 |
| 汽車 / 機車 / 巴士 / 卡車 | 各類型數量 |
| FPS | 即時幀率 |
| 串流網址 | 來源網址 |

---

## 📁 專案結構

```
.
├── traffic_monitor.py   # 主程式
├── requirements.txt     # 套件需求
├── README.md            # 說明文件
├── .gitignore           # 排除模型與資料檔
└── traffic_data_*.csv   # 自動產生（gitignore 排除）
```

> 模型檔（`.weights`, `.cfg`, `.names`）首次執行時自動下載，無需放入 git。

---

## 🔑 操作快捷鍵

| 按鍵 | 功能 |
|------|------|
| `q` | 結束程式 |
| `Ctrl+C` | 強制中斷（終端機） |

---
