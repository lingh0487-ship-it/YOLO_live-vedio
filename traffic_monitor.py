"""
高雄交通監控系統 - 無 PyTorch 依賴版 + CSV 輸出
支援自訂串流網址（命令列參數 / 互動式輸入）
"""

import cv2
import time
import os
import urllib.request
import csv
import argparse
import sys
from datetime import datetime

print("=" * 50)
print("高雄交通監控系統初始化")
print("=" * 50)

# ============= 解析命令列參數 =============
def parse_args():
    parser = argparse.ArgumentParser(
        description="高雄交通監控系統 - 即時車輛偵測",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help=(
            "串流網址（RTSP / HTTP / MJPEG 均可）\n"
            "範例：\n"
            "  --url rtsp://192.168.1.1:554/stream\n"
            "  --url https://trafficvideo2.tainan.gov.tw/e0029830\n"
            "  --url 0   (使用本機攝影機)\n"
            "若未提供，程式會以互動方式詢問。"
        )
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=5,
        metavar="秒",
        help="每隔幾秒儲存一次 CSV（預設 5）"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.3,
        metavar="0~1",
        help="偵測信心閾值（預設 0.3）"
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="無頭模式：不開啟視窗，僅輸出 CSV（適合伺服器）"
    )
    return parser.parse_args()

# ============= 取得串流網址 =============
DEFAULT_URL = 'https://trafficvideo2.tainan.gov.tw/e0029830'

def get_stream_url(args_url: str | None) -> str:
    """優先順序：命令列參數 > 互動輸入 > 預設值"""
    if args_url is not None:
        # 允許用 "0" 代表本機攝影機
        url = 0 if args_url.strip() == "0" else args_url.strip()
        print(f"✓ 使用命令列串流網址: {url}")
        return url

    print(f"\n預設串流網址: {DEFAULT_URL}")
    user_input = input("請輸入串流網址（直接按 Enter 使用預設值）: ").strip()

    if user_input == "":
        print(f"✓ 使用預設網址: {DEFAULT_URL}")
        return DEFAULT_URL
    elif user_input == "0":
        print("✓ 使用本機攝影機 (index 0)")
        return 0
    else:
        print(f"✓ 使用自訂網址: {user_input}")
        return user_input

# ============= 自動下載模型檔案 =============
def download_model_files():
    files = {
        "yolov4-tiny.weights": "https://github.com/AlexeyAB/darknet/releases/download/yolov4/yolov4-tiny.weights",
        "yolov4-tiny.cfg":     "https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4-tiny.cfg",
        "coco.names":          "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }
    for filename, url in files.items():
        if not os.path.exists(filename):
            print(f"下載 {filename}...")
            try:
                urllib.request.urlretrieve(url, filename)
                print(f"✓ {filename} 下載完成")
            except Exception as e:
                print(f"✗ 下載 {filename} 失敗: {e}")
                return False
        else:
            print(f"✓ {filename} 已存在")
    return True

# ============= 載入模型 =============
print("\n檢查模型檔案...")
if not download_model_files():
    print("\n✗ 無法下載模型檔案，請手動下載後重試。")
    sys.exit(1)

print("\n載入 YOLO 模型...")
try:
    import numpy as np
    net = cv2.dnn.readNet("yolov4-tiny.weights", "yolov4-tiny.cfg")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    print("✓ 模型載入成功")
except Exception as e:
    print(f"✗ 模型載入失敗: {e}")
    sys.exit(1)

# ============= 常數設定 =============
VEHICLE_CLASS_NAMES = ["car", "motorcycle", "bus", "truck"]
VEHICLE_COLORS = {
    "Car":        (0, 255, 0),
    "Motorcycle": (255, 0, 0),
    "Bus":        (0, 165, 255),
    "Truck":      (0, 0, 255)
}

# ============= CSV 工具 =============
def init_csv(filename: str):
    with open(filename, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow(['時間戳記', '日期', '時刻', '總車輛數', '汽車', '機車', '巴士', '卡車', 'FPS', '串流網址'])
    print(f"✓ CSV 檔案已建立: {filename}")

def write_to_csv(filename: str, vehicle_counts: dict, fps: float, stream_url):
    try:
        now = datetime.now()
        total = sum(vehicle_counts.values())
        with open(filename, 'a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            writer.writerow([
                now.strftime('%Y-%m-%d %H:%M:%S'),
                now.strftime('%Y-%m-%d'),
                now.strftime('%H:%M:%S'),
                total,
                vehicle_counts['Car'],
                vehicle_counts['Motorcycle'],
                vehicle_counts['Bus'],
                vehicle_counts['Truck'],
                f'{fps:.2f}',
                stream_url
            ])
    except Exception as e:
        print(f"CSV 寫入錯誤: {e}")

# ============= 車輛偵測 =============
def detect_vehicles(frame, confidence_threshold: float = 0.3):
    try:
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)

        layer_names = net.getLayerNames()
        unconnected = net.getUnconnectedOutLayers()
        if isinstance(unconnected, np.ndarray):
            output_layers = (
                [layer_names[i - 1] for i in unconnected]
                if len(unconnected.shape) == 1
                else [layer_names[i[0] - 1] for i in unconnected]
            )
        else:
            output_layers = [layer_names[i - 1] for i in unconnected]

        outputs = net.forward(output_layers)

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                conf = float(scores[class_id])
                if conf > confidence_threshold and class_id < len(classes) and classes[class_id] in VEHICLE_CLASS_NAMES:
                    cx = int(detection[0] * width)
                    cy = int(detection[1] * height)
                    w  = int(detection[2] * width)
                    h  = int(detection[3] * height)
                    boxes.append([cx - w // 2, cy - h // 2, w, h])
                    confidences.append(conf)
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, confidence_threshold, 0.4)
        vehicle_counts = {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}

        if len(indices) > 0:
            if isinstance(indices, tuple):
                indices = indices[0] if indices else []
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()

            for i in indices:
                i = int(i)
                x, y, w, h = boxes[i]
                label = classes[class_ids[i]]
                if label in VEHICLE_CLASS_NAMES:
                    vname = label.capitalize()
                    vehicle_counts[vname] += 1
                    color = VEHICLE_COLORS[vname]
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    text = f'{vname} {confidences[i]:.2f}'
                    tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                    cv2.rectangle(frame, (x, y - 20), (x + tw, y), color, -1)
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame, vehicle_counts
    except Exception as e:
        print(f"偵測錯誤: {e}")
        return frame, {"Car": 0, "Motorcycle": 0, "Bus": 0, "Truck": 0}

# ============= 繪製 HUD =============
def draw_info(frame, vehicle_counts: dict, fps: float = 0, csv_status: str = "", stream_url: str = ""):
    try:
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (5, 10), (220, 200), (0, 0, 0), -1)

        y = 30
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y += 25
        total = sum(vehicle_counts.values())
        cv2.putText(frame, f'Total: {total}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        for vtype, count in vehicle_counts.items():
            if count > 0:
                y += 25
                cv2.putText(frame, f'{vtype}: {count}', (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, VEHICLE_COLORS[vtype], 2)

        if csv_status:
            y += 25
            cv2.putText(frame, csv_status, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # 顯示截斷後的網址
        short_url = (str(stream_url)[:35] + '…') if len(str(stream_url)) > 36 else str(stream_url)
        cv2.putText(frame, short_url, (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        # 圖例
        legend_y = h - 130
        cv2.rectangle(frame, (5, legend_y - 10), (180, h - 25), (0, 0, 0), -1)
        cv2.putText(frame, 'Legend:', (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        for vtype, color in VEHICLE_COLORS.items():
            legend_y += 25
            cv2.rectangle(frame, (10, legend_y - 15), (30, legend_y - 5), color, -1)
            cv2.putText(frame, vtype, (35, legend_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    except Exception as e:
        print(f"繪製錯誤: {e}")
    return frame

# ============= 主程式 =============
def main():
    args = parse_args()
    stream_url = get_stream_url(args.url)

    print("\n" + "=" * 50)
    print("系統啟動")
    print("=" * 50)

    csv_filename = f"traffic_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    init_csv(csv_filename)

    print(f"\n連接串流: {stream_url}")
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("✗ 無法開啟串流，請確認網址是否正確。")
        return

    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("✓ 串流連接成功")
    if not args.no_display:
        print("按 'q' 鍵結束程式\n")

    frame_count   = 0
    start_time    = time.time()
    fps           = 0
    last_save     = time.time()
    csv_status    = ""

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("連線中斷，重新連接...")
                time.sleep(2)
                cap.release()
                cap = cv2.VideoCapture(stream_url)
                continue

            frame, vehicle_counts = detect_vehicles(frame, args.confidence)

            frame_count += 1
            if frame_count % 10 == 0:
                fps = frame_count / (time.time() - start_time)

            now = time.time()
            if now - last_save >= args.save_interval:
                write_to_csv(csv_filename, vehicle_counts, fps, stream_url)
                last_save  = now
                csv_status = "CSV saved"
                print(f"[儲存] {datetime.now().strftime('%H:%M:%S')} | Total: {sum(vehicle_counts.values())} | {vehicle_counts}")

            if not args.no_display:
                frame = draw_info(frame, vehicle_counts, fps, csv_status, stream_url)
                if csv_status and now - last_save > 1:
                    csv_status = ""
                cv2.imshow('Traffic Monitor', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                if frame_count % 30 == 0:
                    total = sum(vehicle_counts.values())
                    print(f"[{frame_count:04d}] FPS: {fps:5.1f} | Vehicles: {total:3d} | {vehicle_counts}")

    except KeyboardInterrupt:
        print("\n程式被中斷")
    finally:
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()
        print(f"\n資料已儲存至: {csv_filename}")
        print("系統關閉")

if __name__ == "__main__":
    main()
