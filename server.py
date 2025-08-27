import os
import datetime
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from fastapi import FastAPI, Request, HTTPException
import torch
import cv2
import numpy as np
from PIL import Image
import easyocr
from sqlalchemy import Column, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect
import socket  # Thêm import socket


# --- Config ---
BASE_RECEIVED_FOLDER = 'received_images'
BASE_DETECTED_FOLDER = 'detected_images'
TEST_IMAGES_FOLDER = 'test_images'
DATABASE_URL = "sqlite:///D:/FolderBackup/School/Semester_9/IOP490/Project/ESP32Server/detection_results.db"
MAX_WORKERS = os.cpu_count() * 2


# Cấu hình IP và port ESP32 để gửi dữ liệu JSON qua socket TCP
ESP32_IP = '192.168.137.123'  # Thay IP ESP32 của bạn
ESP32_PORT = 80               # Thay port đúng


# Hàm gửi dữ liệu JSON tới ESP32 qua socket TCP
def send_command_to_esp32(plate_number: str, lock_number: int):
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((ESP32_IP, ESP32_PORT))
            json_data = '{"plate":"%s", "lock":%d}' % (plate_number, lock_number)
            s.sendall(json_data.encode('utf-8'))
            print(f"🚀 Đã gửi dữ liệu tới ESP32: {json_data}")
    except Exception as e:
        print(f"⚠️ Lỗi gửi dữ liệu tới ESP32: {e}")


def init_model_reader():
    """Khởi tạo model YOLO, OCR reader và tạo bảng database.
       Có thể gọi thủ công ngoài FastAPI hoặc lúc khởi động server."""
    global model, reader
    from ultralytics import YOLO
    import torch
    import easyocr

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Khởi tạo mô hình trên thiết bị: {device}")
    model = YOLO('yolov5su.pt', verbose=False)
    model.model.to(device).eval()
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    print("Mô hình YOLO và EasyOCR reader đã được khởi tạo")
    create_tables()


# Khởi tạo biến toàn cục
model = None
reader = None
LOCK_REGIONS = {1: (0, 0), 2: (0, 0), 3: (0, 0)}  # Giá trị mặc định cho LOCK_REGIONS


# SQLite DB
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()


# Session
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


# Bảng Detection (bỏ cột plate_confidence)
class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String)
    filename = Column(String)
    bike_count = Column(Integer)
    locks = Column(String)
    plate = Column(String)
    lock_number = Column(Integer)
    image_path = Column(String)
    station = Column(String)


# Tạo bảng nếu chưa tồn tại
def create_tables():
    inspector = inspect(engine)
    if not inspector.has_table("detections"):
        print("Bảng detections chưa tồn tại, đang tạo...")
        Base.metadata.create_all(bind=engine)
    else:
        print("Bảng detections đã tồn tại, không tạo lại.")


# Tạo thư mục cha nếu chưa tồn tại
for folder in [BASE_RECEIVED_FOLDER, BASE_DETECTED_FOLDER, TEST_IMAGES_FOLDER]:
    os.makedirs(folder, exist_ok=True)


def define_regions(width):
    """Xác định vùng khóa dựa trên chiều rộng ảnh."""
    LOCK_REGIONS.clear()
    LOCK_REGIONS.update({
        1: (0, width // 3),
        2: (width // 3, 2 * width // 3),
        3: (2 * width // 3, width)
    })
    print(f"📏 Vùng khóa cho chiều rộng {width}: {LOCK_REGIONS}")


# Khởi tạo FastAPI
app = FastAPI()


# Khởi tạo mô hình và reader
@app.on_event("startup")
async def load_models():
    from ultralytics import YOLO
    global model, reader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🌟 Sử dụng thiết bị: {device}")
    model = YOLO('yolov5su.pt', verbose=False)
    model.model.to(device).eval()
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    print("💾 Tạo bảng cơ sở dữ liệu...")
    create_tables()
    print("✅ Mô hình và bảng cơ sở dữ liệu đã được khởi tạo.")


# Thread pool cho tác vụ CPU
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)


async def process_image_bytes(data: bytes, station: str = None, filename: str = None) -> dict:
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

    # Định tên file gốc với timestamp
    if not station:
        station = "Unknown"
    if not filename:
        filename = f"image_{station}_{timestamp}.jpg"
    else:
        filename = f"{filename.rsplit('.', 1)[0]}_{timestamp}.jpg"

    print(f"📥 Nhận được {len(data)} bytes dữ liệu ảnh")
    if len(data) < 100:
        raise HTTPException(status_code=400, detail="Dữ liệu ảnh không hợp lệ: quá nhỏ")

    # Giải mã ảnh
    t0 = time.perf_counter()
    arr = np.frombuffer(data, np.uint8)
    img_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise HTTPException(status_code=400, detail="Không thể giải mã ảnh")
    decode_ms = int((time.perf_counter() - t0) * 1000)
    print(f"🖼️ Giải mã ảnh thành công: {decode_ms} ms")

    # Xác định vùng khóa và vẽ lên ảnh
    h, w, _ = img_cv.shape
    define_regions(w)
    t1 = time.perf_counter()
    zoned_image = img_cv.copy()  # Sao chép để vẽ vùng khóa
    for x in [w // 3, 2 * w // 3]:
        cv2.line(zoned_image, (x, 0), (x, h), (0, 255, 0), 2)
    for i, x in enumerate([w // 6, w // 2, 5 * w // 6], start=1):
        cv2.putText(zoned_image, str(i), (x - 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    draw_ms = int((time.perf_counter() - t1) * 1000)
    print(f"✅ Vẽ vùng khóa thành công trong {draw_ms} ms")

    # Lưu ảnh đã chia slot (zoned_image) vào thư mục received_images (thay vì ảnh gốc)
    raw_dir = os.path.join(BASE_RECEIVED_FOLDER, station)
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, filename)
    if not cv2.imwrite(raw_path, zoned_image):
        raise HTTPException(status_code=500, detail=f"Không thể lưu ảnh chia slot tại {raw_path}")
    print(f"✅ Lưu ảnh chia slot tại {raw_path}")

    # Thư mục detected và đường dẫn ảnh final
    det_dir = os.path.join(BASE_DETECTED_FOLDER, station)
    os.makedirs(det_dir, exist_ok=True)
    final_image_path = None

    # Các phần phát hiện, OCR, vẽ nhãn giữ nguyên như cũ
    t2 = time.perf_counter()
    pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    results = model.predict(pil_img, imgsz=640)
    yolo_ms = int((time.perf_counter() - t2) * 1000)
    print(f"🔎 Thời gian YOLO: {yolo_ms} ms")

    lock_has = {ln: False for ln in LOCK_REGIONS}
    boxes = results[0].boxes
    names = model.names
    bike_lock_map = {}

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        name = names[cls_id]
        if name == 'bicycle':
            xyxy = boxes.xyxy[i].cpu().numpy()
            xmin, _, xmax, _ = xyxy
            cx = (xmin + xmax) / 2
            for ln, (rxmin, rxmax) in LOCK_REGIONS.items():
                if rxmin <= cx <= rxmax:
                    lock_has[ln] = True
                    bike_lock_map[i] = {'lock': ln, 'center_x': cx}
                    print(f"🚲 Phát hiện xe đạp tại khóa {ln} (center_x: {cx})")
                    break

    detected = [ln for ln, v in lock_has.items() if v]
    print(f"📊 Phát hiện {len(detected)} xe đạp tại các khóa: {detected}")

    det_image = img_cv.copy()

    # ---- Vẽ bounding box xe đạp ----
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        if names[cls_id] == 'bicycle':
            xyxy = boxes.xyxy[i].cpu().numpy()
            xmin, ymin, xmax, ymax = map(int, xyxy)
            cv2.rectangle(det_image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)  # Xanh lá cho xe đạp
            cv2.putText(det_image, 'bicycle', (xmin, ymin - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # OCR trong bounding box của từng xe đạp thay vì trên toàn ảnh
    plates_sent = []
    plate_positions = []
    used_locks = set()

    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        if names[cls_id] == 'bicycle':
            xyxy = boxes.xyxy[i].cpu().numpy()
            xmin, ymin, xmax, ymax = map(int, xyxy)

            # Cắt ảnh vùng bounding box xe đạp
            crop_img = img_cv[ymin:ymax, xmin:xmax]

            # OCR trong vùng bounding box
            ocr_results = reader.readtext(crop_img, allowlist='0123456789')

            # Xử lý kết quả OCR trong bounding box đó
            for item in ocr_results:
                if len(item) != 3:
                    print(f"🚫 Bỏ qua kết quả OCR không hợp lệ trong bounding box {i}: {item}")
                    continue
                bbox, text, _ = item
                if len(text) != 2 or not text.isdigit():
                    print(f"🚫 Bỏ qua biển số không hợp lệ trong bounding box {i}: '{text}'")
                    continue

                cx_crop = sum(pt[0] for pt in bbox) / len(bbox)
                cy_crop = sum(pt[1] for pt in bbox) / len(bbox)

                # Tính tọa độ tuyệt đối trong ảnh gốc
                cx = cx_crop + xmin
                cy = cy_crop + ymin

                # Gán biển số cho khóa tương ứng
                assigned = False
                for lock_num, (rxmin, rxmax) in LOCK_REGIONS.items():
                    if rxmin <= cx <= rxmax and lock_has[lock_num] and lock_num not in used_locks:
                        print(f"🔗 Gán biển số '{text}' cho khóa {lock_num} (plate_center_x: {cx}) trong bounding box {i}")
                        send_command_to_esp32(text, lock_num)
                        plate_positions.append((text, lock_num, cx, cy))
                        plates_sent.append((text, lock_num))
                        used_locks.add(lock_num)
                        assigned = True
                        break
                if not assigned:
                    print(f"❓ Biển số '{text}' (center_x: {cx}) không gán được trong bounding box {i}")

    # Vẽ nhãn biển số lên ảnh nếu có
    if plates_sent:
        for plate, lock_num, cx, y in plate_positions:
            text_pos = (int(cx - 50), int(y - 10))
            cv2.putText(det_image, f"Plate: {plate} (Lock {lock_num})", text_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        final_image_path = os.path.join(det_dir, f"final_{filename}")
        if not cv2.imwrite(final_image_path, det_image):
            print(f"❌ Lỗi: Không thể lưu ảnh chú thích tại {final_image_path}")
            final_image_path = None
        else:
            print(f"✅ Lưu ảnh chú thích tại {final_image_path}")
    else:
        final_image_path = None

    # Lưu thông tin vào database giữ nguyên
    if bike_lock_map:
        with SessionLocal() as db_session:
            try:
                for i, info in bike_lock_map.items():
                    lock_num = info['lock']
                    plate = next((p[0] for p in plates_sent if p[1] == lock_num), "x")
                    detection = Detection(
                        timestamp=datetime.datetime.now().isoformat(),
                        filename=filename,
                        bike_count=len(detected),
                        locks='|'.join(map(str, detected)) if detected else '-',
                        plate=plate,
                        lock_number=lock_num,
                        image_path=final_image_path,
                        station=station
                    )
                    db_session.add(detection)
                db_session.commit()
                print(f"[INFO] 💾 Lưu vào cơ sở dữ liệu thành công: {len(bike_lock_map)} bản ghi")
            except Exception as e:
                print(f"[ERROR] ❌ Lỗi khi lưu vào cơ sở dữ liệu: {e}")
                db_session.rollback()
                raise HTTPException(status_code=500, detail=f"Lỗi lưu cơ sở dữ liệu: {str(e)}")

    return {
        'filename': filename,
        'detected_bikes': len(detected),
        'locks_with_bikes': detected,
        'plates': plates_sent,
        'station': station,
        'image_path': final_image_path
    }


@app.post('/upload_binary')
async def upload_binary(request: Request):
    """Xử lý tải lên ảnh nhị phân thô."""
    data = await request.body()
    if not data:
        raise HTTPException(status_code=400, detail='Không nhận được dữ liệu')

    filename = request.headers.get('X-Filename', None)
    station = request.headers.get('X-Station', 'Unknown')  # Ưu tiên lấy station từ header

    print(f"📩 Nhận yêu cầu với filename: {filename}, station: {station}")
    result = await process_image_bytes(data, station, filename)

    return {
        'status': 'success',
        'plates': result['plates'],
        'locks': result['locks_with_bikes'],
        **result
    }


def get_local_ip():
    """Lấy địa chỉ IP cục bộ của máy đang chạy server."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


if __name__ == '__main__':
    ip = get_local_ip()
    port = 5000
    print(f"🌐 Server đang chạy tại http://{ip}:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)


