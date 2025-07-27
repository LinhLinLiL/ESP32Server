import os
import datetime
import socket
import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
import uvicorn
from fastapi import FastAPI, UploadFile, HTTPException
import torch
import cv2
import numpy as np
from PIL import Image
import easyocr
from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy import inspect  # Thêm để kiểm tra bảng tồn tại
import json  ### NEW: Thêm import json để sử dụng json.dumps trong send_to_esp32

# --- Config ---
BASE_RECEIVED_FOLDER = 'received_images'
BASE_DETECTED_FOLDER = 'detected_images'
TEST_IMAGES_FOLDER = 'test_images'
DATABASE_URL = "sqlite:///D:/FolderBackup/School/Semester_9/IOP490/Project/ESP32Server/detection_results.db"
ESP32_IP = '192.168.137.185'
ESP32_PORT = 80
MAX_WORKERS = os.cpu_count() * 2

model = None
reader = None

def init_model_reader():
    global model, reader
    from ultralytics import YOLO
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Khoi tao mo hinh tren: {device}")
    model = YOLO('yolov5su.pt', verbose=False)
    model.model.to(device).eval()
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    print("Mo hinh va OCR Reader da khoi tao")
    create_tables()

# SQLite DB
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
Base = declarative_base()

# Session
SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)

# Bang Detection
class Detection(Base):
    __tablename__ = "detections"
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(String)
    filename = Column(String)
    bike_count = Column(Integer)
    locks = Column(String)
    plate = Column(String)
    plate_confidence = Column(Float)
    lock_number = Column(Integer)
    image_path = Column(String)
    station = Column(String)  # Them cot station

# Tao bang neu chua ton tai, xoa neu da ton tai
def create_tables():
    inspector = inspect(engine)  # Sử dụng inspector để kiểm tra bảng
    if inspector.has_table("detections"):
        print("Bảng detections đã tồn tại, xóa và tạo lại...")
        Base.metadata.drop_all(bind=engine, tables=[Detection.__table__])  # Xóa bảng
    Base.metadata.create_all(bind=engine)  # Tạo bảng mới
    print("Bảng cơ sở dữ liệu đã được tạo hoặc tạo lại.")

# Tao thu muc cha neu chua ton tai
for folder in [BASE_RECEIVED_FOLDER, BASE_DETECTED_FOLDER, TEST_IMAGES_FOLDER]:
    os.makedirs(folder, exist_ok=True)

### NEW: Thêm hàm send_to_esp32 để gửi JSON thực qua TCP đến ESP32
def send_to_esp32(plate, lock_num):
    max_retries = 3  ### NEW: Số lần thử lại tối đa
    retry_delay = 2  ### NEW: Thời gian chờ giữa các lần thử (giây)
    timeout = 10  ### NEW: Timeout 10 giây cho kết nối ổn định

    for attempt in range(1, max_retries + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(timeout)
                print(f"📡 Thử kết nối TCP lần {attempt} đến ESP32 ({ESP32_IP}:{ESP32_PORT})")
                s.connect((ESP32_IP, ESP32_PORT))
                data = {"plate": str(plate), "lock": int(lock_num)}  # Đảm bảo plate là chuỗi, lock là số
                json_data = json.dumps(data)
                s.sendall(json_data.encode('utf-8'))
                print(f"📤 Gửi JSON thành công: {json_data} đến ESP32 ({ESP32_IP}:{ESP32_PORT})")
                return True
        except socket.timeout:
            print(f"❌ Lỗi: Timeout (lần {attempt}/{max_retries}) khi kết nối đến ESP32 tại {ESP32_IP}:{ESP32_PORT}")
            if attempt < max_retries:
                print(f"⏳ Chờ {retry_delay} giây trước khi thử lại...")
                time.sleep(retry_delay)
        except socket.error as e:
            print(f"❌ Lỗi TCP (lần {attempt}/{max_retries}) khi gửi đến ESP32: {e}")
            if attempt < max_retries:
                print(f"⏳ Chờ {retry_delay} giây trước khi thử lại...")
                time.sleep(retry_delay)
    print(f"❌ Thất bại sau {max_retries} lần thử gửi TCP đến ESP32")
    return False

async def process_image_bytes(data: bytes, station: str = None) -> dict:
    """Xu ly anh de phat hien xe dap va bien so."""
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
   
    # Nhận diện station từ dữ liệu (giả định tên file chứa station, ví dụ: station_Alpha_...)
    if not station:
        filename = f"image_{timestamp}.jpg"  # Giả định nếu không có station
        station = "Unknown"
        if filename.startswith('station_Alpha_'):
            station = 'Alpha'
        elif filename.startswith('station_Beta_'):
            station = 'Beta'
    else:
        filename = f"image_{station}_{timestamp}.jpg"

    # Kiem tra du lieu anh
    if len(data) < 100:
        raise HTTPException(status_code=400, detail="Du lieu anh khong hop le: qua nho")
   
    # Giai ma anh
    t0 = time.perf_counter()
    arr = np.frombuffer(data, np.uint8)
    img_cv = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_cv is None:
        raise HTTPException(status_code=400, detail="Khong the giai ma anh")
    decode_ms = int((time.perf_counter() - t0) * 1000)

    # Luu anh goc vào folder tương ứng với station
    raw_dir = os.path.join(BASE_RECEIVED_FOLDER, station)
    os.makedirs(raw_dir, exist_ok=True)
    raw_path = os.path.join(raw_dir, filename)
    cv2.imwrite(raw_path, img_cv)
    print(f"📥 Luu anh goc tai {raw_path}")

    # Xac dinh vung va ve duong
    h, w, _ = img_cv.shape
    define_regions(w)
    t1 = time.perf_counter()
    for x in [w//3, 2*w//3]:
        cv2.line(img_cv, (x, 0), (x, h), (0, 255, 0), 2)
    for i, x in enumerate([w//6, w//2, 5*w//6], start=1):
        cv2.putText(img_cv, str(i), (x-10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    draw_ms = int((time.perf_counter() - t1) * 1000)
    cv2.imwrite(raw_path, img_cv)
    print(f"🎨 Ve vung va nhan tren anh: {raw_path}")

    # Phat hien YOLO
    t2 = time.perf_counter()
    pil_img = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    results = model.predict(pil_img, imgsz=640)
    yolo_ms = int((time.perf_counter() - t2) * 1000)

    lock_has = {ln: False for ln in LOCK_REGIONS}
    boxes = results[0].boxes
    names = model.names
    bike_centers = []
    bike_lock_map = {}

    # Phat hien xe dap va gan vao vung khoa
    for i in range(len(boxes)):
        cls_id = int(boxes.cls[i])
        name = names[cls_id]
        if name == 'bicycle':
            xyxy = boxes.xyxy[i].cpu().numpy()
            xmin, _, xmax, _ = xyxy
            cx = (xmin + xmax) / 2
            bike_centers.append(cx)
            for ln, (rxmin, rxmax) in LOCK_REGIONS.items():
                if rxmin <= cx <= rxmax:
                    lock_has[ln] = True
                    bike_lock_map[i] = {'lock': ln, 'center_x': cx}
                    print(f"🚲 Phat hien xe dap tai khoa {ln} (center_x: {cx})")
                    break

    detected = [ln for ln, v in lock_has.items() if v]
    print(f"📊 Phat hien {len(detected)} xe dap tai cac khoa: {detected}")

    # OCR tren toan bo anh
    t3 = time.perf_counter()
    ocr_results = reader.readtext(img_cv, allowlist='0123456789')
    ocr_ms = int((time.perf_counter() - t3) * 1000)
    print(f"⏱️ Thoi gian EasyOCR: {ocr_ms} ms")
    print(f"🔍 Ket qua OCR tren toan anh: {ocr_results}")

    # Sao chep anh goc de chu thich
    det_image = img_cv.copy()

    # Ve khung YOLO
    for box in boxes:
        if names[int(box.cls[0])] == 'bicycle':
            xyxy = box.xyxy[0].cpu().numpy()
            xmin, ymin, xmax, ymax = xyxy
            cv2.rectangle(det_image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)

    # Luu anh da phat hien voi khung vao folder tuong ung voi station
    det_dir = os.path.join(BASE_DETECTED_FOLDER, station)
    os.makedirs(det_dir, exist_ok=True)
    final_image_path = os.path.join(det_dir, f"final_{filename}")
    if not cv2.imwrite(final_image_path, det_image):
        print(f"❌ Loi: Khong the luu anh tai {final_image_path}")
        final_image_path = None
    else:
        print(f"✅ Luu anh chu thich voi khung YOLO tai {final_image_path}")

    # Xu ly bien so hop le
    socket_ms = 0
    plates_sent = []
    plate_positions = []
    plate_centers = []
    used_locks = set()
    for item in ocr_results:
        if len(item) != 3:
            print(f"🚫 Bo qua ket qua OCR khong hop le: {item}")
            continue
        bbox, text, conf = item
        if len(text) != 2 or not text.isdigit():
            print(f"🚫 Bo qua bien so khong hop le: '{text}' (do dai: {len(text)}, la so: {text.isdigit()})")
            continue
        if conf < 0.3:
            print(f"🚫 Bo qua bien so '{text}' do do tin cay thap: {conf}")
            continue
        cx = sum(pt[0] for pt in bbox) / len(bbox)
        plate_centers.append(cx)
        assigned = False
        for lock_num, (rxmin, rxmax) in LOCK_REGIONS.items():
            if rxmin <= cx <= rxmax and lock_has[lock_num] and lock_num not in used_locks:
                print(f"🔗 Gan bien so '{text}' cho khoa {lock_num} (plate_center_x: {cx}, conf: {conf})")
                plate_positions.append((text, lock_num, cx, bbox[0][1]))
                ### NEW: Thay [MO PHONG] bằng gửi TCP thực đến ESP32
                if send_to_esp32(text, lock_num):
                    plates_sent.append((text, lock_num))
                    used_locks.add(lock_num)
                    assigned = True
                else:
                    print(f"⚠️ Gửi biển số '{text}' cho khóa {lock_num} thất bại")
                ### OLD: Dòng giả lập
                # print(f"[MO PHONG] 📤 Gui bien so '{text}' cho khoa {lock_num} den ESP32")
                # plates_sent.append((text, lock_num))
                # used_locks.add(lock_num)
                # assigned = True
                break
        if not assigned:
            print(f"❓ Bien so '{text}' (center_x: {cx}, conf: {conf}) khong gan duoc: khong co khoa hoac khong co xe")

    # Ve bien so len anh
    for plate, lock_num, cx, y in plate_positions:
        text_pos = (int(cx - 50), int(y - 10))
        cv2.putText(det_image, f"Plate: {plate} (Lock {lock_num})", text_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    # Luu anh cuoi cung voi bien so
    if final_image_path and not cv2.imwrite(final_image_path, det_image):
        print(f"❌ Loi: Khong the cap nhat anh voi bien so tai {final_image_path}")
        final_image_path = None
    else:
        print(f"✅ Cap nhat anh voi bien so tai {final_image_path}")

    if not plates_sent:
        print("⚠️ Khong tim thay bien so hop le hoac gui that bai")
    else:
        print(f"📦 Da gui bien so: {plates_sent}")

    # Luu vao SQL
    db_session = SessionLocal()
    try:
        for i, info in bike_lock_map.items():
            lock_num = info['lock']
            plate = next((p[0] for p in plates_sent if p[1] == lock_num), "x")
            plate_confidence = float(next((item[2] for item in ocr_results if item[1] == plate), 0.0)) if plate != "x" else 0.0
            detection = Detection(
                timestamp=datetime.datetime.now().isoformat(),
                filename=filename,
                bike_count=len(detected),
                locks='|'.join(map(str, detected)) if detected else '-',
                plate=plate,
                plate_confidence=plate_confidence,
                lock_number=lock_num,
                image_path=final_image_path,
                station=station
            )
            db_session.add(detection)
        db_session.commit()
        print(f"[INFO] 💾 Luu vao co so du lieu thanh cong: {len(bike_lock_map)} ban ghi")
    except Exception as e:
        print(f"[ERROR] ❌ Loi khi luu vao co so du lieu: {e}")
        db_session.rollback()
    finally:
        db_session.close()

    return {
        'filename': filename,
        'detected_bikes': len(detected),
        'locks_with_bikes': detected,
        'plates': plates_sent,
        'station': station,
        'image_path': final_image_path
    }

# Khoi tao FastAPI
app = FastAPI()

# Khoi tao mo hinh
@app.on_event("startup")
async def load_models():
    from ultralytics import YOLO
    global model, reader
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🌟 Su dung thiet bi: {device}")
    model = YOLO('yolov5su.pt', verbose=False)
    model.model.to(device).eval()
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    print("💾 Tao bang co so du lieu...")
    create_tables()
    print("✅ Bang co so du lieu da duoc tao hoac da ton tai.")

# Thread pool cho tac vu CPU
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
LOCK_REGIONS = {}

def define_regions(width):
    """Xac dinh vung khoa dua tren chieu rong anh."""
    LOCK_REGIONS.clear()
    LOCK_REGIONS.update({
        1: (0, width//3),
        2: (width//3, 2*width//3),
        3: (2*width//3, width)
    })
    print(f"📏 Vung khoa cho chieu rong {width}: {LOCK_REGIONS}")

@app.post('/upload_binary')
async def upload_binary(file: UploadFile):
    """Xu ly tai len anh nhi phan."""
    # #################################################################################
    # Phần này dành cho thiết bị thật: Nhận dữ liệu ảnh từ ESP32
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail='Khong nhan duoc du lieu')
    # Lấy tên file từ header X-Filename (ưu tiên hơn file.filename)
    filename = file.filename or file.headers.get('X-Filename', 'unknown.jpg')
    station = "Unknown"
    if "station_Alpha" in filename:
        station = "Alpha"
    elif "station_Beta" in filename:
        station = "Beta"
    # #################################################################################
    result = await process_image_bytes(data, station)
    return {'status': 'success', **result}

if __name__ == '__main__':
    uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=False)