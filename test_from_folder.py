import os
import asyncio
import datetime
import logging
import shutil
import sys
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


# --- Logging setup ---
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('processing.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


# --- Import tu server ---
from server import process_image_bytes, init_model_reader


# --- Cau hinh ---
WATCHED_FOLDERS = {
    'Alpha': {'input': 'test_image_Alpha', 'output': 'test_detected_image_Alpha'},
    'Beta': {'input': 'test_image_Beta', 'output': 'test_detected_image_Beta'}
}
MAX_CONCURRENT_TASKS = 4


class ImageFileHandler(FileSystemEventHandler):
    def __init__(self, folder_station_map, loop, queue):
        self.folder_station_map = folder_station_map
        self.loop = loop
        self.queue = queue


    def on_created(self, event):
        if event.is_directory or not event.src_path.lower().endswith(('.jpg', '.png')):
            return


        # Lay ten file va xac dinh tram tu ten file
        filename = os.path.basename(event.src_path)
        station = None
        if filename.startswith('station_Alpha_'):
            station = 'Alpha'
        elif filename.startswith('station_Beta_'):
            station = 'Beta'
        else:
            logger.warning(f"🚫 Bo qua file {filename}: khong khop dinh dang (station_Alpha_* hoac station_Beta_*)")
            return


        # Kiem tra file nam dung thu muc
        folder = os.path.dirname(event.src_path)
        expected_folder = self.folder_station_map.get(station, {}).get('input')
        if expected_folder and os.path.normpath(folder) != os.path.normpath(expected_folder):
            logger.warning(f"🚫 File {filename} trong thu muc sai {folder}, ky vong {expected_folder}")
            return


        asyncio.run_coroutine_threadsafe(
            self.queue.put((station, event.src_path)), self.loop
        )
        logger.info(f"📥 [+] [{station}] Da xep hang: {filename}")


# Khoi tao mo hinh
init_model_reader()


async def process_image_queue(queue):
    while True:
        station, path = await queue.get()
        try:
            await asyncio.sleep(0.5)
            initial_size = os.path.getsize(path)
            await asyncio.sleep(0.5)
            if os.path.getsize(path) != initial_size:
                logger.warning(f"⏳ File {path} dang duoc ghi, bo qua")
                continue


            # Đọc ảnh từ file và chuyển thành bytes
            with open(path, 'rb') as f:
                data = f.read()


            # Gọi process_image_bytes với station
            result = await process_image_bytes(data, station=station)
            logger.info(f"✅ [+] [{station}] Da xu ly: {os.path.basename(path)} | Ket qua: {result}")


            # Di chuyen anh da xu ly den thu muc tuong ung
            result_image_path = result.get('image_path')
            if result_image_path and os.path.exists(result_image_path):
                dst_folder = WATCHED_FOLDERS[station]['output']
                os.makedirs(dst_folder, exist_ok=True)
                dst_path = os.path.join(dst_folder, os.path.basename(result_image_path))
                try:
                    shutil.move(result_image_path, dst_path)
                    logger.info(f"📤 [+] [{station}] Di chuyen anh den: {dst_path}")
                except Exception as e:
                    logger.error(f"❌ [x] Loi khi di chuyen anh {result_image_path} den {dst_path}: {e}")
            else:
                logger.warning(f"⚠️ Khong tim thay anh da xu ly tai {result_image_path} de di chuyen cho {path}")


        except Exception as e:
            logger.error(f"❌ [x] Loi khi xu ly [{station}] {os.path.basename(path)}: {e}")
        finally:
            queue.task_done()


async def main():
    loop = asyncio.get_event_loop()
    image_queue = asyncio.Queue()


    observers = []
    for station, folders in WATCHED_FOLDERS.items():
        input_folder = folders['input']
        output_folder = folders['output']
        if not os.path.exists(input_folder):
            os.makedirs(input_folder, exist_ok=True)
            logger.info(f"📁 Tao thu muc dau vao: {input_folder}")
        if not os.path.exists(output_folder):
            os.makedirs(output_folder, exist_ok=True)
            logger.info(f"📁 Tao thu muc dau ra: {output_folder}")


        handler = ImageFileHandler(WATCHED_FOLDERS, loop, image_queue)
        observer = Observer()
        observer.schedule(handler, input_folder, recursive=False)
        observer.start()
        observers.append(observer)
        logger.info(f"👀 [+] Theo doi thu muc: {input_folder} cho tram: {station}")


    # Khoi dong workers
    workers = [asyncio.create_task(process_image_queue(image_queue)) for _ in range(MAX_CONCURRENT_TASKS)]


    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("⏹️ Tat trinh theo doi...")
        for observer in observers:
            observer.stop()
        for worker in workers:
            worker.cancel()
        await image_queue.join()


    for observer in observers:
        observer.join()


if __name__ == '__main__':
    asyncio.run(main())

