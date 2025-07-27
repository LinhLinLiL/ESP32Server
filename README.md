cách clone git 
1. mới clone 
git clone --recurse-submodules https://github.com/LinhLinLiL/ESP32Server.git
 2. nếu đã clone rồi 
 git submodule update --init --recursive
Hệ thống nhận diện xe đạp và biển số
Hệ thống sử dụng ESP32-CAM gửi ảnh đến server FastAPI, server xử lý ảnh (YOLO, OCR) và gửi biển số qua TCP đến ESP32 để kiểm tra và cập nhật Firebase.
Yêu cầu

Phần cứng: ESP32-CAM, ESP32, cảm biến siêu âm HY-SRF05, LED, relay.
Môi trường:
Python 3.8+.
Arduino IDE cho ESP32 và ESP32-CAM.
WiFi mạng lin (SSID: lin, mật khẩu: 12345678).


Thư viện Python:
fastapi, uvicorn, torch, ultralytics, opencv-python, pillow, easyocr, sqlalchemy.


Thư viện Arduino:
WiFi, HTTPClient, esp_camera (ESP32-CAM).
WiFi, Firebase_ESP_Client, ArduinoJson (ESP32).



Cài đặt
1. Chuẩn bị server

Cài đặt Python và thư viện:pip install fastapi uvicorn torch ultralytics opencv-python pillow easyocr sqlalchemy


Tải mô hình YOLO:
Đảm bảo file yolov5su.pt trong thư mục chạy server.


Cấu hình IP:
Kiểm tra IP máy chạy server (ví dụ: 192.168.0.102) bằng ipconfig (Windows) hoặc ifconfig (Linux).
Cập nhật serverName trong mã ESP32-CAM thành IP server (dòng: const char* serverName = "http://192.168.0.102:5000/upload_binary";).



2. Chuẩn bị ESP32-CAM

Cài Arduino IDE:
Cài thêm board ESP32 (esp32 by Espressif).
Cài thư viện esp_camera.


Upload mã:
Mở file mã ESP32-CAM, cập nhật ssid và password nếu cần.
Upload vào ESP32-CAM qua Arduino IDE.


Kiểm tra:
Mở Serial Monitor (115200 baud), xác nhận WiFi kết nối và IP.



3. Chuẩn bị ESP32

Cài thư viện Arduino:
Cài Firebase_ESP_Client, ArduinoJson.


Upload mã:
Mở file mã ESP32, kiểm tra WIFI_SSID, WIFI_PASSWORD, và thông tin Firebase.
Upload vào ESP32.


Kiểm tra IP:
Mở Serial Monitor, xác nhận WiFi.localIP() là 192.168.137.185. Nếu khác, cập nhật ESP32_IP trong mã server FastAPI.



4. Cấu hình Firebase

Tạo dự án Firebase, lấy API_KEY, DATABASE_URL, USER_EMAIL, USER_PASSWORD.
Đảm bảo cấu hình Firebase trong mã ESP32 khớp với dự án.
Thêm dữ liệu returnBikeId (ví dụ: "12") vào các đường dẫn 1/returnBikeId, 2/returnBikeId, 3/returnBikeId.

Chạy hệ thống

Chạy server FastAPI:python server.py


Server chạy trên http://0.0.0.0:5000.
Log hiển thị nhận ảnh, xử lý YOLO/OCR, và gửi JSON đến ESP32.


Chạy ESP32-CAM:
Nguồn cấp cho ESP32-CAM, camera chụp và gửi ảnh mỗi 6 giây đến /upload_binary.


Chạy ESP32:
Nguồn cấp cho ESP32, lắng nghe TCP trên cổng 80, nhận JSON { "plate": str, "lock": int }, kiểm tra với Firebase và siêu âm, cập nhật isValid.



Kiểm tra

Mạng:
Đảm bảo ESP32-CAM, ESP32, và server cùng WiFi lin.
Ping 192.168.0.102 (server) và 192.168.137.185 (ESP32) từ máy tính.


Log:
ESP32-CAM: Log Serial Monitor hiển thị "WiFi connected", "HTTP Response code: 200".
Server: Log nhận ảnh, xử lý, gửi JSON { "plate": "12", "lock": 1 }.
ESP32: Log nhận JSON, kiểm tra returnBikeId, đo khoảng cách siêu âm, cập nhật isValid.


Firebase: Kiểm tra 1/isValid, 2/isValid, 3/isValid cập nhật đúng (true nếu biển số khớp và khoảng cách < 4cm).

Lưu ý

Đảm bảo IP server và ESP32 đúng, cùng subnet (ví dụ: 192.168.0.x).
Nếu lỗi kết nối, kiểm tra WiFi, firewall, hoặc tăng timeout trong send_to_esp32 (FastAPI) hoặc handleTcpClient (ESP32).
Đảm bảo mô hình yolov5su.pt tải đúng và thư viện EasyOCR hoạt động.
