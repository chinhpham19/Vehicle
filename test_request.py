import numpy as np, asyncio
import requests, random
from requests.adapters import HTTPAdapter, Retry
from multiprocessing import Process
import logging, cv2

# Đọc dữ liệu từ webcam or video
video_path = './test_image/cars.mp4'
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(video_path)

# Thiết lập hạn chế kết nối cho requests
session = requests.Session()
retry = Retry(connect=3, backoff_factor=0.5)
adapter = HTTPAdapter(max_retries=retry)
session.mount('http://', adapter)
session.mount('https://', adapter)

# Cấu hình logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

colors = [(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)) for j in range(10)]

def draw_box(image: np.ndarray, box: np.ndarray, color: tuple[int, int, int] = (0, 0, 255), thickness: int = 2) -> np.ndarray:
    x1, y1, x2, y2 = map(int, box)
    line_length = 15

    # Top-left corner
    cv2.line(image, (x1, y1), (x1 + line_length, y1), color, thickness)
    cv2.line(image, (x1, y1), (x1, y1 + line_length), color, thickness)

    # Top-right corner
    cv2.line(image, (x2, y1), (x2 - line_length, y1), color, thickness)
    cv2.line(image, (x2, y1), (x2, y1 + line_length), color, thickness)

    # Bottom-left corner
    cv2.line(image, (x1, y2), (x1 + line_length, y2), color, thickness)
    cv2.line(image, (x1, y2), (x1, y2 - line_length), color, thickness)

    # Bottom-right corner
    cv2.line(image, (x2, y2), (x2 - line_length, y2), color, thickness)
    cv2.line(image, (x2, y2), (x2, y2 - line_length), color, thickness)

    return image

async def post_server():
    """
    This Python function continuously reads frames from the webcam, encodes them as JPEG, and sends them to an API
    as a POST request. The result is saved to a file and printed to the console.
    """
    frame_nmr = -1
    ret = True
    while ret:
        frame_nmr += 1
        ret, frame = cap.read()
        frame = cv2.resize(frame, (1020, 500))
        if not ret:
            logger.error("Failed to read frame from webcam")
            break

        # Gửi yêu cầu POST đến API với tệp ảnh JPEG
        imencoded = cv2.imencode(".jpg", frame)[1]
        files = {'file': ('image.jpg', imencoded.tobytes(), 'image/jpeg')}
        try:
            response = session.request(
                "POST",
                "http://localhost:6000/detections",
                files=files
            )

            # Kiểm tra phản hồi
            if response.status_code == 200:
                result = response.json()
                logger.info(f"Kết quả từ API: {result}")
                if result:
                    for i in range(0, len(result)):
                        bbox = result[i]['predictions']['boxes']
                        tracking_id = result[i]['predictions']['tracking_ids']
                        confidence = result[i]['predictions']['confidence']
                        # Vẽ hình chữ nhật lên khung hình
                        xmin, ymin = bbox[0], bbox[1]
                        thickness = 2  # Độ dày đường viền
                        frame = draw_box(frame, bbox, (colors[int(tracking_id) % len(colors)]), thickness)
                        label = "{}-{}".format(tracking_id, confidence)
                        cv2.putText(frame, str(label), (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
                # Hiển thị ảnh kết quả trên webcam
                cv2.imshow("faces", frame)
                
                # Thoát nếu nhấn phím 'q'
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                logger.error(f"Đã xảy ra lỗi: {response.status_code} - {response.text}")
        except requests.RequestException as e:
            logger.error(f"Request failed: {str(e)}")

    # Giải phóng camera và đóng cửa sổ hiển thị
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(post_server())