import socket
import cv2
import base64
import json
from picamera2 import Picamera2

SERVER_HOST = "192.168.11.1"  # PC IP
SERVER_PORT = 5005

def main():
    # Initialize PiCamera2
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()
    print("PiCamera2 started.")

    # Setup TCP socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((SERVER_HOST, SERVER_PORT))
    server_socket.listen(1)
    print("Waiting for PC connection...")
    conn, addr = server_socket.accept()
    print(f"PC connected: {addr}")

    try:
        while True:
            # Capture frame
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Encode to JPEG
            _, buffer = cv2.imencode(".jpg", frame)
            img_b64 = base64.b64encode(buffer).decode("utf-8")

            # Wrap into JSON payload (no detections here)
            payload = {"image": img_b64}
            payload_bytes = json.dumps(payload).encode("utf-8")

            # Send size + payload
            size_str = str(len(payload_bytes)).ljust(16).encode()
            conn.sendall(size_str + payload_bytes)

            # Show live stream locally (optional)
            cv2.imshow("RPi Camera Stream", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        conn.close()
        server_socket.close()
        picam2.stop()
        cv2.destroyAllWindows()
        print("✅ RPi server closed.")

if __name__ == "__main__":
    main()
