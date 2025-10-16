import socket
import base64
import cv2
import numpy as np
import json
import os
from datetime import datetime
from ultralytics import YOLO 

HOST = "192.168.11.15"
PORT = 5005

model = YOLO(r"C:\Users\randa\OneDrive\Documents\GitHub\SC2079-MDP-Group-11\best.pt")
SAVE_DIR = r"C:\Users\randa\OneDrive\Documents\GitHub\SC2079-MDP-Group-11\Image Recognition\Detected_Task2"
os.makedirs(SAVE_DIR, exist_ok=True)

def receive_full_message(sock):
    length_bytes = sock.recv(16)
    if not length_bytes:
        return None

    length = int(length_bytes.decode("utf-8").strip())
    data_bytes = b""

    while len(data_bytes) < length:
        chunk = sock.recv(length - len(data_bytes))
        if not chunk:
            break
        data_bytes += chunk

    if len(data_bytes) != length:
        print("Incomplete message received")
        return None

    return json.loads(data_bytes.decode("utf-8"))


def send_json(sock, data):
    msg = json.dumps(data).encode("utf-8")
    header = f"{len(msg):<16}".encode("utf-8")
    sock.sendall(header + msg)


def handle_client(conn, addr):
    print(f"Connected to RPi client: {addr}")
    try:
        while True:
            message = receive_full_message(conn)
            if not message:
                break

            # Decode base64 image
            frame_b64 = message.get("image")
            if not frame_b64:
                continue
            img_data = base64.b64decode(frame_b64)
            np_arr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            img = cv2.flip(img, -1)

            # Run YOLO detection
            results = model(img, verbose=False)
            detected_class = "None"

            for box in results[0].boxes:
                cls_idx = int(box.cls[0])
                print(cls_idx)  
                if cls_idx == 27:
                    detected_class = "right_arrow"
                    print("Should be right")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, "Right Arrow", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    break
                elif cls_idx == 28:
                    detected_class = "left_arrow"
                    print("Should be left")
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, "Left Arrow", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    break

            # Save image for inspection
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(SAVE_DIR, f"detection_{detected_class}_{timestamp}.jpg")
            cv2.imwrite(save_path, img)
            print(f"Saved detection image: {save_path}")

            # Send back class_id result
            send_json(conn, {"class_id": detected_class})
            print(f"Sent result: {detected_class}")

    except (ConnectionResetError, EOFError):
        print("RPi disconnected.")
    finally:
        conn.close()
        print("Connection closed.")

def main():
    print("Starting PC Server...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"Listening on {HOST}:{PORT}")

    while True:
        conn, addr = server_socket.accept()
        handle_client(conn, addr)


if __name__ == "__main__":
    main()