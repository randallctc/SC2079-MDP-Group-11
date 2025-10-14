import socket
import cv2
import base64
import json
import struct
import time
from collections import deque
from picamera2 import Picamera2
import os

# RPi IP and port (server)
HOST = "192.168.11.1"
PORT = 5005

# Directory to save detected images
SAVE_DIR = r"C:\Users\randa\OneDrive\Documents\GitHub\SC2079-MDP-Group-11\Detected_Task2"
os.makedirs(SAVE_DIR, exist_ok=True)

def send_frame(conn, frame):
    """Encode frame as base64 and send with length header"""
    _, buffer = cv2.imencode(".jpg", frame)
    img_b64 = base64.b64encode(buffer).decode("utf-8")
    data = img_b64.encode('utf-8')
    msg = struct.pack('!I', len(data)) + data
    try:
        conn.sendall(msg)
    except (BrokenPipeError, ConnectionResetError):
        raise ConnectionResetError("Client disconnected during send")

def main():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()
    print("[INFO] PiCamera2 started")

    # Setup TCP server
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST, PORT))
    server_socket.listen(1)
    print(f"[INFO] Waiting for PC connection at {HOST}:{PORT} ...")

    conn, addr = server_socket.accept()
    print(f"[INFO] Connected to PC: {addr}")

    last_detected_class = None
    last_detection_time = 0
    cooldown_period = 10  # seconds
    stable_buffer = deque(maxlen=5)  # recent detections for stabilization
    required_stability = 2  # consecutive same detections to confirm

    try:
        conn.setblocking(False)
        while True:
            # Capture frame
            frame = picam2.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Send frame to PC
            try:
                send_frame(conn, frame)
            except ConnectionResetError:
                print("[WARN] PC disconnected. Waiting for reconnection...")
                conn.close()
                conn, addr = server_socket.accept()
                print(f"[INFO] Reconnected to PC: {addr}")
                conn.setblocking(False)
                continue

            # Try to receive detection result
            try:
                data = conn.recv(1024)
                if data:
                    message = json.loads(data.decode('utf-8'))
                    class_id = message.get("class_id")
                    current_time = time.time()

                    # Add to rolling buffer for stabilization
                    stable_buffer.append(class_id)

                    # Check if class_id is stable enough
                    if len(stable_buffer) >= required_stability and all(
                        c == class_id for c in list(stable_buffer)[-required_stability:]
                    ):
                        # Only act if cooldown passed or new class detected
                        if class_id != last_detected_class or (current_time - last_detection_time) >= cooldown_period:
                            if class_id == "None":
                                pass  # ignore
                            elif class_id == "right_arrow":
                                print("Turning right")
                                # Save frame for testing
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                cv2.imwrite(os.path.join(SAVE_DIR, f"detection_right_{timestamp}.jpg"), frame)
                            elif class_id == "left_arrow":
                                print("Turning left")
                                # Save frame for testing
                                timestamp = time.strftime("%Y%m%d_%H%M%S")
                                cv2.imwrite(os.path.join(SAVE_DIR, f"detection_left_{timestamp}.jpg"), frame)

                            last_detected_class = class_id
                            last_detection_time = current_time

            except BlockingIOError:
                # No message received yet
                pass
            except ConnectionResetError:
                print("[WARN] PC disconnected. Waiting for reconnection...")
                conn.close()
                conn, addr = server_socket.accept()
                print(f"[INFO] Reconnected to PC: {addr}")
                conn.setblocking(False)
                continue

            # Optional local preview
            cv2.imshow("RPi Camera", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    finally:
        picam2.stop()
        conn.close()
        server_socket.close()
        cv2.destroyAllWindows()
        print("[INFO] RPi server closed.")

if __name__ == "__main__":
    main()
