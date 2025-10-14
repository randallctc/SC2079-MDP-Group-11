import socket
import cv2
import base64
import json
import time
from picamera2 import Picamera2


HOST = "192.168.11.1"
PORT = 5005

def send_json(sock, data):
    msg = json.dumps(data).encode("utf-8")
    header = f"{len(msg):<16}".encode("utf-8")
    sock.sendall(header + msg)


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
        return None
    return json.loads(data_bytes.decode("utf-8"))


def snap_and_send(picamera, sock):
    frame = picamera.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", frame)
    frame_b64 = base64.b64encode(buffer).decode("utf-8")

    # Send image as JSON with header
    send_json(sock, {"image": frame_b64})
    print("Image sent to PC for classification")

    # Wait for response
    response = receive_full_message(sock)
    if response:
        return response.get("class_id", None)
    return None


def main():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(main={"size": (640, 480)}))
    picam2.start()
    time.sleep(1)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.connect((HOST, PORT))
    print("Connected to PC server")

    try:
        while True:
            user_input = input("Press Enter to capture or q to quit: ")
            if user_input.lower() == 'q':
                break

            class_id = snap_and_send(picam2, sock)
            if class_id:
                if class_id == "None":
                    print("No arrow detected.")
                elif class_id == "right_arrow":
                    print("Turning RIGHT")
                elif class_id == "left_arrow":
                    print("Turning LEFT")
                else:
                    print(f"Unknown class: {class_id}")

    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        picam2.stop()
        sock.close()
        print("RPi client closed.")


if __name__ == "__main__":
    main()
