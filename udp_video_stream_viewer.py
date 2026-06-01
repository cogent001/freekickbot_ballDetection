import ctypes
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import xml.sax.saxutils

import cv2
import numpy as np


UDP_IP = "0.0.0.0"
VIDEO_PORT = 5005
MAX_DGRAM = 2**16

DEVICE_SSID = "IKC_A0003"
DEVICE_PASSWORD = "12345678"
DEVICE_AP_IP = "192.168.4.1"
CONNECT_TIMEOUT_SEC = 25.0
WIFI_PROFILE_NAME = f"videoStream_{DEVICE_SSID}"

WINDOW_NAME = "videoStream UDP Viewer"
EXPECTED_FRAME_SIZE = (640, 480)


def is_windows() -> bool:
    return os.name == "nt"


def is_linux() -> bool:
    return sys.platform.startswith("linux")


def command_exists(command: str) -> bool:
    return shutil.which(command) is not None


def run_netsh(args, timeout=10):
    cmd = ["netsh", *args]
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=timeout,
    )


def run_cmd(args, timeout=10):
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
        timeout=timeout,
    )


def is_admin() -> bool:
    if not is_windows():
        return False
    try:
        return bool(ctypes.windll.shell32.IsUserAnAdmin())
    except Exception:
        return False


def make_wifi_profile_xml(profile_name: str, ssid: str, password: str) -> str:
    profile_name_escaped = xml.sax.saxutils.escape(profile_name)
    ssid_escaped = xml.sax.saxutils.escape(ssid)
    password_escaped = xml.sax.saxutils.escape(password)

    if password:
        auth_block = f"""
            <authentication>WPA2PSK</authentication>
            <encryption>AES</encryption>
            <useOneX>false</useOneX>
        """
        key_block = f"""
        <sharedKey>
            <keyType>passPhrase</keyType>
            <protected>false</protected>
            <keyMaterial>{password_escaped}</keyMaterial>
        </sharedKey>
        """
    else:
        auth_block = """
            <authentication>open</authentication>
            <encryption>none</encryption>
            <useOneX>false</useOneX>
        """
        key_block = ""

    return f"""<?xml version="1.0"?>
<WLANProfile xmlns="http://www.microsoft.com/networking/WLAN/profile/v1">
    <name>{profile_name_escaped}</name>
    <SSIDConfig>
        <SSID>
            <name>{ssid_escaped}</name>
        </SSID>
    </SSIDConfig>
    <connectionType>ESS</connectionType>
    <connectionMode>manual</connectionMode>
    <MSM>
        <security>
            <authEncryption>{auth_block}
            </authEncryption>
            {key_block}
        </security>
    </MSM>
</WLANProfile>
"""


def add_wifi_profile(profile_name: str, ssid: str, password: str) -> None:
    xml_text = make_wifi_profile_xml(profile_name, ssid, password)
    profile_path = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False, encoding="utf-8") as profile:
            profile.write(xml_text)
            profile_path = profile.name

        result = run_netsh(["wlan", "delete", "profile", f"name={profile_name}"], timeout=5)
        if result.returncode != 0:
            pass

        result = run_netsh(["wlan", "add", "profile", f"filename={profile_path}", "user=current"])
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
    finally:
        if profile_path:
            try:
                os.remove(profile_path)
            except OSError:
                pass


def connect_wifi(ssid: str, password: str) -> None:
    if is_windows():
        print(f"[WiFi] Preparing profile '{WIFI_PROFILE_NAME}' for SSID '{ssid}'")
        add_wifi_profile(WIFI_PROFILE_NAME, ssid, password)

        print(f"[WiFi] Connecting to '{ssid}'")
        result = run_netsh(["wlan", "connect", f"name={WIFI_PROFILE_NAME}", f"ssid={ssid}"])
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())

    elif is_linux():
        if not command_exists("nmcli"):
            print("[WiFi] 'nmcli' not found. Connect to the device AP manually.")
        else:
            print(f"[WiFi] Connecting to '{ssid}' via nmcli")
            connect_cmd = ["nmcli", "dev", "wifi", "connect", ssid]
            if password:
                connect_cmd.extend(["password", password])
            result = run_cmd(connect_cmd, timeout=15)

            if result.returncode != 0:
                fallback = run_cmd(["nmcli", "con", "up", "id", ssid], timeout=10)
                if fallback.returncode != 0:
                    detail = (
                        result.stderr.strip()
                        or result.stdout.strip()
                        or fallback.stderr.strip()
                        or fallback.stdout.strip()
                    )
                    raise RuntimeError(detail or "Failed to connect Wi-Fi with nmcli")
    else:
        print("[WiFi] Unsupported OS for auto connect. Connect to the device AP manually.")

    deadline = time.monotonic() + CONNECT_TIMEOUT_SEC
    while time.monotonic() < deadline:
        ip = get_softap_ip()
        if ip is not None:
            print(f"[WiFi] Connected. Host IP: {ip}")
            return
        time.sleep(0.5)

    raise TimeoutError(f"Timed out waiting for IP address from '{ssid}'")


def disconnect_wifi(ssid: str) -> None:
    if is_windows():
        status = run_netsh(["wlan", "show", "interfaces"], timeout=5)
        if ssid not in status.stdout:
            run_netsh(["wlan", "delete", "profile", f"name={WIFI_PROFILE_NAME}"], timeout=5)
            return

        print(f"[WiFi] Disconnecting from '{ssid}'")
        run_netsh(["wlan", "disconnect"], timeout=5)
        run_netsh(["wlan", "delete", "profile", f"name={WIFI_PROFILE_NAME}"], timeout=5)
        return

    if is_linux():
        if not command_exists("nmcli"):
            print("[WiFi] Skipping auto disconnect (nmcli not available).")
            return

        status = run_cmd(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"], timeout=5)
        active_lines = [line for line in status.stdout.splitlines() if line.startswith("yes:")]
        if not any(line.split(":", 1)[1] == ssid for line in active_lines):
            return

        print(f"[WiFi] Disconnecting from '{ssid}'")
        run_cmd(["nmcli", "con", "down", "id", ssid], timeout=10)
        return


def get_softap_ip(ap_ip=DEVICE_AP_IP):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((ap_ip, 1))
        ip = s.getsockname()[0]
        if ip.startswith("192.168.4."):
            return ip
        return None
    except OSError:
        return None
    finally:
        s.close()


def decode_jpeg(packet: bytes):
    if len(packet) < 100:
        return None
    if packet[0] != 0xFF or packet[1] != 0xD8:
        return None

    data = np.frombuffer(packet, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def video_receiver():
    video_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        video_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1024 * 1024)
    except OSError:
        pass
    video_sock.bind((UDP_IP, VIDEO_PORT))
    video_sock.settimeout(1.0)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, *EXPECTED_FRAME_SIZE)
    print(f"[UDP] Listening on {UDP_IP}:{VIDEO_PORT}")
    print("[KEY] Press ESC to quit and disconnect Wi-Fi")

    frame_count = 0
    last_stat_time = time.monotonic()
    first_frame_seen = False

    try:
        while True:
            try:
                packet, addr = video_sock.recvfrom(MAX_DGRAM)
            except socket.timeout:
                key = cv2.waitKey(1) & 0xFF
                if key == 27:
                    break
                continue

            frame = decode_jpeg(packet)
            if frame is None:
                continue

            if not first_frame_seen:
                height, width = frame.shape[:2]
                print(f"[Video] First decoded frame: {width}x{height}, jpeg={len(packet)} bytes")
                if (width, height) != EXPECTED_FRAME_SIZE:
                    print(f"[Video] WARNING: expected {EXPECTED_FRAME_SIZE[0]}x{EXPECTED_FRAME_SIZE[1]}, got {width}x{height}")
                cv2.resizeWindow(WINDOW_NAME, width, height)
                first_frame_seen = True

            frame_count += 1
            now = time.monotonic()
            if now - last_stat_time >= 5.0:
                fps = frame_count / (now - last_stat_time)
                print(f"[UDP] {fps:.1f} fps from {addr[0]}:{addr[1]} last={len(packet)} bytes")
                frame_count = 0
                last_stat_time = now

            cv2.imshow(WINDOW_NAME, frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
    finally:
        video_sock.close()
        cv2.destroyAllWindows()


def main():
    if is_windows() and not is_admin():
        print("[WiFi] netsh Wi-Fi profile changes may require an elevated terminal.")
    if is_linux() and not command_exists("nmcli"):
        print("[WiFi] nmcli is not installed. Auto connect/disconnect is disabled.")

    try:
        connect_wifi(DEVICE_SSID, DEVICE_PASSWORD)
        video_receiver()
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1
    finally:
        disconnect_wifi(DEVICE_SSID)

    return 0


if __name__ == "__main__":
    sys.exit(main())
