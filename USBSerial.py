#!/usr/bin/env python3
"""USB serial HEX packet TX/RX example for ESP32-C3.

Usage examples:
  python USBSerial.py
  python USBSerial.py --port /dev/ttyACM0 --baud 115200 --tx "AA 55 01 02 03 04"
  python USBSerial.py --interval 0.2 --newline
"""

import argparse
import signal
import sys
import time

import serial


def parse_hex_string(hex_text: str) -> bytes:
    """Convert a hex string like 'AA 55 01 02' or 'aa550102' into bytes."""
    cleaned = hex_text.replace(" ", "").replace("0x", "")
    if len(cleaned) % 2 != 0:
        raise ValueError("HEX string length must be even.")
    return bytes.fromhex(cleaned)


def to_hex(data: bytes) -> str:
    return " ".join(f"{b:02X}" for b in data)


def to_ascii_safe(data: bytes) -> str:
    return "".join(chr(b) if 32 <= b <= 126 else "." for b in data)


def main() -> int:
    parser = argparse.ArgumentParser(description="ESP32-C3 USB serial HEX packet sender/receiver")
    parser.add_argument("--port", default="/dev/ttyACM0", help="Serial device path")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate")
    parser.add_argument("--timeout", type=float, default=0.2, help="Read timeout seconds")
    parser.add_argument(
        "--tx",
        default="48 45 4C 4C 4F 20 45 53 50 33 32 0A",
        help="TX HEX packet (default: 'HELLO ESP32\\n')",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="TX interval seconds")
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=0.02,
        help="RX polling interval seconds when no data is available",
    )
    parser.add_argument(
        "--reconnect-delay",
        type=float,
        default=1.0,
        help="Seconds to wait before retrying serial reconnect",
    )
    parser.add_argument(
        "--sigint-exit-window",
        type=float,
        default=2.0,
        help="Require 2 Ctrl+C presses within this many seconds to exit",
    )
    parser.add_argument(
        "--newline",
        action="store_true",
        help="Append '\\n' to TX packet if not already present",
    )
    args = parser.parse_args()

    try:
        tx_packet = parse_hex_string(args.tx)
    except ValueError as exc:
        print(f"[ERROR] Invalid --tx HEX string: {exc}")
        return 1

    if args.newline and not tx_packet.endswith(b"\n"):
        tx_packet += b"\n"

    print("[INFO] Start RX-only loop. Press Ctrl+C to stop.")
    print(f"[INFO] TX template configured but disabled: {to_hex(tx_packet)}")

    ser = None
    rx_buffer = bytearray()
    last_sigint_time = 0.0
    stop_requested = False

    def handle_sigint(_signum, _frame):
        nonlocal last_sigint_time, stop_requested
        now = time.monotonic()
        if now - last_sigint_time <= args.sigint_exit_window:
            stop_requested = True
            return
        last_sigint_time = now
        print(
            f"\n[WARN] Ctrl+C detected. Press Ctrl+C again within {args.sigint_exit_window:.1f}s to stop."
        )

    signal.signal(signal.SIGINT, handle_sigint)

    try:
        while True:
            if stop_requested:
                print("\n[INFO] Stopped by user.")
                break

            if ser is None or not ser.is_open:
                try:
                    print(f"[INFO] Open serial: {args.port} @ {args.baud}")
                    ser = serial.Serial(args.port, args.baud, timeout=args.timeout)
                    ser.reset_input_buffer()
                    ser.reset_output_buffer()
                    rx_buffer.clear()
                    print("[INFO] Serial connected.")
                except (serial.SerialException, OSError) as exc:
                    print(f"[ERROR] Cannot open serial port: {exc}")
                    time.sleep(args.reconnect_delay)
                    continue

            try:
                # RX-only mode: keep TX disabled to reduce traffic/noise.
                # ser.write(tx_packet)
                # ser.flush()
                # print(f"[TX] {to_hex(tx_packet)}")

                waiting = ser.in_waiting
                if waiting <= 0:
                    time.sleep(args.poll_interval)
                    continue

                rx_data = ser.read(waiting)
                if not rx_data:
                    time.sleep(args.poll_interval)
                    continue

                rx_buffer.extend(rx_data)

                # Extract fixed 12-byte frames: b"TU" + 8-byte payload + b"$$".
                while True:
                    start = rx_buffer.find(b"TU")
                    if start < 0:
                        if len(rx_buffer) > 4096:
                            del rx_buffer[:-64]
                        break

                    if start > 0:
                        del rx_buffer[:start]

                    if len(rx_buffer) < 12:
                        break

                    frame = bytes(rx_buffer[:12])
                    if frame[-2:] == b"$$":
                        print(f"[RX HEX]   {to_hex(frame)}")
                        print(f"[RX ASCII] {to_ascii_safe(frame)}")
                        del rx_buffer[:12]
                    else:
                        del rx_buffer[:1]
            except (serial.SerialException, OSError) as exc:
                print(f"[ERROR] Serial read failed: {exc}")
                try:
                    ser.close()
                except Exception:
                    pass
                ser = None
                print(f"[INFO] Reconnecting in {args.reconnect_delay:.1f}s...")
                time.sleep(args.reconnect_delay)
    finally:
        signal.signal(signal.SIGINT, signal.default_int_handler)
        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass
        print("[INFO] Serial closed.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
