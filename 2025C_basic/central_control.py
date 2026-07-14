#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Central control: 读取串口指令，调度对应的检测脚本。
  A → basic_pi_pc_debug.py       (物体形状尺寸测量)
  B → detect_squares_pi_single_frame_v7.py  (多正方形检测)
  C → numbered_square_pi_v3.py   (指定编号正方形检测)
"""

import os
import sys
import time
import atexit
import subprocess
import serial

try:
    import RPi.GPIO as GPIO
    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False


SERIAL_PORT = '/dev/ttyAMA0'
BAUDRATE = 115200

PID_FILE = '/tmp/central_control.pid'

# 三个脚本与本文件放在同一目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COMMANDS = {
    'A': 'basic_pi_pc_debug.py',
    'B': 'detect_squares_pi_single_frame_v7.py',
    'C': 'numbered_square_pi_v3.py',
}

# GPIO 引脚（BCM 编号）
LED1_PIN = 18   # LED1: 启动闪烁两次后常亮，程序结束熄灭
LED2_PIN = 23   # LED2: 执行子脚本时点亮，执行完熄灭


def setup_gpio():
    """初始化 GPIO 引脚"""
    if not GPIO_AVAILABLE:
        return
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(LED1_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(LED2_PIN, GPIO.OUT, initial=GPIO.LOW)


def cleanup_gpio():
    """清理 GPIO，熄灭所有 LED"""
    if not GPIO_AVAILABLE:
        return
    GPIO.output(LED1_PIN, GPIO.LOW)
    GPIO.output(LED2_PIN, GPIO.LOW)
    GPIO.cleanup()


def led1_blink_twice():
    """LED1 闪烁两次后保持常亮"""
    if not GPIO_AVAILABLE:
        return
    for _ in range(2):
        GPIO.output(LED1_PIN, GPIO.HIGH)
        time.sleep(0.3)
        GPIO.output(LED1_PIN, GPIO.LOW)
        time.sleep(0.3)
    GPIO.output(LED1_PIN, GPIO.HIGH)  # 常亮


def acquire_lock():
    """确保只有一个 central_control 实例在运行"""
    if os.path.exists(PID_FILE):
        with open(PID_FILE, 'r') as f:
            old_pid = f.read().strip()
        if old_pid:
            try:
                os.kill(int(old_pid), 0)  # 检查进程是否存在
                print(f'ERROR: central_control already running (PID={old_pid})')
                print(f'  If sure it is not, remove {PID_FILE} and retry.')
                sys.exit(1)
            except (OSError, ValueError, ProcessLookupError):
                os.remove(PID_FILE)  # 进程已不存在，清理残留文件

    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    atexit.register(lambda: os.path.exists(PID_FILE) and os.remove(PID_FILE))


def open_serial():
    """打开串口并等待硬件稳定，避免首字节丢失"""
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.5)
    ser.reset_input_buffer()
    time.sleep(0.1)
    return ser


def read_line_from_serial(ser):
    """
    稳健读取一行：先读所有可用字节拼到缓冲区，再取完整行。
    避免因 timeout 过短导致一条消息被拆成多段。
    """
    buf = b''
    while True:
        waiting = ser.in_waiting
        if waiting > 0:
            chunk = ser.read(waiting)
            buf += chunk
        else:
            # 没有更多可用数据，尝试读一个字节等待新数据
            byte = ser.read(1)
            if byte:
                buf += byte
                continue
            else:
                # timeout 到期，无新数据
                break

    if not buf:
        return None

    # 取第一个完整行（以 \n 分隔）
    lines = buf.split(b'\n')
    # 把不完整的剩余部分放回缓冲区... 但我们没有地方放。
    # 简化处理：返回第一个非空行，丢弃不完整尾部。
    for line in lines:
        stripped = line.replace(b'\r', b'').strip()
        if stripped:
            return stripped.decode('utf-8', errors='ignore')

    return None


def run_script(script_name):
    """用当前 Python 解释器运行同目录下的脚本。"""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    python = sys.executable
    print(f'--- Running: {python} {script_path} ---')

    # LED2 点亮：开始执行子脚本
    if GPIO_AVAILABLE:
        GPIO.output(LED2_PIN, GPIO.HIGH)

    subprocess.run([python, script_path], cwd=SCRIPT_DIR)

    # LED2 熄灭：子脚本执行完毕
    if GPIO_AVAILABLE:
        GPIO.output(LED2_PIN, GPIO.LOW)

    print(f'--- Finished: {script_name} ---\n')


def main():
    acquire_lock()

    # 初始化 GPIO
    setup_gpio()
    atexit.register(cleanup_gpio)

    # LED1 闪烁两次后常亮
    led1_blink_twice()

    print('Central control started.')
    print('Commands: Key=A → basic  |  Key=B → detect_squares  |  Key=C → numbered_square')
    print('Waiting for serial command (format: Key=X)...')

    ser = open_serial()

    try:
        while True:
            line = read_line_from_serial(ser)
            if not line:
                continue

            print(f'Received: {line}')

            # 检测到 # 时退出程序
            if '#' in line:
                print('Exit command (#) received, shutting down...')
                break

            # 解析 Key=X 格式，取等号后面的第一个字符
            cmd = None
            if 'Key=' in line:
                val = line.split('Key=', 1)[1].strip()
                if val:
                    cmd = val[0].upper()

            if cmd not in COMMANDS:
                print(f'  Unknown command, waiting...')
                continue

            print(f'  Command=Key={cmd} → {COMMANDS[cmd]}')

            # 关闭串口让子脚本使用
            ser.close()
            time.sleep(0.2)

            run_script(COMMANDS[cmd])

            # 子脚本结束后重新打开串口
            time.sleep(0.2)
            ser = open_serial()
            print('Waiting for next command...')

    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        if ser.is_open:
            ser.close()
        cleanup_gpio()


if __name__ == '__main__':
    main()
