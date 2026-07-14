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
import subprocess
import serial
import RPi.GPIO as GPIO


SERIAL_PORT = '/dev/ttyAMA0'
BAUDRATE = 115200

# GPIO pins (BCM numbering, 高电平点亮)
LED1_PIN = 18   # BCM 18 → LED1: 程序运行状态灯
LED2_PIN = 23   # BCM 23 → LED2: 子脚本执行状态灯

# 三个脚本与本文件放在同一目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COMMANDS = {
    'A': 'basic_pi_pc_debug.py',
    'B': 'detect_squares_pi_single_frame_v7.py',
    'C': 'numbered_square_pi_v3.py',
}


def open_serial():
    """打开串口并等待硬件稳定，避免首字节丢失"""
    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
    ser.reset_input_buffer()    # 清空缓冲区残留数据
    time.sleep(0.1)             # 等待 UART 硬件稳定
    return ser


def setup_gpio():
    """初始化 GPIO 引脚"""
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(LED1_PIN, GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(LED2_PIN, GPIO.OUT, initial=GPIO.LOW)


def led1_blink_twice():
    """LED1 闪烁两次（亮 0.3s / 灭 0.3s）"""
    for _ in range(2):
        GPIO.output(LED1_PIN, GPIO.HIGH)
        time.sleep(0.3)
        GPIO.output(LED1_PIN, GPIO.LOW)
        time.sleep(0.3)


def cleanup_gpio():
    """清理 GPIO 资源"""
    GPIO.cleanup()


def run_script(script_name):
    """用当前 Python 解释器运行同目录下的脚本。"""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    python = sys.executable
    print(f'--- Running: {python} {script_path} ---')

    # LED2 点亮，表示正在执行子脚本
    GPIO.output(LED2_PIN, GPIO.HIGH)

    subprocess.run([python, script_path], cwd=SCRIPT_DIR)

    # LED2 熄灭，表示执行完毕
    GPIO.output(LED2_PIN, GPIO.LOW)

    print(f'--- Finished: {script_name} ---\n')


def main():
    # 初始化 GPIO
    setup_gpio()

    # LED1 闪烁两次，表示程序启动
    led1_blink_twice()

    # LED1 常亮，表示程序正在运行中
    GPIO.output(LED1_PIN, GPIO.HIGH)

    print('Central control started.')
    print('Commands: Key=A → basic  |  Key=B → detect_squares  |  Key=C → numbered_square')
    print('Waiting for serial command (format: Key=X)...')

    ser = open_serial()

    try:
        while True:
            raw = ser.readline()
            line = raw.decode('utf-8', errors='ignore').strip()
            if not line:
                continue

            print(f'Received: {line}')

            # 解析 Key=X 格式，取等号后面的第一个字符
            cmd = None
            if 'Key=' in line:
                val = line.split('Key=', 1)[1].strip()
                if val:
                    cmd = val[0].upper()  # 取第一个字符，统一转大写

            if cmd not in COMMANDS:
                print(f'  Unknown command, waiting...')
                continue

            print(f'  Command=Key={cmd} → {COMMANDS[cmd]}')

            # 关闭串口让子脚本使用
            ser.close()
            time.sleep(0.2)

            run_script(COMMANDS[cmd])

            # 子脚本结束后重新打开串口（使用稳定初始化）
            time.sleep(0.2)
            ser = open_serial()
            print('Waiting for next command...')

    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        if ser.is_open:
            ser.close()
        # 熄灭 LED1 并清理 GPIO
        GPIO.output(LED1_PIN, GPIO.LOW)
        cleanup_gpio()


if __name__ == '__main__':
    main()
