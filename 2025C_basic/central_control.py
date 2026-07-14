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


SERIAL_PORT = '/dev/ttyAMA0'
BAUDRATE = 115200

# 三个脚本与本文件放在同一目录
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

COMMANDS = {
    'A': 'basic_pi_pc_debug.py',
    'B': 'detect_squares_pi_single_frame_v7.py',
    'C': 'numbered_square_pi_v3.py',
}


def run_script(script_name):
    """用当前 Python 解释器运行同目录下的脚本。"""
    script_path = os.path.join(SCRIPT_DIR, script_name)
    python = sys.executable
    print(f'--- Running: {python} {script_path} ---')
    subprocess.run([python, script_path], cwd=SCRIPT_DIR)
    print(f'--- Finished: {script_name} ---\n')


def main():
    print('Central control started.')
    print('Commands: Key=A → basic  |  Key=B → detect_squares  |  Key=C → numbered_square')
    print('Waiting for serial command (format: Key=X)...')

    ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)

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

            # 子脚本结束后重新打开串口
            time.sleep(0.2)
            ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=0.1)
            print('Waiting for next command...')

    except KeyboardInterrupt:
        print('\nStopped.')
    finally:
        if ser.is_open:
            ser.close()


if __name__ == '__main__':
    main()
