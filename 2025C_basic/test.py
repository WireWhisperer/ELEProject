import time
import serial
from gpiozero import Device

Device.pin_factory = 'mock'

ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.1)

# 初始化
final_d = 80.9
final_x = 12.79
final_type = "guanrueiqi"

# 存放接收到的 key 等号后面的值，如收到 "key=1" 则存 "1"
received_key = ""

try:
    for i in range(0, 10):
        # ----- 发送 -----
        msg = f"D={final_d:.1f},x={final_x:.2f},type={final_type}\r\n"
        ser.write(msg.encode('utf-8'))

        # ----- 接收 -----
        raw = ser.readline()
        line = raw.decode('utf-8', errors='ignore').strip()
        if line:
            print(f"收到: {line}")
            # 如果是 key=xxx 格式就存起来
            if "Key=" in line:
                val = line.split("Key=", 1)[1].strip()
                if val:
                    received_key = val[0]  # 只取第一个字符，如 "1"、"A"

        time.sleep(0.5)

except KeyboardInterrupt:
    print("Program stopped.")
finally:
    print("received_key="+received_key)
    ser.close()
