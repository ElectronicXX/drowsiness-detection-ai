import cv2
import os
import random

# =========================
# 📁 创建文件夹
# =========================
paths = [
    "dataset/train/open",
    "dataset/train/closed",
    "dataset/val/open",
    "dataset/val/closed"
]

for path in paths:
    os.makedirs(path, exist_ok=True)

# =========================
# 🎥 摄像头
# =========================
cap = cv2.VideoCapture(0)

count_open = 0
count_closed = 0

print("按 O = open | 按 C = closed | 按 Q = 退出")
print("自动分配：80% train / 20% val")

# =========================
# 🔄 循环采集
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Collect Data", frame)

    key = cv2.waitKey(1)

    # =========================
    # 👀 OPEN
    # =========================
    if key == ord('o'):
        if random.random() < 0.8:
            folder = "dataset/train/open"
        else:
            folder = "dataset/val/open"

        filename = f"{folder}/open_{count_open}.jpg"
        cv2.imwrite(filename, frame)

        count_open += 1
        print(f"Saved OPEN → {folder}")

    # =========================
    # 😴 CLOSED
    # =========================
    elif key == ord('c'):
        if random.random() < 0.8:
            folder = "dataset/train/closed"
        else:
            folder = "dataset/val/closed"

        filename = f"{folder}/closed_{count_closed}.jpg"
        cv2.imwrite(filename, frame)

        count_closed += 1
        print(f"Saved CLOSED → {folder}")

    # =========================
    # ❌ 退出
    # =========================
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("✅ 数据收集完成！")