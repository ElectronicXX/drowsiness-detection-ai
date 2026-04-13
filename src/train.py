import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt
import pandas as pd

# =========================
# 📁 数据路径
# =========================
train_dir = "dataset/train"
val_dir = "dataset/val"

if not os.path.exists(train_dir):
    print("❌ train dataset 不存在")
    exit()

# =========================
# 📦 加载数据
# =========================
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(48, 48),
    batch_size=32
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(48, 48),
    batch_size=32
)

# =========================
# 🔄 数据增强（稍微调稳🔥）
# =========================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# =========================
# 🧠 CNN模型（稳定版🔥）
# =========================
model = models.Sequential([
    layers.Input(shape=(48, 48, 3)),

    layers.Rescaling(1./255),

    data_augmentation,

    layers.Conv2D(16, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),   # 🔥提升表达能力
    layers.Dropout(0.5),

    layers.Dense(2, activation='softmax')
])

# =========================
# ⚙️ 编译模型
# =========================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# =========================
# 🛑 EarlyStopping（更稳🔥）
# =========================
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=3,
    restore_best_weights=True
)

# =========================
# 💾 保存最佳模型（关键🔥🔥🔥）
# =========================
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    "model/best_model.keras",
    monitor="val_accuracy",
    save_best_only=True,
    mode="max"
)

# =========================
# 🚀 训练模型
# =========================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[early_stop, checkpoint]
)

# =========================
# 💾 保存最终模型
# =========================
os.makedirs("model", exist_ok=True)
model.save("model/last_model.keras")

print("✅ 模型训练完成！")
print("🏆 最佳模型已保存：model/best_model.keras")

# =========================
# 📊 可视化
# =========================
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(10,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Train Accuracy')
plt.plot(epochs_range, val_acc, label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

# Loss
plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Train Loss')
plt.plot(epochs_range, val_loss, label='Val Loss')
plt.legend()
plt.title('Loss')

plt.show()

# =========================
# 📁 保存训练记录
# =========================
df = pd.DataFrame(history.history)
df.to_csv("training_log.csv", index=False)

print("📊 训练数据已保存到 training_log.csv")