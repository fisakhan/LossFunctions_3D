import math
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import cv2
import matplotlib

# Set non-interactive backend for headless frame rendering
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2

# ====================================================================
# 1. Dataset Preprocessing 
(train_x, train_y), (test_x, test_y) = tf.keras.datasets.mnist.load_data()
train_x = train_x.astype(np.float32) / 255.0
test_x = test_x.astype(np.float32) / 255.0

train_y = tf.keras.utils.to_categorical(train_y, num_classes=10)
test_y = tf.keras.utils.to_categorical(test_y, num_classes=10)

label = tf.argmax(train_y, axis=1).numpy()
colors = cm.rainbow(np.linspace(0, 1, 10))

# 2. Cosine Similarity Layer
# =====================================================================
class CosSimLayer(tf.keras.layers.Layer):

    def __init__(
        self, num_classes, regularizer=None, name="NormLayer", **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self._n_classes = num_classes
        self._regularizer = regularizer

    def build(self, embedding_shape):
        self._w = self.add_weight(
            shape=(embedding_shape[-1], self._n_classes),
            initializer="glorot_uniform",
            trainable=True,
            regularizer=self._regularizer,
            name="cosine_weights",
        )

    def call(self, embedding, training=None):
        x = tf.nn.l2_normalize(embedding, axis=1, name="normalize_prelogits")
        w = tf.nn.l2_normalize(self._w, axis=0, name="normalize_weights")
        return tf.matmul(x, w, name="cosine_similarity"), x

# =====================================================================
# 3. Loss Functions
class CurricularFaceLoss(tf.keras.losses.Loss):

    def __init__(
        self,
        scale=30.0,
        margin=0.5,
        alpha=0.99,
        name="CurricularFaceLoss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.scale = scale
        self.margin = margin
        self.alpha = alpha
        self.t = tf.Variable(0.0, trainable=False, dtype=tf.float32)
        self.eps = 1e-7

    def positive_forward(self, y_logit_masked):
        theta_margin = tf.math.acos(y_logit_masked) + self.margin
        return tf.math.cos(theta_margin)

    def negative_forward(self, y_logit_pos_masked, y_logit):
        hard_sample_mask = y_logit_pos_masked < y_logit
        return tf.where(
            hard_sample_mask, tf.square(y_logit) + self.t * y_logit, y_logit
        )

    def forward(self, y_true, y_logit):
        y_true = tf.cast(y_true, tf.float32)
        y_logit = tf.cast(y_logit, tf.float32)
        y_logit = tf.clip_by_value(y_logit, -1.0 + self.eps, 1.0 - self.eps)

        y_logit_masked = tf.expand_dims(
            tf.reduce_sum(y_true * y_logit, axis=1), axis=1
        )
        y_logit_pos_masked = self.positive_forward(y_logit_masked)
        y_logit_neg = self.negative_forward(y_logit_pos_masked, y_logit)

        r = tf.reduce_mean(y_logit_pos_masked)
        self.t.assign(self.alpha * r + (1.0 - self.alpha) * self.t)

        y_true_bool = tf.cast(y_true, dtype=tf.bool)
        return tf.where(y_true_bool, y_logit_pos_masked, y_logit_neg)

    def __call__(self, y_true, y_logit):
        y_logit_fixed = self.forward(y_true, y_logit)
        loss = tf.nn.softmax_cross_entropy_with_logits(
            y_true, y_logit_fixed * self.scale
        )
        return tf.reduce_mean(loss)


class UnifiedCrossEntropyLoss(tf.keras.losses.Loss):

    def __init__(
        self,
        scale=30.0,
        m_pos=0.4,
        m_neg=0.1,
        name="UnifiedCrossEntropyLoss",
        **kwargs,
    ):
        super().__init__(name=name, **kwargs)
        self.scale = scale
        self.m_pos = m_pos
        self.m_neg = m_neg
        self.t = tf.Variable(0.0, trainable=True, dtype=tf.float32)
        self.eps = 1e-7

    def __call__(self, y_true, y_logit):
        y_true = tf.cast(y_true, tf.float32)
        y_logit = tf.cast(y_logit, tf.float32)
        y_logit = tf.clip_by_value(y_logit, -1.0 + self.eps, 1.0 - self.eps)

        pos_logit = y_logit - (self.m_pos + self.t)
        neg_logit = y_logit + (self.m_neg - self.t)

        y_true_bool = tf.cast(y_true, dtype=tf.bool)
        adjusted_logits = tf.where(y_true_bool, pos_logit, neg_logit)

        loss = tf.nn.softmax_cross_entropy_with_logits(
            y_true, adjusted_logits * self.scale
        )
        return tf.reduce_mean(loss)


def standard_softmax_loss(y_true, y_logit, scale=30.0):
    loss = tf.nn.softmax_cross_entropy_with_logits(y_true, y_logit * scale)
    return tf.reduce_mean(loss)


# =====================================================================
# 4. Model Builder & Instances
def build_model(embedding_dim=3):
    model_input = layers.Input(shape=(28, 28))
    x = layers.Reshape((28, 28, 1))(model_input)
    x = layers.ZeroPadding2D(padding=2)(x)
    x = layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-5),
    )(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-5),
    )(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu",
        kernel_regularizer=l2(1e-5),
    )(x)
    x = layers.MaxPool2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(embedding_dim)(x)
    cosine_sim, embedding = CosSimLayer(10)(x)
    return tf.keras.models.Model(model_input, [cosine_sim, embedding])


model_softmax = build_model()
model_curricular = build_model()
model_uniface = build_model()

opt_softmax = tf.optimizers.Adam()
opt_curricular = tf.optimizers.Adam()
opt_uniface = tf.optimizers.Adam()

curricular_loss_fn = CurricularFaceLoss()
uniface_loss_fn = UnifiedCrossEntropyLoss()

# =====================================================================
# 5. Training Step Definitions
@tf.function
def train_step_softmax(batch_x, batch_y):
    with tf.GradientTape() as tape:
        cosine_sim, _ = model_softmax(batch_x, training=True)
        loss = standard_softmax_loss(batch_y, cosine_sim)
    grads = tape.gradient(loss, model_softmax.trainable_variables)
    opt_softmax.apply_gradients(zip(grads, model_softmax.trainable_variables))
    return loss


@tf.function
def train_step_curricular(batch_x, batch_y):
    with tf.GradientTape() as tape:
        cosine_sim, _ = model_curricular(batch_x, training=True)
        loss = curricular_loss_fn(batch_y, cosine_sim)
    grads = tape.gradient(loss, model_curricular.trainable_variables)
    opt_curricular.apply_gradients(
        zip(grads, model_curricular.trainable_variables)
    )
    return loss


@tf.function
def train_step_uniface(batch_x, batch_y):
    with tf.GradientTape() as tape:
        cosine_sim, _ = model_uniface(batch_x, training=True)
        loss = uniface_loss_fn(batch_y, cosine_sim)
    trainable_vars = model_uniface.trainable_variables + [uniface_loss_fn.t]
    grads = tape.gradient(loss, trainable_vars)
    opt_uniface.apply_gradients(zip(grads, trainable_vars))
    return loss

# =====================================================================
# 6. Frame Rendering Helper
def render_epoch_frame(embeddings, epoch, title_prefix, camera_angle=30):
    fig = plt.figure(figsize=(8, 8), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    for i in range(10):
        mask = label == i
        class_pts = embeddings[mask]

        ax.scatter(
            class_pts[:, 0],
            class_pts[:, 1],
            class_pts[:, 2],
            color=colors[i],
            alpha=0.35,
            s=10,
        )

        mean_vec = np.mean(class_pts, axis=0)
        norm = np.linalg.norm(mean_vec)
        label_pos = (mean_vec / norm) * 1.12 if norm > 0 else mean_vec

        ax.text(
            label_pos[0],
            label_pos[1],
            label_pos[2],
            str(i),
            color="black",
            fontsize=13,
            weight="bold",
            horizontalalignment="center",
            verticalalignment="center",
            bbox=dict(
                boxstyle="circle,pad=0.2",
                facecolor=colors[i],
                edgecolor="black",
                alpha=0.85,
            ),
        )

    # Wireframe unit sphere
    u = np.linspace(0, 2 * np.pi, 25)
    v = np.linspace(0, np.pi, 25)
    ax.plot_wireframe(
        np.outer(np.cos(u), np.sin(v)),
        np.outer(np.sin(u), np.sin(v)),
        np.outer(np.ones(np.size(u)), np.cos(v)),
        color="gray",
        alpha=0.08,
        linewidth=0.5,
    )

    ax.set_title(f"{title_prefix} - Epoch {epoch:02d}", fontsize=13, pad=15)
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    ax.view_init(elev=20, azim=camera_angle)

    fig.canvas.draw()
    rgba_buffer = fig.canvas.buffer_rgba()
    frame = np.asarray(rgba_buffer, dtype=np.uint8)
    plt.close(fig)

    return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

# =====================================================================
# 7. Training & Frame Collection Loop
n_epochs = 50
batch_size = 1024
n_batches = math.ceil(len(train_x) / batch_size)

frames_softmax = []
frames_curricular = []
frames_uniface = []

print("Training models and recording epoch frames...")

for epoch in range(1, n_epochs + 1):
    for j in range(n_batches):
        batch_x = train_x[batch_size * j : batch_size * (j + 1)]
        batch_y = train_y[batch_size * j : batch_size * (j + 1)]

        train_step_softmax(batch_x, batch_y)
        train_step_curricular(batch_x, batch_y)
        train_step_uniface(batch_x, batch_y)

    # Extract 3D embeddings at end of current epoch
    _, emb_s = model_softmax(train_x, training=False)
    _, emb_c = model_curricular(train_x, training=False)
    _, emb_u = model_uniface(train_x, training=False)

    # Slowly rotate camera view angle across epochs
    cam_angle = (epoch * 4) % 360

    # Render frame for each loss function
    frames_softmax.append(
        render_epoch_frame(
            emb_s.numpy(), epoch, "Standard Softmax", camera_angle=cam_angle
        )
    )
    frames_curricular.append(
        render_epoch_frame(
            emb_c.numpy(), epoch, "CurricularFace", camera_angle=cam_angle
        )
    )
    frames_uniface.append(
        render_epoch_frame(
            emb_u.numpy(), epoch, "UniFace (UCE)", camera_angle=cam_angle
        )
    )

    print(f"Recorded frame for Epoch [{epoch:02d}/{n_epochs}]")

# =====================================================================
# 8. Export Frames to MP4 Video Files
def save_video(frames, filename, fps=5):
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)

    # Hold the last epoch frame for 3 extra seconds
    for _ in range(fps * 3):
        out.write(frames[-1])

    out.release()
    print(f"Video saved successfully: {filename}")


save_video(frames_softmax, "softmax_evolution.mp4", fps=5)
save_video(frames_curricular, "curricularface_evolution.mp4", fps=5)
save_video(frames_uniface, "uniface_evolution.mp4", fps=5)
