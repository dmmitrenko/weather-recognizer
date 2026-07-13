import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.applications.efficientnet import preprocess_input
import gradio as gr

MODEL_PATH = "models/weather_classifier.keras"
GATE_PATH = "models/gate_config.json"
IMG_SIZE = 224

model = tf.keras.models.load_model(MODEL_PATH)
with open(GATE_PATH) as f:
    gate = json.load(f)

class_names = gate["class_names"]
threshold = gate["threshold"]

last_dense = model.layers[-1]
logit_layer = layers.Dense(last_dense.units, activation=None, name="logits")
logits_out = logit_layer(model.layers[-2].output)
logit_layer.set_weights(last_dense.get_weights())
logit_model = Model(model.input, logits_out)


def predict(pil_image):
    if pil_image is None:
        return {}, ""
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img, dtype=np.float32)
    x = preprocess_input(x)[None]
    logits = logit_model.predict(x, verbose=0)[0]
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    msp = float(probs.max())

    if msp < threshold:
        msg = (
            f"Not a weather image - gate score {msp:.3f} < threshold {threshold:.3f}.\n"
            f"Try a photo where sky, precipitation, or frost is clearly visible."
        )
        return {}, msg

    top3 = {class_names[i]: float(probs[i]) for i in np.argsort(-probs)[:3]}
    verdict = (
        f"Weather detected - gate score {msp:.3f} >= threshold {threshold:.3f}.\n"
        f"Top-1: {class_names[int(np.argmax(probs))]}"
    )
    return top3, verdict


CSS_CLASSES = ", ".join(class_names)

with gr.Blocks(theme=gr.themes.Base(), css_paths=["custom.css"],
               title="Weather Image Classifier") as demo:
    with gr.Column(elem_id="wc-card"):
        with gr.Row(elem_id="wc-header"):
            gr.Markdown("# Weather Classifier")

        with gr.Row(equal_height=True):
            image = gr.Image(type="pil", label="Photo", height=500,
                             sources=["upload"], elem_id="wc-image")
            with gr.Column(elem_id="wc-analysis"):
                label = gr.Label(num_top_classes=3, label="Analysis")
                verdict = gr.Textbox(label="Gate verdict", lines=3, interactive=False)

        gr.Markdown(
            f"<small style='color:#8b96a3;font-family:monospace'>{CSS_CLASSES}</small>"
        )

    image.change(fn=predict, inputs=image, outputs=[label, verdict])

if __name__ == "__main__":
    demo.launch()
