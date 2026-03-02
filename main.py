import gradio as gr
from infer import infer

# A simple wrapper to save the uploaded image and pass its path to the infer function


def recognize(img):
    img.save("temp.png")
    return infer("temp.png")


if __name__ == "__main__":
    gr.Interface(
        fn=recognize,
        inputs=gr.Image(type="pil"),
        outputs=gr.Textbox(label="LaTeX"),
        title="Handwritten Formula Recognition (CROHME)"
    ).launch()
