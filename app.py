import gradio as gr
import numpy as np
from PIL import Image
from engine.search import ImageSearchModule
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

def check_dirs():
    dirs = {
        "Data": (PROJECT_ROOT / "data"),
        "Images": (PROJECT_ROOT / "data" / "images"),
        "Features": (PROJECT_ROOT / "data" / "features")
    }
    for dir_name, dir_path in dirs.items():
        if not dir_path.exists():
            raise FileNotFoundError(f"{dir_name} directory not found: {dir_path}")

    print("All data directories exist ✅")


check_dirs()

# Initialize the ImageSearchModule
search = ImageSearchModule(
    image_embeddings_dir=str(PROJECT_ROOT / "data/features"),
    original_images_dir=str(PROJECT_ROOT / "data/images"),
)
print("Add image embeddings and caption embeddings to vector database")
search.add_images()


def search_images(input_data, search_type):
    if search_type == "image" and input_data is not None:
        # Fix: Get the file path directly from the input data
        results = search.search_by_image(input_data, top_k=10, similarity_threshold=0)
    elif search_type == "text" and input_data.strip():
        results = search.search_by_text(input_data, top_k=10, similarity_threshold=0)
    else:
        return [(Image.new("RGB", (100, 100), color="gray"), "No results")] * 10

    images_with_captions = []
    for image_name, similarity in results:
        image_path = os.path.join(search.original_images_dir, f"resized_{image_name}")
        matching_files = [
            f
            for f in os.listdir(search.original_images_dir)
            if f.startswith(f"resized_{image_name}")
        ]
        if matching_files:
            img = Image.open(
                os.path.join(search.original_images_dir, matching_files[0])
            )
            images_with_captions.append((img, f"Similarity: {similarity:.2f}"))
        else:
            images_with_captions.append(
                (Image.new("RGB", (100, 100), color="gray"), "Image not found")
            )

    # Pad the results if less than 10 images are found
    while len(images_with_captions) < 10:
        images_with_captions.append(
            (Image.new("RGB", (100, 100), color="gray"), "No result")
        )

    return images_with_captions


with gr.Blocks() as demo:
    gr.Markdown("# Image Search App")
    with gr.Tab("Image Search"):
        # Fix: Change input type to 'filepath'
        image_input = gr.Image(type="filepath", label="Upload an image")
        image_button = gr.Button("Search by Image")

    with gr.Tab("Text Search"):
        text_input = gr.Textbox(label="Enter text query")
        text_button = gr.Button("Search by Text")

    gallery = gr.Gallery(
        label="Search Results",
        show_label=False,
        elem_id="gallery",
        columns=2,
        height="auto",
    )

    image_button.click(
        fn=search_images,
        inputs=[image_input, gr.Textbox(value="image", visible=False)],
        outputs=[gallery],
    )

    text_button.click(
        fn=search_images,
        inputs=[text_input, gr.Textbox(value="text", visible=False)],
        outputs=[gallery],
    )

demo.launch()
