import asyncio
import os
import logging
from PIL import Image
import torch
from transformers import (
    CLIPProcessor,
    CLIPModel,
    BlipProcessor,
    BlipForConditionalGeneration,
)
from sentence_transformers import SentenceTransformer
import numpy as np
import aiofiles
import json
from abc import ABC, abstractmethod
from typing import Set, Tuple
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = "cpu"


@dataclass
class State:
    processed_files: Set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {"processed_files": list(self.processed_files)}

    @staticmethod
    def from_dict(state_dict: dict) -> "State":
        return State(processed_files=set(state_dict.get("processed_files", [])))


class ImageProcessor(ABC):
    @abstractmethod
    def process(self, image: Image.Image) -> np.ndarray:
        pass


class CLIPImageProcessor(ImageProcessor):
    def __init__(self):
        self.model = CLIPModel.from_pretrained(
            "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
        ).to(device)
        self.processor = CLIPProcessor.from_pretrained(
            "wkcn/TinyCLIP-ViT-8M-16-Text-3M-YFCC15M"
        )
        print("Initialized CLIP model and processor")

    def process(self, image: Image.Image) -> np.ndarray:
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        outputs = self.model.get_image_features(**inputs)
        return outputs.detach().cpu().numpy()


class ImageCaptioningProcessor(ImageProcessor):
    def __init__(self):
        self.image_caption_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(device)
        self.image_caption_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        )
        self.text_embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2", device=device
        )
        print("Initialized BLIP model and processor")

    def process(self, image: Image.Image) -> np.ndarray:
        inputs = self.image_caption_processor(images=image, return_tensors="pt").to(
            device
        )
        output = self.image_caption_model.generate(**inputs)
        caption = self.image_caption_processor.decode(
            output[0], skip_special_tokens=True
        )
        # embedding dim 384
        return self.text_embedding_model.encode(caption).flatten()


class ImageFeatureExtractor:
    def __init__(
        self,
        clip_processor: CLIPImageProcessor,
        caption_processor: ImageCaptioningProcessor,
        max_queue_size: int = 100,
        checkpoint_file: str = "checkpoint.json",
    ):
        self.clip_processor = clip_processor
        self.caption_processor = caption_processor
        self.image_queue = asyncio.Queue(maxsize=max_queue_size)
        self.processed_images_queue = asyncio.Queue()
        self.checkpoint_file = checkpoint_file
        self.state = self.load_state()
        self.executor = ProcessPoolExecutor()
        self.total_images = 0
        self.processed_count = 0
        print(
            "Initialized ImageFeatureExtractor with checkpoint file:", checkpoint_file
        )

    async def image_loader(self, input_folder: str):
        print(f"Loading images from {input_folder}")
        for filename in os.listdir(input_folder):
            if "resized_" in filename and filename not in self.state.processed_files:
                try:
                    file_path = os.path.join(input_folder, filename)
                    await self.image_queue.put((filename, file_path))
                    self.total_images += 1
                    print(f"Loaded image {filename} into queue")
                except Exception as e:
                    logger.error(f"Error loading image {filename}: {e}")
        await self.image_queue.put(None)  # Sentinel to signal end of images
        print(f"Total images to process: {self.total_images}")

    async def image_processor_worker(self, loop: asyncio.AbstractEventLoop):
        while True:
            item = await self.image_queue.get()
            if item is None:
                await self.image_queue.put(None)  # Propagate sentinel
                break
            filename, file_path = item
            try:
                print(f"Processing image {filename}")
                image = Image.open(file_path)
                clip_embedding, caption_embedding = await asyncio.gather(
                    loop.run_in_executor(
                        self.executor, self.clip_processor.process, image
                    ),
                    loop.run_in_executor(
                        self.executor, self.caption_processor.process, image
                    ),
                )
                await self.processed_images_queue.put(
                    (filename, clip_embedding, caption_embedding)
                )
                print(f"Processed image {filename}")
            except Exception as e:
                logger.error(f"Error processing image {filename}: {e}")
            finally:
                self.image_queue.task_done()

    async def save_processed_images(self, output_folder: str):
        while self.processed_count < self.total_images:
            filename, clip_embedding, caption_embedding = (
                await self.processed_images_queue.get()
            )
            try:
                clip_output_path = os.path.join(
                    output_folder, f"{os.path.splitext(filename)[0]}_clip.npy"
                )
                caption_output_path = os.path.join(
                    output_folder, f"{os.path.splitext(filename)[0]}_caption.npy"
                )

                await asyncio.gather(
                    self.save_embedding(clip_output_path, clip_embedding),
                    self.save_embedding(caption_output_path, caption_embedding),
                )

                self.state.processed_files.add(filename)
                self.save_state()
                self.processed_count += 1
                print(f"Saved processed embeddings for {filename}")
            except Exception as e:
                logger.error(f"Error saving processed image {filename}: {e}")
            finally:
                self.processed_images_queue.task_done()

    async def save_embedding(self, output_path: str, embedding: np.ndarray):
        async with aiofiles.open(output_path, "wb") as f:
            await f.write(embedding.tobytes())

    def load_state(self) -> State:
        try:
            with open(self.checkpoint_file, "r") as f:
                state_dict = json.load(f)
                print("Loaded state from checkpoint")
                return State.from_dict(state_dict)
        except (FileNotFoundError, json.JSONDecodeError):
            print("No checkpoint found, starting with empty state")
            return State()

    def save_state(self):
        with open(self.checkpoint_file, "w") as f:
            json.dump(self.state.to_dict(), f)
            print("Saved state to checkpoint")

    async def run(
        self,
        input_folder: str,
        output_folder: str,
        loop: asyncio.AbstractEventLoop,
        num_workers: int = 2,
    ):
        os.makedirs(output_folder, exist_ok=True)
        print(f"Output folder {output_folder} created")

        tasks = [
            loop.create_task(self.image_loader(input_folder)),
            loop.create_task(self.save_processed_images(output_folder)),
        ]
        tasks.extend(
            [
                loop.create_task(self.image_processor_worker(loop))
                for _ in range(num_workers)
            ]
        )

        await asyncio.gather(*tasks)


class ImageFeatureExtractorFactory:
    @staticmethod
    def create() -> ImageFeatureExtractor:
        print(
            "Creating ImageFeatureExtractor with CLIPImageProcessor and ImageCaptioningProcessor"
        )
        return ImageFeatureExtractor(CLIPImageProcessor(), ImageCaptioningProcessor())


async def main(loop: asyncio.AbstractEventLoop, input_folder: str, output_folder: str):
    print("Starting main function")

    extractor = ImageFeatureExtractorFactory.create()

    try:
        await extractor.run(input_folder, output_folder, loop)
    except Exception as e:
        logger.error(f"An error occurred during execution: {e}")
    finally:
        logger.info("Image processing completed.")


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    print("Event loop created and set")
    input_folder = str(PROJECT_ROOT / "data/images")
    output_folder = str(PROJECT_ROOT / "data/features")
    loop.run_until_complete(main(loop, input_folder, output_folder))
    loop.close()
    print("Event loop closed")
