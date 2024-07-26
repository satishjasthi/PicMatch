import csv
from pathlib import Path
import time
import json
import os, io

import aiofiles
import aiohttp
import asyncio
from PIL import Image
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass


@dataclass
class ProcessState:
    urls_processed: int = 0
    images_downloaded: int = 0
    images_saved: int = 0
    images_resized: int = 0


class ImageProcessor(ABC):
    @abstractmethod
    def process(self, image: bytes, filename: str) -> None:
        pass


class ImageSaver(ImageProcessor):
    async def process(self, image: bytes, filename: str) -> None:
        async with aiofiles.open(filename, "wb") as f:
            await f.write(image)


def resize_image(image: bytes, filename: str, max_size: int = 300) -> None:
    with Image.open(io.BytesIO(image)) as img:
        img.thumbnail((max_size, max_size))
        img.save(filename, optimize=True, quality=85)


class RateLimiter:
    """
    High-Level Concept: The Token Bucket Algorithm
    ==============================================
    The Rate_Limiter class implements what's known as the "Token Bucket" algorithm. Imagine you have a bucket that can hold a certain number of tokens. Here's how it works:

    The bucket is filled with tokens at a constant rate.
    When you want to perform an action (in our case, make an API request), you need to take a token from the bucket.
    If there's a token available, you can perform the action immediately.
    If there are no tokens, you have to wait until a new token is added to the bucket.
    The bucket has a maximum capacity, so tokens don't accumulate indefinitely when not used.

    This mechanism allows for both steady-state rate limiting and handling short bursts of activity.

    In the constructor:
    ===================
    rate: is how many tokens we add per time period (e.g., 10 tokens per second)

    per: is the time period (usually 1 second)

    burst: is the bucket size (maximum number of tokens)

    We start with a full bucket (self.tokens = burst)
    We note the current time (self.updated_at)

    Logic:
    ======
    1. Calculate how much time has passed since we last updated the token count.

    2. Add tokens based on the time passed and our rate:
        self.tokens += time_passed * (self.rate / self.per)

    3. If we've added too many tokens, cap it at our maximum (burst size).

    4. Update our "last updated" time.

    5. If we have at least one token:
        Remove a token (self.tokens -= 1)
        Return immediately, allowing the API call to proceed

    6. If we don't have a token:
        Calculate how long we need to wait for the next token
        Sleep for that duration

    Let's walk through an example:
    ==============================
    Suppose we set up our RateLimiter like this:

    Copylimiter = RateLimiter(rate=10, per=1, burst=10)

    This means:
    - We allow 10 requests per second on average
    - We can burst up to 10 requests at once
    - After the burst, we'll be limited to 1 request every 0.1 seconds

    Now, imagine a sequence of API calls:

    1. The first 10 calls will happen immediately (burst capacity)
    2. The 11th call will wait for 0.1 seconds (time to generate 1 token)
    3. Subsequent calls will each wait about 0.1 seconds

    If there's a pause in API calls, tokens will accumulate (up to the burst limit), allowing for another burst of activity.

    This mechanism ensures that:
    1. We respect the average rate limit (10 per second in this example)
    2. We can handle short bursts of activity (up to 10 at once)
    3. We smoothly regulate requests when operating at capacity
    """

    def __init__(self, rate: float, per: float = 1.0, burst: int = 1):
        self.rate = rate
        self.per = per
        self.burst = burst
        self.tokens = burst
        self.updated_at = time.monotonic()

    async def wait(self):
        while True:
            now = time.monotonic()
            time_passed = now - self.updated_at
            self.tokens += time_passed * (self.rate / self.per)
            if self.tokens > self.burst:
                self.tokens = self.burst
            self.updated_at = now

            if self.tokens >= 1:
                self.tokens -= 1
                return
            else:
                await asyncio.sleep((1 - self.tokens) / (self.rate / self.per))


class ImagePipeline:
    def __init__(
        self,
        txt_file: str,
        loop: asyncio.AbstractEventLoop,
        max_concurrent_downloads: int = 10,
        max_workers: int = max(os.cpu_count() - 4, 4),
        rate_limit: float = 10,
        rate_limit_period: float = 1,
        downloaded_images_dir: str = "",
    ):
        self.txt_file = txt_file
        self.loop = loop
        self.url_queue = asyncio.Queue(maxsize=1000)
        self.image_queue = asyncio.Queue(maxsize=100)
        self.semaphore = asyncio.Semaphore(max_concurrent_downloads)
        self.state = ProcessState()
        self.state_file = "pipeline_state.json"
        self.saver = ImageSaver()
        self.process_pool = ProcessPoolExecutor(max_workers=max_workers)
        self.rate_limiter = RateLimiter(
            rate=rate_limit, per=rate_limit_period, burst=max_concurrent_downloads
        )
        self.downloaded_images_dir = Path(downloaded_images_dir)

    async def url_feeder(self):
        try:
            print(f"Starting to read URLs from {self.txt_file}")
            async with aiofiles.open(self.txt_file, mode="r") as f:
                line_number = 0
                async for line in f:
                    line_number += 1
                    if line_number <= self.state.urls_processed:
                        continue

                    url = line.strip()
                    if url:  # Skip empty lines
                        await self.url_queue.put(url)
                        self.state.urls_processed += 1

                        # Check if we need to wait for the queue to have space
                        if self.url_queue.qsize() >= self.url_queue.maxsize - 1:
                            await asyncio.sleep(0.1)
        except Exception as e:
            print(f"Error in url_feeder: {e}")
        finally:
            await self.url_queue.put(None)

    async def image_downloader(self):
        print("Starting image downloader")
        async with aiohttp.ClientSession() as session:
            while True:
                url = await self.url_queue.get()
                if url is None:
                    print("Finished downloading images")
                    await self.image_queue.put(None)
                    break
                try:
                    await self.rate_limiter.wait()  # Wait for rate limit
                    async with self.semaphore:
                        async with session.get(url) as response:
                            if response.status == 200:
                                image = await response.read()
                                await self.image_queue.put((image, url))
                                self.state.images_downloaded += 1
                                if self.state.images_downloaded % 100 == 0:
                                    print(
                                        f"Downloaded {self.state.images_downloaded} images"
                                    )
                except Exception as e:
                    print(f"Error downloading {url}: {e}")
                finally:
                    self.url_queue.task_done()

    async def image_processor(self):
        print("Starting image processor")
        while True:
            item = await self.image_queue.get()
            if item is None:
                print("Finished processing images")
                break
            image, url = item
            filename = os.path.basename(url)
            if not filename.lower().endswith((".png", ".jpg", ".jpeg")):
                filename += ".png"
            try:
                # Save the original image
                await self.saver.process(
                    image, str(self.downloaded_images_dir / f"original_{filename}")
                )
                self.state.images_saved += 1
                if self.state.images_resized % 100 == 0:
                    print(f"Processed {self.state.images_resized} images")

                # Resize the image using the process pool
                # loop = asyncio.get_running_loop()
                await self.loop.run_in_executor(
                    self.process_pool,
                    resize_image,
                    image,
                    str(self.downloaded_images_dir / f"resized_{filename}"),
                )
                self.state.images_resized += 1
            except Exception as e:
                print(f"Error processing {url}: {e}")
            finally:
                self.image_queue.task_done()

    def save_state(self):
        with open(self.state_file, "w") as f:
            json.dump(asdict(self.state), f)

    def load_state(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, "r") as f:
                self.state = ProcessState(**json.load(f))

    async def run(self):
        print("Starting pipeline")
        self.load_state()
        print(f"Loaded state: {self.state}")
        tasks = [
            asyncio.create_task(self.url_feeder()),
            asyncio.create_task(self.image_downloader()),
            asyncio.create_task(self.image_processor()),
        ]
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"Pipeline error: {e}")
        finally:
            self.save_state()
            print(f"Final state: {self.state}")
            self.process_pool.shutdown()
        print("Pipeline finished")


if __name__ == "__main__":
    from pathlib import Path

    PROJECT_ROOT = Path(__file__).resolve().parent
    loop = asyncio.get_event_loop()
    text_file = PROJECT_ROOT / "data/image_urls.txt"
    if not text_file.exists():
        import pandas as pd

        dataframe = pd.read_csv(PROJECT_ROOT / "data/photos.tsv000", sep="\t")
        num_image_urls = len(dataframe)
        print(f"Number of image urls: {num_image_urls}")
        with open(text_file, "w") as f:
            for url in dataframe["photo_image_url"]:
                f.write(url + "\n")
    print("Started downloading images")
    pipeline = ImagePipeline(
        txt_file=text_file,
        loop=loop,
        rate_limit=100,
        rate_limit_period=1,
        downloaded_images_dir=str(PROJECT_ROOT / "data/data/images"),
    )
    # asyncio.run(pipeline.run())
    loop.run_until_complete(pipeline.run())
    print("Finished downloading images")
