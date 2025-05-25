import http.server
import socketserver
import random
from threading import Thread
import numpy as np
import time
from model import Generator
import torch
import cv2

PORT = 8001
CACHE_SIZE = 5


class GenerationCache(Thread):
    def __init__(self):
        super().__init__()
        self.cache = []
        self.generator = Generator()
        self.generator.eval()

    def run(self):
        print("Starting image generation thread")

        while True:
            if len(self.cache) < CACHE_SIZE:
                seed = random.randint(0, 2**32 - 1)
                print(f"Starting generation with seed {seed}, total cache size: {len(self.cache)}")

                with torch.no_grad():
                    image = self.generator(seed)

                image = image.astype(np.float32)
                image = cv2.GaussianBlur(image, (3, 3), 0)
                self.cache.append(image)
            else:
                time.sleep(1.0)

    def get_image(self):
        if len(self.cache) == 0:
            return None
        else:
            image = self.cache.pop(0)
            return image


CACHE = GenerationCache()


class RequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-type', 'text/plain')
            self.end_headers()
            self.wfile.write(b"good")

        elif self.path == '/data':
            image = CACHE.get_image()

            if image is None:
                self.send_response(204)
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                self.wfile.write(b"wait")
                return

            self.send_response(200)
            self.send_header('Content-type', 'application/octet-stream')
            self.end_headers()
            self.wfile.write(image.tobytes())
            
        else:
            self.send_error(404, "File not found")


def main():
    CACHE.start()

    with socketserver.TCPServer(("", PORT), RequestHandler) as httpd:
        print("Serving at port", PORT)
        httpd.serve_forever()


if __name__ == "__main__":
    main()