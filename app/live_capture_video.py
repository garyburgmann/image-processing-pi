""" Used to capture viedeo with pi camera """
from typing import Generator
import io

import picamera


class LiveCaptureVideo:
    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        framerate: int = 30
    ):
        self._width = width
        self._height = height
        self._framerate = framerate

    def frames(self) -> Generator[Image, None, None]:
        with picamera.PiCamera(
            resolution=(self._width, self._height),
            framerate=self._framerate
        ) as camera:
            camera.start_preview()
            try:
                stream = io.BytesIO()
                for _ in camera.capture_continuous(
                    stream,
                    format='jpeg',
                    use_video_port=True
                ):
                    stream.seek(0)

                    yield Image.open(stream).convert('RGB')

                    stream.seek(0)
                    stream.truncate()
            except Exception as exc:
                print(exc)
            finally:
                camera.stop_preview()
