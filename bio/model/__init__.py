# bio-ai, Apache-2.0 license
# Filename: bio/model/FastAPIBaseModel.py
# Description: Model classes for FastAPI served models

from bio.logger import info, debug
import io
from PIL import Image


class FastAPIBaseModel:

    def __init__(self, endpoint: str):
        """
        FastAPI model base class
        :param endpoint: Endpoint for the FastAPI app, e.g. 'localhost:3000/predict_to_json'
        """
        # Make sure there is a / at the end of the path for handling file uploads
        if not endpoint.endswith('/'):
            endpoint = endpoint + '/'
        info(f'Initializing FastAPIBaseModel with endpoint: {endpoint}')
        self.endpoint = endpoint

    def predict_bytes(self, image_bytes: bytes, threshold: float, top_n: int) -> dict:
        pass


class YOLOv5(FastAPIBaseModel):

    def __init__(self, endpoint: str):
        """YOLOv5 model"""
        super().__init__(endpoint)

    def predict_file(self, image_path: str, threshold: float) -> dict:
        # Convert to bytes
        image = Image.open(image_path)
        byte_io = io.BytesIO()
        image.save(byte_io, format='JPEG')
        byte_io.seek(0)
        results = super().predict_bytes(byte_io, threshold)
        image.close()

        # Filter results by confidence
        results = [result for result in results['result'] if result['confidence'] >= threshold]

        return results


class KClassify(FastAPIBaseModel):

    def __init__(self, endpoint: str):
        """Keras Classify model"""
        super().__init__(endpoint)

    def predict_bytes(self, image_bytes: bytes, threshold: float, top_n: int) -> dict:
        """
        Predict image
        :param image_bytes: image bytes
        :param threshold: score threshold
        :param top_n: Number of top predictions to return, sorted by score in descending order up to 5
        """
        results = super().predict_bytes(image_bytes)
        debug(results)

        # Filter results by confidence
        results = [result for result in results['result'] if result['score'] >= threshold]

        # Sort results by score in descending order
        results = sorted(results, key=lambda k: k['score'], reverse=True)

        # Return the top n results
        results = results[:top_n]

        return results

    def predict_file(self, image_path: str, threshold: float, top_n: int) -> dict:
        """
        Predict image
        :param image_path: image path
        :param threshold: score threshold
        :param top_n: Number of top predictions to return, sorted by score in descending order up to 5
        """
        # Convert to bytes
        image = Image.open(image_path)
        byte_io = io.BytesIO()
        image.save(byte_io, format='JPEG')
        byte_io.seek(0)
        results = self.predict_bytes(byte_io, threshold, top_n)
        image.close()
        return results
