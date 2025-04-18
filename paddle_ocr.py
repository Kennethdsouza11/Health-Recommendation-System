from paddleocr import PaddleOCR
from typing import Optional, List
import logging
from pathlib import Path
from dataclasses import dataclass
import os
from functools import lru_cache

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OCRConfig:
    """Configuration for OCR processing."""
    DEFAULT_LANG: str = 'en'
    USE_ANGLE_CLS: bool = True
    DET_MODEL_DIR: Optional[str] = None
    REC_MODEL_DIR: Optional[str] = None
    CLS_MODEL_DIR: Optional[str] = None
    MAX_IMAGE_SIZE: int = 4096  # Maximum dimension for image processing
    CACHE_SIZE: int = 100  # Number of OCR results to cache

class OCRProcessor:
    """OCR processor with caching and error handling."""
    def __init__(self, config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig()
        self._validate_paths()
        self.ocr = self._initialize_ocr()

    def _validate_paths(self):
        """Validate model paths if provided."""
        for path in [self.config.DET_MODEL_DIR, self.config.REC_MODEL_DIR, self.config.CLS_MODEL_DIR]:
            if path and not os.path.exists(path):
                logger.warning(f"Model path does not exist: {path}")

    def _initialize_ocr(self) -> PaddleOCR:
        """Initialize PaddleOCR with configuration."""
        try:
            return PaddleOCR(
                lang=self.config.DEFAULT_LANG,
                use_angle_cls=self.config.USE_ANGLE_CLS,
                det_model_dir=self.config.DET_MODEL_DIR,
                rec_model_dir=self.config.REC_MODEL_DIR,
                cls_model_dir=self.config.CLS_MODEL_DIR
            )
        except Exception as e:
            logger.error(f"Failed to initialize OCR: {str(e)}")
            raise

    def _validate_image(self, img_path: str) -> bool:
        """Validate image file."""
        try:
            path = Path(img_path)
            if not path.exists():
                logger.error(f"Image file not found: {img_path}")
                return False
            if not path.is_file():
                logger.error(f"Path is not a file: {img_path}")
                return False
            return True
        except Exception as e:
            logger.error(f"Error validating image: {str(e)}")
            return False

    @lru_cache(maxsize=OCRConfig.CACHE_SIZE)
    def extract_text_from_image(self, img_path: str, lang: Optional[str] = None) -> Optional[str]:
        """
        Extract text from an image using PaddleOCR with caching.
        
        Args:
            img_path: Path to the image file.
            lang: Language for OCR (defaults to config.DEFAULT_LANG).
            
        Returns:
            Extracted text or None if extraction failed.
        """
        try:
            # Validate input
            if not self._validate_image(img_path):
                return None

            # Perform OCR
            result = self.ocr.ocr(img_path, cls=True)
            
            # Handle empty results
            if not result or not result[0]:
                logger.warning(f"No text detected in image: {img_path}")
                return ""

            # Extract and join text
            detected_texts = [line[1][0] for line in result[0] if line and line[1]]
            final_text = " ".join(detected_texts)

            return final_text

        except Exception as e:
            logger.error(f"OCR processing failed for {img_path}: {str(e)}")
            return None

    def batch_process(self, img_paths: List[str], lang: Optional[str] = None) -> List[Optional[str]]:
        """
        Process multiple images in batch.
        
        Args:
            img_paths: List of image paths to process.
            lang: Language for OCR.
            
        Returns:
            List of extracted texts (None for failed extractions).
        """
        results = []
        for img_path in img_paths:
            try:
                text = self.extract_text_from_image(img_path, lang)
                results.append(text)
            except Exception as e:
                logger.error(f"Batch processing failed for {img_path}: {str(e)}")
                results.append(None)
        return results

# Example usage
if __name__ == "__main__":
    try:
        # Initialize processor
        processor = OCRProcessor()
        
        # Process single image
        img_path = "example.jpg"
        text = processor.extract_text_from_image(img_path)
        if text is not None:
            print(f"Extracted text: {text}")
        else:
            print("Text extraction failed")
            
        # Process multiple images
        img_paths = ["image1.jpg", "image2.jpg"]
        results = processor.batch_process(img_paths)
        for path, text in zip(img_paths, results):
            print(f"{path}: {text if text is not None else 'Failed'}")
            
    except Exception as e:
        logger.error(f"Application failed: {str(e)}")


