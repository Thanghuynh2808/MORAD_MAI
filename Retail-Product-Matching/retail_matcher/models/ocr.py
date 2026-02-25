import os
import cv2
import numpy as np
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
import easyocr

class PriceTagParser:
    def __init__(self, yolo_model_path="weight/best.pt"):
        """
        Initialize the Price Tag recognition system.
        Includes YOLO model for price area detection and OCR (PaddleOCR + EasyOCR for reading text).
        """
        print("Initializing OCR modules (PaddleOCR & EasyOCR)...")
        self.paddle_engine = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
        self.easy_engine = easyocr.Reader(['en'])
        
        print(f"Loading YOLO model from {yolo_model_path}...")
        try:
            self.model = YOLO(yolo_model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None

    def preprocess_crop(self, img):
        """
        Preprocess cropped image before inputting into OCR
        img: numpy array (BGR Image)
        """
        if img is None or img.size == 0 or img.shape[0] == 0 or img.shape[1] == 0:
            return img

        h, w = img.shape[:2]

        # b1: Resize: Scale crop up to at least 150px height to avoid small text
        target_height = 150
        if h < target_height:
            scale = target_height / h
            new_w = int(w * scale)
            # Use INTER_CUBIC for smoother text sharpening
            img = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

        # b2: Grayscale + CLAHE (Contrast Limited Adaptive Histogram Equalization)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        # b3: Angle correction
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)
        
        if lines is not None:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                angles.append(angle)
                
            median_angle = np.median(angles)
            
            if abs(median_angle) > 1.0 and abs(median_angle) < 45.0:
                (h, w) = gray.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                gray = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

        # b4: Denoise: reduce image noise from camera
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

        return denoised

    def combine_crops(self, crops, cols=4):
        """
        Combine a list of cropped images into one large grid image.
        """
        # Filter empty/None images
        crops = [c for c in crops if c is not None and c.size > 0]
        if not crops:
            return None
        
        # Calculate grid cell size based on the largest image in the batch
        max_w = max(c.shape[1] for c in crops)
        max_h = max(c.shape[0] for c in crops)
        
        padded_crops = []
        for c in crops:
            # Pad both horizontally and vertically for uniformity
            h, w = c.shape[:2]
            padded = cv2.copyMakeBorder(c, 0, max_h - h, 0, max_w - w, cv2.BORDER_CONSTANT, value=255)
            # Draw a thin border around each crop for easy distinction
            cv2.rectangle(padded, (0, 0), (max_w-1, max_h-1), (220, 220, 220), 1)
            padded_crops.append(padded)
        
        # Split the list into rows
        rows = []
        for i in range(0, len(padded_crops), cols):
            batch = padded_crops[i:i+cols]
            # If the last row is missing images, fill with white images
            while len(batch) < cols:
                blank = np.full((max_h, max_w, 3), 255, dtype=np.uint8)
                batch.append(blank)
            rows.append(cv2.hconcat(batch))
        
        # Vertically concatenate all rows
        combined = cv2.vconcat(rows)
        return combined

    def extract_prices(self, text):
        """
        Use regex to extract valid currency values from text.
        Remove barcodes/SKUs (length >= 9 digits).
        """
        text = text.replace(" ", "")
        matches = re.findall(r'\d+[.,\d]*\d+|\d+', text)
        
        valid_prices = []
        for m in matches:
            m = m.strip(".,")
            digits_only = re.sub(r'[.,]', '', m)
            # Only take numbers with 3 to 8 digits (to avoid barcodes)
            if 3 <= len(digits_only) < 9:
                # Get actual value (int float) for Min/Max comparison
                try:
                    # If both comma and dot exist, assume the last one is the decimal separator.
                    # In VN, prices usually don't have decimals, so we convert everything to raw string digits.
                    val = int(digits_only)
                    valid_prices.append((m, val))
                except ValueError:
                    pass
                    
        return valid_prices

    def run_ocr(self, crop_img):
        """
        Recognize text on crop image using PaddleOCR with EasyOCR fallback.

        Returns
        -------
        dict or None
            Structured result for backend consumption::

                {
                    "price": str,          # highest detected price string, e.g. "15,000"
                    "discount_price": str | None,  # second price if present
                    "original_price": str | None,
                    "confidence": float,
                    "source": "Paddle" | "EasyOCR",
                    "all_prices": [(price_str, int_val), ...]
                }

            Returns ``None`` if no valid price value was found.
        """
        paddle_res = self.paddle_engine.ocr(crop_img, cls=True)

        valid_results = []
        source = "Paddle"

        if paddle_res and paddle_res[0] is not None:
            for line in paddle_res[0]:
                box, (text, conf) = line
                if conf >= 0.5:
                    found_prices = self.extract_prices(text)
                    for (p_str, p_val) in found_prices:
                        valid_results.append({
                            "box": box,
                            "text": p_str,
                            "val": p_val,
                            "conf": conf,
                            "raw_text": text
                        })

        avg_conf = np.mean([r["conf"] for r in valid_results]) if valid_results else 0.0

        # Fallback to EasyOCR if empty or low confidence
        if not valid_results or avg_conf < 0.7:
            easy_res = self.easy_engine.readtext(crop_img)
            easy_valid_results = []
            if easy_res:
                for line in easy_res:
                    box, text, conf = line
                    if conf >= 0.5:
                        found_prices = self.extract_prices(text)
                        for (p_str, p_val) in found_prices:
                            easy_valid_results.append({
                                "box": box,
                                "text": p_str,
                                "val": p_val,
                                "conf": conf,
                                "raw_text": text
                            })

            easy_avg_conf = np.mean([r["conf"] for r in easy_valid_results]) if easy_valid_results else 0.0
            if not valid_results or (easy_valid_results and easy_avg_conf > avg_conf):
                valid_results = easy_valid_results
                source = "EasyOCR"
                avg_conf = easy_avg_conf
            else:
                source = "Paddle"

        if not valid_results:
            return None

        # Deduplicate prices by integer value, keep highest confidence
        unique_prices = {}
        for r in valid_results:
            if r["val"] not in unique_prices or r["conf"] > unique_prices[r["val"]]["conf"]:
                unique_prices[r["val"]] = r

        sorted_vals = sorted(unique_prices.keys(), reverse=True)
        all_prices = [(unique_prices[v]["text"], v) for v in sorted_vals]

        result = {
            "price": all_prices[0][0] if all_prices else None,
            "original_price": all_prices[0][0] if len(all_prices) >= 2 else None,
            "discount_price": all_prices[1][0] if len(all_prices) >= 2 else None,
            "confidence": float(avg_conf),
            "source": source,
            "all_prices": all_prices,
        }
        # Convenience key: if two prices, "price" = the cheaper (discount) one
        if len(all_prices) >= 2:
            result["price"] = all_prices[1][0]   # discount price is the lower value

        return result

    def run_ocr_visual(self, crop_img):
        """
        Same as run_ocr but returns a visualized numpy image instead of a dict.
        Used by process_folder() for batch demo / debugging.
        Returns None if no price found.
        """
        ocr_data = self.run_ocr(crop_img)
        if ocr_data is None:
            return None

        # Rebuild vis_img
        if len(crop_img.shape) == 2:
            vis_img = cv2.cvtColor(crop_img, cv2.COLOR_GRAY2BGR)
        else:
            vis_img = crop_img.copy()
        h, w = vis_img.shape[:2]

        # Re-run detection just for box drawing (lightweight; results already cached above)
        # We use the source info from ocr_data to decide color
        source = ocr_data["source"]
        color = (0, 255, 0) if source == "Paddle" else (255, 0, 0)

        # Build display labels
        display_texts = []
        if ocr_data.get("original_price") and ocr_data.get("discount_price"):
            display_texts.append(f"Ori:  {ocr_data['original_price']}")
            display_texts.append(f"Disc: {ocr_data['discount_price']}")
        elif ocr_data.get("price"):
            display_texts.append(f"Price: {ocr_data['price']}")

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        line_height = 20
        pad_h = max((len(display_texts) * line_height) + 35, 60)

        padded = cv2.copyMakeBorder(vis_img, 0, pad_h, 0, 0, cv2.BORDER_CONSTANT, value=(255, 255, 255))
        current_y = h + 20
        for txt in display_texts:
            cv2.putText(padded, txt, (5, current_y), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
            current_y += line_height
        cv2.putText(padded, f"[{source}] Conf: {ocr_data['confidence']:.2f}",
                    (5, current_y), font, 0.4, (200, 0, 0), 1, cv2.LINE_AA)
        return padded

    def process_image(self, img_or_path, output_folder=None):
        """
        Process 1 image: detect tag bounding boxes, preprocess crops, run OCR.

        Returns
        -------
        list[dict]
            Each dict is a structured price-tag result ready for the backend::

                {
                    "box": [x1, y1, x2, y2],   # tag location in original image coords
                    "price": str | None,
                    "original_price": str | None,
                    "discount_price": str | None,
                    "confidence": float,
                    "source": str,
                }
        """
        if self.model is None:
            print("YOLO model not initialized successfully!")
            return []

        # Accept both file path and numpy array
        if isinstance(img_or_path, np.ndarray):
            original_img = img_or_path
            img_name = "array_input"
        else:
            original_img = cv2.imread(str(img_or_path))
            img_name = os.path.basename(str(img_or_path))

        if original_img is None:
            print(f"Cannot read image: {img_or_path}")
            return []

        results = self.model.predict(source=original_img, save=False, verbose=False)
        tag_dicts = []

        for result in results:
            if output_folder and img_name != "array_input":
                save_path = os.path.join(output_folder, f"det_{img_name}")
                result.save(filename=save_path)

            boxes = result.boxes.xyxy.cpu().numpy()

            for box in boxes:
                x1, y1, x2, y2 = map(int, box[:4])
                h_img, w_img = original_img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)

                crop_img = original_img[y1:y2, x1:x2]
                processed = self.preprocess_crop(crop_img)

                if processed is not None:
                    # Bug 5 fix: run_ocr now returns structured dict, not a visual image
                    ocr_data = self.run_ocr(processed)
                    if ocr_data is not None:
                        ocr_data["box"] = [x1, y1, x2, y2]  # original image coords
                        tag_dicts.append(ocr_data)

        return tag_dicts

    def process_folder(self, input_folder, output_folder):
        """
        Process all images in a folder and combine VISUAL results (for demo/debugging).
        Uses run_ocr_visual() to produce annotated images.
        """
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created directory: {output_folder}")

        extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(extensions)]

        if not image_files:
            print(f"No images found in folder {input_folder}")
            return

        all_visual_crops = []
        print(f"Found {len(image_files)} images. Starting pipeline...")

        for img_name in image_files:
            img_path = os.path.join(input_folder, img_name)
            print(f"Processing: {img_name}")

            original_img = cv2.imread(img_path)
            if original_img is None or self.model is None:
                continue

            results = self.model.predict(source=original_img, save=False, verbose=False)
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box[:4])
                    h_i, w_i = original_img.shape[:2]
                    crop_img = original_img[max(0,y1):min(h_i,y2), max(0,x1):min(w_i,x2)]
                    processed = self.preprocess_crop(crop_img)
                    if processed is not None:
                        vis = self.run_ocr_visual(processed)
                        if vis is not None:
                            all_visual_crops.append(vis)

        if all_visual_crops:
            print(f"Combining {len(all_visual_crops)} crop pieces...")
            combined_result = self.combine_crops(all_visual_crops, cols=4)
            if combined_result is not None:
                combined_path = os.path.join(output_folder, "combined_crops_result.jpg")
                cv2.imwrite(combined_path, combined_result)
                print(f"-> Combined result saved at: {combined_path}")
        else:
            print("No objects found to crop.")

if __name__ == "__main__":
    parser = PriceTagParser(yolo_model_path="weight/best.pt")
    parser.process_folder(input_folder="images", output_folder="result")
    print("\nCompleted!")
