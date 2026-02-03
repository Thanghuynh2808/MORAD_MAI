import cv2
import numpy as np
from retail_matcher.utils.common import get_class_color

class Visualizer:
    @staticmethod
    def draw_match(annotated_img, x1, y1, x2, y2, best_res, obb_poly=None):
        color = get_class_color(best_res['class'])
        
        # Visualize using OBB if available, otherwise use rectangle
        overlay = annotated_img.copy()
        if obb_poly is not None:
            cv2.fillPoly(overlay, [obb_poly], color)
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.3, annotated_img, 0.7, 0, annotated_img)

        # Draw border
        if obb_poly is not None:
            cv2.polylines(annotated_img, [obb_poly], True, color, 2)
        else:
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)

        label = f"{best_res['class']}: {best_res['score']:.2f}"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        cv2.rectangle(annotated_img, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), color, -1)
        cv2.putText(annotated_img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return annotated_img

    @staticmethod
    def draw_unknown(annotated_img, x1, y1, x2, y2, obb_poly=None):
        overlay = annotated_img.copy()
        if obb_poly is not None:
            cv2.fillPoly(overlay, [obb_poly], (0, 0, 0))
        else:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated_img, 0.5, 0, annotated_img)

        # Draw border
        if obb_poly is not None:
            cv2.polylines(annotated_img, [obb_poly], True, (128, 128, 128), 2)
        else:
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (128, 128, 128), 2)

        # Draw question mark
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        cv2.putText(annotated_img, "?", (center_x - 15, center_y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

        label = "Unknown"
        (text_width, text_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_img, (x1, y1 - text_height - 10), (x1 + text_width + 10, y1), (128, 128, 128), -1)
        cv2.putText(annotated_img, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return annotated_img
