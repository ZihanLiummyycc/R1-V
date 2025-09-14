import cv2
import numpy as np
import os
import json
import glob
import yaml
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import argparse



def load_config(config_path: str) -> Optional[Dict]:
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except:
        return None


def apply_mask(img: np.ndarray, region: Tuple[int, int, int, int], 
               bg_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    if region is None:
        return img
    
    masked = img.copy()
    x_min, y_min, x_max, y_max = region
    
    if y_min > 0:
        masked[:y_min, :] = bg_color
    if y_max < img.shape[0]:
        masked[y_max:, :] = bg_color
    if x_min > 0:
        masked[:, :x_min] = bg_color
    if x_max < img.shape[1]:
        masked[:, x_max:] = bg_color
    
    return masked


def extract_fixed_ui_elements(img: np.ndarray, fixed_ui_config: Dict) -> List[Dict]:
    
    fixed_objects = []
    
    for element_name, element_config in fixed_ui_config.items():
        bbox = element_config.get('bbox')
        element_type = element_config.get('type', element_name)
        
        if bbox and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # 确保坐标在图像范围内
            x1 = max(0, min(x1, img.shape[1]))
            y1 = max(0, min(y1, img.shape[0]))
            x2 = max(0, min(x2, img.shape[1]))
            y2 = max(0, min(y2, img.shape[0]))
            
            if x2 > x1 and y2 > y1:  
                fixed_objects.append({
                    "class": element_name,
                    "confidence": 1.0,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "area": int((x2 - x1) * (y2 - y1))
                })
    
    return fixed_objects

def verify(obj_class,params,img: np.ndarray,x,y,w,h) -> Dict[str, float]:

    if img.size == 0:
        raise ValueError(f"Image is empty for class {obj_class} at ({x}, {y}, {w}, {h})")
    
    area = img.shape[0] * img.shape[1]

    area_range = params.get('area_range', (0, float('inf')))
    if not (area_range[0] <= area <= area_range[1]):
        # print(f"Skipping {obj_class} due to area {area} not in range {area_range}")
        return {"class": "unknown", "confidence": 1.0}
    
    width_range = params.get('width_range', (0, float('inf')))
    height_range = params.get('height_range', (0, float('inf')))
    if not (width_range[0] <= w <= width_range[1]) or not (height_range[0] <= h <= height_range[1]):
        # print(f"Skipping {obj_class} due to width {w} or height {h} not in range {width_range}, {height_range}")
        return {"class": "unknown", "confidence": 1.0}
    
    legal_region = params.get('legal_region', None)
    if legal_region:
        x_min, y_min, x_max, y_max = legal_region
        if not (x_min <= x < x_max and y_min <= y < y_max):
            print(f"Skipping {obj_class} due to position ({x}, {y}) not in legal region {legal_region}")
            return {"class": "unknown", "confidence": 1.0}

    return {"class": obj_class, "confidence": 1.0}


    


def detect_objects(objects_config, img_path: str, bg_colors: List[Tuple[int, int, int]], 
                   legal_region: Optional[Tuple[int, int, int, int]] = None,
                   min_area: int = 1) -> Tuple[np.ndarray, List[Dict]]:
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Cannot load {img_path}")
    
    objects = []
    for obj_class, params in objects_config.items():
        mask = np.ones(img.shape[:2], dtype=np.uint8)
        for color in bg_colors:
            mask &= ~np.all(img == color, axis=2)
        if legal_region:
            x_min, y_min, x_max, y_max = legal_region
            mask[:y_min, :] = 0
            mask[y_max:, :] = 0
            mask[:, :x_min] = 0
            mask[:, x_max:] = 0
            
        obj_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        
        # 添加对exclude_colors的支持
        if params.get('exclude_colors'):
            # 反向逻辑：创建包含所有非排除颜色的mask
            obj_mask = np.ones(img.shape[:2], dtype=np.uint8)
            for exclude_color in params['exclude_colors']:
                obj_mask &= ~np.all(img == exclude_color, axis=-1).astype(np.uint8)
        elif params.get('colors'):
            # 原有逻辑：包含指定颜色
            for color in params['colors']:
                obj_mask |= np.all(img == color, axis=-1).astype(np.uint8)
        
        mask &= obj_mask
        kernel_size = params.get('morphology_kernel', 7)  
        
        if params.get('connect_horizontal', True):
            h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size * 2, 1))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, h_kernel)
        
        if params.get('connect_vertical', True):
            v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_size * 2))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, v_kernel)
        
        ellipse_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ellipse_kernel)
        
        dilate_size = params.get('dilate_size', 2)

        vertical_dilate = params.get('vertical_dilate_size', dilate_size)
        if vertical_dilate != dilate_size:
            v_dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_dilate * 2))
            mask = cv2.dilate(mask, v_dilate_kernel, iterations=1)

        dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
        mask = cv2.dilate(mask, dilate_kernel, iterations=1)
        mask = cv2.erode(mask, dilate_kernel, iterations=1)
        
        # get connected components
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                x = stats[i, cv2.CC_STAT_LEFT]
                y = stats[i, cv2.CC_STAT_TOP]
                w = stats[i, cv2.CC_STAT_WIDTH]
                h = stats[i, cv2.CC_STAT_HEIGHT]
                
                crop = img[y:y+h, x:x+w]
                result = verify(obj_class, params, crop, x, y, w, h)
                
                x_offset = params.get('bbox_offset_x', 0)
                y_offset = params.get('bbox_offset_y', 0)

                objects.append({
                    "class": result["class"],
                    "confidence": float(result["confidence"]),
                    "bbox": [int(x + x_offset), int(y + y_offset), int(x+w + x_offset), int(y+h + y_offset)],
                    "area": int(w * h)
                })
    return img, objects

        
    
    

    
    # n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    
    # objects = []
    # for i in range(1, n_labels):
    #     # if stats[i, cv2.CC_STAT_AREA] >= min_area:
    #     x = stats[i, cv2.CC_STAT_LEFT]
    #     y = stats[i, cv2.CC_STAT_TOP]
    #     w = stats[i, cv2.CC_STAT_WIDTH]
    #     h = stats[i, cv2.CC_STAT_HEIGHT]
        
    #     crop = img[y:y+h, x:x+w]
    #     result = classify(objects_config, crop,x,y,w,h)
        
    #     objects.append({
    #         "class": result["class"],
    #         "confidence": float(result["confidence"]),
    #         "bbox": [int(x), int(y), int(x+w), int(y+h)],
    #         "area": int(w * h)
    #     })
    
    # return img, objects


def annotate(img: np.ndarray, objects: List[Dict], legal_region: Optional[Tuple] = None, 
             fixed_objects: List[Dict] = None) -> Tuple[np.ndarray, Dict[str, int]]:
    result = img.copy()
    counts = {}
    
    
    for obj in objects:
        cls = obj["class"]
        x1, y1, x2, y2 = obj["bbox"]
        
        color = (0, 255, 0)  # Default color for bounding boxes
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 1)  # Changed thickness to 1

        # Draw class label
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.2
        font_color = (255, 255, 255)  # White text
        thickness = 1
        cv2.putText(result, cls, (x1, y1 - 5), font, font_scale, font_color, thickness)
        
        counts[cls] = counts.get(cls, 0) + 1
    
    
    if fixed_objects:
        for obj in fixed_objects:
            cls = obj["class"]
            x1, y1, x2, y2 = obj["bbox"]
            
            color = (255, 0, 0)  # 蓝色框用于固定UI
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 1)

            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.2
            font_color = (0, 255, 255)  # 黄色文字
            thickness = 1
            cv2.putText(result, cls, (x1, y1 - 5), font, font_scale, font_color, thickness)
            
            counts[cls] = counts.get(cls, 0) + 1
    
    
    if legal_region:
        x1, y1, x2, y2 = legal_region
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 0), 1)
    
    return result, counts



def process_images(input_dir: str, output_dir: str, config_path: str):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "annotated"), exist_ok=True)
    # os.makedirs(os.path.join(output_dir, "crops"), exist_ok=True)
    
    config = load_config(config_path)
    legal_region = config.get('legal_region', None)
    fixed_ui_config = config.get('fixed_ui_elements', {})
    
    images = glob.glob(os.path.join(input_dir, "*.png"))
    results = []
    total_counts = {}
    
    for i, img_path in enumerate(tqdm(images, desc="Processing images")):
        
        # try:
        objects_config = config.get('objects', {})
        img, objects = detect_objects(objects_config,img_path, config.get('background_colors', []),legal_region)
        

        fixed_objects = extract_fixed_ui_elements(img, fixed_ui_config)

        # Filter out unknown objects
        objects = [obj for obj in objects if obj["class"] != "unknown"]
        base = os.path.splitext(os.path.basename(img_path))[0]
        
        annotated, counts = annotate(img, objects, legal_region, fixed_objects)
        cv2.imwrite(os.path.join(output_dir, "annotated", f"{base}.png"), annotated)
        
        # for j, obj in enumerate(objects):
        #     x1, y1, x2, y2 = obj["bbox"]
        #     crop = img[y1:y2, x1:x2]
        #     cv2.imwrite(os.path.join(output_dir, "crops", 
        #                 f"{base}_obj{j}_{obj['class']}.png"), crop)
        
        for cls, cnt in counts.items():
            total_counts[cls] = total_counts.get(cls, 0) + cnt
        
        all_objects = objects + fixed_objects
        results.append({
            "image": os.path.basename(img_path),
            "objects": all_objects,
            "counts": counts
        })
            
        # except Exception as e:
        #     print(f"Error: {img_path}: {e}")
    
    summary = {
        "total_images": len(results),
        "total_objects": sum(total_counts.values()),
        "class_counts": total_counts,
        "results": results
    }
    
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)
    
    return summary


def main(input_dir: str, config_path: str):
    # Configure paths here
    output_dir = input_dir + "_labelled"
    
    print("AirRaid Object Detection")
    print("-" * 30)
    
    summary = process_images(input_dir, output_dir, config_path)
    
    print(f"\nProcessed {summary['total_images']} images")
    print(f"Detected {summary['total_objects']} objects")
    print("\nClass distribution:")
    for cls, cnt in sorted(summary['class_counts'].items()):
        pct = cnt / summary['total_objects'] * 100
        print(f"  {cls}: {cnt} ({pct:.1f}%)")
    
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify objects in images using connected components')
    parser.add_argument('--input_dir', required=True, help='Path to the input directory containing images')
    parser.add_argument('--config_path', required=True, help='Path to the configuration YAML file')
    args = parser.parse_args()
    main(args.input_dir, args.config_path)
