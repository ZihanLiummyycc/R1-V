import cv2
import argparse
import sys

def click_event(event, x, y, flags, param):
    """Mouse callback function to handle click events"""
    if event == cv2.EVENT_LBUTTONDOWN:
        img = param['image']
        color = img[y, x]
        print(f"Clicked at (x={x}, y={y}), BGR color={color.tolist()}")
        
        # Optional: Mark the clicked point
        marked_img = img.copy()
        cv2.circle(marked_img, (x, y), 1, (0, 255, 0), -1)
        cv2.imshow("Image", marked_img)

def main(image_path):
    """Main function to display image and handle clicks"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {image_path}")
        
        cv2.imshow("Image", img)
        cv2.setMouseCallback("Image", click_event, param={'image': img})
        
        print("Click on points of interest (press any key to exit)...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Get coordinates of points in an image')
    parser.add_argument('--img', required=True, help='Path to the image file')
    args = parser.parse_args()
    
    main(args.img)