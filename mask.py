import cv2
import numpy as np

def main():
    # Set the paths here
    segmentation_path = "/home/angus/ConceptExtraction-framework/segment/results/00/example_0_segmentation.png"  # Replace with your segmentation image path
    original_path = "/home/angus/ConceptExtraction-framework/segment/results/00/example_0_input.jpg"   # Replace with your original image path

    # Read segmentation and original images
    seg_img = cv2.imread(segmentation_path)
    orig_img = cv2.imread(original_path)
    if seg_img is None or orig_img is None:
        print("Error reading one of the images.")
        return
    
    # Resize both images to 512 x 512 if they are not already
    seg_img = cv2.resize(seg_img, (512, 512))
    orig_img = cv2.resize(orig_img, (512, 512))
    
    # Convert segmentation image into a label map using unique colors
    colors, labels_flat = np.unique(seg_img.reshape(-1, 3), axis=0, return_inverse=True)
    label_map = labels_flat.reshape(seg_img.shape[:2])
    
    # Merge similar colors: increase tolerance for more overmerging
    tolerance = 40
    merged_labels = {}
    merge_groups = {}
    for label, color in enumerate(colors):
        found = False
        for rep in merge_groups:
            rep_color = colors[rep]
            if np.linalg.norm(color - rep_color) < tolerance:
                merged_labels[label] = rep
                merge_groups[rep].append(label)
                found = True
                break
        if not found:
            merged_labels[label] = label
            merge_groups[label] = [label]
    
    # Build final label map using merged labels
    final_label_map = label_map.copy()
    for orig_label in range(len(colors)):
        if orig_label in merged_labels:
            final_label_map[label_map == orig_label] = merged_labels[orig_label]
    
    # For each unique merged label, merge connected components, apply dilation and further combine nearby components
    unique_labels = np.unique(final_label_map)
    for lbl in unique_labels:
        # Create a binary mask for current label
        merged_mask = (final_label_map == lbl).astype(np.uint8)
        # Apply slight dilation for overmerging segments
        kernel = np.ones((3,3), np.uint8)
        merged_mask = cv2.dilate(merged_mask, kernel, iterations=1)
        
        # First connected components pass with size filtering
        num_labels, comp_labels, stats, _ = cv2.connectedComponentsWithStats(merged_mask, connectivity=8)
        size_threshold = 50  # Adjust threshold as needed
        final_mask = np.zeros_like(merged_mask)
        for i in range(1, num_labels):  # Skip the background component
            if stats[i, cv2.CC_STAT_AREA] >= size_threshold:
                final_mask = cv2.bitwise_or(final_mask, (comp_labels == i).astype(np.uint8))
        
        # Additional merging: apply morphological closing to further merge adjacent components
        closing_kernel = np.ones((5,5), np.uint8)
        closed_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, closing_kernel, iterations=2)
        
        # Second connected components pass to refine merged regions
        num_labels2, comp_labels2, stats2, _ = cv2.connectedComponentsWithStats(closed_mask, connectivity=8)
        refined_mask = np.zeros_like(closed_mask)
        for i in range(1, num_labels2):
            if stats2[i, cv2.CC_STAT_AREA] >= size_threshold:
                refined_mask = cv2.bitwise_or(refined_mask, (comp_labels2 == i).astype(np.uint8))
        
        mask_img = refined_mask * 255
        mask_3ch = cv2.merge([mask_img, mask_img, mask_img])
        masked_img = cv2.bitwise_and(orig_img, mask_3ch)
        
        cv2.imwrite(f"./ConceptExtraction-framework/datasets/mask_cluster_{lbl}.png", masked_img)
        print(f"Saved mask_cluster_{lbl}.png")
        cv2.imwrite(f"./ConceptExtraction-framework/datasets/mask_{lbl}.png", mask_img)
        print(f"Saved binary_mask_{lbl}.png")

if __name__ == "__main__":
    main()