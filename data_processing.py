import cv2
import os
from descriptor import glcm, bitdesc
import numpy as np

def extract_features(image_path, descriptor_funcs):
    print(f"Reading image: {image_path}")
    try:
        img = cv2.imread(image_path, 0)
        if img is not None:
            combined_features = []
            for descriptor_func in descriptor_funcs:
                features = descriptor_func(image_path)
                combined_features.extend(features)
            
            return combined_features
        else:
            print(f"Failed to read image: {image_path}")
            return None
    except Exception as e:
        print(f"Error reading image {image_path}: {e}")
        return None

def process_datasets(root_folder):
    glcm_features_list = []
    bit_features_list = []
    combined_features_list = []
    
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.lower().endswith(('.jpg', '.png', '.jpeg')):
                image_rel_path = os.path.join(root, file)
                print(f"Processing file: {image_rel_path}")
                if os.path.isfile(image_rel_path):
                    try:
                        folder_name = os.path.basename(os.path.dirname(image_rel_path))
                        glcm_features = extract_features(image_rel_path, [glcm])
                        bit_features = extract_features(image_rel_path, [bitdesc])
                        combined_features = extract_features(image_rel_path, [glcm, bitdesc])
                        
                        if glcm_features is not None:
                            glcm_features = glcm_features + [folder_name, image_rel_path]
                            glcm_features_list.append(glcm_features)
                        if bit_features is not None:
                            bit_features = bit_features + [folder_name, image_rel_path]
                            bit_features_list.append(bit_features)
                        if combined_features is not None:
                            combined_features = combined_features + [folder_name, image_rel_path]
                            combined_features_list.append(combined_features)
                    except Exception as e:
                        print(f"Error processing file {image_rel_path}: {e}")
                else:
                    print(f"File does not exist: {image_rel_path}")
    
    glcm_signatures = np.array(glcm_features_list)
    bit_signatures = np.array(bit_features_list)
    combined_signatures = np.array(combined_features_list)
    
    print(f"GLCM features shape: {glcm_signatures.shape}")
    print(f"BIT features shape: {bit_signatures.shape}")
    print(f"Combined features shape: {combined_signatures.shape}")
    
    np.save('glcm_signatures.npy', glcm_signatures)
    np.save('bit_signatures.npy', bit_signatures)
    np.save('combined_signatures.npy', combined_signatures)
    
    print('Successfully stored!')

process_datasets('Projet1_Dataset')
