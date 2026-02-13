
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

#setup function
def load_video_frames(frames_path, max_frames=None):
    
    #Load image frames from a folder and convert to grayscale.
    frames_dir = Path(frames_path)
    #Get all image files and sort them 
    image_files = sorted(frames_dir.glob('*.jpg'))
    # Sort numerically 
    try:
        image_files = sorted(image_files, key=lambda x: int(''.join(filter(str.isdigit, x.stem))))
    except:
        image_files = sorted(image_files)
    
    frames = []
    for img_path in image_files:
        # Read image
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        
        #Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        frames.append(gray.astype(np.float32) / 255.0)  # Normalize to [0,1]
        if max_frames and len(frames) >= max_frames:
            break
    
    print(f"Loaded {len(frames)} from {frames_path}")
    return frames

def create_gaussian_derivative_1d(sigma):
    
    #Create 1D Gaussian derivative kernel

    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    center = kernel_size // 2
    
    x = np.arange(kernel_size) - center
    #Derivative of Gaussian: -x/(sigma^2) * exp(-x^2/(2*sigma^2))
    kernel = -x / (sigma**2) * np.exp(-x**2 / (2 * sigma**2))
    kernel = kernel / np.sum(np.abs(kernel))  #normalize
    
    return kernel

#temporal derivative
def compute_temporal_derivative_simple(frames, frame_idx):
    """
    Compute temporal derivative using simple [-1, 0, 1] filter.
    Returns: temporal derivative image
    """
    if frame_idx == 0 or frame_idx >= len(frames) - 1:
        return np.zeros_like(frames[frame_idx])
    
    # Simple derivative: 0.5 * (next_frame - prev_frame)
    derivative = 0.5 * (frames[frame_idx + 1] - frames[frame_idx - 1])
    
    return derivative
#temporal derivative but gaussian 
def compute_temporal_derivative_gaussian(frames, frame_idx, t_sigma):
    
    #get temporal derivative using Gaussian derivative filter.
   
    
    kernel = create_gaussian_derivative_1d(t_sigma)
    kernel_size = len(kernel)
    half_size = kernel_size // 2
    
    #enough frames on both sides
    if frame_idx < half_size or frame_idx >= len(frames) - half_size:
        return np.zeros_like(frames[frame_idx])
    
    #apply kernel across temporal dimension
    derivative = np.zeros_like(frames[frame_idx])
    
    for i, weight in enumerate(kernel):
        offset = i - half_size
        derivative += weight * frames[frame_idx + offset]
    
    return derivative

#pick threshold
def estimate_noise_std(derivatives):
    """
    Estimate standard deviation of background noise from temporal derivatives.
    Most pixels are background with small temporal changes, model them as Gaussian noise with zero mean
    Returns: Estimated standard deviation of noise
    """
    #Using median absolute deviation for robust estimation
    mad = np.median(np.abs(derivatives))
    
    #for Gaussian distribution: std = 1.4826*MAD
    std_estimate = 1.4826 * mad
    
    return std_estimate

def select_threshold(derivatives, n_std=3.0):
    #Select threshold based on noise statistics.
 
    noise_std = estimate_noise_std(derivatives)
    threshold = n_std * noise_std
    return threshold

#motion detection process
def detect_motion(frames, frame_idx, temporal_method='simple', t_sigma=1.0, 
                  threshold_method='adaptive', manual_threshold=None, n_std=3.0):
    """
    Complete motion detection pipeline for a single frame.
    Args: frames: List of video frames, frame_idx: Index of frame to process
        temporal_method: 'simple' or 'gaussian', t_sigma: Temporal Gaussian std dev (if using gaussian)
        threshold_method: 'adaptive' or 'manual', manual_threshold: Manual threshold value
        n_std: Number of std devs for adaptive threshold
    Returns: Dictionary with results
    """
    #1- Compute temporal derivative
    if temporal_method == 'simple':
        derivative = compute_temporal_derivative_simple(frames, frame_idx)
    elif temporal_method == 'gaussian':
        derivative = compute_temporal_derivative_gaussian(frames, frame_idx, t_sigma)
    else:
        raise ValueError(f"Unknown temporal method: {temporal_method}")
    
    #2- Compute absolute value
    abs_derivative = np.abs(derivative)
    
    #3- Select threshold
    if threshold_method == 'adaptive':
        threshold = select_threshold(abs_derivative, n_std)
    else:
        threshold = manual_threshold
    
    #4-Create binary mask
    mask = (abs_derivative > threshold).astype(np.uint8)
    
    #5- Overlay mask on original frame
    result_frame = frames[frame_idx].copy()
    overlay = np.stack([result_frame, result_frame, result_frame], axis=2)
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], mask.astype(np.float32))
    
    return {
        'derivative': derivative,
        'abs_derivative': abs_derivative,
        'threshold': threshold,
        'mask': mask,
        'overlay': overlay,
        'noise_std': estimate_noise_std(abs_derivative)
    }

#the 15pt part, temporal derivative filter
def experiment_temporal_filters(frames, frame_idx):
    
    #Compare simple and Gaussian temporal derivative filters - 3 values of sigma
    
    
    #Test parameters-3 different t_sigma values
    t_sigmas = [0.5, 1.0, 2.0]
    fig, axes = plt.subplots(2, len(t_sigmas) + 1, figsize=(16, 8))
    
    #Simple filter: 0.5[-1, 0, 1]
    result_simple = detect_motion(frames, frame_idx, temporal_method='simple')
    
    axes[0, 0].imshow(result_simple['abs_derivative'], cmap='hot')
    axes[0, 0].set_title('Simple Filter\n[-1, 0, 1]')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(result_simple['mask'], cmap='gray')
    axes[1, 0].set_title(f"Mask (t={result_simple['threshold']:.4f})")
    axes[1, 0].axis('off')
    print(f"\n\n--------")

    print(f"Temporal derivative filter- frame: "+ str(frame_idx))
    print(f"\nSimple Filter:")
    print(f"   Threshold: {result_simple['threshold']:.4f}")
    print(f"   Noise std: {result_simple['noise_std']:.4f}")
    print(f"   Motion pixels: {np.sum(result_simple['mask'])} ({100*np.mean(result_simple['mask']):.2f}%)")
    
    #Gaussian filters with diff sigma values
    for i, t_sigma in enumerate(t_sigmas): #1.0 and 2.0
        result = detect_motion(frames, frame_idx, 
                              temporal_method='gaussian', t_sigma=t_sigma)
        
        axes[0, i+1].imshow(result['abs_derivative'], cmap='hot')
        axes[0, i+1].set_title(f'Gaussian Derivative\nσ_t={t_sigma}')
        axes[0, i+1].axis('off')
        
        axes[1, i+1].imshow(result['mask'], cmap='gray')
        axes[1, i+1].set_title(f"Mask (t={result['threshold']:.4f})")
        axes[1, i+1].axis('off')
        
        print(f"Gaussian Filter (σ_t={t_sigma}):")
        print(f"  Threshold: {result['threshold']:.4f}")
        print(f"  Noise std: {result['noise_std']:.4f}")
        print(f"  Motion pixels: {np.sum(result['mask'])} ({100*np.mean(result['mask']):.2f}%)")
    
    plt.tight_layout()
    plt.savefig('temporal_filters_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    


  
def main():
    print("Motion detection using temporal derivative")
    frames_path = "C:/Users/abhis/Documents/5639/EnterExitCrossingPaths2cor/EnterExitCrossingPaths2cor"  
    

    frames = load_video_frames(frames_path, max_frames=None)

    test_frame_indices = [50, 100, 150, 200, 250]
    for idx in test_frame_indices:
         print(f"\nframe {idx}")
         result = detect_motion(frames, idx, temporal_method='simple')
         print(f"  motion detected: {100*np.mean(result['mask']):.2f}%")

    #Run temporal filters experiment 
    for idx in test_frame_indices:
        experiment_temporal_filters(frames, idx)
    
    #middle frame for basic outline demonstration
    demo_frame_idx = len(frames) // 2
    
    #2 Apply temporal derivative
    print("\n2-Compute temporal derivative")
    result = detect_motion(frames, demo_frame_idx, temporal_method='simple')
    
    # 3/4 Threshold and combine with original
    print(f"3-Thresholding (threshold = {result['threshold']:.4f})")
    print(f"4-Combining mask with original frame")
    
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    
    axes[0].imshow(frames[demo_frame_idx], cmap='gray')
    axes[0].set_title('1. Original Frame\n(Grayscale)')
    axes[0].axis('off')
    
    axes[1].imshow(result['abs_derivative'], cmap='hot')
    axes[1].set_title('2. Temporal Derivative\n(Absolute Value)')
    axes[1].axis('off')
    
    axes[2].imshow(result['mask'], cmap='gray')
    axes[2].set_title('3. Binary Mask\n(Thresholded)')
    axes[2].axis('off')
    
    axes[3].imshow(result['overlay'])
    axes[3].set_title('4. Final Result\n(Mask + Original)')
    axes[3].axis('off')
    
    plt.tight_layout()
    plt.savefig('basic_outline_result.png', dpi=150, bbox_inches='tight')
    plt.show()

    print("\nResults summary:")
    print(f"  Estimated noise std: {result['noise_std']:.4f}")
    print(f"  Adaptive threshold: {result['threshold']:.4f}")
    print(f"  Detected motion pixels: {np.sum(result['mask'])} ({100*np.mean(result['mask']):.2f}%)")

if __name__ == "__main__":
    main()
