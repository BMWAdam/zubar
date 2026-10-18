import cv2
import numpy as np
from matplotlib import pyplot as plt

RADIUS_100UM = 5  
RADIUS_200UM = 10  

def create_circular_mask(rows, cols, radius):
    """Creates a centered 2D ideal circular floating-point mask."""
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist_from_center = np.sqrt((X - ccol)**2 + (Y - crow)**2)
    return (dist_from_center <= radius).astype(float)

def create_gaussian_mask(rows, cols, sigma):
    """Creates a centered 2D Gaussian floating-point mask."""
    crow, ccol = rows // 2, cols // 2
    Y, X = np.ogrid[:rows, :cols]
    dist_sq = (X - ccol)**2 + (Y - crow)**2
    return np.exp(-dist_sq / (2.0 * (sigma ** 2)))

def main():
    print("Running zubar-sim...")

    img_bgr = cv2.imread('clonka_zubata.jpeg')
    if img_bgr is None:
        print("Error: Could not read 'clonka_zubata.jpeg'. Check the file path.")
        return
    
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    rows, cols = img.shape

    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log1p(np.abs(fshift))

    mask_100 = create_circular_mask(rows, cols, RADIUS_100UM)
    mask_200 = create_circular_mask(rows, cols, RADIUS_200UM)

    fshift_lpf_100 = fshift * mask_100
    img_lpf_100 = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_lpf_100)))
    fshift_hpf_100 = fshift * (1.0 - mask_100)
    img_hpf_100 = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_hpf_100)))

    fshift_lpf_200 = fshift * mask_200
    img_lpf_200 = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_lpf_200)))
    fshift_hpf_200 = fshift * (1.0 - mask_200)
    img_hpf_200 = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_hpf_200)))

    gauss_mask_100 = create_gaussian_mask(rows, cols, RADIUS_100UM)
    gauss_mask_200 = create_gaussian_mask(rows, cols, RADIUS_200UM)

    fshift_gauss_lpf_100 = fshift * gauss_mask_100
    img_gauss_lpf_100 = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_gauss_lpf_100)))
    fshift_gauss_hpf_100 = fshift * (1.0 - gauss_mask_100)
    img_gauss_hpf_100 = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_gauss_hpf_100)))

    fshift_gauss_lpf_200 = fshift * gauss_mask_200
    img_gauss_lpf_200 = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_gauss_lpf_200)))
    fshift_gauss_hpf_200 = fshift * (1.0 - gauss_mask_200)
    img_gauss_hpf_200 = np.abs(np.fft.ifft2(np.fft.ifftshift(fshift_gauss_hpf_200)))

    plt.figure(figsize=(24, 7))

    plt.subplot(2, 7, 1), plt.imshow(img, cmap='gray')
    plt.title('Input'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 7, 2), plt.imshow(mask_100, cmap='gray')
    plt.title(f'Kruh 1 (r={RADIUS_100UM}px)'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 7, 3), plt.imshow(img_lpf_100, cmap='jet')
    plt.title('LPF 1'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 7, 4), plt.imshow(img_hpf_100, cmap='jet')
    plt.title('HPF 1'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 7, 5), plt.imshow(gauss_mask_100, cmap='gray')
    plt.title(f'Gauss 1 ($\sigma$={RADIUS_100UM})'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 7, 6), plt.imshow(img_gauss_lpf_100, cmap='jet')
    plt.title('G_LPF 1'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 7, 7), plt.imshow(img_gauss_hpf_100, cmap='jet')
    plt.title('G_HPF 1'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 7, 8), plt.imshow(magnitude_spectrum, cmap='magma')
    plt.title('FFT'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 7, 9), plt.imshow(mask_200, cmap='gray')
    plt.title(f'Kruh 2 (r={RADIUS_200UM}px)'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 7, 10), plt.imshow(img_lpf_200, cmap='jet')
    plt.title('LPF 2'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 7, 11), plt.imshow(img_hpf_200, cmap='jet')
    plt.title('HPF 2'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 7, 12), plt.imshow(gauss_mask_200, cmap='gray')
    plt.title(f'Gauss 2 ($\sigma$={RADIUS_200UM})'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 7, 13), plt.imshow(img_gauss_lpf_200, cmap='jet')
    plt.title('G_LPF 2'), plt.xticks([]), plt.yticks([])

    plt.subplot(2, 7, 14), plt.imshow(img_gauss_hpf_200, cmap='jet')
    plt.title('G_ HPF 2'), plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()