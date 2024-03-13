import cv2
import numpy as np
import sys
import os

def genGaussiankernel(width, sigma):
    x = np.arange(-int(width/2), int(width/2)+1, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, x)
    kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d

def draw_cross(im, fixations, color=(0, 255, 0), thickness=2, size=3):
    """
    Draw a cross at the location of each fixation.

    Parameters:
    - im: The image to draw on.
    - fixations: The list of fixation points.
    - color: The color of the cross (default is green).
    - thickness: The thickness of the lines (default is 2).
    - size: The size of the cross (default is 20).
    """
    for (x, y) in fixations:
        # Draw a line from top to bottom
        cv2.line(im, (x, y-size), (x, y+size), color, thickness)
        # Draw a line from left to right
        cv2.line(im, (x-size, y), (x+size, y), color, thickness)

    return im 

def pyramid(im, sigma=1, prNum=1):
    height_ori, width_ori, ch = im.shape
    G = im.copy()
    pyramids = [G]
    
    # gaussian blur
    Gaus_kernel2D = genGaussiankernel(0.5, sigma)
    
    # downsample
    for i in range(1, prNum):
        G = cv2.filter2D(G, -1, Gaus_kernel2D)
        G = cv2.resize(G, (int(width_ori/2**i), int(height_ori/2**i)))
        pyramids.append(G)
    
    # upsample
    for i in range(1, prNum):
        curr_im = pyramids[i]
        curr_im = cv2.resize(curr_im, (width_ori, height_ori))
        curr_im = cv2.filter2D(curr_im, -1, Gaus_kernel2D)
        pyramids[i] = curr_im

    return pyramids

def foveat_img(im, fixs, sigma):
    """
    im: input image
    fixs: sequences of fixations of form [(x1, y1), (x2, y2), ...]
    
    This function outputs the foveated image with given input image and fixations.
    """
    # sigma=0.1 # CHANGE INTENSITY HERE
    prNum = 6
    As = pyramid(im, sigma, prNum)
    height, width, _ = im.shape
    
    # compute coef
    p = 7.5
    k = 5
    alpha = 5

    x = np.arange(0, width, 1, dtype=np.float32)
    y = np.arange(0, height, 1, dtype=np.float32)
    x2d, y2d = np.meshgrid(x, y)
    theta = np.sqrt((x2d - fixs[0][0]) ** 2 + (y2d - fixs[0][1]) ** 2) / p
    for fix in fixs[1:]:
        theta = np.minimum(theta, np.sqrt((x2d - fix[0]) ** 2 + (y2d - fix[1]) ** 2) / p)
    R = alpha / (theta + alpha)
    
    Ts = []
    for i in range(1, prNum):
        Ts.append(np.exp(-((2 ** (i-3)) * R / sigma) ** 2 * k))
    Ts.append(np.zeros_like(theta))

    # omega
    omega = np.zeros(prNum)
    for i in range(1, prNum):
        omega[i-1] = np.sqrt(np.log(2)/k) / (2**(i-3)) * sigma

    omega[omega>1] = 1

    # layer index
    layer_ind = np.zeros_like(R)
    for i in range(1, prNum):
        ind = np.logical_and(R >= omega[i], R <= omega[i - 1])
        layer_ind[ind] = i

    # B
    Bs = []
    for i in range(1, prNum):
        Bs.append((0.5 - Ts[i]) / (Ts[i-1] - Ts[i] + 1e-5))

    # M
    Ms = np.zeros((prNum, R.shape[0], R.shape[1]))

    for i in range(prNum):
        ind = layer_ind == i
        if np.sum(ind) > 0:
            if i == 0:
                Ms[i][ind] = 1
            else:
                Ms[i][ind] = 1 - Bs[i-1][ind]

        ind = layer_ind - 1 == i
        if np.sum(ind) > 0:
            Ms[i][ind] = Bs[i][ind]

    print('num of full-res pixel', np.sum(Ms[0] == 1))
    # generate periphery image
    im_fov = np.zeros_like(As[0], dtype=np.float32)
    for M, A in zip(Ms, As):
        for i in range(3):
            im_fov[:, :, i] += np.multiply(M, A[:, :, i])

    im_fov = im_fov.astype(np.uint8)
    return im_fov


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Wrong format: python retina_transform.py [image_path] [xc] [yc] [sigma]")
        exit(-1)

    im_path = sys.argv[1]
    _, ext = os.path.splitext(im_path)

    if ext.lower() not in ['.jpg', '.jpeg', '.png']:
        print(f"Skipping non-image file: {im_path}")
    else:
        im = cv2.imread(im_path)
        if im is None:
            print(f"Failed to load image: {im_path}")
            sys.exit(1)

        sig = float(sys.argv[4])
        xc, yc = int(sys.argv[2]), int(sys.argv[3])

        im = foveat_img(im, [(xc, yc)], sig)
        #im = draw_cross(im, [(xc, yc)])
        
        destination = '/home/qw3971/clevr2/image_generation/new_retina/'

        transf_image_path = os.path.join(destination, os.path.basename(im_path))
        cv2.imwrite(transf_image_path, im)
        

        '''
        # Encode the image to JPEG format in memory
        is_success, buffer = cv2.imencode(".png", im)

        # Make sure the encoding was successful
        if not is_success:
            print(f"Error converting image to PNG", file=sys.stderr)
            sys.exit(1)

        # Write the JPEG data to stdout
        sys.stdout.buffer.write(buffer.tobytes())
        '''

#0.2, 100-200 each time. 