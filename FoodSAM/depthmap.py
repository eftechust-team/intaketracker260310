import os
import cv2
import numpy as np
import torch
from PIL import Image
import depth_pro
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

import argparse

import open3d as o3d
import trimesh

parser = argparse.ArgumentParser(
    description=(
        "RUN depth pro model for monocular depth map generation and output depthmap matrix, heatmap, and focal length value"
    )
)

parser.add_argument(
    "--img_path",
    type=str,
    default=None,
    help="dir name of imgs.",
)
parser.add_argument(
    "--output",
    type=str,
    default='../data/M1',
    help=(
        "Path to the directory where results will be output. Output will be a folder "
    ),
)

args = parser.parse_args()


if __name__ == "__main__":
    # i = 'A1'
    # name_list = ['A1']
    # name_list = ['A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9',\
    #              'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10',\
    # name_list = ['M1-1', 'M2-1', 'M3', 'M4-1', 'M5', 'M8-1', 'M9']
    # for i in name_list:

    image_path = args.img_path
    # sam_checkpoint = "sam_vit_h_4b8939.pth"
    # device = 'cpu'

    # binary_image_path = f'./{i}.png'
    # binary_image_path = f'./data/{i}/enhance_mask.png'
    # binary_image = cv2.imread(binary_image_path, cv2.IMREAD_GRAYSCALE)
    # mask = (binary_image > 0).astype(np.uint8)

    # image = cv2.imread(image_path)
    # output_image = np.zeros_like(image)
    # output_image[mask > 0] = image[mask > 0]
    # cv2.imwrite(f"masked_image_{i}.png", output_image)
    # print(mask)

    print("Running Depth Pro...")

    model, transform = depth_pro.create_model_and_transforms()
    model.eval()

    image, _, f_px = depth_pro.load_rgb(image_path)
    image = transform(image)

    prediction = model.infer(image, f_px=f_px)
    depth = prediction["depth"].numpy()  # Depth in [m].
    np.save(f"{args.output}/depth.npy", depth)
    # loaded_arr = np.load(f"./data/{i}/depth.npy")
    # print(loaded_arr)
    focallength_px = prediction["focallength_px"]  # Focal length in pixels.
    np.save(f"{args.output}/focal_length.npy", focallength_px)
    # loaded_tensor = np.load(f"{args.output}/focal_length.npy")
    # print(loaded_tensor)

    c_x = depth.shape[1] / 2  # Principal point x
    c_y = depth.shape[0] / 2  # Principal point y

    min_depth = np.min(depth)
    max_depth = np.max(depth)
    normalized_depth = (depth - min_depth) / (max_depth - min_depth)

    # Scale to [0, 255]
    depth_image = ((1 - normalized_depth) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(depth_image, cv2.COLORMAP_JET)

    # Save the depth image
    cv2.imwrite(f'{args.output}/depth_map.png', heatmap)

    print('Image Finished!')