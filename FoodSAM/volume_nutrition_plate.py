import numpy as np
import cv2
import csv
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
import argparse

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)

def save_pointcloud_image(points, output_path='pointcloud.png'):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')

    # Scatter plot
    ax.scatter(points[:,0], points[:,1], points[:,2], s=1, c='black', alpha=0.7)

    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Equal aspect ratio
    max_range = (points.max(axis=0) - points.min(axis=0)).max() / 2.0
    mid_x = (points[:,0].max() + points[:,0].min()) * 0.5
    mid_y = (points[:,1].max() + points[:,1].min()) * 0.5
    mid_z = (points[:,2].max() + points[:,2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    # Set background color and grid
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    ax.grid(False)
    ax.axis('off')

    # Save to file
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"Point cloud image saved to {output_path}")

# --- Nutrition dataset loader ---
def load_food_dataset_from_csv(csv_path):
    """
    Load food dataset CSV with columns:
    category_id, density, calories, protein, carbohydrates, fat
    
    Returns dict mapping food_label.lower() to nutrient dict.
    """
    dataset = {}
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            food_label = row['category_id'].strip().lower()
            try:
                density = float(row['density'])
            except:
                density = None
            try:
                calories = float(row['calories'])
            except:
                calories = 0
            try:
                protein = float(row['protein'])
            except:
                protein = 0
            try:
                carbs = float(row['carbohydrates'])
            except:
                carbs = 0
            try:
                fat = float(row['fat'])
            except:
                fat = 0

            dataset[food_label] = {
                'density_g_per_ml': density,
                'nutrients_per_g': {
                    'calories_kcal': calories,
                    'protein_g': protein,
                    'carbohydrates_g': carbs,
                    'fat_g': fat,
                }
            }
    return dataset

# --- Point cloud and volume estimation (from previous step) ---
def depth_mask_to_point_cloud(depth, mask, f, cx, cy):
    H, W = depth.shape
    u_coords, v_coords = np.meshgrid(np.arange(W), np.arange(H))

    valid = (mask == 1) & (depth > 0)
    u_valid = u_coords[valid]
    v_valid = v_coords[valid]
    d_valid = depth[valid]

    X = (u_valid - cx) * d_valid / f
    Y = (v_valid - cy) * d_valid / f
    Z = d_valid

    points = np.stack((X, Y, Z), axis=-1)
    return points

def pca_plane_estimation(points):
    pca = PCA(n_components=3)
    pca.fit(points)
    normal = pca.components_[-1]
    point_on_plane = np.mean(points, axis=0)
    d = -point_on_plane.dot(normal)
    return np.append(normal, d)

def align_plane_with_axis(plane_params, target_axis=np.array([0, 0, 1])):
    normal = plane_params[:3]
    normal = normal / np.linalg.norm(normal)
    target_axis = target_axis / np.linalg.norm(target_axis)

    v = np.cross(normal, target_axis)
    c = np.dot(normal, target_axis)
    s = np.linalg.norm(v)

    if s == 0:
        R = np.eye(3)
    else:
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (s ** 2))

    point_on_plane = -plane_params[3] * normal
    translation = -point_on_plane

    return translation, R

def pc_to_volume(points):
    if len(points) < 4:
        return 0.0, None
    try:
        hull = ConvexHull(points)
        return hull.volume, hull.simplices
    except:
        return 0.0, None

def estimate_volume(input_image, depth, mask, f, cx, cy, plate_diameter_prior=0.3, relax_param=0.01):
    # Detect plate ellipse in mask
    image = cv2.imread(input_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("[!] No contours detected in mask.")
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    if len(contour) < 5:
        print("[!] Not enough points for ellipse fitting.")
        return 0.0

    ellipse = cv2.fitEllipse(contour)
    print(ellipse)
    
    # ellipse = ((961.4000244140625, 1856.5565185546875), (3282.461181640625, 4413.37939453125), 120.22563171386719)
    ellipse = ((1480, 1750), (2720, 2430), 0)
    (cx_ellipse, cy_ellipse), (major_axis, minor_axis), angle = ellipse
    
    print(ellipse)
    theta = np.radians(angle)
    pt1 = (int(cx_ellipse + major_axis/2 * np.cos(theta)),
           int(cy_ellipse + major_axis/2 * np.sin(theta)))
    pt2 = (int(cx_ellipse - major_axis/2 * np.cos(theta)),
           int(cy_ellipse - major_axis/2 * np.sin(theta)))
    
    # import matplotlib.pyplot as plt
    
    img_copy = image.copy()
    cv2.ellipse(img_copy,
                (int(cx_ellipse), int(cy_ellipse)),
                (int(major_axis // 2), int(minor_axis // 2)),
                angle,
                0, 360,
                (0, 0, 255), 10)
    # plt.figure(figsize=(8, 8))
    # plt.imshow(img_copy)
    # plt.savefig('test.png')
    cv2.imwrite(f'{args.output}/circle_plate.png', img_copy)

    def pixel_to_3d(u,v):
        z = depth[v,u]
        if z <= 0:
            return None
        x = (u - cx) * z / f
        y = (v - cy) * z / f
        return np.array([x,y,z])

    pt1_3d = pixel_to_3d(pt1[0], pt1[1])
    pt2_3d = pixel_to_3d(pt2[0], pt2[1])

    if pt1_3d is None or pt2_3d is None:
        print("[!] Invalid 3D points from ellipse.")
        return 0.0

    plate_diameter = np.linalg.norm(pt1_3d - pt2_3d)
    if plate_diameter == 0:
        print("[!] Plate diameter zero.")
        return 0.0

    scaling = plate_diameter_prior / plate_diameter
    print(scaling)
    # scaling = 0.8
    depth_scaled = depth * scaling

    

    food_points = depth_mask_to_point_cloud(depth_scaled, mask, f, cx, cy)
    if len(food_points) < 4:
        print("[!] Not enough food points for volume estimation.")
        return 0.0
    save_pointcloud_image(food_points, f'{args.output}/food_pointcloud.png')

    plane_params = pca_plane_estimation(food_points)
    translation, rotation = align_plane_with_axis(plane_params, np.array([0,0,1]))
    food_points_transformed = (food_points + translation).dot(rotation.T)

    z_sorted_indices = np.argsort(food_points_transformed[:, 2])
    adjust_idx = int(len(food_points_transformed) * relax_param)
    height_shift = np.abs(food_points_transformed[z_sorted_indices[adjust_idx], 2])
    food_points_transformed[:, 2] -= height_shift

    above_plane_points = food_points_transformed[food_points_transformed[:, 2] > 0]
    volume_m3, _ = pc_to_volume(above_plane_points)

    return volume_m3

# --- Nutrition calculation ---
def calculate_volume_weight_nutrition(input_image, depth, mask, f, cx, cy, food_data, plate_diameter_prior=0.3):
    volume_m3 = estimate_volume(input_image, depth, mask, f, cx, cy, plate_diameter_prior=plate_diameter_prior)
    volume_ml = volume_m3 * 1e6  # m³ to ml

    if volume_ml == 0 or food_data is None:
        return 0.0, 0.0, {k: 0.0 for k in ['calories_kcal', 'protein_g', 'carbohydrates_g', 'fat_g']}

    density = food_data.get('density_g_per_ml')
    if density is None or density == 0:
        print("[!] Missing or zero density in food data.")
        return volume_ml, 0.0, {k: 0.0 for k in ['calories_kcal', 'protein_g', 'carbohydrates_g', 'fat_g']}

    weight_g = volume_ml * density
    nutrition = {nutrient: val_per_g * weight_g for nutrient, val_per_g in food_data['nutrients_per_g'].items()}

    return volume_ml, weight_g, nutrition

# --- Main processing for multiple masks ---
def calculate_for_multiple_masks_and_food_ids(input_image, depth, mask_id_pairs, f, cx, cy, dataset, plate_diameter_prior=0.3):
    results = []
    for mask_path, category_id in mask_id_pairs:
        food_data = dataset.get(str(category_id).lower())
        # print(food_data)
        if food_data is None:
            print(f"Warning: category_id '{category_id}' not found. Skipping mask '{mask_path}'")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"Warning: Mask file '{mask_path}' not found or unreadable. Skipping.")
            continue
        mask = (mask > 0).astype(np.uint8)

        print(mask.size)

        if depth.shape != mask.shape:
            print(f"Warning: Depth and mask shape mismatch for mask: {mask_path}. Skipping.")
            continue

        volume, weight, nutrition = calculate_volume_weight_nutrition(
            input_image, depth, mask, f, cx, cy, food_data, plate_diameter_prior=plate_diameter_prior)

        results.append({
            'category_id': category_id,
            'mask_path': mask_path,
            'volume_ml': volume,
            'weight_g': weight,
            'nutrition': nutrition,
        })
    return results

parser = argparse.ArgumentParser(
    description=(
        "merge mask for each food item"
    )
)

parser.add_argument(
    "--output",
    type=str,
    default='../data/M1',
    help=(
        "Path to the directory where results will be output. Output will be a folder "
    ),
)
parser.add_argument(
    "--plate_size",
    type=float,
    default=0.2,
    help=(
        "Path to the directory where results will be output. Output will be a folder "
    ),
)
parser.add_argument(
    "--food_id_list",
    type=str,
    default='0,1,2',
    help=(
        "Path to the directory where results will be output. Output will be a folder "
    ),
)

args = parser.parse_args()

# --- Example usage ---
if __name__ == "__main__":
    import os

    dataset_csv_path = 'food_full_data_revised.csv'
    dataset = load_food_dataset_from_csv(dataset_csv_path)

    food_id_list = args.food_id_list.split(',')
    print(food_id_list)
    input_image = f'{args.output}/input.jpg'

    focal_length_px = np.load(f"{args.output}/focal_length.npy").item()
    depth = np.load(f"{args.output}/depth.npy").astype(np.float32)

    cx = depth.shape[1] / 2
    cy = depth.shape[0] / 2

    mask_food_pairs = [(f'{args.output}/merged_mask/{id}.png', id) for id in food_id_list]

    results = calculate_for_multiple_masks_and_food_ids(
        input_image, depth, mask_food_pairs, focal_length_px, cx, cy, dataset, plate_diameter_prior=args.plate_size)

    for res in results:
        print(f"Mask: {res['mask_path']} - Category ID: {res['category_id']}")
        print(f"  Volume: {res['volume_ml']:.2f} ml")
        print(f"  Weight: {res['weight_g']:.2f} g")
        print(f"  Nutrition:")
        for k, v in res['nutrition'].items():
            print(f"    {k}: {v:.2f}")
        print()