import hashlib
import shutil
import os
import numpy as np
import tempfile
import zipfile
import logging
import sys
import json
from pathlib import Path
import typing
from typing import NamedTuple, Union, cast, Dict
from ..types import Dataset, DatasetFeature, EvaluationProtocol, Method, RenderOutput, Iterable
from ..types import camera_model_to_int, new_cameras, FrozenSet
from ._common import DatasetNotFoundError, get_default_viewer_transform, new_dataset
from plyfile import PlyData, PlyElement

BLENDER_SCENES = {"lego_people"}
BLENDER_SPLITS = {"train", "test"}

C0 = 0.28209479177387814
class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def SH2RGB(sh):
    return sh * C0 + 0.5
def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def load_blender_dataset(path: Union[Path, str], split: str, **kwargs):
    assert isinstance(path, (Path, str)), "path must be a pathlib.Path or str"
    path = Path(path)

    scene = "lego_people"
    if scene not in BLENDER_SCENES:
        raise DatasetNotFoundError(f"Scene {scene} not found in nerf_synthetic dataset. Supported scenes: {BLENDER_SCENES}.")
    for dsplit in BLENDER_SPLITS:
        if not (path / f"transforms_{dsplit}.json").exists():
            raise DatasetNotFoundError(f"Path {path} does not contain a blender dataset. Missing file: {path / f'transforms_{dsplit}.json'}")

    assert split in BLENDER_SPLITS, "split must be one of 'train' or 'test'"

    with (path / f"transforms_{split}.json").open("r", encoding="utf8") as fp:
        meta = json.load(fp)

    cams = []
    image_paths = []
    for _, frame in enumerate(meta["frames"]):
        fprefix = path / frame["file_path"]
        image_paths.append(str(fprefix) + ".png")
        cams.append(np.array(frame["transform_matrix"], dtype=np.float32))

    w = h = 800
    image_sizes = np.array([w, h], dtype=np.int32)[None].repeat(len(cams), axis=0)
    nears_fars = np.array([2, 6], dtype=np.float32)[None].repeat(len(cams), axis=0)
    fx = fy = 0.5 * w / np.tan(0.5 * float(meta["camera_angle_x"]))
    cx = cy = 0.5 * w
    intrinsics = np.array([fx, fy, cx, cy], dtype=np.float32)[None].repeat(len(cams), axis=0)
    c2w = np.stack(cams)[:, :3, :4]

    # Convert from OpenGL to OpenCV coordinate system
    c2w[..., 0:3, 1:3] *= -1

    viewer_transform, viewer_pose = get_default_viewer_transform(c2w, "object-centric")

    # generate random pointclouds
    ply_path = os.path.join(path, "points3d.ply")
    # if not os.path.exists(ply_path):
    # Since this data set has no colmap data, we start with random points
    num_pts = 150_000
    print(f"Generating random point cloud ({num_pts})...")

    # We create random points inside the bounds of the synthetic Blender scenes
    xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
    shs = np.random.random((num_pts, 3)) / 255.0
    # pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

    storePly(ply_path, xyz, SH2RGB(shs) * 255)
    # try:
    #     pcd = fetchPly(ply_path)
    # except:
    #     pcd = None


    points3D_xyz, points3D_rgb = (xyz).astype(np.float32), (SH2RGB(shs)*255).astype(np.uint8)

    # import pdb; pdb.set_trace()
    return new_dataset(
                cameras=new_cameras(
                    poses=c2w,
                    intrinsics=intrinsics,
                    camera_types=np.full(len(cams), camera_model_to_int("pinhole"), dtype=np.int32),
                    distortion_parameters=np.zeros((len(cams), 0), dtype=np.float32),
                    image_sizes=image_sizes,
                    nears_fars=nears_fars,
                ),
                image_paths_root=str(path),
                image_paths=image_paths,
                points3D_xyz=points3D_xyz,
                points3D_rgb=points3D_rgb,
                sampling_mask_paths=None,
                metadata={
                    "name": "blender",
                    "scene": scene,
                    "color_space": "srgb",
                    "type": "object-centric",
                    "evaluation_protocol": "nerf",
                    "expected_scene_scale": 4,
                    "viewer_transform": viewer_transform,
                    "viewer_initial_pose": viewer_pose,
                    "background_color": np.array([255, 255, 255], dtype=np.uint8),
                },
            )


def download_blender_dataset(path: str, output: Path):
    if path == "blender":
        extract_prefix = "nerf_synthetic/"
    elif path.startswith("blender/") and len(path) > len("blender/"):
        scene_name = path[len("blender/") :]
        if scene_name not in BLENDER_SCENES:
            raise DatasetNotFoundError(f"Scene {scene_name} not found in nerf_synthetic dataset. Supported scenes: {BLENDER_SCENES}.")
        extract_prefix = f"nerf_synthetic/{scene_name}/"
    else:
        raise DatasetNotFoundError(f"Dataset path must be equal to 'blender' or must start with 'blender/'. It was {path}")

    # https://drive.google.com/uc?id=18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG
    blender_file_id = "18JxhpWD-4ZmuFKLzKlAw-w5PpzZxXOcG"
    file_sha256 = "f01fd1b4ab045b0d453917346f26f898657bb5bec4834b95fdad1f361826e45e"
    try:
        import gdown
    except ImportError:
        logging.fatal("Please install gdown: pip install gdown")
        sys.exit(2)

    url = f"https://drive.google.com/uc?id={blender_file_id}"
    output_tmp = str(output) + ".tmp"
    if os.path.exists(output_tmp):
        shutil.rmtree(output_tmp)
    os.makedirs(output_tmp)
    has_member = False
    with tempfile.TemporaryDirectory() as tmpdir:
        gdown.download(url, output=tmpdir + "/blender_data.zip")

        # Verify hash
        with open(tmpdir + "/blender_data.zip", "rb") as f:
            hasher = hashlib.sha256()
            for blk in iter(lambda: f.read(4096), b""):
                hasher.update(blk)
            if hasher.hexdigest() != file_sha256:
                raise RuntimeError(f"Hash of {tmpdir + '/blender_data.zip'} does not match {file_sha256}")

        # Extract files
        logging.info("Blender dataset downloaded and verified")
        logging.info(f"Extracting blender dataset: {tmpdir + '/blender_data.zip'}")
        with zipfile.ZipFile(tmpdir + "/blender_data.zip", "r") as zip_ref:
            for member in zip_ref.infolist():
                if member.filename.startswith(extract_prefix) and len(member.filename) > len(extract_prefix):
                    member.filename = member.filename[len(extract_prefix) :]
                    zip_ref.extract(member, output_tmp)
                    has_member = True
    if not has_member:
        raise RuntimeError(f"Path {path} not found in nerf_synthetic dataset.")
    if os.path.exists(str(output)):
        shutil.rmtree(str(output))
    os.rename(str(output) + ".tmp", str(output))
    logging.info(f"Downloaded {path} to {output}")

def horizontal_half_dataset(dataset: Dataset, left: bool = True) -> Dataset:
    intrinsics = dataset["cameras"].intrinsics.copy()
    image_sizes = dataset["cameras"].image_sizes.copy()
    image_sizes[:, 0] //= 2
    if not left:
        intrinsics[:, 2] -= image_sizes[:, 0]
    def get_slice(img, w):
        if left:
            return img[:, :w]
        else:
            return img[:, w:]
    dataset = dataset.copy()
    dataset.update(cast(Dataset, dict(
        cameras=dataset["cameras"].replace(
            intrinsics=intrinsics,
            image_sizes=image_sizes),
        images=[get_slice(img, w) for img, w in zip(dataset["images"], image_sizes[:, 0])],
        sampling_masks=[get_slice(mask, w) for mask, w in zip(dataset["sampling_masks"], image_sizes[:, 0])] if dataset["sampling_masks"] is not None else None,
    )))
    return dataset

class NerfWEvaluationProtocol(EvaluationProtocol):
    def __init__(self):
        from wildgaussians.evaluation import compute_metrics
        self._compute_metrics = compute_metrics

    def get_name(self):
        return "nerfw"

    def render(self, method: Method, dataset: Dataset) -> Iterable[RenderOutput]:
        optimization_dataset = horizontal_half_dataset(dataset, left=True)
        optim_iterator = method.optimize_embeddings(optimization_dataset)
        if optim_iterator is None:
            # Method does not support optimization
            for pred in method.render(dataset["cameras"]):
                yield pred
            return

        for i, optim_result in enumerate(optim_iterator):
            # Render with the optimzied result
            for pred in method.render(dataset["cameras"][i:i+1], embeddings=[optim_result["embedding"]]):
                yield pred

    def evaluate(self, predictions: Iterable[RenderOutput], dataset: Dataset) -> Iterable[Dict[str, Union[float, int]]]:
        for i, prediction in enumerate(predictions):
            gt = dataset["images"][i]
            color = prediction["color"]

            background_color = dataset["metadata"].get("background_color", None)
            color_srgb = image_to_srgb(color, np.uint8, color_space="srgb", background_color=background_color)
            gt_srgb = image_to_srgb(gt, np.uint8, color_space="srgb", background_color=background_color)
            w = gt_srgb.shape[1]
            metrics = self._compute_metrics(color_srgb[:, (w//2):], gt_srgb[:, (w//2):])
            yield metrics

    def accumulate_metrics(self, metrics: Iterable[Dict[str, Union[float, int]]]) -> Dict[str, Union[float, int]]:
        acc = {}
        for i, data in enumerate(metrics):
            for k, v in data.items():
                acc[k] = (acc.get(k, 0) * i + v) / (i + 1)
        return acc