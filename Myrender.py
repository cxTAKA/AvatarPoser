"""Examples of using pyrender for viewing and offscreen rendering.
"""
import glob
import math
import shutil
from tqdm import tqdm

import cv2
import pyglet
import matplotlib.pyplot as plt
import os
import numpy as np
import trimesh
import trimesh.transformations as tf
from pyrender import PerspectiveCamera, \
    DirectionalLight, SpotLight, PointLight, \
    MetallicRoughnessMaterial, \
    Primitive, Mesh, Node, Scene, \
    Viewer, OffscreenRenderer, RenderFlags

pyglet.options['shadow_window'] = False


def manually_render():
    folder_path = "C:\\Users\TAKA\Pictures\Poser\\test\\"

    # 获取文件夹中所有条目
    entries = os.listdir(folder_path)

    # 过滤出所有子文件夹并排序
    directories = sorted([entry for entry in entries if os.path.isdir(os.path.join(folder_path, entry))])

    # 打印子文件夹数量和名称
    print(f"样本数量: {len(directories)}")
    for video_dir in directories:
        sample_path = os.path.join(folder_path, video_dir)
        print(sample_path)
        render_video_from_obj(sample_path=sample_path, fps=60, resolution=[1200, 1200])



def render_video_from_obj(sample_path=None, fps=None, resolution=None):
    gt_video_filepath = os.path.join(sample_path, 'gt.avi')
    pred_video_filepath = os.path.join(sample_path, 'pred.avi')

    if not os.path.exists(gt_video_filepath):
        render_video(os.path.join(sample_path, 'gt'), gt_video_filepath, fps, resolution)

    if not os.path.exists(pred_video_filepath):
        render_video(os.path.join(sample_path, 'pred'), pred_video_filepath, fps, resolution)


def render_video(sample_path=None, target_file_path=None, fps=None, resolution=None):
    img_array = []
    scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))
    cam = PerspectiveCamera(yfov=(np.pi / 3.0))
    rad_x_extra = math.pi / 2  # 绕y轴额外旋转的角度
    rad_z_extra = math.pi / 4  # 绕z轴额外旋转的角度
    rotation_matrix_x_extra = tf.rotation_matrix(rad_x_extra, (1, 0, 0))  # 绕y轴旋转
    rotation_matrix_z_extra = tf.rotation_matrix(rad_z_extra, (0, 0, 1))  # 绕z轴旋转
    move_matrix = cam_pose = np.array([
        [1.0, 0.0, 0.0, 3],
        [0.0, 1.0, 0.0, -3],
        [0.0, 0.0, 1.0, 1],
        [0, 0.0, 0.0, 1.0],
    ])
    cam_pose = np.dot(move_matrix, np.dot(rotation_matrix_z_extra, rotation_matrix_x_extra))

    direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
    spot_l = SpotLight(color=np.ones(3), intensity=10.0,
                       innerConeAngle=np.pi / 16, outerConeAngle=np.pi / 6)

    checker_trimesh = trimesh.load("C:\\Users\TAKA\Pictures\Poser\checker.obj")
    checker_mesh = Mesh.from_trimesh(checker_trimesh)
    checker_scale = 0.1
    checker_pose = np.array([
        [checker_scale, 0.0, 0.0, -2],
        [0.0, checker_scale, 0.0, -2],
        [0.0, 0.0, checker_scale, 0],
        [0.0, 0.0, 0.0, 1],
    ])

    scene.add(direc_l, pose=cam_pose)
    scene.add(spot_l, pose=cam_pose)
    scene.add(cam, pose=cam_pose)
    scene.add(checker_mesh, pose=checker_pose)

    r = OffscreenRenderer(viewport_width=resolution[0], viewport_height=resolution[1])

    files_list = os.listdir(sample_path)
    print(f"动画长度: {len(files_list)}")
    index = 0
    for id in tqdm(range(len(files_list))):
        index = id
        body_file = os.path.join(sample_path, str(index) + '.obj')
        if not os.path.exists(body_file):
            break
        body_trimesh = trimesh.load(body_file)
        body_mesh = Mesh.from_trimesh(body_trimesh)
        body_pose = np.array([
            [1.0, 0.0, 0.0, 0],
            [0.0, 1.0, 0.0, 0],
            [0.0, 0.0, 1.0, 0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        rad_y_extra = 0  # 绕y轴额外旋转的角度
        rad_z_extra = math.pi / 2  # 绕z轴额外旋转的角度
        rotation_matrix_y_extra = tf.rotation_matrix(rad_y_extra, (0, 1, 0))  # 绕y轴旋转
        rotation_matrix_z_extra = tf.rotation_matrix(rad_z_extra, (0, 0, 1))  # 绕z轴旋转
        body_pose = np.dot(rotation_matrix_z_extra, np.dot(rotation_matrix_y_extra, body_pose))
        body_mesh_node = scene.add(body_mesh, pose=body_pose)

        body_image, depth = r.render(scene)
        body_image = body_image.astype(np.uint8)
        body_image = cv2.cvtColor(body_image, cv2.COLOR_BGR2RGB)
        img_array.append(body_image)

        scene.remove_node(body_mesh_node)

    out = cv2.VideoWriter(target_file_path, cv2.VideoWriter_fourcc(*'DIVX'), fps, resolution)
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    print(f"渲染视频 完成 {target_file_path} end with {index}")
    # if os.path.exists(sample_path):
    #     # 删除文件夹及其中的所有文件
    #     shutil.rmtree(sample_path)
    #     print(f"文件夹 {sample_path} 已成功删除")
    # else:
    #     print(f"文件夹 {sample_path} 不存在")


def main():
    # ==============================================================================
    # Mesh creation
    # ==============================================================================

    # ------------------------------------------------------------------------------
    # Creating textured meshes from trimeshes
    # ------------------------------------------------------------------------------

    body_path = "C:\\Users\TAKA\Pictures\Poser\\test"
    # s body trimesh
    s_body_trimesh = trimesh.load(body_path + "\\0\pred\\0.obj")

    s_body_mesh = Mesh.from_trimesh(s_body_trimesh)
    ##  x  指向摄像机 向前   y 横向平行摄像机 向后  z纵向平行摄像机 向上
    s_body_pose = np.array([
        [1.0, 0.0, 0.0, 0],
        [0.0, 1.0, 0.0, 0],
        [0.0, 0.0, 1.0, 0],
        [0.0, 0.0, 0.0, 1.0],
    ])

    rad_y_extra = 0  # 绕y轴额外旋转的角度
    rad_z_extra = math.pi / 2  # 绕z轴额外旋转的角度
    rotation_matrix_y_extra = tf.rotation_matrix(rad_y_extra, (0, 1, 0))  # 绕y轴旋转
    rotation_matrix_z_extra = tf.rotation_matrix(rad_z_extra, (0, 0, 1))  # 绕z轴旋转
    s_body_pose = np.dot(rotation_matrix_z_extra, np.dot(rotation_matrix_y_extra, s_body_pose))

    # checker mesh
    checker_trimesh = trimesh.load("C:\\Users\TAKA\Pictures\Poser\checker.obj")
    checker_mesh = Mesh.from_trimesh(checker_trimesh)
    checker_scale = 0.1
    checker_pose = np.array([
        [checker_scale, 0.0, 0.0, -2],
        [0.0, checker_scale, 0.0, -2],
        [0.0, 0.0, checker_scale, 0],
        [0.0, 0.0, 0.0, 1],
    ])
    # Fuze trimesh
    fuze_trimesh = trimesh.load('pyrender/examples/models/fuze.obj')
    fuze_mesh = Mesh.from_trimesh(fuze_trimesh)

    # Drill trimesh
    drill_trimesh = trimesh.load('pyrender/examples/models/drill.obj')
    drill_mesh = Mesh.from_trimesh(drill_trimesh)
    drill_pose = np.eye(4)
    drill_pose[0, 3] = 0.1
    drill_pose[2, 3] = -np.min(drill_trimesh.vertices[:, 2])

    # Wood trimesh
    wood_trimesh = trimesh.load('pyrender/examples/models/wood.obj')
    wood_mesh = Mesh.from_trimesh(wood_trimesh)

    # Water bottle trimesh
    bottle_gltf = trimesh.load('pyrender/examples/models/WaterBottle.glb')
    bottle_trimesh = bottle_gltf.geometry[list(bottle_gltf.geometry.keys())[0]]
    bottle_mesh = Mesh.from_trimesh(bottle_trimesh)
    bottle_pose = np.array([
        [1.0, 0.0, 0.0, 0.1],
        [0.0, 0.0, -1.0, -0.16],
        [0.0, 1.0, 0.0, 0.13],
        [0.0, 0.0, 0.0, 1.0],
    ])

    # ------------------------------------------------------------------------------
    # Creating meshes with per-vertex colors
    # ------------------------------------------------------------------------------
    boxv_trimesh = trimesh.creation.box(extents=0.1 * np.ones(3))
    boxv_vertex_colors = np.random.uniform(size=(boxv_trimesh.vertices.shape))
    boxv_trimesh.visual.vertex_colors = boxv_vertex_colors
    boxv_mesh = Mesh.from_trimesh(boxv_trimesh, smooth=False)

    # ------------------------------------------------------------------------------
    # Creating meshes with per-face colors
    # ------------------------------------------------------------------------------
    boxf_trimesh = trimesh.creation.box(extents=0.1 * np.ones(3))
    boxf_face_colors = np.random.uniform(size=boxf_trimesh.faces.shape)
    boxf_trimesh.visual.face_colors = boxf_face_colors
    boxf_mesh = Mesh.from_trimesh(boxf_trimesh, smooth=False)

    # ------------------------------------------------------------------------------
    # Creating meshes from point clouds
    # ------------------------------------------------------------------------------
    points = trimesh.creation.icosphere(radius=0.05).vertices
    point_colors = np.random.uniform(size=points.shape)
    points_mesh = Mesh.from_points(points, colors=point_colors)

    # ==============================================================================
    # Light creation
    # ==============================================================================

    direc_l = DirectionalLight(color=np.ones(3), intensity=1.0)
    spot_l = SpotLight(color=np.ones(3), intensity=10.0,
                       innerConeAngle=np.pi / 16, outerConeAngle=np.pi / 6)
    point_l = PointLight(color=np.ones(3), intensity=10.0)

    # ==============================================================================
    # Camera creation
    # ==============================================================================

    cam = PerspectiveCamera(yfov=(np.pi / 3.0))
    rad_x_extra = math.pi / 2  # 绕y轴额外旋转的角度
    rad_z_extra = math.pi / 4  # 绕z轴额外旋转的角度
    rotation_matrix_x_extra = tf.rotation_matrix(rad_x_extra, (1, 0, 0))  # 绕y轴旋转
    rotation_matrix_z_extra = tf.rotation_matrix(rad_z_extra, (0, 0, 1))  # 绕z轴旋转
    move_matrix = cam_pose = np.array([
        [1.0, 0.0, 0.0, 3],
        [0.0, 1.0, 0.0, -3],
        [0.0, 0.0, 1.0, 1],
        [0, 0.0, 0.0, 1.0],
    ])
    cam_pose = np.dot(move_matrix, np.dot(rotation_matrix_z_extra, rotation_matrix_x_extra))
    print(cam_pose)
    # ==============================================================================
    # Scene creation
    # ==============================================================================

    scene = Scene(ambient_light=np.array([0.02, 0.02, 0.02, 1.0]))

    # ==============================================================================
    # Adding objects to the scene
    # ==============================================================================

    # ------------------------------------------------------------------------------
    # By manually creating nodes
    # ------------------------------------------------------------------------------
    fuze_node = Node(mesh=fuze_mesh, translation=np.array([0.1, 0.15, -np.min(fuze_trimesh.vertices[:, 2])]))
    scene.add_node(fuze_node)
    boxv_node = Node(mesh=boxv_mesh, translation=np.array([-0.1, 0.10, 0.05]))
    scene.add_node(boxv_node)
    boxf_node = Node(mesh=boxf_mesh, translation=np.array([-0.1, -0.10, 0.05]))
    scene.add_node(boxf_node)

    # ------------------------------------------------------------------------------
    # By using the add() utility function
    # ------------------------------------------------------------------------------
    drill_node = scene.add(drill_mesh, pose=drill_pose)
    bottle_node = scene.add(bottle_mesh, pose=bottle_pose)
    wood_node = scene.add(wood_mesh)
    direc_l_node = scene.add(direc_l, pose=cam_pose)
    spot_l_node = scene.add(spot_l, pose=cam_pose)
    scene.add(s_body_mesh, pose=s_body_pose)
    scene.add(checker_mesh, pose=checker_pose)
    # ==============================================================================
    # Using the viewer with a default camera
    # ==============================================================================

    # v = Viewer(scene, shadows=False)

    # ==============================================================================
    # Using the viewer with a pre-specified camera
    # ==============================================================================
    cam_node = scene.add(cam, pose=cam_pose)
    v = Viewer(scene, central_node=drill_node)

    # ==============================================================================
    # Rendering offscreen from that camera
    # ==============================================================================

    # r = OffscreenRenderer(viewport_width=640 * 2, viewport_height=480 * 2)
    # color, depth = r.render(scene)
    #
    # plt.figure()
    # plt.imshow(color)
    # plt.show()

    # ==============================================================================
    # Segmask rendering
    # ==============================================================================

    # nm = {node: 20 * (i + 1) for i, node in enumerate(scene.mesh_nodes)}
    # seg = r.render(scene, RenderFlags.SEG, nm)[0]
    # plt.figure()
    # plt.imshow(seg)
    # plt.show()

    # r.delete()


if __name__ == '__main__':
    manually_render()
