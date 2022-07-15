import glob
import os
import sys
import numpy as np
import cv2

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time
import random
from queue import Queue, Empty
import copy
import logging
import argparse
import open3d as o3d

IM_WIDTH = 256
IM_HEIGHT = 256
LABEL_COLORS = np.array([
    (0, 0, 0),  # None
    (70, 70, 70),  # Building
    (100, 40, 40),  # Fences
    (55, 90, 80),  # Other
    (255, 255, 0),  # Pedestrian
    (153, 153, 153),  # Pole
    (157, 234, 50),  # RoadLines
    (0, 0, 255),  # Road
    (255, 255, 255),  # Sidewalk
    (0, 155, 0),  # Vegetation
    (255, 0, 0),  # Vehicle
    (102, 102, 156),  # Wall
    (220, 220, 0),  # TrafficSign
    (70, 130, 180),  # Sky
    (0, 0, 0),  # Ground
    (150, 100, 100),  # Bridge
    (230, 150, 140),  # RailTrack
    (180, 165, 180),  # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160),  # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),  # Water
    (145, 170, 100),  # Terrain
]) / 255.0

def process_img(image):
    i = np.array(image.raw_data)
    # print(dir(image))
    i = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i2 = i[:, :, :3]
    return i2


def gen_points(point_cloud, world, lidar_id, vehicle_id, ego_pose, bool, frame,save_dir, view=0):
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
    non_ego = np.array(data['ObjIdx']) != vehicle_id
    points = np.array([data['x'], data['y'], data['z']]).T
    points = points[non_ego, :]

    # Add noise (2 centimeters)
    points += np.random.uniform(-0.02, 0.02, size=points.shape)
    pc = points.reshape(-1, 3)

    labels = np.array(data['ObjTag'])[non_ego]
    actor_list = world.get_actors()
    lidar_loc = actor_list.find(lidar_id).get_transform()
    to_world = np.array(lidar_loc.get_matrix())

    if not bool:
        to_ego = np.linalg.inv(ego_pose)
        to_ego = np.matmul(to_ego, to_world)
    else:
        to_ego = np.linalg.inv(to_world)
        to_ego = np.matmul(to_ego, to_world)

    pc = np.dot(to_ego[:3, :3], pc.T).T + to_ego[:3, 3]
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(pc)
    point_list.colors = o3d.utility.Vector3dVector(LABEL_COLORS[labels])

    # -----------------------------------Save Raw Data---------------------------------------------
    instances = np.array(data['ObjIdx'])[non_ego]
    np.save(save_dir + "/instances" + str(view) + "/" + str(frame), instances)
    np.save(save_dir + "/labels" + str(view) + "/" + str(frame), labels)
    np.save(save_dir + "/velodyne" + str(view) + "/" + str(frame), points)
    np.save(save_dir + "/pose" + str(view) + "/" + str(frame), to_world)
    # extra calculated to save velocity (from Umi github CarlaUtils.py)
    velocities=[]
    to_ego = np.array(lidar_loc.get_inverse_matrix())[:3, :3]
    tags = np.unique(np.array(data['ObjIdx']))
    for id in tags:
        if id ==0: continue
        actor=actor_list.find(int(id))
        actor_vel=actor.get_velocity()
        actor_vel=np.asarray([actor_vel.x,actor_vel.y,actor_vel.z])
        vel=np.matmul(to_ego,actor_vel)
        velocities.append([id,vel[0],vel[1],vel[2]])
    np.save(save_dir + "/velocities" + str(view) + "/" + str(frame), velocities)
    if bool:
        return to_world, point_list
    return ego_pose, point_list


# Semantic lidar
def generate_lidar_bp(world, delta):
    lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    lidar_bp.set_attribute('upper_fov', str(2))
    lidar_bp.set_attribute('lower_fov', str(-25))
    lidar_bp.set_attribute('channels', str(64.0))
    lidar_bp.set_attribute('range', str(50))
    lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
    lidar_bp.set_attribute('points_per_second', str(200000))
    return lidar_bp


def lidar_to_bev(lidar, min_x=-24, max_x=24, min_y=-16, max_y=16, pixels_per_meter=4, hist_max_per_pixel=10):
    xbins = np.linspace(min_x, max_x + 1, (max_x - min_x) * pixels_per_meter + 1)
    ybins = np.linspace(min_y, max_y + 1, (max_y - min_y) * pixels_per_meter + 1)
    # Compute histogram of x and y coordinates of points.
    hist = np.histogramdd(lidar[..., :2], bins=(xbins, ybins))[0]
    # Clip histogram
    hist[hist > hist_max_per_pixel] = hist_max_per_pixel
    # Normalize histogram by the maximum number of points in a bin we care about.
    overhead_splat = hist / hist_max_per_pixel * 255.
    # Return splat in X x Y orientation, with X parallel to car axis, Y perp, both parallel to ground.
    return overhead_splat[::-1, :]


def merge_visualize_data(rgb, lidar):
    canvas = np.array(rgb[..., ::-1])

    if lidar is not None:
        lidar_viz = lidar_to_bev(lidar).astype(np.uint8)
        lidar_viz = cv2.cvtColor(lidar_viz, cv2.COLOR_GRAY2RGB)
        canvas = np.concatenate([canvas, cv2.resize(lidar_viz.astype(np.uint8), (canvas.shape[0], canvas.shape[0]))],
                                axis=1)
    return canvas


def process_lidar(data):
    i = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
    i = copy.deepcopy(i)
    i = np.reshape(i, (int(i.shape[0] / 4), 4))
    return i


def recursive_listen(sensor_data, sensor_queue, sensor_name):
    sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))


def main():
    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--tm_port',
        default=8000,
        type=int,
        help='Traffic Manager Port (default: 8000)')
    argparser.add_argument(
        '-t', '--town',
        metavar='T',
        default=5,
        type=int,
        help='Town map')
    argparser.add_argument(
        '-v', '--num-vehicle',
        metavar='V',
        default=500,
        type=int,
        help='Number of Vehicles')
        # store dir
    argparser.add_argument(
        '--save-dir',
        type=str,
        default="/home/allenzj/Carla_Data",
        help='save directory for raw data')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()

    try:
        actor_list = []
        sensor_list = []
        sensor_queue = Queue()
        lidar_queue = Queue()

        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA synchronous mode
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)
        spectator = world.get_spectator()
        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)

        blueprint_library = world.get_blueprint_library()
        bp = blueprint_library.filter("model3")[0]
        # print(bp)

        # spawn car
        spawn_point = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, spawn_point)

        vehicle.apply_control(carla.VehicleControl(throttle=1, steer=0))
        actor_list.append(vehicle)

        # ------------------------------------------------------------------------
        # spawn sensors
        # --------------------------------------------RGB Cameras--------------------------------------------------
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", f"{IM_WIDTH}")
        cam_bp.set_attribute("image_size_y", f"{IM_HEIGHT}")
        cam_bp.set_attribute("fov", "110")

        # camera locations
        cam_spawn_point1 = carla.Transform(carla.Location(z=10), carla.Rotation(pitch=270, yaw=0, roll=0))
        cam_spawn_point2 = carla.Transform(carla.Location(x=-3, z=3), carla.Rotation(pitch=0, yaw=0, roll=0))

        # spawn cameras
        sensor_cam1 = world.spawn_actor(cam_bp, cam_spawn_point1, attach_to=vehicle)
        sensor_cam2 = world.spawn_actor(cam_bp, cam_spawn_point2, attach_to=vehicle)

        # camera listen() & append sensor_list
        sensor_cam1.listen(lambda data: recursive_listen(data, sensor_queue, "rgb_top"))
        sensor_list.append(sensor_cam1)
        sensor_cam2.listen(lambda data: recursive_listen(data, sensor_queue, "rgb_back"))
        sensor_list.append(sensor_cam2)

        # -----------------------------------------------lidar-------------------------------------------------------
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", "64")
        lidar_bp.set_attribute("points_per_second", "200000")
        lidar_bp.set_attribute("range", "32")
        # lidar_bp.set_attribute("rotation_frequency", str(int(1 / settings.fixed_delta_seconds)))
        lidar_bp.set_attribute("rotation_frequency", str(int(1 / 0.05)))

        # lidar location
        lidar_spawn_point1 = carla.Transform(carla.Location(z=2))

        # spawn lidar
        lidar_01 = world.spawn_actor(lidar_bp, lidar_spawn_point1, attach_to=vehicle)

        # lidar listen() & append sensor_list
        lidar_01.listen(lambda data: recursive_listen(data, sensor_queue, "lidar_01"))
        sensor_list.append(lidar_01)

        # -----------------------------------semantic lidar for 3d mapping---------------------------------------------
        NUM_SENSORS = 5
        views = np.arange(NUM_SENSORS)
        s_lidars = []
        for i in range(NUM_SENSORS):
            if i == 0:  # Onboard sensor
                offsets = [-0.5, 0.0, 1.8]
            else:
                offsets = np.random.uniform([-20, -20, 1], [20, 20, 5], [3, ])

            lidar_bp = generate_lidar_bp(args, world, blueprint_library, 0.05)
            # # Location of lidar, fixed to vehicle
            lidar_transform = carla.Transform(carla.Location(x=offsets[0], y=offsets[1], z=offsets[2]))
            lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

            # Add callback
            s_lidars.append(lidar)
            s_lidars[i].listen(lambda data, view=views[i]: lidar_queue.put([data, view]))

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name='Segmented Scene',
            width=960,
            height=540,
            left=480,
            top=270)
        vis.get_render_option().background_color = [0.0, 0.0, 0.0]
        vis.get_render_option().point_size = 3

        # ----------------------------------------------Recording--------------------------------------------------
        # log_name = "trial2.log"
        # client.start_recorder(log_name, True)

        # -----------------------------------Run Visualization---------------------------------------------
        frame = 0
        while True:
            world.tick()
            world_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % world_frame)
            point_list = o3d.geometry.PointCloud()

            try:
                ego_pose = None
                indicator = True
                for _ in range(len(s_lidars)):
                    data, view = lidar_queue.get()
                    """ maybe not needed
                    # record time
                    timestamp = world.get_snapshot().timestamp
                    time_file = open(args.save_dir+ '/times' + str(view) + '.txt', 'a')
                    time_file.write(str(frame) + ", " + str(timestamp) + "\n")
                    time_file.close()
                    """
                    ego_pose, point_list_2 = gen_points(data, world, s_lidars[view].id, vehicle.id, ego_pose,
                                                        indicator,frame,args.save_dir,view=view)
                    point_list += point_list_2
                    indicator = False
                if frame == 0:
                    geometry = o3d.geometry.PointCloud(point_list)
                    vis.add_geometry(geometry)

                geometry.points = point_list.points
                geometry.colors = point_list.colors
                vis.update_geometry(geometry)
                for i in range(1):
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.005)

                rgbs = []
                lidars = []

                for i in range(0, len(sensor_list)):
                    s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
                    sensor_type = s_name.split('_')[0]
                    if sensor_type == "rgb":
                        rgbs.append(process_img(s_data))
                    elif sensor_type == "lidar":
                        lidars.append(process_lidar(s_data))

                rgb = np.concatenate(rgbs, axis=1)[..., :3]
                lidar = np.concatenate(lidars, axis=1)[..., :3]
                cv2.imshow("vizs", merge_visualize_data(rgb, lidar))
                cv2.waitKey(100)

                # print(world.get_actors().filter("sensor.*"))

                # rgb_file_name = "/home/zl3466/carla/Unreal/CarlaUE4/Saved/multi_sensor_log/rgb/" + str(world_frame) + ".png"
                # cv2.imwrite(rgb_file_name, rgb[..., ::-1])
                # lidar_file_name = "/home/zl3466/carla/Unreal/CarlaUE4/Saved/multi_sensor_log/lidar/" + str(world_frame) + ".npy"
                # np.save(lidar_file_name, lidar)

            except Empty:
                print("inappropriate sensor data")

            frame += 1

    finally:
        # client.stop_recorder()
        world.apply_settings(original_settings)
        for actor in actor_list:
            actor.destroy()
        for sensor in sensor_list:
            sensor.destroy()
        for s_lidar in s_lidars:
            s_lidar.destroy()
        print("all done")

if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
