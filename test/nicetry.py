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
    (255, 255, 255),  # None
    (70, 70, 70),  # Building
    (100, 40, 40),  # Fences
    (55, 90, 80),  # Other
    (220, 20, 60),  # Pedestrian
    (153, 153, 153),  # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),  # Vehicle
    (102, 102, 156),  # Wall
    (220, 220, 0),  # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),  # Ground
    (150, 100, 100),  # Bridge
    (230, 150, 140),  # RailTrack
    (180, 165, 180),  # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160),  # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),  # Water
    (145, 170, 100),  # Terrain
]) / 255.0

intersection_locations = {
    "Town05": [
        [31.80, 34.10, 29.15, 27.54, 29.15, 31.55, 31.80, 100.42, 151.52, -51.85, -47.7, -47.7, -123.95, -124.55,
         -124.55, -123.65, -123.65, -176.99, -188.95, -177.29],
        [-203.60, -145.40, -88.45, 1.06, 89.10, 137.33, 190.10, 62.90, 5.31, -88.45, 2.10, 88.30, -138.00, -88.45, 2.10,
         88.30, 146.60, -91.66, 2.10, 87.86]
    ],
    "Town03": [
        [83.50, 1.40, -77.97, -0.2],
        [-134.70, -135.94, 132.95, 132.50]
    ],
    "Town04": [
        [201.50, 201.50, 201.50, 260.00, 256.20, 312.60, 312.60, 311.90, 350.10],
        [-311.35, -248.85, -172.40, -248.80, -170.90, -248.85, -170.90, -120.30, -170.90]  # 1, 2, 5, 6, 7, 8, 9
    ]
}


def get_actor_blueprints(world, filter, generation):
    bps = world.get_blueprint_library().filter(filter)

    if generation.lower() == "all":
        return bps

    # If the filter returns only one bp, we assume that this one needed
    # and therefore, we ignore the generation
    if len(bps) == 1:
        return bps

    try:
        int_generation = int(generation)
        # Check if generation is in available generations
        if int_generation in [1, 2]:
            bps = [x for x in bps if int(x.get_attribute('generation')) == int_generation]
            return bps
        else:
            print("   Warning! Actor Generation is not valid. No actor will be spawned.")
            return []
    except:
        print("   Warning! Actor Generation is not valid. No actor will be spawned.")
        return []


def process_img(image):
    i = np.array(image.raw_data)
    # print(dir(image))
    i = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i2 = i[:, :, :3]
    return i2


# use to save raw data
def save_raw_data(point_cloud, world, lidar_id, vehicle_id, frame, save_dir, view=0, image=None):
    # check if the incoming lidar is semantic or non-semantic
    actor_list = world.get_actors()
    lidar_loc = actor_list.find(lidar_id).get_transform()
    to_world = np.array(lidar_loc.get_matrix())

    if isinstance(view, str):
        sensor_type = view.split('_')[0]
        if sensor_type == "rgb":
            # TODO save camera data
            # new added save rgb data as image
            # image.save_to_disk(save_dir + str(frame) + ".jpg")
            return
        elif sensor_type == "lidar":
            # non-semantic lidar
            # save velodyne, intensity, and pose
            data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
                ('x', np.float32), ('y', np.float32), ('z', np.float32),
                ('intensity', np.float32)]))
            points = np.array([data['x'], data['y'], data['z']]).T
            points += np.random.uniform(-0.02, 0.02, size=points.shape)
            intensity = np.array(data['intensity'])
            np.save(save_dir + "/velodyne_" + str(view) + "/" + str(frame), points)
            np.save(save_dir + "/intensity_" + str(view) + "/" + str(frame), intensity)
            np.save(save_dir + "/pose_" + str(view) + "/" + str(frame), to_world)
    else:
        # semantic lidar
        # save instance, label, velodyne, pose, and velocities
        data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))
        non_ego = np.array(data['ObjIdx']) != vehicle_id
        points = np.array([data['x'], data['y'], data['z']]).T
        points = points[non_ego, :]

        # Add noise (2 centimeters)
        points += np.random.uniform(-0.02, 0.02, size=points.shape)
        labels = np.array(data['ObjTag'])[non_ego]

        # -----------------------------------Save Raw Data---------------------------------------------
        instances = np.array(data['ObjIdx'])[non_ego]
        # object index in raw data, non_ego is 1 if vehicle_id not much, else 0.
        np.save(save_dir + "/instances_" + str(view) + "/" + str(frame), instances)
        # object labels in raw data, non_ego is 1 if vehicle_id not much, else 0.
        np.save(save_dir + "/labels_" + str(view) + "/" + str(frame), labels)
        # x,y,z in raw data, added noise, if ego take all data, else ignore the index 0
        np.save(save_dir + "/velodyne_" + str(view) + "/" + str(frame), points)
        # lidar location
        np.save(save_dir + "/pose_" + str(view) + "/" + str(frame), to_world)
        # extra calculated to save velocity (from Umi github CarlaUtils.py)

        velocities = []
        to_ego = np.array(lidar_loc.get_inverse_matrix())[:3, :3]
        tags = np.unique(np.array(data['ObjIdx']))
        for id in tags:
            if id == 0: continue
            actor = actor_list.find(int(id))
            actor_vel = actor.get_velocity()
            actor_vel = np.asarray([actor_vel.x, actor_vel.y, actor_vel.z])
            vel = np.matmul(to_ego, actor_vel)
            velocities.append([id, vel[0], vel[1], vel[2]])
        # velocity at x,y,z position
        np.save(save_dir + "/velocities_" + str(view) + "/" + str(frame), velocities)


def gen_points(point_cloud, world, lidar_id, vehicle_id, ego_pose, indicator):
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

    if not indicator:
        to_ego = np.linalg.inv(ego_pose)
        to_ego = np.matmul(to_ego, to_world)
    else:
        to_ego = np.linalg.inv(to_world)
        to_ego = np.matmul(to_ego, to_world)

    pc = np.dot(to_ego[:3, :3], pc.T).T + to_ego[:3, 3]
    point_list = o3d.geometry.PointCloud()
    point_list.points = o3d.utility.Vector3dVector(pc)
    point_list.colors = o3d.utility.Vector3dVector(LABEL_COLORS[labels])
    if indicator:
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


# def recursive_listen(sensor_data, sensor_queue, sensor_name):
#     sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))

# TODO: save camera and non-semantic lidar data during listen callback
#  'the_sensor' is the sensor object, the id of which substitutes the original 'sensor_list[sensor_index].id' for save_raw_data()
def recursive_listen(sensor_data, sensor_queue, sensor_name, vehicle, index, x_max, x_min, y_max, y_min, world,
                     frame, save_dir, the_sensor, vis):
    if vis:
        sensor_queue.put((sensor_data.frame, sensor_name, sensor_data))
    sensor_type = sensor_name.split('_')[0]
    if at_intersection(vehicle, index, x_max, x_min, y_max, y_min):
        # TODO: not sure if we need to distinguish between camera and lidar here
        #  if we do, this is the way, just FYI.
        #  we use this method to tell whether the incoming sensor is a camera or lidar in save_raw_data()
        if sensor_type == "rgb":
            save_raw_data(None, world, None, None, frame, save_dir, view=sensor_name, image=sensor_data)
        elif sensor_type == "lidar":
            save_raw_data(sensor_data, world, the_sensor.id, vehicle.id, frame, save_dir, view=sensor_name)


# TODO: save semantic lidar data during listen callback
def semantic_lidar_callback(data, view, lidar_queue, vehicle, index, x_max, x_min, y_max, y_min, world, s_lidars, frame,
                            s_lidar_save_dir, vis):
    if vis:
        lidar_queue.put([data, view])
    if at_intersection(vehicle, index, x_max, x_min, y_max, y_min):
        save_raw_data(data, world, s_lidars[view].id, vehicle.id, frame, s_lidar_save_dir, view=view, image=None)


def spawn_rgb_cam(world, cam_bp, size_x, size_y, fov, transform, vehicle=None):
    cam_bp.set_attribute("image_size_x", f"{size_x}")
    cam_bp.set_attribute("image_size_y", f"{size_y}")
    cam_bp.set_attribute("fov", str(fov))
    if vehicle is not None:
        cam = world.spawn_actor(cam_bp, transform, attach_to=vehicle)
    else:
        cam = world.spawn_actor(cam_bp, transform)
    return cam


def spawn_lidar(world, lidar_bp, channel, pps, l_range, freq, transform, vehicle=None):
    lidar_bp.set_attribute("channels", str(channel))
    lidar_bp.set_attribute("points_per_second", str(pps))
    lidar_bp.set_attribute("range", str(l_range))
    lidar_bp.set_attribute("rotation_frequency", str(freq))
    if vehicle is not None:
        lidar = world.spawn_actor(lidar_bp, transform, attach_to=vehicle)
    else:
        lidar = world.spawn_actor(lidar_bp, transform)
    return lidar


def at_intersection(actor, index, x_max, x_min, y_max, y_min):
    sensor_x = actor.get_transform().location.x
    sensor_y = actor.get_transform().location.y
    return x_max[index] >= sensor_x >= x_min[index] and y_max[index] >= sensor_y >= y_min[index]


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
        default=2,
        type=int,
        help='Number of Vehicles')
    argparser.add_argument(
        '--generationv',
        metavar='G',
        default='All',
        help='restrict to certain vehicle generation (values: "1","2","All" - default: "All")')
    argparser.add_argument(
        '-n', '--intersection-num',
        metavar='N',
        default=1,
        type=int,
        help='intersection number')
    argparser.add_argument(
        '--save-dir',
        type=str,
        default="/home/zl3466/Data",
        help='save directory for raw data')
    argparser.add_argument(
        '-i', '--target-vehicle',
        metavar='i',
        default=-1,
        type=int,
        help='target vehicle id (default 0)')
    args = argparser.parse_args()

    logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()

    try:
        sensor_list = []
        s_lidars = []
        sensor_queue = Queue()
        lidar_queue = Queue()
        frame = 0

        original_settings = world.get_settings()
        settings = world.get_settings()

        # We set CARLA synchronous mode
        settings.fixed_delta_seconds = 0.1
        settings.synchronous_mode = True
        world.apply_settings(settings)
        spectator = world.get_spectator()
        tm = client.get_trafficmanager(args.tm_port)
        tm.set_synchronous_mode(True)

        tm.set_global_distance_to_leading_vehicle(2.5)

        blueprint_library = world.get_blueprint_library()

        # ----------------------------------spawn vehicles autopilot-------------------------------------------------
        batch = []
        vehicles_list = []
        spawn_points = world.get_map().get_spawn_points()
        number_of_spawn_points = len(spawn_points)

        if args.num_vehicle < number_of_spawn_points:
            random.shuffle(spawn_points)
        elif args.num_vehicle > number_of_spawn_points:
            msg = 'requested %d vehicles, but could only find %d spawn points'
            logging.warning(msg, args.number_of_vehicles, number_of_spawn_points)
            args.num_vehicle = number_of_spawn_points

        blueprints = get_actor_blueprints(world, 'vehicle.*', args.generationv)
        blueprints = sorted(blueprints, key=lambda bp: bp.id)

        SpawnActor = carla.command.SpawnActor
        SetAutopilot = carla.command.SetAutopilot
        FutureActor = carla.command.FutureActor

        for n, transform in enumerate(spawn_points):
            if n >= args.num_vehicle:
                break
            blueprint = random.choice(blueprints)

            blueprint.set_attribute('role_name', 'autopilot')

            # spawn the cars and set their autopilot and light state all together
            batch.append(SpawnActor(blueprint, transform)
                         .then(SetAutopilot(FutureActor, True, tm.get_port())))

        for response in client.apply_batch_sync(batch, False):
            if response.error:
                logging.error(response.error)
            else:
                vehicles_list.append(response.actor_id)

        # get ego vehicle
        while len(world.get_actors().filter('vehicle.*')) < args.num_vehicle:
            world.tick()
            print('waiting')
        vehicle_list = world.get_actors().filter('vehicle.*')
        target_id = args.target_vehicle
        if target_id == -1:
            target_vehicle = vehicle_list[0]
        else:
            target_vehicle = world.get_actor(target_id)


        # -------------------------------------Identify Target Intersection-------------------------------------------
        Intersection_Index = args.intersection_num - 1
        # the location of 20 intersections in the global coordinate in Town
        Intersection_x = intersection_locations['Town0' + str(args.town)][0]
        Intersection_y = intersection_locations['Town0' + str(args.town)][1]
        the_x = Intersection_x[Intersection_Index]
        the_y = Intersection_y[Intersection_Index]

        Intersection_r = 20.00  # 18 meters
        Intersection_x_max = [i + Intersection_r for i in Intersection_x]
        Intersection_x_min = [i - Intersection_r for i in Intersection_x]
        Intersection_y_max = [i + Intersection_r for i in Intersection_y]
        Intersection_y_min = [i - Intersection_r for i in Intersection_y]

        # ------------------------------------------------------------------------
        # spawn sensors
        # --------------------------------------------RGB Cameras--------------------------------------------------
        cam_bp = blueprint_library.find("sensor.camera.rgb")
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")

        # multiple cars
        for vehicle in vehicle_list:
            if vehicle.id == target_vehicle.id:
                vis = True
            else:
                vis = False

            vehicle_save_dir = args.save_dir + "/" + str(vehicle.id)
            if not os.path.exists(vehicle_save_dir):
                os.makedirs(vehicle_save_dir)

            # -----------------------------------create directory for saving data-------------------------------------
            #TODO: create directories before using vehicle->Nonsemantic/Semantic->name->actual data

            cam_name = ["rgb_top", "rgb_back", "rgb_bev"]
            cam_save_dir = vehicle_save_dir + "/rgb"
            for name in cam_name:
                if not os.path.exists(cam_save_dir + "/" + name):
                    os.makedirs(cam_save_dir + "/" + name)

            # camera locations & rotations
            cam_spawn_point0 = carla.Transform(carla.Location(z=10), carla.Rotation(pitch=270, yaw=0, roll=0))
            cam_spawn_point1 = carla.Transform(carla.Location(x=-3, z=3), carla.Rotation(pitch=0, yaw=0, roll=0))
            cam_spawn_point2 = carla.Transform(carla.Location(x=the_x, y=the_y, z=40),
                                               carla.Rotation(pitch=270, yaw=0, roll=0))

            # spawn cameras
            sensor_cam0 = spawn_rgb_cam(world, cam_bp, IM_WIDTH, IM_HEIGHT, 110, cam_spawn_point0, vehicle)
            sensor_cam1 = spawn_rgb_cam(world, cam_bp, IM_WIDTH, IM_HEIGHT, 110, cam_spawn_point1, vehicle)
            sensor_cam2 = spawn_rgb_cam(world, cam_bp, IM_WIDTH, IM_HEIGHT, 110, cam_spawn_point2)

            # TODO: implement new recursive_listen() here; fill the extra attributes
            # camera listen()
            sensor_cam0.listen(
                lambda data, vehicle=vehicle, frame=frame, save_dir=cam_save_dir, sensor=sensor_cam0, the_vis=vis:
                recursive_listen(data, sensor_queue, "rgb_top", vehicle, args.town, Intersection_x_max,
                                 Intersection_x_min, Intersection_y_max, Intersection_y_min, world, frame,
                                 save_dir + "rgb_top", sensor, the_vis))
            sensor_cam1.listen(
                lambda data, vehicle=vehicle, frame=frame, save_dir=cam_save_dir, sensor=sensor_cam1, the_vis=vis:
                recursive_listen(data, sensor_queue, "rgb_back", vehicle, args.town, Intersection_x_max,
                                 Intersection_x_min, Intersection_y_max, Intersection_y_min, world, frame,
                                 save_dir + "rgb_back", sensor, the_vis))
            sensor_cam2.listen(
                lambda data, vehicle=vehicle, frame=frame, save_dir=cam_save_dir, sensor=sensor_cam2, the_vis=vis:
                recursive_listen(data, sensor_queue, "rgb_bev", vehicle, args.town, Intersection_x_max,
                                 Intersection_x_min, Intersection_y_max, Intersection_y_min, world, frame,
                                 save_dir + "rgb_bev", sensor, the_vis))

            """
            sensor_cam0.listen(lambda data: recursive_listen(data, sensor_queue, "rgb_top"))
            sensor_cam1.listen(lambda data: recursive_listen(data, sensor_queue, "rgb_back"))
            sensor_cam2.listen(lambda data: recursive_listen(data, sensor_queue, "rgb_bev"))
            """

            # append sensor_list
            sensor_list.append(sensor_cam0)
            sensor_list.append(sensor_cam1)
            sensor_list.append(sensor_cam2)

            # TODO: ??????????????????index ?????? ??????????????????list
            # -----------------------------------------------lidar------------------------------------------------------
            lidar_save_dir = vehicle_save_dir + "/NonSemantic"

            # lidar number (default 1)
            for i in range(1):
                view = 'lidar_' + str(i)
                if not os.path.exists(lidar_save_dir + "/velodyne_" + view):
                    os.makedirs(lidar_save_dir + "/velodyne_" + view)
                if not os.path.exists(lidar_save_dir + "/pose_" + view):
                    os.makedirs(lidar_save_dir + "/pose_" + view)
                if not os.path.exists(lidar_save_dir + "/intensity_" + view):
                    os.makedirs(lidar_save_dir + "/intensity_" + view)


            # lidar location & rotation
            lidar_spawn_point0 = carla.Transform(carla.Location(z=2))

            # spawn lidar
            lidar_0 = spawn_lidar(world, lidar_bp, 64, 200000, 32, int(1 / settings.fixed_delta_seconds),
                                  lidar_spawn_point0, vehicle)

            # TODO: implement new recursive_listen() here; fill the extra attributes
            # lidar listen()
            lidar_0.listen(
                lambda data, vehicle=vehicle, frame=frame, save_dir=lidar_save_dir, sensor=lidar_0, the_vis=vis:
                recursive_listen(data, sensor_queue, "lidar_0", vehicle, args.town, Intersection_x_max,
                                 Intersection_x_min, Intersection_y_max, Intersection_y_min, world, frame,
                                 save_dir + "lidar_0", sensor, the_vis))

            #lidar_0.listen(lambda data: recursive_listen(data, sensor_queue, "lidar_0"))

            # lidar append sensor_list
            sensor_list.append(lidar_0)

            # -----------------------------------semantic lidar for 3d mapping------------------------------------------
            s_lidar_save_dir = vehicle_save_dir + "/Semantic"

            NUM_SENSORS = 5
            views = np.arange(NUM_SENSORS)

            for i in range(NUM_SENSORS):
                # -----------------------------------save raw data-------------------------------------
                # only velocity need extra code, all others used in gen_points to create 3d scene
                if not os.path.exists(s_lidar_save_dir + "/velocities_" + str(i)):
                    os.makedirs(s_lidar_save_dir + "/velocities_" + str(i))
                if not os.path.exists(s_lidar_save_dir + "/labels_" + str(i)):
                    os.makedirs(s_lidar_save_dir + "/labels_" + str(i))
                if not os.path.exists(s_lidar_save_dir + "/instances_" + str(i)):
                    os.makedirs(s_lidar_save_dir + "/instances_" + str(i))
                if not os.path.exists(s_lidar_save_dir + "/velodyne_" + str(i)):
                    os.makedirs(s_lidar_save_dir + "/velodyne_" + str(i))
                if not os.path.exists(s_lidar_save_dir + "/pose_" + str(i)):
                    os.makedirs(s_lidar_save_dir + "/pose_" + str(i))

                # -----------------------------------spawn the semantic lidars-------------------------------------
                if i == 0:  # Onboard sensor
                    offsets = [-0.5, 0.0, 1.8]
                else:
                    offsets = np.random.uniform([-20, -20, 1], [20, 20, 5], [3, ])

                lidar_bp = generate_lidar_bp(world, 0.05)
                # # Location of lidar, fixed to vehicle
                lidar_transform = carla.Transform(carla.Location(x=offsets[0], y=offsets[1], z=offsets[2]))
                lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

                # Add callback
                s_lidars.append(lidar)
                #s_lidars[i].listen(lambda data, view=views[i]: lidar_queue.put([data, view]))

                # TODO: implement semantic_lidar_callback() here in listen()
                s_lidars[i].listen(
                    lambda data, view=views[i], vehicle=vehicle, frame=frame, save_dir=s_lidar_save_dir,
                           sensor=s_lidars[i], the_vis=vis:
                    semantic_lidar_callback(data, view, lidar_queue, vehicle, args.town, Intersection_x_max,
                                            Intersection_x_min, Intersection_y_max, Intersection_y_min, world,
                                            s_lidars, frame, save_dir, the_vis))

        # window for dense point cloud
        vis0 = o3d.visualization.Visualizer()
        vis0.create_window(
            window_name='Dense Segmented Scene',
            width=IM_WIDTH * 2,
            height=IM_HEIGHT * 2,
            left=480,
            top=270)
        vis0.get_render_option().background_color = [0.0, 0.0, 0.0]
        vis0.get_render_option().point_size = 3

        # window for sparse point cloud
        vis1 = o3d.visualization.Visualizer()
        vis1.create_window(
            window_name='Sparse Segmented Scene',
            width=IM_WIDTH * 2,
            height=IM_HEIGHT * 2,
            left=480 + IM_WIDTH * 2,
            top=270)
        vis1.get_render_option().background_color = [0.0, 0.0, 0.0]
        vis1.get_render_option().point_size = 3

        # ----------------------------------------------Recording--------------------------------------------------
        # log_name = "trial2.log"
        # client.start_recorder(log_name, True)

        # -----------------------------------Run Visualization---------------------------------------------

        # -----------------------------------save video of visualization---------------------------------------------
        cap = cv2.VideoCapture(0)
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out0 = cv2.VideoWriter('sensor_view.avi', fourcc, 10, (IM_WIDTH * len(sensor_list), IM_HEIGHT))
        out1 = cv2.VideoWriter('point_cloud_dense_vs_sparse.avi', fourcc, 10, (IM_WIDTH * 4, IM_HEIGHT * 2))
        # out2 = cv2.VideoWriter('point_cloud_sparse.avi', fourcc, 10, (IM_WIDTH*2, IM_HEIGHT*2))
        while True:
            world.tick()

            world_frame = world.get_snapshot().frame
            print("\nWorld's frame: %d" % world_frame)
            point_list = o3d.geometry.PointCloud()
            sparse_point_list = o3d.geometry.PointCloud()

            try:
                ego_pose = None
                indicator = True

                for i in range(len(s_lidars)):
                    data, view = lidar_queue.get()
                    ego_pose, point_list_2 = gen_points(data, world, s_lidars[view].id, vehicle.id, ego_pose, indicator)
                    point_list += point_list_2
                    indicator = False

                    # the sparse point cloud gets only the data from the first lidar
                    if i == 0:
                        sparse_point_list += point_list_2
                    # sparse_point_list = point_list

                if frame == 0:
                    geometry0 = o3d.geometry.PointCloud(point_list)
                    geometry1 = o3d.geometry.PointCloud(sparse_point_list)
                    vis0.add_geometry(geometry0)
                    vis1.add_geometry(geometry1)

                geometry0.points = point_list.points
                geometry0.colors = point_list.colors
                vis0.update_geometry(geometry0)
                for i in range(1):
                    vis0.poll_events()
                    vis0.update_renderer()
                    time.sleep(0.005)

                # get the image and scale and convert to uint8 type
                o3d_screenshot_dense = vis0.capture_screen_float_buffer()
                o3d_screenshot_dense = (255.0 * np.asarray(o3d_screenshot_dense)).astype(np.uint8)

                geometry1.points = sparse_point_list.points
                geometry1.colors = sparse_point_list.colors
                vis1.update_geometry(geometry1)
                for i in range(1):
                    vis1.poll_events()
                    vis1.update_renderer()
                    time.sleep(0.005)

                o3d_screenshot_sparse = vis1.capture_screen_float_buffer()
                o3d_screenshot_sparse = (255.0 * np.asarray(o3d_screenshot_sparse)).astype(np.uint8)

                canvas = np.concatenate((o3d_screenshot_dense, o3d_screenshot_sparse), axis=1)
                # cv2.imshow("Semantic Lidar Dense/Sparse Comparison", canvas)
                out1.write(canvas)

                rgbs = []
                lidars = []
                # print(sensor_queue.queue)

                for i in range(0, len(sensor_list)):
                    s_frame, s_name, s_data = sensor_queue.get(True, 1.0)
                    # print(s_frame, s_name)
                    sensor_type = s_name.split('_')[0]
                    if sensor_type == "rgb":
                        rgbs.append(process_img(s_data))
                    elif sensor_type == "lidar":
                        lidars.append(process_lidar(s_data))
                rgb = np.concatenate(rgbs, axis=1)[..., :3]
                lidar = np.concatenate(lidars, axis=1)[..., :3]
                cv2.imshow("vizs", merge_visualize_data(rgb, lidar))
                out0.write(merge_visualize_data(rgb, lidar))
                cv2.waitKey(100)



            except Empty:
                print("inappropriate sensor data")

            frame += 1

    finally:
        # client.stop_recorder()
        cap.release()
        out0.release()
        out1.release()
        # out2.release()
        world.apply_settings(original_settings)
        for vehicle in vehicle_list:
            vehicle.destroy()
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
