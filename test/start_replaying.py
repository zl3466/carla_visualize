import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


from nicetry import *
IM_WIDTH = 256
IM_HEIGHT = 256



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
        '-s', '--start',
        metavar='S',
        default=0.0,
        type=float,
        help='starting time (default: 0.0)')
    argparser.add_argument(
        '-d', '--duration',
        metavar='D',
        default=0.0,
        type=float,
        help='duration (default: 0.0)')
    argparser.add_argument(
        '-f', '--recorder-filename',
        metavar='F',
        default="test1.log",
        help='recorder filename (test1.log)')
    argparser.add_argument(
        '-c', '--camera',
        metavar='C',
        default=0,
        type=int,
        help='camera follows an actor (ex: 82)')
    argparser.add_argument(
        '-x', '--time-factor',
        metavar='X',
        default=1.0,
        type=float,
        help='time factor (default 1.0)')
    argparser.add_argument(
        '-i', '--ignore-hero',
        action='store_true',
        help='ignore hero vehicles')
    argparser.add_argument(
        '-v', '--target-vehicle',
        metavar='V',
        default=0,
        type=int,
        help='target vehicle id (default 0)')
    argparser.add_argument(
        '-t', '--typical-sensors',
        metavar='T',
        default=False,
        type=bool,
        help='spawn typical sensors (default True)')
    argparser.add_argument(
        '--spawn-sensors',
        action='store_true',
        help='spawn sensors in the replayed world')
    # store dir
    argparser.add_argument(
        '--save-dir',
        type=str,
        default="/home/allenzj/Carla_Data",
        help='save directory for raw data')
    args = argparser.parse_args()
    try:
        actor_list = []
        sensor_list = []
        sensor_queue = Queue()
        lidar_queue = Queue()
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        # set the time factor for the replayer
        client.set_replayer_time_factor(args.time_factor)

        # set to ignore the hero vehicles or not
        client.set_replayer_ignore_hero(args.ignore_hero)

        # Show the most important events in the recording.
        # print(client.show_recorder_file_info(args.recorder_filename, False))

        # replay the session
        print(client.replay_file(args.recorder_filename, args.start, args.duration, args.camera, args.spawn_sensors))
        world = client.get_world()

        # We set CARLA syncronous mode
        original_settings = world.get_settings()
        settings = world.get_settings()
        settings.fixed_delta_seconds = 0.05
        settings.synchronous_mode = True
        world.apply_settings(settings)

        spectator = world.get_spectator()

        blueprint_library = world.get_blueprint_library()

        # ------------------------------------------------------------------------
        # wait for actors to spawn properly
        if args.typical_sensors:
            while len(world.get_actors().filter('vehicle.*')) == 0:
                world.tick()
                print('waiting')
            # print(world.get_actors().filter('vehicle.*'))

            vehicle = world.get_actor(args.target_vehicle)
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
            lidar_bp.set_attribute("rotation_frequency", str(int(1 / settings.fixed_delta_seconds)))
            # lidar_bp.set_attribute("rotation_frequency", str(int(1 / 0.05)))

            # lidar location
            lidar_spawn_point1 = carla.Transform(carla.Location(z=2))

            # spawn lidar
            lidar_01 = world.spawn_actor(lidar_bp, lidar_spawn_point1, attach_to=vehicle)

            # lidar listen() & append sensor_list
            lidar_01.listen(lambda data: recursive_listen(data, sensor_queue, "lidar_01"))
            sensor_list.append(lidar_01)

            # -----------------------------------semantic lidar for 3d mapping-----------------------------------------
            NUM_SENSORS = 5
            views = np.arange(NUM_SENSORS)
            s_lidars = []
            for i in range(NUM_SENSORS):
                # create dir to save raw data
                # only velocity need extra code, all others used in gen_points to create 3d scene
                if not os.path.exists(args.save_dir + "/velocities" + str(i)):
                    os.mkdir(args.save_dir + "/velocities" + str(i))
                if not os.path.exists(args.save_dir + "/instances" + str(i)):
                    os.mkdir(args.save_dir + "/instances" + str(i))
                if not os.path.exists(args.save_dir + "/labels" + str(i)):
                    os.mkdir(args.save_dir + "/labels" + str(i))
                if not os.path.exists(args.save_dir + "/velodyne" + str(i)):
                    os.mkdir(args.save_dir + "/velodyne" + str(i))
                if not os.path.exists(args.save_dir + "/pose" + str(i)):
                    os.mkdir(args.save_dir + "/pose" + str(i))
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

            # -----------------------------------Run Visualization---------------------------------------------
            frame = 0
            while True:
                spectator.set_transform(sensor_cam2.get_transform())
                world_snapshot = world.tick()
                world_frame = world.get_snapshot().frame
                print("\nWorld's frame: %d" % world_frame)
                point_list = o3d.geometry.PointCloud()

                try:
                    ego_pose = None
                    indicator = True
                    for _ in range(len(s_lidars)):
                        data, view = lidar_queue.get()
                        #TODO：
                        #save_raw_data(data,world,lidar_id,vehicle-id,save_dir,frame,view)
                        #TODO: 看看可不可以合并到一起
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

                except Empty:
                    print("inappropriate sensor data")
                frame += 1

    finally:

        # --------------

        # Destroy actors

        # --------------

        world.apply_settings(original_settings)
        for actor in actor_list:
            actor.destroy()
        for sensor in sensor_list:
            sensor.destroy()
        for s_lidar in s_lidars:
            s_lidar.destroy()
        print('\nNothing to be done.')


if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')

# python3 start_replaying.py -t True -v 126 -f sumo-carla-town5-20.log