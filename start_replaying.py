#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import glob
import os
import sys
import time

import cv2
import numpy as np
from queue import Queue, Empty
import copy

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import argparse
import cv2
from queue import Queue, Empty
import copy


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
    args = argparser.parse_args()
    try:
        actor_list = []
        sensor_list = []
        sensor_queue = Queue()
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
        # spawn sensors

        # wait for actors to spawn properly
        count = 30
        if args.typical_sensors:
            while len(world.get_actors().filter('vehicle.*')) == 0:
                world.tick()
                print('waiting')
            # while count > 0:
            #     if len(world.get_actors().filter('vehicle.*')) > 0:
            #         print(world.get_actors().filter('vehicle.*'))
            #         break
            #     else:
            #         world.tick()
            #         print('waiting')
            #         count -= 1
            # if count == 0 and len(world.get_actors()) == 0:
            #     raise ValueError('Actors not populated in time')

            print(world.get_actors().filter('vehicle.*'))
            
            vehicle = world.get_actor(args.target_vehicle)
            actor_list.append(vehicle)

            # cameras
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

            # lidar
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

            while True:
                world_snapshot = world.tick()
                world_frame = world.get_snapshot().frame
                print("\nWorld's frame: %d" % world_frame)
                # print(world.get_actors().filter('vehicle.*'))

                try:
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

    finally:

        # --------------

        # Destroy actors

        # --------------

        # if ego_vehicle is not None:
        #     if ego_cam is not None:
        #         ego_cam.stop()
        #         ego_cam.destroy()
        #     ego_vehicle.destroy()
        world.apply_settings(original_settings)
        for actor in actor_list:
            actor.destroy()
        for sensor in sensor_list:
            sensor.destroy()
        print("all done")

        print('\nNothing to be done.')


IM_WIDTH = 256
IM_HEIGHT = 256


def process_img(image):
    i = np.array(image.raw_data)
    # print(dir(image))
    i = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i2 = i[:, :, :3]
    # cv2.imshow('', i2)
    # cv2.waitKey(1)
    return i2


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



if __name__ == '__main__':

    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
