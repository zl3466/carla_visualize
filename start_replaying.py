#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
import glob
import os
import sys
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
        '--spawn-sensors',
        action='store_true',
        help='spawn sensors in the replayed world')
    args = argparser.parse_args()
    try:
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
        ego_vehicle = world.get_actor(1937)

        spectator = world.get_spectator()
        world_snapshot = world.wait_for_tick()
        spectator.set_transform(ego_vehicle.get_transform())

        cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        cam_location = carla.Location(2, 0, 1)
        cam_rotation = carla.Rotation(0, 180, 0)
        cam_transform = carla.Transform(cam_location, cam_rotation)
        cam_bp.set_attribute("image_size_x", str(1920))
        cam_bp.set_attribute("image_size_y", str(1080))
        cam_bp.set_attribute("fov", str(105))
        ego_cam = world.spawn_actor(cam_bp, cam_transform, attach_to=ego_vehicle,
                                    attachment_type=carla.AttachmentType.Rigid)
        ego_cam.listen(lambda image: process_img(image))

        while True:
            world_snapshot = world.wait_for_tick()



    finally:

        # --------------

        # Destroy actors

        # --------------

        if ego_vehicle is not None:
            if ego_cam is not None:
                ego_cam.stop()
                ego_cam.destroy()
            ego_vehicle.destroy()

        print('\nNothing to be done.')


IM_WIDTH = 256
IM_HEIGHT = 256


def process_img(image):
    i = np.array(image.raw_data)
    # print(dir(image))
    i = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i2 = i[:, :, :3]
    cv2.imshow(i2)
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
