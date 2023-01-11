from PIL import Image
import argparse
import sys

import gym
import numpy as np
import pyglet
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv

from DuckieDetector import DuckieDetector
from DuckieDetectorML import DuckieDetectorML
from LaneDetector import LaneDetector

from random import *

class Simulator:

    env = None
    key_handler = None

    duckieDetector = None
    duckieDetectorML = None

    decidedAction = []

    stop = False

    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--env-name", default=None)
        parser.add_argument("--map-name", default="udem1")
        parser.add_argument("--distortion", default=False, action="store_true")
        parser.add_argument("--camera_rand", default=False, action="store_true")
        parser.add_argument("--draw-curve", action="store_true", help="draw the lane following curve")
        parser.add_argument("--draw-bbox", action="store_true", help="draw collision detection bounding boxes")
        parser.add_argument("--domain-rand", action="store_true", help="enable domain randomization")
        parser.add_argument("--dynamics_rand", action="store_true", help="enable dynamics randomization")
        parser.add_argument("--frame-skip", default=1, type=int, help="number of frames to skip")
        parser.add_argument("--seed", default=1, type=int, help="seed")
        args = parser.parse_args()

        self.duckieDetector = DuckieDetector()
        self.duckieDetectorML = DuckieDetectorML()
        self.laneDetector = LaneDetector()

        self.env = DuckietownEnv(
            seed=args.seed,
            map_name=args.map_name,
            draw_curve=args.draw_curve,
            draw_bbox=args.draw_bbox,
            domain_rand=args.domain_rand,
            frame_skip=args.frame_skip,
            distortion=args.distortion,
            camera_rand=args.camera_rand,
            dynamics_rand=args.dynamics_rand,
        )

        #self.env = gym.make(args.env_name)

        self.env.reset()
        self.env.render()

        self.key_handler = key.KeyStateHandler()
        self.env.unwrapped.window.push_handlers(self.key_handler)

        pyglet.clock.schedule_interval(self.update, 1.0 / self.env.unwrapped.frame_rate)

        self.decidedAction = np.array([0.0, 0.0])

        # Enter main event loop
        pyglet.app.run()

    def update(self, dt):
        """
        This function is called at every frame to handle
        movement/stepping and redrawing
        """
        wheel_distance = 0.102
        min_rad = 0.08

        action = np.array([0.0, 0.0])

        action = self.decidedAction

        #print(f"Action: {action}")

        if self.key_handler[key.UP]:
            action += np.array([0.44, 0.0])
        if self.key_handler[key.DOWN]:
            action -= np.array([0.44, 0])
        if self.key_handler[key.LEFT]:
            action += np.array([0, 1])
        if self.key_handler[key.RIGHT]:
            action -= np.array([0, 1])
        if self.key_handler[key.SPACE]:
            action = np.array([0, 0])

        if self.key_handler[key.BACKSPACE] or self.key_handler[key.SLASH]:
            self.env.reset()
            self.env.render()
        elif self.key_handler[key.PAGEUP]:
            self.env.unwrapped.cam_angle[0] = 0
        elif self.key_handler[key.ESCAPE]:
            self.env.close()
            sys.exit(0)

        """v1 = action[0]
        v2 = action[1]
        # Limit radius of curvature
        if v1 == 0 or abs(v2 / v1) > (min_rad + wheel_distance / 2.0) / (min_rad - wheel_distance / 2.0):
            # adjust velocities evenly such that condition is fulfilled
            delta_v = (v2 - v1) / 2 - wheel_distance / (4 * min_rad) * (v1 + v2)
            v1 += delta_v
            v2 -= delta_v

        action[0] = v1
        action[1] = v2"""

        # Speed boost
        if self.key_handler[key.LSHIFT]:
            action *= 1.5

        obs, reward, done, info = self.env.step(action)
        #print("step_count = %s, reward=%.3f" % (self.env.unwrapped.step_count, reward))

        frame = Image.fromarray(obs)
        #self.duckieDetector.detect(frame)
        duckie_distance = self.duckieDetectorML.detect(frame)
        left_line, right_line = self.laneDetector.detect(frame)

        print(f"duckie_distance: {duckie_distance}")

        #print(f"LeftLine: {left_line} RightLine: {right_line}")
        if right_line is None:
            self.decidedAction = np.array([0.0, -1.0])
        else:
            x1, y1, x2, y2 = right_line
            slope = (y2 - y1) / (x2 - x1)

            p1=np.array([x1,y1])
            p2=np.array([x2,y2])
            p3=np.array([320,480])

            d = np.abs(np.cross(p2-p1,p3-p1)/np.linalg.norm(p2-p1))
            #print(f"Distance to RightLine: {d}")

            if d < 200:
                self.decidedAction = np.array([0.0, 1.0])
            elif d > 300:
                self.decidedAction = np.array([0.0, -1.0])
            else:
                self.decidedAction = np.array([1.0, 1.0])

        if duckie_distance != None and duckie_distance < 250:
            print("STOP")
            self.stop = True

        if self.stop == True:
            self.decidedAction = np.array([0.0, 0.0])

        if self.key_handler[key.RETURN]:
            x = randint(1, 10000)
            im = Image.fromarray(obs)
            im.save("images/screen" + str(x) + ".png")

        if done:
            #print("done!")
            self.env.reset()
            self.env.render()

        self.env.render()
