import math as m
import random as r
from hw2_geometry import *


class Car():
    def __init__(self) -> None:
        self.diameter = 6
        self.angle_min = -90
        self.angle_max = 270
        self.wheel_min = -40
        self.wheel_max = 40
        self.xini_max = 4.5
        self.xini_min = -4.5

        self.reset()

    @property
    def radius(self):
        return self.diameter/2

    def reset(self):
        self.angle = 90
        self.wheel_angle = 0

        xini_range = (self.xini_max - self.xini_min - self.diameter)
        left_xpos = self.xini_min + self.radius
        self.xpos = r.random()*xini_range + left_xpos  # random x pos [-3, 3]
        self.ypos = 0

    def setWheelAngle(self, angle):
        self.wheel_angle = angle if self.wheel_min <= angle <= self.wheel_max else (
            self.wheel_min if angle <= self.wheel_min else self.wheel_max)

    def setPosition(self, newPosition: Point2D):
        self.xpos = newPosition.x
        self.ypos = newPosition.y

    def getPosition(self, point='center') -> Point2D:
        if point == 'right':
            right_angle = self.angle - 45
            right_point = Point2D(self.radius, 0).rorate(right_angle)
            return self.getPosition('center') + right_point

        elif point == 'left':
            left_angle = self.angle + 45
            left_point = Point2D(self.radius, 0).rorate(left_angle)
            return self.getPosition('center') + left_point

        elif point == 'front':
            front_point = Point2D(self.diameter, 0).rorate(self.angle)
            return self.getPosition('center') + front_point
        else:
            return Point2D(self.xpos, self.ypos)

    def getWheelPosPoint(self):
        wx = m.cos((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius+self.xpos
        wy = m.sin((-self.wheel_angle+self.angle)/180*m.pi) * \
            self.radius+self.ypos
        return Point2D(wx, wy)

    def setAngle(self, new_angle):
        new_angle %= 360
        if new_angle > self.angle_max:
            new_angle -= self.angle_max - self.angle_min
        self.angle = new_angle

    def tick(self):
        car_angle = self.angle/180*m.pi
        wheel_angle = self.wheel_angle/180*m.pi
        new_x = self.xpos + m.cos(car_angle+wheel_angle) + \
            m.sin(wheel_angle)*m.cos(car_angle)

        new_y = self.ypos + m.sin(car_angle+wheel_angle) - \
            m.sin(wheel_angle)*m.cos(car_angle)

        # seem as a car
        new_angle = (car_angle - m.asin(2*m.sin(wheel_angle) /
                     (self.diameter*1.5)))*180/m.pi

        # seem as a circle
        # new_angle = (car_angle - m.asin(2*m.sin(wheel_angle) /
        #              (self.radius)))*180/m.pi

        # new_angle %= 360
        # if new_angle > self.angle_max:
        #     new_angle -= self.angle_max - self.angle_min

        self.xpos = new_x
        self.ypos = new_y
        self.setAngle(new_angle)


class Playground():
    def __init__(self):
        # for rander play ground
        self.render_func = None

        # to handle history path
        self.history_list = []
        self.history_car_init_pos = None
        self.store_history_path = False
        self.history_filename = 'playground_path_history.txt'

        # read path lines
        self.path_line_filename = "軌道座標點.txt"
        self.readPathLines()
        self.decorate_lines = [
            Line2D(-6, 0, 6, 0),  # start line
            Line2D(0, 0, 0, -3),  # middle line
        ]

        self.car = Car()
        self.reset()

    def setDefaultLine(self):
        print('use default lines')
        # default lines
        self.destination_line = Line2D(18, 40, 30, 37)

        self.lines = [
            Line2D(-6, -3, 6, -3),
            Line2D(6, -3, 6, 10),
            Line2D(6, 10, 30, 10),
            Line2D(30, 10, 30, 50),
            Line2D(18, 50, 30, 50),
            Line2D(18, 22, 18, 50),
            Line2D(-6, 22, 18, 22),
            Line2D(-6, -3, -6, 22),
        ]

        self.car_init_pos = None
        self.car_init_angle = None

    def readPathLines(self):

        try:
            with open(self.path_line_filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # get init pos and angle
                pos_angle = [float(v) for v in lines[0].split(',')]
                self.car_init_pos = Point2D(*pos_angle[:2])
                self.car_init_angle = pos_angle[-1]

                # get destination line
                dp1 = Point2D(*[float(v) for v in lines[1].split(',')])
                dp2 = Point2D(*[float(v) for v in lines[2].split(',')])
                self.destination_line = Line2D(dp1, dp2)

                # get wall lines
                self.lines = []
                inip = Point2D(*[float(v) for v in lines[3].split(',')])
                for strp in lines[4:]:
                    p = Point2D(*[float(v) for v in strp.split(',')])
                    line = Line2D(inip, p)
                    inip = p
                    self.lines.append(line)
        except Exception:
            self.setDefaultLine()

    def setRenderFunc(self, func):
        self.render_func = func

    def render(self):
        if self.render_func is not None:
            self.render_func()

    def close(self):
        pass

    def sampleAction(self):
        return r.randint(0, self.n_actions-1)

    @property
    def n_actions(self):
        return (self.car.wheel_max - self.car.wheel_min + 1)

    @property
    def input_shape(self):
        return (len(self.state),)

    @ property
    def state(self):
        front_dist = - 1 if len(self.front_intersects) == 0 else self.car.getPosition(
            'front').distToPoint2D(self.front_intersects[0])
        right_dist = - 1 if len(self.right_intersects) == 0 else self.car.getPosition(
            'right').distToPoint2D(self.right_intersects[0])
        left_dist = - 1 if len(self.left_intersects) == 0 else self.car.getPosition(
            'left').distToPoint2D(self.left_intersects[0])

        return [front_dist, right_dist, left_dist]

    def checkRewardDoneIntersect(self):
        if self.done:
            return self.reward, self.done

        cpos = self.car.getPosition('center')     # center point of the car
        cfront_pos = self.car.getPosition('front')
        cright_pos = self.car.getPosition('right')
        cleft_pos = self.car.getPosition('left')
        radius = self.car.radius

        isAtDestination = cpos.isInRect(
            self.destination_line.p1, self.destination_line.p2
        )
        done = False if not isAtDestination else True
        reward = 0.01 if not isAtDestination else 2 + \
            (4-0.02*self.total_step)

        front_intersections, find_front_inter = [], True
        right_intersections, find_right_inter = [], True
        left_intersections, find_left_inter = [], True
        for wall in self.lines:  # chack every line in play ground
            dToLine = cpos.distToLine(wall)
            p1, p2 = wall.p1, wall.p2
            dp1, dp2 = (cpos-p1).length, (cpos-p2).length
            wall_len = wall.length

            # touch conditions
            p1_touch = (dp1 < radius)
            p2_touch = (dp2 < radius)
            body_touch = (
                dToLine < radius and (dp1 < wall_len and dp2 < wall_len)
            )
            front_touch, front_t, front_u = Line2D(
                cpos, cfront_pos).lineOverlap(wall)
            right_touch, right_t, right_u = Line2D(
                cpos, cright_pos).lineOverlap(wall)
            left_touch, left_t, left_u = Line2D(
                cpos, cleft_pos).lineOverlap(wall)

            if p1_touch or p2_touch or body_touch or front_touch:
                if not done:
                    done = True
                    reward = -1 - (3-0.015*self.total_step)

            # find all intersections
            if find_front_inter and front_u and 0 <= front_u <= 1:
                front_inter_point = (p2 - p1)*front_u+p1
                if front_t:
                    if front_t > 1:  # select only point in front of the car
                        front_intersections.append(front_inter_point)
                    elif front_touch:  # if overlapped, don't select any point
                        front_intersections = []
                        find_front_inter = False

            if find_right_inter and right_u and 0 <= right_u <= 1:
                right_inter_point = (p2 - p1)*right_u+p1
                if right_t:
                    if right_t > 1:  # select only point in front of the car
                        right_intersections.append(right_inter_point)
                    elif right_touch:  # if overlapped, don't select any point
                        right_intersections = []
                        find_right_inter = False

            if find_left_inter and left_u and 0 <= left_u <= 1:
                left_inter_point = (p2 - p1)*left_u+p1
                if left_t:
                    if left_t > 1:  # select only point in front of the car
                        left_intersections.append(left_inter_point)
                    elif left_touch:  # if overlapped, don't select any point
                        left_intersections = []
                        find_left_inter = False

        self.setIntersections(front_intersections,
                              left_intersections,
                              right_intersections)

        # results
        self.done = done
        self.reward = reward

        # store history when done
        if done and self.store_history_path:
            self.storeHistory()
        return reward, done

    def storeHistory(self, filename=None):
        if self.history_car_init_pos is None:
            return

        if filename is None:
            filename = self.history_filename
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(str(self.history_car_init_angle) + '\n')
            f.write(
                f'{self.history_car_init_pos.x}, {self.history_car_init_pos.y}' + '\n')

            for his in self.history_list:
                line = str(his)[1:-1] + '\n'
                f.write(line)

    def loadHistory(self, filename=None):
        if filename is None:
            filename = self.history_filename

        with open(filename, 'r', encoding='utf-8') as f:
            self.history_car_init_angle = float(f.readline())

            self.history_car_init_pos = Point2D(
                *[float(v) for v in f.readline().split(',')]
            )
            self.history_list = []
            for line in f.readlines():
                his = line.split(',')
                self.read_histroy_list.append([float(v) for v in his])

    def resetHistory(self):
        self.history_car_init_pos = self.car.getPosition()
        self.history_car_init_angle = self.car.angle
        self.history_list = []
        self.read_histroy_list = []

    def reset(self):
        self.done = False
        self.reward = 0
        self.total_step = 0
        self.car.reset()
        if self.car_init_pos and self.car_init_angle:
            self.setCarPosAndAngle(self.car_init_pos, self.car_init_angle)
        # history recording
        self.resetHistory()

        self.checkRewardDoneIntersect()
        return self.state

    def spawnCarAt(self, position: Point2D = None, angle=None):
        self.reset()
        self.setCarPosAndAngle(position, angle, True)

    def setCarPosAndAngle(self, position: Point2D = None, angle=None, reset_history=False):
        if position:
            self.car.setPosition(position)
        if angle:
            self.car.setAngle(angle)

        if reset_history:
            self.resetHistory()

        self.checkRewardDoneIntersect()

    def setIntersections(self, front_inters, left_inters, right_inters):
        self.front_intersects = sorted(front_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('front')))
        self.right_intersects = sorted(right_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('right')))
        self.left_intersects = sorted(left_inters, key=lambda p: p.distToPoint2D(
            self.car.getPosition('left')))

    def calWheelAngleFromAction(self, action):
        angle = self.car.wheel_min + \
            action*(self.car.wheel_max-self.car.wheel_min) / \
            (self.n_actions-1)
        return angle

    def step(self, action=None):
        if action:
            angle = self.calWheelAngleFromAction(action=action)
            self.car.setWheelAngle(angle)

        if not self.done:
            # history recording
            self.history_list.append([
                self.car.getPosition().x,
                self.car.getPosition().y,
                *self.state,
                self.car.wheel_angle])

            self.car.tick()

            # calcualte total step it take this round
            self.total_step += 1
            self.checkRewardDoneIntersect()
            return self.state, self.reward, self.done, []
        else:
            return self.state, 0, self.done, []
