import time
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QBrush, QPainter, QPen, QColor, QResizeEvent
from PyQt5.QtCore import QPoint, QRect, Qt, QObject, pyqtSignal
import math as m
import typing
import sys
import random as r
import threading
# my modules
from hw2_utils import resource_path
from hw2_playground import Playground, Point2D
from hw2_playground_agent import train_save_agent, load_my_agent


main_ui_path = resource_path('./hw2_qtui/main.ui')
main_ui_class, main_window_class = uic.loadUiType(main_ui_path)


class PlayGroundCanvas():
    class CanvasEventHandler(QObject):
        playgroundChangeEvent = pyqtSignal()

    def __init__(self, parent=None, playground=None):
        # setting playground
        self.playground = playground if playground else Playground()

        def renderFunc():
            self.update()
            time.sleep(0.02)  # to slow down animation
        self.playground.setRenderFunc(renderFunc)
        self.eventHandler = self.CanvasEventHandler()

        # setting canvas
        self.c = QWidget(parent=parent)  # generate a widget to paint
        self.draw_support_lines = True

        # setting default parameters
        self.px_start = -1
        self.py_start = -1
        self.mouse_pos_x = 0
        self.mouse_pos_y = 0
        # setting display range
        self.physic_y_max = 60
        self.physic_y_min = -10
        self.physic_x_max = 45
        self.physic_x_min = -45

        # setting shape, geometry and event
        self._setCanvasBodyGeometry()
        self._setCanvasBodyEvent()
        self._setPhysicsGeometry()
        self.resetPlayground()

    def setDrawSupportLine(self, draw: bool):
        self.draw_support_lines = draw

    def update(self):
        self.c.update()

    def resetPlayground(self):
        self.playground.reset()

        self.eventHandler.playgroundChangeEvent.emit()
        self.update()

    def tick(self):
        self.playground.step()
        self.eventHandler.playgroundChangeEvent.emit()
        self.update()

    def _setPhysicsGeometry(self):
        _, _, canvas_width, canvas_height = self.c.geometry().getRect()
        self.y_unit = canvas_height/(self.physic_y_max - self.physic_y_min)
        self.x_unit = canvas_width/(self.physic_x_max - self.physic_x_min)
        self.x_orin = int(-self.physic_x_min * self.x_unit)
        self.y_orin = int(self.physic_y_max * self.y_unit)

    def _setCanvasBodyGeometry(self):
        self.c.setStyleSheet("background: black;")

    def _setCanvasBodyEvent(self):
        self.c.paintEvent = self._paintEvent
        self.c.mouseMoveEvent = self._mouseMoveEvent
        self.c.mousePressEvent = self._mousePressEvent
        self.c.mouseReleaseEvent = self._mouseReleaseEvent
        self.c.resizeEvent = self._resizeEvent

    def _resizeEvent(self, event: QResizeEvent):
        self._setPhysicsGeometry()
        self.c.update()

    def _transToPixelQPoint(self, point: Point2D):
        px, py = point.x, point.y
        x = px * self.x_unit
        y = -py * self.y_unit
        return QPoint(int(self.x_orin + x), int(self.y_orin + y))

    def _transToPixelLength(self, length, direction='x'):
        if direction == 'x':
            return length * self.x_unit
        else:
            return length * self.y_unit

    def _paintEvent(self, event):  # paint canvas event after update(clear)
        painter = QPainter()
        painter.begin(self.c)
        # draw background
        pen = QPen()
        pen.setWidth(4)
        # draw destination line
        dl = self.playground.destination_line
        p1, p2 = dl.p1, dl.p2
        startQp = self._transToPixelQPoint(p1)
        endQp = self._transToPixelQPoint(p2)
        x, y = startQp.x(), startQp.y()
        w, h = abs(endQp.x()-x), abs(endQp.y()-y)
        painter.setBrush(QBrush(QColor('red')))
        painter.drawRect(QRect(x, y, w, h))
        painter.setBrush(QBrush())

        # draw wall lines
        pen.setColor(QColor('blue'))
        painter.setPen(pen)
        for line in self.playground.lines:
            p1 = line.p1
            p2 = line.p2
            startQp = self._transToPixelQPoint(p1)
            endQp = self._transToPixelQPoint(p2)
            painter.drawLine(startQp, endQp)

        # draw decorate lines
        pen.setColor(QColor('red'))
        painter.setPen(pen)
        for line in self.playground.decorate_lines:
            p1, p2 = line.p1, line.p2
            startQp = self._transToPixelQPoint(p1)
            endQp = self._transToPixelQPoint(p2)
            painter.drawLine(startQp, endQp)

        # get car positions
        car_point = self.playground.car.getPosition()
        car_front_point = self.playground.car.getPosition('front')
        car_right_point = self.playground.car.getPosition('right')
        car_left_point = self.playground.car.getPosition('left')
        wheel_point = self.playground.car.getWheelPosPoint()
        carQp = self._transToPixelQPoint(car_point)
        car_frontQp = self._transToPixelQPoint(car_front_point)
        car_rightQp = self._transToPixelQPoint(car_right_point)
        car_leftQp = self._transToPixelQPoint(car_left_point)
        car_wheelQp = self._transToPixelQPoint(wheel_point)

        # draw car body
        pen.setColor(QColor('white'))
        pen.setWidth(4)
        painter.setPen(pen)
        rx = self._transToPixelLength(self.playground.car.diameter/2, 'x')
        ry = self._transToPixelLength(self.playground.car.diameter/2, 'y')
        painter.drawEllipse(carQp, rx, ry)

        # draw car support lines
        if self.draw_support_lines:
            pen.setColor(QColor('lightgreen'))
            pen.setWidth(2)
            pen.setStyle(Qt.DotLine)
            painter.setPen(pen)
            # min dist one
            front_points = self.playground.front_intersects[:1]
            # min dist one
            right_points = self.playground.right_intersects[:1]
            left_points = self.playground.left_intersects[:1]   # min dist one
            for p in front_points:
                interQp = self._transToPixelQPoint(p)
                painter.drawLine(car_frontQp, interQp)

            for p in right_points:
                interQp = self._transToPixelQPoint(p)
                painter.drawLine(car_rightQp, interQp)

            for p in left_points:
                interQp = self._transToPixelQPoint(p)
                painter.drawLine(car_leftQp, interQp)
            pen.setStyle(Qt.SolidLine)

        # draw car line
        pen.setColor(QColor('white'))
        pen.setWidth(3)
        painter.setPen(pen)
        painter.drawLine(carQp, car_frontQp)
        if self.draw_support_lines:
            painter.drawLine(carQp, car_leftQp)
            painter.drawLine(carQp, car_rightQp)

            # draw car wheel angle
            pen.setColor(QColor('yellow'))
            painter.setPen(pen)
            painter.drawLine(carQp, car_wheelQp)

        # draw mouse guide line
        if self.px_start >= 0 and self.py_start >= 0:
            pen.setColor(QColor('white'))
            pen.setWidth(5)
            painter.setPen(pen)
            painter.drawLine(self.px_start,
                             self.py_start,
                             self.mouse_pos_x,
                             self.mouse_pos_y)

        painter.end()

    def _mousePressEvent(self, event):
        self.px_start = event.pos().x()
        self.py_start = event.pos().y()

    def _mouseReleaseEvent(self, event):
        if self.px_start != self.mouse_pos_x or self.py_start != self.mouse_pos_y:
            # put car at any where
            self.playground.reset()
            px = self.px_start
            py = self.py_start
            fx = (px-self.x_orin)/self.x_unit
            fy = -(py-self.y_orin)/self.y_unit

            # calcualte angle between [1 0] and mouse line
            x_off = self.mouse_pos_x - self.px_start
            y_off = self.mouse_pos_y - self.py_start
            posMouse = Point2D(x_off, y_off)

            length = posMouse.length
            new_angle = m.acos(x_off / (length + 1e-10))*180/m.pi
            new_angle = new_angle if self.mouse_pos_y <= self.py_start else -new_angle
            # self.playground.setCarPosAndAngle(Point2D(fx, fy), new_angle)
            self.playground.spawnCarAt(Point2D(fx, fy), new_angle)
            self.px_start = -1
            self.py_start = -1

            self.eventHandler.playgroundChangeEvent.emit()
            self.update()
        else:
            self.px_start = -1
            self.py_start = -1

    def _mouseMoveEvent(self, event):
        self.mouse_pos_x = event.pos().x()
        self.mouse_pos_y = event.pos().y()

        self.update()


class MyMainWindow(main_window_class):
    def __init__(self, main_ui_class) -> None:
        super().__init__()
        self.is_using_thread = False
        self.is_auto_predict = False
        self.auto_run_thread = None
        self.playground = Playground()
        self.playground_agent = load_my_agent()

        self.main_ui = main_ui_class()
        self.main_ui.setupUi(self)

        self._setUpExtraWidgets()
        self._setUpUiEvent()

    def _setEndThreadRunning(self):
        self.is_using_thread = False
        self.auto_run_thread = None
        self.main_ui.control_panel_hlayout.setEnabled(True)
        self.main_ui.next_step_btn.setEnabled(True)
        self.main_ui.reset_btn.setEnabled(True)
        self.main_ui.auto_predict_cbox.setEnabled(True)

        self.main_ui.predict_btn.setEnabled(
            not self.main_ui.auto_predict_cbox.isChecked())
        self.main_ui.wheel_angle_slider.setEnabled(
            not self.main_ui.auto_predict_cbox.isChecked())

        self.canvas.resetPlayground()

    def _setStartThreadRunning(self):
        self.is_using_thread = True
        self.main_ui.control_panel_hlayout.setEnabled(False)
        self.main_ui.next_step_btn.setEnabled(False)
        self.main_ui.reset_btn.setEnabled(False)
        self.main_ui.predict_btn.setEnabled(False)
        self.main_ui.auto_predict_cbox.setEnabled(False)
        self.main_ui.wheel_angle_slider.setEnabled(False)

    def _trainAgentToEnd(self):
        if self.is_using_thread:
            return

        self._setStartThreadRunning()

        train_save_agent(self.playground, self.playground_agent)

        self._setEndThreadRunning()

    def _startAutoRunOneRound(self):
        if self.is_using_thread:
            return
        self._setStartThreadRunning()
        self.main_ui.auto_predict_cbox.setChecked(True)

        self.canvas.update()
        while not self.playground.done:
            action = self._agentPredictAction()
            self._playgroundNextStep(action)
        self.canvas.update()

        self._setEndThreadRunning()

    def _startTrainingAgent(self):
        if self.is_using_thread:
            return
        self.auto_run_thread = threading.Thread(
            target=self._startAutoRunOneRound)
        # self.train_thread = threading.Thread(target=self.trainAgentToEnd)
        self.auto_run_thread.start()

    def _agentPredictAction(self):
        return self.playground_agent.get_action(self.playground.state)

    def _setUpExtraWidgets(self):
        self.canvas = PlayGroundCanvas(playground=self.playground)
        self.main_ui.main_canvas_holder.addWidget(self.canvas.c)

        slider = self.main_ui.wheel_angle_slider
        slider.setMinimum(self.playground.car.wheel_min)
        slider.setMaximum(self.playground.car.wheel_max)

    def _playgroundNextStep(self, action=None):
        state, reward, done, info = self.playground.step(action)
        self.playground.render()  # with time proposed

        if self.is_auto_predict:
            self._setPredictAngle()

        return state, reward, done, info

    def _setPredictAngle(self):
        action = self._agentPredictAction()
        angle = self.playground.calWheelAngleFromAction(action)
        self._setWheelAngle(angle)

    def _setUpUiEvent(self):
        # self.main_ui.next_step_btn.clicked.connect(self.canvas.tick)
        self.main_ui.next_step_btn.clicked.connect(
            lambda: self._playgroundNextStep())

        def changeWheelAngle():
            self.playground.car.setWheelAngle(
                self.main_ui.wheel_angle_slider.value()
            )
            self.canvas.update()
            self.main_ui.wheel_angle_label.setText(
                str(self.main_ui.wheel_angle_slider.value()))
        self.main_ui.wheel_angle_slider.valueChanged[int].connect(
            changeWheelAngle)

        self.main_ui.reset_btn.clicked.connect(self.canvas.resetPlayground)
        self.main_ui.train_btn.clicked.connect(self._startTrainingAgent)

        self.main_ui.predict_btn.clicked.connect(self._setPredictAngle)

        def setAutoPredictMode():
            self.is_auto_predict = self.main_ui.auto_predict_cbox.isChecked()
            self.main_ui.predict_btn.setEnabled(
                not self.main_ui.auto_predict_cbox.isChecked())
            self.main_ui.wheel_angle_slider.setEnabled(
                not self.main_ui.auto_predict_cbox.isChecked())
            if self.is_auto_predict:
                self._setPredictAngle()
        self.main_ui.auto_predict_cbox.stateChanged.connect(setAutoPredictMode)

        def drawSupportLineChanged():
            self.canvas.setDrawSupportLine(
                self.main_ui.draw_support_lines_cbox.isChecked())
            self.canvas.update()

        self.main_ui.draw_support_lines_cbox.stateChanged.connect(
            drawSupportLineChanged)

        def handlePlaygroundChanged():
            if not self.is_auto_predict:
                value = self.main_ui.wheel_angle_slider.value()
                self._setWheelAngle(value)
            else:
                self._setPredictAngle()

        self.canvas.eventHandler.playgroundChangeEvent.connect(
            handlePlaygroundChanged)

        def runningHistory():
            self.main_ui.auto_predict_cbox.setChecked(False)
            self.main_ui.store_path_cbox.setChecked(False)
            self._setStartThreadRunning()
            self.playground.reset()
            try:
                self.playground.loadHistory()
                init_pos = self.playground.history_car_init_pos

                init_angle = self.playground.history_car_init_angle

                self.playground.setCarPosAndAngle(init_pos, init_angle)

                for his in self.playground.read_histroy_list:
                    angle = his[-1]
                    self.playground.car.setWheelAngle(angle)
                    self._setWheelAngle(angle)
                    self.playground.step()
                    self.playground.render()

            except FileNotFoundError:
                print('no history data')
            finally:
                self._setEndThreadRunning()

        def startRunHistory():
            if self.is_using_thread:
                return
            self.auto_run_thread = threading.Thread(target=runningHistory)
            self.auto_run_thread.start()

        def setStoreHistory():
            self.playground.store_history_path = self.main_ui.store_path_cbox.isChecked()
        self.main_ui.store_path_cbox.stateChanged.connect(setStoreHistory)

        self.main_ui.run_history_btn.clicked.connect(startRunHistory)

    def _setWheelAngle(self, value):  # we can use methods to change wheel angle
        # to invoke slider value change event
        v = self.main_ui.wheel_angle_slider.value()
        if v == value:
            self.main_ui.wheel_angle_slider.setValue(v+1)

        self.main_ui.wheel_angle_slider.setValue(int(value))


class MyHw2App(QApplication):
    def __init__(self, argv: typing.List[str]) -> None:
        super().__init__(argv)

        # create ui and window obj after app created
        self.main_window = MyMainWindow(main_ui_class)

    def start(self):
        self.main_window.show()
        self.exec()


def main():
    app = MyHw2App(sys.argv)
    app.start()


if __name__ == "__main__":
    main()
