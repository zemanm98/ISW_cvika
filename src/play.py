from time import time

from PyQt5 import QtGui, QtCore
from PyQt5.QtCore import QCoreApplication, Qt
from PyQt5.QtGui import QFont
from PyQt5.QtSvg import QSvgWidget
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QBoxLayout, QSpacerItem, QPushButton, QCheckBox, \
    QFrame, QTableWidget, QTableWidgetItem, QAbstractItemView, QRadioButton, QLabel

from src import graphics as svg
from src.environment import RealTimeEnvironment, Environment
from src.settings import *

ESCAPE_KEY = ord('\x1b')
R_KEY = ord('r')
REALTIME_ENVS = {cls.__name__ for cls in RealTimeEnvironment.__subclasses__()}


class AspectRatioWidget(QWidget):
    def __init__(self, widget, parent):
        super().__init__(parent)
        self.aspect_ratio = widget.size().width() / widget.size().height()
        self.setLayout(QBoxLayout(QBoxLayout.LeftToRight, self))
        #  add spacer, then widget, then spacer
        self.layout().addItem(QSpacerItem(0, 0))
        self.layout().addWidget(widget)
        self.layout().addItem(QSpacerItem(0, 0))

    def resizeEvent(self, e):
        w = e.size().width()
        h = e.size().height()

        if w / h > self.aspect_ratio:  # too wide
            self.layout().setDirection(QBoxLayout.LeftToRight)
            widget_stretch = h * self.aspect_ratio
            outer_stretch = (w - widget_stretch) / 2 + 0.5
        else:  # too tall
            self.layout().setDirection(QBoxLayout.TopToBottom)
            widget_stretch = w / self.aspect_ratio
            outer_stretch = (h - widget_stretch) / 2 + 0.5

        self.layout().setStretch(0, int(outer_stretch))
        self.layout().setStretch(1, int(widget_stretch))
        self.layout().setStretch(2, int(outer_stretch))


class Simulator(QMainWindow):
    STEPWISE_AUTOPLAY_DELAY = 0.5
    NO_ACTION = -1
    app = QApplication([])

    def __init__(self, env: Environment, agent=None, fps=0.0):
        super().__init__()
        self.__app = Simulator.app
        self.__current_keybinds = None
        self.setGeometry(0, 0, WIDTH, HEIGHT)
        self.__svg_canvas = QSvgWidget(parent=self)
        self.__svg_canvas.setGeometry(0, 0, WIDTH, HEIGHT)

        self.__should_run = True
        self.__stepwise = fps == 0.0
        self.__fps = fps
        self.__env = env
        self.__default_action = -1
        self.__env_state = None
        self.__renderer = svg.SVGAnimator(WIDTH, HEIGHT, RENDER_PRECISION)
        self.__player_mode = 0
        self.__agent = agent
        self.__prepared_agent_action = -1
        self.__agent_should_play = False
        self.__last_played_action = -1

        if self.__stepwise:
            if isinstance(env, RealTimeEnvironment):
                env_name = type(env).__name__
                raise ValueError("Environment %s is realtime (one of %s). For %s, fps must be non-zero." % (
                    env_name, REALTIME_ENVS, env_name))

            self.__default_action = -1
            self.__process_keypress = self.__process_keypress_stepwise
            self.__update = self.__update_stepwise
            self.__render_first = self.__render
            self.__stepwise_autoplay_target_time = 0
        else:
            if not isinstance(env, RealTimeEnvironment):
                env_name = type(env).__name__
                raise ValueError("Environment %s is not realtime (one of %s). For %s, fps must be equal to zero." % (
                    env_name, REALTIME_ENVS, env_name))

            self.__default_action = env.idle_action
            self.__process_keypress = self.__process_keypress_continuous
            self.__update = self.__update_continuous

            # continuous simulator attrs
            self.__start_time = self.__update_time = 0
            self.__dt = 1 / self.__fps
            self.__render_first = self.__render_first_continuous

        self.__bind_keybinds(self.__env)
        self.__player_action = self.__default_action
        self.__reward = 0

        self.__build_layout()

    def __build_layout(self):
        main = QWidget(self)
        helper = QWidget(main)
        game = AspectRatioWidget(self.__svg_canvas, main)
        reset_box = QWidget(helper)
        agent_control_box = QWidget(helper)
        last_action_box = QWidget(agent_control_box)

        # MAIN

        main_layout = QBoxLayout(QBoxLayout.LeftToRight)
        main_layout.addWidget(game)
        separator = QFrame()
        separator.setFrameShape(QFrame.VLine)
        main_layout.addWidget(separator)
        main_layout.addWidget(helper)
        main.setLayout(main_layout)

        # TABLE

        self.__action_table = QTableWidget(self.__env.num_actions + 1, 4, helper)
        self.__action_table.verticalHeader().setVisible(False)
        self.__action_table.horizontalHeader().setVisible(False)
        self.__action_table.setItem(0, 0, QTableWidgetItem("action #"))
        self.__action_table.setItem(0, 1, QTableWidgetItem("binding"))
        self.__action_table.setItem(0, 2, QTableWidgetItem("valid"))
        self.__action_table.setItem(0, 3, QTableWidgetItem("description"))
        self.__action_table.setColumnWidth(0, 60)
        self.__action_table.setColumnWidth(1, 70)
        self.__action_table.setColumnWidth(2, 50)
        self.__action_table.setColumnWidth(3, 200)

        for a in range(self.__env.num_actions):
            self.__action_table.setItem(a + 1, 3, QTableWidgetItem(self.__env.action_help_text[a]))
            self.__action_table.setItem(a + 1, 2, QTableWidgetItem("--"))
            self.__action_table.setItem(a + 1, 0, QTableWidgetItem(str(a)))

        for (b, a) in self.__env.bindings:
            self.__action_table.setItem(a + 1, 1, QTableWidgetItem(b))

        self.__action_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.__action_table.setFocusPolicy(Qt.NoFocus)

        bold = QFont()
        bold.setBold(True)
        italic = QFont()
        italic.setItalic(True)

        for i in range(3):
            self.__action_table.item(0, i).setFont(bold)

        for i in range(self.__env.num_actions):
            self.__action_table.item(i + 1, 0).setFont(italic)

        # HELPER

        helper_layout = QBoxLayout(QBoxLayout.TopToBottom, parent=main)
        helper_layout.addWidget(reset_box)
        helper_layout.addWidget(agent_control_box)
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.HLine)
        helper_layout.addWidget(separator2)
        helper_layout.addWidget(self.__action_table)
        helper.setLayout(helper_layout)
        helper.setFixedWidth(400)

        # RESET

        reset_layout = QBoxLayout(QBoxLayout.TopToBottom)
        reset_button = QPushButton("Reset environment (or press R)", parent=reset_box)
        reset_button.clicked.connect(self.__restart_env)
        self.__autoreset_checkbox = QCheckBox("Reset automatically", parent=reset_box)
        self.__autoreset_checkbox.clicked.connect(lambda: reset_button.setEnabled(not reset_button.isEnabled()))
        reset_layout.addWidget(self.__autoreset_checkbox)
        reset_layout.addWidget(reset_button)
        reset_box.setLayout(reset_layout)
        reset_box.setFixedHeight(90)

        # AC

        last_action_layout = QBoxLayout(QBoxLayout.LeftToRight)
        agent_control_layout = QBoxLayout(QBoxLayout.TopToBottom)

        self.__agent_control_button = QPushButton("Perform agent step", parent=agent_control_box)
        last_action_label = QLabel("Last action: ", parent=last_action_box)
        last_action_label.setFixedWidth(80)
        self.__last_action_text = QLabel("--", parent=last_action_box)
        agent_control_radio = QRadioButton("Agent control", parent=agent_control_box)
        player_control_radio = QRadioButton("Player control", parent=agent_control_box)
        self.__agent_control_interrupted_autoplay_checkbox = QCheckBox("Interrupted autoplay", parent=agent_control_box)

        last_action_layout.addWidget(last_action_label)
        last_action_layout.addWidget(self.__last_action_text)
        last_action_box.setLayout(last_action_layout)
        last_action_box.setFixedHeight(50)

        agent_control_layout.addWidget(player_control_radio)
        agent_control_layout.addWidget(agent_control_radio)
        agent_control_layout.addWidget(self.__agent_control_interrupted_autoplay_checkbox)
        agent_control_layout.addWidget(self.__agent_control_button)
        agent_control_layout.addWidget(last_action_box)
        agent_control_box.setLayout(agent_control_layout)

        player_control_radio.setChecked(True)
        self.__agent_control_interrupted_autoplay_checkbox.setEnabled(False)
        self.__agent_control_button.setEnabled(False)
        self.__agent_control_interrupted_autoplay_checkbox.clicked.connect(self.__toggle_autoplay_wrapper)
        agent_control_radio.toggled.connect(self.__agent_radio)
        player_control_radio.toggled.connect(self.__player_radio)
        self.__agent_control_button.clicked.connect(self.__make_agent_action)
        agent_control_box.setFixedHeight(200)

        self.setCentralWidget(main)
        
        game.setFixedWidth(680)

        game.setFocus()

    def keyPressEvent(self, a0: QtGui.QKeyEvent) -> None:
        try:
            v = ord(a0.text())
        except TypeError:
            return
        self.__current_keybinds[v](a0)

    def run(self):
        self.__restart_env()
        while self.__should_run:
            self.__update()

            if self.__autoreset_checkbox.isEnabled() and self.__autoreset_checkbox.isChecked() and self.__env_state[3]:
                self.__autoreset_checkbox.setEnabled(False)
                QtCore.QTimer.singleShot(1000, self.__restart_env_auto)

            self.__app.processEvents()

    def __process_keypress_stepwise(self, action):
        self.__player_action = action
        # print(self.__next_action)

    def __process_keypress_continuous(self, action):
        self.__player_action = action
        # print(self.__next_action)

    def __restart_env(self):
        self.__agent_should_play = False
        self.__env_state = self.__env.set_to_initial_state()

        self.__update_action_table_validity()

        if self.__agent is not None:
            self.__prepared_agent_action = self.__agent.best_action(self.__env_state[0], self.__env_state[2])

        if not self.__stepwise:
            self.__start_time = time()
            self.__update_time = self.__start_time + 2 + self.__dt
            if self.__agent_control_interrupted_autoplay_checkbox.isChecked():
                self.__player_action = self.NO_ACTION
            else:
                self.__player_action = self.__default_action

        elif self.__agent_control_interrupted_autoplay_checkbox.isChecked():
            self.__stepwise_autoplay_target_time = time() + 1 / self.__env.animator_render_fps
            self.__player_action = self.__default_action
        else:
            self.__player_action = self.__default_action

        self.__reward = 0
        self.__render_first()

    def __update_action_table_validity(self):
        for a in range(self.__env.num_actions):
            if a in self.__env_state[2] and not self.__env_state[3]:
                self.__action_table.item(a + 1, 2).setText("True")
                self.__action_table.item(a + 1, 2).setBackground(QtGui.QColor(60, 150, 60))
            else:
                self.__action_table.item(a + 1, 2).setText("False")
                self.__action_table.item(a + 1, 2).setBackground(QtGui.QColor(150, 60, 60))

    def __restart_env_auto(self):
        self.__restart_env()
        self.__autoreset_checkbox.setEnabled(True)

    def __player_radio(self, ev):
        if ev:
            self.__agent_control_button.setEnabled(False)
            self.__agent_control_interrupted_autoplay_checkbox.setEnabled(False)
            self.__toggle_player()

    def __agent_radio(self, ev):
        if ev:
            self.__agent_control_button.setEnabled(self.__stepwise and
                                                   not self.__agent_control_interrupted_autoplay_checkbox.isChecked())
            self.__agent_control_interrupted_autoplay_checkbox.setEnabled(True)
            if self.__stepwise:
                self.__stepwise_autoplay_target_time = time() + 1 / self.__env.animator_render_fps
            self.__toggle_player()

    def __toggle_player(self):
        if self.__agent is None:
            print("Agent not supplied.")
        else:
            if self.__player_mode == 0:
                self.__player_mode = 1
                self.__prepared_agent_action = self.__agent.best_action(self.__env_state[0], self.__env_state[2])
            else:
                self.__player_mode = 0
                self.__player_action = self.NO_ACTION if self.__stepwise else self.__default_action
            self.__render()

    def __toggle_autoplay_wrapper(self, ev):
        self.__agent_control_button.setEnabled(not ev and self.__stepwise)

        if ev:
            if self.__stepwise:
                self.__stepwise_autoplay_target_time = time() + 1 / self.__env.animator_render_fps
            else:
                self.__player_action = self.NO_ACTION

    def __make_agent_action(self):
        if self.__agent is None:
            print("Agent not supplied.")
        else:
            if self.__player_mode == 0:
                print("Agent not currently playing.")
            else:
                self.__agent_should_play = True
                
    def close_app(self):
        self.__should_run = False
        QCoreApplication.exit(0)
        
    def closeEvent(self, event):
        self.close_app()
        event.accept()

    def __unbind_keybinds(self):
        self.__current_keybinds = [lambda e: print("Unbound key " + e.text())] * 256

        self.__current_keybinds[ESCAPE_KEY] = lambda e: self.close_app()
        self.__current_keybinds[R_KEY] = lambda e: self.__restart_env()

    def __bind_keybinds(self, env):
        self.__unbind_keybinds()

        for src, dest in env.bindings:
            if src != ESCAPE_KEY:  # just to make sure the app is cleanly exitable
                self.__current_keybinds[ord(src)] = lambda e, v=dest: self.__process_keypress(v)

    def __on_last_action(self, a, player):
        self.__last_action_text.setText(f"{a} - {self.__env.action_help_text[a]} ({player=})")

    def __update_stepwise(self):
        # don't update if terminal, until user inputs an action or if last input action is invalid
        if self.__env_state[3]:
            return

        if self.__player_mode == 0:
            if self.__player_action == self.NO_ACTION or self.__player_action not in self.__env_state[2]:
                return
            self.__env_state = self.__env.act(self.__player_action)
            self.__on_last_action(self.__player_action, True)
        else:
            if self.__agent_control_interrupted_autoplay_checkbox.isChecked():
                target_action = None
                player = False
                if self.__player_action != self.NO_ACTION and self.__player_action in self.__env_state[2]:
                    target_action = self.__player_action
                    player = True
                elif time() > self.__stepwise_autoplay_target_time:
                    target_action = self.__prepared_agent_action

                if target_action is not None:
                    self.__env_state = self.__env.act(target_action)
                    if not self.__env_state[3]:
                        self.__prepared_agent_action = self.__agent.best_action(self.__env_state[0],
                                                                                self.__env_state[2])
                    self.__stepwise_autoplay_target_time = time() + 1 / self.__env.animator_render_fps
                    self.__on_last_action(target_action, player)
                else:
                    return
            else:
                if not self.__agent_should_play:
                    return
                self.__env_state = self.__env.act(self.__prepared_agent_action)
                self.__on_last_action(self.__prepared_agent_action, False)

                if not self.__env_state[3]:
                    self.__prepared_agent_action = self.__agent.best_action(self.__env_state[0], self.__env_state[2])
                self.__agent_should_play = False

        self.__reward += self.__env_state[1]
        self.__player_action = self.NO_ACTION
        self.__update_action_table_validity()
        self.__render()

    def __update_continuous(self):
        if self.__env_state[3]:
            # 'pause' if terminal
            return

        if self.__player_mode == 1 and self.__prepared_agent_action == self.NO_ACTION:
            self.__prepared_agent_action = self.__agent.best_action(self.__env_state[0], self.__env_state[2])

        if time() > self.__update_time:
            self.__update_time += self.__dt

            if self.__player_mode == 0:
                if self.__player_action not in self.__env_state[2]:
                    # if invalid input, fall back to default (idle)
                    self.__player_action = self.__default_action
                self.__env_state = self.__env.act(self.__player_action)
                self.__on_last_action(self.__player_action, True)

            else:
                if self.__agent_control_interrupted_autoplay_checkbox.isChecked() \
                        and self.__player_action != self.NO_ACTION \
                        and self.__player_action in self.__env_state[2]:
                    self.__env_state = self.__env.act(self.__player_action)
                    self.__on_last_action(self.__player_action, True)
                    self.__last_played_action = -1
                else:
                    self.__env_state = self.__env.act(self.__prepared_agent_action)
                    self.__on_last_action(self.__prepared_agent_action, False)
                    self.__last_played_action = self.__prepared_agent_action
                self.__prepared_agent_action = self.NO_ACTION

            self.__reward += self.__env_state[1]

            if self.__agent_control_interrupted_autoplay_checkbox.isChecked():
                self.__player_action = self.NO_ACTION
            else:
                self.__player_action = self.__default_action

            self.__update_action_table_validity()
            self.__render()

    def __render_first_continuous(self):
        canvas = self.__env.render_current_state()
        canvas.translate(ACTIONS_X1, ACTIONS_Y1)
        canvas.stroke_weight(1)
        canvas.fill("black")
        canvas.text("Prepare yourself.", ACTIONS_WIDTH * 0.5, ACTIONS_HEIGHT * 0.5, 20)
        svg_text = self.__renderer.build_one(canvas)
        svg_bytes = bytes(svg_text, encoding='utf-8')
        self.__svg_canvas.load(svg_bytes)

    def __render(self):
        canvas = self.__env.render_current_state()
        if self.__player_mode == 1 and not self.__env_state[3]:
            canvas.translate(ACTIONS_X1, ACTIONS_Y1)
            self.__agent.render_action_distribution(canvas, self.__prepared_agent_action if
            self.__prepared_agent_action > 0 else self.__last_played_action)
        svg_text = self.__renderer.build_one(canvas)
        svg_bytes = bytes(svg_text, encoding='utf-8')
        self.__svg_canvas.load(svg_bytes)