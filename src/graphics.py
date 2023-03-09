import math
from io import StringIO

from numpy import linspace

CAP_BUTT = "butt"
CAP_SQUARE = "square"
CAP_ROUND = "round"


class Frame:
    def __init__(self):
        self.instructions = []

    def stroke(self, v: str):
        self.instructions.append((0, v))

    def fill(self, v: str):
        self.instructions.append((1, v))

    def cap(self, v: str):
        self.instructions.append((2, v))

    def stroke_weight(self, v: float):
        self.instructions.append((3, v))

    def stroke_dasharray(self, v):
        self.instructions.append((4, v))

    def translate(self, x: float, y: float):
        self.instructions.append((5, x, y))

    def rotate(self, a: float, x: float, y: float):
        self.instructions.append((6, a, x, y))

    def scale(self, x: float, y: float):
        self.instructions.append((7, x, y))

    def clear_transform(self):
        self.instructions.append((8,))

    def line(self, x1, y1, x2, y2):
        self.instructions.append((9, x1, y1, x2, y2))

    def arrow(self, x1, y1, x2, y2, size, margin, color, text=None):
        angle = math.atan2(y2 - y1, x2 - x1)
        x1 += margin * math.cos(angle)
        y1 += margin * math.sin(angle)
        x2 -= margin * math.cos(angle)
        y2 -= margin * math.sin(angle)

        self.stroke(color)
        self.line(x1, y1, x2, y2)

        x3 = x2 - size * math.cos(angle + math.pi / 4)
        y3 = y2 - size * math.sin(angle + math.pi / 4)
        self.line(x2, y2, x3, y3)

        x4 = x2 - size * math.cos(angle - math.pi / 4)
        y4 = y2 - size * math.sin(angle - math.pi / 4)
        self.line(x2, y2, x4, y4)

        if text is not None:
            self.fill('#ffffff')
            self.stroke('none')
            diff = (-11 if angle > 0 else 11)
            mid_x = (x1 + x2) * 0.5 - 20 + diff
            mid_y = (y1 + y2) * 0.5 - 5 + diff
            self.rectangle(mid_x, mid_y, 40, 10)
            self.fill('none')
            self.stroke('blue')
            self.text(text, mid_x + 20, mid_y + 7, 6)

        self.stroke('black')

    def rectangle(self, x1, y1, w, h):
        self.instructions.append((10, x1, y1, w, h))

    def ellipse(self, x1, y1, rx, ry):
        self.instructions.append((11, x1, y1, rx, ry))

    def polyline(self, coords):
        self.instructions.append((12, coords))

    def text(self, text, x, y, size):
        """
        Always centered at [x,y].
        """
        self.instructions.append((13, x, y, size, text))


class SVGAnimator:
    def __init__(self, width, height, precision):
        self.__width = width
        self.__height = height
        self.__precision = precision
        self.__state_names = ["stroke", "fill", "stroke-linecap", "stroke-width", "stroke-dasharray"]
        self.__state_updated = False
        self.__state_changing_instructions = 9
        self.__working_frames = None
        self.__dt = None
        self.__duration = None
        self.__process_instruction = [self.__process_state_change, self.__process_state_change,
                                      self.__process_state_change, self.__process_state_change,
                                      self.__process_state_change,

                                      self.__process_translate, self.__process_rotate,
                                      self.__process_scale, self.__process_clear,

                                      self.__process_line, self.__process_rectangle, self.__process_ellipse,
                                      self.__process_polyline, self.__process_text]
        self.__reset_state()
        self.__translate_format = "translate(%." + str(precision) + "f %." + str(precision) + "f)"
        self.__rotate_format = "rotate(%." + str(precision) + "f %." + str(precision) + "f %." + str(precision) + "f)"
        self.__scale_format = "scale(%." + str(precision) + "f %." + str(precision) + "f)"
        self.__line_format = "\t\t\t<line x1=\"%." + str(precision) + "f\" y1=\"%." + str(precision) + "f\" x2=\"%." +\
                             str(precision) + "f\" y2=\"%." + str(precision) + "f\" />\n"
        self.__rectangle_format = "\t\t\t<rect x=\"%." + str(precision) + "f\" y=\"%." + str(precision) +\
                                  "f\" width=\"%." + str(precision) + "f\" height=\"%." + str(precision) + "f\" />\n"
        self.__ellipse_format = "\t\t\t<ellipse cx=\"%." + str(precision) + "f\" cy=\"%." + str(precision) +\
                                "f\" rx=\"%." + str(precision) + "f\" ry=\"%." + str(precision) + "f\" />\n"
        self.__text_format = "\t\t\t<text x=\"%." + str(precision) + "f\" y=\"%." + str(precision) + \
                             "f\" font-size=\"%d\" font-family=\"monospace\" " \
                             "dominant-baseline=\"middle\" text-anchor=\"middle\">%s</text>\n"
        self.__coord_format = "%." + str(precision) + "f,%." + str(precision) + "f"

    def __process_state_change(self, f, args):
        self.__state_updated = True
        self.__state[args[0]] = args[1]

    def __process_translate(self, f, args):
        self.__state_updated = True
        self.__transform_queue.append(self.__translate_format % (args[1], args[2]))

    def __process_rotate(self, f, args):
        self.__state_updated = True
        self.__transform_queue.append(self.__rotate_format % (args[1], args[2], args[3]))

    def __process_scale(self, f, args):
        self.__state_updated = True
        self.__transform_queue.append(self.__scale_format % (args[1], args[2]))

    def __process_clear(self, f, args):
        self.__state_updated = True
        self.__transform_queue.clear()

    def __write_state_group(self, f):
        f.write("\t\t<g ")
        for i in range(len(self.__state)):
            if self.__state[i] is not None:
                f.write("%s=\"%s\" " % (self.__state_names[i], str(self.__state[i])))
        f.write("transform=\"%s\" >\n" % " ".join(self.__transform_queue))

    def __preprocess_object(self, f):
        if self.__state_updated:
            f.write("\t\t</g>\n")
            self.__write_state_group(f)
            self.__state_updated = False

    def __process_line(self, f, args):
        self.__preprocess_object(f)
        f.write(self.__line_format % tuple(args[1:]))

    def __process_rectangle(self, f, args):
        self.__preprocess_object(f)
        f.write(self.__rectangle_format % tuple(args[1:]))

    def __process_ellipse(self, f, args):
        self.__preprocess_object(f)
        f.write(self.__ellipse_format % tuple(args[1:]))

    def __process_polyline(self, f, args):
        self.__preprocess_object(f)
        f.write("\t\t\t<polyline points=\"%s\"/>\n" % " ".join((self.__coord_format % (c[0], c[1])) for c in args[1]))

    def __process_text(self, f, args):
        self.__preprocess_object(f)

        f.write(self.__text_format % tuple(args[1:]))

    def __append_frame(self, f, frame, keytimes, frame_id):
        f.write("\t<g>\n")
        self.__render(f, frame)

        values = ["none"] * (len(self.__working_frames) + 1)
        values[frame_id] = "inline"
        values = ";".join(values)

        f.write("\t<animate id=\"frame%d\" attributeName=\"display\" values=\"%s\" keyTimes=\"%s\" dur=\"%.2fs\" "
                "begin=\"0s\" repeatCount=\"indefinite\"/>\n\t</g>\n" % (frame_id, values, keytimes, self.__duration))

    def __reset_state(self):
        self.__state = ["black", "none", "square", 1.0, None]
        self.__transform_queue = []

    def __render(self, f, frame):
        self.__reset_state()
        i = 0
        while i < len(frame.instructions):
            instruction = frame.instructions[i]
            if instruction[0] >= self.__state_changing_instructions:
                self.__write_state_group(f)
                self.__state_updated = False
                break

            self.__process_instruction[instruction[0]](f, instruction)
            i += 1

        while i < len(frame.instructions):
            instruction = frame.instructions[i]
            self.__process_instruction[instruction[0]](f, instruction)
            i += 1

        f.write("\t\t</g>\n")

    def build_full(self, frames, filename: str = None, fps: float = 1, delay: float = 0.0, frame_slice: slice = slice(None)):
        self.__working_frames = frames[frame_slice]
        self.__dt = 1 / fps
        self.__duration = len(self.__working_frames) / fps + delay

        keytimes = list(linspace(0, 1 / (delay / (len(self.__working_frames) / fps) + 1), len(self.__working_frames)))
        keytimes.append(1.0)
        keytimes = ";".join(str(kt) for kt in keytimes)

        if filename is not None:
            f = open(filename, "w")
        else:
            f = StringIO()

        f.write("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\">\n"
                % (self.__width, self.__height))

        for frame_id, frame in enumerate(self.__working_frames):
            self.__append_frame(f, frame, keytimes, frame_id)

        f.write("</svg>")

        if filename is not None:
            f.close()
        else:
            ret = f.getvalue()
            f.close()
            return ret

    def build_one(self, frame):
        f = StringIO()
        f.write("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%d\" height=\"%d\">\n"
                % (self.__width, self.__height))
        self.__render(f, frame)
        f.write("</svg>")

        f.seek(0)
        return f.read()