import math
from time import sleep
from serial.tools import list_ports

import numpy as np
import pydobot
import cv2

np.set_printoptions(threshold=np.inf)

class Camera():
    def __init__(self, cam_id):
        print('Camera starting')
        self.cam = cv2.VideoCapture(cam_id)

    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        print('Camera exiting')
        self.cam.release()
        cv2.destroyAllWindows()

    def grap_frame(self):
        ret, frame = self.cam.read()
        if not ret:
            print("Failed to grab frame")
            return
        return frame.copy()
    
    def frame_edges(self):
        frame = self.grap_frame()
        
        # convert to hsv to get gray color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # gray color mask
        lower_gray = np.array([0,0,0])
        upper_gray = np.array([180,50,110])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)

        # edges
        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        edges = cv2.Canny(blur, 100, 200)

        return edges
        

    def frame_red(self):
        frame = self.grap_frame()

        # hsv color select red colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170,50,50])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2

        return mask

    def frame_blue(self):
        frame = self.grap_frame()

        # hsv color select blue colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100,50,50])
        upper_blue = np.array([130,255,255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        return mask

    def frame_green(self):
        frame = self.grap_frame()

        # hsv color select green colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40,50,50])
        upper_green = np.array([80,255,255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        return mask

        
    def center_of_mass(self, image):
        # find contours in the binary image
        contours, hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        # calculate moment of all contours and find the biggest
        max_area = 0
        max_contour = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > max_area:
                max_area = area
                max_contour = cnt
        
        if max_contour is None:
            return None

        # find the center of mass
        M = cv2.moments(max_contour)
        if M["m00"] == 0:
            return None
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        return (cx, cy)

    def combined_all(self):
        frame = self.grap_frame()
        edges = self.frame_edges()
        edges = self.thick(edges, 30, 30)

        red_mass = self.center_of_mass(self.frame_red())
        blue_mass = self.center_of_mass(self.frame_blue())
        green_mass = self.center_of_mass(self.frame_green())

        #combine the images
        if red_mass is not None:
            cv2.circle(frame, red_mass, 7, (255, 255, 255), -1)
        if blue_mass is not None:
            cv2.circle(frame, blue_mass, 7, (255, 255, 255), -1)
        if green_mass is not None:
            cv2.circle(frame, green_mass, 7, (255, 255, 255), -1)

        # edges borders over the frame
        frame[edges != 0] = (0, 0, 255)

        return frame

    def display_capture(self, image_func):
        print("Press q or esc to quit")
        cv2.namedWindow("Video")
        while True:
            cv2.imshow("Video", image_func())
            
            k = cv2.waitKey(1)
            if k%256 == 27 or k%256 == ord('q'):
                print("Escape hit, closing window")
                break
        
        cv2.destroyAllWindows()

    # dilate the image to make it thick
    @staticmethod
    def thick(image, width, height):
        kernel = np.ones((width,height),np.uint8)
        dilation = cv2.dilate(image,kernel,iterations = 1)
        return dilation 

class Robot(pydobot.Dobot):
    def __init__(self) -> None:
        available_ports = list_ports.comports()

        if available_ports is None:
            print("Ingen porte at starte på")

        port = available_ports[0].device
        print(f'Starting robot on port: {port}')

        try:
            super().__init__(port=port, verbose=False)
        except Exception as e:
            print(f'Error on startup: {e}')
            exit()

        print('Started successfully')

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print('Robot exiting')
        self.close()

class TreePathfinder():
    def __init__(self) -> None:
        pass

    def __enter__(self):
        print("TreePathfinder starting")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print('TreePathfinder exiting')

    def check_point_collision(self, image, point: tuple[int, int]):
        if not (0 <= point[0] < image.shape[0] and 0 <= point[1] < image.shape[1]):
            #print(1)
            return True
        
        #print(2)
        #print(image[point[0], point[1]], image[point[1], point[0]])
        return image[point[1], point[0]]
    
    def find_path(self, image, start_point: tuple[int, int], end_point: tuple[int, int]):
        image = Camera.thick(image, 30, 30)
        image = np.asarray(image, dtype=bool)
        #print(image)

        # check if the start and end point is valid
        if self.check_point_collision(image, start_point):
            print("Start point is in collision")
            return None

        if self.check_point_collision(image, end_point):
            print("End point is in collision")
            return None

        # find the path
        path = []
        current_point = start_point
        max_tries = 10000
        while current_point != end_point:
            max_tries -= 1
            print(max_tries)
            if max_tries <= 0:
                print("To many tries")
                return path

            path.append(current_point)

            # find the next point
            next_point = None
            for x in range(-1, 2):
                for y in range(-1, 2):
                    if x == 0 and y == 0:
                        continue

                    new_point = (current_point[0] + x, current_point[1] + y)
                    if not self.check_point_collision(image, new_point):
                        next_point = new_point
                        break

                if next_point is not None:
                    break

            if next_point is None:
                print("No path found")
                return None

            current_point = next_point

        return path
        

if __name__ == '__main__':
    with Camera(1) as camera:
        #with Robot() as robot:
        robot = None
        with TreePathfinder() as pathfinder:
            while True:
                input_var = input("Command: ")
                match input_var:
                    case "exit" | "e" | "quit" | "q":   
                        break
                    case "show" | "s":
                        input_show = input("Show (r, g, b, (edge, e)): ")
                        match input_show:
                            case "r":
                                camera.display_capture(camera.frame_red)
                            case "g":
                                camera.display_capture(camera.frame_green)
                            case "b":
                                camera.display_capture(camera.frame_blue)
                            case "edge" | "e":
                                camera.display_capture(camera.frame_edges)
                            case "all" | "a":
                                camera.display_capture(camera.combined_all)
                            case _:
                                print("Invalid input")
                    case "move" | "m":
                        input_move = input("Move x, y, z, r: ")
                        s_input = input_move.split(",")
                        robot.move_to(x=float(s_input[0]), y=float(s_input[1]), z=float(s_input[2]), r=float(s_input[3]), wait=False)
                    case "path" | "p":
                        cx_start, cy_start = camera.center_of_mass(camera.frame_red())
                        cx_end, cy_end = camera.center_of_mass(camera.frame_blue())

                        path = pathfinder.find_path(camera.frame_edges(), (cx_start, cy_start), (cx_end, cy_end))

                        if path is None:
                            print("No path found")
                            continue
                        
                        # display the path as image
                        image = camera.grap_frame()

                        # draw the path
                        for point in path:
                            image[point[0], point[1]] = (0, 255, 0)
                        
                        cv2.imshow("Path", image)
                        k = cv2.waitKey(1)
                        
                    case _:
                        print("Invalid input")


'''
x, y, z, *_ = robot.pose()
cx, cy = camera.center_of_mass(camera.frame_red())

robot.move_to(x + 10, y + 10, z, r=90, wait=True)

x2, y2, z2, *_ = robot.pose()
cx2, cy2 = camera.center_of_mass(camera.frame_red())

# calculate camera and pixel ratio
camera_ratio = math.sqrt((x2 - x)**2 + (y2 - y)**2) / math.sqrt((cx2 - cx)**2 + (cy2 - cy)**2)

print(f'Camera ratio: {camera_ratio}')

#move robot to the center of mass of the red object
robot.move_to(x + (cx - camera.grap_frame().shape[0] // 2) * camera_ratio, y + (cy - camera.grap_frame().shape[1] // 2) * camera_ratio, z, r=90, wait=True)

robot.close()
'''