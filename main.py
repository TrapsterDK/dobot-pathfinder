import math
from time import sleep
from serial.tools import list_ports

import numpy as np
import pydobot
import cv2

np.set_printoptions(threshold=np.inf)

class Camera():
    def __init__(self, cam_id_or_img):
        match cam_id_or_img:
            case int():
                print('Camera starting')
                self.cam = cv2.VideoCapture(cam_id_or_img)

            case str():
                print('Image starting')
                self.cam = cv2.imread(cam_id_or_img)

            case _: 
                raise Exception('Invalid argument for camera')


    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_value, traceback):
        if isinstance(self.cam, cv2.VideoCapture):
            print('Camera exiting')
            self.cam.release()
        else:
            print('Image exiting')

    def grap_frame(self):

        if isinstance(self.cam, cv2.VideoCapture):
            ret, frame = self.cam.read()
            if not ret:
                print("Failed to grab frame")
                return
            return frame.copy()
        else:
            return self.cam.copy()
        
    
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
        edges = self.thick(edges, 3, 3)

        red_mass = self.center_of_mass(self.frame_red())
        blue_mass = self.center_of_mass(self.frame_blue())
        green_mass = self.center_of_mass(self.frame_green())

        #combine the images
        if red_mass is not None:
            cv2.circle(frame, red_mass, 7, (0, 255, 0), -1)
        if blue_mass is not None:
            cv2.circle(frame, blue_mass, 7, (0, 0, 255), -1)
        if green_mass is not None:
            cv2.circle(frame, green_mass, 7, (255, 0, 0), -1)

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

    #check if point colides with in the image
    def check_collision(self, image, point):
        #check if point is in the image
        if not (0 <= point[0] < image.shape[1]) or not (0 <= point[1] < image.shape[0]):
            return False

        return image[point[1], point[0]]

    # find the path from start to end  
    def find_path(self, image, start, end):
        image = Camera.thick(image, 30, 30)

        # show the image
        cv2.namedWindow("Accept Y or cancel N")
        while True:
            cv2.imshow("Accept Y or cancel N", image)
            
            k = cv2.waitKey(1)
            if k%256 == 27 or k%256 == ord('q'):
                print("Escape hit, closing window")
                break
            
        cv2.destroyAllWindows()
            
        image = np.asarray(image, dtype=bool)

        #check if start and end is in the image
        if self.check_collision(image, start):
            print("Start is not in the image")
            return None
        
        if self.check_collision(image, end):
            print("End is not in the image")
            return None
        
        path = []
        current = start
        max_a = 2000
        while current != end:
            max_a-=1
            if max_a < 0:
                print("Max a reached")
                return path

            path.append(current)
            if not self.check_collision(image, (current[0] + 1, current[1])):
                current = (current[0] + 1, current[1])
            elif not self.check_collision(image, (current[0] - 1, current[1])):
                current = (current[0] - 1, current[1])
            elif not self.check_collision(image, (current[0], current[1] + 1)):
                current = (current[0], current[1] + 1)
            elif not self.check_collision(image, (current[0], current[1] - 1)):
                current = (current[0], current[1] - 1)
            else:
                return path

        return path
        


if __name__ == '__main__':
    with Camera("lol.png") as camera:
        #with Robot() as robot:
        robot = None
        with TreePathfinder() as pathfinder:
            while True:
                input_var = input("Command: ")
                match input_var:
                    case "exit" | "e" | "quit" | "q":   
                        break
                    case "show" | "s":
                        input_show = input("Show (r, g, b, (edge, e), (all, a)): ")
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
                        print(path)
                        for point in path:
                            image[point[1], point[0]] = (0, 255, 0)
                        
                        cv2.imshow("Path", image)
                        cv2.waitKey(0) 
                        
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