import math
from time import sleep
from serial.tools import list_ports

import numpy as np
import pydobot
import cv2
from pathfinding.core.diagonal_movement import DiagonalMovement
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

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
        
    @staticmethod
    def frame_edges(image):
        image = image.copy()

        # convert to hsv to get gray color
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # gray color mask
        lower_gray = np.array([0,0,0])
        upper_gray = np.array([180,50,110])
        mask = cv2.inRange(hsv, lower_gray, upper_gray)

        # edges
        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        edges = cv2.Canny(blur, 100, 200)

        return edges
        
    @staticmethod
    def frame_red(image):
        image = image.copy()

        # hsv color select red colors
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0,50,50])
        upper_red = np.array([10,255,255])
        mask1 = cv2.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170,50,50])
        upper_red = np.array([180,255,255])
        mask2 = cv2.inRange(hsv, lower_red, upper_red)

        mask = mask1 + mask2

        return mask

    @staticmethod
    def frame_blue(image):
        image = image.copy()

        # hsv color select blue colors
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100,50,50])
        upper_blue = np.array([130,255,255])
        mask = cv2.inRange(hsv, lower_blue, upper_blue)

        return mask

    @staticmethod
    def frame_green(image):
        image = image.copy()

        # hsv color select green colors
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40,50,50])
        upper_green = np.array([80,255,255])
        mask = cv2.inRange(hsv, lower_green, upper_green)

        return mask

    @staticmethod
    def center_of_mass(image):
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

    @staticmethod
    def combined_all(image):
        image = image.copy()

        edges = Camera.frame_edges(image)
        edges = Camera.thick(edges, 3, 3)

        red_mass = Camera.center_of_mass(Camera.frame_red(image))
        blue_mass = Camera.center_of_mass(Camera.frame_blue(image))
        green_mass = Camera.center_of_mass(Camera.frame_green(image))

        #combine the images
        if red_mass is not None:
            cv2.circle(image, red_mass, 7, (0, 255, 0), -1)
        if blue_mass is not None:
            cv2.circle(image, blue_mass, 7, (0, 0, 255), -1)
        if green_mass is not None:
            cv2.circle(image, green_mass, 7, (255, 0, 0), -1)

        # edges borders over the frame
        image[edges != 0] = (0, 0, 255)

        return image

    def display_capture(self, image_func):
        print("Press q or esc to quit")
        cv2.namedWindow("Video")
        cv2.setWindowProperty("Video", cv2.WND_PROP_TOPMOST, 1)
        while True:
            cv2.imshow("Video", image_func(self.grap_frame()))
            
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


# find the path from start to end  
def find_path(image, cube_width=50, cube_height=50):  
    edges = Camera.frame_edges(image)
    start = Camera.center_of_mass(Camera.frame_red(image))
    end = Camera.center_of_mass(Camera.frame_blue(image))

    if start is None:
        print("No start found")
        return None

    if end is None:
        print("No end found")
        return None

    # show the image
    display_image = image.copy()

    edges3 = Camera.thick(edges, 3, 3)
    display_image[edges3 != 0] = (0, 255, 0)

    cv2.circle(display_image, start, 7, (255, 255, 255), -1)
    cv2.circle(display_image, end, 7, (255, 255, 255), -1)

    # display in window that is in foreground
    cv2.namedWindow("Accept Y or cancel N")
    cv2.setWindowProperty("Accept Y or cancel N", cv2.WND_PROP_TOPMOST, 1)
    while True:
        cv2.imshow("Accept Y or cancel N", display_image)
        
        k = cv2.waitKey(1)
        if k%256 == 27 or k%256 == ord('q') or k%256 == ord('n'):
            print("Cancellig path finding")
            return

        elif k%256 == ord('y'):
            print("Finding path")
            break

    cv2.destroyAllWindows()
    
    # process the image
    thick_edges = Camera.thick(edges, cube_width, cube_height)

    # display the image
    grid = Grid(matrix=np.invert(thick_edges.astype(bool)))
    start = grid.node(start[0], start[1])
    end = grid.node(end[0], end[1])

    finder = AStarFinder(diagonal_movement=DiagonalMovement.always)
    path, runs = finder.find_path(start, end, grid)

    return path

def path_to_mask(path, shape):
    mask = np.zeros(shape, dtype=np.uint8)
    for point in path:
        mask[point[1], point[0]] = 255
    return mask


if __name__ == '__main__':
    with Camera("lol.png") as camera:
        #with Robot() as robot:
        robot = None
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
                    image = camera.grap_frame()
                    path = find_path(image)

                    if path is None:
                        print("No path found")
                        continue
                    
                    # display the path as image
                    path_mask = path_to_mask(path, image.shape[:2])
                    path_mask = Camera.thick(path_mask, 3, 3)

                    display_image = image.copy()
                    display_image[path_mask != 0] = (0, 255, 0)

                    
                    cv2.namedWindow("Path q to cancel, y to execute")
                    cv2.setWindowProperty("Path q to cancel, y to execute", cv2.WND_PROP_TOPMOST, 1)
                    while True:
                        cv2.imshow("Path q to cancel, y to execute", display_image)
                        
                        k = cv2.waitKey(1)
                        if k%256 == 27 or k%256 == ord('q') or k%256 == ord('n'):
                            print("Not executing path")
                            break

                        elif k%256 == ord('y'):
                            print("Executing path")

                            for point in path:
                                robot.move_to(x=point[0], y=point[1], z=30, r=0, wait=True)
                            break

                    cv2.destroyAllWindows()
                    
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