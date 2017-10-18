# import cv2413.cv2 as cv2
import cv310.cv2 as cv2
import numpy as np
import sys
import math
from scipy import spatial
import time

import pyzed.camera as zcam
import pyzed.defines as sl
import pyzed.types as tp
import pyzed.core as core

sample_width = 5
sample_height = 5

sample_width = sample_width / 2
sample_height = sample_height / 2


class ZED():

    def __init__(self):
        self.zed, self.image, self.depth, self.point_cloud = self.init_zed()

    def init_zed(self):
        # Create a PyZEDCamera object
        zed = zcam.PyZEDCamera()

        # Create a PyInitParameters object and set configuration parameters
        init_params = zcam.PyInitParameters()
        init_params.depth_mode = sl.PyDEPTH_MODE.PyDEPTH_MODE_QUALITY  # Use PERFORMANCE depth mode
        init_params.coordinate_units = sl.PyUNIT.PyUNIT_METER  # Use milliliter units (for depth measurements)
        init_params.camera_fps = 30
        init_params.camera_resolution = sl.PyRESOLUTION.PyRESOLUTION_HD1080
        init_params.depth_minimum_distance = 0.3
        init_params.coordinate_system = sl.PyCOORDINATE_SYSTEM.PyCOORDINATE_SYSTEM_IMAGE

        # Open the camera
        err = zed.open(init_params)
        if err != tp.PyERROR_CODE.PySUCCESS:
            exit(1)

        # Create and set PyRuntimeParameters after opening the camera
        self.runtime_parameters = zcam.PyRuntimeParameters()
        self.runtime_parameters.sensing_mode = sl.PySENSING_MODE.PySENSING_MODE_FILL # Use STANDARD sensing mode

        image = core.PyMat()
        depth = core.PyMat()
        point_cloud = core.PyMat()

        return zed, image, depth, point_cloud

    def close(self):
        self.zed.close()

    def grab(self):
        return self.zed.grab(self.runtime_parameters) == tp.PyERROR_CODE.PySUCCESS

    def retrieve_processed(self):
        # Retrieve left image
        self.zed.retrieve_image(self.image, sl.PyVIEW.PyVIEW_LEFT)
        # Retrieve depth map. Depth is aligned on the left image
        self.zed.retrieve_measure(self.depth, sl.PyMEASURE.PyMEASURE_DEPTH)
        # Retrieve colored point cloud. Point cloud is aligned on the left image.
        self.zed.retrieve_measure(self.point_cloud, sl.PyMEASURE.PyMEASURE_XYZ)
        depth_data = self.depth.get_data()
        depth_clipped = np.clip(depth_data, a_min=None, a_max=5.0)
        depth_clipped *= 51
        depth_clipped = 255 - depth_clipped
        return self.image.get_data(), depth_clipped, self.point_cloud.get_data()

class SkinColorAnalyzer():
    def __init__(self, img_, window_name_, h_tolerance_=0.2, s_tolerance_=0.2, v_tolerance_=0.2):
        self.h_range_low_raw = [0,0]
        self.h_range_high_raw = [179,179]
        self.h_range_raw = [255,0]
        self.s_range_raw = [255,0]
        self.v_range_raw = [255,0]
        self.samples = 0
        self.h_range_low = [0,0]
        self.h_range_high = [179,179]
        self.h_range = [255,0]
        self.s_range = [255,0]
        self.v_range = [255,0]
        self.h_tolerance = h_tolerance_
        self.s_tolerance = s_tolerance_
        self.v_tolerance = v_tolerance_
        self.img = img_
        self.dis_img = cv2.cvtColor(img_, cv2.COLOR_HSV2BGR)
        self.sample_height = 10 // 2
        self.sample_width = 10 // 2
        self.window_name = window_name_
    def update_range(self, range_processed, range_raw, tolerance, value, mx):
        range_raw[0] = min(range_raw[0], value)
        range_raw[1] = max(range_raw[1], value)
        margin = (range_raw[1] - range_raw[0]) * tolerance / 2.0
        range_processed[0] = max(range_raw[0] - margin, 0)
        range_processed[1] = min(range_raw[1] + margin, mx)
    # def update_range_h(self, range_low_processed, range_high_processed, range_low_raw, range_high_raw, tolerance, value):
    #     if value >= 0 and value < 90
    def add_sample(self, x, y):
        # hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        tl_x = max(0, x - self.sample_width)
        tl_y = max(0, y - self.sample_height)
        br_x = min(self.img.shape[1], x + self.sample_width)
        br_y = min(self.img.shape[0], y + self.sample_height)
        hsv_patch = self.img[tl_y:br_y, tl_x:br_x]
        cv2.rectangle(self.dis_img, (tl_x, tl_y), (br_x, br_y), (0,255,0), 1)
        cv2.imshow(self.window_name, self.dis_img)
        h_med = np.median(hsv_patch[:,:,0])
        s_med = np.median(hsv_patch[:,:,1])
        v_med = np.median(hsv_patch[:,:,2])
        print(h_med, s_med, v_med)
        self.update_range(self.h_range, self.h_range_raw, self.h_tolerance, h_med, 179)
        self.update_range(self.s_range, self.s_range_raw, self.s_tolerance, s_med, 255)
        self.update_range(self.v_range, self.v_range_raw, self.v_tolerance, v_med, 255)
        self.samples += 1
        print("Sample: %d" % self.samples)
        print("H range: min: %d, max: %d" % (self.h_range[0], self.h_range[1]))
        print("S range: min: %d, max: %d" % (self.s_range[0], self.s_range[1]))
        print("V range: min: %d, max: %d" % (self.v_range[0], self.v_range[1]))

def find_furthest_pair(points):
    mx_distance = 0.0
    mx_pt1 = None
    mx_pt2 = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            x1, y1 = points[i][0]
            x2, y2 = points[j][0]
            dist = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            if dist > mx_distance:
                mx_pt1 = (x1, y1)
                mx_pt2 = (x2, y2)
                mx_distance = dist
    return mx_pt1, mx_pt2

if __name__ == '__main__': 
    has_img = False
    zed = ZED()
    # img = cv2.imread(sys.argv[1])
    while not has_img:
        if zed.grab():
            has_img = True
        time.sleep(0.005)
    img, depth, pcd = zed.retrieve_processed()
    img = img[:,:,:3]
    pcd = pcd[:,:,:3]
    new_height = 480
    new_width = int(img.shape[1] * new_height // img.shape[0])
    img = cv2.resize(img, (new_width, new_height))
    pcd = cv2.resize(pcd, (new_width, new_height))
    hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.namedWindow("sampler")
    skanalyzer = SkinColorAnalyzer(hsvimg, "sampler")
    def click_to_sample(event, x, y, flags, param):
        global skanalyzer
        if event == cv2.EVENT_LBUTTONDOWN:
            skanalyzer.add_sample(x, y)
    cv2.setMouseCallback("sampler", click_to_sample)
    cv2.imshow("sampler", img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    indices = np.stack(np.mgrid[0:img.shape[0], 0:img.shape[1]], axis=2)
    while True:
        data_ready = zed.grab()
        key = cv2.waitKey(5) & 0xFF
        if key == ord('q'):
            zed.close()
            exit(0)
        elif key == ord('s'):
            cv2.imwrite("1_sample_the_color.png", img)
            cv2.imwrite("2_apply_median_filter.png", median_filtered)
            cv2.imwrite("3_closing.png", closing)
            cv2.imwrite("4_find_all_contours.png", contoured)
            cv2.imwrite("5_select_biggest_contour.png", one_contoured)
            cv2.imwrite("6_find_furthest_pair_of_points.png", link_two_furthest)
            cv2.imwrite("7_use_2d_tree_to_find_nearest_samples.png", sample_points)
        elif key == ord('g'):
            if data_ready:
                img, depth, pcd = zed.retrieve_processed()
                img = img[:,:,:3]
                pcd = pcd[:,:,:3]
                new_height = 480
                new_width = int(img.shape[1] * new_height // img.shape[0])
                print(img.shape)
                img = cv2.resize(img, (new_width, new_height))
                pcd = cv2.resize(pcd, (new_width, new_height))
                hsvimg = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
                skanalyzer = SkinColorAnalyzer(hsvimg, "sampler")
                cv2.imshow("sampler", img)
        elif key == ord('p') and skanalyzer.samples > 5:
            hand_mask = cv2.inRange(hsvimg, 
                (skanalyzer.h_range[0], skanalyzer.s_range[0], skanalyzer.v_range[0]),
                (skanalyzer.h_range[1], skanalyzer.s_range[1], skanalyzer.v_range[1]))
            # hand_mask = np.reshape(hand_mask, (hand_mask.shape[0], hand_mask.shape[1], 1))
            res = cv2.bitwise_and(img, img, mask=hand_mask)
            grey = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
            binary = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY)[1]
            cv2.imshow("color_filtered", binary)
            median_filtered = cv2.medianBlur(binary, 5)
            cv2.imshow("median_filtered", median_filtered)
            closing = cv2.morphologyEx(median_filtered, cv2.MORPH_CLOSE, kernel)
            cv2.imshow("morph_closing", closing)
            im2, contours, hierarchy = cv2.findContours(closing,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            contoured = np.copy(img)
            cv2.drawContours(contoured, contours,-1,(0,255,0),2)  
            cv2.imshow("contoured", contoured)
            selected_contoured = None
            selected_contoured_area = 0.0
            for contour in contours:
                contour_area = cv2.contourArea(contour)
                if contour_area > selected_contoured_area:
                    selected_contour = contour
                    selected_contoured_area = contour_area
            if selected_contour.any() != None:
                one_contoured = np.copy(img)
                cv2.drawContours(one_contoured, [selected_contour],-1,(0,255,0),1)  
                hull = cv2.convexHull(selected_contour)
                cv2.drawContours(one_contoured, [hull],-1,(0,0,255),2) 
                # p1, p2 is in (col, row) i.e.(x, y) format! 
                p1, p2 = find_furthest_pair(hull)
                cv2.imshow("one_contoured", one_contoured)
                link_two_furthest = np.copy(one_contoured)
                cv2.line(link_two_furthest, p1, p2, (255,0,0),3)
                cv2.imshow("link_two_furthest", link_two_furthest)

                sample_points = np.copy(img)
                flated_hand_points = indices[binary==255]
                tree = spatial.KDTree(flated_hand_points)
                _, neighbors = tree.query([p1[::-1], p2[::-1]], 20)
                p1_nei_set = pcd[flated_hand_points[neighbors[0]][:,0], flated_hand_points[neighbors[0]][:,1]]
                p2_nei_set = pcd[flated_hand_points[neighbors[1]][:,0], flated_hand_points[neighbors[1]][:,1]]
                print(p1_nei_set)
                print(p2_nei_set)
                p1_mean = np.mean(p1_nei_set, axis=0)
                p2_mean = np.mean(p2_nei_set, axis=0)
                print(p1_mean, p2_mean)
                dist = np.sqrt(np.sum(np.square(p1_mean - p2_mean)))
                # direction_vector = (p1_mean - p2_mean)
                # ref_point = p1_mean
                # t = np.linspace(0, 100, 500)
                # coef = np.reshape(direction_vector, (3,1))
                # line_points = 
                print(dist)
                for neighbor in neighbors:
                    for pt in neighbor:
                        # dst = pcd[flated_hand_points[pt][0],flated_hand_points[pt][1],2]
                        sample_points[flated_hand_points[pt][0],flated_hand_points[pt][1],:] = np.array([0,255,0])#red(bad point)
                        # if dst > 0 and dst < 0.8:
                        #     sample_points[flated_hand_points[pt][0],flated_hand_points[pt][1],:] = np.array([0,255,0])#green(good point)
                        # else:
                        #     sample_points[flated_hand_points[pt][0],flated_hand_points[pt][1],:] = np.array([0,0,255])#red(bad point)
                cv2.imshow("final_sampled_pts", sample_points)
            else:
                print("no valid contour")
