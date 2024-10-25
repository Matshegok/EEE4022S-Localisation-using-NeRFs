#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger 
import numpy as np
import os
import cv2
import json 
import math
import time
from scipy.spatial.transform import Rotation as R
from voyager_interfaces.srv import SaveImageAndPose 


#used to determine which images to save
DEFAULT_DISTANCE_THRESHOLD = 0.1
DEFAULT_ANGLE_THRESHOLD = 5.0 #angle in degrees


class data_collect(Node): 
    def __init__(self):
        super().__init__("data_collect") 
        self.get_logger().info("Running Data Collect:")
        self.setup_parameters()
        self.setup_publishers()
       # self.setup_subscribers()

        ###
        #self.sub_node = rclpy.create_node('sub_node')
        
    def setup_parameters(self):
        self.declare_parameter('wait_for_ready', False) 
        self.ready =  self.get_parameter('wait_for_ready').value
        self.get_logger().info("ready: "+str(self.ready))
        self.last_odom = None
        self.current_odom = None
        self.zero_odom_offset = None
        self.declare_parameter('distance_threshold', DEFAULT_DISTANCE_THRESHOLD)
        self.declare_parameter('angle_threshold_deg', DEFAULT_ANGLE_THRESHOLD)
        self.distance_threshold =self.get_parameter('distance_threshold')
        self.angle_threshold = math.radians( float(self.get_parameter('angle_threshold_deg').value) )
        self.get_logger().info("angle: "+str(self.angle_threshold))
        self.get_logger().info("distance: "+ str(self.distance_threshold.value ))
        self.nerf_images_array = None
        self.nerf_poses_array = None

    def setup_publishers(self): #ignoring the response from subscriber for now
        self.save_image_and_pose = self.create_client(SaveImageAndPose, 'save_image_pose')
        #self.save_image_and_pose = self.sub_node.create_client(SaveImageAndPose, 'save_image_pose')

        while not self.save_image_and_pose.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service \'save_image_pose\' not available, waiting...')
        self.get_logger().info('Service \'save_image_pose\' is available')
        self.request = SaveImageAndPose.Request()
    
    def setup_subscribers(self):
        if  self.ready:
            self.srv_ready = self.create_service(Trigger, 'ready_data_collect', self.on_ready)
            pass
        self.sub_odom = self.create_subscription(Odometry, '/odometry', self.process_odom_data, 10)
        self.sub_odom
        self.sub_images = self.create_subscription(Image, '/webcam/image_raw', self.process_image_data, 10)
        self.sub_images

    #not sure what to trigger this method yet... but nice to have for now
    def on_ready(self, request, response):
        self.get_logger().info("on ready")
        if  not self.ready:
            self.ready = True
            return response(sucess=True)
        else:
            return response(success=False, message="Data collect already started.")
        
    
    def process_odom_data(self, msg):
        if self.ready:
            if self.current_odom is None:
                self.zero_odom_offset = msg.pose.pose
            self.current_odom = self.subtract_odom(msg, self.zero_odom_offset)
            if(self.pose_to_matrix(self.current_odom)[2,3]>0):
                self.get_logger().info("------panic----------")
                self.get_logger().info(str(self.current_odom))
                self.get_logger().info(str(self.pose_to_matrix(self.current_odom)))
                
           
    # To process the array of NeRF images and poses
    def process_nerf_data(self):
        
        #Getting the images and poses arra
        nerf_images_array = self.nerf_images_array
        nerf_poses_array = self.nerf_poses_array
        
        # Getting the length of the nerf_images_array and nerf_poses_array
        array_length = len(nerf_images_array)
        
        #Iterate through arrays for processing
        for i in range(array_length):
            
            #The current image and pose based on index i
            current_image = nerf_images_array[i]
            current_pose = nerf_poses_array[i]
            
            #Now saving the iamge and pose 
            self.current_odom = current_pose
            self.save_data(current_image)
            
            
    # Process the nerf images into into the node
    def process_nerf_images(self, image_path):
        # Array to store the loaded images
        nerf_images_array = []
    
        # Get list of all files in the directory
        image_files = sorted([f for f in os.listdir(image_path) if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Iterate over each file, load the image, and append it to the array
        for image_file in image_files:
            full_image_path = os.path.join(image_path, image_file)
            
            # Read the image using OpenCV
            image = cv2.imread(full_image_path)
            
            # Append the image to the array
            if image is not None:
                nerf_images_array.append(image)
            else:
                self.get_logger().warn(f"Could not load image: {full_image_path}")
        
        self.nerf_images_array = nerf_images_array
        

    # Process the nerf_camera_paths into an array of poses
    def process_nerf_poses(self, camera_paths):
        
        # Array to store the processed poses
        nerf_poses_array = []
    
        # Load the JSON data containing camera paths and poses
        with open(camera_paths, 'r') as file:
            data = json.load(file)
        
        # Iterate over all camera paths (in case there are more than one)
        for path in data['camera_path']:
            # Extract the 'camera_to_world' matrix (which is in flattened 4x4 format)
            camera_to_world = path['camera_to_world']
            
            # Reshape the flat list into a 4x4 matrix (numpy array)
            pose_matrix = np.array(camera_to_world).reshape(4, 4)
            nerf_pose = self.matrix_to_pose(pose_matrix)
            
            # Append the pose matrix to the array of poses
            nerf_poses_array.append(nerf_pose)
        
        self.nerf_poses_array = nerf_poses_array
        

    def subtract_odom(self, odom, odom_frame_to_subtract):
        odom_frame=self.pose_to_matrix(odom.pose.pose)
        subtracted_odom = np.matmul(self.inverse(self.pose_to_matrix(odom_frame_to_subtract)) , odom_frame)
        #subtracted_odom1 = np.matmul(np.linalg.inv(self.pose_to_matrix(odom_frame_to_subtract)) , odom_frame)
        pose = self.matrix_to_pose(subtracted_odom)
        return  pose
    
    
    def process_image_data(self, msg):
        if self.ready:
            if self.last_odom is None:
                if self.current_odom is not None:
                    self.save_data(msg)
                return
            current_frame = self.current_odom 
            old_frame = self.last_odom 
            difference = np.matmul(self.inverse(self.pose_to_matrix(old_frame)) , self.pose_to_matrix(current_frame))

            #delta_distance = math.sqrt(math.pow(difference[0,3],2)+math.pow(difference[1,3],2)+math.pow(difference[2,3],2))
            delta_distance = math.sqrt(math.pow(self.pose_to_matrix(old_frame)[0,3]-self.pose_to_matrix(current_frame)[0,3],2)+
                                       math.pow(self.pose_to_matrix(old_frame)[1,3]-self.pose_to_matrix(current_frame)[1,3],2)+
                                       math.pow(self.pose_to_matrix(old_frame)[2,3]-self.pose_to_matrix(current_frame)[2,3],2))
        

            rotVec,_ = cv2.Rodrigues(difference[:3, :3])
            delta_theta=abs(rotVec[2]/math.pi*180)
            self.get_logger().info("delta-angle: :"+str(delta_theta))

            if delta_distance > self.distance_threshold.value or delta_theta > self.angle_threshold:
                self.get_logger().info("saved")
                self.save_data(msg)
                
    # creating the function to process the NeRF daa 

    def save_data(self, img):

        self.request.pose=self.current_odom
        self.request.image=img
        self.response = self.save_image_and_pose.call_async(self.request)
        
        
        
# not working at the moment     
###
#        rclpy.spin_until_future_complete(self.sub_node, self.response)
#        self.get_logger().info("response: "+ str(self.response.success))
#        if not self.response.success:
#            rclpy.logging.get_logger('data_collection').error('Data Collection - couldn\'t save data. Err: %s' % self.response.message)
#
        self.last_odom = self.current_odom


    def pose_to_matrix(self, pose):
        quartenion = np.array([pose.orientation.x, pose.orientation.y, 
            pose.orientation.z, pose.orientation.w])
        #converting quarternion to 3x3 transform
        small_matrix = R.from_quat(quartenion).as_matrix()

        #adding translation parts to transform 
        matrix = np.identity(4)
        matrix[:3, :3] = small_matrix
        matrix[0, 3] = pose.position.x
        matrix[1, 3] = pose.position.y
        matrix[2, 3] = pose.position.z
        return matrix

    def matrix_to_pose(self, matrix):
        small_matrix  = R.from_matrix(matrix[:3, :3])
        quart=small_matrix.as_quat()
        pose = Pose()
        pose.position.x = matrix[0,3]
        pose.position.y = matrix[1,3]
        pose.position.z = matrix[2,3]
        pose.orientation.x = quart[0]
        pose.orientation.y = quart[1]
        pose.orientation.z = quart[2]
        pose.orientation.w = quart[3]
        return pose

    def inverse(self, pose):
        transpose = np.zeros((3, 3))  
        for i in range(3):
            for j in range(3):
                transpose[i,j] = pose[j,i]
        positions=np.zeros((3,1))
        positions[0,0]=pose[0,3]
        positions[1,0]=pose[1,3]
        positions[2,0]=pose[2,3]
        new=-np.matmul(transpose,positions)
        final_matrix = np.identity(4)
        final_matrix[:3, :3] = transpose
        final_matrix[:3, [3]]=new
        return final_matrix



def main(args=None):
    rclpy.init(args=args)
    node = data_collect() 
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()