from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.filechooser import FileChooser
from kivy.uix.image import Image

import time
from skimage import transform
import numpy as np
import tensorflow as tf
import math
import cv2
import os

class MainWindow(Screen):
    pass

class TakePhotoWindow(Screen):    
    
    def capture(self):
        '''
        Function to capture the images and give them the names
        according to their captured time and date.
        '''
        camera = self.ids['camera']
        timestr = time.strftime("%Y%m%d_%H%M%S")
        #camera.export_to_png("image/IMG_{}.png".format(timestr))
        camera.export_to_png("image/counting.png")
        #TakePhotoWindow.switch_(self, ['D:\\INOVASI IT 2020\\Kivy_CountingIron\\image\\IMG_{}.png'.format(timestr)])
        TakePhotoWindow.switch_(self,['D:\\Inovasi Project\\Kivy_CountingIron\\image\\counting.png'])
        print("Captured")

    def switch_(self, source):
        #here you can insert any python logic you like        
        self.parent.current = 'Counting'
        print (source[0])
        #self.ids.image.clear_widgets()
        #self.ids.image.source = source[0]

class FileChooserWindow(Screen):   
    
    def load(self, path, filename):
        img = cv2.imread(filename[0])
        cv2.imwrite("image/counting.png",img)        
        FileChooserWindow.switch(self,['D:\\Inovasi Project\\Kivy_CountingIron\\image\\counting.png'])
        print(filename)
        print(filename[0])
            
    def selected(self,filename):
        self.ids.image.source = filename[0]

    def switch(self,filename):
        #here you can insert any python logic you like        
        self.parent.current = 'Counting'
        self.ids.image.source = filename[0]
        self.ids.image.reload()

class CountingWindow(Screen):    
    

    def detection(img_original,stride):
        patch_size = 71
        height = np.size(img_original,0)
        width = np.size(img_original,1)
        img_original = transform.resize(img_original,(height,width))   
        imgs=[]
        coordidate = []
        for i in range(patch_size,height-patch_size,stride):
            for j in range(patch_size,width-patch_size,stride):
                img_original_patch = img_original[int(i-(patch_size-1)/2):int(i+(patch_size-1)/2+1),int(j-(patch_size-1)/2):int(j+(patch_size-1)/2+1),:]
                imgs.append(img_original_patch)
                coordidate.append([i,j])
        data = np.asarray(imgs,np.float32)
        output =[]

        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        config.log_device_placement = True

        #os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        
        with tf.compat.v1.Session(config=config) as sess:
        #with tf.compat.v1.Session() as sess:
            saver = tf.compat.v1.train.import_meta_graph('model/model.ckpt.meta')
            saver.restore(sess, tf.compat.v1.train.latest_checkpoint('model/'))
                    
            graph = tf.compat.v1.get_default_graph() # tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            vol_slice =1500
            num_slice = math.ceil(np.size(data,0)/vol_slice)
            for i in range(0,num_slice,1):
                if i+1 != num_slice:
                    data_temp = data[i*vol_slice:(i+1)*vol_slice]            
                else:
                    data_temp = data[i*vol_slice:np.size(data,0)]
                    
                feed_dict = {x:data_temp}
                logits = graph.get_tensor_by_name("logits_eval:0")
                classification_result = sess.run(logits,feed_dict)
                output_temp = tf.argmax(classification_result,1).eval()
                output = np.hstack((output,output_temp))   
        candidate_center = []
        for i in range(len(output)):
            if output[i] == 1:
                candidate_center.append(coordidate[i])            
        return np.array(candidate_center) #the matrix of candidate center

    # the function of clustering, which gets the group of candidate center
    def clustering(candidate_center,threshold_dis):
        x = candidate_center[:,1]
        y = candidate_center[:,0]
        group_distance = []
        for i in range(len(candidate_center)):
            xpoint, ypoint = x[i], y[i]
            xTemp, yTemp = x, y 
            distance = np.sqrt(pow((xpoint-xTemp),2)+pow((ypoint-yTemp),2))
            distance_matrix = np.vstack((np.array(range(len(candidate_center))),distance))
            distance_matrix = np.transpose(distance_matrix)
            distance_sort = distance_matrix[distance_matrix[:,1].argsort()] 
            distance_sort = np.delete(distance_sort,0,axis = 0)
            thre_matrix = distance_sort[distance_sort[:,1]<=threshold_dis]
            thre_point = thre_matrix[:,0]
            thre_point = thre_point.astype(np.int)
            thre_point = thre_point.tolist()
            thre_point.insert(0,i)
            group_distance.append(thre_point)
        
        group_clustering = [[]] 
        
        for i in range(len(candidate_center)):
            m1 = group_distance[i]
            for j in range(len(group_clustering)):
                m2 = group_clustering[j]
                com = set(m2).intersection(set(m1))
                if len(com) == 0:
                    if j == len(group_clustering)-1:
                        group_clustering.append(m1)
                else:
                    m = set(m1).union(set(m2))
                    group_clustering[j] = []
                    group_clustering[j] = list(m)
                    break
        group_clustering.pop(0)
        return group_clustering  #the group of candiate center

    #the function of clustering the final center
    def center_clustering(candidate_center,group_clustering):
        final_result = []
        for i in range(len(group_clustering)): 
            points_coord = candidate_center[group_clustering[i]]
            xz = points_coord[:,1] 
            yz = points_coord[:,0]
            x_mean = np.mean(xz)
            y_mean = np.mean(yz)
            final_result.append([y_mean,x_mean])
        final_result = np.array(final_result)
        final_result = final_result.astype(np.int)
        return final_result # the matrix of final center of steel bars

    # the function of showing the result, include the result of candidate center, the bounding-box of clustering, the center of clustering
    def show_original_red_point(img_original,candidate_center):        
        for i in range(len(candidate_center)):
            cv2.circle(img_original,(candidate_center[i,1],candidate_center[i,0]),2,(0,0,255),-1)        
        cv2.imwrite("result/resultoriginal.png", img_original)
        
    def show_green_box(img_original,candidate_center,group_clustering):        
        for i in range(len(candidate_center)):
            cv2.circle(img_original,(candidate_center[i,1],candidate_center[i,0]),2,(0,0,255),-1)
        for i in range(len(group_clustering)):
            points_coord = candidate_center[group_clustering[i]]
            xz = points_coord[:,1] 
            yz = points_coord[:,0]
            #cv2.putText(img_original, "#{}".format(i), (int(xz) - 10, int(yz)),
		#cv2.FONT_HERSHEY_SIMPLEX, 0.2, (0, 0, 255), 2)
            cv2.rectangle(img_original,(min(xz)-5,min(yz)-5),(max(xz)+5,max(yz)+5),(0,255,0))        
        cv2.imwrite("result/resultgreenbox.png", img_original)
        return len(group_clustering)
        #print(len(group_clustering))
            
    def show_clustering_red_point(img_original,center_cluster):
        for i in range(len(center_cluster)):
            cv2.circle(img_original,(center_cluster[i,1],center_cluster[i,0]),2,(0,0,255),-1)
        cv2.imwrite("result/resultclustering.png", img_original)

    def counting(self):
        img_original = cv2.imread('image/counting.png')
        stride = 6 #the parameter of slide window stride
        candidate_center = CountingWindow.detection(img_original,stride)   
        distance_threshold = 20 # the parameter of distance clustering threshold
        group_clustering = CountingWindow.clustering(candidate_center,distance_threshold)       
        center_cluster = CountingWindow.center_clustering(candidate_center,group_clustering)        
        total = CountingWindow.show_green_box(img_original,candidate_center,group_clustering) 
        self.lbl_Total.text = str("Results :{} ".format(total))
        self.ids.image.source = "result/resultgreenbox.png"
        

class WindowManager(ScreenManager):
    pass

kv = Builder.load_file("my.kv")

class CountingApp(App):
    def build(self):
        return kv

if __name__ == "__main__":
    CountingApp().run()
