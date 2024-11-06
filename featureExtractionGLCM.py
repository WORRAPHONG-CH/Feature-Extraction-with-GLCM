# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 13:22:33 2022

@author: Worra
"""

import cv2
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from operator import itemgetter
from scipy.spatial import distance
from skimage.feature import greycomatrix , greycoprops


def thVal1(th1):
    pass
def thVal2(th2):
    pass

cap  = cv2.VideoCapture(2)

cv2.namedWindow("Trackbar",cv2.WINDOW_FREERATIO)
cv2.namedWindow("Blur(Gaussian)",cv2.WINDOW_FREERATIO)
cv2.namedWindow("Threshold",cv2.WINDOW_FREERATIO)
cv2.namedWindow("Erosion",cv2.WINDOW_FREERATIO)

cv2.createTrackbar("Th1","Trackbar",130,255,thVal1)
cv2.createTrackbar("Th2","Trackbar",230,255,thVal2)
cv2.createTrackbar("Th","Trackbar",200,255,thVal1)
cv2.createTrackbar("area1","Trackbar",5000,10000,thVal1)
cv2.createTrackbar("area2","Trackbar",18000,50000,thVal1)
name_count = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # Set config camera and frame
        frame = cv2.resize(frame,(400,300))
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),1)
        thval1 = cv2.getTrackbarPos("Th1","Trackbar")
        thval2 = cv2.getTrackbarPos("Th2","Trackbar")
        th0 = cv2.getTrackbarPos("Th","Trackbar")
        thersh, th = cv2.threshold(blur,th0,255,cv2.THRESH_BINARY)
        
        # Use 3 algorithms to post-processing frame in camera
        kernel = np.ones((3,3),np.uint8)
        erode = cv2.erode(th,kernel,iterations=2)
        canny = cv2.Canny(erode,thval1,thval2)
        
        frame_contour = frame.copy()
        contours, hierarchy = cv2.findContours(erode,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        
        # Find contour of interested area
        biggest = []
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            area1 = cv2.getTrackbarPos("area1","Trackbar")
            area2 = cv2.getTrackbarPos("area2","Trackbar")
            
            # Approximate corner point according to interested rea
            if area > area1 and area < area2:
                #print(area)
                cv2.drawContours(frame_contour,cnt,-1,(0,255,0),2)
                length_con = cv2.arcLength(cnt,True)
                edge = cv2.approxPolyDP(cnt,0.02*length_con,True)#aprox point(edge)
                if area > max_area and len(edge) == 4:
                    biggest = edge
                    max_area = area
                    #print(biggest)
                    
        if len(biggest) != 0:
            cornner = cv2.drawContours(frame_contour,biggest,-1,(0,0,255),3)                
            #new point 
            biggest = biggest.reshape((4,2))
            #print("general:",biggest)
            #biggest_new = sorted(biggest,key=itemgetter(0))
            biggest_new = sorted(biggest,key=itemgetter(1))#sort array y
            #print("sort:",biggest_new)
            for t in range(len(biggest)):
                text = str(t)
                pt_biggest = tuple(biggest_new[t])
                cv2.putText(frame_contour,text,pt_biggest,cv2.FONT_HERSHEY_COMPLEX,0.5,(0,255,0),2)
            
            cv2.line(frame_contour,tuple(biggest_new[2]),tuple(biggest_new[0]),(255,0,255),2)
            cv2.line(frame_contour,tuple(biggest_new[1]),tuple(biggest_new[0]),(255,0,255),2)
            cv2.line(frame_contour,tuple(biggest_new[3]),tuple(biggest_new[1]),(255,0,255),2)
            cv2.line(frame_contour,tuple(biggest_new[3]),tuple(biggest_new[2]),(255,0,255),2)
            
            cv2.namedWindow("Warp Perspective",cv2.WINDOW_FREERATIO)
            warp_frame = frame.copy()
            
            # Get 4 corners
            pts1 = np.float32(biggest_new)
            p01 = np.round(distance.euclidean(pts1[0],pts1[1]),decimals=2)
            p02 = np.round(distance.euclidean(pts1[0],pts1[2]),decimals=2)
           
            # Find proper corner to perspective warping
            if p01 > p02:
                pts2 = np.float32([[900,800],[900,0],[0,800],[0,0]])
            elif p01 < p02:
                pts2 = np.float32([[900,0],[0,0],[900,800],[0,800]])
                
            matrix = cv2.getPerspectiveTransform(pts1,pts2)
            result = cv2.warpPerspective(warp_frame,matrix,(900,800))
            cv2.imshow("Warp Perspective",result)
                    
        # Display post-process image in each windows
        cv2.imshow("Blur(Gaussian)",blur)
        cv2.imshow("Threshold",th)
        cv2.imshow("Erosion",erode)
        cv2.imshow("Canny",canny)
        cv2.imshow("Contour",frame_contour)
        
        button = cv2.waitKey(1)

        # Press ESC to close all windows
        if button & 0xFF == 27:
            break
        
        # Press "q" to save images
        elif button & 0xFF == ord("q"):
            img_name = "Picture{}.png".format(name_count)
            cv2.imwrite(img_name,result)
            print("!!Pic{} written!!".format(name_count))
            name_count +=1
        
        # Press "e" to analyze and compare data with GLCM and graph
        elif button & 0xFF == ord("e"):
            glcm_listBig =[]
            r_list,g_list,b_list = [],[],[]
            
            img_list= []

            for i in range(name_count):
                img = cv2.imread("Picture{}.png".format(i))
                print("read img {}".format(i))
                grayF = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                grayF = (grayF/32).astype(np.uint8)
                b = img[:,:,0]
                g = img[:,:,1]
                r = img[:,:,2]
                b_list.append(b)
                g_list.append(g)
                r_list.append(r)
                img_list.append(img)
                
            # skimage.feature.texture.greycomatrix(image, distances(pixelที่ติดกัน), angles(0,90), levels=256, symmetric=False, normed=False)
                glcm = greycomatrix(grayF,[1],[0],levels = 8,normed = True)
                contrast = greycoprops(glcm, 'contrast')
                dissimilar = greycoprops(glcm, 'dissimilarity')
                homo = greycoprops(glcm, 'homogeneity')
                asm = greycoprops(glcm, 'ASM')
                energy = greycoprops(glcm, 'energy')
                correlation = greycoprops(glcm,'correlation')
                glcmlist = [contrast,dissimilar,homo,asm,energy,correlation]
                glcm_listBig.append(glcmlist)
                
            name = ["contrast","dissimilarity","homogeneity","ASM","energy","correlation"]
            
            # Plot graph to provide GLCM detail
            plt.figure("GLCM {} img".format(len(glcm_listBig)),figsize=(7,5))#(figsize=(w,h))
            for j in range(len(glcm_listBig)):
                glcm_listplot = np.reshape(glcm_listBig[j],-1)
                plt.plot(name,glcm_listplot,label="Pic{}".format(j))
                plt.xlabel("GLCM")
                plt.ylabel("Value(normallized)")
                plt.title("GLCM Plot")
                plt.legend()
                
            hist_list = []
                
            colors = ["red", "green", "blue"]
            channel = np.arange(0,3,1,"uint8")#np.arrage(start,stop,step,type)

            for k in range(len(img_list)):
                plt.figure("Pic{}".format(k+1),figsize=(5,3))
                image = img_list[k]
                for c,ch in zip(colors,channel):
                    histogram, bin_edges = np.histogram(image[:,:,ch], bins=256, range=(0, 256))
                    hist_list.append(histogram)
                    plt.plot(bin_edges[0:-1], histogram,color = c)#bin_edge ->(length(hist)+1).
                    plt.xlabel("Value")
                    plt.ylabel("Pixel")
                    plt.xlim([0,255])
                    plt.title("Color Plot")
                    plt.legend(colors)
            print("!!Already Plot!!")
        elif button & 0xFF == ord("r"):
            #GLCM(Prepare data before doing dataframe)
            contrast_list = list(map(itemgetter(0),glcm_listBig))
            dissim_list = list(map(itemgetter(1),glcm_listBig))
            homo_list = list(map(itemgetter(2),glcm_listBig))
            ASM_list = list(map(itemgetter(3),glcm_listBig))
            energy_list = list(map(itemgetter(4),glcm_listBig))
            corre_list = list(map(itemgetter(5),glcm_listBig))
    
            # COLOR(Prepare data before dataframe)
            list_colorB = []
            ind_color = []
            for h in range(len(img_list)):
                list_color = list(zip(hist_list[h],hist_list[h+1],hist_list[h+2]))
                list_colorB.append(list_color)
                ind = "Picture{}".format(h)
                ind_color.append(ind)
                
            # Prepare rows and columns
            dict_glcm = {"Contrast":contrast_list,"Dissimilarity":dissim_list,"Homogeneity":homo_list,"ASM":ASM_list,"energy":energy_list,"correlation":corre_list}
            df_glcm = pd.DataFrame(dict_glcm)
            print(df_glcm)
                
            df_color = pd.DataFrame()#,columns=ind_color
            for d in range((len(img_list))):
                df_color[d] = list_colorB[d]
                print("d:",d)
                
            df_color = df_color.transpose()
    
            df_glcm.to_csv("Data_GLCM.csv",header = True,index = True)
            df_color.to_csv("Data_Color.csv",header = True,index = True)
            print("!!Writen to csv file!!")
    else:
        print("Fail")
        break

cap.release()
cv2.destroyAllWindows()




