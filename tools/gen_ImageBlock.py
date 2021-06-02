#coding:utf-8
from PIL import Image
import numpy as np
from skimage import measure
from tqdm import *
import time
import os
import multiprocessing as mp
import argparse

def parse_args():
    parser = argparse.ArgumentParser('complete the binary edge map into a closed edge map')
    parser.add_argument('--RawEdge_path',type=str)
    parser.add_argument('--save_path',type=str)
    parser.add_argument('--search_radius',type=int)
    return parser.parse_args()

def icolor(G):
    white = (255)
    black = (0)
    if G == white:
        return (1)
    else:
        return (0)

def whether_connect_inside(u,v,np,cut_num):
    # determine whether the pixel is locally connected to the end point
    np1=np.copy()
    lab=measure.label(np1,connectivity=2)
    a=lab[u,v]
    b=lab[cut_num,cut_num]
    if a==b:
        return True
    else:
        return False

def connect_point(x,y,m,n,d,np):
    # use the simplest DDA method to connect two pixels
    if x==m and y==n:
        return 0
    elif x==m and y!=n:
        grad_x=abs(x-m)
        gradx=(m-x)/d
        grad_y=abs(y-n)
        grady=(n-y)/d
        diry=int(grady/abs(grady))
    elif x!=m and y==n:
        grad_x = abs(x - m)
        gradx = (m - x) / d
        dirx = int(gradx / abs(gradx))
        grad_y = abs(y - n)
        grady = (n - y) / d
    else:
        grad_x=abs(x-m)
        gradx=(m-x)/d
        dirx=int(gradx/abs(gradx))
        grad_y=abs(y-n)
        grady=(n-y)/d
        diry=int(grady/abs(grady))

        if grad_x>grad_y:
            # determine whether to connect with a straight line
            if grad_x==1:
                for i in range(grad_x-1):
                    np[x+dirx*(i+1),y]=255
            else:
                # use diagonal lines to connect two pixels
                y_plus=0
                for i in range(grad_x-1):
                    if (i+1)*abs(grady)-y_plus>1:
                        y_plus=y_plus+1
                        np[x+dirx*(i+1),y+diry*y_plus]=255
                    else:
                        np[x + dirx * (i + 1), y + diry*y_plus] = 255
        else:
            if grad_y == 1:
                for i in range(grad_y - 1):
                    np[x , y+diry*(i+1)] = 255
            else:
                x_plus = 0
                for i in range(grad_y - 1):
                    if (i + 1) * abs(gradx) - x_plus > 1:
                        x_plus = x_plus + 1
                        np[x + dirx*x_plus, y+diry*(i+1)] = 255
                    else:
                        np[x + dirx*x_plus, y+diry*(i+1)] = 255

def deal_per_pic(pic_file,args):

    print('begin deal: '+pic_file)
    cut_num=args.search_radius
    # load binary edge map
    img_path = os.path.join(args.RawEdge_path,pic_file)
    img = Image.open(img_path)
    img_np = np.asarray(img)
    h, w = img_np.shape

    # divid the connected valued pixels into a connected component
    img_use = img_np.copy()
    labels = measure.label(img_use, connectivity=2)

    labels_list=[x for x in range(labels.max()+1)]
    labels_num = len(labels_list)

    # find the endpoints in each connected area
    #pbar = tqdm(total=labels_num, desc='labels')
    for i in range(labels_num - 1):
        i = i + 1
        a = np.argwhere(labels == labels_list[i])
        #pbar.update(1)
        #pba = tqdm(total=len(a), desc='cut')
        for j in range(len(a)):
            #pba.update(1)
            x, y = a[j]
            if x > (cut_num - 1) and x < (h - cut_num) and y > (cut_num - 1) and y < (w - cut_num):
                if (
                        icolor(img_use[x + 1, y]) +
                        icolor(img_use[x, y + 1]) +
                        icolor(img_use[x - 1, y]) +
                        icolor(img_use[x, y - 1]) +
                        icolor(img_use[x - 1, y - 1]) +
                        icolor(img_use[x + 1, y - 1]) +
                        icolor(img_use[x - 1, y + 1]) +
                        icolor(img_use[x + 1, y + 1])
                ) > 2:
                    continue
                else:
                    # search for valued pixels in the area centered on this pixel
                    wait_connect_list = []
                    for m in range(-cut_num, cut_num + 1, 1):
                        for n in range(-cut_num, cut_num + 1, 1):
                            if icolor(img_use[x + m, y + n]) == 1:
                                wait_connect_list.append([x + m, y + n])
                    if len(wait_connect_list) > 0:
                        # divide the set of pixels to be connected into a set of pixels in the connected
                        # domain and a set of pixels outside the connected domain
                        wait_connect_list_out = []
                        wait_connect_list_in = []
                        wait_distance_out = []
                        wait_distance_in = []
                        for k in range(len(wait_connect_list)):
                            if labels[wait_connect_list[k][0], wait_connect_list[k][1]] != labels[x, y]:
                                wait_connect_list_out.append([wait_connect_list[k][0], wait_connect_list[k][1]])
                                wait_distance_out.append(
                                    ((wait_connect_list[k][0] - x) ** 2 + (wait_connect_list[k][1] - y) ** 2) ** 0.5)
                            else:
                                wait_connect_list_in.append([wait_connect_list[k][0], wait_connect_list[k][1]])
                                wait_distance_in.append(
                                    ((wait_connect_list[k][0] - x) ** 2 + (wait_connect_list[k][1] - y) ** 2) ** 0.5)
                        # find the nearest pixel from 'wait_connect_list_out' and connect it to the end point
                        if len(wait_connect_list_out) > 0:
                            wait_out_min_ind = wait_distance_out.index(min(wait_distance_out))
                            connect_point(x, y, wait_connect_list_out[wait_out_min_ind][0],
                                          wait_connect_list_out[wait_out_min_ind][1],
                                          wait_distance_out[wait_out_min_ind], img_use)
                        # find the nearest pixel from 'wait_connect_list_in' and connect it to the end point
                        if len(wait_connect_list_in) > 0:
                            img_local = img_use[x - cut_num:x + cut_num + 1, y - cut_num:y + cut_num + 1]
                            # wait_in_min=min(wait_distance_in)
                            # wait_in_min_ind=wait_distance_in.index(min(wait_distance_in))
                            wait_distance_in_np = np.array(wait_distance_in)
                            wait_distance_in_sort = sorted(wait_distance_in)
                            wait_distance_in_sort_ind = np.argsort(wait_distance_in_np)
                            for r in range(len(wait_connect_list_in)):
                                # determine if the candidate pixel is in the local connected domain of the end point, and connect them if it is not
                                pix_x = wait_connect_list_in[wait_distance_in_sort_ind[r]][0]
                                pix_y = wait_connect_list_in[wait_distance_in_sort_ind[r]][1]
                                if whether_connect_inside(pix_x - x + cut_num, pix_y - y + cut_num, img_local,
                                                          cut_num) == False:
                                    connect_point(x, y, pix_x, pix_y, wait_distance_in_sort[r], img_use)
                                    pass
    #pbar.close()
    img_save = Image.fromarray(img_use)
    img_save_path = os.path.join(args.save_path,pic_file)
    img_save.save(img_save_path)

def main():
    args=parse_args()

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    de_connect_file=os.listdir(args.RawEdge_path)
    for i in de_connect_file:
        deal_per_pic(i,args)

if __name__ == '__main__':
    main()








