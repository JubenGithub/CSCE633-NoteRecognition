import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
"""
def segment(segpiece = 2, image, pixels):
    seg_image = []
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    return
def DFSsegment(boundingbox, x, y, width, height, pixels):
    if x < 0 or x > width:
        return
    if y < 0 or y > height:
        return
    if pixels[y][x][0]< 125:
        DFSsegment(boundingbox, x+1, y, width, height)
        DFSsegment(boundingbox, x, y+1, width, height)
        DFSsegment(boundingbox, x+1, y+1, width, height)
        DFSsegment(boundingbox, x-1, y+1, width, height)
        DFSsegment(boundingbox, x-1, y-1, width, height)
        DFSsegment(boundingbox, x+1, y-1, width, height)
        DFSsegment(boundingbox, x, y-1, width, height)
        DFSsegment(boundingbox, x-1, y, width, height)
    return
"""
def pad_64_image(cropim, cropwidth, cropheight):
    padimage = Image.new("RGBA", (64, 64), (255, 255, 255, 255))
    if cropwidth > cropheight:
        newheight = int(cropheight*(64/cropwidth))
        cropim = cropim.resize((64, newheight), Image.ANTIALIAS)
        padding =(64 - newheight)/2
        padimage.paste(cropim, (0, int(padding)))
    else:
        newwidth = int(cropwidth*(64/cropheight))
        cropim = cropim.resize((newwidth, 64), Image.ANTIALIAS)
        padding =(64 - newwidth)/2
        padimage.paste(cropim, (int(padding), 0))
    return padimage

def crop_image(bounding_width = 5, segmentation_width = 3, filename = 'test_4.png'):
    crop_image = []
    im = Image.open(filename)
    pixels = list(im.getdata())
    width, height = im.size
    pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
    projection = [0]*width
    for h in range(height):
        for w in range(width):
            if pixels[h][w][0] < 125:
                projection[w]+=1
#im.show()
    #segment projection

    score_segment = []
    left = 0
    right = 0
    count_zero = 0
    for index, p in enumerate(projection):
        if p != 0:
            count_zero = 0
            if left == 0:
                left = index
            else:
                right = index
        else:
            count_zero+=1
            if count_zero >= segmentation_width:
                if left != right:
                    score_segment.append([left, right])
                left = 0
                right = 0
    for inf in score_segment:
        box =(inf[0], 0, inf[1] ,height)
        newim = im.crop(box)
        ver_projection = [0]*height
        pixels = list(newim.getdata())
        width, height = newim.size
        #padd the segment to ideal size
        newh = 0
        newb = 0
        pixels = [pixels[i * width:(i + 1) * width] for i in range(height)]
        for h in range(height):
            for w in range(width):
                if pixels[h][w][0] < 125:
                    ver_projection[h]+=1
            if ver_projection[h] != 0:
                if newh == 0:
                    newh = h-1
                else:
                    newb = h+1
        box = (0, newh, width, newb)
        cropim = newim.crop(box)
        cropheight = newb - newh
        cropwidth = width
        padimage = pad_64_image(cropim, cropwidth, cropheight)
#padimage.show()
        crop_image.append(padimage)
    return crop_image
    #print(pixels)
