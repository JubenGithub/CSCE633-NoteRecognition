from PIL import Image, ImageDraw
import os
def draw(file, p1, p2):
    class_name = file.readline()
    class_name = class_name.replace('\n', '')

    lx = []
    ly = []
    minx = 256
    miny = 256
    maxx = 0
    maxy = 0
    for line in file:
        fx = -1
        fy = -1
        xy = line.split(';')
        x = []
        y = []
        for co in xy:
            try:
                intx, inty = co.split(',')
                x.append(int(intx))
                y.append(int(inty))
                minx = min(minx, int(intx))
                miny = min(miny, int(inty))
                maxx = max(maxx, int(intx))
                maxy = max(maxy, int(inty))
            except ValueError:
                break
        lx.append(x)
        ly.append(y)
    maxw = maxx - minx
    maxh = maxy - miny
    border = max(maxw, maxh)+2
    im = Image.new('RGBA', (border, border), (255, 255, 255, 255))
    draw = ImageDraw.Draw(im)
    centerx = (maxx + minx)/2
    centery = (maxy + miny)/2
    disx = centerx - (border)/2
    disy = centery - (border)/2
    for k in range(0, len(lx)):
        for lk in range(0, len(lx[k])):
            lx[k][lk] = lx[k][lk]-disx
            ly[k][lk] = ly[k][lk]-disy
    for k in range(0, len(lx)):
        for lk in range(1, len(lx[k])):
            draw.line([(lx[k][lk-1], ly[k][lk-1]), (lx[k][lk], ly[k][lk])], fill=(0, 0, 0, 255), width=2)
    path_name = class_name
    file_name = "/"+str(p1)+ "-" +str(p2)+".png"
    im = im.resize((64, 64), resample=3)
    im.thumbnail((64, 64))

    try:
        im.save(str(path_name+file_name))
    except IOError:
        os.mkdir(path_name)
        im.save(str(path_name+file_name))

p1 = 1
p2 = 1
while p1 <= 100 :
    p2 = 1
    while 1 :
        try:
            path = "HOMUS/"+str(p1) + "/" + str(p1)+"-" + str(p2) + ".txt"
            file = open(path, 'r')
            draw(file, p1, p2)
        except IOError:
            p1 = p1+1
            break
        p2 = p2+1
#print path


#im = Image.new('RGBA', (400, 400), (0, 255, 0, 0))
#draw = ImageDraw.Draw(im)
#draw.line((100,200, 150,300), fill=128)
#im.show()
