import os
from muscima.io import parse_cropobject_list
import itertools
import numpy
import matplotlib.pyplot as plt



# Change this to reflect wherever your MUSCIMA++ data lives

#docs = [parse_cropobject_list(f) for f in cropobject_fnames]

# Bear in mind that the outlinks are integers, only valid within the same document.
# Therefore, we define a function per-document, not per-dataset.
def extract_notes_from_doc(cropobjects, count):
    """Finds all ``(full-notehead, stem)`` pairs that form
        quarter or half notes. Returns two lists of CropObject tuples:
        one for quarter notes, one of half notes.
        
        :returns: quarter_notes, half_notes
        """
    objs = []
    _cropobj_dict = {c.objid: c for c in cropobjects}
    for c in cropobjects:
        count = get_staffs(c, count, cropobjects, _cropobj_dict)
    return count

"""
    notes = []
    if (c.clsname == 'beam'):
    for o in c.inlinks:
    
    """
def get_staffs(c, count, cropobjects, _cropobj_dict):
    if c.clsname != "staff":
        return count
    staffobj = []
    for obj in c.inlinks:
        c_obj =_cropobj_dict[obj]
        if "rest" in c_obj.clsname or "clef" in c_obj.clsname or "signature" in c_obj.clsname:
            staffobj.append(c_obj)
        if c_obj.clsname == "notehead-empty" or c_obj.clsname == "notehead-full":
            for out in c_obj.outlinks:
                oc_obj = _cropobj_dict[out]
                if "stem" in oc_obj.clsname:
                    noteheadcount = 0
                    for link in oc_obj.inlinks:
                        if "notehead" in _cropobj_dict[link].clsname:
                            noteheadcount+=1
                    if noteheadcount >1:
                        print "chord found, return"
                        return count # has chord, get rid of this staff group
                if "beam" in oc_obj.clsname or "stem" in oc_obj.clsname or "dot" in oc_obj.clsname or "flag" in oc_obj.clsname:
                    staffobj.append(oc_obj)
            staffobj.append(c_obj)
    staffobj.append(c)
    withstaff_image = get_image(staffobj, 0)
    withoutstaff_image = get_image(staffobj, 1)
    save_mask(withstaff_image, count, "WithStaff")
    save_mask(withoutstaff_image, count, "WithoutStaff")
    count+=1
    return count
def get_image(cropobjects, withoutstaff, margin=1):
    """Paste the cropobjects' mask onto a shared canvas.
        There will be a given margin of background on the edges."""
    
    # Get the bounding box into which all the objects fit
    top = min([c.top for c in cropobjects])
    left = min([c.left for c in cropobjects])
    bottom = max([c.bottom for c in cropobjects])
    right = max([c.right for c in cropobjects])
    if withoutstaff == 1:
        cropobjects.pop()
    # Create the canvas onto which the masks will be pasted
    height = bottom - top + 2 * margin
    width = right - left + 2 * margin
    canvas = numpy.zeros((height, width), dtype='uint8')
    for c in cropobjects:
        # Get coordinates of upper left corner of the CropObject
        # relative to the canvas
        _pt = c.top - top + margin
        _pl = c.left - left + margin
        # We have to add the mask, so as not to overwrite
        # previous nonzeros when symbol bounding boxes overlap.
        canvas[_pt:_pt+c.height, _pl:_pl+c.width] += c.mask
    
    canvas[canvas > 0] = 2
    canvas[canvas == 0] = 1
    canvas[canvas == 2] = 0
    
    return canvas

def save_mask(mask, i, directory):
    fig = plt.figure(frameon=False)
    #fig.set_size_inches(5,5)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    filestring = directory+"/"+str(i)+".png"
    plt.imshow(mask, cmap='gray')
    try:
        fig.savefig(filestring, transparent=True, dpi=256)
    except IOError:
        os.mkdir(directory)
        fig.savefig(filestring, transparent=True, dpi=256)
    plt.close(fig)

def show_mask(mask):
    plt.imshow(mask, cmap='gray', interpolation='nearest')
    plt.show()


CROPOBJECT_DIR = os.path.join(os.environ['HOME'], '/Users/jeanhsu/Documents/csce633/csce633/633project/v1.0/data/cropobjects_withstaff')
cropobject_fnames = [os.path.join(CROPOBJECT_DIR, f) for f in os.listdir(CROPOBJECT_DIR)]
print "dic"
count = 0
for f in cropobject_fnames:
    cropobjects = parse_cropobject_list(f)
    count = extract_notes_from_doc(cropobjects, count)

