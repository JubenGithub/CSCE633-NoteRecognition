import os
from muscima.io import parse_cropobject_list
import itertools
import numpy
import matplotlib.pyplot as plt



# Change this to reflect wherever your MUSCIMA++ data lives
CROPOBJECT_DIR = os.path.join(os.environ['HOME'], '/Users/jeanhsu/Documents/csce633/csce633/633project/v1.0/data/cropobjects_withstaff')
cropobject_fnames = [os.path.join(CROPOBJECT_DIR, f) for f in os.listdir(CROPOBJECT_DIR)]
print "dic"
docs = [parse_cropobject_list(f) for f in cropobject_fnames]
print "dic2"
# Bear in mind that the outlinks are integers, only valid within the same document.
# Therefore, we define a function per-document, not per-dataset.
def extract_notes_from_doc(cropobjects, count_e, count_s, count_t):
    """Finds all ``(full-notehead, stem)`` pairs that form
        quarter or half notes. Returns two lists of CropObject tuples:
        one for quarter notes, one of half notes.
        
        :returns: quarter_notes, half_notes
        """
    print "generate file"
    _cropobj_dict = {c.objid: c for c in cropobjects}
    for c in cropobjects:
        if (c.clsname == 'notehead-full'):
            _has_stem = False
            _has_beam = False
            stem_obj = None
            beam_objs = []
            for o in c.outlinks:
                _o_obj = _cropobj_dict[o]
                if _o_obj.clsname == 'stem':
                    _has_stem = True
                    stem_obj = _o_obj
                elif _o_obj.clsname == 'beam':
                    _has_beam = True
                    beam_objs.append(_o_obj)
            if _has_stem and _has_beam:
                if len(stem_obj.inlinks) == 1 and len(beam_objs) == 1:
                    eightth_note = (c, stem_obj, beam_objs[0])
                    e_image = get_image(eightth_note)
                    save_mask(e_image, count_e,"Compound_Single_Eighteenth_Note")
                    count_e = count_e+1
                elif len(stem_obj.inlinks) == 1 and len(beam_objs) == 2:
                    sixteenth_note = (c, stem_obj, beam_objs[0], beam_objs[1])
                    s_image = get_image(sixteenth_note)
                    save_mask(s_image, count_s, "Compound_Single_Sixteenth_Note")
                    count_s = count_s+1

                elif len(stem_obj.inlinks) == 1 and len(beam_objs) == 3:
                    thirtytwo_note = (c, stem_obj, beam_objs[0], beam_objs[1], beam_objs[2])
                    t_image = get_image(thirtytwo_note)
                    save_mask(t_image, count_t, "Compound_Single_Thirtytwo_Note")
                    count_t = count_t+1
    return count_e, count_s, count_t

"""
    notes = []
    if (c.clsname == 'beam'):
    for o in c.inlinks:
    
"""

def get_image(cropobjects, margin=1):
    """Paste the cropobjects' mask onto a shared canvas.
        There will be a given margin of background on the edges."""
    
    # Get the bounding box into which all the objects fit
    top = min([c.top for c in cropobjects])
    left = min([c.left for c in cropobjects])
    bottom = max([c.bottom for c in cropobjects])
    right = max([c.right for c in cropobjects])
    
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
    fig.set_size_inches(5,5)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    filestring = directory+"/"+str(i)+".png"
    plt.imshow(mask, cmap='gray')
    try:
        fig.savefig(filestring)
    except IOError:
        os.mkdir(directory)
        fig.savefig(filestring, bbox_inches='tight', transparent=True, dpi=51)
    plt.close(fig)

def show_mask(mask):
    plt.imshow(mask, cmap='gray', interpolation='nearest')
    plt.show()

def save_masks(masks, directory):
    for i, mask in enumerate(masks):
        fig = plt.figure(frameon=False)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        filestring = directory+"/"+str(i)+".png"
        plt.imshow(mask, cmap='gray')
        try:
            fig.savefig(filestring)
        except IOError:
            os.mkdir(directory)
            fig.savefig(filestring, bbox_inches='tight', transparent=True, dpi = 1)
        plt.close(fig)

count_e = 0
count_s = 0
count_t = 0
for cropobjects in docs:
    count_e, count_s, count_t = extract_notes_from_doc(cropobjects, count_e, count_s, count_t)
