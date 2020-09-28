'''
Interface for custom reconstruction nets. Implement the __init__ and predict methods, then run with --custom_recon_net=True.
'''

class Custom_Recon_Net():
    
    def __init__(self):
        pass
    
    '''
    reconstruct an object given the four channel representation, or use the rgb, depth, and object segmentation to implement custom preprocessing. 
    Return a 128x128x128 occupancy grid.
    '''
    def predict_voxels(self, sphr_projs, four_channel_rep, rgb, depth, segs, cam_mat, cam_pos):
        pass