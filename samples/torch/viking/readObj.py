import os
import sys

import numpy as np

###################
import texture, material, util
import torch



def readObj(fname):
    """
    Args:
    fname: path to the obj file
    Returns:
    material_dict : list of dict of materials, each element in the list is material type
    np.array(mface, dtype = np.int32) : index of materials of each faces (N, )
    np.array(vface, dtype = np.int32) : vertex face (N, 3)
    np.array(tface, dtype = np.int32) : texture face (N, 3)
    np.array(nface, np.int32) : normal face (N, 3)
    vertex : #vertex x 3
    normal : #vn x 3
    texcoord : #vt x 2
    """

    material_dict = [
        {
            'name' : '_default_mat',
            'bsdf' : 'falcor',
            'kd'   : texture.Texture2D(torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32, device='cuda')),
            'ks'   : texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
        }
    ]
    normal, texcoord, mtl = None, None, None
    with open(fname, 'r') as f:
        parsed = [line.split('\n') for line in f.readlines()]
        # filter out empty lines
        parsed = [list(filter(None, line)) for line in parsed]
        parsed = [line for line in parsed if line]
        parsed = [line[0].split(' ') for line in parsed]
        vertex = np.array([line[1:] for line in parsed if line[0] == 'v'], dtype = np.float32)
    
        normal = np.array([line[1:] for line in parsed if line[0] == 'vn'], dtype = np.float32)
        texcoord = np.array([line[1:] for line in parsed if line[0] == 'vt'], dtype = np.float32)
 
        if texcoord.size != 0:
            texcoord[:,1] = np.ones((1, texcoord[:,1].shape[0]),dtype = np.float32) - texcoord[:,1]  

        mtl = [line[1] for line in parsed if line[0] =='mtllib']
        for mm in mtl:
            material_dict += material.load_mtl(os.path.join(os.path.dirname(fname), mtl[0]), True)

        def _find_mat(name):
            for mat in material_dict:
                if name == mat['name']:
                    return mat
            return material_dict[0]
        
        vface, tface, nface, mface = [], [], [], []
        matIdx = -1
        for line in parsed:
            if line[0] == 'usemtl':
                mat = _find_mat(line[1])
                matIdx = material_dict.index(mat)

            if line[0] == 'f':
                tmp = line[1:]
               
                nv = len(tmp)
                vv = tmp[0].split('/')
                v0 = int(vv[0]) - 1
                if(len(vv) > 1):
                    t0 = int(vv[1]) - 1
                    n0 = int(vv[2]) - 1
                else:
                    t0 = -1
                    n0 = -1
                for i in range(nv - 2):
                    vv = tmp[i+1].split('/')
                    v1 = int(vv[0]) - 1
                    if(len(vv) > 1):
                        t1 = int(vv[1]) - 1
                        n1 = int(vv[2]) - 1
                    else:
                        t1 = -1
                        n1 = -1

                    vv = tmp[i+2].split('/')
                    v2 = int(vv[0]) - 1
                    if(len(vv) > 1):
                        t2 = int(vv[1]) - 1
                        n2 = int(vv[2]) - 1
                    else:
                        t2 = -1
                        n2 = -1 
                    mface.append(matIdx)
                    vface.append((v0,v1,v2))
                    tface.append((t0,t1,t2))
                    nface.append((n0,n1,n2))

        return {'material' : material_dict, \
                'mface' : np.array(mface, dtype = np.int32), \
                'vface' : np.array(vface, dtype = np.int32), \
                'tface' : np.array(tface, dtype = np.int32), \
                'nface' : np.array(nface, np.int32), \
                'vertex' : vertex, \
                'normal' : normal, \
                'texcoord' : texcoord} 

