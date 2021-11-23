import os
import sys

import numpy as np

###################
import texture, material, util



def readObj(fname):
    material_dict = []
    with open(fname, 'r') as f:
        parsed = [line.split('\n') for line in f.readlines()]
        # filter out empty lines
        parsed = [list(filter(None, line)) for line in parsed]
        parsed = [line for line in parsed if line]
        parsed = [line[0].split(' ') for line in parsed]
        vertex = np.array([line[1:] for line in parsed if line[0] == 'v'], dtype = np.float32)
    
        normal = np.array([line[1:] for line in parsed if line[0] == 'vn'], dtype = np.float32)
        texcoord = np.array([line[1:] for line in parsed if line[0] == 'vt'], dtype = np.float32)
        texcoord[:,1] = np.ones((1, texcoord[:,1].shape[0]),dtype = np.float32) - texcoord[:,1]  

        mtl = [line[1] for line in parsed if line[0] =='usemtl']
        for mm in mtl:
            material_dict += material.load_mtl(os.path.join(os.path.dirname(fname), line[1]), True)

        def _find_mat(name):
            for mat in material_dict:
                if name == mat['name']:
                    return mat
        
        vface, tface, nface, mface = [], [], [], []
        for line in parsed:
            if line[0] == 'usemtl':
                mat = _find_mat(line[1])
                matIdx = material_dict.index(mat)

            if line[0] == 'f':
                tmp = line[1:]
                nv = len(tmp)
                vv = tmp[0].split('/')
                v0 = int(vv[0]) - 1
                t0 = int(vv[1]) - 1 if vv[1] else -1
                n0 = int(vv[2]) - 1 if vv[2] else -1
                for i in range(nv - 2):
                    vv = tmp[i+1].split('/')
                    v1 = int(vv[0]) - 1
                    t1 = int(vv[1]) - 1 if vv[1] else -1
                    n1 = int(vv[2]) - 1 if vv[2] else -1
                    vv = tmp[i+2].split('/')
                    v2 = int(vv[0]) - 1
                    t2 = int(vv[1]) - 1 if vv[1] else -1
                    n2 = int(vv[2]) - 1 if vv[2] else -1
                    mface.append(matIdx)
                    vface.append((v0,v1,v2))
                    tface.append((t0,t1,t2))
                    nface.append((n0,n1,n2))

        return material_dict, mface, tface, nface, vertex, normal, texcoord 


