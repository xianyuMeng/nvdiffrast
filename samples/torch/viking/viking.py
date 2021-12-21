import argparse
import os
import sys

import numpy as np
import torch

###################
import util
import nvdiffrast.torch as dr

import Obj

import pdb

proj_mtx = util.projection(x=0.4, f=1000.0)

def printlog(log_fn, data):
    print(data)
    log_fn.write(data + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', help = 'output dir', default = './')
    parser.add_argument('--ref', help = 'path to the high-res mesh', required = True)
    parser.add_argument('--base', help = 'path to the low-res mesh', required = True)
    parser.add_argument('--log', help = 'log file name', default = 'log.txt')
    parser.add_argument('--b', help = 'batch size', default = 1, type = int)
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok = True)
    log_fn = open(os.path.join(args.outdir,args.log), 'wt')

    opt_viking(log_fn, args.ref, args.base, args.outdir, batch = args.b)
    


def transform_pos(mtx, pos):
    pdb.set_trace()
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    #t_mtx = torch.from_numpy(mtx).cuda()
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    #(x, y, z) -> (x, y, z, 1)
    #output : (1, num_vertices, 4)
    out = torch.matmul(posw, t_mtx.permute(2, 1, 0))
    return out.permute(2, 1, 0)

def render(glctx, mtx, vertex, triangle, texcoord,uv_idx, tex, resolution: int):
    pdb.set_trace()
    vertex_clip    = transform_pos(mtx, vertex)
    #vertex_clip (1, num_vertices, 3, 1)
    # glctx, pos, tri, resolution, ranges=None, grad_db=True
    rast_out, rast_db = dr.rasterize(glctx,vertex_clip.contiguous() , triangle , resolution=[resolution, resolution])
    # rast_db : shape [minibatch_size, height, width, 4] and contain said derivatives in order (du/dX, du/dY, dv/dX, dv/dY)
    # attr, rast, tri, rast_db=None, diff_attrs=None
    #interpolate_db  [minibatch_size, height, width, 2 * len(diff_attrs)]. The derivatives of the first selected attribute A will be on channels 0 and 1 as (dA/dX, dA/dY), etc.
    color, interpolate_db = dr.interpolate(texcoord[None, ...], rast_out, uv_idx, rast_db = rast_db, diff_attrs='all')
    texture = dr.texture(tex[None, ...], color, interpolate_db, filter_mode='linear-mipmap-linear')
    texture = texture * torch.clamp(rast_out[..., -1:], 0, 1)
    return texture

def buildMtx(batch = 1, proj_mtx = proj_mtx, radius = 3.5):
    #Randomly generate rotation/translation matrix
    mvp = np.zeros((batch, 4, 4),dtype = np.float32)
    campos = np.zeros((batch, 3), dtype = np.float32)
    lightpos = np.zeros((batch, 3), dtype = np.float32)
    for b in range(batch):
        r_rot = util.random_rotation_translation(0.25)
        r_mv = np.matmul(util.translate(0, 0, radius), r_rot)
        mvp[b] = np.matmul(proj_mtx, r_mv).astype(np.float32)
        campos[b] = np.linalg.inv(r_mv)[:3, 3]
        lightpos[b] = util.cosine_sample(campos[b]) * radius
    return mvp, campos, lightpos


def opt_viking(
        log_fn,
        ref,
        base,
        output_dir,
        batch=1,
        resolution=256):
    #material_dict, mface, vface, tface, nface, vertex, normal, texcoord = readObj(ref)

    ref_mesh = Obj.readObj(ref)
    base_mesh = Obj.readObj(base)


    # Create position/triangle index tensors
    triangles = torch.from_numpy(ref_mesh['vface'].astype(np.int32)).cuda()
    vertex = torch.from_numpy(ref_mesh['vertex'].astype(np.float32)).cuda()
    uv_idx = torch.from_numpy(ref_mesh['tface'].astype(np.int32)).cuda()
    texcoord = torch.from_numpy(ref_mesh['texcoord']).cuda()
    
    pdb.set_trace()

    #triangles_opt = torch.from_numpy(base_mesh['vface'].astype(np.int32)).cuda()
    #vertex_opt = torch.from_numpy(base_mesh['vertex'].astype(np.float32)).cuda()
    tex = torch.full((512, 512, 3), 0.2, device='cuda', requires_grad=True)

    glctx = dr.RasterizeGLContext()

    ang = 0.0



##########################################################################################
    


    # Render.
    max_iter = 20000
    
    ang = 0.0
    texloss_avg = []
    for it in range(max_iter + 1):
        mvp, campos, lightpos = buildMtx(batch = 1)
        color_opt = render(glctx, mvp, vertex.contiguous(), triangles.contiguous(), texcoord.contiguous(), uv_idx.contiguous(), tex, resolution = resolution)
        img_opt = color_opt[0].detach().cpu().numpy()[::-1]
        util.save_image(os.path.join(output_dir, '%d.png'%it), img_opt)

           


    

if __name__ == "__main__":
    main()


