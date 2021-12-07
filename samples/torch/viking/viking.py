import argparse
import os
import sys

import numpy as np
import torch

###################
import util
import nvdiffrast.torch as dr

import readObj

def printlog(log_fn, data):
    print(data)
    log_fn.write(data + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', help = 'output dir', default = './')
    parser.add_argument('--ref', help = 'path to the high-res mesh', required = True)
    parser.add_argument('--base', help = 'path to the low-res mesh', required = True)
    parser.add_argument('--log', help = 'log file name', default = 'log.txt')
    args = parser.parse_args()

    if args.input and os.path.exists(args.input):
        print('File : {}\n'.format(args.input))
    else:
        print('No input\n')
        return
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir, exist_ok = True)
        log_fn = open(os.path.join(args.outdir,args.log), 'wt')
    


def transform_pos(mtx, pos):
    t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
    posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
    return torch.matmul(posw, t_mtx.t())[None, ...]

def render(glctx, mtx, vertex, triangle, texcoord, resolution: int):
    vertex_clip    = transform_pos(mtx, vertex)
    # glctx, pos, tri, resolution, ranges=None, grad_db=True
    rast_out, rast_db = dr.rasterize(glctx,vertex_clip , triangle , resolution=[resolution, resolution])
    # rast_db : shape [minibatch_size, height, width, 4] and contain said derivatives in order (du/dX, du/dY, dv/dX, dv/dY)
    # attr, rast, tri, rast_db=None, diff_attrs=None
    #interpolate_db  [minibatch_size, height, width, 2 * len(diff_attrs)]. The derivatives of the first selected attribute A will be on channels 0 and 1 as (dA/dX, dA/dY), etc.
    color, interpolate_db = dr.interpolate(texcoord[None, ...], rast_out, rast_db = rast_db)
    #return color




def opt_viking(
        log_fn,
        ref,
        base,
        output_dir):
    #material_dict, mface, vface, tface, nface, vertex, normal, texcoord = readObj(ref)

    ref_mesh = readObj(ref)
    base_mesh = readObj(base)


    # Create position/triangle index tensors
    triangles = torch.from_numpy(ref_mesh['vface'].astype(np.int32)).cuda()
    vertex = torch.from_numpy(ref_mesh['vertex'].astype(np.int32)).cuda()
    uv_idx = torch.from_numpy(ref_mesh['tface'].astype(np.int32)).cuda()
    texcoord = torch.from_numpy(ref_mesh['texcoord']).cuda()

    triangles_opt = torch.from_numpy(base_mesh['vface'].astype(np.int32)).cuda()
    vertex_opt = torch.from_numpy(base_mesh['vertex'].astype(np.int32)).cuda()
    #tex     = torch.from_numpy(tex.astype(np.float32)).cuda()
    #tex_opt = torch.full(tex.shape, 0.2, device='cuda', requires_grad=True)
    glctx = dr.RasterizeGLContext()

    ang = 0.0

    # Adam optimizer for texture with a learning rate ramp.
    #optimizer    = torch.optim.Adam([tex_opt], lr=lr_base)
    #scheduler    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda x: lr_ramp**(float(x)/float(max_iter)))

    # Render.
    max_iter = 20000
    ang = 0.0
    texloss_avg = []
    for it in range(max_iter + 1):
        # Random rotation/translation matrix for optimization.
        r_rot = util.random_rotation_translation(0.25)

        # Smooth rotation for display.
        a_rot = np.matmul(util.rotate_x(-0.4), util.rotate_y(ang))
        dist = np.random.uniform(0.0, 48.5)

        # Modelview and modelview + projection matrices.
        proj  = util.projection(x=0.4, n=1.0, f=200.0)
        r_mv  = np.matmul(util.translate(0, 0, -1.5-dist), r_rot)
        r_mvp = np.matmul(proj, r_mv).astype(np.float32)
        a_mv  = np.matmul(util.translate(0, 0, -3.5), a_rot)
        a_mvp = np.matmul(proj, a_mv).astype(np.float32)

        # Measure texture-space RMSE loss
#        with torch.no_grad():
#            texmask = torch.zeros_like(tex)
#            tr = tex.shape[1]//4
#            texmask[tr+13:2*tr-13, 25:-25, :] += 1.0
#            texmask[25:-25, tr+13:2*tr-13, :] += 1.0
#            # Measure only relevant portions of texture when calculating texture
#            # PSNR.
#            texloss = (torch.sum(texmask * (tex - tex_opt)**2)/torch.sum(texmask))**0.5 # RMSE within masked area.
#            texloss_avg.append(float(texloss))
#
        # Render reference and optimized frames. Always enable mipmapping for reference.
        color = render()

        # Reduce the reference to correct size.
        while color.shape[1] > res:
            color = util.bilinear_downsample(color)

        # Compute loss and perform a training step.
        #loss = torch.mean((color - color_opt)**2) # L2 pixel loss.
        #optimizer.zero_grad()
        #loss.backward()
        #optimizer.step()
        #scheduler.step()

        # Print/save log.
        #if log_interval and (it % log_interval == 0):
        #    texloss_val = np.mean(np.asarray(texloss_avg))
        #    texloss_avg = []
        #    psnr = -10.0 * np.log10(texloss_val**2) # PSNR based on average RMSE.
        #    s = "iter=%d,loss=%f,psnr=%f" % (it, texloss_val, psnr)
        #    print(s)
        #    if log_file:
        #        log_file.write(s + '\n')

        ## Show/save image.
        #display_image = display_interval and (it % display_interval == 0)
        #save_image = imgsave_interval and (it % imgsave_interval == 0)
        #save_texture = texsave_interval and (it % texsave_interval) == 0

        #if display_image or save_image:
        #    ang = ang + 0.1

        #    with torch.no_grad():
        #        result_image = render(glctx, a_mvp, vtx_pos, pos_idx, vtx_uv, uv_idx, tex_opt, res, enable_mip, max_mip_level)[0].cpu().numpy()[::-1]

        #        if display_image:
        #            util.display_image(result_image, size=display_res, title='%d / %d' % (it, max_iter))
        #        if save_image:
        #            util.save_image(out_dir + '/' + (imgsave_fn % it), result_image)

        #        if save_texture:
        #            texture = tex_opt.cpu().numpy()[::-1]
        #            util.save_image(out_dir + '/' + (texsave_fn % it), texture)


    # Done.
    #if log_file:
    #    log_file.close()
    


    

if __name__ == "__main__":
    main()


