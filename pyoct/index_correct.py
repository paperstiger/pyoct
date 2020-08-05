import numpy
import scipy.interpolate

class InterpHelper(object):
    def __init__(self, shape, ray_interp, vol_interp):
        self._shape = shape
        self._ray_interp = ray_interp
        self._vol_interp = vol_interp

    def __getitem__(self, key):
        runs = []
        for (s, k) in zip(self._shape, key):
            if isinstance(k, slice):
                runs.append(numpy.arange(k.start or 0, k.stop or s, k.step or 1))
            else:
                runs.append([k])

        idx = numpy.stack(numpy.meshgrid(*runs, indexing='ij'), axis=-1)

        return self(idx)

    def __call__(self, xi):
        return self._vol_interp(self._ray_interp(xi))

    @property
    def shape(self):
        return self._shape

def index_correct_volume(oct, **kwargs):
    if 'correction' in kwargs:
        (ray_interp, vol_interp) = kwargs.pop('correction')
    elif 'segmentation' in kwargs:
        (ray_interp, vol_interp) = make_index_correction(oct, **kwargs)
    else:
        raise RuntimeError('missing correction or segmentation')

    scan_idx = numpy.stack(numpy.meshgrid(*[numpy.arange(0, r) for r in oct['volume'].shape], indexing='ij'), axis=-1)

    return vol_interp(ray_interp(scan_idx))

def index_correction(oct, **kwargs):
    return InterpHelper(oct['volume'].shape, *make_index_correction(oct, **kwargs))

def make_index_correction(oct, **kwargs):
    if 'rays' in kwargs and 'ascans' in kwargs:
        rays = kwargs.pop('rays')
        ascans = kwargs.pop('ascans')
    elif 'segmentation' in kwargs:
        seg = kwargs.pop('segmentation')

        if len(seg.get('topLayers', [])) > 0 and len(seg.get('botLayers', [])) > 0:
            rays = [numpy.swapaxes(seg[k][:, numpy.newaxis, ::10, :], 0, 2) for k in ['topLayersViz', 'botLayersViz']]

            rays.insert(0, rays[0].copy())
            rays[0][..., 1] = 0
            rays = numpy.array(rays)

            ascans = [numpy.swapaxes(seg[k][:, numpy.newaxis, ::10, :], 0, 2) for k in ['topLayers', 'botLayers']]
            (_, _, zidx) = numpy.meshgrid(range(ascans[0].shape[0]), [0], range(ascans[0].shape[2]), indexing='ij')
            for (i, ascan) in enumerate(ascans):
                ascans[i] = numpy.concatenate((ascan, zidx[..., numpy.newaxis]), axis=-1)

            ascans.insert(0, ascans[0].copy())
            ascans[0][..., 1] = 0
            ascans = numpy.array(ascans, dtype=numpy.float32)

            vol_dim = numpy.array([oct[k] for k in ['xlength', 'ylength', 'zlength']])
            ascan_shape = [ seg['dsXdim'] - 1, seg['dsYdim'] - 1, seg['dsZdim'] - 1 ]
            ascan_pitch = vol_dim / ascan_shape
            ascans = ascans * ascan_pitch
        else:
            (rays, ascans, valid) = raytrace(seg, **kwargs)
            rays = rays[:, valid, :]
            ascans = ascans[:, valid, :]
    else:
        raise RuntimeError('missing rays/ascans or segmentation')

    vol_shape = oct['volume'].shape
    vol_dim = numpy.array([oct[k] for k in ['xlength', 'ylength', 'zlength']])
    vol_pitch = vol_dim / vol_shape

    # use pixel coordinates
    rays /= vol_pitch
    ascans /= vol_pitch

    # set up the distortion interpolator
    ray_interp = scipy.interpolate.LinearNDInterpolator(
        rays.reshape((-1, 3)),
        ascans.reshape((-1, 3)),
        fill_value=-1,
    )

    # set up the volume interpolator
    vol_interp = scipy.interpolate.RegularGridInterpolator(
        [numpy.arange(0, r) for r in vol_shape],
        oct['volume'],
        method='linear',
        bounds_error=False,
        fill_value=0
    )

    return (ray_interp, vol_interp)

def compute_normals(normals, normals_idx, layer, patch):
    batch = {}
    batch_idx = {}

    (sx, sz) = patch
    sx = max(sx, 4)
    sz = max(sz, 4)

    for i in range(normals.shape[0]):
        for j in range(normals.shape[2]):
            x = normals_idx[i, 0, j, 0]
            z = normals_idx[i, 0, j, 2]

            bnds = [slice(max(0, q - sq), min(q + sq, u)) for (q, sq, u) in zip([x, z], [sx, sz], [layer.shape[0], layer.shape[2]])]
            pts = layer[bnds[0], 0, bnds[1]].reshape((-1, 3))

            batch.setdefault(pts.shape[0], []).append(pts)
            batch_idx.setdefault(pts.shape[0], []).append((i, j))

    for k in batch.keys():
        pts = numpy.hstack(batch[k])
        pts -= numpy.mean(pts, axis=0)
        pts = pts.T.reshape((len(batch[k]), 3, -1))

        (u, _, _) = numpy.linalg.svd(pts, full_matrices=False)
        idx = numpy.array(batch_idx[k])
        normals[idx[:, 0], 0, idx[:, 1]] = u[:, :, 2]

    # flip normal if facing wrong direction
    normals[normals[..., 1] > 0] *= -1

def raytrace(seg, **kwargs):
    # process parameters
    ray_resolution = kwargs.pop('ray_resolution', (25, 25))
    ray_sample_distance = kwargs.pop('ray_sample_distance', seg['ylength'])
    ray_sample_count = kwargs.pop('ray_sample_count', 250)
    precompute_normals = kwargs.pop('precompute_normals', False)

    # gather volume/segmentation acqusition info
    vol_dim = numpy.array([seg[k] for k in ['xlength', 'ylength', 'zlength']])

    layers = [numpy.swapaxes(seg[k][:, numpy.newaxis, :, :], 0, 2) for k in ['topLayersViz', 'botLayersViz']]
    layers.append(vol_dim[1] * numpy.ones_like(layers[-1]))
    lyr_dim = numpy.array([vol_dim[0], 0, vol_dim[2]])
    lyr_pitch = lyr_dim / layers[0].shape[:-1]
    normals_patch_scale = kwargs.pop('normals_patch_scale', (0.05, 0.05))
    normals_patch = kwargs.pop('normals_patch', (int(lyr_dim[0] * normals_patch_scale[0] / lyr_pitch[0]), int(lyr_dim[2] * normals_patch_scale[1] / lyr_pitch[2])))

    # initialize rays and their origins
    res = [ray_resolution[0], 1, ray_resolution[1]]
    rays = numpy.tile([[[0, 1, 0]]], res + [1])
    valid = numpy.ones((res[0], 1, res[2]), dtype=numpy.bool)
    origins = numpy.stack(numpy.meshgrid(*[numpy.linspace(0, l, r) for (l, r) in zip(vol_dim, res)], indexing='ij'), axis=-1)
    origins[..., 1] = 0

    # needed indexing arrays for later
    (sample_idx_x, sample_idx_z) = numpy.meshgrid(range(origins.shape[0]), range(origins.shape[2]), indexing='ij')
    sample_idx_x = sample_idx_x[:, numpy.newaxis, :]
    sample_idx_z = sample_idx_z[:, numpy.newaxis, :]

    segments = [origins]
    # ref: http://hyperphysics.phy-astr.gsu.edu/hbase/vision/eyescal.html
    refractive_indices = kwargs.pop('refractive_indices', [1, 1.376, 1.336])

    for i in range(len(layers)):
        layer = layers[i]
        n1 = refractive_indices[i]
        if i + 1 < len(refractive_indices):
            n2 = refractive_indices[i+1]
        else:
            n2 = n1

        # sample along the ray
        ts = numpy.linspace(0, ray_sample_distance, ray_sample_count)
        samples = origins + ts[numpy.newaxis, :, numpy.newaxis, numpy.newaxis] * rays

        # map (X, Z) into index to lookup in layer
        layer_idx = (samples / (lyr_pitch + 1e-9)).astype(numpy.intp)
        layer_idx[..., 1] = 0

        # mark invalid samples
        invalid = numpy.zeros(samples.shape[:-1], dtype=numpy.bool)
        for d in range(3):
            mask = layer_idx[..., d] >= layer.shape[d]
            layer_idx[mask, d] = layer.shape[d] - 1
            invalid[mask] = True

            mask = layer_idx[..., d] < 0
            layer_idx[mask, d] = 0
            invalid[mask] = True

        # find layer hit
        hit = layer[layer_idx[..., 0], layer_idx[..., 1], layer_idx[..., 2], 1] < samples[..., 1]
        first_hit_idx = numpy.argmax(hit, axis=1)[:, numpy.newaxis, :]

        any_invalid = numpy.any(invalid, axis=1, keepdims=True)
        first_invalid_idx = numpy.argmin(invalid, axis=1)[:, numpy.newaxis, :]

        mask = numpy.logical_and(first_invalid_idx <= first_hit_idx, any_invalid)
        valid[mask] = False

        # update origins and rays
        origins = samples[sample_idx_x, first_hit_idx, sample_idx_z, :]
        segments.append(origins)

        normals_idx = layer_idx[sample_idx_x, first_hit_idx, sample_idx_z, :]
        if precompute_normals:
            # compute all normals
            normals = numpy.zeros_like(layer)
            compute_normals(normals, layer_idx, layer, normals_patch)
            normals = normals[normals_idx[..., 0], normals_idx[..., 1], normals_idx[..., 2]]
        else:
            # compute only the needed normals
            normals = numpy.zeros_like(rays)
            compute_normals(normals, normals_idx, layer, normals_patch)

        # ref: https://en.wikipedia.org/wiki/Snell%27s_law#Vector_form
        r = n1 / n2
        c = -numpy.sum(normals * rays, axis=-1, keepdims=True)
        radicand = 1 - r**2 * (1 - c**2)

        mask = radicand[..., 0] < 0
        valid[mask] = False
        radicand[mask] = 0

        rays = r * rays + (r*c - numpy.sqrt(radicand)) * normals

    segments = numpy.array(segments)

    # generate the rays as they appear in the volume
    ascans = segments.copy()
    ascans[1:, ...] = ascans[0, ...]
    dists = numpy.sum(numpy.diff(segments, axis=0)**2, axis=-1)**0.5 * numpy.array(refractive_indices)[:, numpy.newaxis, numpy.newaxis, numpy.newaxis]
    ascans[1:, ..., 1] = numpy.cumsum(dists, axis=0)

    return (segments, ascans, valid)

def plot_raytrace(rays, fig=None):
    if fig is None:
        fig = pyplot.figure()

    fig.clf()
    ax = fig.gca(projection='3d')

    for x in range(rays.shape[1]):
        for z in range(rays.shape[3]):
            ax.plot(rays[:, x, 0, z, 0], rays[:, x, 0, z, 2], rays[:, x, 0, z, 1], '-x')

    # for layer in layers[:-1]:
    #     ax.plot_surface(layer[:, 0, :, 0], layer[:, 0, :, 2], layer[:, 0, :, 1])

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('z (mm)')
    ax.set_zlabel('y (mm)')

    ax.invert_zaxis()
    ax.invert_xaxis()

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Qt5Agg')
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib import pyplot

    import segio
    import broct

    seg_path = r"D:\Izatt\Journal\DALK_TBME\Data\1_octseg_11-20-2018_2_20_27_PM.seg"
    seg = next(segio.volumes(open(seg_path, 'rb')))
    layers = [numpy.swapaxes(seg[k][:, numpy.newaxis, :, :], 0, 2) for k in ['topLayersViz', 'botLayersViz']]

    oct_path = r"D:\Izatt\Journal\DALK_TBME\Data\1_oct_11-20-2018_2_20_27_PM.broct"
    oct = next(broct.volumes(open(oct_path, 'rb')))[1]
    oct['volume'] = numpy.flip(numpy.swapaxes(oct['volume'], 0, 2), axis=1)


    raw_vol = oct['volume']
    ic_vol = index_correction(oct, segmentation=seg)

    pos = 0.75
    dim = 2

    idx = (slice(None), slice(None), int(raw_vol.shape[2] * pos))
    raw_bscan = raw_vol[idx]
    ic_bscan = ic_vol[idx]

    vol_shape = [seg[k] for k in ['xdim', 'ydim', 'zdim']]
    vol_dim = numpy.array([seg[k] for k in ['xlength', 'ylength', 'zlength']])
    vol_pitch = vol_dim / vol_shape

    pyplot.figure()
    idx = [int(layers[0].shape[dim] * pos) if dim == i else slice(None) for i in range(3)]

    pyplot.subplot(2,1,1)
    pyplot.title('Raw')
    pyplot.autoscale(tight=True)
    pyplot.imshow(numpy.squeeze(raw_bscan.T), extent=[0, vol_dim[0], vol_dim[1], 0])
    for layer in layers:
        pyplot.plot(layer[idx + [0]], layer[idx + [1]])

    pyplot.subplot(2,1,2)
    pyplot.title('IC')
    pyplot.autoscale(tight=True)
    pyplot.imshow(numpy.squeeze(ic_bscan.T), extent=[0, vol_dim[0], vol_dim[1], 0])
    for layer in layers:
        pyplot.plot(layer[idx + [0]], layer[idx + [1]])

    pyplot.show()
