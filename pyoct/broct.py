import numpy as np
import sys
import os

#name, size, type
header_fields = [
    ('meta',        1,  np.int32),
    ('xdim',        1,  np.int32),
    ('ydim',        1,  np.int32),
    ('zdim',        1,  np.int32),
    ('xmin',        1,  np.int32),
    ('xmax',        1,  np.int32),
    ('ymin',        1,  np.int32),
    ('ymax',        1,  np.int32),
    ('zmin',        1,  np.int32),
    ('zmax',        1,  np.int32),
    ('inactive',    1,  np.int32),
    ('xlength',     1,  np.float64),
    ('ylength',     1,  np.float64),
    ('zlength',     1,  np.float64),
    ('scan_type',   1,  np.int32) ,
    ('big_xdim',    1,  np.int32),
    ('big_xmin',    1,  np.int32),
    ('big_xmax',    1,  np.int32),
    ('big_inactive',1,  np.int32),
    ('roi',         1,  np.int32),
]

def volumes(f, skip=None):
    skip = skip or 0

    #read in entire header
    header = {x[0]: np.fromfile(f, dtype=x[2], count=x[1])[0] for x in header_fields}

    #read in mapping of bscans
    header['scan_map'] = np.fromfile(f, dtype=np.int32, count=header['zdim'])
    note_len = np.fromfile(f, dtype=np.int32, count=1)[0]
    header['note'] = np.fromfile(f, dtype=np.uint8, count=note_len)

    xmax = header['xmax']
    xmin = header['xmin']
    ymax = header['ymax']
    ymin = header['ymin']
    zdim = header['zdim']
    bigXmax = header['big_xmax']
    bigXmin = header['big_xmin']
    roi = header['roi']
    scan_type = header['scan_type']

    #size of the volume
    vSize = (xmax-xmin+1)*(ymax-ymin+1)*zdim
    avgSize = (bigXmax-bigXmin+1)*(ymax-ymin+1)*roi

    if scan_type == 5:
        totalSize = (ymax - ymin +1)*2
    elif scan_type == 7:
        vSize = (xmax-xmin+1)*(ymax-ymin+1)*(zdim-roi)
        totalSize = vSize + avgSize + zdim
    else:
        totalSize = vSize

    volumes_start = f.tell()
    f.seek(0, 2)
    volumes_end = f.tell()
    volumes_count = (volumes_end - volumes_start) // totalSize

    if skip < 0:
        skip += volumes_count

    skip = max(min(skip, volumes_count), 0)
    f.seek(volumes_start + totalSize * skip, 0)

    while True:
        if not f.peek(1):
            break

        result = header.copy()

        if scan_type == 5:
            result['volume'] = np.fromfile(f, dtype=np.float32, count=vSize).reshape((2, ymax-ymin+1))
        elif scan_type == 7:
            result['volume'] = np.fromfile(f, dtype=np.int8, count=vSize).reshape((zdim-roi, ymax-ymin+1, xmax-xmin+1))
            # result['average'] = np.fromfile(f, dtype=np.int8, count=avgSize).reshape((roi, ymax-ymin+1, bigXmax-bigXmin+1))
            # result['map'] = np.fromfile(f, dtype=np.int32, count=zdim)
        else:
            result['volume'] = np.fromfile(f, dtype=np.int8, count=vSize).reshape((zdim, ymax-ymin+1, xmax-xmin+1))

        yield (skip, result)
        skip += 1

def _postprocess(args, img):
     # white and black levels
    img = (img - args.black) / (args.white - args.black)
    img = numpy.clip(img, 0, 1)

    # flips
    if args.flip_vertical:
        img = img[::-1, :]
    if args.flip_vertical:
        img = img[:, ::-1]

    # rotation
    if args.rotate > 0:
        img = numpy.rot90(img, args.rotate % 90)

    return (255 * img).astype(numpy.uint8)

if __name__ == '__main__':
    import os
    from glob import glob
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

    import numpy
    from imageio import imwrite

    parser = ArgumentParser(description='parser for BROCT files', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('path', nargs='+', help='path to BROCT file')
    parser.add_argument('--volume', '-v', type=int, help='first volume index to read', default=0)
    parser.add_argument('--count', '-c', type=int, help='volumes to read')
    parser.add_argument('--white', '-w', type=int, help='white level', default=100)
    parser.add_argument('--black', '-b', type=int, help='black level', default=54)
    parser.add_argument('--flip-vertical', '-fv', action='store_true', help='flip B-scan vertically')
    parser.add_argument('--flip-horizontal', '-fh', action='store_true', help='flip B-scan horizontal')
    parser.add_argument('--rotate', '-r', type=int, help='rotate B-scan in 90 degree increments', default=0)
    parser.add_argument('--average', '-a', type=int, help='average adjacent B-scans', default=-1)

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--svp', action='store_true', help='dump an SVP')
    group.add_argument('--scan', nargs='?', type=int, help='B-scan index to dump')
    group.add_argument('--scan-all', action='store_true', help='dump all B-scans')
    group.add_argument('--mip', action='store_true', help='dump a MIP')

    args = parser.parse_args()

    chunks = max(1, args.average)

    if args.count == 0:
        raise SystemExit(0)

    for path in args.path:
        for path in glob(path):
            for (vidx, vol) in volumes(open(path, 'rb'), skip=args.volume):
                data = vol['volume']

                if args.svp:
                    print(path, vidx, 'SVP', end=' ')

                    svp = numpy.mean(data, axis=1)
                    svp = _postprocess(args, svp)

                    out = '{}_{:03d}_svp.png'.format(os.path.splitext(path)[0], vidx)
                    dpi = (1e3 * vol['xlength'] / svp.shape[0], 1e3 * vol['zlength'] / svp.shape[1])
                    imwrite(out, svp, dpi=dpi)
                    print('->', out)
                elif args.mip:
                    print(path, vidx, 'MIP', end=' ')

                    mip = numpy.amax(data, axis=1)
                    mip = _postprocess(args, mip)

                    out = '{}_{:03d}_mip.png'.format(os.path.splitext(path)[0], vidx)
                    dpi = (1e3 * vol['xlength'] / mip.shape[0], 1e3 * vol['zlength'] / mip.shape[1])
                    imwrite(out, mip, dpi=dpi)
                    print('->', out)
                else:
                    if args.scan_all:
                        bidxs = range(data.shape[0])
                    elif args.scan is None or args.scan < 0:
                        bidxs = [data.shape[0] // 2]
                    else:
                        bidxs = [args.scan]

                    for bidx in bidxs[::chunks]:
                        print(path, vidx, bidx, end=' ')
                        bscan = data[bidx, :, :].astype(numpy.float32)
                        for i in range(1, chunks):
                            bscan += data[bidx + i, :, :].astype(numpy.float32)
                        bscan /= chunks

                        bscan = _postprocess(args, bscan)

                        out = '{}_{:03d}_{:03d}.png'.format(os.path.splitext(path)[0], vidx, bidx // chunks)
                        dpi = (1e3 * vol['ylength'] / bscan.shape[0], 1e3 * vol['xlength'] / bscan.shape[1])
                        imwrite(out, bscan, dpi=dpi)
                        print('->', out)

                if args.count is not None and args.count > 0:
                    if vidx - args.volume + 1 >= args.count:
                        break
