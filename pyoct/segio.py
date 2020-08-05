'''
Parser for BROCT segmentation files.

```
from segio import volumes

for volume in volumes(open('a.seg', rb')):
    print(volume['percentDepth'])
```

'''

import os
from struct import unpack, calcsize, Struct
from glob import glob
from copy import deepcopy

import numpy

# header field format
HEADER_FIELDS = [
    ('type',    'i'),
    ('dsXdim',  'i'),
    ('dsYdim',  'i'),
    ('dsZdim',  'i'),
    ('number',  'i'),
    ('xdim',    'i'),
    ('ydim',    'i'),
    ('zdim',    'i'),
    ('xlength', 'f'),
    ('ylength', 'f'),
    ('zlength', 'f'),
    ('arbSize', 'i'),
    ('readSeg', 'i'),
]
HEADER_FORMAT = '<' + ''.join([fmt for (name, fmt) in HEADER_FIELDS])
_header_parser = Struct(HEADER_FORMAT)

# segmentation array format
SEGMENTATION_ARRAYS = [
    ('topLayersViz',  ('dsZdim',  'dsXdim', 3), numpy.float32),
    ('botLayersViz',  ('dsZdim',  'dsXdim', 3), numpy.float32),
    ('topLayers',     ('dsZdim',  'dsXdim', 2), numpy.int32),
    ('botLayers',     ('dsZdim',  'dsXdim', 2), numpy.int32),
    ('topLayerArbMM', ('arbSize', 3),           numpy.float32),
    ('botLayerArbMM', ('arbSize', 3),           numpy.float32),
]

# tracking matrix format
TRACKING_MATRICES = [
    ('toolXform', (4, 4), numpy.float32),
    ('baseXform', (4, 4), numpy.float32),
    ('origin2OCT', (4, 4), numpy.float32),
    ('ee2Tip', (4, 4), numpy.float32),
]

# footer field format
SEGMENTATION_FIELDS = [
    ('topDistancePoint', 'ff'),
    ('botDistancePoint', 'ff'),
    ('topLayerPtIdx',    'i'),
    ('botLayerPtIdx',    'i'),
    ('apex',             'fff'),
]
SEGMENTATION_FORMAT = '<' + ''.join([fmt for (name, fmt) in SEGMENTATION_FIELDS])
_segmentation_parser = Struct(SEGMENTATION_FORMAT)

FOOTER_FIELDS = [
    ('dsToolLocation',   'ii'),
    ('percentDepth',     'f'),
]
FOOTER_FORMAT = '<' + ''.join([fmt for (name, fmt) in FOOTER_FIELDS])
_footer_parser = Struct(FOOTER_FORMAT)

def _mul(l):
    p = 1
    for x in l:
        p *= x
    return p

def _group(fields, values):
    out = []
    i = 0

    for (name, fmt) in fields:
        n = len(fmt)
        if n > 1:
            out.append((name, fmt, values[i:i + n]))
            i += n
        else:
            out.append((name, fmt, values[i]))
            i += 1

    return out

def _peek(f, n=1):
    data = f.read(n)
    f.seek(-len(data), os.SEEK_CUR)
    return data

def _read_array(f, shape, dtype):
    return numpy.fromfile(f, dtype=dtype, count=_mul(shape)).reshape(shape)

def _specialize(header, template):
    return [(field, [header.get(dim, dim) for dim in dims], type_) for (field, dims, type_) in template]

def _parse_header(f):
    # read in the whole header
    header = _header_parser.unpack(f.read(_header_parser.size))
    header = {name: value for (name, fmt, value) in _group(HEADER_FIELDS, header)}

    return header

def parse_volume(f):
    '''
    Parse one complete volume from the file.
    '''

    volume = {}

    # parse header for volume read dimensions
    header = _parse_header(f)
    volume.update(header)

    if header.get('readSeg'):

        # read in the segmentation arrays
        arrays = _specialize(header, SEGMENTATION_ARRAYS)
        arrays = {name: _read_array(f, shape, dtype) for (name, shape, dtype) in arrays}
        volume.update(arrays)

        # read in the segmentation fields
        segmentation = _segmentation_parser.unpack(f.read(_segmentation_parser.size))
        segmentation = {name: value for (name, fmt, value) in _group(SEGMENTATION_FIELDS, segmentation)}
        volume.update(segmentation)

    # read in the footer fields
    footer = _footer_parser.unpack(f.read(_footer_parser.size))
    footer = {name: value for (name, fmt, value) in _group(FOOTER_FIELDS, footer)}
    volume.update(footer)

    # read in the transformation matrices
    # NOTE: matrices are stored in column-major format
    matrices = {name: _read_array(f, shape, dtype) for (name, shape, dtype) in _specialize(header, TRACKING_MATRICES)}
    volume.update(matrices)

    return volume

def skip_volume(f):
    '''
    Skip one complete volume in the file without parsing it.
    '''

    # find the skip size
    skip = size_volume(f)

    # skip ahead
    f.seek(skip, os.SEEK_CUR)

def size_volume(f):
    '''
    Compute the size of a volume in the file without advancing the reader pointer.
    '''

    # read the header to compute the size
    header = _parse_header(f)

    # rewind the file pointer
    f.seek(-_header_parser.size, os.SEEK_CUR)

    # compute the size
    size = _header_parser.size + _footer_parser.size
    dynamic = _specialize(header, TRACKING_MATRICES)

    # handle optional fields
    if header.get('readSeg'):
        size += _segmentation_parser.size
        dynamic += _specialize(header, SEGMENTATION_ARRAYS)

    # include sizes of dynamic fields
    for (_, shape, dtype) in dynamic:
        size += _mul(shape) * dtype(0).itemsize

    return size

def volumes(f, skip=0):
    '''
    Generator for iterating through volumes in the file. Parses on demand.

    Optionally skips `skip` volumes at the start of the file.
    '''

    # skip as needed
    for i in range(skip):
        if not _peek(f):
            break

        skip_volume(f)

    # sequentially read each volume
    while _peek(f):
        yield parse_volume(f)

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='parser for BROCT segmentation files', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('path', nargs='+', help='path to segmentation file')
    parser.add_argument('--skip', '-s', type=int, help='number of volumes to skip', default=0)
    parser.add_argument('--count', '-c', type=int, help='number of volumes to read', default=-1)
    args = parser.parse_args()

    if args.count == 0:
        raise SystemExit(0)

    for path in args.path:
        for path in glob(path):
            for (i, volume) in enumerate(volumes(open(path, 'rb'), skip=args.skip)):
                print('{0} {1:3d}: '
                        '{2[number]:3d} {2[dsXdim]:3d}x{2[dsYdim]:3d}x{2[dsZdim]:3d}  '
                        '({3[0]:5.2f}, {3[1]:5.2f}, {3[2]:5.2f})  {4:6.1f}%  ({5[0]:4.2f}, {5[1]:4.2f}, {5[2]:4.2f})'.format(
                            path, args.skip + i, volume, volume['toolXform'][:3, 3], volume['percentDepth'] * 100, volume.get('apex', (0,0,0))))

                if args.count >= 0 and i + 1 >= args.count:
                    break
