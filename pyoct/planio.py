'''
Parser for serialized PlannerInput/PlannerOutput files.

```
import planio

for (i, o) in planio.plans(open('a.bin', 'rb')):
    print(i['robotNeedlePose'])
```

'''

import os
from struct import unpack, calcsize
from glob import glob

import numpy

def _peek(f, n=1):
    data = f.read(n)
    f.seek(-len(data), os.SEEK_CUR)
    return data

def _read_raw(f, dtype):
    return unpack(dtype, f.read(calcsize(dtype)))[0]

def _read_point(f):
    return numpy.fromfile(f, dtype=numpy.float64, count=3)

def _read_1d_array(f):
    n = _read_raw(f, 'Q')
    return numpy.fromfile(f, dtype=numpy.float64, count=3*n).reshape((n, 3))

def _read_2d_array(f):
    n = _read_raw(f, 'Q')
    return numpy.array([ _read_1d_array(f) for i in range(n)])

def _read_pose(f):
    return numpy.fromfile(f, dtype=numpy.float64, count=16).reshape((4, 4)).T

def _read_waypoint(f):
    return (_read_raw(f, 'd'), _read_pose(f))

def parse_input(f):
    '''
    Parse one complete PlannerInput from the file.
    '''

    ins = {}

    ins['topSurface'] = _read_2d_array(f)
    ins['botSurface'] = _read_2d_array(f)

    ins['topLayerArb'] = _read_1d_array(f)
    ins['botLayerArb'] = _read_1d_array(f)

    ins['trackedApex'] = _read_point(f)
    ins['targetOffset'] = _read_point(f)

    ins['trackedNeedlePose'] = _read_pose(f)
    ins['robotNeedlePose'] = _read_pose(f)

    ins['robotNeedleLinearVelocity'] = _read_point(f)
    ins['robotNeedleAngularVelocity'] = _read_point(f)

    ins['timeScaleFactor'] = _read_raw(f, 'd')

    n = _read_raw(f, 'Q')
    ins['activePlan'] = [_read_waypoint(f) for i in range(n)]
    ins['activeWaypoint'] = _read_waypoint(f)

    ins['timestamp'] = _read_raw(f, 'Q')
    ins['volume_number'] = _read_raw(f, 'i')

    return ins

def parse_output(f):
    '''
    Parse one complete PlannerOutput from the file.
    '''

    outs = {}

    outs['success'] = _read_raw(f, '?')

    n = _read_raw(f, 'Q')
    outs['newPlan'] = [_read_waypoint(f) for i in range(n)]

    outs['swapTime'] = _read_raw(f, 'd')
    outs['timestamp'] = _read_raw(f, 'Q')

    return outs

def plans(f, skip=0):
    '''
    Generator for iterating through inputs in the file. Parses every input as it goes.

    Optionally skips `skip` inputs at the start of the file.
    '''

    # skip as needed
    for i in range(skip):
        if not _peek(f):
            break

        parse_input(f)
        parse_output(f)

    # sequentially read each input
    while _peek(f):
        yield (parse_input(f), parse_output(f))

if __name__ == '__main__':
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(description='parser for planner input files', formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument('path', nargs='+', help='path to planner input file')
    parser.add_argument('--skip', '-s', type=int, help='number of inputs to skip', default=0)
    parser.add_argument('--count', '-c', type=int, help='number of inputs to read', default=-1)
    args = parser.parse_args()

    if args.count == 0:
        raise SystemExit(0)

    for path in args.path:
        for path in glob(path):
            for (i, ios) in enumerate(plans(open(path, 'rb'), skip=args.skip)):
                print('{0} {1:3d} {2[timestamp]}: {2[activeWaypoint][0]:5.2f} {3} {2[trackedApex]} -> {4[swapTime]:0.2f}'
                        .format(path, args.skip + i, ios[0], ios[0]['robotNeedlePose'][0:3, 3], ios[1]))

                if args.count >= 0 and i + 1 >= args.count:
                    break
