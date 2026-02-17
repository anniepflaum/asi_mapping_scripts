import functools
from pathlib import Path
from datetime import datetime

import numpy as np

import matlab.engine


@functools.cache
def matlab_engine():
    """
    only cached because used by Pytest in multiple tests
    """
    cwd = Path(__file__).parent
    eng = matlab.engine.start_matlab("-nojvm")
    eng.addpath(eng.genpath(str(cwd)), nargout=0)
    return eng


@functools.cache
def has_matmap3d(eng) -> bool:
    cwd = Path(__file__).parent
    d = cwd.parents[3] / "matmap3d"
    print(f"Looking in {d} for matmap3d")

    if d.is_dir():
        eng.addpath(str(d), nargout=0)
        return True

    return False


@functools.cache
def has_aerospace(eng) -> bool:
    return eng.has_matlab_toolbox("Aerospace Toolbox")


@functools.cache
def has_mapping(eng) -> bool:
    return eng.has_matlab_toolbox("Mapping Toolbox")


def matlab_ecef2eci(eng, matmap3d: bool, utc: datetime, ecef):
    if matmap3d:
        return eng.matmap3d.ecef2eci(utc, *ecef, nargout=3)

    return np.array(eng.ecef2eci(utc, np.asarray(ecef), nargout=1)).squeeze()


def matlab_eci2ecef(eng, matmap3d: bool, utc: datetime, eci):
    if matmap3d:
        return eng.matmap3d.eci2ecef(utc, *eci, nargout=3)

    return np.array(eng.eci2ecef(utc, np.asarray(eci), nargout=1)).squeeze()
