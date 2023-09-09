# 分析 dmrg measurement code
from collections import namedtuple
import numpy as np
open_bc = "open_bc"
#
# Model-specific code for the Heisenberg XXZ chain
class HeisenbergSpinHalfXXZChain(object):
    dtype = 'd'  # double-precision floating point
    d = 2  # single-site basis size

    Sz1 = np.array([[0.5, 0], [0, -0.5]], dtype)  # single-site S^z
    Sp1 = np.array([[0, 1], [0, 0]], dtype)  # single-site S^+

    sso = {"Sz": Sz1, "Sp": Sp1, "Sm": Sp1.transpose()}  # single-site operators
    def __init__(self, J=1., Jz=None, hz=0., hx=0., boundary_condition=open_bc):
        """
        `hz` can be either a number (for a constant magnetic field) or a
        callable (which is called with the site index and returns the
        magnetic field on that site).  The same goes for `hx`.
        """
        if Jz is None:
            Jz = J
        self.J = J
        self.Jz = Jz
        self.boundary_condition = boundary_condition
#
def get_site_class(L, boundary_condition):
    if boundary_condition == open_bc:
        sites_by_area = (
            range(0, L //2 - 1), # left block
            [L // 2 - 1], # left bare site
            range(L // 2 + 1, L)[::-1], # right block
            [L // 2], # right bare site
        )
    else:
        # PBC algorithm
        sites_by_area = (
            range(0, L//2 -1), # left block
            [L // 2 - 1], # left bare site
            range(L // 2, L - 1)[::-1], # right block
            [L - 1] # right bare site
        )
    site_class = [None] * L
    for i, heh in enumerate(sites_by_area):
        for site_index in heh:
            site_class[site_index] = i
    assert None not in site_class
    return site_class
#
def canonicalize(meas_desc):
    # this performs a stable sort on operators by site (so we are assuming
    # that operators on different sites commute),
    # By doning this, we will be able to combing operators ASAP as a block grows
    print("sorted(meas_desc, key=lambda x:x[0]):", sorted(meas_desc, key=lambda x:x[0]))
    return sorted(meas_desc, key=lambda x:x[0])
#
InnerObject = namedtuple("InnerObject", ["site_indices", "operator_names", "area"])
#
class OuterObject(object):
    def __init__(self, site_indices, operator_names, area, built=False):
        self.obj = InnerObject(site_indices,operator_names,area)
        self.built = built
#================================================================
if __name__ == "__main__":
    L = 8
    boundary_condition = "open_bc"
    # first make a list of which operators we need to consider on each site.
    # This way we wont have to look at every single operator each time we add a site
    #
    # we are also checking that each operator and site index is valid. 
    # we are copying the measurements so that we can modify them.
    #
    # we also make note of which of the four blocks each site is in 
    if boundary_condition is "open_bc":
        print("-------For the open boundary condition-------")
        boundary_condition = "open_bc"
        site_class = get_site_class(L=L, boundary_condition=boundary_condition)
        print("site_class:", site_class)
        #
    else:
        print("-------For the periodic boundary condition----------")
        boundary_condition = "periodic_bc"
        site_class = get_site_class(L=L, boundary_condition=boundary_condition)
        print("site_class:", site_class)
    #
    # figure out which sites are where
    LEFT_BLOCK, LEFT_SITE, RIGHT_BLOCK, RIGHT_SITE = 0, 1, 2, 3 # dui site 类型进行定义
    #
    measurements = ([[(i, "Sz")] for i in range(L)] +
                    [[(i, "Sz"), (j, "Sz")] for i in range(L) for j in range(L)] +
                    [[(i, "Sp"), (j, "Sm")] for i in range(L) for j in range(L)])
    #
    model = HeisenbergSpinHalfXXZChain(J=1., Jz=1., boundary_condition=boundary_condition)
    measurements_by_site = {}
    processed_measurements = []
    for meas_desc in measurements:
        site_indices = set()
        for site_index, operator_name in meas_desc:
            if operator_name not in model.sso:
                raise RuntimeError("Unknown operator: %s" % operator_name)
            if site_index not in range(L):
                raise RuntimeError("Unknown site index: %r (L = %r)" % (site_index, L))
            site_indices.add(site_index)
        print("meas_desc:", meas_desc)
        print("site_indices:", site_indices)
        measurement = [OuterObject((site_index,), (operator_name,),
                                site_class[site_index], False)
                        for site_index, operator_name in canonicalize(meas_desc)]
        processed_measurements.append(measurement)
        print("site_indices:", site_indices)
        for site_index in site_indices:
            measurements_by_site.setdefault(site_index, []).append(measurement)
    assert len(measurements) == len(processed_measurements)