# 观察周期边界条件下，无限大 enlarge block 的过程
#
from __future__ import print_function, division  # requires Python >= 2.6
#
import sys
from collections import namedtuple
from collections.abc import Callable
from itertools import chain
#
import numpy as np
from scipy.sparse import kron, identity, lil_matrix
from scipy.sparse.linalg import eigsh  # Lanczos routine from ARPACK
#
open_bc = 0
periodic_bc = 1
#
# We will use python's "namedtuple" to represent the Block and EnlargedBlock
# objects
Block = namedtuple("Block", ["length", "basis_size", "operator_dict", "basis_sector_array"])
EnlargedBlock = namedtuple("EnlargedBlock", ["length", "basis_size", "operator_dict", "basis_sector_array"])

def is_valid_block(block):
    if len(block.basis_sector_array) != block.basis_size:
        return False
    for op in block.operator_dict.values():
        if op.shape[0] != block.basis_size or op.shape[1] != block.basis_size:
            return False
    return True

# This function should test the same exact things, so there is no need to
# repeat its definition.
is_valid_enlarged_block = is_valid_block

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
        if isinstance(hz, Callable):
            self.hz = hz
        else:
            self.hz = lambda site_index: hz
        if isinstance(hx, Callable):
            self.hx = hx
        else:
            self.hx = lambda site_index: hx

        if hx == 0:
            # S^z sectors corresponding to the single site basis elements
            self.single_site_sectors = np.array([0.5, -0.5])
        else:
            # S^z is not conserved
            self.single_site_sectors = np.array([0, 0])

    def H1(self, site_index):
        half_hz = .5 * self.hz(site_index)
        half_hx = .5 * self.hx(site_index)
        return np.array([[half_hz, half_hx], [half_hx, -half_hz]], self.dtype)

    def H2(self, Sz1, Sp1, Sz2, Sp2):  # two-site part of H
        """Given the operators S^z and S^+ on two sites in different Hilbert spaces
        (e.g. two blocks), returns a Kronecker product representing the
        corresponding two-site term in the Hamiltonian that joins the two sites.
        """
        return (
            (self.J / 2) * (kron(Sp1, Sp2.conjugate().transpose()) +
                            kron(Sp1.conjugate().transpose(), Sp2)) +
            self.Jz * kron(Sz1, Sz2)
        )

    def initial_block(self, site_index):
        if self.boundary_condition == open_bc:
            # conn refers to the connection operator, that is, the operator on the
            # site that was most recently added to the block.  We need to be able
            # to represent S^z and S^+ on that site in the current basis in order
            # to grow the chain.
            operator_dict = {
                "H": self.H1(site_index),
                "conn_Sz": self.Sz1,
                "conn_Sp": self.Sp1,
            }
        else:
            # Since the PBC block needs to be able to grow in both directions,
            # we must be able to represent the relevant operators on both the
            # left and right sites of the chain.
            operator_dict = {
                "H": self.H1(site_index),
                "l_Sz": self.Sz1,
                "l_Sp": self.Sp1,
                "r_Sz": self.Sz1,
                "r_Sp": self.Sp1,
            }
        return Block(length=1, basis_size=self.d, operator_dict=operator_dict,
                    basis_sector_array=self.single_site_sectors)

    def enlarge_block(self, block, direction, bare_site_index):
        """This function enlarges the provided Block by a single site, returning an
        EnlargedBlock.
        """
        mblock = block.basis_size
        o = block.operator_dict

        # Create the new operators for the enlarged block.  Our basis becomes a
        # Kronecker product of the Block basis and the single-site basis.  NOTE:
        # `kron` uses the tensor product convention making blocks of the second
        # array scaled by the first.  As such, we adopt this convention for
        # Kronecker products throughout the code.
        if self.boundary_condition == open_bc:
            enlarged_operator_dict = {
                "H": kron(o["H"], identity(self.d)) +
                    kron(identity(mblock), self.H1(bare_site_index)) +
                    self.H2(o["conn_Sz"], o["conn_Sp"], self.Sz1, self.Sp1),
                "conn_Sz": kron(identity(mblock), self.Sz1),
                "conn_Sp": kron(identity(mblock), self.Sp1),
            }
        else:
            assert direction in ("l", "r")
            if direction == "l":
                enlarged_operator_dict = {
                    "H": kron(o["H"], identity(self.d)) +
                        kron(identity(mblock), self.H1(bare_site_index)) +
                        self.H2(o["l_Sz"], o["l_Sp"], self.Sz1, self.Sp1),
                    "l_Sz": kron(identity(mblock), self.Sz1),
                    "l_Sp": kron(identity(mblock), self.Sp1),
                    "r_Sz": kron(o["r_Sz"], identity(self.d)),
                    "r_Sp": kron(o["r_Sp"], identity(self.d)),
                }
            else:
                enlarged_operator_dict = {
                    "H": kron(o["H"], identity(self.d)) +
                        kron(identity(mblock), self.H1(bare_site_index)) +
                        self.H2(o["r_Sz"], o["r_Sp"], self.Sz1, self.Sp1),
                    "l_Sz": kron(o["l_Sz"], identity(self.d)),
                    "l_Sp": kron(o["l_Sp"], identity(self.d)),
                    "r_Sz": kron(identity(mblock), self.Sz1),
                    "r_Sp": kron(identity(mblock), self.Sp1),
                }

        # This array keeps track of which sector each element of the new basis is
        # in.  `np.add.outer()` creates a matrix that adds each element of the
        # first vector with each element of the second, which when flattened
        # contains the sector of each basis element in the above Kronecker product.
        enlarged_basis_sector_array = np.add.outer(block.basis_sector_array, self.single_site_sectors).flatten()

        return EnlargedBlock(length=(block.length + 1),
                            basis_size=(block.basis_size * self.d),
                            operator_dict=enlarged_operator_dict,
                            basis_sector_array=enlarged_basis_sector_array)

    def construct_superblock_hamiltonian(self, sys_enl, env_enl):
        sys_enl_op = sys_enl.operator_dict
        env_enl_op = env_enl.operator_dict
        if self.boundary_condition == open_bc:
            # L**R
            H_int = self.H2(sys_enl_op["conn_Sz"], sys_enl_op["conn_Sp"],
                            env_enl_op["conn_Sz"], env_enl_op["conn_Sp"])
        else:
            assert self.boundary_condition == periodic_bc
            # L*R*
            H_int = (self.H2(sys_enl_op["r_Sz"], sys_enl_op["r_Sp"], 
                            env_enl_op["l_Sz"], env_enl_op["l_Sp"]) +
                    self.H2(sys_enl_op["l_Sz"], sys_enl_op["l_Sp"], 
                            env_enl_op["r_Sz"], env_enl_op["r_Sp"]))
        return (kron(sys_enl_op["H"], identity(env_enl.basis_size)) +
                kron(identity(sys_enl.basis_size), env_enl_op["H"]) +
                H_int)
#
def rotate_and_truncate(operator, transformation_matrix):
    """Transforms the operator to the new (possibly truncated) basis given by
    `transformation_matrix`.
    """
    return transformation_matrix.conjugate().transpose().dot(operator.dot(transformation_matrix))

def index_map(array):
    """Given an array, returns a dictionary that allows quick access to the
    indices at which a given value occurs.

    Example usage:

    >>> by_index = index_map([3, 5, 5, 7, 3])
    >>> by_index[3]
    [0, 4]
    >>> by_index[5]
    [1, 2]
    >>> by_index[7]
    [3]
    """
    d = {}
    for index, value in enumerate(array):
        d.setdefault(value, []).append(index)
    return d

def graphic(boundary_condition, sys_block, env_block, direction="r"):
    """Returns a graphical representation of the DMRG step we are about to
    perform, using '=' to represent the system sites, '-' to represent the
    environment sites, and '**' to represent the two intermediate sites.
    """
    order = {"r": 1, "l": -1}[direction]
    l_symbol, r_symbol = ("=", "-")[::order]
    l_length, r_length = (sys_block.length, env_block.length)[::order]

    if boundary_condition == open_bc:
        return (l_symbol * l_length) + "+*"[::order] + (r_symbol * r_length)
    else:
        return (l_symbol * l_length) + "+" + (r_symbol * r_length) + "*"

def bare_site_indices(boundary_condition, sys_block, env_block, direction):
    """Returns the site indices of the two bare sites: first the one in the
    enlarged system block, then the one in the enlarged environment block.
    """
    order = {"r": 1, "l": -1}[direction]
    l_block, r_block = (sys_block, env_block)[::order]
    if boundary_condition == open_bc:
        l_site_index = l_block.length
        r_site_index = l_site_index + 1
        sys_site_index, env_site_index = (l_site_index, r_site_index)[::order]
    else:
        sys_site_index = l_block.length
        env_site_index = l_block.length + r_block.length + 1
    
    print("sys_site_index:", sys_site_index)
    print("env_site_index:", env_site_index)

    
    g = graphic(boundary_condition, sys_block, env_block, direction)
    assert sys_site_index == g.index("+")
    assert env_site_index == g.index("*")

    return sys_site_index, env_site_index

def arpack_diagonalize_superblock(hamiltonian, guess):
    # Call ARPACK to find the superblock ground state.  ("SA" means find the
    # "smallest in amplitude" eigenvalue.)
    w, v = eigsh(hamiltonian, k=1, which="SA", v0=guess)
    return w[0], v.flatten()

StepInfo = namedtuple("StepInfo", ["truncation_error", "overlap", "energy", "L"])

def single_dmrg_step(model, sys, env, m, direction, 
                    diagonalize_superblock, target_sector=None, 
                    psi0_guess=None, callback=None):
    """Perform a single DMRG step using `sys` as the system and `env` as the
    environment, keeping a maximum of `m` states in the new basis.  If
    `psi0_guess` is provided, it will be used as a starting vector for the
    Lanczos algorithm.
    """
    assert is_valid_block(sys)
    assert is_valid_block(env)

    # Determine the site indices of the two bare sites
    sys_site_index, env_site_index = bare_site_indices(model.boundary_condition, sys, env, direction)
    print("H1 sys_site_index:", model.H1(sys_site_index))
    print("H1 env_site_index:", model.H1(env_site_index))
    # Enlarge each block by a single site.
    sys_enl = model.enlarge_block(sys, direction, sys_site_index)
    sys_enl_basis_by_sector = index_map(sys_enl.basis_sector_array)
    env_enl = model.enlarge_block(env, direction, env_site_index)
    env_enl_basis_by_sector = index_map(env_enl.basis_sector_array)

    assert is_valid_enlarged_block(sys_enl)
    assert is_valid_enlarged_block(env_enl)

    m_sys_enl = sys_enl.basis_size
    m_env_enl = env_enl.basis_size

    # Construct the full superblock Hamiltonian.
    superblock_hamiltonian = model.construct_superblock_hamiltonian(sys_enl, env_enl)

    if target_sector is not None:
        # Build up a "restricted" basis of states in the target sector and
        # reconstruct the superblock Hamiltonian in that sector.
        sector_indices = {} # will contain indices of the new (restricted) basis
                            # for which the enlarged system is in a given sector
        restricted_basis_indices = []  # will contain indices of the old (full) basis, which we are mapping to
        for sys_enl_sector, sys_enl_basis_states in sys_enl_basis_by_sector.items():
            sector_indices[sys_enl_sector] = []
            env_enl_sector = target_sector - sys_enl_sector
            if env_enl_sector in env_enl_basis_by_sector:
                for i in sys_enl_basis_states:
                    i_offset = m_env_enl * i  # considers the tensor product structure of the superblock basis
                    for j in env_enl_basis_by_sector[env_enl_sector]:
                        current_index = len(restricted_basis_indices)  # about-to-be-added index of restricted_basis_indices
                        sector_indices[sys_enl_sector].append(current_index)
                        restricted_basis_indices.append(i_offset + j)

        if not restricted_basis_indices:
            raise RuntimeError("There are zero states in the restricted basis.")

        restricted_superblock_hamiltonian = superblock_hamiltonian[:, restricted_basis_indices][restricted_basis_indices, :]
        if psi0_guess is not None:
            restricted_psi0_guess = psi0_guess[restricted_basis_indices]
        else:
            restricted_psi0_guess = None

    else:
        # Our "restricted" basis is really just the original basis.  The only
        # thing to do is to build the `sector_indices` dictionary, which tells
        # which elements of our superblock basis correspond to a given sector
        # in the enlarged system.
        sector_indices = {}
        restricted_basis_indices = range(m_sys_enl * m_env_enl)
        for sys_enl_sector, sys_enl_basis_states in sys_enl_basis_by_sector.items():
            sector_indices[sys_enl_sector] = [] # m_env_enl
            for i in sys_enl_basis_states:
                sector_indices[sys_enl_sector].extend(range(m_env_enl * i, m_env_enl * (i + 1)))

        restricted_superblock_hamiltonian = superblock_hamiltonian
        restricted_psi0_guess = psi0_guess

    if restricted_superblock_hamiltonian.shape == (1, 1):
        restricted_psi0 = np.array([[1.]], dtype=model.dtype)
        energy = restricted_superblock_hamiltonian[0, 0]
    else:
        energy, restricted_psi0 = diagonalize_superblock(restricted_superblock_hamiltonian, restricted_psi0_guess)

    # Construct each block of the reduced density matrix of the system by
    # tracing out the environment
    rho_block_dict = {}
    for sys_enl_sector, indices in sector_indices.items():
        if indices: # if indices is nonempty
            psi0_sector = restricted_psi0[indices]
            # We want to make the (sys, env) indices correspond to (row,
            # column) of a matrix, respectively.  Since the environment
            # (column) index updates most quickly in our Kronecker product
            # structure, psi0_sector is thus row-major ("C style").
            psi0_sector = psi0_sector.reshape([len(sys_enl_basis_by_sector[sys_enl_sector]), -1], order="C")
            rho_block_dict[sys_enl_sector] = np.dot(psi0_sector, psi0_sector.conjugate().transpose())

    # Diagonalize each block of the reduced density matrix and sort the
    # eigenvectors by eigenvalue.
    possible_eigenstates = []
    for sector, rho_block in rho_block_dict.items():
        evals, evecs = np.linalg.eigh(rho_block)
        current_sector_basis = sys_enl_basis_by_sector[sector]
        for eval, evec in zip(evals, evecs.transpose()):
            possible_eigenstates.append((eval, evec, sector, current_sector_basis))
    possible_eigenstates.sort(reverse=True, key=lambda x: x[0])  # largest eigenvalue first

    # Build the transformation matrix from the `m` overall most significant
    # eigenvectors.  It will have sparse structure due to the conserved quantum
    # number.
    my_m = min(len(possible_eigenstates), m)
    transformation_matrix = lil_matrix((sys_enl.basis_size, my_m), dtype=model.dtype)
    new_sector_array = np.zeros((my_m,), model.dtype)  # lists the sector of each
                                                        # element of the new/truncated basis
    for i, (eval, evec, sector, current_sector_basis) in enumerate(possible_eigenstates[:my_m]):
        for j, v in zip(current_sector_basis, evec):
            transformation_matrix[j, i] = v
        new_sector_array[i] = sector
    # Convert the transformation matrix to a more efficient internal
    # representation.  `lil_matrix` is good for constructing a sparse matrix
    # efficiently, but `csr_matrix` is better for performing quick
    # multiplications.
    transformation_matrix = transformation_matrix.tocsr()

    # Calculate the truncation error
    truncation_error = 1 - sum([x[0] for x in possible_eigenstates[:my_m]])

    # Rotate and truncate each operator.
    new_operator_dict = {}
    for name, op in sys_enl.operator_dict.items():
        new_operator_dict[name] = rotate_and_truncate(op, transformation_matrix)

    newblock = Block(length=sys_enl.length,
                    basis_size=my_m,
                    operator_dict=new_operator_dict,
                    basis_sector_array=new_sector_array)

    # Construct psi0 (that is, in the full superblock basis) so we can use it
    # later for eigenstate prediction.
    psi0 = np.zeros([m_sys_enl * m_env_enl], model.dtype)
    for i, z in enumerate(restricted_basis_indices):
        psi0[z] = restricted_psi0[i]
    assert np.all(psi0[restricted_basis_indices] == restricted_psi0)

    # Determine the overlap between psi0_guess and psi0
    if psi0_guess is not None:
        overlap = np.absolute(np.dot(psi0_guess.conjugate().transpose(), psi0).item())
        overlap /= np.linalg.norm(psi0_guess) * np.linalg.norm(psi0)  # normalize it
    else:
        overlap = None

    L = sys_enl.length + env_enl.length
    callback(StepInfo(truncation_error=truncation_error,
                    overlap=overlap,
                    energy=energy,
                    L=L))

    return newblock, energy, transformation_matrix, psi0, restricted_basis_indices
#
def default_callback(step_info):
    print("truncation error", step_info.truncation_error)
    if step_info.overlap is not None:
        print("overlap |<psi0_guess|psi0>| =", step_info.overlap)
    print("E/L =", step_info.energy / step_info.L)
    print("E   =", step_info.energy)
    sys.stdout.flush()

def default_graphic_callback(current_graphic):
    print(current_graphic)
    sys.stdout.flush()
#
if __name__ == "__main__":
    # 观察周期边界条件下，无限大 enlarge block 的过程
    np.set_printoptions(precision=10, suppress=True, threshold=10000, linewidth=300)
    print("-------------enlarge block in periodic boundary condition--------------")
    boundary_condition = periodic_bc
    J = 1.0
    Jz = 1.0
    L = 20
    model = HeisenbergSpinHalfXXZChain(J=J, Jz=Jz, boundary_condition=boundary_condition)
    #
    m_warmup = 10
    target_sector = 0
    m_sweep_list = [30]
    for m in m_sweep_list:
        #dmrg = DMRG(model, L=L, m_warmup=m_warmup, target_sector=target_sector,
        #            callback=default_callback, graphic_callback=default_graphic_callback)
        block_disk = {}  # "disk" storage for Block objects
        trmat_disk = {}  # "disk" storage for transformation matrices
        block = model.initial_block(0)
        print("block:\n", block)
        assert block.length == 1
        block_disk["l", 1] = block
        diagonalize_superblock=arpack_diagonalize_superblock
        callback=default_callback 
        graphic_callback=default_graphic_callback
        while 2 * block.length < L:
            # Perform a single DMRG step and save the new Block to "disk"
            if graphic_callback is not None:
                graphic_callback(graphic(model.boundary_condition, block, block))
            current_L = 2 * block.length + 2  # current superblock length
            if target_sector is not None:
                # assumes the value is extensive
                current_target_sector = int(target_sector) * current_L // L
            else:
                current_target_sector = None
            block, energy, transformation_matrix, psi0, rbi = single_dmrg_step(
                model, block, block, m=m_warmup, direction="r", 
                diagonalize_superblock=diagonalize_superblock, 
                target_sector=current_target_sector, 
                callback=callback)
            block_disk["l", block.length] = block
            block_disk["r", block.length] = block
        # Assuming a site-dependent Hamiltonian, the infinite system algorithm
        # above actually used the wrong superblock Hamiltonian, since the left
        # block was mirrored and used as the environment.  This mistake will be
        # fixed during the finite system algorithm sweeps below as long as we begin
        # with the correct initial block of the right-hand system.
        if model.boundary_condition == open_bc:
            right_initial_block_site_index = L - 1  # right-most site
        else:
            right_initial_block_site_index = L - 2  # second site from right
        # 这里在代码设计上主要考虑到了无序磁场的情况，如果磁场是无序的，那么L-2个格点的单格点hzSz 和
        # L-1个格点的单格点 hzSz 不一样，对于周期边界条件来说，从右开始的初始块矩阵必须的 L-2 个格点的算符矩阵。
        block_disk["r", 1] = model.initial_block(right_initial_block_site_index)
        print("right_initial_block_site_index:", right_initial_block_site_index)
        for index in list(block_disk.keys()):
            print("index:", index, "block_size:", block_disk[index][1])
        print("block:\n", block_disk[("r",10)])
        #
        print("block:", )