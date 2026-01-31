"""
PDB parsing and featurization utilities for protforge MPNN module.
"""

import numpy as np
import torch
from prody import parsePDB, writePDB, confProDy, AtomGroup

from .constants import (
    restype_1to3,
    restype_3to1,
    restype_int_to_str,
    restype_str_to_int,
    restype_name_to_atom14_names,
    element_dict,
    atom_order,
    _ATOM_NAMES,
)

confProDy(verbosity="none")


def get_seq_rec(S: torch.Tensor, S_pred: torch.Tensor, mask: torch.Tensor):
    """
    Compute sequence recovery rate.

    S : true sequence shape=[batch, length]
    S_pred : predicted sequence shape=[batch, length]
    mask : mask to compute average over the region shape=[batch, length]

    Returns: averaged sequence recovery shape=[batch]
    """
    match = S == S_pred
    average = torch.sum(match * mask, dim=-1) / torch.sum(mask, dim=-1)
    return average


def get_score(S: torch.Tensor, log_probs: torch.Tensor, mask: torch.Tensor):
    """
    Compute sequence score (cross entropy loss).

    S : true sequence shape=[batch, length]
    log_probs : predicted log probabilities shape=[batch, length, 21]
    mask : mask to compute average over the region shape=[batch, length]

    Returns:
        average_loss : averaged categorical cross entropy (CCE) [batch]
        loss_per_residue : per position CCE [batch, length]
    """
    S_one_hot = torch.nn.functional.one_hot(S, 21)
    loss_per_residue = -(S_one_hot * log_probs).sum(-1)  # [B, L]
    average_loss = torch.sum(loss_per_residue * mask, dim=-1) / (
        torch.sum(mask, dim=-1) + 1e-8
    )
    return average_loss, loss_per_residue


def write_full_PDB(
    save_path: str,
    X: np.ndarray,
    X_m: np.ndarray,
    b_factors: np.ndarray,
    R_idx: np.ndarray,
    chain_letters: np.ndarray,
    S: np.ndarray,
    other_atoms=None,
    icodes=None,
    force_hetatm=False,
):
    """
    Write protein structure to PDB file.

    save_path : path where the PDB will be written to
    X : protein atom xyz coordinates shape=[length, 14, 3]
    X_m : protein atom mask shape=[length, 14]
    b_factors: shape=[length, 14]
    R_idx: protein residue indices shape=[length]
    chain_letters: protein chain letters shape=[length]
    S : protein amino acid sequence shape=[length]
    other_atoms: other atoms parsed by prody
    icodes: a list of insertion codes for the PDB; e.g. antibody loops
    """
    S_str = [restype_1to3[restype_int_to_str[AA]] for AA in S]

    X_list = []
    b_factor_list = []
    atom_name_list = []
    element_name_list = []
    residue_name_list = []
    residue_number_list = []
    chain_id_list = []
    icodes_list = []

    for i, AA in enumerate(S_str):
        sel = X_m[i].astype(np.int32) == 1
        total = np.sum(sel)
        tmp = np.array(restype_name_to_atom14_names[AA])[sel]
        X_list.append(X[i][sel])
        b_factor_list.append(b_factors[i][sel])
        atom_name_list.append(tmp)
        element_name_list += [AA[:1] for AA in list(tmp)]
        residue_name_list += total * [AA]
        residue_number_list += total * [R_idx[i]]
        chain_id_list += total * [chain_letters[i]]
        icodes_list += total * [icodes[i]]

    X_stack = np.concatenate(X_list, 0)
    b_factor_stack = np.concatenate(b_factor_list, 0)
    atom_name_stack = np.concatenate(atom_name_list, 0)

    protein = AtomGroup()
    protein.setCoords(X_stack)
    protein.setBetas(b_factor_stack)
    protein.setNames(atom_name_stack)
    protein.setResnames(residue_name_list)
    protein.setElements(element_name_list)
    protein.setOccupancies(np.ones([X_stack.shape[0]]))
    protein.setResnums(residue_number_list)
    protein.setChids(chain_id_list)
    protein.setIcodes(icodes_list)

    if other_atoms:
        other_atoms_g = AtomGroup()
        other_atoms_g.setCoords(other_atoms.getCoords())
        other_atoms_g.setNames(other_atoms.getNames())
        other_atoms_g.setResnames(other_atoms.getResnames())
        other_atoms_g.setElements(other_atoms.getElements())
        other_atoms_g.setOccupancies(other_atoms.getOccupancies())
        other_atoms_g.setResnums(other_atoms.getResnums())
        other_atoms_g.setChids(other_atoms.getChids())
        if force_hetatm:
            other_atoms_g.setFlags("hetatm", other_atoms.getFlags("hetatm"))
        writePDB(save_path, protein + other_atoms_g)
    else:
        writePDB(save_path, protein)


def _get_aligned_coordinates(protein_atoms, CA_dict: dict, atom_name: str):
    """
    Get aligned coordinates for a specific atom type.

    protein_atoms: prody atom group
    CA_dict: mapping between chain_residue_idx_icodes and integers
    atom_name: atom to be parsed; e.g. CA
    """
    atom_atoms = protein_atoms.select(f"name {atom_name}")

    if atom_atoms is not None:
        atom_coords = atom_atoms.getCoords()
        atom_resnums = atom_atoms.getResnums()
        atom_chain_ids = atom_atoms.getChids()
        atom_icodes = atom_atoms.getIcodes()

    atom_coords_ = np.zeros([len(CA_dict), 3], np.float32)
    atom_coords_m = np.zeros([len(CA_dict)], np.int32)

    if atom_atoms is not None:
        for i in range(len(atom_resnums)):
            code = atom_chain_ids[i] + "_" + str(atom_resnums[i]) + "_" + atom_icodes[i]
            if code in CA_dict:
                atom_coords_[CA_dict[code], :] = atom_coords[i]
                atom_coords_m[CA_dict[code]] = 1

    return atom_coords_, atom_coords_m


def parse_PDB(
    input_path: str,
    device: str = "cpu",
    chains: list = [],
    parse_all_atoms: bool = False,
    parse_atoms_with_zero_occupancy: bool = False
):
    """
    Parse PDB file and extract protein features.

    input_path : path for the input PDB
    device: device for the torch.Tensor
    chains: a list specifying which chains need to be parsed; e.g. ["A", "B"]
    parse_all_atoms: if False parse only N,CA,C,O otherwise all 37 atoms
    parse_atoms_with_zero_occupancy: if True atoms with zero occupancy will be parsed
    """
    atom_types = _ATOM_NAMES[:-1] if parse_all_atoms else ["N", "CA", "C", "O"]

    atoms = parsePDB(input_path)
    if not parse_atoms_with_zero_occupancy:
        atoms = atoms.select("occupancy > 0")
    if chains:
        chain_sel = " or ".join(f"chain {c}" for c in chains)
        atoms = atoms.select(chain_sel)

    protein_atoms = atoms.select("protein")
    backbone = protein_atoms.select("backbone")
    other_atoms = atoms.select("not protein and not water")
    water_atoms = atoms.select("water")

    CA_atoms = protein_atoms.select("name CA")
    CA_resnums = CA_atoms.getResnums()
    CA_chain_ids = CA_atoms.getChids()
    CA_icodes = CA_atoms.getIcodes()

    CA_dict = {
        f"{CA_chain_ids[i]}_{CA_resnums[i]}_{CA_icodes[i]}": i
        for i in range(len(CA_resnums))
    }

    xyz_37 = np.zeros([len(CA_dict), 37, 3], np.float32)
    xyz_37_m = np.zeros([len(CA_dict), 37], np.int32)

    for atom_name in atom_types:
        xyz, xyz_m = _get_aligned_coordinates(protein_atoms, CA_dict, atom_name)
        xyz_37[:, atom_order[atom_name], :] = xyz
        xyz_37_m[:, atom_order[atom_name]] = xyz_m

    N = xyz_37[:, atom_order["N"], :]
    CA = xyz_37[:, atom_order["CA"], :]
    C = xyz_37[:, atom_order["C"], :]
    O = xyz_37[:, atom_order["O"], :]

    N_m = xyz_37_m[:, atom_order["N"]]
    CA_m = xyz_37_m[:, atom_order["CA"]]
    C_m = xyz_37_m[:, atom_order["C"]]
    O_m = xyz_37_m[:, atom_order["O"]]

    mask = N_m * CA_m * C_m * O_m  # must all 4 atoms exist

    chain_labels = np.array(CA_atoms.getChindices(), dtype=np.int32)
    R_idx = np.array(CA_resnums, dtype=np.int32)
    S = CA_atoms.getResnames()
    S = [restype_3to1.get(AA, "X") for AA in list(S)]
    S = np.array([restype_str_to_int[AA] for AA in list(S)], np.int32)
    X = np.concatenate([N[:, None], CA[:, None], C[:, None], O[:, None]], 1)

    # Parse ligand/other atoms
    try:
        Y = np.array(other_atoms.getCoords(), dtype=np.float32)
        Y_t = list(other_atoms.getElements())
        Y_t = np.array([element_dict.get(y_t.upper(), 0) for y_t in Y_t], dtype=np.int32)
        Y_m = (Y_t != 1) * (Y_t != 0)
        Y = Y[Y_m, :]
        Y_t = Y_t[Y_m]
        Y_m = Y_m[Y_m]
    except:
        Y = np.zeros([1, 3], np.float32)
        Y_t = np.zeros([1], np.int32)
        Y_m = np.zeros([1], np.int32)

    output_dict = {
        "X": torch.tensor(X, device=device, dtype=torch.float32),
        "mask": torch.tensor(mask, device=device, dtype=torch.int32),
        "Y": torch.tensor(Y, device=device, dtype=torch.float32),
        "Y_t": torch.tensor(Y_t, device=device, dtype=torch.int32),
        "Y_m": torch.tensor(Y_m, device=device, dtype=torch.int32),
        "R_idx": torch.tensor(R_idx, device=device, dtype=torch.int32),
        "chain_labels": torch.tensor(chain_labels, device=device, dtype=torch.int32),
        "chain_letters": CA_chain_ids,
        "S": torch.tensor(S, device=device, dtype=torch.int32),
        "xyz_37": torch.tensor(xyz_37, device=device, dtype=torch.float32),
        "xyz_37_m": torch.tensor(xyz_37_m, device=device, dtype=torch.int32),
    }

    # Chain masks
    chain_list = sorted(set(output_dict["chain_letters"]))
    mask_c = [
        torch.tensor(
            [chain == item for item in output_dict["chain_letters"]],
            device=device,
            dtype=bool,
        )
        for chain in chain_list
    ]
    output_dict["mask_c"] = mask_c
    output_dict["chain_list"] = chain_list

    return output_dict, backbone, other_atoms, CA_icodes, water_atoms


def _get_nearest_neighbours(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms):
    """Get nearest ligand atoms for each residue."""
    device = CB.device
    mask_CBY = mask[:, None] * Y_m[None, :]  # [A,B]
    L2_AB = torch.sum((CB[:, None, :] - Y[None, :, :]) ** 2, -1)
    L2_AB = L2_AB * mask_CBY + (1 - mask_CBY) * 1000.0

    nn_idx = torch.argsort(L2_AB, -1)[:, :number_of_ligand_atoms]
    L2_AB_nn = torch.gather(L2_AB, 1, nn_idx)
    D_AB_closest = torch.sqrt(L2_AB_nn[:, 0])

    Y_r = Y[None, :, :].repeat(CB.shape[0], 1, 1)
    Y_t_r = Y_t[None, :].repeat(CB.shape[0], 1)
    Y_m_r = Y_m[None, :].repeat(CB.shape[0], 1)

    Y_tmp = torch.gather(Y_r, 1, nn_idx[:, :, None].repeat(1, 1, 3))
    Y_t_tmp = torch.gather(Y_t_r, 1, nn_idx)
    Y_m_tmp = torch.gather(Y_m_r, 1, nn_idx)

    Y_out = torch.zeros([CB.shape[0], number_of_ligand_atoms, 3], dtype=torch.float32, device=device)
    Y_t_out = torch.zeros([CB.shape[0], number_of_ligand_atoms], dtype=torch.int32, device=device)
    Y_m_out = torch.zeros([CB.shape[0], number_of_ligand_atoms], dtype=torch.int32, device=device)

    num_nn_update = Y_tmp.shape[1]
    Y_out[:, :num_nn_update] = Y_tmp
    Y_t_out[:, :num_nn_update] = Y_t_tmp
    Y_m_out[:, :num_nn_update] = Y_m_tmp

    return Y_out, Y_t_out, Y_m_out, D_AB_closest


def featurize(
    input_dict,
    cutoff_for_score=8.0,
    use_atom_context=True,
    number_of_ligand_atoms=16,
    model_type="protein_mpnn",
):
    """
    Featurize protein dictionary for model input.

    input_dict: parsed protein dictionary from parse_PDB
    cutoff_for_score: distance cutoff for ligand context (angstroms)
    use_atom_context: whether to use ligand atom context
    number_of_ligand_atoms: number of nearest ligand atoms to include
    model_type: model type (protein_mpnn, ligand_mpnn, etc.)
    """
    output_dict = {}

    if model_type == "ligand_mpnn":
        mask = input_dict["mask"]
        Y = input_dict["Y"]
        Y_t = input_dict["Y_t"]
        Y_m = input_dict["Y_m"]
        N = input_dict["X"][:, 0, :]
        CA = input_dict["X"][:, 1, :]
        C = input_dict["X"][:, 2, :]

        # Compute CB position
        b = CA - N
        c = C - CA
        a = torch.cross(b, c, axis=-1)
        CB = -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA

        Y, Y_t, Y_m, D_XY = _get_nearest_neighbours(CB, mask, Y, Y_t, Y_m, number_of_ligand_atoms)
        mask_XY = (D_XY < cutoff_for_score) * mask * Y_m[:, 0]
        output_dict["mask_XY"] = mask_XY[None,]

        if "side_chain_mask" in input_dict:
            output_dict["side_chain_mask"] = input_dict["side_chain_mask"][None,]

        output_dict["Y"] = Y[None,]
        output_dict["Y_t"] = Y_t[None,]
        output_dict["Y_m"] = Y_m[None,]

        if not use_atom_context:
            output_dict["Y_m"] = 0.0 * output_dict["Y_m"]

    elif model_type in ("per_residue_label_membrane_mpnn", "global_label_membrane_mpnn"):
        output_dict["membrane_per_residue_labels"] = input_dict["membrane_per_residue_labels"][None,]

    # Renumber R_idx to handle insertion codes
    R_idx_list = []
    count = 0
    R_idx_prev = -100000
    for R_idx in list(input_dict["R_idx"]):
        if R_idx_prev == R_idx:
            count += 1
        R_idx_list.append(R_idx + count)
        R_idx_prev = R_idx
    R_idx_renumbered = torch.tensor(R_idx_list, device=input_dict["R_idx"].device)

    output_dict["R_idx"] = R_idx_renumbered[None,]
    output_dict["R_idx_original"] = input_dict["R_idx"][None,]
    output_dict["chain_labels"] = input_dict["chain_labels"][None,]
    output_dict["S"] = input_dict["S"][None,]
    output_dict["chain_mask"] = input_dict["chain_mask"][None,]
    output_dict["mask"] = input_dict["mask"][None,]
    output_dict["X"] = input_dict["X"][None,]

    if "xyz_37" in input_dict:
        output_dict["xyz_37"] = input_dict["xyz_37"][None,]
        output_dict["xyz_37_m"] = input_dict["xyz_37_m"][None,]

    return output_dict
