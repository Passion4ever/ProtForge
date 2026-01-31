"""
Constants for protforge MPNN module.
Amino acid mappings, chemical elements, and atom naming conventions.
"""

# Amino acid 1-letter to 3-letter mapping
_AA_1TO3 = "A:ALA,R:ARG,N:ASN,D:ASP,C:CYS,Q:GLN,E:GLU,G:GLY,H:HIS,I:ILE,L:LEU,K:LYS,M:MET,F:PHE,P:PRO,S:SER,T:THR,W:TRP,Y:TYR,V:VAL,X:UNK"
restype_1to3 = dict(item.split(":") for item in _AA_1TO3.split(","))

# Amino acid ordering (alphabetical except X at end)
_AA_ORDER = "ACDEFGHIKLMNPQRSTVWYX"
restype_str_to_int = {aa: i for i, aa in enumerate(_AA_ORDER)}
restype_int_to_str = {i: aa for i, aa in enumerate(_AA_ORDER)}
alphabet = list(_AA_ORDER)

# Non-standard amino acid mappings (3-letter to 1-letter)
_NONSTANDARD_AA = {
    "MSE": "M", "SEC": "C", "PYL": "K",  # Selenomethionine, Selenocysteine, Pyrrolysine
    "SEP": "S", "TPO": "T", "PTR": "Y",  # Phosphorylated
    "CSO": "C", "CSD": "C", "CME": "C", "OCS": "C", "CSS": "C",  # Modified cysteine
    "HYP": "P", "HYL": "K",  # Hydroxylated
    "MLY": "K", "M3L": "K", "MLZ": "K",  # Methylated lysine
    "PCA": "E", "KCX": "K", "LLP": "K", "NEP": "H", "AIB": "A",  # Other
    "DAL": "A", "DVA": "V", "DLE": "L", "DIL": "I", "DPR": "P",  # D-amino acids
    "DSN": "S", "DTH": "T", "MEN": "N", "CGU": "E", "FME": "M",
}
restype_3to1 = {v: k for k, v in restype_1to3.items() if k != "X"}
restype_3to1.update(_NONSTANDARD_AA)

# Chemical elements (periodic table)
_ELEMENTS = (
    "H,He,Li,Be,B,C,N,O,F,Ne,Na,Mg,Al,Si,P,S,Cl,Ar,K,Ca,"
    "Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Ga,Ge,As,Se,Br,Kr,"
    "Rb,Sr,Y,Zr,Nb,Mb,Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Sb,Te,I,Xe,"
    "Cs,Ba,La,Ce,Pr,Nd,Pm,Sm,Eu,Gd,Tb,Dy,Ho,Er,Tm,Yb,Lu,"
    "Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi,Po,At,Rn,"
    "Fr,Ra,Ac,Th,Pa,U,Np,Pu,Am,Cm,Bk,Cf,Es,Fm,Md,No,Lr,"
    "Rf,Db,Sg,Bh,Hs,Mt,Ds,Rg,Cn,Uut,Fl,Uup,Lv,Uus,Uuo"
).upper().split(",")
element_dict = {el: i + 1 for i, el in enumerate(_ELEMENTS)}
element_dict_rev = {i + 1: el for i, el in enumerate(_ELEMENTS)}

# Atom ordering for protein structures (37 atoms)
_ATOM_NAMES = (
    "N,CA,C,CB,O,CG,CG1,CG2,OG,OG1,SG,CD,CD1,CD2,ND1,ND2,OD1,OD2,SD,"
    "CE,CE1,CE2,CE3,NE,NE1,NE2,OE1,OE2,CH2,NH1,NH2,OH,CZ,CZ2,CZ3,NZ,OXT"
).split(",")
atom_order = {name: i for i, name in enumerate(_ATOM_NAMES)}

# Atom names for each residue type (14 atoms max, for PDB writing)
restype_name_to_atom14_names = {
    "ALA": ["N", "CA", "C", "O", "CB", "", "", "", "", "", "", "", "", ""],
    "ARG": ["N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2", "", "", ""],
    "ASN": ["N", "CA", "C", "O", "CB", "CG", "OD1", "ND2", "", "", "", "", "", ""],
    "ASP": ["N", "CA", "C", "O", "CB", "CG", "OD1", "OD2", "", "", "", "", "", ""],
    "CYS": ["N", "CA", "C", "O", "CB", "SG", "", "", "", "", "", "", "", ""],
    "GLN": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2", "", "", "", "", ""],
    "GLU": ["N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2", "", "", "", "", ""],
    "GLY": ["N", "CA", "C", "O", "", "", "", "", "", "", "", "", "", ""],
    "HIS": ["N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2", "", "", "", ""],
    "ILE": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1", "", "", "", "", "", ""],
    "LEU": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "", "", "", "", "", ""],
    "LYS": ["N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ", "", "", "", "", ""],
    "MET": ["N", "CA", "C", "O", "CB", "CG", "SD", "CE", "", "", "", "", "", ""],
    "PHE": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "", "", ""],
    "PRO": ["N", "CA", "C", "O", "CB", "CG", "CD", "", "", "", "", "", "", ""],
    "SER": ["N", "CA", "C", "O", "CB", "OG", "", "", "", "", "", "", "", ""],
    "THR": ["N", "CA", "C", "O", "CB", "OG1", "CG2", "", "", "", "", "", "", ""],
    "TRP": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE2", "CE3", "NE1", "CZ2", "CZ3", "CH2"],
    "TYR": ["N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ", "OH", "", ""],
    "VAL": ["N", "CA", "C", "O", "CB", "CG1", "CG2", "", "", "", "", "", "", ""],
    "UNK": ["", "", "", "", "", "", "", "", "", "", "", "", "", ""],
}

# Atoms present for each residue type (derived from atom14 names, filtering empty strings)
residue_atoms = {
    resname: [atom for atom in atoms if atom]
    for resname, atoms in restype_name_to_atom14_names.items()
}
