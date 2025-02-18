import hydra
import torch
import pandas as pd


from MeMDLM.src.scripts.infill import Infiller

cfg_pth = '/workspace/sg666/MeMDLM/MeMDLM/configs'
data_path = "/workspace/sg666/MeMDLM/MeMDLM/data/aquaporins"

def read_fasta(fasta_path):
    sequences = {}
    with open(fasta_path, "r") as file:
        current_id, current_seq = None, []
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                if current_id:  # Save previous sequence
                    sequences[current_id] = "".join(current_seq)
                current_id = line[1:]  # Remove ">" and set new ID
                current_seq = []  # Reset sequence list
            else:
                current_seq.append(line)
        if current_id:  # Save last sequence – split it to get only the ID from the PDB_Chain format
            sequences[current_id] = "".join(current_seq)
    sequences = {id.split("_")[-1]: seq for id, seq in sequences.items()}
    return sequences


def get_exposed_residues(csv_path):
    residues = pd.read_csv(csv_path)['exposed'].tolist()
    exposed_residues = {}
    for residue in residues:
        chain, residue_id = residue.split("_")
        if chain not in exposed_residues:
            exposed_residues[chain] = []
        exposed_residues[chain].append(int(residue_id))
    return {k: sorted(v) for k, v in exposed_residues.items()}


def optimize_tm_region(sequences, exposed_residues, infiller, monomer: bool, guidance: bool):
    if monomer:
        optimized = {k: {"OG Sequence": v, "Optimized Sequence": None} for k, v in sequences.items() if k=="A"}
    else:
        optimized = {k: {"OG Sequence": v, "Optimized Sequence": None} for k, v in sequences.items()}
    for id, dct in optimized.items():
        mask_indices = exposed_residues.get(id, [])
        infilled_seq, _ = infiller.infill(sequence=dct['OG Sequence'], mask_indices=mask_indices, guidance=guidance)
        dct['Optimized Sequence'] = infilled_seq
    return optimized


def optimize_soluble_region(sequences, exposed_residues, infiller, monomer: bool, guidance: bool):
    if monomer:
        optimized = {k: {"OG Sequence": v, "Optimized Sequence": None} for k, v in sequences.items() if k=="A"}
    else:
        optimized = {k: {"OG Sequence": v, "Optimized Sequence": None} for k, v in sequences.items()}
    print(optimized)
    for id, dct in optimized.items():
        print(f'dct id: {id}')
        mask_indices = exposed_residues.get(id, [])
        print(f'exposed res ids: {mask_indices}')
        fasta_idxs = [i for i, c in enumerate(sequences[id]) if c.islower()]
        print(f'fasta idxs: {fasta_idxs}')
        sol_idxs = sorted(list(set(mask_indices + fasta_idxs)))
        print(f'sol idxs: {sol_idxs}')
        infilled_seq, _ = infiller.infill(sequence=dct['OG Sequence'], mask_indices=sol_idxs, guidance=guidance)
        dct['Optimized Sequence'] = infilled_seq
    return optimized


@hydra.main(version_base=None, config_path=cfg_pth, config_name='config')
def main(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    infiller = Infiller(config, device)
    
    sequences = read_fasta(data_path + "/EcAqpZ_annotation.fasta")
    mono_exposed_res = get_exposed_residues(data_path + "/EcAqpZmonomer_lipid_exposed_residues.csv")
    tetra_exposed_res = get_exposed_residues(data_path + "/EcAqpZtetramer_lipid_exposed_residues.csv")
    
    # tetra_tms = optimize_tm_region(sequences, tetra_exposed_res, infiller, guidance=True)
    # monomer_tms = optimize_tm_region(sequences, mono_exposed_res, infiller, guidance=True)
    # tetra_sol = optimize_soluble_region(sequences, tetra_exposed_res, infiller, guidance=True)
    mono_sol = optimize_soluble_region(sequences, mono_exposed_res, infiller, monomer=True, guidance=True)

    # print(tetra_tms)
    # print(monomer_tms)
    # print(tetra_sol)
    print(mono_sol)

if __name__ == "__main__":
    main()
