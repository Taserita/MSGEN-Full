# ------------------------------------------------------------------------------
# Joint Molecular Conformer Generation for MSGEN Framework with GeoDiff(Stage 1 + Stage 2)
#
# This script performs *both* Stage 1 (backbone generation) and Stage 2 (full-atom
# generation) in a single run. The Stage 1 model generates heavy-atom-only structures,
# and the output is used to guide the Stage 2 model in reconstructing full conformations,
# including hydrogens.
#
# Note:
# - You may run Stage 1 and Stage 2 separately if needed.
# - To decouple, first generate backbone conformations and save them.
#   Then load them as guidance input in a separate Stage 2-only script.
#
# This joint pipeline is useful for direct evaluation.
# ------------------------------------------------------------------------------
import os
import argparse
import pickle
import yaml
import torch
from glob import glob
from tqdm.auto import tqdm
from easydict import EasyDict

from models.epsnet import *
from utils.datasets import *
from utils.transforms import *
from utils.misc import *

def num_confs(num:str):
    if num.endswith('x'):
        return lambda x:x*int(num[:-1])
    elif int(num) > 0: 
        return lambda x:int(num)
    else:
        raise ValueError()

def H_upsampling(pos_gen, batch, tau=3):
    # H upsampling
    ptr = batch.ptr

    up_pos = torch.tensor([]).to(pos_gen.device)
    for i in range(len(ptr) - 1):
        st = ptr[i]
        ed = ptr[i + 1]
        
        pos_i = pos_gen[st:ed]
        h_c = batch.h_connection[i]

        original_num = batch.orig_size[i]

        index = batch.orig_idx[st:ed]

        pos = torch.zeros(size=(original_num, 3)).to(pos_gen.device)
        pos[index] = pos_i
            
        for h_id, heavy in h_c.items():
            gen_pos = torch.randn(1, 3)
            gen_pos = gen_pos / torch.norm(gen_pos)
            delta = (tau * gen_pos).to(pos_gen.device)
            pos[h_id] = pos[heavy] + delta 
        
        up_pos = torch.cat((up_pos, pos), dim=0)
        
    return up_pos

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt1', type=str, help='path for loading the stage1 checkpoint')
    parser.add_argument('ckpt2', type=str, help='path for loading the stage2 checkpoint')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--num_confs', type=num_confs, default=num_confs('2x'))
    parser.add_argument('--test_set', type=str, default=None)
    parser.add_argument('--start_idx', type=int, default=800)
    parser.add_argument('--end_idx', type=int, default=1000)
    parser.add_argument('--out_dir', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--clip', type=float, default=1000.0)
    parser.add_argument('--n_steps', type=int, default=5000,
                    help='sampling num steps; for DSM framework, this means num steps for each noise scale')
    parser.add_argument('--global_start_sigma', type=float, default=0.5,
                    help='enable global gradients only when noise is low')
    parser.add_argument('--w_global', type=float, default=1.0,
                    help='weight for global gradients')
    # Parameters for DDPM
    parser.add_argument('--sampling_type', type=str, default='ld',
                    help='generalized, ddpm_noisy, ld: sampling method for DDIM, DDPM or Langevin Dynamics')
    parser.add_argument('--eta', type=float, default=1.0,
                    help='weight for DDIM and DDPM: 0->DDIM, 1->DDPM')
    args = parser.parse_args()

    # Stage 1
    ckpt_1 = torch.load(args.ckpt1)
    config_path = glob(os.path.join(os.path.dirname(os.path.dirname(args.ckpt1)), '*.yml'))[0]
    with open(config_path, 'r') as f:
        config1 = EasyDict(yaml.safe_load(f))

    # Stage 2
    ckpt_2 = torch.load(args.ckpt2)
    config_path = glob(os.path.join(os.path.dirname(os.path.dirname(args.ckpt2)), '*.yml'))[0]
    with open(config_path, 'r') as f:
        config2 = EasyDict(yaml.safe_load(f))

    seed_all(config1.train.seed)
    log_dir = os.path.dirname(os.path.dirname(args.ckpt2))

    # Logging
    output_dir = get_new_log_dir(log_dir, 'sample', tag=args.tag)
    logger = get_logger('test', output_dir)
    logger.info(args)

    # Datasets
    logger.info('Loading datasets...')
    transforms_1 = Compose([
        CountNodesPerGraph(),
        AddHigherOrderEdges(order=config1.model.edge_order), # Offline edge augmentation
    ])

    transforms_2 = Compose([
        CountNodesPerGraph(),
        AddHigherOrderEdges(order=config2.model.edge_order), # Offline edge augmentation
    ])

    if args.test_set is None:
        BBtest_set = PackedBackboneConformationDataset(config1.dataset.test, transform=transforms_1)
        test_set = PackedConformationDataset(config2.dataset.test, transform=transforms_2)
        # test_set = PackedConditionalConformationDataset(config.dataset.test, transform=transforms_2)
    else:
        BBtest_set = PackedBackboneConformationDataset(args.test_set, transform=transforms_1)
        test_set = PackedConformationDataset(args.test_set, transform=transforms_2)
    
    # Model for stage 1 and stage 2
    logger.info('Loading model...')

    model = get_model(ckpt_1['config'].model).to(args.device)
    model.load_state_dict(ckpt_1['model'])

    model2 = get_model(ckpt_2['config'].model).to(args.device)
    model2.load_state_dict(ckpt_2['model'])

    test_set_selected = []
    BBtest_set_selected = []
    for i, data in enumerate(test_set):
        if not (args.start_idx <= i < args.end_idx): continue
        test_set_selected.append(data)
        BBtest_set_selected.append(BBtest_set[i])

    done_smiles = set()
    results = []
    if args.resume is not None:
        with open(args.resume, 'rb') as f:
            results = pickle.load(f)
        for data in results:
            done_smiles.add(data.smiles)
    
    for i, (data, databb) in enumerate(tqdm(zip(test_set_selected, BBtest_set_selected), total=len(test_set_selected))):
        if data.smiles in done_smiles:
            logger.info('Molecule#%d is already done.' % i)
            continue

        num_refs = data.pos_ref.size(0) // data.num_nodes
        num_samples = args.num_confs(num_refs)
        
        # BB generation
        BB_input = databb.clone()
        BB_input['pos_ref'] = None
        batch = repeat_data(BB_input, num_samples).to(args.device)

        clip_local = None
        for _ in range(2):  # Maximum number of retry
            try:
                pos_init = torch.randn(batch.num_nodes, 3).to(args.device)
                pos_gen, pos_gen_traj = model.langevin_dynamics_sample(
                    atom_type=batch.atom_type,
                    pos_init=pos_init,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    extend_order=False, # Done in transforms.
                    n_steps=args.n_steps, # args.n_steps for stage 1
                    step_lr=1e-6,
                    w_global=args.w_global,
                    global_start_sigma=args.global_start_sigma,
                    clip=args.clip,
                    clip_local=clip_local,
                    sampling_type=args.sampling_type,
                    eta=args.eta
                )
                break   # No errors occured, break the retry loop
            except FloatingPointError:
                clip_local = 20
                logger.warning('Retrying with local clipping.')
        
        # pos_gen = (batch.num_nodes, 3)
        n = pos_gen.shape[0] // num_samples
        pos_gen = pos_gen.reshape(num_samples, n, 3)

        # Molecular Generation
        data_input = data.clone()
        data_input['pos_ref'] = None

        # Upsampling only for H
        # up_pos = H_upsampling(pos_gen, batch)

        # Chemical Upsampling
        coarse_I = databb.orig_idx.to(args.device)
        tot_num = len(data.atom_type)
        fine_I = torch.arange(tot_num).to(args.device)
        up_pos = [Chemical_upsampling(data_input.edge_index, coarse_I, fine_I, pos_gen[i], tot_num) for i in range(num_samples)]
        up_pos = torch.cat(up_pos, dim=0).to(args.device)

        batch = repeat_data(data_input, num_samples).to(args.device)

        clip_local = None
        for _ in range(2):  # Maximum number of retry
            try:
                pos_init = torch.randn(batch.num_nodes, 3).to(args.device)
                assert pos_init.shape == up_pos.shape
                
                pos_gen, pos_gen_traj = model2.langevin_dynamics_sample(
                    atom_type=batch.atom_type,
                    pos_init=pos_init,
                    pos_guide=up_pos,
                    bond_index=batch.edge_index,
                    bond_type=batch.edge_type,
                    batch=batch.batch,
                    num_graphs=batch.num_graphs,
                    extend_order=False, # Done in transforms.
                    n_steps=args.n_steps,
                    step_lr=1e-6,
                    w_global=args.w_global,
                    global_start_sigma=args.global_start_sigma,
                    clip=args.clip,
                    clip_local=clip_local,
                    sampling_type=args.sampling_type,
                    eta=args.eta
                )
                pos_gen = pos_gen.cpu()
                data.pos_gen = pos_gen
                data.pos_guide = up_pos # consider the supervision influence
                results.append(data)
                done_smiles.add(data.smiles)

                save_path = os.path.join(output_dir, 'samples_%d.pkl' % i)
                logger.info('Saving samples to: %s' % save_path)
                with open(save_path, 'wb') as f:
                    pickle.dump(results, f)

                break   # No errors occured, break the retry loop
            except FloatingPointError:
                clip_local = 20
                logger.warning('Retrying with local clipping.')
                
    save_path = os.path.join(output_dir, 'samples_all.pkl')
    logger.info('Saving samples to: %s' % save_path)


    def get_mol_key(data):
        for i, d in enumerate(test_set_selected):
            if d.smiles == data.smiles:
                return i
        return -1
    results.sort(key=get_mol_key)

    with open(save_path, 'wb') as f:
        pickle.dump(results, f)
        