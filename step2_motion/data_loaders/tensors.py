# This code is based on https://github.com/GuyTevet/motion-diffusion-model
import torch
import numpy

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def collate(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['inp'] for b in notnone_batches]
    if 'lengths' in notnone_batches[0]:
        lenbatch = [b['lengths'] for b in notnone_batches]
    else:
        lenbatch = [len(b['inp'][0][0]) for b in notnone_batches]


    databatchTensor = collate_tensors(databatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = databatchTensor
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenbatchTensor}}

    if 'text' in notnone_batches[0]:
        textbatch = [b['text'] for b in notnone_batches]
        cond['y'].update({'text': textbatch})

    if 'tokens' in notnone_batches[0]:
        textbatch = [b['tokens'] for b in notnone_batches]
        cond['y'].update({'tokens': textbatch})

    if 'action' in notnone_batches[0]:
        actionbatch = [b['action'] for b in notnone_batches]
        cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # collate action textual names
    if 'action_text' in notnone_batches[0]:
        action_text = [b['action_text']for b in notnone_batches]
        cond['y'].update({'action_text': action_text})

    if 'seq_name' in notnone_batches[0]:
        seq_name = [b['seq_name']for b in notnone_batches]
        cond['y'].update({'seq_name': seq_name})
    
    if 'obj_points' in notnone_batches[0]:
        obj_points = [b['obj_points']for b in notnone_batches]
        cond['y'].update({'obj_points': torch.as_tensor(obj_points)})
    
    if 'hint' in notnone_batches[0] and notnone_batches[0]['hint'] is not None:
        hint = [b['hint'] for b in notnone_batches]
        # 找到最大的k值
        max_k = notnone_batches[0]['max_abs_num']
        # 对每个hint进行padding
        padded_hint = []
        for h in hint:
            t, k = h.shape
            pad_width = ((0, 0), (0, max_k*3*8 - k))  # 在k维度上padding
            padded = numpy.pad(h, pad_width, mode='constant', constant_values=0)
            padded_hint.append(padded)
        cond['y'].update({'hint': torch.as_tensor(numpy.array(padded_hint))})
    if 'filename' in notnone_batches[0]:
        filename = [b['filename'] for b in notnone_batches]
        cond['y'].update({'filename': filename})
    if 'position' in notnone_batches[0]:
        position = [b['position'] for b in notnone_batches]
        cond['y'].update({'position': position})
    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
        'hint': b[-4],
        'max_abs_num': b[-3],
        'filename': b[-2],
        'position': b[-1],
    } for b in batch]
    return collate(adapted_batch)

