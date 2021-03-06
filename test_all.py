from pyfaidx import Fasta
from pysam import VariantFile
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from collections import defaultdict
from .model import SpliceAI, CL_max
from bisect import bisect
from .utils import one_hot_encode
import torch
import os


class VCFDataset(Dataset):
    def __init__(self, vcf, fasta, annotations, cov=1001):
        self.fasta = Fasta(fasta)
        self.wid = CL_max + cov
        df = pd.read_csv(annotations, sep='\t')
        annotations = defaultdict(lambda: [])
        self.annotations = {}
        self.pos_ref = {}
        self.rev_dict = {'N': 'N', 'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C'}
        for _, a in df.iterrows():
            chrom = a['CHROM']
            if not chrom.startswith('chr'):
                chrom = 'chr' + chrom
            annotations[chrom].append((a['TX_START'] + 1, (a['TX_END'], a['#NAME'], a['STRAND'])))
        for k in annotations:
            v = annotations[k]
            self.annotations[k] = dict(v + [("TX_STARTS", sorted([x[0] for x in v]))])
        vcf = VariantFile(vcf)
        self.records = []
        for record in vcf:
            for alt in record.alts:
                chrom = record.chrom
                if not chrom.startswith('chr'):
                    chrom = 'chr' + chrom
                self.records.append((chrom, record.pos, record.ref, alt))

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        chrom, pos, ref, alt = self.records[idx]
        seq = self.fasta[chrom][pos - self.wid // 2 - 1:pos + self.wid // 2].seq
        tx_starts = self.annotations[chrom]["TX_STARTS"]
        idx_pos = bisect(tx_starts, pos) - 1
        if idx_pos == -1:
            return
        tx_start = tx_starts[idx_pos]
        tx_end, name, strand = self.annotations[chrom][tx_start]
        if pos > tx_end:
            return
        ref = '' if ref == '<NON_REF>' else ref
        alt = '' if alt == '.' else alt
        mask_left, mask_right = max(self.wid // 2 + tx_start - pos, 0), max(self.wid // 2 - tx_end + pos, 0)
        x_ref = 'N' * mask_left + seq[mask_left:self.wid - mask_right] + 'N' * mask_right
        pad = 'N' * max(0, len(ref) - len(alt))
        x_alt = x_ref[:self.wid // 2] + alt + x_ref[self.wid // 2 + len(ref):]
        x_alt = (x_alt + pad)[:self.wid]
        if strand == '-':
            x_ref = ''.join(self.rev_dict[x] for x in x_ref[::-1].upper())
            x_alt = ''.join(self.rev_dict[x] for x in x_alt[::-1].upper())
        return chrom[3:], pos, alt, name, torch.tensor(strand == '-'), torch.tensor(one_hot_encode(x_ref)), torch.tensor(one_hot_encode(x_alt)),\
               torch.tensor(len(ref)), torch.tensor(len(alt))


def custom_collate(batch):
    batch = list(filter(lambda x:x is not None, batch))
    if batch:
        return default_collate(batch)
    else:
        return None


def test(in_vcf, out_file='output', fasta_ref='pytorch_spliceai_test/hg19.fa',
         grch='pytorch_spliceai_test/annotations/grch37.txt', cov=1001):
    dataloader = DataLoader(VCFDataset(in_vcf, fasta_ref, grch, cov), batch_size=32, collate_fn=custom_collate)
    folder = os.path.dirname(os.path.realpath(__file__))
    model_states = [folder + '/model_{}.pth'.format(flank) for flank in ['flank_0', 'flank_5', 'flank_10', 'orig']]
    models = [SpliceAI() for state in model_states]
    for model, state in zip(models, model_states):
        model.load_state_dict(torch.load(state))
        model.eval()
    cuda = torch.cuda.is_available()
    if cuda:
        for i, model in enumerate(models):
            model.cuda(i)
    fs = [open(out_file + '.{}'.format(x), 'w+') for x in [0, 5, 10, 'orig']]
    for f in fs:
        f.write('These include delta scores (DS) and delta positions (DP) for '
                 'acceptor gain (AG), acceptor loss (AL), donor gain (DG), and donor loss (DL).\n'
                 'Format: CHROME|POS|ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|DP_AG|DP_AL|DP_DG|DP_DL">\n')
    for batch in dataloader:
        if batch:
            chroms, pos, alts, names, strands, x_refs, x_alts, ref_lens, alt_lens = batch
        else:
            continue
        x_r = []
        x_a = []
        r_l = []
        a_l = []
        if cuda:
            for i in range(len(models)):
                x_r.append(x_refs.cuda(i))
                x_a.append(x_alts.cuda(i))
                r_l.append(ref_lens.cuda(i))
                a_l.append(alt_lens.cuda(i))
        else:
            x_r = [x_refs] * len(models)
            x_a = [x_alts] * len(models)
            r_l = [ref_lens] * len(models)
            a_l = [alt_lens] * len(models)
        with torch.no_grad():
            for i, model in enumerate(models):
                y_refs = torch.softmax(model(x_r[i]), 1)
                y_alts = torch.softmax(model(x_a[i]), 1)
                y_refs[strands] = y_refs[strands].flip((2,))
                y_alts[strands] = y_alts[strands].flip((2,))
                for row in range(y_refs.shape[0]):
                    alt_len, ref_len = a_l[i][row], r_l[i][row]
                    if alt_len > ref_len:
                        diff = torch.cat(
                            (y_alts[row, 1:, :cov // 2 + ref_len - 1],
                             torch.max(y_alts[row, 1:, cov//2 + ref_len - 1: cov//2 + alt_len], 1, keepdim=True)[0],
                             y_alts[row, 1:, cov // 2 + alt_len:]), 1) - \
                               y_refs[row, 1:, :ref_len - alt_len]
                    elif alt_len < ref_len:
                        z = torch.zeros(2, ref_len - alt_len)
                        if cuda:
                            z = z.cuda(i)
                        diff = torch.cat((y_alts[row, 1:, :cov // 2 + alt_len], z,
                                          y_alts[row, 1:, cov // 2 + alt_len:alt_len - ref_len]), 1) - y_refs[row,
                                                                                                       1:, :]
                    else:
                        diff = y_alts[row, 1:, :] - y_refs[row, 1:, :]
                    diff = diff.cpu().numpy()
                    idx_ag = diff[0, :].argmax()
                    idx_al = diff[0, :].argmin()
                    idx_dg = diff[1, :].argmax()
                    idx_dl = diff[1, :].argmin()
                    ag = diff[0, idx_ag]
                    al = diff[0, idx_al]
                    dg = diff[1, idx_dg]
                    dl = diff[1, idx_dl]
                    fs[i].write("{}|{}|{}|{}|{:.2f}|{:.2f}|{:.2f}|{:.2f}|{}|{}|{}|{}\n".format(
                        chroms[row],
                        pos[row],
                        names[row],
                        alts[row],
                        ag,
                        -al,
                        dg,
                        -dl,
                        idx_ag - cov//2,
                        idx_al - cov//2,
                        idx_dg - cov//2,
                        idx_dl - cov//2))