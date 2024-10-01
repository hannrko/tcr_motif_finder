import numpy as np
import pandas as pd
import os
import scipy as sp

def sub_motif(motif, cdr3, saa=4, eaa=4):
    k = len(motif)
    l = len(cdr3)
    if saa + eaa > l:
        # don't insert motif if we have to overwrite start or end amino acids
        # start and end amino acids default 4 from each end
        print("CDR3 too short")
        return cdr3
    if saa + k + eaa > l:
        end_i = l - eaa
    else:
        end_i = saa + k
    return cdr3[:saa] + motif + cdr3[end_i:]

class RepertoireMotifInserter:
    def __init__(self, rep):
        # rep must be list of seqs or dict with seq keys and count values
        if isinstance(rep, list):
            self.cdr3s = rep
            self.counts = np.ones(len(rep))
        elif isinstance(rep, dict):
            self.cdr3s = list(rep.keys())
            self.counts = list(rep.values())
        self.seq_ls = np.array(list(map(len, self.cdr3s)))

    def insert(self, motifs, insrt_nums, saa=4, eaa=4):
        k = len(motifs)
        # only choose clones with sufficient length
        sub_choice_options = np.arange(len(self.cdr3s))[self.seq_ls >= (k + saa + eaa)]
        if len(sub_choice_options) < sum(insrt_nums):
            print("Can't sub in desired number of motifs, add more CDR3s or reduce sub_num")
            return self.cdr3s
        # sample clones without replacement so motif not inserted into same seq multiple times
        # also different motifs are not inserted into same seq
        sub_choices = np.random.choice(sub_choice_options, int(sum(insrt_nums)), replace=False)
        # expand motifs
        motifs_exp = np.repeat(motifs, insrt_nums)
        for sc, motif in zip(sub_choices, motifs_exp):
            self.cdr3s[sc] = sub_motif(motif, self.cdr3s[sc], saa=saa, eaa=eaa)
        return self.cdr3s

class DatasetMotifInserter:
    def __init__(self, inp_mat, svdir, fpaths=None, fdir=None, meta_dir=None, aacdr3_name="aaCDR3", rs=None):
        # this needs motif input file...
        # the input file just gives a number for each motif and each sample
        # separate code to generate it
        if fpaths is None:
            self.fdirs = fdir
            self.fnames = os.listdir(fdir)
            self.fpaths = [os.path.join(fdir, fn) for fn in self.fnames]
        else:
            self.fpaths = fpaths
            self.fnames = [os.path.split(fp)[1] for fp in self.fpaths]
            self.fdirs = np.unique([os.path.split(fp)[0] for fp in self.fpaths])
        self.svdir = svdir
        if not os.path.exists(self.svdir):
            os.mkdir(self.svdir)
        self.snames = [fn.split(".")[0] for fn in self.fnames]
        self.rs = rs
        if rs is not None:
            np.random.seed(rs)
        self.inp_mat = inp_mat.loc[self.snames]
        self.aacdr3_name = aacdr3_name

    def check_paths(self):
        print(f"{len(self.fnames)} repertoires to be loaded from {self.fdirs}")
        print(f"Motif-inserted repertoires to be saved to {self.svdir}")

    def insert_rep(self, i):
        # read repertoire
        rep = pd.read_csv(self.fpaths[i], index_col=0)
        # get motif info as series
        motifs = self.inp_mat.loc[self.snames[i]]
        # insert motifs into copy
        rep_insrt = RepertoireMotifInserter(list(rep[self.aacdr3_name].values))
        rep_final = rep
        rep_final[self.aacdr3_name] = rep_insrt.insert(motifs.index, motifs.values)
        # save new repertoire
        rep_sv_path = os.path.join(self.svdir, self.fnames[i])
        rep_final.to_csv(rep_sv_path)

    def insert_dataset(self):
        for i in range(len(self.fpaths)):
            self.insert_rep(i)

def exclusive_motifs(motifs, mot_ratio, n_motifs, sam_names):
    n_sam = len(sam_names)
    mot_prop = np.array(mot_ratio) / sum(mot_ratio)
    mot_num = np.round(mot_prop * n_sam)
    mot_ind = np.concatenate([[0], np.cumsum(mot_num)]).astype(int)
    cols = []
    for i, mot in enumerate(motifs):
        col = np.zeros(n_sam)
        col[mot_ind[i]:mot_ind[i+1]] = n_motifs
        cols.append(col)
    mot_nums = np.array(cols).T
    return pd.DataFrame(index=sam_names, columns=motifs, data=mot_nums)

# p_motifs and p_mot_ratio should be arrays?
# try constructing for setting ON or OFF for set of samples and each motif
def binom_motifs(motifs, n_seq, p_motifs, sam_names):
    n_sam = len(sam_names)
    mot_nums = np.array([sp.stats.binom.rvs(n=n_seq, p=p_motifs[i], size=n_sam) for i, mot in enumerate(motifs)])
    return pd.DataFrame(index=sam_names, columns=motifs, data=mot_nums.T)

def split_exclusive(values, split_nums):
    return np.repeat(values, split_nums)
