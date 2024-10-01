import numpy as np
import pandas as pd
import os
import itertools
import aa_alph_redu.aa_alph_reducer as aaar
import matplotlib.pyplot as plt
from interactive_scatter import InteractiveScatter
import seaborn as sns
from matplotlib.colors import LogNorm

class MotifFinder:
    def __init__(self, seqs, k, seq_name, include_unobs=False):
        # could add some checks here so we don't run into computational difficulty
        self.k = k
        self.dsn = seq_name
        # decompose into kmers
        self.kmers, self.kmer_counts = self.to_kmers(seqs)
        # compare to possible kmers
        if include_unobs:
            poss_kmers = self.all_kmers()
            unobs_kmers = np.setdiff1d(poss_kmers, self.kmers)
            print(f"{len(unobs_kmers)} out of {len(poss_kmers)} {self.k}mers not observed")
            self.kmers = np.concatenate((self.kmers, unobs_kmers))
            self.kmer_counts = np.concatenate((self.kmer_counts, np.zeros(len(unobs_kmers))))

    def find_similar(self, ra_size, ra_enc, max_gc_lim=[None, 100], g_size_lim=[0,None], kmer_path="kmer_groups", no_cys=False, no_ra_repeat=False):
        # define reduced alphabet
        aaclus_kwargs = {"aa_enc": ra_enc, "cmethod": "average", "cmetric": "euclidean"}
        motif_ra = aaar.ReducedAAAlphabet(aaclus_kwargs, ra_size, k=self.k)
        # convert the kmers
        ra_kmers = np.array([motif_ra.convert(seq) for seq in self.kmers])
        kmer_tab = pd.DataFrame(data=np.column_stack((self.kmers, self.kmer_counts)), columns=["Motif", "Count"])
        kmer_tab["RAParent"] = ra_kmers
        kmer_tab = kmer_tab.astype({"Motif": "str", "Count": "float64", "RAParent": "str"})
        # Use tidy data approach to get table of motifs
        kmer_group_tab = kmer_tab.groupby("RAParent").agg(MaxCount=pd.NamedAgg(column="Count", aggfunc="max"),
                                                          GroupSize=pd.NamedAgg(column="Motif", aggfunc="size"))
        kmer_group_tab = kmer_group_tab.reset_index()
        # combine tabs
        kmer_tab = kmer_tab.merge(kmer_group_tab, on="RAParent", how="left")
        # use other conditions to reduce motif list
        if no_cys:
            kmer_tab = self._apply_condition(kmer_tab, self._aa_not, "Motif", ("C"))
        if no_ra_repeat:
            kmer_tab = self._apply_condition(kmer_tab, self._distinct_aa, "RAParent")
        # use initial conditions to select kmer groups for interactive selection
        isc = InteractiveScatter(x_lim=max_gc_lim, y_lim=g_size_lim, axs_names=["GroupSize", "MaxCount"])
        gdata, gdata_c = np.unique(list(zip(kmer_tab["GroupSize"].values, kmer_tab["MaxCount"].values)),
                                   return_counts=True, axis=0)
        gs_data = gdata[:, 0]
        gc_data = gdata[:, 1]
        gs_lim, gc_lim = isc.zoom_plot(gs_data, gc_data, gdata_c)
        gs_mask = self._inside_mask(kmer_tab["GroupSize"].values.astype(int), *gs_lim)
        gc_mask = self._inside_mask(kmer_tab["MaxCount"].values.astype(float), *gc_lim)
        final_sel = np.logical_and(gs_mask, gc_mask)
        final_kmer_tab = kmer_tab[final_sel]
        final_kmer_tab = final_kmer_tab.sort_values(by=["MaxCount","GroupSize","RAParent", "Count"], ignore_index=True)
        tidy_ra_size = str(ra_size) if motif_ra.final_alph.alph_catg == "single" else "-".join([str(ras) for ras in ra_size])
        tidy_mc_lims = self._get_tidy_lims(final_kmer_tab["MaxCount"])
        tidy_gs_lims = self._get_tidy_lims(final_kmer_tab["GroupSize"])
        tidy_is_cys = "no_cys" if no_cys else ""
        tidy_is_repeat = "no_ra_repeat" if no_ra_repeat else ""
        fn = f"{self.dsn}_{self.k}mer_motif_{ra_enc}RA{tidy_ra_size}groups_{tidy_is_cys}_{tidy_is_repeat}_maxcount{tidy_mc_lims}_size{tidy_gs_lims}.csv"
        self._save_kmers(final_kmer_tab, kmer_path, fn)
        return kmer_tab, final_kmer_tab

    def _save_kmers(self, kmer_df, kmer_path, kmer_df_name):
        if not os.path.exists(kmer_path):
            os.mkdir(kmer_path)
        kmer_df.to_csv(os.path.join(kmer_path, kmer_df_name))

    def _inside_mask(self, nums, lb, ub):
        above_lb = nums >= lb
        below_ub = nums <= ub
        inside = np.logical_and(above_lb, below_ub)
        return inside

    def _get_tidy_lims(self, vals):
        return f"{round(min(vals))}-{round(max(vals))}"

    def to_kmers(self, seqs):
        # kmerise each sequence
        kmers = list(itertools.chain(*map(self.seq_to_kmers, seqs)))
        # count them up
        return np.unique(kmers, return_counts=True)

    def seq_to_kmers(self, seq):
        return [seq[i:i+self.k] for i in range(len(seq) - self.k + 1)]

    def all_kmers(self):
        aa = "ACDEFGHIKLMNPQRSTVWY"
        return [''.join(comb) for comb in itertools.product(aa, repeat=self.k)]

    def plot_kmer_dist(self):
        plt.figure()
        ord_counts = np.flip(np.sort(self.kmer_counts))
        plt.plot(range(len(ord_counts)), np.log(ord_counts))
        plt.show()

    def _apply_condition(self, motif_df, func, motif_name, func_args=None):
        if func_args is None:
            func_args = ()
        motif_cond = motif_df[motif_name].apply(func, args=func_args)
        return motif_df[motif_cond]

    def _distinct_aa(self, seq):
        dstnct_aa = np.unique(list(seq))
        return len(dstnct_aa) == self.k

    def _aa_not(self, seq, aa):
        return aa not in seq
