import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

class CDR3_content:
    def __init__(self, seqs, aaCDR3_name="aaCDR3", len_from=None, len_to=None, save=None, sv_kwargs=None):
        if save is None:
            self.sv_flag = False
            self.sv_path = ""
        else:
            self.sv_flag = True
            self.sv_path = save
        if sv_kwargs is None:
            self.sv_kwargs = {}
        else:
            self.sv_kwargs = sv_kwargs
        #self.aaCDR3n = aaCDR3_name
        self.seqs = seqs.astype(str)
        self.seq_ls = np.char.str_len(self.seqs)
        self.length_counts = self.len_dist()
        if len_from is None:
            self.len_from = int(input("Consider lengths from (inclusive):"))
        else:
            self.len_from = len_from
        if len_to is None:
            self.len_to = int(input("Consider lengths up to (inclusive):"))
        else:
            self.len_to = len_to
        self.lengths = list(range(self.len_from, self.len_to + 1))
        self.dist_kept = sum([self.length_counts[l] for l in self.lengths])/len(seqs)
        print(self.dist_kept)
        self.cdr3_by_length = dict(zip(self.lengths, map(self.IMGT_CDR3_by_length, self.lengths)))
        # use max length to get correct order of IMGT positions
        self.IMGT_order = self.get_IMGT_pos(self.len_to)


    def len_dist(self, colour="k"):
        plt.figure()
        l, c = np.unique(self.seq_ls, return_counts=True)
        print(l, c)
        plt.bar(l, c, color=colour)
        plt.xlabel("CDR3 length")
        plt.ylabel("Frequency")
        self._save_or_show_plot("CDR3 length distribution")
        return dict(zip(l, c))

    def get_IMGT_pos(self, l):
        IMGT_pos = pd.read_csv("IMGT_pos.csv", index_col=0, dtype=np.float64)
        seq_range = IMGT_pos.columns[range(l)]
        return IMGT_pos[seq_range].loc[l].astype(str)

    def _map_to_IMGT(self, aa_grid, l):
        aa_grid.columns = self.get_IMGT_pos(l)
        return aa_grid

    def IMGT_CDR3_by_length(self, l):
        cdr3_l = self.seqs[self.seq_ls == l]
        aa_grid = list(map(list, cdr3_l))
        aa_grid = pd.DataFrame(data=list(aa_grid))
        IMGT_aa_grid = self._map_to_IMGT(aa_grid, l)
        return IMGT_aa_grid

    def _aa_perplex(self, aa_obs):
        # dict containing all possible aa
        unq_aa_obs, c_aa_obs = np.unique(aa_obs, return_counts=True)
        aa_dict = dict(zip(unq_aa_obs, c_aa_obs))
        c_obs = np.array(list(aa_dict.values()))
        n_obs = sum(c_obs)
        p = c_obs / n_obs
        pp_all = np.power(p, -p)
        perplex = np.prod(pp_all)
        return perplex

    # perpexity
    # plots: per length, for all, subplots per length
    def single_length_perplexity(self, l):
        aa_grid = self.cdr3_by_length[l]
        return aa_grid.apply(self._aa_perplex, axis=0)

    def all_length_perplexity(self):
        all_aa_grid = pd.concat(list(self.cdr3_by_length.values()), axis=0)
        # final IMGT grid has right order of positions
        all_aa_grid = all_aa_grid[self.IMGT_order]
        all_aa_grid = all_aa_grid.fillna("_")
        perplex_by_pos = all_aa_grid.apply(self._aa_perplex, axis=0)
        return perplex_by_pos

    def perplexity_bar(self, l=None):
        if l is None:
            ls_tidy = f"Perpexity given CDR3 lengths between {self.len_from} and {self.len_to}"
            ppp = self.all_length_perplexity()
        else:
            ls_tidy = f"Perpexity given CDR3 length {l}"
            ppp = self.single_length_perplexity(l)
        fig, axs = plt.subplots(1, figsize=(15,8))
        axs.bar(ppp.index, ppp.values)
        axs.set_xlabel("IMGT position")
        #axs.set_title(ls_tidy)
        self._save_or_show_plot(ls_tidy)

    def _locate_IMGT_pos(self, IMGT_pos):
        ord_rank = np.arange(len(self.IMGT_order))
        return [ord_rank[self.IMGT_order == pos]  for pos in IMGT_pos]

    def perplexity_scatter(self, fgsz=(10,6)):
        all_ppp = [self.single_length_perplexity(l) for l in self.lengths]
        all_ppp = pd.concat(all_ppp, axis=1)
        all_ppp.columns = self.lengths
        all_ppp["Pos"] = all_ppp.index
        all_ppp_long = pd.melt(all_ppp, id_vars="Pos")
        # this needs to be changed
        all_ppp_long = all_ppp_long.sort_values(by="Pos", ascending=True, key=self._locate_IMGT_pos)
        fig, ax = plt.subplots(1, figsize=fgsz)
        lc = [(self.length_counts[sc_len]/1000) for sc_len in all_ppp_long["variable"]]
        sc = ax.scatter(all_ppp_long["Pos"], all_ppp_long["value"], c=all_ppp_long["variable"], s=lc, cmap="turbo", alpha=0.8)
        legend1 = ax.legend(*sc.legend_elements(),loc="upper right", title="CDR3 length", frameon=False)
        ax.add_artist(legend1)
        ax.set_xlabel("IMGT position")
        ax.set_ylabel("bits")
        ax.tick_params(axis='x', rotation=90)
        pname = f"Perpexity given each of CDR3 lengths {self.len_from} to {self.len_to}"
        #ax.set_title(pname)
        plt.tight_layout()
        self._save_or_show_plot(pname)

    def _save_or_show_plot(self, plot_name):
        if self.sv_flag:
            plot_fname = plot_name.replace(" ", "_") + ".png"
            plot_fpath = os.path.join(self.sv_path, plot_fname)
            plt.savefig(plot_fpath, **self.sv_kwargs)
            plt.close()
        else:
            plt.title(plot_name)
            plt.show()
