from IPython.display import clear_output
import numpy as np
import matplotlib.pyplot as plt

class ExperimentLogger:
    def log(self, values):
        for k, v in values.items():
            if k not in self.__dict__:
                self.__dict__[k] = [v]
            else:
                self.__dict__[k] += [v]



def display_train_stats(cfl_stats, communication_rounds):
    clear_output(wait=False)
    
    plt.figure(figsize=(12,4))
    
    plt.subplot(1,2,1)
    acc_mean = np.mean(cfl_stats.acc_server, axis=1)
    acc_std = np.std(cfl_stats.acc_server, axis=1)
    plt.fill_between(cfl_stats.rounds, acc_mean-acc_std, acc_mean+acc_std, alpha=0.5, color="C0")
    plt.plot(cfl_stats.rounds, acc_mean, color="C0")
    
    if "split" in cfl_stats.__dict__:
        for s in cfl_stats.split:
            plt.axvline(x=s, linestyle="-", color="k", label=r"Split")
    
    
    plt.text(x=communication_rounds, y=1, ha="right", va="top", s="foo")
    
    plt.xlabel("Communication Rounds")
    plt.ylabel("Accuracy")
    
    plt.xlim(0, communication_rounds)
    plt.ylim(0,1)
    
    # plt.plot(cfl_stats.rounds, cfl_stats.mean_norm, color="C1", label=r"$\|\sum_i\Delta W_i \|$")
    # plt.plot(cfl_stats.rounds, cfl_stats.max_norm, color="C2", label=r"$\max_i\|\Delta W_i \|$")
    
    # plt.legend()
    
    plt.show()
